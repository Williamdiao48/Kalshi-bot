"""
Build temperature-delta training data for the delta forecast model.

For each (city, date, hour H) where we have IEM observations:
  - Compute running_obs: running max (high markets) or running min (low markets) up to hour H
  - Label: delta_to_final = actual_final - running_obs (how much more temp will change today)
    High markets: delta >= 0 always (running max is monotone non-decreasing)
    Low markets:  delta <= 0 always (running min is monotone non-increasing)
  - Features: running obs level, momentum, model forecast remainders, climatology

No band ceiling required — rows emitted for all hours regardless of current running obs.
This enables the model to fire before the NO signal crosses the ceiling, extending
coverage to anticipatory entries and summer markets where the signal is less clean.

At inference: P(NO wins) = P(running_obs + delta > band_ceil)
             = P(delta > band_ceil - running_obs)
computed from model's predicted delta distribution.

Input:  data/backtest/band_arb_hist_cache.json
Output: data/backtest/delta_training_data.csv

Usage:
  venv/bin/python scripts/build_delta_training_data.py
  venv/bin/python scripts/build_delta_training_data.py --high-only
  venv/bin/python scripts/build_delta_training_data.py --start 2026-03-27  # spring-only for comparison
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CACHE_PATH = Path("data/backtest/band_arb_hist_cache.json")
OUT_CSV    = Path("data/backtest/delta_training_data.csv")

HOURS = list(range(4, 23))

STATION_TO_CITY = {
    "LAX": "lax", "DEN": "den", "MDW": "chi", "NYC": "ny",  "MIA": "mia",
    "AUS": "aus", "DAL": "dal", "BOS": "bos", "HOU": "hou", "DFW": "dfw",
    "SFO": "sfo", "SEA": "sea", "PHX": "phx", "PHL": "phl", "ATL": "atl",
    "MSP": "msp", "DCA": "dca", "LAS": "las", "OKC": "okc", "SAT": "sat",
    "MSY": "msy",
}

CSV_FIELDS = [
    "metric", "date", "hour_utc", "is_high", "month", "city",
    "running_obs",         # running max/min at this hour
    "delta_1h",            # change in running_obs vs 1h ago
    "delta_2h",            # change in running_obs vs 2h ago
    "hours_to_close",      # hours until market close (22 - hour_utc)
    "hrrr_remaining",      # hrrr_f - running_obs (how much more HRRR expects)
    "gfs_remaining",       # gfs_f - running_obs
    "consensus_remaining", # median(models) - running_obs
    "model_spread",        # max - min across daily model forecasts
    "obs_vs_hrrr_h",       # running_obs - HRRR hourly forecast at this hour
    "obs_vs_gfs_h",        # running_obs - GFS hourly forecast at this hour
    "recent_hrrr_mae_7d",  # rolling 7-day mean |HRRR_daily - actual|
    "clim_p25",            # 25th pct of historical delta_to_final for (city, month, hour)
    "clim_p50",            # median historical delta
    "clim_p75",            # 75th pct historical delta
    "actual_f",            # IEM official daily max/min (for reference)
    "delta_to_final",      # LABEL: actual_f - running_obs
]


def find_cache_key(cache: dict, prefix: str) -> str | None:
    return next((k for k in cache if k.startswith(prefix)), None)


def get_hourly_obs(cache: dict, station: str, date: str) -> dict[int, float]:
    key = find_cache_key(cache, f"hourly_{station}_")
    if not key:
        return {}
    return {int(h): float(v) for h, v in cache[key].get(date, {}).items()}


def get_actual(cache: dict, station: str, is_high: bool, date: str) -> float | None:
    suffix = "high" if is_high else "low"
    key = next((k for k in cache if k.startswith(f"actual_{station}_") and k.endswith(f"_{suffix}")), None)
    if not key:
        return None
    v = cache[key].get(date)
    return float(v) if v is not None else None


def get_hrrr_daily(cache: dict, city: str, is_high: bool, date: str) -> float | None:
    direction = "high" if is_high else "low"
    key = find_cache_key(cache, f"hrrr_temp_{direction}_{city}_")
    if not key:
        return None
    v = cache[key].get(date)
    return float(v) if v is not None else None


def get_om_models(cache: dict, city: str, is_high: bool, date: str) -> dict[str, float]:
    direction = "high" if is_high else "low"
    key = find_cache_key(cache, f"om_temp_{direction}_{city}_")
    if not key:
        return {}
    om = cache[key]
    result = {}
    for model_name, dates in om.items():
        v = dates.get(date)
        if v is not None:
            result[model_name] = float(v)
    return result


def get_hourly_fc(cache: dict, source: str, city: str, is_high: bool, date: str) -> dict[int, float]:
    direction = "high" if is_high else "low"
    key = find_cache_key(cache, f"hourly_fc_{source}_temp_{direction}_{city}_")
    if not key:
        return {}
    return {int(h): float(v) for h, v in cache[key].get(date, {}).items()}


def build_remaining_climatology(cache: dict) -> dict[tuple, list[float]]:
    """
    For each (city, is_high, month, hour): list of observed delta_to_final values.
    delta_to_final = actual_final - running_obs_at_hour
    High: positive (how much more max rises)
    Low:  negative (how much more min falls)
    Built from all available IEM data in cache.
    """
    clim: dict[tuple, list[float]] = defaultdict(list)

    for station, city in STATION_TO_CITY.items():
        for is_high in (True, False):
            suffix = "high" if is_high else "low"
            fn = max if is_high else min

            hourly_key = find_cache_key(cache, f"hourly_{station}_")
            actual_key = next(
                (k for k in cache if k.startswith(f"actual_{station}_") and k.endswith(f"_{suffix}")),
                None,
            )
            if not hourly_key or not actual_key:
                continue

            hourly_data = cache[hourly_key]
            actual_data = cache[actual_key]

            for date_str, obs in hourly_data.items():
                actual_f = actual_data.get(date_str)
                if actual_f is None:
                    continue
                actual_f = float(actual_f)
                month = int(date_str[5:7])

                running = None
                for h in HOURS:
                    v = obs.get(str(h))
                    if v is not None:
                        running = fn([running, float(v)]) if running is not None else float(v)
                    if running is not None:
                        delta = actual_f - running
                        clim[(city, is_high, month, h)].append(delta)

    return clim


def clim_percentiles(clim: dict, city: str, is_high: bool, month: int, hour: int) -> tuple[float, float, float]:
    """Return (p25, p50, p75) of historical delta_to_final."""
    vals = clim.get((city, is_high, month, hour), [])
    if not vals:
        # sensible defaults: high markets typically rise 2-4°F more, low fall 1-3°F more
        if is_high:
            return 0.5, 2.0, 4.0
        else:
            return -3.0, -1.5, -0.5
    vals_sorted = sorted(vals)
    n = len(vals_sorted)
    return (
        round(vals_sorted[max(0, int(n * 0.25))], 2),
        round(vals_sorted[n // 2], 2),
        round(vals_sorted[min(n - 1, int(n * 0.75))], 2),
    )


def compute_rolling_hrrr_mae(cache: dict, city: str, is_high: bool, station: str, window: int = 7) -> dict[str, float]:
    direction = "high" if is_high else "low"
    hrrr_key = find_cache_key(cache, f"hrrr_temp_{direction}_{city}_")
    actual_key = next(
        (k for k in cache if k.startswith(f"actual_{station}_") and k.endswith(f"_{direction}")),
        None,
    )
    if not hrrr_key or not actual_key:
        return {}

    hrrr = cache[hrrr_key]
    actual = cache[actual_key]

    day_errors: dict[str, float] = {}
    for d, hf in hrrr.items():
        af = actual.get(d)
        if hf is not None and af is not None:
            try:
                day_errors[d] = abs(float(hf) - float(af))
            except (TypeError, ValueError):
                pass

    rolling: dict[str, float] = {}
    for d in day_errors:
        dt = datetime.strptime(d, "%Y-%m-%d")
        past = [
            day_errors[(dt - timedelta(days=i)).strftime("%Y-%m-%d")]
            for i in range(1, window + 1)
            if (dt - timedelta(days=i)).strftime("%Y-%m-%d") in day_errors
        ]
        if past:
            rolling[d] = round(sum(past) / len(past), 2)
    return rolling


def build_rows(
    station: str,
    city: str,
    is_high: bool,
    cache: dict,
    clim: dict,
    start: str,
    end: str,
) -> list[dict]:
    fn = max if is_high else min
    direction = "high" if is_high else "low"
    metric = f"temp_{direction}_{city}"

    hourly_all = get_hourly_obs.__wrapped__ if hasattr(get_hourly_obs, "__wrapped__") else None

    hourly_key = find_cache_key(cache, f"hourly_{station}_")
    actual_key = next(
        (k for k in cache if k.startswith(f"actual_{station}_") and k.endswith(f"_{direction}")),
        None,
    )
    if not hourly_key or not actual_key:
        return []

    hourly_all_dates: dict[str, dict] = cache[hourly_key]
    actual_all: dict[str, float] = cache[actual_key]

    rolling_mae = compute_rolling_hrrr_mae(cache, city, is_high, station)

    rows = []
    common_dates = sorted(set(hourly_all_dates) & set(actual_all))

    for date_str in common_dates:
        if date_str < start or date_str > end:
            continue

        actual_f = actual_all.get(date_str)
        if actual_f is None:
            continue
        actual_f = float(actual_f)

        obs_by_hour: dict[int, float] = {
            int(h): float(v) for h, v in hourly_all_dates[date_str].items()
        }

        hrrr_f = get_hrrr_daily(cache, city, is_high, date_str)
        om_models = get_om_models(cache, city, is_high, date_str)
        fc_vals = ([hrrr_f] if hrrr_f is not None else []) + list(om_models.values())
        if not fc_vals:
            continue
        consensus_f = median(fc_vals)
        spread = max(fc_vals) - min(fc_vals) if len(fc_vals) > 1 else 0.0

        hrrr_hourly = get_hourly_fc(cache, "hrrr", city, is_high, date_str)
        gfs_hourly  = get_hourly_fc(cache, "gfs",  city, is_high, date_str)

        month = int(date_str[5:7])
        running: dict[int, float] = {}
        cur = None

        for h in HOURS:
            v = obs_by_hour.get(h)
            if v is not None:
                cur = fn([cur, v]) if cur is not None else v
            if cur is not None:
                running[h] = cur

        for h in HOURS:
            if h not in running:
                continue

            ro = running[h]
            delta_to_final = round(actual_f - ro, 2)

            r_prev1 = running.get(h - 1)
            r_prev2 = running.get(h - 2)
            delta_1h = round(ro - r_prev1, 2) if r_prev1 is not None else 0.0
            delta_2h = round(ro - r_prev2, 2) if r_prev2 is not None else 0.0

            hrrr_h = hrrr_hourly.get(h)
            gfs_h  = gfs_hourly.get(h)
            gfs_f  = om_models.get("gfs_seamless")

            clim_p25, clim_p50, clim_p75 = clim_percentiles(clim, city, is_high, month, h)

            rows.append({
                "metric":             metric,
                "date":               date_str,
                "hour_utc":           h,
                "is_high":            1 if is_high else 0,
                "month":              month,
                "city":               city,
                "running_obs":        round(ro, 2),
                "delta_1h":           delta_1h,
                "delta_2h":           delta_2h,
                "hours_to_close":     max(0, 22 - h),
                "hrrr_remaining":     round(hrrr_f - ro, 2) if hrrr_f is not None else "",
                "gfs_remaining":      round(gfs_f  - ro, 2) if gfs_f  is not None else "",
                "consensus_remaining":round(consensus_f - ro, 2),
                "model_spread":       round(spread, 2),
                "obs_vs_hrrr_h":      round(ro - hrrr_h, 2) if hrrr_h is not None else "",
                "obs_vs_gfs_h":       round(ro - gfs_h,  2) if gfs_h  is not None else "",
                "recent_hrrr_mae_7d": rolling_mae.get(date_str, 3.0),
                "clim_p25":           clim_p25,
                "clim_p50":           clim_p50,
                "clim_p75":           clim_p75,
                "actual_f":           actual_f,
                "delta_to_final":     delta_to_final,
            })

    return rows


def main(start: str, end: str, high_only: bool, low_only: bool) -> None:
    if not CACHE_PATH.exists():
        print(f"Cache not found: {CACHE_PATH}")
        print("Run build_band_arb_training_data.py first.")
        return

    print("Loading cache...")
    cache = json.loads(CACHE_PATH.read_text())

    print("Building remaining-delta climatology from IEM history...")
    clim = build_remaining_climatology(cache)
    n_city_buckets = len(clim)
    print(f"  {n_city_buckets} (city, is_high, month, hour) buckets")

    all_rows: list[dict] = []
    pairs = []
    if not low_only:
        pairs += [(s, c, True) for s, c in STATION_TO_CITY.items()]
    if not high_only:
        pairs += [(s, c, False) for s, c in STATION_TO_CITY.items()]

    for station, city, is_high in pairs:
        rows = build_rows(station, city, is_high, cache, clim, start, end)
        label = "high" if is_high else "low"
        if rows:
            n_pos = sum(1 for r in rows if r["delta_to_final"] >= 0)
            print(f"  {label} {city}: {len(rows):,} rows  "
                  f"median_delta={sorted(r['delta_to_final'] for r in rows)[len(rows)//2]:+.1f}°F")
        all_rows.extend(rows)

    if not all_rows:
        print("No rows generated.")
        return

    Path("data/backtest").mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(all_rows)

    high_rows = [r for r in all_rows if r["is_high"] == 1]
    low_rows  = [r for r in all_rows if r["is_high"] == 0]
    print(f"\nExported {len(all_rows):,} rows → {OUT_CSV}")
    if high_rows:
        hi_deltas = sorted(r["delta_to_final"] for r in high_rows)
        n = len(hi_deltas)
        print(f"  High: {n:,} rows  "
              f"p25={hi_deltas[n//4]:+.1f}  p50={hi_deltas[n//2]:+.1f}  p75={hi_deltas[3*n//4]:+.1f}°F")
    if low_rows:
        lo_deltas = sorted(r["delta_to_final"] for r in low_rows)
        n = len(lo_deltas)
        print(f"  Low:  {n:,} rows  "
              f"p25={lo_deltas[n//4]:+.1f}  p50={lo_deltas[n//2]:+.1f}  p75={lo_deltas[3*n//4]:+.1f}°F")

    # Distribution by month
    from collections import defaultdict
    by_month: dict[int, list] = defaultdict(list)
    for r in all_rows:
        by_month[int(r["month"])].append(r["delta_to_final"])
    print("\nMedian delta_to_final by month (+ = still rising, - = still falling):")
    for m in sorted(by_month):
        vals = sorted(by_month[m])
        n = len(vals)
        print(f"  Month {m:>2}:  n={n:>6,}  "
              f"p25={vals[n//4]:+.1f}  median={vals[n//2]:+.1f}  p75={vals[3*n//4]:+.1f}°F")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",     default="2022-01-01")
    parser.add_argument("--end",       default="2026-12-31")
    parser.add_argument("--high-only", action="store_true")
    parser.add_argument("--low-only",  action="store_true")
    args = parser.parse_args()
    main(args.start, args.end, args.high_only, args.low_only)
