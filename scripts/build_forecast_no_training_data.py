"""
Build hour-by-hour training data for a forecast_no ML model.

For each (city, date, hour H) where the NO signal has fired
(running obs > band_ceil), emit one training row capturing what
the bot knows at that moment:

  - How far above the band ceiling the running obs is (margin_f)
  - Whether the temperature is still moving (delta_1h, delta_2h)
  - What time it is (hour_utc, hours_to_close)
  - What the model forecasts say relative to the band ceiling

Label: won = 1 if actual final daily max/min > band_ceil (NO wins)

Multiple rows per (city, date) at different hours show how model
confidence should evolve as the day progresses and the observation
becomes more "locked in."

Input:  data/backtest/band_arb_hist_cache.json  (from build_band_arb_training_data.py)
Output: data/backtest/forecast_no_training_data.csv

Usage:
  venv/bin/python scripts/build_forecast_no_training_data.py
  venv/bin/python scripts/build_forecast_no_training_data.py --high-only
  venv/bin/python scripts/build_forecast_no_training_data.py --low-only
"""

import argparse
import csv
import json
import os
import sys
from datetime import date
from pathlib import Path
from statistics import median

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.build_band_arb_training_data import IEM_STATIONS, _ENTRY_HOURS

CACHE_PATH = Path("data/backtest/band_arb_hist_cache.json")
OUT_CSV    = Path("data/backtest/forecast_no_training_data.csv")

# Hours UTC to evaluate signal at. Covers morning through late afternoon
# for all timezones (ET 4am–8pm, PT 9pm–1pm next day).
HOURS_TO_CHECK = list(range(4, 23))

# Minimum margin (running_obs - band_ceil) for the NO signal to be active.
MIN_MARGIN = 0.5

CSV_FIELDS = [
    "metric", "date", "hour_utc", "is_high", "month", "city",
    "running_obs",           # METAR running max/min at this hour
    "band_ceil",             # simulated band ceiling
    "margin_f",              # running_obs - band_ceil (how far above)
    "delta_1h",              # change in running_obs vs 1h ago (still moving?)
    "delta_2h",              # change in running_obs vs 2h ago
    "hours_above_ceil",      # consecutive hours running_obs has been above band_ceil
    "hours_to_close",        # approximate hours until market close (22 - hour_utc)
    "obs_vs_hrrr_h",         # running_obs minus HRRR's hourly forecast at this hour
    "obs_vs_gfs_h",          # running_obs minus GFS's hourly forecast at this hour
    "actual_f",              # IEM official daily max/min (label source)
    "hrrr_vs_ceil",          # HRRR daily forecast - band_ceil
    "gfs_vs_ceil",           # GFS daily forecast - band_ceil
    "ecmwf_vs_ceil",         # ECMWF daily forecast - band_ceil
    "consensus_vs_ceil",     # median forecast - band_ceil
    "model_spread",          # max - min across available models
    "n_models",
    "n_models_above_ceil",   # models predicting above ceiling (NO safer)
    "won",                   # 1 = NO wins (actual_f > band_ceil)
]


def load_cache() -> dict:
    if not CACHE_PATH.exists():
        print(f"Cache not found: {CACHE_PATH}")
        print("Run build_band_arb_training_data.py first.")
        sys.exit(1)
    return json.loads(CACHE_PATH.read_text())


def running_obs(hourly: dict[str, float], up_to_hour: int, is_high: bool) -> float | None:
    """Running max (is_high) or min (not is_high) up to and including up_to_hour."""
    temps = [v for h, v in hourly.items() if int(h) <= up_to_hour]
    if not temps:
        return None
    return max(temps) if is_high else min(temps)


def build_rows(
    metric: str,
    is_high: bool,
    station: str,
    cache: dict,
    start: str,
    end: str,
) -> list[dict]:
    hourly_key      = f"hourly_{station}_{start}_{end}"
    actual_key      = f"actual_{station}_{start}_{end}_{'high' if is_high else 'low'}"
    om_key          = f"om_{metric}_{start}_{end}_{'high' if is_high else 'low'}"
    hrrr_key        = f"hrrr_{metric}_2022-01-01_{end}_{'high' if is_high else 'low'}"
    hrrr_hourly_key = f"hourly_fc_hrrr_{metric}_{start}_{end}"
    gfs_hourly_key  = f"hourly_fc_gfs_{metric}_{start}_{end}"

    hourly_cache:      dict[str, dict[str, float]] = cache.get(hourly_key, {})
    actual_cache:      dict[str, float]            = cache.get(actual_key, {})
    om_cache:          dict[str, dict[str, float]] = cache.get(om_key, {})
    hrrr_cache:        dict[str, float]            = cache.get(hrrr_key, {})
    # Hourly forecast caches: {date: {hour: temp_f}} — empty dict if not yet fetched
    hrrr_hourly_cache: dict[str, dict[str, float]] = cache.get(hrrr_hourly_key, {})
    gfs_hourly_cache:  dict[str, dict[str, float]] = cache.get(gfs_hourly_key, {})

    if not hourly_cache or not actual_cache:
        return []

    city = metric.split("_")[-1]  # "temp_high_lax" → "lax"

    rows = []
    common_dates = sorted(set(hourly_cache) & set(actual_cache))

    for d in common_dates:
        if d < start or d > end:
            continue

        actual = actual_cache[d]
        hourly: dict[int, float] = {int(h): v for h, v in hourly_cache[d].items()}

        # Hourly forecast temps for this date {hour: temp_f}
        hrrr_hourly_d: dict[int, float] = {int(h): v for h, v in hrrr_hourly_cache.get(d, {}).items()}
        gfs_hourly_d:  dict[int, float] = {int(h): v for h, v in gfs_hourly_cache.get(d,  {}).items()}

        # Daily model forecasts for this date
        hrrr_f  = hrrr_cache.get(d)
        gfs_f   = om_cache.get("gfs_seamless",  {}).get(d)
        ecmwf_f = om_cache.get("ecmwf_ifs025", {}).get(d)
        gem_f   = om_cache.get("gem_seamless",  {}).get(d)
        icon_f  = om_cache.get("icon_seamless", {}).get(d)

        model_vals = [v for v in (hrrr_f, gfs_f, ecmwf_f, gem_f, icon_f) if v is not None]
        if len(model_vals) < 2:
            continue

        consensus = median(model_vals)
        spread    = max(model_vals) - min(model_vals)

        dt = date.fromisoformat(d)
        month = dt.month

        prev_obs: dict[int, float | None] = {}
        hours_above: int = 0  # consecutive hours signal has been active

        for h in HOURS_TO_CHECK:
            obs = running_obs(hourly, h, is_high)
            if obs is None:
                continue

            band_ceil = round(obs) - 1
            margin    = obs - band_ceil
            if margin < MIN_MARGIN:
                prev_obs[h] = obs
                hours_above = 0
                continue

            hours_above += 1

            obs_1h_ago = prev_obs.get(h - 1)
            obs_2h_ago = prev_obs.get(h - 2)
            delta_1h   = round(obs - obs_1h_ago, 2) if obs_1h_ago is not None else ""
            delta_2h   = round(obs - obs_2h_ago, 2) if obs_2h_ago is not None else ""

            # obs vs hourly model forecast at this specific hour
            hrrr_h = hrrr_hourly_d.get(h)
            gfs_h  = gfs_hourly_d.get(h)
            obs_vs_hrrr_h = round(obs - hrrr_h, 2) if hrrr_h is not None else ""
            obs_vs_gfs_h  = round(obs - gfs_h,  2) if gfs_h  is not None else ""

            n_above = sum(1 for v in model_vals if v > band_ceil)

            rows.append({
                "metric":            metric,
                "date":              d,
                "hour_utc":          h,
                "is_high":           1 if is_high else 0,
                "month":             month,
                "city":              city,
                "running_obs":       round(obs, 2),
                "band_ceil":         band_ceil,
                "margin_f":          round(margin, 2),
                "delta_1h":          delta_1h,
                "delta_2h":          delta_2h,
                "hours_above_ceil":  hours_above,
                "hours_to_close":    max(0, 22 - h),
                "obs_vs_hrrr_h":     obs_vs_hrrr_h,
                "obs_vs_gfs_h":      obs_vs_gfs_h,
                "actual_f":          actual,
                "hrrr_vs_ceil":      round(hrrr_f  - band_ceil, 2) if hrrr_f  is not None else "",
                "gfs_vs_ceil":       round(gfs_f   - band_ceil, 2) if gfs_f   is not None else "",
                "ecmwf_vs_ceil":     round(ecmwf_f - band_ceil, 2) if ecmwf_f is not None else "",
                "consensus_vs_ceil": round(consensus - band_ceil, 2),
                "model_spread":      round(spread, 2),
                "n_models":          len(model_vals),
                "n_models_above_ceil": n_above,
                "won": 1 if actual > band_ceil else 0,
            })

            prev_obs[h] = obs

    return rows


def main(start: str, end: str, high_only: bool, low_only: bool) -> None:
    cache = load_cache()

    # Infer date range from cache keys if not specified
    all_rows: list[dict] = []

    from kalshi_bot.cities import CITIES, LOW_CITIES
    metrics: list[tuple[str, str, bool]] = []
    if not low_only:
        for metric in CITIES:
            if metric in IEM_STATIONS:
                metrics.append((metric, IEM_STATIONS[metric][0], True))
    if not high_only:
        from kalshi_bot.cities import LOW_CITIES
        for metric in LOW_CITIES:
            if metric in IEM_STATIONS:
                metrics.append((metric, IEM_STATIONS[metric][0], False))

    for metric, station, is_high in metrics:
        rows = build_rows(metric, is_high, station, cache, start, end)
        won  = sum(r["won"] for r in rows)
        tag  = "KXHIGH" if is_high else "KXLOWT"
        if rows:
            print(f"{tag} {metric}: {len(rows)} rows  WR={100*won/len(rows):.1f}%")
        all_rows.extend(rows)

    if not all_rows:
        print("No rows generated. Check cache.")
        return

    Path("data/backtest").mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(all_rows)

    total_won = sum(r["won"] for r in all_rows)
    high_rows = [r for r in all_rows if r["is_high"]]
    low_rows  = [r for r in all_rows if not r["is_high"]]

    print(f"\nExported {len(all_rows):,} rows → {OUT_CSV}")
    print(f"Overall WR: {100*total_won/len(all_rows):.1f}%")
    if high_rows:
        hw = sum(r["won"] for r in high_rows)
        print(f"  KXHIGH: {len(high_rows):,} rows  WR={100*hw/len(high_rows):.1f}%")
    if low_rows:
        lw = sum(r["won"] for r in low_rows)
        print(f"  KXLOWT: {len(low_rows):,} rows  WR={100*lw/len(low_rows):.1f}%")

    # WR by hour
    from collections import defaultdict
    by_hour: dict[int, list[int]] = defaultdict(list)
    for r in all_rows:
        by_hour[r["hour_utc"]].append(r["won"])
    print("\nWR by hour UTC:")
    for h in sorted(by_hour):
        grp = by_hour[h]
        wr = sum(grp) / len(grp)
        print(f"  {h:>2}UTC  n={len(grp):>5}  WR={100*wr:>5.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",     default="2022-01-01")
    parser.add_argument("--end",       default="2026-05-31")
    parser.add_argument("--high-only", action="store_true")
    parser.add_argument("--low-only",  action="store_true")
    args = parser.parse_args()
    main(args.start, args.end, args.high_only, args.low_only)
