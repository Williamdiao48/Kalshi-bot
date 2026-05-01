#!/usr/bin/env python3
"""Backtest the parabolic diurnal temperature trajectory model.

Pulls the last N days of individual METAR observations from the FAA
Aviation Weather API, then simulates the trajectory projection at each
hourly checkpoint during the warming phase. Compares projected peak to
the actual daily maximum to measure:

  - Projection error distribution (mean, std, percentiles)
  - Directional accuracy (did the model correctly predict the direction
    of market movement relative to a given strike threshold?)
  - Time-of-day accuracy (is the model more reliable at 10 AM vs 2 PM?)
  - City-level variance (which cities is the model confident on?)

Usage:
  cd /path/to/Kalshi-Bot
  venv/bin/python scripts/backtest_trajectory.py [--days 7] [--city all|chi|ny|...]

Output:
  - Summary table: city × hour → mean error, std, n_samples
  - Overall statistics and recommendation for TRAJ_MIN_EDGE_F
  - Optional CSV dump for further analysis
"""

from __future__ import annotations

import argparse
import asyncio
import math
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import aiohttp

import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from kalshi_bot.news.noaa import CITIES

# ICAO weather station IDs for METAR fetching (index 4 in the old local CITIES dict).
CITY_STATION: dict[str, str] = {
    "temp_high_lax": "KLAX",
    "temp_high_den": "KDEN",
    "temp_high_chi": "KMDW",
    "temp_high_ny":  "KNYC",
    "temp_high_mia": "KMIA",
    "temp_high_aus": "KAUS",
    "temp_high_bos": "KBOS",
    "temp_high_hou": "KHOU",
    "temp_high_dfw": "KDFW",
    "temp_high_sfo": "KSFO",
    "temp_high_sea": "KSEA",
    "temp_high_phx": "KPHX",
    "temp_high_phl": "KPHL",
    "temp_high_atl": "KATL",
    "temp_high_msp": "KMSP",
    "temp_high_dca": "KDCA",
    "temp_high_las": "KLAS",
    "temp_high_okc": "KOKC",
    "temp_high_sat": "KSAT",
    "temp_high_msy": "KMSY",
}

# Per-city diurnal parameters (used by the trajectory model)
# dawn_hour: local hour when daily min typically occurs
# peak_hour: local hour when daily max typically occurs
DIURNAL_PARAMS: dict[str, tuple[int, int]] = {
    "temp_high_mia": (7, 14),
    "temp_high_lax": (7, 15),
    "temp_high_phx": (6, 15),
    # All others: dawn=6, peak=16 (default)
}

_METAR_URL = "https://aviationweather.gov/api/data/metar"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _c_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def _linreg_slope(xs: list[float], ys: list[float]) -> float | None:
    """Return slope (y per x unit) via ordinary least squares. None if degenerate."""
    n = len(xs)
    if n < 2:
        return None
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    num = sum((xs[i] - x_mean) * (ys[i] - y_mean) for i in range(n))
    den = sum((xs[i] - x_mean) ** 2 for i in range(n))
    if den < 1e-9:
        return None
    return num / den


def parabolic_projected_peak(
    current_temp: float,
    slope_fph: float,     # °F per hour (positive = warming)
    local_hour_frac: float,
    dawn_hour: int,
    peak_hour: int,
) -> float | None:
    """Project daily peak using the parabolic diurnal model.

    Model: T(t) = T_min + A*(2u - u²) where u = phase ∈ [0,1].
    dT/dt = A * 2(1-u) / (t_peak - t_dawn)

    Remaining rise: ΔT = slope * hours_to_peak * (1 - u) / 2

    Returns None if already past the peak or warming_duration is zero.
    """
    warming_duration = peak_hour - dawn_hour
    if warming_duration <= 0:
        return None
    hours_to_peak = peak_hour - local_hour_frac
    if hours_to_peak <= 0:
        return None
    u_now = max(0.0, min(1.0, (local_hour_frac - dawn_hour) / warming_duration))
    tod_factor = (1.0 - u_now) / 2.0
    projected_rise = slope_fph * hours_to_peak * tod_factor
    return current_temp + projected_rise


# ---------------------------------------------------------------------------
# METAR fetch
# ---------------------------------------------------------------------------

async def fetch_metar_series(
    session: aiohttp.ClientSession,
    station_ids: list[str],
    hours: int,
) -> dict[str, list[tuple[float, float]]]:
    """Fetch METAR observations for the given stations.

    Returns: station_id → sorted list of (epoch_timestamp, temp_f)
    """
    params = {
        "ids":    ",".join(station_ids),
        "format": "json",
        "hours":  str(hours),
    }
    try:
        async with session.get(
            _METAR_URL,
            params=params,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            resp.raise_for_status()
            records: list[dict] = await resp.json()
    except Exception as exc:
        print(f"  METAR fetch error: {exc}", file=sys.stderr)
        return {}

    by_station: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for rec in records:
        station = rec.get("icaoId", "")
        temp_c = rec.get("temp")
        obs_time = rec.get("obsTime")
        if not station or temp_c is None or obs_time is None:
            continue
        by_station[station].append((float(obs_time), _c_to_f(float(temp_c))))

    # Sort each series by time ascending
    for sid in by_station:
        by_station[sid].sort(key=lambda x: x[0])

    return by_station


# ---------------------------------------------------------------------------
# Per-day simulation
# ---------------------------------------------------------------------------

def simulate_day(
    obs_series_today: list[tuple[float, float]],  # all today's (epoch, temp_f)
    city_tz: ZoneInfo,
    dawn_hour: int,
    peak_hour: int,
    lookback_hours: float = 2.0,
    min_obs: int = 3,
    min_slope_fph: float = 0.3,
    eval_hours: list[int] | None = None,
) -> list[dict]:
    """Simulate the trajectory model for one city-day.

    At each evaluation hour during the warming window, compute the slope
    from the prior `lookback_hours` of observations and project the peak.
    Return a list of result dicts — one per evaluation hour.
    """
    if eval_hours is None:
        eval_hours = list(range(10, 17))  # 10 AM to 4 PM local

    if not obs_series_today:
        return []

    # Actual daily max (ground truth)
    actual_max = max(temp for _, temp in obs_series_today)

    results = []
    for eval_local_h in eval_hours:
        # Find the approximate epoch for eval_local_h on this day
        # Use the first observation's epoch to anchor the local date
        first_epoch = obs_series_today[0][0]
        first_dt = datetime.fromtimestamp(first_epoch, tz=timezone.utc).astimezone(city_tz)
        eval_dt = first_dt.replace(hour=eval_local_h, minute=0, second=0, microsecond=0)
        eval_epoch = eval_dt.timestamp()

        # Only use observations up to and including eval_epoch
        available = [(t, temp) for t, temp in obs_series_today if t <= eval_epoch]
        if not available:
            continue

        current_temp = available[-1][1]
        current_epoch = available[-1][0]

        # Lookback window
        cutoff = eval_epoch - lookback_hours * 3600
        recent = [(t, temp) for t, temp in available if t >= cutoff]

        if len(recent) < min_obs:
            results.append({
                "eval_hour":    eval_local_h,
                "skip_reason":  f"insufficient_obs ({len(recent)} < {min_obs})",
                "actual_max":   actual_max,
                "current_temp": current_temp,
            })
            continue

        # Compute slope via linear regression (seconds → °F/hr conversion)
        ts_sec = [t for t, _ in recent]
        temps   = [temp for _, temp in recent]
        slope_per_sec = _linreg_slope(ts_sec, temps)
        if slope_per_sec is None:
            continue
        slope_fph = slope_per_sec * 3600

        if slope_fph <= min_slope_fph:
            results.append({
                "eval_hour":    eval_local_h,
                "skip_reason":  f"not_warming (slope={slope_fph:.2f}°F/h)",
                "actual_max":   actual_max,
                "current_temp": current_temp,
                "slope_fph":    slope_fph,
            })
            continue

        local_hour_frac = eval_local_h  # exactly on the hour for simplicity

        proj = parabolic_projected_peak(
            current_temp, slope_fph, local_hour_frac, dawn_hour, peak_hour
        )
        if proj is None:
            continue

        # Also compute naive linear projection for comparison
        hours_to_peak = max(0.0, peak_hour - local_hour_frac)
        naive_proj = current_temp + slope_fph * hours_to_peak

        error_parabolic = proj - actual_max
        error_naive     = naive_proj - actual_max

        results.append({
            "eval_hour":        eval_local_h,
            "actual_max":       actual_max,
            "current_temp":     current_temp,
            "slope_fph":        round(slope_fph, 2),
            "u_now":            round(max(0.0, min(1.0, (local_hour_frac - dawn_hour) / (peak_hour - dawn_hour))), 3),
            "tod_factor":       round((1.0 - max(0.0, min(1.0, (local_hour_frac - dawn_hour) / (peak_hour - dawn_hour)))) / 2.0, 3),
            "projected_parabolic": round(proj, 2),
            "projected_naive":     round(naive_proj, 2),
            "error_parabolic":  round(error_parabolic, 2),
            "error_naive":      round(error_naive, 2),
            "obs_in_window":    len(recent),
        })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_backtest(
    days: int,
    city_filter: str | None,
    min_obs: int,
    lookback_h: float,
    min_slope_fph: float,
    start_hour: int = 10,
    end_hour: int = 16,
    min_edge_filter: float = 0.0,
    csv_out: str | None = None,
) -> None:
    hours_to_fetch = days * 24 + 6  # extra buffer for timezone offsets

    metrics = list(CITIES.keys())
    if city_filter and city_filter != "all":
        tag = f"temp_high_{city_filter}"
        metrics = [m for m in metrics if m == tag]
        if not metrics:
            print(f"Unknown city filter: {city_filter}. Use one of: {', '.join(k.replace('temp_high_', '') for k in CITIES)}")
            return

    station_ids = [CITY_STATION[m] for m in metrics if m in CITY_STATION]

    print(f"\nFetching {hours_to_fetch}h of METAR observations for {len(metrics)} cities...")
    async with aiohttp.ClientSession() as session:
        raw = await fetch_metar_series(session, station_ids, hours_to_fetch)

    if not raw:
        print("No data returned from METAR API.")
        return

    # Map station → metric
    station_to_metric = {CITY_STATION[m]: m for m in metrics if m in CITY_STATION}

    # Aggregate results
    # Structure: metric → list of result dicts
    all_results: list[dict] = []
    days_per_metric: dict[str, int] = {}

    for station_id, series in raw.items():
        metric = station_to_metric.get(station_id)
        if metric is None:
            continue

        city_name, _, _, city_tz = CITIES[metric]
        dawn_hour, peak_hour = DIURNAL_PARAMS.get(metric, (6, 16))

        # Split series into per-day buckets (local date)
        by_day: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for epoch, temp_f in series:
            local_date = datetime.fromtimestamp(epoch, tz=timezone.utc).astimezone(city_tz).strftime("%Y-%m-%d")
            by_day[local_date].append((epoch, temp_f))

        days_per_metric[metric] = len(by_day)

        for date_str, day_series in sorted(by_day.items()):
            day_results = simulate_day(
                day_series,
                city_tz,
                dawn_hour,
                peak_hour,
                lookback_hours=lookback_h,
                min_obs=min_obs,
                min_slope_fph=min_slope_fph,
                eval_hours=list(range(start_hour, end_hour)),
            )
            for r in day_results:
                r["metric"] = metric
                r["city"]   = city_name
                r["date"]   = date_str
            all_results.extend(day_results)

    # Filter to rows with projections
    projected = [r for r in all_results if "error_parabolic" in r]
    # Apply min_edge_filter: only keep rows where the projected peak is at
    # least min_edge_filter°F above the current temp (simulates the
    # TRAJ_MIN_EDGE_F gate that decides whether to surface a signal at all).
    if min_edge_filter > 0:
        before = len(projected)
        projected = [
            r for r in projected
            if (r["projected_parabolic"] - r["current_temp"]) >= min_edge_filter
        ]
        print(f"\n  min_edge_filter={min_edge_filter}°F applied: {before} → {len(projected)} rows")
    skipped = [r for r in all_results if "skip_reason" in r]

    print(f"\nTotal evaluation points (hours {start_hour}–{end_hour-1} local): {len(all_results)}")
    print(f"  Projected (slope > {min_slope_fph}°F/h, obs ≥ {min_obs}): {len(projected)}")
    print(f"  Skipped (not warming or insufficient obs): {len(skipped)}")

    if not projected:
        print("\nNo projected results — try reducing --min-slope or --min-obs.")
        return

    # ---------------------------------------------------------------------------
    # Summary statistics
    # ---------------------------------------------------------------------------

    print("\n" + "=" * 72)
    print("OVERALL MODEL ACCURACY")
    print("=" * 72)

    para_errors = [r["error_parabolic"] for r in projected]
    naive_errors = [r["error_naive"] for r in projected]
    n = len(para_errors)

    def stats(errs: list[float]) -> dict:
        errs_s = sorted(errs)
        mean = sum(errs) / len(errs)
        variance = sum((e - mean) ** 2 for e in errs) / len(errs)
        std = math.sqrt(variance)
        abs_errs = [abs(e) for e in errs]
        mae = sum(abs_errs) / len(abs_errs)
        p50 = errs_s[len(errs_s) // 2]
        p75 = errs_s[int(len(errs_s) * 0.75)]
        p90 = errs_s[int(len(errs_s) * 0.90)]
        p10 = errs_s[int(len(errs_s) * 0.10)]
        over = sum(1 for e in errs if e > 0) / len(errs)
        return dict(mean=mean, std=std, mae=mae, p10=p10, p50=p50, p75=p75, p90=p90, over_pct=over)

    ps = stats(para_errors)
    ns = stats(naive_errors)

    print(f"\n{'':30s} {'Parabolic':>12s} {'Naive linear':>12s}")
    print(f"  {'n samples':30s} {n:>12d} {n:>12d}")
    print(f"  {'Mean error (°F)':30s} {ps['mean']:>12.2f} {ns['mean']:>12.2f}")
    print(f"  {'Std dev (°F)':30s} {ps['std']:>12.2f} {ns['std']:>12.2f}")
    print(f"  {'MAE (°F)':30s} {ps['mae']:>12.2f} {ns['mae']:>12.2f}")
    print(f"  {'P10 error (°F)':30s} {ps['p10']:>12.2f} {ns['p10']:>12.2f}")
    print(f"  {'P50 error / median (°F)':30s} {ps['p50']:>12.2f} {ns['p50']:>12.2f}")
    print(f"  {'P75 error (°F)':30s} {ps['p75']:>12.2f} {ns['p75']:>12.2f}")
    print(f"  {'P90 error (°F)':30s} {ps['p90']:>12.2f} {ns['p90']:>12.2f}")
    print(f"  {'% overestimates':30s} {ps['over_pct']:>12.1%} {ns['over_pct']:>12.1%}")

    # ---------------------------------------------------------------------------
    # Accuracy by hour of day
    # ---------------------------------------------------------------------------

    print("\n" + "=" * 72)
    print("ACCURACY BY EVALUATION HOUR  (parabolic model)")
    print("=" * 72)
    print(f"  {'Hour':>6s} {'n':>5s} {'Mean err':>10s} {'MAE':>8s} {'Std':>8s} {'% over':>8s}  Note")
    print("  " + "-" * 60)

    by_hour: dict[int, list[float]] = defaultdict(list)
    for r in projected:
        by_hour[r["eval_hour"]].append(r["error_parabolic"])

    for h in sorted(by_hour):
        errs = by_hour[h]
        s = stats(errs)
        note = ""
        if s["mae"] < 2.0:
            note = "✓ reliable"
        elif s["mae"] < 3.5:
            note = "~ marginal"
        else:
            note = "✗ unreliable"
        print(f"  {h:>4}h  {len(errs):>5d}  {s['mean']:>+8.2f}°F  {s['mae']:>6.2f}°F  {s['std']:>6.2f}°F  {s['over_pct']:>7.1%}  {note}")

    # ---------------------------------------------------------------------------
    # Accuracy by city
    # ---------------------------------------------------------------------------

    print("\n" + "=" * 72)
    print("ACCURACY BY CITY  (parabolic model, all hours combined)")
    print("=" * 72)
    print(f"  {'City':20s} {'n':>5s} {'Mean err':>10s} {'MAE':>8s} {'Std':>8s} {'% over':>8s}")
    print("  " + "-" * 65)

    by_city: dict[str, list[float]] = defaultdict(list)
    for r in projected:
        by_city[r["city"]].append(r["error_parabolic"])

    for city in sorted(by_city, key=lambda c: sum(abs(e) for e in by_city[c]) / len(by_city[c])):
        errs = by_city[city]
        s = stats(errs)
        print(f"  {city:20s} {len(errs):>5d}  {s['mean']:>+8.2f}°F  {s['mae']:>6.2f}°F  {s['std']:>6.2f}°F  {s['over_pct']:>7.1%}")

    # ---------------------------------------------------------------------------
    # Parabolic vs naive improvement
    # ---------------------------------------------------------------------------

    print("\n" + "=" * 72)
    print("PARABOLIC vs NAIVE: error comparison by hour")
    print("=" * 72)
    print(f"  {'Hour':>6s}  {'Para MAE':>10s}  {'Naive MAE':>10s}  {'Para wins?':>12s}")
    print("  " + "-" * 50)

    by_hour_naive: dict[int, list[float]] = defaultdict(list)
    for r in projected:
        by_hour_naive[r["eval_hour"]].append(r["error_naive"])

    for h in sorted(by_hour):
        p_mae = sum(abs(e) for e in by_hour[h]) / len(by_hour[h])
        n_mae = sum(abs(e) for e in by_hour_naive.get(h, [1e9])) / max(1, len(by_hour_naive.get(h, [1])))
        winner = "✓ parabolic" if p_mae < n_mae else "✗ naive better"
        print(f"  {h:>4}h  {p_mae:>10.2f}°F  {n_mae:>10.2f}°F  {winner}")

    # ---------------------------------------------------------------------------
    # Recommended TRAJ_MIN_EDGE_F calibration
    # ---------------------------------------------------------------------------

    print("\n" + "=" * 72)
    print("CALIBRATION RECOMMENDATION")
    print("=" * 72)

    reliable_hours = [h for h, errs in by_hour.items()
                      if (sum(abs(e) for e in errs) / len(errs)) < 2.5]
    unreliable_hours = [h for h in sorted(by_hour) if h not in reliable_hours]

    overall_mae = ps["mae"]
    # Recommended min edge: 2× MAE gives ~P75 confidence the projection is correct
    recommended_edge = round(overall_mae * 2, 1)

    print(f"\n  Overall MAE (parabolic): {overall_mae:.2f}°F")
    print(f"  Recommended TRAJ_MIN_EDGE_F: {recommended_edge}°F  (2× MAE)")
    print(f"  Reliable hours (MAE < 2.5°F): {reliable_hours}")
    print(f"  Unreliable hours:             {unreliable_hours}")
    if unreliable_hours:
        safest_end = min(unreliable_hours)
        print(f"  Recommended TRAJ_END_LOCAL_HOUR: {safest_end}  (cut off before unreliable zone)")

    # ---------------------------------------------------------------------------
    # Optional CSV dump
    # ---------------------------------------------------------------------------

    if csv_out:
        import csv
        fieldnames = [
            "date", "city", "metric", "eval_hour",
            "current_temp", "actual_max", "slope_fph",
            "u_now", "tod_factor",
            "projected_parabolic", "projected_naive",
            "error_parabolic", "error_naive",
            "obs_in_window",
        ]
        with open(csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(projected)
        print(f"\n  Detailed results written to: {csv_out}")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest the parabolic diurnal temperature trajectory model."
    )
    parser.add_argument("--days",       type=int,   default=7,
                        help="Days of METAR history to fetch (default 7; API max ~14)")
    parser.add_argument("--city",       type=str,   default="all",
                        help="City code to filter (e.g. chi, ny) or 'all'")
    parser.add_argument("--min-obs",    type=int,   default=3,
                        help="Min observations in lookback window (default 3)")
    parser.add_argument("--lookback",   type=float, default=2.0,
                        help="Lookback window in hours for slope calculation (default 2.0)")
    parser.add_argument("--min-slope",  type=float, default=0.3,
                        help="Min warming slope in °F/h to generate projection (default 0.3)")
    parser.add_argument("--start-hour", type=int,   default=10,
                        help="First local hour to evaluate (default 10)")
    parser.add_argument("--end-hour",   type=int,   default=16,
                        help="Last local hour to evaluate, exclusive (default 16)")
    parser.add_argument("--min-edge",   type=float, default=0.0,
                        help="Only include projected points where |projected_peak - actual_max| "
                             "would justify a trade (simulates TRAJ_MIN_EDGE_F gate). Default 0 = off.")
    parser.add_argument("--csv",        type=str,   default=None,
                        help="Optional path to write detailed CSV results")
    args = parser.parse_args()

    asyncio.run(run_backtest(
        days=args.days,
        city_filter=args.city if args.city != "all" else None,
        min_obs=args.min_obs,
        lookback_h=args.lookback,
        min_slope_fph=args.min_slope,
        start_hour=args.start_hour,
        end_hour=args.end_hour,
        min_edge_filter=args.min_edge,
        csv_out=args.csv,
    ))


if __name__ == "__main__":
    main()
