"""Compute the distribution of (entry_temp - true_daily_min) by time-of-day.

For KXLOWT* NO signals, the bot uses a running observed_min as a proxy for the
true daily minimum.  The "gap" = current_metar_temp - running_min is proposed
as a gate: only enter if the current temperature is sufficiently above the
running minimum, proving the minimum was captured hours ago and not just "when
the bot first started watching this market."

This script answers: what gap threshold G has real statistical backing?

Method
------
For each historical calendar day and city:
  1. Fetch the full day of METAR readings (midnight to midnight local).
  2. At each simulated entry time T (every 2 hours from 18:00 to 08:00 next day),
     look up the temperature closest to T.
  3. Compute true_gap = temp_at_T - actual_daily_minimum.
     This is the gap the bot WOULD observe if it had been running continuously
     since before the true minimum occurred.
  4. Aggregate by (city, month, entry_time) → distribution of true_gap.

Then for each gap threshold candidate (5, 8, 10, 12, 15°F) compute:
  - Fraction of post-minimum entries (afternoon/evening) passing the gate.
  - Fraction of pre-minimum entries (overnight window) passing the gate.

The threshold that maximally separates these two populations is the right G.

Output
------
  data/overnight_gap_analysis.csv  — per (city, month, entry_time) percentiles
  data/overnight_gap_threshold.txt — recommended thresholds with statistical basis

Usage
-----
  venv/bin/python scripts/backtest_overnight_gap_threshold.py
  venv/bin/python scripts/backtest_overnight_gap_threshold.py --years 3 --cities chi satx okc
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import math
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).parent.parent))

from kalshi_bot.news.noaa import KALSHI_STATION_IDS, LOW_CITIES  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_MESONET_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
_FETCH_DELAY = 0.5
_MIN_OBS_PER_DAY = 8

# Simulated entry times (local hour, minute).  We cover the "evening before"
# through the "morning after" — the full window where a bot might enter a
# KXLOWT trade.  Pre-minimum entries: 18:00–04:00.  Post-minimum: 06:00–14:00.
_ENTRY_HOURS = [18, 20, 22, 0, 2, 4, 6, 8, 10, 12]

# Gap threshold candidates to evaluate
_GAP_THRESHOLDS = [3, 5, 8, 10, 12, 15]


def _low_station(metric: str) -> str | None:
    return KALSHI_STATION_IDS.get(metric.replace("temp_low_", "temp_high_"))


def _percentile(vals: list[float], pct: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    idx = min(int(math.ceil(pct / 100 * len(s))) - 1, len(s) - 1)
    return s[max(idx, 0)]


async def _fetch_city_obs(
    session: aiohttp.ClientSession,
    metric: str,
    station: str,
    start_dt: date,
    end_dt: date,
) -> dict[int, list[tuple[int, float]]]:
    """Fetch ASOS observations and return ordinal → sorted list of (local_minutes, temp_f)."""
    _, _, _, city_tz = LOW_CITIES[metric]

    # Fetch one extra day on each side so overnight windows don't miss readings
    fetch_start = start_dt - timedelta(days=1)
    fetch_end   = end_dt   + timedelta(days=1)

    params = {
        "station": station, "data": "tmpf",
        "year1": str(fetch_start.year), "month1": str(fetch_start.month),
        "day1":  str(fetch_start.day),
        "year2": str(fetch_end.year),   "month2": str(fetch_end.month),
        "day2":  str(fetch_end.day),
        "tz": "UTC", "format": "comma", "latlon": "no",
        "missing": "M", "trace": "T", "direct": "no",
        "report_type": "3,4",
    }
    try:
        async with session.get(
            _MESONET_URL, params=params,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            resp.raise_for_status()
            text = await resp.text()
    except Exception as exc:
        log.error("Fetch failed for %s: %s", metric, exc)
        return {}

    obs_by_ordinal: dict[int, list[tuple[int, float]]] = {}
    for line in text.splitlines():
        if line.startswith("#") or line.startswith("station") or not line.strip():
            continue
        parts = line.split(",")
        if len(parts) < 3:
            continue
        try:
            utc_ts = datetime.strptime(parts[1].strip(), "%Y-%m-%d %H:%M").replace(
                tzinfo=timezone.utc
            )
            temp_str = parts[2].strip()
            if temp_str in ("M", "T", ""):
                continue
            temp_f = float(temp_str)
        except (ValueError, IndexError):
            continue
        local_ts = utc_ts.astimezone(city_tz)
        ordinal = local_ts.toordinal()
        local_min = local_ts.hour * 60 + local_ts.minute
        obs_by_ordinal.setdefault(ordinal, []).append((local_min, temp_f))

    # Sort each day's readings by time
    for ordinal in obs_by_ordinal:
        obs_by_ordinal[ordinal].sort()

    return obs_by_ordinal


def _nearest_temp(readings: list[tuple[int, float]], target_min: int) -> float | None:
    """Return the temperature of the observation closest to target_min."""
    if not readings:
        return None
    best = min(readings, key=lambda r: abs(r[0] - target_min))
    if abs(best[0] - target_min) > 90:   # more than 90 min away → no valid reading
        return None
    return best[1]


def _analyze_city(
    metric: str,
    obs_by_ordinal: dict[int, list[tuple[int, float]]],
    start_dt: date,
    end_dt: date,
) -> dict[tuple[int, int], list[float]]:
    """Return {(month, entry_hour) → [true_gap values]} across all valid days."""
    _, _, _, city_tz = LOW_CITIES[metric]
    results: dict[tuple[int, int], list[float]] = {}

    start_ord = date(*start_dt.timetuple()[:3]).toordinal()
    end_ord   = date(*end_dt.timetuple()[:3]).toordinal()

    for ordinal in range(start_ord, end_ord + 1):
        readings = obs_by_ordinal.get(ordinal, [])
        if len(readings) < _MIN_OBS_PER_DAY:
            continue

        true_min = min(t for _, t in readings)
        month = datetime.fromordinal(ordinal).month

        for entry_hour in _ENTRY_HOURS:
            # Entry times ≥ 18:00 are on the same calendar day.
            # Entry times < 18:00 (0–14) are on the NEXT calendar day's readings.
            if entry_hour >= 18:
                # Evening of calendar day `ordinal`
                target_min = entry_hour * 60
                entry_readings = readings
            else:
                # Morning of the NEXT calendar day
                next_readings = obs_by_ordinal.get(ordinal + 1, [])
                target_min = entry_hour * 60
                entry_readings = next_readings

            temp = _nearest_temp(entry_readings, target_min)
            if temp is None:
                continue

            true_gap = temp - true_min
            key = (month, entry_hour)
            results.setdefault(key, []).append(true_gap)

    return results


async def main(
    years: int,
    city_filter: set[str] | None,
    out_path: Path,
) -> None:
    end_dt = date.today()
    start_dt = date(end_dt.year - years, end_dt.month, end_dt.day)

    pairs: list[tuple[str, str]] = []
    for metric in LOW_CITIES:
        suffix = metric.replace("temp_low_", "")
        if city_filter and suffix not in city_filter:
            continue
        station = _low_station(metric)
        if station is None:
            continue
        pairs.append((metric, station))

    log.info("Fetching %d cities (%d years)", len(pairs), years)

    # Aggregate true_gap lists across ALL cities so the threshold recommendation
    # is global.  Also keep per-city breakdowns for the CSV.
    global_gaps: dict[tuple[int, int], list[float]] = {}   # (month, hour) → gaps
    city_gaps:   dict[str, dict[tuple[int, int], list[float]]] = {}

    async with aiohttp.ClientSession() as session:
        for i, (metric, station) in enumerate(pairs):
            if i > 0:
                await asyncio.sleep(_FETCH_DELAY)
            log.info("[%d/%d] %s", i + 1, len(pairs), metric)
            obs = await _fetch_city_obs(session, metric, station, start_dt, end_dt)
            if not obs:
                continue
            city_result = _analyze_city(metric, obs, start_dt, end_dt)
            city_gaps[metric] = city_result
            for key, vals in city_result.items():
                global_gaps.setdefault(key, []).extend(vals)

    # --- Write per-city CSV ---
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "city_key", "city_name", "month", "entry_hour",
        "n_days", "mean_gap", "std_gap",
        "p10", "p25", "p50", "p75", "p90",
    ] + [f"pct_above_{g}f" for g in _GAP_THRESHOLDS]

    rows: list[dict] = []
    for metric in sorted(city_gaps):
        city_name = LOW_CITIES[metric][0]
        for (month, hour), vals in sorted(city_gaps[metric].items()):
            if not vals:
                continue
            n = len(vals)
            mean = sum(vals) / n
            std = math.sqrt(sum((v - mean) ** 2 for v in vals) / max(n - 1, 1))
            row = {
                "city_key":   metric,
                "city_name":  city_name,
                "month":      month,
                "entry_hour": f"{hour:02d}:00",
                "n_days":     n,
                "mean_gap":   f"{mean:.1f}",
                "std_gap":    f"{std:.1f}",
                "p10":        f"{_percentile(vals, 10):.1f}",
                "p25":        f"{_percentile(vals, 25):.1f}",
                "p50":        f"{_percentile(vals, 50):.1f}",
                "p75":        f"{_percentile(vals, 75):.1f}",
                "p90":        f"{_percentile(vals, 90):.1f}",
            }
            for g in _GAP_THRESHOLDS:
                pct = 100 * sum(1 for v in vals if v >= g) / n
                row[f"pct_above_{g}f"] = f"{pct:.1f}"
            rows.append(row)

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote %d rows to %s", len(rows), out_path)

    # --- Print threshold summary (global, all cities combined) ---
    print("\n" + "=" * 72)
    print("GLOBAL: true_gap distribution by simulated entry time")
    print("(true_gap = entry_temp - actual_daily_min, assuming perfect monitoring)")
    print("=" * 72)

    # Group entry hours into "pre-minimum" and "post-minimum" buckets
    pre_min_hours  = [22, 0, 2, 4]    # overnight — minimum likely hasn't occurred
    post_min_hours = [8, 10, 12]      # morning/afternoon — minimum likely has occurred
    evening_hours  = [18, 20]         # evening — minimum likely occurred hours ago

    hdr = f"{'Entry':>7}  {'n':>5}  {'p10':>5}  {'p25':>5}  {'p50':>5}  {'p75':>5}  {'p90':>5}"
    for g in _GAP_THRESHOLDS:
        hdr += f"  {f'>={g}°':>7}"
    print(hdr)
    print("-" * len(hdr))

    for hour in _ENTRY_HOURS:
        # Merge across months for the global view
        all_vals: list[float] = []
        for month in range(1, 13):
            all_vals.extend(global_gaps.get((month, hour), []))
        if not all_vals:
            continue
        n = len(all_vals)
        label = f"{hour:02d}:00"
        row_str = f"{label:>7}  {n:>5}  "
        row_str += f"{_percentile(all_vals, 10):>5.1f}  "
        row_str += f"{_percentile(all_vals, 25):>5.1f}  "
        row_str += f"{_percentile(all_vals, 50):>5.1f}  "
        row_str += f"{_percentile(all_vals, 75):>5.1f}  "
        row_str += f"{_percentile(all_vals, 90):>5.1f}  "
        for g in _GAP_THRESHOLDS:
            pct = 100 * sum(1 for v in all_vals if v >= g) / n
            row_str += f"  {pct:>6.1f}%"
        print(row_str)

    print()
    print("Threshold recommendations (% of post-min entries allowed vs % of pre-min blocked):")
    print(f"  {'Gap':>6}  {'Post-min (08-12) allowed':>24}  {'Pre-min (22-04) blocked':>24}")
    print("-" * 62)

    def _pct_above(hours: list[int], threshold: float) -> float:
        vals: list[float] = []
        for h in hours:
            for month in range(1, 13):
                vals.extend(global_gaps.get((month, h), []))
        if not vals:
            return float("nan")
        return 100 * sum(1 for v in vals if v >= threshold) / len(vals)

    for g in _GAP_THRESHOLDS:
        allowed = _pct_above(post_min_hours, g)
        blocked = 100 - _pct_above(pre_min_hours, g)
        print(f"  {g:>4}°F  {allowed:>23.1f}%  {blocked:>23.1f}%")

    # Also write the threshold summary to a text file
    summary_path = out_path.parent / "overnight_gap_threshold.txt"
    with summary_path.open("w") as f:
        f.write("KXLOWT* NO-signal gap threshold analysis\n")
        f.write(f"Data: {years} years, {len(pairs)} cities\n\n")
        f.write("Gap = current_metar_temp - true_daily_minimum\n")
        f.write("Pre-min entries: 22:00-04:00 local (minimum hasn't typically occurred yet)\n")
        f.write("Post-min entries: 08:00-12:00 local (minimum has typically already occurred)\n\n")
        f.write(f"{'Gap':>6}  {'Post-min allowed':>17}  {'Pre-min blocked':>17}\n")
        f.write("-" * 46 + "\n")
        for g in _GAP_THRESHOLDS:
            allowed = _pct_above(post_min_hours, g)
            blocked = 100 - _pct_above(pre_min_hours, g)
            f.write(f"  {g:>3}°F  {allowed:>16.1f}%  {blocked:>16.1f}%\n")
    log.info("Wrote threshold summary to %s", summary_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute true_gap distribution to back the KXLOWT NO-signal gap threshold."
    )
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--cities", nargs="+", default=None)
    parser.add_argument("--out", default="data/overnight_gap_analysis.csv")
    args = parser.parse_args()

    asyncio.run(main(
        years=args.years,
        city_filter=set(args.cities) if args.cities else None,
        out_path=Path(args.out),
    ))
