"""Compute when daily high temperatures typically occur, by city and month.

For each Kalshi temperature city, fetches historical ASOS observations from the
Iowa State Environmental Mesonet and finds the local time of each day's maximum
temperature. Aggregates by (city, month) to produce percentile distributions at
minute-level precision.

Outputs:
  data/peak_hour_analysis.csv  — full percentile table (HH:MM, std in minutes)
  data/peak_hour_p90.py        — Python dict literal ready to import into the bot

The p90 threshold means: on 90% of historical days, the daily high had already
occurred by this local time. This replaces the hardcoded 4:30 PM peak_past gate
with city- and season-aware thresholds.

Data source: Iowa State Environmental Mesonet (no auth required)
  https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py

Usage:
  venv/bin/python scripts/backtest_peak_hour.py
  venv/bin/python scripts/backtest_peak_hour.py --years 3
  venv/bin/python scripts/backtest_peak_hour.py --years 5 --cities chi bos ny
  venv/bin/python scripts/backtest_peak_hour.py --years 5 --dict
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

from kalshi_bot.news.noaa import CITIES, KALSHI_STATION_IDS  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_MESONET_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
# Sequential fetches — IEM is a shared academic resource; be polite
_FETCH_DELAY = 0.5  # seconds between city requests
# Skip local dates with fewer than this many valid observations (data outage)
_MIN_OBS_PER_DAY = 8


def _minutes_to_hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _percentile(sorted_vals: list[int], pct: float) -> int:
    """Return the pct-th percentile of a sorted list (nearest-rank method)."""
    if not sorted_vals:
        return 0
    idx = min(int(math.ceil(pct / 100 * len(sorted_vals))) - 1, len(sorted_vals) - 1)
    return sorted_vals[max(idx, 0)]


async def _fetch_city_obs(
    session: aiohttp.ClientSession,
    metric: str,
    station: str,
    start_dt: date,
    end_dt: date,
) -> dict[tuple[int, int], int]:
    """Fetch all obs for one station and return (month, date_ordinal) → peak_minutes.

    peak_minutes is minutes since local midnight of the earliest observation
    that tied for the daily maximum temperature.
    Days with fewer than _MIN_OBS_PER_DAY valid readings are excluded.
    """
    city_entry = CITIES.get(metric)
    if city_entry is None:
        log.warning("No CITIES entry for %s — skipping", metric)
        return {}
    _, _, _, city_tz = city_entry

    params = {
        "station":     station,
        "data":        "tmpf",
        "year1":       str(start_dt.year),
        "month1":      str(start_dt.month),
        "day1":        str(start_dt.day),
        "year2":       str(end_dt.year),
        "month2":      str(end_dt.month),
        "day2":        str(end_dt.day),
        "tz":          "UTC",
        "format":      "comma",
        "latlon":      "no",
        "missing":     "M",
        "trace":       "T",
        "direct":      "no",
        "report_type": "3,4",   # routine METAR + SPECI (special obs at any minute)
    }

    try:
        async with session.get(
            _MESONET_URL,
            params=params,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            resp.raise_for_status()
            text = await resp.text()
    except Exception as exc:
        log.error("Mesonet fetch failed for %s (%s): %s", metric, station, exc)
        return {}

    # Parse CSV: station,valid,tmpf
    # valid is a UTC timestamp "YYYY-MM-DD HH:MM"
    # Group observations by (local_date_ordinal) → list of (local_minutes, temp_f)
    obs_by_date: dict[int, list[tuple[int, float]]] = {}
    month_by_ordinal: dict[int, int] = {}

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
        local_minutes = local_ts.hour * 60 + local_ts.minute
        obs_by_date.setdefault(ordinal, []).append((local_minutes, temp_f))
        month_by_ordinal[ordinal] = local_ts.month

    # For each local date, find the time of the daily maximum
    peak_by_date: dict[tuple[int, int], int] = {}
    for ordinal, readings in obs_by_date.items():
        if len(readings) < _MIN_OBS_PER_DAY:
            continue
        max_temp = max(t for _, t in readings)
        # Earliest reading tied at the max (conservative — "when did peak first occur")
        peak_minutes = min(m for m, t in readings if t == max_temp)
        month = month_by_ordinal[ordinal]
        peak_by_date[(month, ordinal)] = peak_minutes

    valid_days = len(peak_by_date)
    log.info("  %s (%s): %d valid days", metric, station, valid_days)
    return peak_by_date


def _aggregate(peak_by_date: dict[tuple[int, int], int]) -> dict[int, dict]:
    """Aggregate (month, ordinal) → peak_minutes into per-month statistics."""
    by_month: dict[int, list[int]] = {}
    for (month, _), minutes in peak_by_date.items():
        by_month.setdefault(month, []).append(minutes)

    result: dict[int, dict] = {}
    for month, minutes_list in by_month.items():
        s = sorted(minutes_list)
        n = len(s)
        mean_min = sum(s) / n
        variance = sum((x - mean_min) ** 2 for x in s) / max(n - 1, 1)
        std_min = math.sqrt(variance)
        result[month] = {
            "n": n,
            "mean": mean_min,
            "std": std_min,
            "p50": _percentile(s, 50),
            "p75": _percentile(s, 75),
            "p90": _percentile(s, 90),
            "p95": _percentile(s, 95),
        }
    return result


async def main(
    years: int,
    city_filter: set[str] | None,
    out_path: Path,
    emit_dict: bool,
) -> None:
    end_dt = date.today()
    start_dt = date(end_dt.year - years, end_dt.month, end_dt.day)

    pairs: list[tuple[str, str]] = []
    for metric, station in KALSHI_STATION_IDS.items():
        if city_filter:
            suffix = metric.replace("temp_high_", "")
            if suffix not in city_filter:
                continue
        pairs.append((metric, station))

    log.info(
        "Fetching %d cities from %s to %s (%d years)",
        len(pairs), start_dt, end_dt, years,
    )

    all_stats: dict[str, dict[int, dict]] = {}

    async with aiohttp.ClientSession() as session:
        for i, (metric, station) in enumerate(pairs):
            if i > 0:
                await asyncio.sleep(_FETCH_DELAY)
            log.info("[%d/%d] %s (%s)", i + 1, len(pairs), metric, station)
            peak_by_date = await _fetch_city_obs(session, metric, station, start_dt, end_dt)
            if peak_by_date:
                all_stats[metric] = _aggregate(peak_by_date)

    # --- Write CSV ---
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "city_key", "city_name", "station", "month",
        "n_days", "mean_peak_time", "std_minutes",
        "p50", "p75", "p90", "p95",
    ]
    rows: list[dict] = []
    for metric, monthly in sorted(all_stats.items()):
        city_name = CITIES[metric][0] if metric in CITIES else metric
        station = KALSHI_STATION_IDS.get(metric, "?")
        for month in sorted(monthly):
            s = monthly[month]
            rows.append({
                "city_key":       metric,
                "city_name":      city_name,
                "station":        station,
                "month":          month,
                "n_days":         s["n"],
                "mean_peak_time": _minutes_to_hhmm(round(s["mean"])),
                "std_minutes":    round(s["std"]),
                "p50":            _minutes_to_hhmm(s["p50"]),
                "p75":            _minutes_to_hhmm(s["p75"]),
                "p90":            _minutes_to_hhmm(s["p90"]),
                "p95":            _minutes_to_hhmm(s["p95"]),
            })

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote %d rows to %s", len(rows), out_path)

    # --- Write Python dict literal ---
    if emit_dict:
        dict_path = out_path.parent / "peak_hour_p90.py"
        lines = [
            "# Auto-generated by scripts/backtest_peak_hour.py — do not edit manually",
            "# p90 threshold: local time by which 90% of daily highs have been observed",
            "",
            "# Minutes since local midnight",
            "P90_MINUTES: dict[str, dict[int, int]] = {",
        ]
        for metric in sorted(all_stats):
            monthly = all_stats[metric]
            inner = ", ".join(
                f"{m}: {monthly[m]['p90']}"
                for m in sorted(monthly)
            )
            lines.append(f'    "{metric}": {{{inner}}},')
        lines.append("}")
        lines.append("")
        lines.append("# Decimal local hour (minutes / 60) — compare to local_hour + local_minute/60")
        lines.append("P90_DECIMAL_HOUR: dict[str, dict[int, float]] = {")
        for metric in sorted(all_stats):
            monthly = all_stats[metric]
            inner = ", ".join(
                f"{m}: {monthly[m]['p90'] / 60:.2f}"
                for m in sorted(monthly)
            )
            lines.append(f'    "{metric}": {{{inner}}},')
        lines.append("}")
        lines.append("")

        dict_path.write_text("\n".join(lines))
        log.info("Wrote dict literal to %s", dict_path)

        # Also print a summary to stdout
        print("\n=== P90 peak times by city and month (HH:MM local) ===")
        header = f"{'City':<22}" + "".join(f"  {m:>2}" for m in range(1, 13))
        print(header)
        print("-" * len(header))
        for metric in sorted(all_stats):
            city_name = CITIES[metric][0] if metric in CITIES else metric
            monthly = all_stats[metric]
            row = f"{city_name:<22}"
            for m in range(1, 13):
                if m in monthly:
                    h = monthly[m]["p90"] // 60
                    row += f"  {h:02d}"
                else:
                    row += "   --"
            print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute peak temperature time distributions by city and month."
    )
    parser.add_argument(
        "--years", type=int, default=5,
        help="Lookback period in years (default: 5)",
    )
    parser.add_argument(
        "--cities", nargs="+", default=None,
        help="City suffixes to process, e.g. --cities chi bos ny (default: all)",
    )
    parser.add_argument(
        "--out", default="data/peak_hour_analysis.csv",
        help="Output CSV path (default: data/peak_hour_analysis.csv)",
    )
    parser.add_argument(
        "--dict", action="store_true",
        help="Also write data/peak_hour_p90.py and print summary table",
    )
    args = parser.parse_args()

    asyncio.run(main(
        years=args.years,
        city_filter=set(args.cities) if args.cities else None,
        out_path=Path(args.out),
        emit_dict=args.dict,
    ))
