"""Compute when daily minimum temperatures typically occur, by city and month.

For each Kalshi overnight-low city (KXLOWT* markets), fetches historical ASOS
observations from the Iowa State Environmental Mesonet and finds the local time
of each calendar day's minimum temperature.  Aggregates by (city, month) to
produce percentile distributions at minute-level precision.

Outputs:
  data/overnight_low_analysis.csv  — full percentile table (HH:MM, std in minutes)
  data/overnight_low_p90.py        — Python dict literal ready to import into the bot

Interpretation
--------------
The NWS CLI daily minimum is measured over the local calendar day (midnight to
midnight).  ASOS data shows the diurnal minimum typically occurs in the pre-dawn
window — around 04:00–07:00 local time in most cities.

The p90 threshold means: on 90% of historical days, the daily minimum had already
occurred by this local time.  If the bot enters a KXLOWT* NO signal at, say,
21:00 local, but the p90 minimum occurrence time is 07:00, the bot's running
observed_min almost certainly does NOT reflect the actual final daily minimum —
the coldest reading is still hours in the future (or already occurred before the
bot started monitoring the market that day).

This replaces guesswork about "before 08:00 UTC" with city- and season-aware
thresholds grounded in historical data.

Data source: Iowa State Environmental Mesonet (no auth required)
  https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py

Usage:
  venv/bin/python scripts/backtest_overnight_low_hour.py
  venv/bin/python scripts/backtest_overnight_low_hour.py --years 3
  venv/bin/python scripts/backtest_overnight_low_hour.py --years 5 --cities chi bos ny
  venv/bin/python scripts/backtest_overnight_low_hour.py --years 5 --dict
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


def _low_station(metric: str) -> str | None:
    """Return the ASOS station ID for a temp_low_* metric.

    LOW_CITIES shares the same physical stations as CITIES; the IDs are stored
    under the corresponding temp_high_* key in KALSHI_STATION_IDS.
    """
    high_key = metric.replace("temp_low_", "temp_high_")
    return KALSHI_STATION_IDS.get(high_key)


async def _fetch_city_obs(
    session: aiohttp.ClientSession,
    metric: str,
    station: str,
    start_dt: date,
    end_dt: date,
) -> dict[tuple[int, int], int]:
    """Fetch all obs for one station and return (month, date_ordinal) → trough_minutes.

    trough_minutes is minutes since local midnight of the earliest observation
    that tied for the daily minimum temperature on that calendar day.
    Days with fewer than _MIN_OBS_PER_DAY valid readings are excluded.
    """
    city_entry = LOW_CITIES.get(metric)
    if city_entry is None:
        log.warning("No LOW_CITIES entry for %s — skipping", metric)
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
        "report_type": "3,4",   # routine METAR + SPECI
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

    # For each local calendar day, find the time of the daily minimum.
    # The NWS CLI daily min uses midnight-to-midnight local time, so we group
    # the same way.  The minimum typically occurs in the pre-dawn window
    # (00:00–08:00 local) — this is the window that matters for gate timing.
    trough_by_date: dict[tuple[int, int], int] = {}
    for ordinal, readings in obs_by_date.items():
        if len(readings) < _MIN_OBS_PER_DAY:
            continue
        min_temp = min(t for _, t in readings)
        # Earliest reading tied at the minimum (conservative — first time it's established)
        trough_minutes = min(m for m, t in readings if t == min_temp)
        month = month_by_ordinal[ordinal]
        trough_by_date[(month, ordinal)] = trough_minutes

    log.info("  %s (%s): %d valid days", metric, station, len(trough_by_date))
    return trough_by_date


def _aggregate(trough_by_date: dict[tuple[int, int], int]) -> dict[int, dict]:
    """Aggregate (month, ordinal) → trough_minutes into per-month statistics."""
    by_month: dict[int, list[int]] = {}
    for (month, _), minutes in trough_by_date.items():
        by_month.setdefault(month, []).append(minutes)

    result: dict[int, dict] = {}
    for month, minutes_list in by_month.items():
        s = sorted(minutes_list)
        n = len(s)
        mean_min = sum(s) / n
        variance = sum((x - mean_min) ** 2 for x in s) / max(n - 1, 1)
        std_min = math.sqrt(variance)
        result[month] = {
            "n":   n,
            "mean": mean_min,
            "std":  std_min,
            "p10":  _percentile(s, 10),
            "p25":  _percentile(s, 25),
            "p50":  _percentile(s, 50),
            "p75":  _percentile(s, 75),
            "p90":  _percentile(s, 90),
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
    for metric in LOW_CITIES:
        suffix = metric.replace("temp_low_", "")
        if city_filter and suffix not in city_filter:
            continue
        station = _low_station(metric)
        if station is None:
            log.warning("No station mapping for %s — skipping", metric)
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
            trough_by_date = await _fetch_city_obs(session, metric, station, start_dt, end_dt)
            if trough_by_date:
                all_stats[metric] = _aggregate(trough_by_date)

    # --- Write CSV ---
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "city_key", "city_name", "station", "month",
        "n_days", "mean_trough_time", "std_minutes",
        "p10", "p25", "p50", "p75", "p90",
    ]
    rows: list[dict] = []
    for metric, monthly in sorted(all_stats.items()):
        city_name = LOW_CITIES[metric][0] if metric in LOW_CITIES else metric
        station = _low_station(metric) or "?"
        for month in sorted(monthly):
            s = monthly[month]
            rows.append({
                "city_key":        metric,
                "city_name":       city_name,
                "station":         station,
                "month":           month,
                "n_days":          s["n"],
                "mean_trough_time": _minutes_to_hhmm(round(s["mean"])),
                "std_minutes":     round(s["std"]),
                "p10":             _minutes_to_hhmm(s["p10"]),
                "p25":             _minutes_to_hhmm(s["p25"]),
                "p50":             _minutes_to_hhmm(s["p50"]),
                "p75":             _minutes_to_hhmm(s["p75"]),
                "p90":             _minutes_to_hhmm(s["p90"]),
            })

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote %d rows to %s", len(rows), out_path)

    if not emit_dict:
        return

    # --- Write Python dict literal ---
    dict_path = out_path.parent / "overnight_low_p90.py"
    lines = [
        "# Auto-generated by scripts/backtest_overnight_low_hour.py — do not edit manually",
        "# p90 threshold: local time by which 90% of daily minima have already occurred",
        "# Use this to gate KXLOWT* NO signals: the bot's running observed_min is only",
        "# reliable for a NO signal if the current local time is PAST this threshold",
        "# (meaning the overnight low has almost certainly already been recorded).",
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
    lines.append("# Decimal local hour — compare to local_hour + local_minute/60")
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

    # Print summary table
    print("\n=== P90 overnight-low occurrence times by city and month (HH:MM local) ===")
    print("(On 90% of historical days the daily min had already occurred by this time)")
    header = f"{'City':<22}" + "".join(f"  {m:>2}" for m in range(1, 13))
    print(header)
    print("-" * len(header))
    for metric in sorted(all_stats):
        city_name = LOW_CITIES[metric][0] if metric in LOW_CITIES else metric
        monthly = all_stats[metric]
        row = f"{city_name:<22}"
        for m in range(1, 13):
            if m in monthly:
                h = monthly[m]["p90"] // 60
                row += f"  {h:02d}"
            else:
                row += "   --"
        print(row)

    print("\n=== P50 (median) overnight-low occurrence times ===")
    print(header)
    print("-" * len(header))
    for metric in sorted(all_stats):
        city_name = LOW_CITIES[metric][0] if metric in LOW_CITIES else metric
        monthly = all_stats[metric]
        row = f"{city_name:<22}"
        for m in range(1, 13):
            if m in monthly:
                h = monthly[m]["p50"] // 60
                row += f"  {h:02d}"
            else:
                row += "   --"
        print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute overnight minimum temperature time distributions by city and month."
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
        "--out", default="data/overnight_low_analysis.csv",
        help="Output CSV path (default: data/overnight_low_analysis.csv)",
    )
    parser.add_argument(
        "--dict", action="store_true",
        help="Also write data/overnight_low_p90.py and print summary table",
    )
    args = parser.parse_args()

    asyncio.run(main(
        years=args.years,
        city_filter=set(args.cities) if args.cities else None,
        out_path=Path(args.out),
        emit_dict=args.dict,
    ))
