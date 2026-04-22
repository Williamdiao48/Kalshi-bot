"""Fetch historical hourly METAR temperature data from Iowa State Mesonet.

For each Kalshi temperature city, downloads all ASOS/METAR observations for
the requested date range and computes the running daily maximum at each local
hour — mirroring exactly how the live METAR module builds its obs_values.

The resulting CSV is the input for backtest_band_arb_yes.py: it tells us what
temperature the bot would have seen at e.g. 10 AM, 12 PM, 2 PM on each
historical day, which lets us simulate when the YES band-arb signal would have
fired and at what clearance.

Data source: Iowa State Environmental Mesonet (no auth required)
  https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py

Output: data/mesonet_hourly.csv
  city_metric, date, local_hour, running_max_f

  Each row represents the running daily maximum at the START of that local hour
  (i.e., the max of all observations from midnight through hour H:59).

Usage:
  venv/bin/python scripts/fetch_mesonet_history.py
  venv/bin/python scripts/fetch_mesonet_history.py --days 60
  venv/bin/python scripts/fetch_mesonet_history.py --start 2026-02-01 --end 2026-04-13
  venv/bin/python scripts/fetch_mesonet_history.py --cities chi bos ny --days 30
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
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
_SEMAPHORE = asyncio.Semaphore(5)

# Only emit hourly snapshots for hours that are useful for the band-arb signal.
# The bot's pre-lock window is 6 h before close (~5 PM local); earliest
# meaningful temperature reading is ~6 AM after morning low.
_HOUR_START = 6
_HOUR_END = 22  # inclusive — capture through late afternoon


def _c_to_f(temp_c: float) -> float:
    return temp_c * 9.0 / 5.0 + 32.0


async def _fetch_city_obs(
    session: aiohttp.ClientSession,
    metric: str,
    station: str,
    start_dt: date,
    end_dt: date,
) -> list[dict]:
    """Fetch all observations for one station between start_dt and end_dt (inclusive).

    Returns list of {city_metric, date, local_hour, running_max_f} rows.
    """
    city_entry = CITIES.get(metric)
    if city_entry is None:
        log.warning("No CITIES entry for %s — skipping", metric)
        return []
    _, _, _, city_tz = city_entry

    params = {
        "station":  station,
        "data":     "tmpf",          # temperature in °F
        "year1":    str(start_dt.year),
        "month1":   str(start_dt.month),
        "day1":     str(start_dt.day),
        "year2":    str(end_dt.year),
        "month2":   str(end_dt.month),
        "day2":     str(end_dt.day),
        "tz":       "UTC",
        "format":   "comma",
        "latlon":   "no",
        "missing":  "M",
        "trace":    "T",
        "direct":   "no",
        "report_type": "1,3",       # METAR + special obs
    }

    async with _SEMAPHORE:
        try:
            async with session.get(
                _MESONET_URL,
                params=params,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                resp.raise_for_status()
                text = await resp.text()
        except Exception as exc:
            log.error("Mesonet fetch failed for %s (%s): %s", metric, station, exc)
            return []

    # Parse CSV response.
    # Format: station,valid,tmpf
    # valid is UTC timestamp "YYYY-MM-DD HH:MM"
    obs_by_date_hour: dict[str, dict[int, list[float]]] = {}
    lines = text.splitlines()
    for line in lines:
        if line.startswith("#") or line.startswith("station") or not line.strip():
            continue
        parts = line.split(",")
        if len(parts) < 3:
            continue
        try:
            utc_ts = datetime.strptime(parts[1].strip(), "%Y-%m-%d %H:%M")
            utc_ts = utc_ts.replace(tzinfo=timezone.utc)
            local_ts = utc_ts.astimezone(city_tz)
            local_date_str = local_ts.date().isoformat()
            local_hour = local_ts.hour
            temp_str = parts[2].strip()
            if temp_str in ("M", "T", ""):
                continue
            temp_f = float(temp_str)
        except (ValueError, IndexError):
            continue

        obs_by_date_hour.setdefault(local_date_str, {}).setdefault(local_hour, []).append(temp_f)

    # For each date, build running daily max at each local hour.
    rows: list[dict] = []
    for date_str in sorted(obs_by_date_hour):
        hour_obs = obs_by_date_hour[date_str]
        running_max: float | None = None
        for hour in range(0, 24):
            temps = hour_obs.get(hour, [])
            if temps:
                hour_max = max(temps)
                running_max = hour_max if running_max is None else max(running_max, hour_max)
            # Only emit rows for useful hours
            if _HOUR_START <= hour <= _HOUR_END and running_max is not None:
                rows.append({
                    "city_metric":   metric,
                    "date":          date_str,
                    "local_hour":    hour,
                    "running_max_f": round(running_max, 2),
                })

    log.info("  %s (%s): %d city-days, %d hourly rows",
             metric, station, len(obs_by_date_hour), len(rows))
    return rows


async def main(
    start_dt: date,
    end_dt: date,
    city_filter: list[str] | None,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build list of (metric, station) pairs to fetch
    pairs: list[tuple[str, str]] = []
    for metric, station in KALSHI_STATION_IDS.items():
        if city_filter:
            # --cities accepts short suffixes like "chi", "bos", "ny"
            suffix = metric.replace("temp_high_", "")
            if suffix not in city_filter:
                continue
        pairs.append((metric, station))

    log.info("Fetching %d cities from %s to %s", len(pairs), start_dt, end_dt)

    async with aiohttp.ClientSession() as session:
        tasks = [
            _fetch_city_obs(session, metric, station, start_dt, end_dt)
            for metric, station in pairs
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    all_rows: list[dict] = []
    for (metric, _), result in zip(pairs, results):
        if isinstance(result, Exception):
            log.error("Error fetching %s: %s", metric, result)
        else:
            all_rows.extend(result)

    all_rows.sort(key=lambda r: (r["city_metric"], r["date"], r["local_hour"]))

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["city_metric", "date", "local_hour", "running_max_f"]
        )
        writer.writeheader()
        writer.writerows(all_rows)

    log.info("Wrote %d rows to %s", len(all_rows), out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch historical hourly METAR data from Iowa State Mesonet."
    )
    parser.add_argument(
        "--days", type=int, default=60,
        help="Fetch this many days ending today (default: 60). Overridden by --start/--end.",
    )
    parser.add_argument(
        "--start", type=str, default=None,
        help="Start date YYYY-MM-DD (overrides --days)",
    )
    parser.add_argument(
        "--end", type=str, default=None,
        help="End date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--cities", nargs="+", default=None,
        help="City suffixes to fetch, e.g. --cities chi bos ny (default: all)",
    )
    parser.add_argument(
        "--out", default="data/mesonet_hourly.csv",
        help="Output CSV path (default: data/mesonet_hourly.csv)",
    )
    args = parser.parse_args()

    end_date = date.fromisoformat(args.end) if args.end else date.today()
    start_date = (
        date.fromisoformat(args.start) if args.start
        else end_date - timedelta(days=args.days)
    )

    asyncio.run(main(start_date, end_date, args.cities, Path(args.out)))
