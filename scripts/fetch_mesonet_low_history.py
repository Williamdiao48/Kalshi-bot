"""Fetch historical hourly METAR temperature data for KXLOWT cities from Iowa State Mesonet.

For each Kalshi daily-low city, downloads all ASOS/METAR observations for the
requested date range and computes the running daily MINIMUM at each local hour —
mirroring exactly how the live METAR module builds its obs_values for KXLOWT.

The resulting CSV feeds backtest_kxlowt_entry_timing.py: it lets us reconstruct
which hours the bot's noaa_observed YES signal would have fired (running_min inside
the nominal band) and at what edge, so the exit-parameter grid search runs only on
bot-eligible entries rather than all 316 markets indiscriminately.

Data source: Iowa State Environmental Mesonet (no auth required)
  https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py

Station IDs are the hard-coded Kalshi resolution stations from LOW_CITIES in
kalshi_bot/news/noaa.py (comments).  They match the stations Kalshi uses to
settle KXLOWT markets and that the live bot uses for noaa_observed signals.

Output: data/mesonet_low_hourly.csv
  city_metric, date, local_hour, running_min_f

  Each row is the running daily minimum at the START of that local hour
  (i.e., the min of all observations from midnight through hour H:59 inclusive).

Usage:
  venv/bin/python scripts/fetch_mesonet_low_history.py
  venv/bin/python scripts/fetch_mesonet_low_history.py --start 2026-04-06 --end 2026-04-22
  venv/bin/python scripts/fetch_mesonet_low_history.py --cities chi bos nyc --days 30
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).parent.parent))

from kalshi_bot.news.noaa import LOW_CITIES  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_MESONET_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
_SEMAPHORE = asyncio.Semaphore(1)   # Mesonet enforces ~1 req/s; serialize to avoid 429
_FETCH_DELAY = 1.5                  # seconds between requests

# Station IDs for KXLOWT settlement stations.
# Source: comments in LOW_CITIES dict in kalshi_bot/news/noaa.py.
# These are the same stations Kalshi uses to settle KXLOWT markets.
LOW_STATION_IDS: dict[str, str] = {
    "temp_low_lax": "KLAX",
    "temp_low_den": "KDEN",
    "temp_low_chi": "KMDW",
    "temp_low_ny":  "KNYC",
    "temp_low_mia": "KMIA",
    "temp_low_aus": "KAUS",
    "temp_low_bos": "KBOS",
    "temp_low_hou": "KHOU",
    "temp_low_dfw": "KDFW",
    "temp_low_sfo": "KSFO",
    "temp_low_sea": "KSEA",
    "temp_low_phx": "KPHX",
    "temp_low_phl": "KPHL",
    "temp_low_atl": "KATL",
    "temp_low_msp": "KMSP",
    "temp_low_dca": "KDCA",
    "temp_low_las": "KLAS",
    "temp_low_okc": "KOKC",
    "temp_low_sat": "KSAT",
    "temp_low_msy": "KMSY",
}

# Backtest city-slug → metric name (slugs from KXLOWT ticker middle section)
SLUG_TO_METRIC: dict[str, str] = {
    "lax": "temp_low_lax",
    "den": "temp_low_den",
    "chi": "temp_low_chi",
    "nyc": "temp_low_ny",
    "ny":  "temp_low_ny",
    "mia": "temp_low_mia",
    "aus": "temp_low_aus",
    "bos": "temp_low_bos",
    "hou": "temp_low_hou",
    "dfw": "temp_low_dfw",
    "sfo": "temp_low_sfo",
    "sea": "temp_low_sea",
    "phx": "temp_low_phx",
    "phl": "temp_low_phl",
    "atl": "temp_low_atl",
    "min": "temp_low_msp",   # backtest uses "min", noaa uses "msp"
    "msp": "temp_low_msp",
    "dc":  "temp_low_dca",
    "dca": "temp_low_dca",
    "las": "temp_low_las",
    "okc": "temp_low_okc",
    "sat": "temp_low_sat",
    "nola": "temp_low_msy",
    "msy": "temp_low_msy",
}


async def _fetch_city_obs(
    session: aiohttp.ClientSession,
    metric: str,
    station: str,
    start_dt: date,
    end_dt: date,
) -> list[dict]:
    """Fetch all observations for one station and return running-min rows."""
    city_entry = LOW_CITIES.get(metric)
    if city_entry is None:
        log.warning("No LOW_CITIES entry for %s — skipping", metric)
        return []
    _, _, _, city_tz = city_entry

    params = {
        "station":  station,
        "data":     "tmpf",
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
        "report_type": "1,3",
    }

    text = None
    for attempt in range(4):
        async with _SEMAPHORE:
            await asyncio.sleep(_FETCH_DELAY * (2 ** attempt))
            try:
                async with session.get(
                    _MESONET_URL,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    if resp.status == 429:
                        log.warning("429 for %s attempt %d — backing off", metric, attempt + 1)
                        continue
                    resp.raise_for_status()
                    text = await resp.text()
                    break
            except Exception as exc:
                log.warning("Mesonet attempt %d failed for %s: %s", attempt + 1, metric, exc)
    if text is None:
        log.error("All retries failed for %s (%s)", metric, station)
        return []

    # Parse CSV: station,valid(UTC),tmpf
    obs_by_date_hour: dict[str, dict[int, list[float]]] = {}
    for line in text.splitlines():
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

    # Build running daily MIN at each local hour (all 24 hours — trough can occur 2-6 AM)
    rows: list[dict] = []
    for date_str in sorted(obs_by_date_hour):
        hour_obs = obs_by_date_hour[date_str]
        running_min: float | None = None
        for hour in range(0, 24):
            temps = hour_obs.get(hour, [])
            if temps:
                hour_min = min(temps)
                running_min = hour_min if running_min is None else min(running_min, hour_min)
            if running_min is not None:
                rows.append({
                    "city_metric":   metric,
                    "date":          date_str,
                    "local_hour":    hour,
                    "running_min_f": round(running_min, 2),
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

    pairs: list[tuple[str, str]] = []
    for metric, station in LOW_STATION_IDS.items():
        if city_filter:
            suffix = metric.replace("temp_low_", "")
            if suffix not in city_filter:
                continue
        pairs.append((metric, station))

    log.info("Fetching %d KXLOWT cities  %s → %s", len(pairs), start_dt, end_dt)

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
            f, fieldnames=["city_metric", "date", "local_hour", "running_min_f"]
        )
        writer.writeheader()
        writer.writerows(all_rows)

    log.info("Wrote %d rows to %s", len(all_rows), out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=17,
                        help="Days ending today (default 17 = Apr 6-22 backtest window)")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--cities", nargs="+", default=None,
                        help="City suffixes, e.g. --cities chi bos ny msp atl")
    parser.add_argument("--out", default="data/mesonet_low_hourly.csv")
    args = parser.parse_args()

    end_date   = date.fromisoformat(args.end)   if args.end   else date(2026, 4, 22)
    start_date = date.fromisoformat(args.start) if args.start else date(2026, 4, 6)

    asyncio.run(main(start_date, end_date, args.cities, Path(args.out)))
