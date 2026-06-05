"""Fetch historical temperature data from Iowa State Mesonet.

Builds running daily maximum at each local hour — the entry-condition input
for backtest_band_arb_yes.py.  Two output modes:

  metar     — routine METAR + SPECI only (report_type 3,4).
              Typically 1 reading per hour, sub-hour only on significant changes.
              0.1°C precision.  → data/mesonet_hourly_metar.csv

  combined  — METAR + SPECI + ASOS 5-minute automated (report_type 1,3,4).
              Reading every ~5 minutes.  Automated readings are integer °C
              (±0.5°C = ±0.9°F), but update fast enough to catch intra-hour
              spikes that METAR misses.  → data/mesonet_hourly_combined.csv

Running both lets the backtest compare entry conditions with and without the
ASOS 5-minute signal, matching what the live bot now sees.

Data source: Iowa State Environmental Mesonet (no auth required)
  https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py

IEM report_type values:
  1 = ASOS automated 5-minute observations
  3 = Routine METAR (hourly)
  4 = SPECI (special observations on significant weather changes)

Output columns: city_metric, date, local_hour, running_max_f
  Each row = running daily max at the START of that local hour
  (max of all observations from local midnight through H:59).
  local_hour uses DST-aware city timezone (same as live bot and peak_hour script).

Usage:
  venv/bin/python scripts/fetch_mesonet_history.py              # both modes, 60 days
  venv/bin/python scripts/fetch_mesonet_history.py --mode metar
  venv/bin/python scripts/fetch_mesonet_history.py --mode combined
  venv/bin/python scripts/fetch_mesonet_history.py --days 90 --cities lax sfo
  venv/bin/python scripts/fetch_mesonet_history.py --start 2026-02-01 --end 2026-05-24
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
_FETCH_DELAY = 2.5  # seconds between requests — Iowa State Mesonet rate limit
_MAX_RETRIES = 3

_HOUR_START = 6
_HOUR_END   = 22

# IEM report_type strings per mode
_REPORT_TYPES = {
    "metar":    "3,4",    # routine METAR + SPECI
    "combined": "1,3,4",  # ASOS 5-min + routine METAR + SPECI
}

_OUT_PATHS = {
    "metar":    Path("data/mesonet_hourly_metar.csv"),
    "combined": Path("data/mesonet_hourly_combined.csv"),
}


async def _fetch_city_obs(
    session:     aiohttp.ClientSession,
    semaphore:   asyncio.Semaphore,
    metric:      str,
    station:     str,
    start_dt:    date,
    end_dt:      date,
    report_type: str,
) -> list[dict]:
    """Fetch all observations for one station and return running-max rows.

    Returns list of {city_metric, date, local_hour, running_max_f}.
    """
    city_entry = CITIES.get(metric)
    if city_entry is None:
        log.warning("No CITIES entry for %s — skipping", metric)
        return []
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
        "report_type": report_type,
    }

    text: str | None = None
    for attempt in range(_MAX_RETRIES):
        async with semaphore:
            try:
                async with session.get(
                    _MESONET_URL,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    if resp.status == 429:
                        wait = _FETCH_DELAY * (2 ** attempt)
                        log.warning("429 for %s — retry %d/%d in %.1fs", metric, attempt + 1, _MAX_RETRIES, wait)
                        await asyncio.sleep(wait)
                        continue
                    resp.raise_for_status()
                    text = await resp.text()
            except Exception as exc:
                log.error("Mesonet fetch failed for %s (%s): %s", metric, station, exc)
                return []
            await asyncio.sleep(_FETCH_DELAY)
        break
    if text is None:
        log.error("Mesonet fetch failed for %s (%s) after %d retries", metric, station, _MAX_RETRIES)
        return []

    # Parse CSV: station,valid,tmpf  (valid is UTC "YYYY-MM-DD HH:MM")
    obs_by_date_hour: dict[str, dict[int, list[float]]] = {}
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

    # Build running daily max at each local hour.
    rows: list[dict] = []
    for date_str in sorted(obs_by_date_hour):
        hour_obs = obs_by_date_hour[date_str]
        running_max: float | None = None
        for hour in range(0, 24):
            temps = hour_obs.get(hour, [])
            if temps:
                running_max = (
                    max(temps) if running_max is None
                    else max(running_max, max(temps))
                )
            if _HOUR_START <= hour <= _HOUR_END and running_max is not None:
                rows.append({
                    "city_metric":   metric,
                    "date":          date_str,
                    "local_hour":    hour,
                    "running_max_f": round(running_max, 2),
                })

    log.info("  %s (%s) [%s]: %d city-days, %d hourly rows",
             metric, station, report_type, len(obs_by_date_hour), len(rows))
    return rows


async def _run_mode(
    mode:        str,
    start_dt:    date,
    end_dt:      date,
    city_filter: list[str] | None,
    out_path:    Path,
) -> None:
    report_type = _REPORT_TYPES[mode]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pairs: list[tuple[str, str]] = []
    for metric, station in KALSHI_STATION_IDS.items():
        if city_filter:
            suffix = metric.replace("temp_high_", "")
            if suffix not in city_filter:
                continue
        pairs.append((metric, station))

    log.info(
        "[%s] Fetching %d cities %s → %s (report_type=%s)",
        mode, len(pairs), start_dt, end_dt, report_type,
    )

    semaphore = asyncio.Semaphore(1)
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            *[_fetch_city_obs(session, semaphore, m, s, start_dt, end_dt, report_type)
              for m, s in pairs],
            return_exceptions=True,
        )

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

    log.info("[%s] Wrote %d rows to %s", mode, len(all_rows), out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch historical METAR/ASOS temperature data from Iowa State Mesonet."
    )
    parser.add_argument(
        "--mode", choices=["metar", "combined", "both"], default="both",
        help=(
            "metar = METAR+SPECI only → mesonet_hourly_metar.csv; "
            "combined = ASOS-5min+METAR+SPECI → mesonet_hourly_combined.csv; "
            "both = run both (default)"
        ),
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
        help="City suffixes to fetch, e.g. --cities chi bos ny lax (default: all)",
    )
    args = parser.parse_args()

    end_date   = date.fromisoformat(args.end)   if args.end   else date.today()
    start_date = (
        date.fromisoformat(args.start) if args.start
        else end_date - timedelta(days=args.days)
    )

    modes_to_run = ["metar", "combined"] if args.mode == "both" else [args.mode]
    for mode in modes_to_run:
        asyncio.run(_run_mode(mode, start_date, end_date, args.cities, _OUT_PATHS[mode]))
