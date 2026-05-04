"""Fetch historical GFS/ECMWF/ICON daily-max temperature forecasts from Open-Meteo.

Downloads model analysis/forecast data for each Kalshi temperature city,
computes the local-time daily maximum for each day, and saves to CSV.

NOTE: Open-Meteo's historical-forecast-api returns model analysis output
(~1°F MAE vs actual), not true day-ahead forecasts (~3-5°F MAE). The relative
ranking of models should hold but absolute win rates will be optimistic.

Output: data/openmeteo_forecasts.csv
  city_metric, date, model, forecast_high_f

  'date' is the LOCAL measurement date (same as the Kalshi ticker date,
  which is band["date"] - 1 day from kxhigh_bands.csv convention).

Usage:
  venv/bin/python scripts/fetch_openmeteo_forecast_history.py
  venv/bin/python scripts/fetch_openmeteo_forecast_history.py --days 90
  venv/bin/python scripts/fetch_openmeteo_forecast_history.py --start 2026-02-01 --end 2026-04-30
  venv/bin/python scripts/fetch_openmeteo_forecast_history.py --cities dca chi bos
  venv/bin/python scripts/fetch_openmeteo_forecast_history.py --models gfs_seamless ecmwf_ifs
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

from kalshi_bot.news.noaa import CITIES  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_API_URL   = "https://historical-forecast-api.open-meteo.com/v1/forecast"
_SEMAPHORE = asyncio.Semaphore(1)
_DELAY     = 0.6   # seconds between requests — Open-Meteo free tier

_ALL_MODELS = ["gfs_seamless", "ecmwf_ifs", "icon_seamless", "gfs_hrrr", "gem_seamless"]

# Daytime hours (local) used for daily-max computation — matches mesonet convention.
_HOUR_START = 6
_HOUR_END   = 22  # inclusive


async def _fetch_city_model(
    session: aiohttp.ClientSession,
    metric: str,
    lat: float,
    lon: float,
    city_tz,
    model: str,
    start_dt: date,
    end_dt: date,
) -> list[dict]:
    """Fetch hourly temps for one city × model and return daily-max rows."""
    params = {
        "latitude":          lat,
        "longitude":         lon,
        "hourly":            "temperature_2m",
        "temperature_unit":  "fahrenheit",
        "start_date":        start_dt.isoformat(),
        "end_date":          end_dt.isoformat(),
        "models":            model,
        "timezone":          "UTC",
    }

    async with _SEMAPHORE:
        try:
            async with session.get(
                _API_URL,
                params=params,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status == 429:
                    log.warning("Rate-limited (%s/%s) — waiting 10s", metric, model)
                    await asyncio.sleep(10.0)
                    return []
                resp.raise_for_status()
                data = await resp.json()
        except Exception as exc:
            log.error("Fetch error %s/%s: %s", metric, model, exc)
            return []
        await asyncio.sleep(_DELAY)

    hourly = data.get("hourly", {})
    times  = hourly.get("time", [])
    temps  = hourly.get("temperature_2m", [])

    # Group temps by local date → compute daily max for hours 6–22.
    daily: dict[str, list[float]] = {}
    for t_str, temp in zip(times, temps):
        if temp is None:
            continue
        try:
            utc_dt    = datetime.strptime(t_str, "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)
            local_dt  = utc_dt.astimezone(city_tz)
            local_hr  = local_dt.hour
            date_str  = local_dt.date().isoformat()
        except (ValueError, AttributeError):
            continue
        if _HOUR_START <= local_hr <= _HOUR_END:
            daily.setdefault(date_str, []).append(temp)

    rows: list[dict] = []
    for date_str, day_temps in sorted(daily.items()):
        if not day_temps:
            continue
        rows.append({
            "city_metric":     metric,
            "date":            date_str,
            "model":           model,
            "forecast_high_f": round(max(day_temps), 2),
        })

    log.info("  %-22s %-16s → %3d days", metric, model, len(rows))
    return rows


async def main(
    start_dt: date,
    end_dt: date,
    city_filter: list[str] | None,
    models: list[str],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pairs: list[tuple[str, float, float, object, str]] = []
    for metric, (_, lat, lon, tz) in CITIES.items():
        if not metric.startswith("temp_high_"):
            continue
        if city_filter:
            suffix = metric.replace("temp_high_", "")
            if suffix not in city_filter:
                continue
        for model in models:
            pairs.append((metric, lat, lon, tz, model))

    log.info("Fetching %d city×model combos from %s to %s", len(pairs), start_dt, end_dt)

    async with aiohttp.ClientSession() as session:
        tasks = [
            _fetch_city_model(session, metric, lat, lon, tz, model, start_dt, end_dt)
            for metric, lat, lon, tz, model in pairs
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    all_rows: list[dict] = []
    for (metric, *_, model), result in zip(pairs, results):
        if isinstance(result, Exception):
            log.error("Error for %s/%s: %s", metric, model, result)
        else:
            all_rows.extend(result)

    all_rows.sort(key=lambda r: (r["city_metric"], r["date"], r["model"]))

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["city_metric", "date", "model", "forecast_high_f"]
        )
        writer.writeheader()
        writer.writerows(all_rows)

    log.info("Wrote %d rows to %s  (%d cities × %d models)",
             len(all_rows), out_path, len(CITIES), len(models))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch historical GFS/ECMWF/ICON daily-max forecasts from Open-Meteo."
    )
    parser.add_argument(
        "--days", type=int, default=90,
        help="How many days back to fetch (default: 90). Overridden by --start/--end.",
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
        help="City suffixes, e.g. --cities dca chi bos (default: all)",
    )
    parser.add_argument(
        "--models", nargs="+", default=_ALL_MODELS,
        help=f"Models to fetch (default: {_ALL_MODELS})",
    )
    parser.add_argument(
        "--out", default="data/openmeteo_forecasts.csv",
        help="Output CSV path (default: data/openmeteo_forecasts.csv)",
    )
    args = parser.parse_args()

    end_date = date.fromisoformat(args.end) if args.end else date.today()
    start_date = (
        date.fromisoformat(args.start) if args.start
        else end_date - timedelta(days=args.days)
    )

    asyncio.run(main(start_date, end_date, args.cities, args.models, Path(args.out)))
