"""WeatherAPI.com forecast high temperature fetcher.

Provides a reliable replacement for OWM as a second independent weather
forecast source.  WeatherAPI's free tier allows 1 million calls/month
(vs OWM's slot-based approach which frequently returns no data for today).

Set the env var ``WEATHERAPI_KEY`` to enable (free key at weatherapi.com).
If the key is absent the module returns an empty list and the poll continues.

API:
    GET https://api.weatherapi.com/v1/forecast.json
        ?key={KEY}
        &q={lat},{lon}
        &days=3
        &aqi=no
        &alerts=no

Response (relevant fields):
    {
      "forecast": {
        "forecastday": [
          {
            "date": "2026-03-22",
            "day": {
              "maxtemp_f": 63.1
            }
          },
          ...
        ]
      }
    }

Each forecastday entry covers one calendar day in the location's local time.
We emit one DataPoint per day (up to WEATHERAPI_FORECAST_DAYS days), carrying
forecast_date and forecast_offset in metadata so the date guard in main.py
aligns it to the correct Kalshi market.

Caching
-------
Results are cached for WEATHERAPI_CACHE_MINUTES (default 15 minutes).
WeatherAPI updates forecasts every few hours so a 15-minute cache wastes
no meaningful signal while staying well within free-tier limits.

Source tag: ``"weatherapi"``
Counted as a forecast corroboration source alongside "noaa", "owm",
"nws_hourly", "open_meteo", and "hrrr" in the consensus filter in main.py.

Environment variables
---------------------
  WEATHERAPI_KEY             API key from weatherapi.com (required).
  WEATHERAPI_FORECAST_DAYS   Days to emit (1–3 on free tier, up to 10 on paid).
                             Default: 3.
  WEATHERAPI_CACHE_MINUTES   Cache TTL in minutes.  Default: 15.
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import aiohttp

from ..data import DataPoint
from .noaa import CITIES

_BASE_URL = "https://api.weatherapi.com/v1/forecast.json"

WEATHERAPI_FORECAST_DAYS: int = min(10, max(1, int(
    os.environ.get("WEATHERAPI_FORECAST_DAYS", "3")
)))
WEATHERAPI_CACHE_MINUTES: int = int(os.environ.get("WEATHERAPI_CACHE_MINUTES", "15"))

# Per-city cache: metric → (monotonic_time, list[DataPoint])
_city_cache: dict[str, tuple[float, list[DataPoint]]] = {}


async def _fetch_city_forecast(
    session: aiohttp.ClientSession,
    metric: str,
    city_name: str,
    lat: float,
    lon: float,
    api_key: str,
    city_tz: "ZoneInfo",
) -> list[DataPoint]:
    """Fetch WeatherAPI forecast daily highs for one city.

    Returns one DataPoint per forecast day (up to WEATHERAPI_FORECAST_DAYS).
    Returns empty list on failure or missing API key.
    """
    cache_ttl = WEATHERAPI_CACHE_MINUTES * 60
    now = time.monotonic()
    cached = _city_cache.get(metric)
    if cached is not None:
        cache_ts, cache_pts = cached
        if (now - cache_ts) < cache_ttl:
            return cache_pts

    params = {
        "key":     api_key,
        "q":       f"{lat:.4f},{lon:.4f}",
        "days":    str(WEATHERAPI_FORECAST_DAYS),
        "aqi":     "no",
        "alerts":  "no",
    }
    try:
        async with session.get(
            _BASE_URL,
            params=params,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
    except aiohttp.ClientResponseError as exc:
        logging.error("WeatherAPI HTTP error %s for %s: %s", exc.status, city_name, exc.message)
        return []
    except Exception as exc:
        logging.warning("WeatherAPI fetch failed for %s: %s", city_name, exc)
        return []

    forecastdays = (data.get("forecast") or {}).get("forecastday") or []
    if not forecastdays:
        logging.warning("WeatherAPI: empty forecastday for %s", city_name)
        return []

    # Anchor to the city's local date, not UTC, because WeatherAPI returns
    # forecastday dates in local time. Using UTC as anchor drops the local-today
    # entry for PT cities during 0:00–8:00 UTC (day_offset computes as -1).
    today_local = datetime.now(city_tz).date()
    points: list[DataPoint] = []
    summary_parts: list[str] = []

    for entry in forecastdays:
        date_str = entry.get("date", "")
        maxtemp_f = (entry.get("day") or {}).get("maxtemp_f")
        if not date_str or maxtemp_f is None:
            continue

        try:
            entry_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            continue

        day_offset = (entry_date - today_local).days
        if day_offset < 0 or day_offset >= WEATHERAPI_FORECAST_DAYS:
            continue

        value = float(maxtemp_f)
        # as_of = noon in the city's local timezone (matches NOAA convention).
        as_of = datetime(
            entry_date.year, entry_date.month, entry_date.day,
            12, 0, 0, tzinfo=city_tz,
        ).astimezone(timezone.utc).isoformat()

        label = "today" if day_offset == 0 else f"day+{day_offset}"
        summary_parts.append(f"{label}={value:.1f}°F")

        points.append(DataPoint(
            source="weatherapi",
            metric=metric,
            value=value,
            unit="°F",
            as_of=as_of,
            metadata={
                "city":            city_name,
                "forecast_date":   date_str,
                "forecast_offset": day_offset,
            },
        ))

    if summary_parts:
        logging.info("WeatherAPI [%s]: %s", city_name, "  ".join(summary_parts))
    else:
        logging.warning("WeatherAPI: no forecast data for %s", city_name)

    _city_cache[metric] = (now, points)
    return points


async def fetch_city_forecasts(session: aiohttp.ClientSession) -> list[DataPoint]:
    """Fetch WeatherAPI forecast daily highs for all tracked Kalshi cities.

    Returns empty list if WEATHERAPI_KEY is not set.
    Runs all city fetches concurrently.  Cities that fail are skipped.

    Returns:
        List of DataPoints (one per city per forecast day).
    """
    api_key = os.environ.get("WEATHERAPI_KEY", "")
    if not api_key:
        logging.warning("WEATHERAPI_KEY not set — skipping WeatherAPI fetch.")
        return []

    tasks = [
        _fetch_city_forecast(session, metric, city_name, lat, lon, api_key, city_tz)
        for metric, (city_name, lat, lon, city_tz) in CITIES.items()
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    data_points: list[DataPoint] = []
    for metric, result in zip(CITIES.keys(), results):
        if isinstance(result, Exception):
            logging.error("WeatherAPI fetch error for %s: %s", metric, result)
        elif result:
            data_points.extend(result)

    return data_points
