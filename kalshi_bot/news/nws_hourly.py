"""NWS Hourly Point Forecast fetcher.

Uses the free NWS API (no API key required) to fetch per-hour temperature
forecasts for all tracked Kalshi cities.  This provides a third independent
forecast source that updates every hour — more granular than NOAA's daily
period forecast and independent from the HRRR model.

When NOAA's day-1 forecast is stale or wrong (e.g. predicted 49°F while the
observed high is already 62°F), the NWS hourly forecast often agrees with HRRR
and the observed reading.  With nws_hourly in the corroboration pool, a NOAA
vs HRRR deadlock can be broken when nws_hourly agrees with HRRR.

API flow
--------
1. GET https://api.weather.gov/points/{lat},{lon}
   → properties.forecastHourly   (hourly forecast grid URL)

2. GET {forecastHourly}
   → features[]: each has startTime (ISO-8601) and temperature (°F)

We filter features whose startTime date matches the target date in the city's
local timezone, then take the maximum temperature across all hours as the
forecast daily high.  This is the same aggregation logic used by HRRR.

Caching
-------
Results are cached for NWS_HOURLY_CACHE_MINUTES (default 20 minutes).  The
NWS hourly grid updates once per hour so a 20-minute cache wastes no signal.
The points lookup (step 1) is cached indefinitely per (lat, lon) pair since
grid office assignments do not change.

Source tag: ``"nws_hourly"``
Counted as a forecast corroboration source alongside "noaa", "owm",
"open_meteo", and "hrrr" in the consensus filter in main.py.

Environment variables
---------------------
  NWS_HOURLY_CACHE_MINUTES   Cache TTL in minutes.  Default: 20.
  NWS_HOURLY_FORECAST_DAYS   Number of forecast days to emit (1–7).  Default: 3.
                             Day 0 = today; day 1 = tomorrow; etc.
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import aiohttp

from ..data import DataPoint
from .noaa import CITIES

_HEADERS = {"User-Agent": "kalshi-bot/1.0 (educational; contact: user@example.com)"}
_POINTS_URL = "https://api.weather.gov/points/{lat},{lon}"

NWS_HOURLY_CACHE_MINUTES: int = int(os.environ.get("NWS_HOURLY_CACHE_MINUTES", "20"))
NWS_HOURLY_FORECAST_DAYS: int = min(7, max(1, int(
    os.environ.get("NWS_HOURLY_FORECAST_DAYS", "3")
)))

# Wall-clock UTC ISO timestamp of the most recent successful fetch per city.
# Stored in DataPoint metadata as "fetched_at" so main.py can detect staleness
# regardless of cache hits — cached DataPoints carry the original fetch time.
_city_fetch_time: dict[str, str] = {}

# Per-city cache: metric → (monotonic_time, list[DataPoint])
_city_cache: dict[str, tuple[float, list[DataPoint]]] = {}

# Persistent points cache: (lat, lon) → hourly forecast URL
_points_cache: dict[tuple[float, float], str] = {}


async def _get_hourly_url(
    session: aiohttp.ClientSession,
    lat: float,
    lon: float,
) -> str | None:
    """Return the NWS hourly forecast grid URL for a lat/lon, with caching."""
    key = (round(lat, 4), round(lon, 4))
    if key in _points_cache:
        return _points_cache[key]

    url = _POINTS_URL.format(lat=f"{lat:.4f}", lon=f"{lon:.4f}")
    try:
        async with session.get(
            url,
            headers=_HEADERS,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
    except Exception as exc:
        logging.warning("NWS Hourly: points lookup failed for (%.4f,%.4f): %s", lat, lon, exc)
        return None

    hourly_url = (data.get("properties") or {}).get("forecastHourly")
    if not hourly_url:
        logging.warning("NWS Hourly: no forecastHourly URL for (%.4f,%.4f)", lat, lon)
        return None

    _points_cache[key] = hourly_url
    return hourly_url


async def _fetch_city_hourly(
    session: aiohttp.ClientSession,
    metric: str,
    city_name: str,
    lat: float,
    lon: float,
    city_tz: ZoneInfo,
) -> list[DataPoint]:
    """Fetch NWS hourly forecast daily highs for one city.

    Returns one DataPoint per forecast day (up to NWS_HOURLY_FORECAST_DAYS).
    Each DataPoint carries forecast_date and forecast_offset in metadata so
    the date guard in main.py aligns it to the correct Kalshi market.
    """
    cache_ttl = NWS_HOURLY_CACHE_MINUTES * 60
    now = time.monotonic()
    cached = _city_cache.get(metric)
    if cached is not None:
        cache_ts, cache_pts = cached
        if (now - cache_ts) < cache_ttl:
            return cache_pts

    fetch_wall_time = datetime.now(timezone.utc).isoformat()

    hourly_url = await _get_hourly_url(session, lat, lon)
    if not hourly_url:
        return []

    try:
        async with session.get(
            hourly_url,
            headers=_HEADERS,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
    except Exception as exc:
        logging.warning("NWS Hourly fetch failed for %s: %s", city_name, exc)
        return []

    features = (data.get("properties") or {}).get("periods") or []
    if not features:
        logging.warning("NWS Hourly: empty periods for %s", city_name)
        return []

    # Build day → max_temp mapping using LOCAL dates from each period's timestamp.
    day_highs: dict[str, float] = {}

    for period in features:
        start_raw = period.get("startTime", "")
        temp = period.get("temperature")
        temp_unit = period.get("temperatureUnit", "F")
        if temp is None or not start_raw:
            continue
        temp_f = float(temp) if temp_unit == "F" else float(temp) * 9 / 5 + 32
        try:
            dt = datetime.fromisoformat(start_raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            date_str = dt.strftime("%Y-%m-%d")  # local date from timestamp
        except ValueError:
            continue
        if date_str in day_highs:
            day_highs[date_str] = max(day_highs[date_str], temp_f)
        else:
            day_highs[date_str] = temp_f

    points: list[DataPoint] = []
    summary_parts: list[str] = []

    # Compute local today directly from the city's timezone — avoids the fragile
    # heuristic that checked which of [UTC-1day, UTC-today] appeared in day_highs
    # (which could pick the wrong day if old forecast periods were in the data).
    local_today = datetime.now(city_tz).date()

    for day_offset in range(NWS_HOURLY_FORECAST_DAYS):
        target_date = local_today + timedelta(days=day_offset)
        date_str = target_date.strftime("%Y-%m-%d")
        if date_str not in day_highs:
            continue
        value = day_highs[date_str]

        # as_of = noon in city's local timezone (matches NOAA/Open-Meteo convention)
        as_of = datetime(
            target_date.year, target_date.month, target_date.day,
            12, 0, 0, tzinfo=city_tz,
        ).astimezone(timezone.utc).isoformat()

        label = "today" if day_offset == 0 else f"day+{day_offset}"
        summary_parts.append(f"{label}={value:.0f}°F")

        points.append(DataPoint(
            source="nws_hourly",
            metric=metric,
            value=value,
            unit="°F",
            as_of=as_of,
            metadata={
                "city":            city_name,
                "forecast_date":   date_str,
                "forecast_offset": day_offset,
                "fetched_at":      fetch_wall_time,
            },
        ))

    if summary_parts:
        logging.info("NWS Hourly [%s]: %s", city_name, "  ".join(summary_parts))
    else:
        logging.warning("NWS Hourly: no hourly data for %s", city_name)

    _city_cache[metric] = (now, points)
    return points


async def fetch_city_forecasts(session: aiohttp.ClientSession) -> list[DataPoint]:
    """Fetch NWS hourly forecast daily highs for all tracked Kalshi cities.

    Runs all city fetches concurrently.  Cities that fail are skipped without
    raising — a partial result is still useful.

    Returns:
        List of DataPoints (one per city per forecast day).
    """
    tasks = [
        _fetch_city_hourly(session, metric, city_name, lat, lon, city_tz)
        for metric, (city_name, lat, lon, city_tz) in CITIES.items()
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    data_points: list[DataPoint] = []
    for metric, result in zip(CITIES.keys(), results):
        if isinstance(result, Exception):
            logging.error("NWS Hourly fetch error for %s: %s", metric, result)
        elif result:
            data_points.extend(result)

    return data_points
