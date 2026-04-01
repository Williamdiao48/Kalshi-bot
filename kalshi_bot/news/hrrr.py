"""HRRR-proxy hourly temperature forecast for signal quality gating.

The NWS hourly forecast endpoint (``/forecast/hourly``) uses the
High-Resolution Rapid Refresh (HRRR) model as its primary driver for the
contiguous US, updating every 15–60 minutes.  By comparing its implied
daytime high against the 12-hourly NWS daily forecast (from noaa.py), we
estimate inter-model spread — a proxy for ensemble disagreement that the
deterministic daily forecast alone cannot surface.

Gate logic (applied in main.py)
--------------------------------
    spread = |daily_forecast_high − hourly_hrrr_high|

    If spread >= HRRR_MAX_SPREAD_F  →  suppress the NOAA forecast signal.

``noaa_observed`` DataPoints (station readings) and ``nws_alert`` DataPoints
are NEVER gated — they are ground truth or NWS-issued confidence, not raw
model output.

Shared cache
------------
This module imports ``CITIES``, ``_gridpoint_cache``, ``_resolve_gridpoint``,
and ``_HEADERS`` from noaa.py.  The cache is populated once per process when
noaa.py runs ``fetch_city_forecasts`` — by the time ``fetch_hourly_highs``
is called (both tasks run concurrently in the same gather), gridpoints are
already resolved or will be resolved by the shared ``_resolve_gridpoint``
call here.  Either way, only one HTTP ``/points/{lat},{lon}`` call per city
is ever made per process lifetime.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

import aiohttp

from ..data import DataPoint
from .noaa import CITIES, _gridpoint_cache, _resolve_gridpoint, _HEADERS

# Local-time daytime window used to identify today's peak temperature from
# the hourly forecast.  Periods outside [_DAY_START, _DAY_END) are skipped.
_DAY_START_HOUR: int = 8
_DAY_END_HOUR:   int = 21


def to_data_points(hourly_highs: dict[str, float], as_of: str) -> list[DataPoint]:
    """Convert a fetch_hourly_highs result dict into DataPoints for numeric matching.

    Emits one DataPoint per city with source="hrrr".  These DataPoints flow
    into the forecast consensus and corroboration gates in main.py, so a lone
    NOAA signal that is confirmed by the HRRR hourly model will satisfy
    FORECAST_CORROBORATION_MIN=2 without requiring OWM or Open-Meteo.
    """
    return [
        DataPoint(
            source   = "hrrr",
            metric   = metric,
            value    = high_f,
            unit     = "°F",
            as_of    = as_of,
            metadata = {},
        )
        for metric, high_f in hourly_highs.items()
    ]


async def _fetch_hourly_high(
    session: aiohttp.ClientSession,
    city_name: str,
    hourly_url: str,
) -> float | None:
    """Return today's daytime hourly maximum temperature (°F) from the NWS.

    Fetches the hourly forecast and scans all periods whose ``startTime``
    falls between ``_DAY_START_HOUR`` and ``_DAY_END_HOUR`` in the period's
    own local timezone.  Returns the maximum temperature seen, or None if
    the endpoint is unavailable or returns no matching periods.
    """
    if not hourly_url:
        return None
    try:
        async with session.get(
            hourly_url,
            headers=_HEADERS,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
    except Exception as exc:
        logging.warning("HRRR hourly fetch failed for %s: %s", city_name, exc)
        return None

    max_f: float | None = None

    for period in data.get("properties", {}).get("periods", []):
        try:
            start = datetime.fromisoformat(period["startTime"])
        except (KeyError, ValueError, TypeError):
            continue

        # Filter to periods that fall on today in the period's own local
        # timezone (fromisoformat preserves the UTC offset).  Using the UTC
        # date would incorrectly exclude late-evening periods for cities in
        # negative UTC offsets (e.g. 8 PM ET = 1 AM UTC the next day).
        today_local = datetime.now(start.tzinfo).date() if start.tzinfo else datetime.now(timezone.utc).date()
        if start.date() != today_local:
            continue
        local_hour = start.hour
        if local_hour < _DAY_START_HOUR or local_hour >= _DAY_END_HOUR:
            continue

        unit = period.get("temperatureUnit", "F")
        temp = period.get("temperature")
        if temp is None:
            continue
        temp_f = float(temp) if unit == "F" else float(temp) * 9.0 / 5.0 + 32.0
        if max_f is None or temp_f > max_f:
            max_f = temp_f

    if max_f is not None:
        logging.debug("HRRR [%s]: hourly daytime high = %.1f°F", city_name, max_f)
    return max_f


async def fetch_hourly_highs(
    session: aiohttp.ClientSession,
) -> dict[str, float]:
    """Return today's HRRR-derived daytime high (°F) for each tracked city.

    Re-uses the gridpoint cache populated by noaa.py so no extra
    ``/points/{lat},{lon}`` API calls are made when both modules run in the
    same poll cycle.

    Returns a dict mapping ``metric`` → ``hourly_daytime_high_F``.
    Metrics for which the hourly fetch fails are omitted.
    """
    async def fetch_one(
        metric: str, city_name: str, lat: float, lon: float
    ) -> tuple[str, float | None]:
        gridpoint = await _resolve_gridpoint(session, metric, lat, lon)
        if gridpoint is None:
            return metric, None
        hourly_url = gridpoint.get("forecast_hourly", "")
        high = await _fetch_hourly_high(session, city_name, hourly_url)
        return metric, high

    tasks = [
        fetch_one(metric, city_name, lat, lon)
        for metric, (city_name, lat, lon, *_) in CITIES.items()
    ]
    raw = await asyncio.gather(*tasks, return_exceptions=True)

    hourly_highs: dict[str, float] = {}
    for metric, result in zip(CITIES.keys(), raw):
        if isinstance(result, Exception):
            logging.error("HRRR gather error for %s: %s", metric, result)
        else:
            _, high = result  # type: ignore[misc]
            if high is not None:
                hourly_highs[metric] = high

    if hourly_highs:
        logging.info(
            "HRRR: hourly daytime highs resolved for %d/%d city(ies).",
            len(hourly_highs), len(CITIES),
        )
    return hourly_highs
