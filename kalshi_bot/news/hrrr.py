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
# the hourly forecast.  Periods outside [_DAY_START, _DAY_END) are skipped
# for the daily HIGH, but all 24 hours are used for the daily LOW (overnight
# lows occur before 8 AM and would be missed by the daytime window).
_DAY_START_HOUR: int = 8
_DAY_END_HOUR:   int = 21


def to_data_points(
    hourly_highs: dict[str, float],
    as_of: str,
    hourly_lows: dict[str, float] | None = None,
) -> list[DataPoint]:
    """Convert fetch_hourly_temps result dicts into DataPoints for numeric matching.

    Emits one DataPoint per city for highs (source="hrrr", metric="temp_high_*").
    If hourly_lows is provided, also emits one DataPoint per city for lows
    (metric="temp_low_*") so KXLOWT markets have HRRR as a forecast source for
    the contradiction gate and corroboration check in main.py.
    """
    points = [
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
    if hourly_lows:
        for metric, low_f in hourly_lows.items():
            points.append(DataPoint(
                source   = "hrrr",
                metric   = metric,
                value    = low_f,
                unit     = "°F",
                as_of    = as_of,
                metadata = {},
            ))
    return points


async def _fetch_hourly_temps(
    session: aiohttp.ClientSession,
    city_name: str,
    hourly_url: str,
) -> tuple[float | None, float | None]:
    """Return today's daytime high and daily low temperature (°F) from NWS hourly.

    High: maximum across periods in [_DAY_START_HOUR, _DAY_END_HOUR) local time.
    Low:  minimum across ALL 24 hours — overnight lows occur before 8 AM so the
          daytime window would miss them.

    Returns (max_f, min_f); either may be None if data is unavailable.
    """
    if not hourly_url:
        return None, None
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
        return None, None

    max_f: float | None = None
    min_f: float | None = None

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

        unit = period.get("temperatureUnit", "F")
        temp = period.get("temperature")
        if temp is None:
            continue
        temp_f = float(temp) if unit == "F" else float(temp) * 9.0 / 5.0 + 32.0

        # Daily low: track across ALL hours (overnight lows before 8 AM)
        if min_f is None or temp_f < min_f:
            min_f = temp_f

        # Daily high: track only the daytime window (8 AM – 9 PM local)
        local_hour = start.hour
        if local_hour < _DAY_START_HOUR or local_hour >= _DAY_END_HOUR:
            continue
        if max_f is None or temp_f > max_f:
            max_f = temp_f

    if max_f is not None:
        logging.debug("HRRR [%s]: hourly daytime high = %.1f°F", city_name, max_f)
    if min_f is not None:
        logging.debug("HRRR [%s]: daily low = %.1f°F", city_name, min_f)
    return max_f, min_f


async def _fetch_hourly_high(
    session: aiohttp.ClientSession,
    city_name: str,
    hourly_url: str,
) -> float | None:
    """Backward-compatible wrapper — returns only the daytime high."""
    high, _ = await _fetch_hourly_temps(session, city_name, hourly_url)
    return high


async def fetch_hourly_temps(
    session: aiohttp.ClientSession,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return today's HRRR-derived daytime high and daily low (°F) for each city.

    Re-uses the gridpoint cache populated by noaa.py so no extra
    ``/points/{lat},{lon}`` API calls are made when both modules run in the
    same poll cycle.

    Returns:
        (hourly_highs, hourly_lows) — each a dict mapping metric → temp_F.
        Highs use ``temp_high_*`` keys; lows use ``temp_low_*`` keys.
        Metrics for which the fetch fails are omitted from both dicts.
    """
    async def fetch_one(
        metric: str, city_name: str, lat: float, lon: float
    ) -> tuple[str, float | None, float | None]:
        gridpoint = await _resolve_gridpoint(session, metric, lat, lon)
        if gridpoint is None:
            return metric, None, None
        hourly_url = gridpoint.get("forecast_hourly", "")
        high, low = await _fetch_hourly_temps(session, city_name, hourly_url)
        return metric, high, low

    tasks = [
        fetch_one(metric, city_name, lat, lon)
        for metric, (city_name, lat, lon, *_) in CITIES.items()
    ]
    raw = await asyncio.gather(*tasks, return_exceptions=True)

    hourly_highs: dict[str, float] = {}
    hourly_lows:  dict[str, float] = {}
    for metric, result in zip(CITIES.keys(), raw):
        if isinstance(result, Exception):
            logging.error("HRRR gather error for %s: %s", metric, result)
        else:
            _, high, low = result  # type: ignore[misc]
            if high is not None:
                hourly_highs[metric] = high
            if low is not None:
                low_metric = metric.replace("temp_high_", "temp_low_")
                if low_metric != metric:
                    hourly_lows[low_metric] = low

    if hourly_highs:
        logging.info(
            "HRRR: hourly temps resolved for %d/%d city(ies) (highs=%d lows=%d).",
            len(hourly_highs), len(CITIES), len(hourly_highs), len(hourly_lows),
        )
    return hourly_highs, hourly_lows


async def fetch_hourly_highs(
    session: aiohttp.ClientSession,
) -> dict[str, float]:
    """Backward-compatible wrapper — returns only the highs dict."""
    highs, _ = await fetch_hourly_temps(session)
    return highs
