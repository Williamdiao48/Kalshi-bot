"""Open-Meteo standard weather forecast fetcher.

Uses the Open-Meteo Forecast API (free, no API key required) to provide
7-day daily high temperature forecasts for all tracked Kalshi cities.

Switched from the ensemble API (ensemble-api.open-meteo.com) which rate-limits
aggressively (HTTP 429) when 9 cities are fetched every 60 s.  The standard
forecast API (api.open-meteo.com) allows 10,000 req/day with no per-minute
throttle at normal usage.

Tradeoff vs ensemble API
------------------------
The ensemble API returned 31 model members whose spread gave a physics-based
uncertainty estimate.  The standard API returns a single deterministic forecast
value per day.  The spread quality gate (OPEN_METEO_MAX_SPREAD_F) and the
``open_meteo_tight`` high-confidence source are removed — they relied on spread
data that no longer exists.  For the day-2+ forecasts where Open-Meteo is used
as a corroborating source, the spread gate was already bypassed (only today's
forecast day was eligible for ``open_meteo_tight``), so the practical impact is
minimal.

Two DataPoints are emitted per city per forecast day:

  ``source="open_meteo"``
      value = forecast daily maximum temperature (°F).
      metadata["forecast_date"]   = target date (YYYY-MM-DD), used by the
                                    date guard in main.py to align to the
                                    correct Kalshi market.
      metadata["forecast_offset"] = days from today (0 = today, 1 = tomorrow …).

Extended forecast days (OPEN_METEO_FORECAST_DAYS, default 7)
-------------------------------------------------------------
Open-Meteo returns up to 16 daily highs in a single call.  Emitting day-1
through day-6 DataPoints provides corroboration for the NOAA day-2…day-7
extended-forecast signals, which otherwise fail the FORECAST_CORROBORATION_MIN=2
gate (lone signal) and are suppressed.  The date guard in main.py aligns each
DataPoint to the market with the matching resolution date.

API
---
  GET https://api.open-meteo.com/v1/forecast
      ?latitude={lat}&longitude={lon}
      &daily=temperature_2m_max
      &temperature_unit=fahrenheit
      &timezone=UTC
      &forecast_days=7

Response (relevant part)::

    {
      "daily": {
        "time": ["2026-03-21", ..., "2026-03-27"],
        "temperature_2m_max": [75.2, 77.0, 79.1, ...]
      }
    }

Environment variables
---------------------
  OPEN_METEO_FORECAST_DAYS       Number of forecast days to emit (1–16).  Default: 7.
                                 Day 0 = today; days 1–6 provide corroboration for
                                 the NOAA extended-forecast pipeline.
  OPEN_METEO_CACHE_MINUTES       How long to cache blended results.  Default: 10.
  OPEN_METEO_MODELS_CACHE_MINUTES  How long to cache model-specific results.  Default: 30.
                                 Model forecasts update every 1–6 hours; 30 min avoids
                                 excessive API calls (84 requests per cycle) while still
                                 catching meaningful intra-day model updates.
"""

import asyncio
import logging
import os
from ..utils import env_int
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import aiohttp

from ..data import DataPoint
from ..cities import CITIES, LOW_CITIES, CITY_TZ_STRINGS as _CITY_TZ_STRINGS  # same city registry as NOAA

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Number of forecast days to emit (1 = today only, up to 16).
OPEN_METEO_FORECAST_DAYS: int = min(16, max(1, int(
    os.environ.get("OPEN_METEO_FORECAST_DAYS", "7")
)))

# How long to cache blended results before re-fetching (default: 10 minutes).
OPEN_METEO_CACHE_MINUTES: int = env_int("OPEN_METEO_CACHE_MINUTES", 10)

# How long to cache model-specific results (default: 30 minutes).
# Model-specific fetches make 4 × N_cities requests so a longer TTL keeps
# the daily request count well within the free tier (10k/day).
OPEN_METEO_MODELS_CACHE_MINUTES: int = int(
    os.environ.get("OPEN_METEO_MODELS_CACHE_MINUTES", "30")
)

_BASE_URL = "https://api.open-meteo.com/v1/forecast"

# Model-specific sources: (Open-Meteo model name, DataPoint source name).
# These correspond to the models validated in the backtest (scripts/backtest_forecast_accuracy.py).
_FORECAST_MODELS: list[tuple[str, str]] = [
    ("gfs_seamless",  "open_meteo_gfs"),
    ("ecmwf_ifs",     "open_meteo_ecmwf"),
    ("icon_seamless", "open_meteo_icon"),
    ("gem_seamless",  "open_meteo_gem"),
]

# Semaphore to limit concurrent model-specific requests (shared across all cities × models).
_model_sem = asyncio.Semaphore(4)

# _CITY_TZ_STRINGS is imported from noaa (covers both temp_high_* and temp_low_*).


def _warn_tz_fallback(metric: str) -> str:
    """Log a warning and return 'UTC' when a metric has no configured timezone.

    Using UTC as a fallback misaligns forecast dates for most US cities (e.g.,
    treating 11 PM ET as the next UTC day).  This function makes the silent
    fallback visible so the mapping can be added to _CITY_TZ_STRINGS.
    """
    logging.warning(
        "open_meteo: no timezone configured for metric %r — "
        "falling back to UTC; forecast dates may be misaligned",
        metric,
    )
    return "UTC"


# ---------------------------------------------------------------------------
# Module-level caches
# ---------------------------------------------------------------------------
_cache_time: float = 0.0             # monotonic timestamp of last successful blended fetch
_cache_points: list[DataPoint] = []  # last successful blended result set

_models_cache_time: float = 0.0
_models_cache_points: list[DataPoint] = []


# ---------------------------------------------------------------------------
# Per-city fetch
# ---------------------------------------------------------------------------

async def _fetch_city_forecast(
    session:      aiohttp.ClientSession,
    metric:       str,
    city_name:    str,
    lat:          float,
    lon:          float,
    city_tz:      ZoneInfo,
    city_tz_str:  str,
) -> list[DataPoint]:
    """Fetch standard high temperature forecasts for one city across multiple days.

    Emits one DataPoint per forecast day (up to OPEN_METEO_FORECAST_DAYS days).
    Each DataPoint carries ``forecast_date`` and ``forecast_offset`` in metadata
    so the date guard in main.py aligns it to the correct Kalshi market.  The
    ``as_of`` timestamp is noon in the city's local timezone, matching the NOAA
    extended-forecast convention.

    Returns list of DataPoints (may be empty on error).
    """
    params = {
        "latitude":           f"{lat:.4f}",
        "longitude":          f"{lon:.4f}",
        "daily":              "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max",
        "temperature_unit":   "fahrenheit",
        "precipitation_unit": "inch",    # request inches directly; no mm→in conversion needed
        "timezone":           city_tz_str,  # local tz so daily dates match Kalshi settlement
        "forecast_days":      str(OPEN_METEO_FORECAST_DAYS),
    }

    try:
        async with session.get(
            _BASE_URL,
            params=params,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
    except Exception as exc:
        logging.warning("Open-Meteo fetch failed for %s: %s", city_name, exc)
        return []

    daily = data.get("daily", {})
    times: list[str] = daily.get("time", [])
    temps: list = daily.get("temperature_2m_max", [])
    mins:  list = daily.get("temperature_2m_min", [])
    precip_totals: list = daily.get("precipitation_sum", [])
    precip_probs:  list = daily.get("precipitation_probability_max", [])

    if not times or not temps:
        logging.warning("Open-Meteo: empty daily response for %s", city_name)
        return []

    # Find index for today's LOCAL date — anchor to city tz so the date aligns
    # with what Kalshi settles against (local calendar day, not UTC).
    today_str = datetime.now(city_tz).strftime("%Y-%m-%d")
    try:
        today_idx = times.index(today_str)
    except ValueError:
        logging.debug(
            "Open-Meteo: today (%s) not in response for %s — dates=%s",
            today_str, city_name, times,
        )
        return []

    points: list[DataPoint] = []
    summary_parts: list[str] = []

    for day_offset in range(OPEN_METEO_FORECAST_DAYS):
        day_idx = today_idx + day_offset
        if day_idx >= len(times) or day_idx >= len(temps):
            break
        forecast_date_str = times[day_idx]
        temp_val = temps[day_idx]
        if temp_val is None:
            continue

        value = float(temp_val)

        # as_of = noon in the city's local timezone (matches NOAA extended-forecast
        # convention so the date guard in numeric_matcher treats both sources the same).
        # On parse failure skip the point rather than falling back to now() — a bad
        # as_of would fool the date guard into matching this forecast to the wrong market.
        try:
            forecast_noon_local = datetime.strptime(
                forecast_date_str, "%Y-%m-%d"
            ).replace(hour=12, tzinfo=city_tz)
            as_of = forecast_noon_local.astimezone(timezone.utc).isoformat()
        except ValueError:
            logging.warning(
                "open_meteo: unparseable forecast date %r for %s day+%d — skipping point",
                forecast_date_str, metric, day_offset,
            )
            continue

        label = "today" if day_offset == 0 else f"day+{day_offset}"
        summary_parts.append(f"{label}={value:.0f}°F")

        points.append(DataPoint(
            source   = "open_meteo",
            metric   = metric,
            value    = value,
            unit     = "°F",
            as_of    = as_of,
            metadata = {
                "city":            city_name,
                "forecast_date":   forecast_date_str,
                "forecast_offset": day_offset,
            },
        ))

        # Also emit the daily low for the same date.
        if day_idx < len(mins) and mins[day_idx] is not None:
            low_metric = metric.replace("temp_high_", "temp_low_")
            if low_metric != metric:  # only emit when the replacement matched
                points.append(DataPoint(
                    source   = "open_meteo",
                    metric   = low_metric,
                    value    = float(mins[day_idx]),
                    unit     = "°F",
                    as_of    = as_of,
                    metadata = {
                        "city":            city_name,
                        "forecast_date":   forecast_date_str,
                        "forecast_offset": day_offset,
                    },
                ))

        # Emit daily precipitation total (inches).
        if day_idx < len(precip_totals) and precip_totals[day_idx] is not None:
            precip_metric = metric.replace("temp_high_", "precip_total_")
            if precip_metric != metric:
                points.append(DataPoint(
                    source   = "open_meteo",
                    metric   = precip_metric,
                    value    = float(precip_totals[day_idx]),
                    unit     = "in",
                    as_of    = as_of,
                    metadata = {
                        "city":            city_name,
                        "forecast_date":   forecast_date_str,
                        "forecast_offset": day_offset,
                    },
                ))

        # Emit precipitation probability (0.0–100.0 percent).
        if day_idx < len(precip_probs) and precip_probs[day_idx] is not None:
            prob_metric = metric.replace("temp_high_", "precip_prob_")
            if prob_metric != metric:
                points.append(DataPoint(
                    source   = "open_meteo",
                    metric   = prob_metric,
                    value    = float(precip_probs[day_idx]),
                    unit     = "%",
                    as_of    = as_of,
                    metadata = {
                        "city":            city_name,
                        "forecast_date":   forecast_date_str,
                        "forecast_offset": day_offset,
                    },
                ))

    if summary_parts:
        logging.debug("Open-Meteo [%s]: %s", city_name, "  ".join(summary_parts))
    else:
        logging.warning("Open-Meteo: no forecast data for %s", city_name)

    return points


# ---------------------------------------------------------------------------
# Model-specific forecast fetch (GFS, ECMWF, ICON, GEM)
# ---------------------------------------------------------------------------

async def _fetch_city_model_high(
    session:      aiohttp.ClientSession,
    metric:       str,
    city_name:    str,
    lat:          float,
    lon:          float,
    city_tz:      ZoneInfo,
    city_tz_str:  str,
    om_model:     str,
    source_name:  str,
) -> list[DataPoint]:
    """Fetch today's daily-max forecast for one city from one specific model.

    Returns a single-element list on success, empty list on any failure.
    Only today (forecast_offset=0) is emitted — model-specific sources are
    used exclusively for same-day forecast_no signal generation.
    """
    params = {
        "latitude":         f"{lat:.4f}",
        "longitude":        f"{lon:.4f}",
        "daily":            "temperature_2m_max",
        "temperature_unit": "fahrenheit",
        "timezone":         city_tz_str,
        "forecast_days":    "3",   # buffer; we only use index 0 (today)
        "models":           om_model,
    }
    async with _model_sem:
        try:
            async with session.get(
                _BASE_URL,
                params=params,
                timeout=aiohttp.ClientTimeout(total=20),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
        except Exception as exc:
            logging.debug(
                "Open-Meteo model fetch failed %s/%s: %s", city_name, om_model, exc
            )
            return []

    daily = data.get("daily", {})
    times: list[str] = daily.get("time", [])
    temps: list     = daily.get("temperature_2m_max", [])

    if not times or not temps:
        return []

    today_str = datetime.now(city_tz).strftime("%Y-%m-%d")
    try:
        today_idx = times.index(today_str)
    except ValueError:
        return []

    temp_val = temps[today_idx] if today_idx < len(temps) else None
    if temp_val is None:
        return []

    try:
        forecast_noon_local = datetime.strptime(today_str, "%Y-%m-%d").replace(
            hour=12, tzinfo=city_tz
        )
        as_of = forecast_noon_local.astimezone(timezone.utc).isoformat()
    except ValueError:
        return []

    return [DataPoint(
        source   = source_name,
        metric   = metric,
        value    = float(temp_val),
        unit     = "°F",
        as_of    = as_of,
        metadata = {
            "city":            city_name,
            "forecast_date":   today_str,
            "forecast_offset": 0,
            "om_model":        om_model,
        },
    )]


async def _fetch_city_model_low(
    session:      aiohttp.ClientSession,
    metric:       str,
    city_name:    str,
    lat:          float,
    lon:          float,
    city_tz:      ZoneInfo,
    city_tz_str:  str,
    om_model:     str,
    source_name:  str,
) -> list[DataPoint]:
    """Fetch today's GFS daily-min forecast for one low-temp city.

    Mirrors _fetch_city_model_high but requests temperature_2m_min.  Only the
    GFS model is needed — the result feeds the warm-NO clearance gate in
    strike_arb.find_band_arbs(), stored in _gfs_morning_snap via main.py.
    Returns a single-element list on success, empty list on any failure.
    """
    params = {
        "latitude":         f"{lat:.4f}",
        "longitude":        f"{lon:.4f}",
        "daily":            "temperature_2m_min",
        "temperature_unit": "fahrenheit",
        "timezone":         city_tz_str,
        "forecast_days":    "3",
        "models":           om_model,
    }
    async with _model_sem:
        try:
            async with session.get(
                _BASE_URL,
                params=params,
                timeout=aiohttp.ClientTimeout(total=20),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
        except Exception as exc:
            logging.debug(
                "Open-Meteo low model fetch failed %s/%s: %s", city_name, om_model, exc
            )
            return []

    daily = data.get("daily", {})
    times: list[str] = daily.get("time", [])
    mins: list = daily.get("temperature_2m_min", [])

    if not times or not mins:
        return []

    today_str = datetime.now(city_tz).strftime("%Y-%m-%d")
    try:
        today_idx = times.index(today_str)
    except ValueError:
        return []

    min_val = mins[today_idx] if today_idx < len(mins) else None
    if min_val is None:
        return []

    try:
        forecast_noon_local = datetime.strptime(today_str, "%Y-%m-%d").replace(
            hour=12, tzinfo=city_tz
        )
        as_of = forecast_noon_local.astimezone(timezone.utc).isoformat()
    except ValueError:
        return []

    return [DataPoint(
        source   = source_name,
        metric   = metric,
        value    = float(min_val),
        unit     = "°F",
        as_of    = as_of,
        metadata = {
            "city":            city_name,
            "forecast_date":   today_str,
            "forecast_offset": 0,
            "om_model":        om_model,
        },
    )]


async def fetch_model_forecasts(
    session: aiohttp.ClientSession,
) -> list[DataPoint]:
    """Fetch model-specific daily-high forecasts (GFS, ECMWF, ICON, GEM) for all cities.

    Emits DataPoints with source names ``open_meteo_gfs``, ``open_meteo_ecmwf``,
    ``open_meteo_icon``, ``open_meteo_gem`` — one per city per model for today
    (forecast_offset=0 only).

    Also emits ``open_meteo_gfs`` DataPoints for all temp_low_* metrics using
    daily-min temperature, to feed the warm-NO GFS clearance gate in
    strike_arb.find_band_arbs().

    These are used by ``find_forecast_nos()`` in strike_arb.py as additional
    independent corroboration sources, validated by the backtest in
    scripts/backtest_forecast_accuracy.py.

    Cached for OPEN_METEO_MODELS_CACHE_MINUTES (default 30 min) to stay within
    the free-tier request budget (10k/day).
    """
    global _models_cache_time, _models_cache_points

    cache_ttl = OPEN_METEO_MODELS_CACHE_MINUTES * 60
    now = time.monotonic()
    if _models_cache_points and (now - _models_cache_time) < cache_ttl:
        logging.debug(
            "Open-Meteo models: serving cached results (%ds old, TTL=%ds)",
            int(now - _models_cache_time), cache_ttl,
        )
        return _models_cache_points

    tasks = [
        _fetch_city_model_high(
            session, metric, city_name, lat, lon,
            city_tz=city_tz,
            city_tz_str=_CITY_TZ_STRINGS.get(metric) or _warn_tz_fallback(metric),
            om_model=om_model,
            source_name=source_name,
        )
        for metric, (city_name, lat, lon, city_tz) in CITIES.items()
        if metric.startswith("temp_high_")
        for om_model, source_name in _FORECAST_MODELS
    ]

    # GFS daily-min for low-temp cities (warm-NO clearance gate only)
    tasks += [
        _fetch_city_model_low(
            session, metric, city_name, lat, lon,
            city_tz=city_tz,
            city_tz_str=_CITY_TZ_STRINGS.get(metric) or _warn_tz_fallback(metric),
            om_model="gfs_seamless",
            source_name="open_meteo_gfs",
        )
        for metric, (city_name, lat, lon, city_tz) in LOW_CITIES.items()
        if metric.startswith("temp_low_")
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    points: list[DataPoint] = []
    for result in results:
        if isinstance(result, Exception):
            logging.debug("Open-Meteo model fetch exception: %s", result)
        elif result:
            points.extend(result)

    if points:
        _models_cache_time = now
        _models_cache_points = points
        logging.debug(
            "Open-Meteo model forecasts: %d DataPoints (%d city×model fetches)",
            len(points), len(tasks),
        )
    elif _models_cache_points:
        stale_age = int(now - _models_cache_time)
        logging.warning(
            "Open-Meteo models: all fetches failed; serving stale cache (%ds old)",
            stale_age,
        )
        _models_cache_time = now - cache_ttl // 2
        return _models_cache_points

    return points


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def fetch_city_forecasts(
    session: aiohttp.ClientSession,
) -> list[DataPoint]:
    """Fetch standard high temperature forecasts for all tracked Kalshi cities.

    Results are cached for OPEN_METEO_CACHE_MINUTES (default 10 min).
    Forecasts update every 1–6 hours so a 10-minute cache loses no signal.

    Runs all city fetches concurrently.  Cities that fail are skipped without
    raising — a partial result is still useful for the cities that succeed.

    Returns:
        List of DataPoints (one per city per forecast day).
    """
    global _cache_time, _cache_points

    cache_ttl = OPEN_METEO_CACHE_MINUTES * 60
    now = time.monotonic()
    if _cache_points and (now - _cache_time) < cache_ttl:
        logging.debug(
            "Open-Meteo: serving cached results (%ds old, TTL=%ds)",
            int(now - _cache_time), cache_ttl,
        )
        return _cache_points

    tasks = [
        _fetch_city_forecast(
            session, metric, city_name, lat, lon,
            city_tz=city_tz,
            city_tz_str=_CITY_TZ_STRINGS.get(metric) or _warn_tz_fallback(metric),
        )
        for metric, (city_name, lat, lon, city_tz) in CITIES.items()
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    data_points: list[DataPoint] = []
    for metric, result in zip(CITIES.keys(), results):
        if isinstance(result, Exception):
            logging.error("Open-Meteo fetch error for %s: %s", metric, result)
        elif result:
            data_points.extend(result)

    if data_points:
        # Successful (full or partial) fetch — update cache.
        _cache_time = now
        _cache_points = data_points
    elif _cache_points:
        # Complete failure but stale cache exists — serve it and back off
        # for half the TTL before retrying to avoid hammering a down API.
        stale_age = int(now - _cache_time)
        logging.warning(
            "Open-Meteo: all cities failed; serving stale cache (%ds old) "
            "and backing off for %ds",
            stale_age, cache_ttl // 2,
        )
        _cache_time = now - cache_ttl // 2
        return _cache_points

    return data_points
