"""FAA Aviation Weather METAR real-time temperature fetcher.

Provides faster observed-temperature readings than the NOAA Weather API.
METAR observations are published within 1–2 minutes of the actual station
reading; the NOAA Weather API aggregates them with a 5–10 minute lag.
That head-start is significant when the edge window on a temperature
crossing is 10–30 minutes.

All 9 Kalshi cities are fetched in a single HTTP request using comma-
separated station IDs — one round-trip vs nine for the NOAA approach.

API
---
  GET https://aviationweather.gov/api/data/metar
      ?ids=KLAX,KDEN,KMDW,KNYC,KMIA,KAUS,KDAL,KBOS,KHOU
      &format=json
      &hours={METAR_LOOKBACK_HOURS}

Response: JSON array, one object per observation, with fields:
  icaoId   — ICAO station code (e.g. "KNYC")
  temp     — air temperature in °C  (None when sensor unavailable)
  obsTime  — Unix epoch of the observation

We filter each station's observations to today's local date for that
city, then take the running maximum as the observed daily high —
identical logic to noaa_observed but using a faster data source.

Integration
-----------
Source tag: ``"metar"``

Treated identically to ``"noaa_observed"`` throughout the pipeline:
  • ``_PASS_THROUGH`` in the consensus filter — never blocked by
    forecast corroboration gates.
  • ``_OBS_CONFIRMED`` for obs-consensus pre-pass.
  • ``_obs_value`` lookup for cross-source conflict detection.
  • Same exit thresholds (PT=0.50 YES, SL=0.05 YES, SL=0.60 bare).

Caching
-------
Results are cached for METAR_CACHE_SECONDS (default 90 s).  METAR
stations report every 20–60 minutes, so 90 s is fresh without hammering
the API.  The full 9-station fetch is one HTTP call so the cost is low.

Environment variables
---------------------
  METAR_CACHE_SECONDS    Cache TTL in seconds.  Default: 90.
  METAR_LOOKBACK_HOURS   Hours of METAR history to request.  Default: 24.
                         Must cover the full local day for all US
                         timezones (UTC-5 to UTC-8), so ≥ 24 is safe.
"""

import asyncio
import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timezone

import aiohttp

from ..data import DataPoint
from .noaa import CITIES, KALSHI_STATION_IDS

_BASE_URL = "https://aviationweather.gov/api/data/metar"

METAR_CACHE_SECONDS: int = int(os.environ.get("METAR_CACHE_SECONDS", "30"))
METAR_LOOKBACK_HOURS: int = int(os.environ.get("METAR_LOOKBACK_HOURS", "24"))

# Reverse map: ICAO station ID → metric key (e.g. "KNYC" → "temp_high_ny")
_STATION_TO_METRIC: dict[str, str] = {
    v: k for k, v in KALSHI_STATION_IDS.items()
}

# Module-level cache: (monotonic_ts, list[DataPoint])
_cache_time: float = 0.0
_cache_points: list[DataPoint] = []


def _filter_cache_to_today(now_utc: datetime) -> list[DataPoint]:
    """Return only cached DataPoints still valid for today's local date.

    Called before serving stale cache to prevent yesterday's daily high/low
    from flowing into obs_values after a local midnight rollover.
    Handles both temp_high_* and temp_low_* metrics (CITIES only has temp_high_*).
    """
    valid = []
    for dp in _cache_points:
        lookup_metric = dp.metric.replace("temp_low_", "temp_high_")
        city_entry = CITIES.get(lookup_metric)
        if city_entry is None:
            continue
        _, _, _, city_tz = city_entry
        local_today_str = now_utc.astimezone(city_tz).date().strftime("%Y-%m-%d")
        cached_date = (dp.metadata or {}).get("local_date", "")
        if cached_date == local_today_str:
            valid.append(dp)
    return valid


def _c_to_f(temp_c: float) -> float:
    return temp_c * 9.0 / 5.0 + 32.0


async def fetch_city_forecasts(session: aiohttp.ClientSession) -> list[DataPoint]:
    """Fetch METAR observed daily-high temperatures for all tracked cities.

    Returns one DataPoint per city whose station has at least one valid
    temperature reading for today's local date.  Cities with no data are
    silently skipped — a partial result is still useful.

    Results are cached for METAR_CACHE_SECONDS to avoid hammering the API
    on every 60-second poll cycle.
    """
    global _cache_time, _cache_points

    now = time.monotonic()
    now_utc = datetime.now(timezone.utc)

    if _cache_points and (now - _cache_time) < METAR_CACHE_SECONDS:
        return _cache_points

    station_ids = list(KALSHI_STATION_IDS.values())
    params = {
        "ids":    ",".join(station_ids),
        "format": "json",
        "hours":  str(METAR_LOOKBACK_HOURS),
    }

    try:
        async with session.get(
            _BASE_URL,
            params=params,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            resp.raise_for_status()
            records: list[dict] = await resp.json()
    except Exception as exc:
        logging.warning("METAR fetch failed: %s", exc)
        if _cache_points:
            stale = _filter_cache_to_today(now_utc)
            logging.info(
                "METAR: serving stale cache after fetch failure "
                "(%d/%d cities still valid for local today)",
                len(stale), len(_cache_points),
            )
            return stale
        return _cache_points

    # Group observations by station ID
    by_station: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        station = rec.get("icaoId", "")
        if station:
            by_station[station].append(rec)

    points: list[DataPoint] = []
    summary_parts: list[str] = []

    for station_id, obs_list in by_station.items():
        metric = _STATION_TO_METRIC.get(station_id)
        if metric is None:
            continue

        # Look up city name and local timezone from CITIES
        city_entry = CITIES.get(metric)
        if city_entry is None:
            continue
        city_name, _lat, _lon, city_tz = city_entry

        # Local "today" for this city
        local_today = now_utc.astimezone(city_tz).date()

        # Find the max and min temperatures across all observations for today,
        # and collect the time-series for trajectory projection.
        daily_max_f: float | None = None
        daily_min_f: float | None = None
        obs_today: list[tuple[float, float]] = []  # (epoch_timestamp, temp_f)
        for obs in obs_list:
            temp_c = obs.get("temp")
            if temp_c is None:
                continue
            obs_epoch = obs.get("obsTime")
            if obs_epoch is None:
                continue
            obs_dt = datetime.fromtimestamp(obs_epoch, tz=timezone.utc)
            obs_local_date = obs_dt.astimezone(city_tz).date()
            if obs_local_date != local_today:
                continue
            temp_f = _c_to_f(float(temp_c))
            obs_today.append((float(obs_epoch), temp_f))
            if daily_max_f is None or temp_f > daily_max_f:
                daily_max_f = temp_f
            if daily_min_f is None or temp_f < daily_min_f:
                daily_min_f = temp_f

        if daily_max_f is None:
            logging.debug("METAR: no today observations for %s (%s)", station_id, city_name)
            continue

        obs_today.sort(key=lambda x: x[0])  # ascending by time

        # as_of = current UTC time (real-time observation, not a noon anchor)
        as_of = now_utc.isoformat()
        date_str = local_today.strftime("%Y-%m-%d")

        summary_parts.append(f"{city_name}={daily_max_f:.1f}°F(hi) {daily_min_f:.1f}°F(lo)")

        points.append(DataPoint(
            source="metar",
            metric=metric,
            value=daily_max_f,
            unit="°F",
            as_of=as_of,
            metadata={
                "city":          city_name,
                "station":       station_id,
                "observed_max":  daily_max_f,
                "local_date":    date_str,
                "obs_series":    obs_today,  # list of (epoch, temp_f) for trajectory
            },
        ))

        # Emit daily minimum as temp_low_* DataPoint (same station, same fetch)
        if daily_min_f is not None:
            low_metric = metric.replace("temp_high_", "temp_low_")
            if low_metric != metric:
                points.append(DataPoint(
                    source="metar",
                    metric=low_metric,
                    value=daily_min_f,
                    unit="°F",
                    as_of=as_of,
                    metadata={
                        "city":         city_name,
                        "station":      station_id,
                        "observed_min": daily_min_f,
                        "local_date":   date_str,
                        "obs_series":   obs_today,
                    },
                ))

    if summary_parts:
        logging.info("METAR observed highs: %s", "  ".join(summary_parts))
    else:
        logging.warning("METAR: no valid temperature observations returned")

    if points:
        _cache_time = now
        _cache_points = points
    elif _cache_points:
        stale_age = int(now - _cache_time)
        stale = _filter_cache_to_today(now_utc)
        logging.warning(
            "METAR: all stations returned no data; serving stale cache "
            "(%ds old, %d/%d cities valid for local today)",
            stale_age, len(stale), len(_cache_points),
        )
        _cache_points[:] = stale  # drop expired entries in-place

    return _cache_points
