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
import re
from ..utils import env_int
import time
from collections import defaultdict
from datetime import datetime, time as _dtime, timedelta, timezone
from zoneinfo import ZoneInfo

import aiohttp

from ..data import DataPoint
from ..cities import CITIES, KALSHI_STATION_IDS

_BASE_URL = "https://aviationweather.gov/api/data/metar"
# NWS raw METAR text files — published within ~2 min of observation vs ~20 min for
# the FAA ADDS JSON maxT field.  Used to pull 6-hr synoptic groups (10TTT/20TTT) sooner.
_RAW_METAR_BASE = "https://tgftp.nws.noaa.gov/data/observations/metar/stations"

METAR_CACHE_SECONDS: int = env_int("METAR_CACHE_SECONDS", 30)
METAR_LOOKBACK_HOURS: int = env_int("METAR_LOOKBACK_HOURS", 24)
# Cache raw NWS text slightly shorter than ADDS so every ADDS refresh also re-checks raw.
_RAW_SIX_HR_CACHE_SECS: int = env_int("RAW_METAR_CACHE_SECONDS", 25)

# Reverse map: ICAO station ID → metric key (e.g. "KNYC" → "temp_high_ny")
_STATION_TO_METRIC: dict[str, str] = {
    v: k for k, v in KALSHI_STATION_IDS.items()
}

# NWS resolves markets midnight-to-midnight Local Standard Time (LST) year-round,
# regardless of DST.  During summer, DST-aware local midnight is 1 h earlier than
# NWS midnight, creating a gap window where the bot would include readings that
# NWS attributes to the previous calendar day.  We fix this by always bucketing
# observations using the standard-time UTC offset (no DST adjustment).
def _lst_tz(city_tz: ZoneInfo) -> timezone:
    """Return a fixed-offset timezone matching the city's standard time (no DST)."""
    std_offset = city_tz.utcoffset(datetime(2000, 1, 15))  # January = standard time
    return timezone(std_offset)

# Module-level cache: (monotonic_ts, list[DataPoint])
_cache_time: float = 0.0
_cache_points: list[DataPoint] = []

# Staging buffer for individual METAR observations from the last fresh API fetch.
# Populated inside fetch_city_forecasts() on cache misses; consumed by take_obs_rows().
# Each entry: (station, metric, obs_at_iso, temp_f)
_pending_obs_rows: list[tuple[str, str, str, float]] = []

# Per-station raw NWS text cache: station_id → (monotonic_ts, obs_utc|None, max_c|None, min_c|None)
_raw_six_hr_cache: dict[str, tuple[float, "datetime | None", "float | None", "float | None"]] = {}


def take_obs_rows() -> list[tuple[str, str, str, float]]:
    """Return individual METAR observation rows from the last fresh API fetch and clear the buffer.

    Called by main.py after fetch_city_forecasts() to retrieve rows for
    metar_obs_log.  Returns an empty list on cache hits or if no fresh data
    is available.
    """
    rows = _pending_obs_rows[:]
    _pending_obs_rows.clear()
    return rows


def _parse_raw_six_hr_temps(
    text: str,
) -> tuple["datetime | None", "float | None", "float | None"]:
    """Parse a NWS raw METAR text file into (obs_utc, max_c, min_c).

    File format (two lines):
        YYYY/MM/DD HH:MM
        KXXX DDHHMMZ ... RMK ... 10TTT 20TTT ...

    Remarks groups 10TTT / 11TTT (6-hr max) and 20TTT / 21TTT (6-hr min)
    encode temperature in tenths of °C; second digit is sign (0=+, 1=-).
    Returns (None, None, None) if the header cannot be parsed.
    max_c / min_c are None when the group is absent (e.g. non-synoptic obs).
    """
    lines = text.strip().splitlines()
    if len(lines) < 2:
        return None, None, None
    try:
        obs_utc = datetime.strptime(lines[0].strip(), "%Y/%m/%d %H:%M").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None, None, None

    metar_line = lines[1]
    rmk_idx = metar_line.find(" RMK ")
    if rmk_idx == -1:
        return obs_utc, None, None

    max_c = min_c = None
    for token in metar_line[rmk_idx + 5 :].split():
        if re.fullmatch(r"1[01]\d{3}", token):
            sign = int(token[1])
            max_c = int(token[2:]) / 10.0 * (-1 if sign else 1)
        elif re.fullmatch(r"2[01]\d{3}", token):
            sign = int(token[1])
            min_c = int(token[2:]) / 10.0 * (-1 if sign else 1)

    return obs_utc, max_c, min_c


async def _fetch_raw_six_hr(
    session: aiohttp.ClientSession,
    station_id: str,
) -> tuple["datetime | None", "float | None", "float | None"]:
    """Fetch and parse the NWS raw METAR text for a single station.

    Returns (obs_utc, max_c, min_c); None values when unavailable or when the
    observation is not a synoptic report.  Cached per station for
    _RAW_SIX_HR_CACHE_SECS seconds.
    """
    now = time.monotonic()
    cached = _raw_six_hr_cache.get(station_id)
    if cached is not None and (now - cached[0]) < _RAW_SIX_HR_CACHE_SECS:
        return cached[1], cached[2], cached[3]

    url = f"{_RAW_METAR_BASE}/{station_id}.TXT"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=8)) as resp:
            if resp.status != 200:
                return None, None, None
            text = await resp.text()
    except Exception:
        return None, None, None

    obs_utc, max_c, min_c = _parse_raw_six_hr_temps(text)
    _raw_six_hr_cache[station_id] = (now, obs_utc, max_c, min_c)
    return obs_utc, max_c, min_c


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
        local_today_str = now_utc.astimezone(_lst_tz(city_tz)).date().strftime("%Y-%m-%d")
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

    _pending_obs_rows.clear()

    station_ids = list(KALSHI_STATION_IDS.values())
    params = {
        "ids":    ",".join(station_ids),
        "format": "json",
        "hours":  str(METAR_LOOKBACK_HOURS),
    }

    # Run ADDS JSON fetch and per-station raw NWS text fetches concurrently.
    # Raw text (tgftp.nws.noaa.gov) typically publishes 6-hr synoptic groups
    # within ~2-5 min of the observation vs ~20 min for the ADDS JSON maxT field.
    async def _do_adds() -> tuple[list[dict], datetime]:
        async with session.get(
            _BASE_URL,
            params=params,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            resp.raise_for_status()
            data: list[dict] = await resp.json()
        return data, datetime.now(timezone.utc)

    _gather = await asyncio.gather(
        _do_adds(),
        *[_fetch_raw_six_hr(session, sid) for sid in station_ids],
        return_exceptions=True,
    )
    _adds_result = _gather[0]
    if isinstance(_adds_result, Exception):
        logging.warning("METAR fetch failed: %s", _adds_result)
        if _cache_points:
            stale = _filter_cache_to_today(now_utc)
            logging.debug(
                "METAR: serving stale cache after fetch failure "
                "(%d/%d cities still valid for local today)",
                len(stale), len(_cache_points),
            )
            return stale
        return _cache_points
    records, fetch_wall_utc = _adds_result

    # Build raw six-hr dict: station → (obs_utc, max_c, min_c)
    _raw_six_hr: dict[str, tuple[datetime | None, float | None, float | None]] = {}
    for _sid, _res in zip(station_ids, _gather[1:]):
        if not isinstance(_res, Exception):
            _raw_six_hr[_sid] = _res  # type: ignore[assignment]

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
        lst = _lst_tz(city_tz)

        # Local "today" using NWS Local Standard Time (LST) — no DST adjustment.
        # NWS resolves midnight-to-midnight LST year-round, so we must use the
        # same boundary or readings in the DST gap hour count as "today" for us
        # but "yesterday" for NWS settlement.
        local_today = now_utc.astimezone(lst).date()

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
            obs_local_date = obs_dt.astimezone(lst).date()
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

        # 6-hour period max/min from ADDS API synoptic reports (maxT/minT fields).
        # These cover temperature extremes between hourly METAR readings — more
        # accurate than the running max for cases where a peak occurs mid-hour.
        # Only populated on synoptic-hour observations (00, 06, 12, 18 UTC).
        #
        # IMPORTANT: Each synoptic report's maxT/minT covers the *prior* 6 hours
        # (period = [obsTime−6h, obsTime]).  The first report whose obsTime is
        # "local today" has a period that starts in yesterday evening local time
        # (e.g. 12:00 UTC → 04:00 PST "today", but period starts at 06:00 UTC =
        # 22:00 PST yesterday).  Including that report's maxT would import a stale
        # high from the previous evening — exactly what triggered trade #154
        # (KXHIGHTSEA T59: Seattle was 59°F at 10 PM May 12; that maxT appeared
        # in the 12Z report and overrode the 4 AM running max of 52°F).
        # Fix: require BOTH obsTime and (obsTime − 6 h) to fall on local_today,
        # so only fully-intraday synoptic periods are used.
        six_hr_highs_f = [
            round(float(obs["maxT"]) * 9 / 5 + 32, 2)
            for obs in obs_list
            if obs.get("maxT") is not None
            and datetime.fromtimestamp(obs["obsTime"], tz=timezone.utc).astimezone(lst).date() == local_today
            and datetime.fromtimestamp(obs["obsTime"] - 6 * 3600, tz=timezone.utc).astimezone(lst).date() == local_today
        ]
        six_hr_max_f: float | None = max(six_hr_highs_f) if six_hr_highs_f else None

        six_hr_lows_f = [
            round(float(obs["minT"]) * 9 / 5 + 32, 2)
            for obs in obs_list
            if obs.get("minT") is not None
            and datetime.fromtimestamp(obs["obsTime"], tz=timezone.utc).astimezone(lst).date() == local_today
            and datetime.fromtimestamp(obs["obsTime"] - 6 * 3600, tz=timezone.utc).astimezone(lst).date() == local_today
        ]
        six_hr_min_f: float | None = min(six_hr_lows_f) if six_hr_lows_f else None

        # Overlay raw NWS text 6-hr groups — same data, but published ~2-5 min
        # after the observation vs ~20 min for the FAA ADDS JSON maxT field.
        # Apply the same date guard (both obsTime and obsTime-6h on local_today).
        _raw = _raw_six_hr.get(station_id)
        if _raw:
            _raw_obs_utc, _raw_max_c, _raw_min_c = _raw
            if _raw_obs_utc is not None:
                _raw_local = _raw_obs_utc.astimezone(lst).date()
                _raw_m6h_local = (_raw_obs_utc - timedelta(hours=6)).astimezone(lst).date()
                if _raw_local == local_today and _raw_m6h_local == local_today:
                    if _raw_max_c is not None:
                        _raw_max_f = round(_raw_max_c * 9 / 5 + 32, 2)
                        if six_hr_max_f is None or _raw_max_f > six_hr_max_f:
                            logging.info(
                                "METAR raw %s: 6hr max %.2f°F (ADDS JSON had %s)",
                                station_id,
                                _raw_max_f,
                                f"{six_hr_max_f:.2f}°F" if six_hr_max_f is not None else "none",
                            )
                            six_hr_max_f = _raw_max_f
                    if _raw_min_c is not None:
                        _raw_min_f = round(_raw_min_c * 9 / 5 + 32, 2)
                        if six_hr_min_f is None or _raw_min_f < six_hr_min_f:
                            six_hr_min_f = _raw_min_f

        # Anchor as_of to noon LST on the observation date — same convention as
        # forecast sources — so numeric_matcher's DateGuard agrees on which market
        # day this belongs to regardless of DST.  Using now_utc caused a 1-hour
        # window after midnight CDT (but before midnight LST) where May N data
        # was matched against the May N+1 market.
        as_of = datetime.combine(local_today, _dtime(12, 0), tzinfo=lst).isoformat()
        date_str = local_today.strftime("%Y-%m-%d")

        # Log the most recent observation timestamp (what the API claims) AND
        # the wall-clock time we received the HTTP response (when we actually
        # had the data).  The gap between the two is FAA ADDS ingest delay —
        # e.g. "obs=22:53Z delivered=23:54Z" would have flagged trade #117.
        _latest_epoch = obs_today[-1][0] if obs_today else None
        if _latest_epoch is not None:
            _latest_dt = datetime.fromtimestamp(_latest_epoch, tz=timezone.utc)
            _obs_label = (
                f"obs={_latest_dt.strftime('%H:%MZ')}"
                f" delivered={fetch_wall_utc.strftime('%H:%M:%SZ')}"
            )
        else:
            _obs_label = "obs=? delivered=?"
        _six_max_label = f"6hrHi={six_hr_max_f:.1f}°F" if six_hr_max_f is not None else "6hrHi=none"
        _six_min_label = f"6hrLo={six_hr_min_f:.1f}°F" if six_hr_min_f is not None else "6hrLo=none"
        summary_parts.append(
            f"{city_name}={daily_max_f:.1f}°F(hi) {daily_min_f:.1f}°F(lo)"
            f" {_six_max_label} {_six_min_label} [{_obs_label}]"
        )

        # Stage individual observations for metar_obs_log (consumed by take_obs_rows).
        for obs_epoch, obs_temp_f in obs_today:
            obs_at_iso = datetime.fromtimestamp(obs_epoch, tz=timezone.utc).isoformat()
            _pending_obs_rows.append((station_id, metric, obs_at_iso, obs_temp_f))

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
                "six_hr_max_f":  six_hr_max_f,  # 6-hr synoptic max (None if not reported)
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
                        "city":           city_name,
                        "station":        station_id,
                        "observed_min":   daily_min_f,
                        "current_temp_f": obs_today[-1][1],  # most recent reading
                        "local_date":     date_str,
                        "obs_series":     obs_today,
                        "six_hr_min_f":   six_hr_min_f,  # 6-hr synoptic min (None if not reported)
                    },
                ))

    if summary_parts:
        logging.debug("METAR observed highs: %s", "  ".join(summary_parts))
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
