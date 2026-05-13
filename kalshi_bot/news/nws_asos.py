"""NWS ASOS station observations — 5-minute cadence.

Supplements metar.py with raw sensor data from the NWS observation API,
which updates every ~5 minutes vs the METAR schedule of every 20–60 minutes.
The NWS also publishes a synchronized observation at :53 of each hour.

Root cause for this module: trade #117 (KXHIGHTMIN-26MAY11-B67.5) suffered
a catastrophic loss because the FAA METAR for KMSP had not refreshed when the
sensor ticked from 68°F to 69°F at 23:53 UTC.  The bot saw the stale 68°F
value, misread the crashed price as a band-arb YES, and entered as the market
settled at 0¢.  The NWS ASOS API would have reported the 69°F reading within
5 minutes of the sensor tick.

API
---
  GET https://api.weather.gov/stations/{ICAO}/observations?limit=12
  Headers:
    User-Agent: (KalshiBot/1.0, williamdiao32@g.ucla.edu)
    Accept: application/geo+json

Response: GeoJSON FeatureCollection.  Each Feature's ``properties`` contains:
  timestamp                 — ISO 8601 UTC string
  temperature.value         — Celsius float or null
  maxTemperatureLast6Hours  — Celsius float or null (at synoptic hours)
  minTemperatureLast6Hours  — Celsius float or null (at synoptic hours)

Observations are returned newest-first.  ``limit=12`` covers ~1 hour
(12 × 5-min intervals).

Integration
-----------
Source tag: ``"nws_asos"``

Treated identically to ``"metar"`` throughout the pipeline:
  • ``_PASS_THROUGH`` in numeric_matcher consensus filter.
  • ``_OBS_CONFIRMED`` for obs-consensus pre-pass.
  • ``_CONTRARIAN_EXEMPT`` in main.py.
  • ``_OBS_OVER_SRCS`` cap in main.py.
  • HRRR observed-exceeds gate in main.py.
  • Same KXLOWT source guards in numeric_matcher.py.

In the fast loop, obs_values are merged with METAR by taking the
max for highs and min for lows — whichever source has the fresher
(higher/lower) reading wins.

Caching
-------
Per-station cache with TTL of NWS_ASOS_CACHE_SECONDS (default 240 s).
At :53–:54 of each hour, ``should_force_refresh()`` returns True so the
caller can bypass the cache to catch the synchronized METAR-aligned obs.

Environment variables
---------------------
  NWS_ASOS_ENABLED        Set to "false" to disable (default: true).
  NWS_ASOS_CACHE_SECONDS  Per-station cache TTL.  Default: 240 (4 min).
"""

import asyncio
import logging
import os
import time
from datetime import datetime, time as _dtime, timezone
from zoneinfo import ZoneInfo

import aiohttp

from ..data import DataPoint
from ..cities import CITIES, KALSHI_STATION_IDS
from ..utils import env_int

_ASOS_BASE = "https://api.weather.gov/stations/{icao}/observations"

_HEADERS = {
    "User-Agent": "(KalshiBot/1.0, williamdiao32@g.ucla.edu)",
    "Accept": "application/geo+json",
}

NWS_ASOS_ENABLED: bool = os.environ.get("NWS_ASOS_ENABLED", "true").lower() not in ("false", "0", "no")
NWS_ASOS_CACHE_SECONDS: int = env_int("NWS_ASOS_CACHE_SECONDS", 240)

# Per-station cache: icao → (monotonic_fetch_time, list[DataPoint])
_station_cache: dict[str, tuple[float, list[DataPoint]]] = {}

# Reverse map: ICAO → metric key (same as metar.py builds internally)
_STATION_TO_METRIC: dict[str, str] = {v: k for k, v in KALSHI_STATION_IDS.items()}


def _lst_tz(city_tz: ZoneInfo) -> timezone:
    """Fixed-offset timezone for NWS Local Standard Time (no DST)."""
    std_offset = city_tz.utcoffset(datetime(2000, 1, 15))
    return timezone(std_offset)


def _c_to_f(temp_c: float) -> float:
    return temp_c * 9.0 / 5.0 + 32.0


def should_force_refresh(now: datetime | None = None) -> bool:
    """Return True during the :53–:54 window of each hour.

    NWS publishes a synchronized METAR-aligned observation at :53 each hour.
    Bypassing the cache at this moment ensures the fast loop catches it
    within one 10-second cycle after publication.
    """
    t = now or datetime.now(timezone.utc)
    return t.minute in (53, 54)


async def _fetch_station(
    session: aiohttp.ClientSession,
    icao: str,
    metric_key: str,
    city_tz: ZoneInfo,
    now_utc: datetime,
) -> list[DataPoint]:
    """Fetch one station's ASOS observations and return running-max DataPoints.

    Returns [] on any error — callers gracefully fall back to METAR.
    """
    lst = _lst_tz(city_tz)
    local_today = now_utc.astimezone(lst).date()

    url = _ASOS_BASE.format(icao=icao)
    try:
        async with session.get(
            url,
            params={"limit": "12"},
            headers=_HEADERS,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status == 429:
                logging.warning("NWS ASOS: rate-limited (429) for %s — skipping.", icao)
                return []
            resp.raise_for_status()
            data = await resp.json(content_type=None)
        # Wall-clock time the HTTP response was fully received.
        fetch_wall_utc = datetime.now(timezone.utc)
    except aiohttp.ClientResponseError as exc:
        logging.debug("NWS ASOS HTTP %s for %s: %s", exc.status, icao, exc.message)
        return []
    except Exception as exc:
        logging.debug("NWS ASOS fetch error for %s: %s", icao, exc)
        return []

    features = data.get("features", [])
    if not features:
        logging.debug("NWS ASOS: no features returned for %s", icao)
        return []

    city_entry = CITIES.get(metric_key)
    if city_entry is None:
        return []
    city_name = city_entry[0]

    daily_max_f: float | None = None
    daily_min_f: float | None = None
    latest_ts: str | None = None
    six_hr_max_f: float | None = None
    six_hr_min_f: float | None = None
    obs_count = 0
    # Track integer-Celsius and precise-Celsius maxima separately.
    # 5-minute automated readings return whole-number Celsius (±0.5°C = ±0.9°F
    # uncertainty).  The :53 METAR-synced reading uses 0.1°C precision.
    # Integer readings must go through the synoptic_celsius path in find_band_arbs
    # rather than obs_values to avoid false NO signals near band ceilings.
    daily_max_celsius_int: int | None = None   # running max from integer °C readings
    daily_max_f_precise: float | None = None   # running max from decimal °C readings

    for feat in features:
        props = feat.get("properties", {})

        ts_str = props.get("timestamp", "")
        try:
            obs_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            continue

        obs_local_date = obs_dt.astimezone(lst).date()
        if obs_local_date != local_today:
            continue

        temp_obj = props.get("temperature") or {}
        temp_c = temp_obj.get("value")
        if temp_c is None:
            continue

        temp_c_f = float(temp_c)
        temp_f = _c_to_f(temp_c_f)
        obs_count += 1

        if latest_ts is None:
            latest_ts = ts_str

        if daily_max_f is None or temp_f > daily_max_f:
            daily_max_f = temp_f
        if daily_min_f is None or temp_f < daily_min_f:
            daily_min_f = temp_f

        # Classify: integer Celsius (5-min automated) vs decimal (METAR-synced :53)
        if abs(temp_c_f - round(temp_c_f)) < 0.01:
            int_c = round(temp_c_f)
            if daily_max_celsius_int is None or int_c > daily_max_celsius_int:
                daily_max_celsius_int = int_c
        else:
            if daily_max_f_precise is None or temp_f > daily_max_f_precise:
                daily_max_f_precise = temp_f

        # Synoptic 6-hour extremes (populated at the :00 mark of certain hours)
        max6_obj = props.get("maxTemperatureLast6Hours") or {}
        max6_c = max6_obj.get("value")
        if max6_c is not None:
            max6_f = _c_to_f(float(max6_c))
            if six_hr_max_f is None or max6_f > six_hr_max_f:
                six_hr_max_f = max6_f

        min6_obj = props.get("minTemperatureLast6Hours") or {}
        min6_c = min6_obj.get("value")
        if min6_c is not None:
            min6_f = _c_to_f(float(min6_c))
            if six_hr_min_f is None or min6_f < six_hr_min_f:
                six_hr_min_f = min6_f

    if daily_max_f is None:
        logging.debug("NWS ASOS: no today observations for %s (%s)", icao, city_name)
        return []

    # Anchor as_of to noon LST — same convention as metar.py
    as_of = datetime.combine(local_today, _dtime(12, 0), tzinfo=lst).isoformat()
    date_str = local_today.strftime("%Y-%m-%d")

    # Log the most recent observation timestamp (what the API claims) and the
    # wall-clock delivery time (when we actually received the HTTP response).
    # The gap between the two is real API ingest/delivery latency.
    if latest_ts:
        try:
            _asos_obs_ts = latest_ts[:16].replace("T", " ").rstrip("+").rstrip("0") + "Z"
        except Exception:
            _asos_obs_ts = latest_ts[:16]
    else:
        _asos_obs_ts = "?"
    logging.debug(
        "NWS ASOS %s (%s): high=%.1f°F  obs=%d  [obs=%s delivered=%s]",
        icao, city_name, daily_max_f, obs_count,
        _asos_obs_ts, fetch_wall_utc.strftime("%H:%M:%SZ"),
    )

    # Only emit temp_high_* DataPoints.  With limit=12 (~60 min of history) the
    # running minimum across the window is not the day's true overnight low —
    # it reflects only the last hour of readings.  The fast loop merge uses
    # min() so a stale high ASOS value can't worsen the METAR running minimum.
    # But an inflated temp_low DataPoint in the slow loop (numeric_matcher path)
    # could generate wrong NO signals.  METAR (24-hr lookback) is the correct
    # source for daily running minimums; nws_asos supplements only HIGH peaks.
    #
    # Celsius precision metadata:
    #   synoptic_celsius_max — integer °C running max (from 5-min automated readings).
    #     Callers must route this through the synoptic_celsius path in find_band_arbs
    #     (which uses F_low = (C−0.5)×1.8+32 to account for ±0.9°F uncertainty)
    #     rather than obs_values, to avoid false NO signals near band ceilings.
    #   precise_max_f — running max from 0.1°C-precision readings (:53 METAR-synced).
    #     Safe to add to obs_values directly.  None when only integer readings exist.
    return [
        DataPoint(
            source="nws_asos",
            metric=metric_key,
            value=daily_max_f,
            unit="°F",
            as_of=as_of,
            metadata={
                "city":                 city_name,
                "station":              icao,
                "observed_max":         daily_max_f,
                "local_date":           date_str,
                "obs_count":            obs_count,
                "six_hr_max_f":         six_hr_max_f,
                "synoptic_celsius_max": daily_max_celsius_int,
                "precise_max_f":        daily_max_f_precise,
            },
        )
    ]


async def fetch_city_observations(
    session: aiohttp.ClientSession,
    *,
    force: bool = False,
) -> list[DataPoint]:
    """Fetch NWS ASOS observations for all tracked Kalshi temperature cities.

    All stations are fetched concurrently.  Per-station results are cached
    for NWS_ASOS_CACHE_SECONDS.  Pass ``force=True`` to bypass the cache
    (used at :53 of each hour to catch the synchronized NWS observation).

    Returns [] if NWS_ASOS_ENABLED is false or all stations fail.
    """
    if not NWS_ASOS_ENABLED:
        return []

    now = time.monotonic()
    now_utc = datetime.now(timezone.utc)

    tasks = []
    cached_results: list[DataPoint] = []

    for metric_key, icao in KALSHI_STATION_IDS.items():
        cached = _station_cache.get(icao)
        if not force and cached is not None and (now - cached[0]) < NWS_ASOS_CACHE_SECONDS:
            cached_results.extend(cached[1])
            continue
        city_entry = CITIES.get(metric_key)
        if city_entry is None:
            continue
        city_tz = city_entry[3]
        tasks.append((icao, metric_key, city_tz))

    if not tasks:
        return cached_results

    async def _fetch_and_cache(icao: str, metric_key: str, city_tz: ZoneInfo) -> list[DataPoint]:
        pts = await _fetch_station(session, icao, metric_key, city_tz, now_utc)
        _station_cache[icao] = (now, pts)
        return pts

    results = await asyncio.gather(
        *[_fetch_and_cache(icao, mkey, ctz) for icao, mkey, ctz in tasks],
        return_exceptions=True,
    )

    fresh_points: list[DataPoint] = []
    for r in results:
        if isinstance(r, Exception):
            logging.debug("NWS ASOS gather error: %s", r)
        elif isinstance(r, list):
            fresh_points.extend(r)

    all_points = cached_results + fresh_points
    logging.debug("NWS ASOS: %d DataPoint(s) across %d station(s).", len(all_points), len(KALSHI_STATION_IDS))
    return all_points
