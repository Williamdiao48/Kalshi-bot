"""NOAA/NWS weather forecast + observed temperature fetcher.

Fetches three data streams per city:

  1. FORECAST       — NWS daily forecast high for today (Day 1).
  2. FORECAST_D2–D7 — NWS daily forecast highs for days 2–7 (tomorrow through
                      6 days out), emitted as source="noaa_day2" … "noaa_day7".
  3. OBSERVED       — Maximum temperature actually recorded at the nearest NWS
                      observation station since midnight UTC today.

The effective value returned is ``max(forecast_high, observed_max)``.
Because the day's high can never be *less* than what has already been recorded,
the observed max provides a hard lower bound on the outcome.

When the observed max is the dominant value (i.e. real readings are at or above
the forecast), the DataPoint source is set to ``"noaa_observed"`` instead of
``"noaa"``.  The trade executor uses this to apply a much tighter uncertainty
model (σ ≈ 0.5 °F vs 4 °F for a raw forecast), which greatly increases
confidence in the implied probability and helps the market-disagreement filter
surface genuine mispricings late in the trading day.

API flow
--------
1. GET https://api.weather.gov/points/{lat},{lon}
   → properties.forecast          (daily forecast URL)
   → properties.observationStations (nearby stations list URL)

2. GET {observationStations}
   → features[0].properties.stationIdentifier  (nearest station)
   → builds https://api.weather.gov/stations/{id}/observations

3. GET {forecast_url}          → today's forecast high
4. GET {observations_url}?start={midnight_utc}&limit=100
   → max of all temperature.value readings converted from °C to °F
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")
_CT = ZoneInfo("America/Chicago")
_MT = ZoneInfo("America/Denver")
_PT = ZoneInfo("America/Los_Angeles")
_PHX = ZoneInfo("America/Phoenix")    # no DST observed in Arizona

import aiohttp

from ..data import DataPoint

# NWS requires a descriptive User-Agent or requests get rejected
_HEADERS = {"User-Agent": "kalshi-bot/1.0 (educational; contact: user@example.com)"}

# Cities with Kalshi KXHIGH* temperature markets.
# metric key → (display name, latitude, longitude)
#
# Verified 2026-03-09 by exhaustive probe of all plausible KXHIGH{CODE}
# ticker prefixes across 30-day horizon — Kalshi currently offers daily
# high-temp markets only for these 9 cities.  DAL/BOS/HOU are inactive
# at time of writing but included so forecasts are ready if they return.
# To add a new city: add a row here AND add the ticker prefix mapping in
# market_parser.py → TICKER_TO_METRIC and _NUMERIC_PATTERN_PREFIXES.
CITIES: dict[str, tuple[str, float, float, ZoneInfo]] = {
    # Coordinates are the official NWS ASOS airport stations that Kalshi
    # uses for temperature market settlement.  Using city-centre coordinates
    # caused station mismatch (e.g. downtown LA vs. coastal KLAX), producing
    # systematic forecast errors of 3–8°F.
    # Timezone is used to compute local midnight for the observation window
    # (NWS API start= parameter) so the full calendar day's readings are
    # captured, not just the hours since midnight UTC.
    #
    # Consistently active (confirmed 2026-03-09)
    "temp_high_lax": ("Los Angeles", 33.9425, -118.4081, _PT),  # KLAX airport
    "temp_high_den": ("Denver",      39.8561, -104.6737, _MT),  # KDEN airport
    "temp_high_chi": ("Chicago",     41.7868,  -87.7522, _CT),  # KMDW Midway
    "temp_high_ny":  ("New York",    40.7789,  -73.9692, _ET),  # KNYC Central Park
    "temp_high_mia": ("Miami",       25.7959,  -80.2870, _ET),  # KMIA airport
    "temp_high_aus": ("Austin",      30.1975,  -97.6664, _CT),  # KAUS airport
    # Previously inactive; kept for old KXHIGHDAL series (Love Field settlement)
    "temp_high_dal": ("Dallas",      32.8479,  -96.8514, _CT),  # KDAL Love Field
    "temp_high_bos": ("Boston",      42.3643,  -71.0052, _ET),  # KBOS airport
    "temp_high_hou": ("Houston",     29.6454,  -95.2789, _CT),  # KHOU Hobby
    # KXHIGHTDAL settles against DFW (confirmed by rules_primary), not Love Field
    "temp_high_dfw": ("Dallas/Fort Worth", 32.8998,  -97.0403, _CT),  # KDFW
    # New cities — active from 2026-04
    "temp_high_sfo": ("San Francisco", 37.6190, -122.3750, _PT),   # KSFO
    "temp_high_sea": ("Seattle",       47.4502, -122.3088, _PT),   # KSEA
    "temp_high_phx": ("Phoenix",       33.4373, -112.0078, _PHX),  # KPHX (no DST)
    "temp_high_phl": ("Philadelphia",  39.8729,  -75.2437, _ET),   # KPHL
    "temp_high_atl": ("Atlanta",       33.6407,  -84.4277, _ET),   # KATL
    "temp_high_msp": ("Minneapolis",   44.8848,  -93.2223, _CT),   # KMSP
    "temp_high_dca": ("Washington DC", 38.8512,  -77.0402, _ET),   # KDCA
    "temp_high_las": ("Las Vegas",     36.0840, -115.1537, _PT),   # KLAS
    "temp_high_okc": ("Oklahoma City", 35.3931,  -97.6007, _CT),   # KOKC
    "temp_high_sat": ("San Antonio",   29.5337,  -98.4698, _CT),   # KSAT
    "temp_high_msy": ("New Orleans",   29.9934,  -90.2580, _CT),   # KMSY
}

# Derived look-ups used by open_meteo.py, backtest scripts, and audit scripts.
CITY_TZ: dict[str, ZoneInfo] = {k: v[3] for k, v in CITIES.items()}
CITY_TZ_STRINGS: dict[str, str] = {k: str(v[3]) for k, v in CITIES.items()}
# Include daily-low variants (same timezone as the corresponding daily-high city).
CITY_TZ_STRINGS.update({
    k.replace("temp_high_", "temp_low_"): tz
    for k, tz in list(CITY_TZ_STRINGS.items())
    if k.startswith("temp_high_")
})

# ---------------------------------------------------------------------------
# Per-city seasonal sigma (NWS day-1 MAE, °F)
# ---------------------------------------------------------------------------
#
# Source: NWS NDFD verification data and published regional climate literature.
# Each tuple contains 12 values (Jan … Dec); index = month − 1.
# These replace the flat 4°F global default with city/season-specific values:
#
#   Coastal / tropical cities (marine stabilisation → lower σ):
#     LAX  — persistent marine layer; σ ≈ 2.5–3.5°F
#     MIA  — subtropical, small annual swing; σ ≈ 2.5–3.2°F
#     HOU  — Gulf-coast humid, moderate; σ ≈ 3.0–4.5°F
#
#   Continental / high-variability cities (frontal passages → higher σ):
#     DEN  — high elevation, rapid pressure changes; σ ≈ 3.5–6.5°F
#     CHI  — Great-Lakes amplification, active storm track; σ ≈ 3.0–5.5°F
#     DAL  — continental but fast-moving fronts; σ ≈ 3.5–5.5°F
#
#   Mixed cities:
#     NY   — coastal moderating but nor'easters; σ ≈ 3.0–5.0°F
#     BOS  — storm-track coastal; σ ≈ 3.0–5.0°F
#     AUS  — semi-arid, variable spring/fall; σ ≈ 3.0–5.0°F
#
# Summer months (Jun–Aug) are most predictable for all cities; March and
# October show the highest variability due to seasonal transition fronts.
#
# To add a new city: add a row here with the matching `temp_high_xxx` metric
# key.  The fallback (no entry) is the global TEMP_FORECAST_SIGMA (default 4°F).

_CITY_SIGMA_F: dict[str, tuple[float, ...]] = {
    #                   Jan   Feb   Mar   Apr   May   Jun   Jul   Aug   Sep   Oct   Nov   Dec
    "temp_high_lax": (3.2,  3.0,  3.2,  3.5,  3.0,  2.5,  2.5,  2.5,  2.8,  3.2,  3.0,  3.2),
    "temp_high_den": (5.5,  5.5,  6.5,  6.0,  5.5,  4.5,  3.5,  3.5,  5.0,  5.5,  5.5,  5.5),
    "temp_high_chi": (5.0,  5.5,  5.5,  5.0,  4.5,  3.5,  3.0,  3.0,  4.0,  5.0,  5.5,  5.5),
    "temp_high_ny":  (4.5,  4.5,  5.0,  4.5,  4.0,  3.5,  3.0,  3.0,  3.5,  4.5,  4.5,  4.5),
    "temp_high_mia": (3.0,  3.0,  3.2,  3.2,  3.0,  2.8,  2.5,  2.5,  2.8,  3.0,  3.0,  3.0),
    "temp_high_aus": (4.5,  4.5,  5.0,  5.0,  4.5,  3.5,  3.0,  3.0,  4.0,  4.5,  4.5,  4.5),
    "temp_high_dal": (5.0,  5.0,  5.5,  5.5,  5.0,  4.0,  3.5,  3.5,  4.5,  5.0,  5.0,  5.0),
    "temp_high_bos": (4.5,  5.0,  5.0,  4.5,  4.0,  3.5,  3.0,  3.0,  3.5,  4.5,  5.0,  5.0),
    "temp_high_hou": (4.0,  4.0,  4.5,  4.5,  4.5,  3.5,  3.0,  3.0,  3.5,  4.0,  4.0,  4.0),
    # New cities (initial estimates based on climate type; calibrate after 30 days)
    #                   Jan   Feb   Mar   Apr   May   Jun   Jul   Aug   Sep   Oct   Nov   Dec
    "temp_high_dfw": (5.0,  5.0,  5.5,  5.5,  5.0,  4.0,  3.5,  3.5,  4.5,  5.0,  5.0,  5.0),
    "temp_high_sfo": (4.0,  4.0,  4.5,  4.5,  4.0,  3.5,  3.5,  3.5,  4.0,  4.5,  4.5,  4.0),
    "temp_high_sea": (4.0,  4.0,  4.5,  4.5,  4.0,  3.5,  3.0,  3.0,  3.5,  4.5,  4.5,  4.0),
    "temp_high_phx": (3.5,  3.5,  4.0,  4.5,  4.5,  3.5,  3.0,  3.0,  3.5,  4.0,  3.5,  3.5),
    "temp_high_phl": (4.5,  4.5,  5.0,  4.5,  4.0,  3.5,  3.0,  3.0,  3.5,  4.5,  4.5,  4.5),
    "temp_high_atl": (4.0,  4.0,  4.5,  4.0,  4.0,  3.5,  3.0,  3.0,  3.5,  4.0,  4.0,  4.0),
    "temp_high_msp": (6.0,  6.0,  6.5,  5.5,  5.0,  4.0,  3.5,  3.5,  4.5,  5.5,  6.0,  6.0),
    "temp_high_dca": (4.0,  4.5,  5.0,  4.5,  4.0,  3.5,  3.0,  3.0,  3.5,  4.5,  4.5,  4.0),
    "temp_high_las": (3.5,  3.5,  4.0,  4.5,  4.5,  3.5,  3.0,  3.0,  3.5,  4.0,  3.5,  3.5),
    "temp_high_okc": (5.0,  5.0,  5.5,  5.5,  5.0,  4.0,  3.5,  3.5,  4.5,  5.0,  5.0,  5.0),
    "temp_high_sat": (4.5,  4.5,  5.0,  5.0,  4.5,  3.5,  3.0,  3.0,  4.0,  4.5,  4.5,  4.5),
    "temp_high_msy": (4.0,  4.0,  4.5,  4.0,  4.0,  3.5,  3.0,  3.0,  3.5,  4.0,  4.0,  4.0),
    # Daily low sigma (~10% higher than daily high — overnight lows slightly harder to forecast)
    #                   Jan   Feb   Mar   Apr   May   Jun   Jul   Aug   Sep   Oct   Nov   Dec
    "temp_low_lax": (3.5,  3.2,  3.5,  3.8,  3.2,  2.8,  2.8,  2.8,  3.0,  3.5,  3.2,  3.5),
    "temp_low_den": (6.0,  6.0,  7.0,  6.5,  6.0,  5.0,  4.0,  4.0,  5.5,  6.0,  6.0,  6.0),
    "temp_low_chi": (5.5,  6.0,  6.0,  5.5,  5.0,  4.0,  3.5,  3.5,  4.5,  5.5,  6.0,  6.0),
    "temp_low_ny":  (5.0,  5.0,  5.5,  5.0,  4.5,  4.0,  3.5,  3.5,  4.0,  5.0,  5.0,  5.0),
    "temp_low_mia": (3.2,  3.2,  3.5,  3.5,  3.2,  3.0,  2.8,  2.8,  3.0,  3.2,  3.2,  3.2),
    "temp_low_aus": (5.0,  5.0,  5.5,  5.5,  5.0,  4.0,  3.5,  3.5,  4.5,  5.0,  5.0,  5.0),
    "temp_low_bos": (5.0,  5.5,  5.5,  5.0,  4.5,  4.0,  3.5,  3.5,  4.0,  5.0,  5.5,  5.5),
    "temp_low_hou": (4.5,  4.5,  5.0,  5.0,  5.0,  4.0,  3.5,  3.5,  4.0,  4.5,  4.5,  4.5),
    "temp_low_dfw": (5.5,  5.5,  6.0,  6.0,  5.5,  4.5,  4.0,  4.0,  5.0,  5.5,  5.5,  5.5),
    "temp_low_sfo": (4.5,  4.5,  5.0,  5.0,  4.5,  4.0,  4.0,  4.0,  4.5,  5.0,  5.0,  4.5),
    "temp_low_sea": (4.5,  4.5,  5.0,  5.0,  4.5,  4.0,  3.5,  3.5,  4.0,  5.0,  5.0,  4.5),
    "temp_low_phx": (4.0,  4.0,  4.5,  5.0,  5.0,  4.0,  3.5,  3.5,  4.0,  4.5,  4.0,  4.0),
    "temp_low_phl": (5.0,  5.0,  5.5,  5.0,  4.5,  4.0,  3.5,  3.5,  4.0,  5.0,  5.0,  5.0),
    "temp_low_atl": (4.5,  4.5,  5.0,  4.5,  4.5,  4.0,  3.5,  3.5,  4.0,  4.5,  4.5,  4.5),
    "temp_low_msp": (6.5,  6.5,  7.0,  6.0,  5.5,  4.5,  4.0,  4.0,  5.0,  6.0,  6.5,  6.5),
    "temp_low_dca": (4.5,  5.0,  5.5,  5.0,  4.5,  4.0,  3.5,  3.5,  4.0,  5.0,  5.0,  4.5),
    "temp_low_las": (4.0,  4.0,  4.5,  5.0,  5.0,  4.0,  3.5,  3.5,  4.0,  4.5,  4.0,  4.0),
    "temp_low_okc": (5.5,  5.5,  6.0,  6.0,  5.5,  4.5,  4.0,  4.0,  5.0,  5.5,  5.5,  5.5),
    "temp_low_sat": (5.0,  5.0,  5.5,  5.5,  5.0,  4.0,  3.5,  3.5,  4.5,  5.0,  5.0,  5.0),
    "temp_low_msy": (4.5,  4.5,  5.0,  4.5,  4.5,  4.0,  3.5,  3.5,  4.0,  4.5,  4.5,  4.5),
}

_CITY_SIGMA_FALLBACK: float = 4.0  # °F — used when metric not in table


def get_forecast_sigma(metric: str, month: int) -> float:
    """Return the NWS day-1 forecast MAE (1-sigma, °F) for a city and month.

    Args:
        metric: Full metric key, e.g. ``"temp_high_lax"`` or ``"temp_high_ny"``.
        month:  Calendar month, 1 = January … 12 = December.

    Returns:
        Calibrated σ in °F.  Falls back to ``_CITY_SIGMA_FALLBACK`` (4°F) for
        cities not in the table (e.g. newly-added cities before calibration data
        is available).
    """
    row = _CITY_SIGMA_F.get(metric)
    if row is None:
        return _CITY_SIGMA_FALLBACK
    return row[month - 1]


# ---------------------------------------------------------------------------
# Kalshi resolution station IDs (hard-coded)
# ---------------------------------------------------------------------------
#
# Kalshi's temperature markets settle against a specific NWS station as named
# in the market's rules_primary text (e.g. "Central Park, New York … as
# reported by the National Weather Service's Climatological Report").  The
# dynamic _resolve_obs_url path picks features[0] from the nearby-stations
# list, which can return a different (often lower-elevation) station and
# produce systematic 3–8°F mismatch errors.
#
# These hard-coded station identifiers bypass the dynamic resolution entirely.
# They were verified against Kalshi market rules_primary text and NWS ASOS
# records.  Update if Kalshi changes its resolution station for any city.
KALSHI_STATION_IDS: dict[str, str] = {
    "temp_high_lax": "KLAX",   # Los Angeles International Airport
    "temp_high_den": "KDEN",   # Denver International Airport (5,431 ft)
    "temp_high_chi": "KMDW",   # Chicago Midway International Airport
    "temp_high_ny":  "KNYC",   # New York — Central Park (per Kalshi market rules)
    "temp_high_mia": "KMIA",   # Miami International Airport
    "temp_high_aus": "KAUS",   # Austin-Bergstrom International Airport
    "temp_high_dal": "KDAL",   # Dallas Love Field Airport
    "temp_high_bos": "KBOS",   # Boston Logan International Airport
    "temp_high_hou": "KHOU",   # Houston William P. Hobby Airport
    "temp_high_dfw": "KDFW",   # Dallas/Fort Worth International Airport (KXHIGHTDAL)
    # New cities — active from 2026-04
    "temp_high_sfo": "KSFO",   # San Francisco International Airport
    "temp_high_sea": "KSEA",   # Seattle-Tacoma International Airport
    "temp_high_phx": "KPHX",   # Phoenix Sky Harbor International Airport
    "temp_high_phl": "KPHL",   # Philadelphia International Airport
    "temp_high_atl": "KATL",   # Hartsfield-Jackson Atlanta International Airport
    "temp_high_msp": "KMSP",   # Minneapolis-Saint Paul International Airport
    "temp_high_dca": "KDCA",   # Ronald Reagan Washington National Airport
    "temp_high_las": "KLAS",   # Harry Reid International Airport (Las Vegas)
    "temp_high_okc": "KOKC",   # Will Rogers World Airport (Oklahoma City)
    "temp_high_sat": "KSAT",   # San Antonio International Airport
    "temp_high_msy": "KMSY",   # Louis Armstrong New Orleans International Airport
}

# Cities with Kalshi KXLOWT* daily low temperature markets.
# Same coordinates/timezones as CITIES (same settlement stations).
# Do NOT add temp_low_* entries to KALSHI_STATION_IDS — that breaks METAR's
# reverse map.  Low-temp fetchers derive the station ID from the corresponding
# temp_high_* key.
LOW_CITIES: dict[str, tuple[str, float, float, ZoneInfo]] = {
    "temp_low_lax": ("Los Angeles",      33.9425, -118.4081, _PT),   # KLAX
    "temp_low_den": ("Denver",           39.8561, -104.6737, _MT),   # KDEN
    "temp_low_chi": ("Chicago",          41.7868,  -87.7522, _CT),   # KMDW
    "temp_low_ny":  ("New York",         40.7789,  -73.9692, _ET),   # KNYC
    "temp_low_mia": ("Miami",            25.7959,  -80.2870, _ET),   # KMIA
    "temp_low_aus": ("Austin",           30.1975,  -97.6664, _CT),   # KAUS
    "temp_low_bos": ("Boston",           42.3643,  -71.0052, _ET),   # KBOS
    "temp_low_hou": ("Houston",          29.6454,  -95.2789, _CT),   # KHOU
    "temp_low_dfw": ("Dallas/Fort Worth",32.8998,  -97.0403, _CT),   # KDFW
    "temp_low_sfo": ("San Francisco",    37.6190, -122.3750, _PT),   # KSFO
    "temp_low_sea": ("Seattle",          47.4502, -122.3088, _PT),   # KSEA
    "temp_low_phx": ("Phoenix",          33.4373, -112.0078, _PHX),  # KPHX (no DST)
    "temp_low_phl": ("Philadelphia",     39.8729,  -75.2437, _ET),   # KPHL
    "temp_low_atl": ("Atlanta",          33.6407,  -84.4277, _ET),   # KATL
    "temp_low_msp": ("Minneapolis",      44.8848,  -93.2223, _CT),   # KMSP
    "temp_low_dca": ("Washington DC",    38.8512,  -77.0402, _ET),   # KDCA
    "temp_low_las": ("Las Vegas",        36.0840, -115.1537, _PT),   # KLAS
    "temp_low_okc": ("Oklahoma City",    35.3931,  -97.6007, _CT),   # KOKC
    "temp_low_sat": ("San Antonio",      29.5337,  -98.4698, _CT),   # KSAT
    "temp_low_msy": ("New Orleans",      29.9934,  -90.2580, _CT),   # KMSY
}

# In-process caches — all resolved once per process lifetime
_gridpoint_cache: dict[str, dict[str, str]] = {}  # metric → {forecast, obs_stations}
_obs_url_cache:   dict[str, str] = {}              # metric → station observations URL
# Tracks which metrics have already had the dynamic-vs-hard-coded drift check
# performed.  Only checked once per process to avoid repeated API calls.
_station_drift_checked: set[str] = set()


# ---------------------------------------------------------------------------
# Gridpoint + station resolution (cached)
# ---------------------------------------------------------------------------

async def _resolve_gridpoint(
    session: aiohttp.ClientSession, metric: str, lat: float, lon: float
) -> dict[str, str] | None:
    """Fetch and cache NWS gridpoint data for a lat/lon.

    Returns a dict with ``forecast`` and ``obs_stations`` keys, or None on
    failure.  A single HTTP call provides both URLs, so we avoid duplicating
    the request that was previously made only for the forecast URL.
    """
    if metric in _gridpoint_cache:
        return _gridpoint_cache[metric]

    url = f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}"
    try:
        async with session.get(
            url, headers=_HEADERS, timeout=aiohttp.ClientTimeout(total=10)
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
        props = data["properties"]
        result: dict[str, str] = {
            "forecast":         props["forecast"],
            "obs_stations":     props["observationStations"],
            "forecast_hourly":  props.get("forecastHourly", ""),
        }
        _gridpoint_cache[metric] = result
        logging.debug("NOAA gridpoint resolved for %s", metric)
        return result
    except Exception as exc:
        logging.error("NOAA gridpoint lookup failed for %s: %s", metric, exc)
        return None


async def _resolve_obs_url(
    session: aiohttp.ClientSession, metric: str, obs_stations_url: str
) -> str | None:
    """Resolve and cache the observations endpoint for the Kalshi resolution station.

    Uses the hard-coded KALSHI_STATION_IDS mapping when available (preferred) to
    ensure the observation data comes from the same station Kalshi resolves against.
    Falls back to dynamic API resolution (nearest station) for any city not in the
    hard-coded table.
    """
    if metric in _obs_url_cache:
        return _obs_url_cache[metric]

    # Prefer the hard-coded station ID — avoids dynamic nearest-station lookup
    # which can return a different (e.g. lower-elevation) station.
    station_id = KALSHI_STATION_IDS.get(metric)
    if station_id:
        obs_url = f"https://api.weather.gov/stations/{station_id}/observations"
        _obs_url_cache[metric] = obs_url
        logging.debug("NOAA obs station (hard-coded) for %s: %s", metric, station_id)
        # One-time drift check: compare the hard-coded station against whatever
        # the dynamic nearest-station API would return.  Logs a WARNING if they
        # differ so we know when Kalshi may have changed their resolution station.
        if metric not in _station_drift_checked:
            _station_drift_checked.add(metric)
            asyncio.ensure_future(
                _check_station_drift(session, metric, station_id, obs_stations_url)
            )
        return obs_url

    # Dynamic fallback for cities not in the hard-coded table.
    try:
        async with session.get(
            obs_stations_url, headers=_HEADERS, timeout=aiohttp.ClientTimeout(total=10)
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
        features = data.get("features", [])
        if not features:
            logging.warning("NOAA: no observation stations found for %s", metric)
            return None
        station_id = features[0]["properties"]["stationIdentifier"]
        obs_url = f"https://api.weather.gov/stations/{station_id}/observations"
        _obs_url_cache[metric] = obs_url
        logging.debug("NOAA obs station (dynamic) for %s: %s", metric, station_id)
        return obs_url
    except Exception as exc:
        logging.error("NOAA obs station lookup failed for %s: %s", metric, exc)
        return None


async def _check_station_drift(
    session: aiohttp.ClientSession,
    metric: str,
    hard_coded_id: str,
    obs_stations_url: str,
) -> None:
    """Compare the hard-coded Kalshi station against the dynamic NWS nearest-station.

    Called once per process per city (via ensure_future) so the comparison
    doesn't block the main fetch path.  Logs a WARNING if the dynamic API
    returns a different station — this is the signal that Kalshi may have
    changed their resolution station and KALSHI_STATION_IDS needs updating.
    """
    try:
        async with session.get(
            obs_stations_url, headers=_HEADERS, timeout=aiohttp.ClientTimeout(total=10)
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
        features = data.get("features", [])
        if not features:
            return
        dynamic_id = features[0]["properties"]["stationIdentifier"]
        if dynamic_id != hard_coded_id:
            logging.warning(
                "NOAA station drift detected for %s: hard-coded=%s  NWS nearest=%s"
                " — verify Kalshi resolution station and update KALSHI_STATION_IDS"
                " if needed",
                metric, hard_coded_id, dynamic_id,
            )
        else:
            logging.debug(
                "NOAA station drift check OK for %s: hard-coded matches NWS nearest (%s)",
                metric, hard_coded_id,
            )
    except Exception as exc:
        logging.debug("NOAA station drift check failed for %s: %s", metric, exc)


# ---------------------------------------------------------------------------
# Data fetchers
# ---------------------------------------------------------------------------

async def _fetch_observed_max_today(
    session: aiohttp.ClientSession, city_name: str, obs_url: str, city_tz: ZoneInfo
) -> tuple[float | None, int | None]:
    """Return (precision_max_f, synoptic_max_c) recorded at the station since LST midnight.

    precision_max_f: max °F from METAR-aligned (non-round-5-minute) readings, which
    report decimal °C (0.1°C precision).  These are the only readings used for the
    precision running max.

    synoptic_max_c: running max integer °C from 5-minute synoptic readings.  These
    round to the nearest °C (±0.5°C → ±0.9°F) but update every 5 minutes vs. hourly
    for METAR.  Returned as a raw integer so callers can apply range math
    ([N-0.5, N+0.499]°C) to determine band membership without absorbing rounding error.

    Queries the NWS observations endpoint with a ``start`` parameter equal to
    midnight in Local Standard Time (LST).  The NWS CLI daily period runs from
    midnight LST to midnight LST regardless of DST — during Daylight Saving Time
    that means the period starts at 01:00 local clock time.  Using clock midnight
    (00:00 local) during DST would include the last hour of the *previous* LST
    day and produce a spuriously high running max (e.g. the 71.6°F overnight
    warm reading at 00:55 EDT on April 19 belongs to the NWS April 18 period).

    Returns (None, None) if the endpoint is unavailable or returns no readings for
    the LST day so far (correct behaviour before 01:00 local clock during DST).
    """
    local_now = datetime.now(city_tz)
    # Shift to LST midnight: if DST is active, midnight LST = 01:00 local clock.
    dst_offset = local_now.dst() or timedelta(0)
    lst_hour = 1 if dst_offset.total_seconds() > 0 else 0
    midnight_lst = local_now.replace(hour=lst_hour, minute=0, second=0, microsecond=0)
    # Express as UTC ISO-8601 for NWS API compatibility
    midnight_as_utc = midnight_lst.astimezone(timezone.utc).isoformat()

    try:
        async with session.get(
            obs_url,
            params={"start": midnight_as_utc, "limit": 100},
            headers=_HEADERS,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
    except Exception as exc:
        logging.warning("NOAA observed temp fetch failed for %s: %s", city_name, exc)
        return None, None

    max_f: float | None = None
    synoptic_max_c: int | None = None
    for feature in data.get("features", []):
        props = feature.get("properties") or {}
        ts = props.get("timestamp", "")
        try:
            ts_min = datetime.fromisoformat(ts).minute
        except (ValueError, TypeError):
            ts_min = -1

        temp_c_raw = props.get("temperature", {}).get("value")

        if ts_min >= 0 and ts_min % 5 == 0:
            # 5-minute synoptic reading: integer °C — track running max separately,
            # do not use for precision max.
            if temp_c_raw is not None:
                tc_int = int(round(float(temp_c_raw)))
                if synoptic_max_c is None or tc_int > synoptic_max_c:
                    synoptic_max_c = tc_int
            continue

        # METAR-aligned: decimal °C, use for precision max
        if temp_c_raw is not None:
            temp_f = float(temp_c_raw) * 9.0 / 5.0 + 32.0
            if max_f is None or temp_f > max_f:
                max_f = temp_f

    if max_f is not None:
        logging.debug("NOAA [%s]: observed max today = %.1f°F", city_name, max_f)
    return max_f, synoptic_max_c


async def _fetch_high_temp(
    session: aiohttp.ClientSession,
    metric: str,
    city_name: str,
    forecast_url: str,
    obs_url: str | None,
    city_tz: ZoneInfo,
) -> list[DataPoint]:
    """Fetch forecast high and observed max; return one DataPoint per stream.

    Returns up to two DataPoints:

      * ``source="noaa"``          — NWS day-1 forecast high.  High MAE (~3-4°F),
                                     so only useful when the edge vs. the strike
                                     is large (see TEMP_FORECAST_MIN_EDGE in main.py).

      * ``source="noaa_observed"`` — Maximum temperature actually recorded at the
                                     nearest station since midnight UTC today.
                                     Because the day's high can never be *less* than
                                     what has already been observed, this value is a
                                     hard lower bound on the final outcome.  When the
                                     observed max already exceeds a market's strike the
                                     result is locked in; when it doesn't, the signal
                                     is most reliable within ~4 hours of market close
                                     (see TEMP_OBSERVED_MAX_HOURS in main.py).

    The two DataPoints are emitted separately so the pipeline can apply
    different minimum-edge and time-to-close gates to each.
    """
    # --- Forecast ---
    try:
        async with session.get(
            forecast_url, headers=_HEADERS, timeout=aiohttp.ClientTimeout(total=10)
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
    except Exception as exc:
        logging.error("NOAA forecast fetch failed for %s: %s", city_name, exc)
        return []

    periods: list[dict[str, Any]] = data.get("properties", {}).get("periods", [])
    daytime_periods = [p for p in periods if p.get("isDaytime", False)]
    if not daytime_periods:
        logging.warning("NOAA: no daytime period found for %s", city_name)
        return []

    daytime = daytime_periods[0]
    forecast_temp: float = float(daytime["temperature"])
    unit_raw: str = daytime.get("temperatureUnit", "F")
    unit = f"°{unit_raw}"
    period_name: str = daytime.get("name", "")
    short_forecast: str = daytime.get("shortForecast", "")

    # --- Observed max so far today ---
    observed_max: float | None = None
    synoptic_max_c: int | None = None
    if obs_url:
        observed_max, synoptic_max_c = await _fetch_observed_max_today(session, city_name, obs_url, city_tz)

    today_noon_local = datetime.now(city_tz).replace(hour=12, minute=0, second=0, microsecond=0)
    as_of = today_noon_local.astimezone(timezone.utc).isoformat()

    # Build extended forecast DataPoints for days 2–7 (NWS returns up to 7
    # daytime periods in a single response).  Day N's as_of is set to noon ET
    # on the target date so the date guard in main.py matches against the right
    # Kalshi market.  NWS MAE grows ~40% per additional day so the sigma model
    # in trade_executor scales uncertainty accordingly.
    # Reference date: today in the city's local timezone.  Used to compute
    # the true day_offset for each extended period regardless of whether the
    # NWS "Today" daytime period has already expired.
    #
    # Bug fixed: the old code used array position (enumerate start=2) to assign
    # day_offset, which was wrong when fetched after ~6 PM local.  At that
    # point the NWS "Today" period has expired, so daytime_periods[0] is already
    # tomorrow and daytime_periods[1] is the day after tomorrow.  The old code
    # would label day-after-tomorrow data as "noaa_day2" (tomorrow's label),
    # then set as_of = tomorrow noon — so April 1 data was matched against the
    # March 31 market and produced forecasts 20°F off from reality.
    #
    # Fix: derive day_offset from each period's actual startTime relative to
    # today_local, and set as_of from the period's own date (not today_noon_et).
    # Use LST date (not clock date) to match NWS CLI daily period boundaries.
    # During DST, midnight LST = 01:00 local clock, so before 01:00 the LST
    # date is yesterday.  Subtracting the DST offset gives the LST datetime.
    _now_city = datetime.now(city_tz)
    _dst = _now_city.dst() or timedelta(0)
    today_local = (_now_city - _dst).date()

    extended_summary: list[str] = []
    extended_points: list[DataPoint] = []
    for dp in daytime_periods[1:7]:
        try:
            ext_temp = float(dp["temperature"])
            period_start_dt = datetime.fromisoformat(
                dp["startTime"].replace("Z", "+00:00")
            ).astimezone(city_tz)
            period_start = period_start_dt.date()
        except (KeyError, ValueError, TypeError):
            continue
        # Skip evening periods (start hour >= 14:00 local).  NWS occasionally
        # marks late-afternoon/evening periods as isDaytime=True; their
        # temperature reflects the NEXT day's overnight low rather than the
        # current day's high, and their startTime.date() is still today/tomorrow
        # — causing the fixed day_offset to collide with the actual daytime
        # forecast.  Daytime high periods always start between 06:00–14:00 local.
        if period_start_dt.hour >= 14:
            continue
        day_offset = (period_start - today_local).days + 1
        if day_offset < 2 or day_offset > 7:
            continue  # skip stale or too-far-out periods
        # as_of is noon local time on the period's actual date so the date guard
        # in main.py matches this signal to the correct Kalshi market.
        as_of_ext = datetime(
            period_start.year, period_start.month, period_start.day,
            12, 0, 0, tzinfo=city_tz,
        ).astimezone(timezone.utc).isoformat()
        extended_summary.append(f"day{day_offset}={ext_temp:.0f}°F")
        extended_points.append(DataPoint(
            source=f"noaa_day{day_offset}",
            metric=metric,
            value=ext_temp,
            unit=unit,
            as_of=as_of_ext,
            metadata={
                "city":           city_name,
                "forecast_day":   day_offset,
                "period":         dp.get("name", ""),
                "short_forecast": dp.get("shortForecast", ""),
                "forecast_high":  ext_temp,
            },
        ))

    logging.info(
        "NOAA [%s]: day1=%.1f%s  %s  observed_max=%s",
        city_name,
        forecast_temp, unit,
        "  ".join(extended_summary) if extended_summary else "no extended",
        f"{observed_max:.1f}°F" if observed_max is not None else "N/A",
    )

    points: list[DataPoint] = []

    # Day 1 Forecast DataPoint.
    points.append(DataPoint(
        source="noaa",
        metric=metric,
        value=forecast_temp,
        unit=unit,
        as_of=as_of,
        metadata={
            "city":           city_name,
            "period":         period_name,
            "short_forecast": short_forecast,
            "forecast_high":  forecast_temp,
            "observed_max":   observed_max,
        },
    ))

    # Extended forecast DataPoints (days 2–7).
    points.extend(extended_points)

    # Observed DataPoint — emitted separately when station data is available.
    # The numeric_matcher and main.py treat this with higher confidence than
    # the raw forecast, applying looser edge thresholds and time-to-close gates.
    if observed_max is not None:
        points.append(DataPoint(
            source="noaa_observed",
            metric=metric,
            value=observed_max,
            unit=unit,
            as_of=as_of,
            metadata={
                "city":              city_name,
                "forecast_high":     forecast_temp,
                "observed_max":      observed_max,
                "local_date":        str(today_local),
                "synoptic_celsius":  synoptic_max_c,
            },
        ))

    return points


async def _fetch_observed_min_today(
    session: aiohttp.ClientSession, city_name: str, obs_url: str, city_tz: ZoneInfo
) -> float | None:
    """Return the min temperature (°F) recorded at the station since LST midnight.

    Analogous to _fetch_observed_max_today() but tracks the running minimum.
    Used to provide a hard upper bound on today's daily low (the actual low
    can only stay at or below the running minimum as the morning progresses).
    Uses LST midnight as the window start (same reasoning as _fetch_observed_max_today).
    """
    local_now = datetime.now(city_tz)
    # Shift to LST midnight: if DST is active, midnight LST = 01:00 local clock.
    dst_offset = local_now.dst() or timedelta(0)
    lst_hour = 1 if dst_offset.total_seconds() > 0 else 0
    midnight_lst = local_now.replace(hour=lst_hour, minute=0, second=0, microsecond=0)
    # Cap query window to LST midnight→5 AM local so we only see overnight
    # observations.  Without this, a call at 10 PM includes the afternoon high
    # (e.g. Phoenix 84°F) which becomes the spurious "observed minimum".
    cutoff_local = midnight_lst.replace(hour=5 + lst_hour)
    end_local = min(local_now, cutoff_local)

    try:
        async with session.get(
            obs_url,
            params={
                "start": midnight_lst.astimezone(timezone.utc).isoformat(),
                "end":   end_local.astimezone(timezone.utc).isoformat(),
                "limit": 20,  # at most ~5 hourly obs in a 5-hour window
            },
            headers=_HEADERS,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
    except Exception as exc:
        logging.warning("NOAA observed min fetch failed for %s: %s", city_name, exc)
        return None

    min_f: float | None = None
    for feature in data.get("features", []):
        temp_c = (feature.get("properties") or {}).get("temperature", {}).get("value")
        if temp_c is not None:
            temp_f = temp_c * 9.0 / 5.0 + 32.0
            if min_f is None or temp_f < min_f:
                min_f = temp_f

    if min_f is not None:
        logging.debug("NOAA [%s]: observed min today = %.1f°F", city_name, min_f)
    return min_f


async def _fetch_low_temp(
    session: aiohttp.ClientSession,
    metric: str,
    city_name: str,
    forecast_url: str,
    obs_url: str | None,
    city_tz: ZoneInfo,
) -> list[DataPoint]:
    """Fetch nighttime forecast low and observed min; return one DataPoint per stream.

    The NWS daily forecast includes nighttime periods whose temperature is the
    overnight low.  The "Tonight" period covers tonight's evening into tomorrow
    morning — the low it forecasts settles as tomorrow's daily low on Kalshi
    (occurring ~4–6 AM local).  Therefore:

      nighttime_periods[0]  →  tomorrow's low   (day_offset = 1, source="noaa")
      nighttime_periods[1]  →  night-after low  (day_offset = 2, source="noaa_day2")
      … up to nighttime_periods[4] → day_offset = 5 (source="noaa_day5")

    Extended lows cover 5 nights (days 1–5) rather than 6 because NWS only
    reliably includes 6–7 nighttime periods per call.

    Observed min (source="noaa_observed", as_of=today noon local): today's
    running minimum since midnight.  Acts as a hard upper bound on the daily
    low (the actual low can only go lower before the day ends).
    """
    try:
        async with session.get(
            forecast_url, headers=_HEADERS, timeout=aiohttp.ClientTimeout(total=10)
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
    except Exception as exc:
        logging.error("NOAA low-temp forecast fetch failed for %s: %s", city_name, exc)
        return []

    periods: list[dict[str, Any]] = data.get("properties", {}).get("periods", [])
    nighttime_periods = [p for p in periods if not p.get("isDaytime", True)]
    if not nighttime_periods:
        logging.warning("NOAA: no nighttime period found for %s", city_name)
        return []

    _now_city_low = datetime.now(city_tz)
    _dst_low = _now_city_low.dst() or timedelta(0)
    today_local = (_now_city_low - _dst_low).date()
    today_noon_local = _now_city_low.replace(hour=12, minute=0, second=0, microsecond=0)
    as_of_today = today_noon_local.astimezone(timezone.utc).isoformat()

    unit = "°F"
    points: list[DataPoint] = []
    extended_summary: list[str] = []

    for np_ in nighttime_periods[:5]:
        try:
            low_temp = float(np_["temperature"])
            period_start_dt = datetime.fromisoformat(
                np_["startTime"].replace("Z", "+00:00")
            ).astimezone(city_tz)
            period_start = period_start_dt.date()
        except (KeyError, ValueError, TypeError):
            continue

        # Nighttime periods start in the evening (hour >= 14).  Skip daytime
        # periods that crept into the nighttime list due to NWS quirks.
        if period_start_dt.hour < 14:
            continue

        # The low in the "Tonight" period is recorded in the *next* morning.
        target_date = period_start + timedelta(days=1)
        day_offset = (target_date - today_local).days
        if day_offset < 1 or day_offset > 5:
            continue

        as_of_ext = datetime(
            target_date.year, target_date.month, target_date.day,
            12, 0, 0, tzinfo=city_tz,
        ).astimezone(timezone.utc).isoformat()

        source = "noaa" if day_offset == 1 else f"noaa_day{day_offset}"
        extended_summary.append(f"night+{day_offset}={low_temp:.0f}°F")

        points.append(DataPoint(
            source=source,
            metric=metric,
            value=low_temp,
            unit=unit,
            as_of=as_of_ext,
            metadata={
                "city":           city_name,
                "forecast_day":   day_offset,
                "period":         np_.get("name", ""),
                "short_forecast": np_.get("shortForecast", ""),
                "forecast_low":   low_temp,
            },
        ))

    # Observed minimum so far today.
    observed_min: float | None = None
    if obs_url:
        # Derive the station obs URL from the corresponding temp_high_* key
        # (avoids adding temp_low_* to KALSHI_STATION_IDS which would break METAR).
        high_metric = metric.replace("temp_low_", "temp_high_")
        station_id = KALSHI_STATION_IDS.get(high_metric)
        if station_id:
            low_obs_url = f"https://api.weather.gov/stations/{station_id}/observations"
        else:
            low_obs_url = obs_url
        observed_min = await _fetch_observed_min_today(session, city_name, low_obs_url, city_tz)

    logging.info(
        "NOAA low [%s]: %s  observed_min=%s",
        city_name,
        "  ".join(extended_summary) if extended_summary else "no nighttime periods",
        f"{observed_min:.1f}°F" if observed_min is not None else "N/A",
    )

    if observed_min is not None:
        points.append(DataPoint(
            source="noaa_observed",
            metric=metric,
            value=observed_min,
            unit=unit,
            as_of=as_of_today,
            metadata={
                "city":         city_name,
                "observed_min": observed_min,
                "local_date":   str(today_local),
            },
        ))

    return points


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def fetch_city_forecasts(
    session: aiohttp.ClientSession,
) -> list[DataPoint]:
    """Fetch today's effective high and low temperature estimates for all tracked cities.

    For each city, resolves the NWS gridpoint URLs (cached), then concurrently
    fetches the forecast and the day's observed max/min temperature.  Returns
    DataPoints for both daily high (CITIES) and daily low (LOW_CITIES).
    """
    async def fetch_one_high(
        metric: str, city_name: str, lat: float, lon: float, city_tz: ZoneInfo
    ) -> list[DataPoint]:
        gridpoint = await _resolve_gridpoint(session, metric, lat, lon)
        if gridpoint is None:
            return []
        obs_url = await _resolve_obs_url(session, metric, gridpoint["obs_stations"])
        return await _fetch_high_temp(
            session, metric, city_name, gridpoint["forecast"], obs_url, city_tz
        )

    async def fetch_one_low(
        metric: str, city_name: str, lat: float, lon: float, city_tz: ZoneInfo
    ) -> list[DataPoint]:
        # Re-use the gridpoint resolved for the corresponding high city.
        high_metric = metric.replace("temp_low_", "temp_high_")
        gridpoint = await _resolve_gridpoint(session, high_metric, lat, lon)
        if gridpoint is None:
            return []
        obs_url = await _resolve_obs_url(session, high_metric, gridpoint["obs_stations"])
        return await _fetch_low_temp(
            session, metric, city_name, gridpoint["forecast"], obs_url, city_tz
        )

    high_tasks = [
        fetch_one_high(metric, city_name, lat, lon, city_tz)
        for metric, (city_name, lat, lon, city_tz) in CITIES.items()
    ]
    low_tasks = [
        fetch_one_low(metric, city_name, lat, lon, city_tz)
        for metric, (city_name, lat, lon, city_tz) in LOW_CITIES.items()
    ]
    results = await asyncio.gather(*high_tasks, *low_tasks, return_exceptions=True)

    all_keys = list(CITIES.keys()) + list(LOW_CITIES.keys())
    data_points: list[DataPoint] = []
    for metric, result in zip(all_keys, results):
        if isinstance(result, Exception):
            logging.error("NOAA fetch error for %s: %s", metric, result)
        elif result:
            data_points.extend(result)

    return data_points
