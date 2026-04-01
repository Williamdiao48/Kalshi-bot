"""NWS CLI (Climatological Report) scraper — gold-standard temperature source.

The NWS publishes a daily Climate Report (CLI text product) for each ASOS
station.  This is the *exact* data source Kalshi references in its market
rules: "as reported by the National Weather Service's Daily Climatological
Report."  Scraping it directly eliminates the risk of the NWS observations
API returning data from a nearby-but-wrong station.

The afternoon/evening preliminary report ("TODAY'S PRELIMINARY CLIMATE DATA")
is published around 5–8 PM local time and contains the running daily maximum
temperature.  We fetch only this preliminary report — it is ground-truth data
from the Kalshi resolution station with no approximation.

API flow
--------
1. GET https://api.weather.gov/products?type=CLI&location={location_id}&limit=5
   → list of recent CLI stubs, newest first; each has ``@id`` and ``issuanceTime``

2. GET {stub["@id"]}
   → full product JSON; key field: ``productText`` containing the temperature table

3. Parse ``MAXIMUM`` from the text table.

Schedule / caching
------------------
The preliminary CLI is published once per afternoon.  We cache per (metric, UTC
date) so subsequent poll cycles within the same day re-use the parsed result
without re-fetching.

Source tag: ``"nws_climo"``
Treated identically to ``"noaa_observed"`` in ``_filter_weather_opportunities``:
high-confidence observation tier, subject to afternoon gate and obs-consensus gate.
"""

import logging
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import aiohttp

from ..data import DataPoint

_HEADERS = {"User-Agent": "kalshi-bot/1.0 (educational; contact: user@example.com)"}
_PRODUCTS_URL = "https://api.weather.gov/products"

# NWS 3-letter location IDs for the CLI product, keyed by Kalshi metric.
# These match the station IDs in noaa.KALSHI_STATION_IDS (same settlement stations).
CLIMO_LOCATIONS: dict[str, tuple[str, str, ZoneInfo]] = {
    # metric              location  display     timezone
    "temp_high_ny":  ("NYC", "New York/Central Park",    ZoneInfo("America/New_York")),
    "temp_high_bos": ("BOS", "Boston",                   ZoneInfo("America/New_York")),
    "temp_high_mia": ("MIA", "Miami",                    ZoneInfo("America/New_York")),
    "temp_high_chi": ("MDW", "Chicago Midway",           ZoneInfo("America/Chicago")),
    "temp_high_dal": ("DAL", "Dallas Love Field",        ZoneInfo("America/Chicago")),
    "temp_high_aus": ("AUS", "Austin",                   ZoneInfo("America/Chicago")),
    "temp_high_hou": ("HOU", "Houston Hobby",            ZoneInfo("America/Chicago")),
    "temp_high_den": ("DEN", "Denver",                   ZoneInfo("America/Denver")),
    "temp_high_lax": ("LAX", "Los Angeles",              ZoneInfo("America/Los_Angeles")),
}

# Cache: (metric, utc_date_str) → parsed max temperature (°F).
# Re-populated once per calendar day per city — avoids re-fetching the same
# CLI product on every 60-second poll cycle.
_cache: dict[tuple[str, str], float] = {}

# Which product IDs have already been fetched and attempted this session.
# Prevents hammering the API when a city's CLI doesn't have today's preliminary yet.
_attempted: dict[tuple[str, str], bool] = {}  # (metric, utc_date_str) → tried


def _parse_max_f(text: str) -> float | None:
    """Extract the daily MAXIMUM temperature (°F) from a NWS CLI product text.

    Handles the two most common CLI table formats:

      Format A (column labels on a separate header line):
        TEMPERATURE (F)      TODAY  YESTERDAY  NORMAL
        MAXIMUM               66       72         53

      Format B (inline values):
        MAXIMUM   MAX 66   MIN 38   AVG 52

    Returns ``None`` if no numeric value follows MAXIMUM or if the value
    is flagged as missing ("MM", "MISSING", etc.).
    """
    # Find "MAXIMUM" (or "MAX TEMP") followed by optional whitespace then digits.
    m = re.search(r'\bMAX(?:IMUM)?\b[\s/A-Z]*(\d+)', text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def _is_todays_preliminary(text: str, today_local: "date") -> bool:  # noqa: F821
    """Return True if the product text covers today's preliminary data.

    Two checks:
      1. The word PRELIMINARY appears (rules out the next-morning final report).
      2. The product's date line (if present) matches today's local date.

    If no explicit date line is found, accept any PRELIMINARY product issued
    today (issuanceTime check is already done by the caller).
    """
    if not re.search(r'\bPRELIMINARY\b', text, re.IGNORECASE):
        return False
    # Many CLI products contain a date line like "MAR 15 2026" — verify it
    # matches today.  If the pattern is absent, trust the issuanceTime check.
    date_m = re.search(
        r'\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+(\d{1,2})\s+(\d{4})\b',
        text, re.IGNORECASE,
    )
    if date_m:
        _MONTH = {
            "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
            "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
        }
        try:
            from datetime import date as _date
            product_date = _date(
                int(date_m.group(3)),
                _MONTH[date_m.group(1).upper()],
                int(date_m.group(2)),
            )
            return product_date == today_local
        except (ValueError, KeyError):
            pass
    return True  # date line absent — trust issuanceTime


async def _fetch_product_text(
    session: aiohttp.ClientSession, product_url: str
) -> str | None:
    """Fetch the full product text from a NWS product URL."""
    try:
        async with session.get(
            product_url, headers=_HEADERS, timeout=aiohttp.ClientTimeout(total=10)
        ) as resp:
            resp.raise_for_status()
            data = await resp.json(content_type=None)
        return data.get("productText") or ""
    except Exception as exc:
        logging.debug("NWS climo: product text fetch failed for %s: %s", product_url, exc)
        return None


async def _fetch_city_climo(
    session:   aiohttp.ClientSession,
    metric:    str,
    location:  str,
    city_name: str,
    city_tz:   ZoneInfo,
) -> DataPoint | None:
    """Fetch and parse today's preliminary CLI maximum temperature for one city."""
    now_utc   = datetime.now(timezone.utc)
    today_key = now_utc.strftime("%Y-%m-%d")
    cache_key = (metric, today_key)

    if cache_key in _cache:
        val = _cache[cache_key]
        return DataPoint(
            source="nws_climo", metric=metric, value=val, unit="°F",
            as_of=now_utc.isoformat(),
            metadata={"location": location, "city_name": city_name, "cached": True},
        )

    if _attempted.get(cache_key):
        return None   # already tried and failed this calendar day

    _attempted[cache_key] = True
    today_local = now_utc.astimezone(city_tz).date()

    try:
        async with session.get(
            _PRODUCTS_URL,
            params={"type": "CLI", "location": location, "limit": "5"},
            headers=_HEADERS,
            timeout=aiohttp.ClientTimeout(total=12),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json(content_type=None)
    except Exception as exc:
        logging.debug("NWS climo: products list fetch failed for %s: %s", location, exc)
        return None

    stubs = data.get("@graph", [])
    if not stubs:
        logging.debug("NWS climo: no CLI products found for %s", location)
        return None

    for stub in stubs:
        issuance_str = stub.get("issuanceTime", "")
        try:
            issuance_dt = datetime.fromisoformat(issuance_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            continue

        # Only consider products issued today (local time).
        if issuance_dt.astimezone(city_tz).date() != today_local:
            continue

        product_url = stub.get("@id") or stub.get("id")
        if not product_url:
            continue

        # If the productText is already in the stub, use it; otherwise fetch.
        text = stub.get("productText") or await _fetch_product_text(session, product_url)
        if not text:
            continue

        if not _is_todays_preliminary(text, today_local):
            continue

        max_f = _parse_max_f(text)
        if max_f is None:
            logging.debug(
                "NWS climo [%s]: could not parse MAXIMUM from today's CLI product",
                city_name,
            )
            continue

        _cache[cache_key] = max_f
        logging.info(
            "NWS climo [%s]: today's preliminary max = %.1f°F (from %s)",
            city_name, max_f, issuance_str,
        )
        return DataPoint(
            source   = "nws_climo",
            metric   = metric,
            value    = max_f,
            unit     = "°F",
            as_of    = issuance_str,
            metadata = {"location": location, "city_name": city_name},
        )

    logging.debug(
        "NWS climo [%s]: no today's preliminary CLI product available yet", city_name
    )
    return None


async def fetch_city_climo(session: aiohttp.ClientSession) -> list[DataPoint]:
    """Fetch NWS CLI daily maximum temperatures for all Kalshi temperature cities.

    Returns a list of DataPoints with ``source="nws_climo"``.  Cities where
    today's preliminary CLI has not yet been published return no DataPoint
    (typically before ~5 PM local time).

    All cities are fetched concurrently.
    """
    import asyncio
    tasks = [
        _fetch_city_climo(session, metric, loc, city, tz)
        for metric, (loc, city, tz) in CLIMO_LOCATIONS.items()
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    points: list[DataPoint] = []
    for metric, result in zip(CLIMO_LOCATIONS, results):
        if isinstance(result, Exception):
            logging.warning("NWS climo fetch error for %s: %s", metric, result)
        elif result is not None:
            points.append(result)
    if points:
        logging.info("NWS climo: fetched %d city reading(s).", len(points))
    return points
