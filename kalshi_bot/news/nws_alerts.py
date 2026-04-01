"""NWS (National Weather Service) active weather alerts fetcher.

Fetches active Extreme and Severe alerts from the NWS API and converts them
into text documents compatible with the keyword-matching pipeline.  Each alert
becomes one document whose ``title`` is "{event}: {areaDesc}" and whose
``abstract`` is the alert headline + description.

The document's ``feed_id`` is set to ``"nws_alerts"`` so it is routed to the
``weather_alerts`` source group in main.py (which targets weather-adjacent
Kalshi markets).

Alert deduplication uses the NWS ``id`` field (a stable URN URL like
``https://api.weather.gov/alerts/urn:oid:2.49.0.1.840.0...``).  Alerts
persist in the NWS feed until they expire, so the SeenDocuments store prevents
the same alert from surfacing on every poll cycle.

API:
    GET https://api.weather.gov/alerts/active
        ?status=actual
        &message_type=alert
        &severity=Extreme,Severe

Response: GeoJSON FeatureCollection; features[].properties contains:
    id          — unique alert URN (used as document_number)
    event       — e.g. "Winter Storm Warning", "Heat Advisory"
    headline    — short description
    description — full text
    severity    — "Extreme" | "Severe" | "Moderate" | "Minor" | "Unknown"
    areaDesc    — comma-separated affected zones/counties
    effective   — ISO-8601 onset time
    expires     — ISO-8601 expiry time
"""

import logging
import re
from datetime import datetime, timezone

import aiohttp

from ..data import DataPoint

_BASE_URL = "https://api.weather.gov/alerts/active"
_HEADERS = {"User-Agent": "kalshi-bot/1.0 (educational; contact: user@example.com)"}

# ---------------------------------------------------------------------------
# City alert signal config
# ---------------------------------------------------------------------------

# Maps metric key → (city display name, area keywords to match in areaDesc).
# A keyword match is case-insensitive substring search of the NWS areaDesc
# field, which lists affected counties/zones as a comma-separated string.
_CITY_AREA_CONFIG: dict[str, tuple[str, list[str]]] = {
    "temp_high_lax": ("Los Angeles", ["Los Angeles", "San Fernando", "Long Beach"]),
    "temp_high_den": ("Denver",      ["Denver", "Adams County", "Arapahoe", "Jefferson County", "Boulder"]),
    "temp_high_chi": ("Chicago",     ["Cook County", "Chicago", "DuPage", "Lake, IL", "Will County"]),
    "temp_high_ny":  ("New York",    ["New York City", "Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island", "Nassau", "Westchester"]),
    "temp_high_mia": ("Miami",       ["Miami-Dade", "Broward", "Palm Beach"]),
    "temp_high_aus": ("Austin",      ["Travis County", "Williamson County", "Hays County"]),
    "temp_high_dal": ("Dallas",      ["Dallas County", "Tarrant County", "Collin County", "Denton County"]),
    "temp_high_bos": ("Boston",      ["Suffolk County", "Middlesex County", "Essex County", "Boston"]),
    "temp_high_hou": ("Houston",     ["Harris County", "Fort Bend County", "Montgomery County", "Galveston"]),
}

# Heat event types that imply an unusually HIGH daily maximum temperature.
_HEAT_EVENTS = {
    "Excessive Heat Warning",
    "Excessive Heat Watch",
    "Heat Advisory",
}

# Cold event types that imply an unusually LOW daily maximum temperature.
_COLD_EVENTS = {
    "Blizzard Warning",
    "Ice Storm Warning",
    "Freeze Warning",
    "Wind Chill Warning",
    "Winter Storm Warning",
    "Arctic Blast",
}

# Regex to extract a daily HIGH temperature from alert description text.
# Matches patterns like:
#   "Highs of 105 to 110 degrees"
#   "temperatures up to 108 degrees"
#   "highs around 102"
#   "maximum temperatures of 100 to 105 degrees"
#   "highs 95 to 100"
_HIGH_TEMP_RE = re.compile(
    r"(?:high(?:s)?|temperature(?:s)?|temp(?:s)?)\s*"
    r"(?:of|up\s+to|around|near|reaching|between|to\s+near)?\s*"
    r"(\d{2,3})"
    r"(?:\s*(?:to|and|-)\s*(\d{2,3}))?",
    re.IGNORECASE,
)

# Regex for cold events: lows or wind chills sometimes contain the high
# indirectly, but we look specifically for daytime-high language.
_LOW_TEMP_RE = re.compile(
    r"(?:low(?:s)?|wind\s+chill(?:s)?|temperatures?\s+drop(?:ping)?\s+to)\s*"
    r"(?:of|to|around|near|as\s+(?:low\s+)?as)?\s*"
    r"(-?\d{1,3})",
    re.IGNORECASE,
)

# Only surface alerts at these severity levels.  "Moderate" and below are
# too common and unlikely to move prediction markets.
_SEVERITY_FILTER = {"Extreme", "Severe"}

# NWS event types we care about.  Filters out unrelated alerts (e.g. Special
# Marine Warnings, Flood Advisories) that have no Kalshi market equivalent.
_RELEVANT_EVENTS = {
    # Heat
    "Excessive Heat Warning",
    "Excessive Heat Watch",
    "Heat Advisory",
    # Cold / winter
    "Winter Storm Warning",
    "Winter Storm Watch",
    "Blizzard Warning",
    "Ice Storm Warning",
    "Freeze Warning",
    "Freeze Watch",
    "Frost Advisory",
    "Wind Chill Warning",
    "Wind Chill Advisory",
    "Arctic Blast",
    # Tropical / hurricane
    "Hurricane Warning",
    "Hurricane Watch",
    "Tropical Storm Warning",
    "Tropical Storm Watch",
    "Hurricane Local Statement",
    # Severe / tornado
    "Tornado Warning",
    "Tornado Watch",
    "Severe Thunderstorm Warning",
    "Severe Thunderstorm Watch",
    # High wind / fire
    "High Wind Warning",
    "Red Flag Warning",
}


async def fetch_alerts(session: aiohttp.ClientSession) -> list[dict]:
    """Fetch active NWS weather alerts and return them as document dicts.

    Returns one dict per alert, structured for the text-matching pipeline:
        document_number  — stable NWS alert ID (URN URL)
        title            — "{event}: {areaDesc}"
        abstract         — headline + description (searchable body text)
        html_url         — direct link to the alert on api.weather.gov
        _source          — "nws_alerts"
        feed_id          — "nws_alerts" (SOURCE_GROUPS routing key)
        severity         — original NWS severity string (metadata)
        effective        — alert onset ISO timestamp
        expires          — alert expiry ISO timestamp

    Returns an empty list on any fetch failure so the poll cycle continues.
    """
    params = {
        "status": "actual",
        "message_type": "alert",
    }

    try:
        async with session.get(
            _BASE_URL,
            params=params,
            headers=_HEADERS,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
    except aiohttp.ClientResponseError as exc:
        logging.error("NWS Alerts HTTP error %s: %s", exc.status, exc.message)
        return []
    except aiohttp.ClientError as exc:
        logging.error("NWS Alerts request error: %s", exc)
        return []

    features = data.get("features", [])
    docs: list[dict] = []

    for feature in features:
        props = feature.get("properties") or {}

        severity = props.get("severity", "Unknown")
        if severity not in _SEVERITY_FILTER:
            continue

        event = props.get("event", "")
        if event not in _RELEVANT_EVENTS:
            continue

        alert_id = props.get("id") or feature.get("id", "")
        area_desc = props.get("areaDesc", "")
        headline = props.get("headline") or ""
        description = props.get("description") or ""
        effective = props.get("effective") or ""
        expires = props.get("expires") or ""

        # Build searchable body text: headline + description
        abstract = f"{headline}\n{description}".strip()

        # Title drives Phase-2 keyword matching (must contain the term)
        title = f"{event}: {area_desc}"

        logging.debug(
            "NWS Alert [%s] %s → %s", severity, event, area_desc[:80]
        )

        docs.append({
            "document_number": alert_id,
            "title":           title,
            "abstract":        abstract,
            "html_url":        alert_id,  # the NWS alert ID is itself a URL
            "_source":         "nws_alerts",
            "feed_id":         "nws_alerts",
            "severity":        severity,
            "effective":       effective,
            "expires":         expires,
        })

    if docs:
        logging.info(
            "NWS Alerts: %d relevant active alert(s) (Extreme/Severe).", len(docs)
        )
    else:
        logging.debug("NWS Alerts: no Extreme/Severe relevant alerts active.")

    return docs


# ---------------------------------------------------------------------------
# Numeric trigger: NWS alerts → DataPoints for KXHIGH markets
# ---------------------------------------------------------------------------

def _parse_high_temp(text: str) -> float | None:
    """Extract the best daily HIGH temperature estimate (°F) from alert text.

    For heat events, matches explicit high-temperature language and returns
    the midpoint of any stated range.  Returns None if no match is found.
    """
    m = _HIGH_TEMP_RE.search(text)
    if not m:
        return None
    lo = float(m.group(1))
    hi = float(m.group(2)) if m.group(2) else lo
    return (lo + hi) / 2.0


def _parse_cold_high(text: str) -> float | None:
    """Estimate today's HIGH from a cold-weather alert description.

    Cold alerts focus on overnight lows and wind chills, not daytime highs.
    We parse the stated low/wind-chill and add a conservative daytime rise
    of 5°F to approximate the day's peak.  Returns None if unparseable.
    """
    m = _LOW_TEMP_RE.search(text)
    if not m:
        return None
    low = float(m.group(1))
    return low + 5.0  # rough daytime rise above overnight low


def _area_matches_city(area_desc: str, keywords: list[str]) -> bool:
    """Return True if any city keyword appears (case-insensitive) in areaDesc."""
    area_lower = area_desc.lower()
    return any(kw.lower() in area_lower for kw in keywords)


async def fetch_city_alert_signals(
    session: aiohttp.ClientSession,
) -> list[DataPoint]:
    """Fetch active NWS heat/cold alerts and emit DataPoints for tracked cities.

    For each city in ``_CITY_AREA_CONFIG``, scans all active Extreme/Severe
    NWS alerts.  When a heat or cold event covers a tracked city's area and
    a temperature can be parsed from the alert description, a DataPoint is
    emitted with ``source="nws_alert"`` and the extracted temperature as the
    value.

    These DataPoints flow into the standard numeric-matching pipeline so the
    bot can trade KXHIGH markets when an official NWS alert confirms the
    direction.  ``_filter_weather_opportunities`` in main.py treats
    ``nws_alert`` signals like ``noaa_observed`` (lower edge threshold,
    always surfaced when YES is locked).

    Returns an empty list on any fetch failure or when no relevant alerts
    are active.
    """
    params = {"status": "actual", "message_type": "alert"}
    try:
        async with session.get(
            _BASE_URL,
            params=params,
            headers=_HEADERS,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
    except Exception as exc:
        logging.warning("NWS alert signals fetch failed: %s", exc)
        return []

    features = data.get("features", [])
    as_of = datetime.now(timezone.utc).isoformat()
    points: list[DataPoint] = []

    for feature in features:
        props = feature.get("properties") or {}
        event     = props.get("event", "")
        severity  = props.get("severity", "Unknown")
        area_desc = props.get("areaDesc", "")
        headline  = props.get("headline") or ""
        description = props.get("description") or ""
        expires   = props.get("expires") or ""

        if severity not in _SEVERITY_FILTER:
            continue

        is_heat = event in _HEAT_EVENTS
        is_cold = event in _COLD_EVENTS
        if not is_heat and not is_cold:
            continue

        full_text = f"{headline}\n{description}"

        # Try to parse the temperature implied by the alert.
        if is_heat:
            temp_f = _parse_high_temp(full_text)
        else:
            temp_f = _parse_cold_high(full_text)

        if temp_f is None:
            logging.debug(
                "NWS alert signal: could not parse temperature from [%s] for %s",
                event, area_desc[:60],
            )
            continue

        # Match alert area to tracked cities.
        for metric, (city_name, keywords) in _CITY_AREA_CONFIG.items():
            if not _area_matches_city(area_desc, keywords):
                continue

            logging.info(
                "NWS alert signal: [%s] covers %s → %.1f°F  (expires %s)",
                event, city_name, temp_f, expires,
            )
            points.append(DataPoint(
                source="nws_alert",
                metric=metric,
                value=temp_f,
                unit="°F",
                as_of=as_of,
                metadata={
                    "city":        city_name,
                    "alert_event": event,
                    "alert_area":  area_desc,
                    "severity":    severity,
                    "expires":     expires,
                },
            ))

    if points:
        logging.info("NWS alert signals: %d DataPoint(s) emitted.", len(points))
    return points
