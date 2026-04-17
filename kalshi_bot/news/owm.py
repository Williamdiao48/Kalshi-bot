"""OpenWeatherMap (OWM) forecast high temperature fetcher.

Provides a second independent weather model to cross-validate NOAA forecasts.
Uses the free OWM 5-day/3-hour forecast endpoint — no paid subscription needed.

Set the env var ``OWM_API_KEY`` to enable (free key at openweathermap.org).
If the key is absent the module returns an empty list and the poll continues.

API:
    GET https://api.openweathermap.org/data/2.5/forecast
        ?lat={lat}&lon={lon}
        &appid=KEY
        &units=imperial      ← °F directly, no conversion needed
        &cnt=16              ← 16 × 3-hour slots = 2 days, enough to cover today

Response (relevant fields):
    {
      "list": [
        {
          "dt_txt": "2026-03-07 15:00:00",   ← UTC timestamp string
          "main": {
            "temp":     75.2,
            "temp_max": 77.1                 ← 3-hour max; we aggregate across day
          }
        },
        ...
      ]
    }

We filter the list for entries whose ``dt_txt`` date matches today's UTC date,
then take the maximum ``main.temp_max`` as the forecast daily high.  This is
less precise than a dedicated daily-summary endpoint but is accurate to ±1–2°F
and sufficient for cross-validation.

The same CITIES dict as noaa.py is used so city coverage stays in sync.
DataPoints are emitted with ``source="owm"`` and the same ``temp_high_*``
metric keys as NOAA.  The consensus filter in main.py uses these to decide
whether NOAA and OWM agree before allowing a trade.
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import aiohttp

from ..data import DataPoint
from .noaa import CITIES  # reuse the same city registry

_BASE_URL = "https://api.openweathermap.org/data/2.5/forecast"


async def _fetch_city_high(
    session: aiohttp.ClientSession,
    metric: str,
    city_name: str,
    lat: float,
    lon: float,
    api_key: str,
    city_tz: ZoneInfo,
) -> DataPoint | None:
    """Fetch the forecast daily high for one city from OWM.

    Filters 3-hour forecast slots to today's city-local date, returns the max
    ``main.temp_max`` as a DataPoint.  Returns None on failure or no data.

    City-local date is used (not UTC) because late in the UTC day the OWM API
    returns slots starting from midnight UTC of the next day — which are still
    "today" in US local time (e.g. April 11 00:00 UTC = April 10 8 PM EDT).
    """
    params = {
        "lat":   lat,
        "lon":   lon,
        "appid": api_key,
        "units": "imperial",
        "cnt":   16,
    }
    try:
        async with session.get(
            _BASE_URL,
            params=params,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
    except aiohttp.ClientResponseError as exc:
        logging.error("OWM HTTP error %s for %s: %s", exc.status, city_name, exc.message)
        return None
    except aiohttp.ClientError as exc:
        logging.error("OWM request error for %s: %s", city_name, exc)
        return None

    slots = data.get("list", [])

    # Diagnose auth/quota failures: OWM returns {"cod": "401"/"429", "message": ...}
    # with HTTP 200 in some cases — no "list" key present.
    if not slots:
        cod = data.get("cod") or data.get("message", "")
        logging.warning(
            "OWM: empty response for %s (cod=%s) — possible auth/quota issue",
            city_name, cod,
        )
        return None

    today_local = datetime.now(city_tz).date()
    daily_high: float | None = None
    slot_dates: set[str] = set()
    for slot in slots:
        dt_txt = slot.get("dt_txt", "")
        try:
            slot_utc = datetime.strptime(dt_txt, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            slot_local_date = slot_utc.astimezone(city_tz).date()
        except ValueError:
            continue
        slot_dates.add(str(slot_local_date))
        if slot_local_date != today_local:
            continue
        slot_max = (slot.get("main") or {}).get("temp_max")
        if slot_max is not None:
            if daily_high is None or slot_max > daily_high:
                daily_high = float(slot_max)

    if daily_high is None:
        logging.warning(
            "OWM: no forecast slots for %s on %s (local) — API returned local dates: %s",
            city_name, today_local, sorted(slot_dates),
        )
        return None

    as_of = datetime.now(timezone.utc).isoformat()
    logging.info("OWM [%s]: forecast high %.1f°F", city_name, daily_high)

    return DataPoint(
        source="owm",
        metric=metric,
        value=daily_high,
        unit="°F",
        as_of=as_of,
        metadata={"city": city_name},
    )


async def fetch_city_forecasts(session: aiohttp.ClientSession) -> list[DataPoint]:
    """Fetch OWM forecast high temperatures for all tracked cities.

    Returns one DataPoint per city on success.  Returns an empty list if
    ``OWM_API_KEY`` is not set or all fetches fail.
    """
    api_key = os.environ.get("OWM_API_KEY", "")
    if not api_key:
        logging.warning("OWM_API_KEY not set — skipping OWM weather fetch.")
        return []

    tasks = [
        _fetch_city_high(session, metric, city_name, lat, lon, api_key, tz)
        for metric, (city_name, lat, lon, tz) in CITIES.items()
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    data_points: list[DataPoint] = []
    for metric, result in zip(CITIES.keys(), results):
        if isinstance(result, Exception):
            logging.error("OWM fetch error for %s: %s", metric, result)
        elif result is not None:
            data_points.append(result)

    return data_points
