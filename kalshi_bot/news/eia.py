"""EIA (U.S. Energy Information Administration) numeric data fetcher.

Fetches the most recent daily spot prices for crude oil and natural gas from
the EIA's open data API v2.  A free API key is required (register at
https://www.eia.gov/opendata/).

Set the env var ``EIA_API_KEY`` to enable this module.  If the key is absent
the module logs a warning and returns an empty list.

API (v2):
    GET https://api.eia.gov/v2/{route}/data/
        ?api_key=KEY
        &frequency=daily
        &data[0]=value
        &facets[series][]=SERIES_ID
        &sort[0][column]=period
        &sort[0][direction]=desc
        &length=5          ← a few rows to skip weekends / missing days

Response:
    {
      "response": {
        "data": [
          {"period": "2026-03-05", "series": "RWTC", "value": 74.12,
           "series-description": "WTI...", "units": "Dollars per Barrel"}
        ]
      }
    }

EIA markets do not trade on weekends; ``length=5`` ensures we always find the
most recent business-day close even after a long holiday weekend.

Tracked series
--------------
RWTC     WTI crude oil spot price ($/bbl)            eia_wti
RNGWHHD  Henry Hub natural gas spot price ($/MMBtu)  eia_natgas
"""

import logging
import os
from datetime import datetime, timezone

import aiohttp

from ..data import DataPoint

_BASE_URL = "https://api.eia.gov/v2"

# route → [(series_id, metric_key, unit, display_label)]
_SERIES_CONFIG: list[tuple[str, str, str, str, str]] = [
    # (route,                          series_id,  metric_key,  unit,        label)
    ("petroleum/pri/spt",              "RWTC",     "eia_wti",     "$/bbl",   "WTI Crude"),
    ("natural-gas/pri/fut",            "RNGWHHD",  "eia_natgas",  "$/MMBtu", "Henry Hub NG"),
]


async def _fetch_series(
    session: aiohttp.ClientSession,
    route: str,
    series_id: str,
    api_key: str,
) -> tuple[float, str] | None:
    """Fetch the most recent non-null value for one EIA series.

    Returns ``(value, period_str)`` or ``None`` on failure / no data.
    """
    url = f"{_BASE_URL}/{route}/data/"
    params = {
        "api_key":              api_key,
        "frequency":            "daily",
        "data[0]":              "value",
        "facets[series][]":     series_id,
        "sort[0][column]":      "period",
        "sort[0][direction]":   "desc",
        "length":               "5",
    }
    try:
        async with session.get(
            url,
            params=params,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
    except aiohttp.ClientResponseError as exc:
        logging.error("EIA HTTP error %s for %s: %s", exc.status, series_id, exc.message)
        return None
    except aiohttp.ClientError as exc:
        logging.error("EIA request error for %s: %s", series_id, exc)
        return None

    rows = (data.get("response") or {}).get("data", [])
    for row in rows:
        raw = row.get("value")
        if raw is None:
            continue
        try:
            return float(raw), row.get("period", "")
        except (TypeError, ValueError):
            continue

    logging.warning("EIA: no valid observation found for %s", series_id)
    return None


async def fetch_prices(session: aiohttp.ClientSession) -> list[DataPoint]:
    """Fetch the latest price for all tracked EIA series.

    Returns one DataPoint per series on success.  Returns an empty list if
    ``EIA_API_KEY`` is not set or if all fetches fail.
    """
    api_key = os.environ.get("EIA_API_KEY", "")
    if not api_key:
        logging.warning("EIA_API_KEY not set — skipping EIA data fetch.")
        return []

    import asyncio
    as_of = datetime.now(timezone.utc).isoformat()
    data_points: list[DataPoint] = []

    async def fetch_one(
        route: str, series_id: str, metric: str, unit: str, label: str
    ) -> DataPoint | None:
        result = await _fetch_series(session, route, series_id, api_key)
        if result is None:
            return None
        value, period_str = result
        logging.info("EIA [%s]: %.4f %s (as of %s)", label, value, unit, period_str)
        return DataPoint(
            source="eia",
            metric=metric,
            value=value,
            unit=unit,
            as_of=period_str or as_of,
            metadata={"series_id": series_id, "period": period_str},
        )

    tasks = [fetch_one(*cfg) for cfg in _SERIES_CONFIG]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for cfg, result in zip(_SERIES_CONFIG, results):
        series_id = cfg[1]
        if isinstance(result, Exception):
            logging.error("EIA fetch error for %s: %s", series_id, result)
        elif result is not None:
            data_points.append(result)

    return data_points
