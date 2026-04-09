"""FRED (Federal Reserve Economic Data) numeric data fetcher.

Fetches the most recent observations for key interest-rate and financial series
from the St. Louis Fed's public REST API.  A free API key is required
(register at https://fred.stlouisfed.org/docs/api/api_key.html).

Set the env var ``FRED_API_KEY`` to enable this module.  If the key is absent
the module logs a warning and returns an empty list so the poll cycle continues
unaffected.

API:
    GET https://api.stlouisfed.org/fred/series/observations
        ?series_id=DGS10
        &api_key=KEY
        &sort_order=desc
        &limit=5          ← grab a few in case the most recent is "." (holiday)
        &file_type=json

Response:
    {"observations": [{"date": "2026-03-05", "value": "4.28"}, ...]}

FRED returns "." for dates where the series has no observation (e.g. weekends
for daily series).  The fetcher skips "." entries and uses the most recent
valid reading.

Tracked series
--------------
DFEDTARU   Federal funds target rate — upper bound (%)         fred_fedfunds
DGS10      10-year Treasury constant maturity rate (%)         fred_dgs10
DGS2       2-year Treasury constant maturity rate (%)          fred_dgs2
ICSA       Initial jobless claims (seasonally adj., thousands)  fred_icsa
           Released every Thursday at 08:30 ET by the DOL.
           Kalshi markets: "Will initial claims be above/below Xk this week?"
NAPM       ISM Manufacturing PMI composite index               ism_manufacturing
           Released 1st business day of month at 10:00 ET.
           Kalshi markets: "Will ISM Manufacturing PMI be above/below 50?"
NMFCI      ISM Services (Non-Manufacturing) PMI index          ism_services
           Released 3rd business day of month at 10:00 ET.
           Kalshi markets: "Will ISM Services PMI be above/below 50?"
"""

import logging
import os
from datetime import datetime, timezone

import aiohttp

from ..data import DataPoint

_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# series_id → (metric key, unit string, display label)
SERIES: dict[str, tuple[str, str, str]] = {
    "DFEDTARU": ("fred_fedfunds", "%",        "Fed Funds Upper"),
    "DGS10":    ("fred_dgs10",    "%",        "10yr Treasury"),
    "DGS2":     ("fred_dgs2",     "%",        "2yr Treasury"),
    # DOL initial jobless claims (thousands, SA) — released every Thursday 08:30 ET.
    # FRED updates ICSA within minutes of the DOL press release.
    "ICSA":     ("fred_icsa",     "k claims", "Initial Claims"),
    # ISM PMI (NAPM / NMFCI) were removed from FRED after ISM revoked the
    # redistribution license.  Fetched directly via the ISM module instead.
    "PCEPI":           ("fred_pce",        "index", "PCE Price Index"),
    # Real GDP growth rate (SAAR, %) — matches KXGDP "Will real GDP increase by
    # more than X% in QN YYYY?" markets.  FRED updates within hours of the BEA
    # advance estimate (last business day of the first month after quarter-end).
    "A191RL1Q225SBEA": ("fred_gdp_growth", "%",     "Real GDP Growth (SAAR)"),
}


async def _fetch_series(
    session: aiohttp.ClientSession,
    series_id: str,
    api_key: str,
) -> tuple[float, str] | None:
    """Fetch the most recent non-null observation for one FRED series.

    Returns ``(value, date_str)`` or ``None`` on failure / no data.
    """
    params = {
        "series_id":  series_id,
        "api_key":    api_key,
        "sort_order": "desc",
        "limit":      "5",
        "file_type":  "json",
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
        logging.error("FRED HTTP error %s for %s: %s", exc.status, series_id, exc.message)
        return None
    except aiohttp.ClientError as exc:
        logging.error("FRED request error for %s: %s", series_id, exc)
        return None

    for obs in data.get("observations", []):
        raw = obs.get("value", ".")
        if raw == ".":
            continue
        try:
            return float(raw), obs.get("date", "")
        except ValueError:
            continue

    logging.warning("FRED: no valid observation found for %s", series_id)
    return None


async def fetch_rates(session: aiohttp.ClientSession) -> list[DataPoint]:
    """Fetch the latest reading for all tracked FRED series.

    Returns one DataPoint per series on success.  Returns an empty list if
    ``FRED_API_KEY`` is not set or if all fetches fail.
    """
    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        logging.warning("FRED_API_KEY not set — skipping FRED data fetch.")
        return []

    import asyncio
    as_of = datetime.now(timezone.utc).isoformat()
    data_points: list[DataPoint] = []

    async def fetch_one(series_id: str) -> DataPoint | None:
        metric, unit, label = SERIES[series_id]
        result = await _fetch_series(session, series_id, api_key)
        if result is None:
            return None
        value, date_str = result
        logging.info("FRED [%s]: %.4f%s (as of %s)", label, value, unit, date_str)
        return DataPoint(
            source="fred",
            metric=metric,
            value=value,
            unit=unit,
            as_of=date_str or as_of,
            metadata={"series_id": series_id, "series_date": date_str},
        )

    tasks = [fetch_one(sid) for sid in SERIES]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for series_id, result in zip(SERIES, results):
        if isinstance(result, Exception):
            logging.error("FRED fetch error for %s: %s", series_id, result)
        elif result is not None:
            data_points.append(result)

    return data_points
