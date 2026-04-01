"""BLS (Bureau of Labor Statistics) economic data fetcher.

Polls key economic series and emits a DataPoint only when a NEW data period
appears (i.e. a fresh monthly release). Seen periods are persisted in SQLite
via the shared SeenDocuments store so restarts don't re-fire old releases.

Series tracked:
    CUUR0000SA0   CPI-U (All Items, not seasonally adjusted) → bls_cpi_u
    CES0000000001 Total Nonfarm Payrolls (seasonally adj.)   → bls_nfp
    LNS14000000   Unemployment Rate (seasonally adjusted)    → bls_unrate

API:
    POST https://api.bls.gov/publicAPI/v2/timeseries/data/

Rate limits:
    No key : 25 queries/day,  10 series/query, 3 years of data
    API key: 500 queries/day, 50 series/query, 20 years of data

Set BLS_API_KEY in .env to unlock higher limits.
"""

import json
import logging
import os
from datetime import datetime, timezone

import aiohttp

from ..data import DataPoint
from ..state import SeenDocuments

_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

# BLS series ID → (canonical metric key, human label, unit)
SERIES: dict[str, tuple[str, str, str]] = {
    "CUUR0000SA0":   ("bls_cpi_u",    "CPI-U",                      "index"),
    "CES0000000001": ("bls_nfp",      "Nonfarm Payrolls",            "thousands"),
    "LNS14000000":   ("bls_unrate",   "Unemployment Rate",           "%"),
    "WPUFD4":        ("bls_ppi_fd",   "PPI Final Demand",            "index"),
    "WPUFD49116":    ("bls_ppi_core", "PPI Core (ex food/energy)",   "index"),
}


def _seen_key(series_id: str, period: str, year: str) -> str:
    """Unique deduplication key for a BLS data point."""
    return f"bls:{series_id}:{year}:{period}"


async def fetch_latest(
    session: aiohttp.ClientSession,
    seen: SeenDocuments,
) -> list[DataPoint]:
    """Fetch the latest value for each tracked BLS series.

    Only returns DataPoints for series/periods not previously seen,
    so this fires exactly once per new monthly release.

    Args:
        session: Shared aiohttp session.
        seen:    Shared deduplication store.

    Returns:
        List of new DataPoints (empty if no new releases detected).
    """
    api_key = os.environ.get("BLS_API_KEY", "")
    payload: dict = {
        "seriesid": list(SERIES.keys()),
        "latest": True,
    }
    if api_key:
        payload["registrationkey"] = api_key

    try:
        async with session.post(
            _API_URL,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            resp.raise_for_status()
            data: dict = await resp.json(content_type=None)
    except aiohttp.ClientResponseError as exc:
        logging.error("BLS HTTP error %s: %s", exc.status, exc.message)
        return []
    except aiohttp.ClientError as exc:
        logging.error("BLS request error: %s", exc)
        return []

    if data.get("status") != "REQUEST_SUCCEEDED":
        logging.error("BLS API returned non-success status: %s", data.get("status"))
        return []

    as_of = datetime.now(timezone.utc).isoformat()
    new_points: list[DataPoint] = []

    for series_result in data.get("Results", {}).get("series", []):
        series_id: str = series_result.get("seriesID", "")
        config = SERIES.get(series_id)
        if config is None:
            continue
        metric, label, unit = config

        latest_data = series_result.get("data", [])
        if not latest_data:
            continue

        # BLS returns newest first
        latest = latest_data[0]
        year: str = latest.get("year", "")
        period: str = latest.get("period", "")   # e.g. "M02" = February
        value_str: str = latest.get("value", "")

        try:
            value = float(value_str)
        except ValueError:
            logging.warning("BLS: could not parse value '%s' for %s", value_str, series_id)
            continue

        seen_key = _seen_key(series_id, period, year)
        if seen.contains(seen_key):
            logging.debug("BLS [%s]: %s %s already seen, skipping.", label, year, period)
            continue

        # New release detected
        period_label = f"{year}-{period}"
        logging.info("BLS NEW RELEASE [%s]: %s %s = %.3f %s", label, year, period, value, unit)

        new_points.append(
            DataPoint(
                source="bls",
                metric=metric,
                value=value,
                unit=unit,
                as_of=period_label,
                metadata={
                    "series_id": series_id,
                    "label": label,
                    "year": year,
                    "period": period,
                    "footnotes": [f.get("text", "") for f in latest.get("footnotes", []) if f],
                },
            )
        )
        seen.mark(seen_key, source="bls")

    return new_points
