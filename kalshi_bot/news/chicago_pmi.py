"""Chicago Business Barometer (Chicago PMI) — leading indicator for ISM Manufacturing.

Chicago PMI is published by MNI Markets on the last business day of each month
at 9:45 AM ET, one business day *before* ISM Manufacturing (published 1st
business day of the next month at 10:00 AM ET).

Historical correlation with ISM Manufacturing: r ≈ 0.85.  The 50-point
boundary separates expansion (>50) from contraction (<50) — same interpretation
as ISM Mfg — so the value compares directly against KXISMMFG market strikes.

Uses FRED series CHIPMINDX (Chicago Business Barometer, monthly, index).
FRED updates within minutes of the MNI/ISM press release.

Emits DataPoint(metric="ism_manufacturing", source="chicago_pmi") so it feeds
directly into KXISMMFG / KXISM market matching.  The release window gate in
main.py exempts source="chicago_pmi" — this is a leading indicator, not the
actual ISM release, so the gate would otherwise block it for most of the month.

FRED_API_KEY must be set; returns empty list otherwise.
"""

import logging
import os
from datetime import datetime, timezone

import aiohttp

from ..data import DataPoint

_FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
_SERIES_ID = "CHIPMINDX"  # Chicago Business Barometer, monthly, index

# Module-level cache: dedup on observation date so the same monthly reading
# is only emitted once per bot restart rather than every 60-second poll cycle.
_last_seen_date: str | None = None
_series_available: bool = True  # set False on first 400 to suppress per-cycle errors


async def fetch_datapoints(session: aiohttp.ClientSession) -> list[DataPoint]:
    """Fetch the latest Chicago PMI from FRED.

    Returns a single DataPoint(metric="ism_manufacturing", source="chicago_pmi")
    on success, or an empty list if FRED_API_KEY is not set, the fetch fails,
    or the observation date has not changed since the last successful fetch (dedup).
    """
    global _last_seen_date, _series_available

    if not _series_available:
        return []

    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        return []

    params = {
        "series_id":  _SERIES_ID,
        "api_key":    api_key,
        "sort_order": "desc",
        "limit":      "3",
        "file_type":  "json",
    }
    try:
        async with session.get(
            _FRED_BASE,
            params=params,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
    except aiohttp.ClientResponseError as exc:
        if exc.status == 400:
            _series_available = False
            logging.warning(
                "Chicago PMI (FRED/%s) returned 400 — series likely removed from FRED; disabling.", _SERIES_ID
            )
        else:
            logging.error(
                "Chicago PMI (FRED/%s) HTTP error %s: %s", _SERIES_ID, exc.status, exc.message
            )
        return []
    except aiohttp.ClientError as exc:
        logging.error("Chicago PMI (FRED/%s) request error: %s", _SERIES_ID, exc)
        return []

    for obs in data.get("observations", []):
        raw = obs.get("value", ".")
        if raw == ".":
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        date_str = obs.get("date", "")
        if date_str == _last_seen_date:
            logging.debug("Chicago PMI (FRED/%s): same observation date %s — skipping", _SERIES_ID, date_str)
            return []
        _last_seen_date = date_str
        regime = "expansion" if value >= 50 else "contraction"
        logging.info(
            "Chicago PMI (Business Barometer): %.1f index (period %s) — %s",
            value, date_str, regime,
        )
        return [
            DataPoint(
                source="chicago_pmi",
                metric="ism_manufacturing",
                value=value,
                unit="index",
                as_of=date_str or datetime.now(timezone.utc).isoformat(),
                metadata={
                    "series_id":   _SERIES_ID,
                    "series_date": date_str,
                    "label":       "Chicago Business Barometer (Chicago PMI)",
                    "regime":      regime,
                },
            )
        ]

    logging.warning("Chicago PMI (FRED/%s): no valid observation found", _SERIES_ID)
    return []
