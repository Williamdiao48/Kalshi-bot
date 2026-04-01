"""ADP National Employment Report — private payrolls pre-signal for KXNFP.

ADP publishes their National Employment Report at 8:15 AM ET on the Wednesday
before the BLS Nonfarm Payrolls release (first Friday of each month).
Historically correlated 60–70% with the final BLS NFP print.

Uses FRED series NPPTTL (ADP National Employment Report, Total Private,
Seasonally Adjusted, monthly, in thousands) as the data source.  FRED updates
within minutes of the ADP press release.

Emits DataPoint(metric="bls_nfp", source="adp") so it feeds directly into the
KXNFP market matching pipeline.  The source label distinguishes it from the
actual BLS release and assigns the appropriate lower reliability weight in
scoring (0.75 vs BLS 0.85).

The release window gate in main.py exempts source="adp" — this signal is
intended to be tradeable in the Wednesday–Friday window between the ADP release
and the BLS release that supersedes it.

FRED_API_KEY must be set; returns empty list otherwise.
"""

import logging
import os
from datetime import date, datetime, timezone

import aiohttp

from ..data import DataPoint

_FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
_SERIES_ID = "NPPTTL"  # ADP Total Private Employment, monthly, thousands (SA)

# Module-level cache: observation date of the last DataPoint emitted.
# If FRED still shows the same observation date as our last fetch, there is no
# new ADP release — return empty to avoid replaying the same signal every 60s.
_last_seen_date: str | None = None


async def fetch_datapoints(session: aiohttp.ClientSession) -> list[DataPoint]:
    """Fetch the latest ADP National Employment Report from FRED.

    Returns a single DataPoint(metric="bls_nfp", source="adp") on success,
    or an empty list if FRED_API_KEY is not set, the fetch fails, or the
    observation date has not changed since the last successful fetch (dedup).
    """
    global _last_seen_date

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
        logging.error("ADP (FRED/%s) HTTP error %s: %s", _SERIES_ID, exc.status, exc.message)
        return []
    except aiohttp.ClientError as exc:
        logging.error("ADP (FRED/%s) request error: %s", _SERIES_ID, exc)
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
            logging.debug("ADP (FRED/%s): same observation date %s — skipping", _SERIES_ID, date_str)
            return []
        # Reject stale data — FRED series NPPTTL was discontinued and returns
        # observations from 2022.  A 180-day staleness guard prevents the bot
        # from acting on years-old ADP data as if it were current.
        try:
            obs_date = date.fromisoformat(date_str)
            age_days = (date.today() - obs_date).days
            if age_days > 180:
                logging.warning(
                    "ADP (FRED/%s): observation date %s is %d days old — "
                    "series may be discontinued; skipping",
                    _SERIES_ID, date_str, age_days,
                )
                return []
        except (ValueError, TypeError):
            pass
        _last_seen_date = date_str
        logging.info(
            "ADP National Employment Report: %.0fk private jobs (period %s)",
            value, date_str,
        )
        return [
            DataPoint(
                source="adp",
                metric="bls_nfp",
                value=value,
                unit="thousands",
                as_of=date_str or datetime.now(timezone.utc).isoformat(),
                metadata={
                    "series_id":   _SERIES_ID,
                    "series_date": date_str,
                    "label":       "ADP Total Private Employment",
                },
            )
        ]

    logging.warning("ADP (FRED/%s): no valid observation found", _SERIES_ID)
    return []
