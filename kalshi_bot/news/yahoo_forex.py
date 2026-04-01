"""Yahoo Finance intraday forex rate fetcher (free, no API key required).

Uses the Yahoo Finance chart API to fetch real-time spot exchange rates
for EUR/USD and USD/JPY throughout the forex trading session.

This source complements the ECB daily fix (frankfurter.py).  Frankfurter
publishes once per day at ~16:00 CET (~10:00 AM ET); this module provides
intraday updates every polling cycle so forex opportunities can be detected
before the ECB fix is published.

API:
    GET https://query1.finance.yahoo.com/v8/finance/chart/{ticker}
    params: interval=1m, range=1d

Kalshi metrics produced:
    rate_eur_usd  ← EURUSD=X   (USD per EUR, ECB-convention)
    rate_usd_jpy  ← USDJPY=X   (JPY per USD)
    rate_gbp_usd  ← GBPUSD=X   (USD per GBP)

Staleness:
    The ECB staleness gate (FOREX_MAX_STALE_DAYS) in main.py applies only to
    source="frankfurter" DataPoints.  Yahoo Finance rates are always current
    and bypass that filter.

Environment variables
---------------------
  YAHOO_FOREX_ENABLED   "true" | "false" — disable without removing the source.
                        Default: "true".
"""

import asyncio
import logging
import os
from datetime import datetime, timezone

import aiohttp

from ..data import DataPoint

YAHOO_FOREX_ENABLED: bool = (
    os.environ.get("YAHOO_FOREX_ENABLED", "true").lower() != "false"
)

_BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"

# Yahoo Finance ticker → (canonical metric, display label, unit)
_SYMBOLS: dict[str, tuple[str, str, str]] = {
    "EURUSD=X": ("rate_eur_usd", "EUR/USD", "EUR/USD"),
    "USDJPY=X": ("rate_usd_jpy", "USD/JPY", "USD/JPY"),
    "GBPUSD=X": ("rate_gbp_usd", "GBP/USD", "GBP/USD"),
}

# Yahoo Finance rejects requests without a realistic User-Agent.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}


async def _fetch_pair(
    session:   aiohttp.ClientSession,
    yf_ticker: str,
    metric:    str,
    label:     str,
    unit:      str,
) -> DataPoint | None:
    """Fetch the current spot rate for one forex pair from Yahoo Finance.

    Returns a DataPoint on success, None on any error.
    """
    url = _BASE_URL.format(ticker=yf_ticker)
    params = {"interval": "1m", "range": "1d"}
    try:
        async with session.get(
            url,
            params=params,
            headers=_HEADERS,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
    except aiohttp.ClientResponseError as exc:
        logging.warning(
            "Yahoo Forex HTTP %s for %s: %s", exc.status, yf_ticker, exc.message
        )
        return None
    except aiohttp.ClientError as exc:
        logging.warning("Yahoo Forex request error for %s: %s", yf_ticker, exc)
        return None

    try:
        result = data["chart"]["result"][0]
    except (KeyError, IndexError, TypeError):
        error = (data.get("chart") or {}).get("error") or {}
        logging.warning(
            "Yahoo Forex: bad response for %s — %s",
            yf_ticker, error.get("description", "unknown error"),
        )
        return None

    meta  = result.get("meta", {})
    price: float | None = meta.get("regularMarketPrice")
    ts:    int   | None = meta.get("regularMarketTime")

    if price is None:
        logging.warning("Yahoo Forex: no regularMarketPrice for %s", yf_ticker)
        return None

    as_of = (
        datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        if ts else
        datetime.now(timezone.utc).isoformat()
    )

    logging.info("Yahoo Forex [%s]: %.5f", label, price)
    return DataPoint(
        source   = "yahoo_forex",
        metric   = metric,
        value    = price,
        unit     = unit,
        as_of    = as_of,
        metadata = {"symbol": yf_ticker},
    )


async def fetch_rates(session: aiohttp.ClientSession) -> list[DataPoint]:
    """Fetch intraday spot rates for all tracked forex pairs from Yahoo Finance.

    Returns one DataPoint per successful fetch.
    """
    if not YAHOO_FOREX_ENABLED:
        return []

    tasks = [
        _fetch_pair(session, ticker, metric, label, unit)
        for ticker, (metric, label, unit) in _SYMBOLS.items()
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    data_points: list[DataPoint] = []
    for (ticker, _), result in zip(_SYMBOLS.items(), results):
        if isinstance(result, Exception):
            logging.warning("Yahoo Forex fetch error for %s: %s", ticker, result)
        elif result is not None:
            data_points.append(result)

    return data_points
