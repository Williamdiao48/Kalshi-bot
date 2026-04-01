"""Coinbase Exchange public price fetcher (no API key required).

Uses the Coinbase Exchange (formerly Coinbase Pro) public REST API to fetch
real-time spot prices for all crypto assets tracked on Kalshi.  Acts as a
secondary price source alongside Binance; together they enable cross-exchange
price confirmation in numeric_matcher.py.

API:
    GET https://api.exchange.coinbase.com/products/{product_id}/ticker

Response (relevant fields):
    {"price": "84123.45", "bid": "84120.00", "ask": "84125.00", ...}

The ``source`` tag is ``"coinbase"`` — the numeric_matcher uses this name
when computing inter-exchange divergence.  The metric keys and units are
identical to those produced by binance.py so both DataPoints map to the
same Kalshi markets.

Rate limit: ~10 requests/second per IP for public endpoints.  We fetch all
8 symbols concurrently — well within the limit.

Environment variables
---------------------
  COINBASE_ENABLED   "true" | "false" — set to "false" to disable without
                     removing the source.  Default: "true".
"""

import asyncio
import logging
import os
from datetime import datetime, timezone

import aiohttp

from ..data import DataPoint

COINBASE_ENABLED: bool = (
    os.environ.get("COINBASE_ENABLED", "true").lower() != "false"
)

_BASE_URL = "https://api.exchange.coinbase.com/products/{product_id}/ticker"

# Coinbase product ID → canonical metric key (same keys as binance.py)
PRODUCTS: dict[str, str] = {
    "BTC-USD":  "price_btc_usd",
    "ETH-USD":  "price_eth_usd",
    "SOL-USD":  "price_sol_usd",
    "XRP-USD":  "price_xrp_usd",
    "DOGE-USD": "price_doge_usd",
    "ADA-USD":  "price_ada_usd",
    "AVAX-USD": "price_avax_usd",
    "LINK-USD": "price_link_usd",
}

# Display labels for logging
_LABELS: dict[str, str] = {
    "BTC-USD":  "BTC",
    "ETH-USD":  "ETH",
    "SOL-USD":  "SOL",
    "XRP-USD":  "XRP",
    "DOGE-USD": "DOGE",
    "ADA-USD":  "ADA",
    "AVAX-USD": "AVAX",
    "LINK-USD": "LINK",
}


async def _fetch_product(
    session:    aiohttp.ClientSession,
    product_id: str,
    metric:     str,
    as_of:      str,
) -> DataPoint | None:
    """Fetch the current price for one Coinbase product.

    Returns a DataPoint on success, None on any error.
    """
    url = _BASE_URL.format(product_id=product_id)
    try:
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
    except aiohttp.ClientResponseError as exc:
        logging.warning(
            "Coinbase HTTP %s for %s: %s", exc.status, product_id, exc.message
        )
        return None
    except aiohttp.ClientError as exc:
        logging.warning("Coinbase request error for %s: %s", product_id, exc)
        return None

    raw = data.get("price")
    if raw is None:
        logging.warning("Coinbase: missing 'price' for %s", product_id)
        return None

    try:
        price = float(raw)
    except (ValueError, TypeError):
        logging.warning("Coinbase: unparseable price %r for %s", raw, product_id)
        return None

    logging.info("Coinbase [%s]: $%.4f", _LABELS[product_id], price)
    return DataPoint(
        source   = "coinbase",
        metric   = metric,
        value    = price,
        unit     = "USD",
        as_of    = as_of,
        metadata = {"product_id": product_id},
    )


async def fetch_prices(session: aiohttp.ClientSession) -> list[DataPoint]:
    """Fetch current USD prices for all tracked crypto assets from Coinbase.

    Fetches all products concurrently.  Individual product failures are
    swallowed (returning no DataPoint for that symbol) so a single bad
    ticker doesn't block the rest of the batch.

    Returns:
        List of DataPoints, one per successful fetch.
    """
    if not COINBASE_ENABLED:
        return []

    as_of = datetime.now(timezone.utc).isoformat()
    tasks = [
        _fetch_product(session, product_id, metric, as_of)
        for product_id, metric in PRODUCTS.items()
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    data_points: list[DataPoint] = []
    for product_id, result in zip(PRODUCTS, results):
        if isinstance(result, Exception):
            logging.warning("Coinbase fetch error for %s: %s", product_id, result)
        elif result is not None:
            data_points.append(result)

    return data_points
