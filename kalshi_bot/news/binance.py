"""Binance public price fetcher (no API key required).

Replaces CoinGecko as the primary crypto price source.  Binance's public
REST API has no authentication requirement, a generous rate limit (~1200
requests/minute), and typically returns prices within 50–100ms — orders of
magnitude faster than CoinGecko's 60-second polling minimum.

A single batch request fetches all tracked symbols simultaneously.

API:
    GET https://api.binance.com/api/v3/ticker/price
        ?symbols=["BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT",
                  "DOGEUSDT","ADAUSDT","AVAXUSDT","LINKUSDT"]

Response:
    [{"symbol": "BTCUSDT", "price": "84123.45"}, ...]

The returned DataPoints are identical in structure to those from CoinGecko
(same metric keys, same unit) so the rest of the pipeline is unaffected.
"""

import json
import logging
from datetime import datetime, timezone

import aiohttp

from ..data import DataPoint

_BASE_URL = "https://api.binance.com/api/v3/ticker/price"

# Last fetched price per symbol — used to compute intra-cycle momentum (pct_change).
# Populated on the first fetch; pct_change is None until the second fetch.
_last_price: dict[str, float] = {}

# Binance trading pair → canonical metric key
SYMBOLS: dict[str, str] = {
    "BTCUSDT":  "price_btc_usd",
    "ETHUSDT":  "price_eth_usd",
    "SOLUSDT":  "price_sol_usd",
    "XRPUSDT":  "price_xrp_usd",
    "DOGEUSDT": "price_doge_usd",
    "ADAUSDT":  "price_ada_usd",
    "AVAXUSDT": "price_avax_usd",
    "LINKUSDT": "price_link_usd",
}

# Display labels for logging
_LABELS: dict[str, str] = {
    "BTCUSDT":  "BTC",
    "ETHUSDT":  "ETH",
    "SOLUSDT":  "SOL",
    "XRPUSDT":  "XRP",
    "DOGEUSDT": "DOGE",
    "ADAUSDT":  "ADA",
    "AVAXUSDT": "AVAX",
    "LINKUSDT": "LINK",
}


async def fetch_prices(session: aiohttp.ClientSession) -> list[DataPoint]:
    """Fetch current USD prices for all tracked crypto assets from Binance.

    Uses the batch ticker endpoint to retrieve all prices in a single HTTP
    request.  Returns one DataPoint per asset on success, empty list on
    failure (so the rest of the poll cycle proceeds unaffected).
    """
    symbols_param = json.dumps(list(SYMBOLS.keys()))

    try:
        async with session.get(
            _BASE_URL,
            params={"symbols": symbols_param},
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            resp.raise_for_status()
            data: list[dict] = await resp.json()
    except aiohttp.ClientResponseError as exc:
        logging.error("Binance HTTP error %s: %s", exc.status, exc.message)
        return []
    except aiohttp.ClientError as exc:
        logging.error("Binance request error: %s", exc)
        return []

    as_of = datetime.now(timezone.utc).isoformat()
    data_points: list[DataPoint] = []

    price_by_symbol = {item["symbol"]: item["price"] for item in data}

    for symbol, metric in SYMBOLS.items():
        raw = price_by_symbol.get(symbol)
        if raw is None:
            logging.warning("Binance: missing price for %s", symbol)
            continue
        price = float(raw)

        # Compute inter-cycle momentum: fractional price change vs. last fetch.
        # None on the first cycle (no previous price available yet).
        prev = _last_price.get(symbol)
        pct_change: float | None = (
            (price - prev) / prev if prev is not None and prev > 0 else None
        )
        _last_price[symbol] = price

        if pct_change is not None:
            logging.info(
                "Binance [%s]: $%.4f  %+.3f%%", _LABELS[symbol], price, pct_change * 100
            )
        else:
            logging.info("Binance [%s]: $%.4f", _LABELS[symbol], price)

        data_points.append(
            DataPoint(
                source="binance",
                metric=metric,
                value=price,
                unit="USD",
                as_of=as_of,
                metadata={"symbol": symbol, "pct_change": pct_change},
            )
        )

    return data_points
