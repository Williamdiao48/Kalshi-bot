"""CoinGecko price fetcher (free public API, no key required).

Returns current USD prices for the crypto assets tracked by Kalshi markets
as DataPoints. A single API call fetches all assets simultaneously.

API:
    GET https://api.coingecko.com/api/v3/simple/price
        ?ids=bitcoin,ethereum,solana,ripple&vs_currencies=usd

Rate limits (free tier): ~30 calls/minute. The polling loop should
respect this; a 60-second poll interval is conservative and safe.
"""

import logging
from datetime import datetime, timezone

import aiohttp

from ..data import DataPoint

_BASE_URL = "https://api.coingecko.com/api/v3/simple/price"

# CoinGecko coin ID → canonical metric key
COINS: dict[str, str] = {
    "bitcoin":  "price_btc_usd",
    "ethereum": "price_eth_usd",
    "solana":   "price_sol_usd",
    "ripple":   "price_xrp_usd",
}

# Ticker symbol labels for logging
_LABELS: dict[str, str] = {
    "bitcoin": "BTC", "ethereum": "ETH", "solana": "SOL", "ripple": "XRP"
}


async def fetch_prices(session: aiohttp.ClientSession) -> list[DataPoint]:
    """Fetch current USD prices for all tracked crypto assets.

    Returns one DataPoint per asset on success, empty list on failure.
    """
    params = {
        "ids": ",".join(COINS.keys()),
        "vs_currencies": "usd",
    }

    try:
        async with session.get(
            _BASE_URL,
            params=params,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            resp.raise_for_status()
            data: dict = await resp.json()
    except aiohttp.ClientResponseError as exc:
        logging.error("CoinGecko HTTP error %s: %s", exc.status, exc.message)
        return []
    except aiohttp.ClientError as exc:
        logging.error("CoinGecko request error: %s", exc)
        return []

    as_of = datetime.now(timezone.utc).isoformat()
    data_points: list[DataPoint] = []

    for coin_id, metric in COINS.items():
        price = data.get(coin_id, {}).get("usd")
        if price is None:
            logging.warning("CoinGecko: missing price for %s", coin_id)
            continue
        price = float(price)
        logging.info("CoinGecko [%s]: $%.4f", _LABELS[coin_id], price)
        data_points.append(
            DataPoint(
                source="coingecko",
                metric=metric,
                value=price,
                unit="USD",
                as_of=as_of,
                metadata={"coin_id": coin_id},
            )
        )

    return data_points
