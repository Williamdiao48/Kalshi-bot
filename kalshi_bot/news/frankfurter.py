"""ECB/Frankfurter exchange rate fetcher (free, no key required).

Frankfurter proxies the European Central Bank's daily reference rates.
Rates update once per ECB business day around 16:00 CET.

API:
    GET https://api.frankfurter.app/latest?from=EUR&to=USD
    GET https://api.frankfurter.app/latest?from=USD&to=JPY

We make one call per base currency to keep the response small and parseable.
"""

import asyncio
import logging
from datetime import datetime, timezone

import aiohttp

from ..data import DataPoint

_BASE_URL = "https://api.frankfurter.app/latest"

# (base_currency, quote_currency) → canonical metric key
PAIRS: list[tuple[str, str, str]] = [
    ("EUR", "USD", "rate_eur_usd"),
    ("USD", "JPY", "rate_usd_jpy"),
    ("GBP", "USD", "rate_gbp_usd"),
]


async def _fetch_pair(
    session: aiohttp.ClientSession,
    base: str,
    quote: str,
    metric: str,
    as_of: str,
) -> DataPoint | None:
    params = {"from": base, "to": quote}
    try:
        async with session.get(
            _BASE_URL,
            params=params,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            resp.raise_for_status()
            data: dict = await resp.json()
    except aiohttp.ClientResponseError as exc:
        logging.error("Frankfurter HTTP error %s for %s/%s: %s", exc.status, base, quote, exc.message)
        return None
    except aiohttp.ClientError as exc:
        logging.error("Frankfurter request error for %s/%s: %s", base, quote, exc)
        return None

    rate = data.get("rates", {}).get(quote)
    if rate is None:
        logging.warning("Frankfurter: missing rate for %s/%s", base, quote)
        return None

    rate = float(rate)
    ecb_date: str = data.get("date", as_of)
    logging.info("Frankfurter [%s/%s]: %.5f (ECB date: %s)", base, quote, rate, ecb_date)

    return DataPoint(
        source="frankfurter",
        metric=metric,
        value=rate,
        unit=f"{base}/{quote}",
        as_of=ecb_date,
        metadata={"base": base, "quote": quote, "ecb_date": ecb_date},
    )


async def fetch_rates(session: aiohttp.ClientSession) -> list[DataPoint]:
    """Fetch current exchange rates for all tracked currency pairs.

    Returns one DataPoint per pair on success.
    """
    as_of = datetime.now(timezone.utc).isoformat()
    tasks = [_fetch_pair(session, base, quote, metric, as_of) for base, quote, metric in PAIRS]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    data_points: list[DataPoint] = []
    for (base, quote, _), result in zip(PAIRS, results):
        if isinstance(result, Exception):
            logging.error("Frankfurter fetch error for %s/%s: %s", base, quote, result)
        elif result is not None:
            data_points.append(result)

    return data_points
