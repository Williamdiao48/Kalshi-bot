import asyncio
import logging
import os
from typing import Any

import aiohttp

from .auth import generate_headers

KALSHI_ENVIRONMENT: str = os.environ.get("KALSHI_ENVIRONMENT", "demo")
KALSHI_API_BASE: str = (
    "https://api.elections.kalshi.com/trade-api/v2"
    if KALSHI_ENVIRONMENT == "production"
    else "https://demo-api.kalshi.co/trade-api/v2"
)

_MARKETS_PATH = "/trade-api/v2/markets"

# ---------------------------------------------------------------------------
# Series tickers for every numeric market type the bot trades.
# These are fetched directly via series_ticker= so they are never lost behind
# the flood of KXMVE sports markets that dominate the default pagination order.
# ---------------------------------------------------------------------------
NUMERIC_SERIES: tuple[str, ...] = (
    # Daily high temperature by city
    "KXHIGHLAX", "KXHIGHDEN", "KXHIGHCHI", "KXHIGHNY", "KXHIGHMIA",
    "KXHIGHDAL", "KXHIGHBOS", "KXHIGHAUS", "KXHIGHOU",
    # Crypto prices
    "KXBTCD", "KXBTC15M", "KXETH15M", "KXSOL15M", "KXXRP15M",
    "KXDOGE15M", "KXDOGE", "KXADA15M", "KXADA",
    "KXAVAX15M", "KXAVAX", "KXLINK15M", "KXLINK",
    # Forex
    "KXEURUSD", "KXUSDJPY", "KXGBPUSD",
    # Economics (BLS / DOL / ISM)
    "KXCPI", "KXNFP", "KXUNRATE", "KXPPI", "KXPCE",
    "KXJOBLESS", "KXICSA",                            # weekly initial jobless claims
    "KXISM", "KXISMMFG", "KXISMSVC",                  # ISM PMI indices
    # Interest rates (FRED)
    "KXFED", "KXFFR", "KXDGS10", "KXDGS2",
    # Energy (EIA)
    "KXWTI", "KXOIL", "KXNATGAS", "KXNG",
    # Equity indices
    "KXSPX", "KXSPXD", "KXNDX", "KXINXD", "KXDOW",
)

# General markets to fetch via pagination for text/political matching.
# This is intentionally capped: sports/entertainment markets dominate the
# default sort order and will never match our political/economic keywords.
_GENERAL_MARKET_LIMIT = 3000
_PAGE_DELAY = 0.25   # seconds between pages; keeps us well under Kalshi rate limits


def _normalize_market(m: dict[str, Any]) -> dict[str, Any]:
    """Normalize price fields: API now returns *_dollars string fields.

    The Kalshi API changed its response format: yes_bid/yes_ask (integer cents)
    were replaced by yes_bid_dollars/yes_ask_dollars (dollar strings like "0.02").
    This function back-fills the old integer-cent fields so the rest of the
    codebase continues to work without changes.
    """
    if m.get("yes_bid") is None and m.get("yes_bid_dollars") is not None:
        try:
            m["yes_bid"] = round(float(m["yes_bid_dollars"]) * 100)
        except (TypeError, ValueError):
            pass
    if m.get("yes_ask") is None and m.get("yes_ask_dollars") is not None:
        try:
            m["yes_ask"] = round(float(m["yes_ask_dollars"]) * 100)
        except (TypeError, ValueError):
            pass
    return m


async def _paginate(
    session: aiohttp.ClientSession,
    *,
    status: str,
    total_limit: int | None = None,
    extra_params: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Cursor-paginate the Kalshi /markets endpoint.

    Args:
        session:       Shared aiohttp session.
        status:        Market status filter ("open", "closed", "settled").
        total_limit:   Stop after collecting this many markets. None = exhaustion.
        extra_params:  Additional query parameters (e.g. series_ticker).

    Returns:
        Collected market dicts (may be fewer than total_limit on API error).
    """
    markets: list[dict[str, Any]] = []
    cursor: str | None = None

    while True:
        if total_limit is not None:
            remaining = total_limit - len(markets)
            if remaining <= 0:
                break
            page_size = min(100, remaining)
        else:
            page_size = 100

        params: dict[str, Any] = {"status": status, "limit": page_size}
        if cursor:
            params["cursor"] = cursor
        if extra_params:
            params.update(extra_params)

        headers = generate_headers("GET", _MARKETS_PATH)

        try:
            async with session.get(
                f"{KALSHI_API_BASE}/markets",
                params=params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 429:
                    logging.warning(
                        "Kalshi markets: rate-limited (429) after %d markets — stopping page.",
                        len(markets),
                    )
                    break
                resp.raise_for_status()
                data = await resp.json()
        except aiohttp.ClientResponseError as exc:
            logging.error("Kalshi markets HTTP error %s: %s", exc.status, exc.message)
            break
        except aiohttp.ClientError as exc:
            logging.error("Kalshi markets request error: %s", exc)
            break

        page = [_normalize_market(m) for m in data.get("markets", [])]
        markets.extend(page)

        cursor = data.get("cursor")
        if not cursor or not page:
            break

        await asyncio.sleep(_PAGE_DELAY)

    return markets


async def fetch_markets_by_series(
    session: aiohttp.ClientSession,
    series_tickers: tuple[str, ...] | list[str] = NUMERIC_SERIES,
    *,
    status: str = "open",
    limit_per_series: int = 200,
) -> list[dict[str, Any]]:
    """Fetch markets for specific series tickers directly.

    Uses the ``series_ticker`` query parameter which bypasses the default
    sort order that buries numeric/weather/crypto markets under thousands of
    sports markets.  Each series is fetched with a small delay to stay within
    Kalshi's rate limits.

    Args:
        session:           Shared aiohttp session.
        series_tickers:    Kalshi series identifiers (e.g. "KXHIGHLAX", "KXBTC15M").
        status:            Market status filter.
        limit_per_series:  Max markets to fetch per series.

    Returns:
        Deduplicated list of market dicts across all requested series.
    """
    all_markets: list[dict[str, Any]] = []
    seen_tickers: set[str] = set()

    for series in series_tickers:
        params: dict[str, Any] = {
            "status": status,
            "limit": min(limit_per_series, 100),
            "series_ticker": series,
        }
        headers = generate_headers("GET", _MARKETS_PATH)
        try:
            async with session.get(
                f"{KALSHI_API_BASE}/markets",
                params=params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status == 429:
                    logging.warning("Series fetch rate-limited for %s — skipping.", series)
                    await asyncio.sleep(2.0)
                    continue
                resp.raise_for_status()
                data = await resp.json()
        except Exception as exc:
            logging.warning("Series fetch error for %s: %s", series, exc)
            await asyncio.sleep(0.2)
            continue

        for m in data.get("markets", []):
            t = m.get("ticker", "")
            if t and t not in seen_tickers:
                seen_tickers.add(t)
                all_markets.append(_normalize_market(m))

        await asyncio.sleep(0.2)

    logging.info(
        "Series fetch: %d market(s) across %d series.", len(all_markets), len(series_tickers)
    )
    return all_markets


async def fetch_markets(
    session: aiohttp.ClientSession,
    *,
    status: str = "open",
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Fetch active markets from Kalshi asynchronously (up to ``limit`` total).

    Args:
        session: Shared aiohttp session.
        status:  Market status filter ("open", "closed", "settled").
        limit:   Max number of markets to return across all pages.

    Returns:
        List of market dicts as returned by the Kalshi API.
    """
    markets = await _paginate(session, status=status, total_limit=limit)
    logging.info("Fetched %d Kalshi markets.", len(markets))
    return markets


async def fetch_all_markets(
    session: aiohttp.ClientSession,
    *,
    status: str = "open",
) -> list[dict[str, Any]]:
    """Fetch markets from Kalshi using a two-pronged strategy:

    1. **Targeted series fetch** — directly fetches all markets for known
       numeric series (weather, crypto, forex, economic, energy) using the
       ``series_ticker`` parameter.  This bypasses the default sort order
       that places 10,000+ KXMVE sports markets before any numeric markets.

    2. **Throttled general pagination** — paginates up to
       ``_GENERAL_MARKET_LIMIT`` general markets at ``_PAGE_DELAY`` seconds
       per page to discover political/text-matchable markets without hitting
       Kalshi's rate limits.

    Results are deduplicated by ticker.

    Args:
        session: Shared aiohttp session.
        status:  Market status filter ("open", "closed", "settled").

    Returns:
        Combined, deduplicated list of market dicts.
    """
    # --- 1. Targeted numeric series (fast, precise) ---
    series_markets = await fetch_markets_by_series(session, status=status)

    # --- 2. General pagination (throttled, for political/text markets) ---
    general_markets = await _paginate(
        session, status=status, total_limit=_GENERAL_MARKET_LIMIT
    )

    # --- 3. Merge, deduplicate ---
    seen = {m["ticker"] for m in series_markets}
    combined = series_markets + [m for m in general_markets if m["ticker"] not in seen]

    logging.info(
        "Full market sync: %d total (%d series + %d general, %d deduped).",
        len(combined),
        len(series_markets),
        len(general_markets),
        len(general_markets) - (len(combined) - len(series_markets)),
    )
    return combined


async def fetch_market_detail(
    session: aiohttp.ClientSession,
    ticker: str,
) -> dict[str, Any] | None:
    """Fetch live detail for a single market ticker.

    Returns the market dict with live yes_bid, yes_ask, volume, last_price.
    Returns None on any error so callers can skip gracefully.
    """
    path = f"/trade-api/v2/markets/{ticker}"
    headers = generate_headers("GET", path)

    try:
        async with session.get(
            f"{KALSHI_API_BASE}/markets/{ticker}",
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
    except aiohttp.ClientResponseError as exc:
        logging.warning("Market detail HTTP %s for %s: %s", exc.status, ticker, exc.message)
        return None
    except aiohttp.ClientError as exc:
        logging.warning("Market detail request error for %s: %s", ticker, exc)
        return None

    market = data.get("market")
    return _normalize_market(market) if market else None
