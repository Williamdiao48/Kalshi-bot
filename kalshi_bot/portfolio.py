"""Kalshi portfolio position fetcher.

Retrieves all current open positions from the Kalshi portfolio API and returns
them as a list of position dicts. This data is used in each poll cycle to:

  - Suppress re-alerting on markets where we already hold a meaningful position
    (configurable via POSITION_SKIP_CONTRACTS in main.py).
  - Display a portfolio summary (position count, total exposure) in the report
    header so the operator always has context when evaluating new signals.
  - Prepare for future order sizing logic that must account for existing
    exposure before placing additional trades.

API endpoint
------------
    GET /trade-api/v2/portfolio/positions

Cursor-paginated; returns all markets where net position != 0. Each page
returns up to 100 positions. For typical accounts (tens to low hundreds of
open positions) this completes in 1–3 requests.

Relevant fields in each position dict
--------------------------------------
    ticker                 str  — Kalshi market ticker
    position               int  — net YES contracts held
                                  > 0 → long YES (bought YES contracts)
                                  < 0 → long NO  (bought NO contracts, i.e. short YES)
    market_exposure        int  — current dollar exposure in cents
    resting_orders_count   int  — count of unfilled open orders in this market
    fees_paid              int  — total fees paid in cents (informational)

Error handling
--------------
Any HTTP or network error returns an empty list rather than propagating an
exception. This allows the portfolio fetch to fail gracefully — the bot
continues operating; opportunities are surfaced without position context,
and the report header will indicate that positions are unavailable.
"""

import logging
from typing import Any

import aiohttp

from .auth import generate_headers

_POSITIONS_PATH = "/trade-api/v2/portfolio/positions"


async def fetch_positions(
    session: aiohttp.ClientSession,
) -> list[dict[str, Any]]:
    """Fetch all open positions from the Kalshi portfolio API.

    Paginates to exhaustion so the complete position set is always returned,
    regardless of how many positions are held.

    Args:
        session: Shared aiohttp session (must have auth credentials configured).

    Returns:
        List of position dicts ordered as returned by the API (newest first).
        Returns an empty list on any error so callers degrade gracefully.
    """
    positions: list[dict[str, Any]] = []
    cursor: str | None = None

    while True:
        params: dict[str, Any] = {"limit": 100}
        if cursor:
            params["cursor"] = cursor

        headers = generate_headers("GET", _POSITIONS_PATH)

        try:
            async with session.get(
                f"https://api.elections.kalshi.com{_POSITIONS_PATH}"
                if _is_production()
                else f"https://demo-api.kalshi.co{_POSITIONS_PATH}",
                params=params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
        except aiohttp.ClientResponseError as exc:
            logging.error(
                "Portfolio positions HTTP error %s: %s", exc.status, exc.message
            )
            return positions  # return whatever we have so far
        except aiohttp.ClientError as exc:
            logging.error("Portfolio positions request error: %s", exc)
            return positions

        page = data.get("market_positions") or data.get("positions") or []
        positions.extend(page)

        cursor = data.get("cursor")
        if not cursor or not page:
            break

    logging.info(
        "Portfolio: %d open position(s) fetched.", len(positions)
    )
    return positions


def build_position_index(positions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index positions by ticker for O(1) lookup during opportunity matching.

    Args:
        positions: Raw position list from fetch_positions().

    Returns:
        Dict mapping ticker → position dict. Tickers with zero net position
        are excluded (the API should not return them, but this is defensive).
    """
    return {
        p["ticker"]: p
        for p in positions
        if p.get("ticker") and p.get("position", 0) != 0
    }


def summarise_portfolio(positions: list[dict[str, Any]]) -> str:
    """Return a one-line human-readable portfolio summary.

    Example: "Portfolio: 12 position(s)  |  $4,821.50 total exposure"

    Args:
        positions: Raw position list from fetch_positions().

    Returns:
        Formatted summary string, or a "(unavailable)" message if the list
        was empty due to a fetch error (caller must distinguish from genuinely
        holding no positions by checking whether an error was logged).
    """
    if not positions:
        return "Portfolio: 0 open position(s)"

    total_exposure_cents = sum(
        abs(p.get("market_exposure", 0)) for p in positions
    )
    total_exposure_dollars = total_exposure_cents / 100.0
    return (
        f"Portfolio: {len(positions)} open position(s)"
        f"  |  ${total_exposure_dollars:,.2f} total exposure"
    )


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _is_production() -> bool:
    """Return True if KALSHI_ENVIRONMENT is set to 'production'."""
    import os
    return os.environ.get("KALSHI_ENVIRONMENT", "demo") == "production"
