"""Combinatorial / logical arbitrage detector.

Identifies price monotonicity violations across same-underlying Kalshi markets
that guarantee a risk-free profit regardless of how the underlying resolves.

Logical constraint
------------------
For "over" markets on the same metric and close date, with k_lo < k_hi:

    P(value > k_lo)  ≥  P(value > k_hi)   (always true by definition)

Therefore:
    bid(YES on k_hi)  ≤  ask(YES on k_lo)

When this constraint is violated — bid(YES on k_hi) > ask(YES on k_lo) —
we can lock in a guaranteed profit:

    Buy YES on k_lo  @ ask(k_lo)          (cost: ask_lo cents)
    Buy NO  on k_hi  @ no_ask(k_hi)       (cost: 100 − bid_hi cents)
    Net cost = ask_lo + (100 − bid_hi)    (strictly < 100 when arb exists)

    If value > k_hi:  YES(k_lo) wins $1, NO(k_hi) loses → profit: 100 − ask_lo − (100−bid_hi)
    If k_lo < value ≤ k_hi:  YES(k_lo) wins, NO(k_hi) wins → profit: 200 − total_cost
    If value ≤ k_lo:  YES(k_lo) loses, NO(k_hi) wins $1 → profit: bid_hi − ask_lo

    Minimum profit across all outcomes = bid_hi − ask_lo > 0

For "under" markets the mirror holds:
    P(value < k_hi)  ≥  P(value < k_lo)   for k_lo < k_hi

    Arb when bid(YES on k_lo) > ask(YES on k_hi)
    Buy YES on k_hi @ ask_hi + Buy NO on k_lo @ (100 − bid_lo)
    Minimum profit = bid_lo − ask_hi > 0

Environment variables
---------------------
  ARB_MIN_PROFIT_CENTS  Minimum guaranteed profit per contract (default 2¢).
                        Lower values produce more opportunities but may not
                        cover transaction costs.  Raise after calibration.
  ARB_EXECUTION_ENABLED Set to 'false' to log arb opportunities without
                        executing them (detect-only mode).  Default: true.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from .market_parser import parse_market, ParsedMarket

ARB_MIN_PROFIT_CENTS: int = int(os.environ.get("ARB_MIN_PROFIT_CENTS", "2"))
ARB_EXECUTION_ENABLED: bool = (
    os.environ.get("ARB_EXECUTION_ENABLED", "true").lower() != "false"
)

CROSSED_BOOK_MIN_PROFIT: int = int(os.environ.get("CROSSED_BOOK_MIN_PROFIT", "2"))
CROSSED_BOOK_ARB_ENABLED: bool = (
    os.environ.get("CROSSED_BOOK_ARB_ENABLED", "true").lower() != "false"
)


@dataclass
class CrossedBookArb:
    """A single market where the order book is crossed (YES_ask + NO_ask < 100¢).

    Buying both YES and NO guarantees a profit of `profit` cents per contract,
    since exactly one side will pay out $1.00 at settlement.

    Execution:
      - Buy YES @ yes_ask
      - Buy NO  @ no_ask  (= 100 − yes_bid)
      Total cost = yes_ask + no_ask < 100¢ → guaranteed profit on settlement.
    """

    ticker: str
    yes_ask: int    # cents — cost to buy YES
    no_ask: int     # cents = 100 − yes_bid — cost to buy NO
    profit: int     # cents = 100 − yes_ask − no_ask (guaranteed at settlement)
    # Min depth across both sides; None if absent from API response.
    available_depth: int | None = None


def find_crossed_book_opportunities(
    markets: list[dict[str, Any]],
    *,
    min_profit: int = CROSSED_BOOK_MIN_PROFIT,
) -> list[CrossedBookArb]:
    """Scan all markets for crossed order books (YES_ask + NO_ask < 100¢).

    A crossed book arises when yes_ask < yes_bid — the ask is lower than the
    bid, which means buying YES and NO together costs less than $1 guaranteed
    payout.  This is rare on functioning exchanges but detectable.

    Args:
        markets:    Full list of open market dicts from the Kalshi API.
                    Each dict must have yes_ask and yes_bid fields (cents).
        min_profit: Minimum guaranteed profit per contract pair (cents).

    Returns:
        List of CrossedBookArb instances sorted highest-profit first.
    """
    result: list[CrossedBookArb] = []
    for m in markets:
        yes_ask = m.get("yes_ask")
        yes_bid = m.get("yes_bid")
        if yes_ask is None or yes_bid is None:
            continue
        yes_ask = int(yes_ask)
        yes_bid = int(yes_bid)
        if yes_ask <= 0 or yes_bid <= 0:
            continue
        no_ask = 100 - yes_bid
        profit = 100 - yes_ask - no_ask   # = yes_bid - yes_ask
        if profit >= min_profit:
            # Depth: minimum contracts available on each side of the book.
            d_yes = m.get("yes_ask_size")
            d_no  = m.get("yes_bid_size")
            depth: int | None = None
            if d_yes is not None and d_no is not None:
                depth = min(int(d_yes), int(d_no))
            result.append(CrossedBookArb(
                ticker=m.get("ticker", ""),
                yes_ask=yes_ask,
                no_ask=no_ask,
                profit=profit,
                available_depth=depth,
            ))
    result.sort(key=lambda x: x.profit, reverse=True)
    return result


@dataclass
class ArbOpportunity:
    """A risk-free arbitrage between two logically-related markets.

    For "over" direction:
      - Buy YES on ticker_lo (lower strike) @ ask_lo_cents
      - Buy NO  on ticker_hi (higher strike) @ (100 − bid_hi_cents)
      Guaranteed profit = bid_hi_cents − ask_lo_cents (> 0 by construction)

    For "under" direction:
      - Buy YES on ticker_hi (higher strike) @ ask_hi_cents
      - Buy NO  on ticker_lo (lower strike)  @ (100 − bid_lo_cents)
      Guaranteed profit = bid_lo_cents − ask_hi_cents (> 0 by construction)
    """

    metric: str
    direction: str        # "over" | "under"
    close_date: str       # ISO date string (from close_time, truncated to date)

    ticker_lo: str        # lower strike market ticker
    ticker_hi: str        # higher strike market ticker
    strike_lo: float
    strike_hi: float

    # For "over": buy YES on lo, buy NO on hi
    # For "under": buy YES on hi, buy NO on lo
    side_lo: str          # "yes" for "over", "no" for "under"
    side_hi: str          # "no" for "over", "yes" for "under"

    # YES-equivalent limit prices (cents, 0–100)
    limit_lo_cents: int   # yes_ask for YES buy; yes_bid for NO buy
    limit_hi_cents: int

    # Cost per contract pair
    cost_lo_cents: int    # actual cost for lo leg
    cost_hi_cents: int    # actual cost for hi leg

    guaranteed_profit_cents: int   # min profit regardless of outcome

    # Minimum available depth across both legs at the limit prices.
    # None when depth data is absent from the ticker_detail dict.
    # Used by execute_arb() to cap contract count at actual book liquidity.
    available_depth: int | None = None


def _yes_bid(detail: dict[str, Any] | None) -> int | None:
    if detail is None:
        return None
    v = detail.get("yes_bid")
    return int(v) if v is not None else None


def _yes_ask(detail: dict[str, Any] | None) -> int | None:
    if detail is None:
        return None
    v = detail.get("yes_ask")
    return int(v) if v is not None else None


def _depth(detail: dict[str, Any] | None, side: str) -> int | None:
    """Return the order book depth (contracts available) for a given side.

    'yes' → yes_ask_size (contracts available to buy YES)
    'no'  → yes_bid_size (contracts available to buy NO, expressed as YES bid depth)
    Returns None if the field is absent.
    """
    if detail is None:
        return None
    field = "yes_ask_size" if side == "yes" else "yes_bid_size"
    v = detail.get(field)
    return int(v) if v is not None else None


def _close_date(detail: dict[str, Any] | None) -> str:
    """Extract the close date (YYYY-MM-DD) from a market detail dict."""
    if detail is None:
        return ""
    ts = detail.get("close_time") or detail.get("expiration_time") or ""
    # ISO timestamps like "2026-03-19T20:00:00Z" → "2026-03-19"
    return ts[:10] if ts else ""


def find_arb_opportunities(
    markets: list[dict[str, Any]],
    ticker_detail: dict[str, dict[str, Any]],
    *,
    min_profit: int = ARB_MIN_PROFIT_CENTS,
) -> list[ArbOpportunity]:
    """Find all combinatorial arbitrage opportunities in the current market set.

    Args:
        markets:        Full list of open market dicts from the Kalshi API.
        ticker_detail:  Mapping from ticker → market detail dict (bid/ask/close_time).
        min_profit:     Minimum guaranteed profit per contract pair (cents).

    Returns:
        List of ArbOpportunity instances, each representing an immediately
        executable risk-free trade.
    """
    # Parse all markets; only keep "over" and "under" directions with a strike.
    parsed: list[tuple[ParsedMarket, dict[str, Any]]] = []
    for m in markets:
        ticker = m.get("ticker", "")
        detail = ticker_detail.get(ticker)
        if detail is None:
            continue
        pm = parse_market(m)
        if pm is None or pm.direction not in ("over", "under"):
            continue
        if pm.strike is None:
            continue
        parsed.append((pm, detail))

    # Group by (metric, direction, close_date)
    groups: dict[tuple[str, str, str], list[tuple[ParsedMarket, dict[str, Any]]]] = {}
    for pm, detail in parsed:
        cd = _close_date(detail)
        if not cd:
            continue
        key = (pm.metric, pm.direction, cd)
        groups.setdefault(key, []).append((pm, detail))

    arbs: list[ArbOpportunity] = []

    for (metric, direction, close_date), group in groups.items():
        if len(group) < 2:
            continue

        # Sort by strike ascending
        group_sorted = sorted(group, key=lambda x: x[0].strike)  # type: ignore[arg-type]

        if direction == "over":
            # For each adjacent (lo, hi) pair: arb if bid(hi) > ask(lo)
            for i in range(len(group_sorted) - 1):
                for j in range(i + 1, len(group_sorted)):
                    pm_lo, detail_lo = group_sorted[i]
                    pm_hi, detail_hi = group_sorted[j]

                    ask_lo = _yes_ask(detail_lo)
                    bid_hi = _yes_bid(detail_hi)
                    if ask_lo is None or bid_hi is None:
                        continue
                    if ask_lo <= 0 or bid_hi <= 0:
                        continue

                    # Arb: bid(hi) > ask(lo) means we can lock in guaranteed profit
                    profit = bid_hi - ask_lo
                    if profit < min_profit:
                        continue

                    # Execution:
                    #   buy YES on lo @ yes_ask (yes_price = ask_lo)
                    #   buy NO  on hi @ no_ask  (yes_price = yes_bid = bid_hi)
                    # Depth: YES side needs ask depth on lo; NO side needs bid depth on hi.
                    d_lo = _depth(detail_lo, "yes")
                    d_hi = _depth(detail_hi, "no")
                    depth = min(d_lo, d_hi) if d_lo is not None and d_hi is not None else None
                    arbs.append(ArbOpportunity(
                        metric=metric,
                        direction=direction,
                        close_date=close_date,
                        ticker_lo=pm_lo.ticker,
                        ticker_hi=pm_hi.ticker,
                        strike_lo=pm_lo.strike,   # type: ignore[arg-type]
                        strike_hi=pm_hi.strike,   # type: ignore[arg-type]
                        side_lo="yes",
                        side_hi="no",
                        limit_lo_cents=ask_lo,           # YES price for YES buy
                        limit_hi_cents=bid_hi,           # YES price for NO buy
                        cost_lo_cents=ask_lo,
                        cost_hi_cents=100 - bid_hi,
                        guaranteed_profit_cents=profit,
                        available_depth=depth,
                    ))

        else:  # "under"
            # For each (lo, hi) pair: arb if bid(lo) > ask(hi)
            # P(below k_lo) ≤ P(below k_hi) for k_lo < k_hi
            # Violation: bid(YES on k_lo) > ask(YES on k_hi)
            for i in range(len(group_sorted) - 1):
                for j in range(i + 1, len(group_sorted)):
                    pm_lo, detail_lo = group_sorted[i]
                    pm_hi, detail_hi = group_sorted[j]

                    bid_lo = _yes_bid(detail_lo)
                    ask_hi = _yes_ask(detail_hi)
                    if bid_lo is None or ask_hi is None:
                        continue
                    if bid_lo <= 0 or ask_hi <= 0:
                        continue

                    profit = bid_lo - ask_hi
                    if profit < min_profit:
                        continue

                    # Execution:
                    #   buy YES on hi @ yes_ask (yes_price = ask_hi)
                    #   buy NO  on lo @ no_ask  (yes_price = yes_bid = bid_lo)
                    # Depth: NO side needs bid depth on lo; YES side needs ask depth on hi.
                    d_lo = _depth(detail_lo, "no")
                    d_hi = _depth(detail_hi, "yes")
                    depth = min(d_lo, d_hi) if d_lo is not None and d_hi is not None else None
                    arbs.append(ArbOpportunity(
                        metric=metric,
                        direction=direction,
                        close_date=close_date,
                        ticker_lo=pm_lo.ticker,
                        ticker_hi=pm_hi.ticker,
                        strike_lo=pm_lo.strike,   # type: ignore[arg-type]
                        strike_hi=pm_hi.strike,   # type: ignore[arg-type]
                        side_lo="no",
                        side_hi="yes",
                        limit_lo_cents=bid_lo,           # YES price for NO buy on lo
                        limit_hi_cents=ask_hi,           # YES price for YES buy on hi
                        cost_lo_cents=100 - bid_lo,
                        cost_hi_cents=ask_hi,
                        guaranteed_profit_cents=profit,
                        available_depth=depth,
                    ))

    # Return highest-profit first
    arbs.sort(key=lambda a: a.guaranteed_profit_cents, reverse=True)
    return arbs
