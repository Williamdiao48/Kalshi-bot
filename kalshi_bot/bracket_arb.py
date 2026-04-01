"""Series bracket arbitrage detector for Kalshi temperature "between" markets.

Kalshi lists dozens of mutually exclusive, collectively exhaustive "between"
brackets for each city/date combination — e.g., all KXHIGHCHI-26APR01-B* markets
for Chicago on April 1.  Exactly one bracket resolves YES at settlement.

Logical constraints
-------------------
Because exactly one bracket must resolve YES:

  sum(YES prices) across all brackets  ≈  100¢

If the sum of YES_ask prices across all brackets is < 100¢, buying every bracket
costs less than the $1 guaranteed payout — risk-free profit.

If the sum of YES_bid prices across all brackets is > 100¢, selling every bracket
(buying NO on each) costs 100n − sum(YES_bid) < 100¢ per bracket on average,
again a risk-free profit.

Bracket grouping
----------------
The Kalshi API returns `event_ticker` (e.g., "KXHIGHCHI-26APR01") on every
market, which already groups all brackets for a given city/date.  We additionally
require:
  - strike_type == "between"
  - ≥ 3 brackets in the group (reduces noise)
  - Contiguous coverage: each bracket's cap_strike equals the next bracket's
    floor_strike (no gaps)

Environment variables
---------------------
  BRACKET_ARB_MIN_PROFIT   Minimum guaranteed profit in cents (default: 2).
  BRACKET_ARB_ENABLED      Set to 'false' for detect-only mode (default: true).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

BRACKET_ARB_MIN_PROFIT: int = int(os.environ.get("BRACKET_ARB_MIN_PROFIT", "2"))
BRACKET_ARB_ENABLED: bool = (
    os.environ.get("BRACKET_ARB_ENABLED", "true").lower() != "false"
)

_MIN_BRACKETS = 3   # require at least this many brackets to qualify a group


@dataclass
class BracketSetArb:
    """A series bracket arbitrage opportunity.

    Either buy all YES (if sum_yes_ask < 100) or buy all NO (if sum_yes_bid > 100).
    Exactly one bracket resolves YES, so one of the purchased contracts pays $1.

    Attributes:
        event_ticker:  Shared event ticker (e.g. "KXHIGHCHI-26APR01").
        tickers:       All bracket tickers in strike order.
        sum_yes_ask:   Total cost (¢) to buy YES on every bracket.
        sum_yes_bid:   Sum of YES bids — total cost of buying all NO =
                       100 × n_brackets − sum_yes_bid.
        yes_profit:    100 − sum_yes_ask  (>0 → buy-all-YES is profitable).
        no_profit:     sum_yes_bid − 100  (>0 → buy-all-NO is profitable).
        side:          "yes" or "no" — which side to trade.
        profit:        Guaranteed profit in cents for the chosen side.
        n_brackets:    Number of brackets in the set.
    """

    event_ticker: str
    tickers: list[str]
    sum_yes_ask: int
    sum_yes_bid: int
    yes_profit: int    # 100 - sum_yes_ask
    no_profit: int     # sum_yes_bid - 100
    side: str          # "yes" | "no"
    profit: int        # max(yes_profit, no_profit) — the actionable profit
    n_brackets: int
    yes_ask_prices: list[int] = field(default_factory=list)   # per-bracket YES ask prices
    yes_bid_prices: list[int] = field(default_factory=list)   # per-bracket YES bid prices


def find_bracket_set_opportunities(
    markets: list[dict[str, Any]],
    *,
    min_profit: int = BRACKET_ARB_MIN_PROFIT,
) -> list[BracketSetArb]:
    """Scan markets for series bracket arbitrage opportunities.

    Args:
        markets:    Full list of open market dicts from the Kalshi API.
                    Each dict should have: event_ticker, strike_type,
                    floor_strike, cap_strike, yes_ask, yes_bid, ticker.
        min_profit: Minimum guaranteed profit per contract set (cents).

    Returns:
        List of BracketSetArb sorted highest-profit first.
    """
    # Group "between" markets by event_ticker
    groups: dict[str, list[dict[str, Any]]] = {}
    for m in markets:
        if m.get("strike_type") != "between":
            continue
        et = m.get("event_ticker", "")
        if not et:
            continue
        groups.setdefault(et, []).append(m)

    results: list[BracketSetArb] = []

    for event_ticker, group in groups.items():
        if len(group) < _MIN_BRACKETS:
            continue

        # Sort by floor_strike; filter out brackets with missing price data
        valid = []
        for m in group:
            fs = m.get("floor_strike")
            cs = m.get("cap_strike")
            ya = m.get("yes_ask")
            yb = m.get("yes_bid")
            if fs is None or cs is None or ya is None or yb is None:
                continue
            if int(ya) <= 0 or int(yb) <= 0:
                continue
            valid.append(m)

        if len(valid) < _MIN_BRACKETS:
            continue

        valid.sort(key=lambda m: float(m["floor_strike"]))

        # Verify contiguous coverage (no gaps between brackets)
        contiguous = True
        for i in range(len(valid) - 1):
            if float(valid[i]["cap_strike"]) != float(valid[i + 1]["floor_strike"]):
                contiguous = False
                break

        if not contiguous:
            logging.debug(
                "Bracket group %s: skipped (non-contiguous strikes)", event_ticker
            )
            continue

        tickers = [m["ticker"] for m in valid]
        yes_asks = [int(m["yes_ask"]) for m in valid]
        yes_bids = [int(m["yes_bid"]) for m in valid]

        sum_yes_ask = sum(yes_asks)
        sum_yes_bid = sum(yes_bids)

        yes_profit = 100 - sum_yes_ask   # buy all YES: profit if sum_ask < 100
        no_profit  = sum_yes_bid - 100   # buy all NO:  profit if sum_bid > 100

        best_profit = max(yes_profit, no_profit)
        if best_profit < min_profit:
            continue

        side = "yes" if yes_profit >= no_profit else "no"

        results.append(BracketSetArb(
            event_ticker=event_ticker,
            tickers=tickers,
            sum_yes_ask=sum_yes_ask,
            sum_yes_bid=sum_yes_bid,
            yes_profit=yes_profit,
            no_profit=no_profit,
            side=side,
            profit=best_profit,
            n_brackets=len(valid),
            yes_ask_prices=yes_asks,
            yes_bid_prices=yes_bids,
        ))

    results.sort(key=lambda r: r.profit, reverse=True)
    return results
