"""Synthetic spread opportunity detector.

Finds pairs of open Kalshi markets on the same underlying metric that, when
traded together (one YES and one NO), create a synthetic range position:

  Leg A (lower strike, buy YES): e.g. "BTC above $94k" YES @ 92¢
  Leg B (upper strike, buy NO):  e.g. "BTC above $98k" NO  @ 85¢

  Together: profit if $94k < BTC < $98k, loss if BTC breaks out.

This is equivalent to an explicit "between" market on [$94k, $98k], but
constructed from two standard binary contracts.  Useful when:

  - No explicit "between" market is listed for the range of interest.
  - Both individual legs already pass the bot's quality and liquidity gates.
  - The combined cost underprices the actual probability of staying in range.

Detection
---------
Pairs NumericOpportunity objects with:
  - Same metric AND same source
  - Same direction ("over" or "under")
  - One leg implied_outcome="YES" (data above lower strike)
  - Other leg implied_outcome="NO" (data below upper strike)
  → Together they define a range the live value currently sits in.

Pricing
-------
  cost_lo_cents = YES ask on lower leg (from ticker_detail["yes_ask"])
  cost_hi_cents = NO ask on upper leg  = 100 − ticker_detail["yes_bid"]
  total_cost    = cost_lo + cost_hi
  max_profit    = 200 − total_cost  (both legs resolve in our favour)

P(win)
------
  P(win) = P(YES on leg_lo) − P(YES direction on leg_hi)
  This equals the probability the value settles in [strike_lo, strike_hi].
  Computation is deferred to trade_executor (avoids circular import).

Environment variables
---------------------
  SPREAD_MIN_RANGE_WIDTH  Minimum width of the synthetic range (in the
                          metric's natural unit, e.g. °F or USD).
                          Pairs closer than this are skipped.  Default: 0.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from .numeric_matcher import NumericOpportunity

SPREAD_MIN_RANGE_WIDTH: float = float(
    os.environ.get("SPREAD_MIN_RANGE_WIDTH", "0.0")
)


@dataclass
class SpreadOpportunity:
    """Two-leg synthetic range position on the same underlying metric.

    For 'over' direction (most common):
      leg_lo: "above strike_lo" YES — data is above the lower bound
      leg_hi: "above strike_hi" NO  — data is below the upper bound
      Wins if the value settles in [strike_lo, strike_hi].

    For 'under' direction (mirror image):
      leg_lo: "below strike_lo" NO  — data is above the lower bound
      leg_hi: "below strike_hi" YES — data is below the upper bound
      Wins if the value settles in [strike_lo, strike_hi].

    P(win) = P(leg_lo underlying direction) − P(leg_hi underlying direction)
             (deferred to trade_executor._implied_p_yes for computation)
    """

    metric:     str
    source:     str
    as_of:      str
    data_value: float
    unit:       str
    direction:  str              # "over" | "under"

    # Lower bound leg and upper bound leg.
    # For "over" spreads: leg_lo.implied_outcome == "YES", leg_hi.implied_outcome == "NO"
    # For "under" spreads: semantics are swapped (see find_spread_opportunities)
    leg_lo: NumericOpportunity
    leg_hi: NumericOpportunity

    strike_lo:  float
    strike_hi:  float
    range_width: float           # strike_hi − strike_lo

    # Pricing in cents (None if market data unavailable)
    cost_lo_cents:    int | None  # cost to buy the lower-bound leg
    cost_hi_cents:    int | None  # cost to buy the upper-bound leg
    total_cost_cents: int | None  # cost_lo + cost_hi
    max_profit_cents: int | None  # 200 − total_cost (both legs win)


# ---------------------------------------------------------------------------
# Pricing helpers
# ---------------------------------------------------------------------------

def _yes_ask(detail: dict[str, Any] | None) -> int | None:
    """Price to buy YES (the ask side)."""
    if detail is None:
        return None
    v = detail.get("yes_ask")
    return int(v) if v is not None else None


def _no_ask(detail: dict[str, Any] | None) -> int | None:
    """Price to buy NO = 100 − yes_bid."""
    if detail is None:
        return None
    bid = detail.get("yes_bid")
    if bid is None:
        return None
    return max(1, 100 - int(bid))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def find_spread_opportunities(
    numeric_opps: list[NumericOpportunity],
    ticker_detail: dict[str, dict[str, Any]],
    *,
    min_range_width: float = SPREAD_MIN_RANGE_WIDTH,
) -> list[SpreadOpportunity]:
    """Find synthetic range positions from pre-filtered numeric opportunities.

    Both legs must have already passed the individual quality gates
    (weather ensemble spread, release window, liquidity, etc.) before
    being passed here.

    Args:
        numeric_opps:    Filtered opportunities (post all quality gates).
        ticker_detail:   Market detail dicts keyed by Kalshi ticker.
        min_range_width: Drop pairs whose range width is below this value.

    Returns:
        All valid spread pairs.  Multiple pairs per metric are returned when
        several adjacent strikes exist (caller ranks by score and takes the
        best one per metric).
    """
    # Index opps by (metric, source, direction) — skip non-striked / direction-only
    groups: dict[tuple[str, str, str], list[NumericOpportunity]] = {}
    for opp in numeric_opps:
        if opp.direction not in ("over", "under"):
            continue
        if opp.implied_outcome not in ("YES", "NO"):
            continue
        if opp.strike is None:
            continue
        key = (opp.metric, opp.source, opp.direction)
        groups.setdefault(key, []).append(opp)

    spreads: list[SpreadOpportunity] = []

    for (metric, source, direction), group in groups.items():
        yes_legs = [o for o in group if o.implied_outcome == "YES"]
        no_legs  = [o for o in group if o.implied_outcome == "NO"]

        if not yes_legs or not no_legs:
            continue

        for yes_leg in yes_legs:
            for no_leg in no_legs:
                assert yes_leg.strike is not None
                assert no_leg.strike is not None

                if direction == "over":
                    # "above X" YES: data > X (lower strike)
                    # "above Y" NO:  data < Y (upper strike)
                    # Spread wins when data settles in [X, Y]
                    if yes_leg.strike >= no_leg.strike:
                        continue
                    lo_strike = yes_leg.strike
                    hi_strike = no_leg.strike
                    leg_lo, leg_hi = yes_leg, no_leg

                    detail_lo = ticker_detail.get(leg_lo.market_ticker)
                    detail_hi = ticker_detail.get(leg_hi.market_ticker)
                    cost_lo = _yes_ask(detail_lo)   # buy YES on lower
                    cost_hi = _no_ask(detail_hi)    # buy NO on upper

                else:  # "under"
                    # "below X" NO:  data > X (lower strike, NO wins)
                    # "below Y" YES: data < Y (upper strike, YES wins)
                    # Spread wins when data settles in [X, Y]
                    if no_leg.strike >= yes_leg.strike:
                        continue
                    lo_strike = no_leg.strike
                    hi_strike = yes_leg.strike
                    leg_lo, leg_hi = no_leg, yes_leg

                    detail_lo = ticker_detail.get(leg_lo.market_ticker)
                    detail_hi = ticker_detail.get(leg_hi.market_ticker)
                    cost_lo = _no_ask(detail_lo)    # buy NO on lower
                    cost_hi = _yes_ask(detail_hi)   # buy YES on upper

                range_width = hi_strike - lo_strike
                if range_width < min_range_width:
                    continue

                total_cost = (
                    cost_lo + cost_hi
                    if cost_lo is not None and cost_hi is not None
                    else None
                )
                max_profit = (200 - total_cost) if total_cost is not None else None

                spreads.append(SpreadOpportunity(
                    metric           = metric,
                    source           = source,
                    as_of            = yes_leg.as_of,
                    data_value       = yes_leg.data_value,
                    unit             = yes_leg.unit,
                    direction        = direction,
                    leg_lo           = leg_lo,
                    leg_hi           = leg_hi,
                    strike_lo        = lo_strike,
                    strike_hi        = hi_strike,
                    range_width      = range_width,
                    cost_lo_cents    = cost_lo,
                    cost_hi_cents    = cost_hi,
                    total_cost_cents = total_cost,
                    max_profit_cents = max_profit,
                ))

    return spreads
