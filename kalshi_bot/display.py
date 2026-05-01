"""Display helpers: console formatting for surfaced opportunities."""

from __future__ import annotations

from .matcher import Opportunity
from .numeric_matcher import NumericOpportunity
from .polymarket_matcher import PolyOpportunity

_SEP = "-" * 64


def fmt_liquidity(detail: dict | None) -> str:
    """Format bid/ask/spread/volume from a market detail dict."""
    if not detail:
        return "  Liquidity: (unavailable)"
    bid = detail.get("yes_bid")
    ask = detail.get("yes_ask")
    vol = detail.get("volume")
    if bid is not None and ask is not None:
        spread = ask - bid
        price_str = f"{bid}¢ bid / {ask}¢ ask  (spread: {spread}¢)"
    else:
        last = detail.get("last_price", "N/A")
        price_str = f"{last}¢ last  (no bid/ask)"
    vol_str = f"  |  Volume: {vol:,}" if vol is not None else ""
    return f"  Liquidity: {price_str}{vol_str}"


def fmt_position(net_position: int) -> str:
    """Format an existing position for inline display (returns empty string if flat)."""
    if net_position == 0:
        return ""
    side = "YES" if net_position > 0 else "NO"
    return f"  Position: {abs(net_position)} {side} contracts held"


def print_text_opportunity(
    idx: int, opp: Opportunity, detail: dict | None, score: float,
    existing_position: int = 0,
) -> None:
    print(_SEP)
    alts = f"  (+{opp.n_alternatives} similar markets)" if opp.n_alternatives else ""
    print(f"  [TEXT #{idx}  score={score:.2f}  display-only]  {opp.topic}  |  {opp.market_ticker}{alts}")
    print(f"  Market:   {opp.market_title}")
    print(fmt_liquidity(detail) + f"  |  Source: {opp.source}")
    pos_line = fmt_position(existing_position)
    if pos_line:
        print(pos_line)
    print(f"  Article:  {opp.doc_title}")
    print(f"  URL:      {opp.doc_url}")


def print_numeric_opportunity(
    idx: int, opp: NumericOpportunity, detail: dict | None, score: float,
    existing_position: int = 0,
) -> None:
    if opp.direction == "between":
        strike_str = f"{opp.strike_lo}–{opp.strike_hi}"
    elif opp.strike is not None:
        strike_str = str(opp.strike)
    else:
        strike_str = "N/A"

    print(_SEP)
    print(f"  [DATA #{idx}  score={score:.2f}]  {opp.metric}  |  {opp.market_ticker}")
    print(f"  Market:   {opp.market_title}")
    print(f"  Live:     {opp.data_value}{opp.unit}  (as of {opp.as_of})")
    print(
        f"  Strike:   {opp.direction.upper()} {strike_str}"
        f"  →  implied {opp.implied_outcome}  (edge {opp.edge:.3f})"
    )
    print(fmt_liquidity(detail) + f"  |  Source: {opp.source}")
    pos_line = fmt_position(existing_position)
    if pos_line:
        print(pos_line)


def print_poly_opportunity(
    idx: int, opp: PolyOpportunity, detail: dict | None, score: float,
    existing_position: int = 0,
) -> None:
    _SOURCE_LABELS = {
        "polymarket": "Polymarket",
        "metaculus":  "Metaculus",
        "manifold":   "Manifold",
        "predictit":  "PredictIt",
    }
    source_label = _SOURCE_LABELS.get(opp.source, opp.source.capitalize())
    if opp.source == "metaculus":
        liq_str = f"{opp.poly_liquidity:.0f} forecasters"
    elif opp.source == "predictit":
        liq_str = f"vol=${opp.poly_liquidity:,.0f}"
    else:
        liq_str = f"liq=${opp.poly_liquidity:,.0f}"
    print(_SEP)
    print(f"  [EXT #{idx}  score={score:.2f}  src={source_label}]  divergence={opp.divergence:.2%}  |  {opp.kalshi_ticker}")
    print(f"  Kalshi:   {opp.kalshi_title}")
    print(f"  Kalshi p: {opp.kalshi_mid:.0f}¢  →  side={opp.implied_side.upper()}")
    print(f"  {source_label} ({opp.poly_p_yes:.1%}  {liq_str}):")
    print(f"    {opp.poly_question}")
    print(f"  Match:    {opp.match_score:.2f}  |  " + fmt_liquidity(detail))
    pos_line = fmt_position(existing_position)
    if pos_line:
        print(pos_line)
