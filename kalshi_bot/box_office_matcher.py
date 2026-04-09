"""Box office market matcher.

Matches weekend box office DataPoints to open Kalshi markets by fuzzy movie-title
similarity.  Returns NumericOpportunity objects that flow into the standard trade
executor without any modification.

Why a custom matcher (not the standard ticker-prefix pipeline):
    Kalshi box office market ticker formats are per-movie slugs (e.g.
    KXBOXMOVIE-AVENGERS-T150).  We cannot enumerate them in TICKER_TO_METRIC in
    advance.  Instead this module filters markets whose title text identifies them
    as box office markets, then matches DataPoints to those markets by movie-name
    Jaccard similarity — the same approach used by polymarket_matcher.py.

Environment variables:
    BOX_OFFICE_MIN_MATCH   Minimum Jaccard title-similarity score [0–1].
                           Default: 0.40 (at least 40% token overlap).
    BOX_OFFICE_MIN_EDGE_M  Minimum gap (in $M) between estimated gross and the
                           Kalshi strike before surfacing an opportunity.
                           Default: 10.0 ($10M ≈ 1 studio-estimate σ).
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from .data import DataPoint
from .numeric_matcher import NumericOpportunity
from .market_parser import _OVER_RE, _UNDER_RE, _BETWEEN_RE, _to_float

BOX_OFFICE_MIN_MATCH: float = float(os.environ.get("BOX_OFFICE_MIN_MATCH", "0.40"))
BOX_OFFICE_MIN_EDGE_M: float = float(os.environ.get("BOX_OFFICE_MIN_EDGE_M", "10.0"))

# Keywords that identify a Kalshi market as a box office market.
_BO_KEYWORDS = ("opening weekend", "box office", "weekend gross", "weekend box", "domestic gross")

# Stopwords stripped before Jaccard similarity.
_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or", "is",
    "will", "be", "its", "by", "with", "from", "that", "this", "it", "are",
    "was", "has", "have", "had", "as", "up", "do", "did", "not", "but",
    "million", "billion", "dollars", "gross", "earn", "make", "open", "weekend",
    "opening", "box", "office", "domestic", "total",
}


def _keywords(text: str) -> set[str]:
    """Lowercase word tokens from text, with stopwords removed."""
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 1}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _is_box_office_market(title: str) -> bool:
    """Return True if the market title looks like a box office question."""
    lower = title.lower()
    return any(kw in lower for kw in _BO_KEYWORDS)


def _extract_movie_name(title: str) -> str:
    """Best-effort extraction of the movie name from a market title.

    Handles patterns like:
      "Will [Movie Title] gross over $150M opening weekend?"
      "Will [Movie] earn more than $200 million at the box office?"
      "[Movie] opening weekend gross > $100M?"
    """
    # Remove everything after a numeric threshold pattern
    clean = re.sub(r"[\$><=]+\s*[\d,\.]+.*", "", title, flags=re.IGNORECASE)
    # Remove leading "Will" / "Can" / "Does"
    clean = re.sub(r"^\s*(?:will|can|does|did|is|was)\s+", "", clean, flags=re.IGNORECASE)
    # Remove trailing filler phrases
    clean = re.sub(
        r"\s+(?:gross|earn|make|open(?:ing)?|at\s+the\s+box\s+office|"
        r"in\s+its\s+opening|domestic|box\s+office|opening\s+weekend).*$",
        "", clean, flags=re.IGNORECASE,
    )
    return clean.strip(" ?!.,")


def _parse_strike(title: str) -> tuple[str, float | None, float | None, float | None]:
    """Extract direction and strike(s) from a market title.

    Returns (direction, strike, strike_lo, strike_hi).
    Dollar amounts like "$150M" or "$150 million" are normalised to millions.
    """
    # Normalise "$150M" → "$150", "$150 million" → "$150"
    norm = re.sub(r"\$\s*([\d,]+)\s*[Mm](?:illion)?", lambda m: f"${m.group(1)}", title)

    if m := _OVER_RE.search(norm):
        return "over", _to_float(m.group(1)), None, None
    if m := _UNDER_RE.search(norm):
        return "under", _to_float(m.group(1)), None, None
    if m := _BETWEEN_RE.search(norm):
        lo, hi = _to_float(m.group(1)), _to_float(m.group(2))
        if lo > hi:
            lo, hi = hi, lo
        return "between", None, lo, hi
    return "unknown", None, None, None


def match_box_office_to_kalshi(
    data_points: list[DataPoint],
    kalshi_markets: list[dict[str, Any]],
) -> list[NumericOpportunity]:
    """Match box office DataPoints to open Kalshi box office markets.

    Args:
        data_points:    Box office DataPoints (source="box_office").
        kalshi_markets: Full list of open Kalshi market dicts.

    Returns:
        NumericOpportunity list for markets where our gross estimate
        disagrees with the strike by at least BOX_OFFICE_MIN_EDGE_M.
    """
    # Filter to only box office DataPoints
    bo_points = [dp for dp in data_points if dp.source == "box_office"]
    if not bo_points:
        return []

    # Filter Kalshi markets to box office candidates
    bo_markets = [m for m in kalshi_markets if _is_box_office_market(m.get("title", ""))]
    if not bo_markets:
        return []

    logging.debug(
        "Box office matcher: %d DataPoint(s), %d candidate market(s).",
        len(bo_points), len(bo_markets),
    )

    # Pre-compute keyword sets for all DataPoints
    dp_kw = {
        dp.metric: (dp, _keywords(dp.metadata.get("movie_title", dp.metric)))
        for dp in bo_points
    }

    opportunities: list[NumericOpportunity] = []
    matched_pairs: set[tuple[str, str]] = set()  # (dp.metric, market_ticker)

    for market in bo_markets:
        ticker = market.get("ticker", "")
        title = market.get("title", "")
        raw_price = market.get("last_price")

        movie_name = _extract_movie_name(title)
        market_kw = _keywords(movie_name)
        if not market_kw:
            continue

        # Find best-matching DataPoint by Jaccard score
        best_score = 0.0
        best_dp: DataPoint | None = None
        for dp, dp_words in dp_kw.values():
            score = _jaccard(market_kw, dp_words)
            if score > best_score:
                best_score = score
                best_dp = dp

        if best_dp is None or best_score < BOX_OFFICE_MIN_MATCH:
            continue

        pair = (best_dp.metric, ticker)
        if pair in matched_pairs:
            continue

        direction, strike, strike_lo, strike_hi = _parse_strike(title)
        if direction == "unknown":
            logging.debug("Box office: no strike parsed from '%s'", title)
            continue

        gross = best_dp.value  # $M

        # Compute edge and implied outcome
        if direction == "over" and strike is not None:
            edge = gross - strike
            implied = "YES" if edge > 0 else "NO"
        elif direction == "under" and strike is not None:
            edge = strike - gross
            implied = "YES" if edge > 0 else "NO"
        elif direction == "between" and strike_lo is not None and strike_hi is not None:
            if gross < strike_lo:
                edge = strike_lo - gross
                implied = "NO"
            elif gross > strike_hi:
                edge = gross - strike_hi
                implied = "NO"
            else:
                edge = min(gross - strike_lo, strike_hi - gross)
                implied = "YES"
        else:
            continue

        edge = abs(edge)
        if edge < BOX_OFFICE_MIN_EDGE_M:
            continue

        matched_pairs.add(pair)
        logging.info(
            "Box office: matched '%s' → %s (score=%.2f, gross=$%.1fM, "
            "strike=%s, edge=$%.1fM, %s)",
            best_dp.metadata.get("movie_title", best_dp.metric),
            ticker, best_score, gross,
            f"${strike}M" if strike is not None else f"${strike_lo}M–${strike_hi}M",
            edge, implied,
        )

        opportunities.append(NumericOpportunity(
            metric=best_dp.metric,
            data_value=gross,
            unit="$M",
            source="box_office",
            as_of=best_dp.as_of,
            market_ticker=ticker,
            market_title=title,
            current_market_price=raw_price,
            direction=direction,
            strike=strike,
            strike_lo=strike_lo,
            strike_hi=strike_hi,
            implied_outcome=implied,
            edge=edge,
            metadata={
                **best_dp.metadata,
                "match_score": round(best_score, 3),
            },
        ))

    return opportunities
