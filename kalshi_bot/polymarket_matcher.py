"""Match external forecast platforms to Kalshi markets and surface price divergences.

Strategy
--------
For each active Kalshi market, we search for a matching question on Polymarket,
Metaculus, or Manifold using keyword-overlap (Jaccard similarity).  When a match
is found and the external platform disagrees by at least the configured divergence
threshold, we emit a PolyOpportunity.

The external price IS the p_estimate: if Polymarket says 70% and Kalshi says
40%, we buy YES on Kalshi using Kelly sizing with p=0.70.  Direction is always
determined by the external platform — we buy whichever side Kalshi has underpriced.

Signal quality hierarchy (highest to lowest):
  1. Polymarket   — real money, financially incentivized, very efficient
  1. PredictIt    — real money, regulated, specialises in US political markets
  2. Metaculus    — reputation-tracked crowd forecasting, well-calibrated
  3. Manifold     — play money (mana), active community but weaker incentives

Each source has independently tunable thresholds (all env-var overridable):

  POLY_MIN_DIVERGENCE / POLY_MIN_LIQUIDITY  / POLY_MIN_MATCH_SCORE
  PDIT_MIN_DIVERGENCE / PDIT_MIN_VOLUME     / PDIT_MIN_MATCH_SCORE
  META_MIN_DIVERGENCE / META_MIN_FORECASTERS / META_MIN_MATCH_SCORE
  MANI_MIN_DIVERGENCE / MANI_MIN_LIQUIDITY   / MANI_MIN_MATCH_SCORE

Matching notes
--------------
Jaccard similarity on word sets is simple and surprisingly effective for
prediction market questions, which tend to use the same noun phrases ("Fed",
"Bitcoin", "Ukraine", "election").  Stopwords and single characters are
stripped before comparison.  Numbers are kept — "2026" and "100000" are
discriminating tokens.
"""

import os
import re
from collections import defaultdict
from dataclasses import dataclass, field

from .news.polymarket import PolyMarket
from .news.metaculus import MetaculusQuestion, META_MIN_DIVERGENCE, META_MIN_MATCH_SCORE
from .news.manifold import ManifoldMarket, MANI_MIN_DIVERGENCE, MANI_MIN_LIQUIDITY, MANI_MIN_MATCH_SCORE
from .news.predictit import PredictItContract, PDIT_MIN_DIVERGENCE, PDIT_MIN_VOLUME, PDIT_MIN_MATCH_SCORE

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

POLY_MIN_DIVERGENCE: float  = float(os.environ.get("POLY_MIN_DIVERGENCE",  "0.20"))
POLY_MAX_DIVERGENCE: float  = float(os.environ.get("POLY_MAX_DIVERGENCE",  "0.65"))
POLY_MIN_LIQUIDITY:  float  = float(os.environ.get("POLY_MIN_LIQUIDITY",   "5000"))
POLY_MIN_MATCH_SCORE: float = float(os.environ.get("POLY_MIN_MATCH_SCORE", "0.40"))


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class PolyOpportunity:
    """A Kalshi market that an external forecast platform is pricing differently.

    The ``source`` field identifies which platform generated the signal:
      "polymarket" — Polymarket (real money, strongest signal)
      "metaculus"  — Metaculus (reputation-based, well-calibrated)
      "manifold"   — Manifold (play money, treated with more skepticism)
    """

    kalshi_ticker:  str
    kalshi_title:   str
    kalshi_mid:     float   # Kalshi mid-price in cents (0–100)
    poly_question:  str     # external platform's question text
    poly_market_id: str     # external platform's market/question ID
    poly_p_yes:     float   # external platform's YES probability (0.0–1.0)
    poly_liquidity: float   # platform liquidity (USD for Poly/EIA, mana for Manifold, forecasters for Metaculus)
    divergence:     float   # |ext_p_yes − kalshi_mid/100|
    implied_side:   str     # "yes" if ext > kalshi, "no" if ext < kalshi
    match_score:    float   # Jaccard similarity (0.0–1.0)
    source:         str = field(default="polymarket")  # "polymarket" | "metaculus" | "manifold"


# ---------------------------------------------------------------------------
# Keyword helpers
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset({
    "will", "the", "a", "an", "be", "is", "are", "was", "were",
    "in", "on", "at", "by", "for", "of", "to", "and", "or",
    "it", "its", "that", "this", "with", "from", "as", "not",
    "no", "yes", "above", "below", "than", "more", "less",
    "have", "has", "had", "do", "does", "did", "would", "could",
    "should", "which", "what", "when", "where", "who", "how",
    "if", "then", "there", "their", "they", "we", "you", "he",
    "she", "his", "her", "our", "your", "any", "all", "over",
    "under", "up", "down", "out", "into", "about", "after",
    "before", "between", "during", "through", "against", "within",
})

# Suffix rules applied longest-first within each group.
# Each entry is (suffix, min_stem_len).
# min_stem_len prevents over-stripping (e.g. "has" → "ha" would be wrong).
#
# Design notes:
#   - "tion" is intentionally absent: strip "ion" instead so that
#     "election" → "elect" (not "elec") and "elections" → "elect" both match.
#   - "es" is intentionally absent: strip "s" instead so that
#     "rates" → "rate" (not "rat") and "rate" stays "rate" — both match.
#   - Rules within the same length are ordered most-specific-first to avoid
#     a shorter-suffix from shadowing a longer one for a different word class.
_SUFFIX_RULES: tuple[tuple[str, int], ...] = (
    # 7-char suffixes
    ("ational", 4),  # relational   → relat
    ("ization", 4),  # legalization → legal
    # 6-char suffixes
    ("ations",  4),  # regulations  → regul
    ("nesses",  4),  # opennesses   → open
    # 5-char suffixes
    ("ation",   4),  # regulation   → regul
    ("ments",   4),  # agreements   → agre, appointments → appoint
    ("alism",   4),  # nationalism  → nation
    ("alist",   4),  # naturalist   → natur
    ("ality",   4),  # formality    → formal
    # 4-char suffixes
    ("ness",    4),  # openness     → open
    ("ment",    4),  # agreement    → agre
    ("ings",    4),  # rulings      → rul
    ("ions",    4),  # elections    → elect
    # 3-char suffixes
    ("ers",     3),  # traders      → trad
    ("ing",     4),  # trading      → trad
    ("ies",     3),  # policies     → polic
    ("ied",     3),  # ratified     → ratif
    ("ion",     4),  # election     → elect  (handles -tion words too)
    ("ous",     4),  # hazardous    → hazard
    ("ive",     4),  # decisive     → decis
    ("ful",     4),  # grateful     → grate
    # 2-char suffixes
    ("ed",      3),  # confirmed    → confirm
    ("ly",      3),  # quickly      → quick
    ("er",      3),  # trader       → trad
    ("al",      4),  # federal      → feder
    ("ic",      3),  # economic     → econom
    # 1-char suffixes
    ("s",       3),  # markets      → market, rates → rate
)


def _stem(word: str) -> str:
    """Apply a single-pass suffix-stripping stem to an English word.

    Purely mechanical — no lookup table.  Fast and dependency-free.
    Numbers and very short words are returned unchanged.
    """
    if not word.isalpha() or len(word) <= 3:
        return word

    for suffix, min_stem in _SUFFIX_RULES:
        if word.endswith(suffix):
            stem = word[: -len(suffix)]
            if len(stem) >= min_stem:
                return stem

    return word


def _keywords(text: str) -> set[str]:
    """Extract meaningful, stemmed tokens from a question or market title."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {_stem(t) for t in tokens if t not in _STOPWORDS and len(t) > 1}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# Generic matching core
# ---------------------------------------------------------------------------

def _match_external(
    kalshi_markets: list[dict],
    external_kw: list[tuple[str, str, float, float, set[str]]],
    min_match_score: float,
    min_divergence: float,
    source: str,
) -> list[PolyOpportunity]:
    """Generic Kalshi-vs-external matching loop using an inverted keyword index.

    Args:
        kalshi_markets:  Raw Kalshi market dicts.
        external_kw:     List of (market_id, question, p_yes, liquidity, keywords).
        min_match_score: Minimum Jaccard similarity to consider a pair.
        min_divergence:  Minimum |ext_p − kalshi_p| to emit an opportunity.
        source:          Source label for the PolyOpportunity.

    Returns:
        At most one PolyOpportunity per Kalshi ticker (highest divergence×score).

    Complexity:
        O(N × K + M × K) where N = Kalshi markets, M = external markets,
        K = average keywords per title. The inverted index eliminates Jaccard
        computation for external markets that share no keywords with the Kalshi
        market — typically the vast majority of pairs.
    """
    if not external_kw:
        return []

    # Build inverted index: keyword → list of external-item indices
    inv_index: dict[str, list[int]] = defaultdict(list)
    for i, (*_, ext_kw) in enumerate(external_kw):
        for word in ext_kw:
            inv_index[word].append(i)

    opportunities: list[PolyOpportunity] = []

    for km in kalshi_markets:
        ticker = km.get("ticker", "")
        title  = km.get("title", "")
        if not ticker or not title:
            continue

        # Exclude ticker prefixes that are unsuitable for external-forecast
        # matching: entertainment/sports/esports markets and numeric price-series
        # markets (crypto, forex, economics, energy, equity).  External-forecast
        # matching is only meaningful for true binary prediction markets (political,
        # regulatory) and KXHIGH* temperature markets.  Numeric series use dedicated
        # data-source signals — Jaccard matches against them are almost always
        # false positives (e.g. Polymarket's "Will BTC hit X?" vs Kalshi's 15-min
        # BTC strike contract are not commensurable).
        if ticker.startswith((
            # Sports / entertainment / esports (habitually illiquid or untradeable)
            "KXMVE",
            "KXNBA", "KXNHL", "KXMLB", "KXATP", "KXWBC",
            "KXLOL", "KXVALORANT",
            "KXTOPSONG", "KXTOP10BIL", "KXRT", "KXMAMDANIM",
            # Crypto price series — BTC and ETH removed (legitimate Poly divergence signals)
            "KXSOL", "KXXRP", "KXDOGE",
            "KXADA", "KXAVAX", "KXLINK",
            # Forex series
            "KXEUR", "KXUSD", "KXGBP",
            # Economics / BLS / DOL / ISM
            "KXCPI", "KXNFP", "KXUNRAT", "KXPPI", "KXPCE",
            "KXJOBLESS", "KXICSA", "KXISM",
            # Interest rates
            "KXFED", "KXFFR", "KXDGS",
            # Energy
            "KXWTI", "KXOIL", "KXNATGAS", "KXNG",
            # Equity indices
            "KXSPX", "KXNDX", "KXINXD", "KXDOW",
        )):
            continue

        # Try last_price first.  If absent, fall back to the bid/ask midpoint —
        # this allows political/economic markets that haven't traded recently
        # (so last_price is None) but DO have a live orderbook to participate
        # in external-forecast matching.  Bare 50¢ assumption is still avoided.
        last    = km.get("last_price")
        yes_bid = km.get("yes_bid")
        yes_ask = km.get("yes_ask")
        try:
            kalshi_mid = float(last)
            if not (0.0 <= kalshi_mid <= 100.0):
                raise ValueError("out of range")
        except (TypeError, ValueError):
            if yes_bid is not None and yes_ask is not None:
                kalshi_mid = (float(yes_bid) + float(yes_ask)) / 2.0
            else:
                # No usable price at all — skip to avoid phantom divergences.
                continue

        kalshi_kw = _keywords(title)
        if not kalshi_kw:
            continue

        # Collect only external items sharing ≥1 keyword with this Kalshi market
        candidate_indices: set[int] = set()
        for word in kalshi_kw:
            for i in inv_index.get(word, []):
                candidate_indices.add(i)

        if not candidate_indices:
            continue

        best_opp: PolyOpportunity | None = None

        for i in candidate_indices:
            ext_id, ext_question, ext_p_yes, ext_liquidity, ext_kw = external_kw[i]
            sim = _jaccard(kalshi_kw, ext_kw)
            if sim < min_match_score:
                continue

            divergence = abs(ext_p_yes - kalshi_mid / 100.0)
            if divergence < min_divergence:
                continue

            implied_side = "yes" if ext_p_yes > kalshi_mid / 100.0 else "no"

            opp = PolyOpportunity(
                kalshi_ticker=ticker,
                kalshi_title=title,
                kalshi_mid=kalshi_mid,
                poly_question=ext_question,
                poly_market_id=ext_id,
                poly_p_yes=ext_p_yes,
                poly_liquidity=ext_liquidity,
                divergence=divergence,
                implied_side=implied_side,
                match_score=sim,
                source=source,
            )
            if best_opp is None or (
                opp.divergence * opp.match_score
                > best_opp.divergence * best_opp.match_score
            ):
                best_opp = opp

        if best_opp is not None:
            opportunities.append(best_opp)

    return opportunities


# ---------------------------------------------------------------------------
# Public matching functions
# ---------------------------------------------------------------------------

def match_poly_to_kalshi(
    poly_markets: list[PolyMarket],
    kalshi_markets: list[dict],
) -> list[PolyOpportunity]:
    """Find Kalshi markets that Polymarket is pricing materially differently."""
    liquid_poly = [pm for pm in poly_markets if pm.liquidity >= POLY_MIN_LIQUIDITY]
    if not liquid_poly:
        return []

    external_kw = [
        (pm.market_id, pm.question, pm.p_yes, pm.liquidity, _keywords(pm.question))
        for pm in liquid_poly
    ]
    opps = _match_external(
        kalshi_markets, external_kw,
        min_match_score=POLY_MIN_MATCH_SCORE,
        min_divergence=POLY_MIN_DIVERGENCE,
        source="polymarket",
    )
    if POLY_MAX_DIVERGENCE > 0:
        capped = [o for o in opps if o.divergence <= POLY_MAX_DIVERGENCE]
        dropped = len(opps) - len(capped)
        if dropped:
            import logging
            logging.info(
                "Polymarket hard cap: blocked %d opportunity(ies) with divergence > %.0f%%"
                " (likely false-positive text matches).",
                dropped, POLY_MAX_DIVERGENCE * 100,
            )
        opps = capped
    return opps


def match_metaculus_to_kalshi(
    meta_questions: list[MetaculusQuestion],
    kalshi_markets: list[dict],
) -> list[PolyOpportunity]:
    """Find Kalshi markets that Metaculus community forecasts disagree with.

    Uses ``forecasters`` as the liquidity proxy — more forecasters = more
    reliable signal.
    """
    if not meta_questions:
        return []

    external_kw = [
        (mq.question_id, mq.title, mq.p_yes, float(mq.forecasters), _keywords(mq.title))
        for mq in meta_questions
    ]
    return _match_external(
        kalshi_markets, external_kw,
        min_match_score=META_MIN_MATCH_SCORE,
        min_divergence=META_MIN_DIVERGENCE,
        source="metaculus",
    )


def match_manifold_to_kalshi(
    mani_markets: list[ManifoldMarket],
    kalshi_markets: list[dict],
) -> list[PolyOpportunity]:
    """Find Kalshi markets that Manifold play-money markets disagree with.

    Higher divergence threshold (MANI_MIN_DIVERGENCE=0.25) accounts for
    reduced signal quality from play-money incentives.
    """
    liquid_mani = [mm for mm in mani_markets if mm.liquidity >= MANI_MIN_LIQUIDITY]
    if not liquid_mani:
        return []

    external_kw = [
        (mm.market_id, mm.question, mm.p_yes, mm.liquidity, _keywords(mm.question))
        for mm in liquid_mani
    ]
    return _match_external(
        kalshi_markets, external_kw,
        min_match_score=MANI_MIN_MATCH_SCORE,
        min_divergence=MANI_MIN_DIVERGENCE,
        source="manifold",
    )


def match_predictit_to_kalshi(
    pdit_contracts: list[PredictItContract],
    kalshi_markets: list[dict],
) -> list[PolyOpportunity]:
    """Find Kalshi markets that PredictIt is pricing materially differently.

    PredictIt is a real-money regulated market focused on US political and
    current-events outcomes.  Signal quality is equivalent to Polymarket —
    financially incentivised, well-calibrated on political questions.

    Lower divergence threshold (PDIT_MIN_DIVERGENCE=0.15 vs Poly's 0.20)
    reflects PredictIt's political specialisation: its prices on Senate
    confirmations, presidential actions, and elections are arguably more
    reliable than Polymarket's for those specific market types.

    Volume filter (PDIT_MIN_VOLUME) removes stale zero-activity contracts
    that haven't traded today and may carry outdated prices.
    """
    active = [c for c in pdit_contracts if c.volume >= PDIT_MIN_VOLUME]
    if not active:
        return []

    external_kw = [
        (c.market_id, c.question, c.p_yes, c.volume, _keywords(c.question))
        for c in active
    ]
    return _match_external(
        kalshi_markets, external_kw,
        min_match_score=PDIT_MIN_MATCH_SCORE,
        min_divergence=PDIT_MIN_DIVERGENCE,
        source="predictit",
    )
