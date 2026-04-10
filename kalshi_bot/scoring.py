"""Opportunity scoring for the Kalshi information-alpha bot.

Each opportunity (text or numeric) is assigned a composite score in [0.0, 1.0]
that estimates how actionable and reliable the signal is. Higher = better.

Four sub-scores are combined via a weighted average. Each sub-score is
independently in [0.0, 1.0] and has a clear semantic meaning:

  SPREAD SCORE
    Measures market liquidity. A 0¢ spread scores 1.0; a spread at or above
    SPREAD_MAX_CENTS (default 20¢) scores 0.0. Linear interpolation between.
    Rationale: a wide spread means you lose a large fraction of any edge just
    crossing the spread to enter the position. Illiquid markets are not worth
    trading even if the signal is correct.

  UNCERTAINTY SCORE
    Measures how much room the market price has to move in response to new
    information. Derived from the bid/ask midpoint: a midpoint of 50¢ scores
    1.0 (maximum uncertainty → maximum potential edge); midpoints near 0¢ or
    100¢ score 0.0 (market is already nearly resolved, signal adds little).
    Formula: 1 - |midpoint - 50| / 50

  TEMPORAL SCORE
    Measures how relevant today's news is to the market's resolution timeline.
    A market closing in 0 days scores 1.0; a market closing at DAYS_OUT_MAX
    scores 0.0. Linear interpolation between.
    Rationale: breaking news matters most for markets resolving soon. A Trump
    headline today is less actionable for a market that resolves in six months.

  SPECIFICITY SCORE  (text only)
    Rewards multi-word matched phrases over single-word terms. Single words
    (e.g. "Bitcoin") score 0.33; two-word phrases score 0.67; three or more
    words (e.g. "Trump executive order") score 1.0.
    Rationale: broader single-word terms produce more false positives. A
    three-word phrase in a headline is a much stronger signal.

  EDGE SCORE  (numeric / poly only)
    Replaces the specificity score for numeric and external-forecast (poly)
    opportunities. Measures how clearly the live data value sits on the YES
    or NO side of the market strike (or how large the cross-platform divergence
    is for poly). Each metric has a reference scale (METRIC_EDGE_SCALES).
    An implied_outcome of UNKNOWN scores 0.0.
    Edge is the primary informational signal for data-driven trades, so it
    carries more weight here than specificity does in text scoring.

Weight comparison (text vs numeric/poly):

  Sub-score      Text    Numeric / Poly
  ----------     -----   --------------
  Spread         0.35    0.30   ← reduced; edge already penalises bad entries
  Uncertainty    0.25    0.25
  Temporal       0.25    0.20   ← reduced; data edges are less time-sensitive
  Specificity    0.15    —
  Edge           —       0.25   ← raised from 0.15; core signal for data trades

If orderbook data (bid/ask) is unavailable for a ticker, the spread and
uncertainty scores default to 0.0 — a hard penalty: we cannot price or
execute a trade without a live orderbook.
"""

from __future__ import annotations

import math

from .matcher import Opportunity
from .numeric_matcher import NumericOpportunity
from .polymarket_matcher import PolyOpportunity


# ---------------------------------------------------------------------------
# Weights — must sum to 1.0 for each scorer
# ---------------------------------------------------------------------------

# Text opportunity weights (sum = 1.0)
WEIGHT_SPREAD:         float = 0.35
WEIGHT_UNCERTAINTY:    float = 0.25
WEIGHT_TEMPORAL:       float = 0.25
WEIGHT_SPECIFICITY:    float = 0.15

# Numeric / poly opportunity weights (sum = 1.0)
# Edge is the primary informational signal for data-driven trades and carries
# more weight than specificity does in text scoring.  Spread and temporal are
# trimmed slightly to compensate.  A small source-reliability weight is added
# to discount signals from noisier or less authoritative data providers.
NUM_WEIGHT_SPREAD:     float = 0.25   # 0.30 → 0.25: 0.05 reallocated to source
NUM_WEIGHT_UNCERTAINTY: float = 0.25
NUM_WEIGHT_TEMPORAL:   float = 0.20
NUM_WEIGHT_EDGE:       float = 0.25
NUM_WEIGHT_SOURCE:     float = 0.05   # new: data-source reliability


# ---------------------------------------------------------------------------
# Calibration constants
# ---------------------------------------------------------------------------

# Spread at which spread_score reaches 0.0 (linear from 0 → SPREAD_MAX_CENTS).
SPREAD_MAX_CENTS: float = 20.0

# Maximum days-to-close used for temporal normalisation.
# Should match MARKET_MAX_DAYS_OUT in main.py; imported callers may pass their
# own value to keep the two in sync.
DAYS_OUT_MAX_DEFAULT: float = 30.0

# Half-life for the exponential temporal decay (days).
# Score = e^(-days / TEMPORAL_HALFLIFE), giving:
#   0 days → 1.000   (signal is maximally timely)
#   1 day  → 0.819
#   2 days → 0.670
#   5 days → 0.368   (one half-life: score halved)
#  10 days → 0.135
#  30 days → 0.002   (essentially zero)
# A 5-day half-life concentrates scoring weight on the next 1–3 days where
# breaking news provides genuine actionable edge.
TEMPORAL_HALFLIFE: float = 5.0

# Reference edge magnitudes per metric prefix.
# An edge equal to this value (or larger) yields edge_score = 1.0.
# Values below it scale linearly toward 0.0.
METRIC_EDGE_SCALES: dict[str, float] = {
    # City daily high/low temperature (°F) — 10°F clearance is a strong signal
    "temp_high": 10.0,
    "temp_low":  10.0,
    # Crypto prices (USD)
    "price_btc":  5_000.0,  # $5k above/below BTC strike
    "price_eth":    300.0,  # $300 above/below ETH strike
    "price_sol":     15.0,  # $15 above/below SOL strike
    "price_xrp":      0.10, # $0.10 above/below XRP strike
    "price_doge":     0.01, # $0.01 above/below DOGE strike (~8% of ~$0.12)
    "price_ada":      0.05, # $0.05 above/below ADA strike (~10% of ~$0.45)
    "price_avax":     2.0,  # $2 above/below AVAX strike (~10% of ~$20)
    "price_link":     1.0,  # $1 above/below LINK strike (~7% of ~$14)
    # Forex rates
    "rate_eur":      0.04,  # 4 cents EUR/USD clearance
    "rate_usd":      4.0,   # 4 JPY USD/JPY clearance
    "rate_gbp":      0.03,  # 3 cents GBP/USD clearance (slightly less volatile than EUR)
    # BLS economic series
    "bls_cpi":       0.5,   # 0.5 CPI index points
    "bls_nfp":     200.0,   # 200k jobs (values are in thousands)
    "bls_unrate":    0.5,   # 0.5 percentage points
    "bls_ppi_fd":    0.5,   # 0.5 index-point edge (same magnitude as CPI)
    "bls_ppi_core":  0.3,   # 0.3 index-point edge
    "fred_pce":      0.3,   # 0.3 index-point edge for PCE
    # DOL / FRED weekly jobless claims
    "fred_icsa":         20.0,  # 20k claims edge (ICSA in thousands; 20k ≈ 1 std dev)
    # ISM PMI indices (index points; 50 = expansion boundary)
    "ism_manufacturing":  5.0,  # 5pp above/below strike = strong signal
    "ism_services":       5.0,  # 5pp above/below strike = strong signal
    # FRED interest rates (% points)
    "fred_fedfunds": 0.25,  # 25bp edge in fed funds rate
    "fred_dgs10":    0.25,  # 25bp edge in 10yr Treasury yield
    "fred_dgs2":     0.25,  # 25bp edge in 2yr Treasury yield
    # EIA energy prices
    "eia_wti":       3.0,   # $3/bbl edge in WTI crude
    "eia_natgas":    0.20,  # $0.20/MMBtu edge in Henry Hub nat gas
    # Equity indices (points)
    "price_spx":    50.0,   # 50 S&P 500 points (≈1% of ~5,200)
    "price_ndx":   200.0,   # 200 Nasdaq points (≈1.1% of ~18,000)
    "price_dow":   300.0,   # 300 Dow Jones points (≈0.7% of ~42,000)
}

# Fallback scale used when no prefix matches.
_DEFAULT_EDGE_SCALE: float = 1.0

# Per-source reliability scores for the source sub-score.
# Ground-truth observed data scores 1.0; play-money/noisy forecasts score 0.5.
_SOURCE_SCORES: dict[str, float] = {
    "noaa_observed": 1.00,   # station-observed temperature — ground truth
    "metar":         0.90,   # FAA real-time station observed max — same ground-truth tier as noaa_observed;
                             # slight discount vs noaa_observed because METAR station may differ from
                             # the NWS CLI station Kalshi uses for settlement
    "nws_alert":     0.95,   # NWS official warning/alert — high-confidence
    "cme_fedwatch":  0.90,   # CME FedWatch futures-implied FOMC probabilities
    "polymarket":    0.90,   # real-money global prediction market
    "binance":       0.85,   # real-time exchange spot price
    "adp":           0.75,   # ADP Employment Report (60-70% corr. with BLS NFP)
    "chicago_pmi":   0.78,   # Chicago Business Barometer (r≈0.85 with ISM Mfg)
    "bls":           0.85,   # official US Bureau of Labor Statistics release
    "fred":          0.85,   # official Federal Reserve / Treasury data
    "predictit":     0.85,   # real-money US-regulated political market
    "frankfurter":   0.80,   # ECB official exchange rates
    "noaa":              0.75,   # NWS model-based daily forecast (MAE ≈ 3–4°F)
    "noaa_day2":         0.68,   # NWS day-2 forecast (MAE ≈ 5–7°F)
    "noaa_day3":         0.63,   # NWS day-3 forecast (MAE ≈ 6–9°F)
    "noaa_day4":         0.58,   # NWS day-4 forecast (MAE ≈ 8–11°F)
    "noaa_day5":         0.55,   # NWS day-5 forecast (MAE ≈ 9–13°F)
    "noaa_day6":         0.52,   # NWS day-6 forecast (MAE ≈ 11–15°F)
    "noaa_day7":         0.50,   # NWS day-7 forecast (MAE ≈ 12–17°F)
    "eia":               0.75,   # official EIA energy spot prices
    "eia_inventory":     0.65,   # EIA weekly inventory change → implied price (trend proxy, ~65% directional accuracy)
    "metaculus":         0.75,   # calibrated crowd forecasting
    "open_meteo":        0.70,   # Open-Meteo standard forecast (deterministic)
    "yahoo_finance":     0.85,   # official index level, ~1–3 min delay
    "yahoo_finance_premarket": 0.70,  # pre-market price (thinner market, wider spread)
    "owm":               0.60,   # third-party weather forecast (less calibrated)
    "manifold":      0.50,   # play-money market (noisy)
}
_DEFAULT_SOURCE_SCORE: float = 0.60  # fallback for unlisted sources

# Reference divergence (|ext_p − kalshi_p|) treated as "full edge" for poly
# scoring. 50pp divergence = edge_score 1.0; smaller divergences scale down.
POLY_DIVERGENCE_REF: float = 0.50


# ---------------------------------------------------------------------------
# Sub-score helpers
# ---------------------------------------------------------------------------

def _source_score(source: str) -> float:
    """Score [0, 1] for data-source reliability (numeric/poly opportunities only).

    Higher scores indicate more authoritative or better-calibrated sources.
    Unlisted sources fall back to _DEFAULT_SOURCE_SCORE (0.60).
    """
    return _SOURCE_SCORES.get(source, _DEFAULT_SOURCE_SCORE)

def _spread_score(bid: float | None, ask: float | None) -> float:
    """Score [0, 1] for bid-ask spread tightness.

    0¢ spread → 1.0; spread >= SPREAD_MAX_CENTS → 0.0.
    Returns 0.0 (hard penalty) when bid/ask data is unavailable — we cannot
    execute a trade without a live price, so missing data is not neutral.
    """
    if bid is None or ask is None:
        return 0.0
    spread = ask - bid
    if spread < 0:
        return 0.0  # corrupt/crossed orderbook
    return max(0.0, 1.0 - spread / SPREAD_MAX_CENTS)


def _uncertainty_score(bid: float | None, ask: float | None) -> float:
    """Score [0, 1] for market uncertainty (distance of midpoint from 50¢).

    Midpoint == 50 → 1.0 (maximum uncertainty, maximum potential edge).
    Midpoint == 0 or 100 → 0.0 (market nearly resolved, little edge available).
    Returns 0.0 (hard penalty) when bid/ask data is unavailable — same
    rationale as _spread_score: missing orderbook means not tradeable.
    """
    if bid is None or ask is None:
        return 0.0
    midpoint = (bid + ask) / 2.0
    return 1.0 - abs(midpoint - 50.0) / 50.0


def _temporal_score(days_to_close: float) -> float:
    """Score [0, 1] for how soon the market resolves.

    Uses exponential decay: score = e^(-days / TEMPORAL_HALFLIFE).

    This concentrates scoring weight on markets closing within the next few
    days, which is where breaking news provides genuine actionable edge.
    Unlike the previous linear model, the score never abruptly drops to 0 at
    a fixed boundary — it decays smoothly and is nearly zero by 30 days.

    Selected values (TEMPORAL_HALFLIFE = 5 days):
      0 days → 1.000   (closes today)
      1 day  → 0.819
      2 days → 0.670
      5 days → 0.368   (one half-life)
     10 days → 0.135
     30 days → 0.002

    The ``max_days`` parameter is retained for API compatibility but is no
    longer used in the formula.  Returns 0.0 for unknown / infinite values.
    """
    if days_to_close == float("inf") or days_to_close < 0:
        return 0.0
    return math.exp(-days_to_close / TEMPORAL_HALFLIFE)


_HIGH_SPEC_SINGLE_WORDS: frozenset[str] = frozenset({
    # Crypto
    "bitcoin", "btc", "ethereum", "eth", "solana", "xrp", "dogecoin", "doge",
    # Economics
    "cpi", "gdp", "nfp", "fomc", "tariff", "inflation", "recession",
    # Politics
    "shutdown", "impeachment", "ceasefire", "filibuster",
    # Sports leagues
    "nba", "nfl", "nhl", "mlb", "ncaa", "wnba",
    # Entertainment
    "billboard", "grammy", "oscar",
})


def _specificity_score(term: str) -> float:
    """Score [0, 1] for matched-term specificity (text opportunities only).

    Word count of the matched phrase, capped at 3:
      1 word  → 0.33  (generic) or 0.67 (well-known specific term)
      2 words → 0.67  (e.g. "Trump tariff", "Senate vote")
      3+ words → 1.0  (e.g. "Trump executive order")

    Well-known single-word domain terms (e.g. "bitcoin", "nba", "cpi") are
    scored at 0.67 because they are unambiguously specific despite being one word.
    """
    words = term.strip().lower().split()
    if len(words) >= 3:
        return 1.0
    elif len(words) == 2:
        return 0.67
    elif words and words[0] in _HIGH_SPEC_SINGLE_WORDS:
        return 0.67
    else:
        return 0.33


def _edge_score(metric: str, edge: float, implied_outcome: str) -> float:
    """Score [0, 1] for numeric signal clarity (numeric opportunities only).

    Looks up the reference scale for the metric prefix, then linearly maps
    the edge magnitude onto [0, 1], capped at 1.0.
    Returns 0.0 for UNKNOWN direction markets (no strike to compare against).
    """
    if implied_outcome == "UNKNOWN":
        return 0.0
    scale = next(
        (v for k, v in METRIC_EDGE_SCALES.items() if metric.startswith(k)),
        _DEFAULT_EDGE_SCALE,
    )
    return min(1.0, edge / scale)


# ---------------------------------------------------------------------------
# Public scoring functions
# ---------------------------------------------------------------------------

def score_text_opportunity(
    opp: Opportunity,
    detail: dict | None,
    days_to_close: float,
) -> float:
    """Compute a composite score in [0.0, 1.0] for a text/keyword opportunity.

    Args:
        opp:          The matched text opportunity.
        detail:       Live market detail dict from fetch_market_detail (may be
                      None if the orderbook fetch failed).
        days_to_close: Fractional days until market close_time. Pass
                       float("inf") if unknown.

    Returns:
        Weighted average of spread, uncertainty, temporal, and specificity
        sub-scores. All weights sum to 1.0.
    """
    bid = detail.get("yes_bid") if detail else None
    ask = detail.get("yes_ask") if detail else None

    term = opp.matched_terms[0] if opp.matched_terms else opp.topic.lower()

    s_spread      = _spread_score(bid, ask)
    s_uncertainty = _uncertainty_score(bid, ask)
    s_temporal    = _temporal_score(days_to_close)
    s_specificity = _specificity_score(term)

    return (
        WEIGHT_SPREAD      * s_spread
        + WEIGHT_UNCERTAINTY * s_uncertainty
        + WEIGHT_TEMPORAL    * s_temporal
        + WEIGHT_SPECIFICITY * s_specificity
    )


def score_numeric_opportunity(
    opp: NumericOpportunity,
    detail: dict | None,
    days_to_close: float,
) -> float:
    """Compute a composite score in [0.0, 1.0] for a numeric/data opportunity.

    Args:
        opp:          The matched numeric opportunity.
        detail:       Live market detail dict from fetch_market_detail (may be
                      None if the orderbook fetch failed).
        days_to_close: Fractional days until market close_time.

    Returns:
        Weighted average of spread, uncertainty, temporal, and edge sub-scores.
        Edge replaces specificity compared to the text scorer.
    """
    bid = detail.get("yes_bid") if detail else None
    ask = detail.get("yes_ask") if detail else None

    s_spread      = _spread_score(bid, ask)
    s_temporal    = _temporal_score(days_to_close)
    s_source      = _source_score(opp.source)

    # Locked-observation signals override the uncertainty sub-score to 1.0
    # when the observation DIRECTLY CONFIRMS the outcome — the market hasn't
    # fully repriced yet, but ground truth has already settled it.
    #
    # Locked YES:
    #   direction=over:    observed max exceeds the strike → YES locked anytime.
    #   direction=between: observed max is inside the band → YES confirmed after
    #                      the 4:30 PM local gate (peak is past).
    #   direction=under:   intentionally excluded — observation only shows the max
    #                      is still below the ceiling, not that the peak is done.
    #                      The market often correctly prices further warming risk.
    #
    # Locked NO (opp.peak_past set by _filter_weather_opportunities()):
    #   direction=over, after 4:30 PM local: observed max is BELOW the strike and
    #   the day's peak is established — outcome is locked NO.  A market pricing
    #   YES at 15¢ still offers 85¢ profit on the NO side; override the penalty.
    _locked_obs_sources = {"noaa_observed", "metar", "nws_alert"}
    _locked_directions  = {"over", "between"}
    if (
        opp.source in _locked_obs_sources
        and opp.implied_outcome == "YES"
        and opp.direction in _locked_directions
    ):
        s_uncertainty = 1.0
    elif (
        opp.source in _locked_obs_sources
        and opp.implied_outcome == "NO"
        and opp.direction == "over"
        and opp.peak_past
    ):
        # Peak is confirmed past 4:30 PM local; observed max < strike → locked NO.
        s_uncertainty = 1.0
    else:
        s_uncertainty = _uncertainty_score(bid, ask)

    # Between-market YES: opp.edge is the clearance (minimum distance from
    # either boundary), as set by numeric_matcher._implied_outcome.  Normalise
    # by half-width so score = 1.0 at centre, 0.0 at the boundary.
    # Between-market NO uses the standard edge score (distance from boundary).
    if (
        opp.direction == "between"
        and opp.implied_outcome == "YES"
        and opp.strike_lo is not None
        and opp.strike_hi is not None
    ):
        half_width = (opp.strike_hi - opp.strike_lo) / 2.0
        clearance = min(
            opp.data_value - opp.strike_lo,
            opp.strike_hi - opp.data_value,
        )
        s_edge = max(0.0, min(1.0, clearance / half_width)) if half_width > 0 else 0.0
    else:
        s_edge = _edge_score(opp.metric, opp.edge, opp.implied_outcome)

    return (
        NUM_WEIGHT_SPREAD      * s_spread
        + NUM_WEIGHT_UNCERTAINTY * s_uncertainty
        + NUM_WEIGHT_TEMPORAL    * s_temporal
        + NUM_WEIGHT_EDGE        * s_edge
        + NUM_WEIGHT_SOURCE      * s_source
    )


def score_poly_opportunity(
    opp: PolyOpportunity,
    detail: dict | None,
    days_to_close: float,
) -> float:
    """Compute a composite score in [0.0, 1.0] for an external-forecast opportunity.

    Uses the same four sub-scores as the numeric scorer so scores are directly
    comparable across opportunity types:
      - Spread / uncertainty from the Kalshi orderbook
      - Temporal proximity of the Kalshi market
      - Divergence magnitude (analogous to edge; POLY_DIVERGENCE_REF = 1.0)

    Args:
        opp:          The matched external-forecast opportunity.
        detail:       Live Kalshi market detail dict (may be None).
        days_to_close: Fractional days until the Kalshi market closes.
    """
    bid = detail.get("yes_bid") if detail else None
    ask = detail.get("yes_ask") if detail else None

    s_spread      = _spread_score(bid, ask)
    s_uncertainty = _uncertainty_score(bid, ask)
    s_temporal    = _temporal_score(days_to_close)
    s_edge        = min(1.0, opp.divergence / POLY_DIVERGENCE_REF)
    s_source      = _source_score(opp.source)

    return (
        NUM_WEIGHT_SPREAD        * s_spread
        + NUM_WEIGHT_UNCERTAINTY * s_uncertainty
        + NUM_WEIGHT_TEMPORAL    * s_temporal
        + NUM_WEIGHT_EDGE        * s_edge
        + NUM_WEIGHT_SOURCE      * s_source
    )
