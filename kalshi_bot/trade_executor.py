"""Dry-run and live trade execution for the Kalshi information-alpha bot.

Design
------
In TRADE_DRY_RUN=true mode (default) every intended trade is persisted to the
``trades`` table in ``opportunity_log.db`` with ``mode='dry_run'``.  No order
is sent to the API.  This lets you accumulate a realistic trade history over
many cycles and evaluate signal quality — "what would the bot have done?" —
before committing real capital.

Set TRADE_DRY_RUN=false to enable live order placement.  The same code path
runs either way: the only branch is whether ``_place_order`` is called.

Tradeable opportunities
-----------------------
Only *numeric* opportunities are automatically traded.  Keyword/text matches
surface *relevant* markets but cannot determine whether the news is YES-bullish
or NO-bullish without sentiment analysis, so they are logged as skipped at
DEBUG level.

Numeric opportunities with ``implied_outcome == "UNKNOWN"`` (direction-only
markets with no strike price) are also skipped — there is no clear edge to act
on.

Kelly sizing
------------
Contract count is determined by the fractional Kelly criterion.  For a binary
bet where we estimate P(win) and the market charges ``cost`` cents per contract:

    raw_kelly  =  (P(win) − cost/100) / (1 − cost/100)
    contracts  =  floor(KELLY_FRACTION × raw_kelly × MAX_POSITION_CENTS / cost)
    contracts  =  min(contracts, TRADE_MAX_CONTRACTS)

  YES buy: P(win) = p_estimate,       cost = yes_ask
  NO buy:  P(win) = 1 − p_estimate,   cost = 100 − yes_bid

``KELLY_FRACTION`` defaults to 0.25 (quarter-Kelly) — a conservative setting
that is typical before empirical win rates are available.  After the dry-run
period, use ``analyze_pnl.py`` to compute per-metric win rates and update
``KELLY_METRIC_PRIORS`` (or ``KELLY_DEFAULT_P``) accordingly.

``p_estimate`` is looked up from ``KELLY_METRIC_PRIORS`` (a JSON dict of
metric_prefix → probability, settable via the ``KELLY_METRIC_PRIORS`` env var),
falling back to ``KELLY_DEFAULT_P`` when the metric is not listed.

Example calibration update after dry run::

    KELLY_METRIC_PRIORS='{"temp_high": 0.68, "price_btc": 0.62}' \\
    KELLY_FRACTION=0.5 \\
    venv/bin/python run.py

Limit pricing
-------------
Limit prices are set at the current ask side so the order is immediately
marketable (aggressor fill):

  YES buy → ``yes_price = yes_ask``   (pay up to ask for YES contracts)
  NO buy  → ``yes_price = yes_bid``   (YES-equivalent price for a NO buy;
                                        the NO ask ≈ 100 − yes_bid)

If the relevant bid/ask field is absent from the orderbook the trade is
skipped rather than placing a blind market order.

Database schema (``trades`` table in ``opportunity_log.db``)
------------------------------------------------------------
  id               INTEGER  primary key
  logged_at        TEXT     ISO-8601 UTC timestamp
  mode             TEXT     'dry_run' | 'live'
  ticker           TEXT     Kalshi market ticker
  side             TEXT     'yes' | 'no'
  count            INTEGER  number of contracts (Kelly-sized)
  limit_price      INTEGER  yes_price in cents (0–100)
  opportunity_kind TEXT     'text' | 'numeric'
  score            REAL     composite score at time of trade decision
  kelly_fraction   REAL     KELLY_FRACTION multiplier used (for audit)
  p_estimate       REAL     P(win) estimate used in Kelly formula
  status           TEXT     'pending' (dry-run) | 'filled' | 'rejected' | 'error'
  order_id         TEXT     Kalshi order ID (live mode only; NULL in dry-run)
  error_msg        TEXT     populated when status = 'error'; NULL otherwise

Kalshi orders API reference
---------------------------
  POST /trade-api/v2/orders
  Body (JSON):
    {
      "ticker":          str,
      "client_order_id": str,   # idempotency UUID
      "action":          "buy",
      "type":            "limit",
      "side":            "yes" | "no",
      "count":           int,
      "yes_price":       int    # cents 0–100; YES-equivalent for both sides
    }
"""

import json
import logging
import math
import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiohttp

from .auth import generate_headers
from .matcher import Opportunity
from .market_parser import TICKER_TO_METRIC as _TICKER_TO_METRIC
from .numeric_matcher import NumericOpportunity
from .polymarket_matcher import PolyOpportunity
from .news import cme_fedwatch
from .news.noaa import get_forecast_sigma
from .scoring import METRIC_EDGE_SCALES
from .spread_matcher import SpreadOpportunity
from .arb_detector import (
    ArbOpportunity, ARB_EXECUTION_ENABLED,
    CrossedBookArb, CROSSED_BOOK_ARB_ENABLED,
)
from .bracket_arb import BracketSetArb, BRACKET_ARB_ENABLED
from .strike_arb import BandArbSignal, BAND_ARB_EXECUTION_ENABLED


# ---------------------------------------------------------------------------
# Configuration — all overridable via environment variables
# ---------------------------------------------------------------------------

# Default True so the bot never touches real capital without an explicit opt-in.
TRADE_DRY_RUN: bool = os.environ.get("TRADE_DRY_RUN", "true").lower() != "false"

# Hard cap on contracts per trade regardless of what Kelly recommends.
# Raise this after validating signal quality in dry-run.
TRADE_MAX_CONTRACTS: int = int(os.environ.get("TRADE_MAX_CONTRACTS", "10"))

# Minimum composite score required to attempt a trade.
# Raised 0.50 → 0.65 → 0.75: each band below the current threshold showed
# near-zero win rate and negative P&L across all resolved trades.
#   0.65–0.70: 35 trades, 2.86% win rate, -$3.80 (polymarket-free)
#   0.70–0.75: 26 trades, 11.54% win rate, -$7.31 (polymarket-free)
# Recent 14-day data confirms: <0.75 is 0% win rate, -$2.80.
# The 0.75–0.80 band shows 63–83% win rate in both windows.
TRADE_MIN_SCORE: float = float(os.environ.get("TRADE_MIN_SCORE", "0.75"))

# Lower minimum score for text opportunities.  Text composite scores are
# structurally capped below numeric ones (no edge component, specificity
# limited to 0.67 for single-word terms) so a single threshold disadvantages
# text unfairly.  Default 0.68 — just above the observed ceiling of crypto
# noise (~0.68) while allowing genuine politics/economics signals through.
# Set to 0 to fall back to TRADE_MIN_SCORE for all opportunity types.
TEXT_TRADE_MIN_SCORE: float = float(os.environ.get("TEXT_TRADE_MIN_SCORE", "0.68"))

# Minimum |p_yes - 0.5| required to act on a text opportunity's classifier signal.
# At AUC 0.675, require at least 20pp confidence (p_yes < 0.30 or > 0.70).
# Set to 0 to trade any non-NEUTRAL signal.
TEXT_MIN_CONFIDENCE: float = float(os.environ.get("TEXT_MIN_CONFIDENCE", "0.20"))

# Higher minimum score for forex (rate_eur_usd, rate_usd_jpy) markets.
# Forex short-duration contracts are driven by intraday moves that the ECB
# daily fix and Yahoo Forex intraday rate can't predict reliably.  All 5
# forex trades in the historical data were stop-lossed for a combined -$3.74.
# Default: 0.80 — requires a much stronger signal before entering a forex bet.
# Set to 0 to disable (falls back to TRADE_MIN_SCORE for forex).
FOREX_MIN_SCORE: float = float(os.environ.get("FOREX_MIN_SCORE", "0.80"))

# Metric prefixes that text/RSS trades should never touch.  Price metrics
# (crypto, forex) are driven by real-time exchange data — an RSS article
# about Bitcoin's inventor carries zero information about the close price.
# The numeric pipeline already handles these via Binance/Coinbase/Yahoo feeds.
# Comma-separated prefixes; default blocks all price_ and rate_ metrics.
_TEXT_SKIP_RAW: str = os.environ.get("TEXT_SKIP_METRIC_PREFIXES", "price_,rate_,eia_,fred_,bls_,ism_")
TEXT_SKIP_METRIC_PREFIXES: tuple[str, ...] = tuple(
    p.strip() for p in _TEXT_SKIP_RAW.split(",") if p.strip()
)

# Higher minimum score for Polymarket-divergence trades.
# Post-improvement polymarket was 5W/38L (-$2.72) even at score ≥ 0.65.
# Wins only appeared at score ≥ 0.82.  Below that the divergence signal
# has shown no predictive value on Kalshi direction.
# Set to 0 to disable (falls back to TRADE_MIN_SCORE for poly).
POLY_MIN_SCORE: float = float(os.environ.get("POLY_MIN_SCORE", "0.82"))

# Higher minimum score for day-ahead (noaa_day2) and longer-range (noaa_day3+)
# temperature forecasts.  Historical win rate on noaa_day2 trades 186–210 was
# ~25% — well below the breakeven rate for typical entry costs.  Raising the
# gate to 0.90 retains only the strongest day-ahead consensus signals.
# Set to 0 to disable (falls back to TRADE_MIN_SCORE).
NOAA_DAY2_MIN_SCORE: float = float(os.environ.get("NOAA_DAY2_MIN_SCORE", "0.90"))

# Minimum score for same-day NOAA forecast (noaa) and NOAA observed (noaa_observed)
# trades.  Score reflects multi-source corroboration; 0.80 requires at least one
# strong corroborating source.  Set to 0 to disable.
NOAA_MIN_SCORE: float = float(os.environ.get("NOAA_MIN_SCORE", "0.80"))
NOAA_OBSERVED_MIN_SCORE: float = float(os.environ.get("NOAA_OBSERVED_MIN_SCORE", "0.80"))

# Block noaa_observed YES trades when the market prices YES at less than this
# number of cents (i.e., market is ≥95% NO).  When YES ask < 5¢ the market has
# already priced in near-certainty of NO; our p_estimate=1.0 always loses.
NOAA_OBSERVED_MIN_YES_ASK: int = int(os.environ.get("NOAA_OBSERVED_MIN_YES_ASK", "5"))

# POLY_MAX_OPEN_PER_UNDERLYING — Maximum number of concurrent open positions
# allowed on the same underlying prefix (e.g. KXUSDJPY, KXBTCD).  Prevents
# the USD/JPY problem: 5 adjacent strikes all fire in one cycle and one bad
# forex move stops them all out.  Default 1 means only the best-scoring strike
# per underlying is held at any time.  Set to 0 to disable.
POLY_MAX_OPEN_PER_UNDERLYING: int = int(
    os.environ.get("POLY_MAX_OPEN_PER_UNDERLYING", "1")
)

# Enable/disable polymarket (and Metaculus/Manifold/PredictIt) trade execution.
# Disabled by default: 67 dry-run trades at 21% win rate produced -$15.04 P&L
# even with POLY_MIN_SCORE=0.82.  The text-similarity matching introduces too
# much noise.  Re-enable with POLY_ENABLED=true once matching quality improves.
POLY_ENABLED: bool = os.environ.get("POLY_ENABLED", "false").lower() == "true"

# Metrics to skip entirely, regardless of score or edge.
# Comma-separated.  Based on observed 0% win rate across all dry-run trades.
# price_doge: 0W/5L (-$1.00).  rate_usd: 0W/5L (-$3.74).
# Override via BLOCKED_METRICS env var (empty string to disable).
BLOCKED_METRICS: set[str] = set(
    m.strip()
    for m in os.environ.get("BLOCKED_METRICS", "rate_usd,price_doge").split(",")
    if m.strip()
)

# Score threshold for "patient" limit entry — posts one tick inside the bid
# (bid+1¢) rather than at the midpoint.  Only applies to non-urgent sources;
# urgent sources (noaa_observed, binance) always cross the spread immediately.
#
# Patient pricing captures 1–4¢ more of the spread on wide-spread markets by
# letting the market come to the bot rather than always paying midpoint.  If
# the patient order doesn't fill within FILL_TIMEOUT_MINUTES it is cancelled
# by the existing poll_open_orders() mechanism.
#
# Only activates when spread ≥ 3¢ (bid+1 < midpoint); on 1–2¢ spread markets
# bid+1 equals or exceeds the midpoint so the two tiers produce the same price.
PASSIVE_PATIENT_SCORE_THRESHOLD: float = float(
    os.environ.get("PASSIVE_PATIENT_SCORE_THRESHOLD", "0.85")
)

# Maximum dollars (in cents) to allocate to a single trade.  Kelly fraction
# scales down from this ceiling.  Default $7.50 = 750 cents.
# Raised from $5.00 after last-10-trade WR hit 80% (+$3.40 net); revisit at
# $10.00 after another 20 trades sustain ≥55% WR.
MAX_POSITION_CENTS: int = int(os.environ.get("MAX_POSITION_CENTS", "750"))

# Drawdown-based position scaling.
# When the equity curve falls below its peak, MAX_POSITION_CENTS is multiplied
# by a factor that decays linearly from 1.0 (at 0% drawdown) to
# DRAWDOWN_MIN_FACTOR (at DRAWDOWN_FULL_REDUCE_PCT drawdown or worse).
# This reduces exposure during losing streaks and restores it automatically
# as equity recovers.  Set DRAWDOWN_SCALING_ENABLED=false to disable.
DRAWDOWN_FULL_REDUCE_PCT: float = float(os.environ.get("DRAWDOWN_FULL_REDUCE_PCT", "0.20"))
DRAWDOWN_MIN_FACTOR: float      = float(os.environ.get("DRAWDOWN_MIN_FACTOR",       "0.25"))
DRAWDOWN_ENABLED: bool          = os.environ.get("DRAWDOWN_SCALING_ENABLED", "true").lower() == "true"
# How many recent resolved trades to use when computing the drawdown equity curve.
# Using all-time history unfairly penalises sizing for losses incurred when the
# bot's signal quality was much lower (e.g. pre-score-gate legacy trades).
# Default 50: covers ~5–7 weeks of recent history at current trade frequency.
# Set to 0 to use all-time history (original behaviour).
DRAWDOWN_LOOKBACK_TRADES: int = int(os.environ.get("DRAWDOWN_LOOKBACK_TRADES", "50"))

# Module-level drawdown factor, updated each cycle by main.py via
# set_drawdown_factor().  Applied to pos_max_cents in all trade paths.
_dd_factor: float = 1.0


def set_drawdown_factor(factor: float) -> None:
    """Update the global drawdown sizing factor.  Called once per poll cycle."""
    global _dd_factor
    _dd_factor = max(DRAWDOWN_MIN_FACTOR, min(1.0, factor))


# Fractional Kelly multiplier.  1.0 = full Kelly (aggressive).
# 0.25 = quarter-Kelly (conservative, recommended before calibration).
# 0.5  = half-Kelly (common practitioner choice after calibration).
KELLY_FRACTION: float = float(os.environ.get("KELLY_FRACTION", "0.25"))

# Elevated Kelly fraction for high-conviction trades (score ≥ KELLY_HIGH_SCORE_THRESHOLD).
# Applies a moderately higher fraction so signals we're most confident in get
# meaningfully larger positions while low-score trades remain conservatively sized.
# Default: 0.33 for scores ≥ 0.80; the standard KELLY_FRACTION applies below that.
KELLY_FRACTION_HIGH: float = float(os.environ.get("KELLY_FRACTION_HIGH", "0.40"))
KELLY_HIGH_SCORE_THRESHOLD: float = float(os.environ.get("KELLY_HIGH_SCORE_THRESHOLD", "0.80"))

# Position sizing overrides for locked-observation trades (noaa_observed, metar,
# nws_climo, nws_alert, eia, eia_inventory).  These positions are near-certain by
# definition: either the observation already confirms the outcome (weather station
# readings, EIA inventory report) or an NWS alert is in effect.  Use a larger
# position ceiling and aggressive Kelly fraction since the edge is structurally
# locked rather than probabilistic.
# LOCKED_OBS_MAX_POSITION_CENTS: $50 (raised from $25 — EIA 100% hold-to-settlement WR).
# LOCKED_OBS_KELLY_FRACTION: 0.75 — aggressive but not full-Kelly.
# LOCKED_OBS_MAX_CONTRACTS: 50 — hard cap (raised from 30).
LOCKED_OBS_MAX_POSITION_CENTS: int = int(os.environ.get("LOCKED_OBS_MAX_POSITION_CENTS", "5000"))
LOCKED_OBS_KELLY_FRACTION: float    = float(os.environ.get("LOCKED_OBS_KELLY_FRACTION",  "0.75"))
LOCKED_OBS_MAX_CONTRACTS: int       = int(os.environ.get("LOCKED_OBS_MAX_CONTRACTS",      "50"))
_LOCKED_OBS_SOURCES: frozenset[str] = frozenset({
    "noaa_observed", "metar", "nws_climo", "nws_alert",
    "eia", "eia_inventory",   # EIA reports are observed commodity data, not forecasts
})

# Maximum concurrent open positions allowed on the same underlying prefix
# when entering same-direction adjacent strikes.  Prevents unlimited stacking
# while allowing the bot to capture edge on 2-3 strikes of a temp ladder.
# Set to 1 to restore the old behavior (single position per underlying).
MAX_SAME_UNDERLYING_OPEN: int = int(os.environ.get("MAX_SAME_UNDERLYING_OPEN", "3"))

# Maximum total cost basis across all currently open positions (cents).
# Guards against correlated blowup when many weather markets are open
# simultaneously (e.g. LAX + LAS + PHX all NO on a heat-dome day — all
# lose together if the dome stalls).  The check fires AFTER Kelly sizing so
# it can compare the exact cost of the proposed trade against the live total.
# Default $150 (15000¢) — allows ~6 locked-obs trades at ~$23 each, or ~15
# standard trades at ~$10 each.  Set to 0 to disable.
MAX_TOTAL_EXPOSURE_CENTS: int = int(os.environ.get("MAX_TOTAL_EXPOSURE_CENTS", "15000"))

# Maximum contracts per leg for guaranteed-profit arb trades.
# Arb sizing is not Kelly-based (P(win)=1.0 by construction) — it is capped
# by available order book depth and this hard limit.  The global
# MAX_TOTAL_EXPOSURE_CENTS guard provides a secondary ceiling.
# Default 20: EUR/USD 37¢ arb at 20 pairs = $12.60 invested, $7.40 guaranteed.
# Bracket arb default is lower (ARB_MAX_CONTRACTS // 2) because it places N
# legs simultaneously and the per-event cost multiplies by bracket count.
ARB_MAX_CONTRACTS: int = int(os.environ.get("ARB_MAX_CONTRACTS", "20"))

# Score-weighted position sizing.  When enabled, the Kelly count is multiplied
# by (score − TRADE_MIN_SCORE) / (1.0 − TRADE_MIN_SCORE), linearly scaling
# positions from 0% at the score floor to 100% at a perfect score.
# This reduces exposure on low-conviction signals without blocking them
# entirely, while high-conviction signals get full Kelly allocation.
# Set SCORE_WEIGHTED_SIZING=false to disable.
SCORE_WEIGHTED_SIZING: bool = os.environ.get("SCORE_WEIGHTED_SIZING", "true").lower() != "false"

# Default P(win) estimate used when no per-metric prior is available.
# 0.60 is a conservative starting point — only slightly above break-even for
# a market priced at ~50¢.  Adjust per-metric after dry-run calibration.
KELLY_DEFAULT_P: float = float(os.environ.get("KELLY_DEFAULT_P", "0.60"))

# Per-metric P(win) priors loaded from JSON.  Keys are metric prefixes
# (e.g. "temp_high", "price_btc").  Set via env var after dry-run:
#   KELLY_METRIC_PRIORS='{"temp_high": 0.68, "price_btc": 0.62}'
# Minimum minutes between trades on the same ticker.  Used as the fallback
# cooldown when a position is still open, and for spread/arb/poly paths.
# Default 240 min (4 hours).  Set to 0 to disable all ticker cooldowns.
TRADE_TICKER_COOLDOWN_MINUTES: int = int(
    os.environ.get("TRADE_TICKER_COOLDOWN_MINUTES", "240")
)
# Cooldown when re-entering the SAME side after a cleanly exited trade.
# Short because the position is gone — position dedup already blocks re-entry
# while open.  30 min is enough to prevent same-cycle spam.
TRADE_TICKER_COOLDOWN_EXITED_MINUTES: int = int(
    os.environ.get("TRADE_TICKER_COOLDOWN_EXITED_MINUTES", "30")
)
# Cooldown when re-entering the OPPOSITE side (direction flip) after exit.
# More conservative than same-direction re-entry but shorter than the full
# 240-min window used while a position is still live.
TRADE_TICKER_COOLDOWN_FLIP_MINUTES: int = int(
    os.environ.get("TRADE_TICKER_COOLDOWN_FLIP_MINUTES", "150")
)
# Real-time model sources exempt from cross-strike cooldown inheritance.
# These are independent of day-ahead NWS forecasts, so a bad noaa_day2 trade
# should not lock out a HRRR signal on a different strike of the same market.
# Comma-separated list; exact-ticker cooldown still applies.
COOLDOWN_CROSS_STRIKE_EXEMPT_SOURCES: frozenset[str] = frozenset(
    s.strip()
    for s in os.environ.get("COOLDOWN_CROSS_STRIKE_EXEMPT_SOURCES", "hrrr").split(",")
    if s.strip()
)

# Adverse peak gate: if the opposing side of a forecast market has been priced
# above this threshold (¢) at any point in the recent look-back window, skip.
# Rationale: a spike to 75¢+ means the market had a moment of strong conviction
# against our signal — even if prices drifted back, that information persists.
# Only applies to sources listed in ADVERSE_PEAK_SOURCES.
# Example: HRRR says 52°F → NO on 66-67°F market.  If YES peaked at 87¢ (the
# market briefly agreed 87% that the high WOULD be 66-67°F), HRRR's model is
# likely wrong and entry should be blocked regardless of current YES price.
# Set ADVERSE_PEAK_THRESHOLD=0 to disable.
ADVERSE_PEAK_THRESHOLD: int = int(
    os.environ.get("ADVERSE_PEAK_THRESHOLD", "75")
)
ADVERSE_PEAK_HOURS: float = float(
    os.environ.get("ADVERSE_PEAK_HOURS", "4.0")
)
ADVERSE_PEAK_SOURCES: frozenset[str] = frozenset(
    s.strip()
    for s in os.environ.get(
        "ADVERSE_PEAK_SOURCES",
        "hrrr,noaa,owm,open_meteo,nws_hourly,weatherapi",
    ).split(",")
    if s.strip()
)

# Minimum temperature forecast edge (°F) before trading weather markets.
# NWS day-1 forecasts have MAE ≈ 3–4°F, so edges smaller than this are
# within normal forecast noise and carry no real signal.
# Set to 0 to disable. Only applies to "temp_high" metrics.
NUMERIC_MIN_TEMP_EDGE: float = float(os.environ.get("NUMERIC_MIN_TEMP_EDGE", "5.0"))

# Late-day UTC hour cutoff for daily temperature-high NO trades.
# After this UTC hour the daily maximum temperature for US ET cities is
# essentially already determined — new NO positions based on forecast data
# risk buying against a settled outcome.  Trade #200 (nws_hourly NO entered
# at 20:11 UTC / 4:11 PM ET) is the canonical failure this prevents.
# Only applies to tickers matching "KXHIGH" (Kalshi daily high markets).
# Default: 19 (3 PM ET / noon PT).  Set to 0 to disable.
TEMP_HIGH_NO_CUTOFF_UTC: int = int(os.environ.get("TEMP_HIGH_NO_CUTOFF_UTC", "19"))

# Minimum disagreement between model-implied P(YES) and market-implied P(YES)
# before trading. Prevents acting on signals the market has already priced in.
# E.g. 0.20 means our model must imply a probability ≥20pp different from the
# market mid. Set to 0 to disable.
NUMERIC_MIN_DISAGREEMENT: float = float(os.environ.get("NUMERIC_MIN_DISAGREEMENT", "0.15"))

# Extreme model-market disagreement guard.
# When our model is very confident (implied_p > EXTREME_DISAGREE_MODEL_P) but
# the market prices our side below EXTREME_DISAGREE_MARKET_CENTS, the market
# is near-certain we are wrong — almost certainly a station mismatch, wrong
# data source, or ticker cross.  Block unconditionally regardless of score.
# Set EXTREME_DISAGREE_MODEL_P=0 to disable.
EXTREME_DISAGREE_MODEL_P: float     = float(os.environ.get("EXTREME_DISAGREE_MODEL_P",     "0.85"))
EXTREME_DISAGREE_MARKET_CENTS: int  = int(  os.environ.get("EXTREME_DISAGREE_MARKET_CENTS", "25"))
# Symmetric NO-side guard: when model is very confident of NO (implied_p_yes < 1 − MODEL_P)
# but the market is pricing YES strongly (yes_bid > this threshold), block the trade.
# Default 45: if market says YES ≥ 45% but model says NO is ≥ 85% certain, trust the market.
# (Lowered from 55 — trade #200 entered at 51¢ YES, just under old 55¢ threshold.)
EXTREME_DISAGREE_NO_BID_MIN: int    = int(  os.environ.get("EXTREME_DISAGREE_NO_BID_MIN",   "45"))

# NWS day-1 temperature forecast uncertainty (1-sigma, °F).
# When TEMP_FORECAST_SIGMA is set in the environment it acts as a flat override
# for all cities and months — useful for quick experimentation.  When absent,
# _temp_forecast_sigma(metric) returns a per-city seasonal value from the
# lookup table in noaa.py (derived from published NWS NDFD verification data).
TEMP_FORECAST_SIGMA: float = float(os.environ.get("TEMP_FORECAST_SIGMA", "4.0"))
_TEMP_SIGMA_IS_OVERRIDE: bool = "TEMP_FORECAST_SIGMA" in os.environ


def _temp_forecast_sigma(metric: str, source: str = "noaa") -> float:
    """Return the NWS forecast σ (°F) for a temp_high metric.

    For extended forecasts (source="noaa_dayN", N≥2) the day-1 sigma is
    scaled up using the empirical NWS MAE growth rate of ~40% per additional
    day beyond day 1:

        multiplier = 1.0 + 0.4 × (N − 1)

        Day 1 (noaa):      1.0×  → ~3.5–5°F depending on city/season
        Day 2 (noaa_day2): 1.4×  → ~5–7°F
        Day 3 (noaa_day3): 1.8×  → ~6–9°F
        Day 4 (noaa_day4): 2.2×  → ~8–11°F
        Day 5 (noaa_day5): 2.6×  → ~9–13°F
        Day 7 (noaa_day7): 3.4×  → ~12–17°F

    Uses the per-city seasonal table from noaa.py unless TEMP_FORECAST_SIGMA
    is explicitly set in the environment (in which case the override wins).
    """
    if _TEMP_SIGMA_IS_OVERRIDE:
        base = TEMP_FORECAST_SIGMA
    else:
        base = get_forecast_sigma(metric, datetime.now(timezone.utc).month)
    if source.startswith("noaa_day"):
        try:
            day = int(source[len("noaa_day"):])
        except ValueError:
            day = 2
        base *= 1.0 + 0.4 * (day - 1)
    return base

# Effective 1-sigma uncertainty (°F) for station-observed temperatures.
# The old value of 0.5°F assumed the NOAA station IS Kalshi's settlement
# station, making any reading above the strike appear near-certain.  In
# practice, Kalshi settles against a specific major-airport ASOS station and
# the bot may query a different nearby station — introducing 1–3°F of
# structural mismatch.  2.0°F keeps observed signals well above forecast
# quality while reflecting the real inter-station uncertainty.
NOAA_OBSERVED_SIGMA: float = float(os.environ.get("NOAA_OBSERVED_SIGMA", "2.0"))

# P(win) estimate for trades driven by observed station data (source="noaa_observed").
# Observed temps are near-certain, so a higher prior is appropriate.
# Overrides KELLY_DEFAULT_P / KELLY_METRIC_PRIORS for observed-source trades.
NOAA_OBSERVED_P: float = float(os.environ.get("NOAA_OBSERVED_P", "0.80"))

# Maximum P(YES) returned by _implied_p_yes() for forecast-only sources
# (any source other than noaa_observed and nws_alert).  The Normal-CDF model
# can return p > 0.999 when the edge is many sigma — but NWS day-1 forecasts
# are not that reliable.  Capping at 0.95 prevents Kelly from over-sizing on
# overconfident forecast signals while leaving observed-data signals uncapped.
FORECAST_MAX_P: float = float(os.environ.get("FORECAST_MAX_P", "0.95"))

# Maximum P(YES) for crypto price signals (source="coinbase", metric="price_*").
# The Normal-CDF model assumes Gaussian returns, but BTC/ETH exhibit fat tails
# (flash crashes, pumps) that the model dramatically underestimates.  A p=0.9995
# from a $6k BTC gap to strike sounds confident but a $6k intraday move is a
# real tail risk.  0.90 is a hard ceiling regardless of edge size or time to close.
CRYPTO_MAX_P: float = float(os.environ.get("CRYPTO_MAX_P", "0.90"))

# Maximum P(YES) for observed-station data (source="noaa_observed").
# The Normal-CDF with σ=2°F can return p≈1.0 for large edges (e.g. observed
# temp 10°F above strike), causing Kelly to over-size positions.  0.97 is
# highly confident while preventing numerical over-betting.  Set to 1.0 to
# disable (not recommended — Kelly explodes when p→1 and cost is small).
NOAA_OBSERVED_MAX_P: float = float(os.environ.get("NOAA_OBSERVED_MAX_P", "0.97"))

# Crypto temporal edge: BTC/ETH/etc. price volatility scales with √t (Brownian
# motion).  CRYPTO_REFERENCE_HOURS is the window the METRIC_EDGE_SCALES values
# were calibrated for (full day).  When hours_to_close < 24, sigma shrinks
# proportionally so late-day positions with large edges score near-certainty
# rather than ~55%.  E.g. BTC $500 above strike with 1h to close:
#   σ_ref = $2,500  →  σ(1h) = $2,500 × √(1/24) = $510  →  z=0.98, p≈0.84
# Set to 0 to disable time-scaling (reverts to old flat-sigma behaviour).
CRYPTO_REFERENCE_HOURS: float = float(os.environ.get("CRYPTO_REFERENCE_HOURS", "24.0"))
# Minimum hours_to_close to use for the sqrt scaling — prevents sigma collapsing
# to zero in the final seconds before market close.
CRYPTO_MIN_HOURS: float = float(os.environ.get("CRYPTO_MIN_HOURS", "0.083"))  # 5 min

# Hours before market close at which forecast-only trades are blocked.
# Late in the day the forecast adds little — real outcomes are already largely
# determined by actual conditions.  Observed-data trades (noaa_observed) are
# exempt because station readings ARE the real conditions.
# Set to 0 to disable.
SAME_DAY_CUTOFF_HOURS: float = float(os.environ.get("SAME_DAY_CUTOFF_HOURS", "4.0"))

# Number of consecutive settled losses in one market category before the circuit
# breaker trips and blocks further trades in that category.  0 = disabled.
# Raised from 3 → 5: at 1–3 contract position sizes, 3 consecutive losses is
# normal variance and triggered too many false pauses.  5 requires a sustained
# losing streak before blocking, while still protecting against real bad runs.
CIRCUIT_BREAKER_CONSECUTIVE_LOSSES: int = int(
    os.environ.get("CIRCUIT_BREAKER_CONSECUTIVE_LOSSES", "5")
)

# How many hours to pause a category after the circuit breaker trips.
CIRCUIT_BREAKER_PAUSE_HOURS: float = float(
    os.environ.get("CIRCUIT_BREAKER_PAUSE_HOURS", "12.0")
)

# Max number of open/unresolved trades allowed in a category before further
# trades are blocked — regardless of settled win/loss history.  This prevents
# accumulating unlimited exposure in categories (e.g. KXMVECROSSCATEGORY)
# whose markets settle slowly and where the settled-loss breaker can't fire.
# Default 5.  Set to 0 to disable.
CIRCUIT_BREAKER_MAX_OPEN: int = int(
    os.environ.get("CIRCUIT_BREAKER_MAX_OPEN", "5")
)

_priors_raw = os.environ.get("KELLY_METRIC_PRIORS", "{}")
try:
    KELLY_METRIC_PRIORS: dict[str, float] = json.loads(_priors_raw)
except json.JSONDecodeError:
    logging.warning("KELLY_METRIC_PRIORS is not valid JSON — using defaults.")
    KELLY_METRIC_PRIORS = {}

# Auto-calibrated source-level priors (source → blended P(win)).
# Populated at runtime by calling refresh_calibrated_priors() on TradeExecutor.
# These take priority over KELLY_DEFAULT_P but yield to explicit KELLY_METRIC_PRIORS.
_calibrated_source_priors: dict[str, float] = {}

# Minutes a resting limit order is allowed to sit unfilled before it is
# cancelled automatically.  Only applies in live mode — passive midpoint orders
# may take a few cycles to fill; aggressive orders should fill within seconds.
# Set to 0 to disable automatic cancellation.
FILL_TIMEOUT_MINUTES: int = int(os.environ.get("FILL_TIMEOUT_MINUTES", "5"))

# Minimum number of contracts that must be available at the relevant side of
# the orderbook before entering a trade.  Prevents the bot from placing a
# 10-contract Kelly-sized order against a 1-contract-thin best ask.
# When the available depth is between MIN_DEPTH_CONTRACTS and the Kelly count,
# the trade is downsized to the available depth rather than skipped entirely.
# Set to 0 to disable.  Default: 3.
MIN_DEPTH_CONTRACTS: int = int(os.environ.get("MIN_DEPTH_CONTRACTS", "3"))

# Minimum bid on our side of the book before entering a trade.
# For a YES buy, this is the yes_bid.  For a NO buy, this is 100 − yes_ask
# (the NO bid).  When the bid is 0–1¢, the market prices our side as
# essentially worthless — the contract is dead and will be stopped out
# immediately.  Historical data: 24/51 stop-losses (47%) fired within
# seconds of entry on contracts where the bid was already 0¢.  Setting
# this to 2 blocks "dead on arrival" entries at the source.
# Set to 0 to disable.  Default: 2.
TRADE_MIN_BID_CENTS: int = int(os.environ.get("TRADE_MIN_BID_CENTS", "2"))

# ---------------------------------------------------------------------------
# Spread (synthetic range) trading
# ---------------------------------------------------------------------------

# Set to "true" to actually execute spread trades.  Default: "false" (detect
# and log only).  Read the pnl_attribution.txt output first to confirm that
# the spread detector is identifying genuine mispricings before enabling.
SPREAD_EXECUTION_ENABLED: bool = (
    os.environ.get("SPREAD_EXECUTION_ENABLED", "false").lower() == "true"
)

# Minimum P(win) required before a spread is executed.  Must exceed break-even
# for the spread's net cost.  At total_cost=120¢, break-even p=120/200=0.60.
SPREAD_MIN_WIN_PROB: float = float(os.environ.get("SPREAD_MIN_WIN_PROB", "0.65"))

# Kelly fraction multiplier for spreads.  Spreads carry more complexity and
# execution risk than single-leg trades, so a more conservative fraction is
# used by default (half of the standard KELLY_FRACTION).
SPREAD_KELLY_FRACTION: float = float(
    os.environ.get("SPREAD_KELLY_FRACTION", str(float(os.environ.get("KELLY_FRACTION", "0.25")) / 2))
)

_ORDERS_PATH = "/trade-api/v2/orders"
_DEFAULT_DB_PATH = Path(__file__).parent.parent / "opportunity_log.db"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_CREATE_TRADES_SQL = """
CREATE TABLE IF NOT EXISTS trades (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at        TEXT    NOT NULL,
    mode             TEXT    NOT NULL CHECK(mode IN ('dry_run', 'live')),
    ticker           TEXT    NOT NULL,
    side             TEXT    NOT NULL CHECK(side IN ('yes', 'no')),
    count            INTEGER NOT NULL,
    limit_price      INTEGER NOT NULL,   -- yes_price in cents (0–100)
    opportunity_kind TEXT    NOT NULL,  -- 'text'|'numeric'|'arb'|'crossed_book'|'bracket_set'|'spread'|'band_arb'
    score            REAL    NOT NULL,
    kelly_fraction   REAL,               -- KELLY_FRACTION multiplier used
    p_estimate       REAL,               -- P(win) used in Kelly formula
    status           TEXT    NOT NULL,   -- 'pending' | 'filled' | 'rejected' | 'error'
    order_id         TEXT,               -- Kalshi order ID (live only; NULL in dry-run)
    error_msg        TEXT,               -- populated on status = 'error'; NULL otherwise
    source           TEXT,               -- data source that triggered the trade
    outcome          TEXT,               -- 'won' | 'lost' | 'void' (NULL = unresolved)
    market_p_entry   REAL,               -- (yes_bid + yes_ask) / 200 at entry; for calibration
    yes_bid_entry    INTEGER,            -- yes_bid cents at entry
    yes_ask_entry    INTEGER,            -- yes_ask cents at entry
    signal_p_yes     REAL,               -- raw _implied_p_yes() output, before Kelly floors/caps/overrides
                                         -- NULL when no CDF model applies (text, poly, UNKNOWN direction)
                                         -- Use this vs market_p_entry vs outcome for longshot bias calibration
    corroborating_sources TEXT           -- comma-separated list of other sources that agreed with the primary
                                         -- e.g. "weatherapi,open_meteo" when noaa_day2 was the anchor
                                         -- NULL for single-source signals (crypto, EIA, arb, etc.)
)
"""

_CREATE_IDX_TRADES_SQL = """
CREATE INDEX IF NOT EXISTS idx_trades_ticker_logged
    ON trades (ticker, logged_at)
"""

_CREATE_CIRCUIT_BREAKERS_SQL = """
CREATE TABLE IF NOT EXISTS circuit_breakers (
    category        TEXT    PRIMARY KEY,
    tripped_at      TEXT    NOT NULL,
    tripped_until   TEXT    NOT NULL,
    trigger_losses  INTEGER NOT NULL
)
"""

# Ordered longest-first so "KXMVECROSSCATEGORY" is matched before "KXMVE".
_CATEGORY_PREFIXES: tuple[str, ...] = (
    "KXMVECROSSCATEGORY",
    "KXHURRICANE", "KXWEATHER", "KXTORNADO",
    "KXNCAAMB", "KXTRUMPSAY",
    "KXHIGH", "KXNBA", "KXNHL", "KXMLB",
    "KXATP", "KXWBC", "KXLOL", "KXVALORANT",
    "KXTOPSONG", "KXTOP10BIL", "KXRT", "KXMAMDANIM",
)


def _ticker_category(ticker: str) -> str:
    """Return the category prefix of a ticker for circuit-breaker grouping.

    Examples:
        "KXHIGHCHI-26MAR05-T45"  → "KXHIGH"
        "KXMVECROSSCATEGORY-..."  → "KXMVECROSSCATEGORY"
    """
    for prefix in _CATEGORY_PREFIXES:
        if ticker.startswith(prefix):
            return prefix
    return ticker.split("-", 1)[0]


# ---------------------------------------------------------------------------
# Signal quality helpers
# ---------------------------------------------------------------------------

def _normal_cdf(z: float) -> float:
    """Standard normal CDF via math.erfc — no external dependencies."""
    return 0.5 * math.erfc(-z / math.sqrt(2))


def _implied_p_yes(opp: NumericOpportunity) -> float | None:
    """Estimate P(YES) from the live data value.

    Over / under markets (single strike):
      Temperature: calibrated normal model, σ = NWS MAE (≈ 4°F forecast, 0.5°F observed).
        z = edge / σ → P(YES direction) = CDF(z)
      All other metrics: linear edge-fraction model using METRIC_EDGE_SCALES.
        edge_fraction = min(1, edge / scale)
        P(YES direction) = 0.5 + 0.5 * edge_fraction

    Between markets (range [lo, hi]):
      The probability that the settlement value falls inside the range, modelled
      as a Normal distribution centred on the current data value with σ derived
      from the same scale used for over/under markets (σ = scale / 2, so that
      an edge equal to the reference scale corresponds to ≈1σ from the boundary).
        P(YES) = P(in range) = CDF((hi − v) / σ) − CDF((lo − v) / σ)
      The return value is P(YES) directly — no implied_outcome flip needed,
      since P(YES) = P(in_range) regardless of whether the value is currently
      inside or outside the range.

    Returns None for direction-only (no-strike / UNKNOWN) markets.
    """
    if opp.implied_outcome == "UNKNOWN":
        return None

    # ------------------------------------------------------------------
    # Between markets — P(YES) = P(value settles within [lo, hi])
    # ------------------------------------------------------------------
    if opp.direction == "between":
        if opp.strike_lo is None or opp.strike_hi is None:
            return None
        if opp.metric.startswith("temp_high"):
            sigma = NOAA_OBSERVED_SIGMA if opp.source in ("noaa_observed", "metar") else _temp_forecast_sigma(opp.metric, opp.source)
        else:
            scale = next(
                (v for k, v in METRIC_EDGE_SCALES.items() if opp.metric.startswith(k)),
                1.0,
            )
            sigma = scale / 2.0
        p_in_range = (
            _normal_cdf((opp.strike_hi - opp.data_value) / sigma)
            - _normal_cdf((opp.strike_lo - opp.data_value) / sigma)
        )
        return max(0.0, min(1.0, p_in_range))

    # ------------------------------------------------------------------
    # Direction-only markets (up/down) — momentum-based CDF estimate
    # ------------------------------------------------------------------
    # opp.edge = abs(pct_change) * price, set by numeric_matcher when
    # Binance inter-cycle momentum resolves implied_outcome to YES/NO.
    # Use the same scale-based CDF model as over/under price markets.
    if opp.direction in ("up", "down"):
        if opp.edge == 0.0:
            return None
        scale = next(
            (v for k, v in METRIC_EDGE_SCALES.items() if opp.metric.startswith(k)),
            1.0,
        )
        sigma = scale / 2.0
        # 15M markets close quickly — temporal scaling tightens sigma dramatically.
        if CRYPTO_REFERENCE_HOURS > 0 and opp.hours_to_close is not None:
            h = max(CRYPTO_MIN_HOURS, opp.hours_to_close)
            sigma *= math.sqrt(h / CRYPTO_REFERENCE_HOURS)
        p_in_direction = _normal_cdf(opp.edge / sigma) if sigma > 0 else 0.5
        result = p_in_direction if opp.implied_outcome == "YES" else 1.0 - p_in_direction
        return min(result, FORECAST_MAX_P)

    # ------------------------------------------------------------------
    # Over / under markets — single strike required
    # ------------------------------------------------------------------
    if opp.strike is None:
        return None

    if opp.metric.startswith("temp_high"):
        # Calibrated model: observed data uses tight sigma; forecasts use NWS MAE.
        sigma = NOAA_OBSERVED_SIGMA if opp.source == "noaa_observed" else _temp_forecast_sigma(opp.metric, opp.source)
        z = opp.edge / sigma
        p_in_direction = _normal_cdf(z)
    else:
        # Generic CDF model for crypto, forex, BLS, FRED, EIA, etc.
        # sigma = scale / 2 so that edge == scale gives p ≈ 0.977 (2σ).
        scale = next(
            (v for k, v in METRIC_EDGE_SCALES.items() if opp.metric.startswith(k)),
            1.0,
        )
        sigma = scale / 2.0
        # Crypto temporal edge: volatility scales with √t (Brownian motion).
        # Shrink sigma as the market approaches close so that a large price
        # edge late in the day produces near-certainty rather than ~55%.
        if (
            CRYPTO_REFERENCE_HOURS > 0
            and opp.metric.startswith("price_")
            and opp.hours_to_close is not None
        ):
            h = max(CRYPTO_MIN_HOURS, opp.hours_to_close)
            time_factor = math.sqrt(h / CRYPTO_REFERENCE_HOURS)
            sigma *= time_factor
            logging.debug(
                "Crypto temporal sigma: %s  h=%.2f  σ_ref=%.1f  σ(t)=%.1f  edge=%.1f  z=%.2f",
                opp.market_ticker, h, scale / 2.0, sigma, opp.edge,
                opp.edge / sigma if sigma > 0 else 0,
            )
        p_in_direction = _normal_cdf(opp.edge / sigma)

    result = p_in_direction if opp.implied_outcome == "YES" else 1.0 - p_in_direction
    # Cap forecast-source probability: the CDF can return p > 0.999 for large
    # edges, but NWS forecasts are not that reliable.  Observed data (with its
    # tighter sigma) and NWS alerts are exempt from the forecast cap, but
    # observed data gets its own (higher) ceiling to prevent Kelly explosion.
    if opp.source not in ("noaa_observed", "metar", "nws_alert") and FORECAST_MAX_P < 1.0:
        result = min(result, FORECAST_MAX_P)
    elif opp.source in ("noaa_observed", "metar") and NOAA_OBSERVED_MAX_P < 1.0:
        result = min(result, NOAA_OBSERVED_MAX_P)
    # Crypto-specific hard ceiling: normal distribution underestimates fat-tail
    # moves (flash crashes, pumps).  p=0.9995 on a BTC gap trade caused $1+ losses
    # when BTC moved $6k intraday.  Cap crypto independently of FORECAST_MAX_P.
    if opp.metric.startswith("price_") and CRYPTO_MAX_P < 1.0:
        result = min(result, CRYPTO_MAX_P)
    return result


# ---------------------------------------------------------------------------
# Kelly sizing (module-level, pure function — easy to unit-test)
# ---------------------------------------------------------------------------

def kelly_contracts(
    win_prob: float,
    cost_cents: int,
    max_cents: int,
    kelly_fraction: float,
    hard_cap: int,
) -> int:
    """Compute the Kelly-optimal contract count for a binary bet.

    Args:
        win_prob:      Estimated probability of winning (0 < p < 1).
        cost_cents:    Cost per contract in cents (the amount we pay).
        max_cents:     Maximum total dollars to allocate (in cents).
        kelly_fraction: Fractional Kelly multiplier (e.g. 0.25 = quarter-Kelly).
        hard_cap:      Absolute maximum contracts regardless of Kelly.

    Returns:
        Number of contracts to buy (0 if Kelly recommends no bet).

    Formula:
        raw_f  = (win_prob − cost/100) / (1 − cost/100)
        contracts = floor(kelly_fraction × raw_f × max_cents / cost_cents)
    """
    if cost_cents <= 0 or cost_cents >= 100:
        return 0

    raw_f = (win_prob - cost_cents / 100.0) / (1.0 - cost_cents / 100.0)
    if raw_f <= 0:
        return 0  # negative edge — don't bet

    contracts = math.floor(kelly_fraction * raw_f * max_cents / cost_cents)
    return min(contracts, hard_cap)


# ---------------------------------------------------------------------------
# Per-cycle filter statistics
# ---------------------------------------------------------------------------

@dataclass
class FilterStats:
    """Counts of opportunities seen and filtered at each gate per poll cycle.

    Reset after each cycle's trade loop via ``reset()``.  ``log_summary()``
    emits a single INFO line so operators can see the full funnel at a glance:

        Trade funnel: 12 seen → 3 traded
          [score:4  dir:1  book:0  edge:2  disagree:1  cutoff:0  kelly:1  cool:0]
    """
    seen:                  int = field(default=0)
    filtered_score:        int = field(default=0)
    filtered_no_direction: int = field(default=0)
    filtered_no_orderbook: int = field(default=0)
    filtered_temp_edge:    int = field(default=0)
    filtered_disagreement:       int = field(default=0)
    filtered_extreme_disagree:   int = field(default=0)
    filtered_same_day:           int = field(default=0)
    filtered_kelly:        int = field(default=0)
    filtered_depth:        int = field(default=0)
    filtered_dead_market:  int = field(default=0)
    filtered_ticker_cool:  int = field(default=0)
    filtered_circuit_break: int = field(default=0)
    filtered_exposure:     int = field(default=0)
    trades_attempted:      int = field(default=0)

    def reset(self) -> None:
        for f in self.__dataclass_fields__:
            setattr(self, f, 0)

    def log_summary(self) -> None:
        if self.seen == 0:
            return
        logging.info(
            "Trade funnel: %d seen → %d traded"
            "  [score:%d  dir:%d  book:%d  edge:%d"
            "  disagree:%d  extreme:%d  cutoff:%d  kelly:%d"
            "  depth:%d  dead:%d  cool:%d  circuit:%d  exposure:%d]",
            self.seen, self.trades_attempted,
            self.filtered_score, self.filtered_no_direction,
            self.filtered_no_orderbook, self.filtered_temp_edge,
            self.filtered_disagreement, self.filtered_extreme_disagree,
            self.filtered_same_day,
            self.filtered_kelly, self.filtered_depth,
            self.filtered_dead_market,
            self.filtered_ticker_cool, self.filtered_circuit_break,
            self.filtered_exposure,
        )


# ---------------------------------------------------------------------------
# TradeExecutor
# ---------------------------------------------------------------------------

class TradeExecutor:
    """Unified dry-run / live trade executor with Kelly position sizing.

    In dry-run mode (default), intended trades are persisted to SQLite so you
    can review what the bot *would* have done without risking capital.  In live
    mode the same path also calls the Kalshi orders API and records the result.

    The ``trades`` table is written to the same ``opportunity_log.db`` file as
    the opportunity log, so a single DB file holds the full audit trail
    including the Kelly fraction and p_estimate used for each trade.

    Thread-safety note: same as ``OpportunityLog`` — single event loop only.

    Usage::

        executor = TradeExecutor()
        await executor.maybe_trade_numeric(session, opp, detail, score)
        executor.close()
    """

    def __init__(self, db_path: Path | str = _DEFAULT_DB_PATH) -> None:
        self._dry_run = TRADE_DRY_RUN
        self._db_path = Path(db_path)
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            isolation_level=None,   # autocommit
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()
        self.stats = FilterStats()
        mode_label = "DRY-RUN" if self._dry_run else "LIVE"
        logging.info(
            "TradeExecutor — mode=%s  kelly=%.2f  default_p=%.2f"
            "  max_pos=$%.2f  hard_cap=%d  min_score=%.2f",
            mode_label, KELLY_FRACTION, KELLY_DEFAULT_P,
            MAX_POSITION_CENTS / 100, TRADE_MAX_CONTRACTS, TRADE_MIN_SCORE,
        )

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(_CREATE_TRADES_SQL)
            self._conn.execute(_CREATE_IDX_TRADES_SQL)
            self._conn.execute(_CREATE_CIRCUIT_BREAKERS_SQL)
            self._migrate_schema()

    def _migrate_schema(self) -> None:
        """Add new columns to an existing trades table if they are missing."""
        # Widen the opportunity_kind CHECK constraint if the old restrictive
        # version is present.  The original constraint only allowed 'text' and
        # 'numeric', which causes every arb/spread/band_arb trade to raise
        # "CHECK constraint failed" and crash the poll cycle.  SQLite cannot
        # ALTER a CHECK constraint in place — a full table rebuild is required.
        tbl_row = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='trades'"
        ).fetchone()
        if tbl_row and "CHECK(opportunity_kind IN ('text', 'numeric'))" in (tbl_row[0] or ""):
            logging.info(
                "Schema migration: rebuilding trades table to remove "
                "restrictive opportunity_kind CHECK constraint."
            )
            with self._conn:
                self._conn.execute(
                    "ALTER TABLE trades RENAME TO _trades_old"
                )
                self._conn.execute(_CREATE_TRADES_SQL)
                # Copy all rows; spread_id / market_p_entry / yes_bid_entry /
                # yes_ask_entry may not exist in the old table — use COALESCE
                # via the pragma column list to copy only what exists.
                old_cols = {
                    row[1]
                    for row in self._conn.execute(
                        "PRAGMA table_info(_trades_old)"
                    ).fetchall()
                }
                shared = ", ".join(
                    c for c in [
                        "id", "logged_at", "mode", "ticker", "side", "count",
                        "limit_price", "opportunity_kind", "score",
                        "kelly_fraction", "p_estimate", "status", "order_id",
                        "error_msg", "source", "outcome", "market_p_entry",
                        "yes_bid_entry", "yes_ask_entry",
                    ]
                    if c in old_cols
                )
                self._conn.execute(
                    f"INSERT INTO trades ({shared}) SELECT {shared} FROM _trades_old"
                )
                self._conn.execute("DROP TABLE _trades_old")
            logging.info("Schema migration complete: trades table rebuilt.")

        existing = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(trades)").fetchall()
        }
        for col, defn in [
            ("kelly_fraction",   "REAL"),
            ("p_estimate",       "REAL"),
            ("source",           "TEXT"),
            ("outcome",          "TEXT"),
            ("fill_price_cents", "INTEGER"),  # actual fill price from Kalshi (live only)
            ("spread_id",        "TEXT"),     # UUID shared by both legs of a spread trade
            ("market_p_entry",   "REAL"),     # (yes_bid + yes_ask) / 200 at entry
            ("yes_bid_entry",    "INTEGER"),  # yes_bid cents at entry
            ("yes_ask_entry",    "INTEGER"),  # yes_ask cents at entry
            ("signal_p_yes",            "REAL"),  # raw _implied_p_yes() before Kelly adjustments
            ("corroborating_sources",   "TEXT"),  # comma-separated other sources that agreed
        ]:
            if col not in existing:
                self._conn.execute(f"ALTER TABLE trades ADD COLUMN {col} {defn}")

    # -----------------------------------------------------------------------
    # Public entry points
    # -----------------------------------------------------------------------

    async def maybe_trade_text(
        self,
        session: aiohttp.ClientSession,
        opp: Opportunity,
        detail: dict | None,
        score: float,
    ) -> None:
        """Evaluate and optionally execute a trade for a text/keyword opportunity.

        Uses the DistilBERT classifier to infer YES/NO direction from the
        (market_title, article_abstract) pair.  Falls back to display-only if
        the model is unavailable or the signal is below TEXT_MIN_CONFIDENCE.
        """
        self.stats.seen += 1

        # Score gate — text uses its own (lower) threshold
        _text_min = TEXT_TRADE_MIN_SCORE if TEXT_TRADE_MIN_SCORE > 0 else TRADE_MIN_SCORE
        if score < _text_min:
            self.stats.filtered_score += 1
            return

        # Metric filter — skip tickers whose metric is handled by the numeric
        # pipeline (price_, rate_, eia_, fred_, bls_, ism_).  RSS articles
        # about e.g. Bitcoin's inventor carry zero price-level information.
        _ticker_prefix = opp.market_ticker.split("-")[0]
        _metric = _TICKER_TO_METRIC.get(_ticker_prefix, "")
        if any(_metric.startswith(p) for p in TEXT_SKIP_METRIC_PREFIXES):
            logging.debug(
                "Trade skip (text/numeric-metric): ticker=%s metric=%s",
                opp.market_ticker, _metric,
            )
            self.stats.filtered_score += 1
            return

        # Classifier direction gate
        from .classifier import get_classifier
        direction, p_yes = get_classifier().predict(opp.market_title, opp.doc_body)
        if direction == "NEUTRAL" or abs(p_yes - 0.5) < TEXT_MIN_CONFIDENCE:
            self.stats.filtered_no_direction += 1
            logging.debug(
                "Trade skip (text/low confidence): ticker=%s  p_yes=%.3f  score=%.2f",
                opp.market_ticker, p_yes, score,
            )
            return

        side = "yes" if direction == "YES" else "no"

        # Orderbook gate
        if not detail:
            self.stats.filtered_no_direction += 1
            return
        bid = detail.get("yes_bid")
        ask = detail.get("yes_ask")
        if bid is None or ask is None:
            self.stats.filtered_no_direction += 1
            return

        # Kelly sizing (standard path — no locked-obs boost for text signals)
        entry_cost = ask if side == "yes" else (100 - bid)
        pos_max    = max(1, int(MAX_POSITION_CENTS * _dd_factor))
        count = kelly_contracts(
            win_prob=p_yes if side == "yes" else (1.0 - p_yes),
            cost_cents=entry_cost,
            max_cents=pos_max,
            kelly_fraction=KELLY_FRACTION,
            hard_cap=TRADE_MAX_CONTRACTS,
        )
        if SCORE_WEIGHTED_SIZING and TRADE_MIN_SCORE < 1.0 and count > 0:
            score_factor = 0.25 + 0.75 * (score - TRADE_MIN_SCORE) / (1.0 - TRADE_MIN_SCORE)
            count = max(1, math.floor(count * score_factor))
        if count < 1:
            self.stats.filtered_kelly += 1
            return

        limit_price = ask if side == "yes" else (100 - ask)

        logging.info(
            "Text opportunity: ticker=%s  side=%s  p_yes=%.3f  score=%.2f  count=%d  src=%s",
            opp.market_ticker, side.upper(), p_yes, score, count, opp.source,
        )

        await self._execute(
            session,
            ticker=opp.market_ticker,
            side=side,
            count=count,
            limit_price=limit_price,
            opportunity_kind="text",
            score=score,
            p_estimate=p_yes,
            source=opp.source,
            yes_bid=int(bid),
            yes_ask=int(ask),
        )

    async def maybe_trade_numeric(
        self,
        session: aiohttp.ClientSession,
        opp: NumericOpportunity,
        detail: dict | None,
        score: float,
    ) -> None:
        """Evaluate and optionally execute a trade for a numeric opportunity.

        Guards (all must pass before an order is attempted):
          1. score >= TRADE_MIN_SCORE
          2. implied_outcome in {"YES", "NO"} (UNKNOWN = no clear edge)
          3. Orderbook data present (need a price to set a limit)
          4. Kelly recommends >= 1 contract

        Args:
            session: Shared aiohttp session.
            opp:     The matched numeric opportunity.
            detail:  Live market detail dict (bid/ask/spread); may be None.
            score:   Composite score from the scoring module.
        """
        self.stats.seen += 1

        if score < TRADE_MIN_SCORE:
            logging.debug(
                "Trade skip (score %.2f < min %.2f): %s",
                score, TRADE_MIN_SCORE, opp.market_ticker,
            )
            self.stats.filtered_score += 1
            return

        if FOREX_MIN_SCORE > 0 and opp.metric.startswith("rate_") and score < FOREX_MIN_SCORE:
            logging.debug(
                "Trade skip (forex score %.2f < FOREX_MIN_SCORE %.2f): %s",
                score, FOREX_MIN_SCORE, opp.market_ticker,
            )
            self.stats.filtered_score += 1
            return

        if (
            NOAA_DAY2_MIN_SCORE > 0
            and opp.source in ("noaa_day2", "noaa_day3", "noaa_day4", "noaa_day5", "noaa_day6", "noaa_day7")
            and score < NOAA_DAY2_MIN_SCORE
        ):
            logging.debug(
                "Trade skip (day-ahead score %.2f < NOAA_DAY2_MIN_SCORE %.2f): %s src=%s",
                score, NOAA_DAY2_MIN_SCORE, opp.market_ticker, opp.source,
            )
            self.stats.filtered_score += 1
            return

        if NOAA_MIN_SCORE > 0 and opp.source == "noaa" and score < NOAA_MIN_SCORE:
            logging.debug(
                "Trade skip (noaa score %.2f < NOAA_MIN_SCORE %.2f): %s",
                score, NOAA_MIN_SCORE, opp.market_ticker,
            )
            self.stats.filtered_score += 1
            return

        if NOAA_OBSERVED_MIN_SCORE > 0 and opp.source == "noaa_observed" and score < NOAA_OBSERVED_MIN_SCORE:
            logging.debug(
                "Trade skip (noaa_observed score %.2f < NOAA_OBSERVED_MIN_SCORE %.2f): %s",
                score, NOAA_OBSERVED_MIN_SCORE, opp.market_ticker,
            )
            self.stats.filtered_score += 1
            return

        if BLOCKED_METRICS and opp.metric in BLOCKED_METRICS:
            logging.debug(
                "Trade skip (metric %s in BLOCKED_METRICS): %s",
                opp.metric, opp.market_ticker,
            )
            return

        if opp.implied_outcome not in ("YES", "NO"):
            logging.debug(
                "Trade skip (implied_outcome=%s, no clear edge): %s",
                opp.implied_outcome, opp.market_ticker,
            )
            self.stats.filtered_no_direction += 1
            return

        if not detail:
            logging.debug("Trade skip (no orderbook data): %s", opp.market_ticker)
            self.stats.filtered_no_orderbook += 1
            return

        # Block noaa_observed YES when market prices YES at < NOAA_OBSERVED_MIN_YES_ASK¢.
        # When YES ask < 5¢ the market is ≥95% certain of NO; our p_estimate=1.0 is
        # almost always wrong in this regime and produces consistent losses.
        if (
            NOAA_OBSERVED_MIN_YES_ASK > 0
            and opp.source == "noaa_observed"
            and opp.implied_outcome == "YES"
        ):
            _yes_ask = detail.get("yes_ask", 0)
            if _yes_ask < NOAA_OBSERVED_MIN_YES_ASK:
                logging.debug(
                    "Trade skip (noaa_observed YES ask %d¢ < min %d¢): %s",
                    _yes_ask, NOAA_OBSERVED_MIN_YES_ASK, opp.market_ticker,
                )
                self.stats.filtered_score += 1
                return

        # --- Minimum temperature edge filter ---
        if opp.metric.startswith("temp_high") and NUMERIC_MIN_TEMP_EDGE > 0:
            if opp.source != "noaa_observed" and opp.edge < NUMERIC_MIN_TEMP_EDGE:
                logging.debug(
                    "Trade skip (temp edge %.1f°F < min %.1f°F): %s",
                    opp.edge, NUMERIC_MIN_TEMP_EDGE, opp.market_ticker,
                )
                self.stats.filtered_temp_edge += 1
                return

        # Compute the model-implied P(YES) once — used for both the disagreement
        # check below and Kelly position sizing later.  For temperature markets
        # this is a calibrated normal-CDF model (σ ≈ 4°F NWS MAE); for other
        # metrics a scaled linear model.  Returns None for direction-only markets
        # (UNKNOWN outcome) where no probability can be computed.
        implied_p = _implied_p_yes(opp)

        # --- Market-NOAA probability disagreement filter ---
        # Locked observed YES trades (noaa_observed/nws_climo/nws_alert YES) are
        # exempt: the temperature has physically exceeded the strike, so the signal
        # is ground truth rather than a probabilistic forecast.  The disagreement
        # gate is designed to avoid chasing already-priced-in forecasts — not to
        # block certainties.
        _is_locked_obs_yes = (
            opp.source in _LOCKED_OBS_SOURCES and opp.implied_outcome == "YES"
        )
        if NUMERIC_MIN_DISAGREEMENT > 0 and implied_p is not None and not _is_locked_obs_yes:
            bid = detail.get("yes_bid")
            ask = detail.get("yes_ask")
            if bid is None or ask is None:
                # Can't compute market_p without a live orderbook — skip check.
                # (Using `or 50` would treat a valid 0¢ bid as 50¢, distorting market_p.)
                pass
            else:
                market_p = (float(bid) + float(ask)) / 200.0
                disagreement = abs(implied_p - market_p)
                if disagreement < NUMERIC_MIN_DISAGREEMENT:
                    logging.debug(
                        "Trade skip (model-market disagreement %.2f < %.2f, "
                        "model_p=%.2f mkt_p=%.2f): %s",
                        disagreement, NUMERIC_MIN_DISAGREEMENT,
                        implied_p, market_p, opp.market_ticker,
                    )
                    self.stats.filtered_disagreement += 1
                    return

        # --- Extreme model-market disagreement guard ---
        # When our model says p > 0.85 but the market prices our side at < 25¢,
        # the crowd is near-certain we are wrong.  This almost always indicates
        # a station mismatch, stale data, or a ticker cross rather than a genuine
        # information edge.  Block regardless of score or edge.
        if EXTREME_DISAGREE_MODEL_P > 0 and implied_p is not None and implied_p > EXTREME_DISAGREE_MODEL_P:
            bid = detail.get("yes_bid")
            ask = detail.get("yes_ask")
            if bid is not None and ask is not None:
                if opp.implied_outcome == "YES":
                    our_side_cents = float(ask)
                else:  # NO buy — our side is 100 − yes_bid
                    our_side_cents = 100.0 - float(bid)
                if our_side_cents < EXTREME_DISAGREE_MARKET_CENTS:
                    logging.info(
                        "Extreme disagree guard: blocked %s"
                        " model_p=%.2f our_side=%.0f¢ < %d¢ threshold"
                        " — likely station/ticker mismatch",
                        opp.market_ticker, implied_p,
                        our_side_cents, EXTREME_DISAGREE_MARKET_CENTS,
                    )
                    self.stats.filtered_extreme_disagree += 1
                    return

        # --- Extreme disagreement guard (NO side) ---
        # Symmetric counterpart to the YES guard above.  When the model is very
        # confident of NO (implied_p_yes < 1 − EXTREME_DISAGREE_MODEL_P) but the
        # market is pricing YES strongly (yes_bid > EXTREME_DISAGREE_NO_BID_MIN),
        # the crowd is near-certain we are wrong.  This catches stale day-2+
        # forecasts that haven't caught up to a market already pricing the event
        # as likely.  Block unconditionally regardless of score or edge.
        if (
            EXTREME_DISAGREE_MODEL_P > 0
            and implied_p is not None
            and opp.implied_outcome == "NO"
            and implied_p < (1.0 - EXTREME_DISAGREE_MODEL_P)
        ):
            bid = detail.get("yes_bid")
            if bid is not None and float(bid) > EXTREME_DISAGREE_NO_BID_MIN:
                logging.info(
                    "Extreme disagree guard (NO): blocked %s"
                    " model_p_yes=%.4f yes_bid=%.0f¢ > %d¢ threshold"
                    " — market strongly disagrees with model NO confidence",
                    opp.market_ticker, implied_p,
                    float(bid), EXTREME_DISAGREE_NO_BID_MIN,
                )
                self.stats.filtered_extreme_disagree += 1
                return

        # --- Same-day expiry cutoff (forecast only) ---
        # Temperature NO signals are exempt: by evening the daily high is already
        # established, so a NO forecast at 10 PM is effectively ground truth — the
        # observed temperature can't drop back below the strike overnight.
        # noaa_observed is always exempt (it IS the observation).
        _temp_no_exempt = (
            opp.metric.startswith("temp_high")
            and opp.implied_outcome == "NO"
        )
        if SAME_DAY_CUTOFF_HOURS > 0 and opp.source != "noaa_observed" and not _temp_no_exempt:
            close_time_str = detail.get("close_time") or detail.get("expiration_time")
            if close_time_str:
                try:
                    close_dt = datetime.fromisoformat(
                        close_time_str.replace("Z", "+00:00")
                    )
                    hours_remaining = (
                        close_dt - datetime.now(timezone.utc)
                    ).total_seconds() / 3600
                    if 0 < hours_remaining < SAME_DAY_CUTOFF_HOURS:
                        logging.debug(
                            "Trade skip (%.1fh to close < cutoff %.1fh, forecast only): %s",
                            hours_remaining, SAME_DAY_CUTOFF_HOURS, opp.market_ticker,
                        )
                        self.stats.filtered_same_day += 1
                        return
                except (ValueError, TypeError):
                    pass

        side, limit_price, entry_tier = self._compute_order_params(opp, detail, score)
        if limit_price is None:
            logging.debug(
                "Trade skip (ask/bid price unavailable): %s", opp.market_ticker
            )
            self.stats.filtered_no_orderbook += 1
            return

        if entry_tier == "patient":
            logging.info(
                "Patient limit entry: %s %s @ %d¢ (bid+1, score=%.2f ≥ threshold=%.2f)",
                opp.market_ticker, side.upper(), limit_price,
                score, PASSIVE_PATIENT_SCORE_THRESHOLD,
            )

        # --- Dead-market bid filter ---
        # If the bid on our side is already 0–1¢, the market prices our
        # position as essentially worthless.  Entering would result in an
        # immediate stop-loss the next cycle.  24/51 historical stop-losses
        # (47%) were caused by entering contracts in this state.
        if TRADE_MIN_BID_CENTS > 0:
            yes_bid = detail.get("yes_bid")
            yes_ask = detail.get("yes_ask")
            if yes_bid is not None and yes_ask is not None:
                our_bid = int(yes_bid) if side == "yes" else (100 - int(yes_ask))
                if our_bid < TRADE_MIN_BID_CENTS:
                    logging.debug(
                        "Trade skip (dead market: our bid %d¢ < min %d¢): %s",
                        our_bid, TRADE_MIN_BID_CENTS, opp.market_ticker,
                    )
                    self.stats.filtered_dead_market += 1
                    return

        # P4: Use CDF-computed probability for Kelly sizing when the model
        # produced one.  This replaces the static KELLY_DEFAULT_P=0.60 with a
        # signal-specific probability derived from actual forecast data.
        # Fall back to per-metric priors only for direction-only markets
        # (UNKNOWN outcome) where implied_p is None.
        p = implied_p if implied_p is not None else self._get_prior_p(opp.metric, opp.source)
        # Observed station data is near-certain — apply the confidence floor to
        # the win-side probability, not unconditionally to P(YES).
        # For YES trades: floor P(YES) from below at NOAA_OBSERVED_P.
        # For NO  trades: cap  P(YES) from above at (1 − NOAA_OBSERVED_P),
        #   which is equivalent to flooring P(NO wins) at NOAA_OBSERVED_P.
        # Applying max(p, 0.80) unconditionally for NO trades was a bug:
        # it made win_prob = 1-0.80 = 0.20, collapsing Kelly to 0 for clear
        # NO signals where the observed temperature is well below the strike.
        if opp.source == "noaa_observed":
            if opp.implied_outcome == "YES":
                p = max(p, NOAA_OBSERVED_P)
            else:
                p = min(p, 1.0 - NOAA_OBSERVED_P)
        # FedWatch: use CME-implied meeting probability as p_estimate for Fed funds markets.
        # Only applied when the Kalshi market resolves close to the next meeting —
        # markets spanning multiple FOMC decisions need a multi-meeting model that
        # CME single-meeting probabilities cannot provide.
        if opp.metric == "fred_fedfunds":
            fomc = cme_fedwatch.get_next_meeting()
            if fomc is not None:
                # Guard: skip override if the market closes more than 14 days after
                # the next meeting date (implies a multi-meeting horizon).
                close_str = detail.get("close_time") or detail.get("expiration_time", "")
                _applies = True
                try:
                    close_dt   = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
                    meeting_dt = datetime.fromisoformat(fomc.date + "T23:59:59+00:00")
                    if close_dt > meeting_dt + timedelta(days=14):
                        _applies = False
                        logging.debug(
                            "FedWatch skip (market closes %s, past next meeting %s): %s",
                            close_dt.date(), fomc.date, opp.market_ticker,
                        )
                except (ValueError, AttributeError):
                    pass  # unparseable close_time — apply override conservatively

                if _applies:
                    if opp.direction == "under":
                        # YES = rate ends up below strike → a cut occurred
                        fedwatch_p = fomc.cut_prob if opp.implied_outcome == "YES" else (fomc.hold_prob + fomc.hike_prob)
                    elif opp.direction == "over":
                        # YES = rate ends up above strike → no cut / a hike
                        fedwatch_p = fomc.hike_prob if opp.implied_outcome == "YES" else (fomc.hold_prob + fomc.cut_prob)
                    else:
                        fedwatch_p = None
                    if fedwatch_p is not None:
                        p = fedwatch_p
                        logging.debug(
                            "FedWatch p override: %.3f (cut=%.2f hold=%.2f hike=%.2f) for %s",
                            p, fomc.cut_prob, fomc.hold_prob, fomc.hike_prob,
                            opp.market_ticker,
                        )
        # cost_cents is what we actually pay per contract
        cost_cents = limit_price if side == "yes" else (100 - limit_price)
        win_prob = p if side == "yes" else (1.0 - p)

        # High-conviction signals get a moderately larger Kelly fraction.
        effective_kelly = (
            KELLY_FRACTION_HIGH
            if KELLY_FRACTION_HIGH > KELLY_FRACTION and score >= KELLY_HIGH_SCORE_THRESHOLD
            else KELLY_FRACTION
        )

        # Locked-observation YES trades override sizing: outcome is structurally
        # near-certain (observed temp already past strike / NWS alert confirmed).
        _is_locked_obs = opp.source in _LOCKED_OBS_SOURCES and opp.implied_outcome == "YES"
        pos_max_cents  = LOCKED_OBS_MAX_POSITION_CENTS if _is_locked_obs else MAX_POSITION_CENTS
        pos_kelly      = LOCKED_OBS_KELLY_FRACTION     if _is_locked_obs else effective_kelly
        pos_hard_cap   = LOCKED_OBS_MAX_CONTRACTS      if _is_locked_obs else TRADE_MAX_CONTRACTS
        pos_max_cents  = max(1, int(pos_max_cents * _dd_factor))

        count = kelly_contracts(
            win_prob=win_prob,
            cost_cents=cost_cents,
            max_cents=pos_max_cents,
            kelly_fraction=pos_kelly,
            hard_cap=pos_hard_cap,
        )
        if SCORE_WEIGHTED_SIZING and TRADE_MIN_SCORE < 1.0 and count > 0:
            # Scale from 0.25 (at TRADE_MIN_SCORE) up to 1.0 (at score=1.0).
            # floor(kelly * 0.25) rounds to 0 for kelly ≤ 3, which would silently
            # block trades that Kelly already approved.  max(1, …) preserves the
            # Kelly decision ("bet something") while still reducing position size
            # for lower-conviction signals.
            # NOTE: only apply when Kelly already approved (count > 0).  If Kelly
            # returned 0 (no edge — e.g. cost_cents >= 100), max(1, ...) must not
            # override that veto — it would allow 100¢-cost contracts with zero EV.
            score_factor = 0.25 + 0.75 * (score - TRADE_MIN_SCORE) / (1.0 - TRADE_MIN_SCORE)
            count = max(1, math.floor(count * score_factor))
            logging.debug(
                "Score-weighted sizing: score=%.2f factor=%.2f → %d contract(s): %s",
                score, score_factor, count, opp.market_ticker,
            )
        if count == 0:
            logging.debug(
                "Trade skip (Kelly=0, no edge at p=%.2f cost=%d¢): %s",
                p, cost_cents, opp.market_ticker,
            )
            self.stats.filtered_kelly += 1
            return

        # --- Orderbook depth check ------------------------------------------
        # Cap count at the contracts available at the relevant side of the book.
        # Applies when MIN_DEPTH_CONTRACTS > 0 and the field is present in the
        # detail dict (Kalshi returns yes_ask_size / yes_bid_size on the best
        # level).  If the field is absent the check is skipped (backwards compat).
        if MIN_DEPTH_CONTRACTS > 0:
            depth_field = "yes_ask_size" if side == "yes" else "yes_bid_size"
            raw_depth = detail.get(depth_field)
            if raw_depth is not None:
                depth = int(raw_depth)
                if depth < MIN_DEPTH_CONTRACTS:
                    logging.debug(
                        "Trade skip (depth %d < min %d at %s): %s",
                        depth, MIN_DEPTH_CONTRACTS, depth_field, opp.market_ticker,
                    )
                    self.stats.filtered_depth += 1
                    return
                if depth < count:
                    logging.debug(
                        "Depth cap: %s %d → %d contracts (available depth=%d)",
                        opp.market_ticker, count, depth, depth,
                    )
                    count = depth

        if TRADE_TICKER_COOLDOWN_MINUTES > 0:
            _cross_strike = opp.source not in COOLDOWN_CROSS_STRIKE_EXEMPT_SOURCES
            last_time, last_side, last_exited = self._last_trade_context(
                opp.market_ticker, cross_strike=_cross_strike
            )
            if last_time is not None:
                # Same-side open trade on a different strike of the same underlying =
                # a complementary position, not a direction flip.  Bypass the time-based
                # cooldown; enforce a per-underlying cap instead.
                _same_side_open_complement = (
                    _cross_strike and not last_exited and last_side == side
                )
                if _same_side_open_complement:
                    _underlying_prefix = opp.market_ticker.rsplit("-", 1)[0] + "-"
                    _open_count = self._count_open_on_underlying(_underlying_prefix)
                    if _open_count >= MAX_SAME_UNDERLYING_OPEN:
                        logging.info(
                            "Trade skip (underlying cap %d/%d open): %s",
                            _open_count, MAX_SAME_UNDERLYING_OPEN, opp.market_ticker,
                        )
                        self.stats.filtered_ticker_cool += 1
                        return
                    # else: fall through — allow the complementary same-direction trade
                else:
                    age_minutes = (datetime.now(timezone.utc) - last_time).total_seconds() / 60
                    # Three-tier cooldown: open position → full; exited same side → short;
                    # exited opposite side (direction flip) → medium.
                    if not last_exited:
                        effective_cooldown = TRADE_TICKER_COOLDOWN_MINUTES
                    elif last_side == side:
                        effective_cooldown = TRADE_TICKER_COOLDOWN_EXITED_MINUTES
                    else:
                        effective_cooldown = TRADE_TICKER_COOLDOWN_FLIP_MINUTES
                    if age_minutes < effective_cooldown:
                        logging.info(
                            "Trade skip (ticker cooldown %.0f min remaining,"
                            " last=%s→new=%s exited=%s): %s",
                            effective_cooldown - age_minutes,
                            last_side or "?", side, last_exited,
                            opp.market_ticker,
                        )
                        self.stats.filtered_ticker_cool += 1
                        return

        if TEMP_HIGH_NO_CUTOFF_UTC > 0 and side == "no" and "KXHIGH" in opp.market_ticker:
            # Block NO entries on daily temp-high markets after the cutoff hour.
            # By 3 PM ET (19:00 UTC) the daily maximum is essentially determined;
            # forecast sources (nws_hourly, HRRR) may still project a low reading
            # that no longer reflects reality, as seen in trade #200.
            if datetime.now(timezone.utc).hour >= TEMP_HIGH_NO_CUTOFF_UTC:
                logging.info(
                    "Trade skip (late-day cutoff: KXHIGH NO after %02d:00 UTC): %s",
                    TEMP_HIGH_NO_CUTOFF_UTC, opp.market_ticker,
                )
                self.stats.filtered_score += 1
                return

        if ADVERSE_PEAK_THRESHOLD > 0 and opp.source in ADVERSE_PEAK_SOURCES:
            # Check if the opposing side was recently priced very high in the
            # opportunity log — a spike to 75¢+ means the market had strong
            # conviction against our signal even if prices have since drifted back.
            since = (
                datetime.now(timezone.utc)
                - timedelta(hours=ADVERSE_PEAK_HOURS)
            ).isoformat()
            if side == "no":
                peak_row = self._conn.execute(
                    "SELECT MAX(yes_bid) FROM opportunities"
                    " WHERE ticker=? AND logged_at>=?",
                    (opp.market_ticker, since),
                ).fetchone()
                peak_val = peak_row[0] if peak_row and peak_row[0] is not None else 0
                if peak_val >= ADVERSE_PEAK_THRESHOLD:
                    logging.info(
                        "Trade skip (adverse peak: YES peaked at %d¢ ≥ %d¢ in last %.0fh): %s",
                        peak_val, ADVERSE_PEAK_THRESHOLD, ADVERSE_PEAK_HOURS,
                        opp.market_ticker,
                    )
                    self.stats.filtered_score += 1
                    return
            else:  # side == "yes"
                peak_row = self._conn.execute(
                    "SELECT MIN(yes_bid) FROM opportunities"
                    " WHERE ticker=? AND logged_at>=?",
                    (opp.market_ticker, since),
                ).fetchone()
                trough_val = peak_row[0] if peak_row and peak_row[0] is not None else 100
                if trough_val <= (100 - ADVERSE_PEAK_THRESHOLD):
                    logging.info(
                        "Trade skip (adverse peak: YES troughed at %d¢ ≤ %d¢ in last %.0fh): %s",
                        trough_val, 100 - ADVERSE_PEAK_THRESHOLD, ADVERSE_PEAK_HOURS,
                        opp.market_ticker,
                    )
                    self.stats.filtered_score += 1
                    return

        if self._is_category_tripped(opp.market_ticker):
            logging.info(
                "Trade skip (circuit breaker active for category %s): %s",
                _ticker_category(opp.market_ticker), opp.market_ticker,
            )
            self.stats.filtered_circuit_break += 1
            return

        if MAX_TOTAL_EXPOSURE_CENTS > 0:
            current_exposure = self._total_open_exposure_cents()
            this_trade_cost  = count * cost_cents
            if current_exposure + this_trade_cost > MAX_TOTAL_EXPOSURE_CENTS:
                logging.info(
                    "Trade skip (aggregate exposure cap: open=%d¢ + this=%d¢ = %d¢ > %d¢ limit): %s",
                    current_exposure, this_trade_cost,
                    current_exposure + this_trade_cost,
                    MAX_TOTAL_EXPOSURE_CENTS, opp.market_ticker,
                )
                self.stats.filtered_exposure += 1
                return

        logging.debug(
            "All filters passed — executing: %s %s %d×%d¢  p=%.3f  score=%.2f  src=%s",
            opp.market_ticker, side.upper(), count, limit_price, p, score, opp.source,
        )
        self.stats.trades_attempted += 1
        await self._execute(
            session=session,
            ticker=opp.market_ticker,
            side=side,
            count=count,
            limit_price=limit_price,
            opportunity_kind="numeric",
            score=score,
            p_estimate=p,
            source=opp.source,
            yes_bid=int(detail["yes_bid"]) if detail and detail.get("yes_bid") is not None else None,
            yes_ask=int(detail["yes_ask"]) if detail and detail.get("yes_ask") is not None else None,
            kelly_fraction=pos_kelly,
            signal_p_yes=implied_p,  # raw CDF model output, before noaa floor / fedwatch override
            corroborating_sources=opp.corroborating_sources or None,
        )

    async def maybe_trade_poly_opportunity(
        self,
        session: aiohttp.ClientSession,
        opp: PolyOpportunity,
        detail: dict | None,
        score: float,
    ) -> None:
        """Evaluate and optionally execute a trade on a Polymarket-vs-Kalshi divergence.

        Uses Polymarket's implied probability directly as p_estimate — no
        separate prior lookup needed.  Direction is determined by which side
        Polymarket says Kalshi has underpriced.

        Guards:
          1. score >= TRADE_MIN_SCORE
          2. score >= POLY_MIN_SCORE (poly-specific floor, default 0.82)
          3. Orderbook data present
          4. Relevant bid/ask available
          5. Kelly recommends >= 1 contract
          6. Ticker cooldown not active
        """
        self.stats.seen += 1

        if score < TRADE_MIN_SCORE:
            logging.debug(
                "Poly skip (score %.2f < min %.2f): %s",
                score, TRADE_MIN_SCORE, opp.kalshi_ticker,
            )
            self.stats.filtered_score += 1
            return

        if POLY_MIN_SCORE > 0 and score < POLY_MIN_SCORE:
            logging.debug(
                "Poly skip (poly score %.2f < POLY_MIN_SCORE %.2f): %s",
                score, POLY_MIN_SCORE, opp.kalshi_ticker,
            )
            self.stats.filtered_score += 1
            return

        # Guard: reject tickers that are not genuine Kalshi markets.
        # Polymarket-sourced hash IDs (e.g. "91F3C908-E60961190D2") are not
        # Kalshi markets and cannot be traded or exited; allowing them through
        # creates phantom open positions in the trade log that are never closed.
        if not opp.kalshi_ticker.startswith("KX"):
            logging.info(
                "Poly skip (non-Kalshi ticker): %s", opp.kalshi_ticker
            )
            self.stats.filtered_score += 1
            return

        # Guard: per-underlying open position cap.
        # When multiple adjacent strikes on the same underlying (e.g. five
        # USD/JPY strikes) all score above threshold in one cycle, betting on
        # all of them simultaneously concentrates risk — one bad move stops them
        # all out.  Limit to POLY_MAX_OPEN_PER_UNDERLYING concurrent positions.
        if POLY_MAX_OPEN_PER_UNDERLYING > 0:
            open_count = self._open_positions_on_underlying(opp.kalshi_ticker)
            if open_count >= POLY_MAX_OPEN_PER_UNDERLYING:
                logging.info(
                    "Poly skip (underlying cap: %d open >= max %d): %s",
                    open_count, POLY_MAX_OPEN_PER_UNDERLYING, opp.kalshi_ticker,
                )
                self.stats.filtered_ticker_cool += 1
                return

        if not detail:
            logging.debug("Poly skip (no orderbook data): %s", opp.kalshi_ticker)
            self.stats.filtered_no_orderbook += 1
            return

        side = opp.implied_side  # "yes" or "no"
        bid  = detail.get("yes_bid")
        ask  = detail.get("yes_ask")

        if bid is None or ask is None:
            logging.debug("Poly skip (ask/bid unavailable): %s", opp.kalshi_ticker)
            self.stats.filtered_no_orderbook += 1
            return

        # Poly signals are never time-critical — use passive midpoint to save
        # roughly half the spread.  Polymarket prices update slowly relative to
        # the Kalshi orderbook, so a passive limit is unlikely to miss a fill.
        limit_price = (int(bid) + int(ask) + 1) // 2
        p           = opp.poly_p_yes if side == "yes" else 1.0 - opp.poly_p_yes

        # --- Dead-market bid filter ---
        if TRADE_MIN_BID_CENTS > 0:
            our_bid = int(bid) if side == "yes" else (100 - int(ask))
            if our_bid < TRADE_MIN_BID_CENTS:
                logging.debug(
                    "Poly skip (dead market: our bid %d¢ < min %d¢): %s",
                    our_bid, TRADE_MIN_BID_CENTS, opp.kalshi_ticker,
                )
                self.stats.filtered_dead_market += 1
                return

        cost_cents = limit_price if side == "yes" else (100 - limit_price)
        win_prob   = p

        count = kelly_contracts(
            win_prob=win_prob,
            cost_cents=cost_cents,
            max_cents=max(1, int(MAX_POSITION_CENTS * _dd_factor)),
            kelly_fraction=KELLY_FRACTION,  # poly trades use standard fraction (no score tiering)
            hard_cap=TRADE_MAX_CONTRACTS,
        )
        if count == 0:
            logging.debug(
                "Poly skip (Kelly=0, no edge at p=%.2f cost=%d¢): %s",
                p, cost_cents, opp.kalshi_ticker,
            )
            self.stats.filtered_kelly += 1
            return

        if TRADE_TICKER_COOLDOWN_MINUTES > 0:
            last = self._last_trade_time(opp.kalshi_ticker)
            if last is not None:
                age_minutes = (datetime.now(timezone.utc) - last).total_seconds() / 60
                if age_minutes < TRADE_TICKER_COOLDOWN_MINUTES:
                    logging.info(
                        "Poly skip (ticker cooldown %.0f min remaining): %s",
                        TRADE_TICKER_COOLDOWN_MINUTES - age_minutes,
                        opp.kalshi_ticker,
                    )
                    self.stats.filtered_ticker_cool += 1
                    return

        if self._is_category_tripped(opp.kalshi_ticker):
            logging.info(
                "Poly skip (circuit breaker active for category %s): %s",
                _ticker_category(opp.kalshi_ticker), opp.kalshi_ticker,
            )
            self.stats.filtered_circuit_break += 1
            return

        self.stats.trades_attempted += 1
        await self._execute(
            session=session,
            ticker=opp.kalshi_ticker,
            side=side,
            count=count,
            limit_price=limit_price,
            opportunity_kind="numeric",
            score=score,
            p_estimate=p,
            source=opp.source,
        )

    async def maybe_trade_spread(
        self,
        session: aiohttp.ClientSession,
        spread: "SpreadOpportunity",
        score: float,
    ) -> None:
        """Evaluate and optionally execute a two-leg synthetic range position.

        Guards (all must pass):
          1. score >= TRADE_MIN_SCORE
          2. Both legs have valid pricing (total_cost_cents not None and > 0)
          3. p_win computed from _implied_p_yes > SPREAD_MIN_WIN_PROB
          4. Kelly sizing yields count >= 1
          5. Ticker cooldown clear for both legs
          6. Circuit breaker clear for both legs

        If SPREAD_EXECUTION_ENABLED is False the method logs the opportunity
        and returns without placing orders (read-only detection mode).
        """
        if score < TRADE_MIN_SCORE:
            logging.debug(
                "Spread skip (score %.2f < min %.2f): %s/%s",
                score, TRADE_MIN_SCORE,
                spread.leg_lo.market_ticker, spread.leg_hi.market_ticker,
            )
            return

        if spread.total_cost_cents is None or spread.total_cost_cents <= 0:
            logging.debug(
                "Spread skip (no pricing): %s/%s",
                spread.leg_lo.market_ticker, spread.leg_hi.market_ticker,
            )
            return

        if spread.max_profit_cents is None or spread.max_profit_cents <= 0:
            logging.debug(
                "Spread skip (non-positive max_profit=%s): %s/%s",
                spread.max_profit_cents,
                spread.leg_lo.market_ticker, spread.leg_hi.market_ticker,
            )
            return

        # P(win) = P(leg_lo YES direction) − P(leg_hi YES direction)
        p_lo = _implied_p_yes(spread.leg_lo)
        p_hi = _implied_p_yes(spread.leg_hi)
        if p_lo is None or p_hi is None:
            logging.debug(
                "Spread skip (cannot compute p_win): %s/%s",
                spread.leg_lo.market_ticker, spread.leg_hi.market_ticker,
            )
            return

        # For "over" spreads: P(in range) = P(above lo) − P(above hi)
        # For "under" spreads: P(in range) = P(below hi) − P(below lo) — same formula
        #   because leg_lo.implied_outcome="NO" → p_lo = P(YES) but we buy NO,
        #   and leg_hi.implied_outcome="YES" → p_hi = P(YES) and we buy YES.
        #   The formula still evaluates correctly: p_lo(NO) - p_hi(YES) maps to
        #   (1-p_lo_yes) ... but the general form works when we note that the
        #   "over" pairing always has YES lower and NO upper, so:
        if spread.direction == "over":
            p_win = p_lo - p_hi          # P(above lo) − P(above hi)
        else:
            p_win = (1.0 - p_lo) - (1.0 - p_hi)  # P(below hi) − P(below lo)

        p_win = max(0.0, p_win)

        logging.info(
            "Spread detected: %s [%s] %s–%s %s  p_win=%.2f  cost=%d¢  max_profit=%d¢  score=%.2f",
            spread.metric, spread.direction,
            spread.strike_lo, spread.strike_hi, spread.unit,
            p_win,
            spread.total_cost_cents, spread.max_profit_cents, score,
        )

        if p_win <= SPREAD_MIN_WIN_PROB:
            logging.info(
                "Spread skip (p_win=%.2f <= min %.2f): %s/%s",
                p_win, SPREAD_MIN_WIN_PROB,
                spread.leg_lo.market_ticker, spread.leg_hi.market_ticker,
            )
            return

        # Kelly sizing for the spread as a unit
        # f* = (p·max_profit − (1-p)·total_cost) / max_profit
        max_p = spread.max_profit_cents
        total_c = spread.total_cost_cents
        raw_kelly = (p_win * max_p - (1.0 - p_win) * total_c) / max_p
        if raw_kelly <= 0:
            logging.debug(
                "Spread skip (Kelly<=0, no edge): %s/%s",
                spread.leg_lo.market_ticker, spread.leg_hi.market_ticker,
            )
            return

        count = int(MAX_POSITION_CENTS / total_c * raw_kelly * SPREAD_KELLY_FRACTION)
        count = max(0, min(count, TRADE_MAX_CONTRACTS))
        if count < 1:
            logging.debug(
                "Spread skip (Kelly count=0): %s/%s",
                spread.leg_lo.market_ticker, spread.leg_hi.market_ticker,
            )
            return

        # Ticker cooldown — check both legs
        for ticker in (spread.leg_lo.market_ticker, spread.leg_hi.market_ticker):
            if TRADE_TICKER_COOLDOWN_MINUTES > 0:
                last = self._last_trade_time(ticker)
                if last is not None:
                    age_minutes = (datetime.now(timezone.utc) - last).total_seconds() / 60
                    if age_minutes < TRADE_TICKER_COOLDOWN_MINUTES:
                        logging.info(
                            "Spread skip (ticker cooldown %.0f min remaining): %s",
                            TRADE_TICKER_COOLDOWN_MINUTES - age_minutes, ticker,
                        )
                        return

        # Circuit breaker — check both legs
        for ticker in (spread.leg_lo.market_ticker, spread.leg_hi.market_ticker):
            if self._is_category_tripped(ticker):
                logging.info(
                    "Spread skip (circuit breaker active for category %s): %s",
                    _ticker_category(ticker), ticker,
                )
                return

        if not SPREAD_EXECUTION_ENABLED:
            logging.info(
                "Spread DETECT-ONLY (SPREAD_EXECUTION_ENABLED=false): "
                "%s [%s] %s–%s %s  count=%d  p_win=%.2f  score=%.2f",
                spread.metric, spread.direction,
                spread.strike_lo, spread.strike_hi, spread.unit,
                count, p_win, score,
            )
            return

        spread_id = str(uuid.uuid4())

        # Determine order sides and limit prices for each leg
        if spread.direction == "over":
            # leg_lo: buy YES @ yes_ask (cost_lo_cents)
            lo_side = "yes"
            lo_limit = spread.cost_lo_cents
            # leg_hi: buy NO  @ no_ask  (cost_hi_cents); limit_price is YES price
            #   no_ask = 100 − yes_bid, so yes_limit = 100 − cost_hi
            hi_side = "no"
            hi_limit = 100 - spread.cost_hi_cents  # YES price for NO order
        else:
            # leg_lo: buy NO  @ no_ask  (cost_lo_cents); YES price = 100 − cost_lo
            lo_side = "no"
            lo_limit = 100 - spread.cost_lo_cents
            # leg_hi: buy YES @ yes_ask (cost_hi_cents)
            hi_side = "yes"
            hi_limit = spread.cost_hi_cents

        self.stats.trades_attempted += 1
        await self._execute(
            session=session,
            ticker=spread.leg_lo.market_ticker,
            side=lo_side,
            count=count,
            limit_price=lo_limit,
            opportunity_kind="spread",
            score=score,
            p_estimate=p_win,
            source=spread.source,
            spread_id=spread_id,
        )
        await self._execute(
            session=session,
            ticker=spread.leg_hi.market_ticker,
            side=hi_side,
            count=count,
            limit_price=hi_limit,
            opportunity_kind="spread",
            score=score,
            p_estimate=p_win,
            source=spread.source,
            spread_id=spread_id,
        )

    async def execute_arb(
        self,
        session: aiohttp.ClientSession,
        arb: "ArbOpportunity",
        count: int | None = None,
    ) -> None:
        """Execute both legs of a combinatorial arbitrage opportunity.

        Both legs are placed aggressively (cross the spread) since arb
        opportunities are fleeting and no signal quality gate applies —
        the profit is guaranteed by logical necessity, not by a model.

        Guards:
          1. Circuit breaker clear for both tickers
          2. Ticker cooldown clear for both tickers
          3. ARB_EXECUTION_ENABLED is True (else detect-only)

        Args:
            session:  aiohttp session.
            arb:      ArbOpportunity from find_arb_opportunities().
            count:    Number of contract pairs to trade (default 1).
                      Caller can raise this after verifying the arb is real.
        """
        logging.info(
            "ARB detected: %s [%s] k_lo=%s k_hi=%s  profit=%d¢/pair"
            "  depth=%s  lo=%s %s@%d¢  hi=%s %s@%d¢",
            arb.metric, arb.direction,
            arb.strike_lo, arb.strike_hi,
            arb.guaranteed_profit_cents,
            arb.available_depth if arb.available_depth is not None else "?",
            arb.ticker_lo, arb.side_lo.upper(), arb.cost_lo_cents,
            arb.ticker_hi, arb.side_hi.upper(), arb.cost_hi_cents,
        )

        # Ticker cooldown — check both legs
        for ticker in (arb.ticker_lo, arb.ticker_hi):
            if TRADE_TICKER_COOLDOWN_MINUTES > 0:
                last = self._last_trade_time(ticker)
                if last is not None:
                    age_min = (datetime.now(timezone.utc) - last).total_seconds() / 60
                    if age_min < TRADE_TICKER_COOLDOWN_MINUTES:
                        logging.info(
                            "ARB skip (ticker cooldown %.0f min remaining): %s",
                            TRADE_TICKER_COOLDOWN_MINUTES - age_min, ticker,
                        )
                        return

        # Circuit breaker — check both legs
        for ticker in (arb.ticker_lo, arb.ticker_hi):
            if self._is_category_tripped(ticker):
                logging.info(
                    "ARB skip (circuit breaker active for category %s): %s",
                    _ticker_category(ticker), ticker,
                )
                return

        # Size by available depth capped at ARB_MAX_CONTRACTS.
        # Depth is the minimum contracts available across both legs at the
        # limit prices; None means depth data was absent from the API response.
        if count is None:
            depth_cap = arb.available_depth if arb.available_depth is not None else ARB_MAX_CONTRACTS
            count = max(1, min(ARB_MAX_CONTRACTS, depth_cap))

        if not ARB_EXECUTION_ENABLED:
            logging.info(
                "ARB DETECT-ONLY (ARB_EXECUTION_ENABLED=false): "
                "%s [%s] k_lo=%s k_hi=%s  profit=%d¢/pair  count=%d",
                arb.metric, arb.direction,
                arb.strike_lo, arb.strike_hi,
                arb.guaranteed_profit_cents, count,
            )
            return

        arb_id = str(uuid.uuid4())
        self.stats.trades_attempted += 1

        await self._execute(
            session=session,
            ticker=arb.ticker_lo,
            side=arb.side_lo,
            count=count,
            limit_price=arb.limit_lo_cents,
            opportunity_kind="arb",
            score=1.0,        # guaranteed profit → perfect score
            p_estimate=1.0,
            source="arb_detector",
            spread_id=arb_id,
        )
        await self._execute(
            session=session,
            ticker=arb.ticker_hi,
            side=arb.side_hi,
            count=count,
            limit_price=arb.limit_hi_cents,
            opportunity_kind="arb",
            score=1.0,
            p_estimate=1.0,
            source="arb_detector",
            spread_id=arb_id,
        )

    async def execute_crossed_book(
        self,
        session: aiohttp.ClientSession,
        arb: "CrossedBookArb",
        count: int | None = None,
    ) -> None:
        """Execute both legs of a crossed-book arbitrage on a single market.

        A crossed book means YES_ask + NO_ask < 100¢, so buying both sides
        guarantees a profit at settlement regardless of outcome.

        Guards:
          1. Circuit breaker clear for this ticker
          2. Ticker cooldown clear
          3. CROSSED_BOOK_ARB_ENABLED is True (else detect-only)

        Args:
            session:  aiohttp session.
            arb:      CrossedBookArb from find_crossed_book_opportunities().
            count:    Number of contract pairs to trade (default 1).
        """
        logging.info(
            "CROSSED-BOOK ARB: %s  YES_ask=%d¢  NO_ask=%d¢  profit=%d¢/pair",
            arb.ticker, arb.yes_ask, arb.no_ask, arb.profit,
        )

        # Ticker cooldown
        if TRADE_TICKER_COOLDOWN_MINUTES > 0:
            last = self._last_trade_time(arb.ticker)
            if last is not None:
                age_min = (datetime.now(timezone.utc) - last).total_seconds() / 60
                if age_min < TRADE_TICKER_COOLDOWN_MINUTES:
                    logging.info(
                        "CROSSED-BOOK skip (ticker cooldown %.0f min remaining): %s",
                        TRADE_TICKER_COOLDOWN_MINUTES - age_min, arb.ticker,
                    )
                    return

        # Circuit breaker
        if self._is_category_tripped(arb.ticker):
            logging.info(
                "CROSSED-BOOK skip (circuit breaker active for category %s): %s",
                _ticker_category(arb.ticker), arb.ticker,
            )
            return

        if count is None:
            depth_cap = arb.available_depth if arb.available_depth is not None else ARB_MAX_CONTRACTS
            count = max(1, min(ARB_MAX_CONTRACTS, depth_cap))

        if not CROSSED_BOOK_ARB_ENABLED:
            logging.info(
                "CROSSED-BOOK DETECT-ONLY (CROSSED_BOOK_ARB_ENABLED=false): "
                "%s  YES_ask=%d¢  NO_ask=%d¢  profit=%d¢/pair  count=%d",
                arb.ticker, arb.yes_ask, arb.no_ask, arb.profit, count,
            )
            return

        spread_id = str(uuid.uuid4())
        self.stats.trades_attempted += 1

        await self._execute(
            session=session,
            ticker=arb.ticker,
            side="yes",
            count=count,
            limit_price=arb.yes_ask,
            opportunity_kind="crossed_book",
            score=1.0,
            p_estimate=1.0,
            source="crossed_book_arb",
            spread_id=spread_id,
        )
        # NO limit price is expressed as a YES-equivalent: yes_bid = 100 - no_ask
        await self._execute(
            session=session,
            ticker=arb.ticker,
            side="no",
            count=count,
            limit_price=100 - arb.no_ask,   # yes_bid
            opportunity_kind="crossed_book",
            score=1.0,
            p_estimate=1.0,
            source="crossed_book_arb",
            spread_id=spread_id,
        )

    async def execute_bracket_set_arb(
        self,
        session: aiohttp.ClientSession,
        arb: "BracketSetArb",
        count: int | None = None,
    ) -> None:
        """Execute all legs of a series bracket arbitrage.

        Buys `side` on every bracket in the set.  Because exactly one bracket
        resolves YES, the winning leg pays $1 while all others pay $0 — and the
        total cost is less than $1, guaranteeing profit.

        Guards:
          1. Circuit breaker clear for ALL bracket tickers
          2. BRACKET_ARB_ENABLED is True (else detect-only)

        Args:
            session:  aiohttp session.
            arb:      BracketSetArb from find_bracket_set_opportunities().
            count:    Contracts per bracket (default 1).
        """
        logging.info(
            "BRACKET ARB: %s  %d brackets  side=%s  sum_yes_ask=%d¢  "
            "sum_yes_bid=%d¢  profit=%d¢",
            arb.event_ticker, arb.n_brackets, arb.side.upper(),
            arb.sum_yes_ask, arb.sum_yes_bid, arb.profit,
        )

        # Circuit breaker — check all legs
        for ticker in arb.tickers:
            if self._is_category_tripped(ticker):
                logging.info(
                    "BRACKET ARB skip (circuit breaker active for %s): %s",
                    _ticker_category(ticker), arb.event_ticker,
                )
                return

        # Bracket arb places count contracts on EVERY bracket simultaneously.
        # Total contracts = count × n_brackets, so use a lower per-bracket cap.
        if count is None:
            count = max(1, ARB_MAX_CONTRACTS // 2)

        if not BRACKET_ARB_ENABLED:
            logging.info(
                "BRACKET ARB DETECT-ONLY (BRACKET_ARB_ENABLED=false): "
                "%s  %d brackets  side=%s  profit=%d¢  count=%d/bracket",
                arb.event_ticker, arb.n_brackets, arb.side.upper(), arb.profit, count,
            )
            return

        spread_id = str(uuid.uuid4())
        self.stats.trades_attempted += 1

        if arb.side == "yes":
            prices = arb.yes_ask_prices
            for ticker, limit_price in zip(arb.tickers, prices):
                await self._execute(
                    session=session,
                    ticker=ticker,
                    side="yes",
                    count=count,
                    limit_price=limit_price,
                    opportunity_kind="bracket_arb",
                    score=1.0,
                    p_estimate=1.0,
                    source="bracket_arb",
                    spread_id=spread_id,
                )
        else:
            # buy NO on each bracket; NO limit price = yes_bid (YES-equivalent)
            for ticker, yes_bid in zip(arb.tickers, arb.yes_bid_prices):
                await self._execute(
                    session=session,
                    ticker=ticker,
                    side="no",
                    count=count,
                    limit_price=yes_bid,
                    opportunity_kind="bracket_arb",
                    score=1.0,
                    p_estimate=1.0,
                    source="bracket_arb",
                    spread_id=spread_id,
                )

    async def maybe_trade_band_arb(
        self,
        session: aiohttp.ClientSession,
        signal: "BandArbSignal",
    ) -> None:
        """Evaluate and optionally execute a band-pass NO trade.

        Called when a KXHIGH "between" or "under" market has been definitively
        passed through by the METAR observed daily high.  The YES side of the
        market will settle to 0¢, so buying NO is near-certain profit.

        Uses LOCKED_OBS sizing (same as noaa_observed/metar YES trades) since
        the underlying evidence is observed station data, not a forecast model.

        Guards:
          1. Ticker cooldown clear (TRADE_TICKER_COOLDOWN_MINUTES)
          2. Circuit breaker clear for this ticker's category
          3. Kelly recommends ≥ 1 contract at the NO ask price
          4. BAND_ARB_EXECUTION_ENABLED is True (else detect-only)

        Args:
            session:  aiohttp session.
            signal:   BandArbSignal from find_band_arbs().
        """
        logging.info(
            "BandArb: %s  obs=%.1f°F > ceil=%.1f°F  NO_ask=%d¢  (%s)",
            signal.ticker, signal.observed_max, signal.band_ceil,
            signal.no_ask, signal.direction,
        )

        # Ticker cooldown
        if TRADE_TICKER_COOLDOWN_MINUTES > 0:
            last = self._last_trade_time(signal.ticker)
            if last is not None:
                age_min = (datetime.now(timezone.utc) - last).total_seconds() / 60
                if age_min < TRADE_TICKER_COOLDOWN_MINUTES:
                    logging.info(
                        "BandArb skip (ticker cooldown %.0f min remaining): %s",
                        TRADE_TICKER_COOLDOWN_MINUTES - age_min, signal.ticker,
                    )
                    return

        # Circuit breaker
        if self._is_category_tripped(signal.ticker):
            logging.info(
                "BandArb skip (circuit breaker active for category %s): %s",
                _ticker_category(signal.ticker), signal.ticker,
            )
            return

        # Kelly sizing: use LOCKED_OBS parameters — this is near-certain observed data
        # win_prob for NO = 1 − p_yes ≈ 0.97 (NOAA_OBSERVED_MAX_P)
        p_win = NOAA_OBSERVED_MAX_P  # P(NO wins) = P(band stays passed through)
        count = kelly_contracts(
            win_prob=p_win,
            cost_cents=signal.no_ask,
            max_cents=max(1, int(LOCKED_OBS_MAX_POSITION_CENTS * _dd_factor)),
            kelly_fraction=LOCKED_OBS_KELLY_FRACTION,
            hard_cap=LOCKED_OBS_MAX_CONTRACTS,
        )
        if count == 0:
            logging.info(
                "BandArb skip (Kelly=0 at p=%.2f no_ask=%d¢): %s",
                p_win, signal.no_ask, signal.ticker,
            )
            return

        if not BAND_ARB_EXECUTION_ENABLED:
            logging.info(
                "BandArb DETECT-ONLY (BAND_ARB_EXECUTION_ENABLED=false): "
                "%s  NO×%d @ %d¢  obs=%.1f°F > ceil=%.1f°F",
                signal.ticker, count, signal.no_ask,
                signal.observed_max, signal.band_ceil,
            )
            return

        self.stats.trades_attempted += 1
        await self._execute(
            session=session,
            ticker=signal.ticker,
            side="no",
            count=count,
            limit_price=signal.yes_bid,   # NO buy: limit_price = yes_bid
            opportunity_kind="arb",
            score=1.0,       # near-certain observed signal
            p_estimate=p_win,
            source="band_arb",
            kelly_fraction=LOCKED_OBS_KELLY_FRACTION,
        )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def refresh_calibrated_priors(self, tracker: object) -> None:
        """Pull auto-calibrated source priors from WinRateTracker.

        Should be called at startup and after each settlement pass so the
        Kelly sizer uses the latest observed win rates rather than the static
        ``KELLY_DEFAULT_P``.

        Args:
            tracker: A ``WinRateTracker`` instance with ``get_calibrated_priors()``.
        """
        global _calibrated_source_priors
        try:
            priors = tracker.get_calibrated_priors()  # type: ignore[attr-defined]
            _calibrated_source_priors = priors
            if priors:
                logging.info(
                    "Calibrated source priors refreshed: %s",
                    {k: f"{v:.4f}" for k, v in priors.items()},
                )
        except Exception as exc:
            logging.warning("Failed to refresh calibrated priors: %s", exc)

    @staticmethod
    def _get_prior_p(metric: str, source: str = "") -> float:
        """Look up P(YES) prior for a metric or source, falling back to the global default.

        Priority order:
          1. ``KELLY_METRIC_PRIORS`` — explicit per-metric prefix override (highest)
          2. ``_calibrated_source_priors`` — auto-calibrated from historical win rates
          3. ``KELLY_DEFAULT_P`` — global fallback

        Args:
            metric: Metric name (e.g. ``"temp_high_ny"``).
            source: Data source name (e.g. ``"noaa_observed"``).
        """
        for prefix, p in KELLY_METRIC_PRIORS.items():
            if metric.startswith(prefix):
                return float(p)
        if source and source in _calibrated_source_priors:
            return _calibrated_source_priors[source]
        return KELLY_DEFAULT_P

    @staticmethod
    def _compute_order_params(
        opp: NumericOpportunity, detail: dict, score: float = 0.0
    ) -> tuple[str, int | None, str]:
        """Return (side, yes_price_cents, tier) for a limit order.

        Three entry tiers (applied in priority order):

        1. **Aggressive** — urgent sources (``noaa_observed``, ``binance``):
           Cross the spread immediately.  Time-sensitive signals may expire
           within minutes; paying the full spread is worth the certainty of fill.
           YES buy → yes_ask;  NO buy → yes_bid.

        2. **Patient** — high-score non-urgent signals (score ≥
           ``PASSIVE_PATIENT_SCORE_THRESHOLD``), only when spread ≥ 3¢:
           Post one tick inside the bid (bid+1¢).  Captures 1–4¢ more of the
           spread by letting the market come to the bot.  If unfilled within
           FILL_TIMEOUT_MINUTES the existing timeout mechanism cancels it.
           Only activates when bid+1 < midpoint (i.e. spread ≥ 3¢).

        3. **Standard passive** — all other cases:
           Post at the ceiling midpoint ``(bid + ask + 1) // 2``.  Saves half
           the spread vs. aggressive at the cost of a slower fill.

        YES / NO price convention: yes_price_cents is always on the YES-scale;
        for NO buys the actual cost per contract is 100 − yes_price_cents.

        Returns ``(side, None, "aggressive")`` when the price field is absent.
        """
        _URGENT_SOURCES = ("noaa_observed", "binance")

        side = opp.implied_outcome.lower()  # "yes" or "no"
        bid  = detail.get("yes_bid")
        ask  = detail.get("yes_ask")

        if bid is None or ask is None:
            return side, None, "aggressive"

        bid_i = int(bid)
        ask_i = int(ask)
        mid_i = (bid_i + ask_i + 1) // 2

        if opp.source in _URGENT_SOURCES:
            # Tier 1 — Aggressive: cross the spread immediately.
            limit_price = ask_i if side == "yes" else bid_i
            tier = "aggressive"
        elif (
            score >= PASSIVE_PATIENT_SCORE_THRESHOLD
            and bid_i + 1 < mid_i          # only when patient < midpoint (spread ≥ 3¢)
            and bid_i + 1 < ask_i          # sanity: don't post at or above ask
        ):
            # Tier 2 — Patient: one tick inside the bid.
            limit_price = bid_i + 1
            tier = "patient"
        else:
            # Tier 3 — Standard passive: ceiling midpoint.
            limit_price = mid_i
            tier = "passive"

        return side, limit_price, tier

    def _is_category_tripped(self, ticker: str) -> bool:
        """Return True if this ticker's market category has an active circuit breaker.

        The breaker trips when the last CIRCUIT_BREAKER_CONSECUTIVE_LOSSES resolved
        trades for the category are all losses.  A trade counts as a loss if Kalshi
        settled it as outcome='lost' OR if the bot exited it via stop-loss before
        settlement.  Once tripped,
        the category is blocked for CIRCUIT_BREAKER_PAUSE_HOURS hours.  The trip
        state is written to the ``circuit_breakers`` table so it survives restarts.

        Design: self-updating — first checks the cached table (fast path), then
        re-evaluates from the trades table if no active record exists, and writes
        a new record if a trip condition is detected.
        """
        if CIRCUIT_BREAKER_CONSECUTIVE_LOSSES <= 0:
            return False

        category = _ticker_category(ticker)

        # --- Fast path: check existing cached record ---
        tripped_at_str: str | None = None
        row = self._conn.execute(
            "SELECT tripped_until, tripped_at FROM circuit_breakers WHERE category = ?",
            (category,),
        ).fetchone()
        if row is not None:
            tripped_at_str = row[1]
            try:
                until = datetime.fromisoformat(row[0])
                if until.tzinfo is None:
                    until = until.replace(tzinfo=timezone.utc)
                if datetime.now(timezone.utc) < until:
                    return True
                # Pause has expired — fall through to re-evaluate, but only
                # against trades logged *after* the trip time.  If no new trades
                # exist (because the category was blocked during the pause), the
                # breaker clears automatically rather than re-tripping forever.
            except (ValueError, TypeError):
                pass

        # --- Evaluate: check last N resolved outcomes for this category ---
        # A trade counts as a "loss" if either:
        #   (a) Kalshi settled it as outcome='lost', or
        #   (b) The bot exited early via stop-loss (exit_reason='stop_loss').
        # This catches rapid stop-loss streaks that don't yet have settled
        # outcomes — e.g. KXDOGE positions exited at 0¢ before settlement.
        #
        # When a previous trip record exists, only look at trades logged AFTER
        # that trip time.  This prevents a permanent deadlock where pre-trip
        # losses keep re-firing the breaker indefinitely even though the pause
        # window has already elapsed.
        after_clause = ""
        after_params: tuple = ()
        if tripped_at_str:
            after_clause = "AND logged_at > ?"
            after_params = (tripped_at_str,)

        # Use exit_pnl_cents as the source of truth when an early exit occurred.
        # outcome='lost' means Kalshi's contract settled against our direction —
        # but a profitable profit-take exit (exit_pnl_cents > 0) is a WIN for
        # the bot even if the underlying contract ultimately resolved NO/YES
        # against us.  Using outcome alone classified all early-exit wins as
        # losses and caused the circuit breaker to deadlock permanently.
        rows = self._conn.execute(
            f"""
            SELECT
                CASE
                    WHEN exit_pnl_cents IS NOT NULL AND exit_pnl_cents > 0 THEN 'won'
                    WHEN exit_pnl_cents IS NOT NULL AND exit_pnl_cents < 0 THEN 'lost'
                    WHEN exit_pnl_cents IS NULL AND outcome = 'won'        THEN 'won'
                    WHEN exit_pnl_cents IS NULL AND outcome = 'lost'       THEN 'lost'
                    ELSE 'other'
                END AS resolved_outcome,
                logged_at
            FROM trades
            WHERE ticker LIKE ?
              AND (exit_pnl_cents IS NOT NULL OR outcome IS NOT NULL)
              {after_clause}
            ORDER BY logged_at DESC
            LIMIT ?
            """,
            (category + "%",) + after_params + (CIRCUIT_BREAKER_CONSECUTIVE_LOSSES,),
        ).fetchall()

        if len(rows) < CIRCUIT_BREAKER_CONSECUTIVE_LOSSES:
            # Not enough resolved history — check open-trade cap instead.
            if CIRCUIT_BREAKER_MAX_OPEN > 0:
                open_count = self._conn.execute(
                    "SELECT COUNT(*) FROM trades WHERE ticker LIKE ? AND outcome IS NULL",
                    (category + "%",),
                ).fetchone()[0]
                if open_count >= CIRCUIT_BREAKER_MAX_OPEN:
                    logging.warning(
                        "Circuit breaker (open-cap) TRIPPED — category=%s  "
                        "%d open trades with no settlements.  Blocking further trades.",
                        category, open_count,
                    )
                    return True
            return False

        if not all(r[0] == "lost" for r in rows):
            return False

        # All N consecutive losses — write trip record
        now = datetime.now(timezone.utc)
        tripped_until = now + timedelta(hours=CIRCUIT_BREAKER_PAUSE_HOURS)
        self._conn.execute(
            """
            INSERT OR REPLACE INTO circuit_breakers
                (category, tripped_at, tripped_until, trigger_losses)
            VALUES (?, ?, ?, ?)
            """,
            (category, now.isoformat(), tripped_until.isoformat(),
             CIRCUIT_BREAKER_CONSECUTIVE_LOSSES),
        )
        logging.warning(
            "Circuit breaker TRIPPED — category=%s  %d consecutive losses.  "
            "Paused until %s UTC.",
            category, CIRCUIT_BREAKER_CONSECUTIVE_LOSSES,
            tripped_until.strftime("%Y-%m-%d %H:%M"),
        )
        return True

    def _count_open_on_underlying(self, underlying_prefix: str) -> int:
        """Count open (not yet exited) trades on any strike of the given underlying prefix."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM trades WHERE ticker LIKE ? AND exited_at IS NULL",
            (underlying_prefix + "%",),
        ).fetchone()
        return row[0] if row else 0

    def _last_trade_context(
        self, ticker: str, cross_strike: bool = True
    ) -> "tuple[datetime | None, str | None, bool]":
        """Return (last_trade_time, side, is_exited) for the most recent trade.

        Also checks any other strike on the same underlying market (e.g. all
        KXBTCD-26MAR1421-* tickers share a cooldown so the bot cannot flip
        direction by trading a different strike 18 minutes later).

        Args:
            ticker:       The market ticker to check.
            cross_strike: When True (default), also checks adjacent strikes on
                          the same underlying prefix.  Pass False for real-time
                          model sources (e.g. HRRR) that are independent of
                          day-ahead forecasts and should not inherit cooldowns
                          from other-model trades on the same underlying.

        Returns:
            last_time:  UTC datetime of last trade, or None if no prior trade.
            side:       'yes' | 'no', or None if no prior trade.
            is_exited:  True if exited_at IS NOT NULL on the most recent trade.
        """
        parts = ticker.rsplit("-", 1)
        underlying_prefix = parts[0] + "-" if len(parts) > 1 else None

        if cross_strike and underlying_prefix:
            row = self._conn.execute(
                "SELECT logged_at, side, exited_at IS NOT NULL, exited_at"
                " FROM trades WHERE ticker = ? OR ticker LIKE ?"
                " ORDER BY logged_at DESC LIMIT 1",
                (ticker, underlying_prefix + "%"),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT logged_at, side, exited_at IS NOT NULL, exited_at"
                " FROM trades WHERE ticker = ? ORDER BY logged_at DESC LIMIT 1",
                (ticker,),
            ).fetchone()

        if row is None:
            return None, None, False

        ts_str, last_side, is_exited, exited_at_str = row

        # For exited trades, use exited_at as the cooldown reference — not
        # logged_at.  A trade held for 3+ hours and then stopped out has its
        # logged_at far in the past, which makes the same-side cooldown appear
        # already elapsed at the moment of exit, allowing immediate re-entry
        # on a signal that was just proven wrong.  Using exited_at restarts
        # the cooldown clock from the actual close of the position.
        ref_str = (exited_at_str if is_exited and exited_at_str else ts_str)
        try:
            dt = datetime.fromisoformat(ref_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            return None, None, False

        return dt, last_side, bool(is_exited)

    def _last_trade_time(self, ticker: str) -> "datetime | None":
        """Return the UTC datetime of the most recent trade for this ticker, or None.

        Thin wrapper around _last_trade_context for callers (spread, arb, poly)
        that only need the timestamp.
        """
        dt, _, _ = self._last_trade_context(ticker)
        return dt

    def _total_open_exposure_cents(self) -> int:
        """Sum of cost basis (count × cost_per_contract) across all open positions.

        For YES trades: cost = limit_price cents.
        For NO  trades: cost = 100 − limit_price cents (limit_price is the YES-
                        equivalent bid used to size the NO order).

        Excludes rejected/error trades; includes dry_run, resting, and filled
        trades that have not yet been exited (exited_at IS NULL).
        """
        row = self._conn.execute(
            """
            SELECT COALESCE(SUM(
                count * CASE WHEN side = 'yes' THEN limit_price
                             ELSE 100 - limit_price END
            ), 0)
            FROM trades
            WHERE exited_at IS NULL
              AND status NOT IN ('rejected', 'error')
            """
        ).fetchone()
        return int(row[0]) if row else 0

    def _open_positions_on_underlying(self, ticker: str) -> int:
        """Count open (not yet exited) positions on the same underlying prefix.

        Uses the ticker prefix up to the last hyphen-separated strike component.
        E.g. KXUSDJPY-26MAR1010-T157.000 → prefix KXUSDJPY-26MAR1010-.
        Returns 0 if the ticker has no hyphen (no prefix to match against).
        """
        parts = ticker.rsplit("-", 1)
        if len(parts) < 2:
            return 0
        prefix = parts[0] + "-"
        row = self._conn.execute(
            "SELECT COUNT(*) FROM trades"
            " WHERE (ticker = ? OR ticker LIKE ?)"
            " AND exited_at IS NULL",
            (ticker, prefix + "%"),
        ).fetchone()
        return row[0] if row else 0

    async def poll_open_orders(self, session: aiohttp.ClientSession) -> None:
        """Check and update all resting live orders from previous cycles.

        Called once per poll cycle (live mode only).  For each trade in
        ``status = 'resting'``:

          - Fetches current status from the Kalshi orders API.
          - Marks ``'filled'`` and records ``fill_price_cents`` on success.
          - Marks ``'rejected'`` when Kalshi reports cancelled / expired.
          - Cancels and marks ``'rejected'`` when the order has been resting
            longer than ``FILL_TIMEOUT_MINUTES`` (stale passive limit order).

        Dry-run trades are never sent to Kalshi so they are skipped entirely.
        """
        if self._dry_run:
            return

        rows = self._conn.execute(
            """
            SELECT id, order_id, ticker, side, count, limit_price, logged_at
            FROM trades
            WHERE status = 'resting' AND mode = 'live' AND order_id IS NOT NULL
            """
        ).fetchall()

        if not rows:
            return

        now = datetime.now(timezone.utc)
        logging.info("Fill confirmation: checking %d resting order(s).", len(rows))

        for row in rows:
            trade_id, order_id, ticker, side, count, limit_price, logged_at_str = row

            try:
                logged_at = datetime.fromisoformat(logged_at_str)
                if logged_at.tzinfo is None:
                    logged_at = logged_at.replace(tzinfo=timezone.utc)
                age_minutes = (now - logged_at).total_seconds() / 60
            except (ValueError, TypeError):
                age_minutes = 0.0

            order = await self._fetch_order_status(session, order_id)
            if order is None:
                # API error — status unknown; retry next cycle.
                continue

            kalshi_status = order.get("status", "")
            fill_price    = order.get("yes_price")

            if kalshi_status == "filled":
                self._conn.execute(
                    "UPDATE trades SET status = 'filled', fill_price_cents = ? WHERE id = ?",
                    (fill_price, trade_id),
                )
                logging.info(
                    "[FILL CONFIRMED] trade #%d  %s %s x%d"
                    "  limit=%d¢  fill=%s¢",
                    trade_id, side.upper(), ticker, count, limit_price, fill_price,
                )

            elif kalshi_status in ("cancelled", "expired", "rejected"):
                self._conn.execute(
                    "UPDATE trades SET status = 'rejected' WHERE id = ?",
                    (trade_id,),
                )
                logging.info(
                    "[ORDER VOID] trade #%d %s %s — Kalshi status: %s",
                    trade_id, side.upper(), ticker, kalshi_status,
                )

            elif FILL_TIMEOUT_MINUTES > 0 and age_minutes > FILL_TIMEOUT_MINUTES:
                # Order is still resting past the timeout — cancel it.
                await self._cancel_order(session, order_id)
                self._conn.execute(
                    "UPDATE trades SET status = 'rejected' WHERE id = ?",
                    (trade_id,),
                )
                logging.info(
                    "[FILL TIMEOUT] Cancelled stale resting order:"
                    " trade #%d %s %s  (age %.0f min > %.0f min limit)",
                    trade_id, side.upper(), ticker,
                    age_minutes, FILL_TIMEOUT_MINUTES,
                )

    async def _fetch_order_status(
        self,
        session: aiohttp.ClientSession,
        order_id: str,
    ) -> dict | None:
        """Return the Kalshi order dict for *order_id*, or None on error."""
        base = (
            "https://api.elections.kalshi.com"
            if os.environ.get("KALSHI_ENVIRONMENT", "demo") == "production"
            else "https://demo-api.kalshi.co"
        )
        path = f"{_ORDERS_PATH}/{order_id}"
        headers = generate_headers("GET", path)
        try:
            async with session.get(
                f"{base}{path}",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data.get("order", {})
        except aiohttp.ClientResponseError as exc:
            logging.warning(
                "Order status fetch failed for %s: HTTP %s", order_id, exc.status
            )
        except aiohttp.ClientError as exc:
            logging.warning("Order status fetch error for %s: %s", order_id, exc)
        return None

    async def _cancel_order(
        self,
        session: aiohttp.ClientSession,
        order_id: str,
    ) -> None:
        """Send a DELETE request to cancel a resting order on Kalshi."""
        base = (
            "https://api.elections.kalshi.com"
            if os.environ.get("KALSHI_ENVIRONMENT", "demo") == "production"
            else "https://demo-api.kalshi.co"
        )
        path = f"{_ORDERS_PATH}/{order_id}"
        headers = generate_headers("DELETE", path)
        try:
            async with session.delete(
                f"{base}{path}",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                resp.raise_for_status()
                logging.debug("Cancelled order %s.", order_id)
        except aiohttp.ClientError as exc:
            logging.warning("Order cancel failed for %s: %s", order_id, exc)

    async def _execute(
        self,
        session: aiohttp.ClientSession,
        ticker: str,
        side: str,
        count: int,
        limit_price: int,
        opportunity_kind: str,
        score: float,
        p_estimate: float,
        source: str = "",
        spread_id: str | None = None,
        yes_bid: int | None = None,
        yes_ask: int | None = None,
        kelly_fraction: float | None = None,
        signal_p_yes: float | None = None,
        corroborating_sources: list[str] | None = None,
    ) -> None:
        """Place the order (or log it in dry-run mode) and persist to SQLite."""
        mode = "dry_run" if self._dry_run else "live"
        logged_at = datetime.now(timezone.utc).isoformat()
        order_id: str | None = None
        status: str
        error_msg: str | None = None

        cost_cents = limit_price if side == "yes" else (100 - limit_price)
        _logged_kelly = kelly_fraction if kelly_fraction is not None else KELLY_FRACTION

        if self._dry_run:
            logging.info(
                "[DRY-RUN] Would buy %d %s @ %d¢ (cost %d¢)  %s"
                "  kelly=%.2f  p=%.2f  score=%.2f  src=%s",
                count, side.upper(), limit_price, cost_cents, ticker,
                _logged_kelly, p_estimate, score, source or "unknown",
            )
            status = "pending"
        else:
            order_id, status, error_msg = await self._place_order(
                session, ticker, side, count, limit_price
            )

        market_p_entry = (
            (float(yes_bid) + float(yes_ask)) / 200.0
            if yes_bid is not None and yes_ask is not None else None
        )

        corroboration_str = (
            ",".join(corroborating_sources) if corroborating_sources else None
        )

        self._conn.execute(
            """
            INSERT INTO trades (
                logged_at, mode, ticker, side, count, limit_price,
                opportunity_kind, score, kelly_fraction, p_estimate,
                status, order_id, error_msg, source, spread_id,
                market_p_entry, yes_bid_entry, yes_ask_entry, signal_p_yes,
                corroborating_sources
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                logged_at, mode, ticker, side, count, limit_price,
                opportunity_kind, score, _logged_kelly, p_estimate,
                status, order_id, error_msg, source or None, spread_id,
                market_p_entry, yes_bid, yes_ask, signal_p_yes,
                corroboration_str,
            ),
        )

    async def _place_order(
        self,
        session: aiohttp.ClientSession,
        ticker: str,
        side: str,
        count: int,
        yes_price: int,
    ) -> tuple[str | None, str, str | None]:
        """POST a limit buy order to the Kalshi API.

        Returns:
            (order_id, status, error_msg) where status is one of
            'filled', 'resting', 'rejected', or 'error'.
        """
        base = (
            "https://api.elections.kalshi.com"
            if os.environ.get("KALSHI_ENVIRONMENT", "demo") == "production"
            else "https://demo-api.kalshi.co"
        )
        headers = generate_headers("POST", _ORDERS_PATH)
        body = {
            "ticker": ticker,
            "client_order_id": str(uuid.uuid4()),
            "action": "buy",
            "type": "limit",
            "side": side,
            "count": count,
            "yes_price": yes_price,
        }

        try:
            async with session.post(
                f"{base}{_ORDERS_PATH}",
                json=body,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                order = data.get("order", {})
                oid = order.get("order_id") or order.get("id")
                order_status = order.get("status", "unknown")
                logging.info(
                    "[LIVE] Order placed: %s %s x%d @ %d¢  id=%s  status=%s",
                    side.upper(), ticker, count, yes_price, oid, order_status,
                )
                return oid, order_status, None

        except aiohttp.ClientResponseError as exc:
            msg = f"HTTP {exc.status}: {exc.message}"
            logging.error("Order failed for %s: %s", ticker, msg)
            return None, "error", msg

        except aiohttp.ClientError as exc:
            msg = str(exc)
            logging.error("Order request error for %s: %s", ticker, msg)
            return None, "error", msg

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()
