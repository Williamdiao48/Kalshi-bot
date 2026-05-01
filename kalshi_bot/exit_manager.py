"""Position exit / profit-taking manager for the Kalshi information-alpha bot.

Called every poll cycle (via DryRunLedger) after open positions are enriched
with live prices.  Checks each open position against configurable thresholds
and, when triggered, records the exit in the database and (in live mode)
places a sell order on Kalshi.

Exit triggers
-------------
  PROFIT_TAKE    — unrealized P&L ≥ EXIT_PROFIT_TAKE × total_cost  (default 50%)
  STOP_LOSS      — unrealized P&L ≤ −EXIT_STOP_LOSS  × total_cost  (default 80%)
  TRAILING_STOP  — peak pct_gain ever exceeded EXIT_TRAILING_DRAWDOWN, and
                   current pct_gain has since fallen ≥ EXIT_TRAILING_DRAWDOWN
                   below that peak.  Default 0.30 — enabled.

Set these via environment variables:

  EXIT_PROFIT_TAKE=0.50                # exit when up 50% on cost
  EXIT_STOP_LOSS=0.80                  # exit when down 80% on cost (0 = disabled)
  EXIT_TRAILING_DRAWDOWN=0.30          # trailing drawdown fraction from peak
  EXIT_TRAILING_NEARCLOSE_HOURS=2.0    # tighten trailing stop inside this window
  EXIT_TRAILING_NEARCLOSE_DRAWDOWN=0.15 # tighter drawdown used near close

Source-specific profit-take
----------------------------
EXIT_SOURCE_PROFIT_TAKE supports composite ``source:side`` keys so each
combination can have its own threshold.  Lookup order:
  1. exact ``source:side`` key  (e.g. ``"noaa_observed:yes"``)
  2. bare ``source`` key        (e.g. ``"noaa_observed"``)
  3. global EXIT_PROFIT_TAKE fallback

Rationale for compiled-in defaults:
  noaa_observed:yes  0.50 — observed temp exceeds the strike; outcome is
                            near-locked but the market can reverse (3/7 exits
                            in dry-run history confirmed this).  50% is
                            aggressive enough to harvest quickly without
                            exiting at the first tick of movement (the old
                            0.33 threshold was too eager on cheap contracts).
  noaa_observed      0.80 — NO side still has uncertainty (temp can still
                            rise); give those more room.
  nws_alert          0.80 — high-confidence but not yet ground truth.
  eia                0.50 — EIA data is public and already priced in; treat as
  eia_inventory      0.50   a moderate-confidence forecast, not a lock.
                            Historical "100% WR" was all far-OTM NO trades;
                            the first genuinely contested signal was a bad trade
                            caused by stale 2022-era spot price data.

Exit columns added to the ``trades`` table
------------------------------------------
  exited_at          TEXT  — UTC ISO-8601 timestamp of exit decision
  exit_price_cents   INT   — conservative exit price (yes_bid or 100−yes_ask)
  exit_pnl_cents     REAL  — P&L captured at exit (signed cents)
  exit_reason        TEXT  — 'profit_take' | 'stop_loss'
  exit_order_id      TEXT  — Kalshi order ID (live mode only; NULL in dry-run)

Performance analysis queries
-----------------------------
After settlement, compare captured vs. hypothetical hold P&L:

  -- For every early exit that later settled, was exiting early better?
  SELECT
      t.id,
      t.ticker,
      t.exit_reason,
      t.exit_pnl_cents,
      CASE
          WHEN t.outcome = t.side           THEN (100 - t.limit_price) * t.count
          WHEN t.outcome IS NOT NULL        THEN -t.limit_price * t.count
          ELSE NULL
      END AS hold_pnl_cents,
      t.exit_pnl_cents - (
          CASE
              WHEN t.outcome = t.side       THEN (100 - t.limit_price) * t.count
              WHEN t.outcome IS NOT NULL    THEN -t.limit_price * t.count
              ELSE NULL
          END
      ) AS exit_improvement_cents
  FROM trades t
  WHERE t.exited_at IS NOT NULL
    AND t.outcome   IS NOT NULL;

  -- Distribution of exit pct_gain by exit reason
  SELECT exit_reason,
         COUNT(*) AS exits,
         AVG(exit_pnl_cents * 1.0 / (
             CASE side
                 WHEN 'yes' THEN limit_price * count
                 ELSE (100 - limit_price) * count
             END
         )) AS avg_pct_gain
  FROM trades
  WHERE exited_at IS NOT NULL
  GROUP BY exit_reason;
"""

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import aiohttp

from .auth import generate_headers

if TYPE_CHECKING:
    import sqlite3
    from .dry_run_ledger import _Trade
    from .numeric_matcher import NumericOpportunity
    from .polymarket_matcher import PolyOpportunity

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EXIT_PROFIT_TAKE: float = float(os.environ.get("EXIT_PROFIT_TAKE", "0.20"))
EXIT_STOP_LOSS: float   = float(os.environ.get("EXIT_STOP_LOSS",   "0.70"))

# Tight stop-loss for YES trades on KXLOWT (daily-low) markets backed by
# observed sources (noaa_observed, metar).  Unlike KXHIGHT observed YES trades
# (where a high above the strike is permanently locked in), a KXLOWT morning
# running-minimum above the strike is not locked — temps can still fall due to
# cold fronts.  A tight SL cuts losses quickly if the market moves against us.
# Set to 0 to disable (falls back to source-specific EXIT_SOURCE_STOP_LOSS).
KXLOWT_OBS_YES_STOP_LOSS: float = float(os.environ.get("KXLOWT_OBS_YES_STOP_LOSS", "0.20"))

# Dynamic stop widening for KXLOWT YES positions far from close.
# KXLOWT markets are illiquid overnight (5–15¢ bid-ask spreads common from
# midnight to 5 AM).  KXLOWT_OBS_YES_STOP_LOSS=0.20 fires on spread noise
# at e.g. 21¢ on a 27¢ entry, stopping out correct signals before the morning
# confirms.  When hours_remaining > KXLOWT_EARLY_STOP_HOURS, override the stop
# to KXLOWT_EARLY_STOP_LOSS so the position can breathe through the illiquid
# window.  The tight 0.20 threshold resumes once within KXLOWT_EARLY_STOP_HOURS.
# Set to 0 to disable (always use normal threshold).
KXLOWT_EARLY_STOP_LOSS: float = float(
    os.environ.get("KXLOWT_EARLY_STOP_LOSS", "0.60")
)
KXLOWT_EARLY_STOP_HOURS: float = float(
    os.environ.get("KXLOWT_EARLY_STOP_HOURS", "4.0")
)

# Dynamic stop widening for KXHIGHT NO positions far from close.
# Default EXIT_STOP_LOSS=0.70 fires when YES_bid rises to ~85¢ on a 59¢ NO entry.
# KXHIGHT NO entered 14h+ before close can see YES_bid spike to 80¢+ intraday on
# thin books before collapsing to 0 at resolution (trade #12: YES_bid peaked 81¢,
# resolved NO 10 min later).  When hours_remaining > KXHIGHT_EARLY_STOP_HOURS,
# override the stop to KXHIGHT_EARLY_STOP_LOSS so the position survives intraday
# noise.  The tighter 0.70 threshold resumes once within KXHIGHT_EARLY_STOP_HOURS.
# Set to 0 to disable (always use EXIT_STOP_LOSS).
KXHIGHT_EARLY_STOP_LOSS: float = float(
    os.environ.get("KXHIGHT_EARLY_STOP_LOSS", "0.85")
)
KXHIGHT_EARLY_STOP_HOURS: float = float(
    os.environ.get("KXHIGHT_EARLY_STOP_HOURS", "4.0")
)

# Capital recycling: force-exit near-settled positions to free capital for new trades.
# Sources eligible for recycling — observed-data sources only (high confidence).
CAPITAL_RECYCLE_SOURCES: frozenset[str] = frozenset(
    s.strip() for s in
    os.environ.get("CAPITAL_RECYCLE_SOURCES", "band_arb,metar,noaa_observed").split(",")
)
# Minimum current NO value (¢) to be eligible.  At 97¢ the YES bid is ≤ 3¢ —
# the market has essentially priced in settlement.  Set to 0 to disable recycling.
CAPITAL_RECYCLE_MIN_NO_VALUE: int = int(os.environ.get("CAPITAL_RECYCLE_MIN_NO_VALUE", "97"))

# Contra-signal exit: if this many independent real-time sources all say NO on
# a ticker where we hold a noaa_day2:yes between position, exit immediately.
# Eligible sources: noaa_observed, metar, hrrr, open_meteo, nws_hourly.
# CONTRA_SIGNAL_MIN_EDGE_F: each source must have at least this edge (°F) on
# its NO signal to count (filters out marginal forecasts near the band edge).
CONTRA_SIGNAL_MIN_SOURCES: int   = int(os.environ.get("CONTRA_SIGNAL_MIN_SOURCES", "2"))
CONTRA_SIGNAL_MIN_EDGE_F:  float = float(os.environ.get("CONTRA_SIGNAL_MIN_EDGE_F",  "3.0"))

# KXLOWT between YES contra-signal exit.
# While holding a KXLOWT between YES position, if KXLOWT_CONTRA_MIN_SOURCES
# independent forecast models update to predict fc_low < strike_lo (the daily
# low will fall BELOW the band floor), exit immediately before the market
# reprices.  Uses a much lower edge threshold than the global counter-signal
# (which is calibrated for over/under markets with wide strike ranges) because
# a 1°F-wide between band means even a 2°F shift below the floor is definitive.
# Only real-time/forecast sources count (hrrr, nws_hourly, noaa, open_meteo,
# weatherapi) — observed sources (metar, noaa_observed) would confirm YES.
# Set KXLOWT_CONTRA_MIN_SOURCES=0 to disable.
KXLOWT_CONTRA_MIN_EDGE_F: float = float(
    os.environ.get("KXLOWT_CONTRA_MIN_EDGE_F", "2.0")
)
KXLOWT_CONTRA_MIN_SOURCES: int = int(
    os.environ.get("KXLOWT_CONTRA_MIN_SOURCES", "2")
)
_KXLOWT_CONTRA_FORECAST_SOURCES: frozenset[str] = frozenset({
    "hrrr", "nws_hourly", "noaa", "noaa_day1", "open_meteo", "weatherapi",
})

# Trailing stop: exit if the position has *ever* been up by at least this
# fraction of cost, and has since drawn back by EXIT_TRAILING_DRAWDOWN below
# that peak.
# Example: EXIT_TRAILING_DRAWDOWN=0.30 means: if the position was up 60% and
# has since fallen back to +30% (a 30pp drawdown from peak), exit to lock in
# the remaining gain.  Also fires if the position was up 40% and has since
# gone flat or negative (drawback ≥ 30pp from peak).
# Default 0.30 — enabled.  Set to 0 to disable.
EXIT_TRAILING_DRAWDOWN: float = float(
    os.environ.get("EXIT_TRAILING_DRAWDOWN", "0.00")
)

# Per-source trailing drawdown overrides (JSON dict).
# Same key semantics as EXIT_SOURCE_PROFIT_TAKE: "source:side" or bare "source".
# Allows sources with smaller expected gains to use a tighter trailing stop
# that activates earlier (lower peak requirement) and exits sooner on reversal.
#
# noaa_day2  0.05 — day-ahead NWS forecast NO trades:
#   These positions sometimes peak at 5–10% overnight before reversing sharply
#   when morning observations contradict the forecast.  The global 30% threshold
#   never activates on those small gains.  A 5% drawdown threshold means: if
#   the trade ever exceeds 5% gain and then falls back 5pp, exit immediately
#   rather than riding it to a stop-loss.
_src_trailing_raw = os.environ.get(
    "EXIT_SOURCE_TRAILING_DRAWDOWN",
    '{"noaa_day2": 0.00, "noaa_day2_early": 0.05,'
    ' "noaa_day2:yes": 0.12, "noaa_day2_early:yes": 0.12,'
    ' "noaa_day2:no": 0.10, "noaa_day2_early:no": 0.10,'
    ' "noaa:no": 0.08, "open_meteo:no": 0.08,'
    ' "noaa_observed:no": 0.30, "hrrr:no": 0.08,'
    ' "noaa": 0.00, "noaa_observed": 0.15, "polymarket": 0.00,'
    ' "binance": 0.20, "coinbase": 0.20}',
)
try:
    EXIT_SOURCE_TRAILING_DRAWDOWN: dict[str, float] = json.loads(_src_trailing_raw)
except json.JSONDecodeError:
    logging.warning("EXIT_SOURCE_TRAILING_DRAWDOWN is not valid JSON — using global default.")
    EXIT_SOURCE_TRAILING_DRAWDOWN = {}

# Time-gated trailing stop tightening.
# Within EXIT_TRAILING_NEARCLOSE_HOURS of market close, use the tighter
# EXIT_TRAILING_NEARCLOSE_DRAWDOWN instead of EXIT_TRAILING_DRAWDOWN.
# Rationale: a position drifting back with 90 min left has no time to
# recover — lock in remaining gains immediately.
# Set EXIT_TRAILING_NEARCLOSE_HOURS=0 to disable this feature.
EXIT_TRAILING_NEARCLOSE_HOURS: float = float(
    os.environ.get("EXIT_TRAILING_NEARCLOSE_HOURS", "2.0")
)
EXIT_TRAILING_NEARCLOSE_DRAWDOWN: float = float(
    os.environ.get("EXIT_TRAILING_NEARCLOSE_DRAWDOWN", "0.15")
)

# Longshot entry protection.
# Contracts entered at a very low price (e.g. 3¢) are binary moonshots: they
# either go to 0¢ or 100¢.  A standard 50% profit-take threshold fires when
# the contract rises from 3¢ to 4.5¢ — capturing almost none of the upside if
# the position is heading to 100¢.  Data from 7 longshot exits: held value
# would have been $6.80 vs $2.60 captured → $4.20 destroyed by early exit.
#
# When the cost per contract is ≤ EXIT_PROFIT_TAKE_LONGSHOT_CENTS, the
# resolved profit-take threshold is multiplied by EXIT_PROFIT_TAKE_LONGSHOT_MULT
# (default 10×), requiring e.g. a 500% gain before exiting a 3¢ contract
# (exit at ~18¢+) — still well below the 100¢ settlement value.
#
# Set EXIT_PROFIT_TAKE_LONGSHOT_CENTS=0 to disable.
EXIT_PROFIT_TAKE_LONGSHOT_CENTS: int = int(
    os.environ.get("EXIT_PROFIT_TAKE_LONGSHOT_CENTS", "10")
)
EXIT_PROFIT_TAKE_LONGSHOT_MULT: float = float(
    os.environ.get("EXIT_PROFIT_TAKE_LONGSHOT_MULT", "10.0")
)

# Per-source profit-take and stop-loss overrides (JSON dicts).
# Keys may be bare source names ("noaa_observed") or composite "source:side"
# keys ("noaa_observed:yes").  Lookup order: source:side → source → global.
#
# noaa_observed:yes  0.50 — observed temp already exceeds the strike; outcome
#                           is near-locked but the market can still reverse.
#                           Harvest at 50% rather than the aggressive 33%,
#                           which fired too early (e.g. id 185: exited at 34%
#                           gain on a contract that settled at 100¢).
# noaa_observed      0.80 — NO side still has residual uncertainty (temp can
#                           still climb); give those positions more room.
# nws_alert          0.80 — high-confidence directional signal but not yet
#                           ground truth; let the position run.
# eia                0.40 — EIA is a public, already-priced-in signal; treat
# eia_inventory      0.40   like a weak forecast (same tier as polymarket).
# noaa / open_meteo  0.50 — raw forecast: symmetric risk each way.
# polymarket / manifold    0.40 — text matches are noisier; cut losses fast.
# noaa_day2:no  0.04 — day-ahead NWS forecast NO trades:
#   The market is betting the temperature won't reach the strike.  These
#   positions can drift positive overnight when models briefly agree, then
#   reverse sharply at sunrise when actual observations contradict the
#   forecast.  Exit at 4% gain — tighter than the original 7% because
#   the reversal often happens before price reaches 7%.  Take whatever
#   small early gain is available rather than waiting for a larger target
#   that frequently never materialises.
#   Paired with EXIT_SOURCE_TRAILING_DRAWDOWN "noaa_day2": 0.05 as a
#   secondary catch if the position doesn't reach the profit-take first.
_pt_raw = os.environ.get(
    "EXIT_SOURCE_PROFIT_TAKE",
    '{"noaa_observed:yes": 0.50, "noaa_observed": 0.75, "metar:yes": 0.50, "metar": 0.80, "nws_alert": 0.80,'
    ' "eia": 0.40, "eia_inventory": 0.40, "noaa_day2:no": 0.07, "noaa_day2_early:no": 0.07,'
    ' "noaa_day2:yes": 0.35, "noaa_day2_early:yes": 0.35,'
    ' "noaa": 0.40, "noaa_day2": 0.20, "polymarket": 0.25,'
    ' "obs_trajectory:yes": 0.30,'
    ' "band_arb:yes": 0.15, "band_arb:no": 2.00,'
    ' "forecast_no": 0.40, "numeric": 0.75,'
    ' "binance": 0.35, "coinbase": 0.35}',
)
_sl_raw = os.environ.get(
    "EXIT_SOURCE_STOP_LOSS",
    '{"noaa_observed:yes": 0.50, "metar:yes": 0.05, "nws_climo:yes": 0.05, "nws_alert:yes": 0.05,'
    ' "noaa_observed": 0.70, "metar": 0.60, "noaa": 0.30, "open_meteo": 0.50,'
    ' "polymarket": 0.20, "manifold": 0.40, "metaculus": 0.50, "eia": 0.50, "eia_inventory": 0.50,'
    ' "noaa_day2:yes": 0.45, "noaa_day2_early:yes": 0.45,'
    ' "noaa_day2:no": 0.55, "noaa_day2_early:no": 0.55,'
    ' "noaa_day2": 0.30, "hrrr": 0.40,'
    ' "band_arb:no": 0.70, "band_arb:yes": 0.70, "forecast_no": 0.90,'
    ' "numeric": 0.70,'
    ' "binance": 0.25, "coinbase": 0.25}',
)
try:
    EXIT_SOURCE_PROFIT_TAKE: dict[str, float] = json.loads(_pt_raw)
except json.JSONDecodeError:
    logging.warning("EXIT_SOURCE_PROFIT_TAKE is not valid JSON — using global default.")
    EXIT_SOURCE_PROFIT_TAKE = {}
try:
    EXIT_SOURCE_STOP_LOSS: dict[str, float] = json.loads(_sl_raw)
except json.JSONDecodeError:
    logging.warning("EXIT_SOURCE_STOP_LOSS is not valid JSON — using global default.")
    EXIT_SOURCE_STOP_LOSS = {}

# Minimum hold time before any exit trigger (profit_take, stop_loss, trailing)
# fires on a newly-logged position.  Prevents the exit manager from stopping out
# a trade on the same cycle it was entered — which happens when the current
# yes_bid is below the limit_price (normal bid-ask spread) and the stop-loss
# threshold is tighter than the spread.
# Default: 2 minutes.  Set to 0 to disable.
EXIT_MIN_HOLD_MINUTES: float = float(
    os.environ.get("EXIT_MIN_HOLD_MINUTES", "2.0")
)

# Counter-signal exit: exit a position when live forecast data has flipped
# direction against the open trade.
#
# COUNTER_SIGNAL_MIN_EDGE — minimum edge (°F for temp, % for crypto/forex) that
#   the new counter-direction signal must exceed.  Set well above the entry
#   threshold so marginal model disagreements don't force premature exits.
#   Default 6.0 — a 6°F opposite-direction edge is roughly 2× the minimum
#   entry edge (5°F), meaning the forecast has shifted convincingly past the
#   strike in the wrong direction.  0 = disabled.
#
# COUNTER_SIGNAL_MIN_SOURCES — number of independent forecast sources that must
#   simultaneously show a counter-direction signal with edge ≥ MIN_EDGE.
#   Default 2: requires at least two models (e.g. HRRR + NWS hourly) to agree
#   before exiting.  A single outlier model never triggers an exit.
#
# COUNTER_SIGNAL_MAX_PROFIT_PCT — if the position is already up by this fraction
#   of cost, skip the counter-signal exit.  The trailing stop will handle
#   re-entry of profits without abandoning a near-settled winner.
#   Default 0.40 (40% gain).  Set to 1.0 to disable this guard.
COUNTER_SIGNAL_MIN_EDGE: float = float(
    os.environ.get("COUNTER_SIGNAL_MIN_EDGE", "6.0")
)
COUNTER_SIGNAL_MIN_SOURCES: int = int(
    os.environ.get("COUNTER_SIGNAL_MIN_SOURCES", "2")
)
COUNTER_SIGNAL_MAX_PROFIT_PCT: float = float(
    os.environ.get("COUNTER_SIGNAL_MAX_PROFIT_PCT", "0.40")
)

# Near-close stop-loss suppression for physically-locked signals.
#
# When a position is within EXIT_STOP_LOSS_NEARCLOSE_HOURS of market close
# AND its source:side is considered irrevocably locked (the underlying
# physical observation cannot reverse), the stop-loss threshold is replaced
# with EXIT_STOP_LOSS_LOCKED_NEARCLOSE (default 0.0 = fully disabled).
#
# This prevents thin-liquidity midnight price spikes from firing a stop-loss
# on signals whose outcome is already determined by physics.
#
# Locked signal examples:
#   band_arb:no    — METAR observed daily max crossed the KXHIGH band ceiling.
#                    The observation is recorded; it cannot be un-crossed.
#   noaa_observed:no on KXLOWT* — the daily minimum temperature cannot rise
#                    during a calendar day.  Any KXLOWT* NO signal is
#                    structurally locked from entry.
#                    Trade #250: Chicago 11 PM min 44.6°F vs 38.5°F ceil,
#                    34 min before midnight close — a transient 18¢ NO spike
#                    fired a false stop-loss.  Market resolved NO.
#
# Set EXIT_STOP_LOSS_NEARCLOSE_HOURS=0 to disable this feature entirely.
EXIT_STOP_LOSS_NEARCLOSE_HOURS: float = float(
    os.environ.get("EXIT_STOP_LOSS_NEARCLOSE_HOURS", "2.0")
)
# Threshold used near close for locked signals.
# 0.0 = fully disabled (never stop out within the near-close window).
EXIT_STOP_LOSS_LOCKED_NEARCLOSE: float = float(
    os.environ.get("EXIT_STOP_LOSS_LOCKED_NEARCLOSE", "0.0")
)
# Floor-price stop-loss suppression: if the current exit price (NO price for
# NO positions, YES bid for YES positions) is at or below this many cents,
# suppress the stop-loss entirely.  At 0-2¢ there is no further downside to
# protect against — the position has already hit rock bottom and can only
# recover toward 100¢ at settlement.  Stopping out locks in the full loss
# with zero chance of recovery, which is strictly worse than holding.
# Example: KXHIGHTATL-26APR19-T70 stopped at 0¢ for -$7.38 when METAR+NOAA
# both confirmed the official high was above the strike.
EXIT_STOP_LOSS_FLOOR_PRICE: int = int(
    os.environ.get("EXIT_STOP_LOSS_FLOOR_PRICE", "2")
)

# Absolute NO-price profit-take for band_arb:no positions.
# band_arb:no trades buy NO when YES is priced above a physical ceiling (e.g. the
# temperature strip can never reach the strike).  The ideal exit is when the market
# fully corrects and YES drops near zero — but waiting for 100¢ risks a late reversal
# that wipes the gain (see trades #87 and #95: peaked at 96–97¢ then reversed to loss).
# When current_mid (= 100 − yes_ask for NO positions) reaches this threshold, exit
# immediately regardless of the percentage gain.
# Set to 0 to disable (falls back to the standard EXIT_SOURCE_PROFIT_TAKE logic).
BAND_ARB_NO_EXIT_PRICE_CENTS: int = int(
    os.environ.get("BAND_ARB_NO_EXIT_PRICE_CENTS", "95")
)

# Sources where intraday price moves are noise — hold to settlement, skip all exits.
# EIA/BLS/Fed/crypto: single data-release resolution; thin books before release
# don't reflect new information.
_HOLD_TO_SETTLEMENT_SOURCES: frozenset[str] = frozenset({
    "eia", "eia_inventory",
    "bls", "fred", "cme_fedwatch", "adp", "chicago_pmi",
    "binance", "coinbase", "coingecko",
    "frankfurter", "yahoo_forex",
    "polymarket", "metaculus", "manifold", "predictit",
})

# Locked weather signals — hold to settlement, stop_loss only.
# These are based on confirmed observed data or official NWS advisories, so
# the market will reprice to full value by settlement.  Exiting early on
# profit-take destroys large EV (band_arb trade 237: 180¢ captured vs 645¢).
# Stop-loss still fires if the market decisively reverses (sensor error).
_LOCKED_STOP_LOSS_ONLY: frozenset[str] = frozenset({
    "noaa_observed", "metar", "nws_climo", "nws_alert",
    # band_arb: temperature physically recorded above ceiling — most locked signal.
    "band_arb",
})

# Forecast and trajectory sources — profit-take ENABLED for YES side.
# These are probabilistic (model-based or trend-projected) rather than locked
# outcomes, so capturing profit when the market reprices toward the target is
# correct EV management.  NO side remains stop-loss-only (directional NO from
# a forecast model is less certain and should not be profit-taken early).
_FORECAST_PROFIT_TAKE_SOURCES: frozenset[str] = frozenset({
    "noaa", "noaa_day1", "noaa_day2",
    "nws_hourly", "hrrr",
    "open_meteo", "weatherapi",
    # obs_trajectory: projects likely peak from current METAR warming trend.
    # Enter YES when slope × parabolic model says peak > strike; profit-take
    # when the market catches up to the trajectory projection (~30% gain).
    "obs_trajectory",
})

_ORDERS_PATH = "/trade-api/v2/orders"


# ---------------------------------------------------------------------------
# Exit event (returned to caller for logging / overview)
# ---------------------------------------------------------------------------

class ExitEvent:
    """Minimal record of a single triggered exit."""

    __slots__ = (
        "trade_id", "ticker", "side", "count",
        "entry", "exit_price", "pnl_cents", "reason",
    )

    def __init__(
        self,
        trade_id:   int,
        ticker:     str,
        side:       str,
        count:      int,
        entry:      int,
        exit_price: int,
        pnl_cents:  float,
        reason:     str,
    ) -> None:
        self.trade_id   = trade_id
        self.ticker     = ticker
        self.side       = side
        self.count      = count
        self.entry      = entry       # limit_price (yes_price) in cents
        self.exit_price = exit_price  # conservative exit price in cents
        self.pnl_cents  = pnl_cents   # signed P&L captured
        self.reason     = reason      # 'profit_take' | 'stop_loss'


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class ExitManager:
    """Checks open positions against exit thresholds each poll cycle.

    Shares the SQLite connection owned by DryRunLedger so all writes land in
    the same WAL transaction without opening a second connection.

    Usage::

        manager = ExitManager(conn, dry_run=True)
        exits = await manager.check_exits(session, enriched_trades)
    """

    _EXIT_COLUMNS: list[tuple[str, str]] = [
        ("exited_at",        "TEXT"),
        ("exit_price_cents", "INTEGER"),
        ("exit_pnl_cents",   "REAL"),
        ("exit_reason",      "TEXT"),
        ("exit_order_id",    "TEXT"),
        # 1 if trade was entered after the daily peak was confirmed (peak_past=True
        # on the originating NumericOpportunity).  Used by _is_locked_signal to
        # suppress stop-loss on KXHIGH:no positions after 4:30 PM local.
        ("peak_past",        "INTEGER"),
    ]

    def __init__(self, conn: "sqlite3.Connection", dry_run: bool = True) -> None:
        self._conn    = conn
        self._dry_run = dry_run
        self._migrate_schema()
        if EXIT_TRAILING_DRAWDOWN > 0:
            if EXIT_TRAILING_NEARCLOSE_HOURS > 0:
                trailing_str = (
                    f"  trailing={EXIT_TRAILING_DRAWDOWN:.0%}"
                    f"  nearclose={EXIT_TRAILING_NEARCLOSE_DRAWDOWN:.0%}"
                    f"@{EXIT_TRAILING_NEARCLOSE_HOURS:.1f}h"
                )
            else:
                trailing_str = f"  trailing={EXIT_TRAILING_DRAWDOWN:.0%}"
        else:
            trailing_str = "  trailing=off"
        src_overrides = len(EXIT_SOURCE_PROFIT_TAKE) + len(EXIT_SOURCE_STOP_LOSS)
        locked_str = (
            f"  locked-SL={EXIT_STOP_LOSS_LOCKED_NEARCLOSE:.0%}"
            f"@{EXIT_STOP_LOSS_NEARCLOSE_HOURS:.1f}h"
            if EXIT_STOP_LOSS_NEARCLOSE_HOURS > 0
            else "  locked-SL=off"
        )
        logging.info(
            "ExitManager — profit_take=%.0f%%  stop_loss=%.0f%%%s%s"
            "  source_overrides=%d  dry_run=%s",
            EXIT_PROFIT_TAKE * 100,
            EXIT_STOP_LOSS  * 100,
            trailing_str,
            locked_str,
            src_overrides,
            dry_run,
        )

    # -----------------------------------------------------------------------
    # Schema migration
    # -----------------------------------------------------------------------

    def _migrate_schema(self) -> None:
        """Add exit columns to the trades table if they are missing."""
        existing = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(trades)").fetchall()
        }
        for col, defn in self._EXIT_COLUMNS:
            if col not in existing:
                self._conn.execute(f"ALTER TABLE trades ADD COLUMN {col} {defn}")

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    async def check_exits(
        self,
        session: aiohttp.ClientSession,
        trades:  "list[_Trade]",
    ) -> list[ExitEvent]:
        """Evaluate all open positions and execute exits where thresholds are met.

        Should be called every poll cycle with the already-enriched trade list
        from ``DryRunLedger._enrich()`` so no extra API calls are needed.

        Trigger evaluation order (first match wins, rest are skipped):
          1. Already settled or exited — skip entirely.
          2. Minimum hold period (``EXIT_MIN_HOLD_MINUTES``) not elapsed — skip.
          3. Near-close suppression — within ``EXIT_STOP_LOSS_NEARCLOSE_HOURS``
             of market close, stop-loss is tightened to ``EXIT_STOP_LOSS_LOCKED_NEARCLOSE``
             (near-zero, allowing the position to ride to settlement).
          4. Floor-price suppression — if current exit price ≤ ``EXIT_STOP_LOSS_FLOOR_PRICE``¢,
             stop-loss is disabled (position is at rock bottom; only upside remains).
          5. Locked-signal suppression — if ``_is_locked_signal()`` returns True
             (daily temperature peak confirmed past), stop-loss is suppressed.
          6. Source-specific profit-take — ``EXIT_SOURCE_PROFIT_TAKE`` per-source overrides.
          7. Profit-take — ``pct_gain >= EXIT_PROFIT_TAKE``.
          8. Trailing stop — drawdown from peak > ``EXIT_TRAILING_DRAWDOWN``.
          9. Stop-loss — ``pct_gain <= -EXIT_STOP_LOSS``.

        Args:
            session: Shared aiohttp session (used only for live sell orders).
            trades:  Enriched ``_Trade`` objects — must have ``current_mid`` populated.
                     Typically the output of ``DryRunLedger._enrich()``.

        Returns:
            List of ``ExitEvent`` for every exit triggered this cycle.  An empty
            list means no thresholds were breached.
        """
        # Load already-exited IDs to prevent double-firing.
        exited_ids: set[int] = {
            row[0]
            for row in self._conn.execute(
                "SELECT id FROM trades WHERE exited_at IS NOT NULL"
            ).fetchall()
        }

        events: list[ExitEvent] = []

        # Compute the earliest logged_at that is eligible for exit checks.
        now_utc = datetime.now(timezone.utc)
        if EXIT_MIN_HOLD_MINUTES > 0:
            from datetime import timedelta
            _min_hold_cutoff = now_utc - timedelta(minutes=EXIT_MIN_HOLD_MINUTES)
        else:
            _min_hold_cutoff = None

        for trade in trades:
            # Skip settled, already-exited, and trades with no live price.
            if trade.settled or trade.current_mid is None:
                continue
            if trade.trade_id in exited_ids:
                continue

            # Skip trades that haven't been held long enough yet.
            if _min_hold_cutoff is not None:
                try:
                    logged = datetime.fromisoformat(
                        trade.logged_at.replace("Z", "+00:00")
                    )
                    if logged > _min_hold_cutoff:
                        continue
                except (AttributeError, ValueError):
                    pass

            cost = trade.total_cost_cents
            if cost <= 0:
                continue

            # Numeric markets resolve on a single external data release.
            # For data-release sources (EIA, Fed, BLS, crypto, forex), intraday
            # price moves are noise — thin books reacting to speculation, not new
            # information.  Hold those to settlement.
            #
            # Weather trades are different: the market continuously reprices as
            # observed temperatures update throughout the day.  A NO position
            # moving from 40¢ to 80¢ YES is the market reacting to real observed
            # data — that IS new information.  Weather trades should be subject to
            # stop_loss and trailing_stop so we don't ride losers to 0¢.
            src = getattr(trade, "source", "") or ""
            if getattr(trade, "opportunity_kind", "") == "numeric" and src in _HOLD_TO_SETTLEMENT_SOURCES:
                continue

            # Arb spread legs: never stop-loss, never trailing-stop.
            # Arb trades are paired legs of a spread (arb_detector, bracket_arb,
            # etc.) identified by a non-null spread_id.  Stopping out one leg
            # breaks the hedge: the other leg continues open, converting a
            # near-certain positive-EV spread into a one-sided loss.
            # Trade #73: BTC arb YES leg stopped out at 24¢ during a 25-minute
            # flash crash; BTC recovered to 93¢ two minutes later and settled YES.
            # The full spread would have been +$3 net; instead it was -$11.
            # Profit-take is still allowed (both legs landing in-the-money is fine).
            _spread_id = getattr(trade, "spread_id", "") or ""
            if _spread_id:
                # Re-classify to profit-take only: skip stop_loss / trailing entirely.
                # We still want to capture profit if the spread moves our way quickly.
                # Only apply to opportunity kinds that represent spread legs.
                _opp_kind = getattr(trade, "opportunity_kind", "") or ""
                if _opp_kind in ("arb", "bracket_arb", "bracket_set", "spread", "crossed_book"):
                    pnl = trade._unrealized_cents()
                    pct = pnl / cost  # cost already validated > 0 above
                    lp = getattr(trade, "limit_price", 100)
                    side = getattr(trade, "side", "") or ""
                    entry_cost = lp if side == "yes" else (100 - lp)
                    pt_thresh = self._resolve_profit_take(src, side, entry_cost)
                    if pct >= pt_thresh:
                        _pt_d = self._profit_take_detail(src, side, entry_cost)
                        event = await self._execute_exit(session, trade, pnl, "profit_take", _pt_d)
                        events.append(event)
                        exited_ids.add(trade.trade_id)
                    continue  # skip stop_loss / trailing regardless

            pnl = trade._unrealized_cents()
            pct = pnl / cost

            # Resolve per-source thresholds (composite "source:side" key first).
            src  = getattr(trade, "source", "") or ""
            side = getattr(trade, "side",   "") or ""
            # entry_cost: cost per contract (YES = limit_price, NO = 100 − limit_price).
            lp = getattr(trade, "limit_price", 100)
            entry_cost = lp if side == "yes" else (100 - lp)
            profit_take_thresh = self._resolve_profit_take(src, side, entry_cost)

            _sl_composite = f"{src}:{side}"
            stop_loss_thresh = EXIT_SOURCE_STOP_LOSS.get(
                _sl_composite,
                EXIT_SOURCE_STOP_LOSS.get(src, EXIT_STOP_LOSS),
            )
            # Track which gate last set the stop_loss_thresh for audit logging.
            _sl_detail = (
                "stop_loss:source"
                if (_sl_composite in EXIT_SOURCE_STOP_LOSS or src in EXIT_SOURCE_STOP_LOSS)
                else "stop_loss:global"
            )

            # Tighter stop-loss for KXLOWT YES trades from observed sources.
            # A morning running-minimum above the strike does not lock in a YES
            # outcome (unlike KXHIGHT) — cold fronts can push the daily low well
            # below the strike later in the day.  Cut losses fast.
            _ticker = getattr(trade, "ticker", "") or ""
            if (
                KXLOWT_OBS_YES_STOP_LOSS > 0
                and "KXLOWT" in _ticker
                and side == "yes"
                and src in ("noaa_observed", "metar")
            ):
                stop_loss_thresh = KXLOWT_OBS_YES_STOP_LOSS
                _sl_detail = "stop_loss:kxlowt_obs_yes"

            # Compute hours to close from the trade's close_time (populated by
            # _enrich).  Used for the time-gated trailing stop below and for
            # near-close locked-signal stop-loss suppression.
            hours_to_close: float | None = None
            ct_str = getattr(trade, "close_time", None)
            if ct_str:
                try:
                    ct = datetime.fromisoformat(ct_str.replace("Z", "+00:00"))
                    hours_to_close = (ct - datetime.now(timezone.utc)).total_seconds() / 3600
                except (ValueError, TypeError):
                    pass

            # Dynamic stop widening for KXLOWT YES positions far from close.
            # Overrides the tight KXLOWT_OBS_YES_STOP_LOSS upward during the
            # illiquid overnight window; once inside KXLOWT_EARLY_STOP_HOURS
            # of close, the tight threshold resumes naturally.
            if (
                KXLOWT_EARLY_STOP_LOSS > 0
                and "KXLOWT" in _ticker
                and side == "yes"
                and hours_to_close is not None
                and hours_to_close > KXLOWT_EARLY_STOP_HOURS
            ):
                if stop_loss_thresh < KXLOWT_EARLY_STOP_LOSS:
                    logging.debug(
                        "[KXLOWT-early-SL] trade #%d %s: %.1fh to close — "
                        "SL %.0f%% → %.0f%% (overnight illiquid window)",
                        trade.trade_id, _ticker,
                        hours_to_close,
                        stop_loss_thresh * 100,
                        KXLOWT_EARLY_STOP_LOSS * 100,
                    )
                    stop_loss_thresh = KXLOWT_EARLY_STOP_LOSS
                    _sl_detail = "stop_loss:kxlowt_early"

            # Dynamic stop widening for KXHIGHT NO positions far from close.
            # KXHIGHT NO trades entered 10–15h before resolution can see YES_bid
            # spike intraday on thin books before collapsing to 0 at settlement.
            # Trade #12 (Boston): YES_bid peaked 81¢ intraday, resolved 0¢ 10
            # minutes later; the 70% default stop fired prematurely at ~85¢.
            if (
                KXHIGHT_EARLY_STOP_LOSS > 0
                and "KXHIGH" in _ticker
                and side == "no"
                and hours_to_close is not None
                and hours_to_close > KXHIGHT_EARLY_STOP_HOURS
            ):
                if stop_loss_thresh < KXHIGHT_EARLY_STOP_LOSS:
                    logging.debug(
                        "[KXHIGH-early-SL] trade #%d %s: %.1fh to close — "
                        "SL %.0f%% → %.0f%% (intraday noise window)",
                        trade.trade_id, _ticker,
                        hours_to_close,
                        stop_loss_thresh * 100,
                        KXHIGHT_EARLY_STOP_LOSS * 100,
                    )
                    stop_loss_thresh = KXHIGHT_EARLY_STOP_LOSS
                    _sl_detail = "stop_loss:kxhight_early"

            # Near-close locked-signal stop-loss suppression.
            # When the market is closing soon and the signal's underlying
            # observation cannot physically reverse, replace the stop-loss
            # threshold with EXIT_STOP_LOSS_LOCKED_NEARCLOSE (default 0 =
            # fully disabled).  This prevents midnight liquidity spikes from
            # stopping out positions that will settle correctly at resolution.
            # See EXIT_STOP_LOSS_NEARCLOSE_HOURS for configuration.
            if (
                EXIT_STOP_LOSS_NEARCLOSE_HOURS > 0
                and hours_to_close is not None
                and 0 <= hours_to_close <= EXIT_STOP_LOSS_NEARCLOSE_HOURS
                and self._is_locked_signal(
                    src, side,
                    getattr(trade, "ticker", ""),
                    peak_past=bool(getattr(trade, "peak_past", False)),
                )
            ):
                if stop_loss_thresh != EXIT_STOP_LOSS_LOCKED_NEARCLOSE:
                    logging.debug(
                        "[locked-SL] trade #%d %s %s: %.1fh to close —"
                        " SL %.0f%% → %.0f%% (locked signal near-close suppression)",
                        trade.trade_id, side.upper(),
                        getattr(trade, "ticker", "?"),
                        hours_to_close,
                        stop_loss_thresh * 100,
                        EXIT_STOP_LOSS_LOCKED_NEARCLOSE * 100,
                    )
                stop_loss_thresh = EXIT_STOP_LOSS_LOCKED_NEARCLOSE
                _sl_detail = "stop_loss:locked_nearclose"

            # Floor-price stop-loss suppression: if the current exit price has
            # already reached rock bottom (≤ EXIT_STOP_LOSS_FLOOR_PRICE cents),
            # there is no further downside to protect against — the position
            # can only recover toward 100¢ at settlement.  Stopping out here
            # locks in the full loss with zero chance of recovery, which is
            # strictly worse than holding.
            # Example: KXHIGHTATL-26APR19-T70 stopped at 0¢ for -$7.38 when
            # METAR+NOAA had both confirmed the high was above the strike.
            _current_exit_price = getattr(trade, "current_mid", None)
            if (
                EXIT_STOP_LOSS_FLOOR_PRICE > 0
                and _current_exit_price is not None
                and _current_exit_price <= EXIT_STOP_LOSS_FLOOR_PRICE
                and stop_loss_thresh > 0
            ):
                logging.debug(
                    "[floor-SL] trade #%d %s %s: exit_price=%d¢ ≤ floor=%d¢"
                    " — suppressing stop-loss (position at rock bottom,"
                    " only upside remains at settlement)",
                    trade.trade_id, side.upper(),
                    getattr(trade, "ticker", "?"),
                    _current_exit_price, EXIT_STOP_LOSS_FLOOR_PRICE,
                )
                stop_loss_thresh = 0.0

            # Locked signals (observed/advisory): hold to settlement, no profit-take.
            # Forecast YES signals: profit-take enabled (model-projected, not locked).
            # Forecast NO signals: stop-loss only (directional NO is less certain).
            is_locked = (src in _LOCKED_STOP_LOSS_ONLY) and not (src == "band_arb" and side == "yes")
            is_forecast_no = src in _FORECAST_PROFIT_TAKE_SOURCES and side == "no"
            suppress_profit_take = is_locked or is_forecast_no

            reason: str | None = None
            detail: str | None = None

            # Absolute-price profit-take for band_arb:no.
            # band_arb:no is normally held to settlement (suppress_profit_take=True),
            # but a near-zero YES price means the arbitrage has fully corrected and
            # further holding risks a reversal.  Exit when NO bid reaches the threshold.
            if (
                BAND_ARB_NO_EXIT_PRICE_CENTS > 0
                and src == "band_arb"
                and side == "no"
                and _current_exit_price is not None
                and _current_exit_price >= BAND_ARB_NO_EXIT_PRICE_CENTS
            ):
                logging.info(
                    "[band_arb-PT] trade #%d %s: NO price %.0f¢ ≥ %d¢ threshold — profit_take",
                    trade.trade_id, getattr(trade, "ticker", "?"),
                    _current_exit_price, BAND_ARB_NO_EXIT_PRICE_CENTS,
                )
                reason = "profit_take"
                detail = "profit_take:band_arb_abs_price"

            if reason is None and not suppress_profit_take and pct >= profit_take_thresh:
                reason = "profit_take"
                detail = self._profit_take_detail(src, side, entry_cost)
            elif reason is None and stop_loss_thresh > 0 and pct <= -stop_loss_thresh:
                reason = "stop_loss"
                detail = _sl_detail
            elif reason is None and (EXIT_TRAILING_DRAWDOWN > 0 or (src in EXIT_SOURCE_TRAILING_DRAWDOWN or f"{src}:{side}" in EXIT_SOURCE_TRAILING_DRAWDOWN)) and not ("KXLOWT" in _ticker and side == "yes"):
                # Trailing stop is NOT gated by suppress_profit_take.
                #
                # suppress_profit_take blocks early profit-taking for locked
                # observed sources (noaa_observed, metar) and forecast NO positions
                # (noaa:no, noaa_day2:no) — correct, since those should hold toward
                # settlement.  But that flag was also blocking trailing stops, which
                # are a *different* mechanism: they protect against a peak gain that
                # has decisively reversed, regardless of whether the signal is locked.
                #
                # Bug evidence: trade #10 (noaa_observed YES, KXLOWTCHI) peaked at
                # +97% at 03:32 UTC then fell 23pp to +74% — the 15% trailing
                # threshold should have fired at 03:35 UTC.  Instead the code skipped
                # the trailing block entirely (suppress_profit_take=True for
                # noaa_observed), and the position rode down to -20% via stop_loss.
                #
                # Similarly, EXIT_SOURCE_TRAILING_DRAWDOWN entries for noaa:no (0.08),
                # noaa_day2:no (0.10), open_meteo:no (0.08) were all
                # dead config — forecast NO positions (is_forecast_no=True →
                # suppress_profit_take=True) never reached this block.
                peak = self._get_peak_pct_gain(trade.trade_id)
                if peak is not None and peak > 0:
                    # Resolve trailing drawdown threshold: source-specific
                    # overrides allow tighter stops for forecast sources that
                    # tend to peak early then reverse (e.g. noaa_day2:no).
                    src_td_key = f"{src}:{side}"
                    src_drawdown = EXIT_SOURCE_TRAILING_DRAWDOWN.get(
                        src_td_key,
                        EXIT_SOURCE_TRAILING_DRAWDOWN.get(src),
                    )
                    # Time-gated trailing stop: use tighter drawdown near close.
                    # We take the minimum of the near-close threshold and the
                    # source-specific threshold so sources already configured
                    # with a tight trailing (e.g. noaa_day2:yes=0.08) are never
                    # accidentally loosened by the near-close override (0.15).
                    base_drawdown = (
                        src_drawdown if src_drawdown is not None
                        else EXIT_TRAILING_DRAWDOWN
                    )
                    if (
                        EXIT_TRAILING_NEARCLOSE_HOURS > 0
                        and hours_to_close is not None
                        and 0 <= hours_to_close < EXIT_TRAILING_NEARCLOSE_HOURS
                    ):
                        drawdown_thresh = min(EXIT_TRAILING_NEARCLOSE_DRAWDOWN, base_drawdown)
                    else:
                        drawdown_thresh = base_drawdown
                    if (
                        drawdown_thresh > 0          # 0.0 = trailing disabled for this source
                        and peak > drawdown_thresh   # was ever meaningfully up
                        and pct < peak - drawdown_thresh  # drawn back by threshold
                    ):
                        reason = "trailing_stop"
                        detail = "trailing_stop"

            if reason is None:
                continue

            event = await self._execute_exit(session, trade, pnl, reason, detail)
            events.append(event)
            exited_ids.add(trade.trade_id)

        if events:
            logging.info("ExitManager: %d exit(s) triggered this cycle.", len(events))

        return events

    async def check_counter_signals(
        self,
        session: aiohttp.ClientSession,
        trades:      "list[_Trade]",
        numeric_opps: "list[NumericOpportunity]",
        poly_opps:    "list[PolyOpportunity]",
    ) -> list[ExitEvent]:
        """Exit positions where current data now contradicts the original trade side.

        Compares each open position against the current cycle's numeric and
        external-forecast opportunities.  When the live data has flipped direction
        — and the signal strength (edge for numeric, divergence for poly) meets or
        exceeds ``COUNTER_SIGNAL_MIN_EDGE`` — the position is exited at the
        current mark-to-market price.

        This is disabled by default (``COUNTER_SIGNAL_MIN_EDGE=0``).  Set it to
        at least your minimum entry edge so noise signals below the entry bar
        cannot force premature exits.  For example::

            COUNTER_SIGNAL_MIN_EDGE=1.0  # require ≥1 unit counter-edge to exit

        Args:
            session:      Shared aiohttp session (live sell orders only).
            trades:       Enriched _Trade objects (current_mid must be populated).
            numeric_opps: Post-gate numeric opportunities from the current cycle.
            poly_opps:    Post-gate external-forecast opportunities from the cycle.

        Returns:
            List of ExitEvent for every counter-signal exit triggered this cycle.
        """
        if COUNTER_SIGNAL_MIN_EDGE == 0:
            return []

        # Build ticker → list[NumericOpportunity] lookup (one ticker may have
        # multiple opps from different data sources — any counter-signal fires).
        numeric_lookup: dict[str, list] = {}
        for opp in numeric_opps:
            numeric_lookup.setdefault(opp.market_ticker, []).append(opp)

        # One poly opp per Kalshi ticker (already deduplicated upstream).
        poly_lookup: dict[str, object] = {
            opp.kalshi_ticker: opp for opp in poly_opps
        }

        # Load already-exited IDs so we never fire twice for the same trade.
        exited_ids: set[int] = {
            row[0]
            for row in self._conn.execute(
                "SELECT id FROM trades WHERE exited_at IS NOT NULL"
            ).fetchall()
        }

        events: list[ExitEvent] = []

        now_utc2 = datetime.now(timezone.utc)
        if EXIT_MIN_HOLD_MINUTES > 0:
            from datetime import timedelta as _td
            _cs_min_hold_cutoff = now_utc2 - _td(minutes=EXIT_MIN_HOLD_MINUTES)
        else:
            _cs_min_hold_cutoff = None

        for trade in trades:
            if trade.settled or trade.current_mid is None:
                continue
            if trade.trade_id in exited_ids:
                continue
            if trade.total_cost_cents <= 0:
                continue

            # Skip trades within the minimum hold window.
            if _cs_min_hold_cutoff is not None:
                try:
                    logged = datetime.fromisoformat(
                        trade.logged_at.replace("Z", "+00:00")
                    )
                    if logged > _cs_min_hold_cutoff:
                        continue
                except (AttributeError, ValueError):
                    pass

            # ---- Numeric counter-signal check --------------------------------
            # Require COUNTER_SIGNAL_MIN_SOURCES independent models to agree
            # on a counter-direction signal before exiting.  A single outlier
            # model (e.g. one HRRR update) can never force an exit alone.
            # Also skip if the position is already deep in profit — the trailing
            # stop will protect those gains without abandoning a near-settled
            # winner on a late noisy signal.
            pnl_cs = trade._unrealized_cents()
            profit_pct = pnl_cs / max(trade.total_cost_cents, 1)
            if profit_pct <= COUNTER_SIGNAL_MAX_PROFIT_PCT:
                counter_sources: list[str] = []
                for opp in numeric_lookup.get(trade.ticker, []):
                    if opp.edge < COUNTER_SIGNAL_MIN_EDGE:
                        continue
                    flipped = (
                        (trade.side == "yes" and opp.implied_outcome == "NO") or
                        (trade.side == "no"  and opp.implied_outcome == "YES")
                    )
                    if flipped:
                        counter_sources.append(opp.source)

                if len(counter_sources) >= COUNTER_SIGNAL_MIN_SOURCES:
                    logging.info(
                        "[COUNTER-SIGNAL] trade #%d %s %s"
                        " — %d sources imply %s (edge>=%.1f): %s",
                        trade.trade_id, trade.side.upper(), trade.ticker,
                        len(counter_sources),
                        "NO" if trade.side == "yes" else "YES",
                        COUNTER_SIGNAL_MIN_EDGE,
                        ", ".join(counter_sources),
                    )
                    event = await self._execute_exit(
                        session, trade, pnl_cs, "counter_signal", "counter_signal"
                    )
                    events.append(event)
                    exited_ids.add(trade.trade_id)

            if trade.trade_id in exited_ids:
                continue  # already handled above

            # ---- KXLOWT between YES contra-signal check ----------------------
            # When 2+ forecast models update to predict fc_low < strike_lo
            # (temperature will fall BELOW the band floor), exit before the
            # market reprices.  Uses a tighter edge threshold than the global
            # counter-signal because a 1°F-wide between band makes even a 2°F
            # shift below the floor definitive.
            if (
                KXLOWT_CONTRA_MIN_SOURCES > 0
                and "KXLOWT" in trade.ticker
                and trade.side == "yes"
                and profit_pct <= COUNTER_SIGNAL_MAX_PROFIT_PCT
            ):
                try:
                    _note = json.loads(
                        getattr(trade, "note", None) or "{}"
                    )
                    _is_between = _note.get("direction") == "between"
                except (json.JSONDecodeError, TypeError):
                    _is_between = False

                if _is_between:
                    _contra_sources: list[str] = []
                    for opp in numeric_lookup.get(trade.ticker, []):
                        if opp.source not in _KXLOWT_CONTRA_FORECAST_SOURCES:
                            continue
                        if opp.implied_outcome != "NO":
                            continue
                        if opp.edge < KXLOWT_CONTRA_MIN_EDGE_F:
                            continue
                        _contra_sources.append(opp.source)

                    _contra_distinct = set(_contra_sources)
                    if len(_contra_distinct) >= KXLOWT_CONTRA_MIN_SOURCES:
                        logging.info(
                            "[KXLOWT-CONTRA] trade #%d YES %s"
                            " — %d forecast sources now predict NO"
                            " (edge>=%.1f°F): %s",
                            trade.trade_id, trade.ticker,
                            len(_contra_distinct),
                            KXLOWT_CONTRA_MIN_EDGE_F,
                            ", ".join(sorted(_contra_distinct)),
                        )
                        _pnl_contra = trade._unrealized_cents()
                        event = await self._execute_exit(
                            session, trade, _pnl_contra, "data_contra", "data_contra"
                        )
                        events.append(event)
                        exited_ids.add(trade.trade_id)

            if trade.trade_id in exited_ids:
                continue

            # ---- Poly counter-signal check -----------------------------------
            poly_opp = poly_lookup.get(trade.ticker)
            if poly_opp is None:
                continue
            if poly_opp.divergence < COUNTER_SIGNAL_MIN_EDGE:
                continue
            flipped = (
                (trade.side == "yes" and poly_opp.implied_side == "no") or
                (trade.side == "no"  and poly_opp.implied_side == "yes")
            )
            if not flipped:
                continue

            pnl = trade._unrealized_cents()
            logging.info(
                "[COUNTER-SIGNAL/poly] trade #%d %s %s"
                " — %s now implies %s  (div=%.1f%% >= min=%.1f%%)",
                trade.trade_id, trade.side.upper(), trade.ticker,
                poly_opp.source, poly_opp.implied_side.upper(),
                poly_opp.divergence * 100, COUNTER_SIGNAL_MIN_EDGE * 100,
            )
            event = await self._execute_exit(
                session, trade, pnl, "counter_signal", "counter_signal:poly"
            )
            events.append(event)
            exited_ids.add(trade.trade_id)

        if events:
            logging.info(
                "ExitManager: %d counter-signal exit(s) triggered this cycle.",
                len(events),
            )
        return events

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------

    @staticmethod
    def _resolve_profit_take(src: str, side: str, entry_cost: int = 100) -> float:
        """Return the profit-take threshold for a given source, side, and entry cost.

        Lookup order:
          1. Composite key ``"source:side"`` in EXIT_SOURCE_PROFIT_TAKE
          2. Bare ``"source"`` key in EXIT_SOURCE_PROFIT_TAKE
          3. Global EXIT_PROFIT_TAKE fallback

        Longshot multiplier: when ``entry_cost`` (cost per contract in cents)
        is ≤ EXIT_PROFIT_TAKE_LONGSHOT_CENTS, the resolved threshold is
        multiplied by EXIT_PROFIT_TAKE_LONGSHOT_MULT to prevent exiting a
        3¢ contract at 5¢ on its way to 100¢.
        """
        composite = f"{src}:{side}"
        if composite in EXIT_SOURCE_PROFIT_TAKE:
            base = EXIT_SOURCE_PROFIT_TAKE[composite]
        else:
            base = EXIT_SOURCE_PROFIT_TAKE.get(src, EXIT_PROFIT_TAKE)

        if (
            EXIT_PROFIT_TAKE_LONGSHOT_CENTS > 0
            and entry_cost <= EXIT_PROFIT_TAKE_LONGSHOT_CENTS
        ):
            base *= EXIT_PROFIT_TAKE_LONGSHOT_MULT
            logging.debug(
                "Longshot profit-take multiplier %.1f× applied"
                " (entry_cost=%d¢ ≤ %d¢) → threshold=%.1f",
                EXIT_PROFIT_TAKE_LONGSHOT_MULT,
                entry_cost,
                EXIT_PROFIT_TAKE_LONGSHOT_CENTS,
                base,
            )
        return base

    @staticmethod
    def _profit_take_detail(src: str, side: str, entry_cost: int = 100) -> str:
        """Return exit_reason_detail string for a profit-take exit."""
        composite = f"{src}:{side}"
        if composite in EXIT_SOURCE_PROFIT_TAKE or src in EXIT_SOURCE_PROFIT_TAKE:
            return "profit_take:source"
        if EXIT_PROFIT_TAKE_LONGSHOT_CENTS > 0 and entry_cost <= EXIT_PROFIT_TAKE_LONGSHOT_CENTS:
            return "profit_take:longshot"
        return "profit_take:global"

    def _get_peak_pct_gain(self, trade_id: int) -> float | None:
        """Return the highest pct_gain ever recorded for this trade in price_snapshots.

        Returns ``None`` if no snapshots exist yet (position opened this cycle).
        """
        row = self._conn.execute(
            "SELECT MAX(pct_gain) FROM price_snapshots WHERE trade_id = ?",
            (trade_id,),
        ).fetchone()
        return row[0] if row and row[0] is not None else None

    def _get_peak_at(self, trade_id: int) -> str | None:
        """Return the snapshot_at timestamp when pct_gain was highest for this trade."""
        row = self._conn.execute(
            """
            SELECT snapshot_at FROM price_snapshots
            WHERE trade_id = ? AND post_exit = 0
            ORDER BY pct_gain DESC LIMIT 1
            """,
            (trade_id,),
        ).fetchone()
        return row[0] if row else None

    @staticmethod
    def _is_locked_signal(src: str, side: str, ticker: str, peak_past: bool = False) -> bool:
        """Return True when this position's underlying observation cannot physically reverse.

        Used to suppress stop-loss near market close.  A signal is "locked"
        when the physical measurement that drove the entry is already recorded
        and cannot change — stopping out on a transient liquidity spike then
        destroys value when the market settles as originally predicted.

        Locked cases:
          band_arb:no    — METAR observed daily max crossed the KXHIGH band ceiling.
                           The sensor reading is recorded; it cannot be un-crossed.
          noaa_observed:no on KXLOWT* — the daily minimum temperature cannot rise
                           within a calendar day (by definition of "minimum").
                           The signal can only get stronger, never weaker.
          noaa_observed:no on KXHIGH* with peak_past=True — the daily max has
                           already been observed (after 4:30 PM local); the high
                           cannot rise further.  Equivalent lock to KXLOWT:no.
                           peak_past is persisted in the trades table at entry.

        Not locked (deliberately excluded):
          noaa_observed:no on KXHIGH* without peak_past — morning entries where the
                           daily max has not yet been recorded (temp can still rise).
          Forecast sources — model projections can be revised by new observations.
        """
        if side != "no":
            return False
        if src == "band_arb":
            return True
        if src == "noaa_observed":
            if "KXLOWT" in ticker or "KXLOW" in ticker:
                return True   # daily min cannot rise — always locked
            if "KXHIGH" in ticker and peak_past:
                return True   # daily max already peaked — locked NO
        return False

    async def check_contra_exits(
        self,
        session:      aiohttp.ClientSession,
        trades:       "list[_Trade]",
        numeric_opps: list,
    ) -> list["ExitEvent"]:
        """Exit noaa_day2:yes between positions when observational consensus flips to NO.

        Called each poll cycle before check_exits().  For every open noaa_day2:yes trade
        on a between market, count how many independent real-time sources (noaa_observed,
        metar, hrrr, open_meteo, nws_hourly) currently predict NO with edge ≥
        CONTRA_SIGNAL_MIN_EDGE_F.  If the count reaches CONTRA_SIGNAL_MIN_SOURCES,
        force-exit immediately — the data that motivated entry has been contradicted.

        This is intentionally asymmetric: entry required noaa_day2 + 1 corroborator;
        exit requires 2 real-time contra sources.  A single dissenting model is noise.
        """
        _CONTRA_SOURCES: frozenset[str] = frozenset({
            "noaa_observed", "metar", "hrrr", "open_meteo", "nws_hourly",
        })
        exited_ids: set[int] = {
            row[0]
            for row in self._conn.execute(
                "SELECT id FROM trades WHERE exited_at IS NOT NULL"
            ).fetchall()
        }
        events: list[ExitEvent] = []
        for trade in trades:
            if trade.settled or trade.current_mid is None:
                continue
            if trade.trade_id in exited_ids:
                continue
            if trade.source != "noaa_day2" or trade.side != "yes":
                continue
            # Only between markets (YES = high inside band).
            try:
                note = json.loads(trade.note) if trade.note else {}
            except (ValueError, TypeError):
                note = {}
            if note.get("direction") != "between":
                continue
            ticker = trade.ticker
            contra = [
                o for o in numeric_opps
                if getattr(o, "market_ticker", None) == ticker
                and getattr(o, "implied_outcome", None) == "NO"
                and getattr(o, "source", None) in _CONTRA_SOURCES
                and getattr(o, "edge", 0.0) >= CONTRA_SIGNAL_MIN_EDGE_F
            ]
            contra_sources = {o.source for o in contra}
            if len(contra_sources) >= CONTRA_SIGNAL_MIN_SOURCES:
                logging.info(
                    "Contra-signal exit: noaa_day2:yes %s — %d sources now predict NO"
                    " (%s); exiting position.",
                    ticker, len(contra_sources), ", ".join(sorted(contra_sources)),
                )
                ev = await self.force_exit(session, trade, reason="data_contra", detail="data_contra")
                if ev is not None:
                    events.append(ev)
        return events

    async def force_exit(
        self,
        session: aiohttp.ClientSession,
        trade:   "_Trade",
        reason:  str = "capital_recycle",
        detail:  str | None = None,
    ) -> "ExitEvent | None":
        """Force-exit a specific trade immediately, bypassing threshold checks.

        Used by capital recycling: when a new trade is blocked by the exposure cap,
        near-settled positions are liquidated to free capital.
        Returns None if the trade is already exited or has no current price.
        """
        if getattr(trade, "exited_at", None) is not None:
            return None
        if getattr(trade, "current_mid", None) is None:
            return None
        pnl = trade._unrealized_cents()
        return await self._execute_exit(session, trade, pnl, reason, detail or reason)

    async def _execute_exit(
        self,
        session: aiohttp.ClientSession,
        trade:   "_Trade",
        pnl:     float,
        reason:  str,
        detail:  str | None = None,
    ) -> ExitEvent:
        """Record the exit in the DB and (in live mode) place a sell order."""
        exit_price = int(trade.current_mid)
        now = datetime.now(timezone.utc).isoformat()
        cost = trade.total_cost_cents
        pct = pnl / cost * 100
        order_id: str | None = None

        if self._dry_run:
            logging.info(
                "[EXIT DRY-RUN] %s  trade #%d  %s %s x%d"
                "  entry=%d¢  exit=%d¢  pnl=%+.0f¢ (%+.0f%%)",
                {"profit_take": "PROFIT-TAKE", "trailing_stop": "TRAILING-STOP"}.get(reason, "STOP-LOSS"),
                trade.trade_id,
                trade.side.upper(),
                trade.ticker,
                trade.count,
                trade.limit_price,
                exit_price,
                pnl,
                pct,
            )
        else:
            order_id = await self._place_sell_order(
                session, trade.ticker, trade.side, trade.count, exit_price
            )

        peak_pct = self._get_peak_pct_gain(trade.trade_id)
        peak_at  = self._get_peak_at(trade.trade_id)

        self._conn.execute(
            """
            UPDATE trades
            SET exited_at          = ?,
                exit_price_cents   = ?,
                exit_pnl_cents     = ?,
                exit_reason        = ?,
                exit_order_id      = ?,
                exit_reason_detail = ?,
                peak_pct_gain      = ?,
                peak_at            = ?,
                exit_yes_bid       = ?,
                exit_yes_ask       = ?
            WHERE id = ?
            """,
            (
                now, exit_price, pnl, reason, order_id,
                detail, peak_pct, peak_at,
                getattr(trade, "yes_bid", None),
                getattr(trade, "yes_ask", None),
                trade.trade_id,
            ),
        )

        return ExitEvent(
            trade_id   = trade.trade_id,
            ticker     = trade.ticker,
            side       = trade.side,
            count      = trade.count,
            entry      = trade.limit_price,
            exit_price = exit_price,
            pnl_cents  = pnl,
            reason     = reason,
        )

    async def _place_sell_order(
        self,
        session:   aiohttp.ClientSession,
        ticker:    str,
        side:      str,
        count:     int,
        yes_price: int,
    ) -> str | None:
        """POST a limit sell order to the Kalshi API.

        For a YES position, ``side='yes'`` and ``yes_price`` is the bid we're
        willing to accept.  For a NO position, ``side='no'`` and ``yes_price``
        is still the yes_price field (Kalshi always uses yes_price in the
        order body regardless of which side you're on).
        """
        base = (
            "https://api.elections.kalshi.com"
            if os.environ.get("KALSHI_ENVIRONMENT", "demo") == "production"
            else "https://demo-api.kalshi.co"
        )
        headers = generate_headers("POST", _ORDERS_PATH)
        body = {
            "ticker":           ticker,
            "client_order_id":  str(uuid.uuid4()),
            "action":           "sell",
            "type":             "limit",
            "side":             side,
            "count":            count,
            "yes_price":        yes_price,
        }

        try:
            async with session.post(
                f"{base}{_ORDERS_PATH}",
                json=body,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                resp.raise_for_status()
                data   = await resp.json()
                order  = data.get("order", {})
                oid    = order.get("order_id") or order.get("id")
                status = order.get("status", "unknown")
                logging.info(
                    "[LIVE EXIT] Sell %s %s x%d @ %d¢  id=%s  status=%s",
                    side.upper(), ticker, count, yes_price, oid, status,
                )
                return oid
        except aiohttp.ClientResponseError as exc:
            logging.error(
                "Sell order failed for %s: HTTP %s: %s",
                ticker, exc.status, exc.message,
            )
        except aiohttp.ClientError as exc:
            logging.error("Sell order request error for %s: %s", ticker, exc)

        return None
