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
  eia                0.95 — WTI oil contracts have extreme intraday volatility
  eia_inventory      0.95   but consistently converge to their true value by
                            settlement.  Only cut if position is nearly
                            worthless (95% loss = essentially terminal).
                            History: 3/3 stop-losses on eia_inventory were
                            premature — all settled as wins.  Raised from
                            0.50 to 0.95 to match eia behavior.

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
    ' "noaa:no": 0.08, "owm:no": 0.08, "open_meteo:no": 0.08,'
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
# eia                0.90 — WTI oil contracts trend toward their settlement
#                           value; both historical EIA profit-takes exited too
#                           early (left 52¢ and 58¢ on table).
# noaa / owm / open_meteo  0.50 — raw forecast: symmetric risk each way.
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
    ' "eia": 0.90, "eia_inventory": 0.90, "noaa_day2:no": 0.07, "noaa_day2_early:no": 0.07,'
    ' "noaa_day2:yes": 0.35, "noaa_day2_early:yes": 0.35,'
    ' "noaa": 0.40, "noaa_day2": 0.20, "polymarket": 0.25,'
    ' "binance": 0.35, "coinbase": 0.35}',
)
_sl_raw = os.environ.get(
    "EXIT_SOURCE_STOP_LOSS",
    '{"noaa_observed:yes": 0.15, "metar:yes": 0.05, "nws_climo:yes": 0.05, "nws_alert:yes": 0.05,'
    ' "noaa_observed": 0.70, "metar": 0.60, "noaa": 0.30, "owm": 0.50, "open_meteo": 0.50,'
    ' "polymarket": 0.20, "manifold": 0.40, "metaculus": 0.50, "eia": 0.95, "eia_inventory": 0.95,'
    ' "noaa_day2:yes": 0.45, "noaa_day2_early:yes": 0.45,'
    ' "noaa_day2:no": 0.55, "noaa_day2_early:no": 0.55,'
    ' "noaa_day2": 0.30, "hrrr": 0.40,'
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

# Minimum edge (numeric) or divergence (poly) for a counter-direction signal
# to trigger an early exit.  0 = disabled (default).  Once enabled, set this
# to at least the same threshold used for entry (NUMERIC_MIN_EDGE / analogues)
# so noise signals below that floor don't force premature exits.
# Example: COUNTER_SIGNAL_MIN_EDGE=1.0 requires the flipped signal to be at
# least 1 unit above the strike before exiting.
# Minimum hold time before any exit trigger (profit_take, stop_loss, trailing)
# fires on a newly-logged position.  Prevents the exit manager from stopping out
# a trade on the same cycle it was entered — which happens when the current
# yes_bid is below the limit_price (normal bid-ask spread) and the stop-loss
# threshold is tighter than the spread.
# Default: 2 minutes.  Set to 0 to disable.
EXIT_MIN_HOLD_MINUTES: float = float(
    os.environ.get("EXIT_MIN_HOLD_MINUTES", "2.0")
)

COUNTER_SIGNAL_MIN_EDGE: float = float(
    os.environ.get("COUNTER_SIGNAL_MIN_EDGE", "0")
)

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
        logging.info(
            "ExitManager — profit_take=%.0f%%  stop_loss=%.0f%%%s"
            "  source_overrides=%d  dry_run=%s",
            EXIT_PROFIT_TAKE * 100,
            EXIT_STOP_LOSS  * 100,
            trailing_str,
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
        from DryRunLedger._enrich() so no extra API calls are needed.

        Args:
            session: Shared aiohttp session (used only for live sell orders).
            trades:  Enriched _Trade objects — must have current_mid populated.

        Returns:
            List of ExitEvent for every exit triggered this cycle.
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

            # Numeric markets (EIA, Fed, CPI, crypto prices, etc.) resolve on a
            # single external data release.  Price moves before settlement are
            # noise — thin books, not new information.  Hold all numeric trades
            # to settlement; no stop_loss, profit_take, or trailing stop.
            if getattr(trade, "opportunity_kind", "") == "numeric":
                continue

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

            # Compute hours to close from the trade's close_time (populated by
            # _enrich).  Used for the time-gated trailing stop below.
            hours_to_close: float | None = None
            ct_str = getattr(trade, "close_time", None)
            if ct_str:
                try:
                    ct = datetime.fromisoformat(ct_str.replace("Z", "+00:00"))
                    hours_to_close = (ct - datetime.now(timezone.utc)).total_seconds() / 3600
                except (ValueError, TypeError):
                    pass

            reason: str | None = None
            if pct >= profit_take_thresh:
                reason = "profit_take"
            elif stop_loss_thresh > 0 and pct <= -stop_loss_thresh:
                reason = "stop_loss"
            elif EXIT_TRAILING_DRAWDOWN > 0 or (src in EXIT_SOURCE_TRAILING_DRAWDOWN or f"{src}:{side}" in EXIT_SOURCE_TRAILING_DRAWDOWN):
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
                        peak > drawdown_thresh   # was ever meaningfully up
                        and pct < peak - drawdown_thresh  # drawn back by threshold
                    ):
                        reason = "trailing_stop"

            if reason is None:
                continue

            event = await self._execute_exit(session, trade, pnl, reason)
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
            for opp in numeric_lookup.get(trade.ticker, []):
                if opp.edge < COUNTER_SIGNAL_MIN_EDGE:
                    continue
                flipped = (
                    (trade.side == "yes" and opp.implied_outcome == "NO") or
                    (trade.side == "no"  and opp.implied_outcome == "YES")
                )
                if not flipped:
                    continue

                pnl = trade._unrealized_cents()
                logging.info(
                    "[COUNTER-SIGNAL] trade #%d %s %s"
                    " — data now implies %s  (edge=%.3f >= min=%.3f)",
                    trade.trade_id, trade.side.upper(), trade.ticker,
                    opp.implied_outcome, opp.edge, COUNTER_SIGNAL_MIN_EDGE,
                )
                event = await self._execute_exit(
                    session, trade, pnl, "counter_signal"
                )
                events.append(event)
                exited_ids.add(trade.trade_id)
                break  # one exit per trade; stop checking further opps

            if trade.trade_id in exited_ids:
                continue  # already handled above

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
                session, trade, pnl, "counter_signal"
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

    def _get_peak_pct_gain(self, trade_id: int) -> float | None:
        """Return the highest pct_gain ever recorded for this trade in price_snapshots.

        Returns ``None`` if no snapshots exist yet (position opened this cycle).
        """
        row = self._conn.execute(
            "SELECT MAX(pct_gain) FROM price_snapshots WHERE trade_id = ?",
            (trade_id,),
        ).fetchone()
        return row[0] if row and row[0] is not None else None

    async def _execute_exit(
        self,
        session: aiohttp.ClientSession,
        trade:   "_Trade",
        pnl:     float,
        reason:  str,
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

        self._conn.execute(
            """
            UPDATE trades
            SET exited_at        = ?,
                exit_price_cents = ?,
                exit_pnl_cents   = ?,
                exit_reason      = ?,
                exit_order_id    = ?
            WHERE id = ?
            """,
            (now, exit_price, pnl, reason, order_id, trade.trade_id),
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
