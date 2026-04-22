"""Live dry-run trading ledger.

After every poll cycle, rewrites ``dry_run_overview.txt`` with a complete
snapshot of all simulated trades, their current P&L, and a running balance
starting from a configurable amount of paper capital.

The file is a plain-text table you can open at any time (or ``tail -f`` if
you redirect bot output to a log file) to see exactly what the bot would have
done with real money.

P&L accounting
--------------
Each trade's outcome depends on its lifecycle state:

  EXITED   — early exit triggered by ExitManager (profit-take or stop-loss).
               P&L is locked at the exit price captured when the exit fired.
               exit_pnl_cents stored in the DB is used directly.

  SETTLED  — realized P&L computed from the API result field:
               YES buy:  gain = (100 − entry)¢ × count  if result=YES
                         loss = −entry¢ × count          if result=NO
               NO buy:   gain = (100 − no_cost)¢ × count  if result=NO
                         loss = −no_cost¢ × count          if result=YES
               (no_cost = 100 − limit_price, since limit_price stores yes_bid)

  OPEN     — unrealized mark-to-market using conservative exit-side price:
               YES buy:  unrealized = (yes_bid − entry)¢ × count
               NO buy:   unrealized = (entry − (100 − yes_ask))¢ × count

  UNKNOWN  — market data unavailable; shown as pending.

Balance
-------
  current_balance = starting_capital
                    + locked_gains  (settled wins + profitable exits)
                    − locked_losses (settled losses + stop-loss exits)
                    + unrealized_pnl  (mark-to-market on purely open trades)

Output file
-----------
  dry_run_overview.txt  in the project root (overwritten every cycle).
  Configurable via DRY_RUN_OVERVIEW_PATH env var.

Price snapshot table
--------------------
Every poll cycle, a row is written to ``price_snapshots`` for each unsettled
trade — including positions already exited via stop-loss or profit-take.
Post-exit rows have ``post_exit=1`` and represent counterfactual price history
(what the market did after we closed).  Schema:

  trade_id         → foreign key into ``trades``
  snapshot_at      → UTC ISO-8601 timestamp
  yes_bid          → Kalshi yes_bid in cents
  yes_ask          → Kalshi yes_ask in cents
  exit_price       → conservative exit price (yes_bid for YES positions,
                     100 − yes_ask for NO positions)
  unrealized_cents → (exit_price − entry) × count  (signed)
  pct_gain         → unrealized_cents / total_cost_cents  (signed ratio)
  days_to_close    → fractional days until market closes at snapshot time

This produces a time-series of price trajectories per position.  Combined
with exit event data in the ``trades`` table, it enables queries such as:

  • At what pct_gain level did each position peak?
  • How quickly did price decay after peak?
  • Did the exit fire before or after the peak?
  • Would a different threshold have outperformed?
"""

import logging
import math
import os
import sqlite3
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp

from .exit_manager import ExitManager, CAPITAL_RECYCLE_SOURCES, CAPITAL_RECYCLE_MIN_NO_VALUE
from .markets import fetch_market_detail

_DEFAULT_DB_PATH       = Path(__file__).parent.parent / "opportunity_log.db"
_DEFAULT_OVERVIEW_PATH = Path(__file__).parent.parent / "dry_run_overview.txt"

# Starting paper capital in cents (default $100.00 = 10_000¢).
STARTING_CAPITAL_CENTS: int = int(
    float(os.environ.get("DRY_RUN_STARTING_CAPITAL", "100")) * 100
)
OVERVIEW_PATH: Path = Path(
    os.environ.get("DRY_RUN_OVERVIEW_PATH", str(_DEFAULT_OVERVIEW_PATH))
)

from .trade_executor import (
    TRADE_DRY_RUN,
    DRAWDOWN_FULL_REDUCE_PCT,
    DRAWDOWN_MIN_FACTOR,
    DRAWDOWN_ENABLED,
    DRAWDOWN_LOOKBACK_TRADES,
    DRAWDOWN_IGNORE_BEFORE_ID,
)


# ---------------------------------------------------------------------------
# Internal trade record
# ---------------------------------------------------------------------------

class _Trade:
    """Single dry-run trade entry with P&L attached after enrichment."""

    __slots__ = (
        "trade_id", "logged_at", "ticker", "side", "count",
        "limit_price", "score", "kelly_fraction", "p_estimate",
        "opportunity_kind", "source", "note",
        # exit state (loaded from DB)
        "exited", "exit_pnl", "exit_reason",
        # settlement state (loaded from DB outcome / live API)
        "settled", "result",
        # live market data (populated by _enrich)
        "current_mid", "yes_bid", "yes_ask", "close_time",
        # market state at entry (for calibration)
        "market_p_entry", "yes_bid_entry", "yes_ask_entry",
        # signal lock flag: True if trade entered after daily peak confirmed
        # (peak_past=True on NumericOpportunity).  Used by exit_manager to
        # suppress stop-loss on post-peak KXHIGH:no positions near close.
        "peak_past",
        # spread identifier: non-empty for arb spread legs.  Used by
        # exit_manager to suppress stop-loss on arb positions that must be
        # held to settlement for the spread to converge.
        "spread_id",
    )

    def __init__(
        self,
        trade_id:         int,
        logged_at:        str,
        ticker:           str,
        side:             str,
        count:            int,
        limit_price:      int,
        score:            float,
        kelly_fraction:   float | None,
        p_estimate:       float | None,
        outcome:          str | None = None,
        exited_at:        str | None = None,
        exit_pnl_cents:   float | None = None,
        exit_reason:      str | None = None,
        source:           str | None = None,
        note:             str | None = None,
        opportunity_kind: str | None = None,
        market_p_entry:   float | None = None,
        yes_bid_entry:    int   | None = None,
        yes_ask_entry:    int   | None = None,
        peak_past:        bool  | None = None,
        spread_id:        str   | None = None,
    ) -> None:
        self.trade_id       = trade_id
        self.logged_at      = logged_at
        self.ticker         = ticker
        self.side           = side          # "yes" | "no"
        self.count          = count
        self.limit_price    = limit_price   # yes_price in cents
        self.score          = score
        self.kelly_fraction = kelly_fraction
        self.p_estimate     = p_estimate
        self.source           = source or ""
        self.note             = note or ""
        self.opportunity_kind = opportunity_kind or ""
        self.market_p_entry   = market_p_entry
        self.yes_bid_entry    = yes_bid_entry
        self.yes_ask_entry    = yes_ask_entry
        self.peak_past        = bool(peak_past)
        self.spread_id        = spread_id or ""

        # Exit state — locked-in P&L from early exit.
        self.exited      = exited_at is not None
        self.exit_pnl    = exit_pnl_cents   # signed cents, or None
        self.exit_reason = exit_reason      # 'profit_take' | 'stop_loss' | None

        # Settlement state — pre-fill from DB outcome if already settled.
        if outcome in ("won", "lost"):
            self.settled = True
            self.result  = side if outcome == "won" else ("no" if side == "yes" else "yes")
        else:
            self.settled = False
            self.result  = None

        # Live market data populated by _enrich().
        self.current_mid: float | None = None
        self.yes_bid:     int | None   = None
        self.yes_ask:     int | None   = None
        self.close_time:  str | None   = None

    # -----------------------------------------------------------------------
    # Derived values
    # -----------------------------------------------------------------------

    @property
    def cost_per_contract(self) -> int:
        """Amount paid per contract in cents."""
        return self.limit_price if self.side == "yes" else (100 - self.limit_price)

    @property
    def total_cost_cents(self) -> int:
        return self.cost_per_contract * self.count

    @property
    def pnl_cents(self) -> float | None:
        """Signed P&L in cents.  Priority: exited > settled > unrealized > None."""
        if self.exited:
            return self.exit_pnl            # locked exit P&L
        if self.settled and self.result is not None:
            return self._realized_cents()   # settlement outcome
        if self.current_mid is not None:
            return self._unrealized_cents() # mark-to-market
        return None

    @property
    def status_label(self) -> str:
        if self.exited:
            tag = "PROFIT-TAKE" if self.exit_reason == "profit_take" else "STOP-LOSS"
            gain = (self.exit_pnl or 0) > 0
            return f"EXITED {tag} ({'GAIN' if gain else 'LOSS'})"
        if self.settled and self.result is not None:
            win = (
                (self.side == "yes" and self.result == "yes") or
                (self.side == "no"  and self.result == "no")
            )
            return f"SETTLED {'WIN ' if win else 'LOSS'}"
        if self.current_mid is not None:
            return "OPEN"
        return "PENDING"

    def _realized_cents(self) -> float:
        if self.side == "yes":
            per = (100 - self.limit_price) if self.result == "yes" else (-self.limit_price)
        else:
            per = (self.limit_price) if self.result == "no" else (self.limit_price - 100)
        return per * self.count

    def _unrealized_cents(self) -> float:
        mid = self.current_mid  # caller must check not None
        if self.side == "yes":
            # YES: current bid minus entry YES price
            per = mid - self.limit_price
        else:
            # NO: current NO exit price (= 100 - yes_ask) minus entry NO cost (= 100 - limit_price)
            per = mid - (100 - self.limit_price)
        return per * self.count


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------

_CREATE_SNAPSHOTS_SQL = """
CREATE TABLE IF NOT EXISTS price_snapshots (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id         INTEGER NOT NULL,
    snapshot_at      TEXT    NOT NULL,
    yes_bid          INTEGER,
    yes_ask          INTEGER,
    exit_price       INTEGER,
    unrealized_cents REAL,
    pct_gain         REAL,
    days_to_close    REAL,
    post_exit        INTEGER NOT NULL DEFAULT 0   -- 1 = trade was already exited; counterfactual row
)
"""

_CREATE_SNAPSHOTS_IDX_SQL = """
CREATE INDEX IF NOT EXISTS idx_snapshots_trade_time
    ON price_snapshots (trade_id, snapshot_at)
"""

# Columns added to price_snapshots after initial release.
_SNAPSHOT_MIGRATIONS: list[tuple[str, str]] = [
    ("pct_gain",      "REAL"),
    ("days_to_close", "REAL"),
    ("post_exit",     "INTEGER NOT NULL DEFAULT 0"),
]


class DryRunLedger:
    """Reads dry-run trades from SQLite, enriches with live market data,
    checks exit thresholds, and writes a human-readable overview file.

    Also persists a price_snapshots row for every open position on each cycle,
    building the time-series dataset needed to evaluate exit strategy
    performance.

    Usage::

        ledger = DryRunLedger()
        await ledger.refresh_and_write(session)
        ledger.close()
    """

    def __init__(
        self,
        db_path: Path | str = _DEFAULT_DB_PATH,
        overview_path: Path | str = OVERVIEW_PATH,
        starting_capital_cents: int = STARTING_CAPITAL_CENTS,
    ) -> None:
        self._db_path = Path(db_path)
        self._overview_path = Path(overview_path)
        self._starting_capital = starting_capital_cents
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            isolation_level=None,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_SNAPSHOTS_SQL)
        self._conn.execute(_CREATE_SNAPSHOTS_IDX_SQL)
        self._migrate_snapshots()

        # ExitManager shares our connection so all writes land in the same DB.
        self._exit_manager = ExitManager(self._conn, dry_run=TRADE_DRY_RUN)

        # Cache of enriched open trades from the last refresh cycle.
        # Populated by refresh_and_write(); read by recyclable_trades().
        self._current_open_trades: list[_Trade] = []

        logging.info(
            "DryRunLedger — overview=%s  starting_capital=$%.2f",
            self._overview_path, starting_capital_cents / 100,
        )

    def _migrate_snapshots(self) -> None:
        """Add new columns to price_snapshots and trades if they are missing."""
        snap_existing = {
            row[1]
            for row in self._conn.execute(
                "PRAGMA table_info(price_snapshots)"
            ).fetchall()
        }
        for col, defn in _SNAPSHOT_MIGRATIONS:
            if col not in snap_existing:
                self._conn.execute(
                    f"ALTER TABLE price_snapshots ADD COLUMN {col} {defn}"
                )

        # Columns that may be missing from the trades table (added over time).
        trade_existing = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(trades)").fetchall()
        }
        for col, defn in [("note", "TEXT")]:
            if col not in trade_existing:
                self._conn.execute(f"ALTER TABLE trades ADD COLUMN {col} {defn}")

    async def refresh_and_write(
        self,
        session: aiohttp.ClientSession,
        *,
        numeric_opps: list | None = None,
        poly_opps: list | None = None,
    ) -> None:
        """Load all trades, enrich with market data, check exits, write overview.

        Args:
            session:      Shared aiohttp session.
            numeric_opps: Current cycle's numeric opportunities (post quality-gate),
                          passed to ExitManager for counter-signal exit checking.
                          None means counter-signal check is skipped this call.
            poly_opps:    Current cycle's external-forecast opportunities,
                          also used for counter-signal exits.
        """
        trades = self._load_trades()
        if not trades:
            # No trades yet — still write a minimal overview so the file exists.
            overview = self._build_overview([])
            self._overview_path.write_text(overview + "\n", encoding="utf-8")
            logging.info(
                "DryRunLedger: no trades yet — wrote empty overview to %s", self._overview_path
            )
            return

        try:
            await self._enrich(session, trades)
            await self._exit_manager.check_exits(session, trades)

            if numeric_opps is not None or poly_opps is not None:
                await self._exit_manager.check_counter_signals(
                    session,
                    trades,
                    numeric_opps=numeric_opps or [],
                    poly_opps=poly_opps or [],
                )

            self._reload_exit_state(trades)
            self._write_snapshots(trades)
            overview = self._build_overview(trades)
            self._overview_path.write_text(overview + "\n", encoding="utf-8")
            logging.info(
                "DryRunLedger: wrote %d bytes to %s", len(overview), self._overview_path
            )

            # Update cache for capital recycling (must be after check_exits so
            # any positions exited this cycle are excluded).
            self._current_open_trades = [
                t for t in trades
                if not t.exited and t.current_mid is not None
            ]
        except Exception as exc:
            logging.error("DryRunLedger: refresh_and_write failed: %s", exc, exc_info=True)

    def recyclable_trades(self) -> "list[_Trade]":
        """Return open positions eligible for forced capital recycling.

        Filters to high-confidence sources (CAPITAL_RECYCLE_SOURCES) whose
        current NO value has reached CAPITAL_RECYCLE_MIN_NO_VALUE — meaning the
        market has essentially priced in settlement already.  Sorted by NO value
        descending so greedy selection exits the most settled positions first.

        Returns an empty list when recycling is disabled (CAPITAL_RECYCLE_MIN_NO_VALUE=0)
        or no enriched trades are cached yet (before the first refresh cycle).
        """
        if CAPITAL_RECYCLE_MIN_NO_VALUE <= 0:
            return []
        candidates = [
            t for t in self._current_open_trades
            if (
                t.source in CAPITAL_RECYCLE_SOURCES
                and t.side == "no"
                and t.current_mid is not None
                and t.current_mid >= CAPITAL_RECYCLE_MIN_NO_VALUE
            )
        ]
        candidates.sort(key=lambda t: t.current_mid, reverse=True)  # type: ignore[arg-type]
        return candidates

    def available_cash_cents(self) -> int:
        """Return available paper cash = starting capital + realized P&L − open exposure.

        Computed purely from the DB (no market prices needed) so it can be
        called cheaply before every trade.  Unrealized P&L is excluded — only
        settled/exited results count as real cash.
        """
        # Realized P&L: early exits (exited_at IS NOT NULL) + Kalshi settlements
        # (outcome IS NOT NULL but no early exit).  Both return capital to the pool.
        row = self._conn.execute(
            """
            SELECT
                COALESCE(SUM(exit_pnl_cents), 0) AS realized_pnl
            FROM trades
            WHERE status NOT IN ('rejected', 'error')
              AND (exited_at IS NOT NULL OR outcome IS NOT NULL)
            """
        ).fetchone()
        realized_cents = int(row[0]) if row else 0

        row2 = self._conn.execute(
            """
            SELECT COALESCE(SUM(
                count * CASE WHEN side = 'yes' THEN limit_price
                             ELSE 100 - limit_price END
            ), 0)
            FROM trades
            WHERE exited_at IS NULL
              AND outcome IS NULL
              AND status NOT IN ('rejected', 'error')
            """
        ).fetchone()
        open_exposure = int(row2[0]) if row2 else 0

        return self._starting_capital + realized_cents - open_exposure

    def close(self) -> None:
        self._conn.close()

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def open_tickers(self) -> set[str]:
        """Return the set of tickers with currently open (unsettled, unexited) dry-run positions."""
        try:
            rows = self._conn.execute(
                """
                SELECT DISTINCT ticker FROM trades
                WHERE mode = 'dry_run'
                  AND outcome IS NULL
                  AND exited_at IS NULL
                """
            ).fetchall()
            return {row[0] for row in rows}
        except Exception:
            return set()

    def _compute_risk_metrics(self, trades: list["_Trade"]) -> dict:
        """Compute risk-adjusted return metrics over resolved (settled + exited) trades.

        Returns a dict with keys: n, sharpe, sortino, max_dd, current_dd,
        win_rate, wins, losses, avg_gain, avg_loss, profit_factor.
        Returns empty dict if fewer than 5 resolved trades exist.
        """
        resolved = [
            t for t in trades
            if (t.exited or (t.settled and t.result is not None))
            and t.total_cost_cents > 0
        ]
        if len(resolved) < 5:
            return {}

        pnl_list: list[float] = []
        returns:  list[float] = []
        for t in resolved:
            pnl = t.exit_pnl if t.exited else t._realized_cents()
            if pnl is None:
                continue
            pnl_list.append(pnl)
            returns.append(pnl / t.total_cost_cents)

        if len(returns) < 5:
            return {}

        # Equity curve and drawdown
        equity = float(self._starting_capital)
        peak   = equity
        max_dd = 0.0
        for pnl in pnl_list:
            equity += pnl
            peak    = max(peak, equity)
            if peak > 0:
                max_dd = max(max_dd, (peak - equity) / peak)
        current_dd = max(0.0, (peak - equity) / peak) if peak > 0 else 0.0

        # Annualisation factor based on actual date range
        n = len(returns)
        try:
            t0 = datetime.fromisoformat(resolved[0].logged_at.replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(resolved[-1].logged_at.replace("Z", "+00:00"))
            years = max((t1 - t0).total_seconds() / 31_557_600, 1 / 365)
            tpy   = n / years
        except (ValueError, AttributeError):
            tpy = 365.0

        mean_r = statistics.mean(returns)
        std_r  = statistics.pstdev(returns)
        sharpe = (mean_r / std_r * math.sqrt(tpy)) if std_r > 0 else None

        neg     = [r for r in returns if r < 0]
        dn_std  = statistics.pstdev(neg) if len(neg) >= 2 else 0.0
        sortino = (mean_r / dn_std * math.sqrt(tpy)) if dn_std > 0 else None

        wins = [p for p in pnl_list if p > 0]
        loss = [p for p in pnl_list if p <= 0]
        gain_sum = sum(wins)
        loss_sum = abs(sum(loss))
        pf = gain_sum / loss_sum if loss_sum > 0 else None

        return {
            "n":             n,
            "sharpe":        sharpe,
            "sortino":       sortino,
            "max_dd":        max_dd,
            "current_dd":    current_dd,
            "win_rate":      len(wins) / n,
            "wins":          len(wins),
            "losses":        len(loss),
            "avg_gain":      statistics.mean(wins) if wins else 0.0,
            "avg_loss":      statistics.mean(loss) if loss else 0.0,
            "profit_factor": pf,
        }

    def current_drawdown_factor(self) -> float:
        """Return a position-size multiplier in [DRAWDOWN_MIN_FACTOR, 1.0].

        Reads resolved trades from the DB, builds the equity curve, and maps
        the current drawdown linearly to a sizing factor.  Returns 1.0 (no
        scaling) when drawdown scaling is disabled or fewer than 5 resolved
        trades exist.
        """
        if not DRAWDOWN_ENABLED:
            return 1.0
        try:
            limit_clause = (
                f"LIMIT {DRAWDOWN_LOOKBACK_TRADES}" if DRAWDOWN_LOOKBACK_TRADES > 0 else ""
            )
            id_clause = (
                f"AND id >= {DRAWDOWN_IGNORE_BEFORE_ID}" if DRAWDOWN_IGNORE_BEFORE_ID > 0 else ""
            )
            rows = self._conn.execute(
                f"""
                SELECT exit_pnl_cents, outcome, side, count, limit_price
                FROM trades
                WHERE mode = 'dry_run'
                  AND (outcome IN ('won', 'lost') OR exited_at IS NOT NULL)
                  AND outcome IS NOT 'void'
                  {id_clause}
                ORDER BY logged_at DESC
                {limit_clause}
                """
            ).fetchall()
            rows = list(reversed(rows))  # restore chronological order
        except Exception:
            return 1.0

        if len(rows) < 5:
            return 1.0

        equity = float(self._starting_capital)
        peak   = equity
        for exit_pnl, outcome, side, count, limit_price in rows:
            if exit_pnl is not None:
                pnl = float(exit_pnl)
            elif outcome == "won":
                pnl = float((100 - limit_price) * count if side == "yes" else limit_price * count)
            elif outcome == "lost":
                pnl = float(-limit_price * count if side == "yes" else -(100 - limit_price) * count)
            else:
                continue
            equity += pnl
            peak    = max(peak, equity)

        if peak <= 0:
            return DRAWDOWN_MIN_FACTOR

        current_dd = max(0.0, (peak - equity) / peak)
        if current_dd <= 0:
            return 1.0

        # Linear: 0% drawdown → 1.0, DRAWDOWN_FULL_REDUCE_PCT → DRAWDOWN_MIN_FACTOR
        t      = min(1.0, current_dd / DRAWDOWN_FULL_REDUCE_PCT)
        factor = 1.0 - t * (1.0 - DRAWDOWN_MIN_FACTOR)
        factor = max(DRAWDOWN_MIN_FACTOR, factor)
        logging.info(
            "Drawdown scaling: equity %.1f%% below peak → sizing at %.0f%% of normal",
            current_dd * 100, factor * 100,
        )
        return factor

    def source_performance_summary(self, n_recent: int = 20) -> dict[str, dict]:
        """Return per-source performance stats over the last n_recent resolved trades.

        Returns a dict keyed by source with:
          n            — total resolved trades for this source
          win_rate     — fraction of trades that were wins (pnl > 0)
          net_pnl_cents — sum of P&L in cents
        Only sources with ≥3 resolved trades are included.
        """
        try:
            rows = self._conn.execute(
                """
                SELECT COALESCE(source, 'unknown') as src,
                       side, "count", limit_price, outcome, exit_reason,
                       exit_pnl_cents
                FROM trades
                WHERE mode = 'dry_run'
                  AND (outcome IS NOT NULL OR exited_at IS NOT NULL)
                  AND outcome IS NOT 'void'
                ORDER BY logged_at ASC
                """
            ).fetchall()
        except Exception:
            return {}

        # Build per-source lists of (is_win, pnl_cents)
        from collections import defaultdict
        buckets: dict[str, list[tuple[bool, float]]] = defaultdict(list)
        for src, side, count, limit_price, outcome, exit_reason, exit_pnl in rows:
            if exit_pnl is not None:
                pnl = float(exit_pnl)
            elif outcome == "won":
                pnl = float((100 - limit_price) * count if side == "yes" else limit_price * count)
            elif outcome == "lost":
                pnl = float(-limit_price * count if side == "yes" else -(100 - limit_price) * count)
            else:
                continue
            buckets[src].append((pnl > 0, pnl))

        result: dict[str, dict] = {}
        for src, records in buckets.items():
            if len(records) < 3:
                continue
            recent = records[-n_recent:]
            wins = sum(1 for is_win, _ in recent if is_win)
            net_pnl = sum(pnl for _, pnl in recent)
            result[src] = {
                "n": len(records),
                "win_rate": wins / len(recent),
                "net_pnl_cents": net_pnl,
            }
        return result

    def recently_exited_tickers(self, cooldown_minutes: int) -> set[str]:
        """Return tickers exited via stop_loss or trailing_stop within cooldown_minutes.

        These tickers are temporarily blocked from re-entry to prevent cascading
        losses from repeatedly entering a position after the market has moved
        against us.  Profit-take exits are excluded — a market that hit its
        profit target may still have upside worth re-entering.
        """
        if cooldown_minutes <= 0:
            return set()
        try:
            rows = self._conn.execute(
                """
                SELECT DISTINCT ticker FROM trades
                WHERE mode = 'dry_run'
                  AND exit_reason IN ('stop_loss', 'trailing_stop')
                  AND datetime(exited_at) > datetime('now', ? || ' minutes')
                """,
                (f"-{cooldown_minutes}",),
            ).fetchall()
            return {row[0] for row in rows}
        except Exception:
            return set()

    def forecast_no_exited_today(self) -> set[str]:
        """Return tickers where a forecast_no position exited today (any exit reason).

        Used to block same-day re-entry after a forecast_no profit-take. Unlike
        recently_exited_tickers() (which only blocks stop-loss exits), this blocks
        ALL exits for forecast_no trades on the current calendar day — profit-takes
        included — because the opportunity has already been captured and re-entering
        at a higher price with less remaining upside degrades expected value.
        """
        try:
            rows = self._conn.execute(
                """
                SELECT DISTINCT ticker FROM trades
                WHERE mode = 'dry_run'
                  AND opportunity_kind = 'forecast_no'
                  AND exited_at IS NOT NULL
                  AND exited_at >= datetime('now', '-30 hours')
                """
            ).fetchall()
            return {row[0] for row in rows}
        except Exception:
            return set()

    def _load_trades(self) -> list[_Trade]:
        try:
            rows = self._conn.execute(
                """
                SELECT id, logged_at, ticker, side, count, limit_price,
                       score, kelly_fraction, p_estimate, outcome,
                       exited_at, exit_pnl_cents, exit_reason, source, note,
                       opportunity_kind, market_p_entry, yes_bid_entry, yes_ask_entry,
                       peak_past, spread_id
                FROM trades
                WHERE mode = 'dry_run'
                ORDER BY logged_at ASC
                """
            ).fetchall()
        except sqlite3.OperationalError as exc:
            logging.error("DryRunLedger: _load_trades failed: %s", exc, exc_info=True)
            return []

        return [
            _Trade(
                trade_id         = row[0],
                logged_at        = row[1],
                ticker           = row[2],
                side             = row[3],
                count            = row[4],
                limit_price      = row[5],
                score            = row[6],
                kelly_fraction   = row[7],
                p_estimate       = row[8],
                outcome          = row[9],
                exited_at        = row[10],
                exit_pnl_cents   = row[11],
                exit_reason      = row[12],
                source           = row[13],
                note             = row[14],
                opportunity_kind = row[15],
                market_p_entry   = row[16],
                yes_bid_entry    = row[17],
                yes_ask_entry    = row[18],
                peak_past        = bool(row[19]) if row[19] is not None else False,
                spread_id        = row[20] if len(row) > 20 else None,
            )
            for row in rows
        ]

    def _reload_exit_state(self, trades: list[_Trade]) -> None:
        """After ExitManager runs, refresh exit fields on any newly exited trades."""
        rows = self._conn.execute(
            "SELECT id, exited_at, exit_pnl_cents, exit_reason FROM trades WHERE exited_at IS NOT NULL"
        ).fetchall()
        exited = {row[0]: (row[1], row[2], row[3]) for row in rows}
        for t in trades:
            if not t.exited and t.trade_id in exited:
                _, pnl, reason = exited[t.trade_id]
                t.exited      = True
                t.exit_pnl    = pnl
                t.exit_reason = reason

    async def _enrich(
        self, session: aiohttp.ClientSession, trades: list[_Trade]
    ) -> None:
        """Fetch live market state for every open, unexited ticker."""
        import asyncio

        # Fetch live data for: (a) unsettled open positions, and (b) exited-but-unsettled
        # positions so we can continue logging counterfactual price snapshots after exit.
        active_tickers = list({
            t.ticker for t in trades
            if not t.settled
        })

        # Serial with 0.5s gap — avoids Kalshi 429s on large portfolios.
        sem = asyncio.Semaphore(1)

        async def _fetch(ticker: str):
            async with sem:
                result = await fetch_market_detail(session, ticker)
                await asyncio.sleep(0.5)
                return result

        results = await asyncio.gather(
            *[_fetch(t) for t in active_tickers],
            return_exceptions=True,
        )

        market_by_ticker: dict[str, dict[str, Any]] = {}
        for ticker, result in zip(active_tickers, results):
            if isinstance(result, dict):
                market_by_ticker[ticker] = result

        for trade in trades:
            if trade.settled:
                continue
            mkt = market_by_ticker.get(trade.ticker)
            if mkt is None:
                continue
            status       = mkt.get("status", "")
            result_field = mkt.get("result", "")
            if status in ("settled", "finalized") and result_field in ("yes", "no"):
                trade.settled = True
                trade.result  = result_field
            else:
                bid = mkt.get("yes_bid")
                ask = mkt.get("yes_ask")
                if bid is not None and ask is not None:
                    # bid=0 + ask=100 is the Kalshi API's post-close pending-settlement
                    # state — the market has stopped trading but hasn't resolved yet.
                    # Treat as no price available: skip current_mid so exit checks and
                    # snapshots are skipped this cycle rather than recording a phantom
                    # –100% unrealized loss.
                    if int(bid) == 0 and int(ask) == 100:
                        continue
                    trade.yes_bid = int(bid)
                    trade.yes_ask = int(ask)
                    if trade.side == "yes":
                        trade.current_mid = float(bid)
                    else:
                        trade.current_mid = float(100 - ask)
                # Capture close_time for days_to_close calculation in snapshots.
                ct = mkt.get("close_time") or mkt.get("expiration_time")
                if ct:
                    trade.close_time = ct

    def _write_snapshots(self, trades: list[_Trade]) -> None:
        """Persist a price snapshot row for every unsettled position.

        Columns written per row:
          - trade_id, snapshot_at, yes_bid, yes_ask, exit_price
          - unrealized_cents  — signed P&L at conservative exit price
          - pct_gain          — unrealized / total_cost  (ratio, e.g. 0.5 = +50%)
          - days_to_close     — fractional days until market close (None if unavailable)
          - post_exit         — 1 if the trade was already exited (counterfactual rows
                                that show what the price did AFTER our stop-loss or
                                profit-take, enabling exit-parameter backtesting)

        Exited-but-unsettled positions are included with post_exit=1 so the
        full price trajectory is available for backtesting.  Only fully
        settled positions are excluded.
        """
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()
        rows = []

        for t in trades:
            if t.settled or t.current_mid is None:
                continue

            exit_price = int(t.current_mid)
            unrealized = t._unrealized_cents()
            cost       = t.total_cost_cents
            pct_gain   = unrealized / cost if cost > 0 else None

            days_to_close: float | None = None
            if t.close_time:
                try:
                    ct = datetime.fromisoformat(
                        t.close_time.replace("Z", "+00:00")
                    )
                    days_to_close = (ct - now).total_seconds() / 86400
                except (ValueError, TypeError):
                    pass

            rows.append((
                t.trade_id, now_iso,
                t.yes_bid, t.yes_ask,
                exit_price, unrealized,
                pct_gain, days_to_close,
                1 if t.exited else 0,
            ))

        if not rows:
            return

        self._conn.executemany(
            """
            INSERT INTO price_snapshots
                (trade_id, snapshot_at, yes_bid, yes_ask, exit_price,
                 unrealized_cents, pct_gain, days_to_close, post_exit)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        n_post = sum(1 for r in rows if r[-1])
        logging.debug(
            "Price snapshots: wrote %d row(s) (%d post-exit counterfactual).",
            len(rows), n_post,
        )

    def _build_overview(self, trades: list[_Trade]) -> str:
        W = "=" * 68
        S = "-" * 68

        exited_trades  = [t for t in trades if t.exited]
        settled_trades = [t for t in trades if t.settled and t.result is not None and not t.exited]
        open_trades    = [t for t in trades if not t.settled and not t.exited]

        def _d(cents: float) -> str:
            sign = "+" if cents >= 0 else ""
            return f"{sign}${cents / 100:.2f}"

        # --- Realized P&L: settled (normal) + exited (early close) ---
        settled_gains  = sum(max(0.0, t._realized_cents()) for t in settled_trades)
        settled_losses = sum(min(0.0, t._realized_cents()) for t in settled_trades)
        exit_gains     = sum(max(0.0, t.exit_pnl) for t in exited_trades if t.exit_pnl is not None)
        exit_losses    = sum(min(0.0, t.exit_pnl) for t in exited_trades if t.exit_pnl is not None)

        realized_gains_cents  = settled_gains  + exit_gains
        realized_losses_cents = settled_losses + exit_losses

        unrealized_cents = sum(
            t._unrealized_cents()
            for t in open_trades
            if t.current_mid is not None
        )

        current_balance_cents = (
            self._starting_capital
            + realized_gains_cents
            + realized_losses_cents   # already negative
            + unrealized_cents
        )

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        buf: list[str] = [
            W,
            "  DRY-RUN LIVE TRADING OVERVIEW",
            f"  Last updated: {now}",
            W,
            f"  Starting capital :  ${self._starting_capital / 100:>8.2f}",
            f"  Realized gains   :  {_d(realized_gains_cents):>9}  "
            f"(settled wins: {_d(settled_gains)}  exits: {_d(exit_gains)})",
            f"  Realized losses  :  {_d(realized_losses_cents):>9}  "
            f"(settled losses: {_d(settled_losses)}  exits: {_d(exit_losses)})",
            f"  Unrealized P&L   :  {_d(unrealized_cents):>9}  (mark-to-market on open trades)",
            S,
            f"  Current balance  :  ${current_balance_cents / 100:>8.2f}",
            S,
            f"  Total trades     :  {len(trades)}",
            f"  Settled (normal) :  {len(settled_trades)}",
            f"  Exited (early)   :  {len(exited_trades)}",
            f"  Open / pending   :  {len(open_trades)}",
            W,
        ]

        rm = self._compute_risk_metrics(trades)
        if rm:
            def _pct(v: float) -> str: return f"{v * 100:.1f}%"
            def _opt(v: object, fmt: str) -> str: return fmt % v if v is not None else "n/a"
            buf.extend([
                "  RISK METRICS",
                S,
                f"  Resolved trades  :  {rm['n']}",
                f"  Sharpe ratio     :  {_opt(rm['sharpe'],  '%.2f')}  (annualized per-trade)",
                f"  Sortino ratio    :  {_opt(rm['sortino'], '%.2f')}",
                f"  Max drawdown     :  -{_pct(rm['max_dd'])}  (from equity peak)",
                f"  Current drawdown :  -{_pct(rm['current_dd'])}",
                f"  Win rate         :  {_pct(rm['win_rate'])}  ({rm['wins']}W / {rm['losses']}L)",
                f"  Avg gain         :  {_d(rm['avg_gain'])}",
                f"  Avg loss         :  {_d(rm['avg_loss'])}",
                f"  Profit factor    :  {_opt(rm['profit_factor'], '%.2f')}  (total gains / total losses)",
                W,
            ])

        buf.extend([
            "  TRADE HISTORY  (oldest → newest)",
            W,
        ])

        running_balance = self._starting_capital
        for t in trades:
            trade_label = str(t.trade_id)
            date_str  = t.logged_at[:16].replace("T", " ")
            side_str  = t.side.upper()
            profit_per = 100 - t.cost_per_contract
            cost_str  = f"paid {t.cost_per_contract}¢ × {t.count}"
            entry_str = f"(+{profit_per}¢ profit/ea)"

            pnl = t.pnl_cents
            if pnl is not None:
                pnl_tag = f"  P&L {_d(pnl)}"
                running_balance += pnl
            else:
                pnl_tag = "  P&L pending"

            bal_str = f"  balance ${running_balance / 100:.2f}"

            buf.append(
                f"  #{trade_label:<4} {date_str}  {side_str:<3}  {t.ticker:<32}"
                f"  {cost_str:<12}  {entry_str}"
            )
            buf.append(
                f"         score={t.score:.2f}  p={t.p_estimate or '?'}"
                f"  [{t.status_label}]{pnl_tag}{bal_str}"
            )
            if t.note:
                buf.append(f"         note: {t.note}")
            if t.exited:
                buf.append(
                    f"         → exited @ {int(t.exit_pnl / t.count + t.cost_per_contract) if t.exit_pnl is not None else '?'}¢"
                    f"  (held to expiry would be unknown until settlement)"
                )
            elif t.settled and t.result is not None:
                buf.append(f"         → resolved {t.result.upper()}")
            elif t.current_mid is not None:
                pct = (t._unrealized_cents() / t.total_cost_cents * 100) if t.total_cost_cents > 0 else 0
                buf.append(f"         → current exit {t.current_mid:.0f}¢  ({pct:+.0f}% on cost)")
            buf.append("")

        buf.extend([
            S,
            f"  Total realized gains  : {_d(realized_gains_cents)}",
            f"  Total realized losses : {_d(realized_losses_cents)}",
            f"  Net realized P&L      : {_d(realized_gains_cents + realized_losses_cents)}",
            W,
        ])

        return "\n".join(buf)
