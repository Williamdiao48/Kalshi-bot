"""Precedence tests for kalshi_bot.exit_manager.ExitManager.check_exits.

check_exits walks a documented "first match wins" ladder (exit_manager.py
docstring, lines ~709-724). The subtle, money-losing bugs in an exit engine
are ordering bugs: a stop-loss firing on a leg that must be held to complete a
hedge, a forced obs-confirmed exit being blocked by the min-hold gate, a
data-release position stopped out on intraday noise. These tests pin the
*ordering* of that ladder, not the many threshold-tuning knobs (those are
neutralized to fixed values by the fixture so only precedence is under test).

The ladder, top to bottom (each rung, when it matches, ends evaluation):
  1. settled / no live price / already-exited      -> skip
  2. force-exit (queued)                            -> exit, bypass ALL gates
  3. min-hold window not elapsed                    -> skip
  4. cost <= 0                                       -> skip
  5. numeric + hold-to-settlement source            -> skip (no stop-loss)
  6. spread leg (arb family)                        -> profit-take ONLY
  7..  stop-loss / profit-take / trailing            -> the normal exits

check_exits consumes pre-enriched _Trade objects (current_mid already set) and
does no network I/O in dry-run; ``session`` is only touched for live sell
orders, so we pass None. Async is driven with asyncio.run (no pytest-asyncio).
"""
import asyncio
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

import kalshi_bot.exit_manager as em
from kalshi_bot.dry_run_ledger import _Trade


def _run(coro):
    return asyncio.run(coro)


# The only trades-table columns check_exits / _execute_exit touch: id +
# exited_at (double-fire guard) and the exit_* columns the UPDATE writes.
# price_snapshots is read by _get_peak_pct_gain / _get_peak_at.
def _make_db(tmp_path):
    conn = sqlite3.connect(tmp_path / "exit_test.db")
    conn.execute(
        """
        CREATE TABLE trades (
            id                 INTEGER PRIMARY KEY,
            exited_at          TEXT,
            exit_price_cents   INTEGER,
            exit_pnl_cents     REAL,
            exit_reason        TEXT,
            exit_order_id      TEXT,
            exit_reason_detail TEXT,
            peak_pct_gain      REAL,
            peak_at            TEXT,
            exit_yes_bid       INTEGER,
            exit_yes_ask       INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE price_snapshots (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id    INTEGER NOT NULL,
            snapshot_at TEXT,
            pct_gain    REAL,
            post_exit   INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.commit()
    return conn


def _insert_row(conn, trade_id, exited_at=None):
    conn.execute("INSERT INTO trades (id, exited_at) VALUES (?, ?)", (trade_id, exited_at))
    conn.commit()


@pytest.fixture
def manager(tmp_path, monkeypatch):
    """An ExitManager wired to a temp DB with every threshold neutralized.

    All the tuning knobs are pinned so that ONLY the precedence ladder decides
    the outcome: global stop-loss at 70%, profit-take at 20%, and every
    source-specific override / near-close / floor / spread / trailing / longshot
    modifier disabled. Individual tests re-enable exactly the one gate they test.
    """
    # Global thresholds (fractions of entry cost).
    monkeypatch.setattr(em, "EXIT_STOP_LOSS", 0.70, raising=True)
    monkeypatch.setattr(em, "EXIT_PROFIT_TAKE", 0.20, raising=True)
    # Gates OFF by default; specific tests turn one on.
    monkeypatch.setattr(em, "EXIT_MIN_HOLD_MINUTES", 0.0, raising=True)
    monkeypatch.setattr(em, "EXIT_STOP_LOSS_NEARCLOSE_HOURS", 0.0, raising=True)
    monkeypatch.setattr(em, "EXIT_STOP_LOSS_FLOOR_PRICE", 0, raising=True)
    monkeypatch.setattr(em, "EXIT_STOP_LOSS_MAX_SPREAD", 0, raising=True)
    monkeypatch.setattr(em, "EXIT_TRAILING_DRAWDOWN", 0.0, raising=True)
    monkeypatch.setattr(em, "EXIT_PROFIT_TAKE_LONGSHOT_CENTS", 0, raising=True)
    monkeypatch.setattr(em, "FORECAST_NO_STOP_LOSS", 0.0, raising=True)
    monkeypatch.setattr(em, "BAND_ARB_NO_EXIT_PRICE_CENTS", 0, raising=True)
    monkeypatch.setattr(em, "BAND_ARB_YES_EXIT_PRICE_CENTS", 0, raising=True)
    monkeypatch.setattr(em, "FORECAST_BAND_YES_EXIT_PRICE_CENTS", 0, raising=True)
    # Per-source override dicts empty so nothing repaints the global thresholds.
    monkeypatch.setattr(em, "EXIT_SOURCE_STOP_LOSS", {}, raising=True)
    monkeypatch.setattr(em, "EXIT_SOURCE_PROFIT_TAKE", {}, raising=True)
    monkeypatch.setattr(em, "EXIT_SOURCE_TRAILING_DRAWDOWN", {}, raising=True)

    conn = _make_db(tmp_path)
    mgr = em.ExitManager(conn, dry_run=True)
    yield mgr
    conn.close()


# A neutral ticker with no KXLOWT / KXHIGH / KXLOW substrings, so the
# temperature-specific stop-loss branches stay dormant.
_TICKER = "TESTMKT-26SEP09"
# An entry logged far in the past, so a min-hold window (when enabled) never
# excludes it unless a test deliberately uses a recent timestamp.
_OLD = "2020-01-01T00:00:00+00:00"


def _trade(
    trade_id=1, side="yes", limit_price=50, count=4, current_mid=50,
    source="rss", opportunity_kind="numeric", spread_id="",
    logged_at=_OLD, yes_bid=None, yes_ask=None,
):
    """Build an enriched _Trade positioned at ``current_mid`` cents.

    With side='yes', limit_price=50: entry cost = 50c/contract, so
    pct_gain = (current_mid - 50) / 50. current_mid=10 -> -80% (stop-loss),
    current_mid=70 -> +40% (profit-take), current_mid=50 -> flat (no exit).
    """
    t = _Trade(
        trade_id=trade_id, logged_at=logged_at, ticker=_TICKER,
        side=side, count=count, limit_price=limit_price,
        score=0.9, kelly_fraction=0.25, p_estimate=0.8,
        source=source, opportunity_kind=opportunity_kind, spread_id=spread_id,
    )
    t.current_mid = current_mid
    t.yes_bid = yes_bid
    t.yes_ask = yes_ask
    return t


# --- bottom of the ladder: the normal exits actually fire --------------------

def test_stop_loss_fires_on_big_loss(manager):
    _insert_row(manager._conn, 1)
    t = _trade(current_mid=10)          # -80% <= -70%
    events = _run(manager.check_exits(None, [t]))
    assert len(events) == 1
    assert events[0].reason == "stop_loss"
    assert events[0].trade_id == 1
    # The exit was persisted to the DB row.
    exited_at = manager._conn.execute(
        "SELECT exited_at FROM trades WHERE id = 1"
    ).fetchone()[0]
    assert exited_at is not None


def test_profit_take_fires_on_gain(manager):
    _insert_row(manager._conn, 1)
    t = _trade(current_mid=70)          # +40% >= +20%
    events = _run(manager.check_exits(None, [t]))
    assert len(events) == 1
    assert events[0].reason == "profit_take"


def test_flat_position_does_not_exit(manager):
    _insert_row(manager._conn, 1)
    t = _trade(current_mid=50)          # 0% — neither threshold met
    assert _run(manager.check_exits(None, [t])) == []


# --- rung 2: force-exit bypasses everything below it -------------------------

def test_force_exit_beats_min_hold_and_thresholds(manager, monkeypatch):
    # Min-hold ON and the trade is brand new (would be skipped by rung 3), and
    # the price is flat (no stop-loss / profit-take would fire). A queued
    # force-exit must still fire immediately, proving it sits above both gates.
    monkeypatch.setattr(em, "EXIT_MIN_HOLD_MINUTES", 1000.0, raising=True)
    _insert_row(manager._conn, 1)
    recent = datetime.now(timezone.utc).isoformat()
    t = _trade(current_mid=50, logged_at=recent)
    manager.request_force_exit(_TICKER, detail="force_exit:obs_contra")

    events = _run(manager.check_exits(None, [t]))
    assert len(events) == 1
    assert events[0].reason == "stop_loss"           # force exits are recorded as stop_loss
    detail = manager._conn.execute(
        "SELECT exit_reason_detail FROM trades WHERE id = 1"
    ).fetchone()[0]
    assert detail == "force_exit:obs_contra"


def test_force_exit_is_consumed_once(manager, monkeypatch):
    # The queued force-exit is popped on use: a second cycle must not re-fire.
    _insert_row(manager._conn, 1)
    manager.request_force_exit(_TICKER, detail="force_exit:once")
    t = _trade(current_mid=50)
    assert len(_run(manager.check_exits(None, [t]))) == 1
    # Second cycle: same flat trade, queue now empty -> no exit.
    t2 = _trade(current_mid=50)
    assert _run(manager.check_exits(None, [t2])) == []


# --- rung 3: min-hold skips a too-recent trade even if it's a loser ----------

def test_min_hold_suppresses_stop_loss_on_recent_trade(manager, monkeypatch):
    monkeypatch.setattr(em, "EXIT_MIN_HOLD_MINUTES", 1000.0, raising=True)
    _insert_row(manager._conn, 1)
    recent = datetime.now(timezone.utc).isoformat()
    t = _trade(current_mid=10, logged_at=recent)     # would stop out at -80%
    assert _run(manager.check_exits(None, [t])) == []


def test_min_hold_allows_old_trade_to_stop_out(manager, monkeypatch):
    # Same window, but an entry older than the cutoff is eligible again.
    monkeypatch.setattr(em, "EXIT_MIN_HOLD_MINUTES", 1000.0, raising=True)
    _insert_row(manager._conn, 1)
    old = (datetime.now(timezone.utc) - timedelta(minutes=2000)).isoformat()
    t = _trade(current_mid=10, logged_at=old)
    events = _run(manager.check_exits(None, [t]))
    assert len(events) == 1 and events[0].reason == "stop_loss"


# --- rung 4: cost <= 0 is skipped --------------------------------------------

def test_zero_cost_position_is_skipped(manager):
    _insert_row(manager._conn, 1)
    # NO at yes-price 100 => NO cost per contract = 100 - 100 = 0.
    t = _trade(side="no", limit_price=100, current_mid=5)
    assert _run(manager.check_exits(None, [t])) == []


# --- rung 5: numeric hold-to-settlement source suppresses stop-loss ----------

def test_hold_to_settlement_source_suppresses_stop_loss(manager):
    _insert_row(manager._conn, 1)
    # eia is a data-release source: intraday moves are noise, hold to settle.
    t = _trade(current_mid=10, source="eia", opportunity_kind="numeric")
    assert _run(manager.check_exits(None, [t])) == []


def test_non_hold_source_same_loss_does_stop_out(manager):
    # Contrast: identical loss on a non-hold source DOES stop out — isolating
    # that the source membership (not the price) drove the suppression above.
    _insert_row(manager._conn, 1)
    t = _trade(current_mid=10, source="rss", opportunity_kind="numeric")
    events = _run(manager.check_exits(None, [t]))
    assert len(events) == 1 and events[0].reason == "stop_loss"


def test_hold_to_settlement_only_applies_to_numeric_kind(manager):
    # The gate is (opportunity_kind == "numeric" AND source in HOLD set). An
    # eia-sourced trade of a different kind is NOT held — it stops out.
    _insert_row(manager._conn, 1)
    t = _trade(current_mid=10, source="eia", opportunity_kind="strike_arb")
    events = _run(manager.check_exits(None, [t]))
    assert len(events) == 1 and events[0].reason == "stop_loss"


# --- rung 6: spread legs never stop-loss, but may profit-take ----------------

def test_spread_leg_is_never_stopped_out(manager):
    _insert_row(manager._conn, 1)
    # A losing arb leg: stopping it out would unhedge the spread -> suppressed.
    t = _trade(current_mid=10, source="rss", opportunity_kind="arb", spread_id="sp1")
    assert _run(manager.check_exits(None, [t])) == []


def test_spread_leg_still_takes_profit(manager):
    _insert_row(manager._conn, 1)
    # A winning arb leg: profit-take is still permitted for spread legs.
    t = _trade(current_mid=70, source="rss", opportunity_kind="arb", spread_id="sp1")
    events = _run(manager.check_exits(None, [t]))
    assert len(events) == 1 and events[0].reason == "profit_take"


def test_spread_gate_only_applies_to_arb_kinds(manager):
    # spread_id set but opportunity_kind is NOT an arb-family kind: the
    # suppression does not apply, so a losing leg stops out normally.
    _insert_row(manager._conn, 1)
    t = _trade(current_mid=10, source="rss", opportunity_kind="numeric", spread_id="sp1")
    events = _run(manager.check_exits(None, [t]))
    assert len(events) == 1 and events[0].reason == "stop_loss"


# --- rung 1: settled / already-exited / no-price are skipped -----------------

def test_already_exited_trade_is_not_refired(manager):
    # The DB row is already stamped exited_at -> the id is in exited_ids and
    # the trade is skipped even though its price is a stop-loss.
    _insert_row(manager._conn, 1, exited_at="2026-09-09T00:00:00+00:00")
    t = _trade(current_mid=10)
    assert _run(manager.check_exits(None, [t])) == []


def test_no_live_price_is_skipped(manager):
    _insert_row(manager._conn, 1)
    t = _trade(current_mid=10)
    t.current_mid = None                 # enrichment couldn't price it
    assert _run(manager.check_exits(None, [t])) == []


def test_settled_trade_is_skipped(manager):
    _insert_row(manager._conn, 1)
    t = _trade(current_mid=10)
    t.settled = True
    assert _run(manager.check_exits(None, [t])) == []
