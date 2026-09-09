"""Regression tests for the band-arb YES executor on one-sided ("under")
markets, where the signal legitimately has ``strike_lo=None``.

This is the exact case that crashed in production (KXHIGHTBOS-...-T76): the
YES executor assumed a two-sided band and touched ``strike_lo`` as a float in
a log format and in the note builder, raising TypeError on the None. These
tests drive a real ``TradeExecutor`` (dry-run, temp SQLite) end to end and
assert the trade is recorded with the open lower bound represented as None —
they fail hard if the None handling regresses.

Async is driven with ``asyncio.run`` so no pytest-asyncio dependency is needed.
``kelly_contracts`` is stubbed to a fixed count so the test deterministically
reaches the note/record path regardless of sizing-constant tuning; Kelly math
is covered separately in test_kelly_sizing.py.
"""
import asyncio
import json

import pytest

import kalshi_bot.trade_executor as te
from kalshi_bot.strike_arb import BandArbSignal


def _run(coro):
    return asyncio.run(coro)


# Columns the production `trades` table has beyond TradeExecutor._init_schema's
# CREATE (they are added by db.run_migrations, which pulls in cross-module
# tables we don't want in a unit test). Mirror them here so the dry-run
# INSERT and the executor's pre-trade queries (e.g. exited_at) work.
_TRADES_EXTRA_COLS = [
    ("fill_price_cents", "INTEGER"), ("spread_id", "TEXT"), ("note", "TEXT"),
    ("exited_at", "TEXT"), ("exit_price_cents", "INTEGER"),
    ("exit_pnl_cents", "REAL"), ("exit_reason", "TEXT"), ("exit_order_id", "TEXT"),
    ("peak_past", "INTEGER"), ("exit_reason_detail", "TEXT"),
    ("peak_pct_gain", "REAL"), ("peak_at", "TEXT"),
    ("exit_yes_bid", "INTEGER"), ("exit_yes_ask", "INTEGER"), ("bug_loss", "INTEGER"),
    ("settled_result", "TEXT"), ("settled_pnl_cents", "REAL"),
]


def _complete_trades_schema(conn):
    existing = {r[1] for r in conn.execute("PRAGMA table_info(trades)")}
    for col, typedef in _TRADES_EXTRA_COLS:
        if col not in existing:
            conn.execute(f"ALTER TABLE trades ADD COLUMN {col} {typedef}")


@pytest.fixture
def executor(tmp_path, monkeypatch):
    # Force a deterministic, side-effect-free environment.
    monkeypatch.setattr(te, "TRADE_DRY_RUN", True, raising=False)
    monkeypatch.setattr(te, "BAND_ARB_EXECUTION_ENABLED", True, raising=False)
    monkeypatch.setattr(te, "MAX_TOTAL_EXPOSURE_CENTS", 0, raising=False)  # skip exposure cap
    # Fixed positive size so we always reach the note/record path.
    monkeypatch.setattr(te, "kelly_contracts", lambda **kw: 5, raising=True)
    ex = te.TradeExecutor(db_path=tmp_path / "trades_test.db")
    _complete_trades_schema(ex._conn)
    yield ex
    ex.close()


def _under_signal(strike_lo):
    """An 'under 76' KXHIGH threshold market, locked in-band (obs 71.06)."""
    return BandArbSignal(
        metric="temp_high_bos",
        ticker="KXHIGHTBOS-26SEP09-T76",
        yes_bid=84, no_ask=16,
        observed_max=71.06, band_ceil=76.0,
        direction="under", city="Boston",
        side="yes", yes_ask=16, hours_to_close=2.0,
        is_locked=True, strike_lo=strike_lo,
        noaa_val=70.0, corr_status="metar_only",
        yes_ask_entry=16,
    )


def _trade_rows(ex, ticker):
    return ex._conn.execute(
        "SELECT side, count, limit_price, note FROM trades WHERE ticker = ?",
        (ticker,),
    ).fetchall()


def test_under_market_none_lower_bound_does_not_crash_and_records(executor):
    signal = _under_signal(strike_lo=None)
    # The bug: this raised TypeError inside the note builder. Must not now.
    _run(executor.maybe_trade_band_arb(session=None, signal=signal))

    rows = _trade_rows(executor, signal.ticker)
    assert len(rows) == 1, "expected exactly one recorded YES trade"
    side, count, limit_price, note_str = rows[0]
    assert side == "yes"
    assert count == 5
    assert limit_price == 16          # yes_ask


def test_under_market_note_stores_open_bounds_as_none(executor):
    _run(executor.maybe_trade_band_arb(session=None, signal=_under_signal(strike_lo=None)))
    (_, _, _, note_str), = _trade_rows(executor, "KXHIGHTBOS-26SEP09-T76")
    note = json.loads(note_str)
    # Open lower bound -> None (JSON null), not a number, and no crash.
    assert note["band_lo_f"] is None
    assert note["margin_lo_f"] is None
    # Ceiling side is still populated normally.
    assert note["band_ceil_f"] == 76.0


def test_numeric_lower_bound_is_populated(executor):
    # Contrast case on the SAME code path: when strike_lo is a real number the
    # note stores it (and the margin) as numbers. Only strike_lo varies vs the
    # None case above, isolating exactly the conditional the fix introduced.
    _run(executor.maybe_trade_band_arb(session=None, signal=_under_signal(strike_lo=54.0)))
    (_, _, _, note_str), = _trade_rows(executor, "KXHIGHTBOS-26SEP09-T76")
    note = json.loads(note_str)
    assert note["band_lo_f"] == 54.0
    assert note["margin_lo_f"] == round(71.06 - 54.0, 2)
