"""Unit tests for kalshi_bot.trade_executor.kelly_contracts.

Kelly sizing is money-handling code with no automated coverage before this.
These tests pin the guards (the audit flagged div-by-zero risk in the sizing
family) and the core formula:

    raw_f     = (win_prob - cost/100) / (1 - cost/100)
    contracts = floor(kelly_fraction * raw_f * max_cents / cost_cents)
"""
import math

from kalshi_bot.trade_executor import kelly_contracts


# --- guards: degenerate cost must return 0, never raise ---------------------

def test_zero_cost_returns_zero_no_raise():
    # cost_cents == 0 would be a div-by-zero in the formula; the guard must
    # catch it first and return 0.
    assert kelly_contracts(win_prob=0.9, cost_cents=0,
                           max_cents=10_000, kelly_fraction=0.25, hard_cap=100) == 0


def test_negative_cost_returns_zero():
    assert kelly_contracts(win_prob=0.9, cost_cents=-5,
                           max_cents=10_000, kelly_fraction=0.25, hard_cap=100) == 0


def test_cost_at_or_above_100_returns_zero():
    # A contract costing >= 100c has no upside; guard returns 0 (also avoids
    # a zero/negative denominator in 1 - cost/100).
    assert kelly_contracts(win_prob=0.99, cost_cents=100,
                           max_cents=10_000, kelly_fraction=0.25, hard_cap=100) == 0
    assert kelly_contracts(win_prob=0.99, cost_cents=150,
                           max_cents=10_000, kelly_fraction=0.25, hard_cap=100) == 0


# --- edge sign: bet only when win_prob beats the implied cost ---------------

def test_negative_edge_returns_zero():
    # cost 60c implies breakeven p = 0.60; a 0.50 estimate is negative edge.
    assert kelly_contracts(win_prob=0.50, cost_cents=60,
                           max_cents=10_000, kelly_fraction=0.25, hard_cap=100) == 0


def test_breakeven_edge_returns_zero():
    # win_prob exactly equal to cost/100 => raw_f == 0 => no bet.
    assert kelly_contracts(win_prob=0.60, cost_cents=60,
                           max_cents=10_000, kelly_fraction=0.25, hard_cap=100) == 0


# --- positive edge: matches the documented formula --------------------------

def test_positive_edge_matches_formula():
    win_prob, cost, max_cents, frac, cap = 0.80, 50, 10_000, 0.25, 10_000
    raw_f = (win_prob - cost / 100.0) / (1.0 - cost / 100.0)
    expected = math.floor(frac * raw_f * max_cents / cost)
    got = kelly_contracts(win_prob=win_prob, cost_cents=cost,
                          max_cents=max_cents, kelly_fraction=frac, hard_cap=cap)
    assert got == expected
    assert got > 0


def test_hard_cap_binds():
    # Huge bankroll but a tight hard cap: result is clamped to the cap.
    got = kelly_contracts(win_prob=0.95, cost_cents=20,
                          max_cents=10_000_000, kelly_fraction=1.0, hard_cap=7)
    assert got == 7


def test_larger_kelly_fraction_never_sizes_smaller():
    # Monotonic in the fraction (below the hard cap).
    small = kelly_contracts(win_prob=0.80, cost_cents=50,
                            max_cents=10_000, kelly_fraction=0.10, hard_cap=10_000)
    big = kelly_contracts(win_prob=0.80, cost_cents=50,
                          max_cents=10_000, kelly_fraction=0.50, hard_cap=10_000)
    assert big >= small


def test_result_is_int():
    got = kelly_contracts(win_prob=0.75, cost_cents=40,
                          max_cents=5_000, kelly_fraction=0.25, hard_cap=1_000)
    assert isinstance(got, int)
