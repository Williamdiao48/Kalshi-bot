"""Unit tests for kalshi_bot.dry_run_ledger._Trade cost & P&L conventions.

These pin the NO/YES price convention that runs through the whole system:
`limit_price` always stores the YES-scale price in cents, so a NO position's
cost per contract is `100 - limit_price`. Getting this sign wrong silently
mis-states P&L, so it is exactly the invariant worth a regression test.
"""
from kalshi_bot.dry_run_ledger import _Trade


def _mk(side, limit_price, count=1, **kw):
    """Construct a _Trade with only the fields these tests exercise."""
    return _Trade(
        trade_id=1, logged_at="2026-09-09T12:00:00Z", ticker="KXHIGHTBOS-26SEP09-T76",
        side=side, count=count, limit_price=limit_price,
        score=0.9, kelly_fraction=0.25, p_estimate=0.8, **kw,
    )


# --- entry cost: the core NO = 100 - limit_price convention -----------------

def test_yes_cost_per_contract_is_limit_price():
    assert _mk("yes", 40).cost_per_contract == 40


def test_no_cost_per_contract_is_complement():
    # NO at yes-price 40 costs 60c per contract.
    assert _mk("no", 40).cost_per_contract == 60


def test_total_cost_scales_with_count():
    assert _mk("no", 40, count=5).total_cost_cents == 60 * 5
    assert _mk("yes", 40, count=5).total_cost_cents == 40 * 5


# --- realized P&L at settlement ---------------------------------------------

def test_yes_win_pays_complement():
    t = _mk("yes", 40, count=3, outcome="won")   # result == "yes"
    assert t.settled and t.result == "yes"
    assert t._realized_cents() == (100 - 40) * 3


def test_yes_loss_costs_entry():
    t = _mk("yes", 40, count=3, outcome="lost")  # result == "no"
    assert t.result == "no"
    assert t._realized_cents() == (-40) * 3


def test_no_win_pays_entry_price():
    t = _mk("no", 40, count=2, outcome="won")    # result == "no"
    assert t.result == "no"
    # NO win => keep the whole NO payout minus cost: net +limit_price per contract.
    assert t._realized_cents() == 40 * 2


def test_no_loss_costs_complement():
    t = _mk("no", 40, count=2, outcome="lost")   # result == "yes"
    assert t.result == "yes"
    assert t._realized_cents() == (40 - 100) * 2


def test_win_is_positive_loss_is_negative_both_sides():
    for side, lp in [("yes", 45), ("no", 45)]:
        assert _mk(side, lp, outcome="won")._realized_cents() > 0
        assert _mk(side, lp, outcome="lost")._realized_cents() < 0


# --- unrealized (mark-to-market) sign logic ---------------------------------

def test_yes_unrealized_is_mid_minus_entry():
    t = _mk("yes", 40, count=2)
    t.current_mid = 55            # YES bid moved up 15c
    assert t._unrealized_cents() == (55 - 40) * 2


def test_no_unrealized_uses_no_entry_cost():
    t = _mk("no", 40, count=2)    # NO entry cost = 60c
    t.current_mid = 70            # current NO exit price
    assert t._unrealized_cents() == (70 - 60) * 2


def test_pnl_priority_exited_over_settled():
    # An exited trade reports its locked exit P&L even if also marked settled.
    t = _mk("yes", 40, count=1, outcome="won",
            exited_at="2026-09-09T13:00:00Z", exit_pnl_cents=12.0)
    assert t.exited
    assert t.pnl_cents == 12.0
