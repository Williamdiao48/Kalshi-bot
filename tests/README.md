# Tests

Unit tests for the bot's pure money-handling logic. First coverage added
2026-09-09 alongside the system-audit fixes.

## Running

```bash
venv/bin/python -m pip install -r requirements-dev.txt   # once
venv/bin/python -m pytest                                # all tests
venv/bin/python -m pytest tests/test_kelly_sizing.py -v  # one file
```

Config lives in `pytest.ini` at the repo root; `conftest.py` puts the repo
root on `sys.path` so `import kalshi_bot...` works from any cwd.

## What's covered

- **`test_kelly_sizing.py`** — `kelly_contracts`: the degenerate-cost guards
  (0, negative, >= 100) that prevent div-by-zero, negative/breakeven-edge
  no-bet, the Kelly formula, and the hard cap.
- **`test_utils.py`** — `parse_iso_dt`: `Z`/offset handling, and a pinned test
  documenting the naive-in/naive-out trap that the exit-loop guards defend
  against.
- **`test_dry_run_pnl.py`** — `_Trade`: the NO cost = `100 - limit_price`
  convention, realized settlement P&L on both sides, and mark-to-market signs.
- **`test_band_arb_executor.py`** — the band-arb YES executor on one-sided
  ("under") markets where `strike_lo=None`: drives a real `TradeExecutor`
  (dry-run, temp DB) end to end and pins that the open lower bound records as
  `None` without crashing (the exact KXHIGHT-...-T76 production bug).
- **`test_exit_precedence.py`** — `ExitManager.check_exits`: the "first match
  wins" ladder. Force-exit bypasses min-hold and thresholds (and is consumed
  once); min-hold, cost≤0, numeric hold-to-settlement, and spread-leg gates
  each suppress the normal stop-loss; spread legs still profit-take; settled /
  already-exited / no-price trades are skipped. Thresholds are pinned in the
  fixture so only *ordering* is under test.

## Scope note

`test_kelly_sizing`, `test_utils`, and `test_dry_run_pnl` are pure-function
tests (no network, no DB, no event loop). `test_band_arb_executor` and
`test_exit_precedence` drive the async paths against a temp SQLite DB with
`asyncio.run` (no pytest-asyncio dependency) and never hit the network —
`session` is `None`, exercised only in dry-run where no live orders are placed.
