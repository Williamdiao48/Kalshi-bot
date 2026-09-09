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

## Scope note

These are pure-function tests (no network, no DB, no event loop). The async
paths — exit precedence, order placement, the band-arb executor's
`strike_lo=None` handling — are the next tier and need light fixtures/fakes.
