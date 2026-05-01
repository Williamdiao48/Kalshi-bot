"""Backtest optimal MAX_POSITION_CENTS for Kelly sizing.

Uses *proportional scaling* from actual recorded counts — avoids re-deriving
win_prob (which is stored inconsistently across signal paths).

For each candidate pos_max, each trade's contract count is scaled as:

    new_count = clip(round(actual_count × pos_max / base_pos_max), 1, hard_cap)

Three paths are swept independently based on stored kelly_fraction:
  standard  (kf≈0.25, kf≈0.40) → base=750, hard_cap=10, sweep MAX_POSITION_CENTS
  locked    (kf≈0.75)          → base=5000, hard_cap=50, sweep LOCKED_OBS_MAX_POSITION_CENTS

P&L per contract is derived from stored exit_pnl_cents (early exits) or
settlement outcome (won/lost).  Bug-loss trades are excluded.

Usage
-----
  venv/bin/python scripts/backtest_kelly_sizing.py
  venv/bin/python scripts/backtest_kelly_sizing.py --db opportunity_log.db
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import statistics
from typing import NamedTuple

# ── current live defaults ──────────────────────────────────────────────────────
CURRENT_STD_POS_MAX    = 750
CURRENT_LOCKED_POS_MAX = 5000
STD_HARD_CAP           = 10
LOCKED_HARD_CAP        = 50
STD_BASE               = 750    # pos_max that produced the actual counts
LOCKED_BASE            = 5000

# Sweep grids
STANDARD_SWEEP = [250, 500, 750, 1000, 1500, 2000, 3000, 5000]
LOCKED_SWEEP   = [1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000]


# ── helpers ────────────────────────────────────────────────────────────────────

def _path(kf: float) -> str:
    if kf >= 0.70:
        return "locked"
    return "standard"


def _pnl_per_contract(side: str, limit_price: int, outcome: str | None,
                      exit_pnl_cents: float | None, actual_count: int) -> float | None:
    if exit_pnl_cents is not None and actual_count > 0:
        return exit_pnl_cents / actual_count
    cost = limit_price if side == "yes" else (100 - limit_price)
    if outcome == "won":
        return float(100 - cost)
    if outcome == "lost":
        return float(-cost)
    return None


def _scale(actual_count: int, base: int, new_pos_max: int, hard_cap: int) -> int:
    if base <= 0:
        return actual_count
    scaled = round(actual_count * new_pos_max / base)
    return max(1, min(hard_cap, scaled))


class TradeRec(NamedTuple):
    id:           int
    path:         str       # "standard" | "locked"
    actual_count: int
    ppc:          float     # P&L per contract (cents)
    source:       str
    side:         str
    limit_price:  int


def _load(db: sqlite3.Connection) -> list[TradeRec]:
    rows = db.execute(
        """
        SELECT id, side, count, limit_price, kelly_fraction,
               outcome, exit_pnl_cents, source
        FROM trades
        WHERE mode = 'dry_run'
          AND (outcome IN ('won','lost') OR exited_at IS NOT NULL)
          AND outcome IS NOT 'void'
          AND COALESCE(bug_loss, 0) = 0
        ORDER BY logged_at ASC
        """
    ).fetchall()
    result = []
    for r in rows:
        kf      = r[4] or 0.25
        path    = _path(kf)
        ppc     = _pnl_per_contract(r[1], r[3], r[5], r[6], r[2])
        if ppc is None:
            continue
        result.append(TradeRec(
            id=r[0], path=path, actual_count=r[2], ppc=ppc,
            source=r[7] or "unknown", side=r[1], limit_price=r[3],
        ))
    return result


# ── simulation ─────────────────────────────────────────────────────────────────

class SimResult(NamedTuple):
    pos_max:     int
    total_pnl:   float
    max_dd_pct:  float
    win_rate:    float
    n_trades:    int
    avg_pnl:     float
    sharpe:      float
    avg_count:   float
    max_single:  float    # largest single-trade loss (cents)


def _simulate(trades: list[TradeRec], pos_max: int, path: str) -> SimResult:
    base     = LOCKED_BASE if path == "locked" else STD_BASE
    hard_cap = LOCKED_HARD_CAP if path == "locked" else STD_HARD_CAP

    equity = 0.0
    peak   = 0.0
    series: list[float] = []
    counts: list[int]   = []

    for t in [t for t in trades if t.path == path]:
        cnt        = _scale(t.actual_count, base, pos_max, hard_cap)
        trade_pnl  = t.ppc * cnt
        equity    += trade_pnl
        peak       = max(peak, equity)
        series.append(trade_pnl)
        counts.append(cnt)

    if not series:
        return SimResult(pos_max, 0, 0, 0, 0, 0, 0, 0, 0)

    n     = len(series)
    wins  = sum(1 for p in series if p > 0)
    dd    = max(0.0, (peak - equity) / (10_000 + max(peak, 1))) * 100
    avg   = equity / n
    worst = min(series)
    try:
        sharpe = (avg / statistics.stdev(series)) * math.sqrt(n) if n > 1 else 0.0
    except statistics.StatisticsError:
        sharpe = 0.0

    return SimResult(
        pos_max=pos_max, total_pnl=equity, max_dd_pct=dd,
        win_rate=wins / n, n_trades=n, avg_pnl=avg, sharpe=sharpe,
        avg_count=sum(counts) / n, max_single=worst,
    )


# ── output ─────────────────────────────────────────────────────────────────────

def _fmt_row(r: SimResult, current: int) -> str:
    marker = " ◄ current" if r.pos_max == current else ""
    return (
        f"  {r.pos_max:>8}  {r.total_pnl/100:>+9.2f}$  {r.avg_pnl/100:>+7.3f}$  "
        f"{r.sharpe:>7.2f}  {r.max_dd_pct:>6.1f}%  {r.win_rate:>7.1%}  "
        f"{r.avg_count:>6.1f}  {r.max_single/100:>8.2f}$  {r.n_trades:>5}{marker}"
    )


def _print_sweep(results: list[SimResult], current: int, label: str) -> None:
    print(f"\n{'═'*98}")
    print(f"  {label}")
    print(f"{'═'*98}")
    print(f"  {'pos_max':>8}  {'TotalPnL':>10}  {'AvgPnL':>8}  "
          f"{'Sharpe':>7}  {'MaxDD%':>7}  {'WinRate':>8}  "
          f"{'AvgCt':>6}  {'WorstTrd':>9}  {'N':>5}")
    print(f"  {'-'*94}")
    for r in results:
        print(_fmt_row(r, current))

    # find best by P&L among non-trivially-small results
    valid = [r for r in results if r.n_trades >= 5]
    if not valid:
        return
    best_pnl = max(valid, key=lambda r: r.total_pnl)
    best_sh  = max(valid, key=lambda r: r.sharpe)
    print(f"\n  Best P&L    → pos_max={best_pnl.pos_max}  "
          f"({best_pnl.total_pnl/100:+.2f}$  sharpe={best_pnl.sharpe:.2f})")
    print(f"  Best Sharpe → pos_max={best_sh.pos_max}  "
          f"(sharpe={best_sh.sharpe:.2f}  P&L={best_sh.total_pnl/100:+.2f}$)")


def _print_source_breakdown(trades: list[TradeRec], pos_max: int, path: str) -> None:
    base     = LOCKED_BASE if path == "locked" else STD_BASE
    hard_cap = LOCKED_HARD_CAP if path == "locked" else STD_HARD_CAP

    cats: dict[str, list[float]] = {}
    for t in [t for t in trades if t.path == path]:
        cnt = _scale(t.actual_count, base, pos_max, hard_cap)
        cats.setdefault(t.source, []).append(t.ppc * cnt)

    if not cats:
        return
    print(f"\n  Source breakdown at pos_max={pos_max} ({path} path):")
    print(f"  {'Source':<20}  {'N':>4}  {'TotalPnL':>10}  {'AvgPnL':>8}  {'WinRate':>8}")
    print(f"  {'-'*60}")
    for src, series in sorted(cats.items(), key=lambda kv: sum(kv[1]), reverse=True):
        n    = len(series)
        tot  = sum(series)
        wr   = sum(1 for p in series if p > 0) / n if n else 0
        print(f"  {src:<20}  {n:>4}  {tot/100:>+9.2f}$  {tot/100/n:>+7.3f}$  {wr:>7.1%}")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest optimal MAX_POSITION_CENTS (proportional scaling)"
    )
    parser.add_argument("--db", default="opportunity_log.db")
    args = parser.parse_args()

    db     = sqlite3.connect(args.db)
    trades = _load(db)

    std_trades    = [t for t in trades if t.path == "standard"]
    locked_trades = [t for t in trades if t.path == "locked"]

    print(f"\nLoaded {len(trades)} closed non-bug trades  "
          f"({len(std_trades)} standard kf≤0.5, {len(locked_trades)} locked kf≥0.7)")

    # baseline sanity check
    baseline_std    = sum(t.ppc * t.actual_count for t in std_trades)
    baseline_locked = sum(t.ppc * t.actual_count for t in locked_trades)
    print(f"Actual P&L — standard: {baseline_std/100:+.2f}$  "
          f"locked: {baseline_locked/100:+.2f}$  "
          f"total: {(baseline_std+baseline_locked)/100:+.2f}$")

    # ── standard sweep ────────────────────────────────────────────────────────
    std_results = [_simulate(trades, p, "standard") for p in STANDARD_SWEEP]
    _print_sweep(std_results, CURRENT_STD_POS_MAX,
                 "STANDARD PATH  (kf≤0.5)  — sweep MAX_POSITION_CENTS  [hard cap: 10 contracts]")

    best_std = max(std_results, key=lambda r: r.total_pnl)
    _print_source_breakdown(trades, best_std.pos_max, "standard")

    # ── locked sweep ──────────────────────────────────────────────────────────
    if locked_trades:
        locked_results = [_simulate(trades, p, "locked") for p in LOCKED_SWEEP]
        _print_sweep(locked_results, CURRENT_LOCKED_POS_MAX,
                     "LOCKED-OBS PATH  (kf≥0.7)  — sweep LOCKED_OBS_MAX_POSITION_CENTS  [hard cap: 50]")

        best_locked = max(locked_results, key=lambda r: r.total_pnl)
        _print_source_breakdown(trades, best_locked.pos_max, "locked")

    # ── combined view ─────────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print("  COMBINED VIEW — select standard × locked pairs")
    print(f"{'═'*70}")
    print(f"  {'std_max':>8}  {'lkd_max':>8}  {'CombPnL':>10}  {'note'}")
    print(f"  {'-'*60}")
    for sp in [250, 500, 750, 1000, 1500, 2000]:
        for lp in [2000, 5000, 10000, 20000]:
            sr = _simulate(trades, sp, "standard").total_pnl
            lr = _simulate(trades, lp, "locked").total_pnl if locked_trades else 0
            marker = " ◄ current" if sp == CURRENT_STD_POS_MAX and lp == CURRENT_LOCKED_POS_MAX else ""
            print(f"  {sp:>8}  {lp:>8}  {(sr+lr)/100:>+9.2f}$  {marker}")

    print(f"\n{'═'*70}")
    print("  Scaling method: proportional  (new_count = round(actual × pos_max / base))")
    print("  Hard caps applied: standard ≤10ct, locked ≤50ct")
    print("  Win rates and signal quality are identical across all pos_max values.")
    print("  Larger pos_max amplifies both wins AND losses — only helpful if edge > 0.")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()
