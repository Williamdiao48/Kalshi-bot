"""Exit strategy optimization analysis.

Reads price_snapshots and trades from opportunity_log.db and answers:

  1. Recovery rate  — Of positions we stopped out of, what % eventually settled
                      as wins?  Broken down by source and exit reason.

  2. Wasted exits   — Profit-takes that later settled as losses: we captured a
                      small gain but would have lost anyway.  Also profit-takes
                      that settled as wins: we might have left money on the table.

  3. Hold vs exit   — For every early exit with a known settlement outcome, was
                      exiting early better than holding to settlement?

  4. Peak gain      — Distribution (p25/median/p75/max) of peak pct_gain reached
                      per trade, broken down by source and side.  Answers: "how
                      high do our best signals typically run before reversing?"

  5. Threshold sim  — Simulate total P&L if we had used profit-take thresholds
                      of 20–80% (in 5pp steps), based on actual price trajectories
                      in price_snapshots.  Shows the optimal harvest point.

  6. Drawdown depth — After a trade peaks, how deep does it draw before the
                      current threshold fires?  Distribution of drawdown depth
                      at the moment of exit.

Usage
-----
  venv/bin/python analyze_exits.py [lookback_days] [output_file]

  # Examples:
  venv/bin/python analyze_exits.py              # all time, prints to stdout
  venv/bin/python analyze_exits.py 14           # last 14 days
  venv/bin/python analyze_exits.py 30 exits.txt # last 30 days → file
"""

from __future__ import annotations

import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median, quantiles
from typing import Optional

_DB_PATH = Path(__file__).parent / "opportunity_log.db"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    id:          int
    logged_at:   str
    ticker:      str
    source:      str
    side:        str       # 'yes' | 'no'
    count:       int
    limit_price: int       # YES price in cents
    exit_reason: str | None
    exit_pnl:    float | None
    outcome:     str | None   # 'won' | 'lost' | None
    cost:        int           # cents we paid (limit_price × count for YES)


@dataclass
class Snapshot:
    trade_id:     int
    snapshot_at:  str
    pct_gain:     float
    days_to_close: float | None


def _cost(side: str, limit_price: int, count: int) -> int:
    if side == "yes":
        return limit_price * count
    return (100 - limit_price) * count


def _hold_pnl(t: Trade) -> float | None:
    """What P&L would we have captured if we held to settlement?"""
    if t.outcome == "won":
        return (100 - t.limit_price) * t.count if t.side == "yes" \
               else t.limit_price * t.count
    if t.outcome == "lost":
        return (-t.limit_price * t.count) if t.side == "yes" \
               else (-(100 - t.limit_price) * t.count)
    return None


def _load(conn: sqlite3.Connection, cutoff: str | None):
    mode = "AND t.mode IN ('dry_run', 'live')"
    cut  = f"AND t.logged_at >= '{cutoff}'" if cutoff else ""

    rows = conn.execute(f"""
        SELECT
            t.id, t.logged_at, t.ticker,
            COALESCE(t.source, 'unknown') AS source,
            t.side, t.count, t.limit_price,
            t.exit_reason, t.exit_pnl_cents, t.outcome
        FROM trades t
        WHERE 1=1 {mode} {cut}
        ORDER BY t.id
    """).fetchall()

    trades: dict[int, Trade] = {}
    for r in rows:
        tid, logged_at, ticker, source, side, count, lp, exit_reason, exit_pnl, outcome = r
        trades[tid] = Trade(
            id=tid, logged_at=logged_at, ticker=ticker, source=source,
            side=side, count=count, limit_price=lp,
            exit_reason=exit_reason, exit_pnl=exit_pnl, outcome=outcome,
            cost=_cost(side, lp, count),
        )

    snap_rows = conn.execute("""
        SELECT trade_id, snapshot_at, pct_gain, days_to_close
        FROM price_snapshots
        WHERE pct_gain IS NOT NULL
        ORDER BY trade_id, snapshot_at
    """).fetchall()

    snaps: dict[int, list[Snapshot]] = defaultdict(list)
    for r in snap_rows:
        tid, sat, pg, dtc = r
        if tid in trades:
            snaps[tid].append(Snapshot(tid, sat, pg, dtc))

    return trades, snaps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pct(n: int, d: int) -> str:
    return f"{100*n/d:.0f}%" if d else "—"


def _fmt_pnl(c: float) -> str:
    sign = "+" if c >= 0 else ""
    return f"{sign}${c/100:.2f}"


def _distribution(values: list[float]) -> str:
    if not values:
        return "no data"
    s = sorted(values)
    q = quantiles(s, n=4) if len(s) >= 4 else [s[0], median(s), s[-1]]
    return (f"p25={q[0]*100:+.0f}%  med={q[1]*100:+.0f}%"
            f"  p75={q[2]*100:+.0f}%  max={max(s)*100:+.0f}%  n={len(s)}")


# ---------------------------------------------------------------------------
# Section 1 — Recovery rate
# ---------------------------------------------------------------------------

def _recovery_rate(trades: dict[int, Trade]) -> list[str]:
    """For stop_loss / trailing_stop exits with a known settlement, did the
    market go on to settle as a win?  If yes, we cut a winner early."""
    lines = ["\n── 1. Recovery Rate After Stop-Loss Exits ─────────────────────────────"]
    lines.append("   (How often did the market SETTLE as 'won' after we cut it?)")
    lines.append("")

    exited = [t for t in trades.values()
              if t.exit_reason in ("stop_loss", "trailing_stop")
              and t.outcome is not None]
    if not exited:
        lines.append("  No settled stop-loss exits found.")
        return lines

    # Global
    recovered = [t for t in exited if t.outcome == "won"]
    lines.append(
        f"  Global: {len(recovered)}/{len(exited)} stop-outs recovered "
        f"({_pct(len(recovered), len(exited))})"
    )
    lines.append("")

    # By source
    by_src: dict[str, list[Trade]] = defaultdict(list)
    for t in exited:
        by_src[t.source].append(t)
    lines.append(f"  {'Source':<20}  {'Stopped':>7}  {'Recovered':>10}  {'Rate':>6}  {'Avg loss at cut':>15}")
    lines.append("  " + "─" * 68)
    for src, lst in sorted(by_src.items(), key=lambda x: -len(x[1])):
        rec   = [t for t in lst if t.outcome == "won"]
        avg_loss = (
            sum(t.exit_pnl for t in lst if t.exit_pnl is not None) /
            len([t for t in lst if t.exit_pnl is not None])
        ) if any(t.exit_pnl is not None for t in lst) else None
        avg_s = f"{avg_loss/100:+.2f}" if avg_loss is not None else "?"
        lines.append(
            f"  {src:<20}  {len(lst):>7}  {len(rec):>10}  "
            f"{_pct(len(rec), len(lst)):>6}  {avg_s:>15}"
        )

    lines.append("")
    lines.append("  Note: a high recovery rate means our stop-loss is too tight;")
    lines.append("  we're cutting winners that would have recovered.")
    return lines


# ---------------------------------------------------------------------------
# Section 2 — Wasted exits
# ---------------------------------------------------------------------------

def _wasted_exits(trades: dict[int, Trade]) -> list[str]:
    lines = ["\n── 2. Profit-Take Exit Audit ──────────────────────────────────────────"]
    lines.append("   (Did we exit at profit while the market still settled against us?)")
    lines.append("")

    pt = [t for t in trades.values()
          if t.exit_reason == "profit_take" and t.outcome is not None]
    if not pt:
        lines.append("  No settled profit-take exits found.")
        return lines

    pt_then_lost = [t for t in pt if t.outcome == "lost"]
    pt_then_won  = [t for t in pt if t.outcome == "won"]

    lines.append(f"  Profit-take exits with settlement:  {len(pt)}")
    lines.append(
        f"  Exited profitable → market settled YES  (good):  {len(pt_then_won)}"
        f"  ({_pct(len(pt_then_won), len(pt))})"
    )
    lines.append(
        f"  Exited profitable → market settled NO   (lucky): {len(pt_then_lost)}"
        f"  ({_pct(len(pt_then_lost), len(pt))})"
    )
    lines.append("")

    # For profit-take-then-won: how much did we leave on table?
    if pt_then_won:
        lines.append("  Left on the table (profit-take settled WIN — could have held):")
        total_captured = sum(t.exit_pnl for t in pt_then_won if t.exit_pnl is not None)
        total_hold     = sum(_hold_pnl(t) for t in pt_then_won if _hold_pnl(t) is not None)
        lines.append(f"    Captured at exit : {_fmt_pnl(total_captured)}")
        lines.append(f"    Would have earned : {_fmt_pnl(total_hold)}")
        left = total_hold - total_captured
        lines.append(f"    Left on table    : {_fmt_pnl(left)} "
                     f"({'over-exited' if left > 0 else 'exit was better'})")

    lines.append("")

    # For profit-take-then-lost: we salvaged something
    if pt_then_lost:
        lines.append("  Lucky exits (profit-take then settled LOSS — exit saved us):")
        total_captured = sum(t.exit_pnl for t in pt_then_lost if t.exit_pnl is not None)
        total_hold     = sum(_hold_pnl(t) for t in pt_then_lost if _hold_pnl(t) is not None)
        saved = total_captured - total_hold
        lines.append(f"    Exit P&L  : {_fmt_pnl(total_captured)}")
        lines.append(f"    Hold P&L  : {_fmt_pnl(total_hold)}")
        lines.append(f"    Saved     : {_fmt_pnl(saved)}")

    return lines


# ---------------------------------------------------------------------------
# Section 3 — Hold vs exit
# ---------------------------------------------------------------------------

def _hold_vs_exit(trades: dict[int, Trade]) -> list[str]:
    lines = ["\n── 3. Hold vs Early Exit — Overall Impact ─────────────────────────────"]
    lines.append("")

    exited = [t for t in trades.values()
              if t.exit_reason is not None
              and t.exit_pnl is not None
              and t.outcome is not None]
    if not exited:
        lines.append("  No settled early-exit trades found.")
        return lines

    better_exit = 0
    better_hold = 0
    total_exit_pnl = 0.0
    total_hold_pnl = 0.0

    for t in exited:
        hold = _hold_pnl(t)
        if hold is None:
            continue
        total_exit_pnl += t.exit_pnl
        total_hold_pnl += hold
        if t.exit_pnl >= hold:
            better_exit += 1
        else:
            better_hold += 1

    n = better_exit + better_hold
    lines.append(f"  Trades compared (settled + exited early): {n}")
    lines.append(f"  Exit was better than hold : {better_exit} ({_pct(better_exit, n)})")
    lines.append(f"  Hold was better           : {better_hold} ({_pct(better_hold, n)})")
    lines.append(f"  Total P&L captured (exits): {_fmt_pnl(total_exit_pnl)}")
    lines.append(f"  Total P&L if held         : {_fmt_pnl(total_hold_pnl)}")
    lines.append(f"  Net early-exit impact     : {_fmt_pnl(total_exit_pnl - total_hold_pnl)}")
    lines.append("")

    # Breakdown by exit_reason
    for reason in ("profit_take", "stop_loss", "trailing_stop"):
        group = [t for t in exited if t.exit_reason == reason and _hold_pnl(t) is not None]
        if not group:
            continue
        ep = sum(t.exit_pnl for t in group)
        hp = sum(_hold_pnl(t) for t in group)
        be = sum(1 for t in group if t.exit_pnl >= _hold_pnl(t))
        lines.append(f"  {reason:<15}  n={len(group):>3}  exit={_fmt_pnl(ep):>9}"
                     f"  hold={_fmt_pnl(hp):>9}  exit_better={_pct(be, len(group)):>5}")

    return lines


# ---------------------------------------------------------------------------
# Section 4 — Peak gain distribution
# ---------------------------------------------------------------------------

def _peak_distribution(
    trades: dict[int, Trade],
    snaps:  dict[int, list[Snapshot]],
) -> list[str]:
    lines = ["\n── 4. Peak Gain Distribution ──────────────────────────────────────────"]
    lines.append("   (How high did each position run before it turned?)")
    lines.append("")

    peaks_by_src:  dict[str, list[float]] = defaultdict(list)
    peaks_by_side: dict[str, list[float]] = defaultdict(list)
    all_peaks: list[float] = []

    for tid, ss in snaps.items():
        if not ss:
            continue
        peak = max(s.pct_gain for s in ss)
        t = trades.get(tid)
        if t is None:
            continue
        all_peaks.append(peak)
        peaks_by_src[t.source].append(peak)
        peaks_by_side[t.side].append(peak)

    if not all_peaks:
        lines.append("  No snapshot data found.")
        return lines

    lines.append(f"  All trades:  {_distribution(all_peaks)}")
    lines.append("")
    lines.append(f"  {'Source':<22}  Distribution")
    lines.append("  " + "─" * 72)
    for src, vals in sorted(peaks_by_src.items(), key=lambda x: -len(x[1])):
        lines.append(f"  {src:<22}  {_distribution(vals)}")

    lines.append("")
    lines.append(f"  {'Side':<22}  Distribution")
    lines.append("  " + "─" * 72)
    for side, vals in sorted(peaks_by_side.items()):
        lines.append(f"  {side.upper():<22}  {_distribution(vals)}")

    return lines


# ---------------------------------------------------------------------------
# Section 5 — Threshold simulation
# ---------------------------------------------------------------------------

def _threshold_sim(
    trades: dict[int, Trade],
    snaps:  dict[int, list[Snapshot]],
) -> list[str]:
    lines = ["\n── 5. Profit-Take Threshold Simulation ────────────────────────────────"]
    lines.append("   (If we had exited at threshold X, what would total P&L be?)")
    lines.append("   Uses actual price_snapshot trajectories for each trade.")
    lines.append("")

    thresholds = [t / 100 for t in range(20, 85, 5)]  # 0.20 … 0.80

    # Only trades that have snapshots AND a settled outcome
    eligible = {
        tid: (trades[tid], ss)
        for tid, ss in snaps.items()
        if tid in trades and trades[tid].outcome is not None
    }
    if not eligible:
        lines.append("  No settled trades with snapshot history found.")
        return lines

    header = f"  {'Threshold':>10}  {'Total P&L':>10}  {'# Exited':>9}  {'# Held to settle':>17}  {'Avg exit pct':>13}"
    lines.append(header)
    lines.append("  " + "─" * 68)

    best_pnl   = None
    best_thresh = None

    for thresh in thresholds:
        total_pnl   = 0.0
        n_exited    = 0
        n_held      = 0
        exit_pcts   = []

        for tid, (t, ss) in eligible.items():
            # Find first snapshot where pct_gain >= threshold
            hit = next((s for s in ss if s.pct_gain >= thresh), None)
            if hit:
                # Simulated exit: capture threshold fraction of cost
                sim_pnl = thresh * t.cost
                total_pnl += sim_pnl
                exit_pcts.append(thresh)
                n_exited += 1
            else:
                # Held to settlement — use actual outcome
                hold = _hold_pnl(t)
                if hold is not None:
                    total_pnl += hold
                n_held += 1

        avg_ep = f"{sum(exit_pcts)/len(exit_pcts)*100:.0f}%" if exit_pcts else "—"
        flag = " ◀ best" if best_pnl is None or total_pnl > best_pnl else ""
        if best_pnl is None or total_pnl > best_pnl:
            best_pnl    = total_pnl
            best_thresh = thresh

        lines.append(
            f"  {thresh*100:>9.0f}%  {_fmt_pnl(total_pnl):>10}"
            f"  {n_exited:>9}  {n_held:>17}  {avg_ep:>13}{flag}"
        )

    lines.append("")
    if best_thresh is not None:
        lines.append(
            f"  Simulated optimal threshold: {best_thresh*100:.0f}%  "
            f"(P&L {_fmt_pnl(best_pnl)})"
        )
    lines.append("")
    lines.append("  Caveats: simulation assumes instant fill at the snapshot mid-price.")
    lines.append("  Actual fills may differ by 1–3¢ (bid/ask spread).")

    return lines


# ---------------------------------------------------------------------------
# Section 6 — Drawdown depth at exit
# ---------------------------------------------------------------------------

def _drawdown_at_exit(
    trades: dict[int, Trade],
    snaps:  dict[int, list[Snapshot]],
) -> list[str]:
    lines = ["\n── 6. Drawdown Depth at Exit ───────────────────────────────────────────"]
    lines.append("   (From peak to exit: how deep was the drawdown when the trigger fired?)")
    lines.append("")

    drawdowns: dict[str, list[float]] = defaultdict(list)

    for tid, ss in snaps.items():
        t = trades.get(tid)
        if t is None or t.exit_reason is None or t.exit_pnl is None or t.cost <= 0:
            continue
        if not ss:
            continue

        peak   = max(s.pct_gain for s in ss)
        actual = t.exit_pnl / t.cost  # pct at actual exit
        drawdown = peak - actual
        if drawdown >= 0:  # only meaningful when we drew back from peak
            drawdowns[t.exit_reason].append(drawdown)

    if not any(drawdowns.values()):
        lines.append("  No exit drawdown data found.")
        return lines

    lines.append(f"  {'Exit reason':<20}  {'n':>4}  Distribution (drawdown from peak at exit)")
    lines.append("  " + "─" * 72)
    for reason in ("profit_take", "stop_loss", "trailing_stop"):
        vals = drawdowns.get(reason, [])
        if not vals:
            continue
        lines.append(f"  {reason:<20}  {len(vals):>4}  {_distribution(vals)}")

    lines.append("")
    lines.append("  Interpretation: a wide drawdown distribution on stop_loss means the")
    lines.append("  stop fires inconsistently — some positions barely dip, others crater.")
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _run(lookback_days: int, output: Optional[str]) -> None:
    if not _DB_PATH.exists():
        print(f"DB not found: {_DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    try:
        cutoff = None
        if lookback_days > 0:
            cutoff = (
                datetime.now(timezone.utc) - timedelta(days=lookback_days)
            ).strftime("%Y-%m-%dT%H:%M:%S")

        trades, snaps = _load(conn, cutoff)
    finally:
        conn.close()

    window = f"last {lookback_days} days" if lookback_days > 0 else "all time"
    exited  = sum(1 for t in trades.values() if t.exit_reason is not None)
    settled = sum(1 for t in trades.values() if t.outcome is not None)
    with_snaps = len(snaps)

    lines = [
        f"=== Exit Strategy Analysis  ({window}) ===",
        f"  Trades in window        : {len(trades)}",
        f"  Trades with early exit  : {exited}",
        f"  Trades with settlement  : {settled}",
        f"  Trades with snapshots   : {with_snaps}",
    ]

    lines += _recovery_rate(trades)
    lines += _wasted_exits(trades)
    lines += _hold_vs_exit(trades)
    lines += _peak_distribution(trades, snaps)
    lines += _threshold_sim(trades, snaps)
    lines += _drawdown_at_exit(trades, snaps)

    report = "\n".join(lines) + "\n"

    if output:
        Path(output).write_text(report, encoding="utf-8")
        print(f"Report written to {output}")
    else:
        print(report)


if __name__ == "__main__":
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    out  = sys.argv[2]     if len(sys.argv) > 2 else None
    _run(days, out)
