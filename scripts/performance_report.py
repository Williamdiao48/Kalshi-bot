#!/usr/bin/env python3
"""Quantitative performance report: Brier score, calibration, Sharpe, edge by source.

Excludes trades marked bug_loss=1.

Usage:
  venv/bin/python scripts/performance_report.py              # all-time, dry_run
  venv/bin/python scripts/performance_report.py 30           # last 30 days
  venv/bin/python scripts/performance_report.py 0 both       # all-time, all modes
"""
from __future__ import annotations

import math
import sqlite3
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).parent.parent
_DB   = _ROOT / "data" / "db" / "opportunity_log.db"


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Trade:
    id:              int
    logged_at:       str
    ticker:          str
    side:            str           # 'yes' | 'no'
    count:           int
    limit_price:     int           # YES-equivalent price 0–100¢
    source:          str
    kind:            str           # opportunity_kind
    p_estimate:      Optional[float]  # bot's P(win) fed to Kelly
    market_p_entry:  Optional[float]  # (yes_bid + yes_ask) / 200 at entry
    outcome:         Optional[str]    # 'won' | 'lost' | None (early exit)
    exit_pnl:        Optional[float]  # signed cents if early exit
    exit_reason:     Optional[str]

    # -- Derived ---------------------------------------------------------------

    @property
    def cost_cents(self) -> float:
        """Capital at risk: what we actually paid."""
        if self.side == "yes":
            return self.limit_price * self.count
        return (100 - self.limit_price) * self.count

    @property
    def pnl_cents(self) -> float:
        if self.exit_pnl is not None:
            return self.exit_pnl
        if self.outcome == "won":
            gain = (100 - self.limit_price) if self.side == "yes" else self.limit_price
            return gain * self.count
        if self.outcome == "lost":
            loss = self.limit_price if self.side == "yes" else (100 - self.limit_price)
            return -loss * self.count
        return 0.0

    @property
    def is_win(self) -> bool:
        if self.exit_pnl is not None:
            return self.exit_pnl > 0
        return self.outcome == "won"

    @property
    def roi(self) -> Optional[float]:
        """Return on capital-at-risk for this trade."""
        c = self.cost_cents
        return self.pnl_cents / c if c > 0 else None

    @property
    def market_p_win(self) -> Optional[float]:
        """Market's implied P(our side wins) at entry."""
        if self.market_p_entry is None:
            return None
        return self.market_p_entry if self.side == "yes" else (1.0 - self.market_p_entry)

    @property
    def edge(self) -> Optional[float]:
        """Our claimed alpha: p_estimate minus market-implied P(win).
        Positive = we think we're better than the market; should correlate with actual wins."""
        if self.p_estimate is None or self.market_p_win is None:
            return None
        return self.p_estimate - self.market_p_win


# ── DB query ──────────────────────────────────────────────────────────────────

def load_trades(db: Path, mode: str, lookback_days: int) -> list[Trade]:
    if not db.exists():
        print(f"DB not found: {db}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(str(db))
    mode_clause = (
        "mode IN ('dry_run','live')" if mode == "both"
        else f"mode = '{mode}'"
    )
    cutoff_clause = ""
    if lookback_days > 0:
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=lookback_days)
        ).strftime("%Y-%m-%dT%H:%M:%S")
        cutoff_clause = f"AND logged_at >= '{cutoff}'"

    sql = f"""
    SELECT id, logged_at, ticker, side, count, limit_price,
           COALESCE(source, 'unknown'),
           COALESCE(opportunity_kind, 'unknown'),
           p_estimate, market_p_entry, outcome,
           exit_pnl_cents, exit_reason
    FROM trades
    WHERE {mode_clause}
      AND COALESCE(bug_loss, 0) = 0
      AND (outcome IN ('won', 'lost') OR exited_at IS NOT NULL)
      AND outcome IS NOT 'void'
      AND status NOT IN ('rejected', 'error')
      {cutoff_clause}
    ORDER BY logged_at ASC
    """
    rows = conn.execute(sql).fetchall()
    conn.close()

    result = []
    for r in rows:
        (id_, logged_at, ticker, side, count, limit_price,
         source, kind, p_estimate, market_p_entry, outcome,
         exit_pnl, exit_reason) = r
        result.append(Trade(
            id=id_, logged_at=logged_at, ticker=ticker, side=side,
            count=count, limit_price=limit_price, source=source, kind=kind,
            p_estimate=p_estimate, market_p_entry=market_p_entry,
            outcome=outcome, exit_pnl=exit_pnl, exit_reason=exit_reason,
        ))
    return result


# ── Risk metrics ──────────────────────────────────────────────────────────────

def risk_metrics(trades: list[Trade]) -> dict:
    rois = [t.roi for t in trades if t.roi is not None]
    if len(rois) < 3:
        return {}

    # Annualise based on actual date span of the trades in scope.
    try:
        t0 = datetime.fromisoformat(trades[0].logged_at.replace("Z", "+00:00"))
        t1 = datetime.fromisoformat(trades[-1].logged_at.replace("Z", "+00:00"))
        years = max((t1 - t0).total_seconds() / 31_557_600, 1 / 365)
        tpy = len(rois) / years      # trades per year
    except Exception:
        tpy = 365.0

    mean_r = statistics.mean(rois)
    std_r  = statistics.pstdev(rois)
    sharpe = (mean_r / std_r * math.sqrt(tpy)) if std_r > 0 else None

    neg     = [r for r in rois if r < 0]
    dn_std  = statistics.pstdev(neg) if len(neg) >= 2 else 0.0
    sortino = (mean_r / dn_std * math.sqrt(tpy)) if dn_std > 0 else None

    pnls   = [t.pnl_cents for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    pf     = sum(wins) / abs(sum(losses)) if losses else None

    # Max drawdown on cumulative net-P&L equity curve.
    equity = 0.0; peak = 0.0; max_dd = 0.0
    for t in trades:
        equity += t.pnl_cents
        peak    = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak)

    return {
        "n":             len(trades),
        "sharpe":        sharpe,
        "sortino":       sortino,
        "max_dd":        max_dd,
        "profit_factor": pf,
        "avg_gain":      statistics.mean(wins)   if wins   else 0.0,
        "avg_loss":      statistics.mean(losses) if losses else 0.0,
        "mean_roi":      mean_r,
        "n_wins":        len(wins),
        "n_losses":      len(losses),
    }


# ── Calibration ───────────────────────────────────────────────────────────────

def brier_score(trades: list[Trade]) -> Optional[float]:
    """Lower = better. 0 = perfect, 0.25 = random guessing."""
    scored = [(t.p_estimate, int(t.is_win)) for t in trades if t.p_estimate is not None]
    if not scored:
        return None
    return sum((p - o) ** 2 for p, o in scored) / len(scored)


_CAL_BUCKETS = ["< 0.60", "0.60–0.70", "0.70–0.80", "0.80–0.90", "0.90–0.95", "≥ 0.95"]

def _cal_key(p: float) -> str:
    if p < 0.60: return "< 0.60"
    if p < 0.70: return "0.60–0.70"
    if p < 0.80: return "0.70–0.80"
    if p < 0.90: return "0.80–0.90"
    if p < 0.95: return "0.90–0.95"
    return "≥ 0.95"


def calibration_table(trades: list[Trade]) -> list[tuple]:
    """Return (label, n, mean_p, actual_wr, diff) per bucket with any data."""
    buckets: dict[str, list[Trade]] = defaultdict(list)
    for t in trades:
        if t.p_estimate is not None:
            buckets[_cal_key(t.p_estimate)].append(t)

    rows = []
    for label in _CAL_BUCKETS:
        ts = buckets.get(label, [])
        if not ts:
            continue
        mean_p    = statistics.mean(t.p_estimate for t in ts)  # type: ignore[arg-type]
        actual_wr = sum(1 for t in ts if t.is_win) / len(ts)
        diff      = actual_wr - mean_p  # positive = beating expectations
        rows.append((label, len(ts), mean_p, actual_wr, diff))
    return rows


# ── Per-group aggregation ─────────────────────────────────────────────────────

@dataclass
class GroupStats:
    pnls:       list[float] = field(default_factory=list)
    edges:      list[float] = field(default_factory=list)
    p_ests:     list[float] = field(default_factory=list)
    outcomes:   list[int]   = field(default_factory=list)  # 1=win 0=loss

    def add(self, t: Trade) -> None:
        self.pnls.append(t.pnl_cents)
        if t.edge is not None:
            self.edges.append(t.edge)
        if t.p_estimate is not None:
            self.p_ests.append(t.p_estimate)
            self.outcomes.append(int(t.is_win))

    @property
    def n(self) -> int: return len(self.pnls)

    @property
    def win_rate(self) -> Optional[float]:
        w = sum(1 for p in self.pnls if p > 0)
        return w / self.n if self.n > 0 else None

    @property
    def total_pnl(self) -> float: return sum(self.pnls)

    @property
    def avg_pnl(self) -> float:
        return statistics.mean(self.pnls) if self.pnls else 0.0

    @property
    def avg_edge(self) -> Optional[float]:
        return statistics.mean(self.edges) if self.edges else None

    @property
    def brier(self) -> Optional[float]:
        if not self.p_ests:
            return None
        return sum((p - o) ** 2 for p, o in zip(self.p_ests, self.outcomes)) / len(self.p_ests)


# ── Formatting helpers ────────────────────────────────────────────────────────

def _pnl(c: float) -> str:
    return f"{'+' if c >= 0 else ''}${c / 100:.2f}"

def _pct(v: Optional[float], decimals: int = 1) -> str:
    return f"{v * 100:.{decimals}f}%" if v is not None else "n/a"

def _f(v: Optional[float], fmt: str = ".2f") -> str:
    return f"{v:{fmt}}" if v is not None else "n/a"


# ── Report builder ────────────────────────────────────────────────────────────

def build_report(trades: list[Trade], mode: str, lookback_days: int) -> str:
    window = f"last {lookback_days} days" if lookback_days > 0 else "all-time"
    lines: list[str] = []

    total_pnl = sum(t.pnl_cents for t in trades)
    n_wins = sum(1 for t in trades if t.is_win)
    n = len(trades)

    # ── Header ────────────────────────────────────────────────────────────────
    lines += [
        f"Performance Report  ({window}, mode={mode}, bug_loss excluded)",
        "=" * 60,
        f"  Trades        : {n}  ({n_wins}W / {n - n_wins}L)",
        f"  Win rate      : {_pct(n_wins / n if n else None)}",
        f"  Net P&L       : {_pnl(total_pnl)}",
        f"  Avg per trade : {_pnl(total_pnl / n if n else 0)}",
    ]

    # ── Risk-adjusted returns ─────────────────────────────────────────────────
    rm = risk_metrics(trades)
    if rm:
        lines += [
            "",
            "Risk-Adjusted Returns",
            "-" * 40,
            f"  Sharpe ratio   : {_f(rm['sharpe'])}",
            "    (annualized; > 1.0 = good, > 2.0 = excellent)",
            f"  Sortino ratio  : {_f(rm['sortino'])}",
            "    (penalizes downside-only; better metric for skewed payoffs)",
            f"  Max drawdown   : -{_pct(rm['max_dd'])}  (peak-to-trough on net P&L curve)",
            f"  Profit factor  : {_f(rm['profit_factor'])}  (gross wins / gross losses; > 1.5 = solid)",
            f"  Avg gain       : {_pnl(rm['avg_gain'])}  per winning trade",
            f"  Avg loss       : {_pnl(rm['avg_loss'])}  per losing trade",
            f"  Mean ROI/trade : {_pct(rm['mean_roi'])}  (return on capital at risk)",
        ]

    # ── Calibration ───────────────────────────────────────────────────────────
    bs  = brier_score(trades)
    cal = calibration_table(trades)

    lines += ["", "Calibration", "-" * 40]

    if bs is not None:
        qual = (
            "Excellent" if bs < 0.05 else
            "Good"      if bs < 0.10 else
            "Moderate (likely overconfident)" if bs < 0.15 else
            "Poor — p_estimate miscalibrated"
        )
        lines.append(f"  Brier score : {bs:.4f}  →  {qual}")
        lines.append("    (0 = perfect, 0.25 = coin-flip; lower is better)")
    else:
        lines.append("  Brier score : n/a  (no p_estimate recorded for resolved trades)")

    if cal:
        lines += [
            "",
            "  Calibration curve  (actual win rate vs. bot's p_estimate):",
            f"  {'p_estimate':>10}  {'n':>5}  {'mean_p':>7}  {'actual':>7}  {'diff':>7}",
            f"  {'-'*10}  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*7}",
        ]
        for label, n_b, mean_p, actual_wr, diff in cal:
            flag = (
                "✓" if abs(diff) < 0.05 else
                "↑ under-conf." if diff > 0 else
                "↓ over-conf."
            )
            lines.append(
                f"  {label:>10}  {n_b:>5}  {mean_p:>6.1%}  {actual_wr:>6.1%}  "
                f"{'+' if diff >= 0 else ''}{diff:>6.1%}  {flag}"
            )
        lines += [
            "",
            "  A well-calibrated bot's curve is a straight line:",
            "  p_estimate 80% → actual win rate ~80%.",
            "  Persistent over-confidence → Kelly over-sizes → larger drawdowns.",
        ]

    # ── By source ─────────────────────────────────────────────────────────────
    by_source: dict[str, GroupStats] = defaultdict(GroupStats)
    for t in trades:
        by_source[t.source].add(t)

    lines += ["", "By Source", "-" * 40]
    w_src = max((len(s) for s in by_source), default=10)
    hdr = f"  {'source':<{w_src}}  {'n':>4}  {'win%':>5}  {'net P&L':>9}  {'avg/trade':>9}  {'avg edge':>9}  {'brier':>6}"
    lines += [hdr, "  " + "-" * (len(hdr) - 2)]

    for src, ss in sorted(by_source.items(), key=lambda x: -x[1].total_pnl):
        wr    = f"{ss.win_rate * 100:.0f}%" if ss.win_rate is not None else "n/a"
        edge  = f"{ss.avg_edge * 100:+.1f}%" if ss.avg_edge is not None else "  n/a"
        brier = f"{ss.brier:.3f}"            if ss.brier   is not None else "  n/a"
        lines.append(
            f"  {src:<{w_src}}  {ss.n:>4}  {wr:>5}  {_pnl(ss.total_pnl):>9}"
            f"  {_pnl(ss.avg_pnl):>9}  {edge:>9}  {brier:>6}"
        )

    lines += [
        "",
        "  avg edge = our p_estimate minus market-implied P(win at entry).",
        "  Positive edge on winning sources → real information alpha.",
        "  High brier on a source → its p_estimate is mis-calibrated (resize Kelly).",
    ]

    # ── By signal type ────────────────────────────────────────────────────────
    by_kind: dict[str, GroupStats] = defaultdict(GroupStats)
    for t in trades:
        by_kind[t.kind].add(t)

    lines += ["", "By Signal Type", "-" * 40]
    w_kind = max((len(k) for k in by_kind), default=10)
    hdr2 = f"  {'kind':<{w_kind}}  {'n':>4}  {'win%':>5}  {'net P&L':>9}  {'avg/trade':>9}  {'brier':>6}"
    lines += [hdr2, "  " + "-" * (len(hdr2) - 2)]

    for kind, ss in sorted(by_kind.items(), key=lambda x: -x[1].total_pnl):
        wr    = f"{ss.win_rate * 100:.0f}%" if ss.win_rate is not None else "n/a"
        brier = f"{ss.brier:.3f}"           if ss.brier   is not None else "  n/a"
        lines.append(
            f"  {kind:<{w_kind}}  {ss.n:>4}  {wr:>5}  {_pnl(ss.total_pnl):>9}"
            f"  {_pnl(ss.avg_pnl):>9}  {brier:>6}"
        )

    # ── YES vs NO ─────────────────────────────────────────────────────────────
    by_side: dict[str, GroupStats] = defaultdict(GroupStats)
    for t in trades:
        by_side[t.side.upper()].add(t)

    lines += ["", "YES vs NO", "-" * 40]
    for side in ["YES", "NO"]:
        ss = by_side.get(side)
        if ss:
            wr = f"{ss.win_rate * 100:.0f}%" if ss.win_rate is not None else "n/a"
            lines.append(
                f"  {side}:  {ss.n} trades,  {wr} win rate,"
                f"  {_pnl(ss.total_pnl)} net,  {_pnl(ss.avg_pnl)} avg/trade"
            )

    lines.append("")
    return "\n".join(lines)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    lookback = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    mode     = sys.argv[2]     if len(sys.argv) > 2 else "dry_run"

    trades = load_trades(_DB, mode, lookback)
    if not trades:
        window = f"last {lookback} days" if lookback > 0 else "all-time"
        print(f"No resolved trades found ({window}, mode={mode}, bug_loss excluded).")
        return

    print(build_report(trades, mode, lookback))


if __name__ == "__main__":
    main()
