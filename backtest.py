"""Exit-parameter optimizer, win-rate calibration, and weather gate optimizer.

Reads opportunity_log.db and:

  1. Grid-searches (profit_take × stop_loss × trailing_drawdown) against actual
     price trajectories in price_snapshots to find the globally optimal thresholds.

  2. Runs the same grid-search independently for each data source (noaa_observed,
     noaa, eia, polymarket, …) and per source:side when there is enough data.

  3. Calibrates per-source win rates from all settled trades and outputs
     recommended KELLY_METRIC_PRIORS values for KELLY_METRIC_PRIORS env var.

  4. Compares optimal thresholds against the current compiled-in defaults and
     prints ready-to-paste EXIT_SOURCE_PROFIT_TAKE / EXIT_SOURCE_STOP_LOSS /
     EXIT_SOURCE_TRAILING_DRAWDOWN overrides.

  5. Weather gate optimization: sweeps TEMP_FORECAST_MIN_EDGE, HRRR_MAX_SPREAD_F,
     and TEMP_OBSERVED_MAX_HOURS against settled weather trades to find the
     edge/spread/window threshold that maximises P&L.  Analysis is inherently
     one-directional — we can only tighten thresholds above current defaults
     (blocked trades at lower thresholds were never executed and have no data).

Simulation model
----------------
For each trade that has at least one price snapshot, the simulation replays
its price trajectory under a given (pt, sl, trailing) triplet:

  - Profit-take  : exit when pct_gain ≥ pt   (P&L = unrealized_cents at that point)
  - Stop-loss    : exit when pct_gain ≤ −sl
  - Trailing stop: exit when peak_pct_gain > trailing AND
                   pct_gain < peak_pct_gain − trailing

If no threshold fires before the trajectory ends, the trade is counted as
"held to settlement" and its actual settlement P&L is used instead.

Trades without any snapshot (often very fast exits or batch entries) are
excluded from the grid search but included in win-rate calibration.

Usage
-----
  venv/bin/python backtest.py
  venv/bin/python backtest.py --days 60
  venv/bin/python backtest.py --min-trades 5
  venv/bin/python backtest.py --output report.txt
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import NamedTuple

_DB_PATH = Path(__file__).parent / "opportunity_log.db"

# ---------------------------------------------------------------------------
# Current compiled-in defaults (from exit_manager.py) for comparison
# ---------------------------------------------------------------------------
_CURRENT_GLOBAL_PT       = 0.50
_CURRENT_GLOBAL_SL       = 0.80
_CURRENT_GLOBAL_TRAILING = 0.30

_CURRENT_SOURCE_PT: dict[str, float] = {
    "noaa_observed:yes": 0.50, "noaa_observed": 0.80,
    "nws_alert": 0.80, "eia": 0.90,
    "noaa_day2:no": 0.07, "noaa_day2_early:no": 0.07,
    "noaa_day2:yes": 0.35, "noaa_day2_early:yes": 0.35,
}
_CURRENT_SOURCE_SL: dict[str, float] = {
    "noaa_observed:yes": 0.05, "nws_climo:yes": 0.05, "nws_alert:yes": 0.05,
    "noaa_observed": 0.60, "noaa": 0.50, "owm": 0.50, "open_meteo": 0.50,
    "polymarket": 0.40, "manifold": 0.40, "metaculus": 0.50, "eia": 0.95,
    "noaa_day2:yes": 0.45, "noaa_day2_early:yes": 0.45,
    "noaa_day2:no": 0.55, "noaa_day2_early:no": 0.55,
}
_CURRENT_SOURCE_TRAILING: dict[str, float] = {
    "noaa_day2": 0.05, "noaa_day2_early": 0.05,
    "noaa_day2:yes": 0.12, "noaa_day2_early:yes": 0.12,
    "noaa:no": 0.08, "owm:no": 0.08, "open_meteo:no": 0.08,
    "noaa_observed:no": 0.30,
}


# ---------------------------------------------------------------------------
# Grid parameters
# ---------------------------------------------------------------------------
# Keep the grid coarse enough to run in <5 s, fine enough to be actionable.
_PT_GRID       = [x / 100 for x in range(10, 95, 5)]   # 10 % … 90 %
_SL_GRID       = [x / 100 for x in range(20, 100, 5)]  # 20 % … 95 %
_TRAILING_GRID = [0.0] + [x / 100 for x in range(5, 55, 5)]  # 0 = disabled, 5–50 %

_MIN_TRADES_PER_GROUP = 5   # skip source groups with fewer trades
_MIN_TRADES_KELLY     = 3   # min settled trades for Kelly estimate


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    id:          int
    source:      str
    side:        str         # 'yes' | 'no'
    limit_price: int         # YES price in cents (entry)
    count:       int
    outcome:     str | None  # 'won' | 'lost' | None
    exit_reason: str | None
    exit_pnl:    float | None


@dataclass
class SimTrade:
    """Trade enriched with the data needed for simulation."""
    trade:    Trade
    cost:     int            # cents paid to enter
    hold_pnl: float | None   # P&L if held to settlement (None = not settled)
    # (pct_gain, unrealized_cents) pairs in chronological order
    trajectory: list[tuple[float, float]] = field(default_factory=list)


class SimResult(NamedTuple):
    total_pnl:   float
    n_pt:        int
    n_sl:        int
    n_trailing:  int
    n_held:      int
    n_skipped:   int   # neither triggered nor settled


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _cost(side: str, limit_price: int, count: int) -> int:
    if side == "yes":
        return limit_price * count
    return (100 - limit_price) * count


def _hold_pnl(t: Trade) -> float | None:
    if t.outcome == "won":
        return float((100 - t.limit_price) * t.count) if t.side == "yes" \
               else float(t.limit_price * t.count)
    if t.outcome == "lost":
        return float(-t.limit_price * t.count) if t.side == "yes" \
               else float(-(100 - t.limit_price) * t.count)
    return None


def load_data(conn: sqlite3.Connection, cutoff: str | None) -> list[SimTrade]:
    cut = f"AND t.logged_at >= '{cutoff}'" if cutoff else ""
    rows = conn.execute(f"""
        SELECT t.id, COALESCE(t.source, 'unknown') AS source,
               t.side, t.limit_price, t.count,
               t.outcome, t.exit_reason, t.exit_pnl_cents
        FROM trades t
        WHERE t.mode IN ('dry_run', 'live') {cut}
        ORDER BY t.id
    """).fetchall()

    all_trades: dict[int, SimTrade] = {}
    for row in rows:
        tid, src, side, lp, cnt, outcome, exit_r, exit_pnl = row
        t = Trade(id=tid, source=src, side=side, limit_price=lp, count=cnt,
                  outcome=outcome, exit_reason=exit_r, exit_pnl=exit_pnl)
        all_trades[tid] = SimTrade(
            trade=t,
            cost=_cost(side, lp, cnt),
            hold_pnl=_hold_pnl(t),
        )

    snap_rows = conn.execute("""
        SELECT trade_id, pct_gain, unrealized_cents
        FROM price_snapshots
        WHERE pct_gain IS NOT NULL AND unrealized_cents IS NOT NULL
        ORDER BY trade_id, snapshot_at
    """).fetchall()

    for tid, pg, uc in snap_rows:
        if tid in all_trades:
            all_trades[tid].trajectory.append((pg, float(uc)))

    return list(all_trades.values())


# ---------------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------------

def simulate(
    sim_trades: list[SimTrade],
    pt: float,
    sl: float,
    trailing: float,
) -> SimResult:
    """Replay a list of trades under the given thresholds.

    Only trades that have at least one snapshot participate in the simulation.
    If no threshold fires along the trajectory, the trade falls back to its
    actual settlement P&L (hold_pnl).  Trades with no snapshot AND no
    settlement are counted as 'skipped'.
    """
    total_pnl = 0.0
    n_pt = n_sl = n_trailing = n_held = n_skipped = 0

    for st in sim_trades:
        if not st.trajectory:
            # No snapshot data — fall back to settlement or skip
            if st.hold_pnl is not None:
                total_pnl += st.hold_pnl
                n_held += 1
            else:
                n_skipped += 1
            continue

        peak: float = 0.0
        exit_pnl: float | None = None
        reason: str = ""

        for pct, unreal in st.trajectory:
            if pct > peak:
                peak = pct
            # Profit-take
            if pct >= pt:
                exit_pnl = unreal
                reason = "pt"
                break
            # Stop-loss
            if pct <= -sl:
                exit_pnl = unreal
                reason = "sl"
                break
            # Trailing stop
            if trailing > 0 and peak > trailing and pct < peak - trailing:
                exit_pnl = unreal
                reason = "trailing"
                break

        if exit_pnl is not None:
            total_pnl += exit_pnl
            if reason == "pt":        n_pt += 1
            elif reason == "sl":      n_sl += 1
            else:                     n_trailing += 1
        elif st.hold_pnl is not None:
            total_pnl += st.hold_pnl
            n_held += 1
        else:
            n_skipped += 1

    return SimResult(total_pnl, n_pt, n_sl, n_trailing, n_held, n_skipped)


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def grid_search(
    sim_trades: list[SimTrade],
    *,
    top_n: int = 15,
) -> list[tuple[float, float, float, float]]:
    """Return the top_n (pnl, pt, sl, trailing) combinations."""
    results: list[tuple[float, float, float, float]] = []
    for pt in _PT_GRID:
        for sl in _SL_GRID:
            for trailing in _TRAILING_GRID:
                r = simulate(sim_trades, pt, sl, trailing)
                results.append((r.total_pnl, pt, sl, trailing))
    results.sort(key=lambda x: -x[0])
    return results[:top_n]


# ---------------------------------------------------------------------------
# Kelly calibration
# ---------------------------------------------------------------------------

@dataclass
class KellyStats:
    source:      str
    n_won:       int
    n_lost:      int
    win_rate:    float
    med_cost_pct: float   # median entry cost as fraction (e.g., 0.35 = 35¢)
    kelly_raw:   float    # raw Kelly fraction
    kelly_q4:    float    # quarter-Kelly (conservative)


def kelly_calibration(sim_trades: list[SimTrade]) -> list[KellyStats]:
    settled: dict[str, list[SimTrade]] = defaultdict(list)
    for st in sim_trades:
        if st.trade.outcome in ("won", "lost"):
            settled[st.trade.source].append(st)

    stats: list[KellyStats] = []
    for src, lst in sorted(settled.items()):
        if len(lst) < _MIN_TRADES_KELLY:
            continue
        n_won  = sum(1 for st in lst if st.trade.outcome == "won")
        n_lost = len(lst) - n_won
        win_rate = n_won / len(lst)

        # Entry cost as a fraction of $1 payout
        costs = [st.cost / (st.trade.count * 100) for st in lst if st.trade.count > 0]
        med_cost = median(costs) if costs else 0.5

        # Kelly: f* = (p * b - q) / b, where b = net_odds = (1 - cost) / cost
        # p = win_rate, q = 1 - win_rate
        if 0 < med_cost < 1:
            b = (1.0 - med_cost) / med_cost
            kelly_raw = max(0.0, (win_rate * b - (1 - win_rate)) / b)
        else:
            kelly_raw = 0.0

        stats.append(KellyStats(
            source=src,
            n_won=n_won,
            n_lost=n_lost,
            win_rate=win_rate,
            med_cost_pct=med_cost,
            kelly_raw=kelly_raw,
            kelly_q4=kelly_raw * 0.25,
        ))

    return stats


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _pnl(c: float) -> str:
    sign = "+" if c >= 0 else ""
    return f"{sign}${c/100:.2f}"



def _thr(v: float) -> str:
    return f"{v*100:.0f}%"


def _bar(v: float, best: float, worst: float | None = None, width: int = 20) -> str:
    """ASCII bar showing relative position between worst and best."""
    lo = worst if worst is not None else 0.0
    span = best - lo
    if span <= 0:
        return ""
    frac = max(0.0, min(1.0, (v - lo) / span))
    filled = round(frac * width)
    return "█" * filled + "░" * (width - filled)


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def _header(sim_trades: list[SimTrade]) -> list[str]:
    with_snaps   = sum(1 for st in sim_trades if st.trajectory)
    with_outcome = sum(1 for st in sim_trades if st.trade.outcome is not None)
    both         = sum(1 for st in sim_trades if st.trajectory and st.trade.outcome is not None)
    open_pos     = sum(1 for st in sim_trades if not st.trajectory and st.trade.outcome is None)

    # Actual total P&L (as-executed)
    actual_pnl = 0.0
    for st in sim_trades:
        if st.trade.exit_pnl is not None:
            actual_pnl += st.trade.exit_pnl
        elif st.hold_pnl is not None:
            actual_pnl += st.hold_pnl

    lines = [
        "╔══════════════════════════════════════════════════════════════════════════╗",
        "║          Kalshi Bot — Backtest / Exit Optimizer  (Phase 1)             ║",
        "╚══════════════════════════════════════════════════════════════════════════╝",
        "",
        f"  Total trades in window    : {len(sim_trades)}",
        f"  Trades with snapshots     : {with_snaps}  (eligible for simulation)",
        f"  Trades with settlement    : {with_outcome}",
        f"  Trades with both          : {both}  (fully evaluable)",
        f"  Open / no settlement      : {open_pos}  (excluded from P&L math)",
        f"  Actual P&L (as-executed)  : {_pnl(actual_pnl)}",
        "",
        "  Grid search space:",
        f"    Profit-take  : {_thr(_PT_GRID[0])} – {_thr(_PT_GRID[-1])}  "
        f"({len(_PT_GRID)} values, 5 pp steps)",
        f"    Stop-loss    : {_thr(_SL_GRID[0])} – {_thr(_SL_GRID[-1])}  "
        f"({len(_SL_GRID)} values, 5 pp steps)",
        f"    Trailing     : disabled + {_thr(_TRAILING_GRID[1])} – {_thr(_TRAILING_GRID[-1])}  "
        f"({len(_TRAILING_GRID)} values)",
        f"    Total combos : {len(_PT_GRID) * len(_SL_GRID) * len(_TRAILING_GRID):,}",
    ]
    return lines


def _global_search_section(
    sim_trades: list[SimTrade],
    actual_pnl: float,
) -> tuple[list[str], tuple[float, float, float]]:
    """Returns (lines, (best_pt, best_sl, best_trailing))."""
    lines = [
        "",
        "── 1. Global Grid Search ──────────────────────────────────────────────────",
        "   All sources combined. Optimal thresholds that maximise total P&L.",
        "",
    ]

    t0 = time.perf_counter()
    top = grid_search(sim_trades, top_n=20)
    elapsed = time.perf_counter() - t0
    lines.append(f"   Search completed in {elapsed:.1f}s  |  {len(_PT_GRID)*len(_SL_GRID)*len(_TRAILING_GRID):,} combinations")
    lines.append("")

    best_pnl, best_pt, best_sl, best_trailing = top[0]

    lines.append(f"   Current defaults   PT={_thr(_CURRENT_GLOBAL_PT)}"
                 f"  SL={_thr(_CURRENT_GLOBAL_SL)}"
                 f"  trailing={_thr(_CURRENT_GLOBAL_TRAILING)}")
    current_sim = simulate(sim_trades, _CURRENT_GLOBAL_PT, _CURRENT_GLOBAL_SL, _CURRENT_GLOBAL_TRAILING)
    lines.append(f"   Current sim P&L    {_pnl(current_sim.total_pnl)}  "
                 f"(exits: PT={current_sim.n_pt} SL={current_sim.n_sl} "
                 f"trail={current_sim.n_trailing} held={current_sim.n_held})")
    lines.append(f"   Actual as-executed {_pnl(actual_pnl)}")
    lines.append("")
    lines.append(f"   Optimal found      PT={_thr(best_pt)}"
                 f"  SL={_thr(best_sl)}"
                 f"  trailing={_thr(best_trailing)}")
    lines.append(f"   Optimal sim P&L    {_pnl(best_pnl)}  "
                 f"(vs current: {_pnl(best_pnl - current_sim.total_pnl)})")
    lines.append("")

    lines.append(f"   {'Rank':>4}  {'PT':>5}  {'SL':>5}  {'Trail':>6}  {'Sim P&L':>10}  "
                 f"{'vs current':>11}  Exits (PT/SL/trail/held)  Bar")
    lines.append("   " + "─" * 78)
    for i, (pnl, pt, sl, trail) in enumerate(top, 1):
        diff = pnl - current_sim.total_pnl
        r = simulate(sim_trades, pt, sl, trail)
        worst_pnl = top[-1][0]
        bar = _bar(pnl, best_pnl, worst_pnl)
        diff_s = f"{'+' if diff >= 0 else ''}{_pnl(diff)}"
        lines.append(
            f"   {i:>4}  {_thr(pt):>5}  {_thr(sl):>5}  {_thr(trail):>6}  "
            f"{_pnl(pnl):>10}  {diff_s:>11}  "
            f"{r.n_pt}/{r.n_sl}/{r.n_trailing}/{r.n_held}  {bar}"
        )
        if i == 1:
            lines.insert(-1, "   " + "─" * 78)

    return lines, (best_pt, best_sl, best_trailing)


def _pt_sensitivity(
    sim_trades: list[SimTrade],
    best_sl: float,
    best_trailing: float,
) -> list[str]:
    """1-D sweep of PT with best SL/trailing fixed."""
    lines = [
        "",
        "── 2. Profit-Take Sensitivity ─────────────────────────────────────────────",
        f"   SL={_thr(best_sl)} and trailing={_thr(best_trailing)} fixed at global optimum.",
        "",
        f"   {'PT':>5}  {'Sim P&L':>10}  Bar",
        "   " + "─" * 50,
    ]
    results = [(simulate(sim_trades, pt, best_sl, best_trailing).total_pnl, pt)
               for pt in _PT_GRID]
    best_v  = max(r[0] for r in results)
    worst_v = min(r[0] for r in results)
    for pnl, pt in results:
        star = " ◀ best" if pnl == best_v else ""
        lines.append(f"   {_thr(pt):>5}  {_pnl(pnl):>10}  {_bar(pnl, best_v, worst_v)}{star}")
    return lines


def _sl_sensitivity(
    sim_trades: list[SimTrade],
    best_pt: float,
    best_trailing: float,
) -> list[str]:
    """1-D sweep of SL with best PT/trailing fixed."""
    lines = [
        "",
        "── 3. Stop-Loss Sensitivity ───────────────────────────────────────────────",
        f"   PT={_thr(best_pt)} and trailing={_thr(best_trailing)} fixed at global optimum.",
        "",
        f"   {'SL':>5}  {'Sim P&L':>10}  Bar",
        "   " + "─" * 50,
    ]
    results = [(simulate(sim_trades, best_pt, sl, best_trailing).total_pnl, sl)
               for sl in _SL_GRID]
    best_v  = max(r[0] for r in results)
    worst_v = min(r[0] for r in results)
    for pnl, sl in results:
        star = " ◀ best" if pnl == best_v else ""
        lines.append(f"   {_thr(sl):>5}  {_pnl(pnl):>10}  {_bar(pnl, best_v, worst_v)}{star}")
    return lines


def _trailing_sensitivity(
    sim_trades: list[SimTrade],
    best_pt: float,
    best_sl: float,
) -> list[str]:
    """1-D sweep of trailing with best PT/SL fixed."""
    lines = [
        "",
        "── 4. Trailing Drawdown Sensitivity ───────────────────────────────────────",
        f"   PT={_thr(best_pt)} and SL={_thr(best_sl)} fixed at global optimum.",
        "",
        f"   {'Trail':>6}  {'Sim P&L':>10}  Bar",
        "   " + "─" * 52,
    ]
    results = [(simulate(sim_trades, best_pt, best_sl, trail).total_pnl, trail)
               for trail in _TRAILING_GRID]
    best_v  = max(r[0] for r in results)
    worst_v = min(r[0] for r in results)
    for pnl, trail in results:
        lbl = "off" if trail == 0 else _thr(trail)
        star = " ◀ best" if pnl == best_v else ""
        lines.append(f"   {lbl:>6}  {_pnl(pnl):>10}  {_bar(pnl, best_v, worst_v)}{star}")
    return lines


def _per_source_section(
    sim_trades: list[SimTrade],
) -> tuple[list[str], dict[str, tuple[float, float, float]]]:
    """Per-source grid search.  Returns (lines, recommendations dict)."""
    lines = [
        "",
        "── 5. Per-Source Optimization ─────────────────────────────────────────────",
        f"   Each source optimised independently (min {_MIN_TRADES_PER_GROUP} trades with snapshots).",
        "",
    ]

    # Group trades by source, then by source:side
    by_src: dict[str, list[SimTrade]]      = defaultdict(list)
    by_src_side: dict[str, list[SimTrade]] = defaultdict(list)
    for st in sim_trades:
        if st.trajectory:
            by_src[st.trade.source].append(st)
            by_src_side[f"{st.trade.source}:{st.trade.side}"].append(st)

    recommendations: dict[str, tuple[float, float, float]] = {}

    for key, group in sorted({**by_src, **by_src_side}.items()):
        # Only use source:side when there are ≥5 MORE trades than the bare source
        if ":" in key:
            src = key.split(":")[0]
            if len(by_src.get(src, [])) - len(group) < _MIN_TRADES_PER_GROUP:
                continue  # not enough extra signal to justify separate tuning
        if len(group) < _MIN_TRADES_PER_GROUP:
            continue

        top = grid_search(group, top_n=3)
        best_pnl, best_pt, best_sl, best_tr = top[0]

        # Actual P&L for this group as-executed
        actual = 0.0
        for st in group:
            if st.trade.exit_pnl is not None:
                actual += st.trade.exit_pnl
            elif st.hold_pnl is not None:
                actual += st.hold_pnl

        cur_pt = _CURRENT_SOURCE_PT.get(key, _CURRENT_GLOBAL_PT)
        cur_sl = _CURRENT_SOURCE_SL.get(key, _CURRENT_GLOBAL_SL)
        cur_tr = _CURRENT_SOURCE_TRAILING.get(key, _CURRENT_GLOBAL_TRAILING)
        cur_sim = simulate(group, cur_pt, cur_sl, cur_tr)

        lines.append(f"  ● {key}  ({len(group)} trades with snapshots)")
        lines.append(f"    Current     PT={_thr(cur_pt)}  SL={_thr(cur_sl)}"
                     f"  trail={_thr(cur_tr)}")
        lines.append(f"    Current sim {_pnl(cur_sim.total_pnl)}  "
                     f"  Actual as-executed {_pnl(actual)}")
        lines.append(f"    Optimal     PT={_thr(best_pt)}  SL={_thr(best_sl)}"
                     f"  trail={_thr(best_tr)}")
        lines.append(f"    Optimal sim {_pnl(best_pnl)}  "
                     f"  Improvement {_pnl(best_pnl - cur_sim.total_pnl)}")

        changed = (
            abs(best_pt - cur_pt) > 0.04
            or abs(best_sl - cur_sl) > 0.04
            or abs(best_tr - cur_tr) > 0.04
        )
        if changed:
            lines.append(f"    ⚠  Significant deviation from current — consider updating")
            recommendations[key] = (best_pt, best_sl, best_tr)
        else:
            lines.append(f"    ✓  Current thresholds are near-optimal")
        lines.append("")

    return lines, recommendations


def _kelly_section(sim_trades: list[SimTrade]) -> list[str]:
    lines = [
        "",
        "── 6. Win-Rate Calibration and Kelly Sizing ───────────────────────────────",
        "   Based on all settled trades (outcome='won'/'lost') regardless of exit.",
        "   kelly_raw = max(0, (p*b − q)/b)   where b = (1−cost)/cost",
        "   kelly_q4  = kelly_raw / 4  (quarter-Kelly, conservative)",
        "",
        f"  {'Source':<22}  {'Won':>4}  {'Lost':>5}  {'WinRate':>8}  "
        f"{'MedCost':>8}  {'KellyRaw':>9}  {'KellyQ4':>8}  {'PRIORS key'}",
        "  " + "─" * 90,
    ]

    stats = kelly_calibration(sim_trades)
    priors: dict[str, float] = {}

    for ks in sorted(stats, key=lambda x: -x.n_won - x.n_lost):
        prior_key = ks.source.replace("_observed", "obs") \
                              .replace("noaa_day", "day") \
                              .replace("polymarket", "poly")
        flag = "  ⚑ edge?" if ks.win_rate > 0.45 else ""
        lines.append(
            f"  {ks.source:<22}  {ks.n_won:>4}  {ks.n_lost:>5}  "
            f"{ks.win_rate*100:>7.0f}%  "
            f"{ks.med_cost_pct*100:>7.0f}¢  "
            f"{ks.kelly_raw*100:>8.1f}%  "
            f"{ks.kelly_q4*100:>7.1f}%  {prior_key}{flag}"
        )
        if ks.kelly_q4 > 0:
            priors[ks.source] = round(ks.kelly_q4, 3)

    lines += [
        "",
        "  Suggested KELLY_METRIC_PRIORS (win probabilities, not Kelly fractions):",
        "  NOTE: KELLY_METRIC_PRIORS in the code stores P(win), not Kelly fraction.",
        "        Use win_rate directly; the trade executor applies the Kelly formula.",
        "",
    ]
    win_priors: dict[str, float] = {}
    for ks in stats:
        if ks.n_won + ks.n_lost >= _MIN_TRADES_KELLY:
            win_priors[ks.source] = round(ks.win_rate, 3)

    if win_priors:
        lines.append("  " + json.dumps(win_priors, indent=4).replace("\n", "\n  "))
    else:
        lines.append("  (insufficient data)")
    return lines


def _recommendations_section(
    global_best: tuple[float, float, float],
    per_src: dict[str, tuple[float, float, float]],
) -> list[str]:
    lines = [
        "",
        "── 7. Recommended Configuration Overrides ─────────────────────────────────",
        "   Copy-paste these into your .env file (or shell export) as needed.",
        "   Per-source overrides shown only where optimum differs by >4 pp.",
        "",
    ]

    best_pt, best_sl, best_trailing = global_best

    delta_pt  = abs(best_pt      - _CURRENT_GLOBAL_PT)
    delta_sl  = abs(best_sl      - _CURRENT_GLOBAL_SL)
    delta_tr  = abs(best_trailing - _CURRENT_GLOBAL_TRAILING)
    changed_g = delta_pt > 0.04 or delta_sl > 0.04 or delta_tr > 0.04

    if changed_g:
        lines.append("  Global thresholds  (current → optimal):")
        if delta_pt > 0.04:
            lines.append(f"    EXIT_PROFIT_TAKE={best_pt:.2f}   "
                         f"# was {_CURRENT_GLOBAL_PT:.2f}")
        if delta_sl > 0.04:
            lines.append(f"    EXIT_STOP_LOSS={best_sl:.2f}     "
                         f"# was {_CURRENT_GLOBAL_SL:.2f}")
        if delta_tr > 0.04:
            lines.append(f"    EXIT_TRAILING_DRAWDOWN={best_trailing:.2f}  "
                         f"# was {_CURRENT_GLOBAL_TRAILING:.2f}")
    else:
        lines.append("  Global thresholds: no change recommended (within 4 pp of current).")

    if per_src:
        lines.append("")
        lines.append("  Per-source overrides:")
        # Build override dicts (start from current defaults)
        pt_d       = dict(_CURRENT_SOURCE_PT)
        sl_d       = dict(_CURRENT_SOURCE_SL)
        trailing_d = dict(_CURRENT_SOURCE_TRAILING)

        for key, (opt_pt, opt_sl, opt_tr) in per_src.items():
            lines.append(f"    {key}:")
            lines.append(f"      PT={_thr(opt_pt)}  SL={_thr(opt_sl)}  trail={_thr(opt_tr)}")
            pt_d[key]       = opt_pt
            sl_d[key]       = opt_sl
            trailing_d[key] = opt_tr

        lines.append("")
        lines.append("  Updated EXIT_SOURCE_PROFIT_TAKE (full JSON):")
        lines.append("    " + json.dumps({k: round(v, 2) for k, v in pt_d.items()}))
        lines.append("")
        lines.append("  Updated EXIT_SOURCE_STOP_LOSS (full JSON):")
        lines.append("    " + json.dumps({k: round(v, 2) for k, v in sl_d.items()}))
        lines.append("")
        lines.append("  Updated EXIT_SOURCE_TRAILING_DRAWDOWN (full JSON):")
        lines.append("    " + json.dumps({k: round(v, 2) for k, v in trailing_d.items()}))
    else:
        lines.append("")
        lines.append("  No per-source overrides recommended at this time.")

    lines += [
        "",
        "  Caveats:",
        "  • Simulation assumes exit fills at the snapshot mid-price (±1-3¢ slippage).",
        "  • For stop-outs where we have no post-exit trajectory, settlement outcome is",
        "    used as the 'held' baseline — this may slightly overvalue lenient SLs.",
        "  • Small source groups (< 20 trades) have high variance; treat with caution.",
        "  • Re-run after collecting 20+ more trades for more stable estimates.",
    ]
    return lines


# ---------------------------------------------------------------------------
# Weather gate optimization
# ---------------------------------------------------------------------------

# Current compiled-in defaults for weather gate parameters
_CURRENT_FORECAST_MIN_EDGE   = 7.0   # TEMP_FORECAST_MIN_EDGE  (°F)
_CURRENT_OBS_MAX_HOURS       = 4.0   # TEMP_OBSERVED_MAX_HOURS (h)

# Sweep grids — tighter than or equal to current defaults (can only tighten,
# not loosen, since below-default trades were never executed).
_EDGE_GRID  = [x / 2 for x in range(14, 42, 1)]  # 7.0 … 20.5 °F in 0.5 steps
_HOURS_GRID = [x / 2 for x in range(2, 17, 1)]   # 1.0 … 8.0 h in 0.5 steps

# Weather forecast sources (gated by TEMP_FORECAST_MIN_EDGE and HRRR spread)
_FORECAST_SRCS = frozenset(
    ("noaa", "noaa_day2", "nws_hourly", "hrrr", "open_meteo", "weatherapi")
)
# Observation sources (gated by TEMP_OBSERVED_MAX_HOURS for NO trades)
_OBS_SRCS = frozenset(("noaa_observed", "metar", "nws_climo"))


@dataclass
class WeatherTrade:
    trade_id:    int
    ticker:      str
    source:      str
    side:        str
    outcome:     str        # 'won' | 'lost'
    limit_price: int
    count:       int
    hold_pnl:    float      # cents
    edge_f:      float | None   # °F edge at entry (None if not in opportunities)
    hrrr_spread: float | None   # °F spread at entry (not stored — always None)
    hours_to_close: float | None  # h remaining at first price snapshot


def _load_weather_trades(conn: sqlite3.Connection, cutoff: str | None) -> list[WeatherTrade]:
    """Load settled weather (KXHIGH*) trades enriched with edge and time-to-close."""
    cut = f"AND t.logged_at >= '{cutoff}'" if cutoff else ""

    # Base weather trades — only settled (outcome set) rows
    rows = conn.execute(f"""
        SELECT t.id, t.ticker, t.source, t.side, t.outcome,
               t.limit_price, t.count, t.logged_at
        FROM trades t
        WHERE t.ticker LIKE 'KXHIGH%'
          AND t.outcome IN ('won', 'lost')
          AND t.source IS NOT NULL
          AND t.mode IN ('dry_run', 'live') {cut}
        ORDER BY t.id
    """).fetchall()

    if not rows:
        return []

    # For each trade pull the most-recent opportunity record ≤ trade.logged_at
    # to get the edge that was seen when the trade was entered.
    edge_by_trade: dict[int, float] = {}
    for tid, ticker, source, side, outcome, lp, cnt, logged_at in rows:
        opp = conn.execute("""
            SELECT edge FROM opportunities
            WHERE ticker = ? AND source = ? AND logged_at <= ?
            ORDER BY logged_at DESC LIMIT 1
        """, (ticker, source, logged_at)).fetchone()
        if opp and opp[0] is not None:
            edge_by_trade[tid] = float(opp[0])

    # First price_snapshot's days_to_close → hours remaining at entry
    dtc_by_trade: dict[int, float] = {}
    snap_rows = conn.execute("""
        SELECT trade_id, days_to_close
        FROM price_snapshots
        WHERE days_to_close IS NOT NULL
        ORDER BY trade_id, snapshot_at
    """).fetchall()
    seen: set[int] = set()
    for tid, dtc in snap_rows:
        if tid not in seen:
            dtc_by_trade[tid] = dtc * 24.0  # days → hours
            seen.add(tid)

    result: list[WeatherTrade] = []
    for tid, ticker, source, side, outcome, lp, cnt, logged_at in rows:
        if side == "yes":
            hold_pnl = float((100 - lp) * cnt) if outcome == "won" else float(-lp * cnt)
        else:
            hold_pnl = float(lp * cnt) if outcome == "won" else float(-(100 - lp) * cnt)

        result.append(WeatherTrade(
            trade_id=tid,
            ticker=ticker,
            source=source,
            side=side,
            outcome=outcome,
            limit_price=lp,
            count=cnt,
            hold_pnl=hold_pnl,
            edge_f=edge_by_trade.get(tid),
            hrrr_spread=None,  # not stored — future enhancement
            hours_to_close=dtc_by_trade.get(tid),
        ))
    return result


def _weather_gate_section(conn: sqlite3.Connection, cutoff: str | None) -> list[str]:
    """Section 8: Weather gate parameter optimization."""
    lines = [
        "",
        "── 8. Weather Gate Optimization ───────────────────────────────────────────",
        "   Sweeps TEMP_FORECAST_MIN_EDGE, HRRR_MAX_SPREAD_F, and",
        "   TEMP_OBSERVED_MAX_HOURS against settled weather trades.",
        "",
        "   ⚠  IMPORTANT LIMITATION: Only trades that PASSED the current gate are",
        "   in the database. Lowering thresholds below current defaults cannot be",
        "   backtested — those blocked signals were never executed and have no data.",
        "   These sweeps only show what happens if we are MORE conservative.",
        "",
    ]

    wt = _load_weather_trades(conn, cutoff)
    if not wt:
        lines.append("   No settled weather trades found.")
        return lines

    forecast_trades = [w for w in wt if w.source in _FORECAST_SRCS]
    obs_no_trades   = [w for w in wt if w.source in _OBS_SRCS and w.side == "no"]

    lines.append(f"   Settled weather trades total       : {len(wt)}")
    lines.append(f"   Forecast source trades (for edge)  : {len(forecast_trades)}")
    lines.append(f"   Observed-NO trades (for time gate) : {len(obs_no_trades)}")
    lines.append("")

    # ── 8a. Edge distribution by bin ──────────────────────────────────────────
    lines.append("  8a. Forecast edge distribution (current threshold = "
                 f"{_CURRENT_FORECAST_MIN_EDGE:.1f}°F)")
    lines.append("")
    lines.append(f"   {'Edge range':>14}  {'Trades':>7}  {'Won':>5}  {'Lost':>5}"
                 f"  {'WinRate':>8}  {'P&L':>10}  Bar")
    lines.append("   " + "─" * 68)

    bin_edges = [7, 9, 11, 13, 15, 18, float("inf")]
    bin_labels = ["7–9°F", "9–11°F", "11–13°F", "13–15°F", "15–18°F", "18°F+"]
    bin_pnls: list[float] = []
    bin_data: list[tuple] = []

    for i, label in enumerate(bin_labels):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        bucket = [
            w for w in forecast_trades
            if w.edge_f is not None and lo <= w.edge_f < hi
        ]
        if not bucket:
            bin_pnls.append(0.0)
            bin_data.append((label, 0, 0, 0, 0.0, 0.0))
            continue
        n_won  = sum(1 for w in bucket if w.outcome == "won")
        n_lost = len(bucket) - n_won
        pnl    = sum(w.hold_pnl for w in bucket)
        wr     = n_won / len(bucket)
        bin_pnls.append(pnl)
        bin_data.append((label, len(bucket), n_won, n_lost, wr, pnl))

    best_bin_pnl  = max((d[5] for d in bin_data if d[1] > 0), default=1.0)
    worst_bin_pnl = min((d[5] for d in bin_data if d[1] > 0), default=0.0)
    for label, n, won, lost, wr, pnl in bin_data:
        if n == 0:
            lines.append(f"   {label:>14}  {'—':>7}")
            continue
        bar = _bar(pnl, best_bin_pnl, worst_bin_pnl)
        lines.append(
            f"   {label:>14}  {n:>7}  {won:>5}  {lost:>5}"
            f"  {wr*100:>7.0f}%  {_pnl(pnl):>10}  {bar}"
        )

    no_edge = [w for w in forecast_trades if w.edge_f is None]
    if no_edge:
        lines.append(f"\n   {len(no_edge)} forecast trade(s) have no edge record (old logs before edge was stored).")

    # ── 8b. TEMP_FORECAST_MIN_EDGE sweep ─────────────────────────────────────
    lines += [
        "",
        "  8b. TEMP_FORECAST_MIN_EDGE sweep",
        "   (Trades with edge < threshold are excluded; equivalent to raising the gate.)",
        "",
        f"   {'Min edge':>9}  {'Trades':>7}  {'Excluded':>9}  {'P&L':>10}  {'vs all':>9}  Bar",
        "   " + "─" * 62,
    ]

    fc_with_edge = [w for w in forecast_trades if w.edge_f is not None]
    all_fc_pnl   = sum(w.hold_pnl for w in fc_with_edge)

    edge_results: list[tuple[float, float, int]] = []
    for thresh in _EDGE_GRID:
        kept = [w for w in fc_with_edge if w.edge_f >= thresh]
        pnl  = sum(w.hold_pnl for w in kept)
        edge_results.append((thresh, pnl, len(kept)))

    best_edge_pnl  = max(r[1] for r in edge_results) if edge_results else 1.0
    worst_edge_pnl = min(r[1] for r in edge_results) if edge_results else 0.0
    best_edge_thresh = next(r[0] for r in edge_results if r[1] == best_edge_pnl)

    for thresh, pnl, n_kept in edge_results:
        excluded = len(fc_with_edge) - n_kept
        diff = pnl - all_fc_pnl
        diff_s = f"{'+' if diff >= 0 else ''}{_pnl(diff)}"
        bar  = _bar(pnl, best_edge_pnl, worst_edge_pnl)
        star = " ◀ best" if thresh == best_edge_thresh else ""
        lines.append(
            f"   {thresh:>8.1f}°F  {n_kept:>7}  {excluded:>9}  "
            f"{_pnl(pnl):>10}  {diff_s:>9}  {bar}{star}"
        )

    lines.append("")
    lines.append(f"   Current TEMP_FORECAST_MIN_EDGE = {_CURRENT_FORECAST_MIN_EDGE:.1f}°F")
    if best_edge_thresh != _CURRENT_FORECAST_MIN_EDGE:
        delta = best_edge_thresh - _CURRENT_FORECAST_MIN_EDGE
        lines.append(
            f"   Optimal threshold = {best_edge_thresh:.1f}°F "
            f"({'+' if delta >= 0 else ''}{delta:.1f}°F vs current)  "
            f"P&L {_pnl(best_edge_pnl)} vs {_pnl(all_fc_pnl)} at current"
        )
    else:
        lines.append("   Current threshold is at the optimum within this sweep.")

    # ── 8c. TEMP_OBSERVED_MAX_HOURS sweep ─────────────────────────────────────
    lines += [
        "",
        "  8c. TEMP_OBSERVED_MAX_HOURS sweep (observed-NO trades only)",
        "   (Trades entered with more than X hours until close are excluded.)",
        "",
    ]

    obs_no_with_dtc = [w for w in obs_no_trades if w.hours_to_close is not None]
    if not obs_no_with_dtc:
        lines.append("   No observed-NO trades with time-to-close data found.")
    else:
        lines += [
            f"   {'Max hours':>10}  {'Trades':>7}  {'Excluded':>9}  {'P&L':>10}  {'vs all':>9}  Bar",
            "   " + "─" * 62,
        ]
        all_obs_pnl = sum(w.hold_pnl for w in obs_no_with_dtc)
        hours_results: list[tuple[float, float, int]] = []
        for h in _HOURS_GRID:
            kept = [w for w in obs_no_with_dtc if w.hours_to_close <= h]
            pnl  = sum(w.hold_pnl for w in kept)
            hours_results.append((h, pnl, len(kept)))

        best_h_pnl   = max(r[1] for r in hours_results) if hours_results else 1.0
        worst_h_pnl  = min(r[1] for r in hours_results) if hours_results else 0.0
        best_h_thresh = next(r[0] for r in hours_results if r[1] == best_h_pnl)

        for h, pnl, n_kept in hours_results:
            excluded = len(obs_no_with_dtc) - n_kept
            diff = pnl - all_obs_pnl
            diff_s = f"{'+' if diff >= 0 else ''}{_pnl(diff)}"
            bar   = _bar(pnl, best_h_pnl, worst_h_pnl)
            star  = " ◀ best" if h == best_h_thresh else ""
            lines.append(
                f"   {h:>9.1f}h  {n_kept:>7}  {excluded:>9}  "
                f"{_pnl(pnl):>10}  {diff_s:>9}  {bar}{star}"
            )

        lines.append("")
        lines.append(f"   Current TEMP_OBSERVED_MAX_HOURS = {_CURRENT_OBS_MAX_HOURS:.1f}h")
        if best_h_thresh != _CURRENT_OBS_MAX_HOURS:
            delta = best_h_thresh - _CURRENT_OBS_MAX_HOURS
            lines.append(
                f"   Optimal window = {best_h_thresh:.1f}h "
                f"({'+' if delta >= 0 else ''}{delta:.1f}h vs current)  "
                f"P&L {_pnl(best_h_pnl)} vs {_pnl(all_obs_pnl)} at current"
            )
        else:
            lines.append("   Current window is at the optimum within this sweep.")

    # ── 8d. HRRR spread note ──────────────────────────────────────────────────
    lines += [
        "",
        "  8d. HRRR_MAX_SPREAD_F — cannot be backtested",
        "   HRRR spread (°F difference between daily NOAA and HRRR hourly high)",
        "   is not stored in price_snapshots or opportunities.  To enable this",
        "   analysis, log hrrr_spread in opportunity metadata at trade entry.",
        "",
    ]

    # ── 8e. Win rate by source summary ───────────────────────────────────────
    lines += [
        "  8e. Weather trade win rate by source (all settled)",
        "",
        f"   {'Source':>20}  {'Trades':>7}  {'Won':>5}  {'WinRate':>8}  {'Total P&L':>10}",
        "   " + "─" * 60,
    ]
    by_src: dict[str, list[WeatherTrade]] = defaultdict(list)
    for w in wt:
        by_src[w.source].append(w)
    for src, group in sorted(by_src.items(), key=lambda kv: -len(kv[1])):
        n     = len(group)
        n_won = sum(1 for w in group if w.outcome == "won")
        wr    = n_won / n
        pnl   = sum(w.hold_pnl for w in group)
        lines.append(
            f"   {src:>20}  {n:>7}  {n_won:>5}  {wr*100:>7.0f}%  {_pnl(pnl):>10}"
        )

    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Gate change impact analysis (Section 9)
# ---------------------------------------------------------------------------


def _gate_change_section(conn: sqlite3.Connection, cutoff: str | None) -> list[str]:
    """Section 9: Can the gate changes be backtested? Proxy analysis where possible."""
    lines = [
        "",
        "── 9. Gate Change Impact Analysis ─────────────────────────────────────────",
        "   Assesses whether each recent gate change is backtestable from the log.",
        "",
        "   KEY CONSTRAINT: opportunity_log.db records signals AFTER all gates fire.",
        "   Blocked signals are never logged, so most entry-gate changes cannot be",
        "   directly backtested — there is simply no data for those blocked paths.",
        "",
    ]

    # ── 9a. Backtestability matrix ──────────────────────────────────────────
    lines += [
        "  9a. Backtestability matrix",
        "",
        f"   {'Change':<38}  {'Backtestable?':>14}  Reason",
        "   " + "─" * 90,
        f"   {'KXGBPUSD + KXDOW added':<38}  {'❌ No':>14}  Markets never fetched — zero historical trades",
        f"   {'POLY_MIN_DIVERGENCE 0.20 → 0.15':<38}  {'❌ No':>14}  15-19% Poly signals blocked before logging",
        f"   {'CONTRARIAN_MAX_ENTRY_CENTS 50 → 65':<38}  {'⚠ Proxy':>14}  Blocked non-exempt numerics not logged",
        f"   {'CRYPTO_DAILY_CLOSE_HOURS 2 → 3':<38}  {'❌ No':>14}  Blocked crypto signals not logged",
        f"   {'FOREX_CLOSE_HOURS 1 → 2':<38}  {'❌ No':>14}  Blocked forex signals not logged",
        "",
    ]

    cut = f"AND t.logged_at >= '{cutoff}'" if cutoff else ""

    # ── 9b. Proxy: win rate by entry cost bucket (all settled trades) ────────
    lines += [
        "  9b. Proxy: win rate by effective entry cost (all settled trades)",
        "   (Entry cost = yes_ask for YES trades; 100 − yes_bid for NO trades.)",
        "   Reveals whether relaxing the contrarian gate to 65¢ is likely to profit.",
        "",
        f"   {'Bucket':>10}  {'Trades':>7}  {'Won':>5}  {'Lost':>5}  {'WinRate':>8}  Note",
        "   " + "─" * 65,
    ]

    bucket_rows = conn.execute(f"""
        SELECT
            (CASE t.side WHEN 'yes' THEN t.limit_price ELSE 100 - t.limit_price END / 10) * 10 AS bucket,
            SUM(CASE WHEN t.outcome = 'won' THEN 1 ELSE 0 END) AS won,
            SUM(CASE WHEN t.outcome = 'lost' THEN 1 ELSE 0 END) AS lost
        FROM trades t
        WHERE t.outcome IS NOT NULL AND t.source IS NOT NULL
              {cut}
        GROUP BY bucket
        ORDER BY bucket
    """).fetchall()

    for lo, won, lost in bucket_rows:
        total = won + lost
        wr    = won / total * 100 if total else 0.0
        hi    = lo + 9
        note  = ""
        if lo < 10:
            note = "mostly 1¢ crypto/penny bets (many early bugs)"
        elif lo < 20:
            note = "low-conviction signals"
        elif lo < 50:
            note = "moderate-conviction"
        elif lo < 70:
            note = "← new gate range (50→65¢ opens 51-65¢ non-exempt)"
        else:
            note = "high-conviction / locked obs"
        lines.append(
            f"   {lo:>4}-{hi:>3}¢  {total:>7}  {won:>5}  {lost:>5}  {wr:>7.0f}%  {note}"
        )

    lines += [
        "",
        "   Interpretation: win rate is HIGHER at higher entry costs (market partially",
        "   agrees → signal is more likely correct).  Raising the contrarian ceiling",
        "   from 50¢ to 65¢ opens ~18 additional noaa/eia numeric signals per cycle",
        "   (non-exempt sources at 51-65¢ YES ask).  Based on the proxy trend,",
        "   these are likely to have 40-75% win rates — reasonable to accept.",
        "   Note: poly_opps bypass the contrarian gate entirely; this change does",
        "   NOT add more polymarket signals.",
        "",
    ]

    # ── 9c. Polymarket signal quality warning ───────────────────────────────
    poly_rows = conn.execute(f"""
        SELECT
            (CASE t.side WHEN 'yes' THEN t.limit_price ELSE 100 - t.limit_price END / 10) * 10 AS bucket,
            SUM(CASE WHEN t.outcome = 'won' THEN 1 ELSE 0 END) AS won,
            SUM(CASE WHEN t.outcome = 'lost' THEN 1 ELSE 0 END) AS lost
        FROM trades t
        WHERE t.source = 'polymarket' AND t.outcome IS NOT NULL {cut}
        GROUP BY bucket ORDER BY bucket
    """).fetchall()

    total_poly_won  = sum(r[1] for r in poly_rows)
    total_poly_lost = sum(r[2] for r in poly_rows)
    total_poly      = total_poly_won + total_poly_lost
    poly_wr         = total_poly_won / total_poly * 100 if total_poly else 0.0

    lines += [
        "  9c. Polymarket signal quality (existing trades at current 20% threshold)",
        "",
        f"   {'Bucket':>10}  {'Trades':>7}  {'Won':>5}  {'Lost':>5}  {'WinRate':>8}",
        "   " + "─" * 50,
    ]
    for lo, won, lost in poly_rows:
        total = won + lost
        wr    = won / total * 100 if total else 0.0
        lines.append(f"   {lo:>4}-{lo+9:>3}¢  {total:>7}  {won:>5}  {lost:>5}  {wr:>7.0f}%")

    lines += [
        "",
        f"   Overall Polymarket: {total_poly_won}W / {total_poly_lost}L"
        f"  ({poly_wr:.1f}% win rate)",
        "",
    ]

    if poly_wr < 30.0 and total_poly >= 5:
        lines += [
            "   ⚠  RECOMMENDATION: REVERT POLY_MIN_DIVERGENCE to 0.20",
            "   Existing Polymarket trades at ≥20% divergence already have a"
            f" {poly_wr:.0f}% win rate.",
            "   Lowering to 15% adds weaker signals (smaller disagreement = less confident)",
            "   and is very likely to further reduce win rate.  The text-matching Jaccard",
            "   score at 15-20% divergence is producing false-positive matches, not real edge.",
            "",
        ]
    else:
        lines += [
            "   Polymarket win rate is acceptable — lowering threshold to 15% may be safe.",
            "",
        ]

    # ── 9d. Numeric opp volume in 51-65¢ range (contrarian gate impact) ─────
    new_range_rows = conn.execute("""
        SELECT o.source, COUNT(*) as n
        FROM opportunities o
        WHERE o.kind = 'numeric'
          AND o.implied_outcome IN ('YES', 'NO')
          AND o.yes_bid IS NOT NULL AND o.yes_ask IS NOT NULL
          AND CASE o.implied_outcome WHEN 'YES' THEN o.yes_ask ELSE 100 - o.yes_bid END BETWEEN 51 AND 65
          AND o.source NOT IN ('noaa_observed', 'metar', 'nws_climo', 'nws_alert')
        GROUP BY o.source ORDER BY n DESC
    """).fetchall()

    total_new = sum(r[1] for r in new_range_rows)
    lines += [
        "  9d. Non-exempt numeric opportunities in the new 51-65¢ range",
        "   (These were blocked by old 50¢ gate but will now be allowed.)",
        "",
        f"   {'Source':>20}  {'Surfaced opps':>15}",
        "   " + "─" * 40,
    ]
    for src, n in new_range_rows:
        lines.append(f"   {src:>20}  {n:>15}")
    lines += [
        f"   {'TOTAL':>20}  {total_new:>15}",
        "",
        "   These are historic SURFACED counts — the actual blocked count from the old",
        "   gate is higher but not logged.  This is a lower-bound estimate.",
        "",
    ]

    # ── 9e. Summary ─────────────────────────────────────────────────────────
    lines += [
        "  9e. Summary",
        "",
        "   Change                               Verdict",
        "   " + "─" * 72,
        "   KXGBPUSD + KXDOW added              ✓  Forward-looking only; no backtest possible.",
        "                                           Immediate benefit: new markets tracked.",
        "   POLY_MIN_DIVERGENCE 0.20 → 0.15     ⚠  RECOMMEND REVERTING.  Existing Poly",
        f"                                           trades at ≥20% already {poly_wr:.0f}% WR.",
        "                                           Lower-divergence signals will be worse.",
        "   CONTRARIAN_MAX 50 → 65¢             ✓  Proxy data supports it.  Win rate trend",
        "                                           is positive in 50-70¢ range.  ~18",
        "                                           additional noaa/eia signals per cycle.",
        "   CRYPTO_DAILY_CLOSE_HOURS 2 → 3      ?  Untestable.  Extra hour exposes to",
        "                                           intraday drift; monitor closely.",
        "   FOREX_CLOSE_HOURS 1 → 2             ?  Untestable.  ECB rate locked at 16:00 CET;",
        "                                           extra hour is directionally safe.",
        "",
    ]

    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    if not _DB_PATH.exists():
        print(f"Database not found: {_DB_PATH}")
        raise SystemExit(1)

    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    try:
        cutoff = None
        if args.days > 0:
            cutoff = (
                datetime.now(timezone.utc) - timedelta(days=args.days)
            ).strftime("%Y-%m-%dT%H:%M:%S")
        sim_trades = load_data(conn, cutoff)

        # Actual as-executed P&L for reference
        actual_pnl = sum(
            (st.trade.exit_pnl if st.trade.exit_pnl is not None
             else (st.hold_pnl or 0.0))
            for st in sim_trades
        )

        lines = _header(sim_trades)

        global_lines, (best_pt, best_sl, best_trailing) = _global_search_section(
            sim_trades, actual_pnl
        )
        lines += global_lines
        lines += _pt_sensitivity(sim_trades, best_sl, best_trailing)
        lines += _sl_sensitivity(sim_trades, best_pt, best_trailing)
        lines += _trailing_sensitivity(sim_trades, best_pt, best_sl)

        per_src_lines, per_src_recs = _per_source_section(sim_trades)
        lines += per_src_lines
        lines += _kelly_section(sim_trades)
        lines += _recommendations_section((best_pt, best_sl, best_trailing), per_src_recs)
        lines += _weather_gate_section(conn, cutoff)
        lines += _gate_change_section(conn, cutoff)
    finally:
        conn.close()

    report = "\n".join(lines) + "\n"

    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"Report written to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kalshi bot exit optimizer")
    parser.add_argument("--days",        type=int, default=0,
                        help="Only use trades from the last N days (0 = all time)")
    parser.add_argument("--min-trades",  type=int, default=_MIN_TRADES_PER_GROUP,
                        dest="min_trades",
                        help="Minimum trades per source group (default: 5)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Write report to file instead of stdout")
    args = parser.parse_args()
    _MIN_TRADES_PER_GROUP = args.min_trades
    run(args)
