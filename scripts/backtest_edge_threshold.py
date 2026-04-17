"""Backtest minimum-edge threshold for forecast_no NO trades.

For each forecast_no trade, reconstructs the qualifying sources and their
edges at signal time by querying raw_forecasts within a ±3-minute window.
Then sweeps FORECAST_NO_MIN_EDGE_F thresholds to answer:

  "If we required every qualifying source to have edge ≥ X°F,
   how many trades would have been blocked, and would win rate / P&L improve?"

Also breaks down win rate and avg P&L by edge bucket so you can see whether
higher-edge signals actually have better outcomes.

Usage:
  venv/bin/python scripts/backtest_edge_threshold.py
  venv/bin/python scripts/backtest_edge_threshold.py --min-edge 6 --max-edge 20
  venv/bin/python scripts/backtest_edge_threshold.py --resolved-only
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DB_PATH  = Path(__file__).parent.parent / "opportunity_log.db"

# Sources that count toward forecast_no source_score (mirrors strike_arb.py)
FORECAST_NO_SOURCES = frozenset({"hrrr", "nws_hourly", "open_meteo", "noaa"})

# Current live minimum edge threshold
CURRENT_MIN_EDGE = 6.0
CURRENT_MIN_SOURCES = 2

SEP = "=" * 78


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _time_window(ts: str, before_minutes: int, after_minutes: int = 1) -> tuple[str, str]:
    """Return (lo, hi) ISO strings for a time window around ts.

    Strips the timezone suffix so SQLite substring comparisons work against
    the T-separated timestamps stored in raw_forecasts.
    """
    from datetime import datetime, timedelta
    # Parse — handle both +00:00 and Z suffixes
    ts_clean = ts.replace("Z", "+00:00")
    dt = datetime.fromisoformat(ts_clean)
    lo = (dt - timedelta(minutes=before_minutes)).strftime("%Y-%m-%dT%H:%M:%S")
    hi = (dt + timedelta(minutes=after_minutes)).strftime("%Y-%m-%dT%H:%M:%S")
    return lo, hi


def load_trades_with_edges(
    db: sqlite3.Connection,
    resolved_only: bool,
    window_minutes: int = 3,
) -> list[dict]:
    """Load forecast_no NO trades and reconstruct per-source edges at signal time."""
    cur = db.cursor()

    where = "t.side='no' AND t.opportunity_kind='forecast_no'"
    if resolved_only:
        where += " AND t.outcome IS NOT NULL"

    cur.execute(f"""
        SELECT t.id, t.ticker, t.logged_at, t.outcome,
               t.exit_pnl_cents, t.exit_reason, t.limit_price
        FROM trades t
        WHERE {where}
        ORDER BY t.id
    """)
    trade_rows = [dict(r) for r in cur.fetchall()]

    trades = []
    for t in trade_rows:
        lo, hi = _time_window(t["logged_at"], before_minutes=window_minutes)
        cur.execute("""
            SELECT source, data_value, edge, strike, direction
            FROM raw_forecasts
            WHERE ticker = ?
              AND source IN ('hrrr','nws_hourly','open_meteo','noaa')
              AND substr(logged_at,1,19) BETWEEN ? AND ?
            ORDER BY source, edge DESC
        """, (t["ticker"], lo, hi))
        raw = cur.fetchall()

        # Deduplicate: keep highest-edge row per source (mirrors strike_arb dedup)
        best_per_source: dict[str, float] = {}
        for r in raw:
            src, edge = r["source"], r["edge"]
            if edge is not None:
                if src not in best_per_source or edge > best_per_source[src]:
                    best_per_source[src] = edge

        t["source_edges"] = best_per_source   # {source: max_edge}
        t["n_sources_raw"] = len(best_per_source)
        t["min_edge_seen"] = min(best_per_source.values()) if best_per_source else None
        t["max_edge_seen"] = max(best_per_source.values()) if best_per_source else None
        trades.append(t)

    return trades


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def would_fire(trade: dict, min_edge: float, min_sources: int = CURRENT_MIN_SOURCES) -> bool:
    """Would this trade have fired under the given min_edge threshold?"""
    qualifying = {
        src: edge
        for src, edge in trade["source_edges"].items()
        if edge >= min_edge
    }
    return len(qualifying) >= min_sources


def settlement_pnl(trade: dict) -> float | None:
    """P&L if held to settlement (cents, 1 contract)."""
    if trade["outcome"] == "won":
        return float(trade["limit_price"])           # NO wins: collect limit_price cents profit
    if trade["outcome"] == "lost":
        return -(100.0 - trade["limit_price"])       # NO loses: lose entry cost
    return trade.get("exit_pnl_cents")               # Proxy: actual recorded exit


def _agg(group: list[dict]) -> dict:
    pnls = [p for t in group if (p := settlement_pnl(t)) is not None]
    n = len(group)
    n_pnl = len(pnls)
    resolved = [t for t in group if t["outcome"] in ("won", "lost")]
    won = [t for t in resolved if t["outcome"] == "won"]
    return {
        "n":        n,
        "n_pnl":    n_pnl,
        "n_resolved": len(resolved),
        "win_rate": len(won) / len(resolved) if resolved else None,
        "avg_pnl":  sum(pnls) / n_pnl if pnls else None,
        "total_pnl": sum(pnls) if pnls else None,
    }


def fmt(d: dict) -> str:
    wr  = f"{d['win_rate']*100:5.1f}%" if d["win_rate"] is not None else "  n/a "
    ap  = f"{d['avg_pnl']:+8.2f}¢" if d["avg_pnl"] is not None else "       n/a"
    tot = f"{d['total_pnl']:+9.1f}¢" if d["total_pnl"] is not None else "        n/a"
    return f"N={d['n']:3d}  res={d['n_resolved']:3d}  Win%={wr}  Avg={ap}  Tot={tot}"


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def report_no_edge_data(trades: list[dict]) -> None:
    missing = [t for t in trades if not t["source_edges"]]
    if missing:
        print(f"\n  WARNING: {len(missing)} trades have no raw_forecasts edge data")
        print("  (early trades or raw_forecasts retention gap — excluded from edge analysis):")
        for t in missing[:5]:
            print(f"    trade {t['id']} {t['ticker']} logged={t['logged_at'][:19]}")
        if len(missing) > 5:
            print(f"    ... and {len(missing)-5} more")


def report_sweep(trades: list[dict]) -> tuple[float, float]:
    """Sweep min_edge thresholds. Returns (best_min_edge, best_min_sources)."""
    thresholds = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0, 18.0, 20.0]
    trades_with_data = [t for t in trades if t["source_edges"]]

    print(f"\n{SEP}")
    print("1. MIN_EDGE THRESHOLD SWEEP  (min_sources=2, deduped per source)")
    print(SEP)
    print(f"  {'MinEdge':>8}  {'Fired':>5}  {'Blocked':>7}  "
          f"{'res':>4}  {'Win%':>6}  {'Avg¢':>9}  {'Tot¢':>10}")

    best_avg = None
    best_threshold = CURRENT_MIN_EDGE

    for thr in thresholds:
        fired   = [t for t in trades_with_data if would_fire(t, thr)]
        blocked = [t for t in trades_with_data if not would_fire(t, thr)]
        d = _agg(fired)
        marker = ""
        if d["avg_pnl"] is not None:
            if best_avg is None or d["avg_pnl"] > best_avg:
                best_avg = d["avg_pnl"]
                best_threshold = thr
                marker = " ← best"
        cur_marker = "  [current]" if abs(thr - CURRENT_MIN_EDGE) < 0.01 else ""
        wr  = f"{d['win_rate']*100:5.1f}%" if d["win_rate"] is not None else "   n/a"
        ap  = f"{d['avg_pnl']:+8.2f}" if d["avg_pnl"] is not None else "      n/a"
        tot = f"{d['total_pnl']:+9.1f}" if d["total_pnl"] is not None else "       n/a"
        print(f"  ≥{thr:>6.1f}°F  {len(fired):>5}  {len(blocked):>7}  "
              f"{d['n_resolved']:>4}  {wr}  {ap}  {tot}{marker}{cur_marker}")

    return best_threshold, CURRENT_MIN_SOURCES


def report_edge_buckets(trades: list[dict]) -> None:
    """Group trades by their minimum qualifying source edge and show outcomes."""
    print(f"\n{SEP}")
    print("2. OUTCOME BY EDGE BUCKET  (min_edge of the WEAKEST qualifying source)")
    print(SEP)

    trades_with_data = [t for t in trades if t["source_edges"]]
    buckets: dict[str, list[dict]] = defaultdict(list)

    for t in trades_with_data:
        # The "weakest link": the lowest edge among qualifying sources (≥ current threshold)
        qualifying_edges = [e for e in t["source_edges"].values() if e >= CURRENT_MIN_EDGE]
        if not qualifying_edges:
            buckets["<6°F (wouldn't fire)"].append(t)
            continue
        min_q = min(qualifying_edges)
        if min_q < 7:
            buckets["6–7°F"].append(t)
        elif min_q < 8:
            buckets["7–8°F"].append(t)
        elif min_q < 10:
            buckets["8–10°F"].append(t)
        elif min_q < 12:
            buckets["10–12°F"].append(t)
        elif min_q < 15:
            buckets["12–15°F"].append(t)
        else:
            buckets["≥15°F"].append(t)

    order = ["<6°F (wouldn't fire)", "6–7°F", "7–8°F", "8–10°F",
             "10–12°F", "12–15°F", "≥15°F"]
    print(f"  {'Bucket':>25}  {'N':>4}  {'res':>4}  {'Win%':>6}  {'Avg¢':>9}  {'Tot¢':>10}")
    for b in order:
        if b not in buckets:
            continue
        d = _agg(buckets[b])
        wr  = f"{d['win_rate']*100:5.1f}%" if d["win_rate"] is not None else "   n/a"
        ap  = f"{d['avg_pnl']:+8.2f}" if d["avg_pnl"] is not None else "      n/a"
        tot = f"{d['total_pnl']:+9.1f}" if d["total_pnl"] is not None else "       n/a"
        print(f"  {b:>25}  {len(buckets[b]):>4}  {d['n_resolved']:>4}  {wr}  {ap}  {tot}")


def report_source_count_sweep(trades: list[dict]) -> None:
    """Sweep min_sources at fixed best edge to see if requiring more consensus helps."""
    print(f"\n{SEP}")
    print("3. MIN_SOURCES SWEEP  (at current min_edge=6°F)")
    print(SEP)
    trades_with_data = [t for t in trades if t["source_edges"]]
    print(f"  {'MinSrc':>7}  {'Fired':>5}  {'Blocked':>7}  "
          f"{'res':>4}  {'Win%':>6}  {'Avg¢':>9}  {'Tot¢':>10}")
    for min_src in [1, 2, 3, 4]:
        fired = [t for t in trades_with_data
                 if would_fire(t, CURRENT_MIN_EDGE, min_src)]
        d = _agg(fired)
        cur_marker = "  [current]" if min_src == CURRENT_MIN_SOURCES else ""
        wr  = f"{d['win_rate']*100:5.1f}%" if d["win_rate"] is not None else "   n/a"
        ap  = f"{d['avg_pnl']:+8.2f}" if d["avg_pnl"] is not None else "      n/a"
        tot = f"{d['total_pnl']:+9.1f}" if d["total_pnl"] is not None else "       n/a"
        print(f"  ≥{min_src:>5} src  {len(fired):>5}  {len([t for t in trades_with_data if not would_fire(t, CURRENT_MIN_EDGE, min_src)]):>7}  "
              f"{d['n_resolved']:>4}  {wr}  {ap}  {tot}{cur_marker}")


def report_joint_sweep(trades: list[dict]) -> None:
    """Joint sweep: min_edge × min_sources."""
    print(f"\n{SEP}")
    print("4. JOINT SWEEP: MIN_EDGE × MIN_SOURCES")
    print(SEP)
    trades_with_data = [t for t in trades if t["source_edges"]]
    thresholds = [6.0, 7.0, 8.0, 10.0, 12.0]
    print(f"  {'Edge':>7}  {'Src':>4}  {'Fired':>5}  "
          f"{'res':>4}  {'Win%':>6}  {'Avg¢':>9}  {'Tot¢':>10}")
    best_avg = None
    best_combo = (CURRENT_MIN_EDGE, CURRENT_MIN_SOURCES)
    rows_out = []
    for thr in thresholds:
        for min_src in [1, 2, 3]:
            fired = [t for t in trades_with_data if would_fire(t, thr, min_src)]
            d = _agg(fired)
            rows_out.append((thr, min_src, fired, d))
            if d["avg_pnl"] is not None:
                if best_avg is None or d["avg_pnl"] > best_avg:
                    best_avg = d["avg_pnl"]
                    best_combo = (thr, min_src)

    for thr, min_src, fired, d in rows_out:
        marker  = " ← best" if (thr, min_src) == best_combo else ""
        cur_tag = "  [cur]" if (thr, min_src) == (CURRENT_MIN_EDGE, CURRENT_MIN_SOURCES) else ""
        wr  = f"{d['win_rate']*100:5.1f}%" if d["win_rate"] is not None else "   n/a"
        ap  = f"{d['avg_pnl']:+8.2f}" if d["avg_pnl"] is not None else "      n/a"
        tot = f"{d['total_pnl']:+9.1f}" if d["total_pnl"] is not None else "       n/a"
        print(f"  ≥{thr:>5.1f}°F  ≥{min_src}src  {len(fired):>5}  "
              f"{d['n_resolved']:>4}  {wr}  {ap}  {tot}{marker}{cur_tag}")


def report_per_trade(trades: list[dict]) -> None:
    """Show each trade's edge profile alongside its outcome."""
    print(f"\n{SEP}")
    print("5. PER-TRADE EDGE PROFILE")
    print(SEP)
    print(f"  {'id':>4}  {'ticker':>32}  {'outcome':>7}  {'pnl¢':>7}  "
          f"{'n_src':>5}  {'min_e':>6}  {'max_e':>6}  sources")
    for t in trades:
        if not t["source_edges"]:
            continue
        q_edges = {s: e for s, e in t["source_edges"].items() if e >= CURRENT_MIN_EDGE}
        src_str = "  ".join(f"{s}={e:.1f}" for s, e in sorted(t["source_edges"].items()))
        pnl = settlement_pnl(t)
        print(f"  {t['id']:>4}  {t['ticker']:>32}  {str(t['outcome']):>7}  "
              f"{pnl:>+7.1f}  " if pnl is not None else
              f"  {t['id']:>4}  {t['ticker']:>32}  {str(t['outcome']):>7}  "
              f"{'n/a':>7}  ", end="")
        print(f"{len(q_edges):>5}  {t['min_edge_seen']:>6.1f}  {t['max_edge_seen']:>6.1f}  {src_str}")


def report_blocked_trades(trades: list[dict], best_threshold: float) -> None:
    """Show which trades would have been blocked at the best threshold."""
    if abs(best_threshold - CURRENT_MIN_EDGE) < 0.01:
        return
    trades_with_data = [t for t in trades if t["source_edges"]]
    would_block = [t for t in trades_with_data
                   if would_fire(t, CURRENT_MIN_EDGE) and not would_fire(t, best_threshold)]
    if not would_block:
        return
    print(f"\n{SEP}")
    print(f"6. TRADES BLOCKED BY RAISING MIN_EDGE TO {best_threshold:.0f}°F  "
          f"(were passing at {CURRENT_MIN_EDGE:.0f}°F)")
    print(SEP)
    wins   = [t for t in would_block if t["outcome"] == "won"]
    losses = [t for t in would_block if t["outcome"] == "lost"]
    print(f"  Blocked: {len(would_block)} trades — {len(wins)} would-win, "
          f"{len(losses)} would-lose, "
          f"{len(would_block)-len(wins)-len(losses)} unresolved")
    print()
    print(f"  {'id':>4}  {'ticker':>32}  {'outcome':>7}  {'pnl¢':>7}  sources at entry")
    for t in sorted(would_block, key=lambda x: (x["outcome"] or "z")):
        pnl = settlement_pnl(t)
        pnl_s = f"{pnl:+7.1f}" if pnl is not None else "    n/a"
        src_str = "  ".join(f"{s}={e:.1f}" for s, e in sorted(t["source_edges"].items()))
        print(f"  {t['id']:>4}  {t['ticker']:>32}  {str(t['outcome']):>7}  "
              f"{pnl_s}  {src_str}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row

    print(f"Loading forecast_no NO trades "
          f"(resolved_only={args.resolved_only}, window=±{args.window}min)…")
    trades = load_trades_with_edges(db, args.resolved_only, args.window)
    db.close()

    has_data = [t for t in trades if t["source_edges"]]
    print(f"Loaded {len(trades)} trades — {len(has_data)} have raw_forecasts edge data.")
    report_no_edge_data(trades)

    best_threshold, best_min_src = report_sweep(trades)
    report_edge_buckets(trades)
    report_source_count_sweep(trades)
    report_joint_sweep(trades)
    report_per_trade(trades)
    report_blocked_trades(trades, best_threshold)

    # Summary
    print(f"\n{SEP}")
    print("SUMMARY")
    print(SEP)
    cur_fired = [t for t in has_data if would_fire(t, CURRENT_MIN_EDGE)]
    best_fired = [t for t in has_data if would_fire(t, best_threshold)]
    cd = _agg(cur_fired)
    bd = _agg(best_fired)
    print(f"  Current: min_edge={CURRENT_MIN_EDGE:.0f}°F  min_sources={CURRENT_MIN_SOURCES}")
    print(f"    {fmt(cd)}")
    print(f"  Optimal: min_edge={best_threshold:.0f}°F  min_sources={best_min_src}")
    print(f"    {fmt(bd)}")
    if cd["avg_pnl"] is not None and bd["avg_pnl"] is not None:
        delta = bd["avg_pnl"] - cd["avg_pnl"]
        print(f"  Avg P&L improvement: {delta:+.2f}¢/trade")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backtest min-edge threshold for forecast_no NO trades."
    )
    parser.add_argument(
        "--resolved-only", action="store_true",
        help="Only include trades with known outcome (won/lost).",
    )
    parser.add_argument(
        "--window", type=int, default=3,
        help="Minutes before signal to look for raw_forecasts edges (default: 3).",
    )
    args = parser.parse_args()
    main(args)
