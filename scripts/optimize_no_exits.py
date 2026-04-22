"""Backtest exit-parameter optimization for historical NO trades.

Replays price_snapshots trajectories for all NO trades stored in
opportunity_log.db, sweeping profit_take, stop_loss, and near-close
trailing thresholds to find the combination that maximizes total P&L.

For each trade the simulation scans snapshots in order:
  - Profit-take fires when pct_gain >= pt_threshold
  - Stop-loss   fires when pct_gain <= -sl_threshold
  - Near-close trailing fires when days_to_close < nc_hours/24 AND
    peak_pct > 0 AND pct_gain < peak_pct - nc_drawdown
  - Settlement  uses outcome ('won'/'lost') if known, else last pct_gain

P&L is reported in cents per 1-contract lot.  Entry cost = 100 - limit_price.

Usage:
  venv/bin/python scripts/optimize_no_exits.py
  venv/bin/python scripts/optimize_no_exits.py --kinds forecast_no
  venv/bin/python scripts/optimize_no_exits.py --kinds forecast_no numeric --min-snaps 10
  venv/bin/python scripts/optimize_no_exits.py --resolved-only
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DB_PATH = Path(__file__).parent.parent / "opportunity_log.db"

# Current live settings (baseline comparison)
CURRENT_PT = 0.20
CURRENT_SL = 0.70
CURRENT_NC_HOURS    = 2.0
CURRENT_NC_DRAWDOWN = 0.15


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_trades(
    db: sqlite3.Connection,
    kinds: list[str] | None,
    min_snaps: int,
    resolved_only: bool,
    from_id: int | None = None,
) -> list[dict]:
    """Load NO trades with their snapshot trajectories."""
    where_clauses = ["t.side = 'no'"]
    params: list = []

    if kinds:
        placeholders = ",".join("?" * len(kinds))
        where_clauses.append(f"t.opportunity_kind IN ({placeholders})")
        params.extend(kinds)

    if resolved_only:
        where_clauses.append("t.outcome IS NOT NULL")

    if from_id is not None:
        where_clauses.append("t.id >= ?")
        params.append(from_id)

    where_sql = " AND ".join(where_clauses)

    cur = db.cursor()
    cur.execute(f"""
        SELECT t.id, t.ticker, t.opportunity_kind, t.source,
               t.limit_price, t.yes_bid_entry, t.yes_ask_entry,
               t.outcome, t.exit_pnl_cents, t.exit_reason,
               COUNT(ps.id) AS n_snaps
        FROM trades t
        JOIN price_snapshots ps ON ps.trade_id = t.id
        WHERE {where_sql}
        GROUP BY t.id
        HAVING n_snaps >= ?
        ORDER BY t.id
    """, (*params, min_snaps))

    trades = [dict(row) for row in cur.fetchall()]

    # Load snapshot trajectories.
    # post_exit=0 rows: price while we held the position (used for exit simulation).
    # post_exit=1 rows: counterfactual price after we already exited (used for "held longer" analysis).
    # Older DBs lack the post_exit column — fall back to loading all rows as pre-exit.
    cur.execute("PRAGMA table_info(price_snapshots)")
    snap_cols = {row["name"] for row in cur.fetchall()}
    has_post_exit = "post_exit" in snap_cols

    for trade in trades:
        if has_post_exit:
            cur.execute("""
                SELECT pct_gain, days_to_close, post_exit
                FROM price_snapshots
                WHERE trade_id = ?
                ORDER BY id
            """, (trade["id"],))
            rows_raw = cur.fetchall()
            trade["snapshots"]      = [(r["pct_gain"], r["days_to_close"]) for r in rows_raw if not r["post_exit"]]
            trade["snapshots_full"] = [(r["pct_gain"], r["days_to_close"]) for r in rows_raw]
        else:
            cur.execute("""
                SELECT pct_gain, days_to_close
                FROM price_snapshots
                WHERE trade_id = ?
                ORDER BY id
            """, (trade["id"],))
            rows_raw = cur.fetchall()
            trade["snapshots"]      = [(r["pct_gain"], r["days_to_close"]) for r in rows_raw]
            trade["snapshots_full"] = trade["snapshots"]

    return trades


# ---------------------------------------------------------------------------
# Per-trade simulation
# ---------------------------------------------------------------------------

def simulate_exit(
    trade: dict,
    profit_take: float,
    stop_loss: float,
    nc_hours: float,
    nc_drawdown: float,
) -> dict | None:
    """Simulate one trade.  Returns None if entry cost is invalid (≤0).

    Uses snapshots_full (pre-exit + post-exit rows) so that simulated thresholds
    higher than the actual exit can still be reached if the market kept moving.
    """
    entry_cost = 100 - trade["limit_price"]
    if entry_cost <= 0:
        return None

    # Use full trajectory (pre- and post-exit) so we can test "what if we held longer"
    snaps     = trade.get("snapshots_full") or trade["snapshots"]
    nc_days   = nc_hours / 24.0
    peak_pct  = 0.0

    for pct_gain, days_to_close in snaps:
        if pct_gain is None:
            continue

        if pct_gain > peak_pct:
            peak_pct = pct_gain

        # Fixed profit-take
        if pct_gain >= profit_take:
            return _result("profit_take", pct_gain, entry_cost, trade)

        # Fixed stop-loss
        if pct_gain <= -stop_loss:
            return _result("stop_loss", pct_gain, entry_cost, trade)

        # Near-close trailing drawdown
        if (days_to_close is not None
                and days_to_close < nc_days
                and peak_pct > 0.0
                and pct_gain < peak_pct - nc_drawdown):
            return _result("near_close_trailing", pct_gain, entry_cost, trade)

    # No exit triggered → settlement or last-known price
    outcome = trade.get("outcome")
    if outcome == "won":
        # NO contract pays 100¢; we paid entry_cost = 100-limit_price
        final_pct = trade["limit_price"] / entry_cost   # = LP / (100-LP)
    elif outcome == "lost":
        final_pct = -1.0
    else:
        # Unresolved — use last snapshot as proxy (may be near-close price)
        last_pct = snaps[-1][0] if snaps else 0.0
        final_pct = last_pct if last_pct is not None else 0.0

    return _result("settlement", final_pct, entry_cost, trade)


def _result(reason: str, pct_gain: float, entry_cost: int, trade: dict) -> dict:
    return {
        "exit_reason":      reason,
        "pct_gain":         pct_gain,
        "pnl_cents":        pct_gain * entry_cost,
        "entry_cost":       entry_cost,
        "opportunity_kind": trade["opportunity_kind"],
        "source":           trade["source"],
        "outcome":          trade.get("outcome"),
    }


# ---------------------------------------------------------------------------
# Sweep helpers
# ---------------------------------------------------------------------------

def run_sweep(
    trades: list[dict],
    profit_takes: list[float],
    stop_losses: list[float],
    nc_hours: float,
    nc_drawdown: float,
) -> dict[tuple[float, float], dict]:
    """Grid search over all (PT, SL) pairs."""
    results: dict[tuple[float, float], dict] = {}
    for pt in profit_takes:
        for sl in stop_losses:
            sims = [
                s for t in trades
                if (s := simulate_exit(t, pt, sl, nc_hours, nc_drawdown)) is not None
            ]
            if not sims:
                continue
            results[(pt, sl)] = _agg(sims)
    return results


def run_sweep_nc(
    trades: list[dict],
    best_pt: float,
    best_sl: float,
    nc_hours_list: list[float],
    nc_drawdown_list: list[float],
) -> dict[tuple[float, float], dict]:
    """Sweep near-close trailing parameters at fixed best PT/SL."""
    results: dict[tuple[float, float], dict] = {}
    for hours in nc_hours_list:
        for draw in nc_drawdown_list:
            sims = [
                s for t in trades
                if (s := simulate_exit(t, best_pt, best_sl, hours, draw)) is not None
            ]
            if not sims:
                continue
            results[(hours, draw)] = _agg(sims)
    return results


def _agg(sims: list[dict]) -> dict:
    n         = len(sims)
    total_pnl = sum(s["pnl_cents"] for s in sims)
    avg_pnl   = total_pnl / n
    wins      = sum(1 for s in sims if s["pnl_cents"] > 0)
    return {
        "n":         n,
        "total_pnl": total_pnl,
        "avg_pnl":   avg_pnl,
        "win_rate":  wins / n,
        "pt_rate":   sum(1 for s in sims if s["exit_reason"] == "profit_take") / n,
        "sl_rate":   sum(1 for s in sims if s["exit_reason"] == "stop_loss") / n,
        "nc_rate":   sum(1 for s in sims if s["exit_reason"] == "near_close_trailing") / n,
        "se_rate":   sum(1 for s in sims if s["exit_reason"] == "settlement") / n,
        "sims":      sims,
    }


def _find_best(sweep: dict) -> tuple:
    return max(sweep, key=lambda k: sweep[k]["avg_pnl"])


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

SEP = "=" * 78

def _hdr(title: str) -> None:
    print(f"\n{SEP}")
    print(title)
    print(SEP)


def print_sweep_table(sweep: dict, current_key: tuple, label_a: str, label_b: str) -> None:
    best_key = _find_best(sweep)
    print(f"  {label_a:>5}  {label_b:>5}  {'N':>6}  {'Win%':>5}  "
          f"{'PT%':>5}  {'SL%':>5}  {'NC%':>5}  {'Avg¢':>8}  {'Total¢':>10}")
    for key in sorted(sweep):
        d   = sweep[key]
        tag = " ← best" if key == best_key else ("  [current]" if key == current_key else "")
        print(f"  {key[0]*100:>5.0f}  {key[1]*100:>5.0f}  {d['n']:>6}  "
              f"{d['win_rate']*100:>5.1f}%  "
              f"{d['pt_rate']*100:>5.1f}%  {d['sl_rate']*100:>5.1f}%  "
              f"{d['nc_rate']*100:>5.1f}%  {d['avg_pnl']:>+8.2f}  "
              f"{d['total_pnl']:>+10.1f}{tag}")


def print_breakdown(
    trades: list[dict],
    best_pt: float,
    best_sl: float,
    nc_hours: float,
    nc_drawdown: float,
    group_key: str,
) -> None:
    from collections import defaultdict
    groups: dict[str, list] = defaultdict(list)
    for t in trades:
        s = simulate_exit(t, best_pt, best_sl, nc_hours, nc_drawdown)
        if s is not None:
            groups[t[group_key]].append(s)

    print(f"  {group_key[:16]:>16}  {'N':>6}  {'Win%':>5}  "
          f"{'PT%':>5}  {'SL%':>5}  {'Avg¢':>8}  {'Total¢':>10}")
    for gk in sorted(groups, key=lambda k: -_agg(groups[k])["avg_pnl"]):
        sims = groups[gk]
        d    = _agg(sims)
        print(f"  {gk[:16]:>16}  {d['n']:>6}  {d['win_rate']*100:>5.1f}%  "
              f"{d['pt_rate']*100:>5.1f}%  {d['sl_rate']*100:>5.1f}%  "
              f"{d['avg_pnl']:>+8.2f}  {d['total_pnl']:>+10.1f}")


def print_worst_trades(sims: list[dict], n: int = 10) -> None:
    worst = sorted(sims, key=lambda s: s["pnl_cents"])[:n]
    print(f"  {'pnl¢':>8}  {'reason':>20}  {'pct':>8}  kind")
    for s in worst:
        print(f"  {s['pnl_cents']:>+8.1f}  {s['exit_reason']:>20}  "
              f"{s['pct_gain']*100:>+7.1f}%  {s['opportunity_kind']}/{s['source']}")


def print_best_trades(sims: list[dict], n: int = 10) -> None:
    best = sorted(sims, key=lambda s: -s["pnl_cents"])[:n]
    print(f"  {'pnl¢':>8}  {'reason':>20}  {'pct':>8}  kind")
    for s in best:
        print(f"  {s['pnl_cents']:>+8.1f}  {s['exit_reason']:>20}  "
              f"{s['pct_gain']*100:>+7.1f}%  {s['opportunity_kind']}/{s['source']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row

    print(f"Loading NO trades (min_snaps={args.min_snaps}, "
          f"kinds={args.kinds or 'all'}, resolved_only={args.resolved_only})…")
    trades = load_trades(db, args.kinds, args.min_snaps, args.resolved_only, args.from_id)
    db.close()

    if not trades:
        print("No trades matched the filters — exiting.")
        return

    n_resolved = sum(1 for t in trades if t["outcome"] in ("won", "lost"))
    print(f"Loaded {len(trades)} trades ({n_resolved} resolved, "
          f"{len(trades)-n_resolved} unresolved/using last-snapshot proxy).")

    # ── 1. Baseline (current params) ─────────────────────────────────────────
    _hdr("1. BASELINE — current live settings")
    base_sims = [
        s for t in trades
        if (s := simulate_exit(t, CURRENT_PT, CURRENT_SL,
                               CURRENT_NC_HOURS, CURRENT_NC_DRAWDOWN)) is not None
    ]
    base = _agg(base_sims)
    print(f"  PT={CURRENT_PT*100:.0f}%  SL={CURRENT_SL*100:.0f}%  "
          f"NC_hours={CURRENT_NC_HOURS}  NC_drawdown={CURRENT_NC_DRAWDOWN*100:.0f}%")
    print(f"  N={base['n']}  Win%={base['win_rate']*100:.1f}%  "
          f"PT%={base['pt_rate']*100:.1f}%  SL%={base['sl_rate']*100:.1f}%  "
          f"NC%={base['nc_rate']*100:.1f}%  Settle%={base['se_rate']*100:.1f}%")
    print(f"  Avg P&L: {base['avg_pnl']:+.2f}¢/trade   Total: {base['total_pnl']:+.1f}¢")

    # ── 2. PT × SL joint sweep ────────────────────────────────────────────────
    profit_takes = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75, 0.90]
    stop_losses  = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    _hdr("2. PT × SL JOINT SWEEP  (NC_hours=2.0  NC_drawdown=15%)")
    sweep = run_sweep(trades, profit_takes, stop_losses,
                      nc_hours=CURRENT_NC_HOURS, nc_drawdown=CURRENT_NC_DRAWDOWN)
    print_sweep_table(sweep, (CURRENT_PT, CURRENT_SL), "PT%", "SL%")
    best_key = _find_best(sweep)
    best_pt, best_sl = best_key

    # ── 3. Near-close trailing sweep ─────────────────────────────────────────
    _hdr(f"3. NEAR-CLOSE TRAILING SWEEP  (PT={best_pt*100:.0f}%  SL={best_sl*100:.0f}%)")
    nc_hours_list    = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    nc_drawdown_list = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    sweep_nc = run_sweep_nc(trades, best_pt, best_sl, nc_hours_list, nc_drawdown_list)
    print_sweep_table(sweep_nc, (CURRENT_NC_HOURS, CURRENT_NC_DRAWDOWN),
                      "Hours", "Draw%")
    best_nc_key   = _find_best(sweep_nc)
    best_nc_hours, best_nc_draw = best_nc_key

    # ── 4. Per-kind breakdown ─────────────────────────────────────────────────
    _hdr(f"4. PER-KIND BREAKDOWN  (PT={best_pt*100:.0f}%  SL={best_sl*100:.0f}%"
         f"  NC={best_nc_hours}h / {best_nc_draw*100:.0f}%)")
    print_breakdown(trades, best_pt, best_sl, best_nc_hours, best_nc_draw,
                    "opportunity_kind")

    # ── 5. Per-source breakdown ────────────────────────────────────────────────
    _hdr(f"5. PER-SOURCE BREAKDOWN  (PT={best_pt*100:.0f}%  SL={best_sl*100:.0f}%"
         f"  NC={best_nc_hours}h / {best_nc_draw*100:.0f}%)")
    print_breakdown(trades, best_pt, best_sl, best_nc_hours, best_nc_draw, "source")

    # ── 6. Per-kind optimal PT/SL ─────────────────────────────────────────────
    _hdr("6. PER-KIND OPTIMAL PT/SL  (independent optimization per kind)")
    from collections import defaultdict
    by_kind: dict[str, list] = defaultdict(list)
    for t in trades:
        by_kind[t["opportunity_kind"]].append(t)

    print(f"  {'Kind':>14}  {'N':>5}  {'BestPT':>7}  {'BestSL':>7}  {'Avg¢':>8}  {'vs.base':>10}")
    for kind in sorted(by_kind):
        ktrades = by_kind[kind]
        ksweep  = run_sweep(ktrades, profit_takes, stop_losses,
                            nc_hours=CURRENT_NC_HOURS, nc_drawdown=CURRENT_NC_DRAWDOWN)
        if not ksweep:
            continue
        kbest = _find_best(ksweep)
        kd    = ksweep[kbest]
        # baseline for this kind
        kbase_sims = [
            s for t in ktrades
            if (s := simulate_exit(t, CURRENT_PT, CURRENT_SL,
                                   CURRENT_NC_HOURS, CURRENT_NC_DRAWDOWN)) is not None
        ]
        kbase_avg = _agg(kbase_sims)["avg_pnl"] if kbase_sims else 0.0
        delta     = kd["avg_pnl"] - kbase_avg
        print(f"  {kind:>14}  {len(ktrades):>5}  "
              f"PT={kbest[0]*100:.0f}%  SL={kbest[1]*100:.0f}%  "
              f"{kd['avg_pnl']:>+8.2f}  {delta:>+9.2f}¢")

    # ── 7. Peak pct_gain analysis ─────────────────────────────────────────────
    _hdr("7. PEAK PCT_GAIN ANALYSIS  (how high did each trade actually get, incl. post-exit?)")
    peaks = []
    for t in trades:
        snaps = t.get("snapshots_full") or t["snapshots"]
        peak  = max((p for p, _ in snaps if p is not None), default=None)
        if peak is not None:
            entry_cost = 100 - t["limit_price"]
            peaks.append({
                "peak": peak,
                "peak_cents": peak * entry_cost,
                "kind": t["opportunity_kind"],
                "outcome": t.get("outcome"),
            })
    if peaks:
        print(f"  Total trades with peak data: {len(peaks)}")
        thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00]
        print(f"  {'Threshold':>10}  {'% reached':>10}  {'N reached':>10}")
        for thr in thresholds:
            reached = sum(1 for p in peaks if p["peak"] >= thr)
            print(f"  ≥{thr*100:>5.0f}%    {reached/len(peaks)*100:>9.1f}%  {reached:>10}")
        avg_peak = sum(p["peak"] for p in peaks) / len(peaks)
        avg_peak_c = sum(p["peak_cents"] for p in peaks) / len(peaks)
        print(f"\n  Avg peak pct_gain: {avg_peak*100:+.1f}%  ({avg_peak_c:+.1f}¢)")

        # By kind
        from collections import defaultdict
        by_k: dict = defaultdict(list)
        for p in peaks:
            by_k[p["kind"]].append(p)
        print(f"\n  {'Kind':>14}  {'N':>5}  {'AvgPeak%':>9}  {'Reach20%':>9}  {'Reach50%':>9}")
        for k in sorted(by_k):
            g = by_k[k]
            avg_p    = sum(x["peak"] for x in g) / len(g)
            reach20  = sum(1 for x in g if x["peak"] >= 0.20) / len(g)
            reach50  = sum(1 for x in g if x["peak"] >= 0.50) / len(g)
            print(f"  {k:>14}  {len(g):>5}  {avg_p*100:>+8.1f}%  "
                  f"{reach20*100:>8.1f}%  {reach50*100:>8.1f}%")

    # ── 8. Best vs worst trades ────────────────────────────────────────────────
    best_all_sims = [
        s for t in trades
        if (s := simulate_exit(t, best_pt, best_sl, best_nc_hours, best_nc_draw)) is not None
    ]
    _hdr(f"8. WORST 10 TRADES  (PT={best_pt*100:.0f}%  SL={best_sl*100:.0f}%)")
    print_worst_trades(best_all_sims, n=10)

    _hdr(f"9. BEST 10 TRADES  (PT={best_pt*100:.0f}%  SL={best_sl*100:.0f}%)")
    print_best_trades(best_all_sims, n=10)

    # ── Summary ────────────────────────────────────────────────────────────────
    _hdr("RECOMMENDATIONS")
    best_d = sweep[best_key]
    delta_avg   = best_d["avg_pnl"] - base["avg_pnl"]
    delta_total = best_d["total_pnl"] - base["total_pnl"]
    print(f"  Current live: PT={CURRENT_PT*100:.0f}%  SL={CURRENT_SL*100:.0f}%  "
          f"NC_hours={CURRENT_NC_HOURS}  NC_drawdown={CURRENT_NC_DRAWDOWN*100:.0f}%")
    print(f"  Current avg P&L:   {base['avg_pnl']:+.2f}¢/trade   "
          f"total={base['total_pnl']:+.1f}¢")
    print()
    print(f"  Optimal PT × SL:   PT={best_pt*100:.0f}%  SL={best_sl*100:.0f}%")
    print(f"  Optimal avg P&L:   {best_d['avg_pnl']:+.2f}¢/trade   "
          f"total={best_d['total_pnl']:+.1f}¢")
    print(f"  Improvement:       {delta_avg:+.2f}¢/trade   {delta_total:+.1f}¢ total")
    print()
    print(f"  Optimal NC params: hours={best_nc_hours:.1f}  drawdown={best_nc_draw*100:.0f}%")
    best_nc_d = sweep_nc[best_nc_key]
    print(f"  Optimal NC avg:    {best_nc_d['avg_pnl']:+.2f}¢/trade   "
          f"total={best_nc_d['total_pnl']:+.1f}¢")

    # Per-kind optimal PT/SL env-var block for easy copy-paste
    print()
    print("  Suggested EXIT_SOURCE_PROFIT_TAKE / EXIT_SOURCE_STOP_LOSS JSON dicts")
    print("  (based on per-kind optimization — review before applying):")
    pt_overrides: dict[str, float] = {}
    sl_overrides: dict[str, float] = {}
    for kind in sorted(by_kind):
        ktrades = by_kind[kind]
        ksweep  = run_sweep(ktrades, profit_takes, stop_losses,
                            nc_hours=best_nc_hours, nc_drawdown=best_nc_draw)
        if not ksweep:
            continue
        kbest_key = _find_best(ksweep)
        kpt, ksl  = kbest_key
        if abs(kpt - best_pt) > 0.01:
            pt_overrides[kind] = kpt
        if abs(ksl - best_sl) > 0.01:
            sl_overrides[kind] = ksl

    import json
    if pt_overrides:
        print(f"  EXIT_SOURCE_PROFIT_TAKE='{json.dumps(pt_overrides)}'")
    if sl_overrides:
        print(f"  EXIT_SOURCE_STOP_LOSS='{json.dumps(sl_overrides)}'")
    if not pt_overrides and not sl_overrides:
        print("  (all kinds share the same optimal PT/SL — no per-kind overrides needed)")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimize exit parameters for historical NO trades."
    )
    parser.add_argument(
        "--kinds", nargs="+", default=None,
        metavar="KIND",
        help="Filter by opportunity_kind (e.g. forecast_no band_arb numeric). "
             "Default: all kinds.",
    )
    parser.add_argument(
        "--min-snaps", type=int, default=3,
        help="Minimum number of price snapshots required per trade (default: 3).",
    )
    parser.add_argument(
        "--resolved-only", action="store_true",
        help="Only include trades with a known outcome (won/lost).",
    )
    parser.add_argument(
        "--from-id", type=int, default=None, metavar="ID",
        help="Only include trades with id >= ID (e.g. --from-id 58 to skip trades "
             "without post-exit snapshot tracking).",
    )
    args = parser.parse_args()
    main(args)
