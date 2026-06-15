#!/usr/bin/env python3
"""Backtest forecast_no entry gate + exit parameter sweep.

Uses 78 settled forecast_no trades from opportunity_log.db.
Each trade's `note` JSON records all gate features at entry time,
allowing parameter sweeps without re-running the full bot.

Parts:
  A — Entry gate sweep (min_edge, model_spread, sources, direction, city)
  B — Exit sweep: profit-take threshold and stop-loss threshold
  C — Combined PT+SL grid
"""

from __future__ import annotations
import itertools
import json
import sqlite3
from pathlib import Path

DB = Path(__file__).parent.parent / "data" / "db" / "opportunity_log.db"
POLICY_CHANGE_DATE = "2026-05-13"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_trades() -> list[dict]:
    conn = sqlite3.connect(DB)
    rows = conn.execute("""
        SELECT
            id, ticker, logged_at, limit_price, count,
            exit_price_cents, exit_pnl_cents, exit_reason,
            corroborating_sources, note,
            peak_pct_gain, peak_at
        FROM trades
        WHERE source = 'forecast_no'
          AND exit_reason IS NOT NULL
          AND exit_pnl_cents IS NOT NULL
          AND count > 0
        ORDER BY id
    """).fetchall()
    conn.close()

    trades = []
    for row in rows:
        (tid, ticker, logged_at, limit_price, count,
         exit_price_cents, exit_pnl_cents, exit_reason,
         corroborating_sources, note_str,
         peak_pct_gain, peak_at) = row

        note = json.loads(note_str or "{}")
        metric = note.get("metric", "")
        city = metric.split("_")[-1] if metric else "?"
        sources = corroborating_sources.split(",") if corroborating_sources else []
        has_om = any("open_meteo" in s for s in sources)

        # exit_price_per_contract: actual exit per contract
        if exit_price_cents is not None and count:
            epc = exit_price_cents / count
        else:
            epc = None

        trades.append({
            "id":              tid,
            "ticker":          ticker,
            "logged_at":       logged_at,
            "limit_price":     limit_price,     # NO_ask at entry (¢/contract)
            "count":           count,
            "exit_price_cents": exit_price_cents,  # total exit value
            "exit_pnl_cents":  exit_pnl_cents,
            "exit_reason":     exit_reason,
            "sources":         sources,
            "has_om":          has_om,
            "peak_pct_gain":   peak_pct_gain,   # max gain fraction during trade
            "peak_at":         peak_at,
            "exit_price_per_contract": epc,
            # Gate features from note
            "model_spread_f":  note.get("model_spread_f"),
            "min_edge_f":      note.get("min_edge_f"),
            "max_edge_f":      note.get("max_edge_f"),
            "source_count":    note.get("source_count"),
            "no_direction":    note.get("no_direction"),  # NO_HIGH | NO_LOW | None
            "direction":       note.get("direction"),      # between | over | under
            "obs_gap_f":       note.get("obs_gap_f"),
            "hours_to_close":  note.get("hours_to_close"),
            "metric":          metric,
            "city":            city,
            "win":             exit_pnl_cents > 0,
            "post_policy":     logged_at >= POLICY_CHANGE_DATE,
        })
    return trades


# ---------------------------------------------------------------------------
# Part A — Entry gate helpers
# ---------------------------------------------------------------------------

def passes_gate(t: dict, cfg: dict) -> bool:
    if t["min_edge_f"] is not None and t["min_edge_f"] < cfg["min_edge_f"]:
        return False
    if t["model_spread_f"] is not None:
        if cfg["spread_max"] is not None and t["model_spread_f"] > cfg["spread_max"]:
            return False
        if cfg["spread_min"] is not None and t["model_spread_f"] < cfg["spread_min"]:
            return False
    if t["source_count"] is not None and t["source_count"] < cfg["min_sources"]:
        return False
    if cfg["require_om"] and not t["has_om"]:
        return False
    if t["city"] in cfg["blacklist"]:
        return False
    if cfg["block_no_high"] and t["no_direction"] == "NO_HIGH":
        return False
    return True


def stats(trades: list[dict]) -> dict:
    if not trades:
        return {"n": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
                "total_pnl": 0.0, "avg_pnl": 0.0}
    wins = sum(1 for t in trades if t["win"])
    total = sum(t["exit_pnl_cents"] for t in trades)
    return {
        "n":         len(trades),
        "wins":      wins,
        "losses":    len(trades) - wins,
        "win_rate":  wins / len(trades),
        "total_pnl": total,
        "avg_pnl":   total / len(trades),
    }


def _fmt(r: dict) -> str:
    return (
        f"n={r['n']:3d}  wins={r['wins']:3d}/{r['losses']:3d}"
        f"  WR={r['win_rate']:5.1%}"
        f"  total={r['total_pnl']:+8.1f}¢  avg={r['avg_pnl']:+7.1f}¢"
    )


# ---------------------------------------------------------------------------
# Part B — Exit simulation helpers
# ---------------------------------------------------------------------------

def sim_pt(trades: list[dict], pt_thresh: float) -> float:
    """Simulate P&L if PT fires at pt_thresh (fraction, e.g. 0.30 = 30%)."""
    total = 0.0
    for t in trades:
        peak = t["peak_pct_gain"]
        if peak is not None and peak >= pt_thresh:
            # PT fires: exit at limit_price × (1 + pt_thresh) per contract
            sim_exit_per = t["limit_price"] * (1 + pt_thresh)
            sim_pnl = (sim_exit_per - t["limit_price"]) * t["count"]
        else:
            sim_pnl = t["exit_pnl_cents"]
        total += sim_pnl
    return total


def sim_sl(trades: list[dict], sl_thresh: float) -> float:
    """Simulate P&L if SL fires at sl_thresh (fraction, e.g. 0.70 = 70% of entry).

    Conservative: assumes continuous descent, so any SL level above actual
    exit price would have fired. Actual improvement likely less (price may gap).
    """
    total = 0.0
    for t in trades:
        if t["exit_reason"] == "stop_loss":
            sl_exit_per = t["limit_price"] * sl_thresh
            # Only improves if SL exit is above actual exit
            actual_per = t["exit_price_per_contract"] if t["exit_price_per_contract"] is not None else 0.0
            sim_exit_per = max(sl_exit_per, actual_per)
            sim_pnl = (sim_exit_per - t["limit_price"]) * t["count"]
        else:
            sim_pnl = t["exit_pnl_cents"]
        total += sim_pnl
    return total


def sim_pt_sl(trades: list[dict], pt_thresh: float, sl_thresh: float) -> float:
    """Simulate both PT and SL simultaneously."""
    total = 0.0
    for t in trades:
        peak = t["peak_pct_gain"]
        if peak is not None and peak >= pt_thresh:
            sim_exit_per = t["limit_price"] * (1 + pt_thresh)
            sim_pnl = (sim_exit_per - t["limit_price"]) * t["count"]
        elif t["exit_reason"] == "stop_loss":
            sl_exit_per = t["limit_price"] * sl_thresh
            actual_per = t["exit_price_per_contract"] if t["exit_price_per_contract"] is not None else 0.0
            sim_exit_per = max(sl_exit_per, actual_per)
            sim_pnl = (sim_exit_per - t["limit_price"]) * t["count"]
        else:
            sim_pnl = t["exit_pnl_cents"]
        total += sim_pnl
    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    trades = load_trades()
    if not trades:
        print("No settled forecast_no trades found.")
        return

    total = stats(trades)
    print(f"Loaded {len(trades)} settled forecast_no trades\n")

    # ==========================================================
    # PART A — Entry gate sweep
    # ==========================================================
    print("=" * 70)
    print("PART A — ENTRY GATE SWEEP")
    print("=" * 70)

    # Baseline
    baseline_cfg = dict(
        min_edge_f=3.0, spread_max=6.0, spread_min=3.0,
        min_sources=2, require_om=True, blacklist=frozenset(), block_no_high=False,
    )
    bl = [t for t in trades if passes_gate(t, baseline_cfg)]
    print(f"\nBaseline (current policy): {_fmt(stats(bl))}")
    print(f"All trades (no filter):    {_fmt(total)}")

    # Parameter grid
    grid = {
        "min_edge_f":    [2.5, 3.0, 3.5, 4.0, 4.5],
        "spread_max":    [5.0, 6.0, 7.0, 8.0],
        "spread_min":    [0.0, 2.0, 3.0, 4.0],
        "min_sources":   [2, 3, 4],
        "require_om":    [True, False],
        "blacklist":     [frozenset(), frozenset({"bos"}), frozenset({"bos", "ny"})],
        "block_no_high": [False, True],
    }

    results = []
    for combo in itertools.product(*grid.values()):
        cfg = dict(zip(grid.keys(), combo))
        filtered = [t for t in trades if passes_gate(t, cfg)]
        if len(filtered) < 8:
            continue
        s = stats(filtered)
        results.append({**cfg, **s})

    results.sort(key=lambda r: r["total_pnl"], reverse=True)

    print(f"\nTop 20 configs by total P&L (n ≥ 8):\n")
    hdr = (f"{'minE':>4} {'spMx':>4} {'spMn':>4} {'om':>5} {'src':>3}"
           f" {'blk':>12} {'noHi':>5} | {'n':>4} {'wins':>5} {'WR':>6}"
           f" {'tot_pnl':>9} {'avg_pnl':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in results[:20]:
        blk = ",".join(sorted(r["blacklist"])) or "none"
        print(
            f"{r['min_edge_f']:4.1f} {r['spread_max']:4.0f} {r['spread_min']:4.0f}"
            f" {str(r['require_om']):>5} {r['min_sources']:3d}"
            f" {blk:>12} {str(r['block_no_high']):>5}"
            f" | {r['n']:4d} {r['wins']:5d} {r['win_rate']:6.1%}"
            f" {r['total_pnl']:9.1f}¢ {r['avg_pnl']:8.1f}¢"
        )

    by_avg = sorted([r for r in results if r["n"] >= 12], key=lambda r: r["avg_pnl"], reverse=True)
    print(f"\nTop 10 configs by avg P&L (n ≥ 12):\n")
    print(hdr)
    print("-" * len(hdr))
    for r in by_avg[:10]:
        blk = ",".join(sorted(r["blacklist"])) or "none"
        print(
            f"{r['min_edge_f']:4.1f} {r['spread_max']:4.0f} {r['spread_min']:4.0f}"
            f" {str(r['require_om']):>5} {r['min_sources']:3d}"
            f" {blk:>12} {str(r['block_no_high']):>5}"
            f" | {r['n']:4d} {r['wins']:5d} {r['win_rate']:6.1%}"
            f" {r['total_pnl']:9.1f}¢ {r['avg_pnl']:8.1f}¢"
        )

    # Direction breakdown
    print("\n--- Direction breakdown ---")
    for nd in ["NO_HIGH", "NO_LOW", None]:
        subset = [t for t in trades if t["no_direction"] == nd]
        if subset:
            s = stats(subset)
            label = nd if nd else "(over/under/none)"
            print(f"  {label:14s}: {_fmt(s)}")

    # City breakdown (worst 8)
    from collections import defaultdict
    city_trades: dict[str, list] = defaultdict(list)
    for t in trades:
        city_trades[t["city"]].append(t)
    city_stats = {c: stats(v) for c, v in city_trades.items()}
    sorted_cities = sorted(city_stats.items(), key=lambda x: x[1]["total_pnl"])
    print("\n--- City breakdown (worst 8 by total P&L) ---")
    for city, s in sorted_cities[:8]:
        print(f"  {city:6s}: {_fmt(s)}")

    # Pre vs post policy
    pre  = [t for t in trades if not t["post_policy"]]
    post = [t for t in trades if t["post_policy"]]
    print(f"\n--- Period breakdown ---")
    print(f"  Pre-{POLICY_CHANGE_DATE}:  {_fmt(stats(pre))}")
    print(f"  Post-{POLICY_CHANGE_DATE}: {_fmt(stats(post))}")

    # Worst trades × which gates block them
    print(f"\n--- Top 10 losses and what blocks them ---")
    worst = sorted(trades, key=lambda t: t["exit_pnl_cents"])[:10]
    gate_labels = {
        "min_edge≥3.5": lambda t: (t["min_edge_f"] or 0) >= 3.5,
        "spread≤5":     lambda t: (t["model_spread_f"] or 999) <= 5.0,
        "src≥3":        lambda t: (t["source_count"] or 0) >= 3,
        "noHigh":       lambda t: t["no_direction"] != "NO_HIGH",
        "no-bos":       lambda t: t["city"] != "bos",
    }
    gl_names = list(gate_labels.keys())
    print(f"  {'id':>4} {'ticker':>30} {'pnl':>7} | " + " ".join(f"{n:>9}" for n in gl_names))
    for t in worst:
        flags = " ".join(
            f"{'BLOCK':>9}" if not fn(t) else f"{'pass':>9}"
            for fn in gate_labels.values()
        )
        print(f"  {t['id']:4d} {t['ticker']:>30} {t['exit_pnl_cents']:+7.0f}¢ | {flags}")

    # ==========================================================
    # PART B — Exit parameter sweep
    # ==========================================================
    print("\n" + "=" * 70)
    print("PART B — EXIT PARAMETER SWEEP")
    print("=" * 70)

    # Peak gain distribution for stop-loss trades
    sl_trades = [t for t in trades if t["exit_reason"] == "stop_loss"]
    print(f"\nStop-loss trades: {len(sl_trades)}")
    peak_positive = [t for t in sl_trades if (t["peak_pct_gain"] or 0) > 0.05]
    print(f"  Had >5% unrealized gain before crash: {len(peak_positive)} "
          f"({len(peak_positive)/max(len(sl_trades),1):.0%})")
    if peak_positive:
        avg_peak = sum(t["peak_pct_gain"] for t in peak_positive) / len(peak_positive)
        print(f"  Avg peak gain among those: {avg_peak:.1%}")

    # PT threshold sweep
    pt_thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.75]
    print(f"\n--- PT threshold sweep (all {len(trades)} trades) ---")
    print(f"  {'PT%':>5} | {'total_pnl':>10} {'improvement':>12} {'trades_captured':>16}")
    actual_total = total["total_pnl"]
    for pt in pt_thresholds:
        sim_total = sim_pt(trades, pt)
        captured = sum(1 for t in trades if (t["peak_pct_gain"] or -1) >= pt)
        print(f"  {pt:5.0%} | {sim_total:+10.1f}¢  {sim_total - actual_total:+12.1f}¢  {captured:16d}/{len(trades)}")

    # SL threshold sweep
    sl_thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    print(f"\n--- SL threshold sweep (conservative: assumes continuous descent) ---")
    print(f"  {'SL%':>5} | {'total_pnl':>10} {'improvement':>12}")
    for sl in sl_thresholds:
        sim_total = sim_sl(trades, sl)
        print(f"  {sl:5.0%} | {sim_total:+10.1f}¢  {sim_total - actual_total:+12.1f}¢")

    # ==========================================================
    # PART C — Combined PT + SL grid
    # ==========================================================
    print("\n" + "=" * 70)
    print("PART C — COMBINED PT + SL GRID")
    print("=" * 70)
    pt_grid = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    sl_grid = [0.40, 0.50, 0.60, 0.70, 0.80]

    print(f"\n  PT\\SL |", end="")
    for sl in sl_grid:
        print(f"  SL={sl:.0%}", end="")
    print()
    print("  " + "-" * (7 + 9 * len(sl_grid)))
    for pt in pt_grid:
        print(f"  PT={pt:.0%} |", end="")
        for sl in sl_grid:
            sim_total = sim_pt_sl(trades, pt, sl)
            print(f" {sim_total:+8.0f}¢", end="")
        print()

    # Best combination
    combos = [
        (pt, sl, sim_pt_sl(trades, pt, sl))
        for pt in pt_grid for sl in sl_grid
    ]
    combos.sort(key=lambda x: x[2], reverse=True)
    best_pt, best_sl, best_pnl = combos[0]
    print(f"\n  Best combo: PT={best_pt:.0%} + SL={best_sl:.0%} → {best_pnl:+.1f}¢"
          f" vs actual {actual_total:+.1f}¢"
          f" (improvement: {best_pnl - actual_total:+.1f}¢ / ${(best_pnl - actual_total)/100:.2f})")

    print("\n[Note: SL simulation is optimistic — assumes price descends continuously")
    print(" to final exit. Actual improvement may be less if price gaps through SL level.]")


if __name__ == "__main__":
    main()
