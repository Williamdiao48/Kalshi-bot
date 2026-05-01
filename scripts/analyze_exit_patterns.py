"""Exit pattern analysis — what entry-time features predict exit quality?

For every early-exited trade, computes how much of the available gain was
captured vs. left on the table, then segments by entry-time features to find
which profit-take thresholds are miscalibrated.

Sections
--------
  1  Data coverage summary
  2  Per-trade exit quality metrics (internal — used by later sections)
  3  Feature dimension report (7 axes: kind_side, entry_cost, score, spread,
     corroboration count, market_p, market_type)
  4  PT threshold simulation per kind_side segment
  5  "Left on table" top-20 list (profit_take exits only)
  6  Recommended EXIT_SOURCE_PROFIT_TAKE override changes

Usage
-----
  venv/bin/python scripts/analyze_exit_patterns.py
  venv/bin/python scripts/analyze_exit_patterns.py --kinds band_arb forecast_no numeric
  venv/bin/python scripts/analyze_exit_patterns.py --from-id 80
  venv/bin/python scripts/analyze_exit_patterns.py --out exit_analysis.txt
"""

from __future__ import annotations

import argparse
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DB_PATH = Path(__file__).parent.parent / "opportunity_log.db"

# Current live EXIT_SOURCE_PROFIT_TAKE entries (global default + per-source).
# These are the thresholds we're evaluating.
CURRENT_PT: dict[str, float] = {
    "noaa_observed:yes":    0.50,
    "noaa_observed":        0.75,
    "metar:yes":            0.50,
    "metar":                0.80,
    "nws_alert":            0.80,
    "eia":                  0.40,
    "eia_inventory":        0.40,
    "noaa_day2:no":         0.07,
    "noaa_day2_early:no":   0.07,
    "noaa_day2:yes":        0.35,
    "noaa_day2_early:yes":  0.35,
    "noaa":                 0.40,
    "noaa_day2":            0.20,
    "polymarket":           0.25,
    "obs_trajectory:yes":   0.30,
    "band_arb:yes":         0.15,
    "band_arb:no":          2.00,
    "forecast_no":          0.40,
    "numeric":              0.75,
    "binance":              0.35,
    "coinbase":             0.35,
    "_global":              0.20,   # EXIT_PROFIT_TAKE global fallback
}

# Simulation: fixed SL (mirrors current live EXIT_STOP_LOSS=0.70).
SIM_STOP_LOSS = 0.70
SIM_NC_HOURS    = 2.0
SIM_NC_DRAWDOWN = 0.15

# PT values to sweep in Section 4.
PT_SWEEP = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00, 2.00]

# Minimum N in a segment to emit recommendations in Section 6.
MIN_N_RECOMMEND = 5
# Minimum difference between optimal and current PT to flag as miscalibrated.
MIN_PT_DELTA = 0.05


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _has_col(cur: sqlite3.Cursor, table: str, col: str) -> bool:
    cur.execute(f"PRAGMA table_info({table})")
    return any(row["name"] == col for row in cur.fetchall())


def load_trades(
    db: sqlite3.Connection,
    min_snaps: int,
    sides: list[str] | None,
    kinds: list[str] | None,
    from_id: int | None,
) -> list[dict]:
    """Load all early-exited trades with snapshot trajectories.

    Includes both YES and NO sides, all opportunity kinds.
    """
    cur = db.cursor()
    cur.row_factory = sqlite3.Row

    where: list[str] = ["t.exited_at IS NOT NULL"]
    params: list = []

    if sides:
        placeholders = ",".join("?" * len(sides))
        where.append(f"t.side IN ({placeholders})")
        params.extend(sides)

    if kinds:
        placeholders = ",".join("?" * len(kinds))
        where.append(f"t.opportunity_kind IN ({placeholders})")
        params.extend(kinds)

    if from_id is not None:
        where.append("t.id >= ?")
        params.append(from_id)

    where_sql = " AND ".join(where)

    # Detect optional columns (added via migrations over time).
    has_corr   = _has_col(cur, "trades", "corroborating_sources")
    has_mpe    = _has_col(cur, "trades", "market_p_entry")
    has_bid_e  = _has_col(cur, "trades", "yes_bid_entry")
    has_ask_e  = _has_col(cur, "trades", "yes_ask_entry")
    has_pp     = _has_col(cur, "trades", "peak_past")

    extra = []
    if has_corr:  extra.append("t.corroborating_sources")
    if has_mpe:   extra.append("t.market_p_entry")
    if has_bid_e: extra.append("t.yes_bid_entry")
    if has_ask_e: extra.append("t.yes_ask_entry")
    if has_pp:    extra.append("t.peak_past")
    extra_sql = (", " + ", ".join(extra)) if extra else ""

    cur.execute(f"""
        SELECT t.id, t.ticker, t.opportunity_kind, t.source,
               t.side, t.count, t.limit_price, t.score,
               t.outcome, t.exit_pnl_cents, t.exit_reason,
               t.exit_price_cents
               {extra_sql},
               COUNT(ps.id) AS n_snaps
        FROM trades t
        JOIN price_snapshots ps ON ps.trade_id = t.id
        WHERE {where_sql}
        GROUP BY t.id
        HAVING n_snaps >= ?
        ORDER BY t.id
    """, (*params, min_snaps))

    trades = [dict(row) for row in cur.fetchall()]

    # Detect post_exit column.
    has_post_exit = _has_col(cur, "price_snapshots", "post_exit")

    for trade in trades:
        if has_post_exit:
            cur.execute("""
                SELECT pct_gain, days_to_close, exit_price, post_exit
                FROM price_snapshots
                WHERE trade_id = ?
                ORDER BY id
            """, (trade["id"],))
            rows = [dict(r) for r in cur.fetchall()]
            trade["snapshots"]      = [(r["pct_gain"], r["days_to_close"]) for r in rows if not r["post_exit"]]
            trade["snapshots_post"] = [(r["pct_gain"], r["days_to_close"]) for r in rows if r["post_exit"]]
            trade["snapshots_full"] = [(r["pct_gain"], r["days_to_close"]) for r in rows]
        else:
            cur.execute("""
                SELECT pct_gain, days_to_close
                FROM price_snapshots
                WHERE trade_id = ?
                ORDER BY id
            """, (trade["id"],))
            rows = [dict(r) for r in cur.fetchall()]
            trade["snapshots"]      = [(r["pct_gain"], r["days_to_close"]) for r in rows]
            trade["snapshots_post"] = []
            trade["snapshots_full"] = rows

    return trades


# ---------------------------------------------------------------------------
# Per-trade metrics (Section 2)
# ---------------------------------------------------------------------------

def _safe_mean(vals: list[float]) -> float | None:
    vals = [v for v in vals if v is not None]
    return statistics.mean(vals) if vals else None


def compute_metrics(trade: dict) -> dict:
    """Derive all exit-quality metrics for one trade."""
    side       = trade["side"]
    lp         = trade["limit_price"]
    count      = trade["count"]
    entry_cost = lp if side == "yes" else (100 - lp)
    if entry_cost <= 0:
        return {}

    total_cost = entry_cost * count

    # Captured gain from the actual exit.
    exit_pnl  = trade.get("exit_pnl_cents") or 0.0
    captured_pct = exit_pnl / total_cost

    # Peak across PRE-exit snapshots only.
    pre_gains = [g for g, _ in trade["snapshots"] if g is not None]
    peak_at_exit_pct = max(pre_gains, default=None)

    # Post-exit snapshot metrics (counterfactual).
    post_gains = [g for g, _ in trade["snapshots_post"] if g is not None]
    post_exit_peak_pct  = max(post_gains, default=None)
    post_exit_final_pct = post_gains[-1] if post_gains else None

    # Global peak across ALL snapshots.
    all_gains = [g for g, _ in trade["snapshots_full"] if g is not None]
    peak_pct = max(all_gains, default=None)

    # Efficiency: how much of the total peak did we capture?
    efficiency: float | None = None
    if peak_pct is not None and peak_pct > 0:
        efficiency = min(1.0, max(0.0, captured_pct / peak_pct))

    # Reversal: did the market move against us AFTER we exited?
    reversal: bool | None = None
    if post_exit_final_pct is not None:
        reversal = post_exit_final_pct < captured_pct

    # Hold-to-settlement (when outcome is known).
    outcome = trade.get("outcome")
    hold_pnl_pct: float | None = None
    exit_improvement: float | None = None
    if outcome == "won":
        # Contract pays 100¢ at settlement; we paid entry_cost.
        hold_pnl_pct = (100 - entry_cost) / entry_cost
        exit_improvement = captured_pct - hold_pnl_pct
    elif outcome == "lost":
        hold_pnl_pct = -1.0
        exit_improvement = captured_pct - hold_pnl_pct

    # Spread at entry.
    bid_e = trade.get("yes_bid_entry")
    ask_e = trade.get("yes_ask_entry")
    spread_at_entry: int | None = (ask_e - bid_e) if (bid_e is not None and ask_e is not None) else None

    # Corroboration count.
    corr_raw = trade.get("corroborating_sources") or ""
    corr_count = len([s for s in corr_raw.split(",") if s.strip()]) if corr_raw.strip() else 0

    # Market type from ticker prefix.
    ticker = trade["ticker"]
    if "KXLOWT" in ticker:
        market_type = "KXLOWT"
    elif "KXHIGH" in ticker:
        market_type = "KXHIGH"
    elif "KXBTC" in ticker or "KXETH" in ticker:
        market_type = "crypto"
    elif "KXWTI" in ticker or "KXGAS" in ticker:
        market_type = "energy"
    elif "KXEURUSD" in ticker or "KXUSDJPY" in ticker or "KXGBPUSD" in ticker:
        market_type = "forex"
    else:
        market_type = "other"

    return {
        "trade_id":           trade["id"],
        "ticker":             ticker,
        "source":             trade.get("source") or "unknown",
        "side":               side,
        "opportunity_kind":   trade.get("opportunity_kind") or "unknown",
        "score":              trade.get("score"),
        "entry_cost":         entry_cost,
        "total_cost":         total_cost,
        "exit_reason":        trade.get("exit_reason") or "unknown",
        "outcome":            outcome,
        # Gain metrics
        "captured_pct":       captured_pct,
        "peak_at_exit_pct":   peak_at_exit_pct,
        "post_exit_peak_pct": post_exit_peak_pct,
        "post_exit_final_pct":post_exit_final_pct,
        "peak_pct":           peak_pct,
        "efficiency":         efficiency,
        "reversal":           reversal,
        "hold_pnl_pct":       hold_pnl_pct,
        "exit_improvement":   exit_improvement,
        # Feature buckets
        "kind_side":          f"{trade.get('opportunity_kind','?')}:{side}",
        "market_type":        market_type,
        "spread_at_entry":    spread_at_entry,
        "corr_count":         corr_count,
        "market_p_entry":     trade.get("market_p_entry"),
        # Raw for snapshot replay
        "_snapshots_full":    trade["snapshots_full"],
        "_limit_price":       lp,
    }


def _entry_cost_bucket(ec: int) -> str:
    if ec <= 15:   return "≤15¢"
    if ec <= 30:   return "16–30¢"
    if ec <= 50:   return "31–50¢"
    if ec <= 70:   return "51–70¢"
    if ec <= 90:   return "71–90¢"
    return ">90¢"


def _score_bucket(s: float | None) -> str:
    if s is None:  return "n/a"
    if s < 0.80:   return "0.75–0.80"
    if s < 0.85:   return "0.80–0.85"
    if s < 0.90:   return "0.85–0.90"
    if s < 0.95:   return "0.90–0.95"
    return "≥0.95"


def _spread_bucket(sp: int | None) -> str:
    if sp is None: return "n/a"
    if sp <= 5:    return "≤5¢"
    if sp <= 10:   return "6–10¢"
    if sp <= 20:   return "11–20¢"
    return ">20¢"


def _corr_bucket(n: int) -> str:
    if n == 0: return "0"
    if n == 1: return "1"
    if n == 2: return "2"
    return "3+"


def _mkt_p_bucket(p: float | None) -> str:
    if p is None:  return "n/a"
    if p < 0.20:   return "<0.20"
    if p < 0.35:   return "0.20–0.35"
    if p < 0.50:   return "0.35–0.50"
    if p < 0.65:   return "0.50–0.65"
    return "≥0.65"


# ---------------------------------------------------------------------------
# PT simulation helpers (Section 4)
# ---------------------------------------------------------------------------

def simulate_pt(
    m: dict,
    pt: float,
    sl: float = SIM_STOP_LOSS,
    nc_hours: float = SIM_NC_HOURS,
    nc_drawdown: float = SIM_NC_DRAWDOWN,
) -> dict:
    """Replay one trade under a given PT threshold. Returns simulated outcome dict."""
    entry_cost = m["entry_cost"]
    snaps = m["_snapshots_full"]
    nc_days = nc_hours / 24.0
    peak_pct = 0.0

    for pct_gain, days_to_close in snaps:
        if pct_gain is None:
            continue
        if pct_gain > peak_pct:
            peak_pct = pct_gain

        if pct_gain >= pt:
            return {"exit_reason": "profit_take", "pct_gain": pct_gain,
                    "pnl": pct_gain * entry_cost}

        if pct_gain <= -sl:
            return {"exit_reason": "stop_loss", "pct_gain": pct_gain,
                    "pnl": pct_gain * entry_cost}

        if (days_to_close is not None and days_to_close < nc_days
                and peak_pct > 0 and pct_gain < peak_pct - nc_drawdown):
            return {"exit_reason": "near_close_trailing", "pct_gain": pct_gain,
                    "pnl": pct_gain * entry_cost}

    # Settlement
    outcome = m.get("outcome")
    side    = m["side"]
    lp      = m["_limit_price"]
    if outcome == "won":
        final_pct = (100 - lp) / lp if side == "yes" else lp / (100 - lp)
    elif outcome == "lost":
        final_pct = -1.0
    else:
        final_pct = snaps[-1][0] if snaps and snaps[-1][0] is not None else 0.0

    return {"exit_reason": "settlement", "pct_gain": final_pct,
            "pnl": final_pct * entry_cost}


def _agg_sims(sims: list[dict]) -> dict:
    n      = len(sims)
    total  = sum(s["pnl"] for s in sims)
    wins   = sum(1 for s in sims if s["pnl"] > 0)
    return {"n": n, "total_pnl": total, "avg_pnl": total / n, "win_rate": wins / n}


def _resolve_current_pt(source: str, side: str) -> float:
    composite = f"{source}:{side}"
    if composite in CURRENT_PT:
        return CURRENT_PT[composite]
    return CURRENT_PT.get(source, CURRENT_PT["_global"])


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

SEP  = "=" * 90
SEP2 = "-" * 90

def _hdr(title: str, out) -> None:
    print(f"\n{SEP}", file=out)
    print(f"  {title}", file=out)
    print(SEP, file=out)


def _pct(v: float | None, decimals: int = 1) -> str:
    if v is None: return "  n/a"
    return f"{v * 100:+{decimals+4}.{decimals}f}%"


def _dim_table(
    metrics: list[dict],
    key_fn,
    axis_name: str,
    out,
) -> None:
    """Print one feature-dimension table (Section 3)."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for m in metrics:
        groups[key_fn(m)].append(m)

    header = (
        f"  {'Bucket':<18}  {'N':>5}  {'N_post':>6}  {'N_set':>5}"
        f"  {'captured%':>9}  {'peak%':>7}  {'effic':>6}"
        f"  {'reversal%':>9}  {'hold%':>7}  {'exit>hold':>9}"
    )
    print(f"\n  Axis: {axis_name}", file=out)
    print("  " + "-" * 86, file=out)
    print(header, file=out)
    print("  " + "-" * 86, file=out)

    for bucket in sorted(groups.keys()):
        grp = groups[bucket]
        n = len(grp)

        post_grp     = [m for m in grp if m["post_exit_peak_pct"] is not None]
        settled_grp  = [m for m in grp if m["hold_pnl_pct"] is not None]

        avg_cap  = _safe_mean([m["captured_pct"] for m in grp])
        avg_peak = _safe_mean([m["peak_pct"] for m in post_grp]) if post_grp else None
        avg_eff  = _safe_mean([m["efficiency"] for m in post_grp if m["efficiency"] is not None]) if post_grp else None

        rev_grp = [m for m in post_grp if m["reversal"] is not None]
        rev_rate = (sum(1 for m in rev_grp if m["reversal"]) / len(rev_grp)) if rev_grp else None

        avg_hold  = _safe_mean([m["hold_pnl_pct"] for m in settled_grp]) if settled_grp else None
        impr_rate = (
            sum(1 for m in settled_grp if m["exit_improvement"] is not None and m["exit_improvement"] > 0)
            / len(settled_grp)
        ) if settled_grp else None

        def _f(v, decimals=1):
            return _pct(v, decimals) if v is not None else "    n/a"

        def _r(v):
            if v is None: return "    n/a"
            return f"{v * 100:6.1f}%"

        print(
            f"  {bucket:<18}  {n:>5}  {len(post_grp):>6}  {len(settled_grp):>5}"
            f"  {_f(avg_cap):>9}  {_f(avg_peak):>7}  {_f(avg_eff):>6}"
            f"  {_r(rev_rate):>9}  {_f(avg_hold):>7}  {_r(impr_rate):>9}",
            file=out,
        )

    print("  " + "-" * 86, file=out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze exit patterns across all trades.")
    parser.add_argument("--db",         default=str(DB_PATH), help="Path to opportunity_log.db")
    parser.add_argument("--min-snaps",  type=int, default=3,  help="Min price_snapshots per trade")
    parser.add_argument("--sides",      nargs="+", choices=["yes", "no"], help="Filter sides")
    parser.add_argument("--kinds",      nargs="+", help="Filter opportunity_kind(s)")
    parser.add_argument("--from-id",    type=int, help="Only trades with id >= this")
    parser.add_argument("--out",        help="Also write output to this file")
    args = parser.parse_args()

    db = sqlite3.connect(args.db)
    db.row_factory = sqlite3.Row

    out_file = open(args.out, "w") if args.out else None

    class _Tee:
        """Write to stdout and optionally a file simultaneously."""
        def __init__(self, *targets):
            self.targets = targets
        def write(self, s):
            for t in self.targets:
                t.write(s)
        def flush(self):
            for t in self.targets:
                t.flush()

    targets = [sys.stdout] + ([out_file] if out_file else [])
    out = _Tee(*targets)

    trades_raw = load_trades(
        db,
        min_snaps=args.min_snaps,
        sides=args.sides,
        kinds=args.kinds,
        from_id=args.from_id,
    )

    metrics_all = [m for t in trades_raw if (m := compute_metrics(t))]

    # -----------------------------------------------------------------------
    # CAVEATS
    # -----------------------------------------------------------------------
    print(f"""
{SEP}
  EXIT PATTERN ANALYSIS
  Generated from: {args.db}
  Filters: sides={args.sides or 'all'}, kinds={args.kinds or 'all'}, min_snaps={args.min_snaps}
{SEP}

  ⚠  CAVEATS — read before drawing conclusions
  {SEP2}
  1. Post-exit counterfactual data (snapshots_post) exists only when the bot
     was still running after the trade was exited. Near-session-end trades
     may show 0 post-exit rows — they are excluded from efficiency/peak metrics.
  2. Settlement outcome (trades.outcome) is NULL for most early exits. The
     hold-to-settlement comparison (Sections 3 + 6) only covers the subset
     where Kalshi eventually settled the market after our exit.
  3. Bucket sample sizes are small (often N<10). All findings are directional
     signals, not statistically significant conclusions.
  4. Optimal PT from Section 4 is in-sample — no cross-validation. Use as a
     directional guide, not a precise calibration target.
  5. The simulation fixes SL={SIM_STOP_LOSS:.0%} and does not co-optimize SL.
     Use optimize_no_exits.py for joint (PT, SL) sweeps.
""", file=out)

    # -----------------------------------------------------------------------
    # SECTION 1 — Data coverage
    # -----------------------------------------------------------------------
    _hdr("SECTION 1 — Data Coverage", out)

    n_total    = len(metrics_all)
    n_post     = sum(1 for m in metrics_all if m["post_exit_peak_pct"] is not None)
    n_settled  = sum(1 for m in metrics_all if m["outcome"] is not None)
    n_pt_exits = sum(1 for m in metrics_all if m["exit_reason"] == "profit_take")
    n_sl_exits = sum(1 for m in metrics_all if m["exit_reason"] == "stop_loss")

    print(f"""
  Total early-exited trades loaded    : {n_total}
  With post-exit snapshot data        : {n_post}   ({n_post/n_total*100:.0f}% of loaded)
  With settlement outcome             : {n_settled}  ({n_settled/n_total*100:.0f}% of loaded)
  Profit-take exits                   : {n_pt_exits}
  Stop-loss exits                     : {n_sl_exits}
  Other exits                         : {n_total - n_pt_exits - n_sl_exits}
""", file=out)

    # By kind
    kind_counts: dict[str, list] = defaultdict(list)
    for m in metrics_all:
        kind_counts[m["kind_side"]].append(m)

    print(f"  {'kind:side':<28}  {'N':>5}  {'N_post':>6}  {'N_set':>5}  {'PT_exits':>8}  {'SL_exits':>8}", file=out)
    print("  " + "-" * 70, file=out)
    for ks in sorted(kind_counts):
        grp = kind_counts[ks]
        print(
            f"  {ks:<28}  {len(grp):>5}"
            f"  {sum(1 for m in grp if m['post_exit_peak_pct'] is not None):>6}"
            f"  {sum(1 for m in grp if m['outcome'] is not None):>5}"
            f"  {sum(1 for m in grp if m['exit_reason']=='profit_take'):>8}"
            f"  {sum(1 for m in grp if m['exit_reason']=='stop_loss'):>8}",
            file=out,
        )

    # -----------------------------------------------------------------------
    # SECTION 3 — Feature dimension report
    # -----------------------------------------------------------------------
    _hdr("SECTION 3 — Feature Dimension Report", out)
    print("""
  For each dimension, columns are computed over all exited trades in that bucket.
  N_post = trades with ≥1 post-exit snapshot.  N_set = trades with settlement outcome.
  efficiency = captured% / peak%.  reversal% = % where market fell after exit.
  exit>hold% = % of settled trades where early exit beat holding to settlement.
""", file=out)

    _dim_table(metrics_all, lambda m: m["kind_side"],                              "kind_side",    out)
    _dim_table(metrics_all, lambda m: _entry_cost_bucket(m["entry_cost"]),         "entry_cost",   out)
    _dim_table(metrics_all, lambda m: _score_bucket(m.get("score")),               "score",        out)
    _dim_table(metrics_all, lambda m: _spread_bucket(m.get("spread_at_entry")),    "spread_entry", out)
    _dim_table(metrics_all, lambda m: _corr_bucket(m["corr_count"]),               "corr_sources", out)
    _dim_table(metrics_all, lambda m: _mkt_p_bucket(m.get("market_p_entry")),      "market_p",     out)
    _dim_table(metrics_all, lambda m: m["market_type"],                            "market_type",  out)

    # -----------------------------------------------------------------------
    # SECTION 4 — PT threshold simulation per kind_side
    # -----------------------------------------------------------------------
    _hdr("SECTION 4 — PT Threshold Simulation (by kind:side)", out)
    print(f"""
  Each row = total and avg P&L if profit-take threshold was set to that value.
  Fixed SL={SIM_STOP_LOSS:.0%}, NC trailing={SIM_NC_DRAWDOWN:.0%}@{SIM_NC_HOURS:.1f}h.
  Uses snapshots_full (pre + post exit) so simulated thresholds > actual can be reached.
  'current' column = actual exit P&L taken.  'hold' = settlement P&L where outcome known.
""", file=out)

    # Collect recommendations for Section 6.
    recommendations: list[dict] = []

    for ks in sorted(kind_counts):
        seg = kind_counts[ks]
        if not seg:
            continue

        src, side = (ks.rsplit(":", 1) + ["?"])[:2] if ":" in ks else (ks, "?")
        current_thresh = _resolve_current_pt(src, side)

        print(f"\n  {ks}  (N={len(seg)}, current_PT={current_thresh:.2f})", file=out)
        header = f"  {'PT':>6}  {'N_sim':>6}  {'total¢':>8}  {'avg¢':>7}  {'win%':>6}  {'PT_exits%':>10}"
        print(header, file=out)
        print("  " + "-" * 60, file=out)

        # Actual captured P&L (baseline).
        actual_pnl  = sum(m["captured_pct"] * m["entry_cost"] for m in seg)
        actual_wins = sum(1 for m in seg if m["captured_pct"] > 0)
        print(
            f"  {'actual':>6}  {len(seg):>6}  {actual_pnl:>8.1f}  "
            f"{actual_pnl/len(seg):>7.2f}  {actual_wins/len(seg)*100:>5.1f}%  —",
            file=out,
        )

        best_pt   = None
        best_pnl  = float("-inf")
        pt_results: dict[float, dict] = {}

        for pt in PT_SWEEP:
            sims = [simulate_pt(m, pt) for m in seg]
            agg  = _agg_sims(sims)
            pt_rate = sum(1 for s in sims if s["exit_reason"] == "profit_take") / len(sims)
            agg["pt_rate"] = pt_rate
            pt_results[pt] = agg

            marker = ""
            if abs(pt - current_thresh) < 1e-6:
                marker = " ← current"
            if agg["avg_pnl"] > best_pnl:
                best_pnl = agg["avg_pnl"]
                best_pt  = pt

            print(
                f"  {pt:>6.2f}  {agg['n']:>6}  {agg['total_pnl']:>8.1f}  "
                f"{agg['avg_pnl']:>7.2f}  {agg['win_rate']*100:>5.1f}%  "
                f"{pt_rate*100:>9.1f}%{marker}",
                file=out,
            )

        # Hold-to-settlement (settled subset only).
        settled = [m for m in seg if m["hold_pnl_pct"] is not None]
        if settled:
            hold_pnl = sum(m["hold_pnl_pct"] * m["entry_cost"] for m in settled)
            hold_wins = sum(1 for m in settled if m["hold_pnl_pct"] > 0)
            print(
                f"  {'hold':>6}  {len(settled):>6}  {hold_pnl:>8.1f}  "
                f"{hold_pnl/len(settled):>7.2f}  {hold_wins/len(settled)*100:>5.1f}%"
                f"  (settled subset only)",
                file=out,
            )

        if best_pt is not None:
            flag = "  *** MISCALIBRATED ***" if abs(best_pt - current_thresh) >= MIN_PT_DELTA else ""
            print(f"\n  → optimal PT: {best_pt:.2f}  (current: {current_thresh:.2f}){flag}", file=out)

        # Collect for Section 6.
        if (best_pt is not None
                and len(seg) >= MIN_N_RECOMMEND
                and abs(best_pt - current_thresh) >= MIN_PT_DELTA):
            # Efficiency and reversal rate for annotation.
            eff_vals = [m["efficiency"] for m in seg if m["efficiency"] is not None]
            rev_vals  = [m["reversal"]   for m in seg if m["reversal"] is not None]
            recommendations.append({
                "key":         ks,
                "src":         src,
                "side":        side,
                "current_pt":  current_thresh,
                "optimal_pt":  best_pt,
                "n":           len(seg),
                "efficiency":  statistics.mean(eff_vals) if eff_vals else None,
                "reversal_rate": (sum(1 for v in rev_vals if v) / len(rev_vals)) if rev_vals else None,
            })

    # -----------------------------------------------------------------------
    # SECTION 5 — "Left on table" top-20 list
    # -----------------------------------------------------------------------
    _hdr("SECTION 5 — Top-20 Missed Gains (profit_take exits, sorted by missed¢)", out)
    print("""
  missed¢ = (post_exit_peak% − captured%) × total_cost.
  reversal=Y means market fell back after we exited — the "miss" was illusory.
  reversal=N means market continued rising — genuine early exit.
""", file=out)

    pt_exits = [
        m for m in metrics_all
        if m["exit_reason"] == "profit_take" and m["post_exit_peak_pct"] is not None
    ]
    for m in pt_exits:
        missed_pct = m["post_exit_peak_pct"] - m["captured_pct"]
        m["_missed_cents"] = missed_pct * m["total_cost"]

    pt_exits.sort(key=lambda m: -m["_missed_cents"])

    header = (
        f"  {'#':>4}  {'ID':>5}  {'ticker':<30}  {'kind:side':<22}"
        f"  {'entry¢':>7}  {'capt%':>6}  {'peak%':>6}  {'missed¢':>8}  {'rev?':>5}"
    )
    print(header, file=out)
    print("  " + "-" * 110, file=out)
    for rank, m in enumerate(pt_exits[:20], 1):
        rev = "Y" if m["reversal"] else ("N" if m["reversal"] is False else "?")
        print(
            f"  {rank:>4}  {m['trade_id']:>5}  {m['ticker']:<30}  {m['kind_side']:<22}"
            f"  {m['entry_cost']:>7}¢  {_pct(m['captured_pct']):>6}  {_pct(m['post_exit_peak_pct']):>6}"
            f"  {m['_missed_cents']:>8.1f}¢  {rev:>5}",
            file=out,
        )

    # -----------------------------------------------------------------------
    # SECTION 6 — Recommended overrides
    # -----------------------------------------------------------------------
    _hdr("SECTION 6 — Recommended EXIT_SOURCE_PROFIT_TAKE Changes", out)
    print(f"""
  Only segments with N ≥ {MIN_N_RECOMMEND} and |optimal − current| ≥ {MIN_PT_DELTA:.2f} are listed.
  efficiency = avg(captured%/peak%).  reversal% = % of exits where market fell after exit.
""", file=out)

    if not recommendations:
        print("  No segments meet the criteria for recommendation.", file=out)
    else:
        print(
            f"  {'key':<28}  {'current':>8}  {'optimal':>8}  {'N':>4}"
            f"  {'efficiency':>10}  {'reversal%':>10}  note",
            file=out,
        )
        print("  " + "-" * 95, file=out)
        for r in sorted(recommendations, key=lambda x: abs(x["optimal_pt"] - x["current_pt"]), reverse=True):
            direction = "↑ RAISE" if r["optimal_pt"] > r["current_pt"] else "↓ LOWER"
            eff_str = f"{r['efficiency']*100:.0f}%" if r["efficiency"] is not None else "n/a"
            rev_str = f"{r['reversal_rate']*100:.0f}%" if r["reversal_rate"] is not None else "n/a"
            print(
                f"  {r['key']:<28}  {r['current_pt']:>8.2f}  {r['optimal_pt']:>8.2f}  {r['n']:>4}"
                f"  {eff_str:>10}  {rev_str:>10}  {direction}",
                file=out,
            )

        print(f"""
  Paste changes into EXIT_SOURCE_PROFIT_TAKE in exit_manager.py.
  Remember: in-sample optimization — treat as directional signal, not exact values.
""", file=out)

    # -----------------------------------------------------------------------
    # Done
    # -----------------------------------------------------------------------
    print(f"\n{SEP}\n  Analysis complete.\n{SEP}\n", file=out)

    if out_file:
        out_file.close()
        print(f"\n[Output also written to: {args.out}]")

    db.close()


if __name__ == "__main__":
    main()
