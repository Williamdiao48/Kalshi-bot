"""Numeric trade pattern analysis — which entry-time features predict win rate and P&L?

Segments all numeric trades by entry-time features already stored in the
``trades`` table (source, side, score, market_p_entry, bid-ask spread,
corroborating_sources, entry cost) and reports win rate and avg P&L per
bucket to identify which gates need tightening.

This script is the repeatable companion to one-off bash analysis; run it
after accumulating new trades to re-validate any live gate changes.

Sections
--------
  1  Data coverage — how many numeric trades, how many resolved, breakout by source
  2  By source:side        — primary signal origin + direction
  3  By score bucket       — composite score at entry [0.75-0.80, 0.80-0.85, ...]
  4  By market_p_entry     — market consensus at entry [<0.20, 0.20-0.35, ...]
  5  By bid-ask spread     — spread at entry [0-3¢, 4-6¢, 7-10¢, >10¢]
  6  By entry cost         — what we paid per contract [≤15¢, 16-30¢, 31-50¢, 51-70¢, >70¢]
  7  By corr_source count  — number of corroborating sources at entry [0, 1, 2, 3+]
  8  By source reliability score — (noaa_observed vs hrrr vs noaa vs owm etc.)
  9  Worst trades          — bottom-20 by pnl for manual review
  10 Gate recommendations  — buckets with N≥5 and 0% or near-0% win rate

Usage
-----
  venv/bin/python scripts/analyze_numeric_patterns.py
  venv/bin/python scripts/analyze_numeric_patterns.py --from-id 58
  venv/bin/python scripts/analyze_numeric_patterns.py --resolved-only
  venv/bin/python scripts/analyze_numeric_patterns.py --out numeric_analysis.txt
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).parent.parent))

DB_PATH = Path(__file__).parent.parent / "opportunity_log.db"

# Minimum N per bucket before reporting (avoids printing noise for N=1).
MIN_REPORT_N = 2

# Minimum N before emitting a gate recommendation.
MIN_GATE_N = 5

# Win-rate ceiling below which a bucket is flagged in gate recommendations.
GATE_WIN_RATE_CEIL = 0.20

# Avg P&L floor (¢) below which a bucket is flagged regardless of win rate.
GATE_PNL_FLOOR = -50.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_trades(
    db: sqlite3.Connection,
    from_id: int | None,
    resolved_only: bool,
) -> list[dict]:
    """Load all numeric trades with resolved P&L."""
    wheres = ["t.opportunity_kind = 'numeric'"]
    params: list = []

    if from_id is not None:
        wheres.append("t.id >= ?")
        params.append(from_id)

    if resolved_only:
        wheres.append("t.outcome IS NOT NULL")
    else:
        # Require either a resolved outcome OR a recorded exit P&L — trades
        # with neither cannot be evaluated.
        wheres.append("(t.outcome IS NOT NULL OR t.exit_pnl_cents IS NOT NULL)")

    where_sql = " AND ".join(wheres)

    rows = db.execute(f"""
        SELECT
            t.id,
            t.ticker,
            t.source,
            t.side,
            t.score,
            t.limit_price,
            t.market_p_entry,
            t.yes_bid_entry,
            t.yes_ask_entry,
            t.corroborating_sources,
            t.outcome,
            t.exit_pnl_cents,
            t.exit_reason,
            t.note
        FROM trades t
        WHERE {where_sql}
        ORDER BY t.id
    """, params).fetchall()

    trades = []
    for row in rows:
        (tid, ticker, source, side, score, limit_price,
         market_p_entry, yes_bid_entry, yes_ask_entry,
         corr_sources, outcome, exit_pnl_cents, exit_reason, note) = row

        # Derive entry cost (what we paid per contract)
        if side == "yes":
            entry_cost = limit_price
        else:
            entry_cost = 100 - limit_price

        # Bid-ask spread at entry
        spread = None
        if yes_bid_entry is not None and yes_ask_entry is not None:
            spread = yes_ask_entry - yes_bid_entry

        # Corroborating source count
        if corr_sources and corr_sources.strip():
            corr_count = len([s for s in corr_sources.split(",") if s.strip()])
        else:
            corr_count = 0

        # P&L in cents (exit P&L is already in cents; for settlement use outcome)
        pnl_cents: float | None = None
        if exit_pnl_cents is not None:
            pnl_cents = float(exit_pnl_cents)
        elif outcome == "won":
            pnl_cents = float(100 - limit_price) if side == "yes" else float(limit_price)
        elif outcome == "lost":
            pnl_cents = -float(entry_cost)

        # Win determination
        won: bool | None = None
        if pnl_cents is not None:
            won = pnl_cents > 0
        elif outcome == "won":
            won = True
        elif outcome == "lost":
            won = False

        # Extra fields from note JSON (if logged)
        note_data: dict = {}
        if note:
            try:
                note_data = json.loads(note)
            except (ValueError, TypeError):
                pass

        trades.append({
            "id": tid,
            "ticker": ticker,
            "source": source or "",
            "side": side,
            "score": score,
            "limit_price": limit_price,
            "market_p_entry": market_p_entry,
            "yes_bid_entry": yes_bid_entry,
            "yes_ask_entry": yes_ask_entry,
            "spread": spread,
            "corr_count": corr_count,
            "corr_sources": corr_sources or "",
            "entry_cost": entry_cost,
            "outcome": outcome,
            "exit_pnl_cents": exit_pnl_cents,
            "exit_reason": exit_reason,
            "pnl_cents": pnl_cents,
            "won": won,
            "note": note_data,
        })

    return trades


# ---------------------------------------------------------------------------
# Bucketing helpers
# ---------------------------------------------------------------------------

def score_bucket(score: float | None) -> str:
    if score is None:
        return "unknown"
    if score < 0.75:
        return "<0.75"
    if score < 0.80:
        return "0.75-0.80"
    if score < 0.85:
        return "0.80-0.85"
    if score < 0.90:
        return "0.85-0.90"
    if score < 0.95:
        return "0.90-0.95"
    return "≥0.95"


_SCORE_ORDER = ["<0.75", "0.75-0.80", "0.80-0.85", "0.85-0.90", "0.90-0.95", "≥0.95"]


def market_p_bucket(mp: float | None) -> str:
    if mp is None:
        return "unknown"
    if mp < 0.20:
        return "<0.20"
    if mp < 0.30:
        return "0.20-0.30"
    if mp < 0.40:
        return "0.30-0.40"
    if mp < 0.50:
        return "0.40-0.50"
    if mp < 0.60:
        return "0.50-0.60"
    if mp < 0.70:
        return "0.60-0.70"
    return "≥0.70"


_MP_ORDER = ["<0.20", "0.20-0.30", "0.30-0.40", "0.40-0.50", "0.50-0.60", "0.60-0.70", "≥0.70"]


def spread_bucket(spread: int | None) -> str:
    if spread is None:
        return "unknown"
    if spread <= 2:
        return "0-2¢"
    if spread <= 4:
        return "3-4¢"
    if spread <= 6:
        return "5-6¢"
    if spread <= 10:
        return "7-10¢"
    return ">10¢"


_SPREAD_ORDER = ["0-2¢", "3-4¢", "5-6¢", "7-10¢", ">10¢", "unknown"]


def entry_cost_bucket(cost: int) -> str:
    if cost <= 10:
        return "≤10¢"
    if cost <= 20:
        return "11-20¢"
    if cost <= 30:
        return "21-30¢"
    if cost <= 50:
        return "31-50¢"
    if cost <= 70:
        return "51-70¢"
    return ">70¢"


_COST_ORDER = ["≤10¢", "11-20¢", "21-30¢", "31-50¢", "51-70¢", ">70¢"]


def corr_count_bucket(n: int) -> str:
    if n == 0:
        return "0"
    if n == 1:
        return "1"
    if n == 2:
        return "2"
    return "3+"


_CORR_ORDER = ["0", "1", "2", "3+"]


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _bucket_stats(trades: list[dict], key_fn) -> dict[str, dict]:
    """Group trades by key_fn and compute stats per bucket."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        groups[key_fn(t)].append(t)

    stats = {}
    for bucket, ts in groups.items():
        evaluable = [t for t in ts if t["won"] is not None]
        wins = [t for t in evaluable if t["won"]]
        pnls = [t["pnl_cents"] for t in evaluable if t["pnl_cents"] is not None]
        stats[bucket] = {
            "n": len(ts),
            "n_eval": len(evaluable),
            "n_win": len(wins),
            "win_rate": len(wins) / len(evaluable) if evaluable else None,
            "avg_pnl": mean(pnls) if pnls else None,
            "total_pnl": sum(pnls) if pnls else None,
        }
    return stats


def _print_bucket_table(
    stats: dict[str, dict],
    order: list[str] | None,
    out,
    flag_bad: bool = True,
) -> list[str]:
    """Print a stats table and return list of flagged bucket keys."""
    if order:
        keys = [k for k in order if k in stats] + [k for k in stats if k not in order]
    else:
        keys = sorted(stats, key=lambda k: -(stats[k]["n"]))

    header = f"  {'Bucket':<18}  {'N':>4}  {'Eval':>4}  {'Win%':>6}  {'AvgP&L':>8}  {'TotalP&L':>10}"
    print(header, file=out)
    print("  " + "-" * (len(header) - 2), file=out)

    flagged = []
    for k in keys:
        s = stats[k]
        if s["n"] < MIN_REPORT_N:
            continue
        win_str = f"{s['win_rate']*100:.0f}%" if s["win_rate"] is not None else "  n/a"
        pnl_str = f"{s['avg_pnl']:+.1f}¢" if s["avg_pnl"] is not None else "   n/a"
        tot_str = f"{s['total_pnl']:+.0f}¢" if s["total_pnl"] is not None else "    n/a"

        flag = ""
        if flag_bad and s["n_eval"] >= MIN_GATE_N:
            if (s["win_rate"] is not None and s["win_rate"] <= GATE_WIN_RATE_CEIL) or \
               (s["avg_pnl"] is not None and s["avg_pnl"] <= GATE_PNL_FLOOR):
                flag = " ◄ FLAG"
                flagged.append(k)

        print(f"  {k:<18}  {s['n']:>4}  {s['n_eval']:>4}  {win_str:>6}  {pnl_str:>8}  {tot_str:>10}{flag}", file=out)

    return flagged


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------

def run(args, out) -> None:
    db = sqlite3.connect(args.db)
    trades = load_trades(db, args.from_id, args.resolved_only)

    if not trades:
        print("No numeric trades found matching the filters.", file=out)
        return

    evaluable = [t for t in trades if t["won"] is not None]
    wins = [t for t in evaluable if t["won"]]
    pnls = [t["pnl_cents"] for t in evaluable if t["pnl_cents"] is not None]

    # -----------------------------------------------------------------
    # Section 1 — Coverage
    # -----------------------------------------------------------------
    print("=" * 70, file=out)
    print("  NUMERIC TRADE PATTERN ANALYSIS", file=out)
    if args.from_id:
        print(f"  Trades with id >= {args.from_id}", file=out)
    if args.resolved_only:
        print("  Resolved trades only", file=out)
    print("=" * 70, file=out)
    print(file=out)
    print("--- Section 1: Data Coverage ---", file=out)
    print(f"  Total numeric trades loaded : {len(trades)}", file=out)
    print(f"  With evaluable P&L          : {len(evaluable)}", file=out)
    print(f"  Wins / Losses               : {len(wins)} / {len(evaluable) - len(wins)}", file=out)
    overall_wr = f"{len(wins)/len(evaluable)*100:.1f}%" if evaluable else "n/a"
    overall_pnl = f"{mean(pnls):+.1f}¢" if pnls else "n/a"
    print(f"  Overall win rate            : {overall_wr}", file=out)
    print(f"  Overall avg P&L             : {overall_pnl}", file=out)
    print(f"  Total P&L                   : {sum(pnls):+.0f}¢" if pnls else "  Total P&L : n/a", file=out)
    print(file=out)

    # Source breakdown for coverage context
    by_source: dict[str, int] = defaultdict(int)
    for t in trades:
        by_source[f"{t['source']}:{t['side']}"] += 1
    print("  Trades by source:side:", file=out)
    for k, n in sorted(by_source.items(), key=lambda x: -x[1]):
        print(f"    {k:<28}  {n}", file=out)
    print(file=out)

    all_flags: list[tuple[str, str]] = []  # (dimension, bucket) pairs

    # -----------------------------------------------------------------
    # Section 2 — By source:side
    # -----------------------------------------------------------------
    print("--- Section 2: By source:side ---", file=out)
    stats = _bucket_stats(evaluable, lambda t: f"{t['source']}:{t['side']}")
    flagged = _print_bucket_table(stats, None, out)
    all_flags.extend(("source:side", b) for b in flagged)
    print(file=out)

    # -----------------------------------------------------------------
    # Section 3 — By score bucket
    # -----------------------------------------------------------------
    print("--- Section 3: By score bucket ---", file=out)
    stats = _bucket_stats(evaluable, lambda t: score_bucket(t["score"]))
    flagged = _print_bucket_table(stats, _SCORE_ORDER, out)
    all_flags.extend(("score", b) for b in flagged)
    print(file=out)

    # -----------------------------------------------------------------
    # Section 4 — By market_p_entry
    # -----------------------------------------------------------------
    print("--- Section 4: By market_p_entry (consensus at entry) ---", file=out)
    stats = _bucket_stats(evaluable, lambda t: market_p_bucket(t["market_p_entry"]))
    flagged = _print_bucket_table(stats, _MP_ORDER, out)
    all_flags.extend(("market_p", b) for b in flagged)
    print(file=out)

    # -----------------------------------------------------------------
    # Section 5 — By bid-ask spread
    # -----------------------------------------------------------------
    print("--- Section 5: By bid-ask spread at entry ---", file=out)
    stats = _bucket_stats(evaluable, lambda t: spread_bucket(t["spread"]))
    flagged = _print_bucket_table(stats, _SPREAD_ORDER, out)
    all_flags.extend(("spread", b) for b in flagged)
    print(file=out)

    # -----------------------------------------------------------------
    # Section 6 — By entry cost
    # -----------------------------------------------------------------
    print("--- Section 6: By entry cost ---", file=out)
    stats = _bucket_stats(evaluable, lambda t: entry_cost_bucket(t["entry_cost"]))
    flagged = _print_bucket_table(stats, _COST_ORDER, out)
    all_flags.extend(("entry_cost", b) for b in flagged)
    print(file=out)

    # -----------------------------------------------------------------
    # Section 7 — By corroborating source count
    # -----------------------------------------------------------------
    print("--- Section 7: By corroborating source count ---", file=out)
    stats = _bucket_stats(evaluable, lambda t: corr_count_bucket(t["corr_count"]))
    flagged = _print_bucket_table(stats, _CORR_ORDER, out)
    all_flags.extend(("corr_count", b) for b in flagged)
    print(file=out)

    # -----------------------------------------------------------------
    # Section 8 — By source (bare, without side) for reliability comparison
    # -----------------------------------------------------------------
    print("--- Section 8: By bare source (all sides combined) ---", file=out)
    stats = _bucket_stats(evaluable, lambda t: t["source"])
    flagged = _print_bucket_table(stats, None, out)
    all_flags.extend(("source", b) for b in flagged)
    print(file=out)

    # -----------------------------------------------------------------
    # Section 9 — Worst 20 trades
    # -----------------------------------------------------------------
    print("--- Section 9: Bottom-20 trades by P&L ---", file=out)
    sorted_by_pnl = sorted(
        [t for t in evaluable if t["pnl_cents"] is not None],
        key=lambda t: t["pnl_cents"],
    )
    hdr = f"  {'id':>4}  {'ticker':<28}  {'src:side':<28}  {'score':>5}  {'mp':>5}  {'sprd':>5}  {'cost':>5}  {'reason':<14}  {'P&L':>8}"
    print(hdr, file=out)
    print("  " + "-" * (len(hdr) - 2), file=out)
    for t in sorted_by_pnl[:20]:
        src_side = f"{t['source']}:{t['side']}"
        mp_str = f"{t['market_p_entry']:.2f}" if t["market_p_entry"] is not None else "  n/a"
        sp_str = f"{t['spread']}¢" if t["spread"] is not None else " n/a"
        reason = t["exit_reason"] or (t["outcome"] or "?")
        print(
            f"  {t['id']:>4}  {t['ticker']:<28}  {src_side:<28}  "
            f"{t['score']:>5.2f}  {mp_str:>5}  {sp_str:>5}  {t['entry_cost']:>4}¢  "
            f"{reason:<14}  {t['pnl_cents']:>+7.0f}¢",
            file=out,
        )
    print(file=out)

    # -----------------------------------------------------------------
    # Section 10 — Gate recommendations
    # -----------------------------------------------------------------
    print("--- Section 10: Gate Recommendations ---", file=out)
    if not all_flags:
        print("  No buckets meet the flag threshold (N≥5, win%≤20% or avg≤-50¢).", file=out)
    else:
        print(
            f"  The following buckets (N≥{MIN_GATE_N}) have ≤{GATE_WIN_RATE_CEIL*100:.0f}% win rate"
            f" or ≤{GATE_PNL_FLOOR:+.0f}¢ avg P&L:", file=out,
        )
        seen: set[tuple[str, str]] = set()
        for dim, bucket in all_flags:
            if (dim, bucket) in seen:
                continue
            seen.add((dim, bucket))
            print(f"    [{dim}] {bucket}", file=out)
        print(file=out)
        print("  Suggested actions:", file=out)

        # Source:side flags
        ss_flags = {b for d, b in all_flags if d == "source:side"}
        if ss_flags:
            for b in sorted(ss_flags):
                src, side = b.rsplit(":", 1)
                print(
                    f"    • {b}: consider a dedicated BLOCKED_METRICS or per-source"
                    f" min-score override (NOAA_OBSERVED_MIN_SCORE, NOAA_MIN_SCORE, etc.)",
                    file=out,
                )

        # Score flags
        sc_flags = {b for d, b in all_flags if d == "score"}
        if sc_flags:
            worst_sc = sorted(sc_flags)[0]
            print(
                f"    • Score {worst_sc} is flagged — consider raising TRADE_MIN_SCORE"
                f" or adding a NUMERIC_MIN_SCORE env gate.",
                file=out,
            )

        # Market_p flags
        mp_flags = {b for d, b in all_flags if d == "market_p"}
        if mp_flags:
            print(
                f"    • market_p buckets {sorted(mp_flags)} are flagged — consider adding"
                f" NUMERIC_MAX_YES_MARKET_P (env gate, currently absent) to block YES"
                f" trades when the market already strongly agrees.",
                file=out,
            )

        # Spread flags
        sp_flags = {b for d, b in all_flags if d == "spread"}
        if sp_flags:
            print(
                f"    • Spread buckets {sorted(sp_flags)} are flagged — consider adding"
                f" NUMERIC_MAX_SPREAD_CENTS (env gate, currently absent) for a hard"
                f" spread ceiling on numeric trades.",
                file=out,
            )

    print(file=out)
    print("=" * 70, file=out)
    print("  Caveats", file=out)
    print("=" * 70, file=out)
    print(
        "  1. P&L uses exit_pnl_cents where available (all exited trades), falling\n"
        "     back to outcome (won/lost at settlement) for hold-to-settlement trades.\n"
        "     Trades with neither are excluded.\n"
        "  2. Sample sizes per bucket may be small (N<5). Treat flagged buckets as\n"
        "     directional signals, not statistically significant findings.\n"
        "  3. All analysis is in-sample. Gate thresholds derived here should be\n"
        "     validated by running the bot for a period before hardening defaults.\n"
        "  4. note JSON fields (model_spread, hours_to_close, etc.) are only\n"
        "     populated for trades logged after the note-logging upgrade. Older\n"
        "     trades will show blanks for those dimensions.",
        file=out,
    )
    print(file=out)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Numeric trade pattern analysis — win rate by entry-time feature buckets."
    )
    parser.add_argument(
        "--db",
        default=str(DB_PATH),
        help="Path to opportunity_log.db (default: project root)",
    )
    parser.add_argument(
        "--from-id",
        type=int,
        default=None,
        metavar="ID",
        help="Only include trades with id >= ID (useful for recency filtering).",
    )
    parser.add_argument(
        "--resolved-only",
        action="store_true",
        help="Only include trades with a settled outcome (won/lost).",
    )
    parser.add_argument(
        "--out",
        default=None,
        metavar="FILE",
        help="Write output to FILE in addition to stdout.",
    )
    args = parser.parse_args()

    if args.out:
        with open(args.out, "w") as f:
            run(args, f)
        # Also print to stdout
        with open(args.out) as f:
            sys.stdout.write(f.read())
    else:
        run(args, sys.stdout)


if __name__ == "__main__":
    main()
