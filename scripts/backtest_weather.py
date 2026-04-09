"""Weather source backtester.

Reads opportunity_log.db and replays all surfaced weather opportunities under
configurable filter settings. Reports counterfactual P&L vs actual history.

Usage:
    venv/bin/python scripts/backtest_weather.py
    venv/bin/python scripts/backtest_weather.py --min-edge 10 --block-sources weatherapi
    venv/bin/python scripts/backtest_weather.py --source noaa_day2 --min-edge 20

Only opportunities where the matching trade has a known outcome (won/lost) are
included — these are the ground-truth data points we can evaluate.

For opportunities that were surfaced but NOT traded (filtered by other gates,
cooldown, etc.), outcome is unknown and they are excluded from the simulation.
"""

import argparse
import sqlite3
from collections import defaultdict
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "opportunity_log.db"

WEATHER_SOURCES = {
    "noaa", "noaa_day2", "noaa_observed", "owm",
    "open_meteo", "nws_hourly", "metar", "hrrr", "weatherapi", "nws_alert",
}


def load_data(db_path: Path) -> list[dict]:
    """Load all weather opportunities with known trade outcomes."""
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute("""
        SELECT
            t.id,
            t.logged_at,
            t.ticker,
            t.side,
            t.limit_price,
            t.count,
            t.source,
            t.outcome,
            t.exit_pnl_cents,
            t.exit_reason,
            o.data_value,
            o.direction,
            o.strike,
            o.strike_lo,
            o.strike_hi,
            o.edge,
            o.yes_bid,
            o.yes_ask
        FROM trades t
        LEFT JOIN opportunities o
            ON o.ticker = t.ticker
            AND o.source = t.source
            AND o.logged_at = (
                SELECT MIN(o2.logged_at) FROM opportunities o2
                WHERE o2.ticker = t.ticker AND o2.source = t.source
            )
        WHERE t.source IN ({})
        AND t.outcome IS NOT NULL
        ORDER BY t.id
    """.format(",".join(f'"{s}"' for s in WEATHER_SOURCES))).fetchall()
    conn.close()

    cols = [
        "id", "logged_at", "ticker", "side", "limit_price", "count", "source",
        "outcome", "exit_pnl_cents", "exit_reason",
        "data_value", "direction", "strike", "strike_lo", "strike_hi", "edge",
        "yes_bid", "yes_ask",
    ]
    return [dict(zip(cols, r)) for r in rows]


def simulate(
    trades: list[dict],
    min_edge: float = 0.0,
    block_sources: set[str] | None = None,
    filter_source: str | None = None,
    max_observed_edge: float | None = None,
) -> dict:
    """Simulate P&L under given filter parameters.

    Args:
        trades:            All resolved weather trades.
        min_edge:          Minimum edge (°F) required to trade.
        block_sources:     Sources to exclude entirely.
        filter_source:     If set, only analyse this source.
        max_observed_edge: If set, cap noaa_observed trades at this edge (°F).
                           High-edge observed readings likely indicate sensor error,
                           not genuine market mispricing.

    Returns:
        dict with per-source stats and totals.
    """
    block_sources = block_sources or set()
    stats: dict[str, dict] = defaultdict(lambda: {
        "trades": 0, "wins": 0, "losses": 0, "pnl": 0.0,
        "skipped_edge": 0, "skipped_block": 0,
    })

    for t in trades:
        src = t["source"]
        if filter_source and src != filter_source:
            continue

        s = stats[src]

        if src in block_sources:
            s["skipped_block"] += 1
            continue

        edge = t["edge"] or 0.0
        if edge < min_edge:
            s["skipped_edge"] += 1
            continue

        # noaa_observed max-edge cap: very high edges on an observed source signal
        # sensor error (e.g. station reporting 36°F when actual was 50°F+), not a
        # genuine market mispricing. Cap prevents trading on bad readings.
        if src == "noaa_observed" and max_observed_edge is not None:
            if edge > max_observed_edge:
                s["skipped_edge"] += 1
                continue

        # Would have traded — use the actual outcome
        pnl = (t["exit_pnl_cents"] or 0) / 100

        s["trades"] += 1
        if t["outcome"] == "won":
            s["wins"] += 1
        else:
            s["losses"] += 1
        s["pnl"] += pnl

    return stats


def print_report(
    stats: dict,
    label: str,
    actual_stats: dict | None = None,
) -> None:
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    print(f"  {'Source':<16} {'Trades':>6} {'W':>4} {'L':>4} {'Win%':>6} {'Net P&L':>9}  {'Skipped':>8}")
    print(f"  {'-'*62}")

    total_trades = total_wins = total_losses = 0
    total_pnl = 0.0

    all_sources = sorted(stats.keys())
    for src in all_sources:
        s = stats[src]
        if s["trades"] == 0 and s["skipped_edge"] == 0 and s["skipped_block"] == 0:
            continue
        total = s["trades"]
        w, l = s["wins"], s["losses"]
        winpct = f"{100*w/total:.0f}%" if total > 0 else "-"
        pnl = s["pnl"]
        skipped = s["skipped_edge"] + s["skipped_block"]
        skip_str = f"({skipped} skip)" if skipped else ""

        total_trades += total
        total_wins += w
        total_losses += l
        total_pnl += pnl

        print(f"  {src:<16} {total:>6} {w:>4} {l:>4} {winpct:>6} {pnl:>+9.2f}  {skip_str:>8}")

    print(f"  {'-'*62}")
    total_winpct = f"{100*total_wins/total_trades:.0f}%" if total_trades > 0 else "-"
    print(f"  {'TOTAL':<16} {total_trades:>6} {total_wins:>4} {total_losses:>4} {total_winpct:>6} {total_pnl:>+9.2f}")

    if actual_stats:
        act_pnl = sum(s["pnl"] for s in actual_stats.values())
        delta = total_pnl - act_pnl
        print(f"\n  vs actual: {act_pnl:+.2f}  |  delta: {delta:+.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest weather source filters")
    parser.add_argument("--min-edge", type=float, default=0.0,
                        help="Minimum edge (°F) to trade (default: 0 = show all)")
    parser.add_argument("--block-sources", nargs="*", default=[],
                        help="Sources to exclude entirely")
    parser.add_argument("--source", type=str, default=None,
                        help="Only analyse this source")
    parser.add_argument("--max-observed-edge", type=float, default=None,
                        help="Cap noaa_observed trades at this edge (°F); "
                             "high edges on observed data indicate sensor error")
    args = parser.parse_args()

    trades = load_data(DB_PATH)
    print(f"\nLoaded {len(trades)} resolved weather trades from {DB_PATH.name}")

    # Baseline: actual history (no filters)
    actual = simulate(trades, min_edge=0.0, filter_source=args.source)
    print_report(actual, "ACTUAL HISTORY (no filters)")

    # If custom params provided, show counterfactual
    if args.min_edge > 0 or args.block_sources or args.max_observed_edge is not None:
        sim = simulate(
            trades,
            min_edge=args.min_edge,
            block_sources=set(args.block_sources),
            filter_source=args.source,
            max_observed_edge=args.max_observed_edge,
        )
        label = f"COUNTERFACTUAL (edge≥{args.min_edge}°F"
        if args.block_sources:
            label += f", block={args.block_sources}"
        if args.max_observed_edge is not None:
            label += f", observed_max_edge={args.max_observed_edge}°F"
        label += ")"
        print_report(sim, label, actual_stats=actual)

    # Always show a few useful presets
    presets = [
        ("Block weatherapi", dict(block_sources={"weatherapi"})),
        ("Block weatherapi + edge≥10°F noaa_day2",
         dict(min_edge=10.0, block_sources={"weatherapi"})),
        ("noaa_observed edge window 5–10°F",
         dict(max_observed_edge=10.0)),
        ("All filters combined",
         dict(block_sources={"weatherapi"}, max_observed_edge=10.0)),
    ]

    for label, kwargs in presets:
        if args.source:
            kwargs["filter_source"] = args.source
        sim = simulate(trades, **kwargs)
        print_report(sim, f"PRESET: {label}", actual_stats=actual)

    # noaa_observed-specific breakdown: wins vs losses by edge bucket
    print(f"\n{'='*65}")
    print("  noaa_observed: edge buckets (post-fix sanity check)")
    print(f"{'='*65}")
    obs_buckets: dict[str, dict] = {
        "0–5°F":   {"w": 0, "l": 0},
        "5–10°F":  {"w": 0, "l": 0},
        "10–20°F": {"w": 0, "l": 0},
        "20°F+":   {"w": 0, "l": 0},
    }
    for t in trades:
        if t["source"] != "noaa_observed":
            continue
        if args.source and args.source != "noaa_observed":
            continue
        edge = t["edge"] or 0.0
        if edge < 5:
            bucket = "0–5°F"
        elif edge < 10:
            bucket = "5–10°F"
        elif edge < 20:
            bucket = "10–20°F"
        else:
            bucket = "20°F+"
        if t["outcome"] == "won":
            obs_buckets[bucket]["w"] += 1
        else:
            obs_buckets[bucket]["l"] += 1
    print(f"  {'Bucket':<10} {'W':>4} {'L':>4} {'Win%':>6}")
    print(f"  {'-'*28}")
    for bucket, d in obs_buckets.items():
        total = d["w"] + d["l"]
        winpct = f"{100*d['w']/total:.0f}%" if total > 0 else "-"
        print(f"  {bucket:<10} {d['w']:>4} {d['l']:>4} {winpct:>6}")

    # Per-source edge distribution for losers vs winners
    print(f"\n{'='*65}")
    print("  EDGE DISTRIBUTION: winners vs losers per source")
    print(f"{'='*65}")
    by_source: dict[str, dict] = defaultdict(lambda: {"win_edges": [], "loss_edges": []})
    for t in trades:
        src = t["source"]
        if args.source and src != args.source:
            continue
        edge = t["edge"] or 0.0
        if t["outcome"] == "won":
            by_source[src]["win_edges"].append(edge)
        else:
            by_source[src]["loss_edges"].append(edge)

    for src in sorted(by_source.keys()):
        d = by_source[src]
        we = sorted(d["win_edges"])
        le = sorted(d["loss_edges"])
        print(f"\n  {src}:")
        print(f"    Wins  ({len(we):2d}): {[round(e,1) for e in we]}")
        print(f"    Losses({len(le):2d}): {[round(e,1) for e in le]}")
        if we and le:
            min_win = min(we)
            max_loss = max(le)
            overlap = min_win <= max_loss
            print(f"    Min win edge: {min_win:.1f}°F  |  Max loss edge: {max_loss:.1f}°F  |  Overlap: {overlap}")


if __name__ == "__main__":
    main()
