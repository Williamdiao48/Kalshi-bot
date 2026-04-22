"""Gate audit backtest: would filtered opportunities have won?

Pulls ALL numeric opportunities from the `opportunities` table (these are
opportunities that passed scoring/liquidity but may have been filtered by
downstream gates), fetches each market's settled result from the Kalshi API,
and reports per-gate win rates and hypothetical P&L.

This answers: "Are we leaving money on the table by being too conservative?"

Gates simulated
---------------
  contrarian    — entry cost > CONTRARIAN_MAX_ENTRY_CENTS (65¢ default)
                  i.e. market already agrees with the signal
  market_floor  — our-side cost < MARKET_MIN_PRICE_CENTS (10¢ default)
                  i.e. market is near-certain we're wrong
  both_pass     — would have been executed (passes both gates)

For each bucket the script reports:
  - N opportunities
  - Win rate (% where implied_outcome matched settled result)
  - Avg entry cost
  - Hypothetical P&L per $1 risked (= (100 - cost) / cost  if win, else -1)
  - Net P&L across all trades in that bucket

Usage
-----
  venv/bin/python scripts/backtest_gates.py
  venv/bin/python scripts/backtest_gates.py --min-score 0.75
  venv/bin/python scripts/backtest_gates.py --source open_meteo
  venv/bin/python scripts/backtest_gates.py --contrarian-cap 80
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import aiohttp

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kalshi_bot.markets import fetch_market_detail

DB_PATH = Path(__file__).parent.parent / "opportunity_log.db"

# Gate thresholds (mirror main.py defaults, overridable via CLI)
CONTRARIAN_MAX_ENTRY_CENTS = 65
MARKET_MIN_PRICE_CENTS = 10

# Sources exempt from the contrarian gate (observed data, not forecasts)
_CONTRARIAN_EXEMPT = frozenset({"noaa_observed", "metar", "nws_climo", "nws_alert"})


def load_opportunities(
    min_score: float,
    source_filter: str | None,
    limit: int,
) -> list[dict]:
    """Load settled numeric opportunities from the DB."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    where_clauses = [
        "kind = 'numeric'",
        "implied_outcome IN ('YES', 'NO')",
        "yes_bid IS NOT NULL",
        "yes_ask IS NOT NULL",
        f"score >= {min_score}",
    ]
    if source_filter:
        where_clauses.append(f"source = '{source_filter}'")

    query = f"""
        SELECT DISTINCT ticker, source, score, yes_bid, yes_ask,
               implied_outcome, direction, strike, strike_lo, strike_hi,
               edge, data_value, logged_at
        FROM opportunities
        WHERE {' AND '.join(where_clauses)}
        ORDER BY logged_at DESC
        LIMIT {limit}
    """
    rows = [dict(r) for r in conn.execute(query).fetchall()]
    conn.close()
    return rows


async def fetch_results(
    opps: list[dict],
    session: aiohttp.ClientSession,
) -> dict[str, str | None]:
    """Fetch settlement result for each unique ticker. Returns ticker → result."""
    tickers = list({o["ticker"] for o in opps})
    print(f"Fetching settlement results for {len(tickers)} unique tickers...")

    results: dict[str, str | None] = {}
    sem = asyncio.Semaphore(10)  # max 10 concurrent API calls

    async def fetch_one(ticker: str) -> None:
        async with sem:
            mkt = await fetch_market_detail(session, ticker)
            if mkt is None:
                results[ticker] = None
                return
            status = mkt.get("status", "")
            result = mkt.get("result", "")
            if status in ("settled", "finalized") and result in ("yes", "no"):
                results[ticker] = result
            else:
                results[ticker] = None  # still open or unknown

    await asyncio.gather(*[fetch_one(t) for t in tickers])
    settled = sum(1 for v in results.values() if v is not None)
    print(f"  {settled}/{len(tickers)} tickers have settled results.\n")
    return results


def classify_gates(opp: dict, contrarian_cap: int, floor: int) -> str:
    """Return which gate bucket this opportunity falls into."""
    implied = opp["implied_outcome"]
    bid = opp["yes_bid"]
    ask = opp["yes_ask"]

    # Entry cost from our side
    if implied == "YES":
        entry_cost = ask
    else:
        entry_cost = 100 - bid

    fails_contrarian = (
        opp["source"] not in _CONTRARIAN_EXEMPT
        and entry_cost > contrarian_cap
    )
    fails_floor = entry_cost < floor

    if fails_floor:
        return "blocked_floor"
    if fails_contrarian:
        return "blocked_contrarian"
    return "would_execute"


def compute_pnl(entry_cost: int, won: bool) -> float:
    """P&L in cents for a single $1-risked equivalent trade."""
    if won:
        return (100 - entry_cost) / entry_cost  # profit ratio
    return -1.0


def print_bucket_report(
    label: str,
    rows: list[tuple[bool, int]],  # (won, entry_cost)
) -> None:
    if not rows:
        print(f"  {label}: no data")
        return

    wins = sum(1 for won, _ in rows if won)
    n = len(rows)
    win_rate = wins / n * 100
    avg_cost = sum(c for _, c in rows) / n

    # P&L assuming $1 risked per trade
    total_pnl = sum(
        (100 - cost) / 100 if won else -(cost / 100)
        for won, cost in rows
    )

    print(f"  {label}:")
    print(f"    N={n}  win_rate={win_rate:.1f}%  avg_entry={avg_cost:.1f}¢  net_pnl=${total_pnl:+.2f} (per $1/trade)")
    print()


async def main(args: argparse.Namespace) -> None:
    opps = load_opportunities(
        min_score=args.min_score,
        source_filter=args.source or None,
        limit=args.limit,
    )
    print(f"Loaded {len(opps)} numeric opportunities (score≥{args.min_score}"
          + (f", source={args.source}" if args.source else "")
          + (f", entry≤{args.max_entry}¢" if args.max_entry > 0 else "")
          + ")\n")

    if not opps:
        print("No opportunities found. Run the bot longer to accumulate data.")
        return

    async with aiohttp.ClientSession() as session:
        results = await fetch_results(opps, session)

    # Only analyse opportunities with a settled result
    settled_opps = [o for o in opps if results.get(o["ticker"]) is not None]
    print(f"{len(settled_opps)}/{len(opps)} opportunities have settled markets.\n")

    if not settled_opps:
        print("No settled markets yet — markets need time to resolve.")
        print("Re-run after today's markets close (~11 PM ET).")
        return

    # Bucket by gate classification
    buckets: dict[str, list[tuple[bool, int]]] = defaultdict(list)
    per_source: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for opp in settled_opps:
        result = results[opp["ticker"]]  # 'yes' or 'no'
        implied = opp["implied_outcome"].lower()
        won = result == implied

        bid = opp["yes_bid"]
        ask = opp["yes_ask"]
        entry_cost = ask if implied == "yes" else (100 - bid)

        if args.max_entry > 0 and entry_cost > args.max_entry:
            continue

        bucket = classify_gates(opp, args.contrarian_cap, args.floor)
        buckets[bucket].append((won, entry_cost))
        per_source[opp["source"]][bucket].append((won, entry_cost))

    # ── Overall gate audit ──────────────────────────────────────────────────
    print("=" * 60)
    print("GATE AUDIT — ALL SOURCES")
    print("=" * 60)
    print(f"  contrarian_cap={args.contrarian_cap}¢  floor={args.floor}¢\n")
    print_bucket_report("would_execute (passes both gates)", buckets["would_execute"])
    print_bucket_report("blocked_contrarian (market agreed, entry too expensive)", buckets["blocked_contrarian"])
    print_bucket_report("blocked_floor (market near-certain against us)", buckets["blocked_floor"])

    # ── Per-source breakdown ────────────────────────────────────────────────
    print("=" * 60)
    print("PER-SOURCE BREAKDOWN")
    print("=" * 60)
    for src in sorted(per_source):
        total = sum(len(v) for v in per_source[src].values())
        if total < 3:
            continue
        print(f"\n  [{src}]  ({total} settled)")
        for bucket in ("would_execute", "blocked_contrarian", "blocked_floor"):
            rows = per_source[src].get(bucket, [])
            if rows:
                wins = sum(1 for w, _ in rows if w)
                avg_cost = sum(c for _, c in rows) / len(rows)
                net = sum((100-c)/100 if w else -(c/100) for w, c in rows)
                print(f"    {bucket}: N={len(rows)}  WR={wins/len(rows)*100:.0f}%  "
                      f"avg={avg_cost:.0f}¢  pnl=${net:+.2f}")

    # ── Verdict ─────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("VERDICT")
    print("=" * 60)
    b_c = buckets["blocked_contrarian"]
    b_f = buckets["blocked_floor"]
    exe = buckets["would_execute"]

    if b_c:
        wr_c = sum(1 for w, _ in b_c if w) / len(b_c) * 100
        if wr_c > 55:
            print(f"  ⚠ Contrarian gate may be too tight: blocked trades won {wr_c:.0f}%")
            print(f"    Consider raising CONTRARIAN_MAX_ENTRY_CENTS above {args.contrarian_cap}¢")
        else:
            print(f"  ✓ Contrarian gate looks correct: blocked trades won only {wr_c:.0f}%")

    if b_f:
        wr_f = sum(1 for w, _ in b_f if w) / len(b_f) * 100
        if wr_f > 55:
            print(f"  ⚠ Market floor gate may be too tight: blocked trades won {wr_f:.0f}%")
            print(f"    Consider lowering MARKET_MIN_PRICE_CENTS below {args.floor}¢")
        else:
            print(f"  ✓ Market floor gate looks correct: blocked trades won only {wr_f:.0f}%")

    if exe:
        wr_e = sum(1 for w, _ in exe if w) / len(exe) * 100
        print(f"  ✓ Would-execute bucket win rate: {wr_e:.0f}% (target >55%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gate audit backtest")
    parser.add_argument("--min-score",      type=float, default=0.70,
                        help="Minimum opportunity score (default 0.70)")
    parser.add_argument("--source",         type=str,   default=None,
                        help="Filter to a single source (e.g. open_meteo)")
    parser.add_argument("--limit",          type=int,   default=5000,
                        help="Max opportunities to load from DB (default 5000)")
    parser.add_argument("--contrarian-cap", type=int,   default=CONTRARIAN_MAX_ENTRY_CENTS,
                        help=f"Contrarian gate cap in cents (default {CONTRARIAN_MAX_ENTRY_CENTS})")
    parser.add_argument("--floor",          type=int,   default=MARKET_MIN_PRICE_CENTS,
                        help=f"Market floor in cents (default {MARKET_MIN_PRICE_CENTS})")
    parser.add_argument("--max-entry",      type=int,   default=0,
                        help="Only include opps where entry cost <= this (e.g. 85 to exclude near-certain observed signals)")
    args = parser.parse_args()
    asyncio.run(main(args))
