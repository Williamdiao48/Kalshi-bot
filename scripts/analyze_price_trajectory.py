#!/usr/bin/env python3
"""Intraday price trajectory analysis for forecast_no stop-loss trades.

For each resolved forecast_no trade that has candlestick data, traces the
YES_ask price from entry to exit and classifies whether stop-losses were
triggered by:
  - TREND stop: adverse move was sustained; market kept moving against us
  - NOISE stop: adverse move was brief and reversed; market closed profitably

Also reports the distribution of max adverse moves, useful for calibrating
the stop-loss threshold.

Requires candlestick data for the traded tickers.  Run
  venv/bin/python scripts/fetch_candlestick_history.py --days 14 --resume
first to populate data/candlesticks.db with settled May markets.

Usage:
    venv/bin/python scripts/analyze_price_trajectory.py
    venv/bin/python scripts/analyze_price_trajectory.py --kind forecast_no
    venv/bin/python scripts/analyze_price_trajectory.py --kind band_arb
    venv/bin/python scripts/analyze_price_trajectory.py --ids 37 40 83 100
    venv/bin/python scripts/analyze_price_trajectory.py --exit-reason stop_loss
"""

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.lib import DEFAULT_DB_PATH, DEFAULT_CANDLESTICK_DB_PATH, parse_iso_ts

# A NO position profits when YES price falls.
# Adverse move = YES ask rising above our entry (pushing our mark-to-market negative).
# We hold NO, so entry cost = 100 - yes_bid_entry.
# Stop-loss fires when YES bid rises enough that our position is down > stop%.

# Classification thresholds
_REVERSAL_CANDLES  = 5    # if the adverse peak reverses within this many candles → noise
_TREND_PERSIST_PCT = 0.50 # adverse move must persist for this fraction of the hold window


def _classify_stop(candles: list[tuple[int, int | None, int | None]],
                   entry_yes_ask: int,
                   exit_ts: int) -> tuple[str, int, int]:
    """Classify a stop-loss exit given the 1-minute candle sequence.

    candles: list of (period_ts, ask_open, bid_close) sorted ascending
    entry_yes_ask: YES ask at entry in cents
    exit_ts: Unix timestamp of exit

    Returns (classification, max_adverse_cents, reversal_candle_idx)
      classification: 'trend' | 'noise' | 'unknown'
      max_adverse_cents: peak YES ask above entry (0 if never exceeded)
      reversal_idx: index of candle where price first fell back below entry after peak
    """
    if not candles:
        return "unknown", 0, -1

    peak_ask = entry_yes_ask
    peak_idx = 0
    for i, (ts, ask, _bid) in enumerate(candles):
        if ask is not None and ask > peak_ask:
            peak_ask = ask
            peak_idx = i

    max_adverse = max(0, peak_ask - entry_yes_ask)
    if max_adverse == 0:
        return "noise", 0, -1  # price never went against us

    # Find when (if ever) price came back below entry after the peak
    reversal_idx = -1
    for i in range(peak_idx + 1, len(candles)):
        _, ask, _ = candles[i]
        if ask is not None and ask <= entry_yes_ask:
            reversal_idx = i
            break

    if reversal_idx == -1:
        # Price never recovered below entry after peak → trend
        return "trend", max_adverse, -1

    candles_to_reversal = reversal_idx - peak_idx
    hold_candles = len(candles)

    if candles_to_reversal <= _REVERSAL_CANDLES:
        return "noise", max_adverse, reversal_idx

    # Peak persisted for a meaningful fraction of the hold window
    if candles_to_reversal / max(hold_candles, 1) >= _TREND_PERSIST_PCT:
        return "trend", max_adverse, reversal_idx

    return "noise", max_adverse, reversal_idx


def analyze(
    opp_conn: sqlite3.Connection,
    candle_conn: sqlite3.Connection,
    kind: str | None = None,
    trade_ids: list[int] | None = None,
    exit_reason_filter: str | None = None,
) -> None:
    query = """
        SELECT id, ticker, opportunity_kind, side, yes_bid_entry, yes_ask_entry,
               limit_price, logged_at, exited_at, exit_reason, exit_pnl_cents, outcome
        FROM trades
        WHERE outcome IS NOT NULL
    """
    params: list = []
    if kind:
        query += " AND opportunity_kind = ?"
        params.append(kind)
    if trade_ids:
        query += f" AND id IN ({','.join('?' * len(trade_ids))})"
        params.extend(trade_ids)
    if exit_reason_filter:
        query += " AND exit_reason = ?"
        params.append(exit_reason_filter)
    query += " ORDER BY id"

    trades = opp_conn.execute(query, params).fetchall()
    print(f"Analyzing {len(trades)} resolved trade(s)…\n")

    results = []
    no_candles = []

    for (trade_id, ticker, opp_kind, side, yes_bid_entry, yes_ask_entry,
         limit_price, logged_at, exited_at, exit_reason, exit_pnl_cents, outcome) in trades:

        entry_ts = parse_iso_ts(logged_at) if logged_at else 0
        exit_ts  = parse_iso_ts(exited_at) if exited_at else 0

        # Trades that resolved at market settlement have exited_at=NULL.
        # Fall back to close_ts from candlesticks.db markets table.
        if entry_ts and not exit_ts:
            mkt_row = candle_conn.execute(
                "SELECT close_ts FROM markets WHERE ticker = ?", (ticker,)
            ).fetchone()
            if mkt_row and mkt_row[0]:
                exit_ts = int(mkt_row[0])

        if not entry_ts or not exit_ts:
            no_candles.append((trade_id, ticker, "no timestamps"))
            continue

        candles = candle_conn.execute(
            "SELECT period_ts, ask_open, bid_close FROM candles "
            "WHERE ticker = ? AND period_ts BETWEEN ? AND ? ORDER BY period_ts",
            (ticker, entry_ts, exit_ts),
        ).fetchall()

        if not candles:
            no_candles.append((trade_id, ticker, "no candles in DB"))
            continue

        # For NO positions: entry_cost = 100 - yes_bid_entry (what we paid per contract)
        # Adverse move for NO = YES ask rising (reduces our mark-to-market value)
        entry_cost = 100 - (yes_bid_entry or limit_price or 50)
        entry_ask  = yes_ask_entry or limit_price or 50

        classification, max_adverse, reversal_idx = _classify_stop(
            candles, entry_ask, exit_ts
        )

        hold_minutes = (exit_ts - entry_ts) // 60
        pnl = (exit_pnl_cents or 0) / 100.0

        results.append({
            "id":            trade_id,
            "ticker":        ticker,
            "kind":          opp_kind,
            "exit_reason":   exit_reason,
            "outcome":       outcome,
            "pnl":           pnl,
            "entry_cost":    entry_cost,
            "max_adverse":   max_adverse,
            "classification": classification,
            "hold_minutes":  hold_minutes,
            "n_candles":     len(candles),
        })

    # ── Per-trade detail ────────────────────────────────────────────────────
    print("═" * 72)
    print("  PER-TRADE DETAIL")
    print("═" * 72)
    for r in results:
        flag = ""
        if r["exit_reason"] == "stop_loss":
            flag = f"  [{r['classification'].upper()} STOP]"
        print(
            f"  #{r['id']:<4d}  {r['ticker']:<38}"
            f"  {(r['exit_reason'] or 'unknown'):<20}  pnl={r['pnl']:+.2f}$"
            f"  max_adv={r['max_adverse']:+3d}¢{flag}"
        )

    if no_candles:
        print(f"\n  No candle data for {len(no_candles)} trade(s):")
        for tid, tkr, reason in no_candles:
            print(f"    #{tid}  {tkr}  ({reason})")

    # ── Stop-loss classification summary ────────────────────────────────────
    stop_trades = [r for r in results if r["exit_reason"] == "stop_loss"]
    if stop_trades:
        trend  = [r for r in stop_trades if r["classification"] == "trend"]
        noise  = [r for r in stop_trades if r["classification"] == "noise"]
        unk    = [r for r in stop_trades if r["classification"] == "unknown"]

        print(f"\n{'═' * 72}")
        print("  STOP-LOSS CLASSIFICATION")
        print(f"{'═' * 72}")
        print(f"  Total stop-loss exits with candle data: {len(stop_trades)}")
        print()

        for label, group in [("TREND (sustained adverse move)", trend),
                              ("NOISE (brief spike, then reversed)", noise),
                              ("UNKNOWN (insufficient candles)", unk)]:
            if not group:
                continue
            total_pnl = sum(r["pnl"] for r in group)
            avg_adv   = sum(r["max_adverse"] for r in group) / len(group)
            print(f"  {label}")
            print(f"    Count:        {len(group)}")
            print(f"    Total P&L:    ${total_pnl:+.2f}")
            print(f"    Avg max adv:  {avg_adv:.1f}¢")
            for r in group:
                print(f"      #{r['id']:<4d}  {r['ticker']:<38}  adv={r['max_adverse']:3d}¢  pnl={r['pnl']:+.2f}$")
            print()

        print("  Interpretation:")
        if trend:
            print(f"    {len(trend)} TREND stop(s): temperature actually moved against the signal.")
            print("    → Entry criteria may need tightening (model_spread_f, obs_gap_f).")
        if noise:
            print(f"    {len(noise)} NOISE stop(s): stop triggered by thin-market repricing,")
            print("    not a real temperature move. Market likely closed profitably.")
            print("    → Consider widening the stop-loss threshold for these markets.")

    # ── Max adverse move distribution ───────────────────────────────────────
    if results:
        print(f"\n{'═' * 72}")
        print("  MAX ADVERSE MOVE DISTRIBUTION  (all trades with candle data)")
        print(f"{'═' * 72}")
        buckets = [0, 1, 6, 11, 16, 21, 31, 51]
        labels  = ["0¢", "1–5¢", "6–10¢", "11–15¢", "16–20¢", "21–30¢", "31–50¢", ">50¢"]
        for i, (lo, label) in enumerate(zip(buckets, labels)):
            hi = buckets[i + 1] if i + 1 < len(buckets) else 9999
            group = [r for r in results if lo <= r["max_adverse"] < hi]
            if not group:
                continue
            wins = sum(1 for r in group if r["outcome"] == "won")
            bar = "█" * len(group)
            print(f"  {label:>8}  {bar:<20}  {len(group):3d} trades  win={wins}/{len(group)}")

        print()
        all_adv = [r["max_adverse"] for r in results]
        print(f"  Median max adverse: {sorted(all_adv)[len(all_adv)//2]}¢")
        print(f"  Mean max adverse:   {sum(all_adv)/len(all_adv):.1f}¢")
        stop_adv = [r["max_adverse"] for r in stop_trades] if stop_trades else []
        if stop_adv:
            print(f"  Median at stop-loss: {sorted(stop_adv)[len(stop_adv)//2]}¢")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", default="forecast_no",
                        help="opportunity_kind to analyze (default: forecast_no)")
    parser.add_argument("--ids", type=int, nargs="+", metavar="ID")
    parser.add_argument("--exit-reason", metavar="REASON",
                        help="Filter to a specific exit_reason (e.g. stop_loss)")
    parser.add_argument("--opp-db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--candle-db", default=str(DEFAULT_CANDLESTICK_DB_PATH))
    args = parser.parse_args()

    opp_conn    = sqlite3.connect(args.opp_db,    isolation_level=None)
    candle_conn = sqlite3.connect(args.candle_db, isolation_level=None)

    try:
        analyze(
            opp_conn, candle_conn,
            kind=args.kind,
            trade_ids=args.ids,
            exit_reason_filter=args.exit_reason,
        )
    finally:
        opp_conn.close()
        candle_conn.close()


if __name__ == "__main__":
    main()
