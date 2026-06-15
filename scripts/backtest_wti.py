#!/usr/bin/env python3
"""Backtest WTI futures signal strategy against historical KXWTI Kalshi markets.

Data sources
------------
  data/wti_candlesticks.db  — Kalshi KXWTI hourly bid/ask candles (~60 days)
  Yahoo Finance CL=F        — WTI front-month hourly prices (same period)

Strategy
--------
  Signal : WTI futures price is >= EDGE_THRESH $/bbl above (YES) or below (NO)
           the market's strike price.
  Entry  : First hourly candle where the edge condition is met AND bid in range.
           YES entry cost = yes_bid_close  (passive limit assumption)
           NO  entry cost = 100 - yes_ask_close  (NO bid = 100 - YES ask)
  Exit   : PT when bid rises >= PT_PCT from entry; else hold to settlement.

Sweep
-----
  EDGE_THRESHOLDS : $/bbl edge needed to enter
  PT_PCTS         : profit-take gain threshold
  MAX_BIDS        : max entry bid (skip expensive positions)

Usage
-----
  venv/bin/python scripts/backtest_wti.py
  venv/bin/python scripts/backtest_wti.py --best 20   # show top-20 combos only
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yfinance as yf  # noqa: E402

CANDLES_DB = Path("data/wti_candlesticks.db")

EDGE_THRESHOLDS = [0.50, 1.00, 1.50, 2.00, 3.00]
PT_PCTS         = [0.10, 0.15, 0.20, 0.25, 0.30]
MAX_BIDS        = [40, 60, 80]
MIN_BID         = 5   # skip near-worthless bids

# Tickers look like:  KXWTI-26MAY2214-T88.99
_STRIKE_RE = re.compile(r"-T([\d.]+)$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_strike(ticker: str) -> float | None:
    m = _STRIKE_RE.search(ticker)
    return float(m.group(1)) if m else None


def _fetch_wti_hourly(start: str, end: str) -> dict[int, float]:
    """Return {hour_epoch: close_price} from Yahoo Finance CL=F."""
    df = yf.download("CL=F", start=start, end=end, interval="1h", progress=False)
    if df.empty:
        return {}
    prices: dict[int, float] = {}
    for ts, row in df.iterrows():
        epoch = int(ts.timestamp())
        hour_epoch = epoch - (epoch % 3600)
        try:
            prices[hour_epoch] = float(row[("Close", "CL=F")])
        except (KeyError, TypeError):
            pass
    return prices


def _wti_at(wti_prices: dict[int, float], period_ts: int) -> float | None:
    """Nearest WTI price at or before period_ts (hourly bucket)."""
    hour = period_ts - (period_ts % 3600)
    return wti_prices.get(hour) or wti_prices.get(hour - 3600)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate(
    ticker: str,
    result: str,
    candles: list[tuple[int, int | None, int | None]],
    wti_prices: dict[int, float],
    edge_thresh: float,
    pt_pct: float,
    max_bid: int,
) -> dict | None:
    """Simulate one market.  Returns a trade dict or None if no entry triggered."""
    strike = _parse_strike(ticker)
    if strike is None:
        return None

    entry_ts: int | None   = None
    entry_side: str | None = None
    entry_cost: int        = 0
    pt_target: int         = 0

    for period_ts, bid_close, ask_close in candles:
        if bid_close is None or ask_close is None:
            continue

        wti = _wti_at(wti_prices, period_ts)
        if wti is None:
            continue

        if entry_ts is None:
            # --- seek entry ---
            yes_edge = wti - strike
            no_edge  = strike - wti

            if yes_edge >= edge_thresh:
                cost = bid_close
                if cost < MIN_BID or cost > max_bid:
                    continue
                entry_ts   = period_ts
                entry_side = "yes"
                entry_cost = cost
                pt_target  = round(cost * (1 + pt_pct))

            elif no_edge >= edge_thresh:
                # NO bid = 100 - YES ask
                no_bid = 100 - ask_close
                if no_bid < MIN_BID or no_bid > max_bid:
                    continue
                entry_ts   = period_ts
                entry_side = "no"
                entry_cost = no_bid
                pt_target  = round(no_bid * (1 + pt_pct))

        else:
            # --- track position for PT ---
            current_bid = bid_close if entry_side == "yes" else (100 - ask_close)
            if current_bid >= pt_target:
                pnl = current_bid - entry_cost
                return {
                    "ticker":    ticker,
                    "side":      entry_side,
                    "entry":     entry_cost,
                    "exit_val":  current_bid,
                    "exit":      "profit_take",
                    "pnl":       pnl,
                    "result":    result,
                }

    if entry_ts is None:
        return None  # no signal triggered

    # Held to settlement
    if entry_side == "yes":
        settle_val = 100 if result == "yes" else 0
    else:
        settle_val = 100 if result == "no" else 0

    pnl = settle_val - entry_cost
    return {
        "ticker":   ticker,
        "side":     entry_side,
        "entry":    entry_cost,
        "exit_val": settle_val,
        "exit":     "settlement",
        "pnl":      pnl,
        "result":   result,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    if not CANDLES_DB.exists():
        print(f"ERROR: {CANDLES_DB} not found. Run fetch_candlestick_history.py first.")
        sys.exit(1)

    con = sqlite3.connect(CANDLES_DB)

    markets: list[tuple] = con.execute(
        "SELECT ticker, series, open_ts, close_ts, result FROM markets ORDER BY open_ts"
    ).fetchall()
    print(f"Loaded {len(markets)} markets from candlesticks DB")

    # Load all candles into memory
    candles_map: dict[str, list] = {}
    for (ticker, *_) in markets:
        candles_map[ticker] = con.execute(
            "SELECT period_ts, bid_close, ask_close FROM candles "
            "WHERE ticker=? ORDER BY period_ts",
            (ticker,),
        ).fetchall()

    total_candles = sum(len(v) for v in candles_map.values())
    print(f"Loaded {total_candles:,} candles")

    # Date range
    min_open = min(m[2] for m in markets)
    max_close = max(m[3] for m in markets)
    start_str = datetime.fromtimestamp(min_open, tz=timezone.utc).date().isoformat()
    end_str   = (datetime.fromtimestamp(max_close, tz=timezone.utc) + timedelta(days=1)).date().isoformat()

    print(f"Fetching WTI hourly {start_str} → {end_str} …")
    wti_prices = _fetch_wti_hourly(start_str, end_str)
    print(f"  {len(wti_prices)} hourly WTI prices\n")

    # --- parameter sweep ---
    sweep_results = []

    for edge_thresh in EDGE_THRESHOLDS:
        for pt_pct in PT_PCTS:
            for max_bid in MAX_BIDS:
                trades: list[dict] = []
                for mkt in markets:
                    ticker, _, _, _, result = mkt
                    if result not in ("yes", "no"):
                        continue
                    candles = candles_map.get(ticker, [])
                    t = simulate(ticker, result, candles, wti_prices, edge_thresh, pt_pct, max_bid)
                    if t is not None:
                        trades.append(t)

                n = len(trades)
                if n == 0:
                    continue

                pt_trades  = [t for t in trades if t["exit"] == "profit_take"]
                st_trades  = [t for t in trades if t["exit"] == "settlement"]
                wins       = [t for t in trades if t["pnl"] > 0]
                total_pnl  = sum(t["pnl"] for t in trades) / 100.0
                avg_pnl    = sum(t["pnl"] for t in trades) / n

                yes_trades = [t for t in trades if t["side"] == "yes"]
                no_trades  = [t for t in trades if t["side"] == "no"]

                sweep_results.append({
                    "edge":      edge_thresh,
                    "pt_pct":    pt_pct,
                    "max_bid":   max_bid,
                    "n":         n,
                    "n_yes":     len(yes_trades),
                    "n_no":      len(no_trades),
                    "pt_rate":   len(pt_trades) / n,
                    "win_rate":  len(wins) / n,
                    "avg_pnl":   avg_pnl,
                    "total_usd": total_pnl,
                })

    # Sort by avg_pnl descending
    sweep_results.sort(key=lambda x: -x["avg_pnl"])

    top = sweep_results[:args.best]
    print(f"{'edge':>5} {'pt%':>5} {'max_b':>5} {'n':>5} {'yes':>5} {'no':>5} "
          f"{'pt%':>6} {'win%':>6} {'avg¢':>7} {'total$':>8}")
    print("-" * 70)
    for r in top:
        print(
            f"{r['edge']:>5.1f} {r['pt_pct']:>4.0%} {r['max_bid']:>5} "
            f"{r['n']:>5} {r['n_yes']:>5} {r['n_no']:>5} "
            f"{r['pt_rate']:>5.1%} {r['win_rate']:>5.1%} "
            f"{r['avg_pnl']:>+7.1f} {r['total_usd']:>+8.2f}"
        )

    # Best param detail
    if sweep_results:
        best = sweep_results[0]
        print(f"\n=== Best combo: edge={best['edge']}  pt={best['pt_pct']:.0%}  max_bid={best['max_bid']}¢ ===")
        print(f"  {best['n']} trades  |  PT rate {best['pt_rate']:.1%}  |  "
              f"Win rate {best['win_rate']:.1%}  |  "
              f"Avg P&L {best['avg_pnl']:+.1f}¢  |  Total ${best['total_usd']:+.2f}")

        # Trade-level breakdown for best params
        print("\n--- Trade detail (best params, sample) ---")
        trades_best: list[dict] = []
        for mkt in markets:
            ticker, _, _, _, result = mkt
            if result not in ("yes", "no"):
                continue
            candles = candles_map.get(ticker, [])
            t = simulate(ticker, result, candles, wti_prices,
                         best["edge"], best["pt_pct"], best["max_bid"])
            if t is not None:
                trades_best.append(t)

        trades_best.sort(key=lambda x: x["pnl"])
        print(f"{'ticker':45} {'side':4} {'entry':>6} {'exit':>6} {'how':>12} {'pnl':>6}")
        print("-" * 85)
        for t in trades_best[:10]:  # worst 10
            print(f"{t['ticker']:45} {t['side']:4} {t['entry']:>6} {t['exit_val']:>6} "
                  f"{t['exit']:>12} {t['pnl']:>+6}")
        print("  …")
        for t in trades_best[-10:]:  # best 10
            print(f"{t['ticker']:45} {t['side']:4} {t['entry']:>6} {t['exit_val']:>6} "
                  f"{t['exit']:>12} {t['pnl']:>+6}")

    # Live bot comparison row (20-trade sample from DB)
    print("\n--- Live bot reference (20 trades, all PT exits) ---")
    print("  Avg entry: 47.9¢  Avg exit: 59.5¢ (YES) / 69.4¢ (NO)  Avg P&L: +33.6¢  Total: $6.72")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest KXWTI WTI futures strategy")
    parser.add_argument("--best", type=int, default=30, help="Show top N combos (default 30)")
    asyncio_args = parser.parse_args()
    main(asyncio_args)
