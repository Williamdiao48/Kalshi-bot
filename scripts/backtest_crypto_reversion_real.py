"""Validate the crypto 15-minute mean-reversion edge against REAL Kalshi data.

The proxy backtest (`backtest_btc_momentum.py`) established, on Yahoo OHLCV, that
crypto 15-minute direction is anti-predictive: fading the prior move (buy YES
after a down-run, NO after an up-run) is positive out to 2 years — but it assumed
a flat 1c spread and a 50c entry.  This script re-runs that test on the actual
Kalshi order book:

  * ENTRY  price  → real yes_bid / yes_ask from the settled-market candlesticks
                    (data/candlesticks_15m.db, built by fetch_candlestick_history.py)
  * OUTCOME       → the real Kalshi settlement result (yes / no)
  * SIGNAL (B)    → the prior-window spot return from Coinbase 1-min candles
                    (Binance is geo-blocked from the US → 451)

Two arms:
  B (main)  — spot-magnitude reversion: fade the prior 15-min spot move, swept by
              threshold, so we see whether the edge really does grow with move size
              once real spreads are paid.
  A (cross) — pure-Kalshi serial autocorrelation: use the PREVIOUS 15-min market's
              own outcome as the up/down signal (no spot data).  Free sanity check.

A validation step first confirms result=="yes" lines up with spot rising over the
market's own window — i.e. that our spot alignment and the contract's "up" polarity
agree — before any P&L is trusted.

Usage:
  venv/bin/python scripts/backtest_crypto_reversion_real.py
  venv/bin/python scripts/backtest_crypto_reversion_real.py --db data/candlesticks_15m.db
"""

from __future__ import annotations

import argparse
import asyncio
import bisect
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import aiohttp

# Real Kalshi trading fee is ~ round(0.07 * price * (1 - price)) cents per contract.
# The proxy script ignored fees; we surface P&L both gross and net of this.
SPREAD_NOTE = "entry pays the real yes_ask (YES) or 100 - yes_bid (NO)"

SERIES_PRODUCT = {
    "KXBTC15M": "BTC-USD",
    "KXETH15M": "ETH-USD",
    "KXSOL15M": "SOL-USD",
}

WINDOW_SEC = 15 * 60  # 15-minute market window
THRESHOLDS = [0.0, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.008]


# ---------------------------------------------------------------------------
# Load Kalshi settled-market data from the candlestick DB
# ---------------------------------------------------------------------------

def load_markets(db_path: str) -> dict[str, list[dict]]:
    """Return {series: [market dicts sorted by open_ts]}.

    Each market dict: ticker, open_ts, close_ts, result, entry_bid, entry_ask
    where entry_bid/ask are the first candle's close (minute 1 of the window).
    """
    con = sqlite3.connect(db_path)
    # First-candle (minute-1) close bid/ask = realistic entry once quotes populate.
    rows = con.execute(
        """
        WITH firstc AS (
            SELECT ticker, bid_close, ask_close,
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY period_ts) rn
            FROM candles
        )
        SELECT m.series, m.ticker, m.open_ts, m.close_ts, m.result,
               f.bid_close, f.ask_close
        FROM markets m
        JOIN firstc f ON f.ticker = m.ticker AND f.rn = 1
        WHERE m.result IN ('yes','no')
        ORDER BY m.series, m.open_ts
        """
    ).fetchall()
    con.close()

    by_series: dict[str, list[dict]] = defaultdict(list)
    for series, ticker, open_ts, close_ts, result, bid, ask in rows:
        if bid is None or ask is None:
            continue
        by_series[series].append({
            "ticker": ticker, "open_ts": int(open_ts), "close_ts": int(close_ts),
            "result": result, "entry_bid": int(bid), "entry_ask": int(ask),
        })
    return by_series


# ---------------------------------------------------------------------------
# Fetch Coinbase 1-minute spot closes over the needed range
# ---------------------------------------------------------------------------

async def fetch_coinbase_1m(
    session: aiohttp.ClientSession, product: str, start_ts: int, end_ts: int,
) -> dict[int, float]:
    """Return {minute_unix_ts: close_price} for product over [start_ts, end_ts].

    Coinbase caps each request at 300 candles, so we page in 300-minute chunks.
    Response rows: [time, low, high, open, close, volume], newest-first.
    """
    url = f"https://api.exchange.coinbase.com/products/{product}/candles"
    prices: dict[int, float] = {}
    chunk = 300 * 60  # 300 one-minute candles per request
    t = start_ts
    while t < end_ts:
        seg_end = min(t + chunk, end_ts)
        params = {
            "granularity": 60,
            "start": _iso(t),
            "end": _iso(seg_end),
        }
        for attempt in range(4):
            try:
                async with session.get(
                    url, params=params, timeout=aiohttp.ClientTimeout(total=20)
                ) as r:
                    if r.status == 429:
                        await asyncio.sleep(1.0 * (attempt + 1))
                        continue
                    r.raise_for_status()
                    data = await r.json()
                break
            except Exception:
                if attempt == 3:
                    data = []
                    break
                await asyncio.sleep(0.5 * (attempt + 1))
        for row in data:
            ts, _low, _high, _open, close, _vol = row[:6]
            prices[int(ts)] = float(close)
        await asyncio.sleep(0.18)  # stay under Coinbase public rate limit
        t = seg_end
    return prices


def _iso(ts: int) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


class SpotIndex:
    """Nearest-earlier lookup over sparse 1-min closes (forward-fill gaps)."""

    def __init__(self, prices: dict[int, float]):
        self._ts = sorted(prices)
        self._px = [prices[t] for t in self._ts]

    def at(self, ts: int) -> float | None:
        if not self._ts:
            return None
        i = bisect.bisect_right(self._ts, ts) - 1
        if i < 0:
            return None
        # Reject if the nearest earlier print is more than 10 min stale.
        if ts - self._ts[i] > 600:
            return None
        return self._px[i]


# ---------------------------------------------------------------------------
# Backtest arms
# ---------------------------------------------------------------------------

def _entry_and_win(m: dict, side: str) -> tuple[int, bool]:
    """(entry_cost_cents, won) for buying `side` on market m at real prices."""
    if side == "yes":
        return m["entry_ask"], (m["result"] == "yes")
    return 100 - m["entry_bid"], (m["result"] == "no")


def _fee(price_c: int) -> float:
    """Kalshi trading fee in cents: 0.07 * p * (1-p), p in dollars, per contract."""
    p = price_c / 100.0
    return 0.07 * p * (1.0 - p) * 100.0


def run_arm_B(markets: list[dict], spot: SpotIndex) -> None:
    """Spot-magnitude reversion, swept by prior-move threshold."""
    # Attach prior-window spot return to each market.
    rated = []
    for m in markets:
        s_open = spot.at(m["open_ts"])
        s_prior = spot.at(m["open_ts"] - WINDOW_SEC)
        if s_open is None or s_prior is None or s_prior <= 0:
            continue
        m = {**m, "prior_ret": s_open / s_prior - 1.0}
        rated.append(m)

    print(f"    rated markets (with spot): {len(rated)} / {len(markets)}")
    print(f"    {'|move|>':>8}  {'n_YES':>6}  {'WR_YES':>7}  "
          f"{'n_NO':>6}  {'WR_NO':>7}  {'trades':>7}  {'gross¢/t':>9}  {'net¢/t':>8}  {'net $':>8}")
    for thr in THRESHOLDS:
        n_yes = n_no = 0
        gross = net = 0.0
        wins_yes = wins_no = 0
        for m in rated:
            r = m["prior_ret"]
            if r < -thr:            # down-run → fade → buy YES
                side = "yes"
            elif r > thr:           # up-run → fade → buy NO
                side = "no"
            else:
                continue
            entry, won = _entry_and_win(m, side)
            pnl = (100 - entry) if won else -entry
            gross += pnl
            net += pnl - _fee(entry)
            if side == "yes":
                n_yes += 1; wins_yes += int(won)
            else:
                n_no += 1; wins_no += int(won)
        trades = n_yes + n_no
        if trades == 0:
            continue
        wr_yes = 100 * wins_yes / n_yes if n_yes else 0.0
        wr_no = 100 * wins_no / n_no if n_no else 0.0
        print(f"    {thr:>8.4f}  {n_yes:>6}  {wr_yes:>6.1f}%  "
              f"{n_no:>6}  {wr_no:>6.1f}%  {trades:>7}  "
              f"{gross/trades:>+9.2f}  {net/trades:>+8.2f}  {net/100:>+8.2f}")


def run_arm_A(markets: list[dict]) -> None:
    """Pure-Kalshi serial autocorrelation: fade the PREVIOUS market's outcome.

    Signal = previous consecutive market's result; only counts pairs whose
    windows are actually adjacent (prev.close_ts == this.open_ts).
    """
    n = wins = 0
    gross = net = 0.0
    for prev, cur in zip(markets, markets[1:]):
        if prev["close_ts"] != cur["open_ts"]:
            continue  # not consecutive (gap in data)
        # Fade prev: prev up (yes) → buy NO now; prev down (no) → buy YES now.
        side = "no" if prev["result"] == "yes" else "yes"
        entry, won = _entry_and_win(cur, side)
        pnl = (100 - entry) if won else -entry
        gross += pnl
        net += pnl - _fee(entry)
        n += 1; wins += int(won)
    if n == 0:
        print("    no consecutive pairs")
        return
    print(f"    pairs={n}  WR={100*wins/n:.1f}%  "
          f"gross={gross/n:+.2f}¢/t  net={net/n:+.2f}¢/t  net_total=${net/100:+.2f}")


def validate_polarity(markets: list[dict], spot: SpotIndex) -> None:
    """Confirm result=='yes' lines up with spot rising over the market's window."""
    agree = total = 0
    for m in markets:
        s_open = spot.at(m["open_ts"])
        s_close = spot.at(m["close_ts"])
        if s_open is None or s_close is None:
            continue
        spot_up = s_close > s_open
        result_yes = m["result"] == "yes"
        agree += int(spot_up == result_yes)
        total += 1
    if total:
        print(f"    polarity check: result=='yes' matches spot-up on "
              f"{100*agree/total:.1f}% of {total} markets "
              f"(≈100% ⇒ yes=up and spot aligned; ≈0% ⇒ inverted)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(db_path: str) -> None:
    by_series = load_markets(db_path)
    if not by_series:
        print(f"No markets in {db_path}. Run fetch_candlestick_history.py first.")
        return

    async with aiohttp.ClientSession() as session:
        for series, markets in by_series.items():
            product = SERIES_PRODUCT.get(series)
            print("=" * 78)
            print(f"  {series}  ({product})  —  {len(markets)} settled markets")
            print("=" * 78)

            start_ts = min(m["open_ts"] for m in markets) - WINDOW_SEC - 120
            end_ts = max(m["close_ts"] for m in markets) + 120
            print(f"  fetching Coinbase 1m spot {product} "
                  f"({(end_ts-start_ts)//3600}h)…", flush=True)
            prices = await fetch_coinbase_1m(session, product, start_ts, end_ts)
            spot = SpotIndex(prices)
            print(f"  spot minutes: {len(prices)}")

            validate_polarity(markets, spot)
            print(f"\n  [B] SPOT-MAGNITUDE REVERSION  ({SPREAD_NOTE}, net = after Kalshi fee)")
            run_arm_B(markets, spot)
            print(f"\n  [A] PURE-KALSHI serial-autocorrelation cross-check")
            run_arm_A(markets)
            print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/candlesticks_15m.db")
    args = ap.parse_args()
    if not Path(args.db).exists():
        print(f"DB not found: {args.db}", file=sys.stderr)
        sys.exit(1)
    asyncio.run(main(args.db))
