"""
Backtest the YES_bid entry price gate for forecast_no using candlestick data.

Two analyses:
  A) Trades we entered (trades table) joined with full price paths from candlesticks.db
     → Shows actual P&L curves, peak gain, PT simulation with real tick data
  B) Proxy backtest across ALL settled weather markets in candlesticks.db
     → Simulates "enter if YES_bid in range X at 06:00 UTC on trade day"
     → No model consensus required — tests whether price regime alone is predictive

Run:
  venv/bin/python scripts/backtest_candle_yesbid_gate.py
"""

import sqlite3
import json
from datetime import datetime, timezone, timedelta
from collections import defaultdict

CANDLES_DB = "data/candlesticks.db"
TRADES_DB  = "data/db/opportunity_log.db"

# Typical entry window: 2–10 AM UTC (bot runs overnight US time)
ENTRY_WINDOW_START_UTC = 2   # hour
ENTRY_WINDOW_END_UTC   = 10  # hour


# ──────────────────────────────────────────────
# A) Price-path analysis for actual entered trades
# ──────────────────────────────────────────────

def load_forecast_no_trades(db: str) -> list[dict]:
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    rows = con.execute("""
        SELECT id, ticker, limit_price, count, exit_pnl_cents, exit_reason,
               exit_price_cents, peak_pct_gain, logged_at, corroborating_sources, note
        FROM trades
        WHERE source = 'forecast_no'
          AND exit_reason IS NOT NULL
        ORDER BY logged_at
    """).fetchall()
    con.close()

    trades = []
    for r in rows:
        try:
            note = json.loads(r["note"] or "{}")
        except Exception:
            note = {}
        trades.append({
            "id":            r["id"],
            "ticker":        r["ticker"],
            "limit_price":   r["limit_price"],    # YES_bid at entry (cents)
            "no_cost":       100 - r["limit_price"],
            "count":         r["count"],
            "exit_pnl":      r["exit_pnl_cents"],
            "exit_reason":   r["exit_reason"],
            "exit_price":    r["exit_price_cents"],
            "peak_pct_gain": r["peak_pct_gain"] or 0.0,
            "logged_at":     r["logged_at"],
            "sources":       r["corroborating_sources"] or "",
            "note":          note,
        })
    return trades


def load_candles_for_ticker(con: sqlite3.Connection, ticker: str) -> list[dict]:
    rows = con.execute("""
        SELECT period_ts, bid_open, bid_close, ask_open, ask_close, price_close
        FROM candles
        WHERE ticker = ?
        ORDER BY period_ts
    """, (ticker,)).fetchall()
    return [{"ts": r[0], "bid_open": r[1], "bid_close": r[2],
             "ask_open": r[3], "ask_close": r[4], "price": r[5]} for r in rows]


def simulate_exit(candles: list[dict], entry_ts: int, no_cost: int, count: int,
                  pt_pct: float = 0.0, sl_pct: float = 0.90) -> dict:
    """
    Simulate holding NO position through candle history.
    - NO position: paid `no_cost` cents, wins if YES_bid → 0
    - Exit triggers:
        - PT: NO implied value = (100 - YES_bid) → gain% = (NO_value / no_cost) - 1 >= pt_pct
        - SL: NO implied value drops below sl_pct * no_cost
    - Settlement: last candle price_close (0 = YES won, 100 = NO won → NO value = 100 - price_close)
    Returns dict with exit_reason, exit_no_value, pnl, peak_gain_pct
    """
    peak_gain = 0.0
    prev_bid = None

    for c in candles:
        if c["ts"] < entry_ts:
            continue
        bid = c["bid_close"] if c["bid_close"] is not None else c["bid_open"]
        if bid is None:
            continue

        no_value = 100 - bid   # current NO bid value
        gain_pct = (no_value / no_cost) - 1.0
        peak_gain = max(peak_gain, gain_pct)

        if pt_pct > 0 and gain_pct >= pt_pct:
            pnl = (no_value - no_cost) * count
            return {"reason": "profit_take", "no_value": no_value, "pnl": pnl, "peak_gain": peak_gain}

        if gain_pct <= (sl_pct - 1.0):   # sl_pct fraction of cost → gain = sl_pct - 1
            # SL threshold: NO value < sl_pct * no_cost
            sl_no_value = no_cost * sl_pct
            if no_value < sl_no_value:
                pnl = (no_value - no_cost) * count
                return {"reason": "stop_loss", "no_value": no_value, "pnl": pnl, "peak_gain": peak_gain}
        prev_bid = bid

    # Settlement
    last = candles[-1]["price"] if candles else None
    if last is None:
        return {"reason": "no_data", "no_value": None, "pnl": None, "peak_gain": peak_gain}
    no_settle = 100 - last
    pnl = (no_settle - no_cost) * count
    reason = "win" if no_settle > no_cost else "loss"
    return {"reason": reason, "no_value": no_settle, "pnl": pnl, "peak_gain": peak_gain}


def part_a(trades: list[dict], candles_con: sqlite3.Connection) -> None:
    print("\n" + "="*80)
    print("PART A — Actual Trades × Candlestick Price Paths")
    print("="*80)

    markets = {}
    for t in trades:
        if t["ticker"] not in markets:
            row = candles_con.execute(
                "SELECT result, open_ts FROM markets WHERE ticker=?", (t["ticker"],)
            ).fetchone()
            markets[t["ticker"]] = row

    # PT sweep using real price paths
    print("\n--- PT Threshold Sweep (real tick data, SL=90%) ---")
    print(f"{'PT':>6} | {'n':>4} {'wins':>4} {'WR':>6} {'PnL($)':>8} | details")
    print("-" * 60)

    for pt in [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        results = []
        for t in trades:
            candles = load_candles_for_ticker(candles_con, t["ticker"])
            if not candles:
                # Fall back to stored data
                if pt == 0.0:
                    results.append(t["exit_pnl"])
                else:
                    results.append(t["exit_pnl"])  # no candle data
                continue

            entry_ts = int(datetime.fromisoformat(t["logged_at"]).timestamp())
            sim = simulate_exit(candles, entry_ts, t["no_cost"], t["count"], pt_pct=pt, sl_pct=0.90)
            if sim["pnl"] is not None:
                results.append(sim["pnl"])
            else:
                results.append(t["exit_pnl"])

        n = len(results)
        wins = sum(1 for p in results if p > 0)
        total = sum(results)
        wr = wins / n if n else 0
        flag = " *" if pt in (0.25, 0.30) else ""
        print(f"{f'{pt*100:.0f}%':>6} | {n:>4} {wins:>4} {wr:>5.1%} {total/100:>+8.2f}{flag}")

    # Price bucket breakdown using candle data to reconstruct entry price
    print("\n--- YES_bid Bucket × Candle Settlement (actual trades only) ---")
    buckets = defaultdict(lambda: {"n": 0, "wins": 0, "pnl": 0})
    for t in trades:
        yb = t["limit_price"]
        bucket = f"{(yb//10)*10}–{(yb//10)*10+9}"
        candles = load_candles_for_ticker(candles_con, t["ticker"])
        if candles:
            entry_ts = int(datetime.fromisoformat(t["logged_at"]).timestamp())
            sim = simulate_exit(candles, entry_ts, t["no_cost"], t["count"], pt_pct=0.0, sl_pct=0.90)
            pnl = sim["pnl"] if sim["pnl"] is not None else t["exit_pnl"]
        else:
            pnl = t["exit_pnl"]
        buckets[bucket]["n"] += 1
        buckets[bucket]["pnl"] += pnl
        if pnl > 0:
            buckets[bucket]["wins"] += 1

    print(f"  {'YES_bid bucket':>16} | {'n':>4} {'WR':>6} {'P&L($)':>8} {'break-even':>10}")
    print("  " + "-"*55)
    for bucket in sorted(buckets):
        b = buckets[bucket]
        wr = b["wins"] / b["n"] if b["n"] else 0
        lo = int(bucket.split("–")[0])
        be = (100 - lo - 5) / 100   # mid-bucket NO cost break-even WR
        flag = " <-- good" if wr >= be and b["n"] >= 3 else ""
        print(f"  {bucket:>16} | {b['n']:>4} {wr:>5.1%} {b['pnl']/100:>+8.2f} {be:>9.1%}{flag}")


# ──────────────────────────────────────────────
# B) Proxy backtest across ALL settled markets
# ──────────────────────────────────────────────

def part_b(candles_con: sqlite3.Connection) -> None:
    print("\n" + "="*80)
    print("PART B — Proxy Backtest: All Settled Weather Markets (no model consensus)")
    print(f"Entry window: {ENTRY_WINDOW_START_UTC:02d}:00–{ENTRY_WINDOW_END_UTC:02d}:00 UTC on market day")
    print("="*80)

    markets = candles_con.execute("""
        SELECT ticker, series, result, open_ts, close_ts, final_price
        FROM markets
        WHERE (series LIKE 'KXHIGH%' OR series LIKE 'KXLOWT%')
          AND result IN ('yes', 'no')
    """).fetchall()

    print(f"\nTotal settled weather markets: {len(markets)}")

    # For each market, find candles in the entry window on the observation day
    # The market observation date is embedded in the ticker (e.g. 26APR07 → Apr 7 2026)
    # The open_ts is the previous day's 14:00 UTC (market opens)
    # Entry window = open_ts + ~16h to open_ts + ~24h = 06:00–14:00 UTC observation day

    buckets = defaultdict(lambda: {"n": 0, "wins": 0, "pnl": 0, "pnls": []})
    skipped = 0

    for m in markets:
        ticker, series, result, open_ts, close_ts, final_price = m
        # Entry window: 06:00–10:00 UTC on observation day
        # Market opens at 14:00 UTC day-1 → observation day 06:00 UTC = open_ts + 16h
        window_start = open_ts + 16 * 3600  # 06:00 UTC obs day
        window_end   = open_ts + 20 * 3600  # 10:00 UTC obs day

        candles = candles_con.execute("""
            SELECT period_ts, bid_open, bid_close
            FROM candles
            WHERE ticker=? AND period_ts >= ? AND period_ts <= ?
            ORDER BY period_ts
            LIMIT 1
        """, (ticker, window_start, window_end)).fetchall()

        if not candles:
            skipped += 1
            continue

        # Use first candle in entry window as proxy entry
        c = candles[0]
        bid = c[2] if c[2] is not None else c[1]
        if bid is None:
            skipped += 1
            continue

        yes_bid_at_entry = bid
        no_cost = 100 - yes_bid_at_entry

        # NO wins when result == 'no'
        win = (result == "no")
        if win:
            pnl_cents = (100 - no_cost)   # per contract: receive 100, paid no_cost
        else:
            pnl_cents = -no_cost

        bucket_lo = (yes_bid_at_entry // 10) * 10
        bucket = f"{bucket_lo}–{bucket_lo+9}"
        buckets[bucket]["n"] += 1
        buckets[bucket]["pnls"].append(pnl_cents)
        buckets[bucket]["pnl"] += pnl_cents
        if win:
            buckets[bucket]["wins"] += 1

    print(f"Skipped (no candle in entry window): {skipped}")
    print(f"\n  {'YES_bid bucket':>16} | {'n':>5} {'WR':>6} {'P&L/trade':>10} {'Total($)':>9} {'break-even':>10}")
    print("  " + "-"*65)

    for bucket in sorted(buckets):
        b = buckets[bucket]
        if b["n"] < 5:
            continue
        wr = b["wins"] / b["n"]
        avg_pnl = b["pnl"] / b["n"]
        total = b["pnl"] / 100
        lo = int(bucket.split("–")[0])
        be = (100 - lo - 5) / 100   # mid-bucket NO cost ÷ 100
        flag = " <-- good" if wr >= be else ""
        print(f"  {bucket:>16} | {b['n']:>5} {wr:>5.1%} {avg_pnl:>+10.1f}¢ {total:>+9.2f}{flag}")

    # Filter: only YES_bid 30–50 range
    print("\n--- YES_bid 30–50¢ detailed breakdown ---")
    sub = [b for bucket, b in buckets.items() if 30 <= int(bucket.split("–")[0]) <= 49]
    n = sum(b["n"] for b in sub)
    wins = sum(b["wins"] for b in sub)
    pnl = sum(b["pnl"] for b in sub)
    if n:
        print(f"  YES_bid 30–50¢: n={n}, WR={wins/n:.1%}, P&L={pnl/100:+.2f}, avg={pnl/n:+.1f}¢/trade")

    sub60 = [b for bucket, b in buckets.items() if 50 <= int(bucket.split("–")[0]) <= 69]
    n2 = sum(b["n"] for b in sub60)
    wins2 = sum(b["wins"] for b in sub60)
    pnl2 = sum(b["pnl"] for b in sub60)
    if n2:
        print(f"  YES_bid 50–70¢: n={n2}, WR={wins2/n2:.1%}, P&L={pnl2/100:+.2f}, avg={pnl2/n2:+.1f}¢/trade")

    sub80 = [b for bucket, b in buckets.items() if 70 <= int(bucket.split("–")[0]) <= 89]
    n3 = sum(b["n"] for b in sub80)
    wins3 = sum(b["wins"] for b in sub80)
    pnl3 = sum(b["pnl"] for b in sub80)
    if n3:
        print(f"  YES_bid 70–90¢: n={n3}, WR={wins3/n3:.1%}, P&L={pnl3/100:+.2f}, avg={pnl3/n3:+.1f}¢/trade")

    # Series breakdown for 35–45¢ bucket
    print("\n--- Per-city WIN rate (YES_bid 35–50¢ proxy entries) ---")
    city_stats = defaultdict(lambda: {"n": 0, "wins": 0, "pnl": 0})
    for m in markets:
        ticker, series, result, open_ts, close_ts, final_price = m
        window_start = open_ts + 16 * 3600
        window_end   = open_ts + 20 * 3600
        candles = candles_con.execute("""
            SELECT period_ts, bid_open, bid_close FROM candles
            WHERE ticker=? AND period_ts >= ? AND period_ts <= ?
            ORDER BY period_ts LIMIT 1
        """, (ticker, window_start, window_end)).fetchall()
        if not candles:
            continue
        c = candles[0]
        bid = c[2] if c[2] is not None else c[1]
        if bid is None or not (35 <= bid <= 50):
            continue
        no_cost = 100 - bid
        win = (result == "no")
        pnl = (100 - no_cost) if win else -no_cost
        city_stats[series]["n"] += 1
        city_stats[series]["pnl"] += pnl
        if win:
            city_stats[series]["wins"] += 1

    print(f"  {'City':>16} | {'n':>4} {'WR':>6} {'P&L($)':>8}")
    print("  " + "-"*40)
    for city in sorted(city_stats, key=lambda c: -city_stats[c]["n"]):
        b = city_stats[city]
        if b["n"] < 2:
            continue
        wr = b["wins"] / b["n"]
        print(f"  {city:>16} | {b['n']:>4} {wr:>5.1%} {b['pnl']/100:>+8.2f}")


def main():
    trades = load_forecast_no_trades(TRADES_DB)
    print(f"Loaded {len(trades)} settled forecast_no trades")

    candles_con = sqlite3.connect(f"file:{CANDLES_DB}?mode=ro", uri=True)

    # Check overlap
    tickers_with_candles = set(
        r[0] for r in candles_con.execute("SELECT DISTINCT ticker FROM markets").fetchall()
    )
    our_tickers = set(t["ticker"] for t in trades)
    overlap = our_tickers & tickers_with_candles
    print(f"Trades with candle data: {len(overlap)}/{len(our_tickers)}")

    part_a(trades, candles_con)
    part_b(candles_con)

    candles_con.close()


if __name__ == "__main__":
    main()
