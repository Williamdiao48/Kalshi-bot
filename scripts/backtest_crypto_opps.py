"""
Backtest KXBTCD opportunities from the opportunity log.

For each unique KXBTCD ticker we logged an opportunity on, fetches the actual
Kalshi settlement result and computes what would have happened if we'd traded
at yes_ask (for implied YES) or (100 - yes_bid) (for implied NO).
"""
import asyncio
import sqlite3
import aiohttp
import time
from collections import defaultdict

from kalshi_bot.auth import generate_headers
from kalshi_bot.markets import KALSHI_API_BASE

DB_PATH = "data/db/opportunity_log.db"


async def fetch_result(session: aiohttp.ClientSession, ticker: str) -> dict | None:
    url = f"{KALSHI_API_BASE}/markets/{ticker}"
    headers = generate_headers("GET", f"/markets/{ticker}")
    try:
        async with session.get(url, headers=headers) as resp:
            if resp.status == 404:
                return None
            if resp.status == 429:
                await asyncio.sleep(2)
                return None
            resp.raise_for_status()
            data = await resp.json()
            m = data.get("market", {})
            return {
                "result": m.get("result"),
                "expiration_value": m.get("expiration_value"),
                "status": m.get("status"),
            }
    except Exception as e:
        print(f"  Error fetching {ticker}: {e}")
        return None


async def fetch_all_results(tickers: list[str]) -> dict[str, dict]:
    results = {}
    async with aiohttp.ClientSession() as session:
        for i, ticker in enumerate(tickers):
            r = await fetch_result(session, ticker)
            if r:
                results[ticker] = r
            if (i + 1) % 50 == 0:
                print(f"  Fetched {i+1}/{len(tickers)}...")
                await asyncio.sleep(0.5)
            else:
                await asyncio.sleep(0.12)  # ~8 req/s to stay under rate limit
    return results


def run_backtest():
    conn = sqlite3.connect(DB_PATH)

    # Load all KXBTCD opportunities — take the FIRST signal per ticker
    # (simulating: we trade on the first opportunity we see)
    opps = conn.execute("""
        SELECT ticker, MIN(logged_at) as first_seen,
               direction, strike, implied_outcome,
               yes_bid, yes_ask, edge, score, data_value
        FROM opportunities
        WHERE ticker LIKE 'KXBTCD%'
        GROUP BY ticker
        ORDER BY first_seen
    """).fetchall()
    conn.close()

    tickers = [r[0] for r in opps]
    print(f"Fetching results for {len(tickers)} unique KXBTCD tickers...")
    results = asyncio.run(fetch_all_results(tickers))
    print(f"Got results for {len(results)} tickers.\n")

    wins = losses = skipped = 0
    total_pnl = 0.0
    not_finalized = 0

    rows = []
    for row in opps:
        ticker, first_seen, direction, strike, implied, yes_bid, yes_ask, edge, score, btc_at_signal = row
        r = results.get(ticker)
        if not r:
            skipped += 1
            continue
        if r["status"] != "finalized":
            not_finalized += 1
            continue

        actual = r["result"]  # "yes" or "no"
        exp_val = r["expiration_value"]

        # What side would we have traded?
        if not implied:
            skipped += 1
            continue
        trade_side = implied.lower()  # "yes" or "no"

        # Entry cost
        if trade_side == "yes":
            entry_cost = yes_ask if yes_ask else None
        else:
            entry_cost = (100 - yes_bid) if yes_bid else None

        if entry_cost is None or implied is None:
            skipped += 1
            continue

        won = (trade_side == actual)
        if won:
            pnl = 100 - entry_cost
            wins += 1
        else:
            pnl = -entry_cost
            losses += 1
        total_pnl += pnl

        rows.append({
            "ticker": ticker,
            "time": first_seen[:16],
            "strike": strike,
            "btc": btc_at_signal,
            "exp_val": exp_val,
            "implied": implied,
            "actual": actual.upper(),
            "entry_cost": entry_cost,
            "pnl": pnl,
            "won": won,
            "score": score,
        })

    total = wins + losses
    print(f"{'='*80}")
    print(f"KXBTCD Opportunity Backtest — first-signal-per-ticker strategy")
    print(f"{'='*80}")
    print(f"Tickers evaluated: {total}  |  Skipped/pending: {skipped + not_finalized}")
    print(f"Wins: {wins}  Losses: {losses}  Win rate: {wins/total*100:.1f}%" if total else "No results")
    print(f"Simulated P&L (1 contract each): {total_pnl:+.0f}¢  avg/trade: {total_pnl/total:+.1f}¢" if total else "")
    print()

    # Break down by score bucket
    print("Win rate by score bucket:")
    buckets = defaultdict(lambda: [0, 0])
    for r in rows:
        b = f"{int(r['score']*10)/10:.1f}+"
        buckets[b][0 if r['won'] else 1] += 1
    for b in sorted(buckets):
        w, l = buckets[b]
        t = w + l
        print(f"  score >= {b}: {w}W {l}L  {w/t*100:.0f}% win rate")

    print()
    # Show losses
    losses_list = [r for r in rows if not r["won"]]
    print(f"Losses ({len(losses_list)}):")
    for r in sorted(losses_list, key=lambda x: x["pnl"]):
        print(f"  {r['ticker']:<38}  btc={r['btc']:.0f}  exp={r['exp_val']}  "
              f"implied={r['implied']}  actual={r['actual']}  cost={r['entry_cost']}¢  pnl={r['pnl']:+.0f}¢")

    print()
    print(f"Wins ({len([r for r in rows if r['won']])}):")
    for r in sorted([r for r in rows if r["won"]], key=lambda x: x["pnl"]):
        print(f"  {r['ticker']:<38}  btc={r['btc']:.0f}  exp={r['exp_val']}  "
              f"implied={r['implied']}  actual={r['actual']}  cost={r['entry_cost']}¢  pnl={r['pnl']:+.0f}¢")


if __name__ == "__main__":
    run_backtest()
