"""
Probe how far back Kalshi's settled B-band temperature markets go.

Bypasses the local cache and paginates all the way to cursor exhaustion
for a sample of series. Reports the full date range and checks for summer 2025.

Usage:
  venv/bin/python scripts/probe_kalshi_history.py
"""

import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import aiohttp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kalshi_bot.auth import generate_headers
from kalshi_bot.markets import KALSHI_API_BASE

# Probe these — a mix of original and newer series, high and low
PROBE_SERIES = [
    "KXHIGHLAX",   # original high series, LA
    "KXHIGHNY",    # original high series, NY
    "KXHIGHCHI",   # original high series, Chicago
    "KXHIGHTBOS",  # new KXHIGHT series, Boston
    "KXLOWTNYC",   # low series, NYC
    "KXLOWTLAX",   # low series, LA
]

OUT_FILE = Path("data/backtest/kalshi_history_probe.json")


def parse_ticker_date(ticker: str) -> str | None:
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})-", ticker)
    if not m:
        return None
    try:
        dt = datetime.strptime(f"20{m.group(1)}{m.group(2)}{m.group(3)}", "%Y%b%d")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return None


async def fetch_all_settled(session: aiohttp.ClientSession, series: str) -> list[dict]:
    """Paginate through ALL settled markets for a series, ignoring local cache."""
    markets = []
    cursor = None
    pages = 0

    while True:
        params = {"status": "settled", "limit": 200, "series_ticker": series}
        if cursor:
            params["cursor"] = cursor

        headers = generate_headers("GET", "/trade-api/v2/markets")
        try:
            async with session.get(
                f"{KALSHI_API_BASE}/markets",
                headers=headers,
                params=params,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 429:
                    print(f"  [{series}] Rate limited at page {pages}, waiting 5s...")
                    await asyncio.sleep(5)
                    continue
                resp.raise_for_status()
                data = await resp.json()
        except Exception as e:
            print(f"  [{series}] Error on page {pages}: {e}")
            break

        batch = data.get("markets", [])
        b_bands = [m for m in batch if "-B" in m.get("ticker", "")]
        markets.extend(b_bands)
        pages += 1

        cursor = data.get("cursor")
        if not cursor or not batch:
            break

        await asyncio.sleep(0.3)

    return markets


async def main():
    results = {}

    async with aiohttp.ClientSession() as session:
        for series in PROBE_SERIES:
            print(f"Fetching {series}...")
            t0 = time.time()
            markets = await fetch_all_settled(session, series)
            elapsed = time.time() - t0

            dates = [parse_ticker_date(m.get("ticker", "")) for m in markets]
            dates = sorted(d for d in dates if d)

            if dates:
                summer = [d for d in dates if d[5:7] in ("06", "07", "08")]
                from collections import Counter
                by_year_month = Counter(d[:7] for d in dates)

                print(f"  {len(markets)} B-bands in {elapsed:.1f}s")
                print(f"  Date range: {dates[0]} → {dates[-1]}")
                print(f"  Summer (Jun–Aug): {len(summer)} markets")
                if summer:
                    print(f"    First summer: {summer[0]}  Last summer: {summer[-1]}")
                print(f"  By year-month (sample):")
                for ym in sorted(by_year_month)[:6]:
                    print(f"    {ym}: {by_year_month[ym]}")
                if len(by_year_month) > 6:
                    print(f"    ... ({len(by_year_month)} months total)")
            else:
                print(f"  No B-band markets found.")

            results[series] = {
                "total": len(markets),
                "dates": dates,
                "markets": markets,
            }
            print()

    # Summary
    print("="*60)
    print("SUMMARY")
    print("="*60)
    earliest_overall = None
    has_summer_2025 = False
    has_summer_2024 = False

    for series, info in results.items():
        dates = info["dates"]
        if not dates:
            print(f"  {series}: NO DATA")
            continue
        summer_25 = [d for d in dates if d.startswith("2025-") and d[5:7] in ("06","07","08")]
        summer_24 = [d for d in dates if d.startswith("2024-") and d[5:7] in ("06","07","08")]
        print(f"  {series}: {info['total']} markets  {dates[0]} → {dates[-1]}  "
              f"S2025={len(summer_25)}  S2024={len(summer_24)}")
        if summer_25:
            has_summer_2025 = True
        if summer_24:
            has_summer_2024 = True
        if earliest_overall is None or dates[0] < earliest_overall:
            earliest_overall = dates[0]

    print()
    if has_summer_2025:
        print("✓ Summer 2025 markets found — can retrain with real summer Kalshi data!")
    elif has_summer_2024:
        print("✓ Summer 2024 markets found — can retrain with real summer Kalshi data!")
    else:
        print("✗ No summer markets found. B-band products launched after summer 2025.")
        if earliest_overall:
            print(f"  Earliest market: {earliest_overall}")
        print("  Will need IEM-based summer simulation instead.")

    # Save full results (without market details to keep file small)
    slim = {s: {"total": v["total"], "dates": v["dates"]} for s, v in results.items()}
    OUT_FILE.write_text(json.dumps(slim, indent=2))
    print(f"\nSaved probe results → {OUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
