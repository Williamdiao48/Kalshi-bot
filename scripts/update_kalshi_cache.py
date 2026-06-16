"""
Safely update the Kalshi markets cache by merging fresh API data with the
existing cache. Markets already in the cache (keyed on ticker) are preserved.
New markets from the API are appended.

This avoids losing the March 27 - April 5 pilot markets that the API no longer
returns in its settled-markets endpoint.

Usage:
  venv/bin/python scripts/update_kalshi_cache.py [--dry-run]
"""

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import aiohttp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kalshi_bot.auth import generate_headers
from kalshi_bot.markets import KALSHI_API_BASE

CACHE_FILE = Path("data/backtest/kalshi_markets_cache.json")

# All series from build_kalshi_no_training_data.py SERIES_MAP
ALL_SERIES = [
    "KXHIGHLAX", "KXHIGHDEN", "KXHIGHCHI", "KXHIGHNY",  "KXHIGHMIA",
    "KXHIGHDAL", "KXHIGHBOS", "KXHIGHAUS", "KXHIGHOU",  "KXHIGHTSFO",
    "KXHIGHTSEA","KXHIGHTBOS","KXHIGHTPHX","KXHIGHTPHIL","KXHIGHTDC",
    "KXHIGHTLV", "KXHIGHTOKC","KXHIGHTDAL","KXHIGHTHOU", "KXHIGHTNOLA",
    "KXHIGHTATL","KXHIGHTMIN","KXHIGHTDFW","KXHIGHTSATX",
    "KXHIGHPHIL",                           # actual Philly high series
    "KXLOWTLAX", "KXLOWTDEN", "KXLOWTCHI", "KXLOWTNYC", "KXLOWTMIA",
    "KXLOWTAUS", "KXLOWTBOS", "KXLOWTHOU", "KXLOWTDFW", "KXLOWTSFO",
    "KXLOWTSEA", "KXLOWTPHX", "KXLOWTPHIL","KXLOWTATL", "KXLOWTMIN",
    "KXLOWTDC",  "KXLOWTLV",  "KXLOWTOKC", "KXLOWTSATX","KXLOWTNOLA",
    "KXLOWTDAL",                            # Dallas low-temp
]


def parse_ticker_date(ticker: str) -> str | None:
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})-", ticker)
    if not m:
        return None
    try:
        dt = datetime.strptime(f"20{m.group(1)}{m.group(2)}{m.group(3)}", "%Y%b%d")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return None


async def fetch_series(session: aiohttp.ClientSession, series: str) -> list[dict]:
    markets, cursor = [], None
    while True:
        params = {"status": "settled", "limit": 200, "series_ticker": series}
        if cursor:
            params["cursor"] = cursor
        headers = generate_headers("GET", "/trade-api/v2/markets")
        try:
            async with session.get(
                f"{KALSHI_API_BASE}/markets", headers=headers, params=params,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 429:
                    await asyncio.sleep(5)
                    continue
                resp.raise_for_status()
                data = await resp.json()
        except Exception as e:
            print(f"    error: {e}")
            break
        batch = [m for m in data.get("markets", []) if "-B" in m.get("ticker", "")]
        markets.extend(batch)
        cursor = data.get("cursor")
        if not cursor or not data.get("markets"):
            break
        await asyncio.sleep(0.3)
    return markets


async def fetch_all(series_list: list[str]) -> dict[str, list[dict]]:
    result = {}
    async with aiohttp.ClientSession() as session:
        for series in series_list:
            markets = await fetch_series(session, series)
            result[series] = markets
    return result


def merge_cache(existing: dict, fresh: dict) -> tuple[dict, dict]:
    """
    Merge fresh API data into existing cache.
    Returns (merged_cache, stats_per_series).
    """
    merged = {}
    stats = {}

    all_series = sorted(set(existing) | set(fresh))
    for series in all_series:
        old_markets = existing.get(series, [])
        new_markets = fresh.get(series, [])

        # Build lookup by ticker
        by_ticker: dict[str, dict] = {m["ticker"]: m for m in old_markets}
        added = 0
        for m in new_markets:
            t = m.get("ticker", "")
            if t and t not in by_ticker:
                by_ticker[t] = m
                added += 1

        merged_list = sorted(by_ticker.values(), key=lambda m: m.get("ticker", ""))
        merged[series] = merged_list

        dates = sorted(
            d for d in (parse_ticker_date(m["ticker"]) for m in merged_list) if d
        )
        stats[series] = {
            "before": len(old_markets),
            "after":  len(merged_list),
            "added":  added,
            "range":  f"{dates[0]} → {dates[-1]}" if dates else "no dates",
        }

    return merged, stats


def main(dry_run: bool) -> None:
    # Load existing cache
    if not CACHE_FILE.exists():
        print("No existing cache found — will create from scratch.")
        existing = {}
    else:
        existing = json.loads(CACHE_FILE.read_text())
        total_old = sum(len(v) for v in existing.values() if isinstance(v, list))
        print(f"Loaded existing cache: {len(existing)} series, {total_old:,} markets")

        # Show current coverage
        all_old_dates = []
        for markets in existing.values():
            if not isinstance(markets, list): continue
            for m in markets:
                d = parse_ticker_date(m.get("ticker", ""))
                if d:
                    all_old_dates.append(d)
        if all_old_dates:
            all_old_dates.sort()
            print(f"  Date range: {all_old_dates[0]} → {all_old_dates[-1]}")

    # Fetch fresh data
    print(f"\nFetching fresh settled markets for {len(ALL_SERIES)} series...")
    fresh = asyncio.run(fetch_all(ALL_SERIES))
    total_fresh = sum(len(v) for v in fresh.values())
    fresh_dates = sorted(
        d for markets in fresh.values()
        for m in markets
        for d in [parse_ticker_date(m.get("ticker", ""))] if d
    )
    print(f"Fetched {total_fresh:,} markets from API")
    if fresh_dates:
        print(f"  API date range: {fresh_dates[0]} → {fresh_dates[-1]}")

    # Merge
    merged, stats = merge_cache(existing, fresh)
    total_merged = sum(len(v) for v in merged.values())
    total_added  = sum(s["added"] for s in stats.values())

    print(f"\nMerge result: {total_merged:,} total markets (+{total_added} new)")

    # Show per-series changes
    changed = {s: v for s, v in stats.items() if v["added"] > 0}
    if changed:
        print("\nSeries with new markets:")
        for s, v in sorted(changed.items()):
            print(f"  {s:<20} {v['before']:>4} → {v['after']:>4}  (+{v['added']:>3})  {v['range']}")
    else:
        print("No new markets found.")

    # Show preserved early markets
    early = [
        (s, v) for s, v in stats.items()
        if v.get("range", "").startswith("2026-03")
    ]
    if early:
        print(f"\nPreserved early (pre-Apr 6) series: {len(early)}")
        for s, v in sorted(early)[:5]:
            print(f"  {s}: {v['range']}")

    if dry_run:
        print("\n[dry-run] No files written.")
        return

    # Backup existing cache
    if CACHE_FILE.exists():
        backup = CACHE_FILE.with_suffix(".json.bak")
        shutil.copy2(CACHE_FILE, backup)
        print(f"\nBacked up existing cache → {backup}")

    CACHE_FILE.write_text(json.dumps(merged))
    print(f"Saved merged cache → {CACHE_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch and report without writing anything")
    args = parser.parse_args()
    main(args.dry_run)
