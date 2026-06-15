"""Discover long-dated Kalshi markets in economics and financials categories.

Queries the Kalshi API two ways:
  1. /events endpoint — filtered by category keywords (economics, financials)
  2. /markets pagination — all open markets, filtered by days-to-close

Prints a grouped summary of series we don't currently trade, with:
  - Series ticker, market count, expiry range, avg spread, avg bid
  - Sample title so you can understand what each series is

Run:
  venv/bin/python scripts/discover_longdated_markets.py
  venv/bin/python scripts/discover_longdated_markets.py --min-days 60
  venv/bin/python scripts/discover_longdated_markets.py --all-categories
"""

import argparse
import asyncio
import json
import sys
import os
from collections import defaultdict
from datetime import datetime, timezone, timedelta

import aiohttp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kalshi_bot.markets import KALSHI_API_BASE, _normalize_market
from kalshi_bot.auth import generate_headers

# Series we already trade / track — excluded from "new discoveries" output
KNOWN_SERIES: set[str] = {
    # Weather
    "KXHIGHLAX", "KXHIGHDEN", "KXHIGHCHI", "KXHIGHNY", "KXHIGHMIA",
    "KXHIGHDAL", "KXHIGHBOS", "KXHIGHAUS", "KXHIGHOU",
    "KXHIGHTSFO", "KXHIGHTSEA", "KXHIGHTBOS", "KXHIGHTPHX", "KXHIGHPHIL",
    "KXHIGHTATL", "KXHIGHTMIN", "KXHIGHTDC", "KXHIGHTLV", "KXHIGHTOKC",
    "KXHIGHTDAL", "KXHIGHTSATX", "KXHIGHTHOU", "KXHIGHTNOLA",
    "KXLOWTLAX", "KXLOWTDEN", "KXLOWTCHI", "KXLOWTNYC", "KXLOWTMIA",
    "KXLOWTAUS", "KXLOWTBOS", "KXLOWTHOU", "KXLOWTDFW",
    "KXLOWTSFO", "KXLOWTSEA", "KXLOWTPHX", "KXLOWTPHIL",
    "KXLOWTATL", "KXLOWTMIN", "KXLOWTDC", "KXLOWTLV",
    "KXLOWTOKC", "KXLOWTSATX", "KXLOWTNOLA",
    # Crypto
    "KXBTCD", "KXBTC15M", "KXETH15M", "KXSOL15M", "KXXRP15M",
    "KXDOGE15M", "KXDOGE", "KXADA15M", "KXADA",
    "KXAVAX15M", "KXAVAX", "KXLINK15M", "KXLINK", "KXBNB15M", "KXBNB",
    # Forex
    "KXEURUSD", "KXUSDJPY", "KXGBPUSD",
    # Economics
    "KXCPI", "KXNFP", "KXADP", "KXUNRATE", "KXPPI", "KXPCE",
    "KXJOBLESS", "KXICSA", "KXISM", "KXISMMFG", "KXISMSVC", "KXGDP",
    # Rates / Fed
    "KXFED", "KXFFR", "KXDGS10", "KXDGS2", "KXFEDDEC", "KXFOMCRAT", "KXCPIREL",
    # Energy
    "KXWTI", "KXOIL", "KXNATGAS", "KXNG",
    # Equity (intraday)
    "KXSPX", "KXSPXD", "KXNDX", "KXINXD", "KXDOW",
    # Sports / entertainment / politics (all text-matched, not numeric)
    "KXNBAGAME", "KXNBA", "KXNHL", "KXMLB", "KXNCAAMB",
    "KXMLBHRR", "KXNHLCHMP", "KXNBACHMP", "KXF1",
    "KXTOPSONG", "KXTOP10BIL", "KXOSCARS", "KXOSCARN", "KXRT",
    "KXTRUMPS", "KXTRUMPM", "KXTRUMPA", "KXTRUTHS", "KXPRESME",
    "KXCABOUT", "KXCONGRE", "KXWHVISI", "KXSCOTUS",
    "KXGOVTSH", "KXGOVTFU", "KXHORMUZ",
    "KXDENMAR", "KXJPNPM-", "KXJAPANH", "KXHOCHUL",
    # Esports (KXMVE*)
}

# Ticker prefixes that are definitely sports/entertainment — skip in market scan
SKIP_PREFIXES: tuple[str, ...] = (
    "KXMVE", "KXNBA", "KXNHL", "KXMLB", "KXNFL", "KXNCAA",
    "KXF1", "KXNASCAR", "KXPGA", "KXNBAGAME",
    "KXTOP", "KXOSCARS", "KXRT", "KXBOX",
    "KXHIGH", "KXLOWT",
    "KXBTC", "KXETH", "KXSOL", "KXDOGE", "KXADA", "KXAVAX", "KXLINK", "KXBNB", "KXXRP",
    "KXSPX", "KXSPXD", "KXNDX", "KXINXD", "KXDOW",
    "KXEURUSD", "KXUSDJPY", "KXGBPUSD",
)

# Categories to search via the events endpoint
ECONOMICS_CATEGORIES = [
    "financials", "economics", "energy", "rates", "housing",
    "commodities", "economic", "finance",
]


def _parse_dt(s: str) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


async def fetch_events_by_category(
    session: aiohttp.ClientSession,
    category: str,
) -> list[dict]:
    """Query the /events endpoint for a given category."""
    path = "/trade-api/v2/events"
    headers = generate_headers("GET", path)
    params = {"status": "open", "limit": 200, "category": category}
    try:
        async with session.get(
            f"{KALSHI_API_BASE}/events",
            params=params,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=20),
        ) as resp:
            if resp.status == 404:
                return []
            resp.raise_for_status()
            data = await resp.json()
            return data.get("events", [])
    except Exception as exc:
        print(f"  [events/{category}] error: {exc}", file=sys.stderr)
        return []


async def paginate_markets(
    session: aiohttp.ClientSession,
    cutoff: datetime,
    max_markets: int = 15000,
) -> list[dict]:
    """Paginate all open markets, returning those with close_time > cutoff."""
    results = []
    cursor = None
    fetched = 0

    while fetched < max_markets:
        path = "/trade-api/v2/markets"
        headers = generate_headers("GET", path)
        params: dict = {"status": "open", "limit": 100}
        if cursor:
            params["cursor"] = cursor

        try:
            async with session.get(
                f"{KALSHI_API_BASE}/markets",
                params=params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 429:
                    await asyncio.sleep(5)
                    continue
                resp.raise_for_status()
                data = await resp.json()
        except Exception as exc:
            print(f"  [paginate] error: {exc}", file=sys.stderr)
            break

        page = data.get("markets", [])
        fetched += len(page)

        for m in page:
            ticker = m.get("ticker", "")
            series = ticker.split("-")[0]

            if any(series.startswith(p) for p in SKIP_PREFIXES):
                continue

            close_dt = _parse_dt(m.get("close_time") or m.get("expiration_time", ""))
            if close_dt and close_dt >= cutoff:
                results.append(_normalize_market(m))

        cursor = data.get("cursor")
        if not cursor or not page:
            break

        if fetched % 1000 == 0:
            print(f"  paginated {fetched} markets so far...", file=sys.stderr)

        await asyncio.sleep(0.15)

    print(f"  total paginated: {fetched} markets", file=sys.stderr)
    return results


def group_and_print(
    markets: list[dict],
    cutoff: datetime,
    show_known: bool = False,
    label: str = "Markets",
) -> None:
    now = datetime.now(timezone.utc)
    by_series: dict[str, list[dict]] = defaultdict(list)

    for m in markets:
        ticker = m.get("ticker", "")
        series = ticker.split("-")[0]
        close_dt = _parse_dt(m.get("close_time") or m.get("expiration_time", ""))
        if close_dt and close_dt >= cutoff:
            by_series[series].append(m)

    print(f"\n{'='*90}")
    print(f"  {label}  (close >= {cutoff.date()})  —  {len(by_series)} series, {sum(len(v) for v in by_series.values())} markets")
    print(f"{'='*90}")

    header = f"  {'Series':<22} {'n':>4}  {'Expiry range':<23}  {'Bid avg':>7}  {'Spread avg':>10}  Title"
    print(header)
    print("  " + "-" * 88)

    for series in sorted(by_series, key=lambda s: -len(by_series[s])):
        is_known = series in KNOWN_SERIES
        if is_known and not show_known:
            continue

        ms = by_series[series]
        close_dates = []
        bids, spreads = [], []

        for m in ms:
            dt = _parse_dt(m.get("close_time") or m.get("expiration_time", ""))
            if dt:
                close_dates.append(dt.date().isoformat())
            b = m.get("yes_bid")
            a = m.get("yes_ask")
            if b is not None:
                bids.append(b)
            if b is not None and a is not None:
                spreads.append(a - b)

        close_range = f"{min(close_dates)}..{max(close_dates)}" if close_dates else "?"
        bid_avg = f"{sum(bids)/len(bids):.0f}¢" if bids else "?"
        spread_avg = f"{sum(spreads)/len(spreads):.1f}¢" if spreads else "?"

        sample_title = ms[0].get("title", ms[0].get("subtitle", ""))[:48]
        flag = " [KNOWN]" if is_known else ""

        print(f"  {series:<22} {len(ms):>4}  {close_range:<23}  {bid_avg:>7}  {spread_avg:>10}  {sample_title}{flag}")


async def main(min_days: int, all_categories: bool, show_known: bool) -> None:
    cutoff = datetime.now(timezone.utc) + timedelta(days=min_days)
    print(f"Discovering long-dated Kalshi markets (close >= {cutoff.date()}, i.e. >{min_days}d out)")

    async with aiohttp.ClientSession() as session:

        # --- 1. Events API by category ---
        print("\n[1] Querying /events by category...", file=sys.stderr)
        event_markets: list[dict] = []
        seen_event_tickers: set[str] = set()
        categories = ECONOMICS_CATEGORIES if not all_categories else [
            "financials", "economics", "energy", "rates", "housing",
            "commodities", "politics", "sports", "entertainment",
        ]
        for cat in categories:
            events = await fetch_events_by_category(session, cat)
            print(f"  /events?category={cat} → {len(events)} events", file=sys.stderr)
            for ev in events:
                # Each event has a list of markets embedded or just a ticker
                for m in ev.get("markets", []):
                    t = m.get("ticker", "")
                    if t and t not in seen_event_tickers:
                        seen_event_tickers.add(t)
                        event_markets.append(_normalize_market(m))
            await asyncio.sleep(0.3)

        if event_markets:
            group_and_print(event_markets, cutoff, show_known=show_known, label="Via /events API")
        else:
            print("\n[events] No markets returned from /events endpoint (may not be supported or markets not embedded).")

        # --- 2. Market pagination ---
        print("\n[2] Paginating /markets for long-dated markets...", file=sys.stderr)
        paged_markets = await paginate_markets(session, cutoff)
        group_and_print(paged_markets, cutoff, show_known=show_known, label="Via /markets pagination (non-sports, non-weather)")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-days", type=int, default=30, help="Minimum days to close (default: 30)")
    parser.add_argument("--all-categories", action="store_true", help="Include sports/politics in events query")
    parser.add_argument("--show-known", action="store_true", help="Include series we already trade in output")
    args = parser.parse_args()
    asyncio.run(main(args.min_days, args.all_categories, args.show_known))
