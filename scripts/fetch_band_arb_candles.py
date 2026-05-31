#!/usr/bin/env python3
"""Fetch Kalshi candlestick data for all KXHIGH tickers (B-bands and T-bands).

Fills gaps in data/band_arb_candle_cache.json so the backtest has price data
for the full Feb–May 2026 window, covering both between-bands and terminal bands.

Fetches hourly candles (period_interval=60) for each ticker's trading day
using the same window as backtest_band_arb_yes_exits.py:
  start = (settlement_date - 1 day) at 12:00 UTC
  end   = settlement_date at 07:00 UTC

Skips tickers already in the cache with non-empty candle data.
Saves progress every 100 tickers so interruptions don't lose work.

Usage:
  venv/bin/python scripts/fetch_band_arb_candles.py
  venv/bin/python scripts/fetch_band_arb_candles.py --from-date 2026-04-01  # skip old tickers
  venv/bin/python scripts/fetch_band_arb_candles.py --refetch-empty  # retry failed/empty
  venv/bin/python scripts/fetch_band_arb_candles.py --dry-run        # count only
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).parent.parent))

from kalshi_bot.auth import generate_headers      # noqa: E402
from kalshi_bot.markets import KALSHI_API_BASE    # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR   = Path(__file__).parent.parent / "data"
CACHE_FILE = DATA_DIR / "band_arb_candle_cache.json"
BANDS_CSV  = DATA_DIR / "kxhigh_bands.csv"

_SEM            = asyncio.Semaphore(4)   # conservative: 4 concurrent requests
SAVE_INTERVAL   = 100                    # save cache every N fetches
SLEEP_PER_REQ   = 0.25                  # seconds between requests
SLEEP_ON_429    = 5.0                   # back-off on rate limit


def load_bands(from_date: str | None = None) -> list[dict]:
    if not BANDS_CSV.exists():
        log.error("kxhigh_bands.csv not found — run build_kxhigh_bands.py first")
        sys.exit(1)
    rows = []
    with BANDS_CSV.open(newline="") as f:
        for r in csv.DictReader(f):
            if from_date and r["date"] < from_date:
                continue
            rows.append({"ticker": r["ticker"], "date": r["date"]})
    return rows


def load_cache() -> dict[str, list[dict]]:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}


def save_cache(cache: dict[str, list[dict]]) -> None:
    CACHE_FILE.write_text(json.dumps(cache))
    log.info("Cache saved (%d tickers, %d non-empty)",
             len(cache), sum(1 for v in cache.values() if v))


async def fetch_candles(
    session: aiohttp.ClientSession,
    ticker: str,
    market_date_str: str,
) -> list[dict]:
    """Fetch hourly candlesticks for the trading day preceding settlement."""
    mkt_date = datetime.strptime(market_date_str, "%Y-%m-%d").date()
    prev_day = mkt_date - timedelta(days=1)
    start_ts = int(datetime(prev_day.year, prev_day.month, prev_day.day,
                            12, 0, tzinfo=timezone.utc).timestamp())
    end_ts   = int(datetime(mkt_date.year, mkt_date.month, mkt_date.day,
                            7, 0, tzinfo=timezone.utc).timestamp())

    series = ticker.rsplit("-", 2)[0]
    path   = f"/trade-api/v2/series/{series}/markets/{ticker}/candlesticks"
    headers = generate_headers("GET", path)
    params  = {"period_interval": 60, "start_ts": start_ts, "end_ts": end_ts}

    async with _SEM:
        for attempt in range(3):
            try:
                async with session.get(
                    f"{KALSHI_API_BASE}/series/{series}/markets/{ticker}/candlesticks",
                    params=params, headers=headers,
                    timeout=aiohttp.ClientTimeout(total=20),
                ) as r:
                    if r.status == 429:
                        log.warning("Rate-limited on %s (attempt %d) — sleeping %.1fs",
                                    ticker, attempt + 1, SLEEP_ON_429)
                        await asyncio.sleep(SLEEP_ON_429)
                        continue
                    if r.status == 404:
                        return []
                    if r.status != 200:
                        log.debug("HTTP %d for %s", r.status, ticker)
                        return []
                    data = await r.json()
                    await asyncio.sleep(SLEEP_PER_REQ)
                    return data.get("candlesticks", [])
            except Exception as exc:
                log.debug("Candle fetch error %s: %s", ticker, exc)
                await asyncio.sleep(1.0)
        return []


async def run(args: argparse.Namespace) -> None:
    bands = load_bands(from_date=args.from_date)
    cache = load_cache()

    log.info("Loaded %d bands, %d cached (%d non-empty)",
             len(bands), len(cache),
             sum(1 for v in cache.values() if v))

    # Determine which tickers need fetching
    to_fetch = []
    for b in bands:
        ticker = b["ticker"]
        existing = cache.get(ticker)
        if existing is None:
            to_fetch.append(b)          # not in cache at all
        elif not existing and args.refetch_empty:
            to_fetch.append(b)          # in cache but empty, and user wants retry

    log.info("%d tickers to fetch%s",
             len(to_fetch),
             " (including empty retries)" if args.refetch_empty else "")

    if args.dry_run:
        log.info("Dry-run: exiting without fetching")
        return

    if not to_fetch:
        log.info("Nothing to fetch — cache is complete")
        return

    fetched = 0
    got_data = 0

    async with aiohttp.ClientSession() as session:
        # Process in small concurrent batches
        batch_size = 8
        for i in range(0, len(to_fetch), batch_size):
            batch = to_fetch[i:i + batch_size]
            tasks = [fetch_candles(session, b["ticker"], b["date"]) for b in batch]
            results = await asyncio.gather(*tasks)

            for b, candles in zip(batch, results):
                cache[b["ticker"]] = candles
                if candles:
                    got_data += 1
                fetched += 1

            if fetched % SAVE_INTERVAL == 0 or i + batch_size >= len(to_fetch):
                log.info("Progress: %d/%d fetched, %d with data",
                         fetched, len(to_fetch), got_data)
                save_cache(cache)

    log.info("Done. Fetched %d tickers, %d had candle data", fetched, got_data)
    save_cache(cache)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-date", metavar="YYYY-MM-DD", default=None,
                        help="Skip tickers with settlement date before this date "
                             "(use to avoid wasting time on tickers beyond the API candle window)")
    parser.add_argument("--refetch-empty", action="store_true",
                        help="Re-fetch tickers already in cache but with no candle data")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count what would be fetched without making API calls")
    args = parser.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
