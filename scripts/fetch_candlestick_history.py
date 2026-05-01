"""Fetch historical 1-minute candlestick data from Kalshi for settled markets.

Downloads OHLC candlestick data for settled markets in specified series and
stores them in a local SQLite database for use by backtest_momentum.py.

The Kalshi historical candlestick endpoint:
  GET /historical/markets/{ticker}/candlesticks
  Params: start_ts, end_ts (Unix seconds), period_interval (1 | 60 | 1440)

All price values from the API are FixedPointDollars strings ("0.5600" = 56¢).
This script converts to integer cents for storage.

Usage
-----
  # Fetch 7 days of Chicago high-temp and BTC markets (test run)
  venv/bin/python scripts/fetch_candlestick_history.py \\
      --series KXHIGHTCHI KXBTCD --days 7 --db data/candlesticks_test.db

  # Full fetch: all temperature + crypto series, 60 days
  venv/bin/python scripts/fetch_candlestick_history.py --days 60

  # Resume a partial fetch without re-downloading already-stored tickers
  venv/bin/python scripts/fetch_candlestick_history.py --days 60 --resume
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sqlite3
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).parent.parent))

from kalshi_bot.auth import generate_headers          # noqa: E402
from kalshi_bot.markets import (                      # noqa: E402
    KALSHI_API_BASE,
    NUMERIC_SERIES,
    _normalize_market,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_MARKETS_PATH = "/trade-api/v2/markets"
_HIST_BASE = "https://api.elections.kalshi.com/trade-api/v2"
# Candlestick fetches are sequential (semaphore=1) — the historical endpoint
# rate-limits very aggressively on concurrent or rapid requests.
_SEMAPHORE = asyncio.Semaphore(1)
# Base delay between candlestick requests (seconds). Increase if 429s persist.
_CANDLE_DELAY = 2.0
# Minimum age (days) for a market to have historical candlesticks available.
# Very recent markets (< 3 days old) return 404 from the historical endpoint.
_MIN_MARKET_AGE_DAYS = 3

# Default series to fetch if --series not specified.
# Covers temperature (most retail activity) and key crypto series.
_DEFAULT_SERIES: tuple[str, ...] = (
    # High temperature
    "KXHIGHTATL", "KXHIGHTCHI", "KXHIGHTNYC", "KXHIGHTBOS", "KXHIGHTMIN",
    "KXHIGHTSEA", "KXHIGHTSFO", "KXHIGHTDAL", "KXHIGHTPHX", "KXHIGHTDC",
    "KXHIGHLAX",  "KXHIGHDEN",  "KXHIGHCHI",  "KXHIGHNY",   "KXHIGHMIA",
    # Low temperature
    "KXLOWTCHI", "KXLOWTNYC", "KXLOWTBOS", "KXLOWTMIN", "KXLOWTATL",
    # Crypto (high retail activity, lots of spikes)
    "KXBTCD", "KXBTC15M", "KXETH15M", "KXSOL15M",
)

# Minimum candles required to bother storing a market (too-short = not useful)
_MIN_CANDLES = 30


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

def init_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("""
        CREATE TABLE IF NOT EXISTS markets (
            ticker      TEXT PRIMARY KEY,
            series      TEXT,
            open_ts     INTEGER,
            close_ts    INTEGER,
            result      TEXT,
            final_price INTEGER
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS candles (
            ticker      TEXT    NOT NULL,
            period_ts   INTEGER NOT NULL,
            bid_open    INTEGER,
            bid_close   INTEGER,
            bid_low     INTEGER,
            bid_high    INTEGER,
            ask_open    INTEGER,
            ask_close   INTEGER,
            price_close INTEGER,
            volume      REAL,
            PRIMARY KEY (ticker, period_ts)
        )
    """)
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_candles_ticker ON candles(ticker, period_ts)"
    )
    con.commit()
    return con


def _existing_tickers(con: sqlite3.Connection) -> set[str]:
    return {r[0] for r in con.execute("SELECT ticker FROM markets")}


# ---------------------------------------------------------------------------
# Price conversion helpers
# ---------------------------------------------------------------------------

def _fp_to_cents(val: str | float | int | None) -> int | None:
    """Convert FixedPointDollars string ("0.5600") to integer cents (56)."""
    if val is None:
        return None
    try:
        return round(float(val) * 100)
    except (ValueError, TypeError):
        return None


def _parse_candle(c: dict) -> dict | None:
    """Extract fields from one candlestick API object into a flat dict."""
    period_ts = c.get("end_period_ts")
    if period_ts is None:
        return None

    bid = c.get("yes_bid") or {}
    ask = c.get("yes_ask") or {}
    price = c.get("price") or {}

    return {
        "period_ts":   int(period_ts),
        "bid_open":    _fp_to_cents(bid.get("open_dollars")),
        "bid_close":   _fp_to_cents(bid.get("close_dollars")),
        "bid_low":     _fp_to_cents(bid.get("low_dollars")),
        "bid_high":    _fp_to_cents(bid.get("high_dollars")),
        "ask_open":    _fp_to_cents(ask.get("open_dollars")),
        "ask_close":   _fp_to_cents(ask.get("close_dollars")),
        "price_close": _fp_to_cents(price.get("close_dollars")),
        "volume":      _safe_float(c.get("volume_fp") or c.get("volume")),
    }


def _safe_float(v) -> float | None:
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _market_close_ts(mkt: dict) -> int:
    """Return the market close time as a Unix timestamp, or 0 on failure."""
    close_str = mkt.get("close_time") or mkt.get("expiration_time", "")
    try:
        return int(datetime.fromisoformat(close_str.replace("Z", "+00:00")).timestamp())
    except (ValueError, AttributeError):
        return 0


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------

async def _fetch_settled_markets(
    session: aiohttp.ClientSession,
    series: str,
    cutoff_ts: int,
) -> list[dict]:
    """Return settled markets for one series settled after cutoff_ts."""
    params = {"status": "settled", "limit": 100, "series_ticker": series}
    path = _MARKETS_PATH
    headers = generate_headers("GET", path)
    async with _SEMAPHORE:
        try:
            async with session.get(
                f"{KALSHI_API_BASE}/markets",
                params=params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=20),
            ) as resp:
                if resp.status == 429:
                    log.warning("Rate-limited fetching %s — sleeping 5s", series)
                    await asyncio.sleep(5.0)
                    return []
                resp.raise_for_status()
                data = await resp.json()
        except Exception as exc:
            log.warning("Error fetching settled markets for %s: %s", series, exc)
            return []

    markets = []
    for m in data.get("markets", []):
        m = _normalize_market(m)
        # Filter to markets that closed after cutoff
        close_str = m.get("close_time") or m.get("expiration_time", "")
        try:
            close_ts = int(
                datetime.fromisoformat(
                    close_str.replace("Z", "+00:00")
                ).timestamp()
            )
        except (ValueError, AttributeError):
            continue
        if close_ts < cutoff_ts:
            continue
        result = m.get("result", "")
        if result not in ("yes", "no"):
            continue
        markets.append(m)

    await asyncio.sleep(0.3)
    log.info("  %s: %d settled markets (since cutoff)", series, len(markets))
    return markets


async def _fetch_candles(
    session: aiohttp.ClientSession,
    series: str,
    ticker: str,
    open_ts: int,
    close_ts: int,
    period: int,
) -> list[dict]:
    """Fetch candlestick data for one market via the series-scoped endpoint.

    Correct endpoint: GET /series/{series}/markets/{ticker}/candlesticks
    (The /historical/markets/{ticker}/candlesticks path returns 404.)
    """
    path = f"/trade-api/v2/series/{series}/markets/{ticker}/candlesticks"
    url = f"{_HIST_BASE}/series/{series}/markets/{ticker}/candlesticks"
    params = {
        "start_ts": open_ts,
        "end_ts":   close_ts,
        "period_interval": period,
    }

    async with _SEMAPHORE:
        headers = generate_headers("GET", path)
        try:
            async with session.get(
                url,
                params=params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 429:
                    wait = _CANDLE_DELAY * 10
                    log.warning("Rate-limited on %s — sleeping %.0fs", ticker, wait)
                    await asyncio.sleep(wait)
                    # Retry once
                    headers = generate_headers("GET", path)
                    async with session.get(
                        url, params=params, headers=headers,
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as resp2:
                        resp2.raise_for_status()
                        data = await resp2.json()
                elif resp.status in (404, 400):
                    log.debug("Skip %s (%d): %s", ticker, resp.status, await resp.text())
                    return []
                else:
                    resp.raise_for_status()
                    data = await resp.json()
        except Exception as exc:
            log.warning("Error fetching candles for %s: %s", ticker, exc)
            return []

    raw = data.get("candlesticks", [])
    candles = [_parse_candle(c) for c in raw]
    candles = [c for c in candles if c is not None]
    await asyncio.sleep(_CANDLE_DELAY)
    return candles


# ---------------------------------------------------------------------------
# Main fetch-and-store loop
# ---------------------------------------------------------------------------

async def fetch_series(
    session: aiohttp.ClientSession,
    con: sqlite3.Connection,
    series: str,
    cutoff_ts: int,
    period: int,
    resume: bool,
) -> tuple[int, int]:
    """Fetch and store one series. Returns (markets_stored, candles_stored)."""
    existing = _existing_tickers(con) if resume else set()
    markets = await _fetch_settled_markets(session, series, cutoff_ts)

    # Drop markets that closed too recently — the historical candlestick endpoint
    # returns 404 for markets settled in the last _MIN_MARKET_AGE_DAYS days.
    now_ts = int(datetime.now(timezone.utc).timestamp())
    min_age_ts = now_ts - _MIN_MARKET_AGE_DAYS * 86400
    markets = [m for m in markets if _market_close_ts(m) < min_age_ts]
    log.info("  %d markets after age filter (>%dd old)", len(markets), _MIN_MARKET_AGE_DAYS)

    markets_stored = 0
    candles_stored = 0

    for mkt in markets:
        ticker = mkt.get("ticker", "")
        if not ticker:
            continue
        if resume and ticker in existing:
            log.debug("  Skip (already stored): %s", ticker)
            continue

        # Parse open/close timestamps
        open_str  = mkt.get("open_time")  or mkt.get("created_time", "")
        close_str = mkt.get("close_time") or mkt.get("expiration_time", "")
        try:
            open_ts  = int(datetime.fromisoformat(open_str.replace("Z",  "+00:00")).timestamp())
            close_ts = int(datetime.fromisoformat(close_str.replace("Z", "+00:00")).timestamp())
        except (ValueError, AttributeError):
            log.debug("  Skip (bad timestamps): %s", ticker)
            continue

        if close_ts <= open_ts:
            continue

        result = mkt.get("result", "")
        final_price = _fp_to_cents(
            mkt.get("last_price") or mkt.get("yes_bid_dollars")
        )

        candles = await _fetch_candles(session, series, ticker, open_ts, close_ts, period)

        if len(candles) < _MIN_CANDLES:
            log.debug("  Skip (only %d candles): %s", len(candles), ticker)
            continue

        # Store market record
        con.execute(
            "INSERT OR REPLACE INTO markets(ticker,series,open_ts,close_ts,result,final_price)"
            " VALUES(?,?,?,?,?,?)",
            (ticker, series, open_ts, close_ts, result, final_price),
        )

        # Store candles
        con.executemany(
            "INSERT OR IGNORE INTO candles"
            "(ticker,period_ts,bid_open,bid_close,bid_low,bid_high,"
            " ask_open,ask_close,price_close,volume)"
            " VALUES(?,?,?,?,?,?,?,?,?,?)",
            [
                (ticker, c["period_ts"], c["bid_open"], c["bid_close"],
                 c["bid_low"], c["bid_high"], c["ask_open"], c["ask_close"],
                 c["price_close"], c["volume"])
                for c in candles
            ],
        )
        con.commit()

        markets_stored += 1
        candles_stored += len(candles)
        log.info("  Stored %s: %d candles  result=%s", ticker, len(candles), result)

    return markets_stored, candles_stored


async def main(args: argparse.Namespace) -> None:
    db_path = Path(args.db)
    con = init_db(db_path)

    cutoff_dt = datetime.now(timezone.utc) - timedelta(days=args.days)
    cutoff_ts = int(cutoff_dt.timestamp())
    log.info(
        "Fetching settled markets since %s (%d days back)",
        cutoff_dt.date().isoformat(), args.days,
    )
    log.info("Series to fetch: %d", len(args.series))
    log.info("Output DB: %s", db_path)
    if args.resume:
        existing = _existing_tickers(con)
        log.info("Resume mode: %d tickers already in DB will be skipped", len(existing))

    total_markets = 0
    total_candles = 0

    async with aiohttp.ClientSession() as session:
        for series in args.series:
            log.info("--- %s ---", series)
            m, c = await fetch_series(
                session, con, series, cutoff_ts, args.period, args.resume
            )
            total_markets += m
            total_candles += c
            log.info("  %s done: %d markets, %d candles stored", series, m, c)

    log.info(
        "Complete. %d markets stored, %d candles stored → %s",
        total_markets, total_candles, db_path,
    )

    # Summary
    n_mkt = con.execute("SELECT COUNT(*) FROM markets").fetchone()[0]
    n_can = con.execute("SELECT COUNT(*) FROM candles").fetchone()[0]
    log.info("DB totals: %d markets, %d candles", n_mkt, n_can)
    con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch Kalshi historical candlestick data for settled markets."
    )
    parser.add_argument(
        "--series", nargs="+", default=list(_DEFAULT_SERIES),
        metavar="SERIES",
        help="Series tickers to fetch (default: all temperature + key crypto series)",
    )
    parser.add_argument(
        "--days", type=int, default=60,
        help="Fetch markets settled in the last N days (default: 60)",
    )
    parser.add_argument(
        "--period", type=int, default=1, choices=[1, 60, 1440],
        help="Candlestick resolution in minutes: 1 | 60 | 1440 (default: 1)",
    )
    parser.add_argument(
        "--db", default="data/candlesticks.db",
        help="Output SQLite database path (default: data/candlesticks.db)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip tickers already present in the database",
    )
    args = parser.parse_args()
    asyncio.run(main(args))
