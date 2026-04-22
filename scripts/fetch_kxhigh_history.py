"""Fetch historical resolved KXHIGH 'between' band market data from Kalshi API.

Downloads all settled KXHIGH/KXHIGHT temperature markets, filters to
direction='between' (interior bands), and records their outcome (yes/no)
along with strike bounds and settlement date.

This provides the ground truth for backtest_band_arb_yes.py: which bands
actually resolved YES vs NO, enabling win-rate analysis by clearance/time.

Output: data/kxhigh_bands.csv
  ticker, metric, date, direction, strike_lo, strike_hi, band_width, result

Usage:
  venv/bin/python scripts/fetch_kxhigh_history.py
  venv/bin/python scripts/fetch_kxhigh_history.py --days 90
  venv/bin/python scripts/fetch_kxhigh_history.py --out data/kxhigh_bands.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).parent.parent))

from kalshi_bot.auth import generate_headers          # noqa: E402
from kalshi_bot.market_parser import parse_market     # noqa: E402
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
_SEMAPHORE = asyncio.Semaphore(5)

# Only temperature series — skip crypto/forex/energy
_TEMP_SERIES: tuple[str, ...] = tuple(
    s for s in NUMERIC_SERIES
    if s.startswith("KXHIGH")
)


async def _fetch_series_settled(
    session: aiohttp.ClientSession,
    series: str,
    limit: int = 200,
) -> list[dict]:
    """Fetch all settled markets for one series."""
    params = {
        "status": "settled",
        "limit":  min(limit, 100),
        "series_ticker": series,
    }
    headers = generate_headers("GET", _MARKETS_PATH)
    async with _SEMAPHORE:
        try:
            async with session.get(
                f"{KALSHI_API_BASE}/markets",
                params=params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=20),
            ) as resp:
                if resp.status == 429:
                    log.warning("Rate-limited on %s — waiting 3s", series)
                    await asyncio.sleep(3.0)
                    return []
                resp.raise_for_status()
                data = await resp.json()
        except Exception as exc:
            log.warning("Fetch error for series %s: %s", series, exc)
            return []

    markets = [_normalize_market(m) for m in data.get("markets", [])]
    log.info("  %s: %d settled markets", series, len(markets))
    await asyncio.sleep(0.25)
    return markets


def _market_to_row(mkt: dict, cutoff_date: str | None) -> dict | None:
    """Parse a settled market dict into a CSV row, or None if not usable."""
    parsed = parse_market(mkt)
    if parsed is None:
        return None
    if parsed.direction != "between":
        return None
    if parsed.strike_lo is None or parsed.strike_hi is None:
        return None
    if not parsed.metric.startswith("temp_high"):
        return None

    result = mkt.get("result", "")
    if result not in ("yes", "no"):
        return None

    # Extract settlement date from close_time or expiration_time
    close_str = mkt.get("close_time") or mkt.get("expiration_time", "")
    try:
        close_dt = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
        date_str = close_dt.astimezone(timezone.utc).date().isoformat()
    except (ValueError, AttributeError):
        return None

    if cutoff_date and date_str < cutoff_date:
        return None

    band_width = round(parsed.strike_hi - parsed.strike_lo, 1)

    return {
        "ticker":     parsed.ticker,
        "metric":     parsed.metric,
        "date":       date_str,
        "direction":  parsed.direction,
        "strike_lo":  parsed.strike_lo,
        "strike_hi":  parsed.strike_hi,
        "band_width": band_width,
        "result":     result,
    }


async def main(days: int, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cutoff_date = (date.today() - timedelta(days=days)).isoformat()
    log.info("Fetching resolved KXHIGH 'between' bands since %s (%d days)",
             cutoff_date, days)
    log.info("Temperature series to fetch: %d", len(_TEMP_SERIES))

    async with aiohttp.ClientSession() as session:
        tasks = [_fetch_series_settled(session, series) for series in _TEMP_SERIES]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    rows: list[dict] = []
    seen_tickers: set[str] = set()
    for series, result in zip(_TEMP_SERIES, results):
        if isinstance(result, Exception):
            log.error("Error for series %s: %s", series, result)
            continue
        for mkt in result:
            ticker = mkt.get("ticker", "")
            if ticker in seen_tickers:
                continue
            seen_tickers.add(ticker)
            row = _market_to_row(mkt, cutoff_date)
            if row:
                rows.append(row)

    rows.sort(key=lambda r: (r["metric"], r["date"], r["strike_lo"]))

    fieldnames = ["ticker", "metric", "date", "direction",
                  "strike_lo", "strike_hi", "band_width", "result"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    yes_count = sum(1 for r in rows if r["result"] == "yes")
    no_count  = sum(1 for r in rows if r["result"] == "no")
    log.info(
        "Wrote %d rows to %s  (YES=%d  NO=%d  overall_yes_rate=%.0f%%)",
        len(rows), out_path, yes_count, no_count,
        yes_count / max(len(rows), 1) * 100,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch resolved KXHIGH 'between' band market history from Kalshi."
    )
    parser.add_argument(
        "--days", type=int, default=60,
        help="How many days back to fetch (default: 60)",
    )
    parser.add_argument(
        "--out", default="data/kxhigh_bands.csv",
        help="Output CSV path (default: data/kxhigh_bands.csv)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.days, Path(args.out)))
