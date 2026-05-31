"""Fetch all settled KXHIGH* markets from Kalshi and write to data/kxhigh_bands.csv.

Includes between-bands (B-suffix, direction="between") and terminal bands
(T-bands: direction="over" for upper threshold, direction="under" for lower threshold).

Pulls historical settled markets for every high-temperature series, parses
direction/strike using market_parser.parse_market(), and merges into the CSV.
Safe to re-run: deduplicates by ticker.

Run:
  venv/bin/python scripts/build_kxhigh_bands.py
  venv/bin/python scripts/build_kxhigh_bands.py --dry-run   # print rows, don't write
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import sys
import time
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).parent.parent))

from kalshi_bot.auth import generate_headers
from kalshi_bot.market_parser import TICKER_TO_METRIC, parse_market

KALSHI_API_BASE = (
    "https://api.elections.kalshi.com/trade-api/v2"
    if os.environ.get("KALSHI_ENVIRONMENT", "production") == "production"
    else "https://demo-api.kalshi.co/trade-api/v2"
)

OUT_CSV = Path(__file__).parent.parent / "data" / "kxhigh_bands.csv"

CSV_FIELDS = ["ticker", "metric", "date", "direction", "strike_lo", "strike_hi", "band_width", "result"]

# All series that resolve to a temp_high_* metric
HIGH_SERIES: list[str] = sorted(
    {prefix for prefix, metric in TICKER_TO_METRIC.items() if metric.startswith("temp_high_")}
)


def _load_existing() -> dict[str, dict]:
    """Load existing CSV → {ticker: row_dict}."""
    if not OUT_CSV.exists():
        return {}
    with OUT_CSV.open(newline="") as f:
        return {r["ticker"]: r for r in csv.DictReader(f)}


async def _fetch_series(
    session: aiohttp.ClientSession,
    series: str,
    *,
    dry_run: bool,
) -> list[dict]:
    """Paginate all settled markets for one series. Returns parsed band rows."""
    rows: list[dict] = []
    cursor: str | None = None
    page_num = 0

    while True:
        params: dict = {"status": "settled", "limit": 100, "series_ticker": series}
        if cursor:
            params["cursor"] = cursor

        headers = generate_headers("GET", "/trade-api/v2/markets")
        for attempt in range(5):
            try:
                async with session.get(
                    f"{KALSHI_API_BASE}/markets",
                    params=params,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status == 429:
                        wait = 2 ** attempt
                        print(f"    [429] waiting {wait}s…", flush=True)
                        await asyncio.sleep(wait)
                        continue
                    resp.raise_for_status()
                    data = await resp.json()
                    break
            except Exception as exc:
                wait = 2 ** attempt
                print(f"    [{type(exc).__name__}] attempt {attempt+1}, waiting {wait}s…", flush=True)
                await asyncio.sleep(wait)
        else:
            print(f"    FAILED after 5 attempts for {series}, stopping.")
            break

        markets = data.get("markets", [])
        page_num += 1

        for m in markets:
            pm = parse_market(m)
            if pm is None or pm.direction not in ("between", "over", "under"):
                continue

            result_raw = m.get("result", "")
            if result_raw not in ("yes", "no"):
                continue

            close_time = m.get("close_time") or m.get("expiration_time") or ""
            close_date = close_time[:10] if close_time else ""
            if not close_date:
                continue

            if pm.direction == "between":
                if pm.strike_lo is None or pm.strike_hi is None:
                    continue
                row = {
                    "ticker":     pm.ticker,
                    "metric":     pm.metric,
                    "date":       close_date,
                    "direction":  "between",
                    "strike_lo":  pm.strike_lo,
                    "strike_hi":  pm.strike_hi,
                    "band_width": round(pm.strike_hi - pm.strike_lo, 2),
                    "result":     result_raw,
                }
            elif pm.direction == "over":
                # Upper T-band: YES if daily high > strike. No ceiling.
                if pm.strike is None:
                    continue
                row = {
                    "ticker":     pm.ticker,
                    "metric":     pm.metric,
                    "date":       close_date,
                    "direction":  "over",
                    "strike_lo":  pm.strike,
                    "strike_hi":  "",
                    "band_width": "",
                    "result":     result_raw,
                }
            else:
                # Under T-band: YES if daily high < strike. No floor.
                if pm.strike is None:
                    continue
                row = {
                    "ticker":     pm.ticker,
                    "metric":     pm.metric,
                    "date":       close_date,
                    "direction":  "under",
                    "strike_lo":  "",
                    "strike_hi":  pm.strike,
                    "band_width": "",
                    "result":     result_raw,
                }
            rows.append(row)

        cursor = data.get("cursor")
        if not cursor or not markets:
            break

        await asyncio.sleep(0.25)

    return rows


async def main(dry_run: bool) -> None:
    existing = _load_existing()
    print(f"Existing rows in CSV: {len(existing):,}")
    print(f"Series to fetch: {len(HIGH_SERIES)}  ({', '.join(HIGH_SERIES)})\n")

    new_rows: list[dict] = []

    async with aiohttp.ClientSession() as session:
        for series in HIGH_SERIES:
            t0 = time.monotonic()
            rows = await _fetch_series(session, series, dry_run=dry_run)
            fresh = [r for r in rows if r["ticker"] not in existing]
            new_rows.extend(fresh)
            elapsed = time.monotonic() - t0
            print(f"  {series:<16} {len(rows):>5} settled  {len(fresh):>5} new  ({elapsed:.1f}s)")
            await asyncio.sleep(0.5)

    total_new = len(new_rows)
    print(f"\nNew rows to add: {total_new:,}")

    if dry_run:
        for r in new_rows[:10]:
            print(" ", r)
        if total_new > 10:
            print(f"  … {total_new-10} more")
        return

    if total_new == 0:
        print("Nothing to add — CSV is up to date.")
        return

    merged = {**existing, **{r["ticker"]: r for r in new_rows}}
    sorted_rows = sorted(merged.values(), key=lambda r: (r["date"], r["metric"], r["ticker"]))

    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(sorted_rows)

    print(f"Wrote {len(sorted_rows):,} rows to {OUT_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Print rows without writing CSV")
    args = parser.parse_args()
    asyncio.run(main(args.dry_run))
