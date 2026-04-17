"""Fetch historical NWS CLI (Climatological Report) actual high temperatures.

For each of the 21 Kalshi temperature cities, queries the NWS Products API for
all available CLI products, parses the daily MAXIMUM temperature, and writes
a CSV of ground-truth actuals that backtest_source_accuracy.py and
backtest_band_arb_yes.py use to evaluate accuracy.

Output: data/nws_cli_actuals.csv
  city_metric, date, actual_high_f

Usage:
  venv/bin/python scripts/fetch_nws_cli_history.py
  venv/bin/python scripts/fetch_nws_cli_history.py --out data/nws_cli_actuals.csv
  venv/bin/python scripts/fetch_nws_cli_history.py --days 60
  venv/bin/python scripts/fetch_nws_cli_history.py --append
"""

import argparse
import asyncio
import csv
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import aiohttp

# ---------------------------------------------------------------------------
# Path setup — allow running from project root without install
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from kalshi_bot.news.nws_climo import CLIMO_LOCATIONS, _parse_max_f  # noqa: E402

# Extended station mapping — 13 cities beyond the original CLIMO_LOCATIONS 8.
# These are the same stations used by the live METAR/NOAA fetchers.
_EXTRA_STATIONS: dict[str, tuple[str, str, ZoneInfo]] = {
    "temp_high_dal":  ("DAL",  "Dallas Love Field",        ZoneInfo("America/Chicago")),
    "temp_high_atl":  ("ATL",  "Atlanta",                  ZoneInfo("America/New_York")),
    "temp_high_sea":  ("SEA",  "Seattle",                  ZoneInfo("America/Los_Angeles")),
    "temp_high_sfo":  ("SFO",  "San Francisco",            ZoneInfo("America/Los_Angeles")),
    "temp_high_phx":  ("PHX",  "Phoenix",                  ZoneInfo("America/Phoenix")),
    "temp_high_phl":  ("PHL",  "Philadelphia",             ZoneInfo("America/New_York")),
    "temp_high_msp":  ("MSP",  "Minneapolis",              ZoneInfo("America/Chicago")),
    "temp_high_dca":  ("DCA",  "Washington DC",            ZoneInfo("America/New_York")),
    "temp_high_las":  ("LAS",  "Las Vegas",                ZoneInfo("America/Los_Angeles")),
    "temp_high_okc":  ("OKC",  "Oklahoma City",            ZoneInfo("America/Chicago")),
    "temp_high_sat":  ("SAT",  "San Antonio",              ZoneInfo("America/Chicago")),
    "temp_high_msy":  ("MSY",  "New Orleans",              ZoneInfo("America/Chicago")),
    "temp_high_dfw":  ("DFW",  "Dallas/Fort Worth",        ZoneInfo("America/Chicago")),
}

# Combined: all 21 Kalshi temperature cities
ALL_LOCATIONS: dict[str, tuple[str, str, ZoneInfo]] = {
    **CLIMO_LOCATIONS,
    **_EXTRA_STATIONS,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_NWS_PRODUCTS_URL = "https://api.weather.gov/products"
_HEADERS = {"User-Agent": "kalshi-bot/1.0 (educational; contact: user@example.com)"}

# NWS rate-limit: 5 concurrent requests is safe; they ask for ≤ 1 req/s per IP
# in practice but tolerate small bursts.  A semaphore of 5 and a 0.2 s delay
# between product-text fetches keeps us well within limits.
_SEMAPHORE = asyncio.Semaphore(5)
_FETCH_DELAY = 0.15  # seconds between individual product-text GETs

# Month abbreviation → int (for parsing CLI date lines)
_MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


# ---------------------------------------------------------------------------
# Date extraction from CLI product text
# ---------------------------------------------------------------------------

def _extract_product_date(text: str, issuance_dt: datetime, city_tz: ZoneInfo) -> str | None:
    """Return the date string (YYYY-MM-DD) this CLI product covers.

    Prefer an explicit date line in the text (e.g. "MAR 15 2026").
    Fall back to the issuance date in the city's local timezone.
    """
    date_m = re.search(
        r"\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+(\d{1,2})\s+(\d{4})\b",
        text, re.IGNORECASE,
    )
    if date_m:
        try:
            month = _MONTH_MAP[date_m.group(1).upper()]
            day = int(date_m.group(2))
            year = int(date_m.group(3))
            from datetime import date
            return date(year, month, day).isoformat()
        except (ValueError, KeyError):
            pass

    # Fall back: use the issuance time's local calendar date.
    # CLI products issued after midnight but covering the *previous* day are
    # the "next-morning final" report — we keep them because they contain the
    # confirmed daily maximum.
    return issuance_dt.astimezone(city_tz).date().isoformat()


# ---------------------------------------------------------------------------
# Fetch all CLI stubs for one city (paginated)
# ---------------------------------------------------------------------------

async def _fetch_stubs(
    session: aiohttp.ClientSession,
    location: str,
) -> list[dict]:
    """Return all CLI product stubs for a location (newest first).

    The NWS /products endpoint does not support true offset pagination — it
    just returns up to `limit` results.  500 is the documented maximum and
    covers ~1.5 years of daily products.
    """
    async with _SEMAPHORE:
        try:
            async with session.get(
                _NWS_PRODUCTS_URL,
                params={"type": "CLI", "location": location, "limit": "500"},
                headers=_HEADERS,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json(content_type=None)
        except Exception as exc:
            log.error("Failed to fetch CLI stubs for %s: %s", location, exc)
            return []

    stubs = data.get("@graph", [])
    log.info("  %s: %d CLI stubs available", location, len(stubs))
    return stubs


# ---------------------------------------------------------------------------
# Fetch and parse one CLI product text
# ---------------------------------------------------------------------------

async def _parse_stub(
    session: aiohttp.ClientSession,
    stub: dict,
    city_tz: ZoneInfo,
) -> tuple[str, float] | None:
    """Fetch product text and return (date_str, max_f) or None."""
    product_url = stub.get("@id") or stub.get("id")
    if not product_url:
        return None

    issuance_str = stub.get("issuanceTime", "")
    try:
        issuance_dt = datetime.fromisoformat(issuance_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        issuance_dt = datetime.now(timezone.utc)

    # If the stub already contains productText, skip the extra HTTP request.
    text = stub.get("productText")
    if not text:
        async with _SEMAPHORE:
            await asyncio.sleep(_FETCH_DELAY)
            try:
                async with session.get(
                    product_url,
                    headers=_HEADERS,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json(content_type=None)
                text = data.get("productText") or ""
            except Exception as exc:
                log.debug("product text fetch failed for %s: %s", product_url, exc)
                return None

    max_f = _parse_max_f(text)
    if max_f is None:
        return None

    date_str = _extract_product_date(text, issuance_dt, city_tz)
    if date_str is None:
        return None

    return date_str, max_f


# ---------------------------------------------------------------------------
# Fetch all historical actuals for one city
# ---------------------------------------------------------------------------

async def _fetch_city_actuals(
    session: aiohttp.ClientSession,
    metric: str,
    location: str,
    city_name: str,
    city_tz: ZoneInfo,
) -> list[dict]:
    """Return list of {city_metric, date, actual_high_f} for one city."""
    stubs = await _fetch_stubs(session, location)
    if not stubs:
        return []

    tasks = [_parse_stub(session, stub, city_tz) for stub in stubs]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Deduplicate by date: if multiple products cover the same date, keep the
    # highest reported max (final report supersedes preliminary).
    best: dict[str, float] = {}
    for result in results:
        if isinstance(result, Exception) or result is None:
            continue
        date_str, max_f = result
        if date_str not in best or max_f > best[date_str]:
            best[date_str] = max_f

    rows = [
        {"city_metric": metric, "date": d, "actual_high_f": v}
        for d, v in sorted(best.items())
    ]
    log.info("  %s (%s): %d date(s) with actuals", city_name, location, len(rows))
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(out_path: Path, append: bool = False, days: int | None = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing rows if appending
    existing: dict[tuple[str, str], float] = {}
    if append and out_path.exists():
        with out_path.open(newline="") as f:
            for row in csv.DictReader(f):
                key = (row["city_metric"], row["date"])
                existing[key] = float(row["actual_high_f"])
        log.info("Loaded %d existing rows (append mode)", len(existing))

    # Date cutoff: only keep rows within --days of today
    cutoff_date: str | None = None
    if days is not None:
        from datetime import date, timedelta
        cutoff_date = (date.today() - timedelta(days=days)).isoformat()
        log.info("Filtering to dates >= %s (%d days)", cutoff_date, days)

    all_rows: list[dict] = []

    async with aiohttp.ClientSession() as session:
        tasks = [
            _fetch_city_actuals(session, metric, loc, city, tz)
            for metric, (loc, city, tz) in ALL_LOCATIONS.items()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    for metric, result in zip(ALL_LOCATIONS, results):
        if isinstance(result, Exception):
            log.error("City fetch error for %s: %s", metric, result)
        else:
            all_rows.extend(result)

    # Merge with existing (new data wins on same date)
    merged: dict[tuple[str, str], float] = dict(existing)
    for row in all_rows:
        key = (row["city_metric"], row["date"])
        if cutoff_date is None or row["date"] >= cutoff_date:
            merged[key] = row["actual_high_f"]

    final_rows = [
        {"city_metric": m, "date": d, "actual_high_f": v}
        for (m, d), v in sorted(merged.items())
        if cutoff_date is None or d >= cutoff_date
    ]

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["city_metric", "date", "actual_high_f"])
        writer.writeheader()
        writer.writerows(final_rows)

    log.info("Wrote %d rows to %s", len(final_rows), out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch historical NWS CLI actuals.")
    parser.add_argument(
        "--out", default="data/nws_cli_actuals.csv",
        help="Output CSV path (default: data/nws_cli_actuals.csv)",
    )
    parser.add_argument(
        "--days", type=int, default=None,
        help="Only include rows within this many days of today (default: all available)",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="Merge with existing CSV rather than overwriting",
    )
    args = parser.parse_args()
    asyncio.run(main(Path(args.out), append=args.append, days=args.days))
