"""Scrape resolved binary questions from Manifold Markets for AI model training data.

Manifold Markets is a free prediction market with thousands of resolved binary
questions across politics, economics, technology, and more.  No authentication
required — all public market data is freely accessible.

Output format is identical to scrape_kalshi_history.py so the rest of the
pipeline (match_news_to_markets.py onwards) works unchanged.

Usage:
    venv/bin/python scripts/scrape_manifold_history.py

Output:
    data/kalshi_markets.jsonl  (same filename — drop-in replacement)

    Fields per record:
        ticker, series_ticker, category, title, subtitle,
        rules_primary, rules_secondary, result, open_time,
        close_time, resolution_time, settlement_value

Environment variables:
    MANIFOLD_PAGE_SIZE        Markets per search page (default: 100, max: 100).
    MANIFOLD_DETAIL_CONCURRENCY  Concurrent individual-market fetches (default: 10).
    MANIFOLD_DELAY_SECONDS    Delay between search pages (default: 0.5).
    MAX_MARKETS               Stop after writing this many questions (default: 5000).
    MANIFOLD_MIN_BETTORS      Minimum unique bettors for quality filter (default: 5).
"""

import asyncio
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUTPUT_FILE = Path(__file__).parent.parent / "data" / "kalshi_markets.jsonl"

MANIFOLD_API_BASE       = "https://api.manifold.markets/v0"
PAGE_SIZE: int          = int(os.environ.get("MANIFOLD_PAGE_SIZE", "100"))
DETAIL_CONCURRENCY: int = int(os.environ.get("MANIFOLD_DETAIL_CONCURRENCY", "10"))
PAGE_DELAY: float       = float(os.environ.get("MANIFOLD_DELAY_SECONDS", "0.5"))
MAX_MARKETS: int        = int(os.environ.get("MAX_MARKETS", "5000"))
MIN_BETTORS: int        = int(os.environ.get("MANIFOLD_MIN_BETTORS", "5"))

# ---------------------------------------------------------------------------
# Category mapping — Manifold group slugs → our training category labels
# ---------------------------------------------------------------------------
_CAT_MAP: dict[str, str] = {
    # Politics
    "politics":                "political_action",
    "us-politics":             "political_action",
    "elections":               "political_action",
    "us-elections":            "political_action",
    "2024-us-election":        "political_action",
    "trump":                   "political_action",
    "congress":                "political_action",
    "senate":                  "political_action",
    "house-of-representatives":"political_action",
    "geopolitics":             "political_action",
    "world":                   "political_action",
    "foreign-policy":          "political_action",
    "war-and-conflict":        "political_action",
    "ukraine":                 "political_action",
    "russia":                  "political_action",
    "china":                   "political_action",
    "middle-east":             "political_action",
    # Regulatory / Legal
    "law":                     "regulatory",
    "us-law":                  "regulatory",
    "supreme-court":           "regulatory",
    "regulation":              "regulatory",
    "fda":                     "regulatory",
    "healthcare":              "regulatory",
    "health":                  "regulatory",
    "environment":             "regulatory",
    "climate":                 "regulatory",
    "climate-change":          "regulatory",
    "energy":                  "regulatory",
    # Corporate / Economic
    "economics":               "corporate",
    "economy":                 "corporate",
    "business":                "corporate",
    "finance":                 "corporate",
    "companies":               "corporate",
    "technology":              "corporate",
    "tech":                    "corporate",
    "ai":                      "corporate",
    "artificial-intelligence": "corporate",
    "openai":                  "corporate",
    "space":                   "corporate",
    "startups":                "corporate",
    "crypto":                  "corporate",
    "stocks":                  "corporate",
}


def _get_category(group_slugs: list[str]) -> str:
    """Map Manifold group slugs to our training category label."""
    for slug in (group_slugs or []):
        s = (slug or "").lower().strip()
        if s in _CAT_MAP:
            return _CAT_MAP[s]
        for key, label in _CAT_MAP.items():
            if key in s or s in key:
                return label
    return "other"


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

def _ms_to_iso(ms: int | None) -> str | None:
    """Convert millisecond epoch timestamp to ISO 8601 string."""
    if ms is None:
        return None
    try:
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()
    except (ValueError, OSError, OverflowError):
        return None


# ---------------------------------------------------------------------------
# Field extraction
# ---------------------------------------------------------------------------

def _extract_fields(m: dict[str, Any]) -> dict[str, Any]:
    """Map a Manifold market detail record to the training record format."""
    resolution = (m.get("resolution") or "").lower()   # "yes" | "no"
    group_slugs = m.get("groupSlugs") or []
    category    = _get_category(group_slugs)
    series      = group_slugs[0] if group_slugs else ""

    return {
        "ticker":           f"MFD-{m.get('id', '')}",
        "series_ticker":    series,
        "category":         category,
        "title":            m.get("question", ""),
        "subtitle":         None,
        "rules_primary":    (m.get("textDescription") or "").strip() or None,
        "rules_secondary":  None,
        "result":           resolution,
        "open_time":        _ms_to_iso(m.get("createdTime")),
        "close_time":       _ms_to_iso(m.get("closeTime")),
        "resolution_time":  _ms_to_iso(m.get("resolutionTime")),
        "settlement_value": 1 if resolution == "yes" else 0,
    }


def _is_usable_search(m: dict[str, Any]) -> bool:
    """Return True if a search-result market is worth fetching details for."""
    if m.get("outcomeType") != "BINARY":
        return False
    if m.get("resolution") not in ("YES", "NO"):
        return False
    if (m.get("uniqueBettorCount") or 0) < MIN_BETTORS:
        return False
    if not m.get("resolutionTime"):
        return False
    return True


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

async def _search_page(
    session: aiohttp.ClientSession,
    offset: int,
) -> list[dict[str, Any]]:
    """Fetch one page of resolved binary markets from the search endpoint."""
    params = {
        "term":         "",
        "filter":       "resolved",
        "contractType": "BINARY",
        "limit":        PAGE_SIZE,
        "offset":       offset,
        "sort":         "score",     # highest-engagement first
    }
    try:
        async with session.get(
            f"{MANIFOLD_API_BASE}/search-markets",
            params=params,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if resp.status == 429:
                logging.warning("Rate limited (429) on search — waiting 30s.")
                await asyncio.sleep(30)
                return await _search_page(session, offset)   # retry once
            if resp.status >= 400:
                body = await resp.text()
                logging.error("Search error %s: %s", resp.status, body[:300])
                return []
            return await resp.json()
    except aiohttp.ClientError as exc:
        logging.error("Search request error: %s", exc)
        return []


async def _fetch_market_detail(
    session: aiohttp.ClientSession,
    market_id: str,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any] | None:
    """Fetch full market details (includes textDescription and groupSlugs)."""
    async with semaphore:
        try:
            async with session.get(
                f"{MANIFOLD_API_BASE}/market/{market_id}",
                timeout=aiohttp.ClientTimeout(total=20),
            ) as resp:
                if resp.status == 404:
                    return None
                if resp.status == 429:
                    await asyncio.sleep(10)
                    return None
                if resp.status >= 400:
                    return None
                return await resp.json()
        except aiohttp.ClientError:
            return None


# ---------------------------------------------------------------------------
# Main pagination loop
# ---------------------------------------------------------------------------

async def _paginate(
    session: aiohttp.ClientSession,
    out_fh: Any,
    existing_tickers: set[str],
) -> tuple[int, int]:
    """Fetch all resolved binary markets and write usable ones to out_fh.

    Returns (total_seen, total_written).
    """
    semaphore  = asyncio.Semaphore(DETAIL_CONCURRENCY)
    total_seen    = 0
    total_written = 0
    offset        = 0
    page_num      = 0

    while total_written < MAX_MARKETS:
        page = await _search_page(session, offset)
        if not page:
            logging.info("Empty search page — done.")
            break

        page_num   += 1
        total_seen += len(page)

        # Filter: only fetch details for usable candidates
        candidates = [m for m in page if _is_usable_search(m)]

        # Concurrently fetch full market details for candidates
        detail_tasks = [
            _fetch_market_detail(session, m["id"], semaphore)
            for m in candidates
        ]
        details = await asyncio.gather(*detail_tasks)

        page_written = 0
        for detail in details:
            if detail is None:
                continue
            # Re-validate after detail fetch (data might differ)
            if detail.get("resolution") not in ("YES", "NO"):
                continue
            fields = _extract_fields(detail)
            if fields["ticker"] in existing_tickers:
                continue
            existing_tickers.add(fields["ticker"])
            out_fh.write(json.dumps(fields) + "\n")
            page_written += 1
            total_written += 1
            if total_written >= MAX_MARKETS:
                break
        out_fh.flush()

        logging.info(
            "Page %3d | offset %5d | candidates: %2d | written: %2d | total: %d",
            page_num, offset, len(candidates), page_written, total_written,
        )

        # Manifold search returns fewer than PAGE_SIZE when exhausted
        if len(page) < PAGE_SIZE:
            logging.info("Last page reached.")
            break

        offset += PAGE_SIZE
        await asyncio.sleep(PAGE_DELAY)

    return total_seen, total_written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    logging.info(
        "Scraping Manifold Markets resolved binary questions (max %d, min %d bettors).",
        MAX_MARKETS, MIN_BETTORS,
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Load existing tickers to avoid overwriting or duplicating
    existing_tickers: set[str] = set()
    if OUTPUT_FILE.exists():
        for line in OUTPUT_FILE.read_text(encoding="utf-8").splitlines():
            try:
                existing_tickers.add(json.loads(line)["ticker"])
            except (json.JSONDecodeError, KeyError):
                pass
    logging.info("Existing markets in file: %d — will append new only.", len(existing_tickers))

    t0 = time.monotonic()

    connector = aiohttp.TCPConnector(limit=DETAIL_CONCURRENCY + 5)
    with OUTPUT_FILE.open("a", encoding="utf-8") as fh:
        async with aiohttp.ClientSession(connector=connector) as session:
            total_seen, total_written = await _paginate(session, fh, existing_tickers)

    elapsed = time.monotonic() - t0
    logging.info(
        "Done. Wrote %d questions (%d seen, %.0f%% pass rate) in %.1fs.",
        total_written, total_seen,
        100 * total_written / max(total_seen, 1),
        elapsed,
    )

    # Post-run stats
    records = [json.loads(l) for l in OUTPUT_FILE.read_text().splitlines() if l]
    yes_count = sum(1 for r in records if r["result"] == "yes")
    no_count  = sum(1 for r in records if r["result"] == "no")
    logging.info("Result breakdown: %d YES, %d NO.", yes_count, no_count)

    cat_counts = Counter(r.get("category", "other") for r in records)
    logging.info("Category breakdown:")
    for cat, count in cat_counts.most_common():
        logging.info("  %-25s %d", cat, count)


if __name__ == "__main__":
    asyncio.run(main())
