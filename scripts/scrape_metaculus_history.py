"""Scrape resolved binary questions from Metaculus for AI model training data.

Metaculus is a public prediction market with thousands of resolved binary
questions across politics, regulation, economics, and geopolitics — far more
text/event history than Kalshi's settled market archive.

Output format is identical to scrape_kalshi_history.py so the rest of the
pipeline (match_news_to_markets.py onwards) works unchanged.

Setup (free, ~2 minutes):
  1. Create a free account at metaculus.com
  2. Go to Settings → API → Create API token
  3. Add to .env:  METACULUS_TOKEN="your-token-here"

Usage:
    venv/bin/python scripts/scrape_metaculus_history.py

Output:
    data/kalshi_markets.jsonl  (same filename — drop-in replacement)

    Fields per record:
        ticker, series_ticker, category, title, subtitle,
        rules_primary, rules_secondary, result, open_time,
        close_time, resolution_time, settlement_value

Environment variables:
    METACULUS_TOKEN           Required. Free from metaculus.com/accounts/profile/
    METACULUS_PAGE_SIZE       Questions per API page (default: 100, max: 100).
    METACULUS_DELAY_SECONDS   Delay between pages (default: 0.5).
    MAX_MARKETS               Stop after writing this many questions (default: 5000).
    METACULUS_MIN_FORECASTERS Minimum number of forecasters for quality filter
                              (default: 5 — filters out near-empty questions).
"""

import asyncio
import json
import logging
import os
import re
import sys
import time
from collections import Counter
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

METACULUS_API_BASE   = "https://www.metaculus.com/api2"
PAGE_SIZE: int       = int(os.environ.get("METACULUS_PAGE_SIZE", "100"))
PAGE_DELAY: float    = float(os.environ.get("METACULUS_DELAY_SECONDS", "0.5"))
MAX_MARKETS: int     = int(os.environ.get("MAX_MARKETS", "5000"))
MIN_FORECASTERS: int = int(os.environ.get("METACULUS_MIN_FORECASTERS", "5"))

# ---------------------------------------------------------------------------
# Category mapping — Metaculus category names → our training category labels
# ---------------------------------------------------------------------------
_CAT_MAP: dict[str, str] = {
    # Politics
    "politics":                "political_action",
    "us politics":             "political_action",
    "us-politics":             "political_action",
    "elections":               "political_action",
    "us elections":            "political_action",
    "2024 us elections":       "political_action",
    "congress":                "political_action",
    "us congress":             "political_action",
    "executive branch":        "political_action",
    "trump administration":    "political_action",
    "geopolitics":             "political_action",
    "world":                   "political_action",
    "international":           "political_action",
    "foreign policy":          "political_action",
    "war and conflict":        "political_action",
    "ukraine":                 "political_action",
    "russia":                  "political_action",
    "china":                   "political_action",
    "middle east":             "political_action",
    # Regulatory / Legal
    "law":                     "regulatory",
    "us law":                  "regulatory",
    "supreme court":           "regulatory",
    "regulation":              "regulatory",
    "fda":                     "regulatory",
    "healthcare":              "regulatory",
    "health":                  "regulatory",
    "environment":             "regulatory",
    "climate":                 "regulatory",
    "energy":                  "regulatory",
    # Corporate / Economic
    "economics":               "corporate",
    "economy":                 "corporate",
    "business":                "corporate",
    "finance":                 "corporate",
    "companies":               "corporate",
    "technology":              "corporate",
    "artificial intelligence": "corporate",
    "space":                   "corporate",
}


def _get_category(categories: list[dict]) -> str:
    """Map Metaculus category list to our training category label."""
    for cat in categories:
        name = (cat.get("name") or cat.get("slug") or "").lower().strip()
        if name in _CAT_MAP:
            return _CAT_MAP[name]
        # Fuzzy match on substrings
        for key, label in _CAT_MAP.items():
            if key in name or name in key:
                return label
    return "other"


# ---------------------------------------------------------------------------
# HTML / rich-text stripping
# ---------------------------------------------------------------------------
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_WS_RE = re.compile(r"\s{2,}")


def _clean_text(text: str | None) -> str | None:
    """Strip HTML tags and collapse whitespace."""
    if not text:
        return text
    text = _HTML_TAG_RE.sub(" ", text)
    text = _MULTI_WS_RE.sub(" ", text)
    return text.strip() or None


# ---------------------------------------------------------------------------
# Field extraction
# ---------------------------------------------------------------------------

def _extract_fields(q: dict[str, Any]) -> dict[str, Any]:
    """Map a Metaculus question to the training record format."""
    qid        = q.get("id", "")
    categories = q.get("categories") or []
    category   = _get_category(categories)
    series     = categories[0].get("slug", "") if categories else ""
    resolution = (q.get("resolution") or "").lower()   # "yes" | "no"

    return {
        "ticker":           f"MET-{qid}",
        "series_ticker":    series,
        "category":         category,
        "title":            q.get("title", ""),
        "subtitle":         None,
        "rules_primary":    _clean_text(q.get("resolution_criteria")),
        "rules_secondary":  _clean_text(q.get("fine_print")),
        "result":           resolution,
        "open_time":        q.get("publish_time") or q.get("created_time"),
        "close_time":       q.get("close_time"),
        "resolution_time":  q.get("resolution_time") or q.get("close_time"),
        "settlement_value": 1 if resolution == "yes" else 0,
    }


def _is_usable(q: dict[str, Any]) -> bool:
    """Return True if this question is suitable for training data."""
    # Must be a binary question resolved yes or no (not ambiguous/annulled)
    if q.get("type") not in ("binary", None):
        return False
    resolution = (q.get("resolution") or "").lower()
    if resolution not in ("yes", "no"):
        return False
    # Require minimum forecaster engagement for quality
    forecasters = q.get("number_of_forecasters") or q.get("predictions_count") or 0
    if forecasters < MIN_FORECASTERS:
        return False
    # Must have a resolution date for the temporal window
    if not (q.get("resolution_time") or q.get("close_time")):
        return False
    return True


# ---------------------------------------------------------------------------
# API pagination
# ---------------------------------------------------------------------------

async def _paginate(
    session: aiohttp.ClientSession,
    token: str,
    out_fh: Any,
) -> tuple[int, int]:
    """Fetch all resolved binary questions from Metaculus.

    Returns (total_seen, total_written).
    """
    headers = {
        "Authorization": f"Token {token}",
        "Accept": "application/json",
    }
    total_seen    = 0
    total_written = 0
    offset        = 0
    page_num      = 0

    while total_written < MAX_MARKETS:
        params = {
            "type":       "binary",
            "limit":      PAGE_SIZE,
            "offset":     offset,
        }

        try:
            async with session.get(
                f"{METACULUS_API_BASE}/questions/",
                params=params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 401:
                    logging.error(
                        "Authentication failed (401). "
                        "Check METACULUS_TOKEN in your .env.\n"
                        "Get a free token at: metaculus.com/accounts/profile/"
                    )
                    break
                if resp.status == 429:
                    logging.warning("Rate limited (429) — waiting 30s.")
                    await asyncio.sleep(30)
                    continue
                if resp.status >= 400:
                    body = await resp.text()
                    logging.error("HTTP error %s — response body: %s", resp.status, body[:500])
                    break
                data = await resp.json()
        except aiohttp.ClientResponseError as exc:
            logging.error("HTTP error %s: %s", exc.status, exc.message)
            break
        except aiohttp.ClientError as exc:
            logging.error("Request error: %s", exc)
            break

        results = data.get("results") or []
        if not results:
            logging.info("No more results.")
            break

        page_num  += 1
        page_written = 0
        for q in results:
            total_seen += 1
            if _is_usable(q):
                out_fh.write(json.dumps(_extract_fields(q)) + "\n")
                page_written += 1
                total_written += 1
                if total_written >= MAX_MARKETS:
                    break
        out_fh.flush()

        logging.info(
            "Page %3d | offset %5d | page written: %2d | total written: %d | total seen: %d",
            page_num, offset, page_written, total_written, total_seen,
        )

        # Check if there are more pages
        if not data.get("next"):
            logging.info("Last page reached.")
            break

        offset += PAGE_SIZE
        await asyncio.sleep(PAGE_DELAY)

    return total_seen, total_written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    token = os.environ.get("METACULUS_TOKEN", "")
    if not token:
        logging.error(
            "METACULUS_TOKEN not set.\n"
            "  1. Create a free account at metaculus.com\n"
            "  2. Go to Settings → API → Create API token\n"
            "  3. Add to .env:  METACULUS_TOKEN=\"your-token-here\""
        )
        sys.exit(1)

    logging.info(
        "Scraping Metaculus resolved binary questions (max %d, min %d forecasters).",
        MAX_MARKETS, MIN_FORECASTERS,
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()

    with OUTPUT_FILE.open("w", encoding="utf-8") as fh:
        async with aiohttp.ClientSession() as session:
            total_seen, total_written = await _paginate(session, token, fh)

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
