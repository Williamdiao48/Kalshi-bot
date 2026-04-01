"""Fetch article body text for matched (market, article) pairs.

⚠  Run this on a HOME MACHINE or VPS — NOT on a SEASnet/university server.
   Fetching 10,000–20,000 URLs from a university IP risks getting the server
   blacklisted by major news outlets and may trigger abuse detection.

Uses trafilatura for extraction (not newspaper3k — better at 2024+ web
structures, paywall detection, and CDN-served HTML).

Expect ~30–40% failure rate from paywalls, dead links, and bot-protection.
Failed URLs are silently skipped — a partial dataset is still useful.

Pipeline position:
    scrape_kalshi_history.py
        → match_news_to_markets.py
            → THIS SCRIPT  ← you are here
                → label_with_claude.py

Usage:
    pip install trafilatura aiohttp
    python scripts/fetch_article_bodies.py

Input:
    data/gdelt_article_list.jsonl   (output of match_news_to_markets.py)

Output:
    data/market_article_pairs.jsonl — one record per successful (market, article)
    pair with fields:
        market_ticker, market_title, market_result, market_category,
        resolution_time, gdelt_query, article_url, article_headline,
        article_domain, article_date, hours_before_resolution,
        article_body, body_char_count

Resume behaviour:
    Already-fetched URLs are tracked in data/.fetched_urls.txt (one URL per
    line).  On restart the script skips those URLs.  This file covers both
    successes and failures so known-bad URLs are not retried.

Environment variables:
    FETCH_CONCURRENCY      Max simultaneous HTTP requests (default: 8).
    DOMAIN_DELAY_SECONDS   Min seconds between requests to same domain (default: 0.5).
    FETCH_TIMEOUT_SECONDS  Per-URL timeout in seconds (default: 20).
    MIN_BODY_CHARS         Discard extractions shorter than this (default: 150).
    MAX_BODY_CHARS         Truncate body text to this length (default: 6000).
"""

import asyncio
import json
import logging
import os
import sys
import time
import urllib.parse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import aiohttp
from dotenv import load_dotenv

try:
    import trafilatura
except ImportError:
    print("trafilatura not installed.  Run: pip install trafilatura")
    sys.exit(1)

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
INPUT_FILE      = Path(__file__).parent.parent / "data" / "gdelt_article_list.jsonl"
OUTPUT_FILE     = Path(__file__).parent.parent / "data" / "market_article_pairs.jsonl"
SEEN_URLS_FILE  = Path(__file__).parent.parent / "data" / ".fetched_urls.txt"

FETCH_CONCURRENCY: int   = int(os.environ.get("FETCH_CONCURRENCY", "8"))
DOMAIN_DELAY: float      = float(os.environ.get("DOMAIN_DELAY_SECONDS", "0.5"))
FETCH_TIMEOUT: float     = float(os.environ.get("FETCH_TIMEOUT_SECONDS", "20"))
MIN_BODY_CHARS: int      = int(os.environ.get("MIN_BODY_CHARS", "150"))
MAX_BODY_CHARS: int      = int(os.environ.get("MAX_BODY_CHARS", "6000"))

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


# ---------------------------------------------------------------------------
# Temporal guard
# ---------------------------------------------------------------------------

def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _passes_temporal_guard(record: dict[str, Any]) -> bool:
    """Return True if article_date is at least 24h before resolution_time.

    Re-enforces the leakage guard from Stage 2B.  GDELT's date filtering is
    approximate and this is a cheap safety net before we spend time fetching.
    """
    art_dt = _parse_iso(record.get("article_date"))
    res_dt = _parse_iso(record.get("resolution_time"))
    if art_dt is None or res_dt is None:
        return True   # can't check → let it through, label_with_claude will validate
    return art_dt < (res_dt - timedelta(hours=24))


# ---------------------------------------------------------------------------
# Fetch + extract
# ---------------------------------------------------------------------------

async def _fetch_html(
    session: aiohttp.ClientSession,
    url: str,
) -> str | None:
    """Fetch raw HTML from a URL.  Returns None on any error."""
    try:
        async with session.get(
            url,
            headers=_HEADERS,
            timeout=aiohttp.ClientTimeout(total=FETCH_TIMEOUT),
            allow_redirects=True,
            max_redirects=5,
        ) as resp:
            if resp.status != 200:
                return None
            # Refuse to download very large pages (likely not articles)
            content_length = resp.headers.get("Content-Length")
            if content_length and int(content_length) > 5_000_000:
                return None
            return await resp.text(errors="replace")
    except Exception:
        return None


def _extract_body(html: str, url: str) -> str | None:
    """Extract main article body from HTML using trafilatura.

    Returns None if extraction fails or the result is too short to be useful.
    """
    text = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=False,
        no_fallback=False,       # allow fallback extractors on hard pages
        favor_precision=True,    # prefer clean text over recall
        deduplicate=True,
    )
    if not text or len(text) < MIN_BODY_CHARS:
        return None
    return text[:MAX_BODY_CHARS]


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

async def _process_url(
    session: aiohttp.ClientSession,
    url: str,
    records: list[dict[str, Any]],
    semaphore: asyncio.Semaphore,
    domain_last_request: dict[str, float],
    out_fh: Any,
    seen_fh: Any,
    counters: dict[str, int],
) -> None:
    """Fetch one URL, extract body, and write all matching market records."""
    domain = urllib.parse.urlparse(url).netloc

    async with semaphore:
        # Per-domain rate limiting — asyncio is single-threaded so this dict
        # access is safe without locks.
        elapsed = time.monotonic() - domain_last_request.get(domain, 0)
        if elapsed < DOMAIN_DELAY:
            await asyncio.sleep(DOMAIN_DELAY - elapsed)
        domain_last_request[domain] = time.monotonic()

        html = await _fetch_html(session, url)

    # Mark URL as seen regardless of success so we don't retry on resume.
    seen_fh.write(url + "\n")
    seen_fh.flush()

    if html is None:
        counters["fetch_failed"] += 1
        return

    # trafilatura.extract is CPU-bound but fast enough (<50ms) that we run it
    # inline rather than in a thread pool.
    body = _extract_body(html, url)
    if body is None:
        counters["extract_failed"] += 1
        return

    # Write one output record per market that referenced this URL.
    written = 0
    for rec in records:
        out_record = {**rec, "article_body": body, "body_char_count": len(body)}
        out_fh.write(json.dumps(out_record) + "\n")
        written += 1

    out_fh.flush()
    counters["success"] += written
    counters["urls_ok"] += 1


async def main() -> None:
    if not INPUT_FILE.exists():
        logging.error(
            "Input file not found: %s\nRun match_news_to_markets.py first.",
            INPUT_FILE,
        )
        sys.exit(1)

    # Load all (market, article) pairs
    all_records: list[dict[str, Any]] = []
    for line in INPUT_FILE.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Apply temporal guard before fetching to avoid wasting requests
        if _passes_temporal_guard(rec):
            all_records.append(rec)

    logging.info("Loaded %d records from %s.", len(all_records), INPUT_FILE)

    # Group records by URL so each URL is fetched exactly once even if it
    # appears for multiple markets.
    url_to_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in all_records:
        url_to_records[rec["article_url"]].append(rec)

    all_urls = list(url_to_records.keys())
    logging.info("Unique article URLs: %d", len(all_urls))

    # Load already-seen URLs (both successes and failures from prior runs)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    seen_urls: set[str] = set()
    if SEEN_URLS_FILE.exists():
        seen_urls = set(SEEN_URLS_FILE.read_text(encoding="utf-8").splitlines())
    logging.info("Already seen (skip): %d URLs.", len(seen_urls))

    remaining_urls = [u for u in all_urls if u not in seen_urls]
    logging.info("URLs to fetch: %d", len(remaining_urls))

    if not remaining_urls:
        logging.info("Nothing to do.")
        return

    semaphore = asyncio.Semaphore(FETCH_CONCURRENCY)
    domain_last_request: dict[str, float] = {}
    counters: dict[str, int] = defaultdict(int)

    connector = aiohttp.TCPConnector(limit=FETCH_CONCURRENCY, ssl=False)
    with (
        OUTPUT_FILE.open("a", encoding="utf-8") as out_fh,
        SEEN_URLS_FILE.open("a", encoding="utf-8") as seen_fh,
    ):
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                _process_url(
                    session,
                    url,
                    url_to_records[url],
                    semaphore,
                    domain_last_request,
                    out_fh,
                    seen_fh,
                    counters,
                )
                for url in remaining_urls
            ]

            # Process with progress logging every 100 completions
            done = 0
            for coro in asyncio.as_completed(tasks):
                await coro
                done += 1
                if done % 100 == 0 or done == len(tasks):
                    total_attempted = done
                    pct_ok = 100 * counters["urls_ok"] / max(total_attempted, 1)
                    logging.info(
                        "Progress: %d/%d URLs  |  ok: %d (%.0f%%)  "
                        "fetch_fail: %d  extract_fail: %d",
                        done, len(remaining_urls),
                        counters["urls_ok"], pct_ok,
                        counters["fetch_failed"], counters["extract_failed"],
                    )

    total_attempted = len(remaining_urls)
    pct_ok = 100 * counters["urls_ok"] / max(total_attempted, 1)
    logging.info(
        "Done.  %d/%d URLs succeeded (%.0f%%).  "
        "%d market-article pairs written to %s.",
        counters["urls_ok"], total_attempted, pct_ok,
        counters["success"], OUTPUT_FILE,
    )

    # Stats breakdown
    if OUTPUT_FILE.exists():
        records_out = [
            json.loads(l) for l in OUTPUT_FILE.read_text().splitlines() if l
        ]
        from collections import Counter
        cat_counts = Counter(r.get("market_category", "other") for r in records_out)
        result_counts = Counter(r.get("market_result") for r in records_out)
        logging.info("Output breakdown by category:")
        for cat, n in cat_counts.most_common():
            logging.info("  %-25s %d", cat, n)
        logging.info("Result distribution: YES=%d  NO=%d",
                     result_counts.get("yes", 0), result_counts.get("no", 0))


if __name__ == "__main__":
    asyncio.run(main())
