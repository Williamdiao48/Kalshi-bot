"""Scrape resolved Kalshi text/event markets for AI model training data.

Fetches settled markets from the Kalshi API, filters out purely numeric markets
(weather, crypto, forex, economics), and saves the remainder to
data/kalshi_markets.jsonl.

Target: 2,000–5,000 resolved text/event markets across political, regulatory,
and corporate event categories.

Usage:
    cd Kalshi-Bot
    venv/bin/python scripts/scrape_kalshi_history.py

Output:
    data/kalshi_markets.jsonl — one JSON object per line with fields:
        ticker, series_ticker, category, title, subtitle,
        rules_primary, rules_secondary, result, open_time,
        close_time, resolution_time, settlement_value

Notes:
    - yes_bid/yes_ask are intentionally excluded: settled markets always return
      0 for these fields and the values are meaningless for training data.
    - rules_primary/rules_secondary are HTML-stripped before saving.
    - resolution_time = expiration_time (the actual settlement timestamp).
      Use this field (not close_time) for the 24h pre-resolution cutoff.
    - Results are written incrementally per page so a crash mid-run does not
      discard already-collected data.
"""

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import aiohttp
from dotenv import load_dotenv

# Allow imports from the project root
import base64
import time as _time

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Use dedicated SCRAPER_ variables so this script never touches the bot's
# demo credentials (KALSHI_KEY_ID / KALSHI_PRIVATE_KEY_STR).
#
# Add these to .env:
#   KALSHI_SCRAPER_KEY_ID=<your production key id>
#   KALSHI_SCRAPER_PRIVATE_KEY_STR=<your production private key (PEM, \\n-escaped)>
#   KALSHI_SCRAPER_ENVIRONMENT=production
#
_SCRAPER_KEY_ID: str = os.environ.get("KALSHI_SCRAPER_KEY_ID", "")
_raw = os.environ.get("KALSHI_SCRAPER_PRIVATE_KEY_STR", "")
_SCRAPER_PRIVATE_KEY_STR: str = _raw.replace("\\n", "\n") if "\\n" in _raw else _raw

KALSHI_ENVIRONMENT: str = os.environ.get("KALSHI_SCRAPER_ENVIRONMENT", "demo")
KALSHI_API_BASE: str = (
    "https://api.elections.kalshi.com/trade-api/v2"
    if KALSHI_ENVIRONMENT == "production"
    else "https://demo-api.kalshi.co/trade-api/v2"
)

# ---------------------------------------------------------------------------
# Auth — standalone, does not share state with kalshi_bot.auth
# ---------------------------------------------------------------------------
_cached_private_key = None


def generate_headers(method: str, path: str) -> dict:
    """RSA-PSS signed headers using the scraper's own production credentials."""
    global _cached_private_key
    if not _SCRAPER_KEY_ID or not _SCRAPER_PRIVATE_KEY_STR:
        logging.warning(
            "KALSHI_SCRAPER_KEY_ID / KALSHI_SCRAPER_PRIVATE_KEY_STR not set. "
            "Requests will be unauthenticated and likely fail."
        )
        return {}
    if _cached_private_key is None:
        _cached_private_key = serialization.load_pem_private_key(
            _SCRAPER_PRIVATE_KEY_STR.encode("utf-8"), password=None
        )
    ts  = str(int(_time.time() * 1000))
    msg = (ts + method + path).encode("utf-8")
    sig = _cached_private_key.sign(
        msg,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
        hashes.SHA256(),
    )
    return {
        "KALSHI-ACCESS-KEY":       _SCRAPER_KEY_ID,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode("utf-8"),
        "KALSHI-ACCESS-TIMESTAMP": ts,
    }
_MARKETS_PATH = "/trade-api/v2/markets"

OUTPUT_FILE = Path(__file__).parent.parent / "data" / "kalshi_markets.jsonl"

# Max markets to collect (None = unlimited)
MAX_MARKETS: int | None = int(os.environ.get("MAX_MARKETS", "10000"))

# Seconds between pages — stay well under Kalshi rate limits
PAGE_DELAY = 0.30

# ---------------------------------------------------------------------------
# Category mapping — series prefix → human label used for stratified splits
# during training.  "unknown" is the catch-all for series not listed here.
# ---------------------------------------------------------------------------
CATEGORY_PREFIXES: dict[str, str] = {
    "KXTRUMPSAY":  "mentions",
    "KXBIDENWORD": "mentions",
    "KXTRUMPWORD": "mentions",
    "KXPOTUS":     "political_action",
    "KXSENATE":    "political_action",
    "KXHOUSE":     "political_action",
    "KXGOV":       "political_action",
    "KXELECT":     "political_action",
    "KXPRES":      "political_action",
    "KXSCOTUS":    "political_action",
    "KXFDA":       "regulatory",
    "KXFTC":       "regulatory",
    "KXSEC":       "regulatory",
    "KXDOJ":       "regulatory",
    "KXEPA":       "regulatory",
    "KXFED":       "regulatory",   # note: different from the numeric KXFED rate market
    "KXCEO":       "corporate",
    "KXIPO":       "corporate",
    "KXMA":        "corporate",    # M&A
    "KXEARNINGS":  "corporate",
}


def _get_category(series_ticker: str) -> str:
    """Map a series ticker to a training category label."""
    s = (series_ticker or "").upper()
    for prefix, category in CATEGORY_PREFIXES.items():
        if s.startswith(prefix):
            return category
    return "other"


# Series ticker prefixes that indicate purely numeric (non-text) markets.
# These resolve on numbers, not on events, so they're useless for the
# semantic signal model.
NUMERIC_PREFIXES: tuple[str, ...] = (
    "KXHIGH",    # daily temperature highs
    "KXBTC",     # Bitcoin price
    "KXETH",     # Ethereum price
    "KXSOL",     # Solana price
    "KXXRP",     # XRP price
    "KXDOGE",    # Dogecoin price
    "KXADA",     # Cardano price
    "KXAVAX",    # Avalanche price
    "KXLINK",    # Chainlink price
    "KXEURUSD",  # EUR/USD forex
    "KXUSDJPY",  # USD/JPY forex
    "KXCPI",     # Consumer Price Index
    "KXNFP",     # Non-farm payrolls
    "KXUNRATE",  # Unemployment rate
    "KXPPI",     # Producer Price Index
    "KXPCE",     # PCE inflation
    "KXJOBLESS", # Jobless claims
    "KXICSA",    # Initial claims
    "KXISM",     # ISM indices
    "KXFED",     # Federal funds rate
    "KXFFR",     # Federal funds rate
    "KXDGS",     # Treasury yields
    "KXWTI",     # WTI crude oil
    "KXOIL",     # Oil price
    "KXNATGAS",  # Natural gas
    "KXNG",      # Natural gas
    "KXSPX",     # S&P 500
    "KXNDX",     # Nasdaq 100
    "KXINXD",    # Index
    "KXMVE",     # Sports
    "KXNBA",     # NBA
    "KXNFL",     # NFL
    "KXNHL",     # NHL
    "KXMLB",     # MLB
    "KXMMA",     # MMA
    "KXSOCCER",  # Soccer
    "KXGOLF",    # Golf
    "KXTENNIS",  # Tennis
    "KXNASCAR",  # NASCAR
    "KXF1",      # Formula 1
    "KXPGA",     # PGA Tour
)


def _is_text_market(market: dict[str, Any]) -> bool:
    """Return True if this is a text/event market (not numeric or sports).

    Checks series_ticker first.  Falls back to the ticker field itself when
    series_ticker is empty — some Kalshi sports markets have a blank
    series_ticker but a ticker that clearly starts with a numeric prefix.
    """
    series = (market.get("series_ticker") or "").upper()
    ticker = (market.get("ticker") or "").upper()
    check  = series if series else ticker
    return not any(check.startswith(p) for p in NUMERIC_PREFIXES)


_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str | None) -> str | None:
    """Strip HTML tags and collapse whitespace. Returns None if input is None."""
    if not text:
        return text
    return " ".join(_HTML_TAG_RE.sub(" ", text).split())


def _extract_fields(market: dict[str, Any]) -> dict[str, Any]:
    """Extract and clean fields needed for training data.

    - yes_bid/yes_ask excluded: always 0 for settled markets, meaningless.
    - rules_primary/rules_secondary HTML-stripped before saving.
    - expiration_time saved as resolution_time for unambiguous downstream use.
    - category inferred from series_ticker for stratified train/val/test splits.
    """
    series = market.get("series_ticker") or ""
    return {
        "ticker":           market.get("ticker"),
        "series_ticker":    series,
        "category":         _get_category(series),
        "title":            market.get("title"),
        "subtitle":         market.get("subtitle"),
        "rules_primary":    _strip_html(market.get("rules_primary")),
        "rules_secondary":  _strip_html(market.get("rules_secondary")),
        "result":           market.get("result"),           # "yes" | "no"
        "open_time":        market.get("open_time"),
        "close_time":       market.get("close_time"),
        "resolution_time":  market.get("expiration_time"),  # use for 24h cutoff
        "settlement_value": market.get("settlement_value"),
    }


async def _paginate_settled(
    session: aiohttp.ClientSession,
    out_fh: Any,
) -> tuple[int, int]:
    """Cursor-paginate all settled markets from the Kalshi API.

    Filters numeric/sports markets and fully-resolved (result=yes/no) markets
    in-flight.  Each qualifying market is written to out_fh immediately so a
    crash mid-run does not discard already-collected pages.

    Returns:
        (total_seen, total_written) counts.
    """
    total_seen = 0
    total_written = 0
    cursor: str | None = None
    page_num = 0

    while True:
        if MAX_MARKETS is not None and total_written >= MAX_MARKETS:
            logging.info("Reached MAX_MARKETS=%d — stopping.", MAX_MARKETS)
            break

        page_size = 100
        params: dict[str, Any] = {"status": "settled", "limit": page_size}
        if cursor:
            params["cursor"] = cursor

        headers = generate_headers("GET", _MARKETS_PATH)

        try:
            async with session.get(
                f"{KALSHI_API_BASE}/markets",
                params=params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 429:
                    logging.warning("Rate limited (429) — waiting 60s then retrying.")
                    await asyncio.sleep(60)
                    continue
                if resp.status == 401:
                    logging.error(
                        "Authentication failed (401). Check KALSHI_KEY_ID and "
                        "KALSHI_PRIVATE_KEY_STR in your .env, and confirm "
                        "KALSHI_ENVIRONMENT=%s is correct.",
                        KALSHI_ENVIRONMENT,
                    )
                    break
                resp.raise_for_status()
                data = await resp.json()
        except aiohttp.ClientResponseError as exc:
            logging.error("HTTP error %s: %s", exc.status, exc.message)
            break
        except aiohttp.ClientError as exc:
            logging.error("Request error: %s", exc)
            break

        page = data.get("markets", [])
        if not page:
            logging.info("Empty page — done.")
            break

        page_num += 1
        total_seen += len(page)

        # Filter: text/event markets with a known result only
        page_written = 0
        for m in page:
            if _is_text_market(m) and m.get("result") in ("yes", "no"):
                out_fh.write(json.dumps(_extract_fields(m)) + "\n")
                page_written += 1
        out_fh.flush()  # ensure page is on disk before fetching the next
        total_written += page_written

        logging.info(
            "Page %3d: %3d fetched, %3d written (total written: %d, total seen: %d)",
            page_num, len(page), page_written, total_written, total_seen,
        )

        cursor = data.get("cursor")
        if not cursor:
            logging.info("No more pages.")
            break

        await asyncio.sleep(PAGE_DELAY)

    return total_seen, total_written


async def main() -> None:
    from collections import Counter

    logging.info(
        "Scraping settled Kalshi text/event markets from %s environment.",
        KALSHI_ENVIRONMENT,
    )
    if KALSHI_ENVIRONMENT != "production":
        logging.warning(
            "KALSHI_ENVIRONMENT=%s — demo environment has very few settled markets. "
            "Set KALSHI_ENVIRONMENT=production in .env for real training data.",
            KALSHI_ENVIRONMENT,
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()

    # Open for writing before pagination starts — incremental flush per page.
    # If the script is interrupted, already-written pages are preserved.
    with OUTPUT_FILE.open("w", encoding="utf-8") as fh:
        async with aiohttp.ClientSession() as session:
            total_seen, total_written = await _paginate_settled(session, fh)

    elapsed = time.monotonic() - t0
    logging.info(
        "Done. Wrote %d resolved text/event markets (from %d total seen) to %s in %.1fs.",
        total_written, total_seen, OUTPUT_FILE, elapsed,
    )

    # Post-run stats from the output file
    records = [json.loads(line) for line in OUTPUT_FILE.read_text().splitlines() if line]
    yes_count = sum(1 for r in records if r["result"] == "yes")
    no_count  = sum(1 for r in records if r["result"] == "no")
    logging.info("Breakdown: %d YES, %d NO.", yes_count, no_count)

    cat_counts = Counter(r.get("category", "other") for r in records)
    logging.info("Category breakdown:")
    for cat, count in cat_counts.most_common():
        logging.info("  %-25s %d", cat, count)

    series_counts = Counter(r.get("series_ticker", "UNKNOWN") for r in records)
    logging.info("Top 20 series:")
    for series, count in series_counts.most_common(20):
        logging.info("  %-30s %d", series, count)


if __name__ == "__main__":
    asyncio.run(main())
