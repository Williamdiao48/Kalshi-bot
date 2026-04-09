"""Match historical Kalshi markets to GDELT news articles.

Two-stage pipeline:

  Stage 2A — Keyword extraction → GDELT query builder
    For each market, parse title + rules and produce a Boolean GDELT query.
    Extracts: quoted terms (exact phrases), named entities (proper nouns /
    acronyms), and action verbs.  Combines with AND.  Attaches the temporal
    window [open_time, resolution_time − 24h].

  Stage 2B — GDELT DOC 2.0 API fetch (article metadata only, no body text)
    Queries GDELT for each market.  Returns article URL, headline, date, and
    source domain.  Body text is NOT fetched here — that is Stage 3
    (fetch_article_bodies.py) which should be run on a home machine.

Usage:
    venv/bin/python scripts/match_news_to_markets.py

Input:
    data/kalshi_markets.jsonl  — output of scrape_kalshi_history.py

Output:
    data/gdelt_article_list.jsonl — one record per (market, article) pair:
        market_ticker, market_title, market_result, market_category,
        resolution_time, gdelt_query, article_url, article_headline,
        article_domain, article_date, hours_before_resolution

Resume behaviour:
    Already-processed market tickers are skipped on restart.  The output file
    is opened in append mode so completed work is never lost.

Environment variables:
    GDELT_DELAY_SECONDS      Delay between GDELT requests (default: 3.0).
    MAX_RECORDS_PER_MARKET   Max articles per market from GDELT (default: 50).
    MAX_MARKETS_PER_CATEGORY Cap on markets queried per kalshi_category (default: 500).
                             Set to 0 to disable. At 3 s/market, 500 cats × 6 = ~25 min.
    GDELT_MAX_RETRIES        Retry attempts on 429 before giving up (default: 3).
    GDELT_RETRY_BASE_S       Base wait seconds on 429 retry, multiplied by attempt (default: 60).
"""

import asyncio
import json
import logging
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
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
INPUT_FILE  = Path(__file__).parent.parent / "data" / "kalshi_resolved_markets.jsonl"
OUTPUT_FILE = Path(__file__).parent.parent / "data" / "gdelt_article_list.jsonl"

GDELT_BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_DELAY: float = float(os.environ.get("GDELT_DELAY_SECONDS", "3.0"))
MAX_RECORDS: int   = min(250, int(os.environ.get("MAX_RECORDS_PER_MARKET", "50")))

# Max markets to query per kalshi_category.  Keeps total runtime manageable
# (~3 s/market × 500 markets = ~25 min).  Set to 0 to disable.
MAX_MARKETS_PER_CATEGORY: int = int(os.environ.get("MAX_MARKETS_PER_CATEGORY", "500"))

# 429 retry config
GDELT_MAX_RETRIES: int   = int(os.environ.get("GDELT_MAX_RETRIES", "3"))
GDELT_RETRY_BASE_S: float = float(os.environ.get("GDELT_RETRY_BASE_S", "60.0"))

# Earliest date GDELT has reliable coverage
_GDELT_EPOCH = datetime(2013, 1, 1, tzinfo=timezone.utc)

# ---------------------------------------------------------------------------
# Stage 2A — Keyword extraction
# ---------------------------------------------------------------------------

# Words that appear capitalised for grammatical reasons, not as proper nouns.
_SENTENCE_CAPS = {
    "will", "would", "could", "should", "the", "a", "an", "this", "that",
    "these", "those", "if", "when", "whether", "by", "in", "on", "at", "to",
    "for", "of", "from", "with", "and", "or", "but", "not", "be", "is",
    "are", "was", "were", "have", "has", "had", "do", "does", "did", "any",
    "all", "both", "either", "no", "yes", "new", "latest", "next", "last",
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "market", "resolves", "resolution", "price",
    # Common sentence-starters that appear capitalised but are not proper nouns
    "there", "some", "before", "since", "during", "after", "its", "their",
    "which", "what", "who", "how", "where", "each", "every", "other", "only",
    "also", "then", "than", "over", "about", "into", "many", "most", "more",
    "less", "several", "another", "such", "even", "much", "various", "certain",
    "given", "based", "due", "prior", "following", "including", "using",
    "being", "having", "within", "without", "between", "through", "against",
    "across", "around", "under", "above", "per", "via", "just", "however",
    "while", "unless", "until", "because", "though", "moreover", "although",
    "thus", "whereas", "once", "very", "quite", "already", "still", "often",
    "always", "never", "sometimes", "usually", "generally", "currently",
    "recently", "finally", "specifically", "simply", "further", "additionally",
    "similarly", "meanwhile", "otherwise", "instead", "rather", "except",
    "despite", "regardless", "according", "must", "please", "note", "see",
    "refer", "refers", "today", "tomorrow", "yesterday", "source", "sources",
    "information", "data", "update", "updates", "background", "criteria",
    "resolve", "resolved", "however", "therefore", "here", "now",
}

# Acronyms that are generic / too common to be useful GDELT terms.
_GENERIC_ACRONYMS = {
    "US", "USA", "UK", "EU", "UN", "AND", "OR", "NOT", "YES", "NO",
    # Timezone / time abbreviations — appear constantly in Manifold rules text
    "PST", "EST", "CST", "MST", "PDT", "EDT", "CDT", "MDT",
    "PT", "ET", "CT", "MT", "UTC", "GMT",
    "AM", "PM",
    # Month abbreviations
    "JAN", "FEB", "MAR", "APR", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
    # Day abbreviations
    "MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN",
}

# Terms that should never appear as GDELT search signals.
# Manifold resolution criteria use template boilerplate that looks like
# proper nouns / acronyms but has no news-search value.
_BOILERPLATE_TERMS: frozenset[str] = frozenset({
    # Manifold template phrases
    "background", "criteria", "resolve", "resolves", "resolved", "resolution",
    "manifold", "coinflip", "precommit", "dagonet", "sourcelang",
    # Generic metadata words
    "update", "updates", "source", "sources", "information", "rules",
    "note", "notes", "please", "check", "see", "refer", "refers",
    # Time/date words too generic for GDELT
    "today", "tomorrow", "yesterday", "date", "time", "period",
    "start", "end", "begin", "close", "deadline", "ended", "started",
    # Prediction market jargon
    "market", "price", "bet", "mana", "profit", "creator", "trader", "traders",
    "bot", "bots", "question", "questions", "answer", "answers",
    # Single-letter-style noise
    "yes", "no", "true", "false",
})

# Political / financial action verbs that are useful search signals.
ACTION_VERBS: frozenset[str] = frozenset({
    "mention", "say", "announce", "sign", "veto", "nominate", "confirm",
    "approve", "reject", "resign", "fire", "appoint", "appoints", "pass",
    "vote", "certify", "invoke", "impose", "sanction", "withdraw", "extend",
    "implement", "deploy", "launch", "release", "complete", "finalize",
    "reach", "acquire", "merge", "file", "sue", "indict", "charge",
    "arrest", "convict", "pardon", "commute", "issue", "declare", "lift",
    "suspend", "block", "allow", "ban", "restrict", "delay", "grant",
    "deny", "accept", "reject", "ratify", "repeal", "replace", "revoke",
    "raise", "cut", "hike", "lower", "increase", "decrease", "halt",
})

_QUOTE_RE     = re.compile(r'["\u2018\u2019\u201c\u201d\u0027]([^"\'`\u2018\u2019\u201c\u201d]{2,40})["\u2018\u2019\u201c\u201d\u0027]')
_PROPER_RE    = re.compile(r'\b([A-Z][a-z]{1,}(?:\s+[A-Z][a-z]{1,})*)\b')  # "Donald Trump"
_ACRONYM_RE   = re.compile(r'\b([A-Z]{2,5})\b')
_WORD_RE      = re.compile(r'\b[a-z]{3,}\b')
_DATE_NUM_RE  = re.compile(r'\b\d[\d/,\-]*\b|\b\d+(st|nd|rd|th)\b', re.IGNORECASE)


def _extract_query_terms(
    title: str,
    rules: str | None,
) -> list[str]:
    """Extract discriminating search terms from a market title and rules.

    Returns a list of at most 5 terms suitable for a GDELT Boolean AND query.
    Quoted terms are returned as exact-phrase strings (e.g. '"tariff"').
    Named entities and acronyms are returned bare.

    Returns empty list if the text yields fewer than 2 useful terms
    (these markets are skipped — their titles are too generic to query safely).
    """
    combined = f"{title or ''} {rules or ''}"

    terms: list[str] = []
    seen_lower: set[str] = set()

    def _add(tok: str, phrase: bool = False) -> None:
        key = tok.lower().strip('"')
        if key and key not in seen_lower and key not in _BOILERPLATE_TERMS:
            seen_lower.add(key)
            terms.append(f'"{tok}"' if phrase else tok)

    # 1. Quoted terms in the market title (highest priority — the author
    #    explicitly named them as the resolution criterion).
    for m in _QUOTE_RE.finditer(title or ""):
        candidate = m.group(1).strip()
        # Skip if it looks like a date or pure number
        if _DATE_NUM_RE.fullmatch(candidate):
            continue
        if len(candidate.split()) <= 4:
            _add(candidate, phrase=True)

    # Remove quoted sections from further processing to avoid double-counting.
    clean = _QUOTE_RE.sub(" ", combined)
    clean = _DATE_NUM_RE.sub(" ", clean)

    # 2. Multi-word proper nouns ("Donald Trump", "Pfizer Inc")
    for m in _PROPER_RE.finditer(clean):
        candidate = m.group(1).strip()
        # Strip leading sentence-opening words ("Will Trump" → "Trump")
        parts = candidate.split()
        while parts and parts[0].lower() in _SENTENCE_CAPS:
            parts.pop(0)
        if not parts:
            continue
        candidate = " ".join(parts)
        if candidate.lower() not in _SENTENCE_CAPS and len(candidate) > 2:
            if " " in candidate:
                _add(candidate, phrase=True)   # multi-word → exact phrase
            else:
                _add(candidate)

    # 3. Acronyms (FDA, FBI, NATO, etc.)
    for m in _ACRONYM_RE.finditer(clean):
        acr = m.group(1)
        if acr not in _GENERIC_ACRONYMS:
            _add(acr)

    # 4. Action verbs — add at most one to narrow topic without over-constraining
    words_lower = _WORD_RE.findall(combined.lower())
    for w in words_lower:
        if w in ACTION_VERBS:
            _add(w)
            break   # one action verb is enough

    return terms[:5]


def _build_gdelt_query(terms: list[str]) -> str | None:
    """Combine terms into a Boolean GDELT AND query.

    Returns None if fewer than 2 terms (query would be too broad).
    Always appends sourcelang:english to exclude non-English sources.
    """
    if len(terms) < 2:
        return None
    return " AND ".join(terms) + " sourcelang:english"


def _gdelt_ts(dt: datetime) -> str:
    """Format a datetime as GDELT's YYYYMMDDHHMMSS timestamp."""
    return dt.strftime("%Y%m%d%H%M%S")


def _parse_iso(ts: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp to a UTC-aware datetime."""
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _parse_gdelt_date(seendate: str) -> datetime | None:
    """Parse GDELT's seendate format: '20260323T194500Z'."""
    try:
        return datetime.strptime(seendate, "%Y%m%dT%H%M%SZ").replace(
            tzinfo=timezone.utc
        )
    except (ValueError, TypeError):
        return None


def _build_time_window(
    open_time: str | None,
    resolution_time: str | None,
) -> tuple[str, str] | None:
    """Return (gdelt_start, gdelt_end) strings or None if window is invalid.

    End = resolution_time − 24h (temporal leakage guard).
    Start is clamped to _GDELT_EPOCH.
    Returns None if the resulting window is < 12 hours wide.
    """
    start_dt = _parse_iso(open_time)
    res_dt   = _parse_iso(resolution_time)
    if start_dt is None or res_dt is None:
        return None

    end_dt = res_dt - timedelta(hours=24)
    start_dt = max(start_dt, _GDELT_EPOCH)

    if (end_dt - start_dt).total_seconds() < 12 * 3600:
        return None  # window too narrow to find meaningful articles

    return _gdelt_ts(start_dt), _gdelt_ts(end_dt)


def _plan_query(market: dict[str, Any]) -> dict[str, Any] | None:
    """Build a complete query plan for one market.

    Returns None if the market should be skipped (bad timestamps, too few
    terms, or window too narrow).
    """
    ticker      = market.get("ticker", "")
    title       = market.get("title", "")
    rules       = market.get("rules_primary", "")
    resolution  = market.get("resolution_time")
    open_time   = market.get("open_time")

    window = _build_time_window(open_time, resolution)
    if window is None:
        return None

    terms = _extract_query_terms(title, rules)
    query = _build_gdelt_query(terms)
    if query is None:
        return None

    return {
        "market_ticker":    ticker,
        "market_title":     title,
        "market_result":    market.get("result"),
        "market_category":  market.get("category", "other"),
        "kalshi_category":  market.get("kalshi_category", "other"),
        "resolution_time":  resolution,
        "gdelt_query":      query,
        "gdelt_start":      window[0],
        "gdelt_end":        window[1],
    }


# ---------------------------------------------------------------------------
# Stage 2B — GDELT DOC 2.0 fetch
# ---------------------------------------------------------------------------

async def _fetch_gdelt(
    session: aiohttp.ClientSession,
    plan: dict[str, Any],
) -> list[dict[str, Any]]:
    """Call GDELT DOC 2.0 API for one query plan.

    Returns a list of article records (one per article URL found).
    Returns empty list on error or no results.
    """
    params = {
        "query":         plan["gdelt_query"],
        "mode":          "artlist",
        "maxrecords":    str(MAX_RECORDS),
        "startdatetime": plan["gdelt_start"],
        "enddatetime":   plan["gdelt_end"],
        "format":        "json",
        "sort":          "DateDesc",
    }

    for attempt in range(1, GDELT_MAX_RETRIES + 1):
        try:
            async with session.get(
                GDELT_BASE_URL,
                params=params,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 429:
                    wait = GDELT_RETRY_BASE_S * attempt
                    logging.warning(
                        "GDELT rate-limited (429) — attempt %d/%d, waiting %.0fs.",
                        attempt, GDELT_MAX_RETRIES, wait,
                    )
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                data = await resp.json(content_type=None)
                break  # success
        except asyncio.TimeoutError:
            logging.warning("GDELT timeout for %s (attempt %d)", plan["market_ticker"], attempt)
            if attempt < GDELT_MAX_RETRIES:
                await asyncio.sleep(GDELT_RETRY_BASE_S)
            continue
        except Exception as exc:
            logging.warning("GDELT fetch error for %s: %s", plan["market_ticker"], exc)
            return []
    else:
        logging.error("GDELT gave up on %s after %d attempts.", plan["market_ticker"], GDELT_MAX_RETRIES)
        return []

    raw_articles = data.get("articles") or []
    if not raw_articles:
        return []

    resolution_dt = _parse_iso(plan["resolution_time"])
    records: list[dict[str, Any]] = []

    for art in raw_articles:
        url       = art.get("url", "").strip()
        headline  = art.get("title", "").strip()
        domain    = art.get("domain", "").strip()
        seendate  = art.get("seendate", "")
        language  = art.get("language", "")

        if not url or not headline:
            continue
        if language and language.lower() not in ("english", ""):
            continue

        article_dt = _parse_gdelt_date(seendate)

        # Re-enforce the temporal leakage guard — GDELT's date filter is
        # approximate and occasionally returns articles outside the window.
        if article_dt and resolution_dt:
            if article_dt >= (resolution_dt - timedelta(hours=24)):
                continue  # too close to or after resolution — skip

        hours_before = None
        if article_dt and resolution_dt:
            hours_before = round(
                (resolution_dt - article_dt).total_seconds() / 3600, 1
            )

        records.append({
            "market_ticker":           plan["market_ticker"],
            "market_title":            plan["market_title"],
            "market_result":           plan["market_result"],
            "market_category":         plan["market_category"],
            "kalshi_category":         plan["kalshi_category"],
            "resolution_time":         plan["resolution_time"],
            "gdelt_query":             plan["gdelt_query"],
            "article_url":             url,
            "article_headline":        headline,
            "article_domain":          domain,
            "article_date":            article_dt.isoformat() if article_dt else seendate,
            "hours_before_resolution": hours_before,
        })

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_processed_tickers(output_path: Path) -> set[str]:
    """Return the set of market_tickers already present in the output file."""
    if not output_path.exists():
        return set()
    tickers: set[str] = set()
    for line in output_path.read_text(encoding="utf-8").splitlines():
        try:
            tickers.add(json.loads(line)["market_ticker"])
        except (json.JSONDecodeError, KeyError):
            pass
    return tickers


async def main() -> None:
    if not INPUT_FILE.exists():
        logging.error(
            "Input file not found: %s\n"
            "Run scrape_kalshi_history.py first.",
            INPUT_FILE,
        )
        sys.exit(1)

    _raw_lines = INPUT_FILE.read_text(encoding="utf-8").splitlines()
    markets = []
    _skipped_corrupt = 0
    for _line in _raw_lines:
        if not _line.strip():
            continue
        try:
            markets.append(json.loads(_line))
        except json.JSONDecodeError:
            _skipped_corrupt += 1
    if _skipped_corrupt:
        logging.warning("Skipped %d corrupt / non-JSON line(s) in %s.", _skipped_corrupt, INPUT_FILE)
    logging.info("Loaded %d markets from %s.", len(markets), INPUT_FILE)

    # Build query plans (Stage 2A)
    plans: list[dict[str, Any]] = []
    skipped_no_query = 0
    for m in markets:
        plan = _plan_query(m)
        if plan:
            plans.append(plan)
        else:
            skipped_no_query += 1

    logging.info(
        "Query plans built: %d viable, %d skipped (too generic / narrow window).",
        len(plans), skipped_no_query,
    )

    # Resume: skip tickers already in the output file
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    processed = _load_processed_tickers(OUTPUT_FILE)
    remaining = [p for p in plans if p["market_ticker"] not in processed]
    logging.info(
        "Already processed: %d.  Remaining: %d.",
        len(processed), len(remaining),
    )

    if not remaining:
        logging.info("Nothing to do — all markets already processed.")
        return

    # Cap markets per kalshi_category to keep runtime manageable
    if MAX_MARKETS_PER_CATEGORY > 0:
        cat_counts: dict[str, int] = defaultdict(int)
        capped: list[dict[str, Any]] = []
        for plan in remaining:
            cat = plan.get("kalshi_category", "other")
            if cat_counts[cat] < MAX_MARKETS_PER_CATEGORY:
                capped.append(plan)
                cat_counts[cat] += 1
        if len(capped) < len(remaining):
            logging.info(
                "Per-category cap (%d): %d → %d markets remaining.",
                MAX_MARKETS_PER_CATEGORY, len(remaining), len(capped),
            )
        remaining = capped

    # Stage 2B — fetch articles
    total_articles = 0
    market_stats: dict[str, int] = defaultdict(int)  # category → article count

    with OUTPUT_FILE.open("a", encoding="utf-8") as fh:
        async with aiohttp.ClientSession() as session:
            for i, plan in enumerate(remaining, 1):
                records = await _fetch_gdelt(session, plan)

                for rec in records:
                    fh.write(json.dumps(rec) + "\n")
                fh.flush()

                total_articles += len(records)
                market_stats[plan["market_category"]] += len(records)

                logging.info(
                    "[%d/%d] %-40s  %3d articles  (query: %s)",
                    i, len(remaining),
                    plan["market_ticker"][:40],
                    len(records),
                    plan["gdelt_query"][:60],
                )

                await asyncio.sleep(GDELT_DELAY)

    logging.info("Done. Total articles written: %d", total_articles)
    logging.info("Articles by category:")
    for cat, count in sorted(market_stats.items(), key=lambda x: -x[1]):
        logging.info("  %-25s %d", cat, count)


if __name__ == "__main__":
    asyncio.run(main())
