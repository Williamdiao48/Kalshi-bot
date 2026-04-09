"""Match historical markets to GDELT news articles via BigQuery GKG table.

Drop-in replacement for match_news_to_markets.py that uses the GDELT 2.0
Global Knowledge Graph (GKG) table in BigQuery instead of the rate-limited
GDELT DOC 2.0 HTTP API.

Advantages over the HTTP API approach:
  - No rate limits (3s delay eliminated)
  - 22,000 markets in ~30 minutes vs ~18 hours
  - Same GDELT data source, updated every 15 minutes
  - Entity matching (GKG AllNames / V2Persons / V2Organizations) is equally
    good or better than full-text search for entity-heavy prediction markets

Output format is identical to match_news_to_markets.py so all downstream
scripts (fetch_article_bodies.py, label_heuristic.py, etc.) work unchanged.

Setup (~15 minutes, one-time):
  1. Create a free Google Cloud project at console.cloud.google.com
  2. Enable the BigQuery API
  3. IAM → Service Accounts → Create → grant "BigQuery Job User" + "BigQuery
     Data Viewer" roles → download JSON key
  4. Add to .env:  GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
                   GCP_PROJECT_ID=your-project-id
  5. pip install google-cloud-bigquery

Cost estimate:
  GKG table is date-partitioned; each quarterly batch scans ~12 GB.
  40 quarters (2015–2024) × 12 GB = ~480 GB total — within the 1 TB/month
  BigQuery free tier.  Subsequent runs cost $0 for already-processed tickers.

Usage:
    venv/bin/python scripts/match_news_bq.py

Environment variables:
    GOOGLE_APPLICATION_CREDENTIALS   Path to GCP service account JSON key.
    GCP_PROJECT_ID                   GCP project to bill BigQuery jobs to.
    MAX_ARTICLES_PER_MARKET          Cap per market (default: 250).
    DRY_RUN                          If "1", print query plans without running.
    MAX_QUARTERS                     Stop after N quarters (smoke-test mode).
"""

import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

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

GCP_PROJECT_ID    = os.environ.get("GCP_PROJECT_ID", "")
MAX_ARTICLES: int = int(os.environ.get("MAX_ARTICLES_PER_MARKET", "250"))
DRY_RUN: bool     = os.environ.get("DRY_RUN", "0") == "1"
MAX_QUARTERS: int = int(os.environ.get("MAX_QUARTERS", "0"))  # 0 = unlimited
# Max markets per BigQuery job — large UNION ALLs can exceed BQ query size limits.
# Quarters with more markets than this are split into multiple jobs automatically.
CHUNK_SIZE: int   = int(os.environ.get("BQ_CHUNK_SIZE", "400"))

# Skip markets whose resolution window ends in the future — GDELT won't have
# articles for them yet and they waste scan budget.
_NOW = datetime.now(tz=timezone.utc)

# GDELT GKG only has reliable English web coverage back to ~2013.
_GDELT_EPOCH = datetime(2013, 1, 1, tzinfo=timezone.utc)

# BQ table — partitioned version scans far less data than the unpartitioned one.
_GKG_TABLE = "gdelt-bq.gdeltv2.gkg_partitioned"


# ---------------------------------------------------------------------------
# Term / pattern extraction (adapted from match_news_to_markets.py)
# ---------------------------------------------------------------------------

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

_GENERIC_ACRONYMS = {
    "US", "USA", "UK", "EU", "UN", "AND", "OR", "NOT", "YES", "NO",
    "PST", "EST", "CST", "MST", "PDT", "EDT", "CDT", "MDT",
    "PT", "ET", "CT", "MT", "UTC", "GMT", "AM", "PM",
    "JAN", "FEB", "MAR", "APR", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
    "MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN",
}

_BOILERPLATE_TERMS: frozenset[str] = frozenset({
    "background", "criteria", "resolve", "resolves", "resolved", "resolution",
    "manifold", "coinflip", "precommit", "dagonet", "sourcelang",
    "update", "updates", "source", "sources", "information", "rules",
    "note", "notes", "please", "check", "see", "refer", "refers",
    "today", "tomorrow", "yesterday", "date", "time", "period",
    "start", "end", "begin", "close", "deadline", "ended", "started",
    "market", "price", "bet", "mana", "profit", "creator", "trader", "traders",
    "bot", "bots", "question", "questions", "answer", "answers",
    "yes", "no", "true", "false",
})

_QUOTE_RE    = re.compile(r'["\u2018\u2019\u201c\u201d\u0027]([^"\'`\u2018\u2019\u201c\u201d]{2,40})["\u2018\u2019\u201c\u201d\u0027]')
_PROPER_RE   = re.compile(r'\b([A-Z][a-z]{1,}(?:\s+[A-Z][a-z]{1,})*)\b')
_ACRONYM_RE  = re.compile(r'\b([A-Z]{2,5})\b')
_DATE_NUM_RE = re.compile(r'\b\d[\d/,\-]*\b|\b\d+(st|nd|rd|th)\b', re.IGNORECASE)


def _extract_entity_patterns(title: str, rules: str | None) -> list[str]:
    """Extract entity search patterns for BigQuery REGEXP_CONTAINS(AllNames, ...).

    Returns a list of at most 5 case-insensitive regex patterns.
    Action verbs are excluded — they're not in GDELT's entity fields.
    Returns empty list if fewer than 2 patterns found (market too generic).
    """
    combined = f"{title or ''} {rules or ''}"
    patterns: list[str] = []
    seen_lower: set[str] = set()

    def _add(term: str) -> None:
        key = term.lower().strip('"')
        if key and key not in seen_lower and key not in _BOILERPLATE_TERMS:
            seen_lower.add(key)
            # Escape regex metacharacters in the literal term, then wrap
            escaped = re.escape(term)
            patterns.append(f"(?i){escaped}")

    # 1. Quoted terms from title
    for m in _QUOTE_RE.finditer(title or ""):
        candidate = m.group(1).strip()
        if _DATE_NUM_RE.fullmatch(candidate):
            continue
        if len(candidate.split()) <= 4:
            _add(candidate)

    clean = _QUOTE_RE.sub(" ", combined)
    clean = _DATE_NUM_RE.sub(" ", clean)

    # 2. Multi-word proper nouns
    for m in _PROPER_RE.finditer(clean):
        candidate = m.group(1).strip()
        parts = candidate.split()
        while parts and parts[0].lower() in _SENTENCE_CAPS:
            parts.pop(0)
        if not parts:
            continue
        candidate = " ".join(parts)
        if candidate.lower() not in _SENTENCE_CAPS and len(candidate) > 2:
            _add(candidate)

    # 3. Acronyms (FDA, NATO, FBI, etc.)
    for m in _ACRONYM_RE.finditer(clean):
        acr = m.group(1)
        if acr not in _GENERIC_ACRONYMS:
            _add(acr)

    # Note: action verbs intentionally dropped — not present in GKG AllNames

    return patterns[:5]


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _gdelt_int(dt: datetime) -> int:
    """Convert datetime to GDELT DATE integer (YYYYMMDDHHMMSS)."""
    return int(dt.strftime("%Y%m%d%H%M%S"))


def _parse_gdelt_int(date_int: int | None) -> datetime | None:
    """Parse GDELT DATE integer back to a UTC datetime."""
    if date_int is None:
        return None
    try:
        return datetime.strptime(str(date_int), "%Y%m%d%H%M%S").replace(
            tzinfo=timezone.utc
        )
    except (ValueError, TypeError):
        return None


def _quarter_start(dt: datetime) -> datetime:
    """Return the first day of dt's calendar quarter, UTC midnight."""
    q_month = ((dt.month - 1) // 3) * 3 + 1
    return datetime(dt.year, q_month, 1, tzinfo=timezone.utc)


def _quarter_end(start: datetime) -> datetime:
    """Return the first day of the next quarter."""
    month = start.month + 3
    year  = start.year + (month - 1) // 12
    month = ((month - 1) % 12) + 1
    return datetime(year, month, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Query plan building
# ---------------------------------------------------------------------------

def _build_plan(market: dict[str, Any]) -> dict[str, Any] | None:
    """Build a BigQuery match plan for one market, or None if unsupported."""
    ticker      = market.get("ticker", "")
    title       = market.get("title", "") or ""
    rules       = market.get("rules_primary")
    open_time   = market.get("open_time")
    res_time    = market.get("resolution_time")

    open_dt = _parse_iso(open_time)
    res_dt  = _parse_iso(res_time)
    if open_dt is None or res_dt is None:
        return None

    end_dt   = res_dt - timedelta(hours=24)  # temporal leakage guard
    start_dt = max(open_dt, _GDELT_EPOCH)

    # Skip markets that haven't resolved yet — GDELT won't have articles
    if end_dt > _NOW:
        return None

    if (end_dt - start_dt).total_seconds() < 12 * 3600:
        return None  # window too narrow

    patterns = _extract_entity_patterns(title, rules)
    if len(patterns) < 2:
        return None  # too generic to query safely

    return {
        "ticker":          ticker,
        "title":           title,
        "result":          market.get("result"),
        "category":        market.get("category", "other"),
        "kalshi_category": market.get("kalshi_category", "other"),
        "resolution_time": res_time,
        "patterns":        patterns,
        "start_dt":        start_dt,
        "end_dt":          end_dt,
        "gdelt_start":     _gdelt_int(start_dt),
        "gdelt_end":       _gdelt_int(end_dt),
    }


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def _load_processed_tickers(path: Path) -> set[str]:
    if not path.exists():
        return set()
    tickers: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            tickers.add(json.loads(line)["market_ticker"])
        except (json.JSONDecodeError, KeyError):
            pass
    return tickers


# ---------------------------------------------------------------------------
# BigQuery execution
# ---------------------------------------------------------------------------

def _escape_bq_string(s: str) -> str:
    """Escape a string literal for embedding inside a BigQuery STRING value."""
    return s.replace("\\", "\\\\").replace("'", "\\'")


def _build_batch_sql(
    plans: list[dict[str, Any]],
    partition_start: datetime,
    partition_end: datetime,
) -> str:
    """Build a single BigQuery SQL statement for a batch of market plans.

    Uses a WITH clause CTE of UNION ALLs so the GKG table is scanned once
    per quarter, with all markets matched in one pass.
    """
    # Build one UNION ALL row per market
    rows: list[str] = []
    for p in plans:
        pats = p["patterns"]
        # Pad to 5 patterns with None sentinel
        while len(pats) < 5:
            pats = pats + [None]
        pat_cols = []
        for pat in pats[:5]:
            if pat is None:
                pat_cols.append("NULL")
            else:
                escaped = _escape_bq_string(pat)
                pat_cols.append(f"'{escaped}'")
        rows.append(
            f"  SELECT '{_escape_bq_string(p['ticker'])}' AS ticker,"
            f" {p['gdelt_start']} AS start_ts,"
            f" {p['gdelt_end']}   AS end_ts,"
            f" {pat_cols[0]} AS pat1,"
            f" {pat_cols[1]} AS pat2,"
            f" {pat_cols[2]} AS pat3,"
            f" {pat_cols[3]} AS pat4,"
            f" {pat_cols[4]} AS pat5"
        )

    union_sql = "\n  UNION ALL\n".join(rows)

    # Partition timestamps for the GKG table filter
    pt_start = partition_start.strftime("%Y-%m-%d")
    # partition_end is exclusive (first day of next quarter)
    pt_end   = (partition_end - timedelta(seconds=1)).strftime("%Y-%m-%d")

    sql = f"""
WITH market_plans AS (
{union_sql}
)
-- GROUP BY (market, url) to deduplicate: the GKG table contains multiple rows
-- per URL (same article ingested at different times). We keep the earliest DATE
-- and any non-null headline found across duplicates.
SELECT
  m.ticker                                                                    AS market_ticker,
  g.DocumentIdentifier                                                        AS article_url,
  ANY_VALUE(g.SourceCommonName)                                               AS article_domain,
  MIN(g.DATE)                                                                 AS date_int,
  ANY_VALUE(REGEXP_EXTRACT(g.Extras, r'<PAGE_TITLE>(.*?)</PAGE_TITLE>'))      AS article_headline
FROM `{_GKG_TABLE}` g
CROSS JOIN market_plans m
WHERE g._PARTITIONTIME BETWEEN TIMESTAMP('{pt_start}') AND TIMESTAMP('{pt_end}')
  AND g.DATE >= m.start_ts
  AND g.DATE <= m.end_ts
  AND g.SourceCollectionIdentifier = 1
  AND g.DocumentIdentifier IS NOT NULL
  AND g.DocumentIdentifier != ''
  AND REGEXP_CONTAINS(g.AllNames, m.pat1)
  AND (m.pat2 IS NULL OR REGEXP_CONTAINS(g.AllNames, m.pat2))
  AND (m.pat3 IS NULL OR REGEXP_CONTAINS(g.AllNames, m.pat3))
  AND (m.pat4 IS NULL OR REGEXP_CONTAINS(g.AllNames, m.pat4))
  AND (m.pat5 IS NULL OR REGEXP_CONTAINS(g.AllNames, m.pat5))
GROUP BY m.ticker, g.DocumentIdentifier
"""
    return sql.strip()


def _run_batch(
    bq_client,
    plans: list[dict[str, Any]],
    partition_start: datetime,
    partition_end: datetime,
    max_articles: int,
) -> list[dict[str, Any]]:
    """Execute one quarterly batch query and return output records."""
    sql = _build_batch_sql(plans, partition_start, partition_end)

    if DRY_RUN:
        logging.info("[DRY RUN] Would execute:\n%s\n", sql[:500])
        return []

    logging.info(
        "Running BQ query: %d markets | partition %s → %s",
        len(plans),
        partition_start.strftime("%Y-%m-%d"),
        partition_end.strftime("%Y-%m-%d"),
    )

    t0 = time.monotonic()
    query_job = bq_client.query(sql)
    rows = list(query_job.result())
    elapsed = time.monotonic() - t0

    bytes_processed = query_job.total_bytes_processed or 0
    logging.info(
        "  → %d rows in %.1fs | %.1f GB scanned | est. cost $%.4f",
        len(rows),
        elapsed,
        bytes_processed / 1e9,
        bytes_processed / 1e12 * 6.25,
    )

    # Index plans by ticker for O(1) lookup
    plan_by_ticker = {p["ticker"]: p for p in plans}

    # Group rows by market_ticker, apply temporal guard, cap at max_articles
    grouped: dict[str, list] = defaultdict(list)
    for row in rows:
        grouped[row.market_ticker].append(row)

    output_records: list[dict[str, Any]] = []
    for ticker, ticker_rows in grouped.items():
        plan = plan_by_ticker.get(ticker)
        if plan is None:
            continue
        res_dt = _parse_iso(plan["resolution_time"])
        kept = 0
        for row in ticker_rows:
            if kept >= max_articles:
                break
            article_dt = _parse_gdelt_int(row.date_int)
            # Re-enforce temporal leakage guard (SQL filter is approximate)
            if article_dt and res_dt:
                if article_dt >= (res_dt - timedelta(hours=24)):
                    continue

            hours_before = None
            if article_dt and res_dt:
                hours_before = round(
                    (res_dt - article_dt).total_seconds() / 3600, 1
                )

            url = (row.article_url or "").strip()
            if not url:
                continue

            output_records.append({
                "market_ticker":           ticker,
                "market_title":            plan["title"],
                "market_result":           plan["result"],
                "market_category":         plan["category"],
                "kalshi_category":         plan["kalshi_category"],
                "resolution_time":         plan["resolution_time"],
                "gdelt_query":             " AND ".join(plan["patterns"][:2]),
                "article_url":             url,
                "article_headline":        (row.article_headline or "").strip(),
                "article_domain":          (row.article_domain or "").strip(),
                "article_date":            article_dt.isoformat() if article_dt else "",
                "hours_before_resolution": hours_before,
            })
            kept += 1

    return output_records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not GCP_PROJECT_ID and not DRY_RUN:
        logging.error(
            "GCP_PROJECT_ID not set.\n"
            "  1. Create a free GCP project at console.cloud.google.com\n"
            "  2. Add GCP_PROJECT_ID=your-project-id to .env"
        )
        sys.exit(1)

    # Import here so the script can be imported/inspected without the package installed
    try:
        from google.cloud import bigquery as bq
    except ImportError:
        logging.error(
            "google-cloud-bigquery not installed.\n"
            "  Run: venv/bin/pip install google-cloud-bigquery"
        )
        sys.exit(1)

    if not INPUT_FILE.exists():
        logging.error("Input file not found: %s", INPUT_FILE)
        sys.exit(1)

    # Load markets
    markets: list[dict[str, Any]] = []
    skipped_corrupt = 0
    for line in INPUT_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            markets.append(json.loads(line))
        except json.JSONDecodeError:
            skipped_corrupt += 1
    if skipped_corrupt:
        logging.warning("Skipped %d corrupt lines in %s.", skipped_corrupt, INPUT_FILE)
    logging.info("Loaded %d markets.", len(markets))

    # Build query plans
    plans: list[dict[str, Any]] = []
    skipped_no_plan = 0
    for m in markets:
        p = _build_plan(m)
        if p:
            plans.append(p)
        else:
            skipped_no_plan += 1
    logging.info(
        "Plans built: %d viable, %d skipped (too generic / bad timestamps).",
        len(plans), skipped_no_plan,
    )

    # Resume: skip already-processed tickers
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    processed = _load_processed_tickers(OUTPUT_FILE)
    remaining = [p for p in plans if p["ticker"] not in processed]
    logging.info("Already processed: %d.  Remaining: %d.", len(processed), len(remaining))

    if not remaining:
        logging.info("Nothing to do — all markets already processed.")
        return

    # Group remaining plans into quarterly buckets (by their window start)
    quarter_buckets: dict[datetime, list[dict[str, Any]]] = defaultdict(list)
    for p in remaining:
        q_start = _quarter_start(p["start_dt"])
        quarter_buckets[q_start].append(p)

    quarters_sorted = sorted(quarter_buckets.keys())
    logging.info("Quarterly batches: %d", len(quarters_sorted))

    if DRY_RUN:
        logging.info("[DRY RUN] Batch overview:")
        total_jobs = 0
        for qs in quarters_sorted:
            qe = _quarter_end(qs)
            n  = len(quarter_buckets[qs])
            chunks = (n + CHUNK_SIZE - 1) // CHUNK_SIZE
            total_jobs += chunks
            logging.info(
                "  %s → %s : %d markets → %d BQ job(s)",
                qs.date(), qe.date(), n, chunks,
            )
        logging.info("Total BQ jobs: %d", total_jobs)
        return

    bq_client = bq.Client(project=GCP_PROJECT_ID)

    total_written = 0
    quarter_count = 0

    with OUTPUT_FILE.open("a", encoding="utf-8") as fh:
        for q_start in quarters_sorted:
            quarter_count += 1
            if MAX_QUARTERS and quarter_count > MAX_QUARTERS:
                logging.info("MAX_QUARTERS=%d reached — stopping.", MAX_QUARTERS)
                break

            q_end = _quarter_end(q_start)
            batch = quarter_buckets[q_start]

            # Split large quarters into chunks to stay under BQ query size limit
            chunks = [batch[i:i + CHUNK_SIZE] for i in range(0, len(batch), CHUNK_SIZE)]
            logging.info(
                "[%d/%d] Quarter %s → %s | %d markets in %d chunk(s)",
                quarter_count, len(quarters_sorted),
                q_start.strftime("%Y-%m-%d"),
                q_end.strftime("%Y-%m-%d"),
                len(batch), len(chunks),
            )

            for ci, chunk in enumerate(chunks, 1):
                try:
                    records = _run_batch(bq_client, chunk, q_start, q_end, MAX_ARTICLES)
                except Exception as exc:
                    logging.error(
                        "Chunk %d/%d of quarter %s failed: %s — skipping.",
                        ci, len(chunks), q_start.date(), exc,
                    )
                    continue

                for rec in records:
                    fh.write(json.dumps(rec) + "\n")
                fh.flush()

                total_written += len(records)
                logging.info(
                    "  Chunk %d/%d: %d records (total so far: %d).",
                    ci, len(chunks), len(records), total_written,
                )

    logging.info("Done. Total article records written: %d", total_written)


if __name__ == "__main__":
    main()
