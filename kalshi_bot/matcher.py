from dataclasses import dataclass, field
from typing import Any


@dataclass
class Opportunity:
    """A potential trading signal derived from a news document matching a Kalshi market."""

    topic: str
    market_ticker: str
    market_title: str
    current_price: Any  # cents or "N/A"
    doc_id: str
    doc_title: str
    doc_url: str
    source: str = "federal_register"
    matched_terms: list[str] = field(default_factory=list)
    n_alternatives: int = 0  # other markets that matched this same (doc, term) pair


def _extract_searchable_text(doc: dict[str, Any]) -> str:
    title = doc.get("title", "")
    abstract = doc.get("abstract") or ""
    return f"{title} {abstract}".lower()


def _price_distance_from_50(market: dict[str, Any]) -> float:
    """Return |last_price - 50| for sorting; missing/unparseable prices sort last."""
    price = market.get("last_price")
    try:
        return abs(float(price) - 50)
    except (TypeError, ValueError):
        return float("inf")


def find_opportunities(
    docs: list[dict[str, Any]],
    markets: list[dict[str, Any]],
    topics: list[str],
    *,
    require_title_match: bool = True,
) -> list[Opportunity]:
    """Match news documents against open Kalshi markets by topic keywords.

    Strategy:
      1. For each document, collect which topics appear in its title+abstract.
      2. Require each matched topic to also appear in the article title alone —
         this eliminates false positives where a keyword is buried in boilerplate
         abstract text unrelated to the actual story.
      3. For each title-confirmed topic, find all markets whose title or ticker
         contains that term.
      4. Emit ONE opportunity per (document, topic) pair — the market with price
         closest to 50¢ (most uncertain, most potential edge). The count of
         alternative markets is stored in n_alternatives for reference.

    This deduplication prevents a single article from spawning hundreds of rows
    for the same term (e.g. "Trump" matching 66 KXTRUMPSAY markets).

    Args:
        docs:    List of news document dicts (must have 'title', 'abstract',
                 'html_url', and a unique ID field).
        markets: List of Kalshi market dicts. Should be pre-filtered to the
                 category appropriate for the docs being matched.
        topics:  List of keyword strings to match against (case-insensitive).

    Returns:
        List of Opportunity objects — at most one per (document, topic) pair.
    """
    opportunities: list[Opportunity] = []
    lower_topics = [t.lower() for t in topics]

    for doc in docs:
        doc_title_lower = doc.get("title", "").lower()
        doc_full_text = _extract_searchable_text(doc)

        # Phase 1: topic must appear somewhere in the article (title or abstract)
        body_matched = [t for t in lower_topics if t in doc_full_text]
        if not body_matched:
            continue

        # Phase 2: topic must also appear in the headline specifically.
        # Skipped for sources (e.g. EDGAR) whose title is a structured stub
        # rather than enriched prose — content lives in the abstract instead.
        if require_title_match:
            title_confirmed = [t for t in body_matched if t in doc_title_lower]
            if not title_confirmed:
                continue
        else:
            title_confirmed = body_matched

        doc_id = doc.get("document_number") or doc.get("id") or doc.get("url", "unknown")
        doc_title = doc.get("title", "")
        doc_url = doc.get("html_url", "")
        doc_source = doc.get("feed_name") or doc.get("_source", "unknown")

        # Phase 3: for each confirmed topic, find all matching markets and pick the best one
        for term in title_confirmed:
            matching = [
                m for m in markets
                if term in m.get("title", "").lower() or term in m.get("ticker", "").lower()
            ]
            if not matching:
                continue

            # Best market = closest price to 50¢ (maximum uncertainty = maximum potential edge)
            best = min(matching, key=_price_distance_from_50)

            opportunities.append(
                Opportunity(
                    topic=term.capitalize(),
                    market_ticker=best.get("ticker", ""),
                    market_title=best.get("title", ""),
                    current_price=best.get("last_price", "N/A"),
                    doc_id=str(doc_id),
                    doc_title=doc_title,
                    doc_url=doc_url,
                    source=doc_source,
                    matched_terms=[term],
                    n_alternatives=len(matching) - 1,
                )
            )

    return opportunities
