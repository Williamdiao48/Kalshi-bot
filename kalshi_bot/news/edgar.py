"""SEC EDGAR real-time 8-K filings fetcher.

Polls the EDGAR current-events Atom feed for new 8-K filings (material events).
8-K filings cover: earnings surprises, CEO/CFO changes, acquisitions/divestitures,
bankruptcies, restatements, FDA rulings, and other market-moving events that
Kalshi often has active markets for.

EDGAR publishes filings within ~5 minutes of SEC receipt — faster than most
news wire services pick them up.  No API key required; EDGAR is public.

API:
    GET https://www.sec.gov/cgi-bin/browse-edgar
        ?action=getcurrent&type=8-K&dateb=&owner=include
        &count=40&search_text=&output=atom

Response: Atom 1.0 feed.  Key fields per entry:
    <title>   — "COMPANY NAME (CIK XXXXXXXXXX) (8-K)"
    <summary> — "Filed: DATE ...\nPeriod of Report: DATE\n..."
    <link>    — filing index page on EDGAR
    <id>      — unique accession URL (used for deduplication)
    <updated> — ISO-8601 timestamp of the filing

Only 8-K filings are requested.  The ``count`` parameter fetches the most
recent N filings so we capture any burst of releases (e.g. end of quarter).

Normalized document format (compatible with matcher.py and SeenDocuments):
    {
        "document_number": str,   # accession URL — unique per filing
        "title":           str,   # "COMPANY NAME filed 8-K"
        "abstract":        str,   # parsed summary text (cleaned)
        "html_url":        str,   # link to EDGAR filing index
        "feed_id":         "edgar",
        "_source":         "edgar",
    }
"""

import html
import logging
import re
import xml.etree.ElementTree as ET
from typing import Any

import aiohttp

_EDGAR_URL = (
    "https://www.sec.gov/cgi-bin/browse-edgar"
    "?action=getcurrent&type=8-K&dateb=&owner=include"
    "&count=40&search_text=&output=atom"
)

_HEADERS = {
    "User-Agent": "kalshi-bot research@example.com",
    "Accept": "application/atom+xml, application/xml, text/xml, */*",
}

_ATOM_NS = "http://www.w3.org/2005/Atom"
_TAG_RE = re.compile(r"<[^>]+>")

# Strip CIK and form-type parentheticals from EDGAR titles:
# "APPLE INC (CIK 0000320193) (8-K)" → "APPLE INC"
_CIK_RE = re.compile(r"\s*\(CIK\s+\d+\)\s*", re.IGNORECASE)
_FORM_RE = re.compile(r"\s*\(8-K[^\)]*\)\s*$", re.IGNORECASE)


def _tag(local: str) -> str:
    return f"{{{_ATOM_NS}}}{local}"


def _clean(text: str | None) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    text = _TAG_RE.sub(" ", text)
    return " ".join(text.split())


def _company_name(raw_title: str) -> str:
    """Extract the company name from an EDGAR Atom title string."""
    name = _CIK_RE.sub("", raw_title)
    name = _FORM_RE.sub("", name)
    return name.strip()


def _parse_atom(xml_bytes: bytes) -> list[dict[str, Any]]:
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        logging.error("EDGAR: XML parse error: %s", exc)
        return []

    filings: list[dict[str, Any]] = []
    for entry in root.findall(_tag("entry")):
        title_el   = entry.find(_tag("title"))
        summary_el = entry.find(_tag("summary"))
        id_el      = entry.find(_tag("id"))

        raw_title = _clean(title_el.text if title_el is not None else "")
        summary   = _clean(summary_el.text if summary_el is not None else "")
        uid       = (id_el.text or "").strip() if id_el is not None else ""

        # Extract link href
        link = ""
        for link_el in entry.findall(_tag("link")):
            href = link_el.get("href", "")
            if href:
                link = href
                break

        if not raw_title and not link:
            continue

        company = _company_name(raw_title)
        title   = f"{company} filed 8-K" if company else raw_title

        filings.append({
            "document_number": uid or link,
            "title":           title,
            "abstract":        f"{company}: {summary}" if summary else title,
            "html_url":        link,
            "feed_id":         "edgar",
            "_source":         "edgar",
        })

    return filings


async def fetch_filings(session: aiohttp.ClientSession) -> list[dict[str, Any]]:
    """Fetch the latest 8-K filings from EDGAR's current-events Atom feed.

    Returns a list of normalized filing dicts.  Returns an empty list on any
    fetch or parse failure so the poll cycle continues uninterrupted.
    """
    try:
        async with session.get(
            _EDGAR_URL,
            headers=_HEADERS,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            resp.raise_for_status()
            content = await resp.read()
    except aiohttp.ClientResponseError as exc:
        logging.error("EDGAR HTTP error %s: %s", exc.status, exc.message)
        return []
    except aiohttp.ClientError as exc:
        logging.error("EDGAR request error: %s", exc)
        return []

    filings = _parse_atom(content)
    logging.info("EDGAR: %d 8-K filing(s) fetched.", len(filings))
    return filings
