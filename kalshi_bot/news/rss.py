"""Generic async RSS/Atom feed fetcher.

Parses both RSS 2.0 and Atom 1.0 feeds into a normalized article dict format
that is compatible with the existing keyword matcher and SeenDocuments store.

Feeds included and their relevance to live Kalshi market categories:

  ap_top       AP News Top Stories   → production politics / economics
  ap_politics  AP Politics           → production election / Congress markets
  reuters      Reuters Top News      → production economics / global events
  bbc          BBC News              → broad breaking news
  npr          NPR News              → US politics / policy
  espn_nba     ESPN NBA              → KXNBA* player-prop markets
  espn_top     ESPN Top             → general sports
  billboard    Billboard News        → KXTOPSONG (song name signals)
  politico_congress   Politico Congress    → production Congress markets
  politico_healthcare Politico Healthcare → healthcare policy markets
  politico_defense    Politico Defense    → defense / national security markets
  politico_economy    Politico Economy    → production economics markets
  politico_energy     Politico Energy     → energy / EPA markets
  politico_politics   Politico Politics   → production election / admin markets
  thehill             The Hill            → production Congress / political markets

Normalized article format (compatible with matcher.py and state.py):
  {
    "document_number": str,   # guid or url — used for deduplication
    "title":           str,
    "abstract":        str,   # description / summary (HTML stripped)
    "html_url":        str,
    "publication_date": str,
    "feed_id":         str,
    "feed_name":       str,
  }
"""

import asyncio
import html
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

import aiohttp

# ---------------------------------------------------------------------------
# Feed registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Feed:
    id: str
    name: str
    url: str


FEEDS: list[Feed] = [
    # Breaking / general news
    Feed("ap_top",      "AP News Top Stories", "https://feeds.apnews.com/rss/apf-topnews"),
    Feed("ap_politics", "AP Politics",          "https://feeds.apnews.com/rss/apf-politics"),
    Feed("reuters",     "Reuters Top News",     "https://feeds.reuters.com/reuters/topNews"),
    Feed("bbc",         "BBC News",             "http://feeds.bbci.co.uk/news/rss.xml"),
    Feed("npr",         "NPR News",             "https://feeds.npr.org/1001/rss.xml"),
    # Politics (production Kalshi markets)
    Feed("politico_congress",  "Politico Congress",  "https://rss.politico.com/congress.xml"),
    Feed("politico_healthcare","Politico Healthcare","https://rss.politico.com/healthcare.xml"),
    Feed("politico_defense",   "Politico Defense",   "https://rss.politico.com/defense.xml"),
    Feed("politico_economy",   "Politico Economy",   "https://rss.politico.com/economy.xml"),
    Feed("politico_energy",    "Politico Energy",    "https://rss.politico.com/energy.xml"),
    Feed("politico_politics",  "Politico Politics",  "https://rss.politico.com/politics-news.xml"),
    Feed("thehill",     "The Hill",             "https://thehill.com/feed/"),
    # Sports (KXNBA* markets)
    Feed("espn_nba",    "ESPN NBA",             "https://www.espn.com/espn/rss/nba/news"),
    Feed("espn_top",    "ESPN Top Stories",     "https://www.espn.com/espn/rss/news"),
    # Entertainment (KXTOPSONG markets)
    Feed("billboard",   "Billboard",            "https://www.billboard.com/feed/"),
]


# ---------------------------------------------------------------------------
# XML namespace helpers
# ---------------------------------------------------------------------------

_ATOM_NS = "http://www.w3.org/2005/Atom"
_CONTENT_NS = "http://purl.org/rss/1.0/modules/content/"

_NS = {
    "atom":    _ATOM_NS,
    "content": _CONTENT_NS,
}


def _tag(ns_uri: str, local: str) -> str:
    return f"{{{ns_uri}}}{local}"


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"<[^>]+>")


def _clean(text: str | None) -> str:
    """Strip HTML tags and unescape HTML entities."""
    if not text:
        return ""
    text = html.unescape(text)
    text = _TAG_RE.sub(" ", text)
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# RSS 2.0 parser
# ---------------------------------------------------------------------------

def _parse_rss(root: ET.Element, feed: Feed) -> list[dict[str, Any]]:
    channel = root.find("channel")
    if channel is None:
        return []

    articles = []
    for item in channel.findall("item"):
        title   = _clean(item.findtext("title"))
        desc    = _clean(item.findtext("description"))
        link    = (item.findtext("link") or "").strip()
        guid    = (item.findtext("guid") or link).strip()
        pubdate = (item.findtext("pubDate") or "").strip()

        # Prefer content:encoded over description when available
        content_encoded = item.find(_tag(_CONTENT_NS, "encoded"))
        if content_encoded is not None and content_encoded.text:
            desc = _clean(content_encoded.text)

        if not title and not link:
            continue

        articles.append({
            "document_number": guid or link,
            "title":           title,
            "abstract":        desc,
            "html_url":        link,
            "publication_date": pubdate,
            "feed_id":         feed.id,
            "feed_name":       feed.name,
        })

    return articles


# ---------------------------------------------------------------------------
# Atom 1.0 parser
# ---------------------------------------------------------------------------

def _parse_atom(root: ET.Element, feed: Feed) -> list[dict[str, Any]]:
    articles = []

    for entry in root.findall(_tag(_ATOM_NS, "entry")):
        title_el   = entry.find(_tag(_ATOM_NS, "title"))
        summary_el = entry.find(_tag(_ATOM_NS, "summary"))
        content_el = entry.find(_tag(_ATOM_NS, "content"))
        id_el      = entry.find(_tag(_ATOM_NS, "id"))
        pub_el     = entry.find(_tag(_ATOM_NS, "published")) or entry.find(_tag(_ATOM_NS, "updated"))

        title   = _clean(title_el.text if title_el is not None else "")
        summary = _clean((content_el or summary_el).text if (content_el or summary_el) is not None else "")
        uid     = (id_el.text or "").strip() if id_el is not None else ""
        pubdate = (pub_el.text or "").strip() if pub_el is not None else ""

        # Link: <link href="..."/> (self-closing) or <link>...</link>
        link = ""
        for link_el in entry.findall(_tag(_ATOM_NS, "link")):
            rel = link_el.get("rel", "alternate")
            if rel == "alternate" or not link:
                link = link_el.get("href", link_el.text or "")

        if not title and not link:
            continue

        articles.append({
            "document_number": uid or link,
            "title":           title,
            "abstract":        summary,
            "html_url":        link.strip(),
            "publication_date": pubdate,
            "feed_id":         feed.id,
            "feed_name":       feed.name,
        })

    return articles


# ---------------------------------------------------------------------------
# Format detection + dispatch
# ---------------------------------------------------------------------------

def _parse_feed_xml(xml_bytes: bytes, feed: Feed) -> list[dict[str, Any]]:
    """Parse raw XML bytes into normalized articles. Handles RSS and Atom."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        logging.error("RSS [%s]: XML parse error: %s", feed.name, exc)
        return []

    local_tag = root.tag.split("}")[-1] if "}" in root.tag else root.tag

    if local_tag == "rss":
        return _parse_rss(root, feed)
    if local_tag == "feed":
        return _parse_atom(root, feed)

    # Some feeds use <rdf:RDF> (RSS 1.0) — treat items like RSS 2.0
    items = root.findall(".//{http://purl.org/rss/1.0/}item") or root.findall(".//item")
    if items:
        # Build a fake RSS root and reuse the parser
        fake_rss = ET.Element("rss")
        fake_channel = ET.SubElement(fake_rss, "channel")
        for item in items:
            fake_channel.append(item)
        return _parse_rss(fake_rss, feed)

    logging.warning("RSS [%s]: unrecognised feed format (root tag: %s)", feed.name, root.tag)
    return []


# ---------------------------------------------------------------------------
# Async fetcher
# ---------------------------------------------------------------------------

async def _fetch_feed(
    session: aiohttp.ClientSession, feed: Feed
) -> list[dict[str, Any]]:
    try:
        async with session.get(
            feed.url,
            timeout=aiohttp.ClientTimeout(total=12),
            headers={"User-Agent": "kalshi-bot/1.0 (educational)"},
            allow_redirects=True,
        ) as resp:
            resp.raise_for_status()
            content = await resp.read()
    except aiohttp.ClientResponseError as exc:
        logging.error("RSS [%s]: HTTP %s — %s", feed.name, exc.status, exc.message)
        return []
    except aiohttp.ClientError as exc:
        logging.error("RSS [%s]: request error — %s", feed.name, exc)
        return []

    articles = _parse_feed_xml(content, feed)
    logging.info("RSS [%s]: %d article(s) fetched.", feed.name, len(articles))
    return articles


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

async def fetch_all_feeds(
    session: aiohttp.ClientSession,
    feeds: list[Feed] = FEEDS,
) -> list[dict[str, Any]]:
    """Fetch all RSS/Atom feeds concurrently and return merged article list.

    Args:
        session: Shared aiohttp session.
        feeds:   Feed list to fetch (defaults to the module-level FEEDS list).

    Returns:
        Flat list of normalized article dicts from all feeds that succeeded.
    """
    tasks = [_fetch_feed(session, feed) for feed in feeds]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    articles: list[dict[str, Any]] = []
    for feed, result in zip(feeds, results):
        if isinstance(result, Exception):
            logging.error("RSS [%s]: unexpected error — %s", feed.name, result)
        else:
            articles.extend(result)

    logging.info("RSS: %d total articles across %d feed(s).", len(articles), len(feeds))
    return articles
