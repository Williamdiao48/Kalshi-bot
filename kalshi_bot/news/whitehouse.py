"""White House Presidential Actions tracker — binary signals for Kalshi EO markets.

Polls the White House briefing room Presidential Actions RSS feed and matches
newly published executive orders, proclamations, and presidential memoranda to
open Kalshi markets that ask "Will Trump sign X?".

Signal encoding
---------------
Seeing an action in the WH feed is a definitive YES signal — the document was
already signed before it was published.

    Affirmative market framing  (default)
        Action published → implied_outcome = "YES"

    Negative market framing
        "Will Trump fail to sign …?" / "Will Congress block …?" etc.
        Action published → implied_outcome = "NO"

    Count-based markets
        "Will Trump sign more than N executive orders?"
        implied_outcome = "UNKNOWN"  (display-only; not traded)

``edge`` reflects the action type:
    1.0 — Executive Order  (highest legal authority; direct market match)
    0.8 — Presidential Memorandum  (binding directive, slightly lower certainty
           of matching a specific Kalshi market question)
    0.7 — Presidential Proclamation  (often ceremonial; may not have a market)
    0.5 — Other / unclassified action

Market detection
----------------
A Kalshi market is classified as a presidential-action market when its title
contains at least one *subject* word from ``_SUBJECT_WORDS`` (Trump, President,
executive, White House, administration, …) AND at least one *predicate* word
from ``_ACTION_PREDICATES``, which covers every natural-language way to express
a presidential signing or directive:

    sign, signs, signed, issue, issues, issued, order, orders, ordered,
    declare, declares, declared, proclaim, proclaims, invoke, invokes,
    impose, imposes, direct, directs, ban, bans, revoke, revokes, enact,
    enacted, implement, implements, sanction, sanctions, authorize, authorizes,
    designate, designates, withdraw, withdraws, establish, establishes,
    create, creates, halt, halts, mandate, mandates, require, requires,
    end, ends, eliminate, eliminates, freeze, freezes, release, expand,
    restrict, rename, terminate, suspend …

Topic matching
--------------
1. Exact EO number match (score = 1.0)
   "Executive Order 14173" or "EO 14173" appears in the market title.

2. Keyword overlap coefficient (score ∈ [0, 1))
   Content words from the EO topic (stripped of type prefix) are compared
   against content words from the market title after stopword removal.
   The overlap coefficient (intersection / min set size) is used so that
   short market titles are not penalised for the EO's verbosity.
   Score ≥ ``WH_MATCH_THRESHOLD`` (default 0.25) required.

3. Topic alias expansion
   Colloquial market language (``immigration``, ``DOGE``, ``tariffs``) is
   expanded to formal EO topic words and vice versa.  A ~50-entry alias table
   covers the most common Trump EO topic clusters:

       Immigration / border, Energy / environment, Trade / tariffs,
       Government reform / DOGE, Defense / national security, Social policy,
       Cryptocurrency / digital assets, Healthcare, Technology / AI,
       Iran / foreign policy, National emergency declarations, …

Source feed
-----------
    GET https://www.whitehouse.gov/briefing-room/presidential-actions/feed/

Standard WordPress RSS 2.0.  No API key required.  Polled every cycle
(``WH_POLL_INTERVAL``, default 60 s).  New documents are deduplicated by
URL so each action fires at most once per process lifetime.

Update notes
------------
No annual updates required — this module is topic-agnostic.  Add entries to
``_TOPIC_ALIASES`` when new recurring EO topic clusters emerge (e.g. AI policy,
space policy).
"""

from __future__ import annotations

import logging
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any

import aiohttp

from ..numeric_matcher import NumericOpportunity


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WH_MATCH_THRESHOLD: float = float(os.environ.get("WH_MATCH_THRESHOLD", "0.25"))
WH_POLL_INTERVAL:   int   = int(os.environ.get("WH_POLL_INTERVAL", "60"))

_FEED_URL = "https://www.whitehouse.gov/briefing-room/presidential-actions/feed/"
_HEADERS  = {"User-Agent": "kalshi-bot/1.0 (educational; contact: user@example.com)"}


# ---------------------------------------------------------------------------
# Presidential action type classification
# ---------------------------------------------------------------------------

# Title prefix patterns → (action_type, edge)
# Checked in order; first match wins.
_ACTION_PREFIXES: list[tuple[re.Pattern[str], str, float]] = [
    (re.compile(r"executive order", re.IGNORECASE),          "executive_order",   1.0),
    (re.compile(r"presidential memorandum", re.IGNORECASE),  "memorandum",        0.8),
    (re.compile(r"presidential proclamation", re.IGNORECASE),"proclamation",      0.7),
    (re.compile(r"\bproclamation\b", re.IGNORECASE),         "proclamation",      0.7),
    (re.compile(r"finding[s]?\b", re.IGNORECASE),            "finding",           0.4),
    (re.compile(r"notice\b", re.IGNORECASE),                 "notice",            0.3),
]

# Prefixes that are never interesting for Kalshi markets (skip early)
_SKIP_ACTION_TYPES: frozenset[str] = frozenset({"finding", "notice"})

# Regex to strip action-type boilerplate from the title to isolate the topic.
# Matches the leading "Executive Order on …", "Proclamation on …", etc.
_TOPIC_STRIP_RE = re.compile(
    r"^(?:executive order|presidential memorandum|presidential proclamation|"
    r"proclamation|presidential notice|finding)\s*"
    r"(?:on|regarding|concerning|relating to|to|for|about|of|entitled|—|–|-|:)?\s*",
    re.IGNORECASE,
)

# Regex to extract a numeric EO number from a title or market title.
# Matches "Executive Order 14173", "EO 14173", "E.O. 14173", "EO14173".
_EO_NUMBER_RE = re.compile(
    r"\b(?:executive\s+order|e\.?o\.?)\s*#?\s*(\d{4,6})\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Market classification word sets
# ---------------------------------------------------------------------------

# Subject: market must contain at least one
_SUBJECT_WORDS: frozenset[str] = frozenset({
    "trump", "president", "presidential", "executive", "whitehouse",
    "administration", "oval", "commander", "potus",
})

# Action predicates: affirmative (action happens → YES for affirmative market)
_AFFIRMATIVE_PREDICATES: frozenset[str] = frozenset({
    # sign / issue
    "sign", "signs", "signed", "signing",
    "issue", "issues", "issued", "issuing",
    "order", "orders", "ordered", "ordering",
    # declare / proclaim
    "declare", "declares", "declared", "declaring", "declaration",
    "proclaim", "proclaims", "proclaimed", "proclaiming",
    # invoke / impose / implement
    "invoke", "invokes", "invoked", "invoking", "invocation",
    "impose", "imposes", "imposed", "imposing",
    "implement", "implements", "implemented", "implementing",
    # direct / mandate / require
    "direct", "directs", "directed", "directing",
    "mandate", "mandates", "mandated", "mandating",
    "require", "requires", "required", "requiring",
    # ban / restrict / freeze
    "ban", "bans", "banned", "banning",
    "restrict", "restricts", "restricted", "restricting", "restriction",
    "freeze", "freezes", "froze", "frozen",
    "halt", "halts", "halted", "halting",
    "suspend", "suspends", "suspended", "suspending", "suspension",
    # create / establish / designate
    "create", "creates", "created", "creating",
    "establish", "establishes", "established", "establishing",
    "designate", "designates", "designated", "designating", "designation",
    # authorize / sanction / withdraw
    "authorize", "authorizes", "authorized", "authorizing", "authorization",
    "sanction", "sanctions", "sanctioned", "sanctioning",
    "withdraw", "withdraws", "withdrew", "withdrawal",
    # revoke / end / eliminate / terminate
    "revoke", "revokes", "revoked", "revoking", "revocation",
    "end", "ends", "ended", "ending",
    "eliminate", "eliminates", "eliminated", "eliminating",
    "terminate", "terminates", "terminated", "terminating",
    "cancel", "cancels", "cancelled", "canceling",
    # expand / release / rename
    "expand", "expands", "expanded", "expanding",
    "release", "releases", "released", "releasing",
    "rename", "renames", "renamed", "renaming",
    "reform", "reforms", "reformed", "reforming",
    # override / reverse
    "override", "overrides", "overrode",
    "reverse", "reverses", "reversed", "reversing",
    "overturn", "overturns", "overturned",
    # use / use executive authority
    "use", "uses",
})

# Negative predicates (trigger inverted implied_outcome)
_NEGATIVE_PREDICATES: frozenset[str] = frozenset({
    "fail", "fails", "failed", "failing",
    "refuse", "refuses", "refused", "refusing",
    "block", "blocks", "blocked", "blocking",
    "prevent", "prevents", "prevented",
    "stop", "stops", "stopped", "stopping",
    "veto", "vetoes", "vetoed",
    "unable", "unwilling",
    "delay", "delays", "delayed",
    "abandon", "abandons", "abandoned",
    "withdraw", "withdraws",  # in market context: "will Trump withdraw the order?"
})

_ALL_PREDICATES: frozenset[str] = _AFFIRMATIVE_PREDICATES | _NEGATIVE_PREDICATES

# Count-based market indicators (→ UNKNOWN implied_outcome)
_COUNT_INDICATORS: frozenset[str] = frozenset({
    "how many", "more than", "at least", "over", "exceed",
    "number of", "total", "count",
})


# ---------------------------------------------------------------------------
# Stopwords for keyword matching
# ---------------------------------------------------------------------------

_STOPWORDS: frozenset[str] = frozenset({
    # articles / prepositions
    "a", "an", "the", "of", "to", "for", "in", "on", "at", "by", "with",
    "into", "from", "as", "up", "about", "against", "through", "during",
    "before", "after", "above", "below", "between", "within", "without",
    "under", "over", "per",
    # conjunctions / modals
    "and", "or", "but", "nor", "so", "yet",
    "will", "would", "could", "should", "shall", "may", "might", "must",
    "can", "do", "does", "did", "have", "has", "had",
    "be", "is", "are", "was", "were", "been", "being",
    # pronouns / determiners
    "it", "its", "this", "that", "these", "those",
    "he", "she", "they", "we", "his", "her", "their", "our",
    "all", "any", "each", "some", "no", "not",
    # action-type words (non-discriminating)
    "executive", "order", "presidential", "president", "trump", "white", "house",
    "administration", "american", "united", "states", "federal", "national",
    "new", "act", "use",
    # time
    "2025", "2026", "2027", "january", "february", "march", "april",
    "may", "june", "july", "august", "september", "october", "november", "december",
    # misc
    "sign", "issue", "via", "related", "certain",
})


# ---------------------------------------------------------------------------
# Topic alias expansion
# Colloquial market terms ↔ formal EO language.  Both directions are covered:
# the alias table is applied to BOTH the EO topic and the market title before
# computing the overlap score.
# ---------------------------------------------------------------------------

_TOPIC_ALIASES: dict[str, list[str]] = {
    # Immigration / border
    "immigration":       ["border", "migrants", "aliens", "deportation", "asylum",
                          "visa", "daca", "dreamers", "sanctuary", "invasion"],
    "border":            ["immigration", "migrants", "wall", "invasion", "cartel",
                          "deportation", "asylum", "securing", "repelling"],
    "invasion":          ["immigration", "border", "migrants", "aliens",
                          "deportation", "securing", "repelling"],
    "repelling":         ["invasion", "border", "immigration", "aliens"],
    "securing":          ["border", "immigration", "security", "protection"],
    "deportation":       ["immigration", "migrants", "removal", "aliens", "dhs"],
    "asylum":            ["immigration", "migrants", "refugee", "protection"],
    "daca":              ["dreamers", "immigration", "deferred", "action"],
    "sanctuary":         ["immigration", "cities", "jurisdictions", "policies"],
    "birthright":        ["citizenship", "immigration", "14th", "amendment"],
    "cartel":            ["drug", "terrorism", "trafficking", "fto", "border"],
    # Energy / environment
    "energy":            ["oil", "gas", "pipeline", "lng", "drilling", "coal",
                          "renewable", "climate", "epa", "emissions", "fossil"],
    "climate":           ["energy", "emissions", "paris", "epa", "environment",
                          "carbon", "net", "zero", "greenhouse"],
    "pipeline":          ["energy", "oil", "gas", "keystone", "infrastructure"],
    "lng":               ["liquefied", "natural", "gas", "energy", "exports"],
    "drilling":          ["energy", "oil", "gas", "offshore", "anwr", "leasing"],
    "paris":             ["climate", "agreement", "accord", "withdrawal", "emissions"],
    "epa":               ["environment", "climate", "emissions", "regulation",
                          "pollution", "clean", "air", "water"],
    # Trade / tariffs
    "tariff":            ["trade", "imports", "china", "canada", "mexico",
                          "section301", "section232", "customs", "duties"],
    "trade":             ["tariff", "imports", "exports", "deficit", "agreement",
                          "usmca", "wto", "commerce"],
    "china":             ["tariff", "trade", "technology", "sanctions", "beijing",
                          "prc", "communist"],
    "sanctions":         ["trade", "foreign", "policy", "china", "russia", "iran",
                          "embargo", "ofac"],
    # Government reform / DOGE
    "doge":              ["efficiency", "government", "spending", "workforce",
                          "federal", "employees", "elon", "musk", "cuts"],
    "efficiency":        ["doge", "government", "reform", "workforce", "spending",
                          "bureaucracy", "streamline"],
    "workforce":         ["federal", "employees", "doge", "government", "cuts",
                          "telework", "remote", "rif"],
    "dei":               ["diversity", "equity", "inclusion", "affirmative",
                          "action", "discrimination"],
    "diversity":         ["dei", "equity", "inclusion", "affirmative", "action"],
    "affirmative":       ["dei", "diversity", "equity", "action", "discrimination"],
    # Defense / national security
    "military":          ["defense", "armed", "forces", "army", "navy", "marines",
                          "national", "security", "pentagon"],
    "defense":           ["military", "armed", "forces", "national", "security",
                          "pentagon", "dod"],
    "emergency":         ["national", "declaration", "border", "disaster",
                          "national security", "insurrection"],
    "national security": ["military", "defense", "intelligence", "nsa", "cia",
                          "homeland", "terrorism"],
    # Crypto / digital assets
    "crypto":            ["bitcoin", "cryptocurrency", "digital", "assets",
                          "blockchain", "reserve", "strategic"],
    "bitcoin":           ["crypto", "cryptocurrency", "digital", "assets"],
    "digital assets":    ["crypto", "bitcoin", "blockchain", "cbdc", "reserve"],
    "strategic reserve": ["crypto", "bitcoin", "digital", "assets", "reserve"],
    # Healthcare
    "healthcare":        ["obamacare", "aca", "medicaid", "medicare", "insurance",
                          "drug", "pricing", "hhs"],
    "obamacare":         ["aca", "affordable", "care", "healthcare", "insurance"],
    "drug pricing":      ["healthcare", "pharmaceutical", "medicare", "insulin",
                          "prescription"],
    # Technology / AI
    "ai":                ["artificial", "intelligence", "technology", "regulation",
                          "machine", "learning", "deepseek"],
    "artificial intelligence": ["ai", "technology", "regulation", "algorithm"],
    "technology":        ["ai", "chips", "semiconductor", "china", "export",
                          "controls", "innovation"],
    # Iran / foreign policy
    "iran":              ["nuclear", "sanctions", "jcpoa", "deal", "middle", "east",
                          "terrorism", "islamic", "republic"],
    "nuclear":           ["iran", "weapons", "treaty", "nonproliferation", "deal"],
    "ukraine":           ["russia", "war", "aid", "ceasefire", "nato", "peace"],
    "russia":            ["ukraine", "sanctions", "nato", "war", "putin"],
    "nato":              ["defense", "alliance", "europe", "military", "article5"],
    # Social policy
    "transgender":       ["gender", "identity", "dei", "bathrooms", "military",
                          "children", "surgery", "transition", "mutilation"],
    "gender":            ["transgender", "identity", "dei", "sex", "children",
                          "mutilation", "surgery"],
    "mutilation":        ["transgender", "gender", "surgery", "children",
                          "chemical", "surgical"],
    "surgical":          ["transgender", "gender", "mutilation", "surgery",
                          "children", "medical"],
    "abortion":          ["reproductive", "rights", "roe", "planned", "parenthood",
                          "pro-life"],
    # Education
    "education":         ["department", "school", "student", "loans", "title",
                          "desegregation"],
    "student loans":     ["education", "forgiveness", "debt", "higher", "department"],
    # Other
    "pardons":           ["pardon", "clemency", "commutation", "jan6", "january6"],
    "jan6":              ["january", "6th", "pardon", "capitol", "riot"],
    "fbi":               ["doj", "weaponization", "law", "enforcement", "federal"],
    "doj":               ["fbi", "justice", "department", "weaponization"],
    "sec":               ["securities", "regulation", "financial", "crypto", "markets"],
    "weaponization":     ["doj", "fbi", "justice", "law", "enforcement",
                          "politicization", "targeting"],
    "unleashing":        ["energy", "oil", "gas", "drilling", "pipeline",
                          "deregulation", "permits"],
    "protecting":        ["children", "safety", "security", "border", "american"],
    "ending":            ["policy", "mandate", "regulation", "program", "dei",
                          "affirmative"],
    "restoring":         ["freedom", "policy", "order", "rights", "rule"],
    "rebuilding":        ["military", "economy", "infrastructure", "america"],
    "strengthening":     ["military", "economy", "border", "security", "alliances"],
    "designating":       ["cartel", "terrorism", "fto", "organization", "sanction"],
    "rescinding":        ["policy", "order", "rule", "regulation", "mandate"],
    "promoting":         ["energy", "economy", "freedom", "growth", "opportunity"],
    "establishing":      ["program", "council", "task", "force", "committee",
                          "policy", "framework"],
    "withdrawing":       ["paris", "treaty", "agreement", "organization", "who"],
}


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

# URLs of WH actions already emitted — never re-emit in this process lifetime
_seen_urls: set[str] = set()


# ---------------------------------------------------------------------------
# Helpers — text cleaning
# ---------------------------------------------------------------------------

_TAG_RE  = re.compile(r"<[^>]+>")
_WS_RE   = re.compile(r"\s+")
import html as _html_module


def _clean(text: str | None) -> str:
    """Strip HTML tags and unescape entities."""
    if not text:
        return ""
    text = _html_module.unescape(text)
    text = _TAG_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# Helpers — action type detection
# ---------------------------------------------------------------------------

def _classify_action(title: str) -> tuple[str, float]:
    """Return (action_type, edge) for a WH briefing room item title.

    Returns ("other", 0.5) when no specific type is detected.
    """
    for pattern, action_type, edge in _ACTION_PREFIXES:
        if pattern.search(title):
            return action_type, edge
    return "other", 0.5


def _extract_topic(title: str) -> str:
    """Strip the action-type boilerplate prefix from a WH title.

    E.g. "Executive Order on Securing the Border and Repelling Invasions"
         → "Securing the Border and Repelling Invasions"
    """
    stripped = _TOPIC_STRIP_RE.sub("", title).strip()
    # Also strip leading dash/colon/em-dash that sometimes follows the prefix
    stripped = re.sub(r"^[—–\-:,]\s*", "", stripped).strip()
    return stripped or title


def _extract_eo_number(text: str) -> str | None:
    """Return the numeric EO number from a title or URL, or None."""
    m = _EO_NUMBER_RE.search(text)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Helpers — market classification
# ---------------------------------------------------------------------------

def _is_presidential_action_market(title: str) -> bool:
    """Return True if a Kalshi market title is a presidential action question.

    Requires at least one subject word (Trump, President, executive, …) AND at
    least one predicate (sign, issue, order, declare, ban, …).  Intentionally
    broad — the match-score gate handles false positives downstream.
    """
    tokens = set(re.findall(r"[a-z]+", title.lower()))
    return bool(tokens & _SUBJECT_WORDS) and bool(tokens & _ALL_PREDICATES)


def _is_count_market(title: str) -> bool:
    """Return True if the market asks about the *count* of executive orders."""
    lower = title.lower()
    return any(ind in lower for ind in _COUNT_INDICATORS)


def _has_negative_framing(title: str) -> bool:
    """Return True if the market asks whether a presidential action will NOT happen."""
    tokens = set(re.findall(r"[a-z]+", title.lower()))
    return bool(tokens & _NEGATIVE_PREDICATES)


# ---------------------------------------------------------------------------
# Helpers — keyword matching
# ---------------------------------------------------------------------------

def _expand_aliases(text: str) -> str:
    """Append alias-expansion words for known EO topic clusters."""
    lower = text.lower()
    extras: list[str] = []
    for key, words in _TOPIC_ALIASES.items():
        if key in lower:
            extras.extend(words)
    if extras:
        return text + " " + " ".join(extras)
    return text


def _content_words(text: str) -> set[str]:
    """Return content words from text after alias expansion and stopword removal."""
    expanded = _expand_aliases(text)
    tokens   = re.findall(r"[a-z0-9]+", expanded.lower())
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 2}


def _overlap_score(set_a: set[str], set_b: set[str]) -> float:
    """Overlap coefficient: intersection / min(|A|, |B|).

    More appropriate than Jaccard for short market titles vs. verbose EO topics.
    """
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / min(len(set_a), len(set_b))


def _match_score(
    eo_topic: str,
    eo_number: str | None,
    market_title: str,
) -> float:
    """Return [0, 1] match score between an EO topic and a Kalshi market title.

    Priority:
        1.0 — EO number appears in market title (e.g. "EO 14173")
        overlap coefficient — keyword overlap after alias expansion
    """
    # EO number exact match
    if eo_number:
        market_num = _extract_eo_number(market_title)
        if market_num == eo_number:
            return 1.0
        # Also check raw number appearance
        if re.search(r"\b" + eo_number + r"\b", market_title):
            return 1.0

    topic_kw  = _content_words(eo_topic)
    market_kw = _content_words(market_title)
    return _overlap_score(topic_kw, market_kw)


# ---------------------------------------------------------------------------
# RSS parsing
# ---------------------------------------------------------------------------

def _parse_wh_feed(xml_bytes: bytes) -> list[dict[str, Any]]:
    """Parse the WH Presidential Actions RSS 2.0 feed.

    Returns a list of dicts with keys:
        url         — canonical link (used for dedup)
        title       — raw item title
        action_type — "executive_order" | "memorandum" | "proclamation" | "other"
        edge        — confidence weight for this action type
        topic       — substantive topic (title with type prefix stripped)
        eo_number   — e.g. "14173" or None
        published   — pubDate string (for display)
        description — item description / summary text
    """
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        logging.warning("WH feed: XML parse error: %s", exc)
        return []

    # Handle namespaced root (<rss> or <feed>)
    channel = root.find("channel")
    if channel is None:
        return []

    items: list[dict[str, Any]] = []
    for item in channel.findall("item"):
        title   = _clean(item.findtext("title"))
        link    = (item.findtext("link") or "").strip()
        pubdate = (item.findtext("pubDate") or "").strip()
        desc    = _clean(item.findtext("description"))

        if not title:
            continue

        action_type, edge = _classify_action(title)
        if action_type in _SKIP_ACTION_TYPES:
            continue

        topic     = _extract_topic(title)
        eo_number = _extract_eo_number(title) or _extract_eo_number(link)

        items.append({
            "url":         link,
            "title":       title,
            "action_type": action_type,
            "edge":        edge,
            "topic":       topic,
            "eo_number":   eo_number,
            "published":   pubdate,
            "description": desc,
        })

    return items


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def find_whitehouse_opportunities(
    session: aiohttp.ClientSession,
    markets: list[dict[str, Any]],
) -> list[NumericOpportunity]:
    """Fetch the WH Presidential Actions feed and match to open Kalshi markets.

    Returns a ``NumericOpportunity`` for each (action, market) pair where the
    match score clears ``WH_MATCH_THRESHOLD``.  ``implied_outcome`` is set
    directly:

        Affirmative market  → "YES"   (action was signed/published)
        Negatively framed   → "NO"    (action happened, but market asks for failure)
        Count-based market  → "UNKNOWN"  (display only, not traded)

    Called every poll cycle.  In-process deduplication prevents re-emitting
    the same presidential action; no separate rate-limit throttle is needed
    since the WH feed itself only changes when a new action is published.

    Args:
        session: Shared aiohttp session.
        markets: Full list of open Kalshi markets from ``fetch_all_markets()``.

    Returns:
        Empty list when: fetch fails, no presidential markets found, or no
        new actions match any open market.
    """
    # Pre-filter Kalshi markets to presidential-action questions
    pa_markets = [m for m in markets if _is_presidential_action_market(m.get("title", ""))]
    if not pa_markets:
        logging.debug("WH: no presidential-action markets found in Kalshi list.")
        return []

    # Fetch the feed
    try:
        async with session.get(
            _FEED_URL,
            headers=_HEADERS,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            resp.raise_for_status()
            xml_bytes = await resp.read()
    except aiohttp.ClientResponseError as exc:
        logging.warning("WH feed HTTP error %s: %s", exc.status, exc.message)
        return []
    except Exception as exc:
        logging.warning("WH feed fetch failed: %s", exc)
        return []

    items = _parse_wh_feed(xml_bytes)
    if not items:
        logging.debug("WH feed: no actionable items parsed.")
        return []

    new_items = [it for it in items if it["url"] not in _seen_urls]
    if not new_items:
        logging.debug("WH feed: %d item(s) fetched, all already seen.", len(items))
        return []

    logging.debug(
        "WH feed: %d new item(s) of %d total.  Matching against %d market(s).",
        len(new_items), len(items), len(pa_markets),
    )

    as_of = datetime.now(timezone.utc).isoformat()
    opps:  list[NumericOpportunity] = []

    for item in new_items:
        topic     = item["topic"]
        eo_number = item["eo_number"]
        edge      = item["edge"]
        action_type = item["action_type"]

        # Find the best-matching Kalshi market
        best_score  = 0.0
        best_market: dict[str, Any] | None = None
        for market in pa_markets:
            score = _match_score(topic, eo_number, market.get("title", ""))
            if score > best_score:
                best_score  = score
                best_market = market

        if best_market is None or best_score < WH_MATCH_THRESHOLD:
            logging.debug(
                "WH: no market matched [%s] (best=%.2f < %.2f) — marking seen.",
                item["title"][:70], best_score, WH_MATCH_THRESHOLD,
            )
            _seen_urls.add(item["url"])
            continue

        ticker = best_market.get("ticker", "")
        title  = best_market.get("title", "")

        # Determine implied_outcome
        if _is_count_market(title):
            implied = "UNKNOWN"
        elif _has_negative_framing(title):
            implied = "NO"
        else:
            implied = "YES"

        opp = NumericOpportunity(
            metric            = f"whitehouse_{action_type}",
            data_value        = 1.0,   # published = signed = certain
            unit              = "",
            source            = "whitehouse",
            as_of             = as_of,
            market_ticker     = ticker,
            market_title      = title,
            current_market_price = best_market.get("last_price", "N/A"),
            direction         = "over",
            strike            = 0.5,
            strike_lo         = None,
            strike_hi         = None,
            implied_outcome   = implied,
            edge              = edge,
        )
        opps.append(opp)
        _seen_urls.add(item["url"])

        logging.info(
            "WH: %s match  '%s'  →  %s  |  market: %s  (score=%.2f)",
            action_type.replace("_", " ").upper(),
            item["title"][:60],
            implied,
            ticker,
            best_score,
        )

    if opps:
        logging.info("WH: %d opportunity(ies) surfaced.", len(opps))

    return opps
