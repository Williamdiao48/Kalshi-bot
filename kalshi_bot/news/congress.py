"""Congress.gov vote tracker — binary signals for Kalshi legislative markets.

Polls the congress.gov Votes API for recent roll call votes and matches them
to Kalshi markets that ask whether specific legislation will pass.  Vote
outcomes are binary (PASSED / FAILED), so ``implied_outcome`` is set directly
rather than derived from a numeric strike comparison.

Signal encoding
---------------
The congressional vote result maps to Kalshi market resolution as follows:

    "Will Congress/Senate/House pass X?"  (affirmative framing — the default)
        PASSED → implied_outcome = "YES"   market resolves YES
        FAILED → implied_outcome = "NO"    market resolves NO

    "Will Congress fail to pass X?" / "Will X be blocked?" (negative framing)
        PASSED → implied_outcome = "NO"
        FAILED → implied_outcome = "YES"

``edge`` reflects vote type confidence:
    1.0 — final passage vote ("On Passage of the Bill", conference report)
    0.6 — strong procedural vote (cloture, motion to proceed — implies final
           passage is likely but not yet confirmed)
    0.4 — weaker procedural votes (amendments, tabling motions)

Market detection
----------------
A Kalshi market is classified as a "congressional vote" market when its title
contains at least one *subject* word from ``_SUBJECT_WORDS`` (Congress, Senate,
House, chamber, legislation, bill, act, law, resolution) AND at least one
*predicate* word from ``_PREDICATE_WORDS`` covering every natural-language way
to express legislative passage:

    Affirmative predicates
        pass, passes, passed, approve, approves, approved, vote, votes, voted,
        adopt, adopts, adopted, ratify, ratifies, ratified, confirm, confirms,
        confirmed, enact, enacted, enacted, sign, signs, signed, advance,
        advances, advanced, clear, clears, cleared, authorize, authorizes,
        authorized, override, overrides, overrode, overturn, overturns
    Negative predicates (trigger negative-framing logic)
        fail, fails, failed, reject, rejects, rejected, block, blocks, blocked,
        veto, vetoes, vetoed, stall, stalls, killed, defeat, defeats, defeated

Title matching
--------------
Match score is computed in priority order:

    1. Bill-number match (score = 1.0)
       The normalised bill number (e.g. "HR1234", "S456", "SJRES12") appears
       in the market title in any of its common display variants:
           "H.R. 1234", "HR 1234", "HR1234", "H.R.1234"
           "S. 456",    "S 456",   "S456"
       Any match → score = 1.0 regardless of keyword overlap.

    2. Keyword Jaccard similarity (score ∈ [0, 1))
       Content words are extracted from both the bill title and the market title
       after removing stopwords.  Jaccard similarity (intersection / union) of
       the two word sets gives the overlap score.  Kept when score >=
       ``CONGRESS_MATCH_THRESHOLD`` (default 0.25, env ``CONGRESS_MATCH_THRESHOLD``).

    3. Popular-name aliases
       Common legislative shorthand (e.g. "CHIPS Act", "IRA", "ARP Act") is
       expanded before tokenising to improve keyword overlap.  The alias table
       (``_POPULAR_NAME_ALIASES``) maps short names to the content words that
       appear in official long titles.

API
---
    congress.gov REST API v3
    GET /v3/vote/{congress}/{chamber}/{sessionNumber}
        ?api_key=KEY&limit=20&sort=updateDate+desc&format=json

Set ``CONGRESS_API_KEY`` env var.  Module is silently skipped when absent.

Rate limiting
-------------
The vote list is fetched at most once per ``CONGRESS_POLL_INTERVAL`` seconds
(default 300 s, env ``CONGRESS_POLL_INTERVAL``).  Detail records are only
fetched for votes not yet seen in this process lifetime, so on a quiet day
only 0–2 HTTP calls beyond the list poll are made per cycle.

Recess gating
-------------
``is_congress_in_recess()`` checks today's date against the hardcoded recess
calendar.  During recesses the list fetch is skipped entirely.

    Update annually
    ---------------
    ``_RECESS_PERIODS`` — add the next Congress's recess calendar each January.
    ``CURRENT_CONGRESS`` — increment on January 3 of every odd year when a new
    Congress is seated.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from datetime import date, datetime, timezone
from typing import Any

import aiohttp

from ..numeric_matcher import NumericOpportunity


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONGRESS_API_KEY: str       = os.environ.get("CONGRESS_API_KEY", "")
CONGRESS_MATCH_THRESHOLD: float = float(os.environ.get("CONGRESS_MATCH_THRESHOLD", "0.25"))
CONGRESS_POLL_INTERVAL: int     = int(os.environ.get("CONGRESS_POLL_INTERVAL", "300"))

_BASE_URL       = "https://api.congress.gov/v3"
_VOTE_LIST_LIMIT = 20

# 119th Congress (Jan 3 2025 – Jan 3 2027).  Increment on Jan 3 of odd years.
CURRENT_CONGRESS: int = 119


# ---------------------------------------------------------------------------
# Vote-type classification
# ---------------------------------------------------------------------------

# Questions that indicate a definitive final passage vote → edge 1.0
_PASSAGE_QUESTIONS: frozenset[str] = frozenset({
    "on passage",
    "on passage of the bill",
    "on the passage of the bill",
    "on passage of the joint resolution",
    "on passage of the concurrent resolution",
    "on passage of the resolution",
    "on final passage",
    "on the joint resolution",
    "on agreeing to the resolution",
    "on agreeing to the concurrent resolution",
    "on the conference report",
    "on the conference report on",
    "on engrossment and third reading of the bill",
})

# Questions for procedural votes that strongly predict final passage → edge 0.6
_PROCEDURAL_QUESTIONS: frozenset[str] = frozenset({
    "on cloture",
    "on the cloture motion",
    "on the motion to invoke cloture",
    "on the motion to proceed",
    "on the motion to concur",
    "on the nomination",
    "on the nomination of",
    "shall the nomination",
    "on advancing",
})


# ---------------------------------------------------------------------------
# Market classification word sets
# ---------------------------------------------------------------------------

# Subject: at least one must appear in the market title
_SUBJECT_WORDS: frozenset[str] = frozenset({
    "congress", "congressional", "senate", "senators",
    "house", "representatives", "chamber", "chambers",
    "legislation", "legislative", "lawmakers", "legislature",
    "bill", "bills", "resolution", "resolutions",
    "act", "acts", "law", "laws", "statute",
})

# Affirmative predicates: PASSED → YES, FAILED → NO
_AFFIRMATIVE_PREDICATES: frozenset[str] = frozenset({
    # pass / approval
    "pass", "passes", "passed", "passing",
    "approve", "approves", "approved", "approving", "approval",
    # vote
    "vote", "votes", "voted", "voting",
    # adopt
    "adopt", "adopts", "adopted", "adopting", "adoption",
    # ratify
    "ratify", "ratifies", "ratified", "ratifying", "ratification",
    # confirm
    "confirm", "confirms", "confirmed", "confirming", "confirmation",
    # enact
    "enact", "enacts", "enacted", "enacting", "enactment",
    # sign
    "sign", "signs", "signed", "signing",
    # advance / clear / move
    "advance", "advances", "advanced", "advancing",
    "clear", "clears", "cleared", "clearing",
    # authorize
    "authorize", "authorizes", "authorized", "authorizing", "authorization",
    # override / overturn veto
    "override", "overrides", "overrode", "overriding",
    "overturn", "overturns", "overturned",
    # fund / budget
    "fund", "funds", "funded", "funding",
    "reauthorize", "reauthorizes", "reauthorized",
    # generic "will it happen"
    "agree", "agrees", "agreed",
})

# Negative predicates: PASSED → NO, FAILED → YES
_NEGATIVE_PREDICATES: frozenset[str] = frozenset({
    "fail", "fails", "failed", "failing", "failure",
    "reject", "rejects", "rejected", "rejecting", "rejection",
    "block", "blocks", "blocked", "blocking",
    "veto", "vetoes", "vetoed", "vetoing",
    "stall", "stalls", "stalled", "stalling",
    "kill", "kills", "killed", "killing",
    "defeat", "defeats", "defeated", "defeating",
    "oppose", "opposes", "opposed", "opposing",
    "prevent", "prevents", "prevented",
    "stop", "stops", "stopped", "stopping",
    "derail", "derails", "derailed",
    "table", "tables", "tabled", "tabling",
    "filibuster", "filibusters", "filibustered",
    "sustain",   # "sustain the veto"
})

# All predicates combined (for _is_congress_market check)
_ALL_PREDICATES: frozenset[str] = _AFFIRMATIVE_PREDICATES | _NEGATIVE_PREDICATES


# ---------------------------------------------------------------------------
# Stopwords — removed before keyword Jaccard comparison
# ---------------------------------------------------------------------------

_STOPWORDS: frozenset[str] = frozenset({
    # articles / prepositions
    "a", "an", "the", "of", "to", "for", "in", "on", "at", "by", "with",
    "into", "from", "as", "up", "about", "against", "through", "during",
    "before", "after", "above", "below", "between", "within", "without",
    # conjunctions
    "and", "or", "but", "nor", "so", "yet", "both", "either", "neither",
    # modals / auxiliaries
    "will", "would", "could", "should", "shall", "may", "might", "must",
    "can", "do", "does", "did", "have", "has", "had", "be", "is", "are",
    "was", "were", "been", "being",
    # pronouns
    "it", "its", "this", "that", "these", "those", "he", "she", "they",
    "we", "his", "her", "their", "our",
    # common legislative stopwords that don't discriminate between bills
    "act", "bill", "law", "resolution", "legislation", "legislative",
    "congress", "congressional", "senate", "house", "amendment",
    "vote", "votes", "voted", "pass", "passes", "passed",
    "approve", "approved", "approves",
    # time
    "by", "before", "2025", "2026", "2027",
    # misc
    "not", "no", "any", "all", "new", "other", "such",
})


# ---------------------------------------------------------------------------
# Popular-name alias expansion
# Expands abbreviations / shorthand bill names into content words that are
# likely to appear in the official long title, improving Jaccard overlap.
# Keys are lowercase; values are additional content words to inject.
# ---------------------------------------------------------------------------

_POPULAR_NAME_ALIASES: dict[str, list[str]] = {
    # Economic / infrastructure
    "ira":              ["inflation", "reduction"],
    "inflation reduction act": ["clean", "energy", "climate", "healthcare"],
    "chips act":        ["semiconductor", "manufacturing", "technology"],
    "chips":            ["semiconductor", "manufacturing"],
    "bipartisan infrastructure":  ["roads", "bridges", "broadband", "water"],
    "infrastructure investment":  ["roads", "bridges", "broadband"],
    "arp":              ["american", "rescue", "plan", "relief"],
    "american rescue plan": ["covid", "relief", "stimulus"],
    "cares act":        ["covid", "relief", "coronavirus"],
    # Healthcare
    "aca":              ["affordable", "care", "healthcare", "insurance"],
    "obamacare":        ["affordable", "care", "healthcare"],
    "medicare for all": ["single", "payer", "healthcare"],
    # Defense / security
    "ndaa":             ["defense", "authorization", "military"],
    "patriot act":      ["terrorism", "surveillance", "security"],
    # Finance / banking
    "dodd frank":       ["financial", "regulation", "banking", "reform"],
    "dodd-frank":       ["financial", "regulation", "banking", "reform"],
    # Immigration
    "dream act":        ["immigration", "dreamers", "daca"],
    "daca":             ["immigration", "dreamers"],
    # Budget
    "cr":               ["continuing", "resolution", "spending"],
    "omnibus":          ["spending", "appropriations", "budget"],
    "debt ceiling":     ["debt", "limit", "borrowing", "treasury"],
    # Tax
    "tcja":             ["tax", "cuts", "jobs"],
    "tax cuts and jobs": ["corporate", "individual", "tax", "reform"],
    # Gun / social
    "bipartisan safer communities": ["gun", "safety", "background"],
    # Trade
    "usmca":            ["trade", "mexico", "canada", "nafta"],
    "cafta":            ["trade", "central", "america"],
    # Energy
    "clean energy":     ["solar", "wind", "renewable", "climate"],
    # Other
    "earmark":          ["appropriations", "project", "local"],
}


# ---------------------------------------------------------------------------
# Congressional calendar
# ---------------------------------------------------------------------------

# 119th Congress recess periods (approx.).  Congress may vote on the day of
# return, so the gate opens one day before the listed end date.
# Update each January with the official congressional calendar.
_RECESS_PERIODS: list[tuple[date, date]] = [
    # 2025 — 1st Session
    (date(2025,  2, 17), date(2025,  2, 24)),   # Presidents' Day
    (date(2025,  4, 11), date(2025,  4, 28)),   # Spring
    (date(2025,  5, 26), date(2025,  6,  2)),   # Memorial Day
    (date(2025,  7,  4), date(2025,  7, 14)),   # Independence Day
    (date(2025,  8,  4), date(2025,  9,  7)),   # August
    (date(2025, 10, 13), date(2025, 10, 20)),   # Columbus Day
    (date(2025, 11, 26), date(2025, 12,  1)),   # Thanksgiving
    (date(2025, 12, 19), date(2026,  1,  5)),   # Holiday / new year
    # 2026 — 2nd Session
    (date(2026,  2, 16), date(2026,  2, 23)),   # Presidents' Day
    (date(2026,  4, 10), date(2026,  4, 20)),   # Spring
    (date(2026,  5, 25), date(2026,  6,  1)),   # Memorial Day
    (date(2026,  7,  3), date(2026,  7, 13)),   # Independence Day
    (date(2026,  8,  3), date(2026,  9,  7)),   # August
    (date(2026, 10,  9), date(2026, 10, 19)),   # Columbus Day
    (date(2026, 11, 25), date(2026, 11, 30)),   # Thanksgiving
    (date(2026, 12, 18), date(2027,  1,  4)),   # Holiday
]


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

# Dedup: (chamber, congress, session, roll_number) — never re-emit
_seen_votes: set[tuple[str, int, int, int]] = set()

# Rate-limit: epoch seconds of last vote-list HTTP fetch
_last_list_fetch: float = 0.0


# ---------------------------------------------------------------------------
# Helpers — session / recess
# ---------------------------------------------------------------------------

def _current_session() -> int:
    """Return the current congressional session number (1 or 2).

    Session 1 starts in January of the odd year a Congress is seated.
    Session 2 starts in January of the following even year.
    """
    return 1 if date.today().year % 2 == 1 else 2


def is_congress_in_recess(today: date | None = None) -> bool:
    """Return True if today falls within a known congressional recess period."""
    d = today or date.today()
    return any(start <= d <= end for start, end in _RECESS_PERIODS)


# ---------------------------------------------------------------------------
# Helpers — bill number normalisation
# ---------------------------------------------------------------------------

# Regex: captures bill type prefix and numeric ID
# Matches: H.R. 1234, HR 1234, HR1234, S. 456, S456, H.J.Res. 12,
#          H.Res.78, S.Res. 99, H.Con.Res. 3, S.Con.Res. 4, S.J.Res. 5
_BILL_NUM_RE = re.compile(
    r"\b"
    r"(H\.?\s*J\.?\s*Res\.?|S\.?\s*J\.?\s*Res\.?|"
    r"H\.?\s*Con\.?\s*Res\.?|S\.?\s*Con\.?\s*Res\.?|"
    r"H\.?\s*Res\.?|S\.?\s*Res\.?|H\.?\s*R\.?|S\.?)"
    r"\s*(\d{1,5})\b",
    re.IGNORECASE,
)

# Map normalised prefix → canonical short form
_PREFIX_CANON: dict[str, str] = {
    "HR":     "HR",
    "S":      "S",
    "HJRES":  "HJRES",
    "SJRES":  "SJRES",
    "HCONRES":"HCONRES",
    "SCONRES":"SCONRES",
    "HRES":   "HRES",
    "SRES":   "SRES",
}


def _normalise_bill_number(text: str) -> str | None:
    """Extract and normalise a bill number from free text.

    Returns canonical form (e.g. "HR1234", "SJRES12") or None if not found.
    """
    m = _BILL_NUM_RE.search(text)
    if not m:
        return None
    raw_prefix = re.sub(r"[\s.]", "", m.group(1)).upper()
    number     = m.group(2)
    canonical  = _PREFIX_CANON.get(raw_prefix, raw_prefix)
    return f"{canonical}{number}"


def _bill_number_from_api(bill_type: str, bill_number: str) -> str | None:
    """Build a canonical bill reference from congress.gov API type + number fields."""
    if not bill_type or not bill_number:
        return None
    prefix = re.sub(r"[\s.]", "", bill_type).upper()
    canonical = _PREFIX_CANON.get(prefix, prefix)
    return f"{canonical}{bill_number}"


# ---------------------------------------------------------------------------
# Helpers — market classification
# ---------------------------------------------------------------------------

def _is_congress_market(title: str) -> bool:
    """Return True if a Kalshi market title is a congressional vote question.

    Requires both a subject word (Congress, Senate, House …) AND a predicate
    (pass, approve, fail, veto …).  The check is intentionally broad — a
    low false-positive rate is more important than zero false positives here
    because the downstream match-score gate will discard mismatches.
    """
    tokens = set(re.findall(r"[a-z]+", title.lower()))
    return bool(tokens & _SUBJECT_WORDS) and bool(tokens & _ALL_PREDICATES)


def _has_negative_framing(title: str) -> bool:
    """Return True if the market is phrased as a negative outcome question.

    E.g. "Will the Senate fail to pass X?" or "Will X be blocked?"
    When True, PASSED → NO and FAILED → YES (inverted from default).
    """
    tokens = set(re.findall(r"[a-z]+", title.lower()))
    return bool(tokens & _NEGATIVE_PREDICATES)


# ---------------------------------------------------------------------------
# Helpers — keyword extraction and matching
# ---------------------------------------------------------------------------

def _expand_aliases(text: str) -> str:
    """Append alias-expansion words for known popular bill names."""
    lower = text.lower()
    extras: list[str] = []
    for alias, words in _POPULAR_NAME_ALIASES.items():
        if alias in lower:
            extras.extend(words)
    if extras:
        return text + " " + " ".join(extras)
    return text


def _content_words(text: str) -> set[str]:
    """Tokenise text and return content words after stopword removal."""
    expanded = _expand_aliases(text)
    tokens   = re.findall(r"[a-z0-9]+", expanded.lower())
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 2}


def _overlap_score(set_a: set[str], set_b: set[str]) -> float:
    """Return overlap coefficient: intersection / min(|A|, |B|).

    Preferred over Jaccard for legislative matching because market titles are
    typically short (1–4 content words) while official bill titles are long.
    Jaccard penalises this asymmetry harshly; the overlap coefficient asks
    "are the market's key words all covered by the bill?" — which is the
    right question.

    A threshold of 0.25 means at least 25 % of the shorter title's content
    words must appear in the other title.
    """
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / min(len(set_a), len(set_b))


def _bill_number_in_title(bill_ref: str, market_title: str) -> bool:
    """Return True if the bill reference appears in the market title.

    Checks the canonical form (e.g. "HR1234") and multiple display variants
    to handle spacing and punctuation differences in Kalshi market titles.
    """
    lower = market_title.lower()
    digits_m = re.search(r"\d+", bill_ref)
    if not digits_m:
        return False
    n = digits_m.group()

    # Derive display variants based on the prefix
    prefix = bill_ref[: -len(n)]
    display_variants: list[str] = [bill_ref.lower()]

    _DISPLAY: dict[str, list[str]] = {
        "HR":     [f"h.r. {n}", f"h.r.{n}", f"hr {n}", f"hr. {n}"],
        "S":      [f"s. {n}",   f"s.{n}",    f"s {n}"],
        "HJRES":  [f"h.j.res. {n}", f"h.j. res. {n}", f"hjres {n}"],
        "SJRES":  [f"s.j.res. {n}", f"s.j. res. {n}", f"sjres {n}"],
        "HCONRES":[f"h.con.res. {n}", f"h. con. res. {n}"],
        "SCONRES":[f"s.con.res. {n}", f"s. con. res. {n}"],
        "HRES":   [f"h.res. {n}", f"h. res. {n}", f"hres {n}"],
        "SRES":   [f"s.res. {n}", f"s. res. {n}", f"sres {n}"],
    }
    display_variants.extend(_DISPLAY.get(prefix, []))

    return any(v in lower for v in display_variants)


def _match_score(
    bill_title: str,
    bill_popular_name: str,
    bill_ref: str | None,
    market_title: str,
) -> float:
    """Return a [0, 1] match score between a vote record and a Kalshi market title.

    Priority order:
        1.0  — bill number found in market title (unambiguous)
        Jaccard — keyword overlap of (bill_title + popular_name) vs market_title
    """
    # 1. Bill-number priority: check the API-derived ref, and also scan the
    #    bill title itself for any embedded number (handles verbose titles like
    #    "H.R. 1234, the Foo Act").
    effective_ref = bill_ref or _normalise_bill_number(bill_title)
    if effective_ref and _bill_number_in_title(effective_ref, market_title):
        return 1.0
    # Also check whether the market title itself contains a bill number that
    # matches a number embedded anywhere in the vote's bill title.
    market_ref = _normalise_bill_number(market_title)
    if market_ref and effective_ref and market_ref == effective_ref:
        return 1.0

    # 2. Keyword Jaccard
    vote_text   = f"{bill_title} {bill_popular_name}"
    vote_kw     = _content_words(vote_text)
    market_kw   = _content_words(market_title)
    return _overlap_score(vote_kw, market_kw)


# ---------------------------------------------------------------------------
# Helpers — vote result parsing
# ---------------------------------------------------------------------------

# All strings that congress.gov uses for PASSED / AGREED / CONFIRMED
_PASSED_STRINGS: frozenset[str] = frozenset({
    "passed",
    "agreed to",
    "bill passed",
    "joint resolution passed",
    "concurrent resolution agreed to",
    "resolution agreed to",
    "nomination confirmed",
    "confirmed",
    "amendment agreed to",
    "passed (revised pursuant to section 19 of rule xx)",
    "passed, without objection",
    "passed by voice vote",
    "bill passed - without objection",
})

_FAILED_STRINGS: frozenset[str] = frozenset({
    "failed",
    "not agreed to",
    "bill failed",
    "joint resolution failed",
    "failed of passage",
    "motion rejected",
    "veto sustained",
    "rejected",
    "table",
    "motion to table agreed to",  # tabling = killing the measure
    "tabled",
})


def _parse_result(result_str: str) -> str | None:
    """Normalise a vote result string to 'PASSED' or 'FAILED'.

    Returns None for result strings that cannot be confidently classified
    (e.g. ties, quorum calls, or truly unknown outcomes).
    """
    r = result_str.strip().lower()

    if r in _PASSED_STRINGS:
        return "PASSED"
    if r in _FAILED_STRINGS:
        return "FAILED"

    # Substring heuristics for novel/verbose result strings
    if any(k in r for k in ("passed", "agreed", "confirmed", "enacted", "adopted")):
        return "PASSED"
    if any(k in r for k in ("failed", "rejected", "not agreed", "veto sustained", "table")):
        return "FAILED"

    return None


def _vote_edge(question: str) -> float:
    """Return the edge weight for a vote based on question type.

    1.0 — final passage vote (definitive outcome)
    0.6 — strong procedural vote (cloture, motion to proceed)
    0.4 — other procedural votes
    """
    q = question.strip().lower()
    if any(pq in q for pq in _PASSAGE_QUESTIONS):
        return 1.0
    if any(pq in q for pq in _PROCEDURAL_QUESTIONS):
        return 0.6
    return 0.4


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

_HEADERS = {"User-Agent": "kalshi-bot/1.0 (educational; contact: user@example.com)"}


async def _fetch_vote_list(
    session: aiohttp.ClientSession,
    chamber: str,
    congress: int,
    session_num: int,
) -> list[dict[str, Any]]:
    """Fetch recent vote summaries for one chamber from congress.gov.

    Returns a list of summary dicts (each has a ``url`` pointing to the full
    vote record).  Returns an empty list on any error.
    """
    url = f"{_BASE_URL}/vote/{congress}/{chamber}/{session_num}"
    try:
        async with session.get(
            url,
            params={
                "api_key": CONGRESS_API_KEY,
                "limit":   _VOTE_LIST_LIMIT,
                "sort":    "updateDate+desc",
                "format":  "json",
            },
            headers=_HEADERS,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json(content_type=None)
    except aiohttp.ClientResponseError as exc:
        logging.warning(
            "Congress [%s]: vote list HTTP %s — %s", chamber, exc.status, exc.message
        )
        return []
    except Exception as exc:
        logging.warning("Congress [%s]: vote list fetch failed: %s", chamber, exc)
        return []

    # The API nests votes under different keys depending on the response shape.
    votes = (
        data.get("votes", {}).get("vote")
        or data.get("votes")
        or data.get("vote")
        or []
    )
    return votes if isinstance(votes, list) else []


async def _fetch_vote_detail(
    session: aiohttp.ClientSession,
    vote_url: str,
) -> dict[str, Any] | None:
    """Fetch the full detail record for a single vote from congress.gov.

    Returns the inner ``vote`` object, or None on any error.
    """
    try:
        async with session.get(
            vote_url,
            params={"api_key": CONGRESS_API_KEY, "format": "json"},
            headers=_HEADERS,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json(content_type=None)
    except Exception as exc:
        logging.warning("Congress: vote detail fetch failed (%s): %s", vote_url, exc)
        return None

    return data.get("vote") or data


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def find_congress_opportunities(
    session: aiohttp.ClientSession,
    markets: list[dict[str, Any]],
) -> list[NumericOpportunity]:
    """Fetch recent congressional votes and match them to open Kalshi markets.

    Called every poll cycle.  Internally rate-limited to one list fetch per
    ``CONGRESS_POLL_INTERVAL`` seconds.  Detail records are only fetched for
    votes not yet seen this process lifetime.

    Args:
        session: Shared aiohttp session (no auth needed — congress.gov is public).
        markets: Full list of open Kalshi markets from ``fetch_all_markets()``.

    Returns:
        List of ``NumericOpportunity`` objects with ``implied_outcome`` set
        to "YES" or "NO" based on the vote result and market framing.
        Empty list when: API key absent, Congress in recess, rate-limited,
        no new votes, or no matching markets found.
    """
    global _last_list_fetch

    if not CONGRESS_API_KEY:
        logging.debug("Congress: CONGRESS_API_KEY not set — skipping.")
        return []

    if is_congress_in_recess():
        logging.debug("Congress: in recess — skipping vote fetch.")
        return []

    now = time.monotonic()
    if now - _last_list_fetch < CONGRESS_POLL_INTERVAL:
        logging.debug(
            "Congress: rate-limited (%.0fs until next list fetch).",
            CONGRESS_POLL_INTERVAL - (now - _last_list_fetch),
        )
        return []
    _last_list_fetch = now

    congress_num = CURRENT_CONGRESS
    session_num  = _current_session()

    # Pre-filter Kalshi markets to congressional-vote questions only.
    congress_markets = [m for m in markets if _is_congress_market(m.get("title", ""))]
    if not congress_markets:
        logging.debug("Congress: no legislative markets detected in Kalshi market list.")
        return []

    logging.debug(
        "Congress: %d legislative market(s) to match.  Fetching votes for %dth Congress, session %d.",
        len(congress_markets), congress_num, session_num,
    )

    # Fetch vote summaries for both chambers concurrently.
    house_votes, senate_votes = await asyncio.gather(
        _fetch_vote_list(session, "house",  congress_num, session_num),
        _fetch_vote_list(session, "senate", congress_num, session_num),
    )
    all_summaries = house_votes + senate_votes

    if not all_summaries:
        logging.debug("Congress: no vote summaries returned.")
        return []

    as_of = datetime.now(timezone.utc).isoformat()
    opps:  list[NumericOpportunity] = []

    for summary in all_summaries:
        # Build dedup key.
        chamber     = (summary.get("chamber") or "").lower()
        congress_n  = int(summary.get("congress", 0))
        session_n   = int(summary.get("sessionNumber") or summary.get("session", 0))
        roll_num    = int(summary.get("rollNumber") or summary.get("number", 0))
        vote_key    = (chamber, congress_n, session_n, roll_num)

        if vote_key in _seen_votes:
            continue

        vote_url = summary.get("url", "")
        if not vote_url:
            _seen_votes.add(vote_key)
            continue

        detail = await _fetch_vote_detail(session, vote_url)
        if not detail:
            continue  # Don't mark seen — allow retry next cycle

        # Extract core fields from the detail record.
        result_str   = (detail.get("result") or "").strip()
        question     = (detail.get("question") or "").strip()
        bill_info    = detail.get("bill") or {}
        bill_type    = (bill_info.get("type") or "").strip()
        bill_number  = str(bill_info.get("number") or "").strip()
        bill_title   = (
            bill_info.get("title")
            or detail.get("title")
            or question
            or ""
        ).strip()
        popular_name = (
            bill_info.get("popularTitle")
            or bill_info.get("popularName")
            or ""
        ).strip()

        # Canonical bill reference for exact number matching.
        bill_ref = _bill_number_from_api(bill_type, bill_number)

        # Parse the binary outcome.
        outcome = _parse_result(result_str)
        if outcome is None:
            logging.debug(
                "Congress: vote %s result %r not classifiable — skipping.",
                vote_key, result_str,
            )
            _seen_votes.add(vote_key)
            continue

        edge = _vote_edge(question)

        logging.debug(
            "Congress: new vote %s [%s] %r → %s  bill=%r  edge=%.1f",
            vote_key, question[:60], result_str, outcome, bill_ref or bill_title[:40], edge,
        )

        # Find the best-matching Kalshi market.
        best_score  = 0.0
        best_market: dict[str, Any] | None = None
        for market in congress_markets:
            score = _match_score(
                bill_title, popular_name, bill_ref, market.get("title", "")
            )
            if score > best_score:
                best_score  = score
                best_market = market

        if best_market is None or best_score < CONGRESS_MATCH_THRESHOLD:
            logging.debug(
                "Congress: no market matched [%s%s] (best=%.2f < %.2f).",
                f"{bill_ref} " if bill_ref else "",
                bill_title[:60],
                best_score,
                CONGRESS_MATCH_THRESHOLD,
            )
            _seen_votes.add(vote_key)
            continue

        ticker = best_market.get("ticker", "")
        title  = best_market.get("title", "")

        # Determine implied_outcome from vote result and market framing.
        negative = _has_negative_framing(title)
        if not negative:
            implied = "YES" if outcome == "PASSED" else "NO"
        else:
            implied = "NO" if outcome == "PASSED" else "YES"

        opp = NumericOpportunity(
            metric            = "congress_vote",
            data_value        = 1.0 if outcome == "PASSED" else 0.0,
            unit              = "",
            source            = "congress",
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
        _seen_votes.add(vote_key)

        logging.info(
            "Congress: vote match  %s '%s' → %s  |  market: %s  (score=%.2f  edge=%.1f)",
            outcome,
            (f"{bill_ref} " if bill_ref else "") + bill_title[:50],
            implied,
            ticker,
            best_score,
            edge,
        )

    if opps:
        logging.info("Congress: %d opportunity(ies) surfaced.", len(opps))

    return opps

