"""Box office weekend chart fetcher.

Scrapes The Numbers weekend chart page for estimated and actual opening-weekend
grosses.  Updated Friday through Monday each week; no API key required.

Data is matched against open Kalshi box office markets in
``kalshi_bot/box_office_matcher.py``.

URL: https://www.the-numbers.com/weekend-chart/

Normalized DataPoint format:
    source   : "box_office"
    metric   : "box_office_" + normalized_title  (e.g. "box_office_avengers_doomsday")
    value    : weekend gross in millions USD (float)
    unit     : "$M"
    as_of    : ISO date of the opening Sunday (e.g. "2026-04-06")
    metadata : {
        "movie_title"   : str,    # raw title from chart
        "rank"          : int,    # chart position this weekend
        "is_estimate"   : bool,   # True before Monday actuals land
        "weekend_date"  : str,    # ISO date
    }

Deduplication key: ``"box_office:{title}:{date}:{gross_rounded}"`` — changes when the
estimate is updated, so a revised Friday estimate re-fires and overwrites the previous one.
A staleness guard (BOX_OFFICE_MAX_STALE_DAYS) drops data older than N days so Monday
actuals don't linger past settlement.
"""

import logging
import os
import re
from datetime import date, datetime
from typing import Any

import aiohttp

from ..data import DataPoint
from ..state import SeenDocuments

_CHART_URL = "https://www.the-numbers.com/weekend-box-office-chart"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; kalshi-bot/1.0; +https://github.com/kalshi-bot)"
    ),
    "Accept": "text/html,application/xhtml+xml",
}

# Drop data older than this many days (prevents stale actuals from matching
# Kalshi markets that have already settled).
BOX_OFFICE_MAX_STALE_DAYS: int = int(os.environ.get("BOX_OFFICE_MAX_STALE_DAYS", "4"))


# ---------------------------------------------------------------------------
# HTML parsing helpers (stdlib only — no BeautifulSoup dependency)
# ---------------------------------------------------------------------------

# The Numbers weekend chart table: each movie row contains cells with:
#   rank | movie title | distributor | weekend gross | ... more columns
#
# We look for the weekend date in the page heading and then extract all
# <tr> rows that contain dollar amounts.

_WEEKEND_DATE_RE = re.compile(
    r"Weekend\s+of\s+(\w+\s+\d{1,2},?\s+\d{4})", re.IGNORECASE
)
_GROSS_CELL_RE = re.compile(r"\$[\d,]+")

# A table row that looks like a movie entry: starts with a rank digit,
# contains a dollar-formatted gross, and has a recognisable title.
_ROW_RE = re.compile(
    r"<tr[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL
)
_TD_RE = re.compile(r"<td[^>]*>(.*?)</td>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_LINK_TEXT_RE = re.compile(r"<a[^>]*>([^<]+)</a>", re.IGNORECASE)


def _strip_tags(html: str) -> str:
    """Remove all HTML tags and normalise whitespace."""
    text = _LINK_TEXT_RE.sub(r"\1", html)  # unwrap links first
    text = _TAG_RE.sub(" ", text)
    return " ".join(text.split())


def _parse_gross(cell: str) -> float | None:
    """Parse a dollar amount cell like '$123,456,789' into millions (float)."""
    m = _GROSS_CELL_RE.search(cell)
    if not m:
        return None
    try:
        raw = m.group(0).replace("$", "").replace(",", "")
        dollars = float(raw)
        return round(dollars / 1_000_000, 3)
    except ValueError:
        return None


def _parse_rank(cell: str) -> int | None:
    text = _strip_tags(cell).strip()
    try:
        return int(text)
    except ValueError:
        return None


def _normalize_title(title: str) -> str:
    """Convert a movie title to a stable metric-key suffix."""
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower())
    return slug.strip("_")


def _parse_weekend_date(html: str) -> str | None:
    """Extract the opening-weekend date from the page heading."""
    m = _WEEKEND_DATE_RE.search(html)
    if not m:
        return None
    raw = m.group(1).strip().rstrip(",")
    for fmt in ("%B %d %Y", "%B %d, %Y"):
        try:
            dt = datetime.strptime(raw, fmt)
            return dt.date().isoformat()
        except ValueError:
            pass
    return None


def _is_stale(weekend_date_str: str) -> bool:
    """Return True if the weekend data is older than BOX_OFFICE_MAX_STALE_DAYS."""
    if BOX_OFFICE_MAX_STALE_DAYS <= 0:
        return False
    try:
        d = date.fromisoformat(weekend_date_str)
        age = (date.today() - d).days
        return age > BOX_OFFICE_MAX_STALE_DAYS
    except ValueError:
        return False


def _parse_chart(html_text: str) -> list[dict[str, Any]]:
    """Parse the HTML of The Numbers weekend chart into a list of movie dicts.

    The Numbers table columns: rank | last-wk-rank | movie title | gross | chg% | theaters

    Returns dicts with keys: title, rank, gross_millions, weekend_date, is_estimate.
    """
    weekend_date = _parse_weekend_date(html_text) or date.today().isoformat()
    is_estimate = date.today().weekday() < 6  # Mon=0; estimate until Sunday actuals land

    results: list[dict[str, Any]] = []
    rank_counter = 0

    for row_m in _ROW_RE.finditer(html_text):
        row_html = row_m.group(1)
        cells = _TD_RE.findall(row_html)
        if len(cells) < 4:
            continue

        rank = _parse_rank(cells[0])
        if rank is None:
            continue

        # Column layout: [0]=rank, [1]=last-wk-rank, [2]=title, [3]=gross, ...
        title_raw = _strip_tags(cells[2]).strip()
        if not title_raw or len(title_raw) < 2:
            continue
        if any(kw in title_raw.lower() for kw in ("distributor", "studio", "gross", "total")):
            continue

        # Gross is in column 3
        gross = _parse_gross(cells[3])
        if gross is None or gross <= 0:
            continue

        rank_counter += 1
        results.append({
            "title": title_raw,
            "rank": rank,
            "gross_millions": gross,
            "weekend_date": weekend_date,
            "is_estimate": is_estimate,
        })

        if rank_counter >= 20:  # top 20 is more than enough
            break

    return results


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

async def fetch_weekend_chart(
    session: aiohttp.ClientSession,
    seen: SeenDocuments,
) -> list[DataPoint]:
    """Fetch The Numbers weekend box office chart and return DataPoints.

    Only returns entries that are (a) not yet seen and (b) not stale.

    Args:
        session: Shared aiohttp session.
        seen:    Shared deduplication store.

    Returns:
        List of DataPoint objects for new or updated box office entries.
    """
    try:
        async with session.get(
            _CHART_URL,
            headers=_HEADERS,
            timeout=aiohttp.ClientTimeout(total=15),
            allow_redirects=True,
        ) as resp:
            resp.raise_for_status()
            html_text = await resp.text(errors="replace")
    except aiohttp.ClientResponseError as exc:
        logging.error("Box office: HTTP %s — %s", exc.status, exc.message)
        return []
    except aiohttp.ClientError as exc:
        logging.error("Box office: request error — %s", exc)
        return []

    movies = _parse_chart(html_text)
    if not movies:
        logging.warning("Box office: no movies parsed from weekend chart.")
        return []

    points: list[DataPoint] = []
    new_keys: list[str] = []

    for movie in movies:
        weekend_date = movie["weekend_date"]
        if _is_stale(weekend_date):
            logging.debug("Box office: skipping stale entry %s (%s)", movie["title"], weekend_date)
            continue

        gross_rounded = round(movie["gross_millions"], 1)
        dedup_key = f"box_office:{movie['title']}:{weekend_date}:{gross_rounded}"

        if seen.contains(dedup_key):
            continue

        points.append(DataPoint(
            source="box_office",
            metric="box_office_" + _normalize_title(movie["title"]),
            value=movie["gross_millions"],
            unit="$M",
            as_of=weekend_date,
            metadata={
                "movie_title": movie["title"],
                "rank": movie["rank"],
                "is_estimate": movie["is_estimate"],
                "weekend_date": weekend_date,
            },
        ))
        new_keys.append(dedup_key)

    if new_keys:
        seen.mark_many(new_keys, source="box_office")

    logging.info(
        "Box office: %d movie(s) on chart, %d new/updated DataPoint(s).",
        len(movies), len(points),
    )
    return points
