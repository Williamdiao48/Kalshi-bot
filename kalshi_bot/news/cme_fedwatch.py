"""CME FedWatch implied FOMC meeting probabilities.

Fetches market-implied probabilities for rate cuts, holds, and hikes at the
next FOMC meeting from the CME FedWatch tool's JSON API.  This is derived from
30-day Federal Funds futures prices traded on CME and represents the market
consensus on Fed policy — a far better p_estimate for KXFED* trades than any
generic prior.

API:
    GET https://www.cmegroup.com/CmeWS/mvc/Quotes/getFedWatch
    → JSON array of upcoming FOMC meetings, each with a probability
      distribution across possible rate outcomes.

No API key required.  CME requires a browser-like User-Agent header.

Probabilities are cached for CME_FEDWATCH_CACHE_MINUTES (default 60) to avoid
hammering the endpoint.  On any fetch or parse failure the module returns None
and the trade executor falls back to the default Kelly prior — no harm done.

Result type
-----------
FOMCMeeting.cut_prob   — probability (0.0–1.0) of at least one 25bp cut
FOMCMeeting.hold_prob  — probability of no change
FOMCMeeting.hike_prob  — probability of at least one 25bp hike

(cut_prob + hold_prob + hike_prob ≈ 1.0 within floating-point tolerance)
"""

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import aiohttp

from ..data import DataPoint

_FEDWATCH_URL = "https://www.cmegroup.com/CmeWS/mvc/Quotes/getFedWatch"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.cmegroup.com/markets/interest-rates/cme-fedwatch-tool.html",
}

CME_FEDWATCH_CACHE_MINUTES: float = float(
    os.environ.get("CME_FEDWATCH_CACHE_MINUTES", "60")
)

# Current Fed Funds upper-bound target rate (%).  The FOMC sets this 8×/year;
# update in .env when the Fed changes rates.  Used to compute the expected
# post-meeting rate from CME FedWatch cut/hold/hike probabilities.
# Default: 4.50% (rate as of 2026-03).
CME_CURRENT_RATE: float = float(os.environ.get("CME_CURRENT_RATE", "4.50"))


@dataclass
class FOMCMeeting:
    """Market-implied probability distribution for the next FOMC rate decision."""

    date: str         # ISO date string, e.g. "2026-03-18"
    cut_prob: float   # probability of ≥1 cut  (0.0–1.0)
    hold_prob: float  # probability of no change
    hike_prob: float  # probability of ≥1 hike


# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------

_cached: FOMCMeeting | None = None
_cached_at: float = 0.0


def get_next_meeting() -> FOMCMeeting | None:
    """Return the cached FOMC data without making a network request.

    Returns None if ``fetch_next_meeting`` has not been called yet or the
    last fetch failed.
    """
    return _cached


def _parse_response(data: list | dict) -> FOMCMeeting | None:
    """Parse CME FedWatch JSON into an FOMCMeeting.

    CME returns either:
      - A list of meeting objects (take index 0 = next meeting)
      - A dict with a "meetings" key

    Each meeting object contains a ``probabilities`` list where each entry
    has a ``change`` (int bp, e.g. -25, 0, +25) and ``probability`` (float,
    0–100 scale).  Negative change = cut, zero = hold, positive = hike.
    """
    meetings: list = []
    if isinstance(data, list):
        meetings = data
    elif isinstance(data, dict):
        meetings = data.get("meetings") or data.get("data") or []

    if not meetings:
        logging.warning("CME FedWatch: empty meetings list in response")
        return None

    # First entry is the next upcoming meeting
    mtg = meetings[0]

    # Extract date — may be under "meeting", "date", "meetingDate", etc.
    date_str = (
        mtg.get("meeting") or mtg.get("date") or
        mtg.get("meetingDate") or "unknown"
    )
    # Trim to date part if it's a datetime string
    if "T" in str(date_str):
        date_str = str(date_str).split("T")[0]

    probs_raw = mtg.get("probabilities") or mtg.get("probability") or []

    cut_prob  = 0.0
    hold_prob = 0.0
    hike_prob = 0.0

    for entry in probs_raw:
        change = entry.get("change", 0)
        raw_p  = entry.get("probability", 0.0)
        # Normalize: CME sometimes uses 0–100 scale, sometimes 0–1
        p = float(raw_p) / 100.0 if float(raw_p) > 1.0 else float(raw_p)

        if change < 0:
            cut_prob  += p
        elif change == 0:
            hold_prob += p
        else:
            hike_prob += p

    total = cut_prob + hold_prob + hike_prob
    if total < 0.01:
        logging.warning("CME FedWatch: all probabilities are zero — ignoring")
        return None

    # Normalise to sum to 1 in case of floating-point drift
    cut_prob  /= total
    hold_prob /= total
    hike_prob /= total

    meeting = FOMCMeeting(
        date=str(date_str),
        cut_prob=cut_prob,
        hold_prob=hold_prob,
        hike_prob=hike_prob,
    )
    logging.info(
        "CME FedWatch [%s]: cut=%.1f%%  hold=%.1f%%  hike=%.1f%%",
        meeting.date,
        meeting.cut_prob  * 100,
        meeting.hold_prob * 100,
        meeting.hike_prob * 100,
    )
    return meeting


async def fetch_next_meeting(
    session: aiohttp.ClientSession,
) -> FOMCMeeting | None:
    """Fetch (or return cached) FOMC meeting probabilities.

    Updates the module-level cache.  Returns None on any failure.
    """
    global _cached, _cached_at

    now = time.monotonic()
    if _cached is not None and (now - _cached_at) < CME_FEDWATCH_CACHE_MINUTES * 60:
        return _cached

    try:
        async with session.get(
            _FEDWATCH_URL,
            headers=_HEADERS,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json(content_type=None)
    except aiohttp.ClientResponseError as exc:
        logging.error("CME FedWatch HTTP error %s: %s", exc.status, exc.message)
        return None
    except aiohttp.ClientError as exc:
        logging.error("CME FedWatch request error: %s", exc)
        return None
    except Exception as exc:
        logging.error("CME FedWatch unexpected error: %s", exc)
        return None

    meeting = _parse_response(data)
    if meeting is not None:
        _cached = meeting
        _cached_at = now

    return meeting


async def fetch_fedwatch_datapoints(
    session: aiohttp.ClientSession,
) -> list[DataPoint]:
    """Convert CME FedWatch probabilities into a DataPoint for KXFED matching.

    The KXFED market format is "Will the upper bound of the federal funds rate
    be above X% following the next FOMC meeting?"  The market prices the rate
    AFTER the meeting, not today's rate.  Using today's raw FRED rate ignores
    the probability of a cut or hike.

    This function computes the *expected* post-meeting rate:
        E[rate] = current_rate + Σ(p_i × change_i)
    where each change_i is in percent (e.g. -0.25 for a 25bp cut).

    The expected rate is emitted as a DataPoint with metric="fred_fedfunds"
    and source="cme_fedwatch".  The numeric_matcher then matches it to all
    open KXFED markets exactly as it would a FRED observation, but without
    being subject to the FOMC release-window gate (since FedWatch probabilities
    are always current, not a point-in-time release).

    Returns an empty list if FedWatch data is unavailable.
    """
    meeting = await fetch_next_meeting(session)
    if meeting is None:
        return []

    # Map cut/hold/hike to bp changes.  We only have cut_prob / hold_prob /
    # hike_prob aggregates (not the full distribution), so approximate:
    #   hold  → 0 bp change
    #   cut   → −25 bp  (most cuts are 25bp; larger cuts are rare enough that
    #                     the approximation error is within model noise)
    #   hike  → +25 bp
    expected_change_pct = (
        meeting.cut_prob  * (-0.25)
        + meeting.hold_prob * 0.0
        + meeting.hike_prob * 0.25
    )
    expected_rate = CME_CURRENT_RATE + expected_change_pct

    logging.info(
        "CME FedWatch DataPoint: current_rate=%.2f%%  E[change]=%.3f%%"
        "  E[rate]=%.3f%%  (cut=%.1f%% hold=%.1f%% hike=%.1f%%)",
        CME_CURRENT_RATE, expected_change_pct, expected_rate,
        meeting.cut_prob  * 100,
        meeting.hold_prob * 100,
        meeting.hike_prob * 100,
    )

    return [
        DataPoint(
            source="cme_fedwatch",
            metric="fred_fedfunds",
            value=expected_rate,
            unit="%",
            as_of=datetime.now(timezone.utc).isoformat(),
            metadata={
                "fomc_date":      meeting.date,
                "cut_prob":       meeting.cut_prob,
                "hold_prob":      meeting.hold_prob,
                "hike_prob":      meeting.hike_prob,
                "current_rate":   CME_CURRENT_RATE,
                "expected_change": expected_change_pct,
            },
        )
    ]
