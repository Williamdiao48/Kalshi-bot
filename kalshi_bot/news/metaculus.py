"""Metaculus community forecast fetcher (read-only, no account needed).

Metaculus is a reputation-based forecasting platform where participants have
Brier-score-tracked forecasting records.  The community median probability on
an active question is a well-calibrated signal — independent of and often
ahead of prediction market prices.

When Metaculus disagrees materially with Kalshi, we treat it as evidence of
mispricing and surface it as an opportunity with the Metaculus median as our
p_estimate.

API:
    GET https://www.metaculus.com/api2/questions/
        ?format=json
        &limit=100
        &status=open

Response: JSON object with a ``results`` list. Key fields per question:
    id                      — integer question ID
    title                   — plain-English question text
    community_prediction    — dict; .full.q2 = community median (0.0–1.0)
    number_of_forecasters   — participant count (quality filter)
    close_time              — ISO-8601 resolution timestamp

Only binary YES/NO questions are usable (continuous / numeric range questions
lack a single YES probability).  We detect these by checking that
``community_prediction.full.q2`` is present (absent on non-binary questions).

Thresholds (env-var overridable):
    META_MIN_DIVERGENCE   Minimum |meta_p − kalshi_p| to surface.  Default 0.20.
    META_MIN_FORECASTERS  Minimum community participants.  Fewer = noisy.  Default 20.
    META_MIN_MATCH_SCORE  Minimum Jaccard keyword similarity.  Default 0.20.
"""

import logging
import os
from dataclasses import dataclass

import aiohttp

_API_URL = "https://www.metaculus.com/api2/questions/"
_FETCH_LIMIT = 300   # broad enough to surface well-forecasted questions across topics

META_MIN_DIVERGENCE: float  = float(os.environ.get("META_MIN_DIVERGENCE",  "0.20"))
META_MIN_FORECASTERS: int   = int(os.environ.get("META_MIN_FORECASTERS",   "20"))
META_MIN_MATCH_SCORE: float = float(os.environ.get("META_MIN_MATCH_SCORE", "0.20"))


@dataclass
class MetaculusQuestion:
    """A single Metaculus binary question with community forecast."""

    question_id:  str
    title:        str
    p_yes:        float   # community median probability (0.0–1.0)
    forecasters:  int     # number of forecasters (higher = more reliable)
    close_time:   str     # ISO-8601 resolution timestamp


async def fetch_questions(session: aiohttp.ClientSession) -> list[MetaculusQuestion]:
    """Fetch active binary Metaculus questions with community forecasts.

    Returns an empty list on any fetch or parse failure.
    """
    params = {
        "format":   "json",
        "limit":    str(_FETCH_LIMIT),
        "status":   "open",
        # Sort by forecaster count descending so the most-vetted questions
        # come first and fill the limit quota before less-reliable ones.
        "ordering": "-number_of_forecasters",
    }
    try:
        async with session.get(
            _API_URL,
            params=params,
            timeout=aiohttp.ClientTimeout(total=15),
            headers={"User-Agent": "kalshi-bot research@example.com"},
        ) as resp:
            resp.raise_for_status()
            data = await resp.json(content_type=None)
    except aiohttp.ClientResponseError as exc:
        logging.error("Metaculus HTTP error %s: %s", exc.status, exc.message)
        return []
    except aiohttp.ClientError as exc:
        logging.error("Metaculus request error: %s", exc)
        return []

    raw_list = data.get("results", []) if isinstance(data, dict) else data
    if not isinstance(raw_list, list):
        logging.error("Metaculus: unexpected response shape")
        return []

    questions: list[MetaculusQuestion] = []
    for item in raw_list:
        try:
            # community_prediction is absent or null before any forecasts exist;
            # q2 (median) is absent on non-binary (continuous) questions.
            cp   = item.get("community_prediction") or {}
            full = cp.get("full") or {}
            q2   = full.get("q2")
            if q2 is None:
                continue

            p_yes = float(q2)
            # Metaculus uses 0–1 scale, but guard against occasional 0–100
            if p_yes > 1.0:
                p_yes /= 100.0

            n_forecasters = int(item.get("number_of_forecasters") or 0)
            if n_forecasters < META_MIN_FORECASTERS:
                continue

            questions.append(MetaculusQuestion(
                question_id=str(item.get("id", "")),
                title=item.get("title", ""),
                p_yes=p_yes,
                forecasters=n_forecasters,
                close_time=str(item.get("close_time") or ""),
            ))
        except (ValueError, TypeError):
            continue

    logging.info("Metaculus: %d active binary question(s) fetched.", len(questions))
    return questions
