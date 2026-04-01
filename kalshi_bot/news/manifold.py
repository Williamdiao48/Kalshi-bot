"""Manifold Markets forecast fetcher (read-only, no account needed).

Manifold is a play-money prediction market.  Because participants use virtual
currency (mana) rather than real money, prices are *less* efficient than
Polymarket or Kalshi — but the platform has a large and active user base, and
on widely-followed events (elections, AI milestones, sports) the crowd signal
is still meaningful.

We treat Manifold as a lower-confidence signal by requiring a higher divergence
threshold (MANI_MIN_DIVERGENCE, default 0.25 vs 0.20 for Polymarket/Metaculus).

API:
    GET https://api.manifold.markets/v0/markets
        ?limit=500
        &sort=last-bet-time
        &filter=open

Response: JSON array of market objects.  Key fields:
    id            — unique market ID string
    question      — plain-English question text
    probability   — current market probability (0.0–1.0) for binary markets
    outcomeType   — "BINARY" | "MULTIPLE_CHOICE" | "FREE_RESPONSE" | "NUMERIC"
    totalLiquidity — total mana in the market (volume quality filter)
    volume        — total mana traded

Only "BINARY" outcome markets are used; multi-choice / free-response skipped.

Thresholds (env-var overridable):
    MANI_MIN_DIVERGENCE   Minimum |mani_p − kalshi_p| to surface.  Default 0.25
                          (higher than Poly/Meta because play money is noisier).
    MANI_MIN_LIQUIDITY    Minimum mana liquidity.  Markets below this threshold
                          have very few active bettors.  Default 1000.
    MANI_MIN_MATCH_SCORE  Minimum Jaccard keyword similarity.  Default 0.20.
    MANI_MAX_PROB         Maximum (and minimum=1-this) allowable Manifold probability.
                          Probabilities beyond this range indicate a frozen, stale,
                          or effectively-resolved Manifold market — not a real signal.
                          Default 0.95 (rejects p > 0.95 or p < 0.05).
"""

import logging
import os
from dataclasses import dataclass

import aiohttp

_API_URL = "https://api.manifold.markets/v0/markets"
_FETCH_LIMIT = 500

MANI_MIN_DIVERGENCE: float  = float(os.environ.get("MANI_MIN_DIVERGENCE",  "0.25"))
# Raised 1000 → 3000: thin markets have too few active bettors and frequently
# produce stale prices that look like arbitrage opportunities.
MANI_MIN_LIQUIDITY:  float  = float(os.environ.get("MANI_MIN_LIQUIDITY",   "3000"))
# Raised 0.20 → 0.35: stricter keyword overlap reduces false topic matches.
MANI_MIN_MATCH_SCORE: float = float(os.environ.get("MANI_MIN_MATCH_SCORE", "0.35"))
MANI_MAX_PROB:        float = float(os.environ.get("MANI_MAX_PROB",        "0.95"))


@dataclass
class ManifoldMarket:
    """A single binary Manifold prediction market."""

    market_id:  str
    question:   str
    p_yes:      float   # current implied probability (0.0–1.0)
    liquidity:  float   # total mana in the market


async def fetch_markets(session: aiohttp.ClientSession) -> list[ManifoldMarket]:
    """Fetch open binary Manifold markets sorted by recent activity.

    Returns an empty list on any fetch or parse failure.
    """
    params = {
        "limit":  str(_FETCH_LIMIT),
        "sort":   "last-bet-time",
        "filter": "open",
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
        logging.error("Manifold HTTP error %s: %s", exc.status, exc.message)
        return []
    except aiohttp.ClientError as exc:
        logging.error("Manifold request error: %s", exc)
        return []

    if not isinstance(data, list):
        logging.error("Manifold: unexpected response shape")
        return []

    markets: list[ManifoldMarket] = []
    for item in data:
        try:
            if item.get("outcomeType") != "BINARY":
                continue

            probability = item.get("probability")
            if probability is None:
                continue

            p_yes = float(probability)
            if p_yes > 1.0:
                p_yes /= 100.0

            # Reject extreme probabilities — these indicate frozen, stale, or
            # effectively-resolved Manifold markets, not tradeable signals.
            if p_yes > MANI_MAX_PROB or p_yes < 1.0 - MANI_MAX_PROB:
                continue

            liquidity = float(
                item.get("totalLiquidity") or item.get("volume") or 0
            )

            markets.append(ManifoldMarket(
                market_id=str(item.get("id", "")),
                question=item.get("question", ""),
                p_yes=p_yes,
                liquidity=liquidity,
            ))
        except (ValueError, TypeError):
            continue

    logging.info("Manifold: %d active binary market(s) fetched.", len(markets))
    return markets
