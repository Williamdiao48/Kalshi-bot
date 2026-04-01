"""Polymarket public market data fetcher (read-only, no account needed).

Fetches active prediction markets from Polymarket's Gamma REST API and
returns them as ``PolyMarket`` objects.  No API key or account required —
all data is public.

The Gamma API endpoint sorts by liquidity so the most price-efficient markets
(fewest arbitrage opportunities) come first.  We use these as an external
calibration signal for Kalshi: if Polymarket is pricing an event at 70% and
Kalshi is at 40%, that gap is our signal.

API:
    GET https://gamma-api.polymarket.com/markets
        ?active=true
        &closed=false
        &limit=500
        &order=liquidityNum
        &ascending=false

Response: JSON array of market objects.  Key fields used:
    id              — unique market ID (string)
    question        — plain-English question, e.g. "Will BTC exceed $100k?"
    outcomePrices   — JSON string e.g. "[\"0.65\", \"0.35\"]" (YES, NO)
    outcomes        — JSON string e.g. "[\"Yes\", \"No\"]"
    liquidityNum    — total USD liquidity (float)
    endDate         — ISO-8601 resolution date
    active          — bool
    closed          — bool

Only binary YES/NO markets are returned; multi-outcome markets are skipped.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import aiohttp

_GAMMA_URL = "https://gamma-api.polymarket.com/markets"
_FETCH_LIMIT = 500   # top markets by liquidity


@dataclass
class PolyMarket:
    """A single Polymarket prediction market with its current implied probability."""

    market_id: str
    question: str
    p_yes: float       # YES probability, 0.0–1.0
    liquidity: float   # USD liquidity in the market
    end_date: str      # ISO-8601 resolution timestamp


async def fetch_markets(session: aiohttp.ClientSession) -> list[PolyMarket]:
    """Fetch the top active Polymarket markets sorted by liquidity.

    Returns a list of binary YES/NO markets with current implied probabilities.
    Returns an empty list on any fetch or parse failure.
    """
    params = {
        "active":    "true",
        "closed":    "false",
        "limit":     str(_FETCH_LIMIT),
        "order":     "liquidityNum",
        "ascending": "false",
    }
    try:
        async with session.get(
            _GAMMA_URL,
            params=params,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
    except aiohttp.ClientResponseError as exc:
        logging.error("Polymarket HTTP error %s: %s", exc.status, exc.message)
        return []
    except aiohttp.ClientError as exc:
        logging.error("Polymarket request error: %s", exc)
        return []

    markets: list[PolyMarket] = []
    for item in data:
        try:
            # Parse prices — stored as a JSON string inside the JSON object
            outcomes_raw = item.get("outcomes", "[]")
            prices_raw   = item.get("outcomePrices", "[]")
            outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
            prices   = json.loads(prices_raw)   if isinstance(prices_raw,   str) else prices_raw

            # Only handle binary markets with interpretable outcome labels.
            # Accept Yes/No, Up/Down (directional), and Over/Under (threshold).
            # Reject sports team-name matchups and esports markets.
            if len(outcomes) != 2 or len(prices) != 2:
                continue

            out0 = str(outcomes[0]).lower()
            out1 = str(outcomes[1]).lower()

            # Map outcome pairs to a YES-equivalent probability.
            # For Yes/No: p_yes = prices[0].
            # For Up/Down and Over/Under: treat "Up"/"Over" as YES.
            # For anything else (team names etc.) skip — can't match Kalshi.
            if out0 in ("yes", "y"):
                p_yes = float(prices[0])
            elif out0 == "up" and out1 == "down":
                p_yes = float(prices[0])   # p_yes = P(up)
            elif out0 == "over" and out1 == "under":
                p_yes = float(prices[0])   # p_yes = P(over)
            elif out0 == "no" and out1 in ("yes", "y"):
                p_yes = float(prices[1])   # inverted layout
            else:
                continue  # team names, custom labels — skip
            # Normalise: Polymarket uses 0–1 scale but occasionally 0–100
            if p_yes > 1.0:
                p_yes /= 100.0

            liquidity = float(item.get("liquidityNum") or item.get("liquidity") or 0)
            end_date  = item.get("endDate") or item.get("end_date") or ""

            markets.append(PolyMarket(
                market_id=str(item.get("id", "")),
                question=item.get("question", ""),
                p_yes=p_yes,
                liquidity=liquidity,
                end_date=end_date,
            ))
        except (ValueError, TypeError, json.JSONDecodeError):
            continue

    logging.info("Polymarket: fetched %d active binary markets.", len(markets))
    return markets
