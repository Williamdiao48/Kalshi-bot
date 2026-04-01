"""PredictIt public market data fetcher (read-only, no account needed).

PredictIt is a real-money regulated prediction market focused on political
and current-events outcomes.  Its prices are financially incentivised and
carry the same signal quality tier as Polymarket — strong evidence of
genuine crowd-sourced probability estimates.

Unlike Polymarket (global, broad) PredictIt skews heavily toward US political
markets: congressional confirmations, election outcomes, presidential actions.
This makes it a particularly valuable complement to Kalshi's political
contract suite.

API:
    GET https://www.predictit.org/api/marketdata/all/

Response: JSON object with a ``markets`` array.  Key fields per market:
    id              — integer market ID
    name            — plain-English market question
    shortName       — abbreviated version
    status          — "Open" | "Closed"
    tradingHalted   — bool; true when market is suspended
    dayVolume       — integer; today's total USD trading volume
    contracts[]     — array of outcome contracts:
        id                — contract ID
        name              — outcome name ("Yes", "No", or candidate name)
        status            — "Open" | "Closed"
        lastTradePrice    — last trade price (0.00–1.00 = YES probability)
        bestBuyYesCost    — current best YES ask
        bestBuyNoCost     — current best NO ask

Binary market detection
-----------------------
PredictIt markets range from simple YES/NO questions (1 open contract) to
multi-outcome races (many contracts, one per candidate).  Only binary markets
are useful as a direct probability signal:

  1. Markets with exactly 1 open contract → binary; use lastTradePrice as p_yes.
  2. Markets with exactly 2 open contracts named "Yes"/"No" → binary; use
     the "Yes" contract's lastTradePrice.
  3. All other shapes → multi-outcome; skipped.

Thresholds (env-var overridable):
    PDIT_MIN_DIVERGENCE   Minimum |pdit_p − kalshi_p| to surface.  Default 0.15.
    PDIT_MIN_VOLUME       Minimum dayVolume (USD) to filter stale markets.  Default 500.
    PDIT_MIN_MATCH_SCORE  Minimum Jaccard similarity.  Default 0.20.
"""

import logging
import os
from dataclasses import dataclass

import aiohttp

_API_URL = "https://www.predictit.org/api/marketdata/all/"

PDIT_MIN_DIVERGENCE:  float = float(os.environ.get("PDIT_MIN_DIVERGENCE",  "0.15"))
PDIT_MIN_VOLUME:      float = float(os.environ.get("PDIT_MIN_VOLUME",      "500"))
PDIT_MIN_MATCH_SCORE: float = float(os.environ.get("PDIT_MIN_MATCH_SCORE", "0.20"))


@dataclass
class PredictItContract:
    """A single PredictIt binary market with its current implied probability."""

    market_id: str
    question:  str   # market name (plain-English question)
    p_yes:     float # YES probability, 0.0–1.0
    volume:    float # dayVolume in USD (liquidity proxy)
    end_date:  str   # ISO-8601 end date, or empty string if not provided


def _extract_binary(market: dict) -> float | None:
    """Return the YES probability for a binary market, or None if not binary.

    Tries two shapes:
      1. Exactly 1 open contract (it IS the YES contract).
      2. Exactly 2 open contracts named "Yes" and "No".
    """
    contracts = market.get("contracts") or []
    open_contracts = [c for c in contracts if c.get("status") == "Open"]

    if len(open_contracts) == 1:
        price = open_contracts[0].get("lastTradePrice")
        if price is None:
            return None
        return float(price)

    if len(open_contracts) == 2:
        names = {c.get("name", "").strip().lower() for c in open_contracts}
        if names == {"yes", "no"}:
            for c in open_contracts:
                if c.get("name", "").strip().lower() == "yes":
                    price = c.get("lastTradePrice")
                    if price is None:
                        return None
                    return float(price)

    return None


async def fetch_contracts(session: aiohttp.ClientSession) -> list[PredictItContract]:
    """Fetch all active binary PredictIt markets.

    A single request returns the full market universe.  Multi-outcome markets
    (races, ranked outcomes) are silently dropped — only binary YES/NO questions
    are returned.

    Returns an empty list on any fetch or parse failure.
    """
    try:
        async with session.get(
            _API_URL,
            timeout=aiohttp.ClientTimeout(total=15),
            headers={"Accept": "application/json"},
        ) as resp:
            resp.raise_for_status()
            data = await resp.json(content_type=None)
    except aiohttp.ClientResponseError as exc:
        logging.error("PredictIt HTTP error %s: %s", exc.status, exc.message)
        return []
    except aiohttp.ClientError as exc:
        logging.error("PredictIt request error: %s", exc)
        return []

    raw_markets = data.get("markets") if isinstance(data, dict) else None
    if not isinstance(raw_markets, list):
        logging.error("PredictIt: unexpected response shape (no 'markets' array).")
        return []

    contracts: list[PredictItContract] = []
    for item in raw_markets:
        try:
            if item.get("status") != "Open":
                continue
            if item.get("tradingHalted"):
                continue

            p_yes = _extract_binary(item)
            if p_yes is None:
                continue  # multi-outcome market — skip

            # Guard against degenerate prices (fully resolved but not yet closed)
            if not (0.01 <= p_yes <= 0.99):
                continue

            day_volume = float(item.get("dayVolume") or 0)

            # end_date: PredictIt doesn't always expose a close date at market level
            end_date = str(item.get("end") or item.get("endDate") or "")

            contracts.append(PredictItContract(
                market_id=str(item.get("id", "")),
                question=item.get("name", ""),
                p_yes=p_yes,
                volume=day_volume,
                end_date=end_date,
            ))
        except (ValueError, TypeError):
            continue

    logging.info("PredictIt: %d active binary market(s) fetched.", len(contracts))
    return contracts
