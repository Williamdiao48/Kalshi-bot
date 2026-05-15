"""Pinnacle Sports NBA moneyline fetcher (no API key required).

Uses the Pinnacle guest API to fetch NBA moneyline odds for prospective
backtesting of the Kalshi convergence strategy: buy when Kalshi lags a
Pinnacle reprice, sell when Kalshi catches up.

Data is logged to the ``nba_snapshots`` table in opportunity_log.db by
``_log_nba_snapshots()`` in main.py.  This module is purely a fetcher;
it does not emit DataPoints and does not interact with the trading pipeline.

API (no auth):
    Matchups: GET https://guest.api.arcadia.pinnacle.com/0.1/leagues/487/matchups
    Prices:   GET https://guest.api.arcadia.pinnacle.com/0.1/matchups/{id}/markets/straight

Environment variables:
    PINNACLE_ENABLED  "true" | "false" — set to "false" to disable.  Default: "true".
    PINNACLE_TIMEOUT  HTTP timeout in seconds.  Default: 10.
"""

import asyncio
import logging
from datetime import datetime, timezone
from ..utils import env_bool, env_int

import aiohttp

PINNACLE_ENABLED: bool = env_bool("PINNACLE_ENABLED", True)
PINNACLE_TIMEOUT: int = env_int("PINNACLE_TIMEOUT", 10)

_NBA_LEAGUE_ID = 487
_BASE = "https://guest.api.arcadia.pinnacle.com/0.1"
_MATCHUPS_URL = f"{_BASE}/leagues/{_NBA_LEAGUE_ID}/matchups"

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; kalshi-bot/1.0)",
    "Accept": "application/json",
}

# 3-letter Kalshi ticker abbreviation → Pinnacle full team name
NBA_ABBREVS: dict[str, str] = {
    "ATL": "Atlanta Hawks",
    "BOS": "Boston Celtics",
    "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "LA Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards",
}

# Reverse map: normalized full name → 3-letter abbrev
_PINNACLE_TO_ABBREV: dict[str, str] = {v.lower(): k for k, v in NBA_ABBREVS.items()}


def _to_implied(american: float) -> float:
    """Convert American moneyline odds to raw implied probability."""
    if american > 0:
        return 100.0 / (american + 100.0)
    return -american / (-american + 100.0)


def _devig(home: float, away: float) -> tuple[float, float]:
    """Remove the vig from a two-way moneyline, returning fair probabilities."""
    h = _to_implied(home)
    a = _to_implied(away)
    total = h + a
    return h / total, a / total


def _parse_nba_ticker(ticker: str) -> tuple[str, str] | None:
    """Return (team1_abbrev, team2_abbrev) parsed from a KXNBAGAME ticker, or None.

    Ticker format: KXNBAGAME-{DATE}{TEAM1}{TEAM2}-{SIDE}
    Example: KXNBAGAME-26MAY15DETCLE-DET → ("DET", "CLE")
    The matchup segment has a date prefix of variable length; the last 6 chars
    are always the two 3-letter team abbreviations.
    """
    parts = ticker.split("-")
    if len(parts) < 3 or not parts[0].startswith("KXNBA"):
        return None
    matchup = parts[1]
    if len(matchup) < 6:
        return None
    pair = matchup[-6:]
    return pair[:3], pair[3:]


async def _fetch_prices(
    session: aiohttp.ClientSession,
    matchup_id: int,
    timeout: aiohttp.ClientTimeout,
) -> dict[str, float] | None:
    """Fetch moneyline prices for one matchup.

    Returns dict with keys "home" and "away" as American odds, or None on error.
    """
    url = f"{_BASE}/matchups/{matchup_id}/markets/straight"
    try:
        async with session.get(url, headers=_HEADERS, timeout=timeout) as resp:
            if resp.status in (404, 422):
                return None
            resp.raise_for_status()
            data = await resp.json(content_type=None)
    except aiohttp.ClientError as exc:
        logging.debug("Pinnacle: price fetch error for matchup %d: %s", matchup_id, exc)
        return None

    # data is a list of market objects; find the moneyline entry
    markets = data if isinstance(data, list) else [data]
    for mkt in markets:
        if mkt.get("type") != "moneyline":
            continue
        prices = {p["designation"]: p["price"] for p in mkt.get("prices", []) if "designation" in p and "price" in p}
        if "home" in prices and "away" in prices:
            return {"home": prices["home"], "away": prices["away"]}
    return None


async def fetch_nba_games(session: aiohttp.ClientSession) -> list[dict]:
    """Fetch upcoming NBA games from Pinnacle with de-vigged moneyline probabilities.

    Returns:
        List of game dicts with keys:
            matchup_id  (int)   Pinnacle internal ID
            game_date   (str)   ISO date "YYYY-MM-DD" (UTC)
            home_team   (str)   Full team name, e.g. "Denver Nuggets"
            away_team   (str)
            home_prob   (float) De-vigged fair probability 0–1
            away_prob   (float)
    """
    if not PINNACLE_ENABLED:
        return []

    timeout = aiohttp.ClientTimeout(total=PINNACLE_TIMEOUT)

    try:
        async with session.get(
            _MATCHUPS_URL,
            headers=_HEADERS,
            timeout=timeout,
            params={"withSpecials": "false"},
        ) as resp:
            resp.raise_for_status()
            matchups = await resp.json(content_type=None)
    except aiohttp.ClientResponseError as exc:
        logging.warning("Pinnacle: HTTP %s fetching matchups — %s", exc.status, exc.message)
        return []
    except aiohttp.ClientError as exc:
        logging.warning("Pinnacle: matchup fetch error — %s", exc)
        return []

    if not isinstance(matchups, list):
        logging.warning("Pinnacle: unexpected matchups response type: %s", type(matchups))
        return []

    # Filter to pre-game matchups; extract team names from participants
    candidates: list[dict] = []
    for m in matchups:
        if m.get("isLive") or m.get("type") != "matchup":
            continue
        if m.get("status") not in ("pending", "upcoming", None):
            continue
        participants = m.get("participants") or []
        home = next((p["name"] for p in participants if p.get("alignment") == "home"), None)
        away = next((p["name"] for p in participants if p.get("alignment") == "away"), None)
        if not home or not away:
            continue
        start_raw = m.get("startTime", "")
        try:
            game_date = datetime.fromisoformat(start_raw.replace("Z", "+00:00")).date().isoformat()
        except (ValueError, AttributeError):
            game_date = datetime.now(timezone.utc).date().isoformat()
        candidates.append({
            "matchup_id": m["id"],
            "game_date": game_date,
            "home_team": home,
            "away_team": away,
        })

    if not candidates:
        logging.debug("Pinnacle: no upcoming NBA matchups found.")
        return []

    # Fetch prices concurrently
    price_tasks = [
        _fetch_prices(session, c["matchup_id"], timeout)
        for c in candidates
    ]
    price_results = await asyncio.gather(*price_tasks, return_exceptions=True)

    games: list[dict] = []
    for candidate, result in zip(candidates, price_results):
        if isinstance(result, Exception) or result is None:
            continue
        try:
            home_prob, away_prob = _devig(result["home"], result["away"])
        except (KeyError, ZeroDivisionError, TypeError):
            continue
        games.append({
            **candidate,
            "home_prob": round(home_prob, 4),
            "away_prob": round(away_prob, 4),
        })

    logging.info(
        "Pinnacle: %d NBA game(s) with moneyline prices (of %d matchups).",
        len(games), len(candidates),
    )
    return games
