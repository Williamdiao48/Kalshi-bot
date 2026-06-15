"""
VLR.gg live Valorant match monitor.

Polls vlrggapi.vercel.app for live match state (map scores).
Scrapes vlr.gg match pages for round-level scores within the current map.
Computes updated series win probability from live state.
Compares against current Kalshi KXVALORANTGAME orderbook prices.

Probability model:
  - Series win prob: recursive formula (exact, handles BO3/BO5/any format)
  - Per-map prob p: extracted numerically from pre-match Kalshi price
  - Round-level: binomial model (equal per-round prob 0.50 — no map bias yet)
    Used to update P(win current map) from current round score.

Signal: model_series_prob - kalshi_ask_price > VALORANT_MIN_EDGE.
"""

import asyncio
import re
from dataclasses import dataclass, field
from functools import lru_cache

import aiohttp
from bs4 import BeautifulSoup

VLRGG_API_BASE = "https://vlrggapi.vercel.app"
VLRGG_WEB_BASE = "https://www.vlr.gg"
KALSHI_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"

# Default signal threshold: model must exceed Kalshi ask by this many cents
VALORANT_MIN_EDGE_CENTS = 8

# Minimum orderbook depth ($) near market price to consider liquid enough to trade
VALORANT_MIN_DEPTH_USD = 500


@dataclass
class RoundState:
    """Current round score within the active map."""
    team1_rounds: int
    team2_rounds: int
    half: int  # 1 = first half (rounds 1-12), 2 = second half (rounds 13-24), 3 = OT


@dataclass
class LiveMatch:
    team1: str
    team2: str
    maps1: int          # maps won by team1
    maps2: int          # maps won by team2
    series_format: int  # 2 = BO3, 3 = BO5
    tournament: str
    match_page: str | None = None
    round_state: RoundState | None = None
    map_name: str | None = None


@dataclass
class ValorantSignal:
    team: str           # team we'd bet YES on
    ticker: str         # Kalshi market ticker
    model_prob: float   # our model's P(team wins series)
    kalshi_ask: float   # current Kalshi YES ask (in cents / 100)
    edge: float         # model_prob - kalshi_ask
    depth_usd: float    # orderbook depth near ask price
    live_match: LiveMatch
    signal_basis: str   # "map_1_0" | "map_2_0" | "round_7_3" etc.


# ── Series probability (exact recursive formula) ──────────────────────────────

@lru_cache(maxsize=4096)
def _series_prob(need_a: int, need_b: int, p100: int) -> float:
    """P(A wins | needs need_a more maps, B needs need_b more). p100 = p * 10000 (int for caching)."""
    if need_a <= 0:
        return 1.0
    if need_b <= 0:
        return 0.0
    p = p100 / 10000.0
    return p * _series_prob(need_a - 1, need_b, p100) + (1 - p) * _series_prob(need_a, need_b - 1, p100)


def series_cond_prob(maps_won_a: int, maps_won_b: int, series_format: int, per_map_p: float) -> float:
    """P(A wins series) given current map score and per-map win probability."""
    need_a = series_format - maps_won_a
    need_b = series_format - maps_won_b
    if need_a <= 0:
        return 1.0
    if need_b <= 0:
        return 0.0
    return _series_prob(need_a, need_b, round(per_map_p * 10000))


def solve_per_map_prob(series_prob: float, series_format: int) -> float:
    """Binary search: find per-map p such that P(win series) = series_prob."""
    lo, hi = 0.001, 0.999
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if series_cond_prob(0, 0, series_format, mid) < series_prob:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# ── Round-level map win probability ──────────────────────────────────────────

def map_win_prob_from_rounds(rounds_a: int, rounds_b: int, half: int,
                              per_round_p: float = 0.50) -> float:
    """
    P(A wins current map) given current round score.

    Uses simple recursive model: each round is Bernoulli(per_round_p).
    Regulation: first to 13. OT at 12-12 (sudden death rounds).
    half=1: both teams still haven't switched sides yet (rounds 1–12 played so far).
    half=2: second half (rounds 13+, sides swapped).
    half=3: overtime.

    Note: per_round_p is symmetric (0.50) by default. A more accurate model
    would adjust for attack/defense win rates on the specific map half, which
    requires historical data outside this module.
    """
    target = 13

    @lru_cache(maxsize=4096)
    def _map_prob(a_need: int, b_need: int, p100: int) -> float:
        if a_need <= 0:
            return 1.0
        if b_need <= 0:
            return 0.0
        p = p100 / 10000.0
        return p * _map_prob(a_need - 1, b_need, p100) + (1 - p) * _map_prob(a_need, b_need - 1, p100)

    a_need = target - rounds_a
    b_need = target - rounds_b

    # Overtime: 12-12 → sudden death rounds (first to 2 ahead, play 2 at a time)
    # Simplified: treat OT as 2-round sudden death with equal probability
    if rounds_a == 12 and rounds_b == 12:
        return 0.50  # exact 50/50 in OT with equal per-round prob

    p100 = round(per_round_p * 10000)
    return _map_prob(a_need, b_need, p100)


def series_prob_with_rounds(maps_a: int, maps_b: int, series_format: int,
                             per_map_p: float,
                             rounds_a: int | None, rounds_b: int | None,
                             half: int = 1) -> float:
    """
    Full series win probability incorporating current map's round score.

    P(A wins series) = P(A wins current map) * P(A wins series | map_a+1, map_b)
                     + P(B wins current map) * P(A wins series | map_a, map_b+1)
    """
    if rounds_a is None or rounds_b is None:
        return series_cond_prob(maps_a, maps_b, series_format, per_map_p)

    p_win_map = map_win_prob_from_rounds(rounds_a, rounds_b, half)

    p_if_win_map = series_cond_prob(maps_a + 1, maps_b, series_format, per_map_p)
    p_if_lose_map = series_cond_prob(maps_a, maps_b + 1, series_format, per_map_p)

    return p_win_map * p_if_win_map + (1 - p_win_map) * p_if_lose_map


# ── VLR.gg API fetching ───────────────────────────────────────────────────────

def _series_format_from_tournament(tournament: str) -> int:
    """BO3 (2) for Americas, BO5 (3) for Pacific/EMEA."""
    return 2 if "americas" in tournament.lower() else 3


async def fetch_live_matches(session: aiohttp.ClientSession) -> list[LiveMatch]:
    """Poll VLR.gg API for currently live matches."""
    try:
        async with session.get(
            f"{VLRGG_API_BASE}/match?q=live_score",
            timeout=aiohttp.ClientTimeout(total=10),
        ) as r:
            data = await r.json()
    except Exception:
        return []

    segments = data.get("data", {}).get("segments", [])
    matches = []
    for seg in segments:
        team1 = seg.get("team1", "")
        team2 = seg.get("team2", "")
        tournament = seg.get("match_event", seg.get("tournament_name", ""))
        match_page = seg.get("match_page", "")

        # Map scores — field names may vary; try several
        try:
            maps1 = int(seg.get("score1", seg.get("map_score1", 0)))
            maps2 = int(seg.get("score2", seg.get("map_score2", 0)))
        except (TypeError, ValueError):
            maps1 = maps2 = 0

        series_format = _series_format_from_tournament(tournament)

        # Round scores within current map.
        # API returns per-side fields: team1_round_t, team1_round_ct, team2_round_t, team2_round_ct
        # "N/A" means that team hasn't played that side yet (first half in progress).
        round_state = None
        try:
            def _parse_r(v) -> int | None:
                s = str(v).strip() if v is not None else ""
                return int(s) if s.isdigit() else None

            t1_t  = _parse_r(seg.get("team1_round_t"))
            t1_ct = _parse_r(seg.get("team1_round_ct"))
            t2_t  = _parse_r(seg.get("team2_round_t"))
            t2_ct = _parse_r(seg.get("team2_round_ct"))

            if t1_ct is None and t2_t is None and (t1_t is not None or t2_ct is not None):
                # First half: team1 on T, team2 on CT
                round_state = RoundState(t1_t or 0, t2_ct or 0, half=1)
            elif t1_t is None and t2_ct is None and (t1_ct is not None or t2_t is not None):
                # First half: team1 on CT, team2 on T
                round_state = RoundState(t1_ct or 0, t2_t or 0, half=1)
            elif t1_t is not None and t1_ct is not None:
                # Second half (or OT): both sides played
                round_state = RoundState((t1_t or 0) + (t1_ct or 0), (t2_t or 0) + (t2_ct or 0), half=2)
        except (TypeError, ValueError):
            pass

        _map_name = seg.get("current_map", seg.get("map_name"))
        map_name = _map_name if _map_name and _map_name.lower() != "unknown" else None

        matches.append(LiveMatch(
            team1=team1, team2=team2,
            maps1=maps1, maps2=maps2,
            series_format=series_format,
            tournament=tournament,
            match_page=match_page or None,
            round_state=round_state,
            map_name=map_name,
        ))

    return matches


async def scrape_round_score(session: aiohttp.ClientSession,
                              match_page: str) -> tuple[RoundState | None, str | None]:
    """
    Scrape VLR.gg match page for current round score and map name.
    Returns (RoundState, map_name) or (None, None) on failure.

    VLR round HTML: each round column has title="X-Y" where X-Y is the
    cumulative score after that round. The last column = current score.
    """
    url = f"{VLRGG_WEB_BASE}/{match_page.lstrip('/')}"
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    try:
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as r:
            if r.status != 200:
                return None, None
            html = await r.text()
    except Exception:
        return None, None

    soup = BeautifulSoup(html, "html.parser")

    # Map name from veto/pick ban line
    map_names = ["Ascent", "Bind", "Breeze", "Fracture", "Haven",
                 "Icebox", "Lotus", "Pearl", "Split", "Sunset", "Abyss"]
    map_name = None
    map_name_div = soup.find("div", class_="map-name-full")
    if map_name_div:
        text = map_name_div.get_text(strip=True)
        for mn in map_names:
            if mn in text:
                map_name = mn
                break

    # Find all vlr-rounds sections (one per map played or in progress)
    round_sections = soup.find_all("div", class_="vlr-rounds")
    if not round_sections:
        return None, map_name

    # The LAST section is the current/most-recent map
    active_section = round_sections[-1]
    round_cols = active_section.find_all("div", class_="vlr-rounds-row-col")[1:]  # skip team name col

    if not round_cols:
        return None, map_name

    # Last rendered round col has the latest score in title="X-Y"
    last_title = None
    for col in reversed(round_cols):
        t = col.get("title", "")
        if re.match(r"^\d+-\d+$", t):
            last_title = t
            break

    if not last_title:
        return None, map_name

    parts = last_title.split("-")
    rounds_a, rounds_b = int(parts[0]), int(parts[1])
    total = rounds_a + rounds_b
    half = 1 if total < 12 else (2 if total < 24 else 3)

    return RoundState(rounds_a, rounds_b, half), map_name


# ── Kalshi market fetching ────────────────────────────────────────────────────

async def fetch_open_valorant_markets(session: aiohttp.ClientSession) -> list[dict]:
    """Fetch all open KXVALORANTGAME markets from Kalshi."""
    from kalshi_bot.auth import generate_headers
    path = "/trade-api/v2/markets"
    headers = generate_headers("GET", path)
    try:
        async with session.get(
            f"{KALSHI_API_BASE}/markets",
            params={"status": "open", "series_ticker": "KXVALORANTGAME", "limit": 100},
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as r:
            data = await r.json()
        return data.get("markets", [])
    except Exception:
        return []


async def fetch_orderbook(session: aiohttp.ClientSession, ticker: str) -> dict:
    """Fetch Kalshi orderbook for a market ticker. Returns yes_bid, yes_ask, depth_usd."""
    from kalshi_bot.auth import generate_headers
    path = f"/trade-api/v2/markets/{ticker}/orderbook"
    headers = generate_headers("GET", path)
    try:
        async with session.get(
            f"{KALSHI_API_BASE}/markets/{ticker}/orderbook",
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as r:
            ob = await r.json()
    except Exception:
        return {}

    ob_data = ob.get("orderbook_fp", {})
    yes_levels = ob_data.get("yes_dollars", [])
    no_levels = ob_data.get("no_dollars", [])

    yes_bid = max((float(p) for p, _ in yes_levels), default=None)
    no_bid = max((float(p) for p, _ in no_levels), default=None)
    yes_ask = (1.0 - no_bid) if no_bid is not None else None

    # Depth within ±5¢ of the ask price
    depth = 0.0
    if yes_ask is not None:
        for p, d in no_levels:
            if abs(float(p) - no_bid) <= 0.05:
                depth += float(d)

    return {
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "no_bid": no_bid,
        "depth_usd": depth,
    }


def _match_kalshi_markets(live: LiveMatch, open_markets: list[dict]) -> dict[str, dict]:
    """
    Fuzzy-match VLR team names to Kalshi YES markets.
    Returns {team_name: market_dict} for both teams if found.
    """
    result = {}
    for team in [live.team1, live.team2]:
        # Normalize: lowercase, strip accents for matching
        team_lower = team.lower().replace("ü", "u").replace("é", "e")
        for m in open_markets:
            title = m.get("title", "").lower().replace("ü", "u").replace("é", "e")
            if team_lower in title and "win the" in title:
                # Check it's the YES market for this team ("Will TEAM win...")
                if title.startswith(f"will {team_lower}"):
                    result[team] = m
                    break
    return result


# ── Main signal computation ───────────────────────────────────────────────────

def match_key(match: LiveMatch) -> str:
    """Stable identifier for a match (team names alphabetically sorted)."""
    return "|".join(sorted([match.team1, match.team2]))


async def compute_signals(
    session: aiohttp.ClientSession,
    live_matches: list[LiveMatch],
    open_markets: list[dict],
    pre_match_cache: dict,
    min_edge: float = VALORANT_MIN_EDGE_CENTS / 100,
    scrape_rounds: bool = True,
) -> list[ValorantSignal]:
    """
    For each live match, compute model probability and compare to Kalshi prices.
    Returns signals where model_prob - kalshi_ask > min_edge.

    pre_match_cache: maintained by caller across poll cycles.
      Keys: match_key(match)  Values: {team: per_map_p}
    Populated the first time we see a match at 0-0 maps (true pre-match).
    Using cached per-map-p prevents back-solving from conditional prices
    (which gives inflated p and false signals if the market already updated).
    """
    signals = []

    for match in live_matches:
        team_markets = _match_kalshi_markets(match, open_markets)
        if len(team_markets) < 2:
            continue

        # Attempt round scraping if API didn't provide round data
        if match.round_state is None and scrape_rounds and match.match_page:
            rs, map_name = await scrape_round_score(session, match.match_page)
            match.round_state = rs
            if map_name:
                match.map_name = map_name
            await asyncio.sleep(0.5)

        mk = match_key(match)

        # Fetch current orderbook for both teams
        obs: dict[str, dict] = {}
        for team, market in team_markets.items():
            obs[team] = await fetch_orderbook(session, market.get("ticker", ""))
            await asyncio.sleep(0.1)

        # Populate pre-match cache when series is at 0-0 with no rounds yet
        if mk not in pre_match_cache:
            if match.maps1 == 0 and match.maps2 == 0 and match.round_state is None:
                cache_entry = {}
                for team in team_markets:
                    yes_ask = obs[team].get("yes_ask")
                    if yes_ask is not None and 0.05 < yes_ask < 0.95:
                        try:
                            cache_entry[team] = solve_per_map_prob(yes_ask, match.series_format)
                        except Exception:
                            pass
                if cache_entry:
                    pre_match_cache[mk] = cache_entry
            else:
                # Match started before we began monitoring — no reliable baseline
                continue

        cached = pre_match_cache.get(mk, {})
        if not cached:
            continue

        # No signal at 0-0 with no rounds (pre-match state, just caching)
        if match.maps1 == 0 and match.maps2 == 0 and match.round_state is None:
            continue

        for team, market in team_markets.items():
            is_team1 = (team == match.team1)
            maps_a = match.maps1 if is_team1 else match.maps2
            maps_b = match.maps2 if is_team1 else match.maps1

            per_map_p = cached.get(team)
            if per_map_p is None:
                continue

            ob = obs[team]
            yes_ask = ob.get("yes_ask")
            depth = ob.get("depth_usd", 0.0)
            if yes_ask is None:
                continue

            if match.round_state is not None:
                r1 = match.round_state.team1_rounds if is_team1 else match.round_state.team2_rounds
                r2 = match.round_state.team2_rounds if is_team1 else match.round_state.team1_rounds
                model_p = series_prob_with_rounds(
                    maps_a, maps_b, match.series_format, per_map_p,
                    r1, r2, match.round_state.half
                )
                basis = f"map_{maps_a}-{maps_b}_rounds_{r1}-{r2}"
            else:
                model_p = series_cond_prob(maps_a, maps_b, match.series_format, per_map_p)
                basis = f"map_{maps_a}-{maps_b}"

            edge = model_p - yes_ask

            if edge >= min_edge and depth >= VALORANT_MIN_DEPTH_USD:
                signals.append(ValorantSignal(
                    team=team,
                    ticker=market.get("ticker", ""),
                    model_prob=model_p,
                    kalshi_ask=yes_ask,
                    edge=edge,
                    depth_usd=depth,
                    live_match=match,
                    signal_basis=basis,
                ))

    return signals
