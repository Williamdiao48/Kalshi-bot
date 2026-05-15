"""
Backtest NBA convergence: Kalshi KXNBAGAME prices vs DraftKings (via ESPN).

For each finalized KXNBAGAME market, fetches:
  - Kalshi hourly candlesticks for the home-team YES market
  - ESPN DraftKings opening moneyline (de-vigged to fair probability)

Simulates a convergence trade: enter at Kalshi market-open (yes_ask), exit at
the last pre-tip-off candle (yes_bid). Measures whether Kalshi prices converge
toward the DraftKings reference probability during the trading window.

Requires KALSHI_ENVIRONMENT=production in .env (demo has no KXNBAGAME history).

Usage:
    venv/bin/python scripts/backtest_nba_convergence.py
"""

import asyncio
import aiohttp
from collections import defaultdict
from datetime import datetime, timezone

from kalshi_bot.auth import generate_headers
from kalshi_bot.markets import KALSHI_API_BASE

# ── Constants ────────────────────────────────────────────────────────────────

MIN_GAP_THRESHOLD = 0.03    # minimum |DK prob - Kalshi mid| (fraction) to trade
MAX_SPREAD        = 0.15    # maximum (ask - bid) allowed at entry — excludes stub-bid opens
MIN_BID           = 0.05    # minimum yes_bid at entry — excludes markets with no buyers yet
CANDLE_INTERVAL   = 60      # minutes (hourly candles)
RATE_LIMIT_KALSHI = 0.12    # seconds between Kalshi requests
RATE_LIMIT_ESPN   = 0.10    # seconds between ESPN requests

_ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
_ESPN_SUMMARY    = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
_TIMEOUT         = aiohttp.ClientTimeout(total=15)

_MON = {
    "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
    "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
    "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
}

# ESPN abbreviation → Kalshi abbreviation (known divergences)
_ESPN_TO_KALSHI: dict[str, str] = {
    "GS":   "GSW",
    "NO":   "NOP",
    "SA":   "SAS",
    "NY":   "NYK",
    "BK":   "BKN",
    "WSH":  "WAS",
    "UTAH": "UTA",
}


# ── Ticker parsing ────────────────────────────────────────────────────────────

def _parse_ticker(ticker: str) -> dict | None:
    """Parse a KXNBAGAME ticker into its components.

    Ticker format: KXNBAGAME-{YY}{MON}{DD}{TEAM1}{TEAM2}-{SIDE}
    Example: KXNBAGAME-26MAR14CHASAS-SAS
      → {date_str: "20260314", team1: "CHA", team2: "SAS", side: "SAS"}
    """
    parts = ticker.split("-")
    if len(parts) < 3 or not parts[0].startswith("KXNBA"):
        return None
    matchup = parts[1]   # e.g. "26MAR14CHASAS"
    side = parts[-1]     # e.g. "SAS"
    if len(matchup) < 13:
        return None
    yy   = matchup[0:2]   # "26"
    mon  = matchup[2:5]   # "MAR"
    dd   = matchup[5:7]   # "14"
    pair = matchup[7:]    # "CHASAS" (6 chars)
    if len(pair) != 6 or mon not in _MON:
        return None
    return {
        "date_str": f"20{yy}{_MON[mon]}{dd}",
        "team1": pair[:3],
        "team2": pair[3:],
        "side": side,
    }


# ── De-vig ───────────────────────────────────────────────────────────────────

def _to_implied(american: float) -> float:
    if american > 0:
        return 100.0 / (american + 100.0)
    return -american / (-american + 100.0)


def _devig(home: float, away: float) -> tuple[float, float]:
    h, a = _to_implied(home), _to_implied(away)
    total = h + a
    return h / total, a / total


# ── Kalshi API ───────────────────────────────────────────────────────────────

async def fetch_historical_markets(session: aiohttp.ClientSession) -> list[dict]:
    """Fetch all finalized KXNBAGAME markets via paginated historical API."""
    markets: list[dict] = []
    cursor = ""
    page = 0
    while True:
        path = "/historical/markets"
        params: dict = {"series_ticker": "KXNBAGAME", "limit": 200}
        if cursor:
            params["cursor"] = cursor
        headers = generate_headers("GET", path)
        try:
            async with session.get(
                f"{KALSHI_API_BASE}{path}",
                params=params,
                headers=headers,
                timeout=_TIMEOUT,
            ) as resp:
                if resp.status == 429:
                    await asyncio.sleep(2.0)
                    continue
                resp.raise_for_status()
                data = await resp.json()
        except Exception as exc:
            print(f"  Error fetching markets page {page}: {exc}")
            break
        batch = data.get("markets", [])
        markets.extend(batch)
        cursor = data.get("cursor", "")
        page += 1
        print(f"  Page {page}: +{len(batch)} markets  (total: {len(markets)})")
        if not cursor:
            break
        await asyncio.sleep(0.2)
    return markets


async def fetch_kalshi_candles(
    session: aiohttp.ClientSession,
    ticker: str,
    start_ts: int,
    end_ts: int,
) -> list[dict]:
    """Fetch hourly Kalshi candles for one ticker from the historical endpoint."""
    path = f"/historical/markets/{ticker}/candlesticks"
    params = {"start_ts": start_ts, "end_ts": end_ts, "period_interval": CANDLE_INTERVAL}
    headers = generate_headers("GET", path)
    try:
        async with session.get(
            f"{KALSHI_API_BASE}{path}",
            params=params,
            headers=headers,
            timeout=_TIMEOUT,
        ) as resp:
            if resp.status in (404, 422):
                return []
            resp.raise_for_status()
            data = await resp.json()
            return sorted(data.get("candlesticks", []), key=lambda c: c["end_period_ts"])
    except Exception as exc:
        print(f"  Candle error for {ticker}: {exc}")
        return []


# ── ESPN API ─────────────────────────────────────────────────────────────────

_scoreboard_cache: dict[str, list[dict]] = {}


def _norm(abbrev: str) -> str:
    """Normalize an ESPN team abbreviation to Kalshi format."""
    return _ESPN_TO_KALSHI.get(abbrev.upper(), abbrev.upper())


async def fetch_espn_event(
    session: aiohttp.ClientSession,
    date_str: str,
    team1: str,
    team2: str,
) -> tuple[int | None, str | None]:
    """Find the ESPN event ID and Kalshi-format home team abbrev for a game.

    Returns (espn_event_id, home_team_kalshi_abbrev) or (None, None).
    Caches scoreboard responses by date to minimize ESPN calls.
    """
    if date_str not in _scoreboard_cache:
        try:
            async with session.get(
                _ESPN_SCOREBOARD,
                params={"dates": date_str},
                timeout=_TIMEOUT,
            ) as resp:
                resp.raise_for_status()
                _scoreboard_cache[date_str] = (await resp.json()).get("events", [])
        except Exception as exc:
            print(f"  ESPN scoreboard error {date_str}: {exc}")
            _scoreboard_cache[date_str] = []
        await asyncio.sleep(RATE_LIMIT_ESPN)

    target = {team1.upper(), team2.upper()}
    for event in _scoreboard_cache[date_str]:
        comps = event.get("competitions", [{}])[0].get("competitors", [])
        abbrev_to_ha: dict[str, str] = {}
        for c in comps:
            abbrev = _norm(c.get("team", {}).get("abbreviation", ""))
            ha = c.get("homeAway", "")
            if abbrev:
                abbrev_to_ha[abbrev] = ha

        if target <= set(abbrev_to_ha.keys()):
            home = next((a for a, ha in abbrev_to_ha.items() if ha == "home"), None)
            return int(event["id"]), home

    return None, None


async def fetch_espn_odds(
    session: aiohttp.ClientSession,
    espn_id: int,
) -> dict | None:
    """Fetch DraftKings open + close moneylines from ESPN game summary.

    Returns {home_open, home_close, away_open, away_close} as American odds,
    or None if not available.
    """
    try:
        async with session.get(
            _ESPN_SUMMARY,
            params={"event": espn_id},
            timeout=_TIMEOUT,
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
    except Exception as exc:
        print(f"  ESPN summary error event {espn_id}: {exc}")
        return None

    for pc in data.get("pickcenter", []):
        ml = pc.get("moneyline")  # lowercase key
        if not ml:
            continue
        try:
            return {
                "home_open":  float(ml["home"]["open"]["odds"]),
                "home_close": float(ml["home"]["close"]["odds"]),
                "away_open":  float(ml["away"]["open"]["odds"]),
                "away_close": float(ml["away"]["close"]["odds"]),
            }
        except (KeyError, TypeError, ValueError):
            continue
    return None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_float(val) -> float | None:
    try:
        f = float(val)
        return f if f > 0 else None
    except (TypeError, ValueError):
        return None


# ── Main ─────────────────────────────────────────────────────────────────────

async def _run() -> None:
    print("Fetching finalized KXNBAGAME markets from Kalshi historical API...")
    async with aiohttp.ClientSession() as session:
        all_markets = await fetch_historical_markets(session)
    print(f"Total KXNBAGAME markets fetched: {len(all_markets)}\n")

    # Group by unique game: (date_str, frozenset({team1, team2})) → list of tickers
    games: dict[tuple, list[str]] = defaultdict(list)
    for m in all_markets:
        ticker = m.get("ticker", "")
        parsed = _parse_ticker(ticker)
        if not parsed:
            continue
        key = (parsed["date_str"], frozenset({parsed["team1"], parsed["team2"]}))
        games[key].append(ticker)

    print(f"Unique games: {len(games)}\n")
    print("Processing games (ESPN match → DK odds → Kalshi candles)...")

    results = []
    no_espn_match = 0
    no_odds = 0
    no_candles = 0
    below_threshold = 0
    illiquid_open = 0

    async with aiohttp.ClientSession() as session:
        for i, ((date_str, teams_set), tickers) in enumerate(sorted(games.items())):
            team1, team2 = sorted(teams_set)

            # 1. Match to ESPN event
            espn_id, home_abbrev = await fetch_espn_event(session, date_str, team1, team2)
            if espn_id is None or home_abbrev is None:
                no_espn_match += 1
                continue

            # 2. Fetch DK odds
            odds = await fetch_espn_odds(session, espn_id)
            await asyncio.sleep(RATE_LIMIT_ESPN)
            if odds is None:
                no_odds += 1
                continue

            # 3. De-vig DK opening line to get fair home probability
            try:
                dk_home_prob, _ = _devig(odds["home_open"], odds["away_open"])
            except ZeroDivisionError:
                no_odds += 1
                continue

            # 4. Select Kalshi ticker for the home team's YES market
            home_ticker = next(
                (t for t in tickers if t.split("-")[-1] == home_abbrev), None
            )
            if home_ticker is None:
                home_ticker = tickers[0]  # fallback

            # 5. Fetch Kalshi hourly candles (±2 days around game date)
            try:
                game_dt = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            start_ts = int(game_dt.timestamp()) - 2 * 86400
            end_ts   = int(game_dt.timestamp()) + 2 * 86400

            candles = await fetch_kalshi_candles(session, home_ticker, start_ts, end_ts)
            await asyncio.sleep(RATE_LIMIT_KALSHI)

            # 6. Filter to candles with valid bid and ask
            valid = []
            for c in candles:
                bid = _safe_float(c.get("yes_bid", {}).get("close"))
                ask = _safe_float(c.get("yes_ask", {}).get("close"))
                if bid is not None and ask is not None:
                    valid.append((c, bid, ask))

            if len(valid) < 2:
                no_candles += 1
                continue

            _, first_bid, first_ask = valid[0]
            _, last_bid,  last_ask  = valid[-1]

            # 7. Skip illiquid opens: stub bid or spread too wide
            if first_bid < MIN_BID or (first_ask - first_bid) > MAX_SPREAD:
                illiquid_open += 1
                continue

            first_mid = (first_bid + first_ask) / 2
            last_mid  = (last_bid + last_ask) / 2

            # 8. Compute gap vs DK using mid-price
            gap = dk_home_prob - first_mid   # positive = Kalshi underpricing home

            if abs(gap) < MIN_GAP_THRESHOLD:
                below_threshold += 1
                continue

            if gap > 0:
                entry  = first_ask          # buy YES: pay yes_ask
                exit_  = last_bid           # sell YES: receive yes_bid
            else:
                entry  = 1.0 - first_bid    # buy NO: pay (1 - yes_bid)
                exit_  = 1.0 - last_ask     # sell NO: receive (1 - yes_ask)

            pnl_cents = (exit_ - entry) * 100
            converged = (gap > 0 and last_mid > first_mid) or (gap < 0 and last_mid < first_mid)

            away_abbrev = next(iter(teams_set - {home_abbrev}), "?") if home_abbrev in teams_set else "?"
            results.append({
                "ticker":       home_ticker,
                "date":         f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}",
                "home":         home_abbrev,
                "away":         away_abbrev,
                "dk_open_prob": round(dk_home_prob * 100, 1),
                "dk_close_prob": round(_devig(odds["home_close"], odds["away_close"])[0] * 100, 1),
                "kalshi_mid":   round(first_mid * 100, 1),
                "kalshi_close": round(last_mid * 100, 1),
                "gap":          round(gap * 100, 1),
                "pnl":          round(pnl_cents, 1),
                "converged":    converged,
                "n_candles":    len(valid),
            })

            if (i + 1) % 25 == 0:
                print(f"  [{i+1}/{len(games)}] processed so far, {len(results)} with signal...")

    # ── Output ────────────────────────────────────────────────────────────────
    W = "=" * 80
    print(f"\n{W}")
    print("NBA CONVERGENCE BACKTEST  —  Kalshi vs DraftKings (via ESPN)")
    print(W)
    print(
        f"Games analyzed: {len(results)}  |  No ESPN match: {no_espn_match}  |  "
        f"No DK odds: {no_odds}  |  No candles: {no_candles}  |  "
        f"Illiquid open (bid<{MIN_BID*100:.0f}¢ or spread>{MAX_SPREAD*100:.0f}¢): {illiquid_open}  |  "
        f"Below {MIN_GAP_THRESHOLD*100:.0f}¢ threshold: {below_threshold}"
    )

    if not results:
        print("\nNo results — check KALSHI_ENVIRONMENT=production in .env")
        return

    n = len(results)
    wins = [r for r in results if r["pnl"] > 0]
    losses = [r for r in results if r["pnl"] <= 0]
    total_pnl = sum(r["pnl"] for r in results)
    conv_n = sum(1 for r in results if r["converged"])
    avg_gap = sum(abs(r["gap"]) for r in results) / n
    avg_move = sum(r["kalshi_close"] - r["kalshi_mid"] for r in results) / n

    print(f"\n── CONVERGENCE SUMMARY {'─'*56}")
    print(f"Games with signal:           {n}")
    print(f"Converged (mid moved toward DK): {conv_n} / {n}  ({conv_n/n*100:.1f}%)")
    print(f"Avg |DK-open gap| (vs mid):  {avg_gap:+.1f}¢")
    print(f"Avg Kalshi mid movement:     {avg_move:+.1f}¢")
    print(f"(Mid = (bid+ask)/2; spread filter: bid≥{MIN_BID*100:.0f}¢, spread≤{MAX_SPREAD*100:.0f}¢)")

    print(f"\n── SIMULATED P&L (enter at open yes_ask, exit at last yes_bid) {'─'*14}")
    print(f"Total P&L:      {total_pnl:+.0f}¢  ({n} trades, 1 contract each)")
    print(f"Win rate:       {len(wins)/n*100:.1f}%  ({len(wins)}W / {len(losses)}L)")
    print(f"Avg P&L/trade:  {total_pnl/n:+.1f}¢")

    print(f"\n── BY GAP BUCKET {'─'*63}")
    print(f"  {'Bucket':<10}  {'N':>4}  {'Converged':>9}  {'Win rate':>9}  {'Avg P&L':>8}")
    for label, lo, hi in [("3–5¢", 3, 5), ("5–10¢", 5, 10), ("10–20¢", 10, 20), ("20¢+", 20, 999)]:
        sub = [r for r in results if lo <= abs(r["gap"]) < hi]
        if not sub:
            continue
        w = sum(1 for r in sub if r["pnl"] > 0)
        avg = sum(r["pnl"] for r in sub) / len(sub)
        conv = sum(1 for r in sub if r["converged"])
        print(
            f"  {label:<10}  {len(sub):>4}  {conv/len(sub)*100:>8.0f}%"
            f"  {w/len(sub)*100:>8.0f}%  {avg:>+7.1f}¢"
        )

    hdr = f"  {'Ticker':<42}  {'Date':<10}  {'Home':>4}  {'Away':>4}  {'DK%':>5}  {'KMid':>5}  {'→':>1}  {'KClose':>6}  {'Gap':>5}  {'P&L':>6}"
    print(f"\n── TOP GAPS (sorted by |gap|, top 20) {'─'*41}")
    print(hdr)
    for r in sorted(results, key=lambda r: abs(r["gap"]), reverse=True)[:20]:
        print(
            f"  {r['ticker']:<42}  {r['date']:<10}  {r['home']:>4}  {r['away']:>4}"
            f"  {r['dk_open_prob']:>5.1f}  {r['kalshi_mid']:>5.1f}  →"
            f"  {r['kalshi_close']:>6.1f}  {r['gap']:>+5.1f}  {r['pnl']:>+6.1f}¢"
        )

    print(f"\n── LOSSES ({len(losses)}) {'─'*67}")
    print(hdr)
    for r in sorted(losses, key=lambda r: r["pnl"])[:20]:
        print(
            f"  {r['ticker']:<42}  {r['date']:<10}  {r['home']:>4}  {r['away']:>4}"
            f"  {r['dk_open_prob']:>5.1f}  {r['kalshi_mid']:>5.1f}  →"
            f"  {r['kalshi_close']:>6.1f}  {r['gap']:>+5.1f}  {r['pnl']:>+6.1f}¢"
        )

    print(f"\n── TOP WINS ({len(wins)}) {'─'*66}")
    print(hdr)
    for r in sorted(wins, key=lambda r: r["pnl"], reverse=True)[:20]:
        print(
            f"  {r['ticker']:<42}  {r['date']:<10}  {r['home']:>4}  {r['away']:>4}"
            f"  {r['dk_open_prob']:>5.1f}  {r['kalshi_mid']:>5.1f}  →"
            f"  {r['kalshi_close']:>6.1f}  {r['gap']:>+5.1f}  {r['pnl']:>+6.1f}¢"
        )

    print()


def run_backtest() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    run_backtest()
