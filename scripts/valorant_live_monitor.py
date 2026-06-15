"""
Valorant live match monitor — observation-only (no trades placed).

Polls VLR.gg every 30 seconds for live match state.
Prints current map/round scores alongside Kalshi prices and model probabilities.
Flags when model_prob - kalshi_ask exceeds the minimum edge threshold.

The probability model:
  1. Records pre-match Kalshi YES ask at match start (0-0 maps) as series probability.
  2. Back-solves per-map win probability p from that baseline.
  3. For subsequent polls, uses cached p to compute conditional series probability
     given current map/round state.
  4. Edge = model_prob - current_kalshi_ask. Signals when edge >= threshold.

Run during a live VCT match:
  venv/bin/python scripts/valorant_live_monitor.py

Options:
  --min-edge 8     Minimum edge in cents to flag (default 8)
  --no-rounds      Skip VLR.gg HTML scraping for within-map round data
  --once           Single scan, useful for testing outside live matches
"""

import argparse
import asyncio
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kalshi_bot.news.vlr import (
    VALORANT_MIN_EDGE_CENTS,
    fetch_live_matches,
    fetch_open_valorant_markets,
    fetch_orderbook,
    map_win_prob_from_rounds,
    match_key,
    scrape_round_score,
    series_cond_prob,
    series_prob_with_rounds,
    solve_per_map_prob,
    _match_kalshi_markets,
)


LOG_DIR = Path(__file__).parent.parent / "logs"


def setup_logger() -> logging.Logger:
    LOG_DIR.mkdir(exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    log_path = LOG_DIR / f"valorant_{date_str}.log"
    logger = logging.getLogger("valorant_monitor")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(fh)
    print(f"Logging to: {log_path}")
    return logger


def fmt_pct(p: float | None) -> str:
    return f"{p*100:.1f}¢" if p is not None else "  --"


def fmt_team(name: str, width: int = 20) -> str:
    return name[:width].ljust(width)


async def print_live_state(
    session: aiohttp.ClientSession,
    pre_match_cache: dict,
    min_edge_cents: float,
    scrape_rounds: bool,
    logger: logging.Logger | None = None,
    min_map_prob: float = 0.0,
) -> int:
    """Print current live match state with model probabilities. Returns match count."""
    now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    live_matches = await fetch_live_matches(session)
    open_markets = await fetch_open_valorant_markets(session)

    if not live_matches:
        print(f"[{now}] No live Valorant matches.")
        if logger:
            logger.info("POLL no_live_matches")
        return 0

    print(f"\n{'='*95}")
    print(f"  VCT Live Monitor — {now}   ({len(live_matches)} live)")
    print(f"{'='*95}")

    for match in live_matches:
        series_label = f"BO{match.series_format * 2 - 1}"
        mk = match_key(match)
        cached = pre_match_cache.get(mk, {})
        cache_status = "cached" if cached else "no baseline"

        print(f"\n  {match.tournament}  [{series_label}]  [{cache_status}]")
        print(f"  Series: {match.team1}  {match.maps1}–{match.maps2}  {match.team2}", end="")

        # Scrape HTML for map name (API often returns "Unknown") and round fallback
        if scrape_rounds and match.match_page and (match.map_name is None or match.round_state is None):
            rs, map_name = await scrape_round_score(session, match.match_page)
            if match.round_state is None and rs is not None:
                match.round_state = rs
            if match.map_name is None and map_name:
                match.map_name = map_name

        if match.map_name:
            print(f"  |  Map: {match.map_name}", end="")
        if match.round_state:
            rs = match.round_state
            half_label = {1: "1st half", 2: "2nd half", 3: "OT"}.get(rs.half, "")
            print(f"  →  rounds {rs.team1_rounds}–{rs.team2_rounds} ({half_label})", end="")
        print()

        # Find Kalshi markets
        team_markets = _match_kalshi_markets(match, open_markets)
        if len(team_markets) < 2:
            print(f"  [!] Kalshi market match failed ({len(team_markets)}/2 found)")
            print(f"      Looking for: {match.team1!r}  /  {match.team2!r}")
            continue

        # Populate cache at pre-match state
        if mk not in pre_match_cache:
            if match.maps1 == 0 and match.maps2 == 0:
                cache_entry = {}
                for team, mkt in team_markets.items():
                    ob = await fetch_orderbook(session, mkt.get("ticker", ""))
                    ya = ob.get("yes_ask")
                    if ya is not None and 0.05 < ya < 0.95:
                        try:
                            cache_entry[team] = solve_per_map_prob(ya, match.series_format)
                        except Exception:
                            pass
                    await asyncio.sleep(0.1)
                if cache_entry:
                    pre_match_cache[mk] = cache_entry
                    print(f"  [+] Pre-match cache populated: "
                          + ", ".join(f"{t}: p_map={v:.3f}" for t, v in cache_entry.items()))

        print()
        print(f"  {'Team':<22} {'Maps':>5}  {'Bid':>8} {'Ask':>8}  {'Model':>8}  {'Edge':>7}  "
              f"{'p_map':>6}  {'Depth$':>8}  Signal")
        print("  " + "-" * 90)

        for team in [match.team1, match.team2]:
            mkt = team_markets.get(team)
            if mkt is None:
                print(f"  {fmt_team(team)}  (no market)")
                continue

            ticker = mkt.get("ticker", "")
            ob = await fetch_orderbook(session, ticker)
            yes_bid = ob.get("yes_bid")
            yes_ask = ob.get("yes_ask")
            depth = ob.get("depth_usd", 0.0)
            await asyncio.sleep(0.1)

            is_team1 = (team == match.team1)
            maps_a = match.maps1 if is_team1 else match.maps2
            maps_b = match.maps2 if is_team1 else match.maps1
            map_score = f"{maps_a}–{maps_b}"

            per_map_p = pre_match_cache.get(mk, {}).get(team)
            model_p = None
            edge = None

            if per_map_p is not None and yes_ask is not None:
                try:
                    if match.round_state is not None:
                        r1 = match.round_state.team1_rounds if is_team1 else match.round_state.team2_rounds
                        r2 = match.round_state.team2_rounds if is_team1 else match.round_state.team1_rounds
                        model_p = series_prob_with_rounds(
                            maps_a, maps_b, match.series_format, per_map_p,
                            r1, r2, match.round_state.half
                        )
                    else:
                        model_p = series_cond_prob(maps_a, maps_b, match.series_format, per_map_p)
                    edge = model_p - yes_ask
                except Exception:
                    pass

            # Optional: suppress round-level signals unless map outcome is near-certain
            round_signal_ok = True
            if match.round_state is not None and min_map_prob > 0:
                rs = match.round_state
                r1 = rs.team1_rounds if is_team1 else rs.team2_rounds
                r2 = rs.team2_rounds if is_team1 else rs.team1_rounds
                map_p = map_win_prob_from_rounds(r1, r2, rs.half)
                if map_p < min_map_prob:
                    round_signal_ok = False

            signal = ""
            if edge is not None and round_signal_ok:
                if edge * 100 >= min_edge_cents and depth >= 500:
                    signal = "*** BUY ***"
                elif edge * 100 >= min_edge_cents:
                    signal = "thin mkt"
                elif edge * 100 >= min_edge_cents / 2:
                    signal = "watch"

            p_map_str = f"{per_map_p:.3f}" if per_map_p is not None else "  --"

            print(f"  {fmt_team(team)}  {map_score:>5}  {fmt_pct(yes_bid):>8} {fmt_pct(yes_ask):>8}  "
                  f"{fmt_pct(model_p):>8}  {fmt_pct(edge):>7}  {p_map_str:>6}  ${depth:>7.0f}  {signal}")

            # Log every row; flag signals prominently
            if logger:
                rs = match.round_state
                round_str = f" rounds={rs.team1_rounds if is_team1 else rs.team2_rounds}-{rs.team2_rounds if is_team1 else rs.team1_rounds}" if rs else ""
                row = (f"STATE {match.team1} vs {match.team2} | {team} maps={map_score}{round_str} "
                       f"bid={fmt_pct(yes_bid)} ask={fmt_pct(yes_ask)} "
                       f"model={fmt_pct(model_p)} edge={fmt_pct(edge)} depth=${depth:.0f}")
                logger.info(row)
                if signal in ("*** BUY ***", "thin mkt", "watch"):
                    logger.info(
                        f"SIGNAL[{signal.strip('* ')}] {team} ticker={ticker} "
                        f"ask={fmt_pct(yes_ask)} model={fmt_pct(model_p)} "
                        f"edge={fmt_pct(edge)} depth=${depth:.0f} basis=map{map_score}{round_str}"
                    )

    print()
    return len(live_matches)


async def prepopulate_cache(
    session: aiohttp.ClientSession,
    pre_match_cache: dict,
    upcoming_segments: list[dict],
    open_markets: list[dict],
) -> None:
    """
    Pre-fill per-map-p cache from current Kalshi prices for upcoming (not-yet-live) matches.
    Matches are identified by fuzzy VLR team name → Kalshi title lookup.
    Only populates entries not already in the cache.
    """
    from kalshi_bot.news.vlr import LiveMatch, _series_format_from_tournament
    for seg in upcoming_segments:
        team1 = seg.get("team1", "")
        team2 = seg.get("team2", "")
        tournament = seg.get("match_event", "")
        if not team1 or not team2:
            continue
        fake_match = LiveMatch(
            team1=team1, team2=team2, maps1=0, maps2=0,
            series_format=_series_format_from_tournament(tournament),
            tournament=tournament,
        )
        mk = match_key(fake_match)
        if mk in pre_match_cache:
            continue
        team_markets = _match_kalshi_markets(fake_match, open_markets)
        if len(team_markets) < 2:
            continue
        cache_entry = {}
        for team, mkt in team_markets.items():
            ob = await fetch_orderbook(session, mkt.get("ticker", ""))
            ya = ob.get("yes_ask")
            if ya is not None and 0.05 < ya < 0.95:
                try:
                    cache_entry[team] = solve_per_map_prob(ya, fake_match.series_format)
                except Exception:
                    pass
            await asyncio.sleep(0.05)
        if cache_entry:
            pre_match_cache[mk] = cache_entry


async def show_upcoming(session: aiohttp.ClientSession) -> None:
    """Print upcoming match prices (for situational awareness while waiting)."""
    open_markets = await fetch_open_valorant_markets(session)
    by_match: dict[str, list[dict]] = defaultdict(list)
    for m in open_markets:
        ticker = m.get("ticker", "")
        mk = "-".join(ticker.split("-")[:-1])
        by_match[mk].append(m)

    if not by_match:
        print("  No open KXVALORANTGAME markets found.")
        return

    # Fetch VLR upcoming list to get match times
    upcoming: list[dict] = []
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get("https://vlrggapi.vercel.app/match?q=upcoming",
                             timeout=aiohttp.ClientTimeout(total=10)) as r:
                data = await r.json()
        upcoming = data.get("data", {}).get("segments", [])
    except Exception:
        pass

    # Build time lookup from VLR team names
    time_map: dict[str, str] = {}
    for seg in upcoming:
        t = seg.get("time_until_match", "")
        for team in [seg.get("team1", ""), seg.get("team2", "")]:
            if team:
                time_map[team.lower()] = t

    print(f"\n{'='*85}")
    print(f"  Upcoming KXVALORANTGAME markets — {datetime.now(timezone.utc).strftime('%H:%M UTC')}")
    print(f"{'='*85}")
    print(f"  {'Match key':<28} {'Team':>6}  {'Bid':>8} {'Ask':>8}  In")
    print("  " + "-" * 65)

    for mk in sorted(by_match)[:14]:
        pair = by_match[mk]
        for m in pair:
            ticker = m.get("ticker", "")
            team_code = ticker.split("-")[-1]
            title = m.get("title", "")
            ob = await fetch_orderbook(session, ticker)
            bid, ask = ob.get("yes_bid"), ob.get("yes_ask")
            # Try to get time from title team name
            time_str = ""
            for t_lower, t_time in time_map.items():
                if t_lower in title.lower():
                    time_str = t_time
                    break
            short_mk = mk.split("GAME-")[-1] if "GAME-" in mk else mk[-28:]
            print(f"  {short_mk:<28} {team_code:>6}  {fmt_pct(bid):>8} {fmt_pct(ask):>8}  {time_str}")
            await asyncio.sleep(0.05)
        print()


async def _fetch_vlr_upcoming(session: aiohttp.ClientSession) -> list[dict]:
    try:
        async with session.get("https://vlrggapi.vercel.app/match?q=upcoming",
                               timeout=aiohttp.ClientTimeout(total=10)) as r:
            data = await r.json()
        return data.get("data", {}).get("segments", [])
    except Exception:
        return []


async def main(min_edge_cents: float, scrape_rounds: bool, once: bool, min_map_prob: float = 0.0) -> None:
    poll_interval = 30  # seconds
    cache_repopulate_interval = 300  # re-check upcoming prices every 5 min
    pre_match_cache: dict = {}
    logger = None if once else setup_logger()

    async with aiohttp.ClientSession() as session:
        # Seed cache from currently-priced upcoming markets
        open_markets = await fetch_open_valorant_markets(session)
        upcoming_segs = await _fetch_vlr_upcoming(session)
        await prepopulate_cache(session, pre_match_cache, upcoming_segs, open_markets)
        if pre_match_cache:
            msg = f"Pre-match cache seeded: {len(pre_match_cache)} match(es) cached"
            print(msg)
            if logger:
                logger.info(msg)

        if once:
            print("=== Upcoming Kalshi Valorant Markets ===")
            await show_upcoming(session)
            print("\n=== Live Match Check ===")
            n = await print_live_state(session, pre_match_cache, min_edge_cents, scrape_rounds)
            if n == 0:
                print("  Run this script during a live VCT match to see live signals.")
            return

        map_prob_str = f"  |  min_map_prob={min_map_prob:.0%}" if min_map_prob > 0 else ""
        print(f"Valorant live monitor  |  poll={poll_interval}s  "
              f"|  min_edge={min_edge_cents:.0f}¢  |  rounds={'ON' if scrape_rounds else 'OFF'}{map_prob_str}")
        print("Ctrl+C to stop.\n")

        polls = 0
        while True:
            try:
                if polls % (cache_repopulate_interval // poll_interval) == 0 and polls > 0:
                    om = await fetch_open_valorant_markets(session)
                    us = await _fetch_vlr_upcoming(session)
                    await prepopulate_cache(session, pre_match_cache, us, om)

                await print_live_state(session, pre_match_cache, min_edge_cents, scrape_rounds, logger, min_map_prob)
                polls += 1
            except Exception as exc:
                err = f"[ERROR] {exc}"
                print(err)
                if logger:
                    logger.error(err)
            await asyncio.sleep(poll_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-edge", type=float, default=VALORANT_MIN_EDGE_CENTS,
                        help=f"Minimum edge in cents to flag (default: {VALORANT_MIN_EDGE_CENTS})")
    parser.add_argument("--no-rounds", action="store_true",
                        help="Skip VLR.gg HTML scraping for round data")
    parser.add_argument("--min-map-prob", type=float, default=0.0,
                        help="Only flag signals when P(win current map) >= this (e.g. 0.90). Default: off")
    parser.add_argument("--once", action="store_true",
                        help="Run once and exit")
    args = parser.parse_args()

    asyncio.run(main(
        min_edge_cents=args.min_edge,
        scrape_rounds=not args.no_rounds,
        once=args.once,
        min_map_prob=args.min_map_prob,
    ))
