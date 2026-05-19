"""NBA Kalshi↔Pinnacle convergence signal generator.

Consumes already-fetched Pinnacle game data and KXNBAGAME Kalshi markets (both
available from the main poll loop).  Produces NBAConvergenceOpportunity objects
when the Kalshi mid-price lags Pinnacle's de-vigged implied probability by
enough to clear the backtested entry filter.

Backtest results (2025-26 season, 217 liquid KXNBAGAME trades):
  - 98.2% convergence rate, average hold time 15.5 hours.
  - Entry filter (approved 2026-05-14):
      gap ≥ 10¢: viable at any spread ≤ 15¢ (win rate 93–100%)
      gap 5–10¢: viable only at spread < 5¢  (win rate 94%)
      gap 5–10¢ + spread 5–10¢: EXCLUDED (62% win rate — insufficient edge)
  - P-values from gap × spread matrix used directly in Kelly sizing.

Env vars:
    NBA_CONVERGENCE_ENABLED  "true" | "false" — default "true".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .utils import env_bool
from .news.pinnacle import _PINNACLE_TO_ABBREV, _parse_nba_ticker

NBA_CONVERGENCE_ENABLED: bool = env_bool("NBA_CONVERGENCE_ENABLED", True)

# Minimum hours until market close to allow entry.  NBA markets close when the
# game ends — if close_time is within this window the game is live and Pinnacle
# is showing in-game lines, not pre-game.  In-game Pinnacle lines move rapidly
# on scoring runs, creating fake gaps that vanish two polls later.
# Default 5h: typical NBA game takes ~2.5h, so 5h before close ≈ game has started.
_MIN_HOURS_TO_CLOSE: float = 5.0

# Maximum hours until market close to allow entry.  Kalshi NBA prices don't
# track Pinnacle pre-game — they only converge once the game is live and
# scoring runs move both books together.  Entering 10+ hours before close
# (6+ hours before tip-off) just takes on directional game risk with no
# convergence mechanism.  Default 8h: tip-off is ~2.5h before close, so
# 8h before close ≈ ~5.5h before tip-off — blocks distant pre-game entries.
_MAX_HOURS_TO_CLOSE: float = 8.0

# Minimum yes_bid in cents to exclude stub-bid / illiquid markets.
_MIN_BID: int = 5

# Backtested win probabilities per (gap_bucket, spread_bucket).
# Source: 217 KXNBAGAME trades, 2025-26 season.
# The (5-10¢, 5-10¢) bucket (62%) is intentionally absent — excluded per user.
_WIN_PROB: dict[tuple[str, str], float] = {
    ("10+", "<5"):    1.00,
    ("10+", "5-10"):  1.00,
    ("10+", "10-15"): 0.93,
    ("5-10", "<5"):   0.94,
}


def _win_prob(gap_abs: float, spread: float) -> float:
    """Return backtested win probability for the given gap and spread (in cents)."""
    gb = "10+" if gap_abs >= 10 else "5-10"
    if spread < 5:
        sb = "<5"
    elif spread < 10:
        sb = "5-10"
    else:
        sb = "10-15"
    return _WIN_PROB.get((gb, sb), 0.0)


def _passes_entry_filter(gap_abs: float, spread: float) -> bool:
    """Return True if the signal clears the backtested entry threshold."""
    if gap_abs >= 10 and spread <= 15:
        return True
    if gap_abs >= 5 and spread < 5:
        return True
    return False


@dataclass
class NBAConvergenceOpportunity:
    kalshi_ticker: str          # e.g. "KXNBAGAME-26MAY15DENGSW-DEN"
    home_team: str              # "Denver Nuggets"
    away_team: str
    game_date: str              # "YYYY-MM-DD"
    pinnacle_prob: float        # de-vigged Pinnacle probability for this YES side (0–1)
    kalshi_mid: float           # (yes_bid + yes_ask) / 2 in cents
    kalshi_bid: int             # yes_bid in cents
    kalshi_ask: int             # yes_ask in cents
    gap: float                  # pinnacle_prob*100 − kalshi_mid  (positive = buy YES)
    open_spread: float          # yes_ask − yes_bid in cents
    side: str                   # "YES" | "NO"
    win_probability: float      # from backtest lookup table
    target_bid: float           # bid target for the convergence exit (in cents)
    source: str = field(default="pinnacle")


def find_opportunities(
    pinnacle_games: list[dict],
    kalshi_nba_markets: list[dict],
) -> list[NBAConvergenceOpportunity]:
    """Identify Kalshi NBA markets lagging Pinnacle's implied probability.

    Args:
        pinnacle_games:      Output of pinnacle.fetch_nba_games().
                             Each dict has: home_team, away_team, home_prob, away_prob.
        kalshi_nba_markets:  Markets from fetch_markets_by_series("KXNBAGAME").
                             Each dict has: ticker, yes_bid, yes_ask.

    Returns:
        List of NBAConvergenceOpportunity (already filtered by entry gate).
    """
    if not NBA_CONVERGENCE_ENABLED:
        return []
    if not pinnacle_games or not kalshi_nba_markets:
        return []

    # Build lookup: frozenset({abbrev1, abbrev2}) → {side_abbrev → market_dict}
    kalshi_by_pair: dict[frozenset, dict[str, dict]] = {}
    for m in kalshi_nba_markets:
        ticker = m.get("ticker", "")
        parsed = _parse_nba_ticker(ticker)
        if not parsed:
            continue
        t1, t2 = parsed
        # The last segment of the ticker is the YES side's team abbreviation.
        side_abbrev = ticker.split("-")[-1]
        key = frozenset({t1, t2})
        kalshi_by_pair.setdefault(key, {})[side_abbrev] = m

    opportunities: list[NBAConvergenceOpportunity] = []

    for game in pinnacle_games:
        home_abbrev = _PINNACLE_TO_ABBREV.get(game["home_team"].lower())
        away_abbrev = _PINNACLE_TO_ABBREV.get(game["away_team"].lower())
        if not home_abbrev or not away_abbrev:
            logging.debug(
                "NBA convergence: no abbrev for '%s' vs '%s'",
                game["home_team"], game["away_team"],
            )
            continue

        pair_key = frozenset({home_abbrev, away_abbrev})
        pair_markets = kalshi_by_pair.get(pair_key)
        if not pair_markets:
            continue

        game_date   = game.get("game_date", "")
        home_prob   = game.get("home_prob", 0.0)
        away_prob   = game.get("away_prob", 0.0)

        # Each team's YES ticker represents: "will THIS team win?"
        # home YES market → check against home_prob
        # away YES market → check against away_prob
        for abbrev, pinnacle_prob in [(home_abbrev, home_prob), (away_abbrev, away_prob)]:
            m = pair_markets.get(abbrev)
            if not m:
                continue

            yes_bid = m.get("yes_bid")
            yes_ask = m.get("yes_ask")
            if yes_bid is None or yes_ask is None:
                continue
            yes_bid = int(yes_bid)
            yes_ask = int(yes_ask)

            # Stub-bid filter: exclude markets with an essentially-zero bid.
            if yes_bid < _MIN_BID:
                continue

            # In-game filter: if the market closes within _MIN_HOURS_TO_CLOSE,
            # the game is live and Pinnacle lines are in-game (not pre-game).
            # In-game Pinnacle lines spike on scoring runs, creating transient
            # fake gaps that vanish two polls later.
            # expected_expiration_time is the actual game end time; close_time
            # is the series-level max expiration (weeks away) and is useless here.
            _close_str = (m.get("expected_expiration_time")
                          or m.get("close_time")
                          or m.get("expiration_time", ""))
            if _close_str:
                try:
                    from .strike_arb import parse_iso_dt
                    _close_dt = parse_iso_dt(_close_str)
                    if _close_dt is not None:
                        _htc = (_close_dt - datetime.now(timezone.utc)).total_seconds() / 3600
                        if _htc < _MIN_HOURS_TO_CLOSE:
                            logging.debug(
                                "NBA convergence skip (game live): %s closes in %.1fh < %.1fh",
                                m.get("ticker", "?"), _htc, _MIN_HOURS_TO_CLOSE,
                            )
                            continue
                        if _htc > _MAX_HOURS_TO_CLOSE:
                            logging.debug(
                                "NBA convergence skip (too early): %s closes in %.1fh > %.1fh",
                                m.get("ticker", "?"), _htc, _MAX_HOURS_TO_CLOSE,
                            )
                            continue
                except Exception:
                    pass

            spread = float(yes_ask - yes_bid)
            mid    = (yes_bid + yes_ask) / 2.0
            gap    = pinnacle_prob * 100.0 - mid  # positive = Kalshi underpricing YES

            gap_abs = abs(gap)

            if not _passes_entry_filter(gap_abs, spread):
                continue

            wp = _win_prob(gap_abs, spread)
            if wp == 0.0:
                # No valid backtest bucket — skip.
                continue

            if gap > 0:
                # Kalshi is underpricing YES → buy YES
                side = "YES"
                target_bid = pinnacle_prob * 100.0 - 2.0
            else:
                # Kalshi is overpricing YES → buy NO
                side = "NO"
                target_bid = (1.0 - pinnacle_prob) * 100.0 - 2.0

            if target_bid <= 0:
                continue

            ticker = m.get("ticker", "")
            opportunities.append(NBAConvergenceOpportunity(
                kalshi_ticker=ticker,
                home_team=game["home_team"],
                away_team=game["away_team"],
                game_date=game_date,
                pinnacle_prob=pinnacle_prob,
                kalshi_mid=round(mid, 2),
                kalshi_bid=yes_bid,
                kalshi_ask=yes_ask,
                gap=round(gap, 2),
                open_spread=spread,
                side=side,
                win_probability=wp,
                target_bid=round(target_bid, 2),
            ))

    logging.info(
        "NBA convergence: %d opportunity(ies) found.", len(opportunities)
    )
    return opportunities
