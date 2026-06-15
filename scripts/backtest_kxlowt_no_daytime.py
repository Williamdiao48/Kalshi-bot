"""Backtest daytime KXLOWT NO entry using confirmed running minimum.

Hypothesis: once the overnight low has already been recorded for the day
(running min > band ceiling + clearance), entering NO is near risk-free
because temperatures can only rise during the afternoon.

For each settled KXLOWT between market, and for each (entry_hour, clearance):
  - entry_hour: local hour at which we check conditions (10–17)
  - clearance:  how far (°F) the running min must be ABOVE the band ceiling
                before we enter; accounts for Kalshi rounding (±0.5°F)
  - If running_min_at_hour > strike_hi + clearance → signal fires
  - Entry price: NO ask from candlesticks at that local hour
  - Outcome: market settlement (YES/NO)

Output: grid of (entry_hour × clearance) showing signal rate, win rate,
avg NO cost, and avg EV.

Usage:
  venv/bin/python scripts/backtest_kxlowt_no_daytime.py
  venv/bin/python scripts/backtest_kxlowt_no_daytime.py --years 3 --months 4 5 6
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sqlite3
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).parent.parent))

from kalshi_bot.news.noaa import LOW_CITIES, KALSHI_STATION_IDS  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_MESONET_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
_FETCH_DELAY  = 0.5
_MIN_OBS_PER_DAY = 10

CANDLES_DB  = Path("data/candlesticks.db")

_ENTRY_HOURS        = list(range(9, 18))     # 9 AM – 5 PM local
_CLEARANCE_VALS     = [0.5, 1.0, 1.5, 2.0, 3.0]


def _low_station(metric: str) -> str | None:
    return KALSHI_STATION_IDS.get(metric.replace("temp_low_", "temp_high_"))


# ---------------------------------------------------------------------------
# ASOS fetch (reused from backtest_kxlowt_yes_gate.py)
# ---------------------------------------------------------------------------

async def _fetch_city_obs(
    session: aiohttp.ClientSession,
    metric:  str,
    station: str,
    start_dt: date,
    end_dt:   date,
) -> dict[int, list[tuple[int, float]]]:
    """Fetch ASOS → ordinal → sorted [(lst_minute, temp_f)]."""
    _, _, _, city_tz = LOW_CITIES[metric]
    std_offset = city_tz.utcoffset(datetime(2000, 1, 15))
    lst_tz = timezone(std_offset)

    fetch_start = start_dt - timedelta(days=1)
    fetch_end   = end_dt   + timedelta(days=1)

    params = {
        "station": station, "data": "tmpf",
        "year1": str(fetch_start.year), "month1": str(fetch_start.month),
        "day1":  str(fetch_start.day),
        "year2": str(fetch_end.year),   "month2": str(fetch_end.month),
        "day2":  str(fetch_end.day),
        "tz": "UTC", "format": "comma", "latlon": "no",
        "missing": "M", "trace": "T", "direct": "no",
        "report_type": "3,4",
    }
    try:
        async with session.get(
            _MESONET_URL, params=params,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            resp.raise_for_status()
            text = await resp.text()
    except Exception as exc:
        log.error("Fetch failed %s (%s): %s", metric, station, exc)
        return {}

    obs_by_ordinal: dict[int, list[tuple[int, float]]] = {}
    for line in text.splitlines():
        if line.startswith("#") or line.startswith("station") or not line.strip():
            continue
        parts = line.split(",")
        if len(parts) < 3:
            continue
        try:
            utc_ts   = datetime.strptime(parts[1].strip(), "%Y-%m-%d %H:%M").replace(
                tzinfo=timezone.utc)
            temp_str = parts[2].strip()
            if temp_str in ("M", "T", ""):
                continue
            temp_f = float(temp_str)
        except (ValueError, IndexError):
            continue
        local_ts = utc_ts.astimezone(lst_tz)
        ordinal  = local_ts.toordinal()
        lst_min  = local_ts.hour * 60 + local_ts.minute
        obs_by_ordinal.setdefault(ordinal, []).append((lst_min, temp_f))

    for ordinal in obs_by_ordinal:
        obs_by_ordinal[ordinal].sort()

    return obs_by_ordinal


# ---------------------------------------------------------------------------
# Candlestick helpers
# ---------------------------------------------------------------------------

def _load_candles(tickers: list[str]) -> dict[str, list[tuple[int, int, int, int, int]]]:
    """Load candles for the given tickers.

    Returns {ticker: [(ts_unix, open, high, low, close), ...]} sorted by ts.
    """
    if not CANDLES_DB.exists():
        log.warning("Candlesticks DB not found: %s", CANDLES_DB)
        return {}
    conn = sqlite3.connect(CANDLES_DB)
    result: dict[str, list] = {}
    placeholders = ",".join("?" * len(tickers))
    rows = conn.execute(
        f"SELECT ticker, period_ts, bid_close, ask_close FROM candles "
        f"WHERE ticker IN ({placeholders}) ORDER BY ticker, period_ts",
        tickers,
    ).fetchall()
    conn.close()
    for ticker, ts, bid, ask in rows:
        result.setdefault(ticker, []).append((ts, bid, ask))
    return result


def _no_ask_at_hour(
    candles: list[tuple[int, int, int]],
    target_unix: int,
    window: int = 3600,
) -> int | None:
    """Return the NO ask (100 − yes_bid) closest to target_unix within ±window."""
    best = None
    best_dt = window + 1
    for ts, bid, ask in candles:
        dt = abs(ts - target_unix)
        if dt <= window and dt < best_dt and bid is not None:
            best_dt = dt
            best = 100 - bid   # NO ask = 100 − yes_bid
    return best


# ---------------------------------------------------------------------------
# Market parsing
# ---------------------------------------------------------------------------

def _parse_ticker(ticker: str) -> tuple[str | None, float | None, float | None]:
    """Extract (metric_key, strike_lo, strike_hi) from KXLOWT ticker."""
    # e.g. KXLOWTNYC-26MAY10-B46.5  or  KXLOWTSATX-26MAY10-T54
    import re
    m = re.match(
        r"KXLOWT([A-Z]+)-(\d{2})[A-Z]{3}(\d{2})-[BT]([\d.]+)$", ticker
    )
    if not m:
        return None, None, None
    city_code = m.group(1).lower()
    strike    = float(m.group(4))
    # Detect between vs under
    is_between = "-B" in ticker
    if is_between:
        return f"temp_low_{city_code}", strike, strike + 1.0
    else:
        return f"temp_low_{city_code}", None, strike   # T = under; YES if low <= strike


def _city_code_map() -> dict[str, str]:
    """Map Kalshi city suffix → metric key prefix."""
    mapping = {}
    for metric in LOW_CITIES:
        suffix = metric.replace("temp_low_", "").upper()
        # handle aliased codes
        mapping[suffix] = metric
    # a few non-obvious mappings
    extras = {
        "NYC": "temp_low_ny",
        "SATX": "temp_low_sat",
        "NOLA": "temp_low_msy",
        "DCA": "temp_low_dca",
        "DFW": "temp_low_dfw",
    }
    mapping.update(extras)
    return mapping


_CITY_MAP = _city_code_map()


def _ticker_to_metric(ticker: str) -> str | None:
    import re
    m = re.match(r"KXLOWT([A-Z]+)-", ticker)
    if not m:
        return None
    code = m.group(1)
    return _CITY_MAP.get(code)


# ---------------------------------------------------------------------------
# Simulation core
# ---------------------------------------------------------------------------

def _simulate(
    ticker:    str,
    outcome:   str,     # 'won' | 'lost'  (from NO perspective)
    strike_lo: float | None,
    strike_hi: float | None,
    candles:   list[tuple[int, int, int]],
    obs:       list[tuple[int, float]],   # sorted [(lst_minute, temp_f)]
    city_tz,
    market_date: date,   # LST date the market settles on
) -> dict:
    """Return {(entry_hour, clearance): SimResult} for this market."""
    if not obs or not candles:
        return {}

    # Only handle "between" markets here (strike_lo and strike_hi both set).
    # For "under" markets (only strike_hi), the logic is similar but skip for now.
    if strike_lo is None or strike_hi is None:
        return {}

    # Kalshi resolves "between": YES if final_low in [strike_lo-0.5, strike_hi+0.5)
    # NO wins if final_low > strike_hi (too warm) OR final_low < strike_lo (too cold)
    # We're only interested in the "too warm" NO case — running_min stays above ceiling.
    band_ceiling = strike_hi  # if running min never drops to strike_hi or below → NO wins

    # Compute running min at each hour of the day (LST)
    # obs is sorted by (lst_minute, temp_f)
    running_min_at_hour: dict[int, float] = {}
    cur_min = float("inf")
    obs_idx = 0
    for hour in range(0, 24):
        cutoff = hour * 60 + 59
        while obs_idx < len(obs) and obs[obs_idx][0] <= cutoff:
            cur_min = min(cur_min, obs[obs_idx][1])
            obs_idx += 1
        if cur_min < float("inf"):
            running_min_at_hour[hour] = cur_min

    # Convert market_date + entry_hour to unix timestamp for candlestick lookup.
    # Use standard time offset (same as LST bucketing).
    std_offset = city_tz.utcoffset(datetime(2000, 1, 15))
    lst_tz = timezone(std_offset)

    results = {}
    for hour in _ENTRY_HOURS:
        rmin = running_min_at_hour.get(hour)
        if rmin is None:
            continue

        # Target unix timestamp for this entry hour
        entry_dt_lst = datetime(
            market_date.year, market_date.month, market_date.day,
            hour, 0, 0, tzinfo=lst_tz
        )
        entry_unix = int(entry_dt_lst.timestamp())

        no_ask = _no_ask_at_hour(candles, entry_unix)
        if no_ask is None or no_ask <= 0 or no_ask >= 100:
            continue

        won_no = (outcome == "won")   # outcome from the DB is from NO's perspective

        for clearance in _CLEARANCE_VALS:
            # Signal fires when running min is already above ceiling + clearance
            if rmin <= band_ceiling + clearance:
                continue   # low might still be in the band

            pnl = (100 - no_ask) if won_no else -no_ask
            results[(hour, clearance)] = {
                "no_ask":  no_ask,
                "rmin":    rmin,
                "pnl":     pnl,
                "outcome": "win" if won_no else "loss",
            }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(
    years:        int,
    months:       list[int] | None,
    city_filter:  set[str] | None,
) -> None:
    end_dt   = date.today()
    start_dt = end_dt - timedelta(days=365 * years)
    month_set = set(months) if months else None

    # Load settled KXLOWT between markets from candlesticks DB
    if not CANDLES_DB.exists():
        log.error("Candlesticks DB not found: %s", CANDLES_DB)
        return

    conn = sqlite3.connect(CANDLES_DB)
    tickers_raw = conn.execute(
        "SELECT DISTINCT ticker FROM candles WHERE ticker LIKE 'KXLOWT%B%'"
    ).fetchall()
    conn.close()

    all_tickers = [r[0] for r in tickers_raw]

    # Derive settlement from the final bid_close:
    #   bid_close >= 90 at last candle → YES resolved (NO lost)
    #   bid_close <= 10 at last candle → NO resolved (NO won)
    conn = sqlite3.connect(CANDLES_DB)
    final_prices = conn.execute(
        "SELECT ticker, bid_close FROM candles c1 "
        "WHERE ticker LIKE 'KXLOWT%B%' "
        "  AND period_ts = (SELECT MAX(period_ts) FROM candles c2 WHERE c2.ticker = c1.ticker)"
    ).fetchall()
    conn.close()

    settlement: dict[str, str] = {}  # ticker → 'won'|'lost' from NO perspective
    for ticker, last_bid in final_prices:
        if last_bid is None:
            continue
        if last_bid >= 90:
            settlement[ticker] = "lost"   # YES won → NO lost
        elif last_bid <= 10:
            settlement[ticker] = "won"    # NO won
        # markets in the middle are ambiguous — skip

    resolved_tickers = [t for t in all_tickers if t in settlement]
    log.info("%d resolved KXLOWT between tickers", len(resolved_tickers))

    # Filter by date range and month
    def _ticker_date(ticker: str) -> date | None:
        import re
        m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})-", ticker)
        if not m:
            return None
        months_abbr = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
                       "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}
        yr  = 2000 + int(m.group(1))   # format: YYMONDD e.g. 26APR07 = Apr 7 2026
        mon = months_abbr.get(m.group(2), 0)
        day = int(m.group(3))
        try:
            return date(yr, mon, day)
        except ValueError:
            return None

    filtered = []
    for t in resolved_tickers:
        d = _ticker_date(t)
        if d is None:
            continue
        if d < start_dt or d > end_dt:
            continue
        if month_set and d.month not in month_set:
            continue
        metric = _ticker_to_metric(t)
        if metric is None:
            continue
        if city_filter and metric not in city_filter:
            continue
        filtered.append((t, d, metric))

    log.info("%d tickers after date/month filter", len(filtered))
    if not filtered:
        log.error("No tickers to analyze.")
        return

    # Group by metric/city
    by_metric: dict[str, list[tuple[str, date]]] = {}
    for ticker, d, metric in filtered:
        by_metric.setdefault(metric, []).append((ticker, d))

    # Preload all candles
    ticker_list = [t for t, _, _ in filtered]
    log.info("Loading %d tickers from candlesticks DB …", len(ticker_list))
    candles_map = _load_candles(ticker_list)

    # Fetch ASOS data city by city
    all_obs: dict[str, dict[int, list[tuple[int, float]]]] = {}
    async with aiohttp.ClientSession() as session:
        for i, (metric, pairs) in enumerate(sorted(by_metric.items()), 1):
            station = _low_station(metric)
            if not station:
                log.warning("No station for %s, skipping", metric)
                continue
            dates = [d for _, d in pairs]
            s_dt = min(dates)
            e_dt = max(dates)
            log.info("[%d/%d] %s (%s)  %s → %s",
                     i, len(by_metric), metric, station, s_dt, e_dt)
            obs = await _fetch_city_obs(session, metric, station, s_dt, e_dt)
            all_obs[metric] = obs
            if i < len(by_metric):
                await asyncio.sleep(_FETCH_DELAY)

    # Simulate
    # Accumulate: {(hour, clearance): [pnl, ...]}
    from collections import defaultdict
    grid: dict[tuple[int, float], list[float]] = defaultdict(list)
    # Also track: total markets analyzed, total signals fired
    total_markets = 0

    for ticker, d, metric in filtered:
        obs_by_ordinal = all_obs.get(metric, {})
        ordinal = d.toordinal()
        obs = obs_by_ordinal.get(ordinal, [])
        if not obs or len(obs) < _MIN_OBS_PER_DAY:
            continue

        candles = candles_map.get(ticker, [])
        if not candles:
            continue

        _, city_tz = LOW_CITIES[metric][0], LOW_CITIES[metric][3]
        outcome = settlement[ticker]

        # Parse band
        _, strike_lo, strike_hi = _parse_ticker(ticker)
        if strike_lo is None or strike_hi is None:
            continue

        total_markets += 1
        sim = _simulate(ticker, outcome, strike_lo, strike_hi,
                        candles, obs, city_tz, d)
        for key, res in sim.items():
            grid[key].append(res["pnl"])

    if not total_markets:
        log.error("No markets with sufficient data.")
        return

    no_res_count = sum(1 for t, _, _ in filtered if settlement.get(t) == "won")
    yes_res_count = sum(1 for t, _, _ in filtered if settlement.get(t) == "lost")

    print(f"\n{'='*78}")
    print(f"  KXLOWT NO — Daytime Entry Backtest")
    print(f"  Markets: {total_markets}  (NO resolved: {no_res_count}, YES resolved: {yes_res_count})")
    print(f"  Date range: {start_dt} → {end_dt}")
    if month_set:
        print(f"  Months: {sorted(month_set)}")
    print(f"{'='*78}")

    # Print grid: rows = clearance, cols = entry_hour
    for clearance in _CLEARANCE_VALS:
        print(f"\n  Clearance ≥ {clearance:.1f}°F above band ceiling")
        print(f"  {'Hour':>5}  {'N':>5}  {'Win%':>6}  {'AvgCost':>8}  {'AvgEV':>8}  {'TotalEV':>9}")
        print(f"  {'-'*55}")
        for hour in _ENTRY_HOURS:
            pnls = grid.get((hour, clearance), [])
            if not pnls:
                print(f"  {hour:>5}h  {'—':>5}")
                continue
            n    = len(pnls)
            wins = sum(1 for p in pnls if p > 0)
            # Reconstruct avg no_ask from pnls: wins → pnl = 100-ask, losses → pnl = -ask
            # Can't recover ask perfectly; use win/loss split
            avg_ev = sum(pnls) / n
            win_pct = wins / n * 100
            total_ev = sum(pnls)
            print(f"  {hour:>5}h  {n:>5}  {win_pct:>5.1f}%  {'—':>8}  {avg_ev:>+7.1f}¢  {total_ev:>+8.0f}¢")
        # Best hour for this clearance
        best_h = max(
            (h for h in _ENTRY_HOURS if grid.get((h, clearance))),
            key=lambda h: sum(grid[(h, clearance)]) / len(grid[(h, clearance)]),
            default=None,
        )
        if best_h:
            pnls = grid[(best_h, clearance)]
            n = len(pnls)
            avg_ev = sum(pnls) / n
            wins   = sum(1 for p in pnls if p > 0)
            print(f"  → best: {best_h}h  n={n}  win={wins/n*100:.0f}%  avgEV={avg_ev:+.1f}¢")

    print(f"\n{'='*78}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backtest daytime KXLOWT NO entry with confirmed running min."
    )
    parser.add_argument("--years",  type=int, default=3)
    parser.add_argument("--months", type=int, nargs="+", default=None)
    parser.add_argument("--cities", nargs="+", default=None)
    args = parser.parse_args()

    asyncio.run(main(
        years=args.years,
        months=args.months,
        city_filter=set(f"temp_low_{c.lower()}" for c in args.cities) if args.cities else None,
    ))
