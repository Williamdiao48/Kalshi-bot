#!/usr/bin/env python3
"""KXLOWT between YES backtest v2 — bid_low/bid_high simulation + exit parameter grid search.

Primary data: data/candlesticks.db (316 settled KXLOWT between markets, Apr 6-22 2026).

v2 improvements over v1:
  - Loads bid_low / bid_high from candles for accurate intra-candle stop/take simulation
  - Aggregates minute candles to hourly (min bid_low, max bid_high per hour)
  - Full grid search over stop_loss × profit_take × trailing_drawdown × min_entry_ask
  - Trailing stop activates only after position moves into profit (matches bot behaviour)
  - Recommendation block compares current bot settings to optimal found

Usage:
    venv/bin/python scripts/backtest_kxlowt_entry_timing.py
    venv/bin/python scripts/backtest_kxlowt_entry_timing.py --section grid
    venv/bin/python scripts/backtest_kxlowt_entry_timing.py --section grid --top 15
    venv/bin/python scripts/backtest_kxlowt_entry_timing.py --city chi --verbose
"""

import argparse
import csv
import re
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).parent.parent
CANDLES_DB = ROOT / "data" / "candlesticks.db"

# ---------------------------------------------------------------------------
# City timezone map
# ---------------------------------------------------------------------------
CITY_TZ: dict[str, ZoneInfo] = {
    "atl": ZoneInfo("America/New_York"),
    "bos": ZoneInfo("America/New_York"),
    "nyc": ZoneInfo("America/New_York"),
    "dc":  ZoneInfo("America/New_York"),
    "mia": ZoneInfo("America/New_York"),
    "ric": ZoneInfo("America/New_York"),
    "nor": ZoneInfo("America/New_York"),
    "buf": ZoneInfo("America/New_York"),
    "cle": ZoneInfo("America/New_York"),
    "pit": ZoneInfo("America/New_York"),
    "ind": ZoneInfo("America/Indiana/Indianapolis"),
    "cin": ZoneInfo("America/New_York"),
    "col": ZoneInfo("America/New_York"),
    "jax": ZoneInfo("America/New_York"),
    "tpa": ZoneInfo("America/New_York"),
    "orl": ZoneInfo("America/New_York"),
    "rdu": ZoneInfo("America/New_York"),
    "clt": ZoneInfo("America/New_York"),
    "chi": ZoneInfo("America/Chicago"),
    "dal": ZoneInfo("America/Chicago"),
    "hou": ZoneInfo("America/Chicago"),
    "min": ZoneInfo("America/Chicago"),
    "msp": ZoneInfo("America/Chicago"),
    "kc":  ZoneInfo("America/Chicago"),
    "stl": ZoneInfo("America/Chicago"),
    "nola": ZoneInfo("America/Chicago"),
    "ok":  ZoneInfo("America/Chicago"),
    "sa":  ZoneInfo("America/Chicago"),
    "aus": ZoneInfo("America/Chicago"),
    "mem": ZoneInfo("America/Chicago"),
    "mil": ZoneInfo("America/Chicago"),
    "omh": ZoneInfo("America/Chicago"),
    "nas": ZoneInfo("America/Chicago"),
    "bir": ZoneInfo("America/Chicago"),
    "phx": ZoneInfo("America/Phoenix"),
    "den": ZoneInfo("America/Denver"),
    "slc": ZoneInfo("America/Denver"),
    "bio": ZoneInfo("America/Denver"),
    "abq": ZoneInfo("America/Denver"),
    "lax": ZoneInfo("America/Los_Angeles"),
    "sfo": ZoneInfo("America/Los_Angeles"),
    "sea": ZoneInfo("America/Los_Angeles"),
    "pdx": ZoneInfo("America/Los_Angeles"),
    "las": ZoneInfo("America/Los_Angeles"),
    "lv":  ZoneInfo("America/Los_Angeles"),
    "det": ZoneInfo("America/Detroit"),
}

DAWN_HOUR = 6

# ---------------------------------------------------------------------------
# Grid search parameters
# ---------------------------------------------------------------------------
STOP_LOSS_GRID   = [0.15, 0.20, 0.30, 0.40, 0.50, 0.60]
# Absolute YES-bid price at which to take profit; None = hold to settlement
PROFIT_TAKE_GRID = [None, 65, 70, 75, 80, 85, 90]
# Trailing drawdown from peak bid (only activates once in profit); None = disabled
TRAILING_GRID    = [None, 0.10, 0.15, 0.20, 0.25]
# Minimum entry ask (cents) — proxy for "market-confirmed" temperature position
MIN_ASK_GRID     = [0, 20, 25, 30]

ENTRY_WINDOWS: dict[str, frozenset[int]] = {
    "evening  16-21h": frozenset(range(16, 22)),
    "overnight 22-5h": frozenset(list(range(22, 24)) + list(range(0, 6))),
    "morning   6-11h": frozenset(range(6, 12)),
    "all hours":       frozenset(range(24)),
}

# ---------------------------------------------------------------------------
# Signal-filter constants (noaa_observed between YES)
# ---------------------------------------------------------------------------

NWS_BUFFER = 0.5        # °F Kalshi adds to each side of the nominal band
MIN_SIGNAL_EDGE = 0.2   # °F minimum clearance from effective upper boundary

MESONET_LOW_CSV = ROOT / "data" / "mesonet_low_hourly.csv"

# Ticker city-slug → noaa metric (handles slug mismatches vs noaa.py naming)
SLUG_TO_METRIC: dict[str, str] = {
    "chi":  "temp_low_chi",
    "nyc":  "temp_low_ny",
    "bos":  "temp_low_bos",
    "atl":  "temp_low_atl",
    "min":  "temp_low_msp",   # backtest slug "min" → noaa metric "msp"
    "msp":  "temp_low_msp",
    "lax":  "temp_low_lax",
    "den":  "temp_low_den",
    "mia":  "temp_low_mia",
    "aus":  "temp_low_aus",
    "bos":  "temp_low_bos",
    "hou":  "temp_low_hou",
    "dfw":  "temp_low_dfw",
    "sfo":  "temp_low_sfo",
    "sea":  "temp_low_sea",
    "phx":  "temp_low_phx",
    "phl":  "temp_low_phl",
    "dc":   "temp_low_dca",
    "las":  "temp_low_las",
    "okc":  "temp_low_okc",
    "sat":  "temp_low_sat",
    "nola": "temp_low_msy",
}

_MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,  "MAY": 5,  "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def parse_ticker_date(ticker: str) -> str | None:
    """KXLOWTCHI-26APR09-B45.5 → '2026-04-09'"""
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})-", ticker)
    if not m:
        return None
    yy, mon, dd = m.group(1), m.group(2), int(m.group(3))
    month = _MONTH_MAP.get(mon)
    if month is None:
        return None
    return f"20{yy}-{month:02d}-{dd:02d}"


def parse_ticker_band(ticker: str) -> tuple[float, float] | None:
    """KXLOWTCHI-26APR09-B45.5 → nominal (lo=45.0, hi=46.0)"""
    m = re.search(r"-B([\d.]+)$", ticker)
    if not m:
        return None
    center = float(m.group(1))
    return center - 0.5, center + 0.5


def load_running_min_lookup(path: Path) -> dict[tuple, float]:
    """Load mesonet_low_hourly.csv → {(metric, date_str, hour_int): running_min_f}."""
    lookup: dict[tuple, float] = {}
    if not path.exists():
        return lookup
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (row["city_metric"], row["date"], int(row["local_hour"]))
            lookup[key] = float(row["running_min_f"])
    return lookup


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def city_from_ticker(ticker: str) -> str:
    return ticker[len("KXLOWT"):].split("-")[0].lower()


def ts_to_local(ts: int, city: str) -> datetime | None:
    tz = CITY_TZ.get(city)
    if tz is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(tz)


def load_markets(con: sqlite3.Connection, city_filter: str | None) -> list[dict]:
    rows = con.execute(
        "SELECT ticker, result, open_ts, close_ts FROM markets "
        "WHERE ticker LIKE 'KXLOWT%B%' AND result IS NOT NULL"
    ).fetchall()
    out = []
    for ticker, result, open_ts, close_ts in rows:
        city = city_from_ticker(ticker)
        if city_filter and city != city_filter.lower():
            continue
        out.append({
            "ticker": ticker, "city": city, "result": result,
            "tz": CITY_TZ.get(city), "open_ts": open_ts, "close_ts": close_ts,
        })
    return out


def load_all_candles_raw(
    con: sqlite3.Connection, tickers: list[str]
) -> dict[str, list[tuple]]:
    """Bulk-load minute candles: ticker → [(ts, bid_close, bid_low, bid_high, ask_close)]."""
    if not tickers:
        return {}
    ph = ",".join("?" * len(tickers))
    rows = con.execute(
        f"SELECT ticker, period_ts, bid_close, bid_low, bid_high, ask_close "
        f"FROM candles WHERE ticker IN ({ph}) ORDER BY ticker, period_ts",
        tickers,
    ).fetchall()
    result: dict[str, list[tuple]] = defaultdict(list)
    for ticker, ts, bid_close, bid_low, bid_high, ask_close in rows:
        result[ticker].append((ts, bid_close, bid_low, bid_high, ask_close))
    return dict(result)


def aggregate_to_hourly(
    raw_candles: list[tuple],  # (ts, bid_close, bid_low, bid_high, ask_close)
    city: str,
) -> list[tuple]:
    """
    Aggregate minute candles to hourly candles.
    Returns sorted list of (ts_first, local_hour, bid_close, bid_low_min, bid_high_max, ask_close_last).
    bid_low_min = min(bid_low) for the hour  → worst price a stop-loss would see
    bid_high_max = max(bid_high) for the hour → best price a profit-take would see
    """
    by_hour: dict[tuple, list[tuple]] = defaultdict(list)
    for ts, bid_close, bid_low, bid_high, ask_close in raw_candles:
        dt = ts_to_local(ts, city)
        if dt is None:
            continue
        key = (dt.year, dt.month, dt.day, dt.hour)
        by_hour[key].append((ts, bid_close, bid_low, bid_high, ask_close))

    hourly = []
    for key in sorted(by_hour.keys()):
        candles = by_hour[key]
        ts_first = candles[0][0]
        local_hour = key[3]
        bid_close = next((c[1] for c in reversed(candles) if c[1] is not None), None)
        bid_low   = min((c[2] for c in candles if c[2] is not None), default=None)
        bid_high  = max((c[3] for c in candles if c[3] is not None), default=None)
        ask_close = next((c[4] for c in reversed(candles) if c[4] is not None), None)
        hourly.append((ts_first, local_hour, bid_close, bid_low, bid_high, ask_close))
    return hourly


# ---------------------------------------------------------------------------
# Trade simulation
# ---------------------------------------------------------------------------

def simulate_trade(
    hourly_after: list[tuple],  # (ts, hour, bid_close, bid_low, bid_high, ask_close)
    entry_ask: float,
    stop_loss: float,           # fraction below entry_ask to trigger hard stop
    profit_take_abs: int | None,# absolute bid cents to take profit; None = hold
    trailing_pct: float | None, # drawdown from peak to trigger trail; None = disabled
    result: str,                # "yes" or "no"
) -> float:
    """
    Simulate a YES position using hourly candles after entry.

    Stop ordering (conservative — adverse move checked first within each candle):
      1. Hard stop-loss:  bid_low <= entry_ask × (1 - stop_loss)
      2. Trailing stop:   bid_low <= peak_bid × (1 - trailing_pct)
                          [only activates once peak_bid > entry_ask]
      3. Profit take:     bid_high >= profit_take_abs

    Returns P&L in cents per contract.
    """
    stop_price = entry_ask * (1 - stop_loss)
    peak_bid   = entry_ask  # track highest bid seen since entry

    for _, hour, bid_close, bid_low, bid_high, ask_close in hourly_after:
        # Update peak from best intra-hour price
        if bid_high is not None and bid_high > peak_bid:
            peak_bid = bid_high

        # 1. Hard stop-loss (worst intra-hour price)
        if bid_low is not None and bid_low <= stop_price:
            return stop_price - entry_ask

        # 2. Trailing stop (only once position is in profit)
        if trailing_pct is not None and peak_bid > entry_ask:
            trail_thresh = peak_bid * (1 - trailing_pct)
            if bid_low is not None and bid_low <= trail_thresh:
                return trail_thresh - entry_ask

        # 3. Profit take (best intra-hour price)
        if profit_take_abs is not None and bid_high is not None and bid_high >= profit_take_abs:
            return float(profit_take_abs) - entry_ask

    # Held to settlement
    return (100.0 if result == "yes" else 0.0) - entry_ask


def _entry_ask(bid_close: int | None, ask_close: int | None) -> float | None:
    """Resolve entry price: prefer ask_close, fall back to bid_close + 3¢."""
    if ask_close is not None and 0 < ask_close < 100:
        return float(ask_close)
    if bid_close is not None and 0 < bid_close < 97:
        return float(bid_close) + 3.0
    return None


# ---------------------------------------------------------------------------
# Analysis 1: EV by local entry hour
# ---------------------------------------------------------------------------

def analyse_ev_by_hour(
    markets: list[dict], all_raw: dict[str, list[tuple]], verbose: bool
) -> None:
    print("\n" + "=" * 72)
    print("EV BY LOCAL ENTRY HOUR  (buy YES, hold to settlement)")
    print("  EV = result×(100−ask) − (1−result)×ask")
    print("  entry_ask = ask_close if available, else bid_close + 3¢")
    print("=" * 72)

    ev_by_hour:  dict[int, list[float]] = defaultdict(list)
    ask_by_hour: dict[int, list[float]] = defaultdict(list)

    for mkt in markets:
        hourly = aggregate_to_hourly(all_raw.get(mkt["ticker"], []), mkt["city"])
        result_val = 1.0 if mkt["result"] == "yes" else 0.0
        seen: set[int] = set()
        for _, hour, bid_close, bid_low, bid_high, ask_close in hourly:
            if hour in seen:
                continue
            seen.add(hour)
            ask = _entry_ask(bid_close, ask_close)
            if ask is None:
                continue
            ev = result_val * (100 - ask) - (1 - result_val) * ask
            ev_by_hour[hour].append(ev)
            ask_by_hour[hour].append(ask)

    if not ev_by_hour:
        print("  No data.")
        return

    print(f"\n  {'Hr':>4}  {'N':>6}  {'AvgEV':>8}  {'AvgAsk':>8}  "
          f"{'%EV>0':>7}  {'MinEV':>8}  {'MaxEV':>8}")
    print("  " + "-" * 64)
    for hour in sorted(ev_by_hour):
        evs  = ev_by_hour[hour]
        asks = ask_by_hour[hour]
        pct_pos = sum(1 for e in evs if e > 0) / len(evs) * 100
        dm = "🌅" if hour >= DAWN_HOUR else "🌙"
        print(f"  {hour:02d} {dm}  {len(evs):>6}  {sum(evs)/len(evs):>+7.1f}¢  "
              f"{sum(asks)/len(asks):>7.0f}¢  {pct_pos:>6.0f}%  "
              f"{min(evs):>+7.1f}¢  {max(evs):>+7.1f}¢")


# ---------------------------------------------------------------------------
# Analysis 2: Price trajectory YES vs NO markets
# ---------------------------------------------------------------------------

def analyse_price_trajectory(
    markets: list[dict], all_raw: dict[str, list[tuple]], verbose: bool
) -> None:
    print("\n" + "=" * 72)
    print("YES BID TRAJECTORY BY LOCAL HOUR  (YES-settling vs NO-settling)")
    print("=" * 72)

    yes_bids: dict[int, list[int]] = defaultdict(list)
    no_bids:  dict[int, list[int]] = defaultdict(list)

    for mkt in markets:
        hourly = aggregate_to_hourly(all_raw.get(mkt["ticker"], []), mkt["city"])
        bucket = yes_bids if mkt["result"] == "yes" else no_bids
        seen: set[int] = set()
        for _, hour, bid_close, bid_low, bid_high, ask_close in hourly:
            if hour in seen:
                continue
            seen.add(hour)
            if bid_close is not None and 0 < bid_close < 100:
                bucket[hour].append(bid_close)

    all_hours = sorted(set(yes_bids) | set(no_bids))
    print(f"\n  {'Hr':>4}  {'YES_avg':>8}  {'YES_n':>6}  "
          f"{'NO_avg':>8}  {'NO_n':>6}  {'Spread':>8}")
    print("  " + "-" * 56)
    for hour in all_hours:
        y = yes_bids.get(hour, [])
        n = no_bids.get(hour, [])
        y_avg = sum(y) / len(y) if y else None
        n_avg = sum(n) / len(n) if n else None
        spread = f"{y_avg - n_avg:+.0f}¢" if y_avg is not None and n_avg is not None else "—"
        y_str = f"{y_avg:>6.0f}¢" if y_avg is not None else "      —"
        n_str = f"{n_avg:>6.0f}¢" if n_avg is not None else "      —"
        dm = "🌅" if hour >= DAWN_HOUR else "🌙"
        print(f"  {hour:02d} {dm}  {y_str}  {len(y):>6}  {n_str}  {len(n):>6}  {spread:>8}")


# ---------------------------------------------------------------------------
# Analysis 3: Overnight min price dip (YES-settling markets)
# ---------------------------------------------------------------------------

def analyse_overnight_dip(
    markets: list[dict], all_raw: dict[str, list[tuple]], verbose: bool
) -> None:
    print("\n" + "=" * 72)
    print("OVERNIGHT MIN PRICE  (YES-settling markets, 20h–06h local)")
    print("  Uses bid_low (worst intra-candle price) for accurate stop simulation")
    print("=" * 72)

    results = []
    for mkt in (m for m in markets if m["result"] == "yes"):
        hourly = aggregate_to_hourly(all_raw.get(mkt["ticker"], []), mkt["city"])
        entry_bid = None
        overnight_lows: list[int] = []

        for _, hour, bid_close, bid_low, bid_high, ask_close in hourly:
            if 16 <= hour <= 20 and entry_bid is None and bid_close:
                entry_bid = bid_close
            if hour >= 20 or hour < 6:
                low = bid_low if bid_low is not None else bid_close
                if low is not None:
                    overnight_lows.append(low)

        if entry_bid is None or not overnight_lows:
            continue

        min_night = min(overnight_lows)
        drop_pct = (entry_bid - min_night) / entry_bid * 100
        results.append({
            "ticker": mkt["ticker"],
            "entry_bid": entry_bid,
            "min_night": min_night,
            "drop_pct": drop_pct,
            "stop_20": min_night < entry_bid * 0.80,
            "stop_50": min_night < entry_bid * 0.50,
        })

    if not results:
        print("  No YES-settling markets with entry + overnight data.")
        return

    n   = len(results)
    s20 = sum(1 for r in results if r["stop_20"])
    s50 = sum(1 for r in results if r["stop_50"])
    print(f"\n  {n} YES-settling markets with 4–8 PM entry + overnight candles")
    print(f"  20% stop (bid_low) fires: {s20}/{n} ({s20/n*100:.0f}%)  — kills winning trades")
    print(f"  50% stop (bid_low) fires: {s50}/{n} ({s50/n*100:.0f}%)  — kills winning trades")
    print(f"  Avg entry bid (4–8 PM):   {sum(r['entry_bid'] for r in results)/n:.0f}¢")
    print(f"  Avg overnight min (low):  {sum(r['min_night'] for r in results)/n:.0f}¢")
    print(f"  Avg dip from entry:       {sum(r['drop_pct'] for r in results)/n:.0f}%")

    if verbose:
        print(f"\n  {'Ticker':<35}  {'Entry':>6}  {'MinLow':>7}  "
              f"{'Drop%':>7}  {'20%SL':>6}  {'50%SL':>6}")
        print("  " + "-" * 72)
        for r in sorted(results, key=lambda x: -x["drop_pct"]):
            print(f"  {r['ticker']:<35}  {r['entry_bid']:>5}¢  {r['min_night']:>6}¢  "
                  f"{r['drop_pct']:>6.0f}%  "
                  f"{'fire':>6}  " if r["stop_20"] else f"{'ok':>6}  ",
                  f"{'fire':>6}" if r["stop_50"] else f"{'ok':>6}")


# ---------------------------------------------------------------------------
# Analysis 4: Stop-loss survival by entry hour
# ---------------------------------------------------------------------------

def analyse_stop_loss_survival(
    markets: list[dict], all_raw: dict[str, list[tuple]], verbose: bool
) -> None:
    print("\n" + "=" * 72)
    print("STOP-LOSS SURVIVAL BY ENTRY HOUR  (YES-settling markets only)")
    print("  Uses bid_low for stop triggers — actual worst intra-candle price")
    print("=" * 72)

    STOPS = [0.20, 0.40, 0.50, 0.60]
    # survival[hour][sl] = [survived, total]
    survival: dict[int, dict[float, list[int]]] = defaultdict(
        lambda: {s: [0, 0] for s in STOPS}
    )

    for mkt in (m for m in markets if m["result"] == "yes"):
        hourly = aggregate_to_hourly(all_raw.get(mkt["ticker"], []), mkt["city"])
        if not hourly:
            continue
        # First candle index per local hour
        first_idx: dict[int, int] = {}
        for i, row in enumerate(hourly):
            hour = row[1]
            if hour not in first_idx:
                first_idx[hour] = i

        for entry_hour, idx in first_idx.items():
            entry_bid = hourly[idx][2]  # bid_close of first candle in that hour
            if entry_bid is None or not (0 < entry_bid < 99):
                continue
            for sl in STOPS:
                stop_price = entry_bid * (1 - sl)
                fired = any(
                    row[3] is not None and row[3] <= stop_price
                    for row in hourly[idx + 1:]
                )
                survival[entry_hour][sl][1] += 1
                if not fired:
                    survival[entry_hour][sl][0] += 1

    if not survival:
        print("  No YES-settling markets found.")
        return

    hdr = "  ".join(f"Surv_{int(s*100)}%" for s in STOPS)
    print(f"\n  {'Hr':>4}  {'N':>5}  {hdr}")
    print("  " + "-" * (12 + 12 * len(STOPS)))
    for hour in sorted(survival):
        s = survival[hour]
        total = s[STOPS[0]][1]
        if total == 0:
            continue
        surv_strs = "  ".join(f"{s[sl][0]/total*100:>7.0f}%" for sl in STOPS)
        dm = "🌅" if hour >= DAWN_HOUR else "🌙"
        print(f"  {hour:02d} {dm}  {total:>5}  {surv_strs}")


# ---------------------------------------------------------------------------
# Analysis 5: Settlement rate by entry price band
# ---------------------------------------------------------------------------

def analyse_settlement_by_price(
    markets: list[dict], all_raw: dict[str, list[tuple]], verbose: bool
) -> None:
    print("\n" + "=" * 72)
    print("SETTLEMENT RATE BY ENTRY PRICE BAND  (all hours)")
    print("  When YES is priced at X¢, how often does it settle YES?")
    print("=" * 72)

    bands = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 55), (55, 75), (75, 95)]
    counts: dict[tuple, list[int]] = {b: [0, 0] for b in bands}

    for mkt in markets:
        hourly = aggregate_to_hourly(all_raw.get(mkt["ticker"], []), mkt["city"])
        result_yes = mkt["result"] == "yes"
        seen: set[int] = set()
        for _, hour, bid_close, bid_low, bid_high, ask_close in hourly:
            if hour in seen:
                continue
            seen.add(hour)
            ask = _entry_ask(bid_close, ask_close)
            if ask is None:
                continue
            for lo, hi in bands:
                if lo <= ask < hi:
                    counts[(lo, hi)][1] += 1
                    if result_yes:
                        counts[(lo, hi)][0] += 1
                    break

    print(f"\n  {'Ask range':>12}  {'N':>6}  {'YES':>6}  {'Win%':>7}  {'AvgEV@mid':>10}")
    print("  " + "-" * 52)
    for lo, hi in bands:
        yes_n, total = counts[(lo, hi)]
        if total == 0:
            continue
        win_pct = yes_n / total * 100
        mid_ask = (lo + hi) / 2
        ev = win_pct / 100 * (100 - mid_ask) - (1 - win_pct / 100) * mid_ask
        print(f"  {lo:>4}–{hi:<4}¢    {total:>6}  {yes_n:>6}  {win_pct:>6.0f}%  {ev:>+9.1f}¢")


# ---------------------------------------------------------------------------
# Analysis 6: Exit parameter grid search
# ---------------------------------------------------------------------------

def _build_entry_table(
    markets: list[dict], all_raw: dict[str, list[tuple]]
) -> list[tuple]:
    """
    Returns list of (ticker, city, local_hour, entry_ask, hourly_after, result)
    — one row per (market × first occurrence of each local hour).
    """
    entries = []
    for mkt in markets:
        hourly = aggregate_to_hourly(all_raw.get(mkt["ticker"], []), mkt["city"])
        if not hourly:
            continue
        seen_hours: set[int] = set()
        for i, row in enumerate(hourly):
            _, hour, bid_close, bid_low, bid_high, ask_close = row
            if hour in seen_hours:
                continue
            seen_hours.add(hour)
            ask = _entry_ask(bid_close, ask_close)
            if ask is None:
                continue
            entries.append((
                mkt["ticker"],
                mkt["city"],
                hour,
                ask,
                hourly[i + 1:],   # candles after this entry point
                mkt["result"],
            ))
    return entries


def _run_grid(
    entry_slice: list[tuple],  # (ticker, city, hour, entry_ask, hourly_after, result)
    top_n: int,
) -> list[tuple]:
    """
    Run the full grid search on an entry slice.
    Returns list of (avg_ev, sl, pt, trail, n, win_rate, profit_factor) sorted by avg_ev desc.
    """
    results = []
    for sl, pt, trail in product(STOP_LOSS_GRID, PROFIT_TAKE_GRID, TRAILING_GRID):
        pnls = [
            simulate_trade(ca, ea, sl, pt, trail, r)
            for (_, _, _, ea, ca, r) in entry_slice
        ]
        n = len(pnls)
        if n == 0:
            continue
        wins      = sum(1 for p in pnls if p > 0)
        avg_ev    = sum(pnls) / n
        tot_gain  = sum(p for p in pnls if p > 0)
        tot_loss  = abs(sum(p for p in pnls if p < 0))
        pf        = tot_gain / tot_loss if tot_loss > 0 else float("inf")
        results.append((avg_ev, sl, pt, trail, n, wins / n, pf))

    results.sort(key=lambda x: -x[0])
    return results[:top_n]


def analyse_exit_grid(
    markets: list[dict], all_raw: dict[str, list[tuple]], top_n: int, verbose: bool
) -> None:
    print("\n" + "=" * 72)
    print("EXIT PARAMETER GRID SEARCH")
    print(f"  stop_loss:   {STOP_LOSS_GRID}")
    print(f"  profit_take: {PROFIT_TAKE_GRID} (cents; None = hold to settlement)")
    print(f"  trailing:    {TRAILING_GRID} (drawdown from peak; None = disabled)")
    print(f"  min_ask:     {MIN_ASK_GRID} (min entry price filter in cents)")
    n_combos = len(STOP_LOSS_GRID) * len(PROFIT_TAKE_GRID) * len(TRAILING_GRID)
    print(f"  Combos:      {n_combos} per (window × min_ask)")
    print(f"  Simulation:  hourly aggregated candles, bid_low stops, bid_high takes")
    print("=" * 72)

    print("\n  Building entry table ...", end="", flush=True)
    all_entries = _build_entry_table(markets, all_raw)
    print(f" {len(all_entries):,} (market × entry_hour) pairs")

    # Current bot settings: SL=50%, no PT, trailing=15%
    CURRENT_SL, CURRENT_PT, CURRENT_TRAIL = 0.50, None, 0.15

    for window_name, window_hours in ENTRY_WINDOWS.items():
        window_entries = [e for e in all_entries if e[2] in window_hours]
        if not window_entries:
            continue

        print(f"\n{'='*72}")
        print(f"WINDOW: {window_name.upper()}  ({len(window_entries):,} entry slots total)")
        print("=" * 72)

        for min_ask in MIN_ASK_GRID:
            filtered = [e for e in window_entries if e[3] >= min_ask]
            if len(filtered) < 20:
                continue

            top = _run_grid(filtered, top_n)

            # Current bot performance on this slice
            cur_pnls  = [simulate_trade(ca, ea, CURRENT_SL, CURRENT_PT, CURRENT_TRAIL, r)
                         for (_, _, _, ea, ca, r) in filtered]
            cur_ev    = sum(cur_pnls) / len(cur_pnls)
            cur_wins  = sum(1 for p in cur_pnls if p > 0)
            cur_tg    = sum(p for p in cur_pnls if p > 0)
            cur_tl    = abs(sum(p for p in cur_pnls if p < 0))
            cur_pf    = cur_tg / cur_tl if cur_tl > 0 else float("inf")

            print(f"\n  min_ask >= {min_ask}¢  ({len(filtered):,} entries)")
            print(f"  Current bot (SL=50% PT=hold trail=15%): "
                  f"avg EV {cur_ev:+.1f}¢  win {cur_wins/len(cur_pnls):.0%}  PF {cur_pf:.2f}x")
            print(f"\n  {'Rank':>4}  {'SL':>5}  {'PT':>5}  {'Trail':>6}  "
                  f"{'N':>5}  {'Win%':>6}  {'AvgEV':>8}  {'PF':>6}")
            print("  " + "-" * 58)
            for rank, (avg_ev, sl, pt, trail, n, wr, pf) in enumerate(top, 1):
                pt_str    = f"{pt}¢"   if pt    is not None else " hold"
                trail_str = f"{trail:.0%}" if trail is not None else " none"
                pf_str    = f"{pf:.2f}x" if pf != float("inf") else "   inf"
                delta     = avg_ev - cur_ev
                marker    = " ←best" if rank == 1 else ""
                print(f"  {rank:>4}  {sl:.0%}  {pt_str:>5}  {trail_str:>6}  "
                      f"{n:>5}  {wr:.0%}  {avg_ev:>+7.1f}¢  {pf_str}  "
                      f"({delta:+.1f}¢ vs cur){marker}")

    # Overall recommendation across all hours, no min_ask filter
    print(f"\n{'='*72}")
    print("RECOMMENDATION SUMMARY")
    print("=" * 72)

    for window_name, window_hours in ENTRY_WINDOWS.items():
        window_entries = [e for e in all_entries if e[2] in window_hours]
        if len(window_entries) < 20:
            continue
        top1 = _run_grid(window_entries, 1)
        if not top1:
            continue
        avg_ev, sl, pt, trail, n, wr, pf = top1[0]
        cur_pnls = [simulate_trade(ca, ea, CURRENT_SL, CURRENT_PT, CURRENT_TRAIL, r)
                    for (_, _, _, ea, ca, r) in window_entries]
        cur_ev = sum(cur_pnls) / len(cur_pnls)
        pt_str    = f"{pt}¢ profit-take"   if pt    is not None else "hold to settlement"
        trail_str = f"{trail:.0%} trailing" if trail is not None else "no trailing"
        improvement = avg_ev - cur_ev
        print(f"\n  {window_name}:")
        print(f"    optimal:  SL={sl:.0%}  {pt_str}  {trail_str}")
        print(f"    avg EV:   {avg_ev:+.1f}¢  (current: {cur_ev:+.1f}¢,  Δ {improvement:+.1f}¢)")
        print(f"    win rate: {wr:.0%}  profit factor: {pf:.2f}x")


# ---------------------------------------------------------------------------
# Analysis 7: Signal-filtered exit grid (noaa_observed between YES entries only)
# ---------------------------------------------------------------------------

def analyse_signal_filtered_grid(
    markets: list[dict],
    all_raw: dict[str, list[tuple]],
    running_min_lookup: dict[tuple, float],
    top_n: int,
    min_edge_f: float,
) -> None:
    """
    Grid search restricted to (market × hour) pairs where the bot's
    noaa_observed between YES signal would have fired.

    Signal fires when:
      effective_lo = strike_lo - NWS_BUFFER
      effective_hi = strike_hi + NWS_BUFFER
      effective_lo <= running_min <= effective_hi
      edge = effective_hi - running_min >= min_edge_f

    This filters out the 81% of market-hours where the temperature wasn't
    near the band, giving results that reflect the bot's actual traded population.
    """
    if not running_min_lookup:
        print("\n  [signal grid] No running_min data — run fetch_mesonet_low_history.py first")
        return

    print("\n" + "=" * 72)
    print("SIGNAL-FILTERED EXIT GRID  (noaa_observed between YES entries only)")
    print(f"  Signal: effective_lo ≤ running_min ≤ effective_hi")
    print(f"  effective band = nominal ± {NWS_BUFFER}°F,  min edge = {min_edge_f}°F")
    print(f"  Data: {MESONET_LOW_CSV.name}")
    print("=" * 72)

    CURRENT_SL, CURRENT_PT, CURRENT_TRAIL = 0.50, None, 0.15

    # Build signal-active entries
    signal_entries: list[tuple] = []
    skipped_no_data = skipped_outside = 0

    for mkt in markets:
        ticker  = mkt["ticker"]
        city    = mkt["city"]
        metric  = SLUG_TO_METRIC.get(city)
        band    = parse_ticker_band(ticker)
        settle_date = parse_ticker_date(ticker)

        if metric is None or band is None or settle_date is None:
            continue

        strike_lo, strike_hi = band
        eff_lo = strike_lo - NWS_BUFFER
        eff_hi = strike_hi + NWS_BUFFER

        hourly = aggregate_to_hourly(all_raw.get(ticker, []), city)
        if not hourly:
            continue

        seen_hours: set[int] = set()
        for i, row in enumerate(hourly):
            _, hour, bid_close, bid_low, bid_high, ask_close = row
            if hour in seen_hours:
                continue
            seen_hours.add(hour)

            ask = _entry_ask(bid_close, ask_close)
            if ask is None:
                continue

            # Look up running min at this (metric, settlement_date, hour)
            running_min = running_min_lookup.get((metric, settle_date, hour))
            if running_min is None:
                skipped_no_data += 1
                continue

            # Check signal: running_min must be inside the effective band with sufficient edge
            if not (eff_lo <= running_min <= eff_hi):
                skipped_outside += 1
                continue
            edge = eff_hi - running_min
            if edge < min_edge_f:
                skipped_outside += 1
                continue

            signal_entries.append((ticker, city, hour, ask, hourly[i + 1:], mkt["result"], edge))

    yes_count = sum(1 for e in signal_entries if e[5] == "yes")
    no_count  = len(signal_entries) - yes_count

    print(f"\n  Signal-active entries: {len(signal_entries):,}")
    print(f"  YES-settling: {yes_count} ({yes_count/max(len(signal_entries),1)*100:.0f}%)")
    print(f"  NO-settling:  {no_count} ({no_count/max(len(signal_entries),1)*100:.0f}%)")
    print(f"  Skipped (no temp data): {skipped_no_data:,}")
    print(f"  Skipped (outside band): {skipped_outside:,}")

    if len(signal_entries) < 10:
        print("\n  Too few signal entries to run grid — check mesonet CSV coverage.")
        return

    # Edge distribution
    edges = [e[6] for e in signal_entries]
    print(f"  Edge: min={min(edges):.2f}°F  avg={sum(edges)/len(edges):.2f}°F  max={max(edges):.2f}°F")

    # Entry hour distribution
    hour_counts: dict[int, list[int]] = defaultdict(lambda: [0, 0])  # [yes, total]
    for e in signal_entries:
        hour_counts[e[2]][1] += 1
        if e[5] == "yes":
            hour_counts[e[2]][0] += 1
    print(f"\n  Entry hour breakdown (YES/total):")
    for hr in sorted(hour_counts):
        y, t = hour_counts[hr]
        dm = "🌅" if hr >= DAWN_HOUR else "🌙"
        bar = "█" * t
        print(f"    {hr:02d} {dm}  {y:>3}/{t:<3}  ({y/t*100:.0f}% YES)  {bar}")

    # Strip edge from entries for grid (grid only needs the 6-tuple)
    grid_entries = [(t, c, h, ea, ca, r) for (t, c, h, ea, ca, r, _) in signal_entries]

    # Full grid across all windows + min_ask combos
    for window_name, window_hours in ENTRY_WINDOWS.items():
        w_entries = [e for e in grid_entries if e[2] in window_hours]
        if len(w_entries) < 5:
            continue

        print(f"\n{'='*72}")
        print(f"SIGNAL GRID — {window_name.upper()}  ({len(w_entries):,} entries)")
        print("=" * 72)

        for min_ask in MIN_ASK_GRID:
            filtered = [e for e in w_entries if e[3] >= min_ask]
            if len(filtered) < 5:
                continue

            top = _run_grid(filtered, top_n)
            cur_pnls = [simulate_trade(ca, ea, CURRENT_SL, CURRENT_PT, CURRENT_TRAIL, r)
                        for (_, _, _, ea, ca, r) in filtered]
            cur_ev   = sum(cur_pnls) / len(cur_pnls)
            cur_wins = sum(1 for p in cur_pnls if p > 0)
            cur_tg   = sum(p for p in cur_pnls if p > 0)
            cur_tl   = abs(sum(p for p in cur_pnls if p < 0))
            cur_pf   = cur_tg / cur_tl if cur_tl > 0 else float("inf")

            wr_yes = sum(1 for e in filtered if e[5] == "yes") / len(filtered)

            print(f"\n  min_ask >= {min_ask}¢  ({len(filtered):,} entries,  {wr_yes:.0%} YES base rate)")
            print(f"  Current bot (SL=50% PT=hold trail=15%): "
                  f"avg EV {cur_ev:+.1f}¢  win {cur_wins/len(cur_pnls):.0%}  PF {cur_pf:.2f}x")
            print(f"\n  {'Rank':>4}  {'SL':>5}  {'PT':>5}  {'Trail':>6}  "
                  f"{'N':>5}  {'Win%':>6}  {'AvgEV':>8}  {'PF':>6}")
            print("  " + "-" * 58)
            for rank, (avg_ev, sl, pt, trail, n, wr, pf) in enumerate(top, 1):
                pt_str    = f"{pt}¢"      if pt    is not None else " hold"
                trail_str = f"{trail:.0%}" if trail is not None else " none"
                pf_str    = f"{pf:.2f}x"  if pf != float("inf") else "   inf"
                delta     = avg_ev - cur_ev
                marker    = " ←best" if rank == 1 else ""
                print(f"  {rank:>4}  {sl:.0%}  {pt_str:>5}  {trail_str:>6}  "
                      f"{n:>5}  {wr:.0%}  {avg_ev:>+7.1f}¢  {pf_str}  "
                      f"({delta:+.1f}¢ vs cur){marker}")

    # Summary recommendation
    print(f"\n{'='*72}")
    print("SIGNAL-FILTERED RECOMMENDATION")
    print("=" * 72)
    for window_name, window_hours in ENTRY_WINDOWS.items():
        w = [e for e in grid_entries if e[2] in window_hours]
        if len(w) < 5:
            continue
        top1 = _run_grid(w, 1)
        if not top1:
            continue
        avg_ev, sl, pt, trail, n, wr, pf = top1[0]
        cur_pnls = [simulate_trade(ca, ea, CURRENT_SL, CURRENT_PT, CURRENT_TRAIL, r)
                    for (_, _, _, ea, ca, r) in w]
        cur_ev = sum(cur_pnls) / len(cur_pnls)
        pt_str    = f"{pt}¢ profit-take"    if pt    is not None else "hold to settlement"
        trail_str = f"{trail:.0%} trailing"  if trail is not None else "no trailing"
        print(f"\n  {window_name}  (n={n}):")
        print(f"    optimal:  SL={sl:.0%}  {pt_str}  {trail_str}")
        print(f"    avg EV:   {avg_ev:+.1f}¢  (current: {cur_ev:+.1f}¢,  Δ {avg_ev-cur_ev:+.1f}¢)")
        print(f"    win rate: {wr:.0%}  profit factor: {pf:.2f}x")


# ---------------------------------------------------------------------------
# Summary + main
# ---------------------------------------------------------------------------

def print_summary(markets: list[dict]) -> None:
    n     = len(markets)
    n_yes = sum(1 for m in markets if m["result"] == "yes")
    n_no  = n - n_yes
    cities = sorted(set(m["city"] for m in markets))
    print(f"\n  {n} settled KXLOWT between markets | "
          f"YES: {n_yes} ({n_yes/n*100:.0f}%) | NO: {n_no} ({n_no/n*100:.0f}%)")
    print(f"  Cities: {', '.join(cities)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show per-market detail in applicable sections")
    parser.add_argument("--city", help="Filter to one city (e.g. chi, nyc, atl)")
    parser.add_argument(
        "--section",
        choices=["ev", "trajectory", "stops", "price", "dip", "grid", "signal", "all"],
        default="all",
        help="Which analysis section to run [default: all]",
    )
    parser.add_argument("--top", type=int, default=10,
                        help="Show top N results in grid search [default: 10]")
    parser.add_argument("--min-edge", type=float, default=MIN_SIGNAL_EDGE,
                        help=f"Min °F edge from band boundary for signal filter [default: {MIN_SIGNAL_EDGE}]")
    args = parser.parse_args()

    if not CANDLES_DB.exists():
        print(f"ERROR: {CANDLES_DB} not found.", file=sys.stderr)
        sys.exit(1)

    con     = sqlite3.connect(str(CANDLES_DB))
    markets = load_markets(con, args.city)
    if not markets:
        print("No settled KXLOWT between markets found.")
        sys.exit(0)

    label = f" (city={args.city})" if args.city else ""
    print(f"\n{'='*72}")
    print(f"KXLOWT BETWEEN YES BACKTEST v2{label}  —  data/candlesticks.db")
    print(f"{'='*72}")
    print_summary(markets)

    print("\n  Bulk-loading candle data ...", end="", flush=True)
    tickers = [m["ticker"] for m in markets]
    all_raw = load_all_candles_raw(con, tickers)
    total   = sum(len(v) for v in all_raw.values())
    print(f" {total:,} rows across {len(all_raw)} tickers")

    run = args.section
    if run in ("ev",         "all"): analyse_ev_by_hour(markets, all_raw, args.verbose)
    if run in ("trajectory", "all"): analyse_price_trajectory(markets, all_raw, args.verbose)
    if run in ("dip",        "all"): analyse_overnight_dip(markets, all_raw, args.verbose)
    if run in ("stops",      "all"): analyse_stop_loss_survival(markets, all_raw, args.verbose)
    if run in ("price",      "all"): analyse_settlement_by_price(markets, all_raw, args.verbose)
    if run in ("grid",       "all"): analyse_exit_grid(markets, all_raw, args.top, args.verbose)
    if run in ("signal",     "all"):
        print("\n  Loading running-min lookup ...", end="", flush=True)
        running_min = load_running_min_lookup(MESONET_LOW_CSV)
        print(f" {len(running_min):,} rows")
        analyse_signal_filtered_grid(
            markets, all_raw, running_min, args.top, args.min_edge
        )

    print()
    con.close()


if __name__ == "__main__":
    main()
