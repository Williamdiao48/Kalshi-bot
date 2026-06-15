#!/usr/bin/env python3
"""Backtest the METAR trajectory gate on full historical KXLOWT dataset.

Uses 690 settled KXLOWT markets from candlesticks.db (Apr–May 2026) and
IEM hourly temperature to answer: does a flat/warming METAR running minimum
at simulated entry time predict losses?

For each market we simulate band_arb entry as the first candle where:
  - YES bid is in the band_arb range (5–35¢, meaning NO is cheap 65–95¢)
  - Running METAR min is above band ceiling (temp looks safe)
  - It is past the daily observation window (hour ≥ 10 UTC)

At that simulated entry we compute the METAR trajectory gate:
  DROP_2H = running_min(entry) - running_min(entry - 2h)
  Gate fires when DROP_2H < -THRESH (running min has been falling → cooling)
  Gate blocks when DROP_2H ≥ -THRESH (flat or warming → risky)

Then compares gated vs ungated win rates across a threshold sweep.

Usage:
  venv/bin/python scripts/backtest_metar_trajectory_gate.py
"""

from __future__ import annotations

import re
import sqlite3
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

CANDLES_DB = Path("data/candlesticks.db")
CACHE_DIR  = Path("data/cache")

_BAND_RE   = re.compile(r"-B([\d.]+)$")
_SERIES_RE = re.compile(r"^(KXLOWT[A-Z]+)-")

SERIES_TO_IEM: dict[str, str] = {
    "KXLOWTATL": "ATL", "KXLOWTAUS": "AUS", "KXLOWTBOS": "BOS",
    "KXLOWTCHI": "MDW", "KXLOWTDC":  "DCA", "KXLOWTDEN": "DEN",
    "KXLOWTHOU": "HOU", "KXLOWTLV":  "LAS", "KXLOWTLAX": "LAX",
    "KXLOWTMIA": "MIA", "KXLOWTMIN": "MSP", "KXLOWTNOLA":"MSY",
    "KXLOWTNYC": "NYC", "KXLOWTOKC": "OKC", "KXLOWTPHIL":"PHL",
    "KXLOWTPHX": "PHX", "KXLOWTSATX":"SAT", "KXLOWTSEA": "SEA",
    "KXLOWTSFO": "SFO",
}

import json

def load_iem_cache(station: str) -> dict[str, dict]:
    """Load IEM hourly data from cache file (built by backtest_band_approach_yes.py)."""
    # Try cached files
    for f in sorted(CACHE_DIR.glob(f"iem_hourly_{station}_*.json"), reverse=True):
        with open(f) as fh:
            return json.load(fh)
    return {}


def band_ceiling(ticker: str) -> float | None:
    m = _BAND_RE.search(ticker)
    return float(m.group(1)) + 1.0 if m else None


def series_name(ticker: str) -> str | None:
    m = _SERIES_RE.match(ticker)
    return m.group(1) if m else None


def compute_running_min_series(
    iem_hourly: dict[str, dict],
    open_ts: int,
    close_ts: int,
) -> dict[str, float | None]:
    """Return {hour_key: running_min_f} across the market window."""
    result: dict[str, float | None] = {}
    running_min: float | None = None
    cur = datetime.fromtimestamp(open_ts, tz=timezone.utc).replace(minute=0, second=0, microsecond=0)
    end = datetime.fromtimestamp(close_ts, tz=timezone.utc)
    while cur <= end:
        key = cur.strftime("%Y-%m-%dT%H:00")
        obs = iem_hourly.get(key)
        if obs:
            t = obs["tmpf"]
            running_min = t if running_min is None else min(running_min, t)
        result[key] = running_min
        cur += timedelta(hours=1)
    return result


def simulate(
    ticker: str,
    open_ts: int,
    close_ts: int,
    result: str,
    candles: list[tuple],
    running_mins: dict[str, float | None],
    iem_hourly: dict[str, dict],   # raw IEM data for slope across full date range
    entry_bid_max: int = 35,
    entry_hour_min: int = 10,
) -> dict | None:
    """Find simulated band_arb entry and compute METAR temperature trajectory."""
    ceiling = band_ceiling(ticker)
    if ceiling is None:
        return None

    for period_ts, bid_close, ask_close in candles:
        if bid_close is None or ask_close is None:
            continue

        dt = datetime.fromtimestamp(period_ts, tz=timezone.utc)
        if dt.hour < entry_hour_min:
            continue

        if bid_close > entry_bid_max or bid_close < 3:
            continue

        hour_key = dt.strftime("%Y-%m-%dT%H:00")
        rmin_now = running_mins.get(hour_key)
        if rmin_now is None:
            continue

        clearance = rmin_now - ceiling
        if clearance < 2.0:
            continue

        # Use raw IEM actual temperature for slope — not running_min.
        # running_mins only covers market window; raw IEM covers the full date range.
        # drop_Nh > 0 means temperature FELL by that amount (cooled).
        raw_now = (iem_hourly.get(hour_key) or {}).get("tmpf")
        drops: dict[int, float | None] = {}
        for hrs in [2, 4, 6, 8]:
            h_ago    = (dt - timedelta(hours=hrs)).strftime("%Y-%m-%dT%H:00")
            raw_then = (iem_hourly.get(h_ago) or {}).get("tmpf")
            drops[hrs] = round(raw_then - raw_now, 2) if (raw_now is not None and raw_then is not None) else None

        return {
            "ticker":         ticker,
            "result":         result,
            "won":            result == "no",
            "ceiling":        ceiling,
            "clearance":      round(clearance, 2),
            "rmin":           round(rmin_now, 2),
            "temp_now":       raw_now,
            "drop_2h":        drops[2],
            "drop_4h":        drops[4],
            "drop_6h":        drops[6],
            "drop_8h":        drops[8],
            "yes_bid":        bid_close,
            "hours_to_close": round((close_ts - period_ts) / 3600, 1),
            "entry_hour_utc": dt.hour,
        }
    return None


def main() -> None:
    if not CANDLES_DB.exists():
        sys.exit(f"ERROR: {CANDLES_DB} not found")
    if not CACHE_DIR.exists():
        sys.exit("ERROR: data/cache/ not found — run backtest_band_approach_yes.py first")

    con = sqlite3.connect(CANDLES_DB)
    markets = con.execute("""
        SELECT ticker, open_ts, close_ts, result
        FROM markets
        WHERE ticker LIKE 'KXLOWT%B%'
          AND result IN ('yes','no')
    """).fetchall()
    print(f"{len(markets)} settled KXLOWT B-band markets")

    # Load IEM caches
    iem_by_series: dict[str, dict] = {}
    for ser, station in SERIES_TO_IEM.items():
        data = load_iem_cache(station)
        if data:
            iem_by_series[ser] = data
    print(f"{len(iem_by_series)} series with IEM temperature data")

    # Load candles
    candles_map: dict[str, list] = {}
    for ticker, *_ in markets:
        rows = con.execute(
            "SELECT period_ts, bid_close, ask_close FROM candles WHERE ticker=? ORDER BY period_ts",
            (ticker,),
        ).fetchall()
        if rows:
            candles_map[ticker] = rows

    # Simulate entries
    all_trades: list[dict] = []
    no_iem = 0
    for ticker, open_ts, close_ts, result in markets:
        ser = series_name(ticker)
        if not ser or ser not in iem_by_series:
            no_iem += 1
            continue
        candles = candles_map.get(ticker, [])
        if not candles:
            continue
        running_mins = compute_running_min_series(iem_by_series[ser], open_ts, close_ts)
        t = simulate(ticker, open_ts, close_ts, result, candles, running_mins,
                     iem_by_series[ser])
        if t:
            all_trades.append(t)

    n_total = len(all_trades)
    n_with_drop = sum(1 for t in all_trades if t["drop_2h"] is not None)
    baseline_wr = sum(1 for t in all_trades if t["won"]) / n_total if n_total else 0
    print(f"\n{n_total} simulated entries  ({no_iem} skipped — no IEM data)")
    print(f"{n_with_drop} have 2-hour METAR trajectory data")
    print(f"Baseline win rate: {baseline_wr:.1%}\n")

    # Sweep: gate = block entry when drop_2h >= -THRESH (not cooling enough)
    import math
    print(f"{'gate_thresh':>12} {'n_kept':>8} {'n_blocked':>10} {'wr_kept':>9} "
          f"{'wr_blocked':>11} {'delta':>7} {'p_val':>7}")
    print("-" * 75)

    for window, field in [(2,"drop_2h"),(4,"drop_4h"),(6,"drop_6h"),(8,"drop_8h")]:
        valid = [t for t in all_trades if t[field] is not None]
        if len(valid) < 10:
            continue
        print(f"\n--- {window}-hour drop window ({len(valid)} trades with data) ---")
        print(f"{'gate_thresh':>12} {'n_kept':>8} {'n_blocked':>10} {'wr_kept':>9} "
              f"{'wr_blocked':>11} {'delta':>7} {'p_val':>7}")
        for thresh in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]:
            kept    = [t for t in valid if t[field] >= thresh]
            blocked = [t for t in valid if t[field] <  thresh]
            if len(kept) < 5 or len(blocked) < 5:
                continue
            wr_k = sum(t["won"] for t in kept)    / len(kept)
            wr_b = sum(t["won"] for t in blocked) / len(blocked)
            p_pool = (sum(t["won"] for t in kept) + sum(t["won"] for t in blocked)) / (len(kept)+len(blocked))
            se = math.sqrt(p_pool*(1-p_pool)*(1/len(kept)+1/len(blocked)))
            z  = (wr_k - wr_b) / se if se > 0 else 0
            p_val = 2*(1 - 0.5*(1 + math.erf(abs(z)/math.sqrt(2))))
            sig = "**" if p_val < 0.05 else ("*" if p_val < 0.15 else "")
            print(f"  drop≥{thresh:+.1f}°F  {len(kept):>8} {len(blocked):>10} "
                  f"{wr_k:>8.1%} {wr_b:>10.1%} {wr_k-wr_b:>+7.1%} {p_val:>6.3f} {sig}")

    # Bucket breakdown for 6h window (most likely to show signal)
    for window, field in [(4,"drop_4h"),(6,"drop_6h"),(8,"drop_8h")]:
        valid = [t for t in all_trades if t[field] is not None]
        if len(valid) < 10:
            continue
        print(f"\n=== Win rate by {window}h METAR drop bucket ({len(valid)} trades) ===")
        print(f"{'bucket':30} {'n':>5} {'wins':>6} {'losses':>7} {'win%':>7}")
        print("-" * 58)
        buckets = [
            (f"temp ROSE >2°F in {window}h",    lambda d: d > 2.0),
            (f"temp rose 0-2°F in {window}h",   lambda d: 0.0 < d <= 2.0),
            (f"flat (±0°F)",                     lambda d: d == 0.0),
            (f"fell 0-2°F in {window}h",         lambda d: -2.0 <= d < 0.0),
            (f"fell 2-5°F in {window}h",         lambda d: -5.0 <= d < -2.0),
            (f"fell >5°F in {window}h",          lambda d: d < -5.0),
        ]
        for label, fn in buckets:
            grp = [t for t in valid if fn(t[field])]
            if not grp:
                continue
            w = sum(t["won"] for t in grp)
            print(f"  {label:30} {len(grp):>5} {w:>6} {len(grp)-w:>7} {w/len(grp):>6.1%}")

    # Overall win rate by result type
    print(f"\n=== Context: what drove the YES settlements? ===")
    yes_trades = [t for t in all_trades if not t["won"]]
    no_trades  = [t for t in all_trades if t["won"]]
    if yes_trades:
        avg_clr_yes = sum(t["clearance"] for t in yes_trades) / len(yes_trades)
        avg_drop_yes = sum(t["drop_2h"] for t in yes_trades if t["drop_2h"] is not None)
        n_drop_yes   = sum(1 for t in yes_trades if t["drop_2h"] is not None)
        print(f"  YES settlers (losses): n={len(yes_trades)}, "
              f"avg clearance={avg_clr_yes:+.2f}°F, "
              f"avg drop_2h={avg_drop_yes/n_drop_yes:+.2f}°F" if n_drop_yes else "")
    if no_trades:
        avg_clr_no  = sum(t["clearance"] for t in no_trades) / len(no_trades)
        avg_drop_no = sum(t["drop_2h"] for t in no_trades if t["drop_2h"] is not None)
        n_drop_no   = sum(1 for t in no_trades if t["drop_2h"] is not None)
        print(f"  NO settlers (wins):    n={len(no_trades)}, "
              f"avg clearance={avg_clr_no:+.2f}°F, "
              f"avg drop_2h={avg_drop_no/n_drop_no:+.2f}°F" if n_drop_no else "")


if __name__ == "__main__":
    main()
