"""Hybrid forecast-momentum backtest using real Kalshi candlestick prices.

Two questions answered
----------------------
1. MARKET EFFICIENCY: For markets that settled YES, was the YES bid systematically
   underpriced in the morning vs. midday?  If yes, there is a momentum edge to
   exploit — the question is only whether we can predict YES early enough.

2. FORECAST ACCURACY (Open-Meteo daily high): Does the actual daily high from
   Open-Meteo archive predict YES settlement well enough to use as an entry filter?
   We compare the Open-Meteo daily high (what the model would eventually converge to)
   against the band boundaries to score signal accuracy.

Note: Open-Meteo archive gives ACTUAL daily max, not a forecast issued at 8am.
It approximates the signal quality of a perfect same-day forecast model.  The
gap between this and reality is captured in the "forecast accuracy" section.

Usage
-----
  venv/bin/python scripts/backtest_forecast_momentum_live.py
  venv/bin/python scripts/backtest_forecast_momentum_live.py --db data/candlesticks_test.db
  venv/bin/python scripts/backtest_forecast_momentum_live.py --entry-hours 7 8 9 10 11 --hold-hours 1 2 4
  venv/bin/python scripts/backtest_forecast_momentum_live.py --cities bos atl lax
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import urllib.request
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent.parent))

from kalshi_bot.news.noaa import CITIES

SERIES_TO_METRIC: dict[str, str] = {
    "KXHIGHTATL": "temp_high_atl", "KXHIGHTCHI": "temp_high_chi",
    "KXHIGHTNYC": "temp_high_ny",  "KXHIGHTBOS": "temp_high_bos",
    "KXHIGHTMIN": "temp_high_msp", "KXHIGHTSEA": "temp_high_sea",
    "KXHIGHTSFO": "temp_high_sfo", "KXHIGHTDAL": "temp_high_dfw",
    "KXHIGHTPHX": "temp_high_phx", "KXHIGHTDC":  "temp_high_dca",
    "KXHIGHLAX":  "temp_high_lax", "KXHIGHDEN":  "temp_high_den",
    "KXHIGHCHI":  "temp_high_chi", "KXHIGHNY":   "temp_high_ny",
    "KXHIGHMIA":  "temp_high_mia", "KXHIGHOU":   "temp_high_hou",
    "KXHIGHBOS":  "temp_high_bos", "KXHIGHAUS":  "temp_high_aus",
    "KXHIGHDAL":  "temp_high_dfw",
}

ABBREV_TO_METRIC: dict[str, str] = {
    "atl": "temp_high_atl", "chi": "temp_high_chi", "nyc": "temp_high_ny",
    "bos": "temp_high_bos", "msp": "temp_high_msp", "sea": "temp_high_sea",
    "sfo": "temp_high_sfo", "dfw": "temp_high_dfw", "phx": "temp_high_phx",
    "dc":  "temp_high_dca", "lax": "temp_high_lax", "den": "temp_high_den",
    "mia": "temp_high_mia", "hou": "temp_high_hou", "aus": "temp_high_aus",
    "msy": "temp_high_msy", "phl": "temp_high_phl", "sat": "temp_high_sat",
    "las": "temp_high_las", "okc": "temp_high_okc", "ny":  "temp_high_ny",
    "min": "temp_high_msp",
}

# ── Open-Meteo daily high ─────────────────────────────────────────────────────
_OM_CACHE: dict[tuple, dict[date, float]] = {}  # (lat,lon) → {date → daily_high_F}


def _fetch_om_daily_highs(lat: float, lon: float, start: date, end: date) -> dict[date, float]:
    key = (round(lat, 4), round(lon, 4))
    cached = _OM_CACHE.setdefault(key, {})
    needed = [d for d in (start + timedelta(n) for n in range((end - start).days + 1))
              if d not in cached]
    if not needed:
        return {d: cached[d] for d in (start + timedelta(n) for n in range((end - start).days + 1)) if d in cached}

    fs, fe = min(needed), max(needed)
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={fs}&end_date={fe}"
        f"&daily=temperature_2m_max&temperature_unit=fahrenheit&timezone=auto"
    )
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            data = json.loads(r.read())
    except Exception as e:
        print(f"  Open-Meteo error: {e}")
        return {}

    for d_str, val in zip(data.get("daily", {}).get("time", []),
                           data.get("daily", {}).get("temperature_2m_max", [])):
        if val is not None:
            cached[date.fromisoformat(d_str)] = val

    return {d: cached[d] for d in (fs + timedelta(n) for n in range((fe - fs).days + 1)) if d in cached}


# ── ticker parsing ─────────────────────────────────────────────────────────────
_MONTH_MAP = {m: i + 1 for i, m in enumerate(
    ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"])}

import re as _re

def _parse_ticker(ticker: str) -> tuple[str | None, date | None, str | None, float | None, float | None]:
    parts = ticker.split("-")
    if len(parts) < 3:
        return None, None, None, None, None
    series, date_seg, strike_seg = parts[0], parts[1], parts[2]
    m = _re.fullmatch(r"(\d{2})([A-Z]{3})(\d{2})", date_seg)
    if not m:
        return series, None, None, None, None
    yr, mon_str, day = m.groups()
    mon = _MONTH_MAP.get(mon_str)
    if not mon:
        return series, None, None, None, None
    try:
        mkt_date = date(2000 + int(yr), mon, int(day))
    except ValueError:
        return series, None, None, None, None

    direction = strike_lo = strike_hi = None
    if strike_seg.startswith("B"):
        try:
            mid = float(strike_seg[1:])
            direction, strike_lo, strike_hi = "between", mid - 0.5, mid + 0.5
        except ValueError:
            pass
    elif strike_seg.startswith("T"):
        try:
            direction, strike_lo = "over", float(strike_seg[1:])
        except ValueError:
            pass
    elif strike_seg.startswith("U"):
        try:
            direction, strike_hi = "under", float(strike_seg[1:])
        except ValueError:
            pass
    return series, mkt_date, direction, strike_lo, strike_hi


# ── forecast signal from actual daily high ─────────────────────────────────────
NWS_BUF = 0.5  # ±0.5°F NWS rounding buffer


def _om_signal(actual_high: float, direction: str,
               strike_lo: float | None, strike_hi: float | None) -> tuple[str, float]:
    """Return (signal, edge_F) using actual daily high as a perfect-forecast proxy."""
    if direction == "over" and strike_lo is not None:
        eff = strike_lo - NWS_BUF
        e = actual_high - eff
        return ("YES" if e >= 0 else "NO"), abs(e)
    if direction == "under" and strike_hi is not None:
        eff = strike_hi + NWS_BUF
        e = eff - actual_high
        return ("YES" if e >= 0 else "NO"), abs(e)
    if direction == "between" and strike_lo is not None and strike_hi is not None:
        eff_lo, eff_hi = strike_lo - NWS_BUF, strike_hi + NWS_BUF
        if eff_lo <= actual_high <= eff_hi:
            return "YES", min(actual_high - eff_lo, eff_hi - actual_high)
        return "NO", min(abs(actual_high - eff_lo), abs(actual_high - eff_hi))
    return "UNKNOWN", 0.0


# ── candle helpers ────────────────────────────────────────────────────────────

def _price_at(candles: list[tuple], ts: int, col: int) -> int | None:
    """Return candle column value at the last candle with period_ts <= ts."""
    val = None
    for c in candles:
        if c[0] <= ts:
            val = c[col]
        else:
            break
    return val


# ── simulation ────────────────────────────────────────────────────────────────

def _simulate(
    candles: list[tuple],
    result: str,
    close_ts: int,
    mkt_date: date,
    tz: ZoneInfo,
    entry_hours: list[int],
    hold_hours: list[int],
) -> list[dict]:
    """For each (entry_hour, hold_hour) pair, record the price move.

    entry price = ask_close at entry_ts  (col 6)
    exit price  = bid_close at exit_ts   (col 2)
    """
    trades = []
    for eh in entry_hours:
        local_entry = datetime(mkt_date.year, mkt_date.month, mkt_date.day,
                               eh, 0, tzinfo=tz)
        entry_ts = int(local_entry.astimezone(timezone.utc).timestamp())
        if entry_ts >= close_ts:
            continue

        entry_ask = _price_at(candles, entry_ts, col=6)  # ask_close
        if entry_ask is None or entry_ask <= 0 or entry_ask >= 100:
            continue

        for hh in hold_hours:
            exit_ts = entry_ts + hh * 3600
            if exit_ts >= close_ts:
                # Held to settlement
                exit_bid = 100 if result == "yes" else 0
                exit_type = "settle"
            else:
                exit_bid = _price_at(candles, exit_ts, col=2)  # bid_close
                if exit_bid is None:
                    continue
                exit_type = f"{hh}h"

            pnl = exit_bid - entry_ask
            trades.append({
                "entry_h":   eh,
                "hold_h":    hh,
                "entry_ask": entry_ask,
                "exit_bid":  exit_bid,
                "pnl":       pnl,
                "exit_type": exit_type,
                "settled_yes": result == "yes",
            })
    return trades


# ── output helpers ────────────────────────────────────────────────────────────

def _grid(all_trades: list[dict], entry_hours: list[int], hold_hours: list[int],
          filter_fn=None, label: str = "") -> None:
    subset = [t for t in all_trades if (filter_fn is None or filter_fn(t))]
    if label:
        print(f"  {label}")
    col_w = 24
    hdr = f"  {'Entry':>7}  " + "".join(f"{'Hold ' + str(h) + 'h':>{col_w}}" for h in hold_hours)
    print(hdr)
    print("  " + "─" * (9 + col_w * len(hold_hours)))
    for eh in entry_hours:
        row = f"  {eh:02d}:00  "
        for hh in hold_hours:
            pnls = [t["pnl"] for t in subset if t["entry_h"] == eh and t["hold_h"] == hh]
            if not pnls:
                row += f"{'—':>{col_w}}"
            else:
                avg = sum(pnls) / len(pnls)
                wr  = sum(1 for p in pnls if p > 0) / len(pnls)
                row += f"{f'{avg:+.1f}¢  {wr:.0%}  n={len(pnls)}':>{col_w}}"
        print(row)
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid forecast-momentum backtest")
    parser.add_argument("--db", default="data/candlesticks.db")
    parser.add_argument("--entry-hours", nargs="+", type=int, default=[7, 8, 9, 10, 11],
                        dest="entry_hours")
    parser.add_argument("--hold-hours", nargs="+", type=int, default=[1, 2, 4],
                        dest="hold_hours")
    parser.add_argument("--min-edge", type=float, default=2.0, dest="min_edge",
                        help="Min OM daily-high edge from effective boundary to include market")
    parser.add_argument("--cities", nargs="+")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: {db_path} not found.")
        sys.exit(1)

    city_filter: set[str] | None = None
    if args.cities:
        city_filter = {ABBREV_TO_METRIC[c.lower()] for c in args.cities
                       if c.lower() in ABBREV_TO_METRIC}

    con = sqlite3.connect(db_path)
    markets = con.execute(
        "SELECT ticker, series, open_ts, close_ts, result FROM markets WHERE result IS NOT NULL"
    ).fetchall()
    markets = [m for m in markets if m[1] in SERIES_TO_METRIC]
    if city_filter:
        markets = [m for m in markets if SERIES_TO_METRIC.get(m[1]) in city_filter]

    print(f"\n{'═'*72}")
    print(f"  FORECAST MOMENTUM BACKTEST  (real Kalshi prices)")
    print(f"{'═'*72}")
    print(f"  DB:          {db_path}  ({len(markets)} settled KXHIGHT markets)")
    print(f"  Entry hours: {args.entry_hours} local")
    print(f"  Hold hours:  {args.hold_hours}")
    print(f"  Min OM edge: {args.min_edge}°F (daily-high vs effective band boundary)")
    print()

    # All simulated trades (both YES and NO settling)
    all_trades: list[dict] = []
    # Forecast signal accuracy
    signal_correct = signal_total = 0

    for ticker, series, open_ts, close_ts, result in markets:
        _, mkt_date, direction, strike_lo, strike_hi = _parse_ticker(ticker)
        if mkt_date is None or direction is None:
            continue
        metric = SERIES_TO_METRIC.get(series)
        if metric is None:
            continue
        city_info = CITIES.get(metric)
        if city_info is None:
            continue
        _, lat, lon, tz = city_info

        # Get actual daily high from Open-Meteo as forecast proxy
        highs = _fetch_om_daily_highs(lat, lon, mkt_date, mkt_date)
        actual_high = highs.get(mkt_date)
        if actual_high is None:
            continue

        signal, edge_f = _om_signal(actual_high, direction, strike_lo, strike_hi)
        if signal not in ("YES", "NO") or edge_f < args.min_edge:
            continue

        # Forecast accuracy
        signal_total += 1
        if (signal == "YES") == (result == "yes"):
            signal_correct += 1

        # Only simulate YES entries (betting the market underprices YES in morning)
        if signal != "YES":
            continue

        candles = con.execute(
            "SELECT period_ts, bid_open, bid_close, bid_low, bid_high, ask_open, ask_close, volume "
            "FROM candles WHERE ticker=? ORDER BY period_ts",
            (ticker,)
        ).fetchall()
        if not candles:
            continue

        trades = _simulate(candles, result, close_ts, mkt_date, tz,
                           args.entry_hours, args.hold_hours)
        for t in trades:
            t["ticker"]  = ticker
            t["series"]  = series
            t["edge_f"]  = round(edge_f, 1)
        all_trades.extend(trades)

    con.close()

    if not all_trades:
        print("  No qualifying trades. Try lowering --min-edge or fetching more data.")
        return

    n_yes = sum(1 for t in all_trades if t["settled_yes"] and t["hold_h"] == args.hold_hours[0])
    n_no  = sum(1 for t in all_trades if not t["settled_yes"] and t["hold_h"] == args.hold_hours[0])

    print(f"  Qualifying markets: {signal_total}  "
          f"(OM signal→YES settling: {n_yes}, OM signal→NO settling: {n_no})")
    if signal_total:
        print(f"  OM forecast accuracy: {signal_correct}/{signal_total} = "
              f"{signal_correct/signal_total:.1%}  (daily-high vs band, edge≥{args.min_edge}°F)")
    print()

    # ── Section 1: YES-settling markets only (signal is correct) ─────────────
    print(f"{'─'*72}")
    print("  SECTION 1 — OM signal=YES, market settled YES  (correct predictions)")
    print("  If avg P&L > 0: morning price undervalued YES → momentum edge exists")
    print(f"{'─'*72}")
    _grid(all_trades, args.entry_hours, args.hold_hours,
          filter_fn=lambda t: t["settled_yes"])

    # ── Section 2: NO-settling markets (signal was wrong) ────────────────────
    print(f"{'─'*72}")
    print("  SECTION 2 — OM signal=YES, market settled NO  (wrong predictions)")
    print("  Shows cost of false positives — how much we'd lose on bad signals")
    print(f"{'─'*72}")
    _grid(all_trades, args.entry_hours, args.hold_hours,
          filter_fn=lambda t: not t["settled_yes"])

    # ── Section 3: Combined (blended expected value) ──────────────────────────
    print(f"{'─'*72}")
    print("  SECTION 3 — ALL trades combined  (realistic expected value)")
    print("  Assumes you enter every market where OM daily-high says YES")
    print(f"{'─'*72}")
    _grid(all_trades, args.entry_hours, args.hold_hours)

    # ── Section 4: Edge breakdown ─────────────────────────────────────────────
    best_hold = max(args.hold_hours)
    print(f"{'─'*72}")
    print(f"  SECTION 4 — Edge breakdown (hold={best_hold}h, YES-settling only)")
    print(f"{'─'*72}")
    buckets: dict[str, list[float]] = defaultdict(list)
    for t in [x for x in all_trades if x["settled_yes"] and x["hold_h"] == best_hold]:
        e = t["edge_f"]
        b = "<2°F" if e < 2 else "2-4°F" if e < 4 else "4-6°F" if e < 6 else "6-8°F" if e < 8 else "≥8°F"
        buckets[b].append(t["pnl"])
    print(f"  {'Edge':>8}  {'N':>5}  {'AvgPnL':>8}  {'WinRate':>8}  {'TotalPnL':>10}")
    print(f"  {'─'*48}")
    for label in ["<2°F", "2-4°F", "4-6°F", "6-8°F", "≥8°F"]:
        pnls = buckets.get(label, [])
        if not pnls:
            continue
        avg = sum(pnls) / len(pnls)
        wr  = sum(1 for p in pnls if p > 0) / len(pnls)
        print(f"  {label:>8}  {len(pnls):>5}  {avg:>+7.1f}¢  {wr:>7.1%}  {sum(pnls):>+9.1f}¢")

    # ── Section 5: By series ──────────────────────────────────────────────────
    series_data: dict[str, list[float]] = defaultdict(list)
    for t in [x for x in all_trades if x["settled_yes"] and x["hold_h"] == best_hold]:
        series_data[t["series"]].append(t["pnl"])
    if len(series_data) > 1:
        print(f"\n{'─'*72}")
        print(f"  SECTION 5 — By series (hold={best_hold}h, YES-settling)")
        print(f"{'─'*72}")
        print(f"  {'Series':<22}  {'N':>5}  {'AvgPnL':>8}  {'WinRate':>8}  {'TotalPnL':>10}")
        print(f"  {'─'*60}")
        for s, pnls in sorted(series_data.items(), key=lambda kv: -sum(kv[1])):
            avg = sum(pnls) / len(pnls)
            wr  = sum(1 for p in pnls if p > 0) / len(pnls)
            print(f"  {s:<22}  {len(pnls):>5}  {avg:>+7.1f}¢  {wr:>7.1%}  {sum(pnls):>+9.1f}¢")

    print(f"\n{'═'*72}")
    print("  Key: avg P&L = profit per trade in cents, win rate = fraction of trades")
    print("  that would have been sold at a gain vs. purchased price (incl. spread).")
    print(f"{'═'*72}\n")


if __name__ == "__main__":
    main()
