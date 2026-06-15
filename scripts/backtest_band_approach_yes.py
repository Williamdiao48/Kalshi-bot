#!/usr/bin/env python3
"""Backtest the Band Approach YES strategy on historical KXLOWT data.

Strategy
--------
Buy YES on a KXLOWT market when:
  1. The METAR running daily minimum is cooling at >= COOL_THRESH °F/hour
  2. The clearance from band ceiling is <= CLR_THRESH °F
  3. YES bid at trigger >= MIN_YES_BID (filters near-worthless entries)

Hold to settlement (no profit-take — analysis showed settlement >> PT).

Data sources
------------
  data/candlesticks.db  — Kalshi KXLOWT hourly YES bid/ask, April–May 2026
  IEM ASOS API          — hourly temperature + dew point per station
  Cache: data/cache/iem_hourly_{station}.json  (avoids re-fetching)

Usage
-----
  venv/bin/python scripts/backtest_band_approach_yes.py
  venv/bin/python scripts/backtest_band_approach_yes.py --no-cache
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sqlite3
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import aiohttp

CANDLES_DB   = Path("data/candlesticks.db")
CACHE_DIR    = Path("data/cache")

_BAND_RE     = re.compile(r"-B([\d.]+)$")
_SERIES_RE   = re.compile(r"^(KXLOWT[A-Z]+)-")

# IEM station → (station_id, network)
SERIES_TO_IEM: dict[str, tuple[str, str]] = {
    "KXLOWTATL":  ("ATL",  "GA_ASOS"),
    "KXLOWTAUS":  ("AUS",  "TX_ASOS"),
    "KXLOWTBOS":  ("BOS",  "MA_ASOS"),
    "KXLOWTCHI":  ("MDW",  "IL_ASOS"),
    "KXLOWTDC":   ("DCA",  "DC_ASOS"),
    "KXLOWTDEN":  ("DEN",  "CO_ASOS"),
    "KXLOWTHOU":  ("HOU",  "TX_ASOS"),
    "KXLOWTLV":   ("LAS",  "NV_ASOS"),
    "KXLOWTLAX":  ("LAX",  "CA_ASOS"),
    "KXLOWTMIA":  ("MIA",  "FL_ASOS"),
    "KXLOWTMIN":  ("MSP",  "MN_ASOS"),
    "KXLOWTNOLA": ("MSY",  "LA_ASOS"),
    "KXLOWTNYC":  ("NYC",  "NY_ASOS"),
    "KXLOWTOKC":  ("OKC",  "OK_ASOS"),
    "KXLOWTPHIL": ("PHL",  "PA_ASOS"),
    "KXLOWTPHX":  ("PHX",  "AZ_ASOS"),
    "KXLOWTSATX": ("SAT",  "TX_ASOS"),
    "KXLOWTSEA":  ("SEA",  "WA_ASOS"),
    "KXLOWTSFO":  ("SFO",  "CA_ASOS"),
}

_IEM_BASE = "https://mesonet.agron.iastate.edu/api/1/asos.json"
_HEADERS  = {"User-Agent": "KalshiBot/1.0 (williamdiao32@g.ucla.edu)"}


# ---------------------------------------------------------------------------
# IEM fetch
# ---------------------------------------------------------------------------

async def fetch_iem_hourly(
    session: aiohttp.ClientSession,
    station: str,
    network: str,
    start: str,   # YYYY-MM-DD
    end: str,
    use_cache: bool = True,
) -> dict[str, dict]:
    """Return {YYYY-MM-DDTHH:00: {tmpf, dwpt}} for each UTC hour."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"iem_hourly_{station}_{start}_{end}.json"

    if use_cache and cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    # Use CGI endpoint with report_type=3 (synoptic hourly METAR obs, one per hour)
    # Use CGI endpoint with report_type=3 (synoptic hourly METAR, one per hour)
    # data=all returns all fields including dwpf (dew point °F)
    cgi_url  = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.strptime(end,   "%Y-%m-%d")
    params = {
        "station":     station,
        "year1":       start_dt.year,  "month1": start_dt.month, "day1": start_dt.day,
        "year2":       end_dt.year,    "month2": end_dt.month,   "day2": end_dt.day,
        "data":        "all",          # includes tmpf AND dwpf
        "tz":          "UTC",
        "format":      "onlycomma",
        "latlon":      "no",
        "missing":     "M",
        "report_type": "3",
    }
    try:
        async with session.get(cgi_url, params=params, headers=_HEADERS,
                               timeout=aiohttp.ClientTimeout(total=30)) as resp:
            resp.raise_for_status()
            text = await resp.text()
    except Exception as exc:
        print(f"  [IEM] {station}: {exc}", file=sys.stderr)
        return {}

    # Parse CSV — header: station,valid,tmpf,dwpf,relh,...
    hourly: dict[str, dict] = {}
    lines = text.strip().splitlines()
    if not lines:
        return {}
    header = [h.strip() for h in lines[0].split(",")]
    tmpf_idx = header.index("tmpf") if "tmpf" in header else None
    dwpf_idx = header.index("dwpf") if "dwpf" in header else None

    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) < 3:
            continue
        ts_str = parts[1].strip()
        try:
            dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
            hour_key = dt.strftime("%Y-%m-%dT%H:00")
        except ValueError:
            continue
        tmpf_raw = parts[tmpf_idx].strip() if tmpf_idx is not None else "M"
        dwpf_raw = parts[dwpf_idx].strip() if dwpf_idx is not None else "M"
        if tmpf_raw == "M":
            continue
        try:
            tmpf_val = float(tmpf_raw)
        except ValueError:
            continue
        dwpf_val = None
        if dwpf_raw not in ("M", ""):
            try:
                dwpf_val = float(dwpf_raw)
            except ValueError:
                pass
        hourly[hour_key] = {"tmpf": tmpf_val, "dwpt": dwpf_val}

    if use_cache:
        with open(cache_file, "w") as f:
            json.dump(hourly, f)

    return hourly


def compute_running_min(
    hourly: dict[str, dict],
    market_open_utc: datetime,
    market_close_utc: datetime,
) -> dict[str, float | None]:
    """Return {hour_key: running_min_f} for each UTC hour in the market window."""
    # KXLOWT settles on the official climate day (midnight–midnight LOCAL).
    # Approximate: use UTC hours within [open_ts, close_ts].
    result: dict[str, float | None] = {}
    running_min: float | None = None
    cur = market_open_utc.replace(minute=0, second=0, microsecond=0)
    while cur <= market_close_utc:
        key = cur.strftime("%Y-%m-%dT%H:00")
        obs = hourly.get(key)
        if obs:
            t = obs["tmpf"]
            running_min = t if running_min is None else min(running_min, t)
        result[key] = running_min
        cur += timedelta(hours=1)
    return result


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def _band_ceiling(ticker: str) -> float | None:
    m = _BAND_RE.search(ticker)
    return float(m.group(1)) + 1.0 if m else None


def _series(ticker: str) -> str | None:
    m = _SERIES_RE.match(ticker)
    return m.group(1) if m else None


def simulate_market(
    ticker: str,
    open_ts: int,
    close_ts: int,
    result: str,
    candles: list[tuple],          # (period_ts, bid_close, ask_close)
    running_mins: dict[str, float | None],
    iem_hourly: dict[str, dict],   # for dew point
    cool_thresh: float,
    clr_thresh: float,
    min_yes_bid: int,
) -> dict | None:
    """Find first trigger and simulate YES buy. Returns trade dict or None."""
    ceiling = _band_ceiling(ticker)
    if ceiling is None:
        return None

    prev_rmin: float | None = None
    triggered = False

    for period_ts, bid_close, ask_close in candles:
        if bid_close is None or ask_close is None:
            continue

        hour_key = datetime.fromtimestamp(period_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:00")
        rmin = running_mins.get(hour_key)
        dwpt = (iem_hourly.get(hour_key) or {}).get("dwpt")

        if rmin is None:
            prev_rmin = None
            continue

        clearance = rmin - ceiling

        # Compute cooling rate (change vs previous hour)
        cooling = (rmin - prev_rmin) if prev_rmin is not None else 0.0
        prev_rmin = rmin

        if triggered:
            continue  # already have an entry, skip

        # Trigger conditions
        if cooling <= cool_thresh and clearance <= clr_thresh and bid_close >= min_yes_bid:
            # Enter YES: pay YES ask
            yes_cost = ask_close
            hours_to_close = (close_ts - period_ts) / 3600

            # Settlement P&L
            settle_val = 100 if result == "yes" else 0
            settle_pnl = settle_val - yes_cost

            triggered = True
            return {
                "ticker":          ticker,
                "ceiling":         ceiling,
                "cooling_at_trig": round(cooling, 2),
                "clearance":       round(clearance, 2),
                "yes_bid":         bid_close,
                "yes_cost":        yes_cost,
                "dew_point":       dwpt,
                "dew_margin":      round(dwpt - ceiling, 2) if dwpt is not None else None,
                "hours_to_close":  round(hours_to_close, 1),
                "settle_pnl":      settle_pnl,
                "settled_yes":     result == "yes",
                "result":          result,
            }

    return None


async def run_backtest(args: argparse.Namespace) -> None:
    con = sqlite3.connect(CANDLES_DB)

    markets = con.execute("""
        SELECT ticker, open_ts, close_ts, result
        FROM markets
        WHERE ticker LIKE 'KXLOWT%B%'
          AND result IN ('yes','no')
    """).fetchall()
    print(f"{len(markets)} settled KXLOWT B-band markets")

    # Group by series to batch IEM fetches
    series_set = {_series(t) for t, *_ in markets if _series(t)}
    series_set = {s for s in series_set if s in SERIES_TO_IEM}
    print(f"{len(series_set)} series needing IEM data")

    # Date range
    all_open  = [m[1] for m in markets]
    all_close = [m[2] for m in markets]
    start_dt  = datetime.fromtimestamp(min(all_open),  tz=timezone.utc)
    end_dt    = datetime.fromtimestamp(max(all_close), tz=timezone.utc)
    start_str = start_dt.date().isoformat()
    end_str   = end_dt.date().isoformat()
    print(f"Date range: {start_str} → {end_str}")

    # Fetch IEM data sequentially (rate-limit-safe: 1 req/2s)
    iem_data: dict[str, dict] = {}
    async with aiohttp.ClientSession() as session:
        for ser in sorted(series_set):
            station, network = SERIES_TO_IEM[ser]
            res = await fetch_iem_hourly(
                session, station, network, start_str, end_str,
                use_cache=not args.no_cache,
            )
            iem_data[ser] = res
            print(f"  {ser} ({station}): {len(res)} hourly obs")
            await asyncio.sleep(2.0)  # respect IEM rate limit

    # Load candles
    all_candles: dict[str, list] = {}
    for ticker, *_ in markets:
        rows = con.execute(
            "SELECT period_ts, bid_close, ask_close FROM candles WHERE ticker=? ORDER BY period_ts",
            (ticker,),
        ).fetchall()
        if rows:
            all_candles[ticker] = rows

    # Parameter sweep
    COOL_THRESHOLDS = [-0.5, -1.0, -1.5, -2.0]
    CLR_THRESHOLDS  = [1.0, 2.0, 3.0, 4.0]
    MIN_YES_BIDS    = [5, 8, 12]

    print(f"\n{'cool_t':>7} {'clr_t':>6} {'min_bid':>8} {'n':>6} "
          f"{'win%':>7} {'avg¢':>8} {'total$':>9} {'dew_ok%':>9}")
    print("-" * 70)

    best_results = []

    for cool_thresh in COOL_THRESHOLDS:
        for clr_thresh in CLR_THRESHOLDS:
            for min_yes_bid in MIN_YES_BIDS:
                trades: list[dict] = []

                for ticker, open_ts, close_ts, result in markets:
                    ser = _series(ticker)
                    if ser not in SERIES_TO_IEM:
                        continue
                    candles = all_candles.get(ticker, [])
                    if not candles:
                        continue

                    hourly = iem_data.get(ser, {})
                    open_dt  = datetime.fromtimestamp(open_ts,  tz=timezone.utc)
                    close_dt = datetime.fromtimestamp(close_ts, tz=timezone.utc)
                    running_mins = compute_running_min(hourly, open_dt, close_dt)

                    t = simulate_market(
                        ticker, open_ts, close_ts, result,
                        candles, running_mins, hourly,
                        cool_thresh, clr_thresh, min_yes_bid,
                    )
                    if t:
                        trades.append(t)

                n = len(trades)
                if n < 5:
                    continue

                wins   = sum(1 for t in trades if t["settled_yes"])
                avg    = sum(t["settle_pnl"] for t in trades) / n
                total  = avg * n / 100
                dew_ok = sum(1 for t in trades if t["dew_margin"] is not None
                             and t["dew_margin"] > 0)
                dew_pct = dew_ok / n if n else 0

                best_results.append((total, cool_thresh, clr_thresh, min_yes_bid,
                                     n, wins, avg, dew_pct))
                print(f"{cool_thresh:>7.1f} {clr_thresh:>6.1f} {min_yes_bid:>8} "
                      f"{n:>6} {wins/n:>6.0%} {avg:>+8.1f} {total:>+9.2f} "
                      f"{dew_pct:>8.0%}")

    best_results.sort(reverse=True)

    if not best_results:
        print("No parameter combos hit n≥5.")
        return

    total, ct, clt, mb, n, wins, avg, dpct = best_results[0]
    print(f"\n→ Best: cool≤{ct}°/h  clr≤{clt}°  min_bid={mb}¢")
    print(f"  {n} trades  {wins}W/{n-wins}L ({wins/n:.0%})  "
          f"avg {avg:+.1f}¢  total ${total:+.2f}")
    print(f"  Dew point margin > 0 (safe): {dpct:.0%} of trades")

    # Detail for best params
    detail_trades: list[dict] = []
    for ticker, open_ts, close_ts, result in markets:
        ser = _series(ticker)
        if ser not in SERIES_TO_IEM:
            continue
        candles = all_candles.get(ticker, [])
        if not candles:
            continue
        hourly = iem_data.get(ser, {})
        open_dt  = datetime.fromtimestamp(open_ts,  tz=timezone.utc)
        close_dt = datetime.fromtimestamp(close_ts, tz=timezone.utc)
        running_mins = compute_running_min(hourly, open_dt, close_dt)
        t = simulate_market(ticker, open_ts, close_ts, result,
                            candles, running_mins, hourly, ct, clt, mb)
        if t:
            detail_trades.append(t)

    detail_trades.sort(key=lambda x: x["settle_pnl"])
    print(f"\n{'ticker':45} {'cost':>5} {'ceil':>5} {'rmin_clr':>9} "
          f"{'cool':>6} {'dew_m':>7} {'pnl':>7} {'result':>7}")
    print("-" * 100)
    for t in detail_trades[:8]:
        dm = f"{t['dew_margin']:+.1f}" if t['dew_margin'] is not None else " N/A"
        print(f"{t['ticker']:45} {t['yes_cost']:>5} {t['ceiling']:>5.1f} "
              f"{t['clearance']:>+9.2f} {t['cooling_at_trig']:>+6.2f} "
              f"{dm:>7} {t['settle_pnl']:>+7.0f} {t['result']:>7}")
    print("  …")
    for t in detail_trades[-8:]:
        dm = f"{t['dew_margin']:+.1f}" if t['dew_margin'] is not None else " N/A"
        print(f"{t['ticker']:45} {t['yes_cost']:>5} {t['ceiling']:>5.1f} "
              f"{t['clearance']:>+9.2f} {t['cooling_at_trig']:>+6.2f} "
              f"{dm:>7} {t['settle_pnl']:>+7.0f} {t['result']:>7}")

    # Dew point analysis
    dew_trades = [t for t in detail_trades if t["dew_margin"] is not None]
    if dew_trades:
        print(f"\n=== Dew point analysis ({len(dew_trades)} trades with dew point data) ===")
        above = [t for t in dew_trades if t["dew_margin"] > 0]
        below = [t for t in dew_trades if t["dew_margin"] <= 0]
        if above:
            wa = sum(1 for t in above if t["settled_yes"])
            print(f"  Dew point ABOVE ceiling (safe zone): {len(above)} trades, "
                  f"{wa}W/{len(above)-wa}L ({wa/len(above):.0%})  "
                  f"avg {sum(t['settle_pnl'] for t in above)/len(above):+.1f}¢")
        if below:
            wb = sum(1 for t in below if t["settled_yes"])
            print(f"  Dew point BELOW ceiling (risk zone): {len(below)} trades, "
                  f"{wb}W/{len(below)-wb}L ({wb/len(below):.0%})  "
                  f"avg {sum(t['settle_pnl'] for t in below)/len(below):+.1f}¢")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cache", action="store_true",
                        help="Re-fetch IEM data instead of using cache")
    args = parser.parse_args()

    if not CANDLES_DB.exists():
        sys.exit(f"ERROR: {CANDLES_DB} not found")

    asyncio.run(run_backtest(args))


if __name__ == "__main__":
    main()
