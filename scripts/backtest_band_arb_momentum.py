#!/usr/bin/env python3
"""Backtest: momentum YES entry on band markets.

Market makers reprice Kalshi band YES markets sharply before ASOS/METAR can
confirm a temperature crossing, using faster data (1-min feeds, airport sensors,
or forecast trajectory extrapolation).  This script detects that repricing signal
in historical Kalshi trade fills and simulates a stall-bail exit strategy.

Data source: Kalshi /trade-api/v2/markets/trades (fill-level history, all settled
             markets available).

Strategy:
  Entry  — YES fill price jumps ≥ MIN_DELTA cents in one 10-second bin, landing
            ≥ JUMP_THRESHOLD, when recent price was < BASELINE_MAX.
  Exit   — bail after STALL_TICKS consecutive bins where the price does not
            improve (≤ previous).  Otherwise, PT or SL as backstops.
"""

from __future__ import annotations

import asyncio
import importlib.util
import itertools
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import NamedTuple
from zoneinfo import ZoneInfo

import aiohttp

sys.path.insert(0, str(Path(__file__).parent.parent))
from kalshi_bot.auth import generate_headers

CACHE_FILE = Path(__file__).parent.parent / "data" / "momentum_backtest_cache.json"

# ---------------------------------------------------------------------------
# Series to scan — temperature B-markets only (no lows)
# ---------------------------------------------------------------------------
HIGH_SERIES: tuple[str, ...] = (
    # Original KXHIGH* series
    "KXHIGHLAX", "KXHIGHDEN", "KXHIGHCHI", "KXHIGHNY",  "KXHIGHMIA",
    "KXHIGHDAL", "KXHIGHBOS", "KXHIGHAUS", "KXHIGHOU",
    # New KXHIGHT* series
    "KXHIGHTSFO", "KXHIGHTSEA", "KXHIGHTBOS", "KXHIGHTPHX", "KXHIGHPHIL",
    "KXHIGHTATL", "KXHIGHTMIN", "KXHIGHTDC",  "KXHIGHTLV",  "KXHIGHTOKC",
    "KXHIGHTDAL", "KXHIGHTSATX", "KXHIGHTHOU", "KXHIGHTNOLA",
)

BIN_SECONDS   = 10       # fast-loop cadence (seconds per price bin)
SPREAD_EST    = 10       # estimated bid/ask spread in cents (bid = fill − SPREAD_EST)
CONTRACTS     = 5        # simulated contract count per trade
MAX_RATE      = 4        # max parallel requests (stay well under Kalshi limit)
LOOKBACK_DAYS = 28       # how many days of settled markets to include

# ---------------------------------------------------------------------------
# City metadata: ticker city-code → (metric_key, IANA timezone)
# ---------------------------------------------------------------------------
_CITY_TO_META: dict[str, tuple[str, str]] = {
    "MIA":  ("temp_high_mia",  "America/New_York"),
    "BOS":  ("temp_high_bos",  "America/New_York"),
    "NY":   ("temp_high_ny",   "America/New_York"),
    "DC":   ("temp_high_dca",  "America/New_York"),
    "PHIL": ("temp_high_phl",  "America/New_York"),
    "ATL":  ("temp_high_atl",  "America/New_York"),
    "CHI":  ("temp_high_chi",  "America/Chicago"),
    "HOU":  ("temp_high_hou",  "America/Chicago"),
    "DAL":  ("temp_high_dal",  "America/Chicago"),
    "NOLA": ("temp_high_msy",  "America/Chicago"),
    "MIN":  ("temp_high_msp",  "America/Chicago"),
    "OKC":  ("temp_high_okc",  "America/Chicago"),
    "SATX": ("temp_high_sat",  "America/Chicago"),
    "AUS":  ("temp_high_aus",  "America/Chicago"),
    "DEN":  ("temp_high_den",  "America/Denver"),
    "PHX":  ("temp_high_phx",  "America/Phoenix"),
    "LV":   ("temp_high_las",  "America/Los_Angeles"),
    "LAX":  ("temp_high_lax",  "America/Los_Angeles"),
    "SFO":  ("temp_high_sfo",  "America/Los_Angeles"),
    "SEA":  ("temp_high_sea",  "America/Los_Angeles"),
}


def _parse_ticker_city(ticker: str) -> str | None:
    """Extract city abbreviation from ticker, handling both KXHIGH and KXHIGHT prefixes."""
    m = re.match(r"KXHIGHT([A-Z]+)-", ticker)
    if m:
        return m.group(1)
    m = re.match(r"KXHIGH([A-Z]+)-", ticker)
    return m.group(1) if m else None


def _parse_ticker_date(ticker: str):
    """Return date from ticker like KXHIGHMIA-26MAY19-B86.5.

    Kalshi format is YYMMMDD: first two digits = year, last two = day.
    e.g. 26MAY19 → May 19, 2026.
    """
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})-", ticker)
    if not m:
        return None
    mon_map = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
               "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}
    try:
        from datetime import date
        return date(2000 + int(m.group(1)), mon_map[m.group(2)], int(m.group(3)))
    except Exception:
        return None


def _load_p75() -> dict[str, dict[int, int]]:
    """Load P75_MINUTES from data/peak_hour_p90.py."""
    p75_path = Path(__file__).parent.parent / "data" / "peak_hour_p90.py"
    if not p75_path.exists():
        return {}
    spec = importlib.util.spec_from_file_location("peak_hour_p90", p75_path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return getattr(mod, "P75_MINUTES", {})


def _p75_window(ticker: str, p75_data: dict) -> tuple[datetime, datetime] | None:
    """Return (utc_start, utc_end) for the ±2h p75 window for this ticker's city/month."""
    city = _parse_ticker_city(ticker)
    if city is None:
        return None
    meta = _CITY_TO_META.get(city)
    if meta is None:
        return None
    metric_key, tz_name = meta
    tdate = _parse_ticker_date(ticker)
    if tdate is None:
        return None
    p75_mins = p75_data.get(metric_key, {}).get(tdate.month)
    if p75_mins is None:
        return None
    tz = ZoneInfo(tz_name)
    local_midnight = datetime(tdate.year, tdate.month, tdate.day, tzinfo=tz)
    p75_local = local_midnight + timedelta(minutes=p75_mins)
    return (
        (p75_local - timedelta(hours=2)).astimezone(timezone.utc),
        (p75_local + timedelta(hours=2)).astimezone(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _save_cache(market_buckets: dict[str, list[tuple[datetime, int]]]) -> None:
    data = {
        ticker: [[ts.isoformat(), price] for ts, price in buckets]
        for ticker, buckets in market_buckets.items()
    }
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(data))
    total_buckets = sum(len(v) for v in market_buckets.values())
    avg_buckets = total_buckets // max(len(market_buckets), 1)
    print(f"  Saved cache → {CACHE_FILE} ({len(market_buckets)} markets, avg {avg_buckets} buckets/market)\n", flush=True)


def _load_cache() -> dict[str, list[tuple[datetime, int]]]:
    data = json.loads(CACHE_FILE.read_text())
    return {
        ticker: [(datetime.fromisoformat(row[0]), row[1]) for row in rows]
        for ticker, rows in data.items()
    }

# ---------------------------------------------------------------------------
# Kalshi API helpers
# ---------------------------------------------------------------------------

async def _get(session: aiohttp.ClientSession, path: str, params: dict, _retries: int = 0) -> dict:
    headers = generate_headers("GET", path)
    url = "https://api.elections.kalshi.com" + path
    async with session.get(url, headers=headers, params=params,
                           timeout=aiohttp.ClientTimeout(total=15)) as r:
        if r.status == 429:
            if _retries >= 10:
                raise RuntimeError(f"429 rate limit after 10 retries: {path}")
            wait = 2 * (2 ** min(_retries, 4))  # 2, 4, 8, 16, 32s cap
            print(f"  [429] rate limited, waiting {wait}s (retry {_retries+1}/10)…", flush=True)
            await asyncio.sleep(wait)
            return await _get(session, path, params, _retries + 1)
        r.raise_for_status()
        return await r.json()


async def fetch_settled_b_tickers(session: aiohttp.ClientSession) -> list[str]:
    """Return all settled B-market tickers across HIGH_SERIES."""
    tickers: list[str] = []
    for si, series in enumerate(HIGH_SERIES, 1):
        print(f"  [{si}/{len(HIGH_SERIES)}] {series}…", flush=True)
        cursor = None
        page = 0
        while True:
            params: dict = {"series_ticker": series, "status": "settled", "limit": 100}
            if cursor:
                params["cursor"] = cursor
            try:
                d = await _get(session, "/trade-api/v2/markets", params)
            except Exception as exc:
                print(f"  [warn] {series}: {exc}", flush=True)
                break
            batch = d.get("markets", [])
            for m in batch:
                t = m["ticker"]
                if "-B" in t:
                    tickers.append(t)
            page += 1
            cursor = d.get("cursor")
            if not cursor or not batch:
                break
            await asyncio.sleep(0.25)
        print(f"       → {len(tickers)} B-tickers so far", flush=True)
    return tickers


async def fetch_trade_history(session: aiohttp.ClientSession, ticker: str) -> list[tuple[datetime, int]]:
    """Return chronological (timestamp, yes_price_cents) pairs for one market."""
    path = "/trade-api/v2/markets/trades"
    fills: list[tuple[datetime, int]] = []
    cursor = None
    while True:
        params: dict = {"ticker": ticker, "limit": 100}
        if cursor:
            params["cursor"] = cursor
        try:
            d = await _get(session, path, params)
        except Exception:
            break
        for t in d.get("trades", []):
            try:
                ts = datetime.fromisoformat(t["created_time"].replace("Z", "+00:00"))
                price = round(float(t["yes_price_dollars"]) * 100)
                fills.append((ts, price))
            except Exception:
                continue
        cursor = d.get("cursor")
        if not cursor or not d.get("trades"):
            break
    fills.sort(key=lambda x: x[0])
    return fills


# ---------------------------------------------------------------------------
# Price binning
# ---------------------------------------------------------------------------

def bin_fills(fills: list[tuple[datetime, int]]) -> list[tuple[datetime, int]]:
    """Collapse fills into BIN_SECONDS-wide buckets.  Take the last fill price
    in each bucket (best approximation of ask at bucket close).  Fill gaps by
    carrying the last known price forward (no phantom signal from missing data).
    """
    if not fills:
        return []
    epoch = fills[0][0]

    def bucket_idx(ts: datetime) -> int:
        return int((ts - epoch).total_seconds() // BIN_SECONDS)

    bucket_map: dict[int, int] = {}
    for ts, price in fills:
        idx = bucket_idx(ts)
        bucket_map[idx] = price  # last fill in bucket wins

    if not bucket_map:
        return []
    max_idx = max(bucket_map)
    buckets: list[tuple[datetime, int]] = []
    last_price = fills[0][1]
    for i in range(max_idx + 1):
        if i in bucket_map:
            last_price = bucket_map[i]
        bucket_ts = datetime(
            epoch.year, epoch.month, epoch.day,
            epoch.hour, epoch.minute, epoch.second,
            tzinfo=epoch.tzinfo
        )
        from datetime import timedelta
        bucket_ts = epoch + timedelta(seconds=i * BIN_SECONDS)
        buckets.append((bucket_ts, last_price))
    return buckets


# ---------------------------------------------------------------------------
# Signal detection and exit simulation
# ---------------------------------------------------------------------------

class TradeResult(NamedTuple):
    ticker:      str
    entry_ts:    datetime
    entry_price: int    # cents
    exit_ts:     datetime
    exit_price:  int    # cents (bid estimate)
    exit_reason: str    # "stall_bail" | "profit_take" | "stop_loss" | "settlement"
    pnl_cents:   float  # per CONTRACTS contracts

    @property
    def win(self) -> bool:
        return self.pnl_cents > 0


def detect_and_simulate(
    ticker: str,
    buckets: list[tuple[datetime, int]],
    *,
    baseline_max: int,
    jump_threshold: int,
    min_jump: int,
    stall_ticks: int,
    pt_frac: float,
    sl_frac: float,
    entry_window: tuple[datetime, datetime] | None = None,
) -> TradeResult | None:
    """Find the first momentum signal in this market's price series and
    simulate the full trade from entry to exit.

    entry_window: if given, only consider entry candidates whose timestamp
    falls within (utc_start, utc_end).  Signals outside this window are
    ignored.  Used to test the ±2h p75 hypothesis.
    """
    if len(buckets) < 4:
        return None

    # Scan for jump signal
    entry_idx: int | None = None
    entry_price: int = 0
    entry_ts: datetime = buckets[0][0]

    for i in range(3, len(buckets)):
        price = buckets[i][1]
        prev  = buckets[i - 1][1]
        # Recent max over prior 3 buckets (t-3, t-2, t-1)
        recent_max = max(buckets[i - 3][1], buckets[i - 2][1], buckets[i - 1][1])
        if (recent_max < baseline_max
                and price >= jump_threshold
                and (price - prev) >= min_jump):
            # Apply p75 window filter if requested
            if entry_window is not None:
                win_start, win_end = entry_window
                if not (win_start <= buckets[i][0] <= win_end):
                    continue
            entry_idx   = i
            entry_price = price
            entry_ts    = buckets[i][0]
            break

    if entry_idx is None:
        return None

    # Bid estimate at entry (we paid ask; we'd exit at bid)
    def bid(price: int) -> int:
        return max(0, price - SPREAD_EST)

    # Simulate exit from entry onwards
    pt_target  = entry_price * (1 + pt_frac)
    sl_floor   = entry_price * sl_frac
    stall_count = 0
    prev_bid   = bid(entry_price)

    for j in range(entry_idx + 1, len(buckets)):
        ts, price = buckets[j]
        curr_bid  = bid(price)

        # Stall bail: bid didn't improve
        if curr_bid <= prev_bid:
            stall_count += 1
        else:
            stall_count = 0
        prev_bid = curr_bid

        if stall_count >= stall_ticks:
            exit_p = curr_bid
            pnl = (exit_p - entry_price) * CONTRACTS
            return TradeResult(ticker, entry_ts, entry_price, ts, exit_p, "stall_bail", pnl)

        if curr_bid >= pt_target:
            exit_p = curr_bid
            pnl = (exit_p - entry_price) * CONTRACTS
            return TradeResult(ticker, entry_ts, entry_price, ts, exit_p, "profit_take", pnl)

        if curr_bid <= sl_floor:
            exit_p = curr_bid
            pnl = (exit_p - entry_price) * CONTRACTS
            return TradeResult(ticker, entry_ts, entry_price, ts, exit_p, "stop_loss", pnl)

    # Held to settlement — YES settled at 100¢ (market was in-band)
    exit_p = 100
    pnl = (exit_p - entry_price) * CONTRACTS
    last_ts = buckets[-1][0]
    return TradeResult(ticker, entry_ts, entry_price, last_ts, exit_p, "settlement", pnl)


# ---------------------------------------------------------------------------
# Stats helper
# ---------------------------------------------------------------------------

def stats(results: list[TradeResult]) -> dict:
    if not results:
        return {"n": 0, "wins": 0, "wr": 0.0, "total": 0.0, "avg": 0.0}
    wins  = sum(1 for r in results if r.win)
    total = sum(r.pnl_cents for r in results)
    return {
        "n":    len(results),
        "wins": wins,
        "wr":   wins / len(results),
        "total": total,
        "avg":  total / len(results),
    }


def fmt(s: dict) -> str:
    return (f"n={s['n']:3d}  WR={s['wr']:5.1%}"
            f"  total={s['total']:+8.1f}¢  avg={s['avg']:+7.1f}¢")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _run_sweep(
    market_buckets: dict[str, list[tuple[datetime, int]]],
    param_grid: dict,
    p75_data: dict | None = None,
    label: str = "",
) -> list[tuple[dict, list[TradeResult]]]:
    """Run the full parameter sweep over market_buckets.

    If p75_data is provided, each ticker's entry is restricted to its ±2h
    p75 window.  Tickers with no p75 data available are skipped entirely
    (to avoid mixing windowed and unwindowed trades).
    """
    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    total = len(combos)
    print(f"  Running {total} param combos over {len(market_buckets)} markets [{label}]…", flush=True)
    sweep: list[tuple[dict, list[TradeResult]]] = []
    for ci, combo in enumerate(combos, 1):
        if ci % 100 == 0 or ci == total:
            print(f"  {ci}/{total} combos done…", flush=True)
        cfg = dict(zip(keys, combo))
        trades: list[TradeResult] = []
        for ticker, buckets in market_buckets.items():
            window: tuple[datetime, datetime] | None = None
            if p75_data is not None:
                window = _p75_window(ticker, p75_data)
                if window is None:
                    continue  # skip tickers with no p75 data in this mode
            r = detect_and_simulate(ticker, buckets, **cfg, entry_window=window)
            if r is not None:
                trades.append(r)
        if len(trades) >= 5:
            sweep.append((cfg, trades))
    sweep.sort(key=lambda x: stats(x[1])["total"], reverse=True)
    print(f"  Sweep [{label}] complete — {len(sweep)} configs with n≥5.", flush=True)
    return sweep


def _print_sweep(sweep_results: list[tuple[dict, list[TradeResult]]], title: str) -> dict | None:
    """Print the top-25 / top-10 tables and best-config detail.  Returns best_cfg."""
    hdr = (f"{'bsln':>4} {'jmp':>4} {'dlt':>4} {'stl':>3} {'PT':>5} {'SL':>5}"
           f" | {'n':>4} {'WR':>6} {'total_¢':>9} {'avg_¢':>8}")

    print(f"\n{'=' * 72}", flush=True)
    print(f"PARAMETER SWEEP — {title} — Top 25 by total P&L  (n ≥ 5)", flush=True)
    print(f"{'=' * 72}", flush=True)
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for cfg, trades in sweep_results[:25]:
        s = stats(trades)
        print(
            f"{cfg['baseline_max']:4d} {cfg['jump_threshold']:4d} {cfg['min_jump']:4d}"
            f" {cfg['stall_ticks']:3d} {cfg['pt_frac']:5.0%} {cfg['sl_frac']:5.0%}"
            f" | {s['n']:4d} {s['wr']:6.1%} {s['total']:9.1f}¢ {s['avg']:8.1f}¢"
        )

    by_avg = sorted(
        [(cfg, trades) for cfg, trades in sweep_results if len(trades) >= 10],
        key=lambda x: stats(x[1])["avg"], reverse=True,
    )
    print(f"\nTop 10 configs by avg P&L  (n ≥ 10)", flush=True)
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for cfg, trades in by_avg[:10]:
        s = stats(trades)
        print(
            f"{cfg['baseline_max']:4d} {cfg['jump_threshold']:4d} {cfg['min_jump']:4d}"
            f" {cfg['stall_ticks']:3d} {cfg['pt_frac']:5.0%} {cfg['sl_frac']:5.0%}"
            f" | {s['n']:4d} {s['wr']:6.1%} {s['total']:9.1f}¢ {s['avg']:8.1f}¢"
        )

    if not sweep_results:
        return None

    best_cfg, best_trades = sweep_results[0]
    print(f"\n{'=' * 72}", flush=True)
    print(f"BEST CONFIG DETAIL: {best_cfg}", flush=True)
    print(f"{'=' * 72}", flush=True)

    from collections import Counter
    reasons = Counter(r.exit_reason for r in best_trades)
    print("\nExit reason breakdown:", flush=True)
    for reason, n in reasons.most_common():
        sub = [r for r in best_trades if r.exit_reason == reason]
        ss  = stats(sub)
        print(f"  {reason:15s}: {ss['n']:3d}  WR={ss['wr']:5.1%}  total={ss['total']:+8.1f}¢  avg={ss['avg']:+7.1f}¢", flush=True)

    entry_prices = [r.entry_price for r in best_trades]
    print("\nEntry price distribution:", flush=True)
    for lo, hi in [(50,60), (60,65), (65,70), (70,75), (75,80), (80,90)]:
        n = sum(1 for p in entry_prices if lo <= p < hi)
        print(f"  {lo}-{hi}¢: {n}", flush=True)

    worst = sorted(best_trades, key=lambda r: r.pnl_cents)[:8]
    print("\nWorst 8 trades:", flush=True)
    for r in worst:
        print(f"  {r.ticker:40s}  entry={r.entry_price:2d}¢  exit={r.exit_price:2d}¢"
              f"  {r.exit_reason:12s}  pnl={r.pnl_cents:+.0f}¢")

    city_pnl: dict[str, list[float]] = defaultdict(list)
    for r in best_trades:
        m = re.match(r"KXHIGHT?([A-Z]+)-", r.ticker)
        city = m.group(1) if m else "?"
        city_pnl[city].append(r.pnl_cents)
    print("\nCity breakdown:", flush=True)
    city_stats = sorted(
        [(city, sum(v), len(v), sum(1 for x in v if x > 0)) for city, v in city_pnl.items()],
        key=lambda x: x[1], reverse=True,
    )
    for city, total, n, wins in city_stats:
        print(f"  {city:8s}: n={n:3d}  WR={wins/n:5.1%}  total={total:+8.1f}¢", flush=True)

    # Stall sensitivity using best entry params
    best_entry = {k: best_cfg[k] for k in ("baseline_max", "jump_threshold", "min_jump")}
    print(f"\n{'=' * 72}", flush=True)
    print(f"STALL EXIT SENSITIVITY  (entry params fixed: {best_entry})", flush=True)
    print(f"{'=' * 72}", flush=True)
    for stall in [1, 2, 3, 999]:
        for pt in [0.15, 0.20, 0.30]:
            for sl in [0.70, 0.80]:
                trades_s = []
                for ticker, buckets in _active_market_buckets.items():
                    window: tuple[datetime, datetime] | None = None
                    if _active_p75 is not None:
                        window = _p75_window(ticker, _active_p75)
                        if window is None:
                            continue
                    r = detect_and_simulate(
                        ticker, buckets,
                        **best_entry,
                        stall_ticks=stall,
                        pt_frac=pt,
                        sl_frac=sl,
                        entry_window=window,
                    )
                    if r is not None:
                        trades_s.append(r)
                if len(trades_s) >= 3:
                    s = stats(trades_s)
                    stall_label = str(stall) if stall < 100 else "∞(no bail)"
                    print(f"  stall={stall_label:10s} PT={pt:.0%} SL={sl:.0%} | {fmt(s)}", flush=True)

    return best_cfg


# Module-level placeholders so _print_sweep's stall sensitivity section can
# reference the current context's market_buckets and p75 filter.
_active_market_buckets: dict[str, list[tuple[datetime, int]]] = {}
_active_p75: dict | None = None


async def main() -> None:
    global _active_market_buckets, _active_p75

    # -----------------------------------------------------------------------
    # Load or fetch market data
    # -----------------------------------------------------------------------
    if CACHE_FILE.exists():
        print(f"Loading cached data from {CACHE_FILE}…", flush=True)
        market_buckets = _load_cache()
        print(f"  Loaded {len(market_buckets)} markets from cache\n", flush=True)
    else:
        sem = asyncio.Semaphore(MAX_RATE)
        completed = 0

        async def guarded_fetch_history(session, ticker, total):
            nonlocal completed
            async with sem:
                fills = await fetch_trade_history(session, ticker)
                await asyncio.sleep(0.1)
            completed += 1
            if completed % 50 == 0 or completed == total:
                print(f"  {completed}/{total} trade histories fetched…", flush=True)
            return ticker, fills

        async with aiohttp.ClientSession() as session:
            print("Fetching settled B-market tickers…", flush=True)
            tickers = await fetch_settled_b_tickers(session)
            print(f"  Found {len(tickers)} settled B-market tickers\n", flush=True)

            now_date = datetime.now(timezone.utc).date()
            cutoff = now_date - timedelta(days=LOOKBACK_DAYS)

            recent = [t for t in tickers if (d := _parse_ticker_date(t)) and d >= cutoff]
            print(f"  Filtered to last {LOOKBACK_DAYS} days: {len(recent)} tickers\n", flush=True)

            print(f"Fetching trade histories (concurrency={MAX_RATE})…", flush=True)
            t0 = time.monotonic()
            tasks = [guarded_fetch_history(session, t, len(recent)) for t in recent]
            raw = await asyncio.gather(*tasks, return_exceptions=True)
            elapsed = time.monotonic() - t0
            print(f"  Done in {elapsed:.1f}s\n", flush=True)

        market_buckets = {}
        for item in raw:
            if isinstance(item, Exception):
                continue
            ticker, fills = item
            if len(fills) < 5:
                continue
            # Restrict to the settlement date only — avoids binning weeks of
            # pre-market fills which inflates bucket count to 13K+.
            tdate = _parse_ticker_date(ticker)
            if tdate is not None:
                fills = [
                    (ts, p) for ts, p in fills
                    if ts.date() == tdate
                ]
            if len(fills) < 5:
                continue
            bkts = bin_fills(fills)
            if bkts:
                market_buckets[ticker] = bkts

        _save_cache(market_buckets)

    print(f"Markets with usable price series: {len(market_buckets)}\n", flush=True)

    # -----------------------------------------------------------------------
    # Entry price distribution (broad look, baseline<55, jump≥10¢, land≥60¢)
    # -----------------------------------------------------------------------
    all_jumps: list[int] = []
    for ticker, bkts in market_buckets.items():
        for i in range(3, len(bkts)):
            price = bkts[i][1]
            prev  = bkts[i - 1][1]
            recent_max = max(bkts[i-3][1], bkts[i-2][1], bkts[i-1][1])
            if recent_max < 55 and price >= 60 and (price - prev) >= 10:
                all_jumps.append(price)
                break

    if all_jumps:
        all_jumps.sort()
        print("Entry price distribution across markets (baseline<55, jump≥10¢, land≥60¢):", flush=True)
        for lo, hi in [(55,65), (65,75), (75,85), (85,95)]:
            n = sum(1 for p in all_jumps if lo <= p < hi)
            print(f"  {lo}-{hi}¢: {n} markets", flush=True)
        print(f"  Total markets with any signal: {len(all_jumps)}\n", flush=True)

    # -----------------------------------------------------------------------
    # Parameter grid
    # -----------------------------------------------------------------------
    param_grid = {
        "baseline_max":   [45, 50, 55],
        "jump_threshold": [60, 65, 70, 75],
        "min_jump":       [15, 20, 25],
        "stall_ticks":    [1, 2, 3],
        "pt_frac":        [0.15, 0.20, 0.30],
        "sl_frac":        [0.70, 0.80],
    }

    # -----------------------------------------------------------------------
    # Sweep 1: ALL TIME (no window filter)
    # -----------------------------------------------------------------------
    _active_market_buckets = market_buckets
    _active_p75 = None
    print("\nStarting sweep 1: ALL TIME…", flush=True)
    sweep_all = _run_sweep(market_buckets, param_grid, p75_data=None, label="ALL TIME")
    _print_sweep(sweep_all, "ALL TIME")

    # -----------------------------------------------------------------------
    # Sweep 2: ±2h P75 WINDOW ONLY
    # -----------------------------------------------------------------------
    p75_data = _load_p75()
    if not p75_data:
        print("\n[warn] Could not load P75_MINUTES from data/peak_hour_p90.py — skipping p75 sweep.\n", flush=True)
    else:
        n_with_p75 = sum(1 for t in market_buckets if _p75_window(t, p75_data) is not None)
        print(f"\nP75 data loaded for {n_with_p75}/{len(market_buckets)} tickers.", flush=True)
        print("Starting sweep 2: P75 ±2h WINDOW…", flush=True)
        _active_p75 = p75_data
        sweep_p75 = _run_sweep(market_buckets, param_grid, p75_data=p75_data, label="P75 ±2h")
        _print_sweep(sweep_p75, "P75 ±2h WINDOW")


if __name__ == "__main__":
    asyncio.run(main())
