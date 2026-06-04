"""
Backtest: sustained heat-wave carryover signal for next-day KXHIGH B-band YES entry.

Strategy:
  - Evening (20-23 UTC = ~4-7pm ET): today's confirmed METAR max rounds to band X
    AND tomorrow's HRRR forecast also rounds to band X
  - Enter YES on the next-day B-band market if ask is in [10, 55]¢
  - Exit at 70¢ YES bid profit-take OR hold to settlement

This differs from forecast_band_yes (same-day morning HRRR signal) by:
  - Entry window: evening vs morning
  - Signal: observed today + forecasted tomorrow vs same-day morning forecast
  - Hypothesis: next-day markets underprice heat-wave persistence at open

Data:
  - data/candlesticks.db                        — 1-min Kalshi candles (Apr 6 – May 8)
  - data/backtest/band_arb_hist_cache.json       — HRRR + actual daily highs
  - data/mesonet_hourly_combined.csv             — running daily max by city/local-hour

Run:
  venv/bin/python scripts/backtest_forecast_band_yes_carryover.py
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

CANDLE_DB  = Path("data/candlesticks.db")
HIST_CACHE = Path("data/backtest/band_arb_hist_cache.json")
MESONET    = Path("data/mesonet_hourly_combined.csv")

# Entry window: evening UTC (after local afternoon confirmation)
ENTRY_HOURS_UTC = list(range(20, 23))   # 20, 21, 22 UTC
ENTRY_MIN_ASK   = 10
ENTRY_MAX_ASK   = 55
BEST_PT         = 70
PT_TARGETS      = [50, 60, 65, 70, 75, 80]

# Local hour at which today's METAR max is "confirmed" (99.4% within 1°F of final)
METAR_CONFIRM_HOUR = 17

_TICKER_RE = re.compile(r"^([A-Z]+)-(\d{2})([A-Z]{3})(\d{2})-B([\d.]+)$")
_MON = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
        "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}

_SERIES_TO_CITY: dict[str, str] = {
    "KXHIGHLAX":   "lax", "KXHIGHDEN":   "den", "KXHIGHCHI":   "chi",
    "KXHIGHNY":    "ny",  "KXHIGHMIA":   "mia", "KXHIGHDAL":   "dal",
    "KXHIGHBOS":   "bos", "KXHIGHAUS":   "aus", "KXHIGHOU":    "hou",
    "KXHIGHTSFO":  "sfo", "KXHIGHTSEA":  "sea", "KXHIGHTBOS":  "bos",
    "KXHIGHTPHX":  "phx", "KXHIGHTPHIL": "phl", "KXHIGHTDC":   "dca",
    "KXHIGHTLV":   "las", "KXHIGHTOKC":  "okc", "KXHIGHTDAL":  "dfw",
    "KXHIGHTHOU":  "hou", "KXHIGHTNOLA": "msy", "KXHIGHTATL":  "atl",
    "KXHIGHTMIN":  "msp", "KXHIGHTDFW":  "dfw", "KXHIGHTSATX": "sat",
}


def parse_ticker(ticker: str):
    m = _TICKER_RE.match(ticker)
    if not m:
        return None
    series, yy, mon, dd, mid = m.groups()
    city = _SERIES_TO_CITY.get(series)
    if not city:
        return None
    try:
        settle_date = date(2000 + int(yy), _MON[mon], int(dd))
    except (KeyError, ValueError):
        return None
    mid_f = float(mid)
    band_lo = int(mid_f - 0.5)
    band_hi = band_lo + 1
    return settle_date, city, band_lo, band_hi


def load_hrrr_cache() -> dict[str, dict[str, float]]:
    """Return {city: {date_str: forecast_high_f}}."""
    raw = json.loads(HIST_CACHE.read_text())
    out: dict[str, dict[str, float]] = {}
    for key, val in raw.items():
        if not key.startswith("hrrr_temp_high_"):
            continue
        # key = hrrr_temp_high_{city}_2022-01-01_2026-05-31_high
        city = key.replace("hrrr_temp_high_", "").split("_")[0]
        if isinstance(val, dict):
            out[city] = {k: float(v) for k, v in val.items() if v is not None}
    return out


def load_mesonet_max(confirm_hour: int = METAR_CONFIRM_HOUR) -> dict[tuple[str, str], float]:
    """Return {(city, date_str): running_max_f at confirm_hour}."""
    out: dict[tuple[str, str], float] = {}
    with open(MESONET) as f:
        for row in csv.DictReader(f):
            if "high" not in row["city_metric"]:
                continue
            if int(row["local_hour"]) != confirm_hour:
                continue
            city = row["city_metric"].replace("temp_high_", "")
            if row["running_max_f"]:
                out[(city, row["date"])] = float(row["running_max_f"])
    return out


def load_candles_by_ticker() -> dict[str, list[tuple[int, int, int]]]:
    """Return {ticker: [(period_ts, bid_high, ask_open)]} sorted by ts."""
    conn = sqlite3.connect(CANDLE_DB)
    cur = conn.cursor()
    cur.execute(
        "SELECT ticker, period_ts, bid_high, ask_open "
        "FROM candles WHERE ticker LIKE 'KXHIGH%B%' ORDER BY ticker, period_ts"
    )
    rows = cur.fetchall()
    conn.close()
    out: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for ticker, ts, bid_high, ask_open in rows:
        out[ticker].append((ts, bid_high or 0, ask_open or 0))
    return out


def find_entry_ask(candles: list[tuple[int, int, int]], entry_date: date) -> int | None:
    """First ask_open in ENTRY_HOURS_UTC on entry_date that is in [MIN,MAX] range."""
    for h in ENTRY_HOURS_UTC:
        window_start = int(datetime(entry_date.year, entry_date.month, entry_date.day,
                                    h, 0, tzinfo=timezone.utc).timestamp())
        window_end   = window_start + 3600
        for ts, _, ask in candles:
            if window_start <= ts < window_end and ENTRY_MIN_ASK <= ask <= ENTRY_MAX_ASK:
                return ask
    return None


def find_first_entry_ts(candles: list[tuple[int, int, int]], entry_date: date) -> int | None:
    """Timestamp of the first qualifying entry candle on entry_date."""
    for h in ENTRY_HOURS_UTC:
        window_start = int(datetime(entry_date.year, entry_date.month, entry_date.day,
                                    h, 0, tzinfo=timezone.utc).timestamp())
        window_end   = window_start + 3600
        for ts, _, ask in candles:
            if window_start <= ts < window_end and ENTRY_MIN_ASK <= ask <= ENTRY_MAX_ASK:
                return ts
    return None


def max_bid_after(candles: list[tuple[int, int, int]], entry_ts: int) -> int:
    """Highest bid_high in candles strictly after entry_ts."""
    return max((c[1] for c in candles if c[0] > entry_ts), default=0)


def settlement_price(ticker: str, conn: sqlite3.Connection) -> int | None:
    """Returns 100 for YES, 0 for NO, None if unknown."""
    cur = conn.cursor()
    cur.execute("SELECT result FROM markets WHERE ticker=?", (ticker,))
    row = cur.fetchone()
    if row is None:
        return None
    return 100 if row[0] == "yes" else 0


# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading data...")
    hrrr      = load_hrrr_cache()
    mesonet   = load_mesonet_max()
    candles   = load_candles_by_ticker()
    conn      = sqlite3.connect(CANDLE_DB)

    # ── signal sweep ──────────────────────────────────────────────────────────
    # mode A: METAR only
    # mode B: METAR + HRRR agreement (both must round to same band)
    results_a: list[dict] = []  # METAR only
    results_b: list[dict] = []  # METAR + HRRR

    tickers = list(candles.keys())
    print(f"Scanning {len(tickers)} B-band tickers...")

    skipped_no_parse     = 0
    skipped_no_metar     = 0
    skipped_no_hrrr      = 0
    skipped_no_entry_ask = 0
    metar_only_signals   = 0

    for ticker in tickers:
        parsed = parse_ticker(ticker)
        if parsed is None:
            skipped_no_parse += 1
            continue
        settle_date, city, band_lo, band_hi = parsed

        # Entry is on the PRIOR day
        entry_date = settle_date - timedelta(days=1)
        entry_date_str = str(entry_date)
        settle_date_str = str(settle_date)

        # ── Gate 1: METAR confirmed max on entry_date ──────────────────────
        metar_max = mesonet.get((city, entry_date_str))
        if metar_max is None:
            skipped_no_metar += 1
            continue
        metar_band = round(metar_max)  # rounded to nearest °F
        metar_match = (band_lo <= metar_band <= band_hi)

        if not metar_match:
            continue
        metar_only_signals += 1

        # ── Gate 2: HRRR forecast for settle_date (D+1) ───────────────────
        hrrr_city = hrrr.get(city, {})
        hrrr_fc = hrrr_city.get(settle_date_str)
        hrrr_match = False
        if hrrr_fc is not None:
            hrrr_band = round(hrrr_fc)
            hrrr_match = (band_lo <= hrrr_band <= band_hi)

        # ── Entry candle ──────────────────────────────────────────────────
        tkr_candles = candles[ticker]
        entry_ask = find_entry_ask(tkr_candles, entry_date)
        if entry_ask is None:
            skipped_no_entry_ask += 1
            continue

        entry_ts = find_first_entry_ts(tkr_candles, entry_date)
        settle   = settlement_price(ticker, conn)
        peak_bid = max_bid_after(tkr_candles, entry_ts) if entry_ts else 0

        record = {
            "ticker":      ticker,
            "settle_date": settle_date_str,
            "entry_date":  entry_date_str,
            "city":        city,
            "band_lo":     band_lo,
            "band_hi":     band_hi,
            "metar_max":   metar_max,
            "hrrr_fc":     hrrr_fc,
            "entry_ask":   entry_ask,
            "peak_bid":    peak_bid,
            "settle":      settle,
        }

        results_a.append(record)
        if hrrr_match:
            results_b.append(record)

    conn.close()

    # ── Reporting ─────────────────────────────────────────────────────────────
    print(f"\nSkipped — no parse:      {skipped_no_parse}")
    print(f"Skipped — no METAR:      {skipped_no_metar}")
    print(f"Skipped — no entry ask:  {skipped_no_entry_ask}")

    def report(label: str, records: list[dict]) -> None:
        if not records:
            print(f"\n{label}: no trades")
            return
        print(f"\n{'='*60}")
        print(f"{label}  ({len(records)} trades)")
        print(f"{'='*60}")

        # PT sweep
        print("\n── Profit-take sweep ──")
        for pt in PT_TARGETS:
            hits   = sum(1 for r in records if r["peak_bid"] >= pt)
            pct    = hits / len(records) * 100
            avg_pnl = sum(
                (pt - r["entry_ask"] if r["peak_bid"] >= pt else (r["settle"] or 0) - r["entry_ask"])
                for r in records
            ) / len(records)
            print(f"  PT={pt}¢  hit={hits:3d}/{len(records)} ({pct:5.1f}%)  avg PnL={avg_pnl:+.1f}¢")

        # Settlement stats
        settled = [r for r in records if r["settle"] is not None]
        if settled:
            yes_cnt = sum(1 for r in settled if r["settle"] == 100)
            print(f"\nSettlement YES rate: {yes_cnt}/{len(settled)} ({yes_cnt/len(settled)*100:.1f}%)")

        # Entry ask distribution
        asks = [r["entry_ask"] for r in records]
        print(f"Entry ask: min={min(asks)}¢  avg={sum(asks)/len(asks):.1f}¢  max={max(asks)}¢")

        # Best PT breakdown
        pt = BEST_PT
        wins   = [r for r in records if r["peak_bid"] >= pt]
        losses = [r for r in records if r["peak_bid"] < pt]
        win_pnl = sum(pt - r["entry_ask"] for r in wins)
        # Losses exit at settlement (or 0 if unknown)
        loss_pnl = sum((r["settle"] or 0) - r["entry_ask"] for r in losses)
        total_pnl_c = win_pnl + loss_pnl
        print(f"\nAt PT={pt}¢:")
        print(f"  Wins:   {len(wins)} trades,  PnL = +{win_pnl/100:.2f}")
        print(f"  Losses: {len(losses)} trades, PnL = {loss_pnl/100:+.2f}")
        print(f"  Total:  {total_pnl_c/100:+.2f}  ({total_pnl_c/len(records)/100*100:+.1f}¢ avg)")

        # Per-city breakdown at best PT
        print(f"\n── Per-city at PT={pt}¢ ──")
        city_records: dict[str, list] = defaultdict(list)
        for r in records:
            city_records[r["city"]].append(r)
        for city, recs in sorted(city_records.items()):
            hits = sum(1 for r in recs if r["peak_bid"] >= pt)
            avg_ask = sum(r["entry_ask"] for r in recs) / len(recs)
            print(f"  {city:4s}  {hits:2d}/{len(recs):2d}  ({hits/len(recs)*100:5.1f}%)  avg ask={avg_ask:.1f}¢")

        # Compare entry ask: does carryover get cheaper prices than morning?
        print(f"\nHeat-wave context (METAR ≥85°F):")
        hot = [r for r in records if r["metar_max"] >= 85]
        cool = [r for r in records if r["metar_max"] < 85]
        if hot:
            hot_pt = sum(1 for r in hot if r["peak_bid"] >= pt)
            print(f"  Hot (≥85°F): {hot_pt}/{len(hot)} ({hot_pt/len(hot)*100:.1f}%)  avg ask={sum(r['entry_ask'] for r in hot)/len(hot):.1f}¢")
        if cool:
            cool_pt = sum(1 for r in cool if r["peak_bid"] >= pt)
            print(f"  Cool (<85°F): {cool_pt}/{len(cool)} ({cool_pt/len(cool)*100:.1f}%)  avg ask={sum(r['entry_ask'] for r in cool)/len(cool):.1f}¢")

    report("MODE A — METAR carryover only (no HRRR gate)", results_a)
    report("MODE B — METAR + HRRR agreement", results_b)

    # Compare ask levels vs same-day morning signal
    print("\n── Entry ask comparison (evening carryover vs morning HRRR) ──")
    if results_b:
        asks_b = [r["entry_ask"] for r in results_b]
        print(f"  Evening entry (carryover):  avg ask={sum(asks_b)/len(asks_b):.1f}¢  n={len(asks_b)}")
    print("  Morning entry (forecast_band_yes): avg ask≈32.8¢  n=294  (from prior backtest)")


if __name__ == "__main__":
    main()
