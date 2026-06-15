"""Backtest: ASOS synoptic-center NO path for KXHIGH between markets.

Question: if we use the integer-°C center value (C × 1.8 + 32) instead of the
conservative lower bound ((C − 0.5) × 1.8 + 32) to trigger band_arb NO, how
often does ASOS fire BEFORE the :53 METAR confirms, and is it worth it?

Current path (conservative): fires when (C−0.5)×1.8+32 ≥ ceiling+0.5
                              i.e., ASOS center ≥ ceiling + 1.4°F
Proposed path (center):      fires when C×1.8+32 ≥ ceiling+0.5
                              i.e., ASOS center ≥ ceiling + 0.5°F  (0.9°F more lenient)

Exit rule for false positives: if the next :53 METAR after entry shows
temp < ceiling + 0.5°F, force-exit at market price.

Data window: May 12–20 (ASOS available from May 12).

Usage:
  venv/bin/python scripts/backtest_synoptic_no.py
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import NamedTuple

import pandas as pd

DB_PATH    = Path("data/db/opportunity_log.db")
CACHE_FILE = Path("data/forecast_no_hist_cache.json")

# ── Ticker parsing ────────────────────────────────────────────────────────────

_B_RE = re.compile(r"KXHIGHT\w+-\d{2}[A-Z]{3}\d{2}-B([\d.]+)$")

def _parse_strike_hi(ticker: str) -> float | None:
    """Return strike_hi for a KXHIGHT between-market ticker, or None."""
    m = _B_RE.match(ticker)
    if not m:
        return None
    return float(m.group(1)) + 0.5   # B66.5 → ceiling = 67.0°F

def _is_integer_celsius(val_f: float) -> bool:
    c = (val_f - 32.0) * 5.0 / 9.0
    return abs(c - round(c)) < 0.05

# ── Load raw_forecasts ────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (asos_df, metar_df) for KXHIGH between markets."""
    conn = sqlite3.connect(DB_PATH)

    print("Loading ASOS integer-°C rows…")
    asos_df = pd.read_sql_query(
        """
        SELECT ticker, metric, data_value, logged_at
        FROM raw_forecasts
        WHERE source = 'nws_asos'
          AND metric LIKE 'temp_high%'
          AND ticker LIKE 'KXHIGHT%'
          AND direction = 'between'
          AND logged_at >= '2026-05-12'
        """,
        conn,
    )
    asos_df["logged_at"] = pd.to_datetime(asos_df["logged_at"], utc=True)
    # Keep only integer-°C readings (synoptic 5-min)
    asos_df = asos_df[asos_df["data_value"].apply(_is_integer_celsius)].copy()
    print(f"  ASOS integer-°C rows: {len(asos_df):,}")

    print("Loading METAR rows…")
    metar_df = pd.read_sql_query(
        """
        SELECT ticker, metric, data_value, logged_at
        FROM raw_forecasts
        WHERE source = 'metar'
          AND metric LIKE 'temp_high%'
          AND logged_at >= '2026-05-12'
        """,
        conn,
    )
    metar_df["logged_at"] = pd.to_datetime(metar_df["logged_at"], utc=True)
    print(f"  METAR rows: {len(metar_df):,}")

    conn.close()
    return asos_df, metar_df

# ── Load fill cache ───────────────────────────────────────────────────────────

def load_fill_cache() -> dict:
    if not CACHE_FILE.exists():
        print("  WARNING: fill cache not found — pricing data unavailable")
        return {}
    with open(CACHE_FILE) as f:
        raw = json.load(f)
    result = {}
    for ticker, d in raw.items():
        result[ticker] = {
            "fills": [(datetime.fromisoformat(ts), price)
                      for ts, price in d.get("fills", [])],
            "settlement": d.get("settlement"),
        }
    return result

def _no_ask_at(fills: list, ts: datetime) -> int | None:
    """Return 100 - yes_bid at the fill closest to (and before) ts."""
    best_price = None
    for fill_ts, price in fills:
        if fill_ts <= ts:
            best_price = price
        else:
            break
    if best_price is None:
        return None
    return 100 - best_price   # yes_bid → NO ask

# ── Per-ticker analysis ───────────────────────────────────────────────────────

class CrossingEvent(NamedTuple):
    ticker: str
    metric: str
    local_date: str
    strike_hi: float
    # ASOS center crossing
    asos_center_time:  datetime | None
    asos_center_val:   float | None
    # Current conservative ASOS crossing (center ≥ ceiling + 1.4)
    asos_conserv_time: datetime | None
    # METAR crossing
    metar_time:  datetime | None
    metar_val:   float | None
    # Market data
    settlement:  str | None   # 'yes' | 'no'
    no_ask_at_asos:   int | None
    no_ask_at_metar:  int | None
    no_ask_at_conserv: int | None

def analyze(asos_df: pd.DataFrame, metar_df: pd.DataFrame,
            fill_cache: dict) -> list[CrossingEvent]:

    # Build METAR lookup: metric → sorted list of (logged_at, data_value)
    metar_by_metric: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
    for _, row in metar_df.iterrows():
        metar_by_metric[row["metric"]].append((row["logged_at"], row["data_value"]))
    for k in metar_by_metric:
        metar_by_metric[k].sort()

    # Group ASOS by (ticker, date)
    asos_df["utc_date"] = asos_df["logged_at"].dt.date
    groups = asos_df.groupby(["ticker", "utc_date"])

    events: list[CrossingEvent] = []
    skipped_no_strike = 0

    for (ticker, utc_date), grp in groups:
        strike_hi = _parse_strike_hi(ticker)
        if strike_hi is None:
            skipped_no_strike += 1
            continue

        metric = grp["metric"].iloc[0]
        grp_sorted = grp.sort_values("logged_at")

        threshold = strike_hi + 0.5        # same threshold METAR uses
        conserv_threshold = strike_hi + 1.4  # lower bound clears ceiling

        # First ASOS center crossing (proposed path)
        asos_cross = grp_sorted[grp_sorted["data_value"] >= threshold]
        asos_center_time = asos_center_val = None
        if not asos_cross.empty:
            first = asos_cross.iloc[0]
            asos_center_time = first["logged_at"]
            asos_center_val  = first["data_value"]

        # First ASOS conservative crossing (current path)
        asos_cross_c = grp_sorted[grp_sorted["data_value"] >= conserv_threshold]
        asos_conserv_time = None
        if not asos_cross_c.empty:
            asos_conserv_time = asos_cross_c.iloc[0]["logged_at"]

        # First METAR crossing for this metric on this date
        date_start = datetime(utc_date.year, utc_date.month, utc_date.day,
                              tzinfo=timezone.utc)
        date_end   = date_start + timedelta(days=1)
        metar_obs  = [
            (t, v) for t, v in metar_by_metric.get(metric, [])
            if date_start <= t < date_end
        ]
        metar_cross = [(t, v) for t, v in metar_obs if v >= threshold]
        metar_time = metar_val = None
        if metar_cross:
            metar_time, metar_val = metar_cross[0]

        # Fill cache
        cache = fill_cache.get(ticker, {})
        fills = cache.get("fills", [])
        settlement = cache.get("settlement")

        no_ask_at_asos   = _no_ask_at(fills, asos_center_time)  if asos_center_time  else None
        no_ask_at_metar  = _no_ask_at(fills, metar_time)         if metar_time         else None
        no_ask_at_conserv = _no_ask_at(fills, asos_conserv_time) if asos_conserv_time else None

        events.append(CrossingEvent(
            ticker=ticker,
            metric=metric,
            local_date=str(utc_date),
            strike_hi=strike_hi,
            asos_center_time=asos_center_time,
            asos_center_val=asos_center_val,
            asos_conserv_time=asos_conserv_time,
            metar_time=metar_time,
            metar_val=metar_val,
            settlement=settlement,
            no_ask_at_asos=no_ask_at_asos,
            no_ask_at_metar=no_ask_at_metar,
            no_ask_at_conserv=no_ask_at_conserv,
        ))

    print(f"  Tickers analyzed: {len(events):,}  (skipped {skipped_no_strike} non-B tickers)")
    return events

# ── Report ────────────────────────────────────────────────────────────────────

def report(events: list[CrossingEvent]) -> None:
    total = len(events)

    # Classify events
    asos_first_metar_confirms  = []   # proposed path fires early, METAR later confirms
    asos_first_metar_contradicts = [] # proposed path fires early, METAR never confirms
    asos_only                  = []   # ASOS crosses but METAR absent (can't classify)
    metar_first                = []   # standard path (METAR fires before or same as ASOS center)
    no_crossing                = []   # neither crosses
    conserv_early              = []   # current conservative ASOS fires before METAR
    conserv_only               = []   # current conservative ASOS fires, no METAR crossing

    for e in events:
        has_asos   = e.asos_center_time  is not None
        has_metar  = e.metar_time        is not None
        has_conserv = e.asos_conserv_time is not None

        if not has_asos and not has_metar:
            no_crossing.append(e)
        elif has_asos and not has_metar:
            asos_first_metar_contradicts.append(e)  # METAR never confirmed
        elif not has_asos and has_metar:
            metar_first.append(e)
        elif e.asos_center_time < e.metar_time:
            asos_first_metar_confirms.append(e)
        else:
            metar_first.append(e)   # METAR same time or earlier

        # Conservative path
        if has_conserv and not has_metar:
            conserv_only.append(e)
        elif has_conserv and has_metar and e.asos_conserv_time < e.metar_time:
            conserv_early.append(e)

    print("\n" + "=" * 65)
    print("SYNOPTIC-CENTER NO BACKTEST — KXHIGHT Between Markets (May 12–20)")
    print("=" * 65)
    print(f"\nTotal (ticker, date) pairs analyzed: {total:,}")
    print(f"  No crossing at all (both stayed below ceiling):  {len(no_crossing):,}")
    print(f"  METAR confirms first (standard path):            {len(metar_first):,}")
    print()
    print("── PROPOSED CENTER PATH (ASOS center ≥ ceiling + 0.5) ──────────")
    print(f"  ASOS fires BEFORE METAR AND METAR confirms:      {len(asos_first_metar_confirms):,}")
    print(f"  ASOS fires BUT METAR never confirms (false +ve): {len(asos_first_metar_contradicts):,}")
    fp_rate = (len(asos_first_metar_contradicts) /
               max(1, len(asos_first_metar_confirms) + len(asos_first_metar_contradicts)))
    print(f"  False positive rate: {fp_rate:.1%}")
    print()
    print("── CURRENT CONSERVATIVE PATH (ASOS center ≥ ceiling + 1.4) ─────")
    print(f"  Conservative fires BEFORE METAR AND confirms:    {len(conserv_early):,}")
    print(f"  Conservative fires BUT METAR never confirms:      {len(conserv_only):,}")

    # Lead time for true early signals (proposed path)
    if asos_first_metar_confirms:
        lead_minutes = [
            (e.metar_time - e.asos_center_time).total_seconds() / 60
            for e in asos_first_metar_confirms
        ]
        lead_minutes.sort()
        n = len(lead_minutes)
        print(f"\n── LEAD TIME for center-path true early signals ({n} events) ──")
        print(f"  Median:  {lead_minutes[n//2]:.1f} min")
        print(f"  p25:     {lead_minutes[n//4]:.1f} min")
        print(f"  p75:     {lead_minutes[3*n//4]:.1f} min")
        print(f"  Max:     {lead_minutes[-1]:.1f} min")

        # Market pricing gap
        pairs_with_prices = [
            e for e in asos_first_metar_confirms
            if e.no_ask_at_asos is not None and e.no_ask_at_metar is not None
        ]
        if pairs_with_prices:
            ask_at_entry   = [e.no_ask_at_asos  for e in pairs_with_prices]
            ask_at_confirm = [e.no_ask_at_metar for e in pairs_with_prices]
            print(f"\n── MARKET PRICING during early window ({len(pairs_with_prices)} events with fills) ──")
            print(f"  NO ask at ASOS entry  — median: {sorted(ask_at_entry)[len(ask_at_entry)//2]}¢"
                  f"  mean: {sum(ask_at_entry)/len(ask_at_entry):.1f}¢")
            print(f"  NO ask at METAR conf  — median: {sorted(ask_at_confirm)[len(ask_at_confirm)//2]}¢"
                  f"  mean: {sum(ask_at_confirm)/len(ask_at_confirm):.1f}¢")
            price_diff = [a - b for a, b in zip(ask_at_entry, ask_at_confirm)]
            print(f"  Price improvement (entry cheaper): "
                  f"median {sorted(price_diff)[len(price_diff)//2]:+d}¢  "
                  f"mean {sum(price_diff)/len(price_diff):+.1f}¢")

            # Settlement breakdown
            won  = sum(1 for e in pairs_with_prices if e.settlement == "no")
            lost = sum(1 for e in pairs_with_prices if e.settlement == "yes")
            print(f"\n── SETTLEMENT of early signals ─────────────────────────────────")
            print(f"  Won (settled NO): {won}   Lost (settled YES): {lost}"
                  f"  WR: {100*won/max(1,won+lost):.1f}%")

    # False positive economics
    if asos_first_metar_contradicts:
        fp_with_asks = [e for e in asos_first_metar_contradicts if e.no_ask_at_asos is not None]
        print(f"\n── FALSE POSITIVE ECONOMICS ({len(asos_first_metar_contradicts)} events) ──")
        if fp_with_asks:
            asks = sorted(e.no_ask_at_asos for e in fp_with_asks)
            print(f"  NO ask at ASOS entry — median: {asks[len(asks)//2]}¢  mean: {sum(asks)/len(asks):.1f}¢")
        # These are cases where METAR never crossed → market likely stays expensive for YES
        # Estimate exit loss: if forced out at 100 - (high YES bid)
        fp_settled = [(e.settlement, e.no_ask_at_asos) for e in asos_first_metar_contradicts]
        won_fp  = sum(1 for s, _ in fp_settled if s == "no")
        lost_fp = sum(1 for s, _ in fp_settled if s == "yes")
        print(f"  Settlement: won={won_fp} lost={lost_fp} (if held to settlement)")

    # Per-city breakdown of early signals
    if asos_first_metar_confirms:
        from collections import Counter
        city_re = re.compile(r"KXHIGHT([A-Z]+)-")
        city_counts = Counter()
        for e in asos_first_metar_confirms:
            m = city_re.match(e.ticker)
            if m:
                city_counts[m.group(1)] += 1
        print(f"\n── CITY BREAKDOWN (center-path early signals) ──────────────────")
        for city, cnt in city_counts.most_common():
            print(f"  {city:<8} {cnt}")

    # Example events
    print(f"\n── SAMPLE TRUE EARLY EVENTS ────────────────────────────────────")
    print(f"  {'ticker':<32} {'ceil':>5} {'asos_val':>8} {'lead_min':>8} {'no_ask_entry':>12} {'outcome'}")
    for e in sorted(asos_first_metar_confirms,
                    key=lambda x: (x.metar_time - x.asos_center_time).total_seconds(),
                    reverse=True)[:15]:
        lead = (e.metar_time - e.asos_center_time).total_seconds() / 60
        print(f"  {e.ticker:<32} {e.strike_hi:>5.1f} {e.asos_center_val:>8.1f}"
              f" {lead:>8.1f}  {str(e.no_ask_at_asos):>12}  {e.settlement or '?'}")

    if asos_first_metar_contradicts:
        print(f"\n── SAMPLE FALSE POSITIVE EVENTS ───────────────────────────────")
        print(f"  {'ticker':<32} {'ceil':>5} {'asos_val':>8} {'no_ask':>6} {'settled'}")
        for e in asos_first_metar_contradicts[:10]:
            print(f"  {e.ticker:<32} {e.strike_hi:>5.1f} {e.asos_center_val:>8.1f}"
                  f"  {str(e.no_ask_at_asos):>6}  {e.settlement or '?'}")


if __name__ == "__main__":
    asos_df, metar_df = load_data()
    fill_cache = load_fill_cache()
    print(f"  Fill cache tickers loaded: {len(fill_cache):,}")

    events = analyze(asos_df, metar_df, fill_cache)
    report(events)
