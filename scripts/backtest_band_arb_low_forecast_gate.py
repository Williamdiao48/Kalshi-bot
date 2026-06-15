#!/usr/bin/env python3
"""Backtest forecast clearance + METAR trajectory filters for band_arb KXLOWT NO trades.

For each band_arb NO trade in the DB we reconstruct what the forecasts were
saying at entry time and whether the METAR running minimum was still falling.
We then sweep thresholds to find which filters would have improved win rate.

Two filters:
  1. Forecast clearance gate  — at entry, all (or N-of-M) forecast models must
     show their predicted low is >= MIN_CLEARANCE °F ABOVE the band ceiling.
     If Open-Meteo says the low will be 56.2°F and the band ceiling is 56.5°F,
     clearance = -0.3°F → we'd skip this trade.

  2. METAR trajectory gate — the METAR running daily minimum must have fallen
     by >= MIN_DROP °F over the 2 hours before entry.  A flat trajectory (no
     cooling) means the overnight low may not reach the required level.

Band ceiling definition
-----------------------
For a KXLOWT B{X}.5 market the effective YES zone (after NWS rounding) is
[X-0.5, X+1.5) → the band ceiling is X + 1.5 and the band floor is X - 0.5.
NO wins (above-band) when the observed low >= X + 1.5.
So: above-band clearance = forecast_low - (band_midpoint + 1.0)
    where band_midpoint = X + 0.5 (e.g. B55.5 → midpoint=55.5, ceiling=56.5)

Usage
-----
  venv/bin/python scripts/backtest_band_arb_low_forecast_gate.py
"""

from __future__ import annotations

import re
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

DB = Path("data/db/opportunity_log.db")

# Sources used for forecast clearance check
FORECAST_SOURCES = ("nws_hourly", "open_meteo", "hrrr")

# How many minutes before entry to look back for forecasts / METAR
FORECAST_LOOKBACK_MIN = 90
METAR_SLOPE_WINDOW_MIN = 120  # slope window for running-min trajectory

# Sweep parameters
CLEARANCE_THRESHOLDS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]  # °F above ceiling
MIN_SOURCES_OPTIONS  = [1, 2, 3]   # how many forecast models must clear
METAR_DROP_OPTIONS   = [0.0, 0.5, 1.0, 2.0]  # °F running-min must drop in 2h


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_BAND_RE = re.compile(r"-B([\d.]+)$")   # matches -B55.5 at end of ticker


def _band_ceiling(ticker: str) -> float | None:
    """Return the effective band ceiling (°F) for an above-band NO trade."""
    m = _BAND_RE.search(ticker)
    if not m:
        return None
    midpoint = float(m.group(1))   # e.g. 55.5
    # YES zone (after NWS rounding): [midpoint-0.5, midpoint+1.5)
    # above-band NO wins when low >= midpoint + 1.5 ... but the market settles
    # at the official CLI obs rounded to whole °F, so effective ceiling = midpoint + 1.0
    return midpoint + 1.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    ticker: str
    entry_at: str         # ISO UTC
    won: bool
    ceiling: float        # band ceiling °F
    # filled in during enrichment:
    forecasts: dict[str, float] = field(default_factory=dict)  # source → forecast °F
    metar_at_entry: float | None = None
    metar_2h_before: float | None = None


def load_trades(con: sqlite3.Connection) -> list[Trade]:
    rows = con.execute("""
        SELECT ticker, logged_at,
               COALESCE(exit_pnl_cents, settled_pnl_cents) AS pnl
        FROM trades
        WHERE source = 'band_arb'
          AND side = 'no'
          AND ticker LIKE 'KXLOWT%B%'
          AND COALESCE(exit_pnl_cents, settled_pnl_cents) IS NOT NULL
        ORDER BY logged_at
    """).fetchall()

    trades = []
    for ticker, entry_at, pnl in rows:
        ceiling = _band_ceiling(ticker)
        if ceiling is None:
            continue
        trades.append(Trade(
            ticker=ticker,
            entry_at=entry_at,
            won=pnl > 0,
            ceiling=ceiling,
        ))
    return trades


def enrich_trades(con: sqlite3.Connection, trades: list[Trade]) -> None:
    """Attach forecast values and METAR slope to each trade."""
    for t in trades:
        entry_dt = datetime.fromisoformat(t.entry_at.replace("Z", "+00:00"))
        lookback_dt = entry_dt - timedelta(minutes=FORECAST_LOOKBACK_MIN)
        slope_dt    = entry_dt - timedelta(minutes=METAR_SLOPE_WINDOW_MIN)

        # raw_forecasts stores ISO timestamps with 'T' separator + timezone offset.
        # Use datetime() in queries to normalise both sides of the comparison.
        entry_str    = entry_dt.strftime("%Y-%m-%dT%H:%M:%S")
        lookback_str = lookback_dt.strftime("%Y-%m-%dT%H:%M:%S")
        slope_str    = slope_dt.strftime("%Y-%m-%dT%H:%M:%S")

        # --- Forecasts at entry time ---
        rows = con.execute("""
            SELECT source, AVG(data_value)
            FROM raw_forecasts
            WHERE ticker = ?
              AND source IN ('nws_hourly', 'open_meteo', 'hrrr')
              AND logged_at BETWEEN ? AND ?
            GROUP BY source
        """, (t.ticker, lookback_str, entry_str)).fetchall()

        for source, val in rows:
            if val is not None:
                t.forecasts[source] = float(val)

        # --- METAR running minimum at entry and 2h before ---
        row_now = con.execute("""
            SELECT AVG(data_value)
            FROM raw_forecasts
            WHERE ticker = ?
              AND source = 'metar'
              AND logged_at BETWEEN ? AND ?
        """, (t.ticker, lookback_str, entry_str)).fetchone()

        row_before = con.execute("""
            SELECT AVG(data_value)
            FROM raw_forecasts
            WHERE ticker = ?
              AND source = 'metar'
              AND logged_at BETWEEN ? AND ?
        """, (t.ticker, slope_str, lookback_str)).fetchone()

        if row_now and row_now[0] is not None:
            t.metar_at_entry = float(row_now[0])
        if row_before and row_before[0] is not None:
            t.metar_2h_before = float(row_before[0])


# ---------------------------------------------------------------------------
# Filter application
# ---------------------------------------------------------------------------

def passes_forecast_gate(t: Trade, min_clearance: float, min_sources: int) -> bool:
    """True if >= min_sources forecast models clear the ceiling by min_clearance."""
    if not t.forecasts:
        return True   # no data → don't filter (conservative)
    n_clear = sum(
        1 for fc in t.forecasts.values()
        if (fc - t.ceiling) >= min_clearance
    )
    return n_clear >= min(min_sources, len(t.forecasts))


def passes_metar_gate(t: Trade, min_drop: float) -> bool:
    """True if the METAR running min fell >= min_drop °F in the 2h before entry."""
    if min_drop <= 0:
        return True
    if t.metar_at_entry is None or t.metar_2h_before is None:
        return True   # no data → don't filter
    drop = t.metar_2h_before - t.metar_at_entry   # positive = cooled
    return drop >= min_drop


# ---------------------------------------------------------------------------
# Sweep + reporting
# ---------------------------------------------------------------------------

def run_sweep(trades: list[Trade]) -> None:
    n_total = len(trades)
    baseline_wr = sum(t.won for t in trades) / n_total if n_total else 0

    print(f"Base dataset: {n_total} band_arb KXLOWT NO trades  "
          f"(baseline win rate: {baseline_wr:.1%})\n")

    # --- Forecast gate sweep ---
    print("=== Forecast clearance gate sweep ===")
    print(f"{'min_clear':>10} {'min_src':>8} {'n':>5} {'kept%':>6} "
          f"{'win%':>6} {'delta':>7} {'wins':>5} {'losses':>7}")
    print("-" * 65)

    best_fcast = None
    for min_clearance in CLEARANCE_THRESHOLDS:
        for min_sources in MIN_SOURCES_OPTIONS:
            filtered = [t for t in trades
                        if passes_forecast_gate(t, min_clearance, min_sources)]
            n = len(filtered)
            if n == 0:
                continue
            wr = sum(t.won for t in filtered) / n
            kept_pct = n / n_total
            delta = wr - baseline_wr
            w = sum(t.won for t in filtered)
            l = n - w
            if best_fcast is None or wr > best_fcast[0]:
                best_fcast = (wr, min_clearance, min_sources, n)
            print(f"{min_clearance:>10.1f} {min_sources:>8} {n:>5} "
                  f"{kept_pct:>5.0%} {wr:>6.1%} {delta:>+7.1%} "
                  f"{w:>5} {l:>7}")

    # --- METAR trajectory gate sweep ---
    print("\n=== METAR trajectory gate sweep ===")
    print(f"{'min_drop_2h':>12} {'n':>5} {'kept%':>6} "
          f"{'win%':>6} {'delta':>7} {'wins':>5} {'losses':>7}")
    print("-" * 55)

    for min_drop in METAR_DROP_OPTIONS:
        filtered = [t for t in trades if passes_metar_gate(t, min_drop)]
        n = len(filtered)
        if n == 0:
            continue
        wr = sum(t.won for t in filtered) / n
        kept_pct = n / n_total
        delta = wr - baseline_wr
        w = sum(t.won for t in filtered)
        l = n - w
        print(f"{min_drop:>12.1f} {n:>5} {kept_pct:>5.0%} "
              f"{wr:>6.1%} {delta:>+7.1%} {w:>5} {l:>7}")

    # --- Combined gate ---
    print("\n=== Combined: best forecast gate + METAR trajectory ===")
    print(f"{'min_clear':>10} {'min_src':>8} {'min_drop':>9} {'n':>5} "
          f"{'kept%':>6} {'win%':>6} {'delta':>7}")
    print("-" * 70)

    for min_clearance in CLEARANCE_THRESHOLDS:
        for min_sources in MIN_SOURCES_OPTIONS:
            for min_drop in METAR_DROP_OPTIONS:
                filtered = [
                    t for t in trades
                    if passes_forecast_gate(t, min_clearance, min_sources)
                    and passes_metar_gate(t, min_drop)
                ]
                n = len(filtered)
                if n < 5:   # too few to be meaningful
                    continue
                wr = sum(t.won for t in filtered) / n
                kept_pct = n / n_total
                delta = wr - baseline_wr
                if delta > 0.05:   # only print meaningful improvements
                    print(f"{min_clearance:>10.1f} {min_sources:>8} {min_drop:>9.1f} "
                          f"{n:>5} {kept_pct:>5.0%} {wr:>6.1%} {delta:>+7.1%}")

    # --- Loss case study ---
    print("\n=== Loss deep-dive: what forecasts said at entry ===")
    losses = [t for t in trades if not t.won]
    print(f"{'ticker':50} {'ceil':>5} {'metar':>6} "
          f"{'nws':>6} {'om':>6} {'hrrr':>6} {'min_clr':>8}")
    print("-" * 100)
    for t in sorted(losses, key=lambda x: x.entry_at):
        nws  = t.forecasts.get("nws_hourly")
        om   = t.forecasts.get("open_meteo")
        hrrr = t.forecasts.get("hrrr")
        vals = [v for v in [nws, om, hrrr] if v is not None]
        min_clr = (min(vals) - t.ceiling) if vals else None
        metar_s = f"{t.metar_at_entry:.1f}" if t.metar_at_entry else "N/A"
        nws_s   = f"{nws:.1f}"  if nws  else "N/A"
        om_s    = f"{om:.1f}"   if om   else "N/A"
        hrrr_s  = f"{hrrr:.1f}" if hrrr else "N/A"
        clr_s   = f"{min_clr:+.1f}" if min_clr is not None else "N/A"
        print(f"{t.ticker:50} {t.ceiling:>5.1f} {metar_s:>6} "
              f"{nws_s:>6} {om_s:>6} {hrrr_s:>6} {clr_s:>8}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not DB.exists():
        print(f"ERROR: DB not found at {DB}")
        sys.exit(1)

    con = sqlite3.connect(DB)

    print("Loading band_arb NO trades…")
    trades = load_trades(con)
    print(f"  {len(trades)} settled trades found")

    print("Enriching with forecast + METAR data…")
    enrich_trades(con, trades)

    no_forecast = sum(1 for t in trades if not t.forecasts)
    has_metar   = sum(1 for t in trades if t.metar_at_entry is not None)
    print(f"  {len(trades) - no_forecast} trades with forecast data  "
          f"({no_forecast} missing)")
    print(f"  {has_metar} trades with METAR data\n")

    run_sweep(trades)


if __name__ == "__main__":
    main()
