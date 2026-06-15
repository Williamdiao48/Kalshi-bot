#!/usr/bin/env python3
"""Backtest: does HRRR run-to-run revision predict Kalshi YES bid movement?

Key question: when HRRR significantly revises its daily-high forecast,
does the Kalshi market reprice in the *same* hour, or is there a 1-2 hour
lag we could exploit?

Methodology
-----------
1. Take the LAST raw_forecasts HRRR reading within each UTC hour for each
   city — this is HRRR's "official" value for that hour's run.
2. Compute HRRR delta = value[H] - value[H-1] (positive = warmer revision).
3. Pull KXHIGHT YES bid at each hour from candlesticks.db.
4. Compute YES bid change in the same hour and the next 1-2 hours.
5. Lag test: does HRRR delta now predict YES bid change now (market is fast)
   or does it predict YES bid change next hour (market lags = exploitable)?

Usage
-----
  venv/bin/python scripts/backtest_hrrr_delta.py
"""

from __future__ import annotations

import re
import sqlite3
import sys
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

FORECAST_DB = Path("data/db/opportunity_log.db")
CANDLES_DB  = Path("data/candlesticks.db")

# Map KXHIGHT series prefix → raw_forecasts metric
SERIES_TO_METRIC: dict[str, str] = {
    "KXHIGHTATL":  "temp_high_atl",
    "KXHIGHAUS":   "temp_high_aus",
    "KXHIGHTBOS":  "temp_high_bos",
    "KXHIGHTCHI":  "temp_high_chi",
    "KXHIGHTDAL":  "temp_high_dal",
    "KXHIGHTDC":   "temp_high_dca",
    "KXHIGHDEN":   "temp_high_den",
    "KXHIGHTHOU":  "temp_high_hou",
    "KXHIGHTLV":   "temp_high_las",
    "KXHIGHLAX":   "temp_high_lax",
    "KXHIGHMIA":   "temp_high_mia",
    "KXHIGHTMIN":  "temp_high_msp",
    "KXHIGHTNOLA": "temp_high_msy",
    "KXHIGHNY":    "temp_high_ny",
    "KXHIGHTOKC":  "temp_high_okc",
    "KXHIGHPHIL":  "temp_high_phl",
    "KXHIGHTPHX":  "temp_high_phx",
    "KXHIGHTSATX": "temp_high_sat",
    "KXHIGHTSEA":  "temp_high_sea",
    "KXHIGHTSFO":  "temp_high_sfo",
}

_SERIES_RE = re.compile(r"^(KXHIGH[A-Z]+)-")


def series_from_ticker(ticker: str) -> str | None:
    m = _SERIES_RE.match(ticker)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_hrrr_hourly(con: sqlite3.Connection) -> pd.DataFrame:
    """Last HRRR reading per (metric, UTC hour)."""
    print("Loading HRRR forecasts…")
    df = pd.read_sql("""
        SELECT
            metric,
            strftime('%Y-%m-%dT%H:00', logged_at) AS hour_utc,
            data_value
        FROM raw_forecasts
        WHERE source = 'hrrr'
          AND metric LIKE 'temp_high%'
    """, con)
    # Take the last reading within each hour (most recent HRRR run data)
    df = (
        df.sort_values("hour_utc")
          .groupby(["metric", "hour_utc"], as_index=False)
          .last()
    )
    df["hour_dt"] = pd.to_datetime(df["hour_utc"], utc=True)
    print(f"  {len(df):,} hourly HRRR rows across {df['metric'].nunique()} cities")
    return df


def load_candles(con: sqlite3.Connection) -> pd.DataFrame:
    """Hourly YES bid per KXHIGHT ticker."""
    print("Loading KXHIGHT candles…")
    df = pd.read_sql("""
        SELECT
            c.ticker,
            c.period_ts,
            c.bid_close  AS yes_bid,
            c.ask_close  AS yes_ask,
            m.result,
            m.open_ts,
            m.close_ts
        FROM candles c
        JOIN markets m ON m.ticker = c.ticker
        WHERE c.ticker LIKE 'KXHIGH%B%'
          AND c.bid_close IS NOT NULL
          AND c.ask_close IS NOT NULL
    """, con)
    df["hour_dt"] = pd.to_datetime(df["period_ts"], unit="s", utc=True).dt.floor("h")
    df["hour_utc"] = df["hour_dt"].dt.strftime("%Y-%m-%dT%H:00")
    df["ser"] = df["ticker"].map(series_from_ticker)
    df["metric"] = df["ser"].map(SERIES_TO_METRIC)
    df = df.dropna(subset=["metric"])
    print(f"  {len(df):,} candle rows across {df['ticker'].nunique()} tickers")
    return df


# ---------------------------------------------------------------------------
# Build panel
# ---------------------------------------------------------------------------

def build_panel(hrrr: pd.DataFrame, candles: pd.DataFrame) -> pd.DataFrame:
    """Join HRRR hourly values with YES bid, compute deltas."""

    # Compute HRRR delta within each (metric, market-day)
    # Sort by metric + hour so shift is correct
    hrrr = hrrr.sort_values(["metric", "hour_utc"]).copy()
    hrrr["hrrr_delta"] = hrrr.groupby("metric")["data_value"].diff()
    hrrr = hrrr.rename(columns={"data_value": "hrrr_value"})

    # Merge candles with HRRR on (metric, hour_utc)
    merged = candles.merge(
        hrrr[["metric", "hour_utc", "hrrr_value", "hrrr_delta"]],
        on=["metric", "hour_utc"],
        how="inner",
    )

    # YES bid change within the same ticker, next 1h and 2h
    merged = merged.sort_values(["ticker", "period_ts"]).reset_index(drop=True)
    merged["yes_bid_delta_0h"] = merged.groupby("ticker")["yes_bid"].diff()      # vs prev hour
    merged["yes_bid_next_1h"]  = merged.groupby("ticker")["yes_bid"].shift(-1)
    merged["yes_bid_next_2h"]  = merged.groupby("ticker")["yes_bid"].shift(-2)
    merged["yes_bid_delta_1h"] = merged["yes_bid_next_1h"] - merged["yes_bid"]
    merged["yes_bid_delta_2h"] = merged["yes_bid_next_2h"] - merged["yes_bid"]

    # Hours remaining to close
    merged["hours_to_close"] = (merged["close_ts"] - merged["period_ts"]) / 3600

    return merged


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def lag_test(panel: pd.DataFrame) -> None:
    """Pearson correlation: does HRRR delta predict YES bid change now vs next hour?"""
    valid = panel.dropna(subset=["hrrr_delta", "yes_bid_delta_0h", "yes_bid_delta_1h"])

    def r(x, y):
        n = len(x)
        if n < 10:
            return float("nan"), float("nan")
        mx, my = x.mean(), y.mean()
        num = ((x - mx) * (y - my)).sum()
        den = (((x - mx)**2).sum() * ((y - my)**2).sum()) ** 0.5
        rho = num / den if den > 0 else 0
        # two-tailed p via t-distribution approx
        t = rho * math.sqrt(n - 2) / math.sqrt(max(1 - rho**2, 1e-9))
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
        return round(rho, 3), round(p, 4)

    r0, p0 = r(valid["hrrr_delta"], valid["yes_bid_delta_0h"])
    r1, p1 = r(valid["hrrr_delta"], valid["yes_bid_delta_1h"])

    print(f"\n=== Lag test: HRRR delta → YES bid change ({len(valid):,} obs) ===")
    print(f"  HRRR Δ(H)   → YES bid Δ(H)   (same hour): r={r0:+.3f}  p={p0:.4f}")
    print(f"  HRRR Δ(H)   → YES bid Δ(H+1) (1h lag):   r={r1:+.3f}  p={p1:.4f}")

    if abs(r0) > abs(r1):
        print("  → Market reprices in the SAME hour as HRRR update (no lag)")
    elif abs(r1) > abs(r0) * 1.1:
        print("  *** Market lags HRRR by ~1h — potential exploitable window ***")
    else:
        print("  → Roughly simultaneous repricing (weak/no lag)")


def bucket_analysis(panel: pd.DataFrame) -> None:
    """For different HRRR revision magnitudes, how does YES bid respond?"""
    valid = panel.dropna(subset=["hrrr_delta", "yes_bid_delta_0h", "yes_bid_delta_1h"])

    print(f"\n=== YES bid response to HRRR revision ===")
    print(f"{'HRRR revision':22} {'n':>6} {'avg_YES_Δ(0h)':>14} {'avg_YES_Δ(1h)':>14} "
          f"{'pct_drop(0h)':>13} {'pct_drop(1h)':>13}")
    print("-" * 90)

    buckets = [
        ("HRRR drops ≥4°F",   lambda d: d <= -4.0),
        ("HRRR drops 2-4°F",  lambda d: -4.0 < d <= -2.0),
        ("HRRR drops 1-2°F",  lambda d: -2.0 < d <= -1.0),
        ("flat (< ±1°F)",     lambda d: -1.0 < d < 1.0),
        ("HRRR rises 1-2°F",  lambda d: 1.0 <= d < 2.0),
        ("HRRR rises 2-4°F",  lambda d: 2.0 <= d < 4.0),
        ("HRRR rises ≥4°F",   lambda d: d >= 4.0),
    ]

    for label, fn in buckets:
        grp = valid[valid["hrrr_delta"].map(fn)]
        if len(grp) < 5:
            continue
        avg0 = grp["yes_bid_delta_0h"].mean()
        avg1 = grp["yes_bid_delta_1h"].mean()
        pct_drop0 = (grp["yes_bid_delta_0h"] < -2).mean()
        pct_drop1 = (grp["yes_bid_delta_1h"] < -2).mean()
        print(f"  {label:22} {len(grp):>6} {avg0:>+14.2f} {avg1:>+14.2f} "
              f"{pct_drop0:>12.0%} {pct_drop1:>12.0%}")


def large_revision_detail(panel: pd.DataFrame) -> None:
    """For large HRRR drops (≥3°F), how quickly does YES bid respond?"""
    big_drops = panel[panel["hrrr_delta"] <= -3.0].dropna(
        subset=["yes_bid_delta_0h", "yes_bid_delta_1h"]
    )
    if big_drops.empty:
        print("\nNo large HRRR drops found.")
        return

    print(f"\n=== Large HRRR drops (≥3°F) — n={len(big_drops)} ===")
    print(f"  Avg HRRR drop:        {big_drops['hrrr_delta'].mean():.1f}°F")
    print(f"  YES bid same hour:    {big_drops['yes_bid_delta_0h'].mean():+.1f}¢ avg  "
          f"  ({(big_drops['yes_bid_delta_0h'] < -3).mean():.0%} dropped >3¢)")
    print(f"  YES bid next hour:    {big_drops['yes_bid_delta_1h'].mean():+.1f}¢ avg  "
          f"  ({(big_drops['yes_bid_delta_1h'] < -3).mean():.0%} dropped >3¢)")

    # Cumulative: how much total does YES bid drop over 2h after a big HRRR drop?
    big_drops_v = big_drops.dropna(subset=["yes_bid_delta_2h"])
    if len(big_drops_v) > 5:
        total_drop_2h = big_drops_v["yes_bid_delta_1h"] + big_drops_v["yes_bid_delta_2h"]
        print(f"  YES bid over 2h post: {total_drop_2h.mean():+.1f}¢ avg")

    print(f"\n  By hours-to-close at time of HRRR drop:")
    print(f"  {'htc_bucket':15} {'n':>5} {'Δ(0h)':>8} {'Δ(1h)':>8} {'pct_0h_drops':>14}")
    big_drops["htc_bucket"] = pd.cut(
        big_drops["hours_to_close"],
        bins=[0, 4, 8, 12, 16, 20, 99],
        labels=["0-4h", "4-8h", "8-12h", "12-16h", "16-20h", "20h+"],
    )
    for label, grp in big_drops.groupby("htc_bucket", observed=True):
        if len(grp) < 3:
            continue
        print(f"  {str(label):15} {len(grp):>5} {grp['yes_bid_delta_0h'].mean():>+8.1f} "
              f"{grp['yes_bid_delta_1h'].mean():>+8.1f} "
              f"{(grp['yes_bid_delta_0h'] < -3).mean():>13.0%}")


def by_city(panel: pd.DataFrame) -> None:
    """Per-city lag summary for big HRRR drops."""
    big_drops = panel[panel["hrrr_delta"] <= -2.0].dropna(
        subset=["yes_bid_delta_0h", "yes_bid_delta_1h"]
    )
    if big_drops.empty:
        return

    print(f"\n=== Per-city: YES bid response to HRRR drop ≥2°F ===")
    print(f"  {'city':12} {'n':>5} {'Δ(0h)':>8} {'Δ(1h)':>8} {'0h>1h?':>8}")
    for metric, grp in big_drops.groupby("metric"):
        if len(grp) < 3:
            continue
        city = metric.replace("temp_high_", "")
        d0 = grp["yes_bid_delta_0h"].mean()
        d1 = grp["yes_bid_delta_1h"].mean()
        faster = "same" if abs(d0) >= abs(d1) else "lags"
        print(f"  {city:12} {len(grp):>5} {d0:>+8.1f} {d1:>+8.1f} {faster:>8}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    for p in [FORECAST_DB, CANDLES_DB]:
        if not p.exists():
            sys.exit(f"ERROR: {p} not found")

    con_f = sqlite3.connect(FORECAST_DB)
    con_c = sqlite3.connect(CANDLES_DB)

    hrrr   = load_hrrr_hourly(con_f)
    candles = load_candles(con_c)

    print("Building panel…")
    panel = build_panel(hrrr, candles)
    print(f"  {len(panel):,} joined rows  "
          f"({panel['hrrr_delta'].notna().sum():,} with HRRR delta)")

    lag_test(panel)
    bucket_analysis(panel)
    large_revision_detail(panel)
    by_city(panel)

    print("\nDone.")


if __name__ == "__main__":
    main()
