#!/usr/bin/env python3
"""Analyze KXLOWT NO market efficiency: do prices lag METAR/forecast signals?

For each KXLOWT B-band market we join hourly Kalshi bid/ask candles with
hourly METAR running-minimum and forecast values.  We then look at:

  1. Correlation: how well does current temperature/forecast clearance predict
     the current NO bid (i.e. is the market efficient at all)?

  2. Lag: does last hour's clearance predict THIS hour's bid BETTER than the
     current hour's clearance? A stronger lag correlation means the market
     adjusts with a delay — exploitable alpha.

  3. Price reaction: after a large METAR drop (temp falls toward band), how
     many hours does it take for the NO bid to fall accordingly?

  4. Signal vs price divergence: find specific candles where clearance is
     high but bid is still low (market underpricing NO) or vice versa.

Output: printed tables + saves CSV to data/analysis/kxlowt_efficiency.csv

Usage:
  venv/bin/python scripts/analyze_kxlowt_market_efficiency.py
"""

from __future__ import annotations

import re
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

CANDLES_DB  = Path("data/candlesticks.db")
FORECAST_DB = Path("data/db/opportunity_log.db")
OUT_CSV     = Path("data/analysis/kxlowt_efficiency.csv")

# Map KXLOWT series prefix → raw_forecasts metric
SERIES_TO_METRIC: dict[str, str] = {
    "KXLOWTATL":  "temp_low_atl",
    "KXLOWTAUS":  "temp_low_aus",
    "KXLOWTBOS":  "temp_low_bos",
    "KXLOWTCHI":  "temp_low_chi",
    "KXLOWTDC":   "temp_low_dca",
    "KXLOWTDEN":  "temp_low_den",
    "KXLOWTHOU":  "temp_low_hou",
    "KXLOWTLV":   "temp_low_las",
    "KXLOWTLAX":  "temp_low_lax",
    "KXLOWTMIA":  "temp_low_mia",
    "KXLOWTMIN":  "temp_low_msp",
    "KXLOWTNOLA": "temp_low_msy",
    "KXLOWTNYC":  "temp_low_ny",
    "KXLOWTOKC":  "temp_low_okc",
    "KXLOWTPHIL": "temp_low_phl",
    "KXLOWTPHX":  "temp_low_phx",
    "KXLOWTSATX": "temp_low_sat",
    "KXLOWTSEA":  "temp_low_sea",
    "KXLOWTSFO":  "temp_low_sfo",
}

_BAND_RE   = re.compile(r"-B([\d.]+)$")
_SERIES_RE = re.compile(r"^(KXLOWT[A-Z]+)-")


def band_ceiling(ticker: str) -> float | None:
    m = _BAND_RE.search(ticker)
    return float(m.group(1)) + 1.0 if m else None


def series(ticker: str) -> str | None:
    m = _SERIES_RE.match(ticker)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Load candles
# ---------------------------------------------------------------------------

def load_candles(con_c: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql("""
        SELECT c.ticker,
               c.period_ts,
               c.bid_close  AS yes_bid,
               c.ask_close  AS yes_ask,
               m.result,
               m.open_ts,
               m.close_ts
        FROM candles c
        JOIN markets m ON m.ticker = c.ticker
        WHERE c.ticker LIKE 'KXLOWT%B%'
          AND c.bid_close IS NOT NULL
          AND c.ask_close IS NOT NULL
    """, con_c)
    df["dt"] = pd.to_datetime(df["period_ts"], unit="s", utc=True)
    df["hour_str"] = df["dt"].dt.strftime("%Y-%m-%dT%H:00:00")
    # NO bid = 100 - yes_ask
    df["no_bid"] = 100 - df["yes_ask"]
    df["ceiling"] = df["ticker"].map(band_ceiling)
    df["ser"]     = df["ticker"].map(series)
    df = df.dropna(subset=["ceiling", "ser"])
    return df


# ---------------------------------------------------------------------------
# Load forecasts — one row per (metric, source, hour)
# ---------------------------------------------------------------------------

def load_forecasts(con_f: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql("""
        SELECT metric, source,
               strftime('%Y-%m-%dT%H:00:00', logged_at) AS hour_str,
               AVG(data_value) AS value
        FROM raw_forecasts
        WHERE metric LIKE 'temp_low%'
          AND source IN ('metar','nws_hourly','open_meteo','hrrr')
        GROUP BY metric, source, hour_str
    """, con_f)
    return df


# ---------------------------------------------------------------------------
# Join
# ---------------------------------------------------------------------------

def build_panel(candles: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # Build forecast lookup: (metric, source, hour_str) → value
    fc_idx = forecasts.set_index(["metric","source","hour_str"])["value"].to_dict()

    for _, c in candles.iterrows():
        metric = SERIES_TO_METRIC.get(c["ser"])
        if metric is None:
            continue

        h = c["hour_str"]
        metar = fc_idx.get((metric, "metar",      h))
        nws   = fc_idx.get((metric, "nws_hourly", h))
        om    = fc_idx.get((metric, "open_meteo", h))
        hrrr  = fc_idx.get((metric, "hrrr",       h))

        ceil = c["ceiling"]

        # Clearances (positive = temp/forecast is ABOVE band ceiling → NO favoured)
        metar_clr = (metar - ceil) if metar is not None else None
        nws_clr   = (nws   - ceil) if nws   is not None else None
        om_clr    = (om    - ceil) if om    is not None else None
        hrrr_clr  = (hrrr  - ceil) if hrrr  is not None else None

        # Best (lowest) forecast clearance — most pessimistic model
        fc_vals = [v for v in [nws_clr, om_clr, hrrr_clr] if v is not None]
        min_fc_clr = min(fc_vals) if fc_vals else None

        # Hours until market close
        hours_to_close = (c["close_ts"] - c["period_ts"]) / 3600

        rows.append({
            "ticker":        c["ticker"],
            "ser":           c["ser"],
            "hour_str":      h,
            "period_ts":     c["period_ts"],
            "result":        c["result"],
            "yes_bid":       c["yes_bid"],
            "no_bid":        c["no_bid"],
            "ceiling":       ceil,
            "hours_to_close": hours_to_close,
            "metar":         metar,
            "metar_clr":     metar_clr,
            "nws_clr":       nws_clr,
            "om_clr":        om_clr,
            "hrrr_clr":      hrrr_clr,
            "min_fc_clr":    min_fc_clr,
        })

    panel = pd.DataFrame(rows)
    panel = panel.sort_values(["ticker","period_ts"]).reset_index(drop=True)

    # Lagged values (previous hour within same ticker)
    panel["metar_clr_lag1"] = panel.groupby("ticker")["metar_clr"].shift(1)
    panel["min_fc_clr_lag1"] = panel.groupby("ticker")["min_fc_clr"].shift(1)
    panel["no_bid_lag1"]    = panel.groupby("ticker")["no_bid"].shift(1)
    panel["metar_delta_1h"] = panel.groupby("ticker")["metar_clr"].diff()   # change vs prev hour

    # Next-hour bid (what we'd get if we enter NOW and exit next hour)
    panel["no_bid_next1h"]  = panel.groupby("ticker")["no_bid"].shift(-1)
    panel["no_bid_next2h"]  = panel.groupby("ticker")["no_bid"].shift(-2)

    return panel


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def print_correlation_table(panel: pd.DataFrame) -> None:
    cols = ["no_bid","metar_clr","metar_clr_lag1","min_fc_clr","min_fc_clr_lag1",
            "metar_delta_1h","hours_to_close"]
    sub = panel[cols].dropna()
    print(f"\n=== Correlation with NO bid ({len(sub):,} hourly obs) ===")
    corr = sub.corr()["no_bid"].drop("no_bid").round(3)
    for col, val in corr.items():
        print(f"  {col:25s}: {val:+.3f}")


def print_lag_analysis(panel: pd.DataFrame) -> None:
    """Key question: does current clearance or LAGGED clearance predict next-hour bid better?"""
    from scipy.stats import pearsonr  # type: ignore

    sub = panel.dropna(subset=["metar_clr","metar_clr_lag1","no_bid_next1h"])

    r_current, _ = pearsonr(sub["metar_clr"],      sub["no_bid_next1h"])
    r_lag,     _ = pearsonr(sub["metar_clr_lag1"], sub["no_bid_next1h"])

    print(f"\n=== METAR clearance → next-hour NO bid (lag test) ===")
    print(f"  Current clearance  → next-hour bid: r={r_current:+.3f}")
    print(f"  Lagged clearance   → next-hour bid: r={r_lag:+.3f}")
    if abs(r_lag) > abs(r_current):
        print("  ** LAG > CURRENT: market adjusts with delay — potential alpha **")
    else:
        print("  Current >= lag: market is approximately efficient on METAR")

    if panel["min_fc_clr"].notna().sum() > 100:
        sub2 = panel.dropna(subset=["min_fc_clr","min_fc_clr_lag1","no_bid_next1h"])
        r_fc_cur, _ = pearsonr(sub2["min_fc_clr"],      sub2["no_bid_next1h"])
        r_fc_lag, _ = pearsonr(sub2["min_fc_clr_lag1"], sub2["no_bid_next1h"])
        print(f"\n=== Forecast clearance → next-hour NO bid ===")
        print(f"  Current fc clearance  → next-hour bid: r={r_fc_cur:+.3f}")
        print(f"  Lagged  fc clearance  → next-hour bid: r={r_fc_lag:+.3f}")
        if abs(r_fc_lag) > abs(r_fc_cur):
            print("  ** LAG > CURRENT: forecast signal lags market — potential alpha **")
        else:
            print("  Market efficiently priced on forecast signal")


def print_price_reaction(panel: pd.DataFrame) -> None:
    """After a large METAR drop (clearance decreases by >=2°F), how does NO bid react?"""
    drops = panel[panel["metar_delta_1h"] <= -2.0].copy()
    if drops.empty:
        print("\n=== No large METAR drops found ===")
        return

    print(f"\n=== Price reaction after METAR drop >=2°F (n={len(drops)}) ===")
    print(f"  Avg NO bid at drop hour:    {drops['no_bid'].mean():.1f}¢")
    print(f"  Avg NO bid next hour:       {drops['no_bid_next1h'].dropna().mean():.1f}¢")
    print(f"  Avg NO bid 2 hours later:   {drops['no_bid_next2h'].dropna().mean():.1f}¢")
    immediate = drops["no_bid_next1h"] - drops["no_bid"]
    print(f"  Avg bid change in 1h:       {immediate.mean():+.1f}¢")
    print(f"  % where bid FALLS (NO gets cheaper) in 1h: "
          f"{(immediate < 0).mean():.0%}")


def print_divergence_opportunities(panel: pd.DataFrame) -> None:
    """Find candles where METAR clearance is high but NO bid is still cheap — market lag."""
    # Only look at markets with >12h remaining (enough time to move)
    sub = panel[(panel["hours_to_close"] > 12) &
                (panel["metar_clr"].notna()) &
                (panel["metar_clr"] >= 3.0) &   # temp is 3°F+ above ceiling
                (panel["no_bid"] <= 25)]          # but NO is priced cheap

    print(f"\n=== Divergence: METAR clearance >=3°F but NO bid <=25¢ "
          f"({len(sub)} candles) ===")
    if sub.empty:
        return

    by_result = sub.groupby("result").agg(
        n=("no_bid","count"),
        avg_no_bid=("no_bid","mean"),
        avg_metar_clr=("metar_clr","mean"),
        avg_hours_left=("hours_to_close","mean"),
        next1h_avg=("no_bid_next1h","mean"),
    ).round(2)
    print(by_result.to_string())

    print(f"\n  If you bought NO at these candles (avg {sub['no_bid'].mean():.1f}¢):")
    wins_settle = ((sub["result"]=="no") & sub["result"].notna()).mean()
    print(f"  Settlement win rate: {wins_settle:.0%}")
    next_bid_change = (sub["no_bid_next1h"] - sub["no_bid"]).mean()
    print(f"  Avg NO bid change next hour: {next_bid_change:+.1f}¢")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not CANDLES_DB.exists():
        print(f"ERROR: {CANDLES_DB} not found")
        sys.exit(1)

    con_c = sqlite3.connect(CANDLES_DB)
    con_f = sqlite3.connect(FORECAST_DB)

    print("Loading KXLOWT candles…")
    candles = load_candles(con_c)
    print(f"  {len(candles):,} candle rows across {candles['ticker'].nunique()} tickers")

    print("Loading forecast/METAR data…")
    forecasts = load_forecasts(con_f)
    print(f"  {len(forecasts):,} forecast rows")

    print("Building hourly panel…")
    panel = build_panel(candles, forecasts)
    print(f"  {len(panel):,} panel rows, "
          f"{panel['metar_clr'].notna().sum()} with METAR, "
          f"{panel['min_fc_clr'].notna().sum()} with forecast")

    # Analysis
    print_correlation_table(panel)
    print_lag_analysis(panel)
    print_price_reaction(panel)
    print_divergence_opportunities(panel)

    # Save
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUT_CSV, index=False)
    print(f"\nPanel saved to {OUT_CSV}")


if __name__ == "__main__":
    main()
