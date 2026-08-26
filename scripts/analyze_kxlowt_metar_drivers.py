#!/usr/bin/env python3
"""Analyze what drives YES bid spikes in KXLOWT band_arb positions.

For each held band_arb NO trade, joins price_snapshots (YES bid over time)
with raw_forecasts (METAR running minimum over time) to compute:

  1. METAR cooling rate (dT/hour) — how fast is the running min falling?
  2. Time-adjusted risk — at current cooling rate, will temp reach band ceiling
     before market close?  risk_score = cooling_rate / hours_to_close
  3. Clearance shrink rate — how fast is the safety margin disappearing?

Key question: does the METAR cooling rate LEAD the YES bid, or does the
market price information in simultaneously?  If METAR leads by 1+ hours,
there's a usable exit signal.

Dew point note
--------------
Dew point is the theoretical floor temperature (air can't cool below dew
point without condensation).  If dew_point > band_ceiling, the overnight low
physically cannot drop into the band — strong safety signal.  We don't
currently log dew point; output flags this as a gap to fill.

Usage:
  venv/bin/python scripts/analyze_kxlowt_metar_drivers.py
"""

from __future__ import annotations

import re
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

DB = Path("data/db/opportunity_log.db")
_BAND_RE = re.compile(r"-B([\d.]+)$")


def band_ceiling(ticker: str) -> float | None:
    m = _BAND_RE.search(ticker)
    return float(m.group(1)) + 1.0 if m else None


def load_trades(con: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql("""
        SELECT id, ticker, limit_price as entry_yes_bid, logged_at,
               COALESCE(exit_pnl_cents, settled_pnl_cents) as pnl,
               settled_result
        FROM trades
        WHERE source='band_arb' AND side='no'
          AND ticker LIKE 'KXLOWT%B%'
          AND limit_price BETWEEN 5 AND 45
          AND COALESCE(exit_pnl_cents, settled_pnl_cents) IS NOT NULL
    """, con)


def load_snapshots(con: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql("""
        SELECT ps.trade_id, ps.snapshot_at, ps.yes_bid, ps.yes_ask
        FROM price_snapshots ps
        JOIN trades t ON t.id = ps.trade_id
        WHERE t.source='band_arb' AND t.side='no'
          AND t.ticker LIKE 'KXLOWT%B%'
    """, con)


def load_metar(con: sqlite3.Connection) -> pd.DataFrame:
    """Load METAR running-minimum readings from raw_forecasts."""
    return pd.read_sql("""
        SELECT rf.ticker, rf.logged_at, rf.data_value as metar_f
        FROM raw_forecasts rf
        WHERE rf.source='metar'
          AND rf.metric LIKE 'temp_low%'
    """, con)


def build_hourly_panel(
    trades: pd.DataFrame,
    snapshots: pd.DataFrame,
    metar: pd.DataFrame,
) -> pd.DataFrame:
    """Build one row per (trade × UTC hour) with YES bid and METAR cooling rate."""

    trades["ceiling"] = trades["ticker"].map(band_ceiling)
    trades = trades.dropna(subset=["ceiling"])
    trades["outcome"] = trades["pnl"].apply(lambda p: "win" if p > 0 else "loss")

    snapshots["dt"] = pd.to_datetime(snapshots["snapshot_at"], utc=True)
    snapshots["hour_floor"] = snapshots["dt"].dt.floor("h")

    metar["dt"] = pd.to_datetime(metar["logged_at"], utc=True)
    metar["hour_floor"] = metar["dt"].dt.floor("h")

    # Hourly average YES bid per trade
    snap_hourly = (
        snapshots.groupby(["trade_id", "hour_floor"])
        .agg(yes_bid=("yes_bid", "mean"), yes_ask=("yes_ask", "mean"))
        .reset_index()
    )

    # Hourly average METAR per ticker
    metar_hourly = (
        metar.groupby(["ticker", "hour_floor"])
        .agg(metar_f=("metar_f", "mean"))
        .reset_index()
    )

    rows = []
    for _, t in trades.iterrows():
        trade_snaps = snap_hourly[snap_hourly["trade_id"] == t["id"]].sort_values("hour_floor")
        trade_metar = metar_hourly[metar_hourly["ticker"] == t["ticker"]].sort_values("hour_floor")

        if trade_snaps.empty or trade_metar.empty:
            continue

        merged = pd.merge(trade_snaps, trade_metar, on="hour_floor", how="inner")
        if len(merged) < 2:
            continue

        merged = merged.sort_values("hour_floor").reset_index(drop=True)
        merged["trade_id"]  = t["id"]
        merged["ticker"]    = t["ticker"]
        merged["ceiling"]   = t["ceiling"]
        merged["outcome"]   = t["outcome"]
        merged["entry_bid"] = t["entry_yes_bid"]

        # Compute entry time in UTC, hours elapsed since entry
        entry_dt = pd.to_datetime(t["logged_at"], utc=True)
        merged["hours_since_entry"] = (merged["hour_floor"] - entry_dt).dt.total_seconds() / 3600

        # METAR cooling rate: dT per hour (running min change vs previous hour)
        merged["metar_delta"] = merged["metar_f"].diff()          # ≤0 when cooling, NaN for first row

        # Clearance from band ceiling
        merged["clearance"] = merged["metar_f"] - merged["ceiling"]

        # YES bid change vs previous hour (what we want to predict)
        merged["yes_bid_delta_next"] = merged["yes_bid"].shift(-1) - merged["yes_bid"]

        rows.append(merged)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def print_correlation_analysis(panel: pd.DataFrame) -> None:
    """How well do METAR rate and clearance predict the next-hour YES bid change?"""
    valid = panel.dropna(subset=["metar_delta", "clearance", "yes_bid_delta_next",
                                  "hours_since_entry"])
    valid = valid[valid["hours_since_entry"] >= 0]

    print(f"\n=== Correlation: what predicts next-hour YES bid change? ===")
    print(f"    ({len(valid):,} hourly observations)")

    features = {
        "metar_f (abs level)":          valid["metar_f"],
        "clearance (metar - ceiling)":  valid["clearance"],
        "metar_delta (cooling rate)":   valid["metar_delta"],
        "hours_since_entry":            valid["hours_since_entry"],
    }
    target = valid["yes_bid_delta_next"]

    for name, col in features.items():
        r = col.corr(target)
        print(f"  {name:40s}: r={r:+.3f}")


def print_lag_test(panel: pd.DataFrame) -> None:
    """Does METAR cooling rate predict YES bid change with a lag?"""
    valid = panel.dropna(subset=["metar_delta", "yes_bid_delta_next"])

    from scipy.stats import pearsonr  # type: ignore

    print(f"\n=== Lag test: METAR cooling rate vs YES bid change ===")
    # Current hour: does METAR delta now predict YES bid delta now?
    r_now, _ = pearsonr(valid["metar_delta"], valid["yes_bid_delta_next"])

    # Lagged: does METAR delta from 1h ago predict YES bid delta now?
    panel_lag = panel.copy()
    panel_lag["metar_delta_lag1"] = panel_lag.groupby("trade_id")["metar_delta"].shift(1)
    panel_lag["metar_delta_lag2"] = panel_lag.groupby("trade_id")["metar_delta"].shift(2)
    valid2 = panel_lag.dropna(subset=["metar_delta_lag1","metar_delta_lag2","yes_bid_delta_next"])

    r_lag1, _ = pearsonr(valid2["metar_delta_lag1"], valid2["yes_bid_delta_next"])
    r_lag2, _ = pearsonr(valid2["metar_delta_lag2"], valid2["yes_bid_delta_next"])

    print(f"  METAR Δ(t)   → YES bid Δ(t+1): r={r_now:+.3f}")
    print(f"  METAR Δ(t-1) → YES bid Δ(t+1): r={r_lag1:+.3f}  (1h lag)")
    print(f"  METAR Δ(t-2) → YES bid Δ(t+1): r={r_lag2:+.3f}  (2h lag)")

    if abs(r_lag1) > abs(r_now) * 1.05:
        print("  *** METAR leads market by ~1h — potential exit signal ***")
    elif abs(r_lag2) > abs(r_now) * 1.05:
        print("  *** METAR leads market by ~2h — potential exit signal ***")
    else:
        print("  Market prices METAR info approximately simultaneously (no exploitable lag)")


def print_threshold_analysis(panel: pd.DataFrame) -> None:
    """When cooling rate exceeds threshold AND clearance is low, how does YES bid move?"""
    print(f"\n=== Threshold analysis: cooling rate × clearance → YES bid next hour ===")
    print(f"{'cooling_rate':>14} {'clearance':>10} {'n':>6} "
          f"{'avg_yes_Δ':>10} {'pct_yes_UP':>11} {'outcome':>8}")
    print("-" * 65)

    valid = panel.dropna(subset=["metar_delta","clearance","yes_bid_delta_next"])

    for cool_thresh in [-0.5, -1.0, -2.0, -3.0]:
        for clr_thresh in [5.0, 3.0, 2.0, 1.0]:
            sub = valid[
                (valid["metar_delta"] <= cool_thresh) &
                (valid["clearance"] <= clr_thresh)
            ]
            if len(sub) < 5:
                continue
            avg_d = sub["yes_bid_delta_next"].mean()
            pct_up = (sub["yes_bid_delta_next"] > 0).mean()
            print(f"  Δtemp ≤{cool_thresh:+.1f}°/h  clr ≤{clr_thresh:.1f}°  "
                  f"{len(sub):>6}  {avg_d:>+10.2f}  {pct_up:>10.0%}  "
                  f"{'↑ SPIKE RISK' if pct_up > 0.5 and avg_d > 1 else ''}")


def print_time_adjusted_risk(panel: pd.DataFrame) -> None:
    """
    Time-adjusted risk score = clearance / hours_to_close
    Interpretation: how many degrees of buffer per remaining hour.
    Low values = danger zone.
    """
    print(f"\n=== Time-adjusted risk: clearance per hour remaining ===")
    print(f"(Assumes linear cooling; dew point would tighten this further)")

    valid = panel.dropna(subset=["clearance","hours_since_entry","yes_bid_delta_next"])
    # Approximate hours to close from entry (KXLOWT markets usually 18-24h)
    # Use hours_since_entry as proxy; risk increases as we approach close
    valid = valid[valid["hours_since_entry"] >= 0]

    # Bucket by time since entry
    valid["h_bucket"] = pd.cut(valid["hours_since_entry"],
                                bins=[0,3,6,9,12,15,18,99],
                                labels=["0-3h","3-6h","6-9h","9-12h","12-15h","15-18h","18h+"])

    print(f"\n  By hour since entry (avg clearance and YES bid change):")
    print(f"  {'hours':>8} {'n':>6} {'avg_clr':>9} {'avg_yes_Δ':>10} {'pct_YES_up':>11}")
    for h_label, grp in valid.groupby("h_bucket", observed=True):
        if len(grp) < 5:
            continue
        print(f"  {h_label:>8} {len(grp):>6} {grp['clearance'].mean():>+9.2f} "
              f"{grp['yes_bid_delta_next'].mean():>+10.2f} "
              f"{(grp['yes_bid_delta_next']>0).mean():>10.0%}")

    print(f"\n  By outcome (win vs loss):")
    print(f"  {'outcome':>8} {'hour_range':>12} {'avg_clr':>9} {'avg_metar_Δ':>12}")
    for outcome, grp in valid.groupby("outcome"):
        for h_label, subg in grp.groupby("h_bucket", observed=True):
            if len(subg) < 3:
                continue
            print(f"  {outcome:>8} {h_label:>12} {subg['clearance'].mean():>+9.2f} "
                  f"{subg['metar_delta'].dropna().mean():>+12.3f}")


def main() -> None:
    if not DB.exists():
        sys.exit(f"ERROR: {DB} not found")

    con = sqlite3.connect(DB)

    print("Loading trades…")
    trades = load_trades(con)
    print(f"  {len(trades)} band_arb KXLOWT NO trades")

    print("Loading price snapshots…")
    snaps = load_snapshots(con)
    print(f"  {len(snaps):,} snapshot rows")

    print("Loading METAR running-min data…")
    metar = load_metar(con)
    print(f"  {len(metar):,} METAR rows covering {metar['ticker'].nunique()} tickers")

    print("Building hourly panel…")
    panel = build_hourly_panel(trades, snaps, metar)
    if panel.empty:
        print("No joined data — check that trade tickers exist in raw_forecasts.")
        return
    print(f"  {len(panel):,} hourly observations, "
          f"{panel['trade_id'].nunique()} trades with joint data")

    print_correlation_analysis(panel)
    print_lag_test(panel)
    print_threshold_analysis(panel)
    print_time_adjusted_risk(panel)

    print(f"""
=== Dew Point Gap ===
Dew point is the theoretical temperature floor — air cannot cool below it
without condensation.  If dew_point > band_ceiling, the overnight low is
physically bounded above the band → NO is safe regardless of cooling rate.

Current status: NOT logged.  To add it:
  1. metar.py: parse 'dewpoint.value' from NWS ASOS GeoJSON response
  2. nws_asos.py: parse dewpoint from properties.dewpoint.value
  3. Store as DataPoint(source='metar', metric='dewpoint_<city>', ...)
  4. Log to raw_forecasts alongside temp_low

Once available: key derived signal =
  dew_margin = dew_point_f - ceiling
  if dew_margin > 0: overnight low cannot reach ceiling → NO is very safe
  if dew_margin < 0: temperature could theoretically fall through ceiling
""")


if __name__ == "__main__":
    main()
