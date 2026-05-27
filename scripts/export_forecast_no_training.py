#!/usr/bin/env python3
"""Export forecast_no training data CSV for logistic regression calibration.

Runs the same data pipeline as backtest_forecast_no_hist.py but outputs a
per-trade CSV of features + outcome instead of summary statistics.

Usage:
    venv/bin/python scripts/export_forecast_no_training.py

Output: data/backtest/forecast_no_training.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from backtest_forecast_no_hist import (  # noqa: E402
    build_obs_state,
    build_signal_snapshots,
    find_first_entries,
    load_fills_and_settlements,
    load_raw_forecasts,
    _parse_ticker,
)

OUTPUT = Path(__file__).parent.parent / "data" / "backtest" / "forecast_no_training.csv"

# Permissive thresholds — expose the full signal distribution to the model.
# The model learns which edge values actually predict wins; we don't pre-filter.
MIN_EDGE_EXPORT   = 0.5
MIN_SOURCES_EXPORT = 2
REQUIRE_OM_EXPORT  = False

# Continuous edge feature columns (matching historical CSV schema)
_MODEL_EDGE_MAP = {
    "edge_ecmwf": "edge_open_meteo_ecmwf",
    "edge_icon":  "edge_open_meteo_icon",
    "edge_gem":   "edge_open_meteo_gem",
    "edge_hrrr":  "edge_hrrr",
}

# Non-GFS sources counted toward n_supporting (positive edge = supports NO)
_SUPPORTING_SOURCES = ["edge_open_meteo_ecmwf", "edge_open_meteo_icon", "edge_open_meteo_gem", "edge_hrrr"]


def _n_supporting(row: pd.Series) -> int:
    return sum(
        1 for c in _SUPPORTING_SOURCES
        if c in row.index and pd.notna(row[c]) and row[c] > 0
    )



def main() -> None:
    # --- Phase A: Load forecast + obs data ---
    df_fc, df_obs = load_raw_forecasts()

    print("Parsing ticker metadata…", flush=True)
    ticker_meta: dict[str, dict] = {}
    for t in df_fc["ticker"].unique():
        m = _parse_ticker(t)
        if m is not None:
            ticker_meta[t] = m
    df_fc = df_fc[df_fc["ticker"].isin(ticker_meta)].copy()
    print(f"  {len(ticker_meta):,} tickers parsed", flush=True)

    obs_state = build_obs_state(df_obs)

    # Pre-filter: only fetch tickers that showed ≥ 1 source with edge ≥ 0.5°F
    signal_tickers = set(df_fc.loc[df_fc["edge"] >= MIN_EDGE_EXPORT, "ticker"].unique())
    all_tickers = [t for t in ticker_meta if t in signal_tickers]
    df_fc = df_fc[df_fc["ticker"].isin(signal_tickers)].copy()
    print(f"  {len(all_tickers):,} signal tickers for fills fetch", flush=True)

    # --- Phase A3+A4: Fetch fills + settlements (uses cache if available) ---
    fills_data = load_fills_and_settlements(all_tickers)

    # --- Phase B: Build snapshots ---
    snap = build_signal_snapshots(df_fc, obs_state, ticker_meta)
    if snap.empty:
        print("ERROR: No signal snapshots — check DB.", flush=True)
        return

    # --- Phase D: First entry per ticker (permissive thresholds) ---
    entries = find_first_entries(snap, fills_data, MIN_EDGE_EXPORT, MIN_SOURCES_EXPORT, REQUIRE_OM_EXPORT)
    if entries.empty:
        print("No qualifying entries found.", flush=True)
        return
    print(f"  {len(entries):,} first entries before feature engineering", flush=True)

    # --- Feature engineering ---
    import math as _math

    # Continuous per-model edge columns (matching historical CSV schema)
    for out_col, src_col in _MODEL_EDGE_MAP.items():
        entries[out_col] = entries[src_col] if src_col in entries.columns else float("nan")

    entries["n_supporting"]   = entries.apply(_n_supporting, axis=1)
    entries["is_no_high"]     = (entries["no_direction"] == "NO_HIGH").astype(int)
    entries["is_high_market"] = entries["is_high"].astype(int)
    entries["won"]            = (entries["settlement"] == "no").astype(int)

    # Seasonality encoding from market date
    entries["month_sin"] = entries["mkt_date"].apply(
        lambda d: round(_math.sin(2 * _math.pi * d.month / 12), 4)
    )
    entries["month_cos"] = entries["mkt_date"].apply(
        lambda d: round(_math.cos(2 * _math.pi * d.month / 12), 4)
    )

    # --- Output ---
    keep = [
        "ticker", "mkt_date", "city",
        "edge_ecmwf", "edge_icon", "edge_gem", "edge_hrrr",
        "model_spread", "n_supporting",
        "is_no_high", "is_high_market", "month_sin", "month_cos", "won",
    ]
    out = entries[keep].copy()

    feature_cols = [
        "edge_ecmwf", "edge_icon", "edge_gem", "edge_hrrr",
        "model_spread", "n_supporting",
        "is_no_high", "is_high_market", "month_sin", "month_cos",
    ]
    # Only require structural features to be non-NaN.
    # Edge columns (edge_*) may be NaN when individual model data wasn't logged —
    # training fills them to 0.0 (neutral), so keep these rows.
    required = ["model_spread", "n_supporting", "is_no_high", "is_high_market",
                "month_sin", "month_cos", "won"]
    before = len(out)
    out = out.dropna(subset=required)
    dropped = before - len(out)
    if dropped:
        print(f"  Dropped {dropped} rows with NaN structural features", flush=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT, index=False)

    n = len(out)
    wr = out["won"].mean()
    print(f"\nExported {n:,} rows to {OUTPUT}", flush=True)
    print(f"Win rate: {wr:.1%}  ({int(out['won'].sum())} wins / {n} total)", flush=True)
    print(f"\nFeature summary:")
    print(out[feature_cols].describe().round(2).to_string())
    print(f"\nCity distribution (top 10):")
    print(out["city"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
