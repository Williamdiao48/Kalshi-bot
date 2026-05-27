#!/usr/bin/env python3
"""Train logistic regression calibration model for forecast_no trades.

Combines historical training CSV (build_forecast_no_training_historical.py) with
live training CSV (export_forecast_no_training.py), fits a
CalibratedClassifierCV(LogisticRegression) model, prints validation metrics,
and saves the model to data/models/forecast_no_cal.pkl.

Features are continuous per-model edge values vs the band ceiling (°F).
edge_hrrr is NaN for historical rows (HRRR unavailable historically); filled
to 0.0 before training = neutral/no signal.

Usage:
    venv/bin/python scripts/train_forecast_no_model.py

Requires: scikit-learn, joblib, pandas, numpy
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

LIVE_CSV = Path(__file__).parent.parent / "data" / "backtest" / "forecast_no_training.csv"
HIST_CSV = Path(__file__).parent.parent / "data" / "backtest" / "forecast_no_training_historical.csv"
OUTPUT   = Path(__file__).parent.parent / "data" / "models" / "forecast_no_cal.pkl"

# Continuous per-model edge features (°F vs band ceiling; positive = supports NO).
# edge_hrrr is NaN for historical rows → filled to 0.0 before training.
# model_spread and n_supporting capture consensus / disagreement.
# month_sin / month_cos capture seasonal forecast accuracy patterns.
FEATURES = [
    "edge_ecmwf",
    "edge_icon",
    "edge_gem",
    "edge_hrrr",
    "model_spread",
    "n_supporting",
    "is_no_high",
    "is_high_market",
    "month_sin",
    "month_cos",
]

TARGET = "won"


def main() -> None:
    if not LIVE_CSV.exists():
        print(f"ERROR: {LIVE_CSV} not found — run export_forecast_no_training.py first", file=sys.stderr)
        sys.exit(1)
    if not HIST_CSV.exists():
        print(f"ERROR: {HIST_CSV} not found — run build_forecast_no_training_historical.py first", file=sys.stderr)
        sys.exit(1)

    df_live = pd.read_csv(LIVE_CSV)
    df_hist = pd.read_csv(HIST_CSV)

    print(f"Live rows: {len(df_live):,}  WR={df_live[TARGET].mean():.1%}")
    print(f"Hist rows: {len(df_hist):,}  WR={df_hist[TARGET].mean():.1%}")

    # Historical first so chronological split keeps live rows in the test set.
    df = pd.concat([df_hist, df_live], ignore_index=True)
    print(f"Combined:  {len(df):,} rows\n")

    missing = [f for f in FEATURES + [TARGET] if f not in df.columns]
    if missing:
        print(f"ERROR: missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    # Fill edge NaN with 0.0 = "model unavailable / no signal" for all edge columns.
    # edge_hrrr: always NaN for historical rows (gfs_hrrr returns GFS historically).
    # edge_ecmwf: NaN when ECMWF not archived for that city/date in Open-Meteo.
    # Same treatment: missing model = neutral, not skip row.
    for col in ["edge_hrrr", "edge_ecmwf", "edge_icon", "edge_gem"]:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    before = len(df)
    df = df.dropna(subset=FEATURES + [TARGET])
    if len(df) < before:
        print(f"Dropped {before - len(df)} rows with NaN features\n")

    X = df[FEATURES].values
    y = df[TARGET].values.astype(int)

    # Chronological split: live rows fall naturally into the test set since they
    # are appended after historical rows.
    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"Train: {len(X_train):,} rows  ({y_train.mean():.1%} WR)")
    print(f"Test:  {len(X_test):,}  rows  ({y_test.mean():.1%} WR)\n")

    # Sigmoid calibration preferred over isotonic for smaller datasets.
    base  = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
    model = CalibratedClassifierCV(base, method="sigmoid", cv=5)
    model.fit(X_train, y_train)

    # --- Validation metrics ---
    probs_test = model.predict_proba(X_test)[:, 1]
    brier      = brier_score_loss(y_test, probs_test)
    baseline   = float(y_test.mean() * (1 - y_test.mean()))
    acc        = float(((probs_test >= 0.5) == y_test).mean())

    print("=" * 55)
    print("Validation metrics (held-out test set)")
    print("=" * 55)
    print(f"  Brier score : {brier:.4f}  (baseline {baseline:.4f}, lower is better)")
    print(f"  Accuracy    : {acc:.1%}")
    print(f"  Test WR     : {y_test.mean():.1%}  Pred avg: {probs_test.mean():.1%}")

    # --- Calibration curve ---
    try:
        fraction_pos, mean_pred = calibration_curve(y_test, probs_test, n_bins=10)
        print(f"\n  Calibration curve (predicted → actual fraction positive):")
        print(f"  {'pred bin':>10}  {'actual':>8}  {'gap':>8}")
        for mp, fp in zip(mean_pred, fraction_pos):
            print(f"  {mp:10.2f}  {fp:8.2f}  {fp - mp:+8.2f}")
    except ValueError:
        print("  (calibration curve requires ≥2 bins with data)")

    # --- Feature coefficients ---
    try:
        coefs = np.mean([est.calibrated_classifiers_[0].estimator.coef_[0]
                         for est in model.calibrated_classifiers_], axis=0)
        print(f"\n  Feature coefficients (positive = predicts win):")
        pairs = sorted(zip(FEATURES, coefs), key=lambda x: -abs(x[1]))
        for feat, coef in pairs:
            bar = "+" * min(20, int(abs(coef) * 5)) if coef > 0 else "-" * min(20, int(abs(coef) * 5))
            print(f"  {feat:>20s}  {coef:+.3f}  {bar}")
    except (AttributeError, IndexError):
        print("  (could not extract LR coefficients from calibrated model)")

    # --- Save ---
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES}, OUTPUT)
    print(f"\nModel saved to {OUTPUT}")

    # --- Sanity checks ---
    no_ask = 67
    import math
    cases = [
        ("Strong: ECMWF+ICON+GEM all +2°F, Jul",  [2.0, 2.0, 2.0, 0.0, 2.0, 3, 1, 1,
                                                     math.sin(2*math.pi*7/12), math.cos(2*math.pi*7/12)]),
        ("Weak:   ECMWF+ICON -1°F (other side), Jan", [-1.0, -1.0, -1.0, 0.0, 1.5, 0, 1, 1,
                                                         math.sin(2*math.pi*1/12), math.cos(2*math.pi*1/12)]),
        ("Mid:    ECMWF +1°F only, HRRR 0, Apr",  [1.0, 0.0, 0.0, 0.0, 1.5, 1, 1, 1,
                                                     math.sin(2*math.pi*4/12), math.cos(2*math.pi*4/12)]),
        ("HRRR confirm: +2°F HRRR, others +1°F",  [1.0, 1.0, 1.0, 2.0, 1.5, 4, 1, 1,
                                                     math.sin(2*math.pi*6/12), math.cos(2*math.pi*6/12)]),
    ]
    print(f"\n{'Signal type':<52}  P(win)  edge@{no_ask}¢")
    for label, vals in cases:
        x = np.array([vals])
        p = float(model.predict_proba(x)[0, 1])
        print(f"  {label:<52}  {p:.3f}   {p - no_ask/100:+.3f}")


if __name__ == "__main__":
    main()
