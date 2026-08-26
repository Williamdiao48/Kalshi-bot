"""
Train the temperature-delta forecast model.

The model predicts delta_to_final = actual_final - running_obs_at_hour:
  High markets: how much more the daily max will rise today (>= 0)
  Low markets:  how much more the daily min will fall today (<= 0)

At inference, P(NO wins) = P(running_obs + delta > band_ceil)
                         = P(delta > band_ceil - running_obs)

Computed using the model's predicted mean delta and a fitted per-(is_high, month)
residual standard deviation (Gaussian approximation).

Outputs:
  data/models/delta_model.pkl          — LightGBM regressor + metadata
  data/models/delta_residual_std.json  — {"{is_high}_{month}": std} lookup

Usage:
  venv/bin/python scripts/train_delta_model.py
  venv/bin/python scripts/train_delta_model.py --high-only
  venv/bin/python scripts/train_delta_model.py --start 2026-03-27  # spring-only comparison run
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from math import sqrt
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_CSV  = Path("data/backtest/delta_training_data.csv")
OUT_PKL   = Path("data/models/delta_model.pkl")
OUT_STD   = Path("data/models/delta_residual_std.json")

FEATURES = [
    "running_obs",
    "delta_1h",
    "delta_2h",
    "hours_to_close",
    "hrrr_remaining",
    "gfs_remaining",
    "consensus_remaining",
    "model_spread",
    "obs_vs_hrrr_h",
    "obs_vs_gfs_h",
    "recent_hrrr_mae_7d",
    "clim_p25",
    "clim_p50",
    "clim_p75",
    "city_enc",
    "is_high",
    "month",
]


def load_data(src: Path, start: str | None, end: str | None, high_only: bool, low_only: bool):
    rows = list(csv.DictReader(src.open()))
    if start:
        rows = [r for r in rows if r["date"] >= start]
    if end:
        rows = [r for r in rows if r["date"] <= end]
    if high_only:
        rows = [r for r in rows if r["is_high"] == "1"]
    elif low_only:
        rows = [r for r in rows if r["is_high"] == "0"]

    print(f"Loaded {len(rows):,} rows "
          f"(high={sum(1 for r in rows if r['is_high']=='1'):,}  "
          f"low={sum(1 for r in rows if r['is_high']=='0'):,})")

    all_cities = sorted(set(r["city"] for r in rows if r.get("city")))
    city_map   = {c: i for i, c in enumerate(all_cities)}
    print(f"Cities: {len(city_map)}  ({', '.join(all_cities)})")

    X_list, y_list, dates, skipped = [], [], [], 0
    for r in rows:
        try:
            cons = float(r["consensus_remaining"])
            X_list.append([
                float(r["running_obs"]),
                float(r["delta_1h"])          if r.get("delta_1h")          else 0.0,
                float(r["delta_2h"])          if r.get("delta_2h")          else 0.0,
                float(r["hours_to_close"]),
                float(r["hrrr_remaining"])    if r.get("hrrr_remaining")    else cons,
                float(r["gfs_remaining"])     if r.get("gfs_remaining")     else cons,
                cons,
                float(r["model_spread"]),
                float(r["obs_vs_hrrr_h"])     if r.get("obs_vs_hrrr_h")     else 0.0,
                float(r["obs_vs_gfs_h"])      if r.get("obs_vs_gfs_h")      else 0.0,
                float(r.get("recent_hrrr_mae_7d") or 3.0),
                float(r["clim_p25"]),
                float(r["clim_p50"]),
                float(r["clim_p75"]),
                float(city_map.get(r.get("city", ""), 0)),
                float(r["is_high"]),
                float(r["month"]),
            ])
            y_list.append(float(r["delta_to_final"]))
            dates.append(r["date"])
        except (ValueError, KeyError):
            skipped += 1

    if skipped:
        print(f"Skipped {skipped} malformed rows")

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32), dates, city_map


def fit_residual_stds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rows_meta: list[dict],
) -> dict[str, float]:
    """
    Fit per-(is_high, month) residual standard deviation.
    Key: "{is_high}_{month}" (e.g. "1_6" = high market, June)
    """
    buckets: dict[str, list[float]] = defaultdict(list)
    for true, pred, r in zip(y_true, y_pred, rows_meta):
        key = f"{r['is_high']}_{r['month']}"
        buckets[key].append(float(true) - float(pred))

    stds = {}
    for key, residuals in buckets.items():
        n = len(residuals)
        mean = sum(residuals) / n
        var  = sum((x - mean) ** 2 for x in residuals) / max(n - 1, 1)
        stds[key] = round(sqrt(var), 3)

    # Print summary
    print("\nResidual std by (is_high, month):")
    for is_high_str in ("1", "0"):
        label = "HIGH" if is_high_str == "1" else "LOW "
        for m in range(1, 13):
            key = f"{is_high_str}_{m}"
            if key in stds:
                n = len(buckets[key])
                print(f"  {label} month={m:>2}: std={stds[key]:.2f}°F  n={n:,}")

    return stds


def train(
    start: str | None,
    end: str | None,
    high_only: bool,
    low_only: bool,
) -> None:
    import lightgbm as lgb
    import pickle

    X, y, dates, city_map = load_data(DATA_CSV, start, end, high_only, low_only)

    if len(X) == 0:
        print("No data loaded.")
        return

    # Chronological train/test split — hold out last 20% of dates
    unique_dates = sorted(set(dates))
    cutoff = unique_dates[int(len(unique_dates) * 0.8)]
    print(f"\nTrain: up to {cutoff}   Test: {cutoff} → {unique_dates[-1]}")

    tr_idx = [i for i, d in enumerate(dates) if d <= cutoff]
    te_idx = [i for i, d in enumerate(dates) if d > cutoff]
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xte, yte = X[te_idx], y[te_idx]
    print(f"Train: {len(Xtr):,}   Test: {len(Xte):,}")

    print("\nTraining LightGBM regressor (MAE objective)...")
    model = lgb.LGBMRegressor(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        objective="regression_l1",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(
        Xtr, ytr,
        eval_set=[(Xte, yte)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
        feature_name=FEATURES,
    )

    pred_te = model.predict(Xte)
    mae  = float(np.mean(np.abs(pred_te - yte)))
    rmse = float(np.sqrt(np.mean((pred_te - yte) ** 2)))
    print(f"\nTest MAE:  {mae:.3f}°F")
    print(f"Test RMSE: {rmse:.3f}°F")

    # Feature importances
    print("\nTop feature importances:")
    for i in np.argsort(model.feature_importances_)[::-1][:10]:
        print(f"  {FEATURES[i]:<28} {model.feature_importances_[i]:.0f}")

    # Residual stds — reload all rows for meta
    all_rows = list(csv.DictReader(DATA_CSV.open()))
    if start:
        all_rows = [r for r in all_rows if r["date"] >= start]
    if end:
        all_rows = [r for r in all_rows if r["date"] <= end]
    if high_only:
        all_rows = [r for r in all_rows if r["is_high"] == "1"]
    elif low_only:
        all_rows = [r for r in all_rows if r["is_high"] == "0"]

    # Only fit residual stds on training set (no test leakage)
    train_meta = [r for r in all_rows if r["date"] <= cutoff]
    y_pred_tr = model.predict(Xtr)
    stds = fit_residual_stds(ytr, y_pred_tr, train_meta)

    # Verify: how well do the stds cover test set residuals?
    residuals_te = pred_te - yte
    print(f"\nTest residual range: {residuals_te.min():+.2f}  to  {residuals_te.max():+.2f}°F")
    print(f"Test residual std:   {float(np.std(residuals_te)):.3f}°F")

    # Show calibration: P(delta > threshold) vs actual win rate at various margins
    print("\nCalibration check — P(NO wins | margin to ceiling):")
    print("(For a live market: if running_obs=78, band_ceil=76, margin_to_ceiling = -2.0)")
    from scipy.stats import norm
    for margin in (-4, -2, -1, 0, 1, 2, 4):
        # Compute P using test set: how often does delta > margin?
        actual_exceed = float(np.mean(yte > margin))
        # Model-based: P(delta > margin) using predicted mean + per-row std lookup
        probs = []
        for pred, is_hi_str, mon_str in zip(
            pred_te,
            [all_rows[i]["is_high"] for i in te_idx],
            [all_rows[i]["month"]   for i in te_idx],
        ):
            std = stds.get(f"{is_hi_str}_{mon_str}", 3.0)
            probs.append(1.0 - norm.cdf(margin, loc=pred, scale=std))
        model_p = float(np.mean(probs))
        print(f"  margin={margin:+.0f}°F  actual={100*actual_exceed:.1f}%  model_p={100*model_p:.1f}%")

    Path("data/models").mkdir(parents=True, exist_ok=True)

    pkg = {
        "model":    model,
        "features": FEATURES,
        "city_map": city_map,
        "training_range": (
            start or unique_dates[0],
            end   or unique_dates[-1],
        ),
        "test_mae":  round(mae,  3),
        "test_rmse": round(rmse, 3),
    }
    OUT_PKL.write_bytes(pickle.dumps(pkg))
    print(f"\nSaved model → {OUT_PKL}")

    OUT_STD.write_text(json.dumps(stds, indent=2))
    print(f"Saved residual stds → {OUT_STD}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",     default=None,   help="filter training start date (YYYY-MM-DD)")
    parser.add_argument("--end",       default=None,   help="filter training end date")
    parser.add_argument("--high-only", action="store_true")
    parser.add_argument("--low-only",  action="store_true")
    args = parser.parse_args()
    train(args.start, args.end, args.high_only, args.low_only)
