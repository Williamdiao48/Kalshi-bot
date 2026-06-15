"""
Train a band_arb YES signal model on historical temperature data.

Input:  data/backtest/band_arb_training.csv  (from build_band_arb_training_data.py)
Output: data/models/band_arb_yes_model.pkl   (LightGBM model + metadata)

The model predicts P(YES wins) given features available at entry time:
  - Where the running obs sits relative to the band (margin_to_ceil)
  - What model forecasts say relative to the band ceiling (consensus_vs_ceil etc.)
  - Inter-model disagreement (model_spread, n_models_above_ceil)
  - Season and market type (month, is_high)

P(YES wins) at runtime is compared against the market's YES ask price to
compute edge = model_P - market_price. Only trade when edge > threshold.

Usage:
  venv/bin/python scripts/train_band_arb_yes_model.py
  venv/bin/python scripts/train_band_arb_yes_model.py --high-only
  venv/bin/python scripts/train_band_arb_yes_model.py --low-only
"""

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb

DATA_CSV  = Path("data/backtest/band_arb_training.csv")
MODEL_OUT = Path("data/models/band_arb_yes_model.pkl")

# Features used at training and inference time.
# All relative (no absolute temperatures) so the model generalises across cities.
FEATURES = [
    # Boundary-relative features use the RISK boundary for each market type:
    #   KXHIGH: risk is temp rising above band_ceil → use vs_ceil
    #   KXLOWT: risk is temp dropping below band_lo  → use vs_floor
    # We include both so the model can learn which is relevant per is_high.
    "consensus_vs_ceil",    # consensus - band_ceil (KXHIGH risk boundary)
    "hrrr_vs_ceil",         # HRRR - band_ceil
    "gfs_vs_ceil",          # GFS - band_ceil
    "consensus_vs_floor",   # consensus - band_lo (KXLOWT risk boundary)
    "hrrr_vs_floor",        # HRRR - band_lo
    "gfs_vs_floor",         # GFS - band_lo
    "ecmwf_vs_ceil",        # ECMWF - band_ceil (computed from ecmwf_f - band_ceil)
    "model_spread",         # max - min across all available models
    "n_models_above_ceil",  # models predicting above band_ceil
    "n_models_below_floor", # models predicting below band_lo
    "n_models",             # total available models
    "is_high",              # 1 = KXHIGH, 0 = KXLOWT
    "month",                # 1-12: seasonal patterns
]


def load_data(path: Path, high_only: bool, low_only: bool) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load CSV, compute derived features, return (X, y, dates)."""
    rows = list(csv.DictReader(path.open()))

    if high_only:
        rows = [r for r in rows if r["is_high"] == "1"]
    elif low_only:
        rows = [r for r in rows if r["is_high"] == "0"]

    print(f"Loaded {len(rows):,} rows  (high={sum(1 for r in rows if r['is_high']=='1'):,} "
          f"low={sum(1 for r in rows if r['is_high']=='0'):,})")

    X_list, y_list, dates = [], [], []

    for r in rows:
        try:
            band_ceil      = float(r["band_ceil"])
            band_lo        = float(r["band_lo"])
            consensus_vs_c = float(r["consensus_vs_ceil"])
            hrrr_vs_c      = float(r["hrrr_vs_ceil"])   if r["hrrr_vs_ceil"]   else consensus_vs_c
            gfs_vs_c       = float(r["gfs_vs_ceil"])    if r["gfs_vs_ceil"]    else consensus_vs_c
            consensus_vs_f = float(r["consensus_vs_floor"])
            hrrr_vs_f      = float(r["hrrr_vs_floor"])  if r["hrrr_vs_floor"]  else consensus_vs_f
            gfs_vs_f       = float(r["gfs_vs_floor"])   if r["gfs_vs_floor"]   else consensus_vs_f
            ecmwf_f        = float(r["ecmwf_f"])         if r["ecmwf_f"]        else None
            ecmwf_vs_c     = (ecmwf_f - band_ceil)       if ecmwf_f is not None else consensus_vs_c
            model_spread   = float(r["model_spread"])
            n_above        = float(r["n_models_above_ceil"])
            n_below        = float(r["n_models_below_floor"])
            n_models       = float(r["n_models"])
            is_high        = float(r["is_high"])
            month          = float(r["month"])
            won            = int(r["won"])
        except (ValueError, KeyError):
            continue

        X_list.append([
            consensus_vs_c,
            hrrr_vs_c,
            gfs_vs_c,
            consensus_vs_f,
            hrrr_vs_f,
            gfs_vs_f,
            ecmwf_vs_c,
            model_spread,
            n_above,
            n_below,
            n_models,
            is_high,
            month,
        ])
        y_list.append(won)
        dates.append(r["date"])

    return np.array(X_list), np.array(y_list), dates


def chronological_split(X, y, dates, test_frac=0.20):
    """Split by date to avoid look-ahead bias."""
    unique_dates = sorted(set(dates))
    cutoff_idx   = int(len(unique_dates) * (1 - test_frac))
    cutoff_date  = unique_dates[cutoff_idx]
    train_mask = np.array([d < cutoff_date for d in dates])
    test_mask  = ~train_mask
    print(f"Train: {train_mask.sum():,} rows (before {cutoff_date})")
    print(f"Test:  {test_mask.sum():,} rows (from  {cutoff_date})")
    return X[train_mask], X[test_mask], y[train_mask], y[test_mask]


def eval_model(name, y_true, y_prob):
    brier = brier_score_loss(y_true, y_prob)
    auc   = roc_auc_score(y_true, y_prob)
    base  = brier_score_loss(y_true, np.full_like(y_prob, y_true.mean()))
    print(f"  {name:<28} Brier={brier:.4f} (baseline={base:.4f})  AUC={auc:.4f}")
    return brier, auc


def print_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        imps = model.feature_importances_
    elif hasattr(model, "coef_"):
        imps = np.abs(model.coef_[0])
    else:
        return
    order = np.argsort(imps)[::-1]
    print("\n  Feature importances:")
    for i in order:
        print(f"    {feature_names[i]:<28} {imps[i]:.4f}")


def print_calibration(name, y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    print(f"\n  Calibration ({name}):")
    print(f"  {'Pred':>8}  {'Actual':>8}  {'Δ':>8}")
    for pt, pp in zip(prob_pred, prob_true):
        delta = pt - pp
        bar = "▲" if delta > 0.03 else ("▼" if delta < -0.03 else "·")
        print(f"  {pt:>8.2f}  {pp:>8.2f}  {delta:>+7.3f} {bar}")


def win_rate_by_bucket(y_true, y_prob, feature_vals, feature_name, bins):
    """Print WR and model P by feature bucket."""
    print(f"\n  WR by {feature_name}:")
    print(f"  {'Bucket':>12}  {'n':>6}  {'WR':>7}  {'Avg P':>7}  {'Edge':>7}")
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (feature_vals >= lo) & (feature_vals < hi)
        if mask.sum() < 5:
            continue
        wr   = y_true[mask].mean()
        avgp = y_prob[mask].mean()
        print(f"  [{lo:+.1f},{hi:+.1f})  {mask.sum():>6}  {wr:>6.1%}  {avgp:>6.1%}  {avgp-wr:>+6.3f}")


def main(high_only: bool, low_only: bool) -> None:
    X, y, dates = load_data(DATA_CSV, high_only, low_only)
    print(f"Overall WR: {y.mean():.1%}\n")

    X_tr, X_te, y_tr, y_te = chronological_split(X, y, dates)

    # ── 1. Logistic Regression (interpretable baseline) ────────────────────
    print("\n=== Logistic Regression ===")
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr",     LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
    ])
    lr_pipe.fit(X_tr, y_tr)
    lr_prob_te = lr_pipe.predict_proba(X_te)[:, 1]
    eval_model("LogReg", y_te, lr_prob_te)

    coefs = lr_pipe.named_steps["lr"].coef_[0]
    print("\n  Coefficients (+ = increases P(YES wins)):")
    order = np.argsort(np.abs(coefs))[::-1]
    for i in order:
        print(f"    {FEATURES[i]:<28} {coefs[i]:>+.4f}")

    # ── 2. LightGBM ────────────────────────────────────────────────────────
    print("\n=== LightGBM ===")
    lgbm = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
    )
    # Train on first 80% of train set; calibrate on remaining 20%
    cal_split = int(len(X_tr) * 0.8)
    lgbm.fit(X_tr[:cal_split], y_tr[:cal_split], feature_name=FEATURES)
    lgbm_prob_te = lgbm.predict_proba(X_te)[:, 1]
    eval_model("LightGBM (raw)", y_te, lgbm_prob_te)
    print_feature_importance(lgbm, FEATURES)

    # Isotonic calibration fitted on held-out 20% of train
    raw_cal = lgbm.predict_proba(X_tr[cal_split:])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_cal, y_tr[cal_split:])
    lgbm_cal_prob = iso.predict(lgbm_prob_te)
    brier_cal, auc_cal = eval_model("LightGBM (calibrated)", y_te, lgbm_cal_prob)

    print_calibration("LightGBM calibrated", y_te, lgbm_cal_prob)

    # ── 3. Analysis: edge vs actual WR ────────────────────────────────────
    print("\n=== Edge analysis (LightGBM calibrated) ===")
    win_rate_by_bucket(
        y_te, lgbm_cal_prob,
        X_te[:, FEATURES.index("consensus_vs_ceil")], "consensus_vs_ceil",
        bins=[-10, -3, -2, -1, 0, 1, 2, 10],
    )
    win_rate_by_bucket(
        y_te, lgbm_cal_prob,
        X_te[:, FEATURES.index("hrrr_vs_ceil")], "hrrr_vs_ceil",
        bins=[-10, -3, -2, -1, 0, 1, 2, 10],
    )
    if not high_only:
        win_rate_by_bucket(
            y_te, lgbm_cal_prob,
            X_te[:, FEATURES.index("consensus_vs_floor")], "consensus_vs_floor",
            bins=[-10, -3, -2, -1, 0, 1, 2, 10],
        )
        win_rate_by_bucket(
            y_te, lgbm_cal_prob,
            X_te[:, FEATURES.index("hrrr_vs_floor")], "hrrr_vs_floor",
            bins=[-10, -3, -2, -1, 0, 1, 2, 10],
        )

    # ── 4. Save best model ─────────────────────────────────────────────────
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "lgbm":         lgbm,
        "isotonic":     iso,
        "features":     FEATURES,
        "brier":        brier_cal,
        "auc":          auc_cal,
        "base_wr":      float(y.mean()),
        "high_only":    high_only,
        "low_only":     low_only,
    }
    MODEL_OUT.write_bytes(pickle.dumps(payload))
    print(f"\nSaved → {MODEL_OUT}  (Brier={brier_cal:.4f}, AUC={auc_cal:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--high-only", action="store_true", help="Train only on KXHIGH rows")
    parser.add_argument("--low-only",  action="store_true", help="Train only on KXLOWT rows")
    args = parser.parse_args()
    main(args.high_only, args.low_only)
