"""
Train forecast_no band model with improved features.

New vs previous version:
  - hours_above_ceil: consecutive hours running_obs has been above band_ceil
  - city_enc:         label-encoded city (captures station-discrepancy patterns)
  - obs_vs_hrrr_h:    running_obs minus HRRR's hourly forecast at trade hour
  - obs_vs_gfs_h:     running_obs minus GFS's hourly forecast at trade hour
  - Separate KXHIGH and KXLOWT models (different base WRs, dynamics)

Output:
  data/models/forecast_no_band_model.pkl       (combined, for backward compat)
  data/models/forecast_no_band_model_high.pkl  (KXHIGH only)
  data/models/forecast_no_band_model_low.pkl   (KXLOWT only)

Usage:
  venv/bin/python scripts/train_forecast_no_band_model.py
  venv/bin/python scripts/train_forecast_no_band_model.py --high-only
  venv/bin/python scripts/train_forecast_no_band_model.py --low-only
"""

import argparse
import csv
import pickle
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

DATA_CSV         = Path("data/backtest/forecast_no_training_data.csv")
DATA_CSV_KALSHI  = Path("data/backtest/forecast_no_training_data_kalshi.csv")
MODEL_OUT   = Path("data/models/forecast_no_band_model.pkl")
MODEL_HIGH  = Path("data/models/forecast_no_band_model_high.pkl")
MODEL_LOW   = Path("data/models/forecast_no_band_model_low.pkl")

FEATURES = [
    "margin_f",           # running_obs - band_ceil
    "delta_1h",           # running_obs change last 1h
    "delta_2h",           # running_obs change last 2h
    "hours_above_ceil",   # consecutive hours signal has been active
    "hour_utc",
    "hours_to_close",
    "obs_vs_hrrr_h",      # running_obs - HRRR hourly forecast at this hour
    "obs_vs_gfs_h",       # running_obs - GFS hourly forecast at this hour
    "hrrr_vs_ceil",       # HRRR daily - band_ceil
    "gfs_vs_ceil",        # GFS daily - band_ceil
    "consensus_vs_ceil",  # median daily forecast - band_ceil
    "model_spread",
    "n_models_above_ceil",
    "recent_hrrr_mae_7d",  # rolling 7-day mean |HRRR - actual| for this city
    "clim_prob_exceed",    # P(further_drop > margin_f | city, month, hour) from 4yr METAR
    "clim_drop_p50",       # median expected additional cooling
    "clim_drop_p75",       # 75th pct expected additional cooling
    "city_enc",           # label-encoded city
    "is_high",
    "month",
]

# Categorical feature indices for LightGBM (city_enc)
CATEGORICAL_FEATURES = ["city_enc"]


def load_data(high_only: bool, low_only: bool, use_kalshi: bool = False):
    src = DATA_CSV_KALSHI if use_kalshi else DATA_CSV
    rows = list(csv.DictReader(src.open()))
    if high_only:
        rows = [r for r in rows if r["is_high"] == "1"]
    elif low_only:
        rows = [r for r in rows if r["is_high"] == "0"]

    print(f"Loaded {len(rows):,} rows  "
          f"(high={sum(1 for r in rows if r['is_high']=='1'):,}  "
          f"low={sum(1 for r in rows if r['is_high']=='0'):,})")

    # Build city encoder from all unique cities in dataset
    all_cities = sorted(set(r["city"] for r in rows if r.get("city")))
    city_map   = {c: i for i, c in enumerate(all_cities)}
    print(f"Cities: {len(city_map)}  ({', '.join(all_cities[:5])}...)")

    X_list, y_list, dates, skipped = [], [], [], 0
    for r in rows:
        try:
            c    = float(r["consensus_vs_ceil"])
            city = city_map.get(r.get("city", ""), 0)
            X_list.append([
                float(r["margin_f"]),
                float(r["delta_1h"])         if r.get("delta_1h")         else 0.0,
                float(r["delta_2h"])         if r.get("delta_2h")         else 0.0,
                float(r.get("hours_above_ceil", 1)),
                float(r["hour_utc"]),
                float(r["hours_to_close"]),
                float(r["obs_vs_hrrr_h"])    if r.get("obs_vs_hrrr_h")    else 0.0,
                float(r["obs_vs_gfs_h"])     if r.get("obs_vs_gfs_h")     else 0.0,
                float(r["hrrr_vs_ceil"])     if r.get("hrrr_vs_ceil")     else c,
                float(r["gfs_vs_ceil"])      if r.get("gfs_vs_ceil")      else c,
                c,
                float(r["model_spread"]),
                float(r["n_models_above_ceil"]),
                float(r.get("recent_hrrr_mae_7d") or 3.0),
                float(r.get("clim_prob_exceed") or 0.15),
                float(r.get("clim_drop_p50") or 2.0),
                float(r.get("clim_drop_p75") or 3.0),
                float(city),
                float(r["is_high"]),
                float(r["month"]),
            ])
            y_list.append(int(r["won"]))
            dates.append(r["date"])
        except (ValueError, KeyError):
            skipped += 1
    if skipped:
        print(f"Skipped {skipped} malformed rows")
    return np.array(X_list), np.array(y_list), dates, city_map


def split(X, y, dates, frac=0.20, random=False):
    if random:
        rng  = np.random.default_rng(42)
        perm = rng.permutation(len(y))
        n_te = int(len(y) * frac)
        te   = np.zeros(len(y), dtype=bool)
        te[perm[:n_te]] = True
        tr   = ~te
        print(f"Train: {tr.sum():,}  Test: {te.sum():,}  (random 80/20)")
        return X[tr], X[te], y[tr], y[te]
    ud  = sorted(set(dates))
    cut = ud[int(len(ud) * (1 - frac))]
    tr  = np.array([d < cut for d in dates])
    print(f"Train: {tr.sum():,}  Test: {(~tr).sum():,}  cutoff={cut}")
    return X[tr], X[~tr], y[tr], y[~tr]


def ev(name, yt, yp):
    b    = brier_score_loss(yt, yp)
    base = brier_score_loss(yt, np.full_like(yp, yt.mean()))
    a    = roc_auc_score(yt, yp)
    print(f"  {name:<32} Brier={b:.4f} (base={base:.4f})  AUC={a:.4f}")
    return b, a


def bucket(label, yt, yp, fv, bins):
    print(f"\n  WR by {label}:")
    print(f"  {'Bucket':>14}  {'n':>6}  {'WR':>7}  {'AvgP':>7}  {'Edge':>7}")
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (fv >= lo) & (fv < hi)
        if m.sum() < 10:
            continue
        wr = yt[m].mean(); ap = yp[m].mean()
        print(f"  [{lo:>5.1f},{hi:>5.1f})  {m.sum():>6}  {wr:>6.1%}  {ap:>6.1%}  {ap-wr:>+6.3f}")


def train_one(label: str, X, y, dates, out_path: Path, city_map: dict, random_split: bool = False):
    print(f"\n{'='*60}")
    print(f"Training: {label}")
    print(f"{'='*60}")
    print(f"Overall WR: {y.mean():.1%}")

    Xtr, Xte, ytr, yte = split(X, y, dates, random=random_split)
    cat_idx = [FEATURES.index(f) for f in CATEGORICAL_FEATURES]

    # Logistic Regression baseline
    print("\n--- Logistic Regression ---")
    lr = Pipeline([("sc", StandardScaler()),
                   ("lr", LogisticRegression(C=0.5, max_iter=1000, random_state=42))])
    lr.fit(Xtr, ytr)
    lp = lr.predict_proba(Xte)[:, 1]
    ev("LogReg", yte, lp)
    coefs = lr.named_steps["lr"].coef_[0]
    print("  Top coefficients:")
    for i in np.argsort(np.abs(coefs))[::-1][:8]:
        print(f"    {FEATURES[i]:<28} {coefs[i]:>+.4f}")

    # LightGBM
    print("\n--- LightGBM ---")
    cs   = int(len(Xtr) * 0.8)
    lgbm = lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.04,
        max_depth=6,
        num_leaves=48,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
    )
    lgbm.fit(
        Xtr[:cs], ytr[:cs],
        feature_name=FEATURES,
        categorical_feature=cat_idx,
    )
    rp = lgbm.predict_proba(Xte)[:, 1]
    ev("LightGBM (raw)", yte, rp)

    print("  Feature importances:")
    for i in np.argsort(lgbm.feature_importances_)[::-1]:
        print(f"    {FEATURES[i]:<28} {lgbm.feature_importances_[i]:.0f}")

    # Isotonic calibration on held-out train slice
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(lgbm.predict_proba(Xtr[cs:])[:, 1], ytr[cs:])
    cp = iso.predict(rp)
    brier, auc = ev("LightGBM (calibrated)", yte, cp)

    # Calibration curve
    pt, pp = calibration_curve(yte, cp, n_bins=10)
    print("\n  Calibration:")
    for a, b in zip(pt, pp):
        d = a - b
        f = "▲" if d > 0.03 else ("▼" if d < -0.03 else "·")
        print(f"    pred={a:.2f}  actual={b:.2f}  Δ={d:+.3f} {f}")

    # Edge analysis
    print("\n  Edge analysis:")
    fi = {f: i for i, f in enumerate(FEATURES)}
    bucket("hours_above_ceil", yte, cp, Xte[:, fi["hours_above_ceil"]], [0,1,2,3,5,10,20])
    bucket("obs_vs_hrrr_h",    yte, cp, Xte[:, fi["obs_vs_hrrr_h"]],   [-5,-2,-1,0,1,2,5,15])
    bucket("hrrr_vs_ceil",     yte, cp, Xte[:, fi["hrrr_vs_ceil"]],    [-5,-2,-1,0,1,2,5,15])
    bucket("consensus_vs_ceil",yte, cp, Xte[:, fi["consensus_vs_ceil"]],[-5,-2,-1,0,1,2,5,15])
    bucket("delta_1h",         yte, cp, Xte[:, fi["delta_1h"]],        [-3,-1,-0.1,0.1,1,3,10])
    bucket("hour_utc",         yte, cp, Xte[:, fi["hour_utc"]],        [4,8,10,12,14,16,18,23])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(pickle.dumps({
        "lgbm":     lgbm,
        "isotonic": iso,
        "features": FEATURES,
        "brier":    brier,
        "auc":      auc,
        "base_wr":  float(y.mean()),
        "city_map": city_map,
    }))
    print(f"\nSaved → {out_path}  (Brier={brier:.4f}, AUC={auc:.4f})")
    return lgbm, iso, brier, auc


def main(high_only: bool, low_only: bool, use_kalshi: bool = False, random_split: bool = False) -> None:
    X, y, dates, city_map = load_data(high_only, low_only, use_kalshi)

    if high_only:
        train_one("KXHIGH only", X, y, dates, MODEL_HIGH, city_map, random_split)
    elif low_only:
        train_one("KXLOWT only", X, y, dates, MODEL_LOW, city_map, random_split)
    else:
        train_one("Combined (KXHIGH + KXLOWT)", X, y, dates, MODEL_OUT, city_map, random_split)
        X_h, y_h, d_h, _ = load_data(high_only=True,  low_only=False, use_kalshi=use_kalshi)
        X_l, y_l, d_l, _ = load_data(high_only=False, low_only=True,  use_kalshi=use_kalshi)
        train_one("KXHIGH only", X_h, y_h, d_h, MODEL_HIGH, city_map, random_split)
        train_one("KXLOWT only", X_l, y_l, d_l, MODEL_LOW,  city_map, random_split)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--high-only",     action="store_true")
    parser.add_argument("--low-only",      action="store_true")
    parser.add_argument("--kalshi",        action="store_true", help="Use real Kalshi band training data")
    parser.add_argument("--random-split",  action="store_true", help="Random 80/20 split instead of chronological")
    args = parser.parse_args()
    main(args.high_only, args.low_only, args.kalshi, args.random_split)
