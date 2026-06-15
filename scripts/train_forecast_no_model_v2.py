"""
Train v2 forecast_no band model with three new alpha features:
  - hrrr_skill_adj       (HRRR confidence adjusted for recent forecast error)
  - min_model_vs_ceil    (conservative consensus floor — min of HRRR and GFS)
  - margin_per_hour_left (safety buffer per unit of remaining risk time)

Reads from: data/backtest/forecast_no_training_data_kalshi_v2.csv
Writes to:
  data/models/forecast_no_band_model_v2_high.pkl
  data/models/forecast_no_band_model_v2_low.pkl

Does NOT overwrite the existing v1 models.

Usage:
  venv/bin/python scripts/train_forecast_no_model_v2.py
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

DATA_CSV   = Path("data/backtest/forecast_no_training_data_kalshi_v2.csv")
MODEL_HIGH = Path("data/models/forecast_no_band_model_v2_high.pkl")
MODEL_LOW  = Path("data/models/forecast_no_band_model_v2_low.pkl")

# All v1 features plus the three new alpha features
FEATURES = [
    # ── existing v1 features ──────────────────────────────────────────
    "margin_f",             # running_obs - band_ceil
    "delta_1h",             # temp change last 1h
    "delta_2h",             # temp change last 2h
    "hours_above_ceil",     # consecutive hours signal has been active
    "hour_utc",
    "hours_to_close",
    "obs_vs_hrrr_h",        # running_obs - HRRR hourly fc at this hour
    "obs_vs_gfs_h",         # running_obs - GFS hourly fc at this hour
    "hrrr_vs_ceil",         # HRRR daily - band_ceil
    "gfs_vs_ceil",          # GFS daily - band_ceil
    "consensus_vs_ceil",    # median daily forecast - band_ceil
    "model_spread",
    "n_models_above_ceil",
    "recent_hrrr_mae_7d",
    "clim_prob_exceed",
    "clim_drop_p50",
    "clim_drop_p75",
    "city_enc",
    "is_high",
    "month",
    # ── new v2 alpha features ─────────────────────────────────────────
    "hrrr_skill_adj",       # hrrr_vs_ceil / (recent_hrrr_mae_7d + 0.5)
    "min_model_vs_ceil",    # min(hrrr_vs_ceil, gfs_vs_ceil)
    "margin_per_hour_left", # margin_f / (hours_to_close + 1)
    "hour_utc_x_is_high",  # hour_utc * is_high: hour 22 is risky for HIGH, safe for LOW
]

CATEGORICAL_FEATURES = ["city_enc"]


def load_data(high_only: bool, low_only: bool):
    if not DATA_CSV.exists():
        raise FileNotFoundError(
            f"{DATA_CSV} not found — run scripts/build_training_data_v2.py first"
        )
    rows = list(csv.DictReader(DATA_CSV.open()))
    if high_only:
        rows = [r for r in rows if r["is_high"] == "1"]
    elif low_only:
        rows = [r for r in rows if r["is_high"] == "0"]

    print(f"Loaded {len(rows):,} rows  "
          f"(high={sum(1 for r in rows if r['is_high']=='1'):,}  "
          f"low={sum(1 for r in rows if r['is_high']=='0'):,})")

    all_cities = sorted(set(r["city"] for r in rows if r.get("city")))
    city_map   = {c: i for i, c in enumerate(all_cities)}
    print(f"Cities ({len(city_map)}): {', '.join(all_cities)}")

    def f(r, k, default=0.0):
        v = r.get(k)
        try:
            return float(v) if v not in (None, "", "nan") else default
        except (ValueError, TypeError):
            return default

    X_list, y_list, dates, skipped = [], [], [], 0
    for r in rows:
        try:
            cons   = f(r, "consensus_vs_ceil")
            hrrr   = f(r, "hrrr_vs_ceil", cons)
            gfs    = f(r, "gfs_vs_ceil", cons)
            mae    = f(r, "recent_hrrr_mae_7d", 3.0)
            margin = f(r, "margin_f")
            hrs    = f(r, "hours_to_close")
            city   = float(city_map.get(r.get("city", ""), 0))

            hour_utc  = f(r, "hour_utc")
            is_high_v = f(r, "is_high")
            X_list.append([
                margin,
                f(r, "delta_1h"),
                f(r, "delta_2h"),
                f(r, "hours_above_ceil", 1.0),
                hour_utc,
                hrs,
                f(r, "obs_vs_hrrr_h"),
                f(r, "obs_vs_gfs_h"),
                hrrr,
                gfs,
                cons,
                f(r, "model_spread"),
                f(r, "n_models_above_ceil"),
                mae,
                f(r, "clim_prob_exceed", 0.15),
                f(r, "clim_drop_p50", 2.0),
                f(r, "clim_drop_p75", 3.0),
                city,
                is_high_v,
                f(r, "month"),
                # v2 features — fall back to derived values if missing from CSV
                f(r, "hrrr_skill_adj",       hrrr / (mae + 0.5)),
                f(r, "min_model_vs_ceil",     min(hrrr, gfs)),
                f(r, "margin_per_hour_left",  margin / (hrs + 1)),
                f(r, "hour_utc_x_is_high",    hour_utc * is_high_v),
            ])
            y_list.append(int(r["won"]))
            dates.append(r["date"])
        except (ValueError, KeyError):
            skipped += 1

    if skipped:
        print(f"Skipped {skipped} malformed rows")
    return np.array(X_list), np.array(y_list), dates, city_map


def split_chronological(X, y, dates, frac=0.20):
    ud  = sorted(set(dates))
    cut = ud[int(len(ud) * (1 - frac))]
    tr  = np.array([d < cut for d in dates])
    print(f"Train: {tr.sum():,}  Test: {(~tr).sum():,}  cutoff={cut}")
    return X[tr], X[~tr], y[tr], y[~tr]


def ev(name, yt, yp):
    b    = brier_score_loss(yt, yp)
    base = brier_score_loss(yt, np.full_like(yp, yt.mean()))
    a    = roc_auc_score(yt, yp)
    print(f"  {name:<35} Brier={b:.4f} (base={base:.4f})  AUC={a:.4f}")
    return b, a


def bucket(label, yt, yp, fv, bins):
    print(f"\n  WR by {label}:")
    print(f"  {'Bucket':>15}  {'n':>6}  {'WR':>7}  {'AvgP':>7}  {'Err':>7}")
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (fv >= lo) & (fv < hi)
        if m.sum() < 10:
            continue
        wr = yt[m].mean(); ap = yp[m].mean()
        print(f"  [{lo:>6.2f},{hi:>6.2f})  {m.sum():>6}  {wr:>6.1%}  {ap:>6.1%}  {ap-wr:>+6.3f}")


def compare_v1(label, yt, yp_v2):
    """Try to load the matching v1 model and compare AUC head-to-head."""
    v1_path = {
        "KXHIGH": Path("data/models/forecast_no_band_model_high.pkl"),
        "KXLOWT": Path("data/models/forecast_no_band_model_low.pkl"),
    }.get(label)
    if v1_path is None or not v1_path.exists():
        return
    with v1_path.open("rb") as fh:
        v1 = pickle.load(fh)
    print(f"\n  ── v1 vs v2 AUC comparison ({label}) ──")
    print(f"  v1 AUC (stored): {v1['auc']:.4f}")
    print(f"  v2 AUC (test):   {roc_auc_score(yt, yp_v2):.4f}")


def train_one(label: str, X, y, dates, out_path: Path, city_map: dict):
    print(f"\n{'='*65}")
    print(f"Training v2: {label}")
    print(f"{'='*65}")
    print(f"Rows: {len(y):,}   Overall WR: {y.mean():.1%}")

    Xtr, Xte, ytr, yte = split_chronological(X, y, dates)
    cat_idx = [FEATURES.index(f) for f in CATEGORICAL_FEATURES]

    # Logistic baseline
    print("\n--- Logistic Regression baseline ---")
    lr = Pipeline([("sc", StandardScaler()),
                   ("lr", LogisticRegression(C=0.5, max_iter=1000, random_state=42))])
    lr.fit(Xtr, ytr)
    lp = lr.predict_proba(Xte)[:, 1]
    ev("LogReg", yte, lp)
    coefs = lr.named_steps["lr"].coef_[0]
    print("  Top 10 coefficients:")
    for i in np.argsort(np.abs(coefs))[::-1][:10]:
        print(f"    {FEATURES[i]:<30} {coefs[i]:>+.4f}")

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

    print("\n  Feature importances (top 15):")
    for i in np.argsort(lgbm.feature_importances_)[::-1][:15]:
        bar = "█" * int(lgbm.feature_importances_[i] / max(lgbm.feature_importances_) * 30)
        print(f"    {FEATURES[i]:<30} {lgbm.feature_importances_[i]:>6.0f}  {bar}")

    # Isotonic calibration
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(lgbm.predict_proba(Xtr[cs:])[:, 1], ytr[cs:])
    cp = iso.predict(rp)
    brier, auc = ev("LightGBM (calibrated)", yte, cp)

    # Calibration curve
    pt, pp = calibration_curve(yte, cp, n_bins=10)
    print("\n  Calibration (predicted vs actual):")
    for a, b_ in zip(pt, pp):
        d = a - b_
        flag = "▲" if d > 0.03 else ("▼" if d < -0.03 else "·")
        print(f"    pred={a:.2f}  actual={b_:.2f}  Δ={d:+.3f} {flag}")

    # New feature buckets
    fi = {feat: i for i, feat in enumerate(FEATURES)}
    bucket("hrrr_skill_adj",       yte, cp, Xte[:, fi["hrrr_skill_adj"]],
           [-10, -2, 0, 1, 2, 4, 8, 20])
    bucket("min_model_vs_ceil",    yte, cp, Xte[:, fi["min_model_vs_ceil"]],
           [-10, -2, 0, 1, 2, 4, 8, 20])
    bucket("margin_per_hour_left", yte, cp, Xte[:, fi["margin_per_hour_left"]],
           [0, 0.2, 0.5, 1.0, 2.0, 5.0, 20.0])
    bucket("clim_prob_exceed",     yte, cp, Xte[:, fi["clim_prob_exceed"]],
           [0, 0.05, 0.10, 0.18, 0.35, 1.0])
    bucket("hrrr_vs_ceil",         yte, cp, Xte[:, fi["hrrr_vs_ceil"]],
           [-5, -1, 0, 1, 2, 5, 15])
    bucket("hours_to_close",       yte, cp, Xte[:, fi["hours_to_close"]],
           [0, 2, 4, 6, 9, 12, 18, 24])

    # v1 comparison
    compare_v1(label, yte, cp)

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


def main(high_only: bool, low_only: bool):
    if high_only:
        X, y, dates, city_map = load_data(high_only=True, low_only=False)
        train_one("KXHIGH", X, y, dates, MODEL_HIGH, city_map)
    elif low_only:
        X, y, dates, city_map = load_data(high_only=False, low_only=True)
        train_one("KXLOWT", X, y, dates, MODEL_LOW, city_map)
    else:
        X_h, y_h, d_h, city_map = load_data(high_only=True,  low_only=False)
        X_l, y_l, d_l, _        = load_data(high_only=False, low_only=True)
        train_one("KXHIGH", X_h, y_h, d_h, MODEL_HIGH, city_map)
        train_one("KXLOWT", X_l, y_l, d_l, MODEL_LOW,  city_map)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--high-only", action="store_true")
    parser.add_argument("--low-only",  action="store_true")
    args = parser.parse_args()
    main(args.high_only, args.low_only)
