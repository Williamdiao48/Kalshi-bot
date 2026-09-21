#!/usr/bin/env python3
"""Phase 3 — ahead-of-time HIGH-market model: distribution over the daily high.

Predicts a CALIBRATED PREDICTIVE DISTRIBUTION for the official daily high at the
09:00-local decision time (features from Phase 2), then integrates it over the
integer band edges to give P(daily high rounds to T) for each candidate band.

Model form (data-efficient for ~2k rows, ROADMAP Phase 3 "point + calibrated
uncertainty"):
  mu    = LightGBM L1 regression  -> robust point estimate of the daily high
  sigma = LightGBM regression on |residual| (of mu), scaled by sqrt(pi/2)
          -> heteroscedastic spread; predictive law is N(mu, sigma)
A single global variance-recalibration factor (from OOF PIT std) is then applied
so the predictive intervals are honest.

Everything is evaluated OUT-OF-SAMPLE by strictly chronological EXPANDING-WINDOW
walk-forward (train only on date_target < the test block). No random splits
(seasonal drift + the leakage risk that sank forecast_band_yes). Both mu and sigma
are fit inside each fold, so sigma never sees the test residuals.

Scored against two baselines the model must add value over BEFORE the market bar
(that comparison is Phase 4):
  - consensus     : N(consensus_median, sigma_consensus)   [raw multi-model median]
  - climatology   : N(clim_high, sigma_clim)               [trailing city mean]

Metrics: point MAE, PIT calibration + interval coverage, CRPS (closed-form
Gaussian), and integer-band multiclass log-loss / Brier.

Usage:
    venv/bin/python scripts/train_lookahead_high_model.py
    venv/bin/python scripts/train_lookahead_high_model.py --features data/lookahead/features_low.parquet

Artifacts (gitignored data/, keep untracked):
    data/lookahead/oof_predictions_high.parquet   # per-row OOS mu/sigma + baselines
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

import lightgbm as lgb

KEYS = ("city", "date_target", "actual_f")
CATEGORICAL = ["city_code", "month"]
SIGMA_FLOOR = 0.75          # deg F; guards against overconfident zero-spread
EPS = 1e-9


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", default="data/lookahead/features_high.parquet")
    p.add_argument("--out", default=None)
    p.add_argument("--burnin-days", type=int, default=45,
                   help="initial train window before first OOS prediction")
    p.add_argument("--step-days", type=int, default=7,
                   help="retrain cadence / test block width (calendar days)")
    p.add_argument("--save-model", default=None,
                   help="also fit mu+sigma on ALL rows and pickle to this path "
                        "(production model for the Phase 5 shadow logger)")
    return p.parse_args()


def lgb_params(objective: str) -> dict:
    # small-data regularized config; identical shape for mu and sigma models
    return dict(
        objective=objective, n_estimators=300, learning_rate=0.03,
        num_leaves=15, min_child_samples=30, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=1.0, min_split_gain=0.0,
        verbosity=-1, n_jobs=1,
    )


def fit_predict_fold(tr: pd.DataFrame, te: pd.DataFrame, feats: list[str]):
    """Fit mu + sigma on `tr`, return (mu, sigma) arrays for `te`."""
    Xtr, ytr = tr[feats], tr["actual_f"].to_numpy()
    Xte = te[feats]

    mu_model = lgb.LGBMRegressor(**lgb_params("regression_l1"))
    mu_model.fit(Xtr, ytr, categorical_feature=CATEGORICAL)
    mu_tr = mu_model.predict(Xtr)
    mu_te = mu_model.predict(Xte)

    # heteroscedastic spread: model |residual|, convert to sigma (E|N|=sigma*sqrt(2/pi))
    abs_res = np.abs(ytr - mu_tr)
    sig_model = lgb.LGBMRegressor(**lgb_params("regression_l1"))
    sig_model.fit(Xtr, abs_res, categorical_feature=CATEGORICAL)
    sig_te = np.maximum(sig_model.predict(Xte), 0.0) * np.sqrt(np.pi / 2.0)
    sig_te = np.maximum(sig_te, SIGMA_FLOOR)
    return mu_te, sig_te


def walk_forward(df: pd.DataFrame, feats: list[str], burnin: int, step: int) -> pd.DataFrame:
    df = df.sort_values("date_target").reset_index(drop=True)
    dates = pd.to_datetime(df["date_target"])
    d0 = dates.min()
    start = d0 + pd.Timedelta(days=burnin)
    end = dates.max()

    out = []
    cur = start
    while cur <= end:
        nxt = cur + pd.Timedelta(days=step)
        tr = df[dates < cur]
        te = df[(dates >= cur) & (dates < nxt)]
        if len(te) and len(tr) >= 50:
            mu, sig = fit_predict_fold(tr, te, feats)
            block = te[list(KEYS)].copy()
            block["mu"] = mu
            block["sigma"] = sig
            out.append(block)
        cur = nxt
    return pd.concat(out, ignore_index=True)


def baseline_distributions(df: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    """Attach consensus & climatology predictive dists (sigma = train residual std)."""
    m = oof.merge(
        df[["city", "date_target", "consensus_median", "clim_high"]],
        on=["city", "date_target"], how="left",
    )
    # global residual std of each baseline over the OOS rows -> honest-ish sigma
    sig_cons = float(np.nanstd(m["consensus_median"] - m["actual_f"]))
    sig_clim = float(np.nanstd(m["clim_high"] - m["actual_f"]))
    m["mu_cons"], m["sig_cons"] = m["consensus_median"], max(sig_cons, SIGMA_FLOOR)
    m["mu_clim"], m["sig_clim"] = m["clim_high"], max(sig_clim, SIGMA_FLOOR)
    return m


def recalibrate(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray):
    """Single global variance-inflation so PIT z has unit std (honest intervals)."""
    z = (y - mu) / sigma
    c = float(np.nanstd(z))
    return sigma * max(c, EPS), c


def crps_gaussian(y, mu, sigma):
    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def band_logloss_brier(y, mu, sigma):
    """Integer-band multiclass scores. P(round(high)=T) = Phi(T+.5) - Phi(T-.5)."""
    y = np.asarray(y); mu = np.asarray(mu); sigma = np.asarray(sigma)
    tgt = np.round(y).astype(int)
    ll = np.empty(len(y)); br = np.empty(len(y))
    for i in range(len(y)):
        m, s, t = mu[i], sigma[i], tgt[i]
        ks = np.arange(int(np.floor(m - 6 * s)), int(np.ceil(m + 6 * s)) + 1)
        p = norm.cdf(ks + 0.5, m, s) - norm.cdf(ks - 0.5, m, s)
        p = np.clip(p, 1e-12, None); p /= p.sum()
        pt = p[ks == t]
        pt = float(pt[0]) if len(pt) else 1e-12
        ll[i] = -np.log(max(pt, 1e-12))
        # multiclass Brier over the support: sum (p - onehot)^2
        oh = (ks == t).astype(float)
        br[i] = float(np.sum((p - oh) ** 2))
    return ll, br


def report(name, y, mu, sigma):
    mae = float(np.nanmean(np.abs(y - mu)))
    crps = float(np.nanmean(crps_gaussian(y, mu, sigma)))
    z = (y - mu) / sigma
    cov = {q: float(np.nanmean(np.abs(z) <= norm.ppf(0.5 + q / 2)))
           for q in (0.50, 0.80, 0.90)}
    ll, br = band_logloss_brier(y, mu, sigma)
    print(f"\n[{name}]")
    print(f"  point MAE     : {mae:6.3f} F")
    print(f"  CRPS          : {crps:6.3f}")
    print(f"  PIT z std     : {float(np.nanstd(z)):6.3f}  (want ~1.0)")
    print(f"  PI coverage   : 50%={cov[0.50]:.2f}  80%={cov[0.80]:.2f}  90%={cov[0.90]:.2f}")
    print(f"  band log-loss : {float(np.nanmean(ll)):6.3f}")
    print(f"  band Brier    : {float(np.nanmean(br)):6.3f}")
    return dict(mae=mae, crps=crps, band_logloss=float(np.nanmean(ll)))


def main() -> None:
    a = parse_args()
    kind = "low" if "low" in Path(a.features).stem else "high"
    out = a.out or f"data/lookahead/oof_predictions_{kind}.parquet"

    df = pd.read_parquet(a.features)
    feats = [c for c in df.columns if c not in KEYS]
    print(f"loaded {len(df):,} rows, {len(feats)} features from {a.features}")
    print(f"walk-forward: burn-in {a.burnin_days}d, step {a.step_days}d")

    oof = walk_forward(df, feats, a.burnin_days, a.step_days)
    m = baseline_distributions(df, oof)
    y = m["actual_f"].to_numpy()

    # global variance-recalibration of the model (Phase-3 calibration step)
    sig_recal, c = recalibrate(m["mu"].to_numpy(), m["sigma"].to_numpy(), y)
    m["sigma_recal"] = sig_recal
    print(f"\nOOS predictions: {len(m):,}  ({m['date_target'].min()} -> {m['date_target'].max()})")
    print(f"variance recalibration factor c = {c:.3f}")

    print("\n" + "=" * 60 + "\nPHASE 3 — OUT-OF-SAMPLE MODEL EVALUATION\n" + "=" * 60)
    report("model (raw)",   y, m["mu"].to_numpy(), m["sigma"].to_numpy())
    r_model = report("model (recal)", y, m["mu"].to_numpy(), sig_recal)
    r_cons = report("consensus",     y, m["mu_cons"].to_numpy(), np.full(len(m), m["sig_cons"].iloc[0]))
    r_clim = report("climatology",   y, m["mu_clim"].to_numpy(), np.full(len(m), m["sig_clim"].iloc[0]))

    print("\n" + "-" * 60)
    print("VERDICT (Phase 3 — model quality, NOT yet vs market):")
    dmae = r_cons["mae"] - r_model["mae"]
    dll = r_cons["band_logloss"] - r_model["band_logloss"]
    print(f"  model vs consensus  MAE: {r_model['mae']:.3f} vs {r_cons['mae']:.3f} "
          f"({'better' if dmae>0 else 'WORSE'} by {abs(dmae):.3f} F)")
    print(f"  model vs consensus  band log-loss: {r_model['band_logloss']:.3f} vs "
          f"{r_cons['band_logloss']:.3f} ({'better' if dll>0 else 'WORSE'} by {abs(dll):.3f})")
    print(f"  (climatology floor: MAE {r_clim['mae']:.3f}, band log-loss {r_clim['band_logloss']:.3f})")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    m.to_parquet(out, index=False)
    print(f"\nwrote OOS predictions -> {out}")

    if a.save_model:
        save_production_model(df, feats, c, a.save_model, kind)


def save_production_model(df: pd.DataFrame, feats: list[str], recal: float,
                          path: str, kind: str) -> None:
    """Fit mu + sigma on ALL rows and pickle for live serving (Phase 5)."""
    import pickle
    Xall, yall = df[feats], df["actual_f"].to_numpy()
    mu_model = lgb.LGBMRegressor(**lgb_params("regression_l1"))
    mu_model.fit(Xall, yall, categorical_feature=CATEGORICAL)
    abs_res = np.abs(yall - mu_model.predict(Xall))
    sig_model = lgb.LGBMRegressor(**lgb_params("regression_l1"))
    sig_model.fit(Xall, abs_res, categorical_feature=CATEGORICAL)
    payload = dict(
        mu_model=mu_model, sigma_model=sig_model, recal=float(recal),
        features=feats, categorical=CATEGORICAL, sigma_floor=SIGMA_FLOOR,
        kind=kind, trained_through=str(pd.to_datetime(df["date_target"]).max().date()),
        n_train=len(df),
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"saved production model ({len(df)} rows, recal {recal:.3f}) -> {path}")


if __name__ == "__main__":
    main()
