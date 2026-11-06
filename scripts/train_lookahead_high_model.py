#!/usr/bin/env python3
"""Phase 3 — ahead-of-time HIGH-market model: distribution over the daily high.

Predicts a CALIBRATED PREDICTIVE DISTRIBUTION for the official (CLI) daily high at
the 09:00-local decision time (features from Phase 2), then integrates it over the
integer band edges to give P(daily high rounds to T) for each candidate band.

Model form (v2):
  mu    = consensus_median + LightGBM-L1( actual - consensus_median )
          The model learns a bounded CORRECTION to the multi-model consensus from
          consensus-relative features, so it follows the forecasts through seasons
          it never saw. (v1 predicted the absolute temperature and, being a tree
          model trained on summer, floored near 65F in autumn.)
  sigma = LightGBM-L1 on |OUT-OF-FOLD residual| * MAD_TO_SIGMA
          OOF residuals come from blocked-by-date K-fold inside the training set.
          An L1 fit estimates the MEDIAN absolute residual, so the Gaussian scale
          is median/0.6745 (MAD_TO_SIGMA = 1.4826). v1 fit IN-SAMPLE residuals
          (shrunk by overfitting) AND used sqrt(pi/2), the MEAN-|N| constant --
          together that is why v1 needed a 2.26x variance inflation.
  recal = global variance factor = std of OOF PIT z over PRIOR folds only (the
          production value uses the most recent RECAL_DAYS). Should sit near 1.

Everything is evaluated OUT-OF-SAMPLE by strictly chronological walk-forward
(train only on date_target < the test block). Baselines: consensus N(consensus,
sigma_c) and the climate normal N(normal, sigma_n).

Usage:
    venv/bin/python scripts/train_lookahead_high_model.py
    venv/bin/python scripts/train_lookahead_high_model.py --train-window-days 60
    venv/bin/python scripts/train_lookahead_high_model.py --save-model data/models/lookahead_high.pkl

Artifacts (gitignored data/, keep untracked):
    data/lookahead/oof_predictions_high.parquet   # per-row OOS mu/sigma + baselines
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scripts.build_lookahead_features as FB  # noqa: E402

MODEL_VERSION = "v2"
KEYS = ("city", "date_target", "actual_f", "consensus_median", "normal_f")
CATEGORICAL = ["city_code"]
SIGMA_FLOOR = 0.75          # deg F; guards against overconfident zero-spread
MAD_TO_SIGMA = 1.0 / norm.ppf(0.75)   # 1.4826: median|N(0,s)| = 0.6745 s
INNER_FOLDS = 5             # blocked-by-date folds for OOF residuals (sigma)
RECAL_MIN_ROWS = 200        # prior OOF rows needed before applying a recal factor
RECAL_DAYS = 30             # production recal factor: most recent N days of OOF
EPS = 1e-9


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", default="data/lookahead/features_high.parquet")
    p.add_argument("--out", default=None)
    p.add_argument("--burnin-days", type=int, default=45,
                   help="initial train window before first OOS prediction")
    p.add_argument("--step-days", type=int, default=7,
                   help="retrain cadence / test block width (calendar days)")
    p.add_argument("--train-window-days", type=int, default=None,
                   help="rolling train window (default: expanding, all prior days)")
    p.add_argument("--save-model", default=None,
                   help="also fit on the (windowed) full history and pickle here")
    return p.parse_args()


def lgb_params(objective: str) -> dict:
    # small-data regularized config; identical shape for mu and sigma models
    return dict(
        objective=objective, n_estimators=300, learning_rate=0.03,
        num_leaves=15, min_child_samples=30, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=1.0, min_split_gain=0.0,
        verbosity=-1, n_jobs=1,
    )


def _fit_mu(X: pd.DataFrame, r: np.ndarray) -> lgb.LGBMRegressor:
    m = lgb.LGBMRegressor(**lgb_params("regression_l1"))
    m.fit(X, r, categorical_feature=CATEGORICAL)
    return m


def oof_residuals(tr: pd.DataFrame, feats: list[str], k: int = INNER_FOLDS) -> np.ndarray:
    """Out-of-fold residuals of the mu model on `tr`, blocked by date.

    All cities of a date share a fold (no same-day cross-city leakage). Folds are
    contiguous date blocks; using later blocks to predict earlier ones only
    informs residual MAGNITUDE for sigma, never the test block.
    """
    dates = np.array(sorted(tr["date_target"].unique()))
    blocks = np.array_split(dates, min(k, len(dates)))
    res = np.full(len(tr), np.nan)
    r = tr[FB.TARGET].to_numpy()
    for b in blocks:
        hold = tr["date_target"].isin(set(b)).to_numpy()
        if hold.all() or not hold.any():
            continue
        m = _fit_mu(tr.loc[~hold, feats], r[~hold])
        res[hold] = r[hold] - m.predict(tr.loc[hold, feats])
    return res


def fit_models(tr: pd.DataFrame, feats: list[str]):
    """mu model on the full window + sigma model on OOF |residual|."""
    mu_model = _fit_mu(tr[feats], tr[FB.TARGET].to_numpy())
    res = oof_residuals(tr, feats)
    ok = ~np.isnan(res)
    sig_model = lgb.LGBMRegressor(**lgb_params("regression_l1"))
    sig_model.fit(tr.loc[ok, feats], np.abs(res[ok]), categorical_feature=CATEGORICAL)
    return mu_model, sig_model


def predict(mu_model, sig_model, df: pd.DataFrame, feats: list[str]):
    """(mu, raw sigma) in deg F: consensus + correction, floored spread."""
    mu = df["consensus_median"].to_numpy() + mu_model.predict(df[feats])
    sig = np.maximum(sig_model.predict(df[feats]), 0.0) * MAD_TO_SIGMA
    return mu, np.maximum(sig, SIGMA_FLOOR)


def _window(df: pd.DataFrame, dates: pd.Series, end, window_days: int | None) -> pd.DataFrame:
    m = dates < end
    if window_days:
        m &= dates >= end - pd.Timedelta(days=window_days)
    return df[m]


def walk_forward(df: pd.DataFrame, feats: list[str], burnin: int, step: int,
                 window_days: int | None) -> pd.DataFrame:
    df = df.sort_values("date_target").reset_index(drop=True)
    dates = pd.to_datetime(df["date_target"])
    cur = dates.min() + pd.Timedelta(days=burnin)
    end = dates.max()

    out: list[pd.DataFrame] = []
    while cur <= end:
        nxt = cur + pd.Timedelta(days=step)
        tr = _window(df, dates, cur, window_days)
        te = df[(dates >= cur) & (dates < nxt)]
        if len(te) and len(tr) >= 200:
            mu_m, sig_m = fit_models(tr, feats)
            mu, sig = predict(mu_m, sig_m, te, feats)
            # recal factor from PRIOR folds only (none until enough history)
            c = 1.0
            if out:
                prev = pd.concat(out, ignore_index=True)
                if len(prev) >= RECAL_MIN_ROWS:
                    c = float(np.nanstd((prev["actual_f"] - prev["mu"]) / prev["sigma"]))
            block = te[list(KEYS)].copy()
            block["mu"], block["sigma"] = mu, sig
            block["recal"] = c
            block["sigma_recal"] = sig * max(c, EPS)
            out.append(block)
        cur = nxt
    return pd.concat(out, ignore_index=True)


def baseline_distributions(oof: pd.DataFrame) -> pd.DataFrame:
    """Consensus & climate-normal predictive dists (sigma = OOS residual std)."""
    m = oof.copy()
    sig_cons = float(np.nanstd(m["consensus_median"] - m["actual_f"]))
    sig_norm = float(np.nanstd(m["normal_f"] - m["actual_f"]))
    m["mu_cons"], m["sig_cons"] = m["consensus_median"], max(sig_cons, SIGMA_FLOOR)
    m["mu_clim"], m["sig_clim"] = m["normal_f"], max(sig_norm, SIGMA_FLOOR)
    return m


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
        oh = (ks == t).astype(float)
        br[i] = float(np.sum((p - oh) ** 2))
    return ll, br


def report(name, y, mu, sigma):
    ok = ~(np.isnan(y) | np.isnan(mu) | np.isnan(sigma))
    y, mu, sigma = y[ok], mu[ok], sigma[ok]
    mae = float(np.mean(np.abs(y - mu)))
    crps = float(np.mean(crps_gaussian(y, mu, sigma)))
    z = (y - mu) / sigma
    cov = {q: float(np.mean(np.abs(z) <= norm.ppf(0.5 + q / 2))) for q in (0.50, 0.80, 0.90)}
    ll, br = band_logloss_brier(y, mu, sigma)
    print(f"\n[{name}]  n={len(y):,}")
    print(f"  point MAE     : {mae:6.3f} F")
    print(f"  CRPS          : {crps:6.3f}")
    print(f"  PIT z std     : {float(np.std(z)):6.3f}  (want ~1.0)")
    print(f"  PI coverage   : 50%={cov[0.50]:.2f}  80%={cov[0.80]:.2f}  90%={cov[0.90]:.2f}")
    print(f"  band log-loss : {float(np.mean(ll)):6.3f}")
    print(f"  band Brier    : {float(np.mean(br)):6.3f}")
    return dict(mae=mae, crps=crps, band_logloss=float(np.mean(ll)))


def by_month(m: pd.DataFrame) -> pd.DataFrame:
    """The seasonal-drift check: model vs consensus per month, + PIT z std."""
    t = m.assign(
        ym=pd.to_datetime(m["date_target"]).dt.to_period("M"),
        e_model=m["mu"] - m["actual_f"],
        e_cons=m["consensus_median"] - m["actual_f"],
        z=(m["actual_f"] - m["mu"]) / m["sigma_recal"],
    )
    return t.groupby("ym").agg(
        n=("mu", "size"), actual=("actual_f", "mean"),
        model_mae=("e_model", lambda s: s.abs().mean()),
        cons_mae=("e_cons", lambda s: s.abs().mean()),
        model_bias=("e_model", "mean"), cons_bias=("e_cons", "mean"),
        z_std=("z", "std"), recal=("recal", "mean"),
    ).round(3)


def main() -> None:
    a = parse_args()
    kind = "low" if "low" in Path(a.features).stem else "high"
    out = a.out or f"data/lookahead/oof_predictions_{kind}.parquet"

    df = pd.read_parquet(a.features).dropna(subset=["actual_f", "consensus_median"])
    feats = FB.feature_columns(df)
    win = f"rolling {a.train_window_days}d" if a.train_window_days else "expanding"
    print(f"loaded {len(df):,} rows, {len(feats)} features from {a.features}")
    print(f"walk-forward: burn-in {a.burnin_days}d, step {a.step_days}d, window {win}")

    oof = walk_forward(df, feats, a.burnin_days, a.step_days, a.train_window_days)
    m = baseline_distributions(oof)
    y = m["actual_f"].to_numpy()
    print(f"\nOOS predictions: {len(m):,}  ({m['date_target'].min()} -> {m['date_target'].max()})")
    print(f"recal factor (prior-folds-only): first {m['recal'].iloc[0]:.3f}, "
          f"last {m['recal'].iloc[-1]:.3f}")

    print("\n" + "=" * 60 + "\nPHASE 3 — OUT-OF-SAMPLE MODEL EVALUATION\n" + "=" * 60)
    report("model (raw sigma)", y, m["mu"].to_numpy(), m["sigma"].to_numpy())
    r_model = report("model (recal)", y, m["mu"].to_numpy(), m["sigma_recal"].to_numpy())
    r_cons = report("consensus", y, m["mu_cons"].to_numpy(), np.full(len(m), m["sig_cons"].iloc[0]))
    r_clim = report("climate normal", y, m["mu_clim"].to_numpy(), np.full(len(m), m["sig_clim"].iloc[0]))

    print("\nPER MONTH (seasonal drift check):")
    print(by_month(m).to_string())
    late = m[pd.to_datetime(m["date_target"]) >= pd.to_datetime(m["date_target"]).max() - pd.Timedelta(days=30)]
    cb = (late["mu"] - late["actual_f"]).groupby(late["city"]).mean().sort_values()
    print(f"\nper-city model bias, last 30 days (want all within +/-1.5F): "
          f"min {cb.iloc[0]:+.2f} ({cb.index[0]}), max {cb.iloc[-1]:+.2f} ({cb.index[-1]})")

    print("\n" + "-" * 60)
    print("VERDICT (Phase 3 — model quality, NOT yet vs market):")
    dmae = r_cons["mae"] - r_model["mae"]
    dll = r_cons["band_logloss"] - r_model["band_logloss"]
    print(f"  model vs consensus  MAE: {r_model['mae']:.3f} vs {r_cons['mae']:.3f} "
          f"({'better' if dmae > 0 else 'WORSE'} by {abs(dmae):.3f} F)")
    print(f"  model vs consensus  band log-loss: {r_model['band_logloss']:.3f} vs "
          f"{r_cons['band_logloss']:.3f} ({'better' if dll > 0 else 'WORSE'} by {abs(dll):.3f})")
    print(f"  (climate-normal floor: MAE {r_clim['mae']:.3f}, band log-loss {r_clim['band_logloss']:.3f})")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    m.to_parquet(out, index=False)
    print(f"\nwrote OOS predictions -> {out}")

    if a.save_model:
        recent = m[pd.to_datetime(m["date_target"]) >= pd.to_datetime(m["date_target"]).max()
                   - pd.Timedelta(days=RECAL_DAYS)]
        c = float(np.nanstd((recent["actual_f"] - recent["mu"]) / recent["sigma"]))
        save_production_model(df, feats, c, a.save_model, kind, a.train_window_days)


def save_production_model(df: pd.DataFrame, feats: list[str], recal: float,
                          path: str, kind: str, window_days: int | None) -> None:
    """Fit mu + sigma on the (windowed) full history and pickle for serving."""
    import pickle
    dates = pd.to_datetime(df["date_target"])
    tr = _window(df, dates, dates.max() + pd.Timedelta(days=1), window_days)
    mu_model, sig_model = fit_models(tr, feats)
    trained_through = str(dates.max().date())
    payload = dict(
        mu_model=mu_model, sigma_model=sig_model, recal=float(recal),
        features=feats, categorical=CATEGORICAL, sigma_floor=SIGMA_FLOOR,
        sigma_scale=MAD_TO_SIGMA,
        kind=kind, trained_through=trained_through, n_train=len(tr),
        target="resid_consensus", version=f"{MODEL_VERSION}@{trained_through}",
        train_window_days=window_days,
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(path).with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(payload, f)
    tmp.replace(path)                       # atomic: the live bot never reads a partial file
    print(f"saved production model {payload['version']} ({len(tr)} rows, "
          f"recal {recal:.3f}) -> {path}")


if __name__ == "__main__":
    main()
