#!/usr/bin/env python3
"""v1 vs v2 lookahead HIGH model: offline head-to-head + ablation of the 3 fixes.

Every variant runs through the SAME expanding walk-forward (burn-in 45d, 7-day
blocks) on the SAME city-days and is scored against the official CLI high (what
Kalshi settles on). The three v1 -> v2 changes are toggled independently:

    L  label   : train on IEM ASOS max (v1)        | CLI daily high (v2)
    T  target  : absolute high, v1 features        | actual - consensus, relative features
    S  sigma   : in-sample |resid| * sqrt(pi/2)    | out-of-fold |resid| * 1.4826

All variants use a PRIOR-FOLDS-ONLY variance recal (computed against the label the
variant trains on, as it would be in production). v1 as originally published used
a full-sample recal, which peeks at the test set, so "v1" here is v1 made honest.

Differences are paired per row and get a day-clustered bootstrap 95% CI (all
cities on one date move together, so days -- not rows -- are the independent unit).

v1's feature builder is recovered from git (V1_COMMIT) so the comparison uses v1's
exact feature definitions, not a re-implementation.

    venv/bin/python scripts/eval_lookahead_ablation.py
    venv/bin/python scripts/eval_lookahead_ablation.py --score-from 2026-11-08
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import scripts.build_lookahead_features as FB  # noqa: E402
import scripts.train_lookahead_high_model as TR  # noqa: E402

V1_COMMIT = "54fc2f3"
V1_PREFIX = "v1__"
SQRT_PI_2 = np.sqrt(np.pi / 2.0)
N_BOOT = 2000

# name: (label, target, sigma)
VARIANTS = {
    "v1":          ("iem", "abs",   "insample"),
    "v1 +L":       ("cli", "abs",   "insample"),
    "v1 +L+T":     ("cli", "resid", "insample"),   # == v2 -S
    "v2":          ("cli", "resid", "oof"),
    "v2 -L":       ("iem", "resid", "oof"),
    "v2 -T":       ("cli", "abs",   "oof"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default="data/db/opportunity_log.db")
    p.add_argument("--v2-features", default="data/lookahead/features_high.parquet")
    p.add_argument("--v1-features", default=None,
                   help="prebuilt v1 parquet (default: rebuild from git V1_COMMIT)")
    p.add_argument("--burnin-days", type=int, default=45)
    p.add_argument("--step-days", type=int, default=7)
    p.add_argument("--score-from", default=None,
                   help="score only OOS rows on/after this date (models still train "
                        "on everything earlier) -- used for the pre-registered "
                        "forward test, which scores only post-design-freeze days")
    p.add_argument("--out", default="data/lookahead/ablation_high.parquet")
    return p.parse_args()


def build_v1_features(db: str) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "v1_build.py"
        src.write_text(subprocess.run(
            ["git", "show", f"{V1_COMMIT}:scripts/build_lookahead_features.py"],
            cwd=ROOT, check=True, capture_output=True, text=True).stdout)
        out = Path(td) / "features_high_v1.parquet"
        subprocess.run([sys.executable, str(src), "--db", db, "--out", str(out)],
                       cwd=ROOT, check=True, env={"PYTHONPATH": str(ROOT)},
                       stdout=subprocess.DEVNULL)
        return pd.read_parquet(out)


def joined_rows(v1: pd.DataFrame, v2: pd.DataFrame):
    """One row per city-day with both labels and both feature sets."""
    v1_feats = [c for c in v1.columns if c not in ("city", "date_target", "actual_f")]
    v1 = v1.rename(columns={c: V1_PREFIX + c for c in v1_feats})
    v1 = v1.rename(columns={"actual_f": "y_iem"})
    for d in (v1, v2):
        d["date_target"] = pd.to_datetime(d["date_target"])
    df = v2.rename(columns={"actual_f": "y_cli"}).merge(
        v1, on=["city", "date_target"], how="inner")
    df = df.dropna(subset=["y_cli", "y_iem", "consensus_median"])
    feats = {
        "abs": ([V1_PREFIX + c for c in v1_feats],
                [V1_PREFIX + "city_code", V1_PREFIX + "month"]),
        "resid": (FB.feature_columns(v2), TR.CATEGORICAL),
    }
    return df.sort_values("date_target").reset_index(drop=True), feats


def _lgb(X, y, cat):
    m = lgb.LGBMRegressor(**TR.lgb_params("regression_l1"))
    m.fit(X, y, categorical_feature=cat)
    return m


def _target(df, label, target):
    y = df[f"y_{label}"].to_numpy()
    return y - df["consensus_median"].to_numpy() if target == "resid" else y


def _oof_abs_resid(tr, feats, cat, r, k=TR.INNER_FOLDS):
    dates = np.array(sorted(tr["date_target"].unique()))
    res = np.full(len(tr), np.nan)
    for b in np.array_split(dates, min(k, len(dates))):
        hold = tr["date_target"].isin(set(b)).to_numpy()
        if hold.all() or not hold.any():
            continue
        m = _lgb(tr.loc[~hold, feats], r[~hold], cat)
        res[hold] = r[hold] - m.predict(tr.loc[hold, feats])
    return res


def fit_predict(tr, te, feats, cat, label, target, sigma):
    r = _target(tr, label, target)
    mu_m = _lgb(tr[feats], r, cat)
    if sigma == "oof":
        res, scale = _oof_abs_resid(tr, feats, cat, r), TR.MAD_TO_SIGMA
    else:
        res, scale = r - mu_m.predict(tr[feats]), SQRT_PI_2
    ok = ~np.isnan(res)
    sig_m = _lgb(tr.loc[ok, feats], np.abs(res[ok]), cat)
    mu = mu_m.predict(te[feats])
    if target == "resid":
        mu = mu + te["consensus_median"].to_numpy()
    sig = np.maximum(np.maximum(sig_m.predict(te[feats]), 0.0) * scale, TR.SIGMA_FLOOR)
    return mu, sig


def walk_forward(df, feats, cat, label, target, sigma, burnin, step):
    dates = df["date_target"]
    cur, end = dates.min() + pd.Timedelta(days=burnin), dates.max()
    out = []
    while cur <= end:
        nxt = cur + pd.Timedelta(days=step)
        tr, te = df[dates < cur], df[(dates >= cur) & (dates < nxt)]
        if len(te) and len(tr) >= 200:
            mu, sig = fit_predict(tr, te, feats, cat, label, target, sigma)
            c = 1.0
            if out:
                prev = pd.concat(out, ignore_index=True)
                if len(prev) >= TR.RECAL_MIN_ROWS:
                    c = float(np.std((prev["y_train"] - prev["mu"]) / prev["sigma_raw"]))
            b = te[["city", "date_target", "y_cli", "consensus_median"]].copy()
            b["y_train"] = te[f"y_{label}"].to_numpy()
            b["mu"], b["sigma_raw"], b["sigma"] = mu, sig, sig * c
            out.append(b)
        cur = nxt
    return pd.concat(out, ignore_index=True)


def row_scores(y, mu, sigma) -> pd.DataFrame:
    ll, _ = TR.band_logloss_brier(y, mu, sigma)
    z = (y - mu) / sigma
    return pd.DataFrame({
        "ae": np.abs(y - mu), "crps": TR.crps_gaussian(y, mu, sigma),
        "ll": ll, "z": z, "cov80": (np.abs(z) <= 1.2816).astype(float),
    })


def boot_ci(diff: pd.Series, days: pd.Series, rng) -> tuple[float, float, float]:
    """Mean paired diff + day-clustered bootstrap 95% CI."""
    g = pd.DataFrame({"d": diff.to_numpy(), "day": days.to_numpy()}).groupby("day")["d"]
    s, n = g.sum().to_numpy(), g.size().to_numpy()
    idx = rng.integers(0, len(s), size=(N_BOOT, len(s)))
    boots = s[idx].sum(1) / n[idx].sum(1)
    return float(diff.mean()), *np.percentile(boots, [2.5, 97.5])


def main() -> None:
    a = parse_args()
    v2 = pd.read_parquet(a.v2_features)
    v1 = pd.read_parquet(a.v1_features) if a.v1_features else build_v1_features(a.db)
    df, feats = joined_rows(v1, v2)
    print(f"joined {len(df):,} city-days, {df['city'].nunique()} cities, "
          f"{df['date_target'].min().date()} -> {df['date_target'].max().date()}")
    print(f"IEM label != CLI on {np.mean(df['y_iem'].round() != df['y_cli']):.1%} of city-days")

    oof = {}
    for name, (label, target, sigma) in VARIANTS.items():
        f, cat = feats[target]
        oof[name] = walk_forward(df, f, cat, label, target, sigma, a.burnin_days, a.step_days)
        print(f"  ran {name:<8} ({label}/{target}/{sigma}): {len(oof[name]):,} OOS rows")

    if a.score_from:
        cut = pd.Timestamp(a.score_from)
        oof = {n: o[o["date_target"] >= cut].reset_index(drop=True) for n, o in oof.items()}
        if oof["v2"].empty:
            print(f"no OOS rows on/after {a.score_from} yet"); return
    base = oof["v2"]
    keys = ["city", "date_target"]
    for n, o in oof.items():          # identical OOS rows across variants
        assert o[keys].equals(base[keys]), n
    y = base["y_cli"].to_numpy()
    scores = {n: row_scores(y, o["mu"].to_numpy(), o["sigma"].to_numpy()) for n, o in oof.items()}
    sig_c = float(np.std(base["consensus_median"] - y))
    scores["consensus"] = row_scores(y, base["consensus_median"].to_numpy(),
                                     np.full(len(y), max(sig_c, TR.SIGMA_FLOOR)))

    print(f"\nOOS {len(y):,} rows / {base['date_target'].nunique()} days "
          f"({base['date_target'].min().date()} -> {base['date_target'].max().date()}), "
          "scored vs CLI\n")
    print(f"{'variant':<11}{'MAE':>7}{'CRPS':>7}{'bandLL':>8}{'z-std':>7}{'cov80':>7}")
    for n, s in scores.items():
        print(f"{n:<11}{s.ae.mean():7.3f}{s.crps.mean():7.3f}{s.ll.mean():8.3f}"
              f"{s.z.std():7.3f}{s.cov80.mean():7.3f}")

    rng = np.random.default_rng(0)
    days = base["date_target"]
    print("\nPAIRED DIFFERENCES (negative = first is better), day-clustered 95% CI:")
    pairs = [("v2", "v1"), ("v1 +L", "v1"), ("v1 +L+T", "v1 +L"), ("v2", "v1 +L+T"),
             ("v2", "v2 -L"), ("v2", "v2 -T"), ("v2", "consensus"), ("v1", "consensus")]
    rows = []
    for x, b in pairs:
        line = f"  {x:>8} - {b:<10}"
        for m in ("ae", "crps", "ll"):
            mean, lo, hi = boot_ci(scores[x][m] - scores[b][m], days, rng)
            sig = "*" if hi < 0 or lo > 0 else " "
            line += f"  {m}: {mean:+.3f} [{lo:+.3f},{hi:+.3f}]{sig}"
            rows.append(dict(a=x, b=b, metric=m, diff=mean, lo=lo, hi=hi))
        print(line)
    print("  (* = CI excludes 0)")

    ym = days.dt.to_period("M")
    print("\nMAE BY MONTH (seasonal-drift check):")
    t = pd.DataFrame({n: s.ae.groupby(ym.to_numpy()).mean() for n, s in scores.items()})
    t.insert(0, "n", scores["v2"].ae.groupby(ym.to_numpy()).size())
    print(t.round(3).to_string())

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(a.out, index=False)
    print(f"\nwrote paired-diff table -> {a.out}")


if __name__ == "__main__":
    main()
