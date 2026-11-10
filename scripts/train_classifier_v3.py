#!/usr/bin/env python3
"""Walk-forward training + honest evaluation for the band-NO classifier v3.

Calibration scheme (used identically in every fold and for the saved model):
LightGBM is fit on the oldest 80% of the fold's training DATES, and the
isotonic calibrator is fit on the newest 20% (which the trees never saw). The
v1/v2 trainer fit isotonic on rows the trees had been trained on and split by
row order on an unsorted CSV; both overstate confidence.

Evaluation is out-of-sample by date: expanding window, 14-day test blocks, one
embargo day. Metrics are clustered by city-day (rows within a day are hourly
re-looks at the same outcome, so row-level CIs would be far too narrow).

Retro market check: same-day rows of the old v2 shadow log carry the market's
NO ask at a known instant. Each is matched to the v3 OOS row for the same ticker
at the latest decision instant at or before it, so v3, the old v2 model and the
market are scored on identical band-instances.

    venv/bin/python scripts/train_classifier_v3.py               # evaluate only
    venv/bin/python scripts/train_classifier_v3.py --save-model  # + write pkls
"""
from __future__ import annotations

import argparse
import pickle
import sqlite3
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kalshi_bot import classifier_v3 as C  # noqa: E402
from scripts.eval_lookahead_vs_market import brier, calib_table, clustered_bootstrap, logloss  # noqa: E402

ROWS = Path("data/classifier_v3/train_rows.parquet")
OOF_OUT = Path("data/classifier_v3/oof_{}.parquet")

PARAMS = dict(
    n_estimators=300, learning_rate=0.03, num_leaves=15, max_depth=5,
    # Rows are hourly re-looks at one outcome per band-day, so a leaf needs many
    # rows before it means anything.
    min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    reg_lambda=5.0, random_state=0, verbose=-1,
)
CAL_FRAC = 0.20
MIN_TRAIN_DAYS = 45
TEST_DAYS = 14
EMBARGO_DAYS = 1


def fit(train: pd.DataFrame) -> tuple[lgb.LGBMClassifier, IsotonicRegression, str]:
    days = np.sort(train["date_target"].unique())
    cut = days[int(len(days) * (1 - CAL_FRAC))]
    fit_part, cal_part = train[train["date_target"] < cut], train[train["date_target"] >= cut]
    m = lgb.LGBMClassifier(**PARAMS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(fit_part[C.FEATURES].astype(float), fit_part["won_no"],
              categorical_feature=["city_code"])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(m.predict_proba(cal_part[C.FEATURES].astype(float))[:, 1], cal_part["won_no"])
    return m, iso, cut


def walk_forward(df: pd.DataFrame) -> pd.DataFrame:
    days = sorted(df["date_target"].unique())
    start = pd.Timestamp(days[0]) + pd.Timedelta(days=MIN_TRAIN_DAYS)
    out = []
    t0 = start
    while t0 <= pd.Timestamp(days[-1]):
        t1 = t0 + pd.Timedelta(days=TEST_DAYS)
        tr = df[pd.to_datetime(df["date_target"]) < t0 - pd.Timedelta(days=EMBARGO_DAYS)]
        te = df[(pd.to_datetime(df["date_target"]) >= t0) & (pd.to_datetime(df["date_target"]) < t1)]
        if len(te) and tr["won_no"].nunique() == 2:
            m, iso, _ = fit(tr)
            te = te.copy()
            te["p"] = C.predict_frame({"lgbm": m, "iso": iso}, te)   # the serving path
            te["p_base"] = tr["won_no"].mean()          # training base rate
            out.append(te)
        t0 = t1
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def report(oof: pd.DataFrame, label: str, n_boot: int) -> None:
    d = oof.reset_index(drop=True).copy()
    y = d["won_no"]
    for c in ("p", "p_base"):
        d[f"br_{c}"], d[f"ll_{c}"] = brier(d[c], y), logloss(d[c], y)
    print(f"\n{label}: {len(d):,} OOS rows, {d.groupby(['city', 'date_target']).ngroups:,} "
          f"city-days, {d['date_target'].min()} -> {d['date_target'].max()}, NO rate {y.mean():.3f}")
    print(f"  v3   Brier {d['br_p'].mean():.4f}  log-loss {d['ll_p'].mean():.4f}  mean p {d['p'].mean():.3f}")
    print(f"  base Brier {d['br_p_base'].mean():.4f}  log-loss {d['ll_p_base'].mean():.4f}")
    for m in ("br", "ll"):
        pt, lo, hi = clustered_bootstrap(d, f"{m}_p", f"{m}_p_base", n_boot)
        print(f"  v3 - base {m}: {pt:+.4f} [{lo:+.4f}, {hi:+.4f}]  (<0 => v3 better)")
    print(calib_table(d, "p", "won_no").to_string())


def market_check(oof: pd.DataFrame, db: str, n_boot: int) -> None:
    """v3 OOS vs the old v2 model vs the market on the same same-day instants."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    sh = pd.read_sql_query(
        "SELECT logged_at, ticker, model_p AS p_v2, market_p_no FROM shadow_model_no_v2 "
        "WHERE outcome IN ('won','lost') AND logged_at >= '2026-06-18'", con)
    con.close()
    sh["t"] = pd.to_datetime(sh["logged_at"], utc=True, format="mixed")
    o = oof.copy()
    o["t"] = pd.to_datetime(o["asof"], utc=True, format="mixed")
    o = o.sort_values("t")
    sh = sh.sort_values("t")
    # latest v3 decision instant at or before the shadow row, same ticker, <= 75 min old
    m = pd.merge_asof(sh, o, on="t", by="ticker", direction="backward",
                      tolerance=pd.Timedelta(minutes=75)).dropna(subset=["p"])
    m = m[m["market_p_no"].between(0.15, 0.85)].reset_index(drop=True)
    if m.empty:
        print("\nretro market check: no overlapping rows")
        return
    y = m["won_no"]
    for c in ("p", "p_v2", "market_p_no"):
        m[f"br_{c}"], m[f"ll_{c}"] = brier(m[c], y), logloss(m[c], y)
    print(f"\nRETRO MARKET CHECK (same-day v2 shadow instants, NO ask in [15,85]c): "
          f"{len(m):,} rows, {m.groupby(['city', 'date_target']).ngroups:,} city-days, "
          f"NO rate {y.mean():.3f}")
    for c, nm in (("p", "v3 OOS"), ("p_v2", "old v2 live"), ("market_p_no", "market (NO ask)")):
        print(f"  {nm:<16} mean p {m[c].mean():.3f}  Brier {m[f'br_{c}'].mean():.4f}  "
              f"log-loss {m[f'll_{c}'].mean():.4f}")
    for a, b in (("p", "market_p_no"), ("p", "p_v2")):
        for k in ("br", "ll"):
            pt, lo, hi = clustered_bootstrap(m, f"{k}_{a}", f"{k}_{b}", n_boot)
            print(f"  {a}-{b} {k}: {pt:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    print("  (the NO ask sits half a spread above the mid, so it slightly overstates P(NO))")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="data/db/opportunity_log.db")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--save-model", action="store_true")
    a = ap.parse_args()
    df = pd.read_parquet(ROWS).sort_values(["date_target", "asof"]).reset_index(drop=True)
    for hi in (1, 0):
        part = df[df["is_high"] == hi]
        name = "HIGH" if hi else "LOW"
        oof = walk_forward(part)
        if oof.empty:
            print(f"{name}: not enough data for a walk-forward")
            continue
        oof.to_parquet(str(OOF_OUT).format(name.lower()), index=False)
        report(oof, f"{name} walk-forward", a.boot)
        market_check(oof, a.db, a.boot)
        if a.save_model:
            m, iso, cut = fit(part)
            gain = pd.Series(m.booster_.feature_importance("gain"), index=C.FEATURES)
            print("  feature gain share:\n" + (gain / gain.sum()).sort_values(ascending=False)
                  .head(10).round(3).to_string())
            bundle = dict(
                version=C.MODEL_VERSION, features=C.FEATURES, is_high=bool(hi),
                lgbm=m, iso=iso, params=PARAMS, cal_from=str(cut),
                trained_from=part["date_target"].min(), trained_through=part["date_target"].max(),
                n_rows=len(part), built_at=datetime.now(timezone.utc).isoformat(),
            )
            path = C.model_path(bool(hi))
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as f:
                pickle.dump(bundle, f)
            print(f"  saved {path}")


if __name__ == "__main__":
    main()
