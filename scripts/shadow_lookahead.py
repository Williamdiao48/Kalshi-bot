#!/usr/bin/env python3
"""Phase 5 — live serving for the ahead-of-time HIGH model (shadow, never trades).

Two jobs, both importable by main.py's poll loop and runnable offline for testing:
  1. build_serve_features(conn, city, date_target)  -> the SAME feature row the
     Phase 2 builder produces, but assembled from live DB state at the 09:00-local
     decision time. It literally reuses build_lookahead_features' functions, so
     serve features are identical to training features (no train/serve skew — the
     failure mode called out for the old forecast_no model).
  2. predict_bands(model, feat_row, bands) -> per-band P(YES) from N(mu, sigma_recal).

Offline self-test (the anti-skew proof): rebuild features for a labeled city-day and
assert they match data/lookahead/features_high.parquet row-for-row.

    venv/bin/python scripts/shadow_lookahead.py --selftest
"""
from __future__ import annotations

import argparse
import pickle
import sqlite3
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scripts.build_lookahead_features as FB  # noqa: E402

CUTOFF = time(9, 0)
MIN_CORE = 3
MAE_WINDOW = 14
MODEL_PATH = "data/models/lookahead_high.pkl"


def load_model(path: str = MODEL_PATH) -> dict | None:
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


def _melted_for_city(conn: sqlite3.Connection, city: str, is_high: int) -> pd.DataFrame:
    """All shadow-log rows for one city (its history is needed for trailing skill)."""
    df = pd.read_sql_query(
        "SELECT logged_at, city, date_target, source, forecast_f, actual_f "
        "FROM forecast_shadow_log WHERE is_high = ? AND city = ?",
        conn, params=(is_high, city))
    df["logged_at"] = pd.to_datetime(df["logged_at"], utc=True, format="mixed")
    df["date_target"] = pd.to_datetime(df["date_target"]).dt.date
    return df


def build_serve_features(conn: sqlite3.Connection, city: str, date_target,
                         is_high: int = 1) -> pd.DataFrame | None:
    """Feature row for (city, date_target) at the 09:00-local cutoff. None if absent."""
    if isinstance(date_target, str):
        date_target = pd.to_datetime(date_target).date()
    df = _melted_for_city(conn, city, is_high)
    if df.empty:
        return None
    latest = FB.latest_before_cutoff(df, CUTOFF)          # enforces logged_at<=cutoff
    wide = FB.pivot_features(latest)
    wide = FB.add_derived(wide, MIN_CORE)                 # n_core>=3 quality gate
    wide = FB.add_trailing_skill(wide, MAE_WINDOW)        # prior-days-only skill
    row = wide[wide["date_target"] == date_target]
    return row if len(row) else None


def predict_dist(model: dict, feat_row: pd.DataFrame):
    """(mu, sigma_recal) for a 1-row feature frame."""
    X = feat_row[model["features"]]
    mu = float(model["mu_model"].predict(X)[0])
    sig = float(model["sigma_model"].predict(X)[0]) * np.sqrt(np.pi / 2.0)
    sig = max(sig, model["sigma_floor"]) * model["recal"]
    return mu, sig


def band_prob(mu: float, sigma: float, lo: float, hi: float) -> float:
    """P(daily high in band) = P(true high in [lo-0.5, hi+0.5))."""
    return float(norm.cdf(hi + 0.5, mu, sigma) - norm.cdf(lo - 0.5, mu, sigma))


def predict_bands(model: dict, feat_row: pd.DataFrame,
                  bands: list[tuple[float, float]]):
    mu, sig = predict_dist(model, feat_row)
    return mu, sig, [(lo, hi, band_prob(mu, sig, lo, hi)) for lo, hi in bands]


# --------------------------------------------------------------------------- #
def selftest(db: str) -> None:
    """Prove serve features == training features for labeled city-days."""
    train = pd.read_parquet("data/lookahead/features_high.parquet")
    feat_cols = [c for c in train.columns if c not in ("city", "date_target", "actual_f")]
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        sample = train.sample(min(8, len(train)), random_state=1)
        maxdiff = 0.0
        for _, r in sample.iterrows():
            row = build_serve_features(con, r["city"], r["date_target"])
            assert row is not None, f"no serve row for {r['city']} {r['date_target']}"
            a = row[feat_cols].to_numpy(dtype=float)[0]
            b = r[feat_cols].to_numpy(dtype=float)
            d = np.nanmax(np.abs(np.nan_to_num(a) - np.nan_to_num(b)))
            maxdiff = max(maxdiff, d)
            print(f"  {r['city']:16s} {r['date_target']}  max|Δfeat|={d:.2e}")
        print(f"\nANTI-SKEW: max feature diff across sample = {maxdiff:.2e} "
              f"({'PASS' if maxdiff < 1e-6 else 'FAIL'})")

        # end-to-end prediction on one row
        model = load_model()
        if model:
            r = sample.iloc[0]
            row = build_serve_features(con, r["city"], r["date_target"])
            mu, sig, bands = predict_bands(model, row,
                                           [(74, 75), (76, 77), (78, 79)])
            print(f"\nexample serve prediction {r['city']} {r['date_target']}: "
                  f"mu={mu:.2f} sigma={sig:.2f} actual={r['actual_f']:.0f}")
            for lo, hi, p in bands:
                print(f"  band [{lo:.0f},{hi:.0f}] P(YES)={p:.3f}")
    finally:
        con.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="data/db/opportunity_log.db")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(a.db)
    else:
        print("use --selftest (live logging runs from main.py, Phase 5 part 4)")


if __name__ == "__main__":
    main()
