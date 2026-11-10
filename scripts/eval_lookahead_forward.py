#!/usr/bin/env python3
"""Phase 6 — the unbiased verdict: model vs market on the FORWARD shadow log.

Reads `shadow_lookahead_high` (the Phase 5 live logger): every open HIGH band,
captured at the 09:00-local decision time with both the model probability
(N(mu, sigma_recal) integrated over the band) and the concurrent market price,
each settled to a realized YES/NO outcome.

Unlike the Phase 4 eval (`eval_lookahead_vs_market.py`), this is NOT
selection-biased. Phase 4 could only use the `opportunities` log — bands the bot
had *flagged* (near-money / apparent-edge) — so it saw a biased slice of the
distribution. The forward logger records the FULL per-day band distribution, so
the model-vs-market Brier / log-loss is finally an honest test.

Because that full distribution includes many tail bands both sides price near
0/1 (easy, and they shrink the average difference), we report TWO cuts:
  (1) ALL bands            — the honest aggregate.
  (2) NEAR-MONEY           — market_p in [0.1, 0.9]; where skill actually lives,
                             and the apples-to-apples comparison with Phase 4.

Metrics per cut: Brier + log-loss with day-CLUSTERED bootstrap CIs (same-day
bands are correlated, so a naive IID CI overstates significance), calibration
tables, and a decision-time P&L proxy at the real book prices, net of a flat 1c fee
and of Kalshi's actual taker-fee schedule (kalshi_bot/fees.py).
When the realized daily high is joinable from `forecast_shadow_log`, we also
report distribution sharpness/calibration: CRPS, PIT z-mean/z-std, 80% coverage.

Run once enough rows have SETTLED (outcome backfilled):
    venv/bin/python scripts/eval_lookahead_forward.py
    venv/bin/python scripts/eval_lookahead_forward.py --min-days 20 --near 0.1 0.9

This reads only settled rows; it is safe to run any time — it just says "not
enough settled data yet" until the forward log has accumulated.
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Reuse the exact Phase-4 statistics so the two evals are directly comparable.
from scripts.eval_lookahead_vs_market import (  # noqa: E402
    brier, logloss, clustered_bootstrap, calib_table, report_pnl, trade_pnl,
)



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default="data/db/opportunity_log.db")
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--min-days", type=int, default=10,
                   help="warn (don't fail) below this many settled city-days")
    p.add_argument("--near", type=float, nargs=2, default=(0.1, 0.9),
                   metavar=("LO", "HI"), help="near-money market_p band")
    p.add_argument("--model", default="v2",
                   help="model_version prefix to score: v2 (default) or v1. "
                        "Never pool versions -- they are different models.")
    return p.parse_args()


def load_rows(db: str, model: str = "v2") -> pd.DataFrame:
    """Settled forward-log rows of one model version, outcome mapped to 0/1.

    Rows logged by the pre-V12 bot carry NULL model_version; they are v1.
    """
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        df = pd.read_sql_query(
            "SELECT city, date_target, ticker, band_lo, band_hi, mu, sigma, "
            "       model_p, market_yes_bid, market_yes_ask, market_p, "
            "       outcome, actual_f "
            "FROM shadow_lookahead_high WHERE outcome IN ('yes','no') "
            "  AND COALESCE(model_version, 'v1') LIKE ?", con, params=(f"{model}%",))
    finally:
        con.close()
    if df.empty:
        return df
    df["y"] = (df["outcome"] == "yes").astype(int)
    df["date_target"] = pd.to_datetime(df["date_target"]).dt.date

    # Before the logger fix, next-day contracts were logged under today's
    # date_target and scored with today's distribution. Keep only rows whose
    # contract date matches date_target.
    ticker_date = pd.to_datetime(
        df["ticker"].str.extract(r"-(\d{2}[A-Z]{3}\d{2})-")[0],
        format="%y%b%d", errors="coerce").dt.date
    matched = ticker_date == df["date_target"]
    if (~matched).any():
        print(f"dropped {int((~matched).sum()):,} of {len(df):,} rows whose contract "
              f"date != date_target (pre-fix mis-dated captures)")
    return df[matched].reset_index(drop=True)


def attach_actual(df: pd.DataFrame, db: str) -> pd.DataFrame:
    """Fill actual_f (the realized daily high) from the official CLI labels.

    The logger leaves actual_f NULL. cli_daily is the Kalshi settlement source
    (IEM max_tmpf, used before, disagreed with it on ~1/3 of days). Needed only
    for CRPS/PIT.
    """
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        lab = pd.read_sql_query(
            "SELECT city, date AS date_target, high_f AS actual_f FROM cli_daily "
            "WHERE high_f IS NOT NULL", con)
    finally:
        con.close()
    if lab.empty:
        return df
    lab["date_target"] = pd.to_datetime(lab["date_target"]).dt.date
    lab = lab.drop_duplicates(["city", "date_target"])
    merged = df.merge(lab, on=["city", "date_target"], how="left",
                      suffixes=("", "_lab"))
    merged["actual_f"] = merged["actual_f"].fillna(merged["actual_f_lab"])
    return merged.drop(columns=[c for c in merged.columns if c.endswith("_lab")])


def crps_gaussian(mu, sigma, x):
    """Closed-form CRPS for a Gaussian predictive dist (lower is better)."""
    sigma = np.maximum(sigma, 1e-6)
    z = (x - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def report_cut(df: pd.DataFrame, label: str, n_boot: int) -> None:
    print("\n" + "=" * 66)
    print(f"CUT: {label}   ({len(df):,} band-instances, "
          f"{df.groupby(['city','date_target']).ngroups} city-days)")
    print("=" * 66)
    if df.empty:
        print("  (no rows in this cut)"); return

    d = df.copy().reset_index(drop=True)
    d["model_p"] = d["model_p"].clip(1e-6, 1 - 1e-6)
    has_mkt = d["market_p"].notna()
    d["market_p"] = d["market_p"].clip(1e-6, 1 - 1e-6)

    d["brier_model"] = brier(d["model_p"], d["y"])
    d["ll_model"] = logloss(d["model_p"], d["y"])
    d["brier_market"] = brier(d["market_p"], d["y"])
    d["ll_market"] = logloss(d["market_p"], d["y"])

    print(f"base rate (band wins): {d['y'].mean():.3f}   "
          f"rows with market price: {int(has_mkt.sum())}/{len(d)}")
    print(f"  model   Brier {d['brier_model'].mean():.4f}   "
          f"log-loss {d['ll_model'].mean():.4f}")
    dm = d[has_mkt]
    if not dm.empty:
        print(f"  market  Brier {dm['brier_market'].mean():.4f}   "
              f"log-loss {dm['ll_market'].mean():.4f}   (on priced rows)")

        # model vs market only where both are defined (priced rows)
        dm = dm.reset_index(drop=True)
        print("\nday-clustered bootstrap, mean(model - market)  [<0 => model beats market]:")
        for metric in ("brier", "ll"):
            pt, lo, hi = clustered_bootstrap(dm, f"{metric}_model", f"{metric}_market", n_boot)
            verdict = ("MODEL BETTER" if hi < 0 else
                       "MARKET BETTER" if lo > 0 else "TIE (CI spans 0)")
            print(f"  {metric:6s} diff {pt:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  -> {verdict}")

    print("\nmodel calibration (decision-time band prob vs realized):")
    print(calib_table(d, "model_p", "y").to_string())
    if not dm.empty:
        print("\nmarket calibration (sanity — should be near-diagonal):")
        print(calib_table(dm, "market_p", "y").to_string())

    report_pnl(*trade_pnl(d["model_p"], d["market_yes_bid"], d["market_yes_ask"], d["y"]))


def report_distribution(df: pd.DataFrame) -> None:
    """CRPS + PIT sharpness/calibration on the realized high (one row per city-day)."""
    d = df.dropna(subset=["actual_f", "mu", "sigma"]).copy()
    if d.empty:
        print("\n(no realized highs joinable — skipping CRPS/PIT)"); return
    # collapse to one predictive dist per city-day (mu/sigma are constant across
    # that day's bands; take the first)
    day = d.groupby(["city", "date_target"], as_index=False).agg(
        mu=("mu", "first"), sigma=("sigma", "first"), actual_f=("actual_f", "first"))
    mu = day["mu"].to_numpy(dtype=float)
    sigma = np.maximum(day["sigma"].to_numpy(dtype=float), 1e-6)
    actual = day["actual_f"].to_numpy(dtype=float)
    z = (actual - mu) / sigma
    crps = crps_gaussian(mu, sigma, actual)
    cov80 = float(((z > norm.ppf(0.1)) & (z < norm.ppf(0.9))).mean())
    print("\n" + "=" * 66)
    print(f"PREDICTIVE DISTRIBUTION  ({len(day)} city-days)")
    print("=" * 66)
    print(f"  point MAE (|actual-mu|): {np.abs(actual - mu).mean():.3f} F")
    print(f"  CRPS (mean):             {crps.mean():.3f}")
    print(f"  PIT z-mean:  {z.mean():+.3f}  (want ~0)")
    print(f"  PIT z-std:   {z.std():.3f}   (want ~1; <1 under-, >1 over-confident)")
    print(f"  80% interval coverage:   {cov80:.3f}  (want ~0.80)")


def main() -> None:
    a = parse_args()
    df = load_rows(a.db, a.model)
    print(f"model_version: {a.model}*")
    if df.empty:
        print("shadow_lookahead_high has no SETTLED rows for this model yet — the "
              "forward log is still accumulating. Re-run after some band days settle.")
        return

    df = attach_actual(df, a.db)
    n_days = df.groupby(["city", "date_target"]).ngroups
    print(f"settled forward rows: {len(df):,}  |  {n_days} city-days  |  "
          f"{df['city'].nunique()} cities  |  "
          f"{df['date_target'].min()} -> {df['date_target'].max()}")
    if n_days < a.min_days:
        print(f"WARNING: only {n_days} settled city-days (< --min-days={a.min_days}). "
              f"Results are preliminary; treat the CIs as noisy until more data lands.")

    report_cut(df, "ALL bands (honest full distribution)", a.boot)
    lo, hi = a.near
    near = df[df["market_p"].between(lo, hi)]
    report_cut(near, f"NEAR-MONEY (market_p in [{lo:.2f}, {hi:.2f}])", a.boot)
    report_distribution(df)

    print("\nNOTE: this is the UNBIASED forward test — every open band is logged, so "
          "no selection bias. The verdict is the NEAR-MONEY cut's clustered bootstrap: "
          "that is where the model must beat the market to have real edge.")


if __name__ == "__main__":
    main()
