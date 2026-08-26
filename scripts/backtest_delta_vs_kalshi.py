"""
Backtest: delta model vs existing Kalshi band model on spring 2026 data.

Evaluates both models on the same 35K rows from forecast_no_training_data_kalshi.csv
(Mar 27 – May 30, 2026, real Kalshi band positions + actual outcomes).

For each row:
  - Existing model: LightGBM classifier trained on band-relative features → P(won) via isotonic
  - Delta model:    LightGBM regression predicting delta_to_final, then
                    P(NO wins) = P(delta > band_ceil - running_obs) via Gaussian CDF

Output:
  - Head-to-head AUC, calibration, and win rates at various probability thresholds
  - Rows where models disagree (one would trade, other wouldn't)
  - Breakdown by is_high, month, margin band
"""

import csv
import json
import pickle
import sys
import os
from collections import defaultdict
from math import sqrt
from pathlib import Path

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, brier_score_loss

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

KALSHI_CSV  = Path("data/backtest/forecast_no_training_data_kalshi.csv")
DELTA_CSV   = Path("data/backtest/delta_training_data.csv")
KALSHI_PKL  = Path("data/models/forecast_no_band_model.pkl")
DELTA_PKL   = Path("data/models/delta_model.pkl")
DELTA_STDS  = Path("data/models/delta_residual_std.json")

KALSHI_FEATURES = [
    "margin_f", "delta_1h", "delta_2h", "hours_above_ceil", "hour_utc",
    "hours_to_close", "obs_vs_hrrr_h", "obs_vs_gfs_h", "hrrr_vs_ceil",
    "gfs_vs_ceil", "consensus_vs_ceil", "model_spread", "n_models_above_ceil",
    "recent_hrrr_mae_7d", "clim_prob_exceed", "clim_drop_p50", "clim_drop_p75",
    "city_enc", "is_high", "month",
]

DELTA_FEATURES = [
    "running_obs", "delta_1h", "delta_2h", "hours_to_close",
    "hrrr_remaining", "gfs_remaining", "consensus_remaining", "model_spread",
    "obs_vs_hrrr_h", "obs_vs_gfs_h", "recent_hrrr_mae_7d",
    "clim_p25", "clim_p50", "clim_p75", "city_enc", "is_high", "month",
]


def load_clim_lookup(delta_csv: Path) -> dict[tuple, tuple]:
    """
    Build {(city, is_high, month, hour_utc): (clim_p25, clim_p50, clim_p75)} from delta CSV.
    clim values are identical for all rows sharing a (city, is_high, month, hour_utc) bucket.
    Only load spring range to match Kalshi data.
    """
    lookup = {}
    print("Building clim lookup from delta training data (spring dates only)...")
    for r in csv.DictReader(delta_csv.open()):
        if r["date"] < "2026-03-27":
            continue
        key = (r["city"], r["is_high"], int(r["month"]), int(r["hour_utc"]))
        if key not in lookup:
            lookup[key] = (
                float(r["clim_p25"]),
                float(r["clim_p50"]),
                float(r["clim_p75"]),
            )
    print(f"  {len(lookup)} (city, is_high, month, hour) buckets loaded")
    return lookup


def build_kalshi_X(rows, city_map):
    X = []
    for r in rows:
        city = city_map.get(r["city"], 0)
        cons = float(r["consensus_vs_ceil"])
        X.append([
            float(r["margin_f"]),
            float(r["delta_1h"])            if r.get("delta_1h")            else 0.0,
            float(r["delta_2h"])            if r.get("delta_2h")            else 0.0,
            float(r.get("hours_above_ceil", 1)),
            float(r["hour_utc"]),
            float(r["hours_to_close"]),
            float(r["obs_vs_hrrr_h"])       if r.get("obs_vs_hrrr_h")       else 0.0,
            float(r["obs_vs_gfs_h"])        if r.get("obs_vs_gfs_h")        else 0.0,
            float(r["hrrr_vs_ceil"])        if r.get("hrrr_vs_ceil")        else cons,
            float(r["gfs_vs_ceil"])         if r.get("gfs_vs_ceil")         else cons,
            cons,
            float(r["model_spread"]),
            float(r["n_models_above_ceil"]),
            float(r.get("recent_hrrr_mae_7d") or 3.0),
            float(r.get("clim_prob_exceed") or 0.15),
            float(r.get("clim_drop_p50")    or 2.0),
            float(r.get("clim_drop_p75")    or 3.0),
            float(city),
            float(r["is_high"]),
            float(r["month"]),
        ])
    return np.array(X, dtype=np.float32)


def build_delta_X(rows, city_map, clim_lookup):
    X = []
    for r in rows:
        city   = city_map.get(r["city"], 0)
        ro     = float(r["running_obs"])
        margin = float(r["margin_f"])  # running_obs - band_ceil
        cons   = float(r["consensus_vs_ceil"])
        clim_key = (r["city"], r["is_high"], int(r["month"]), int(r["hour_utc"]))
        p25, p50, p75 = clim_lookup.get(clim_key, (-3.0, 0.0, 4.0))
        X.append([
            ro,
            float(r["delta_1h"])            if r.get("delta_1h")            else 0.0,
            float(r["delta_2h"])            if r.get("delta_2h")            else 0.0,
            float(r["hours_to_close"]),
            float(r["hrrr_vs_ceil"]) - margin if r.get("hrrr_vs_ceil") else cons - margin,
            float(r["gfs_vs_ceil"])  - margin if r.get("gfs_vs_ceil")  else cons - margin,
            cons - margin,  # consensus_remaining = consensus_vs_ceil - margin_f
            float(r["model_spread"]),
            float(r["obs_vs_hrrr_h"])       if r.get("obs_vs_hrrr_h")       else 0.0,
            float(r["obs_vs_gfs_h"])        if r.get("obs_vs_gfs_h")        else 0.0,
            float(r.get("recent_hrrr_mae_7d") or 3.0),
            p25, p50, p75,
            float(city),
            float(r["is_high"]),
            float(r["month"]),
        ])
    return np.array(X, dtype=np.float32)


def delta_to_prob(pred_delta, rows, stds):
    """P(NO wins) = P(delta > band_ceil - running_obs) for each row."""
    probs = []
    for delta, r in zip(pred_delta, rows):
        threshold = int(r["band_ceil"]) - float(r["running_obs"])  # negative when obs > ceil
        std_key   = f"{r['is_high']}_{r['month']}"
        std       = stds.get(std_key, 3.0)
        p         = 1.0 - norm.cdf(threshold, loc=delta, scale=std)
        probs.append(float(p))
    return np.array(probs)


def win_rate_at_threshold(probs, labels, threshold):
    fired = [(p, l) for p, l in zip(probs, labels) if p >= threshold]
    if not fired:
        return 0.0, 0
    wr = sum(l for _, l in fired) / len(fired)
    return round(wr, 4), len(fired)


def print_threshold_table(name, probs, labels, thresholds=(0.70, 0.75, 0.80, 0.85, 0.90, 0.95)):
    print(f"\n{name} — win rate by probability threshold:")
    print(f"  {'Threshold':>10}  {'Win Rate':>10}  {'Trades':>8}  {'Coverage':>10}")
    total = len(labels)
    for t in thresholds:
        wr, n = win_rate_at_threshold(probs, labels, t)
        print(f"  {t:>10.2f}  {100*wr:>9.1f}%  {n:>8,}  {100*n/total:>9.1f}%")


def print_margin_breakdown(name, probs, labels, rows, threshold=0.80):
    buckets = defaultdict(lambda: [0, 0])  # margin_bucket → [won, total]
    for p, l, r in zip(probs, labels, rows):
        if p < threshold:
            continue
        m = float(r["margin_f"])
        if m < 1:
            bucket = "<1°F"
        elif m < 2:
            bucket = "1-2°F"
        elif m < 3:
            bucket = "2-3°F"
        elif m < 5:
            bucket = "3-5°F"
        else:
            bucket = "5+°F"
        buckets[bucket][0] += l
        buckets[bucket][1] += 1

    order = ["<1°F", "1-2°F", "2-3°F", "3-5°F", "5+°F"]
    print(f"\n{name} (p≥{threshold}) — by margin from ceiling:")
    for b in order:
        won, n = buckets[b]
        if n:
            print(f"  margin {b:>6}: {100*won/n:>5.1f}%  n={n}")


def disagreement_analysis(kalshi_probs, delta_probs, labels, rows, threshold=0.80):
    """Rows where models differ in their trade decision."""
    k_fires = kalshi_probs >= threshold
    d_fires = delta_probs  >= threshold

    # Delta fires but Kalshi doesn't
    d_only = [(p_k, p_d, l, r) for p_k, p_d, l, r, kf, df
              in zip(kalshi_probs, delta_probs, labels, rows, k_fires, d_fires)
              if df and not kf]
    # Kalshi fires but Delta doesn't
    k_only = [(p_k, p_d, l, r) for p_k, p_d, l, r, kf, df
              in zip(kalshi_probs, delta_probs, labels, rows, k_fires, d_fires)
              if kf and not df]

    print(f"\nDisagreements at p≥{threshold}:")
    if k_only:
        wr = sum(l for _, _, l, _ in k_only) / len(k_only)
        print(f"  Kalshi fires, Delta doesn't: {len(k_only):,} trades  WR={100*wr:.1f}%  "
              f"← trades Delta would skip")
    if d_only:
        wr = sum(l for _, _, l, _ in d_only) / len(d_only)
        print(f"  Delta fires, Kalshi doesn't: {len(d_only):,} trades  WR={100*wr:.1f}%  "
              f"← new trades Delta would add")

    both   = sum(1 for kf, df in zip(k_fires, d_fires) if kf and df)
    neither= sum(1 for kf, df in zip(k_fires, d_fires) if not kf and not df)
    print(f"  Both fire:    {both:,}")
    print(f"  Neither fire: {neither:,}")


def main():
    print("Loading models...")
    kalshi_pkg  = pickle.loads(KALSHI_PKL.read_bytes())
    kalshi_lgbm = kalshi_pkg["lgbm"]
    kalshi_iso  = kalshi_pkg.get("isotonic")
    kalshi_cmap = kalshi_pkg["city_map"]

    delta_pkg   = pickle.loads(DELTA_PKL.read_bytes())
    delta_lgbm  = delta_pkg["model"]
    delta_cmap  = delta_pkg["city_map"]
    delta_stds  = json.loads(DELTA_STDS.read_text())

    print(f"Kalshi model trained on: {kalshi_pkg.get('training_range', 'unknown')}")
    print(f"Kalshi AUC: {kalshi_pkg.get('auc', 'N/A'):.4f}  Brier: {kalshi_pkg.get('brier', 'N/A'):.4f}")
    print(f"Delta model trained on: {delta_pkg.get('training_range')}")
    print(f"Delta test MAE: {delta_pkg.get('test_mae')}°F  RMSE: {delta_pkg.get('test_rmse')}°F")

    clim_lookup = load_clim_lookup(DELTA_CSV)

    print(f"\nLoading spring Kalshi data from {KALSHI_CSV}...")
    rows = list(csv.DictReader(KALSHI_CSV.open()))
    labels = np.array([int(r["won"]) for r in rows])
    print(f"  {len(rows):,} rows  WR={100*labels.mean():.1f}%")
    print(f"  High: {sum(1 for r in rows if r['is_high']=='1'):,}  "
          f"Low: {sum(1 for r in rows if r['is_high']=='0'):,}")

    # ── Run Kalshi model ──────────────────────────────────────────────────────
    print("\nRunning existing Kalshi model...")
    Xk = build_kalshi_X(rows, kalshi_cmap)
    kalshi_probs_raw = kalshi_lgbm.predict_proba(Xk)[:, 1]
    if kalshi_iso is not None:
        kalshi_probs = kalshi_iso.predict(kalshi_probs_raw)
    else:
        kalshi_probs = kalshi_probs_raw

    # ── Run Delta model ───────────────────────────────────────────────────────
    print("Running delta model...")
    Xd = build_delta_X(rows, delta_cmap, clim_lookup)
    delta_pred = delta_lgbm.predict(Xd)
    delta_probs = delta_to_prob(delta_pred, rows, delta_stds)

    # ── Overall metrics ───────────────────────────────────────────────────────
    kalshi_auc = roc_auc_score(labels, kalshi_probs)
    delta_auc  = roc_auc_score(labels, delta_probs)
    kalshi_bs  = brier_score_loss(labels, kalshi_probs)
    delta_bs   = brier_score_loss(labels, delta_probs)

    print("\n" + "="*60)
    print("HEAD-TO-HEAD: SPRING 2026 KALSHI DATA")
    print("="*60)
    print(f"\n{'Metric':<30} {'Kalshi model':>14} {'Delta model':>14}")
    print(f"{'─'*58}")
    print(f"{'AUC':<30} {kalshi_auc:>14.4f} {delta_auc:>14.4f}")
    print(f"{'Brier score (lower=better)':<30} {kalshi_bs:>14.4f} {delta_bs:>14.4f}")
    print(f"{'Mean predicted prob':<30} {kalshi_probs.mean():>14.3f} {delta_probs.mean():>14.3f}")
    print(f"{'Actual win rate':<30} {labels.mean():>14.3f} {'':>14}")

    print_threshold_table("Kalshi model", kalshi_probs, labels)
    print_threshold_table("Delta model",  delta_probs,  labels)

    # ── By is_high ────────────────────────────────────────────────────────────
    print("\n" + "─"*60)
    print("Breakdown by market type (at p≥0.80):")
    for is_hi, label in (("1", "HIGH"), ("0", "LOW")):
        idx = [i for i, r in enumerate(rows) if r["is_high"] == is_hi]
        if not idx:
            continue
        kp = kalshi_probs[idx]; dp = delta_probs[idx]; lb = labels[idx]
        k_wr, k_n = win_rate_at_threshold(kp, lb, 0.80)
        d_wr, d_n = win_rate_at_threshold(dp, lb, 0.80)
        print(f"  {label}: Kalshi WR={100*k_wr:.1f}% n={k_n:,}  |  "
              f"Delta WR={100*d_wr:.1f}% n={d_n:,}  (actual WR={100*lb.mean():.1f}%)")

    # ── By month ──────────────────────────────────────────────────────────────
    print("\nBreakdown by month (at p≥0.80):")
    for m in ("3", "4", "5"):
        idx = [i for i, r in enumerate(rows) if r["month"] == m]
        if not idx:
            continue
        kp = kalshi_probs[idx]; dp = delta_probs[idx]; lb = labels[idx]
        k_wr, k_n = win_rate_at_threshold(kp, lb, 0.80)
        d_wr, d_n = win_rate_at_threshold(dp, lb, 0.80)
        print(f"  Month {m}: Kalshi WR={100*k_wr:.1f}% n={k_n:,}  |  "
              f"Delta WR={100*d_wr:.1f}% n={d_n:,}  (actual WR={100*lb.mean():.1f}%)")

    # ── Margin breakdown ──────────────────────────────────────────────────────
    print_margin_breakdown("Kalshi model", kalshi_probs, labels, rows, 0.80)
    print_margin_breakdown("Delta model",  delta_probs,  labels, rows, 0.80)

    # ── Disagreement analysis ─────────────────────────────────────────────────
    disagreement_analysis(kalshi_probs, delta_probs, labels, rows, threshold=0.80)
    disagreement_analysis(kalshi_probs, delta_probs, labels, rows, threshold=0.85)

    # ── Distribution of model probabilities ───────────────────────────────────
    print("\nProbability distribution:")
    print(f"  {'Bucket':<12} {'Kalshi n':>10} {'Kalshi WR':>10} {'Delta n':>10} {'Delta WR':>10}")
    for lo, hi in [(0.0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.85), (0.85, 0.9), (0.9, 1.01)]:
        ki = [i for i, p in enumerate(kalshi_probs) if lo <= p < hi]
        di = [i for i, p in enumerate(delta_probs)  if lo <= p < hi]
        kwr = labels[ki].mean() if ki else 0
        dwr = labels[di].mean() if di else 0
        print(f"  {lo:.2f}–{hi:.2f}:     {len(ki):>9,}  {100*kwr:>9.1f}%  "
              f"{len(di):>9,}  {100*dwr:>9.1f}%")

    # ── Correlation ───────────────────────────────────────────────────────────
    corr = np.corrcoef(kalshi_probs, delta_probs)[0, 1]
    print(f"\nCorrelation between model probabilities: {corr:.3f}")

    # ── Combined ensemble ─────────────────────────────────────────────────────
    ensemble = (kalshi_probs + delta_probs) / 2
    ens_auc  = roc_auc_score(labels, ensemble)
    print(f"\nEnsemble (average) AUC: {ens_auc:.4f}  (vs Kalshi {kalshi_auc:.4f}, Delta {delta_auc:.4f})")
    print("\nEnsemble win rates:")
    print(f"  {'Threshold':>10}  {'Win Rate':>10}  {'Trades':>8}")
    total = len(labels)
    for t in (0.70, 0.75, 0.80, 0.85, 0.90):
        wr, n = win_rate_at_threshold(ensemble, labels, t)
        print(f"  {t:>10.2f}  {100*wr:>9.1f}%  {n:>8,}  ({100*n/total:.1f}%)")


if __name__ == "__main__":
    main()
