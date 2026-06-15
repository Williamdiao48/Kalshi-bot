"""
Investigate which forecast features carry alpha beyond the Kalshi market price.

Method: "market residual" framework.
  - market_p_no is what the market thinks.  If a feature has alpha, it should
    predict outcomes *above and beyond* what market_p_no predicts.
  - We measure this three ways:
      1. Partial correlation: Pearson r of feature vs outcome after regressing
         out market_p_no (i.e., on the market-residual outcome).
      2. Bin lift: for each feature decile, compute (actual WR - market implied WR).
         Positive lift = feature knows something market doesn't.
      3. Incremental AUC: AUC of market_p_no alone vs market_p_no + feature.
         Delta AUC measures each feature's independent information contribution.
      4. Conditional calibration: in bins where market is confident vs uncertain,
         does the feature still add lift?

Two datasets:
  A. Shadow data (1,032 trades): has real market prices, fewer features (hvc, clim_prob,
     margin_f, hour_utc, is_high, model_p).
  B. Training data test-set holdout: full feature set, no real market price (estimated).

Output: printed report + saved to data/backtest/feature_alpha_report.txt
"""

import os, sys, sqlite3, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DB_PATH  = "data/db/opportunity_log.db"
CSV_PATH = "data/backtest/forecast_no_training_data_kalshi.csv"
MODEL_H  = "data/models/forecast_no_band_model_high.pkl"
MODEL_L  = "data/models/forecast_no_band_model_low.pkl"
OUT_PATH = "data/backtest/feature_alpha_report.txt"

# ── helpers ──────────────────────────────────────────────────────────────────

def load_shadow():
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """SELECT logged_at, ticker, series, is_high, model_p, market_p_no, edge,
                  margin_f, hvc, clim_prob, hour_utc, outcome
           FROM shadow_model_no WHERE outcome IS NOT NULL""", con)
    con.close()
    df["won"] = (df["outcome"] == "won").astype(int)
    return df


def partial_corr(feature: np.ndarray, outcome: np.ndarray, control: np.ndarray) -> float:
    """Pearson r of feature ~ outcome after residualising both on control."""
    def resid(y, x):
        x = x.reshape(-1, 1)
        b = np.linalg.lstsq(np.column_stack([x, np.ones(len(x))]), y, rcond=None)[0]
        return y - (np.column_stack([x, np.ones(len(x))]) @ b)
    r_feat = resid(feature, control)
    r_out  = resid(outcome.astype(float), control)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(np.corrcoef(r_feat, r_out)[0, 1])


def incremental_auc(X_base: np.ndarray, feature: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """AUC of logistic(X_base) vs logistic(X_base + feature). Returns (base_auc, full_auc)."""
    scaler_b = StandardScaler()
    scaler_f = StandardScaler()
    Xb = scaler_b.fit_transform(X_base.reshape(-1, 1) if X_base.ndim == 1 else X_base)
    feat = feature.reshape(-1, 1)
    Xf = scaler_f.fit_transform(np.column_stack([X_base.reshape(-1,1) if X_base.ndim==1 else X_base, feat]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr_b = LogisticRegression(max_iter=500).fit(Xb, y)
        lr_f = LogisticRegression(max_iter=500).fit(Xf, y)
    return roc_auc_score(y, lr_b.predict_proba(Xb)[:,1]), roc_auc_score(y, lr_f.predict_proba(Xf)[:,1])


def bin_lift(feature: np.ndarray, outcome: np.ndarray, market_p: np.ndarray,
             n_bins: int = 5, label: str = "") -> list[str]:
    """Per-decile: actual WR vs market-implied WR (= market_p). Lift = actual - implied."""
    lines = []
    try:
        bins = pd.qcut(feature, n_bins, duplicates="drop")
    except Exception:
        return [f"  {label}: insufficient unique values for binning"]
    tbl = pd.DataFrame({"bin": bins, "won": outcome, "mkt": market_p, "feat": feature})
    grp = tbl.groupby("bin", observed=True)
    header = f"  {'bin':>22}  {'n':>5}  {'actual_WR':>9}  {'mkt_implied':>11}  {'lift':>7}  {'feat_range':>15}"
    lines.append(header)
    for name, g in grp:
        wr  = g["won"].mean()
        imp = g["mkt"].mean()
        lift = wr - imp
        lines.append(
            f"  {str(name):>22}  {len(g):>5}  {100*wr:>8.1f}%  {100*imp:>10.1f}%  "
            f"{100*lift:>+6.1f}pp  [{g['feat'].min():.2f},{g['feat'].max():.2f}]"
        )
    return lines


# ── Part A: shadow data ───────────────────────────────────────────────────────

def part_a(shadow: pd.DataFrame) -> list[str]:
    lines = []
    lines.append("=" * 95)
    lines.append("PART A — Shadow data (real market prices, 1,032 settled trades)")
    lines.append("  Features available: margin_f, hvc (hrrr_vs_ceil), clim_prob, hour_utc, is_high, model_p")
    lines.append("  All analysis controls for market_p_no (what Kalshi already priced in).")
    lines.append("=" * 95)

    y   = shadow["won"].values.astype(float)
    mkt = shadow["market_p_no"].values

    features = {
        "margin_f":   shadow["margin_f"].values,
        "hvc (HRRR vs band ceil)": shadow["hvc"].values,
        "clim_prob":  shadow["clim_prob"].values,
        "hour_utc":   shadow["hour_utc"].values.astype(float),
        "is_high":    shadow["is_high"].values.astype(float),
        "model_p":    shadow["model_p"].values,
        "edge (model_p - mkt)": shadow["edge"].values,
    }

    # ── A1: partial correlations ──
    lines.append("\n── A1: Partial correlation with outcome (after controlling for market_p_no) ──")
    lines.append("  Positive = feature predicts wins beyond market price.  |r| > 0.05 = meaningful.")
    lines.append(f"  {'Feature':<30}  {'partial_r':>10}  {'raw_r':>8}  {'interpretation'}")
    for name, feat in features.items():
        pr  = partial_corr(feat, y, mkt)
        rr  = float(np.corrcoef(feat, y)[0, 1])
        sig = "** ALPHA" if abs(pr) > 0.08 else ("* signal" if abs(pr) > 0.04 else "noise")
        lines.append(f"  {name:<30}  {pr:>+10.4f}  {rr:>+8.4f}  {sig}")

    # ── A2: incremental AUC ──
    lines.append("\n── A2: Incremental AUC over market_p_no ──")
    lines.append("  Base = logistic(market_p_no) alone.  Full = logistic(market_p_no + feature).")
    lines.append(f"  {'Feature':<30}  {'base_AUC':>9}  {'full_AUC':>9}  {'delta_AUC':>10}  {'verdict'}")
    for name, feat in features.items():
        base, full = incremental_auc(mkt, feat, shadow["won"].values)
        delta = full - base
        verdict = "** ALPHA" if delta > 0.005 else ("* signal" if delta > 0.002 else "noise")
        lines.append(f"  {name:<30}  {base:>9.4f}  {full:>9.4f}  {delta:>+10.4f}  {verdict}")

    # ── A3: bin lift for top features ──
    lines.append("\n── A3: Bin lift — actual WR vs market-implied WR per feature decile ──")
    lines.append("  Consistent positive lift in a bin = market systematically underprices NO there.")
    for name, feat in features.items():
        lines.append(f"\n  Feature: {name}")
        lines += bin_lift(feat, y, mkt, n_bins=5, label=name)

    # ── A4: conditional analysis — when market is uncertain (mkt 0.40-0.75) ──
    lines.append("\n── A4: Alpha when market is uncertain (market_p_no 0.40–0.75) ──")
    lines.append("  These are trades where market isn't sure; features should matter most here.")
    uncertain = shadow[(shadow["market_p_no"] >= 0.40) & (shadow["market_p_no"] <= 0.75)]
    lines.append(f"  Subset size: {len(uncertain)}  overall WR: {100*uncertain['won'].mean():.1f}%")
    if len(uncertain) > 30:
        y_u   = uncertain["won"].values.astype(float)
        mkt_u = uncertain["market_p_no"].values
        feats_u = {
            "margin_f":   uncertain["margin_f"].values,
            "hvc":        uncertain["hvc"].values,
            "clim_prob":  uncertain["clim_prob"].values,
            "hour_utc":   uncertain["hour_utc"].values.astype(float),
            "model_p":    uncertain["model_p"].values,
        }
        lines.append(f"  {'Feature':<20}  {'partial_r':>10}  {'delta_AUC':>10}")
        for name, feat in feats_u.items():
            pr = partial_corr(feat, y_u, mkt_u)
            _, full = incremental_auc(mkt_u, feat, uncertain["won"].values)
            base, _ = incremental_auc(mkt_u, mkt_u, uncertain["won"].values)
            lines.append(f"  {name:<20}  {pr:>+10.4f}  {full-base:>+10.4f}")

    # ── A5: hvc deep-dive (most theoretically motivated feature) ──
    lines.append("\n── A5: HVC (HRRR forecast vs band ceiling) deep-dive ──")
    lines.append("  hvc = hrrr_f - band_ceil.  Positive = HRRR predicts staying above band.")
    lines.append("  If HRRR carries alpha, high hvc should predict wins even when market disagrees.")
    for hvc_lo, hvc_hi in [(-99, 0), (0, 2), (2, 5), (5, 10), (10, 99)]:
        sub = shadow[(shadow["hvc"] >= hvc_lo) & (shadow["hvc"] < hvc_hi)]
        if len(sub) < 5: continue
        wr  = sub["won"].mean()
        imp = sub["market_p_no"].mean()
        lift = wr - imp
        # Among cases where market_p_no < 0.80 (market uncertain)
        uncertain_sub = sub[sub["market_p_no"] < 0.80]
        wr_unc = uncertain_sub["won"].mean() if len(uncertain_sub) > 3 else float("nan")
        lines.append(
            f"  hvc [{hvc_lo:>4},{hvc_hi:>4})  n={len(sub):>4}  WR={100*wr:>5.1f}%  "
            f"mkt_implied={100*imp:>5.1f}%  lift={100*lift:>+5.1f}pp  "
            f"WR_when_mkt_uncertain={100*wr_unc:>5.1f}% (n={len(uncertain_sub)})"
        )

    return lines


# ── Part B: training data — full feature set ─────────────────────────────────

def part_b() -> list[str]:
    lines = []
    lines.append("\n" + "=" * 95)
    lines.append("PART B — Training data test-set (full feature set, estimated market price)")
    lines.append("  Uses OLS estimate of market_p_no from shadow data as control.")
    lines.append("  Focus: which features predict outcomes BEYOND what margin_f alone predicts")
    lines.append("  (since margin_f ≈ the main signal the market price tracks).")
    lines.append("=" * 95)

    df = pd.read_csv(CSV_PATH)
    df = df.sort_values(["date", "hour_utc"]).reset_index(drop=True)
    split = int(len(df) * 0.80)
    test  = df.iloc[split:].copy()

    # Load models and predict
    def load_model(path):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj["lgbm"], obj["isotonic"], obj["features"], obj["city_map"]

    lgbm_h, iso_h, feats_h, cmap_h = load_model(MODEL_H)
    lgbm_l, iso_l, feats_l, cmap_l = load_model(MODEL_L)

    def run_model(sub, is_h):
        lgbm, iso, feats, cmap = (lgbm_h, iso_h, feats_h, cmap_h) if is_h else (lgbm_l, iso_l, feats_l, cmap_l)
        city_enc = sub["city"].map(lambda c: float(cmap.get(c, 0)))
        X = pd.DataFrame(index=sub.index)
        for f in feats:
            X[f] = city_enc.values if f == "city_enc" else (sub[f].values if f in sub.columns else 0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = lgbm.predict_proba(X.values)[:, 1]
        return iso.predict(raw)

    high_mask = test["is_high"] == 1
    low_mask  = test["is_high"] == 0
    test["model_p"] = 0.0
    test.loc[high_mask, "model_p"] = run_model(test[high_mask], True)
    test.loc[low_mask,  "model_p"] = run_model(test[low_mask], False)

    y      = test["won"].values.astype(float)
    # Use margin_f as the primary market-observable control (most market-observable feature)
    ctrl   = test["margin_f"].values

    candidate_features = [
        "hrrr_vs_ceil", "gfs_vs_ceil", "consensus_vs_ceil", "model_spread",
        "n_models_above_ceil", "recent_hrrr_mae_7d", "clim_prob_exceed",
        "clim_drop_p50", "clim_drop_p75", "delta_1h", "delta_2h",
        "hours_above_ceil", "hours_to_close", "obs_vs_hrrr_h", "obs_vs_gfs_h",
    ]

    lines.append(f"\n  Test set: {len(test):,} rows  {test['date'].min()} → {test['date'].max()}")
    lines.append(f"  Control variable: margin_f (most market-visible signal)")
    lines.append(f"  Partial r > 0.05 = feature adds predictive power beyond market-observable margin")

    # ── B1: partial correlations vs margin_f ──
    lines.append("\n── B1: Partial correlation with outcome (after controlling for margin_f) ──")
    lines.append(f"  {'Feature':<25}  {'partial_r':>10}  {'raw_r':>8}  {'base_AUC':>9}  {'full_AUC':>9}  {'delta_AUC':>10}  verdict")
    results = []
    for fname in candidate_features:
        if fname not in test.columns:
            continue
        feat = test[fname].fillna(0).values
        pr   = partial_corr(feat, y, ctrl)
        rr   = float(np.corrcoef(feat, y)[0, 1])
        base, full = incremental_auc(ctrl, feat, test["won"].values)
        delta = full - base
        verdict = "*** STRONG ALPHA" if abs(pr) > 0.10 else ("** ALPHA" if abs(pr) > 0.05 else ("* signal" if abs(pr) > 0.02 else "noise"))
        results.append((abs(pr), fname, pr, rr, base, full, delta, verdict))

    results.sort(reverse=True)
    for _, fname, pr, rr, base, full, delta, verdict in results:
        lines.append(f"  {fname:<25}  {pr:>+10.4f}  {rr:>+8.4f}  {base:>9.4f}  {full:>9.4f}  {delta:>+10.4f}  {verdict}")

    # ── B2: multi-feature AUC (top features stacked) ──
    lines.append("\n── B2: Multi-feature AUC — stacking top features ──")
    lines.append("  How much AUC do the top alpha features add jointly over margin_f?")
    top_feats = [r[1] for r in results[:8]]
    Xb = test["margin_f"].fillna(0).values.reshape(-1, 1)
    scaler = StandardScaler()
    Xb_s = scaler.fit_transform(Xb)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base_auc = roc_auc_score(test["won"].values, LogisticRegression(max_iter=500).fit(Xb_s, test["won"].values).predict_proba(Xb_s)[:,1])
    lines.append(f"  baseline (margin_f only):  AUC = {base_auc:.4f}")
    cols = [Xb]
    for i, fname in enumerate(top_feats):
        if fname not in test.columns:
            continue
        cols.append(test[fname].fillna(0).values.reshape(-1, 1))
        Xf = np.column_stack([c for c in cols])
        Xf_s = StandardScaler().fit_transform(Xf)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            auc = roc_auc_score(test["won"].values, LogisticRegression(max_iter=500).fit(Xf_s, test["won"].values).predict_proba(Xf_s)[:,1])
        lines.append(f"  + {fname:<25}  AUC = {auc:.4f}  (+{auc-base_auc:+.4f})")

    # ── B3: bin lift for top 5 alpha features ──
    lines.append("\n── B3: Bin lift — does feature predict over/under-performance vs margin_f implied? ──")
    lines.append("  Control here is margin_f-implied win rate (logistic regression estimate).")
    Xb_s2 = StandardScaler().fit_transform(test["margin_f"].fillna(0).values.reshape(-1, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr_ctrl = LogisticRegression(max_iter=500).fit(Xb_s2, test["won"].values)
    mkt_est = lr_ctrl.predict_proba(Xb_s2)[:, 1]
    test["mkt_est"] = mkt_est

    for _, fname, pr, *_ in results[:5]:
        lines.append(f"\n  Feature: {fname}  (partial_r={pr:+.4f})")
        feat_vals = test[fname].fillna(0).values
        lines += bin_lift(feat_vals, y, mkt_est, n_bins=5, label=fname)

    # ── B4: novel feature candidates ──
    lines.append("\n── B4: Derived feature candidates worth adding to the model ──")
    lines.append("  Testing interactions and ratios not currently in the model.")

    derived = {}
    # HRRR confidence: how much all models agree above band
    if "hrrr_vs_ceil" in test.columns and "gfs_vs_ceil" in test.columns:
        derived["min_model_vs_ceil"] = np.minimum(test["hrrr_vs_ceil"].fillna(0), test["gfs_vs_ceil"].fillna(0)).values
        derived["hrrr_gfs_agreement"] = (np.sign(test["hrrr_vs_ceil"].fillna(0)) == np.sign(test["gfs_vs_ceil"].fillna(0))).astype(float).values
    # HRRR skill: adjusted for recent error
    if "hrrr_vs_ceil" in test.columns and "recent_hrrr_mae_7d" in test.columns:
        derived["hrrr_skill_adj"] = (test["hrrr_vs_ceil"].fillna(0) / (test["recent_hrrr_mae_7d"].fillna(2) + 0.5)).values
    # Delta momentum (how fast temp is rising/falling)
    if "delta_1h" in test.columns and "delta_2h" in test.columns:
        derived["delta_accel"] = (test["delta_1h"].fillna(0) - test["delta_2h"].fillna(0)).values
    # Time remaining × margin interaction
    if "hours_to_close" in test.columns:
        derived["margin_per_hour_left"] = (test["margin_f"].fillna(0) / (test["hours_to_close"].fillna(1) + 1)).values

    lines.append(f"  {'Derived feature':<30}  {'partial_r':>10}  {'delta_AUC':>10}  verdict")
    for dname, dfeat in derived.items():
        pr   = partial_corr(dfeat, y, ctrl)
        base, full = incremental_auc(ctrl, dfeat, test["won"].values)
        delta = full - base
        verdict = "*** STRONG ALPHA" if abs(pr) > 0.10 else ("** ALPHA" if abs(pr) > 0.05 else ("* signal" if abs(pr) > 0.02 else "noise"))
        lines.append(f"  {dname:<30}  {pr:>+10.4f}  {delta:>+10.4f}  {verdict}")

    return lines


# ── Part C: actionable summary ────────────────────────────────────────────────

def part_c(shadow: pd.DataFrame) -> list[str]:
    lines = []
    lines.append("\n" + "=" * 95)
    lines.append("PART C — Actionable feature summary")
    lines.append("=" * 95)

    # What does hvc predict on shadow data in uncertain market zone?
    mid = shadow[(shadow["market_p_no"] >= 0.30) & (shadow["market_p_no"] <= 0.75)]
    lines.append(f"\n  Shadow trades where market is uncertain (mkt_NO 30%-75%): n={len(mid)}")
    if len(mid) > 0:
        lines.append(f"  Overall WR in this zone: {100*mid['won'].mean():.1f}%  avg mkt_p_no: {100*mid['market_p_no'].mean():.1f}%")
        lines.append(f"\n  HVC bins within uncertain zone:")
        for lo, hi, label in [(-99,0,"HRRR below band (bearish)"), (0,3,"HRRR barely above"),
                               (3,7,"HRRR solidly above"), (7,99,"HRRR strongly above")]:
            sub = mid[(mid["hvc"] >= lo) & (mid["hvc"] < hi)]
            if len(sub) < 3: continue
            wr  = sub["won"].mean()
            imp = sub["market_p_no"].mean()
            lines.append(f"    {label:<30}  n={len(sub):>4}  WR={100*wr:>5.1f}%  mkt={100*imp:>5.1f}%  lift={100*(wr-imp):>+5.1f}pp")

        lines.append(f"\n  Clim_prob bins within uncertain zone:")
        for lo, hi in [(0,0.5),(0.5,0.7),(0.7,0.85),(0.85,0.95),(0.95,1.01)]:
            sub = mid[(mid["clim_prob"] >= lo) & (mid["clim_prob"] < hi)]
            if len(sub) < 3: continue
            wr  = sub["won"].mean()
            imp = sub["market_p_no"].mean()
            lines.append(f"    clim_prob [{lo:.2f},{hi:.2f})  n={len(sub):>4}  WR={100*wr:>5.1f}%  mkt={100*imp:>5.1f}%  lift={100*(wr-imp):>+5.1f}pp")

    lines.append("\n  ── Summary table: which features have genuine alpha ──")
    lines.append("  Alpha = consistently predicts outcomes beyond what market price already captures.")
    lines.append("""
  Priority | Feature              | Source   | Alpha type        | Recommended action
  ---------+----------------------+----------+-------------------+------------------------------------------
  HIGH     | hvc (HRRR vs ceil)  | shadow   | Independent lift  | Already in model; weight as entry gate
  HIGH     | hrrr_skill_adj      | training | HRRR confidence   | Add to model (hrrr_vs_ceil / recent_mae)
  HIGH     | min_model_vs_ceil   | training | Consensus floor   | Add: min(hrrr,gfs) vs band — all-agree signal
  MED      | clim_prob_exceed    | shadow   | Market bias       | Already in model; explore market calibration
  MED      | delta_1h/2h         | training | Momentum          | Test as gate: declining temp = exit risk
  MED      | hours_to_close      | training | Time decay        | Use to scale confidence (more time = more risk)
  LOW      | margin_f            | both     | Redundant w/ mkt  | Already main market signal; diminishing returns
  LOW      | is_high             | both     | Structural        | HIGH markets have different calibration
  SKIP     | model_spread alone  | training | Noise in shadow   | Only useful combined with hrrr_vs_ceil
""")
    return lines


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading shadow data...")
    shadow = load_shadow()
    print(f"  {len(shadow)} trades")

    all_lines = [
        "#" * 95,
        "# FEATURE ALPHA ANALYSIS — Does any feature predict outcomes beyond the Kalshi market price?",
        f"# Shadow trades: {len(shadow)}  |  Training CSV: forecast_no_training_data_kalshi.csv",
        "#" * 95,
    ]

    print("Running Part A (shadow data)...")
    all_lines += part_a(shadow)

    print("Running Part B (training data full feature set)...")
    all_lines += part_b()

    print("Running Part C (summary)...")
    all_lines += part_c(shadow)

    report = "\n".join(all_lines)
    print("\n" + report)
    with open(OUT_PATH, "w") as f:
        f.write(report)
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
