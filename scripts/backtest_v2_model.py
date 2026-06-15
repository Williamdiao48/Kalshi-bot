"""
Head-to-head backtest: v1 vs v2 forecast_no model.

Data: chronological test-set holdout from forecast_no_training_data_kalshi_v2.csv
      (dates >= 80th-percentile date cutoff — same split used during training).

Sections:
  A. AUC, Brier, WR@threshold — both models on same test rows
  B. Win rate and EV by model_p threshold (sweep 0.60 → 0.95)
  C. Agreement analysis — where do models disagree and who is right?
  D. Feature lift: new v2 features vs actual outcome
  E. Estimated EV using OLS price model from shadow data
  F. Head-to-head by market type (HIGH vs LOW) and time of day

All EV estimates assume $1 contract size, NO cost ≈ OLS-estimated market_p_no.
"""

import os, sys, sqlite3, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CSV_V2   = "data/backtest/forecast_no_training_data_kalshi_v2.csv"
MODEL_V1_H = "data/models/forecast_no_band_model_high.pkl"
MODEL_V1_L = "data/models/forecast_no_band_model_low.pkl"
MODEL_V2_H = "data/models/forecast_no_band_model_v2_high.pkl"
MODEL_V2_L = "data/models/forecast_no_band_model_v2_low.pkl"
DB_PATH    = "data/db/opportunity_log.db"
OUT_PATH   = "data/backtest/v2_backtest_report.txt"

# ── helpers ──────────────────────────────────────────────────────────────────

def load_model(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["lgbm"], obj["isotonic"], obj["features"], obj["city_map"]


def predict(df: pd.DataFrame, lgbm, iso, feats, cmap) -> np.ndarray:
    city_enc = df["city"].map(lambda c: float(cmap.get(c, 0)))
    X = np.zeros((len(df), len(feats)))
    for i, feat in enumerate(feats):
        if feat == "city_enc":
            X[:, i] = city_enc.values
        elif feat in df.columns:
            X[:, i] = df[feat].fillna(0).values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = lgbm.predict_proba(X)[:, 1]
    return iso.predict(raw)


def load_shadow_for_ols() -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    df  = pd.read_sql_query(
        "SELECT market_p_no, margin_f, hvc, hour_utc, is_high FROM shadow_model_no "
        "WHERE outcome IS NOT NULL", con)
    con.close()
    df["no_cost"] = 1.0 - df["market_p_no"]
    return df


def fit_price_ols(shadow: pd.DataFrame):
    """OLS: predict no_cost from (margin_f, hvc, hour_utc, is_high)."""
    X = shadow[["margin_f", "hvc", "hour_utc", "is_high"]].values
    X = np.column_stack([X, np.ones(len(X))])
    y = shadow["no_cost"].values
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def est_no_cost(df: pd.DataFrame, coef: np.ndarray) -> np.ndarray:
    X = df[["margin_f", "hrrr_vs_ceil", "hour_utc", "is_high"]].fillna(0).values
    X = np.column_stack([X, np.ones(len(X))])
    return np.clip(X @ coef, 0.01, 0.99)


def ev_series(won: np.ndarray, no_cost: np.ndarray) -> np.ndarray:
    return won * (1.0 - no_cost) - (1 - won) * no_cost


def thr_stats(model_p: np.ndarray, won: np.ndarray, no_cost: np.ndarray, thr: float) -> dict:
    mask = model_p >= thr
    n    = mask.sum()
    if n == 0:
        return {"n": 0, "wr": 0, "ev_pt": 0, "total_ev": 0, "avg_mp": 0}
    w  = won[mask]
    nc = no_cost[mask]
    ev = ev_series(w, nc)
    return {
        "n":        n,
        "wr":       w.mean(),
        "ev_pt":    ev.mean() * 100,
        "total_ev": ev.sum()  * 100,
        "avg_mp":   model_p[mask].mean(),
    }


# ── load data + split ─────────────────────────────────────────────────────────

def load_test() -> pd.DataFrame:
    df = pd.read_csv(CSV_V2)
    df = df.sort_values(["date", "hour_utc"]).reset_index(drop=True)
    dates = sorted(df["date"].unique())
    cut   = dates[int(len(dates) * 0.80)]
    test  = df[df["date"] >= cut].copy().reset_index(drop=True)
    print(f"Test set: {len(test):,} rows  {test['date'].min()} → {test['date'].max()}")
    print(f"  HIGH: {(test['is_high']==1).sum():,}  LOW: {(test['is_high']==0).sum():,}"
          f"  WR: {test['won'].mean():.1%}")
    return test, cut


# ── sections ─────────────────────────────────────────────────────────────────

def section_a(test: pd.DataFrame, p_v1: np.ndarray, p_v2: np.ndarray) -> list[str]:
    lines = ["=" * 90,
             "SECTION A — Overall AUC, Brier, calibration",
             "=" * 90]
    y = test["won"].values
    base_brier = brier_score_loss(y, np.full(len(y), y.mean()))

    def row(label, p, yt=None):
        yt = yt if yt is not None else y
        auc = roc_auc_score(yt, p)
        b   = brier_score_loss(yt, p)
        bb  = brier_score_loss(yt, np.full(len(yt), yt.mean()))
        return f"  {label:<35} AUC={auc:.4f}  Brier={b:.4f} (base={bb:.4f})"

    lines.append(row("v1 (current deployed model)", p_v1))
    lines.append(row("v2 (new alpha features)",     p_v2))
    lines.append(f"  delta AUC (v2 - v1):                {roc_auc_score(y, p_v2) - roc_auc_score(y, p_v1):>+.4f}")

    # Per-market-type
    for label, mask in [("HIGH only", test["is_high"]==1), ("LOW only", test["is_high"]==0)]:
        m = mask.values if hasattr(mask, "values") else mask
        if m.sum() < 10: continue
        y_sub = y[m]
        lines.append(f"\n  {label} (n={m.sum():,}):")
        lines.append(row(f"  v1", p_v1[m], y_sub))
        lines.append(row(f"  v2", p_v2[m], y_sub))

    return lines


def section_b(test: pd.DataFrame, p_v1: np.ndarray, p_v2: np.ndarray,
              no_cost: np.ndarray) -> list[str]:
    lines = ["\n" + "=" * 90,
             "SECTION B — WR and EV by model_p threshold (sweep)",
             "  EV in cents per $1 contract.  Positive = profitable.",
             "=" * 90]
    y = test["won"].values

    thrs = [0.60, 0.65, 0.70, 0.75, 0.80, 0.82, 0.85, 0.87, 0.90, 0.92, 0.95]
    hdr  = (f"  {'Threshold':>10} | "
            f"{'v1 n':>7} {'v1 WR':>7} {'v1 EV/tr':>9} | "
            f"{'v2 n':>7} {'v2 WR':>7} {'v2 EV/tr':>9} | "
            f"{'WR diff':>8} {'EV diff':>8}")
    lines.append(hdr)
    lines.append("  " + "-" * 87)
    for t in thrs:
        v1 = thr_stats(p_v1, y, no_cost, t)
        v2 = thr_stats(p_v2, y, no_cost, t)
        wr_diff = (v2["wr"] - v1["wr"]) * 100 if v1["n"] and v2["n"] else 0
        ev_diff = v2["ev_pt"] - v1["ev_pt"] if v1["n"] and v2["n"] else 0
        lines.append(
            f"  mp ≥ {t:.2f}   | "
            f"{v1['n']:>7,} {100*v1['wr']:>6.1f}% {v1['ev_pt']:>+8.2f}¢ | "
            f"{v2['n']:>7,} {100*v2['wr']:>6.1f}% {v2['ev_pt']:>+8.2f}¢ | "
            f"{wr_diff:>+7.1f}pp {ev_diff:>+7.2f}¢"
        )
    return lines


def section_c(test: pd.DataFrame, p_v1: np.ndarray, p_v2: np.ndarray,
              no_cost: np.ndarray) -> list[str]:
    lines = ["\n" + "=" * 90,
             "SECTION C — Disagreement analysis: where v1 and v2 differ",
             "  Gate: v1 fires (≥0.80) but v2 doesn't (< threshold), and vice-versa.",
             "=" * 90]
    y = test["won"].values
    GATE = 0.80

    v1_fires = p_v1 >= GATE
    v2_fires = p_v2 >= GATE

    agree_both  = v1_fires & v2_fires
    v1_only     = v1_fires & ~v2_fires
    v2_only     = ~v1_fires & v2_fires
    neither     = ~v1_fires & ~v2_fires

    def ds(mask, label):
        n = mask.sum()
        if n == 0:
            return f"  {label:<40} n=0"
        ev = ev_series(y[mask], no_cost[mask]).mean() * 100
        return (f"  {label:<40} n={n:>6,}  WR={100*y[mask].mean():>5.1f}%  "
                f"EV/tr={ev:>+6.2f}¢  avg_v1={100*p_v1[mask].mean():.1f}%  "
                f"avg_v2={100*p_v2[mask].mean():.1f}%")

    lines.append(ds(agree_both, "Both fire (v1≥0.80 AND v2≥0.80)"))
    lines.append(ds(v1_only,    "v1 fires, v2 doesn't (v2 disagrees)"))
    lines.append(ds(v2_only,    "v2 fires, v1 doesn't (v2 finds new)"))
    lines.append(ds(neither,    "Neither fires"))

    lines.append(f"\n  Implication of switching to v2 gate (≥0.80):")
    lines.append(f"    Trades LOST (v1_only): {v1_only.sum():,}  WR={100*y[v1_only].mean():.1f}%")
    lines.append(f"    Trades GAINED (v2_only): {v2_only.sum():,}  WR={100*y[v2_only].mean():.1f}%")

    # Higher threshold sweep: what if we use v2 ≥ 0.85 as tighter gate?
    lines.append(f"\n  At v2 ≥ 0.85 gate (tighter filter on the new signal):")
    v2_85 = p_v2 >= 0.85
    lines.append(ds(v2_85, "v2 ≥ 0.85"))

    # Where v2 is significantly higher than v1
    big_v2_lift = (p_v2 - p_v1) >= 0.05
    lines.append(f"\n  Where v2 is ≥5pp MORE confident than v1 (n={big_v2_lift.sum():,}):")
    lines.append(ds(big_v2_lift, "v2 >> v1 (+5pp)"))
    big_v1_lift = (p_v1 - p_v2) >= 0.05
    lines.append(f"\n  Where v1 is ≥5pp MORE confident than v2 (n={big_v1_lift.sum():,}):")
    lines.append(ds(big_v1_lift, "v1 >> v2 (+5pp)"))

    return lines


def section_d(test: pd.DataFrame, p_v2: np.ndarray) -> list[str]:
    lines = ["\n" + "=" * 90,
             "SECTION D — New v2 feature lift on test set (actual outcomes)",
             "=" * 90]
    y = test["won"].values

    new_feats = {
        "hrrr_skill_adj":       "HRRR confidence / recent error",
        "min_model_vs_ceil":    "min(HRRR, GFS) vs band — consensus floor",
        "margin_per_hour_left": "margin / (hours_to_close + 1) — buffer per hour",
    }
    for fname, desc in new_feats.items():
        if fname not in test.columns:
            continue
        fv = test[fname].fillna(0).values
        lines.append(f"\n  {fname} — {desc}")
        try:
            bins_ser = pd.qcut(pd.Series(fv), 5, duplicates="drop")
        except Exception:
            continue
        grp = pd.DataFrame({"b": bins_ser, "won": y, "v2p": p_v2, "fv": fv}).groupby("b", observed=True)
        lines.append(f"  {'bin':>22}  {'n':>5}  {'WR':>7}  {'avg_v2p':>8}  {'feat_range':>15}")
        for name, g in grp:
            lines.append(
                f"  {str(name):>22}  {len(g):>5}  {100*g['won'].mean():>6.1f}%  "
                f"{100*g['v2p'].mean():>7.1f}%  [{g['fv'].min():.2f},{g['fv'].max():.2f}]"
            )

    # Clim_prob threshold gate (identified in alpha analysis as a skip signal at >0.18)
    lines.append(f"\n  clim_prob_exceed > 0.18 skip-gate analysis:")
    cp = test["clim_prob_exceed"].fillna(0).values
    for lo, hi in [(0.0, 0.10), (0.10, 0.18), (0.18, 0.30), (0.30, 1.0)]:
        mask = (cp >= lo) & (cp < hi)
        if mask.sum() < 10: continue
        lines.append(f"  clim_prob [{lo:.2f},{hi:.2f})  n={mask.sum():>6,}  "
                     f"WR={100*y[mask].mean():>5.1f}%  avg_v2p={100*p_v2[mask].mean():>5.1f}%")

    return lines


def section_e(test: pd.DataFrame, p_v1: np.ndarray, p_v2: np.ndarray,
              no_cost: np.ndarray) -> list[str]:
    lines = ["\n" + "=" * 90,
             "SECTION E — Estimated EV head-to-head",
             "  NO cost estimated from shadow OLS model.  Values in cents per $1 contract.",
             "=" * 90]
    y  = test["won"].values
    ev = ev_series(y, no_cost)

    def scenario(label, mask):
        n = mask.sum()
        if n == 0: return f"  {label:<50} n=0"
        wr   = y[mask].mean()
        evpt = ev[mask].mean() * 100
        tev  = ev[mask].sum()  * 100
        dep  = no_cost[mask].sum()
        roi  = (ev[mask].sum() / dep) if dep > 0 else 0
        return (f"  {label:<50} n={n:>6,}  WR={100*wr:>5.1f}%  "
                f"EV/tr={evpt:>+6.2f}¢  totEV={tev:>+9.1f}¢  ROI={roi:>+.3f}")

    lines.append(scenario("v1 model ≥ 0.80",             p_v1 >= 0.80))
    lines.append(scenario("v1 model ≥ 0.85",             p_v1 >= 0.85))
    lines.append(scenario("v2 model ≥ 0.80",             p_v2 >= 0.80))
    lines.append(scenario("v2 model ≥ 0.85",             p_v2 >= 0.85))
    lines.append(scenario("v2 model ≥ 0.80 + yes_ask≤55 (cost≥0.45)", (p_v2 >= 0.80) & (no_cost <= 0.55)))
    lines.append(scenario("v1 AND v2 ≥ 0.80 (consensus)", (p_v1 >= 0.80) & (p_v2 >= 0.80)))
    lines.append(scenario("v2 ≥ 0.80, v1 < 0.80 (v2 new finds)",      (p_v2 >= 0.80) & (p_v1 < 0.80)))

    # Optimal v2 threshold
    lines.append(f"\n  Threshold sweep for v2 by total EV:")
    lines.append(f"  {'Threshold':>10}  {'n':>6}  {'WR':>7}  {'EV/tr':>9}  {'total EV':>10}  {'ROI':>8}")
    for t in [0.60, 0.65, 0.70, 0.75, 0.80, 0.82, 0.85, 0.87, 0.90]:
        mask = p_v2 >= t
        n    = mask.sum()
        if n == 0: continue
        wr   = y[mask].mean()
        evpt = ev[mask].mean() * 100
        tev  = ev[mask].sum()  * 100
        dep  = no_cost[mask].sum()
        roi  = ev[mask].sum() / dep if dep > 0 else 0
        lines.append(f"  v2 ≥ {t:.2f}     {n:>6,}  {100*wr:>6.1f}%  {evpt:>+8.2f}¢  {tev:>+9.1f}¢  {roi:>+7.3f}")

    return lines


def section_f(test: pd.DataFrame, p_v1: np.ndarray, p_v2: np.ndarray,
              no_cost: np.ndarray) -> list[str]:
    lines = ["\n" + "=" * 90,
             "SECTION F — Breakdown by market type and time of day",
             "=" * 90]
    y  = test["won"].values
    ev = ev_series(y, no_cost)

    def row(label, mask_base, mp):
        m = mask_base & (mp >= 0.80)
        n = m.sum()
        if n < 5: return f"  {label:<45} n<5"
        wr  = y[m].mean()
        evpt = ev[m].mean() * 100
        return f"  {label:<45} n={n:>5,}  WR={100*wr:>5.1f}%  EV/tr={evpt:>+6.2f}¢"

    high  = test["is_high"] == 1
    low   = test["is_high"] == 0
    early = test["hour_utc"] <= 10     # pre-market / morning
    mid   = (test["hour_utc"] > 10) & (test["hour_utc"] <= 16)
    late  = test["hour_utc"] > 16

    for label, mask in [
        ("HIGH markets", high), ("LOW markets", low),
        ("Early (UTC ≤10)", early), ("Mid (UTC 11-16)", mid), ("Late (UTC >16)", late),
        ("HIGH + early", high & early), ("HIGH + mid", high & mid), ("HIGH + late", high & late),
        ("LOW + early",  low & early),  ("LOW + mid",  low & mid),  ("LOW + late",  low & late),
    ]:
        v1r = row(f"v1 | {label}", mask, p_v1)
        v2r = row(f"v2 | {label}", mask, p_v2)
        lines.append(v1r)
        lines.append(v2r)
        lines.append("")

    return lines


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    test, cut = load_test()
    y = test["won"].values

    print("Loading models...")
    v1h_lgbm, v1h_iso, v1h_feats, v1h_cmap = load_model(MODEL_V1_H)
    v1l_lgbm, v1l_iso, v1l_feats, v1l_cmap = load_model(MODEL_V1_L)
    v2h_lgbm, v2h_iso, v2h_feats, v2h_cmap = load_model(MODEL_V2_H)
    v2l_lgbm, v2l_iso, v2l_feats, v2l_cmap = load_model(MODEL_V2_L)

    high_mask = test["is_high"] == 1
    low_mask  = test["is_high"] == 0

    print("Running inference...")
    p_v1 = np.zeros(len(test))
    p_v2 = np.zeros(len(test))
    p_v1[high_mask] = predict(test[high_mask], v1h_lgbm, v1h_iso, v1h_feats, v1h_cmap)
    p_v1[low_mask]  = predict(test[low_mask],  v1l_lgbm, v1l_iso, v1l_feats, v1l_cmap)
    p_v2[high_mask] = predict(test[high_mask], v2h_lgbm, v2h_iso, v2h_feats, v2h_cmap)
    p_v2[low_mask]  = predict(test[low_mask],  v2l_lgbm, v2l_iso, v2l_feats, v2l_cmap)

    print("Fitting price OLS from shadow data...")
    shadow = load_shadow_for_ols()
    coef   = fit_price_ols(shadow)
    no_cost = est_no_cost(test, coef)
    print(f"  OLS coef: {coef.round(4).tolist()}")
    print(f"  Est NO cost: mean={100*no_cost.mean():.1f}¢  "
          f"shadow actual mean={100*shadow['no_cost'].mean():.1f}¢")

    all_lines = [
        "#" * 90,
        "# V2 MODEL BACKTEST — v1 vs v2 head-to-head on chronological test holdout",
        f"# Test period: {test['date'].min()} → {test['date'].max()}  (cutoff: {cut})",
        f"# Test rows: {len(test):,}  HIGH: {high_mask.sum():,}  LOW: {low_mask.sum():,}",
        f"# Overall WR: {y.mean():.1%}",
        "# EV estimated from OLS price model fitted on 1,032 shadow trades.",
        "#" * 90,
    ]

    print("Building report...")
    all_lines += section_a(test, p_v1, p_v2)
    all_lines += section_b(test, p_v1, p_v2, no_cost)
    all_lines += section_c(test, p_v1, p_v2, no_cost)
    all_lines += section_d(test, p_v2)
    all_lines += section_e(test, p_v1, p_v2, no_cost)
    all_lines += section_f(test, p_v1, p_v2, no_cost)

    all_lines += ["\n" + "=" * 90, "END OF REPORT", "=" * 90]
    report = "\n".join(all_lines)
    print("\n" + report)
    with open(OUT_PATH, "w") as f:
        f.write(report)
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
