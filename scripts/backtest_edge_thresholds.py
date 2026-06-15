"""
Edge-threshold backtest: rigorous analysis of optimal model_p and edge gates.

Part A — Shadow data (1,032 real settled trades, model_p >= 0.80):
  Sweeps model_p floor, edge floor, and combined grids.
  Uses real Kalshi market prices → real entry costs → real EV.

Part B — Model rerun on chronological test-set holdout:
  Runs the trained model on rows AFTER the 80/20 split date (model_p < 0.80 territory).
  No market price available → estimates NO entry price from hours_to_close + margin_f
  via OLS fit on shadow data, then computes approximate EV.

Part C — Combined grid: model_p floor × edge floor on shadow data.

Output: printed report.
"""

import os, sys, sqlite3, pickle, warnings
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DB_PATH    = "data/db/opportunity_log.db"
CSV_PATH   = "data/backtest/forecast_no_training_data_kalshi.csv"
MODEL_H    = "data/models/forecast_no_band_model_high.pkl"
MODEL_L    = "data/models/forecast_no_band_model_low.pkl"

# ── helpers ──────────────────────────────────────────────────────────────────

def load_shadow() -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """SELECT logged_at, ticker, is_high, model_p, market_p_no, edge,
                  margin_f, hvc, clim_prob, hour_utc, outcome
           FROM shadow_model_no WHERE outcome IS NOT NULL""",
        con
    )
    con.close()
    df["won"] = (df["outcome"] == "won").astype(int)
    # entry cost: buying NO costs (1 - market_p_no) dollars per $1 contract
    # (market_p_no = (100-yes_ask)/100; NO price ≈ 1 - market_p_no + spread, but
    # we approximate as 1 - market_p_no for conservatism)
    # Actually: you buy NO at NO_ask ≈ 100 - YES_bid.  market_p_no uses yes_ask.
    # Spread adds cost, so actual NO cost >= (1 - market_p_no).
    # We'll use (1 - market_p_no) as the optimistic (lower-bound cost) approximation.
    df["no_cost"] = 1.0 - df["market_p_no"]   # cents per $1 = fraction of $1
    # Payoff if NO wins: $1 - no_cost. If NO loses: -no_cost.
    df["ev_if_bet"] = df["won"] * (1.0 - df["no_cost"]) - (1 - df["won"]) * df["no_cost"]
    return df


def load_model(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["lgbm"], obj["isotonic"], obj["features"], obj["city_map"]


def summary_stats(sub: pd.DataFrame, label: str) -> dict:
    n = len(sub)
    if n == 0:
        return {"label": label, "n": 0, "wr": None, "avg_model_p": None,
                "avg_mkt_p_no": None, "avg_no_cost_pct": None,
                "total_ev_100c": None, "ev_per_trade_100c": None}
    wins = sub["won"].sum()
    wr   = wins / n
    avg_mkt  = sub["market_p_no"].mean()
    avg_cost = sub["no_cost"].mean()
    # EV in cents assuming $1 contract size
    total_ev = sub["ev_if_bet"].sum() * 100   # cents
    ev_pt    = sub["ev_if_bet"].mean() * 100
    return {
        "label": label, "n": n, "wr": wr,
        "avg_model_p": sub["model_p"].mean(),
        "avg_mkt_p_no": avg_mkt,
        "avg_no_cost_pct": avg_cost * 100,
        "total_ev_100c": total_ev,
        "ev_per_trade_100c": ev_pt,
    }


def fmt(d: dict) -> str:
    if d["n"] == 0:
        return f"  {d['label']:<45}  n=0"
    return (
        f"  {d['label']:<45}  n={d['n']:>5}  "
        f"WR={100*d['wr']:>5.1f}%  "
        f"model_p={100*d['avg_model_p']:>5.1f}%  "
        f"mkt_no={100*d['avg_mkt_p_no']:>5.1f}%  "
        f"NO_cost={d['avg_no_cost_pct']:>5.1f}¢  "
        f"EV/trade={d['ev_per_trade_100c']:>+6.2f}¢  "
        f"total_EV={d['total_ev_100c']:>+8.1f}¢"
    )


# ── Part A: shadow data sweeps ───────────────────────────────────────────────

def part_a(shadow: pd.DataFrame):
    lines = []
    lines.append("=" * 100)
    lines.append("PART A — Shadow data (real settled trades, model_p ≥ 0.80)")
    lines.append("  'EV' assumes $1 contract size.  NO_cost ≈ 1 - market_p_no (lower-bound approximation).")
    lines.append("  Positive EV/trade means the bet has edge; total_EV shows aggregate impact.")
    lines.append("=" * 100)

    # ── A1: model_p floor sweep (edge free) ──
    lines.append("\n── A1: model_p floor sweep (no edge filter) ──")
    lines.append(fmt(summary_stats(shadow, "Baseline: ALL (model_p ≥ 0.80)")))
    for thr in [0.80, 0.82, 0.85, 0.87, 0.90, 0.92, 0.95]:
        sub = shadow[shadow["model_p"] >= thr]
        lines.append(fmt(summary_stats(sub, f"model_p ≥ {thr:.2f}")))

    # ── A2: edge floor sweep (percentage) ──
    lines.append("\n── A2: edge floor sweep (model_p ≥ 0.80 maintained) ──")
    lines.append("  edge = model_p - market_p_no  (positive = model more bullish on NO than market)")
    for thr in [-0.10, -0.05, 0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
        sub = shadow[shadow["edge"] >= thr]
        lines.append(fmt(summary_stats(sub, f"edge ≥ {thr:+.2f} ({thr*100:+.0f}pp)")))

    # ── A3: absolute NO cost sweep (how cheap can you buy NO?) ──
    lines.append("\n── A3: NO entry cost sweep (model_p ≥ 0.80 maintained) ──")
    lines.append("  NO_cost = 1 - market_p_no.  Low cost = market hasn't priced in NO yet.")
    for max_cost in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 1.00]:
        sub = shadow[shadow["no_cost"] <= max_cost]
        lines.append(fmt(summary_stats(sub, f"NO cost ≤ {max_cost*100:.0f}¢ (mkt_NO ≥ {(1-max_cost)*100:.0f}%)")))

    # ── A4: is_high split ──
    lines.append("\n── A4: HIGH vs LOW breakdown (baseline model_p ≥ 0.80) ──")
    for label, sub in [("HIGH markets", shadow[shadow["is_high"]==1]),
                        ("LOW  markets", shadow[shadow["is_high"]==0])]:
        lines.append(fmt(summary_stats(sub, label)))
    for label, sub in [("HIGH, edge ≥ 0", shadow[(shadow["is_high"]==1) & (shadow["edge"]>=0)]),
                        ("LOW,  edge ≥ 0", shadow[(shadow["is_high"]==0) & (shadow["edge"]>=0)]),
                        ("HIGH, edge ≥ +10pp", shadow[(shadow["is_high"]==1) & (shadow["edge"]>=0.10)]),
                        ("LOW,  edge ≥ +10pp", shadow[(shadow["is_high"]==0) & (shadow["edge"]>=0.10)])]:
        lines.append(fmt(summary_stats(sub, label)))

    # ── A5: combined model_p × edge grid ──
    lines.append("\n── A5: Combined grid: model_p floor × edge floor ──")
    mp_thrs   = [0.80, 0.85, 0.90]
    edge_thrs = [-0.05, 0.0, 0.05, 0.10, 0.20]
    header = f"  {'model_p':>9} | " + " | ".join(f"edge≥{e:+.2f}  n/WR/EV" for e in edge_thrs)
    lines.append(header)
    for mp in mp_thrs:
        row = f"  mp≥{mp:.2f}   | "
        cells = []
        for e in edge_thrs:
            sub = shadow[(shadow["model_p"] >= mp) & (shadow["edge"] >= e)]
            if len(sub) == 0:
                cells.append("  n=0           ")
            else:
                wr = 100 * sub["won"].mean()
                ev = sub["ev_if_bet"].mean() * 100
                cells.append(f"n={len(sub):>4} WR={wr:>4.1f}% EV={ev:>+5.2f}¢")
        lines.append(row + " | ".join(cells))

    # ── A6: EV-optimal cutoff analysis ──
    lines.append("\n── A6: Cumulative EV — sorted by EV per trade (best trades first) ──")
    shadow_sorted = shadow.sort_values("ev_if_bet", ascending=False).reset_index(drop=True)
    shadow_sorted["cumEV"] = shadow_sorted["ev_if_bet"].cumsum() * 100
    shadow_sorted["cumN"]  = range(1, len(shadow_sorted) + 1)

    lines.append(f"  {'Top N trades':>14} | {'cumEV (¢)':>12} | {'WR':>7} | {'avg EV/tr':>10} | {'avg edge':>9}")
    for n_top in [50, 100, 200, 344, 500, 767, 1032]:
        sub = shadow_sorted.head(n_top)
        wr  = 100 * sub["won"].mean()
        ev  = sub["ev_if_bet"].mean() * 100
        cum = sub["ev_if_bet"].sum() * 100
        lines.append(f"  {'top ' + str(n_top):>14} | {cum:>12.1f} | {wr:>6.1f}% | {ev:>+9.2f}¢ | {sub['edge'].mean():>+8.3f}")

    return lines


# ── Part B: model rerun on test holdout ─────────────────────────────────────

def part_b(shadow: pd.DataFrame):
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("PART B — Model rerun on chronological test holdout (sub-0.80 territory)")
    lines.append("  Training data chronological 80/20 split; test = later 20%.")
    lines.append("  No live market prices available → NO entry price estimated from OLS on shadow data.")
    lines.append("  EV figures here are APPROXIMATE; use for directional guidance only.")
    lines.append("=" * 100)

    # Load and split training data
    df = pd.read_csv(CSV_PATH)
    df = df.sort_values(["date", "hour_utc"]).reset_index(drop=True)
    split_idx = int(len(df) * 0.80)
    test = df.iloc[split_idx:].copy()
    lines.append(f"\n  Test set: {len(test):,} rows  dates {test['date'].min()} → {test['date'].max()}")

    # Load models
    lgbm_h, iso_h, feats_h, cmap_h = load_model(MODEL_H)
    lgbm_l, iso_l, feats_l, cmap_l = load_model(MODEL_L)

    def run_model(sub, is_h):
        lgbm, iso, feats, cmap = (lgbm_h, iso_h, feats_h, cmap_h) if is_h else (lgbm_l, iso_l, feats_l, cmap_l)
        city_enc = sub["city"].map(lambda c: float(cmap.get(c, 0)))
        X = pd.DataFrame(index=sub.index)
        for f in feats:
            if f == "city_enc":
                X[f] = city_enc.values
            else:
                X[f] = sub[f].values if f in sub.columns else 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = lgbm.predict_proba(X.values)[:, 1]
        cal = iso.predict(raw)
        return cal

    # Run predictions
    high_mask = test["is_high"] == 1
    low_mask  = test["is_high"] == 0
    test["model_p"] = 0.0
    test.loc[high_mask, "model_p"] = run_model(test[high_mask], True)
    test.loc[low_mask,  "model_p"] = run_model(test[low_mask],  False)

    # Estimate NO entry cost from shadow data via OLS
    # Features available in both shadow and training CSV: hour_utc, margin_f, is_high
    # Target: no_cost (= 1 - market_p_no)
    from numpy.linalg import lstsq
    Xs = shadow[["hour_utc", "margin_f", "is_high"]].values
    Xs = np.column_stack([Xs, np.ones(len(Xs))])
    ys = shadow["no_cost"].values
    coef, _, _, _ = lstsq(Xs, ys, rcond=None)

    Xt = test[["hour_utc", "margin_f", "is_high"]].values
    Xt = np.column_stack([Xt, np.ones(len(Xt))])
    est_cost = np.clip(Xt @ coef, 0.01, 0.99)

    test["est_no_cost"] = est_cost
    test["est_market_p_no"] = 1.0 - est_cost
    test["est_ev"] = test["won"] * (1.0 - test["est_no_cost"]) - (1 - test["won"]) * test["est_no_cost"]
    # Approximate edge: model_p vs estimated market
    test["est_edge"] = test["model_p"] - test["est_market_p_no"]

    lines.append(f"  OLS price model coefficients (hour_utc, margin_f, is_high, const):")
    lines.append(f"    {coef.round(4).tolist()}")
    lines.append(f"  Estimated NO cost mean={100*test['est_no_cost'].mean():.1f}¢  "
                 f"shadow actual mean={100*shadow['no_cost'].mean():.1f}¢")

    # model_p distribution on test set
    lines.append(f"\n  model_p distribution on test set:")
    for lo, hi in [(0.0, 0.5), (0.5, 0.6), (0.6, 0.65), (0.65, 0.70),
                   (0.70, 0.75), (0.75, 0.80), (0.80, 0.85), (0.85, 0.90),
                   (0.90, 0.95), (0.95, 1.01)]:
        sub = test[(test["model_p"] >= lo) & (test["model_p"] < hi)]
        if len(sub) == 0:
            continue
        wr  = 100 * sub["won"].mean()
        ev  = sub["est_ev"].mean() * 100
        lines.append(f"  [{lo:.2f},{hi:.2f})  n={len(sub):>6}  WR={wr:>5.1f}%  "
                     f"approx EV/trade={ev:>+6.2f}¢  est_NO_cost={100*sub['est_no_cost'].mean():.1f}¢")

    # Focus on the sub-0.80 zone: is there signal?
    lines.append(f"\n── B2: Sub-0.80 zone: edge gate on estimated edge ──")
    sub80 = test[test["model_p"] < 0.80]
    lines.append(f"  Sub-0.80 rows: {len(sub80):,}  overall WR={100*sub80['won'].mean():.1f}%")
    for e_thr in [-0.10, 0.0, 0.05, 0.10, 0.15, 0.20]:
        sub = sub80[sub80["est_edge"] >= e_thr]
        if len(sub) == 0:
            continue
        wr  = 100 * sub["won"].mean()
        ev  = sub["est_ev"].mean() * 100
        lines.append(f"  sub-0.80, est_edge ≥ {e_thr:+.2f}  n={len(sub):>6}  WR={wr:>5.1f}%  "
                     f"approx EV/trade={ev:>+6.2f}¢")

    # Full model_p sweep with estimated EV
    lines.append(f"\n── B3: Full model_p floor sweep with estimated EV ──")
    for mp in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        sub = test[test["model_p"] >= mp]
        if len(sub) == 0:
            continue
        wr  = 100 * sub["won"].mean()
        ev  = sub["est_ev"].mean() * 100
        lines.append(f"  model_p ≥ {mp:.2f}  n={len(sub):>6}  WR={wr:>5.1f}%  "
                     f"approx EV/trade={ev:>+6.2f}¢  est_edge={sub['est_edge'].mean():>+.3f}")

    return lines


# ── Part C: recommendation framework ────────────────────────────────────────

def part_c(shadow: pd.DataFrame):
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("PART C — Key scenarios compared head-to-head (shadow data, real prices)")
    lines.append("=" * 100)

    scenarios = [
        ("Current (model_p ≥ 0.80, no edge filter)",
         shadow),
        ("model_p ≥ 0.80, edge ≥ 0% (drop neg-edge)",
         shadow[shadow["edge"] >= 0.0]),
        ("model_p ≥ 0.80, edge ≥ +5pp",
         shadow[shadow["edge"] >= 0.05]),
        ("model_p ≥ 0.80, edge ≥ +10pp",
         shadow[shadow["edge"] >= 0.10]),
        ("model_p ≥ 0.85, edge ≥ 0%",
         shadow[(shadow["model_p"] >= 0.85) & (shadow["edge"] >= 0.0)]),
        ("model_p ≥ 0.85, edge ≥ +5pp",
         shadow[(shadow["model_p"] >= 0.85) & (shadow["edge"] >= 0.05)]),
        ("model_p ≥ 0.90, edge ≥ 0%",
         shadow[(shadow["model_p"] >= 0.90) & (shadow["edge"] >= 0.0)]),
        ("model_p ≥ 0.90, edge ≥ +5pp",
         shadow[(shadow["model_p"] >= 0.90) & (shadow["edge"] >= 0.05)]),
        ("HIGH only, model_p ≥ 0.80, edge ≥ 0%",
         shadow[(shadow["is_high"] == 1) & (shadow["edge"] >= 0.0)]),
        ("LOW only, model_p ≥ 0.80, edge ≥ +10pp",
         shadow[(shadow["is_high"] == 0) & (shadow["edge"] >= 0.10)]),
    ]

    # Table header
    lines.append(f"\n  {'Scenario':<50} {'n':>5} {'WR':>7} {'EV/tr':>8} {'totEV':>10} {'drops':>7}")
    lines.append("  " + "-" * 95)
    base_n  = len(shadow)
    base_ev = shadow["ev_if_bet"].sum() * 100
    for name, sub in scenarios:
        n   = len(sub)
        wr  = 100 * sub["won"].mean() if n > 0 else 0
        ev  = sub["ev_if_bet"].mean() * 100 if n > 0 else 0
        tev = sub["ev_if_bet"].sum()  * 100 if n > 0 else 0
        dropped = base_n - n
        lines.append(f"  {name:<50} {n:>5} {wr:>6.1f}% {ev:>+7.2f}¢ {tev:>+9.1f}¢ {dropped:>+7}")

    # What are the negative-edge trades actually worth?
    neg = shadow[shadow["edge"] < 0]
    lines.append(f"\n── Negative-edge trades deep-dive ──")
    lines.append(f"  Count: {len(neg)}  WR: {100*neg['won'].mean():.1f}%  "
                 f"avg EV/trade: {neg['ev_if_bet'].mean()*100:+.2f}¢  "
                 f"total EV: {neg['ev_if_bet'].sum()*100:+.1f}¢")
    lines.append(f"  avg model_p: {neg['model_p'].mean():.3f}  "
                 f"avg market_p_no: {neg['market_p_no'].mean():.3f}  "
                 f"avg NO_cost: {neg['no_cost'].mean()*100:.1f}¢")
    lines.append(f"  Interpretation: market already knows the outcome.  "
                 f"Positive EV but ~{neg['no_cost'].mean()*100:.0f}¢ buy for ~{(1-neg['no_cost'].mean())*100:.0f}¢ max profit.")

    lines.append(f"\n── EV per dollar deployed ──")
    lines.append(f"  (total_EV in cents / total dollars deployed = ROI per dollar)")
    for name, sub in scenarios:
        if len(sub) == 0:
            continue
        deployed = sub["no_cost"].sum()   # total dollars deployed (each trade = no_cost dollars)
        total_ev = sub["ev_if_bet"].sum() * 100 / 100  # in dollars
        roi = total_ev / deployed if deployed > 0 else 0
        lines.append(f"  {name:<50}  ROI={roi:>+.4f}  deployed=${deployed:>7.2f}")

    return lines


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading shadow data from {DB_PATH}...")
    shadow = load_shadow()
    print(f"  {len(shadow)} settled trades, {shadow['won'].sum()} won")

    all_lines = []
    all_lines.append(f"\n{'#'*100}")
    all_lines.append(f"# EDGE-THRESHOLD BACKTEST REPORT")
    all_lines.append(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    all_lines.append(f"# Shadow trades: {len(shadow)} settled  |  WR: {100*shadow['won'].mean():.1f}%")
    all_lines.append(f"# Model_p range: [{shadow['model_p'].min():.3f}, {shadow['model_p'].max():.3f}]")
    all_lines.append(f"# Market_p_no range: [{shadow['market_p_no'].min():.3f}, {shadow['market_p_no'].max():.3f}]")
    all_lines.append(f"# Edge range: [{shadow['edge'].min():.3f}, {shadow['edge'].max():.3f}]")
    all_lines.append(f"{'#'*100}\n")

    all_lines += part_a(shadow)
    all_lines += part_b(shadow)
    all_lines += part_c(shadow)

    all_lines.append("\n" + "=" * 100)
    all_lines.append("END OF REPORT")
    all_lines.append("=" * 100)

    report = "\n".join(all_lines)
    print(report)

    out = "data/backtest/edge_threshold_report.txt"
    with open(out, "w") as f:
        f.write(report)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
