#!/usr/bin/env python3
"""Retro market check: would v2 have closed v1's gap to the market?

Takes the v1 forward-log rows whose contract date matches date_target (the only
v1 rows scored with the right day's distribution, see eval_lookahead_forward.py)
and re-prices the SAME bands with v2's walk-forward out-of-sample distribution
for that city-day (data/lookahead/oof_predictions_high.parquet). All three
forecasters -- v1 as logged live, v2 OOS, the market at capture -- are scored on
identical band-instances against the settled outcome.

CAVEAT: v1's numbers were served live; v2's are reconstructed offline. The v2
features are point-in-time (09:00 cutoff, leak canary ~0) and the walk-forward
only trains on earlier dates, but v2 was designed after looking at this period.
Treat this as supporting evidence; the pre-registered forward test decides.

    venv/bin/python scripts/eval_lookahead_retro_market.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.eval_lookahead_forward import load_rows  # noqa: E402
from scripts.eval_lookahead_vs_market import brier, clustered_bootstrap, logloss  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default="data/db/opportunity_log.db")
    p.add_argument("--oof", default="data/lookahead/oof_predictions_high.parquet")
    p.add_argument("--near", type=float, nargs=2, default=(0.1, 0.9))
    p.add_argument("--boot", type=int, default=2000)
    return p.parse_args()


def compare(d: pd.DataFrame, label: str, n_boot: int) -> None:
    d = d.reset_index(drop=True)
    print(f"\n{label}: {len(d):,} band-instances, "
          f"{d.groupby(['city', 'date_target']).ngroups} city-days")
    for f in ("v1", "v2", "mkt"):
        d[f"br_{f}"] = brier(d[f"p_{f}"], d["y"])
        d[f"ll_{f}"] = logloss(d[f"p_{f}"], d["y"])
        print(f"  {f:<4} Brier {d[f'br_{f}'].mean():.4f}  log-loss {d[f'll_{f}'].mean():.4f}")
    print("  paired diff, city-day-clustered 95% CI (<0 => first is better):")
    for a, b in (("v2", "v1"), ("v1", "mkt"), ("v2", "mkt")):
        for m in ("br", "ll"):
            pt, lo, hi = clustered_bootstrap(d, f"{m}_{a}", f"{m}_{b}", n_boot)
            star = "*" if hi < 0 or lo > 0 else " "
            print(f"    {a}-{b:<4} {m}: {pt:+.4f} [{lo:+.4f}, {hi:+.4f}]{star}")


def main() -> None:
    a = parse_args()
    fwd = load_rows(a.db, "v1").dropna(subset=["market_p", "band_lo", "band_hi"])
    oof = pd.read_parquet(a.oof)[["city", "date_target", "mu", "sigma_recal"]]
    oof["date_target"] = pd.to_datetime(oof["date_target"]).dt.date
    d = fwd.merge(oof.rename(columns={"mu": "mu_v2", "sigma_recal": "sig_v2"}),
                  on=["city", "date_target"], how="inner")
    print(f"v1 date-matched forward rows with a v2 OOS distribution: {len(d):,} of {len(fwd):,}"
          f"  ({d['date_target'].min()} -> {d['date_target'].max()})")
    if d.empty:
        return
    clip = lambda p: np.clip(p, 1e-6, 1 - 1e-6)  # noqa: E731
    d["p_v1"] = clip(d["model_p"])
    d["p_v2"] = clip(norm.cdf(d["band_hi"] + 0.5, d["mu_v2"], d["sig_v2"])
                     - norm.cdf(d["band_lo"] - 0.5, d["mu_v2"], d["sig_v2"]))
    d["p_mkt"] = clip(d["market_p"])

    compare(d, "ALL bands", a.boot)
    lo, hi = a.near
    compare(d[d["p_mkt"].between(lo, hi)], f"NEAR-MONEY (market_p in [{lo}, {hi}])", a.boot)


if __name__ == "__main__":
    main()
