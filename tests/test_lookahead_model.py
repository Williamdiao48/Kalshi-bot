"""Tests for the v2 lookahead HIGH model's label, target, and serving math.

Covers the three fixes that made v2 (see scripts/train_lookahead_high_model.py):
  - CLI labels: "M"/"T"/None parse to missing; only finals (<= max_date) kept.
  - Residual target: predict_dist adds consensus back; trailing skill only sees
    labels LABEL_LAG_DAYS old (what the bot has at the 09:00 cutoff).
  - Sigma: an L1 fit on |residual| estimates the MEDIAN, so the Gaussian scale
    constant must be 1/0.6745, not sqrt(pi/2).
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

import scripts.build_lookahead_features as FB
import scripts.shadow_lookahead as LA
import scripts.train_lookahead_high_model as TR
from kalshi_bot.shadow_forecast import parse_cli_results


# --- CLI label parsing ------------------------------------------------------

def test_parse_cli_handles_missing_and_cutoff():
    payload = {"results": [
        {"valid": "2026-11-01", "high": 61, "low": 48, "high_normal": 58, "low_normal": 44},
        {"valid": "2026-11-02", "high": "M", "low": 50, "high_normal": 58, "low_normal": 44},
        {"valid": "2026-11-03", "high": "M", "low": "M"},          # nothing usable
        {"valid": "2026-11-04", "high": 70, "low": 55},            # past max_date
    ]}
    rows = parse_cli_results("Seattle", "KSEA", payload, "2026-11-03", "now")
    assert [r[1] for r in rows] == ["2026-11-01", "2026-11-02"]
    assert rows[0][3:7] == (61.0, 48.0, 58.0, 44.0)
    assert rows[1][3] is None and rows[1][4] == 50.0


# --- predict_dist: residual target + sigma scale ---------------------------

class _Const:
    def __init__(self, v):
        self.v = v

    def predict(self, X):
        return np.full(len(X), self.v)


def _bundle(mu_corr, med_abs_res, recal=1.0):
    return dict(mu_model=_Const(mu_corr), sigma_model=_Const(med_abs_res),
                features=["x"], sigma_floor=0.75, recal=recal,
                sigma_scale=TR.MAD_TO_SIGMA, target="resid_consensus")


def test_predict_dist_adds_consensus_back():
    row = pd.DataFrame({"x": [0.0], "consensus_median": [48.0]})
    mu, sig = LA.predict_dist(_bundle(mu_corr=1.5, med_abs_res=1.0), row)
    assert mu == pytest.approx(49.5)        # far below any summer label: no floor
    assert sig == pytest.approx(1.4826, abs=1e-3)


def test_predict_dist_applies_floor_then_recal():
    row = pd.DataFrame({"x": [0.0], "consensus_median": [70.0]})
    _, sig = LA.predict_dist(_bundle(0.0, med_abs_res=0.1, recal=1.2), row)
    assert sig == pytest.approx(0.75 * 1.2)


def test_predict_dist_rejects_missing_consensus():
    row = pd.DataFrame({"x": [0.0], "consensus_median": [np.nan]})
    with pytest.raises(ValueError):
        LA.predict_dist(_bundle(0.0, 1.0), row)


def test_mad_to_sigma_matches_median_abs_normal():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 3.0, 400_000)
    assert np.median(np.abs(z)) * TR.MAD_TO_SIGMA == pytest.approx(3.0, rel=0.01)


def test_band_prob_integrates_rounding_edges():
    # band {86, 87} settles on raw highs in [85.5, 87.5)
    p = LA.band_prob(86.5, 1.0, 86, 87)
    assert p == pytest.approx(norm.cdf(1.0) - norm.cdf(-1.0))


# --- trailing skill only sees labels available at the cutoff ---------------

def test_trailing_skill_lags_labels():
    d0 = date(2026, 10, 1)
    n = 10
    df = pd.DataFrame({
        "city": ["Seattle"] * n,
        "date_target": [d0 + timedelta(days=i) for i in range(n)],
        "consensus_median": [60.0] * n,
        # error vs consensus is i, so a leak of day t's label shows up as t
        "actual_f": [60.0 - i for i in range(n)],
    })
    out = FB.add_trailing_skill(df, window=14).set_index("date_target")
    lag = FB.LABEL_LAG_DAYS
    # first lag+2 days have < 3 available labels -> NaN
    assert out["bias14_consensus"].iloc[: lag + 2].isna().all()
    # day t sees errors 0..t-lag only: mean = (t-lag)/2
    t = 8
    assert out["bias14_consensus"].iloc[t] == pytest.approx((t - lag) / 2)


def test_feature_columns_exclude_absolute_levels():
    cols = ["city", "date_target", "actual_f", "normal_f", "resid", "f_noaa",
            "consensus_median", "month", "doy", "morning_obs_max",
            "d_noaa", "mae14_noaa", "bias14_consensus", "consensus_anom", "city_code"]
    feats = FB.feature_columns(pd.DataFrame(columns=cols))
    assert set(feats) == {"d_noaa", "mae14_noaa", "bias14_consensus",
                          "consensus_anom", "city_code"}
