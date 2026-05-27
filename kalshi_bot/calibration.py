"""Logistic regression calibration model for forecast_no trades.

Loads data/models/forecast_no_cal.pkl (trained by scripts/train_forecast_no_model.py)
and exposes win_prob / edge functions used by strike_arb.py.

The model takes continuous per-model edge values (°F vs band ceiling) rather than
binary qualifying flags.  edge_hrrr=NaN is treated as 0.0 (neutral/no signal),
matching the training-time fill used for historical rows.

If the model file is absent the functions return None, and strike_arb.py
falls back to the existing hardcoded p_estimate formula — so the bot
continues to work without a trained model.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

_MODEL_PATH = Path(__file__).parent.parent / "data" / "models" / "forecast_no_cal.pkl"

_bundle: dict | None = None
_load_attempted: bool = False


def _load() -> dict | None:
    global _bundle, _load_attempted
    if _load_attempted:
        return _bundle
    _load_attempted = True
    if not _MODEL_PATH.exists():
        logging.debug("Calibration: model not found at %s — using fallback formula", _MODEL_PATH)
        return None
    try:
        import joblib
        _bundle = joblib.load(_MODEL_PATH)
        features = _bundle.get("features", [])
        logging.info("Calibration: loaded forecast_no model (%d features) from %s", len(features), _MODEL_PATH)
        return _bundle
    except Exception as exc:
        logging.warning("Calibration: failed to load model — %s", exc)
        return None


def forecast_no_win_prob(
    edge_ecmwf: float,
    edge_icon: float,
    edge_gem: float,
    edge_hrrr: float,
    model_spread: float,
    n_supporting: int,
    is_no_high: bool,
    is_high_market: bool,
    month_sin: float,
    month_cos: float,
) -> float | None:
    """Return calibrated P(NO wins) from continuous per-model edge values.

    All edge_* args are °F relative to the band ceiling (positive = supports NO).
    edge_hrrr=NaN is safe — treated as 0.0 (neutral) matching training convention.
    Returns None if model not loaded (bot falls back to hardcoded p_estimate).
    """
    bundle = _bundle if _load_attempted else _load()
    if bundle is None:
        return None
    try:
        import numpy as np
        model    = bundle["model"]
        features = bundle["features"]
        def _safe(v: float) -> float:
            return 0.0 if (v is None or math.isnan(v)) else v

        row = {
            "edge_ecmwf":    _safe(edge_ecmwf),
            "edge_icon":     _safe(edge_icon),
            "edge_gem":      _safe(edge_gem),
            "edge_hrrr":     _safe(edge_hrrr),
            "model_spread":  model_spread,
            "n_supporting":  n_supporting,
            "is_no_high":    int(is_no_high),
            "is_high_market": int(is_high_market),
            "month_sin":     month_sin,
            "month_cos":     month_cos,
        }
        X = np.array([[row[f] for f in features]])
        return float(model.predict_proba(X)[0, 1])
    except Exception as exc:
        logging.warning("Calibration: predict error — %s", exc)
        return None


def forecast_no_edge(
    edge_ecmwf: float,
    edge_icon: float,
    edge_gem: float,
    edge_hrrr: float,
    model_spread: float,
    n_supporting: int,
    is_no_high: bool,
    is_high_market: bool,
    month_sin: float,
    month_cos: float,
    *,
    no_ask: int,
) -> float | None:
    """Return model P(NO wins) − market P(NO wins).

    Positive = model believes we have edge over the market price.
    no_ask is keyword-only and used only in the edge formula, not passed to the model.
    Returns None if the calibration model is not loaded.
    """
    p = forecast_no_win_prob(
        edge_ecmwf, edge_icon, edge_gem, edge_hrrr,
        model_spread, n_supporting, is_no_high, is_high_market,
        month_sin, month_cos,
    )
    return None if p is None else p - (no_ask / 100.0)
