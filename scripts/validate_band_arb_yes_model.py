"""
Validate the band_arb YES model against actual live trades.

Pulls every resolved YES band_arb trade from the DB, reconstructs
forecast features from raw_forecasts logged on the same day, runs
the KXHIGH model, and compares model P(YES wins) against:
  - What the market priced it at (limit_price / 100)
  - Whether it actually won or lost

Usage:
  venv/bin/python scripts/validate_band_arb_yes_model.py
"""

import os
import re
import sys
import pickle
import sqlite3
from datetime import datetime, timezone
from statistics import median
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DB_PATH    = "data/db/opportunity_log.db"
MODEL_PATH = "data/models/band_arb_yes_model.pkl"

# Maps Kalshi series prefix → (metric prefix, city suffix map)
# Used to derive metric key from ticker for raw_forecasts lookup.
_SERIES_TO_METRIC: dict[str, str] = {
    # Original KXHIGH series
    "KXHIGHLAX": "temp_high_lax", "KXHIGHDEN": "temp_high_den",
    "KXHIGHCHI": "temp_high_chi", "KXHIGHNY":  "temp_high_ny",
    "KXHIGHMIA": "temp_high_mia", "KXHIGHDAL": "temp_high_dal",
    "KXHIGHBOS": "temp_high_bos", "KXHIGHAUS": "temp_high_aus",
    "KXHIGHOU":  "temp_high_hou",
    # New KXHIGHT series
    "KXHIGHTSFO": "temp_high_sfo", "KXHIGHTSEA": "temp_high_sea",
    "KXHIGHTBOS": "temp_high_bos", "KXHIGHTPHX": "temp_high_phx",
    "KXHIGHPHIL": "temp_high_phl", "KXHIGHTDC":  "temp_high_dca",
    "KXHIGHTLV":  "temp_high_las", "KXHIGHTOKC": "temp_high_okc",
    "KXHIGHTDAL": "temp_high_dfw", "KXHIGHTSATX":"temp_high_sat",
    "KXHIGHTHOU": "temp_high_hou", "KXHIGHTNOLA":"temp_high_msy",
    "KXHIGHTATL": "temp_high_atl", "KXHIGHTMIN": "temp_high_msp",
    "KXHIGHPHIL": "temp_high_phl",
    # KXLOWT series
    "KXLOWTLAX": "temp_low_lax", "KXLOWTDEN": "temp_low_den",
    "KXLOWTCHI": "temp_low_chi", "KXLOWTNYC": "temp_low_ny",
    "KXLOWTMIA": "temp_low_mia", "KXLOWTAUS": "temp_low_aus",
    "KXLOWTBOS": "temp_low_bos", "KXLOWTHOU": "temp_low_hou",
    "KXLOWTDFW": "temp_low_dfw", "KXLOWTSFO": "temp_low_sfo",
    "KXLOWTSEA": "temp_low_sea", "KXLOWTPHX": "temp_low_phx",
    "KXLOWTPHIL":"temp_low_phl", "KXLOWTATL": "temp_low_atl",
    "KXLOWTMIN": "temp_low_msp", "KXLOWTDC":  "temp_low_dca",
    "KXLOWTLV":  "temp_low_las", "KXLOWTOKC": "temp_low_okc",
    "KXLOWTSATX":"temp_low_sat", "KXLOWTNOLA":"temp_low_msy",
}

def parse_band_ticker(ticker: str):
    """Parse a band_arb ticker into (metric, band_lo, band_ceil) or None."""
    # e.g. KXHIGHTDC-26MAY01-B66.5  or  KXLOWTCHI-26MAY30-B64.5
    m = re.match(r"^([A-Z]+)-\d{2}[A-Z]{3}\d{2}-B(\d+\.?\d*)$", ticker)
    if not m:
        return None
    series, midpoint_str = m.group(1), m.group(2)
    midpoint = float(midpoint_str)
    band_lo  = int(midpoint - 0.5)
    band_ceil = band_lo + 1
    metric = _SERIES_TO_METRIC.get(series)
    if not metric:
        return None
    return metric, band_lo, band_ceil

FORECAST_SOURCES = {
    "hrrr":            "hrrr_f",
    "open_meteo_gfs":  "gfs_f",
    "open_meteo_ecmwf":"ecmwf_f",
    "open_meteo_gem":  "gem_f",
    "open_meteo_icon": "icon_f",
}


def load_model():
    payload = pickle.loads(open(MODEL_PATH, "rb").read())
    return payload["lgbm"], payload["isotonic"], payload["features"]


def get_forecasts(conn, metric: str, trade_date: str) -> dict[str, float]:
    """Get model forecast values for a metric on a given date."""
    rows = conn.execute("""
        SELECT source, AVG(data_value) as val
        FROM raw_forecasts
        WHERE metric = ?
          AND date(logged_at) = ?
          AND source IN ('hrrr','open_meteo_gfs','open_meteo_ecmwf',
                         'open_meteo_gem','open_meteo_icon')
          AND data_value IS NOT NULL
        GROUP BY source
    """, (metric, trade_date)).fetchall()
    return {source: val for source, val in rows}


def build_features(pm, forecasts: dict, feature_names: list) -> list | None:
    """Build feature vector matching FEATURES list. Returns None if not enough data."""
    band_ceil = float(pm.strike_hi)
    band_lo   = float(pm.strike_lo)
    is_high   = 1.0 if pm.metric.startswith("temp_high_") else 0.0

    model_vals = [v for v in forecasts.values() if v is not None]
    if len(model_vals) < 2:
        return None

    consensus = median(model_vals)
    spread    = max(model_vals) - min(model_vals)
    n_above   = sum(1 for v in model_vals if v > band_ceil)
    n_below   = sum(1 for v in model_vals if v < band_lo)

    hrrr_f  = forecasts.get("hrrr")
    gfs_f   = forecasts.get("open_meteo_gfs")
    ecmwf_f = forecasts.get("open_meteo_ecmwf")

    vals = {
        "consensus_vs_ceil":  consensus - band_ceil,
        "hrrr_vs_ceil":       (hrrr_f  - band_ceil) if hrrr_f  is not None else (consensus - band_ceil),
        "gfs_vs_ceil":        (gfs_f   - band_ceil) if gfs_f   is not None else (consensus - band_ceil),
        "consensus_vs_floor": consensus - band_lo,
        "hrrr_vs_floor":      (hrrr_f  - band_lo)   if hrrr_f  is not None else (consensus - band_lo),
        "gfs_vs_floor":       (gfs_f   - band_lo)   if gfs_f   is not None else (consensus - band_lo),
        "ecmwf_vs_ceil":      (ecmwf_f - band_ceil) if ecmwf_f is not None else (consensus - band_ceil),
        "model_spread":       spread,
        "n_models_above_ceil":float(n_above),
        "n_models_below_floor":float(n_below),
        "n_models":           float(len(model_vals)),
        "is_high":            is_high,
        "month":              float(pm.metric and 1),  # will be overridden below
    }
    return [vals[f] for f in feature_names]


def main():
    lgbm, iso, features = load_model()
    conn = sqlite3.connect(DB_PATH)

    trades = conn.execute("""
        SELECT ticker, limit_price, logged_at, outcome, settled_pnl_cents, count
        FROM trades
        WHERE opportunity_kind = 'band_arb'
          AND side = 'yes'
          AND outcome IS NOT NULL
        ORDER BY logged_at
    """).fetchall()

    print(f"{'Ticker':<35} {'Mkt':>4} {'ModelP':>7} {'Edge':>6} {'Out':>5} {'PnL':>7}  Forecasts")
    print("-" * 95)

    results = []
    skipped = 0

    for ticker, limit_price, logged_at_str, outcome, pnl_cents, count in trades:
        parsed = parse_band_ticker(ticker)
        if parsed is None:
            skipped += 1
            continue
        metric, band_lo, band_ceil = parsed

        dt = datetime.fromisoformat(logged_at_str)
        trade_date = dt.date().isoformat()
        month = dt.month

        forecasts = get_forecasts(conn, metric, trade_date)
        if not forecasts:
            skipped += 1
            continue

        # Build a simple namespace to reuse build_features
        class PM:
            pass
        pm = PM()
        pm.metric    = metric
        pm.strike_lo = band_lo
        pm.strike_hi = band_ceil
        pm.direction = "between"

        feat = build_features(pm, forecasts, features)
        if feat is None:
            skipped += 1
            continue

        feat[features.index("month")] = float(month)

        import numpy as np
        X = np.array([feat])
        raw_p  = lgbm.predict_proba(X)[0][1]
        model_p = float(iso.predict([raw_p])[0])

        mkt_p  = limit_price / 100.0
        edge   = model_p - mkt_p
        won    = outcome == "won"

        fc_summary = " ".join(f"{s.replace('open_meteo_','')[:4]}={v:.1f}"
                              for s, v in sorted(forecasts.items()))

        print(f"{ticker:<35} {limit_price:>3}¢ {model_p:>6.1%} {edge:>+5.1%} "
              f"{'WIN' if won else 'LOSS':>5} {(pnl_cents or 0)/100:>+6.2f}$  {fc_summary}")

        results.append({
            "model_p": model_p, "mkt_p": mkt_p, "edge": edge,
            "won": won, "pnl": (pnl_cents or 0) / 100,
        })

    conn.close()

    if not results:
        print("No results.")
        return

    print(f"\n{'='*60}")
    n = len(results)
    wins = sum(1 for r in results if r["won"])
    print(f"Trades evaluated: {n}  (skipped {skipped})")
    print(f"Actual WR:        {100*wins/n:.1f}%")
    print(f"Avg model P:      {100*sum(r['model_p'] for r in results)/n:.1f}%")
    print(f"Avg market P:     {100*sum(r['mkt_p']   for r in results)/n:.1f}%")
    print(f"Avg edge:         {100*sum(r['edge']     for r in results)/n:+.1f}%")
    print(f"Total PnL:        ${sum(r['pnl'] for r in results):+.2f}")

    # Model P buckets vs actual WR
    print(f"\nModel P bucket  →  actual WR:")
    buckets = defaultdict(list)
    for r in results:
        b = int(r["model_p"] * 10) / 10  # floor to 0.1
        buckets[b].append(r["won"])
    for b in sorted(buckets):
        grp = buckets[b]
        wr = sum(grp) / len(grp)
        bar = "█" * sum(grp) + "░" * (len(grp) - sum(grp))
        print(f"  {b:.0%}–{b+.1:.0%}  n={len(grp):>2}  WR={100*wr:>5.1f}%  {bar}")

    # Agreement: model high P and bot won?
    agree = sum(1 for r in results if (r["model_p"] > 0.6) == r["won"])
    print(f"\nModel-bot agreement (model>60% ↔ won): {agree}/{n} = {100*agree/n:.1f}%")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
