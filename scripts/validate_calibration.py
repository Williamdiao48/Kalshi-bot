"""
Validate whether HRRR/GFS RMSE calibration predicts actual trade outcomes.

For each settled forecast_no trade (between-band type only):
  1. Extract city, signal edge, model forecast from note JSON
  2. Compute implied P(win) using city RMSE/bias from historical calibration
  3. Compare implied probabilities to actual win/loss outcomes

A well-calibrated model should show:
  trades with implied P(win)=80% should actually win ~80% of the time, etc.

Run:
  venv/bin/python scripts/validate_calibration.py
"""

import json
import math
import sqlite3
import re
from collections import defaultdict

TRADES_DB = "data/db/opportunity_log.db"

# GFS/HRRR calibration from build_forecast_calibration.py (2022–2026, IEM ASOS)
# Format: city_code → (rmse, bias, stdev)
# stdev ≈ sqrt(rmse² - bias²)
CALIBRATION = {
    "lax": (1.50, -0.45),
    "mia": (2.06, -1.27),
    "hou": (2.25, -0.19),
    "atl": (2.34, -1.12),
    "msy": (2.45, -0.71),
    "ny":  (2.47, +0.35),
    "nyc": (2.47, +0.35),   # alias
    "phl": (2.51, -0.28),
    "bos": (2.56, +0.26),
    "phx": (2.83, -0.59),
    "las": (2.99, +0.77),
    "lv":  (2.99, +0.77),   # alias
    "dfw": (3.01, +0.22),
    "dal": (3.07, +1.18),
    "chi": (3.18, -0.56),
    "sat": (3.04, +0.27),
    "sfo": (3.41, +0.67),
    "aus": (3.29, +0.02),
    "msp": (3.36, +0.30),
    "min": (3.36, +0.30),   # alias
    "sea": (3.69, -0.11),
    "okc": (3.75, +0.84),
    "den": (4.74, +0.02),
}


def norm_cdf(x: float) -> float:
    """Standard normal CDF using math.erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def implied_win_prob(edge: float, direction: str, rmse: float, bias: float) -> float:
    """
    Compute P(NO wins) given:
      edge      = distance from model forecast to nearest band edge (°F)
      direction = "NO_HIGH" (model above band) or "NO_LOW" (model below band)
      rmse, bias = city-level GFS error statistics (model - observed)

    Model error distribution: Normal(bias, sigma) where sigma = sqrt(rmse² - bias²)
    Band width = 1°F

    NO_HIGH: model = band_top + edge
      Win if actual > band_top (error < edge) OR actual < band_bottom (error > edge+1)
      P(win) = Φ((edge - bias)/σ) + [1 - Φ((edge+1 - bias)/σ)]

    NO_LOW: model = band_bottom - edge
      Win if actual < band_bottom (error > -edge) OR actual > band_top (error < -(edge+1))
      P(win) = [1 - Φ((-edge - bias)/σ)] + Φ((-(edge+1) - bias)/σ)
    """
    sigma = math.sqrt(max(rmse**2 - bias**2, 0.01))

    if direction == "NO_HIGH":
        p_above = norm_cdf((edge - bias) / sigma)
        p_below = 1.0 - norm_cdf((edge + 1 - bias) / sigma)
    else:  # NO_LOW
        p_above = norm_cdf((-(edge + 1) - bias) / sigma)
        p_below = 1.0 - norm_cdf((-edge - bias) / sigma)

    return p_above + p_below


def parse_ticker(ticker: str) -> tuple[str | None, str | None, float | None]:
    """
    Returns (city_code, market_type, band_center_or_strike).
    market_type: "between" (B) or "over"/"under" (T)
    Examples:
      KXHIGHMIA-26MAY04-B85.5  → ("mia", "between", 85.5)
      KXHIGHTSEA-26MAY04-B78.5 → ("sea", "between", 78.5)
      KXHIGHTPHX-26MAY04-T81   → ("phx", "touch", 81.0)
    """
    # Strip the series prefix to get city code
    series_map = {
        "KXHIGHLAX": "lax", "KXHIGHDEN": "den", "KXHIGHCHI": "chi",
        "KXHIGHNY":  "ny",  "KXHIGHMIA": "mia", "KXHIGHAUS": "aus",
        "KXHIGHDAL": "dal", "KXHIGHBOS": "bos", "KXHIGHOU":  "hou",
        "KXHIGHTDAL": "dfw", "KXHIGHTSFO": "sfo", "KXHIGHTSEA": "sea",
        "KXHIGHTBOS": "bos", "KXHIGHTPHX": "phx", "KXHIGHTATL": "atl",
        "KXHIGHTMIN": "msp", "KXHIGHTDC":  "dca", "KXHIGHTLV":  "las",
        "KXHIGHTOKC": "okc", "KXHIGHTDAL": "dfw", "KXHIGHTSATX": "sat",
        "KXHIGHTHOU": "hou", "KXHIGHTNOLA": "msy", "KXHIGHPHIL": "phl",
        "KXHIGHTPHIL": "phl",
        "KXLOWTLAX": "lax", "KXLOWTDEN": "den", "KXLOWTCHI": "chi",
        "KXLOWTNYC": "ny",  "KXLOWTMIA": "mia", "KXLOWTAUS": "aus",
        "KXLOWTBOS": "bos", "KXLOWTHOU": "hou", "KXLOWTDFW": "dfw",
        "KXLOWTSFO": "sfo", "KXLOWTSEA": "sea", "KXLOWTPHX": "phx",
        "KXLOWTPHIL": "phl", "KXLOWTATL": "atl", "KXLOWTMIN": "msp",
        "KXLOWTDC":  "dca", "KXLOWTLV":  "las", "KXLOWTOKC": "okc",
        "KXLOWTSATX": "sat", "KXLOWTNOLA": "msy",
    }

    series = ticker.split("-")[0]
    city = series_map.get(series)
    if city is None:
        return None, None, None

    # Extract market suffix (last component)
    suffix = ticker.split("-")[-1]
    if suffix.startswith("B"):
        try:
            return city, "between", float(suffix[1:])
        except ValueError:
            return city, "between", None
    elif suffix.startswith("T"):
        try:
            return city, "touch", float(suffix[1:])
        except ValueError:
            return city, "touch", None
    return city, None, None


def infer_direction(model_forecast: float, band_center: float,
                    market_type: str, strike: float | None) -> str:
    """Infer NO_HIGH vs NO_LOW from model forecast vs band position."""
    if market_type == "touch":
        # Touch market: T81 = "will high reach 81°F?" YES wins if actual ≥ 81
        # We buy NO when model predicts ABOVE strike (model says yes, market disagrees?)
        # Actually: if model > strike, YES likely → we buy NO only if model < strike
        # The edge in sources_detail is model - strike for NO_LOW or strike - model for NO_HIGH
        # From observation: edge always positive, so direction must be inferred from context
        # Conservative: if model > strike, NO_HIGH (model above threshold, betting actual < threshold)
        # This is a special case — skip touch markets for clean validation
        return "SKIP"
    # Between market
    if model_forecast > band_center:
        return "NO_HIGH"
    return "NO_LOW"


def load_trades() -> list[dict]:
    con = sqlite3.connect(TRADES_DB)
    con.row_factory = sqlite3.Row
    rows = con.execute("""
        SELECT id, ticker, limit_price, count, exit_pnl_cents, exit_reason,
               logged_at, note
        FROM trades
        WHERE source = 'forecast_no' AND exit_reason IS NOT NULL
        ORDER BY logged_at
    """).fetchall()
    con.close()

    trades = []
    for r in rows:
        try:
            note = json.loads(r["note"] or "{}")
        except Exception:
            note = {}
        city, mtype, band_val = parse_ticker(r["ticker"])
        if city is None or band_val is None:
            continue

        # Get HRRR forecast from sources_detail (prefer hrrr, fallback to any)
        sources_detail = note.get("sources_detail", [])
        hrrr_forecast = None
        any_forecast  = None
        for src, forecast_val, edge_val in sources_detail:
            if any_forecast is None:
                any_forecast = (forecast_val, edge_val)
            if "hrrr" in src.lower():
                hrrr_forecast = (forecast_val, edge_val)
                break
        best = hrrr_forecast or any_forecast
        if best is None:
            continue
        model_val, edge_val = best

        direction = infer_direction(model_val, band_val, mtype, band_val)

        trades.append({
            "id":        r["id"],
            "ticker":    r["ticker"],
            "city":      city,
            "mtype":     mtype,
            "band_val":  band_val,
            "model_val": model_val,
            "edge":      note.get("min_edge_f", edge_val),
            "direction": direction,
            "win":       r["exit_pnl_cents"] > 0,
            "pnl":       r["exit_pnl_cents"],
            "yes_bid":   r["limit_price"],
            "note":      note,
        })
    return trades


def main():
    trades = load_trades()
    print(f"Loaded {len(trades)} forecast_no trades with parseable tickers")

    # Split: between vs touch
    between = [t for t in trades if t["mtype"] == "between" and t["direction"] != "SKIP"]
    touch   = [t for t in trades if t["mtype"] == "touch"]
    print(f"  Between (B) markets: {len(between)}")
    print(f"  Touch (T) markets:   {len(touch)}")

    # ── Calibration validation for between markets ──
    print(f"\n{'='*70}")
    print("Calibration validation — Between (B) markets only")
    print(f"{'='*70}")

    # Assign implied win probability to each trade
    calibrated = []
    no_calib   = []
    for t in between:
        calib = CALIBRATION.get(t["city"])
        if calib is None:
            no_calib.append(t["city"])
            continue
        rmse, bias = calib
        p_win = implied_win_prob(t["edge"], t["direction"], rmse, bias)
        calibrated.append({**t, "p_win": p_win})

    if no_calib:
        print(f"\n  No calibration for: {sorted(set(no_calib))}")
    print(f"\n  Trades with calibration: {len(calibrated)}")

    # ── Per-city: implied vs actual ──
    print(f"\n--- Per-city: implied P(win) vs actual win rate ---")
    print(f"  {'City':>6}  {'n':>4}  {'Avg P(win)':>10}  {'Actual WR':>10}  {'Delta':>7}  {'Calib?':>7}")
    print("  " + "-"*55)

    city_data = defaultdict(list)
    for t in calibrated:
        city_data[t["city"]].append(t)

    for city in sorted(city_data):
        ts = city_data[city]
        avg_p  = sum(t["p_win"] for t in ts) / len(ts)
        actual = sum(1 for t in ts if t["win"]) / len(ts)
        delta  = actual - avg_p
        flag   = "OK" if abs(delta) < 0.15 else "POOR"
        print(f"  {city:>6}  {len(ts):>4}  {avg_p:>9.1%}  {actual:>9.1%}  {delta:>+6.1%}  {flag:>7}")

    # ── Probability bucket calibration curve ──
    print(f"\n--- Calibration curve: implied P(win) bucket vs actual win rate ---")
    print(f"  {'P(win) bucket':>14}  {'n':>4}  {'actual WR':>10}  {'assessment':>12}")
    print("  " + "-"*48)

    buckets = defaultdict(list)
    for t in calibrated:
        p = t["p_win"]
        bucket = f"{int(p*10)*10}–{int(p*10)*10+9}%"
        buckets[bucket].append(t)

    for bucket in sorted(buckets):
        ts = buckets[bucket]
        actual = sum(1 for t in ts if t["win"]) / len(ts)
        lo = int(bucket.split("–")[0])
        expected = (lo + 5) / 100
        ok = "OK" if abs(actual - expected) < 0.15 else "OFF"
        print(f"  {bucket:>14}  {len(ts):>4}  {actual:>9.1%}  {ok}")

    # ── Edge vs actual win rate ──
    print(f"\n--- Signal edge vs actual win rate (all between trades) ---")
    print(f"  {'Edge bin':>10}  {'n':>4}  {'actual WR':>10}  {'avg P(win)':>10}  {'delta':>7}")
    print("  " + "-"*52)

    edge_buckets = defaultdict(list)
    for t in calibrated:
        e = t["edge"]
        lo = int(e)
        bucket = f"{lo}–{lo+1}°F"
        edge_buckets[bucket].append(t)

    for bucket in sorted(edge_buckets):
        ts = edge_buckets[bucket]
        actual = sum(1 for t in ts if t["win"]) / len(ts)
        avg_p  = sum(t["p_win"] for t in ts) / len(ts)
        delta  = actual - avg_p
        print(f"  {bucket:>10}  {len(ts):>4}  {actual:>9.1%}  {avg_p:>9.1%}  {delta:>+6.1%}")

    # ── City RMSE vs per-city win rate ──
    print(f"\n--- City model accuracy vs observed win rate ---")
    print(f"  {'City':>6}  {'RMSE':>6}  {'n':>4}  {'actual WR':>10}  {'break-even':>11}  {'edge?':>6}")
    print("  " + "-"*55)

    for city in sorted(city_data, key=lambda c: CALIBRATION.get(c, (99, 0))[0]):
        ts = city_data[city]
        rmse, bias = CALIBRATION.get(city, (0, 0))
        actual = sum(1 for t in ts if t["win"]) / len(ts)
        avg_yb = sum(t["yes_bid"] for t in ts) / len(ts)
        be     = (100 - avg_yb) / 100
        flag   = "YES" if actual >= be else "NO"
        print(f"  {city:>6}  {rmse:>5.2f}°  {len(ts):>4}  {actual:>9.1%}  {be:>10.1%}  {flag:>6}")

    # ── Overall summary ──
    print(f"\n--- Overall ---")
    overall_actual  = sum(1 for t in calibrated if t["win"]) / len(calibrated)
    overall_implied = sum(t["p_win"] for t in calibrated) / len(calibrated)
    print(f"  All between trades: n={len(calibrated)}")
    print(f"  Avg implied P(win): {overall_implied:.1%}")
    print(f"  Actual win rate:    {overall_actual:.1%}")
    print(f"  Gap:                {overall_actual - overall_implied:+.1%}")
    print()
    print(f"  Interpretation: if gap ≈ 0, calibration predicts outcomes well.")
    print(f"  Large negative gap = market is pricing correctly, we have no alpha.")
    print(f"  Large positive gap = we're underestimating our edge.")


if __name__ == "__main__":
    main()
