"""Backtest BSM fair-value pricing for long-dated S&P equity Kalshi markets.

No historical Kalshi price data is available from the API, so this backtest
uses a "shadow" approach:

  1. Pull daily S&P 500 + VIX data for 2026 (via yfinance).
  2. Compute BSM fair value for each KXINXY band and KXINXMAXY/KXINXMINY touch
     barrier on every trading day.
  3. Simulate a simple mean-reversion strategy: on a large S&P move day, assume
     Kalshi lags BSM by LAG_DAYS days, then catches up.  Entry = close of move
     day at BSM-lagged price; exit = BSM price LAG_DAYS later.
  4. Report: signal magnitude (how many cents do bands move on big SPX days),
     simulated P&L under different lag / entry-threshold assumptions.

Usage:
  venv/bin/python scripts/backtest_equity_arb.py
  venv/bin/python scripts/backtest_equity_arb.py --lag 2 --threshold 5

Key assumptions:
  - Kalshi price = BSM fair value most of the time (reasonable for liquid bands).
  - On a large S&P move, Kalshi lags by LAG_DAYS before repricing.
  - Entry cost = avg spread observed today (1-3¢ for KXINXY, 3-7¢ for barriers).
  - Risk-free rate = 4.3% (current 1-year Treasury yield).
"""

import argparse
import math
import sys
from datetime import datetime, timezone, date
from typing import NamedTuple

import yfinance as yf
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Market definitions (from live API on 2026-05-16)
# ---------------------------------------------------------------------------

# KXINXY: "between" bands closing Dec 31, 2026 4pm EST
# Band = [lo, hi) where hi is exclusive. Above 9000 and below 4000 also exist.
KXINXY_BANDS = [
    (4000, 4200), (4200, 4400), (4400, 4600), (4600, 4800),
    (4800, 5000), (5000, 5200), (5200, 5400), (5400, 5600),
    (5600, 5800), (5800, 6000), (6000, 6200), (6200, 6400),
    (6400, 6600), (6600, 6800), (6800, 7000), (7000, 7200),
    (7200, 7400), (7400, 7600), (7600, 7800), (7800, 8000),
    (8000, 8200), (8200, 8400), (8400, 8600), (8600, 8800),
    (8800, 9000),
]
KXINXY_ABOVE = 9000  # resolves YES if > 9000
KXINXY_BELOW = 4000  # resolves YES if < 4000
KXINXY_CLOSE = date(2026, 12, 31)
KXINXY_SPREAD = 1.5  # average observed spread (¢)

# KXINXMAXY: one-touch calls — resolves YES if S&P max ever reaches strike before Jan 1
KXINXMAXY_STRIKES = [7600, 7800, 8000, 8200, 8400, 8600, 8800, 9000]
# Note: 7200 and 7400 already settled YES (touched). Only model remaining open ones.
KXINXMAXY_BIDS_TODAY = {
    7600: 75, 7800: 53, 8000: 46, 8200: 38, 8400: 24, 8600: 18, 8800: 13, 9000: 5
}
KXINXMAXY_CLOSE = date(2027, 1, 1)
KXINXMAXY_SPREAD = 3.0

# KXINXMINY: one-touch puts — resolves YES if S&P min ever drops below strike before Jan 1
KXINXMINY_STRIKES = [6300, 6200, 6100, 6000, 5900]
# Note: 6400, 6500, 6600 already settled YES (touched in April crash).
KXINXMINY_BIDS_TODAY = {
    6300: 38, 6200: 30, 6100: 25, 6000: 21, 5900: 19,
}
KXINXMINY_CLOSE = date(2027, 1, 1)
KXINXMINY_SPREAD = 4.0

RISK_FREE = 0.043  # 4.3% — 1-year Treasury yield


# ---------------------------------------------------------------------------
# Pricing formulas
# ---------------------------------------------------------------------------

def days_to_close(from_date: date, close: date) -> float:
    return max((close - from_date).days, 1)


def bsm_band(S: float, lo: float, hi: float, T: float, r: float, sigma: float) -> float:
    """Probability S ends strictly in [lo, hi) under lognormal — European digital."""
    def d2(K: float) -> float:
        return (math.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return norm.cdf(d2(lo)) - norm.cdf(d2(hi))


def bsm_above(S: float, K: float, T: float, r: float, sigma: float) -> float:
    d2 = (math.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return norm.cdf(d2)


def bsm_below(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return 1.0 - bsm_above(S, K, T, r, sigma)


def touch_call(S: float, H: float, T: float, r: float, sigma: float) -> float:
    """Probability S_t ever reaches H > S_0 in [0, T] under risk-neutral GBM.

    Uses the reflection principle under GBM (standard barrier option result):
      P = N(d1) + (S/H)^(2r/σ²-1) ... simplified to:
      P = N((-h + μT)/(σ√T)) + exp(2μh/σ²) * N((-h - μT)/(σ√T))
    where h = log(H/S), μ = r - σ²/2.
    """
    if H <= S:
        return 1.0
    mu = r - 0.5 * sigma ** 2
    h  = math.log(H / S)
    sq = sigma * math.sqrt(T)
    p  = norm.cdf((-h + mu * T) / sq) + math.exp(2 * mu * h / sigma ** 2) * norm.cdf((-h - mu * T) / sq)
    return min(max(p, 0.0), 1.0)


def touch_put(S: float, L: float, T: float, r: float, sigma: float) -> float:
    """Probability S_t ever reaches L < S_0 in [0, T] under risk-neutral GBM.

    Symmetric mirror of touch_call:
      P = N((l + μT)/(σ√T)) + exp(-2μl/σ²) * N((l - μT)/(σ√T))
    where l = log(L/S) < 0, μ = r - σ²/2.
    """
    if L >= S:
        return 1.0
    mu = r - 0.5 * sigma ** 2
    l  = math.log(L / S)   # l < 0
    sq = sigma * math.sqrt(T)
    p  = norm.cdf((l + mu * T) / sq) + math.exp(-2 * mu * l / sigma ** 2) * norm.cdf((l - mu * T) / sq)
    return min(max(p, 0.0), 1.0)


# ---------------------------------------------------------------------------
# Main backtest
# ---------------------------------------------------------------------------

class DaySnapshot(NamedTuple):
    date: date
    spx: float
    vix: float
    # BSM prices (cents) keyed by band label or strike
    inxy: dict    # band_label → bsm_cents
    inxmaxy: dict # strike → bsm_cents
    inxminy: dict # strike → bsm_cents


def compute_snapshot(d: date, spx: float, vix: float) -> DaySnapshot:
    sigma = vix / 100.0

    # KXINXY
    T_inxy = days_to_close(d, KXINXY_CLOSE) / 365.0
    inxy = {}
    for lo, hi in KXINXY_BANDS:
        label = f"B{lo}"
        inxy[label] = round(bsm_band(spx, lo, hi, T_inxy, RISK_FREE, sigma) * 100, 2)
    inxy["ABOVE9000"] = round(bsm_above(spx, KXINXY_ABOVE, T_inxy, RISK_FREE, sigma) * 100, 2)
    inxy["BELOW4000"] = round(bsm_below(spx, KXINXY_BELOW, T_inxy, RISK_FREE, sigma) * 100, 2)

    # KXINXMAXY — only model strikes not yet settled (>= 7600)
    T_touch = days_to_close(d, KXINXMAXY_CLOSE) / 365.0
    inxmaxy = {}
    for K in KXINXMAXY_STRIKES:
        if spx >= K:
            inxmaxy[K] = 100.0  # already above — certainty
        else:
            inxmaxy[K] = round(touch_call(spx, K, T_touch, RISK_FREE, sigma) * 100, 2)

    # KXINXMINY — only model strikes not yet settled (<= 6300)
    inxminy = {}
    for K in KXINXMINY_STRIKES:
        if spx <= K:
            inxminy[K] = 100.0
        else:
            inxminy[K] = round(touch_put(spx, K, T_touch, RISK_FREE, sigma) * 100, 2)

    return DaySnapshot(d, spx, vix, inxy, inxmaxy, inxminy)


def simulate_lag_strategy(
    snapshots: list[DaySnapshot],
    lag_days: int,
    min_spx_move_pct: float,
    entry_spread: float,
    series: str,  # "inxy", "inxmaxy", "inxminy"
) -> dict:
    """Simulate: on big SPX move day, assume Kalshi lags lag_days before catching up.

    Entry: at BSM fair value on move day (plus half-spread cost).
    Exit:  at BSM fair value lag_days later.
    Returns {total_pnl, n_trades, win_rate, avg_pnl_per_trade}.
    """
    trades = []

    for i in range(len(snapshots) - lag_days):
        cur = snapshots[i]
        fut = snapshots[i + lag_days]

        # Check for a large S&P move
        if i == 0:
            continue
        prev = snapshots[i - 1]
        pct_move = (cur.spx - prev.spx) / prev.spx

        if abs(pct_move) < min_spx_move_pct / 100.0:
            continue

        # For each instrument in the series, compute entry and exit BSM price
        if series == "inxy":
            instruments = cur.inxy
            fut_instruments = fut.inxy
        elif series == "inxmaxy":
            instruments = {str(k): v for k, v in cur.inxmaxy.items()}
            fut_instruments = {str(k): v for k, v in fut.inxmaxy.items()}
        else:
            instruments = {str(k): v for k, v in cur.inxminy.items()}
            fut_instruments = {str(k): v for k, v in fut.inxminy.items()}

        for label, entry_bsm in instruments.items():
            if label not in fut_instruments:
                continue
            exit_bsm = fut_instruments[label]

            # Entry: mid-price ≈ BSM on move day.  We buy/sell at mid ± spread/2.
            # Direction: if S&P dropped, upside bands likely fell → buy YES (expect recovery).
            # If S&P rallied, downside bands likely fell → buy NO on min bands.

            if pct_move < 0:
                # S&P down: upside bands underpriced → buy YES at (entry_bsm + spread/2)
                buy_price = entry_bsm + entry_spread / 2
                sell_price = exit_bsm - entry_spread / 2
            else:
                # S&P up: upside bands overpriced → sell YES = buy NO
                buy_price = 100 - entry_bsm + entry_spread / 2  # NO price
                sell_price = 100 - exit_bsm - entry_spread / 2  # NO exit price

            pnl = sell_price - buy_price
            trades.append({
                "date": cur.date,
                "label": label,
                "spx": cur.spx,
                "pct_move": round(pct_move * 100, 2),
                "entry_bsm": entry_bsm,
                "exit_bsm": exit_bsm,
                "pnl": round(pnl, 2),
            })

    if not trades:
        return {"n_trades": 0, "total_pnl": 0, "win_rate": 0, "avg_pnl": 0}

    wins = sum(1 for t in trades if t["pnl"] > 0)
    total = sum(t["pnl"] for t in trades)
    return {
        "n_trades": len(trades),
        "total_pnl": round(total, 2),
        "win_rate": round(wins / len(trades) * 100, 1),
        "avg_pnl": round(total / len(trades), 2),
        "trades": trades,
    }


def main(lag: int, threshold: float, verbose: bool) -> None:
    print("Fetching 2026 S&P 500 + VIX data...")
    raw = yf.download("^GSPC ^VIX", start="2026-01-01", end="2026-05-17", progress=False)
    spx_series = raw["Close"]["^GSPC"].dropna()
    vix_series = raw["Close"]["^VIX"].dropna()
    common = spx_series.index.intersection(vix_series.index)
    print(f"  {len(common)} trading days loaded (Jan 1 → May 16, 2026)")

    # Build daily snapshots
    snapshots: list[DaySnapshot] = []
    for dt in common:
        d = dt.date() if hasattr(dt, "date") else dt
        snap = compute_snapshot(d, float(spx_series[dt]), float(vix_series[dt]))
        snapshots.append(snap)

    # --- Section 1: Snapshot of key instruments today (most recent day) ---
    today = snapshots[-1]
    print(f"\n{'='*70}")
    print(f"BSM Fair Value Snapshot — {today.date}  S&P={today.spx:.0f}  VIX={today.vix:.1f}%")
    print(f"{'='*70}")
    print(f"\nKXINXY year-end bands (current Kalshi bids in brackets):")
    kalshi_bids_inxy = {
        "B7700": 9, "B7900": 10, "B8100": 8, "B7500": 8, "B7300": 4,
        "B7100": 4, "B6900": 4, "B6700": 3, "B8300": 7, "B8500": 5,
    }
    for lo, hi in sorted(KXINXY_BANDS, key=lambda x: x[0]):
        label = f"B{lo}"
        bsm = today.inxy.get(label, 0)
        kalshi = kalshi_bids_inxy.get(label, "?")
        diff = (bsm - kalshi) if isinstance(kalshi, int) else 0
        diff_str = f"{diff:+.1f}¢" if isinstance(kalshi, int) else ""
        print(f"  {label:8s} [{lo:5d}-{hi:5d}]  BSM={bsm:5.1f}¢  Kalshi≈{kalshi}¢  {diff_str}")

    print(f"\nKXINXMAXY one-touch calls (current Kalshi bids in brackets):")
    for K in KXINXMAXY_STRIKES:
        bsm = today.inxmaxy.get(K, 0)
        kalshi = KXINXMAXY_BIDS_TODAY.get(K, "?")
        diff = (bsm - kalshi) if isinstance(kalshi, int) else 0
        diff_str = f"{diff:+.1f}¢" if isinstance(kalshi, int) else ""
        print(f"  strike={K:6d}  BSM={bsm:5.1f}¢  Kalshi≈{kalshi}¢  {diff_str}")

    print(f"\nKXINXMINY one-touch puts (current Kalshi bids in brackets):")
    for K in KXINXMINY_STRIKES:
        bsm = today.inxminy.get(K, 0)
        kalshi = KXINXMINY_BIDS_TODAY.get(K, "?")
        diff = (bsm - kalshi) if isinstance(kalshi, int) else 0
        diff_str = f"{diff:+.1f}¢" if isinstance(kalshi, int) else ""
        print(f"  strike={K:6d}  BSM={bsm:5.1f}¢  Kalshi≈{kalshi}¢  {diff_str}")

    # --- Section 2: Signal magnitude — how much do BSM prices move on big SPX days? ---
    print(f"\n{'='*70}")
    print(f"Signal Magnitude: BSM Δ on Big S&P Move Days (threshold: ±{threshold}%)")
    print(f"{'='*70}")

    # ATM bands (closest to current S&P)
    atm_band = f"B{int(today.spx // 200) * 200}"  # nearest band below current
    focus_bands = ["B7200", "B7400", "B7600", "B7800", "B8000"]
    focus_max = [7600, 7800, 8000, 8200]
    focus_min = [6300, 6200, 6100, 6000]

    big_moves = []
    for i in range(1, len(snapshots)):
        cur, prev = snapshots[i], snapshots[i - 1]
        pct = (cur.spx - prev.spx) / prev.spx * 100
        if abs(pct) >= threshold:
            big_moves.append((cur, prev, pct))

    print(f"  Found {len(big_moves)} days with |S&P move| >= {threshold}%\n")
    print(f"  {'Date':12s}  {'SPX':7s}  {'Δ%':7s}  VIX  {'INXY B7400':>10}  {'INXY B7600':>10}  {'MAX@8000':>9}  {'MIN@6300':>9}")
    print("  " + "-" * 85)
    for cur, prev, pct in big_moves:
        b7400 = cur.inxy.get("B7400", 0) - prev.inxy.get("B7400", 0)
        b7600 = cur.inxy.get("B7600", 0) - prev.inxy.get("B7600", 0)
        max8k = cur.inxmaxy.get(8000, 0) - prev.inxmaxy.get(8000, 0)
        min6k = cur.inxminy.get(6300, 0) - prev.inxminy.get(6300, 0)
        print(f"  {str(cur.date):12s}  {cur.spx:7.0f}  {pct:+6.1f}%  {cur.vix:4.1f}  "
              f"{b7400:>+9.1f}¢  {b7600:>+9.1f}¢  {max8k:>+8.1f}¢  {min6k:>+8.1f}¢")

    # --- Section 3: Price evolution around the April crash ---
    print(f"\n{'='*70}")
    print(f"BSM Evolution Around Key S&P Inflection Points")
    print(f"{'='*70}")
    # Find the minimum S&P day (April crash) and maximum day
    min_snap = min(snapshots, key=lambda s: s.spx)
    max_snap = max(snapshots, key=lambda s: s.spx)
    print(f"  S&P min: {min_snap.spx:.0f} on {min_snap.date}")
    print(f"  S&P max: {max_snap.spx:.0f} on {max_snap.date}")
    print(f"\n  KXINXY B7400 and KXINXMAXY@8000 over time:")
    print(f"  {'Date':12s}  {'SPX':7s}  {'B7400':>8}  {'B7600':>8}  {'MAX@8000':>10}  {'MIN@6300':>10}")
    print("  " + "-" * 65)
    step = max(1, len(snapshots) // 20)
    shown = set()
    key_dates = {min_snap.date, max_snap.date, snapshots[0].date, snapshots[-1].date}
    for i, s in enumerate(snapshots):
        if s.date in key_dates or i % step == 0:
            if s.date not in shown:
                shown.add(s.date)
                print(f"  {str(s.date):12s}  {s.spx:7.0f}  "
                      f"{s.inxy.get('B7400', 0):>8.1f}¢  "
                      f"{s.inxy.get('B7600', 0):>8.1f}¢  "
                      f"{s.inxmaxy.get(8000, 0):>10.1f}¢  "
                      f"{s.inxminy.get(6300, 0):>10.1f}¢")

    # --- Section 4: Lag strategy simulation ---
    print(f"\n{'='*70}")
    print(f"Lag-Strategy Simulation (lag={lag}d, threshold={threshold}%, spread=1.5¢)")
    print(f"{'='*70}")
    print(f"  Assumption: Kalshi price lags BSM by {lag} trading day(s) after a big S&P move.")
    print(f"  Entry on move day at BSM; exit {lag}d later at BSM.\n")

    for series, spread, label in [
        ("inxy",    KXINXY_SPREAD,    "KXINXY  (year-end bands)"),
        ("inxmaxy", KXINXMAXY_SPREAD, "KXINXMAXY (max-touch calls)"),
        ("inxminy", KXINXMINY_SPREAD, "KXINXMINY (min-touch puts)"),
    ]:
        result = simulate_lag_strategy(snapshots, lag, threshold, spread, series)
        print(f"  {label}")
        print(f"    n={result['n_trades']}  total_pnl={result['total_pnl']:+.1f}¢/contract  "
              f"win_rate={result['win_rate']}%  avg={result['avg_pnl']:+.2f}¢")
        if verbose and result.get("trades"):
            worst = sorted(result["trades"], key=lambda t: t["pnl"])[:5]
            best  = sorted(result["trades"], key=lambda t: -t["pnl"])[:5]
            print(f"    Best trades: " + ", ".join(
                f"{t['date']} {t['label']} {t['pnl']:+.1f}¢" for t in best))
            print(f"    Worst trades: " + ", ".join(
                f"{t['date']} {t['label']} {t['pnl']:+.1f}¢" for t in worst))
        print()

    # --- Section 5: Lag sweep ---
    print(f"{'='*70}")
    print(f"Lag Sweep (threshold={threshold}%)")
    print(f"{'='*70}")
    print(f"  {'lag':>5}  {'INXY n':>7}  {'INXY P&L':>10}  {'MAX n':>7}  {'MAX P&L':>10}  {'MIN n':>7}  {'MIN P&L':>10}")
    print("  " + "-" * 65)
    for lag_d in [1, 2, 3, 5, 7]:
        r_inxy  = simulate_lag_strategy(snapshots, lag_d, threshold, KXINXY_SPREAD, "inxy")
        r_max   = simulate_lag_strategy(snapshots, lag_d, threshold, KXINXMAXY_SPREAD, "inxmaxy")
        r_min   = simulate_lag_strategy(snapshots, lag_d, threshold, KXINXMINY_SPREAD, "inxminy")
        print(f"  {lag_d:>5}  {r_inxy['n_trades']:>7}  {r_inxy['total_pnl']:>+10.1f}¢  "
              f"{r_max['n_trades']:>7}  {r_max['total_pnl']:>+10.1f}¢  "
              f"{r_min['n_trades']:>7}  {r_min['total_pnl']:>+10.1f}¢")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lag",       type=int,   default=2,    help="Assumed Kalshi lag days (default: 2)")
    parser.add_argument("--threshold", type=float, default=1.5,  help="Min S&P %% move to trigger (default: 1.5)")
    parser.add_argument("--verbose",   action="store_true",       help="Show best/worst individual trades")
    args = parser.parse_args()
    main(args.lag, args.threshold, args.verbose)
