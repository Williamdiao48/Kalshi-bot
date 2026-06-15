"""Backtest BTC/ETH/SOL momentum signal for KXBTC15M / KXETH15M / KXSOL15M.

KXBTC15M asks: "Will BTC price be up in the next 15 minutes?"
  - Bid/ask spread: ~1¢ (e.g. bid=28, ask=29)
  - Market resets every 15 minutes

Strategy: observe prior N-minute price momentum; buy YES when bullish above
threshold, buy NO when bearish below threshold, hold flat otherwise.

Key question: does prior momentum predict the next 15-minute direction enough
to overcome the ~1¢ spread (requires >50.5% win rate on selected trades)?

Data:
  - 15m OHLCV: last 60 days from Yahoo Finance (~5,700 candles per coin)
  - 1h OHLCV: last 2 years from Yahoo Finance (~17,000 candles) for longer history

Sections:
  A. Base rates and data summary
  B. Win rate by momentum bucket (does stronger momentum → higher win rate?)
  C. Threshold sweep (which min-momentum threshold maximises edge × coverage?)
  D. Lookback sweep (15m / 30m / 1h / 2h / 4h — which is most predictive?)
  E. Time-of-day analysis (crypto momentum patterns vary by hour)
  F. Volatility-adjusted momentum (momentum / recent vol)
  G. Multi-coin comparison (BTC vs ETH vs SOL)
  H. Extended 1h backtest (2-year horizon, direction = up in next hour)
  I. P&L simulation (combining best signals)

Usage:
  venv/bin/python scripts/backtest_btc_momentum.py
  venv/bin/python scripts/backtest_btc_momentum.py --coins BTC ETH
  venv/bin/python scripts/backtest_btc_momentum.py --no-extended
"""

import argparse
import math
import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

SPREAD_CENTS   = 1.0   # observed bid/ask spread on KXBTC15M
ENTRY_COST     = SPREAD_CENTS / 2  # half-spread per side
BREAK_EVEN_WR  = 50 + ENTRY_COST   # win rate needed to profit (50.5%)


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_15m(ticker: str, days: int = 59) -> pd.DataFrame:
    df = yf.download(f"{ticker}-USD", period=f"{days}d", interval="15m", progress=False)
    if df.empty:
        return df
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df = df.dropna()
    return df


def fetch_1h(ticker: str, years: int = 2) -> pd.DataFrame:
    df = yf.download(f"{ticker}-USD", period=f"{years}y", interval="1h", progress=False)
    if df.empty:
        return df
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    return df.dropna()


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Candle return: close vs open of THIS candle (the target)
    df["ret_this"]  = (df["close"] - df["open"]) / df["open"]
    df["up"]        = (df["ret_this"] > 0).astype(int)

    # Prior candle returns (lookback momentum)
    df["ret_c1"]  = df["close"].pct_change(1)   # 1-candle return  (15m or 1h)
    df["ret_c2"]  = df["close"].pct_change(2)   # 2-candle return  (30m or 2h)
    df["ret_c4"]  = df["close"].pct_change(4)   # 4-candle return  (1h  or 4h)
    df["ret_c8"]  = df["close"].pct_change(8)   # 8-candle return  (2h  or 8h)
    df["ret_c16"] = df["close"].pct_change(16)  # 16-candle return (4h  or 16h)

    # Shift by 1: these are computed from data BEFORE the current candle opens
    for col in ["ret_c1", "ret_c2", "ret_c4", "ret_c8", "ret_c16"]:
        df[col] = df[col].shift(1)

    # Rolling volatility (std of last 8 candle returns) — for vol-normalised momentum
    df["vol_8"] = df["ret_c1"].rolling(8).std()

    # Vol-normalised momentum (z-score)
    for col in ["ret_c1", "ret_c2", "ret_c4", "ret_c8", "ret_c16"]:
        df[f"{col}_z"] = df[col] / (df["vol_8"] + 1e-9)

    # Volume momentum: is this candle's volume above recent average?
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(8).mean()

    # Time features
    if hasattr(df.index, "hour"):
        df["hour"] = df.index.hour
        df["day_of_week"] = df.index.dayofweek
    else:
        df["hour"] = df.index.map(lambda x: x.hour if hasattr(x, "hour") else 0)
        df["day_of_week"] = df.index.map(lambda x: x.dayofweek if hasattr(x, "dayofweek") else 0)

    return df.dropna()


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def win_rate_stats(df: pd.DataFrame, mask: pd.Series) -> tuple[int, float, float]:
    """Return (n, win_rate_pct, ev_cents) for rows selected by mask."""
    sub = df[mask]
    n   = len(sub)
    if n == 0:
        return 0, 0.0, 0.0
    wr  = sub["up"].mean() * 100
    ev  = wr - BREAK_EVEN_WR  # cents above break-even per trade
    return n, wr, ev


def threshold_sweep(
    df: pd.DataFrame,
    momentum_col: str,
    thresholds: list[float],
) -> list[dict]:
    rows = []
    for thr in thresholds:
        bull = df[momentum_col] > thr
        bear = df[momentum_col] < -thr

        n_bull, wr_bull, ev_bull = win_rate_stats(df, bull)
        n_bear, wr_bear, ev_bear = win_rate_stats(df, bear & (df["up"] == 0))

        # Treat bear side as "buy NO" — win when up==0
        n_bear_raw = int((bear).sum())
        wr_bear_no = (df[bear]["up"] == 0).mean() * 100 if n_bear_raw > 0 else 0.0
        ev_bear_no = wr_bear_no - BREAK_EVEN_WR

        rows.append({
            "threshold": thr,
            "n_bull": n_bull, "wr_bull": round(wr_bull, 1), "ev_bull": round(ev_bull, 2),
            "n_bear": n_bear_raw, "wr_bear_no": round(wr_bear_no, 1), "ev_bear_no": round(ev_bear_no, 2),
            "n_total": n_bull + n_bear_raw,
            "combined_ev": round((ev_bull * n_bull + ev_bear_no * n_bear_raw) / max(n_bull + n_bear_raw, 1), 2),
        })
    return rows


def sim_pnl(df: pd.DataFrame, momentum_col: str, threshold: float) -> float:
    """Simulate P&L in cents: +49.5 on win, -50.5 on loss."""
    bull = df[momentum_col] > threshold
    bear = df[momentum_col] < -threshold
    total = 0.0
    total += (df[bull]["up"] *  (100 - 50 - ENTRY_COST)
              + (1 - df[bull]["up"]) * -(50 + ENTRY_COST)).sum()
    # Bear: buy NO — win when up==0
    total += ((1 - df[bear]["up"]) * (100 - 50 - ENTRY_COST)
              + df[bear]["up"] * -(50 + ENTRY_COST)).sum()
    return round(float(total), 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_coin(coin: str, df15: pd.DataFrame, df1h: pd.DataFrame) -> None:
    print(f"\n{'='*72}")
    print(f"  {coin}  —  15-minute momentum analysis")
    print(f"{'='*72}")

    # --- A. Data summary ---
    print(f"\n[A] Data")
    print(f"  15m candles : {len(df15)}  ({df15.index[0].date()} → {df15.index[-1].date()})")
    base_wr = df15["up"].mean() * 100
    print(f"  Base win rate (candle up): {base_wr:.1f}%  (expected ~50%)")
    print(f"  Break-even win rate with {SPREAD_CENTS}¢ spread: {BREAK_EVEN_WR:.1f}%")

    # --- B. Win rate by momentum quintile ---
    print(f"\n[B] Win rate by momentum quintile (does stronger signal → higher win rate?)")
    lookbacks = [("1-candle (15m)", "ret_c1"), ("2-candle (30m)", "ret_c2"),
                 ("4-candle (1h)", "ret_c4"), ("8-candle (2h)", "ret_c8")]
    print(f"  {'Lookback':20s}  {'Q1(bearish)':>12}  {'Q2':>8}  {'Q3':>8}  {'Q4':>8}  {'Q5(bullish)':>12}  spread")
    for lbl, col in lookbacks:
        quintiles = pd.qcut(df15[col], 5, labels=False, duplicates="drop")
        wrs = [df15[quintiles == q]["up"].mean() * 100 for q in range(5)]
        spread_wr = wrs[-1] - wrs[0] if len(wrs) == 5 else 0
        vals = "  ".join(f"{w:>6.1f}%" for w in wrs)
        print(f"  {lbl:20s}  {vals}  Δ={spread_wr:+.1f}pp")

    # --- C. Threshold sweep ---
    print(f"\n[C] Threshold sweep — best (lookback, min|momentum|) combination")
    thresholds = [0.0, 0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02]
    print(f"  {'Lookback':20s}  {'Threshold':>10}  {'n_bull':>7}  {'WR_YES':>7}  {'EV_YES':>7}  "
          f"{'n_bear':>7}  {'WR_NO':>7}  {'EV_NO':>7}  {'Comb EV':>8}")
    best_ev = -999
    best_config = None
    for lbl, col in lookbacks:
        rows = threshold_sweep(df15, col, thresholds)
        for r in rows:
            if r["n_bull"] < 20 or r["n_bear"] < 20:
                continue
            marker = ""
            if r["combined_ev"] > best_ev:
                best_ev = r["combined_ev"]
                best_config = (lbl, col, r["threshold"])
                marker = " ◄ best"
            print(f"  {lbl:20s}  {r['threshold']:>10.3f}  {r['n_bull']:>7}  "
                  f"{r['wr_bull']:>6.1f}%  {r['ev_bull']:>+7.2f}¢  "
                  f"{r['n_bear']:>7}  {r['wr_bear_no']:>6.1f}%  {r['ev_bear_no']:>+7.2f}¢  "
                  f"{r['combined_ev']:>+8.2f}¢{marker}")

    # --- D. Vol-normalised momentum sweep ---
    print(f"\n[D] Vol-normalised momentum (momentum ÷ recent volatility)")
    z_thresholds = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    print(f"  {'Lookback':20s}  {'z-thresh':>8}  {'n_bull':>7}  {'WR_YES':>7}  {'EV_YES':>7}  "
          f"{'n_bear':>7}  {'WR_NO':>7}  {'EV_NO':>7}  {'Comb EV':>8}")
    best_z_ev = -999
    best_z_config = None
    for lbl, col in lookbacks:
        zcol = f"{col}_z"
        rows = threshold_sweep(df15, zcol, z_thresholds)
        for r in rows:
            if r["n_bull"] < 20 or r["n_bear"] < 20:
                continue
            marker = ""
            if r["combined_ev"] > best_z_ev:
                best_z_ev = r["combined_ev"]
                best_z_config = (lbl, zcol, r["threshold"])
                marker = " ◄ best"
            print(f"  {lbl:20s}  {r['threshold']:>8.2f}  {r['n_bull']:>7}  "
                  f"{r['wr_bull']:>6.1f}%  {r['ev_bull']:>+7.2f}¢  "
                  f"{r['n_bear']:>7}  {r['wr_bear_no']:>6.1f}%  {r['ev_bear_no']:>+7.2f}¢  "
                  f"{r['combined_ev']:>+8.2f}¢{marker}")

    # --- E. Time-of-day ---
    print(f"\n[E] Time-of-day analysis (UTC hours, best lookback: {lookbacks[1][0]})")
    col = lookbacks[1][1]  # 30m by default
    print(f"  {'Hour (UTC)':>10}  {'n':>6}  {'base_wr':>8}  {'bull_wr':>8}  {'bear_no_wr':>10}  "
          f"{'bull_n':>7}  {'bear_n':>7}")
    for h in range(0, 24, 2):
        hour_mask = (df15["hour"] >= h) & (df15["hour"] < h + 2)
        sub = df15[hour_mask]
        if len(sub) < 30:
            continue
        bwr = sub["up"].mean() * 100
        bull_mask = hour_mask & (df15[col] > 0.003)
        bear_mask = hour_mask & (df15[col] < -0.003)
        n_bull = int(bull_mask.sum())
        n_bear = int(bear_mask.sum())
        bull_wr = df15[bull_mask]["up"].mean() * 100 if n_bull > 5 else float("nan")
        bear_no_wr = (df15[bear_mask]["up"] == 0).mean() * 100 if n_bear > 5 else float("nan")
        print(f"  {h:02d}:00-{h+2:02d}:00   {len(sub):>6}  {bwr:>7.1f}%  "
              f"{bull_wr:>7.1f}%  {bear_no_wr:>9.1f}%  {n_bull:>7}  {n_bear:>7}")

    # --- F. Volume filter ---
    print(f"\n[F] Does high volume confirm momentum? (lookback=30m, threshold=0.003)")
    col = "ret_c2"
    for vol_label, vol_mask_fn in [
        ("low vol  (ratio<0.8)",  lambda d: d["vol_ratio"] < 0.8),
        ("med vol  (0.8-1.5)",    lambda d: (d["vol_ratio"] >= 0.8) & (d["vol_ratio"] < 1.5)),
        ("high vol (ratio>1.5)",  lambda d: d["vol_ratio"] >= 1.5),
    ]:
        vm = vol_mask_fn(df15)
        bull = vm & (df15[col] > 0.003)
        bear = vm & (df15[col] < -0.003)
        wr_bull = df15[bull]["up"].mean() * 100 if bull.sum() > 5 else float("nan")
        wr_bear = (df15[bear]["up"] == 0).mean() * 100 if bear.sum() > 5 else float("nan")
        print(f"  {vol_label}: bull_wr={wr_bull:.1f}%  bear_no_wr={wr_bear:.1f}%  "
              f"n_bull={int(bull.sum())}  n_bear={int(bear.sum())}")

    # --- G. Best config P&L simulation ---
    if best_config:
        lbl, col, thr = best_config
        pnl = sim_pnl(df15, col, thr)
        n_trades = int(((df15[col] > thr) | (df15[col] < -thr)).sum())
        print(f"\n[G] Best raw-momentum config: {lbl}, threshold={thr:.3f}")
        print(f"  Simulated P&L (assuming entry at 50¢): {pnl:+.1f}¢ over {n_trades} trades "
              f"({pnl/max(n_trades,1):+.2f}¢/trade)")
        print(f"  = ${pnl/100:+.2f} at $1/contract")
    if best_z_config:
        lbl, zcol, zthr = best_z_config
        pnl_z = sim_pnl(df15, zcol, zthr)
        n_z = int(((df15[zcol] > zthr) | (df15[zcol] < -zthr)).sum())
        print(f"\n  Best vol-adj config: {lbl} z>{zthr:.2f}")
        print(f"  Simulated P&L: {pnl_z:+.1f}¢ over {n_z} trades "
              f"({pnl_z/max(n_z,1):+.2f}¢/trade)")


def run_extended_1h(coin: str, df1h: pd.DataFrame) -> None:
    """Extended 2-year backtest using hourly data (KXBTC15M proxy: hourly direction)."""
    print(f"\n{'='*72}")
    print(f"  {coin}  —  Extended 1h backtest (2 years, direction = up next hour)")
    print(f"{'='*72}")
    print(f"  1h candles: {len(df1h)}  ({df1h.index[0].date()} → {df1h.index[-1].date()})")
    print(f"  Base win rate: {df1h['up'].mean()*100:.1f}%")

    lookbacks_1h = [
        ("1h",  "ret_c1"), ("2h",  "ret_c2"),
        ("4h",  "ret_c4"), ("8h",  "ret_c8"), ("16h", "ret_c16"),
    ]
    thresholds = [0.0, 0.002, 0.005, 0.01, 0.02, 0.03]

    print(f"\n  Threshold sweep (1h candles):")
    print(f"  {'Lookback':12s}  {'Threshold':>10}  {'n_bull':>7}  {'WR_YES':>7}  {'EV':>7}  "
          f"{'n_bear':>7}  {'WR_NO':>7}  {'EV':>7}  {'CombEV':>8}")
    best_ev = -999
    for lbl, col in lookbacks_1h:
        rows = threshold_sweep(df1h, col, thresholds)
        for r in rows:
            if r["n_bull"] < 50 or r["n_bear"] < 50:
                continue
            marker = " ◄" if r["combined_ev"] > best_ev else ""
            if r["combined_ev"] > best_ev:
                best_ev = r["combined_ev"]
            print(f"  {lbl:12s}  {r['threshold']:>10.3f}  {r['n_bull']:>7}  "
                  f"{r['wr_bull']:>6.1f}%  {r['ev_bull']:>+7.2f}¢  "
                  f"{r['n_bear']:>7}  {r['wr_bear_no']:>6.1f}%  {r['ev_bear_no']:>+7.2f}¢  "
                  f"{r['combined_ev']:>+8.2f}¢{marker}")


def main(coins: list[str], no_extended: bool) -> None:
    print("Fetching crypto OHLCV data from Yahoo Finance...")
    data = {}
    for coin in coins:
        print(f"  {coin}: 15m (60d)...", end=" ", flush=True)
        df15 = fetch_15m(coin)
        df15 = add_features(df15)
        print(f"{len(df15)} candles", end="   ", flush=True)

        df1h = pd.DataFrame()
        if not no_extended:
            print(f"1h (2y)...", end=" ", flush=True)
            df1h = fetch_1h(coin)
            df1h = add_features(df1h)
            print(f"{len(df1h)} candles")
        else:
            print()

        data[coin] = (df15, df1h)

    for coin, (df15, df1h) in data.items():
        if df15.empty:
            print(f"\n{coin}: no 15m data available, skipping.")
            continue
        run_coin(coin, df15, df1h)
        if not no_extended and not df1h.empty:
            run_extended_1h(coin, df1h)

    # Multi-coin comparison if >1 coin
    if len(coins) > 1:
        print(f"\n{'='*72}")
        print(f"  Multi-coin comparison (30m lookback, 0.3% threshold)")
        print(f"{'='*72}")
        print(f"  {'Coin':6s}  {'base_wr':>8}  {'bull_wr':>8}  {'bear_no_wr':>11}  "
              f"{'n_bull':>7}  {'n_bear':>7}  {'ev_bull':>8}  {'ev_bear':>8}")
        for coin, (df15, _) in data.items():
            if df15.empty:
                continue
            col = "ret_c2"
            thr = 0.003
            bull = df15[col] > thr
            bear = df15[col] < -thr
            bwr = df15["up"].mean() * 100
            bull_wr = df15[bull]["up"].mean() * 100 if bull.sum() > 0 else 0
            bear_no_wr = (df15[bear]["up"] == 0).mean() * 100 if bear.sum() > 0 else 0
            print(f"  {coin:6s}  {bwr:>7.1f}%  {bull_wr:>7.1f}%  {bear_no_wr:>10.1f}%  "
                  f"{int(bull.sum()):>7}  {int(bear.sum()):>7}  "
                  f"{bull_wr-BREAK_EVEN_WR:>+8.2f}¢  {bear_no_wr-BREAK_EVEN_WR:>+8.2f}¢")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coins", nargs="+", default=["BTC", "ETH", "SOL"],
                        help="Coins to analyse (default: BTC ETH SOL)")
    parser.add_argument("--no-extended", action="store_true",
                        help="Skip the 2-year hourly backtest (faster)")
    args = parser.parse_args()
    main([c.upper() for c in args.coins], args.no_extended)
