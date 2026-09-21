#!/usr/bin/env python3
"""Phase 4 — the real test: model band probabilities vs the MARKET at decision time.

Beating the raw-consensus baseline (Phase 3) is necessary but not sufficient. The
market already aggregates the public forecasts. This script asks the only question
that matters: at the 09:00-local decision time, does the model's probability for a
band predict the outcome BETTER than the market's own price?

Data reality (audited 2026-09-16): there is no historical table of market prices
for *every* band at the morning decision time. `raw_forecasts.yes_bid` is 0%
populated; `shadow_model_no` prices are post-breach afternoon. The one broad morning
market-price source is the `opportunities` log: KXHIGH band rows with yes_bid/yes_ask
(100% populated) across all hours. BUT it only logs bands the bot *flagged*
(near-money / apparent-edge), so this is a SELECTION-BIASED subset, not the full
per-day distribution. That subset is exactly the near-money region where skill is
hardest and where trading would happen, so it is a fair — if partial — test. The
definitive test is forward shadow logging of all bands (Phase 5).

Per morning band-instance we compare, against the realized outcome:
  market_p = mid (yes_bid+yes_ask)/2       [what you'd actually pay/receive]
  model_p  = N(mu, sigma_recal) integrated over the band [lo-0.5, hi+0.5)
  consensus_p = N(consensus_median, sig_cons) over the same band   [Phase-3 baseline]
Metrics: Brier + log-loss, with day-CLUSTERED bootstrap CIs (same-day band bets are
correlated — the naive IID CI overstates significance), calibration, and a simple
decision-time P&L proxy at real book prices net of Kalshi fees.

Usage:
    venv/bin/python scripts/eval_lookahead_vs_market.py
"""
from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime, time, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kalshi_bot.cities import CITIES  # noqa: E402
from kalshi_bot.market_parser import TICKER_TO_METRIC  # noqa: E402

# ticker prefix -> display city (longest-prefix match; metric -> CITIES display)
_PREFIX_CITY = {pfx: CITIES[m][0] for pfx, m in TICKER_TO_METRIC.items()
                if m in CITIES and pfx.startswith("KXHIGH")}
_PREFIXES = sorted(_PREFIX_CITY, key=len, reverse=True)  # longest first
CITY_TZ = {v[0]: v[3] for v in CITIES.values()}

FEE = 0.01          # Kalshi ~1c/contract proxy (fraction of $1 notional)
EDGE_THR = 0.08     # min model-vs-market edge to "trade" in the P&L proxy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default="data/db/opportunity_log.db")
    p.add_argument("--oof", default="data/lookahead/oof_predictions_high.parquet")
    p.add_argument("--cutoff", default="09:00")
    p.add_argument("--boot", type=int, default=2000)
    return p.parse_args()


def ticker_city(tk: str) -> str | None:
    head = tk.split("-", 1)[0]
    for pfx in _PREFIXES:
        if head == pfx:
            return _PREFIX_CITY[pfx]
    return None


def parse_date(tk: str):
    try:
        return datetime.strptime(tk.split("-")[1], "%y%b%d").date()
    except (IndexError, ValueError):
        return None


def load_market(db: str, cutoff: time) -> pd.DataFrame:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        df = pd.read_sql_query(
            "SELECT logged_at, ticker, strike_lo, strike_hi, yes_bid, yes_ask "
            "FROM opportunities WHERE ticker LIKE 'KXHIGH%' AND direction='between' "
            "AND yes_bid IS NOT NULL AND yes_ask IS NOT NULL "
            "AND strike_lo IS NOT NULL AND strike_hi IS NOT NULL", con)
    finally:
        con.close()
    df["city"] = df["ticker"].map(ticker_city)
    df["date_target"] = df["ticker"].map(parse_date)
    df = df.dropna(subset=["city", "date_target"])
    df["logged_at"] = pd.to_datetime(df["logged_at"], utc=True, format="mixed")

    # keep only snapshots at/ before the 09:00-local decision cutoff, latest per band
    cut = [pd.Timestamp(datetime.combine(d, cutoff)
                        .replace(tzinfo=CITY_TZ[c]).astimezone(timezone.utc))
           for c, d in zip(df["city"], df["date_target"])]
    df["cutoff"] = cut
    df = df[df["logged_at"] <= df["cutoff"]]
    df = df.sort_values("logged_at").groupby("ticker", as_index=False).last()
    return df


def band_prob(mu, sigma, lo, hi):
    """P(high in band) = P(true high in [lo-0.5, hi+0.5))."""
    return norm.cdf(hi + 0.5, mu, sigma) - norm.cdf(lo - 0.5, mu, sigma)


def brier(p, y):
    return (p - y) ** 2


def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def clustered_bootstrap(df, col_a, col_b, n_boot, seed=0):
    """Mean (a - b) with day-clustered bootstrap CI. Positive => a worse than b."""
    rng = np.random.default_rng(seed)
    diffs = (df[col_a] - df[col_b]).to_numpy()
    # df is reset-indexed, so group labels are positional indices into `diffs`
    groups = list(df.groupby(["city", "date_target"]).indices.values())
    point = float(np.mean(diffs))
    boots = np.empty(n_boot)
    n = len(groups)
    for b in range(n_boot):
        pick = rng.integers(0, n, n)
        rows = np.concatenate([groups[i] for i in pick])
        boots[b] = diffs[rows].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi


def calib_table(df, pcol, ycol, bins=(0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0)):
    b = pd.cut(df[pcol], bins=bins, include_lowest=True)
    g = df.groupby(b, observed=True).agg(n=(ycol, "size"), pred=(pcol, "mean"),
                                         obs=(ycol, "mean"))
    return g


def pnl_proxy(df):
    """Trade the model's edge vs the book at decision time, settle at outcome.

    Buy YES at ask when model_p - ask > EDGE_THR; buy NO at (1-bid) when
    bid - model_p > EDGE_THR. Payoff in $1 notional, minus a per-trade fee.
    """
    ask = df["yes_ask"] / 100.0
    bid = df["yes_bid"] / 100.0
    y = df["outcome"].to_numpy()
    mp = df["model_p"].to_numpy()
    pnl = np.zeros(len(df)); traded = np.zeros(len(df), bool); side = np.array([""] * len(df), object)
    buy_yes = mp - ask.to_numpy() > EDGE_THR
    buy_no = bid.to_numpy() - mp > EDGE_THR
    pnl[buy_yes] = np.where(y[buy_yes] == 1, 1 - ask.to_numpy()[buy_yes], -ask.to_numpy()[buy_yes]) - FEE
    pnl[buy_no] = np.where(y[buy_no] == 0, bid.to_numpy()[buy_no], -(1 - bid.to_numpy()[buy_no])) - FEE
    traded = buy_yes | buy_no
    return pnl, traded


def main() -> None:
    a = parse_args()
    hh, mm = (int(x) for x in a.cutoff.split(":"))
    cutoff = time(hh, mm)

    oof = pd.read_parquet(a.oof)
    mk = load_market(a.db, cutoff)
    print(f"morning market band-snapshots (<= {a.cutoff} local): {len(mk):,} "
          f"tickers, {mk['city'].nunique()} cities, "
          f"{mk['date_target'].min()} -> {mk['date_target'].max()}")

    df = mk.merge(oof[["city", "date_target", "actual_f", "mu", "sigma_recal",
                       "consensus_median", "sig_cons"]],
                  on=["city", "date_target"], how="inner").reset_index(drop=True)
    print(f"joined to model OOS window: {len(df):,} band-instances, "
          f"{df.groupby(['city','date_target']).ngroups} city-days")
    if df.empty:
        print("no overlap — nothing to evaluate."); return

    lo, hi = df["strike_lo"].to_numpy(), df["strike_hi"].to_numpy()
    df["outcome"] = ((df["actual_f"] >= lo - 0.5) & (df["actual_f"] < hi + 0.5)).astype(int)
    df["market_p"] = ((df["yes_bid"] + df["yes_ask"]) / 200.0).clip(1e-4, 1 - 1e-4)
    df["model_p"] = band_prob(df["mu"], df["sigma_recal"], lo, hi).clip(1e-6, 1 - 1e-6)
    df["cons_p"] = band_prob(df["consensus_median"], df["sig_cons"], lo, hi).clip(1e-6, 1 - 1e-6)

    for who, pc in [("market", "market_p"), ("model", "model_p"), ("consensus", "cons_p")]:
        df[f"brier_{who}"] = brier(df[pc], df["outcome"])
        df[f"ll_{who}"] = logloss(df[pc], df["outcome"])

    print("\n" + "=" * 64 + "\nPHASE 4 — MODEL vs MARKET (morning decision time)\n" + "=" * 64)
    print(f"base rate (band wins): {df['outcome'].mean():.3f}")
    for who in ("market", "model", "consensus"):
        print(f"  {who:10s} Brier {df[f'brier_{who}'].mean():.4f}   "
              f"log-loss {df[f'll_{who}'].mean():.4f}")

    print("\nday-clustered bootstrap, mean(model - market)  [<0 => model beats market]:")
    for metric in ("brier", "ll"):
        pt, l, h = clustered_bootstrap(df, f"{metric}_model", f"{metric}_market", a.boot)
        verdict = "MODEL BETTER" if h < 0 else ("MARKET BETTER" if l > 0 else "TIE (CI spans 0)")
        print(f"  {metric:6s} diff {pt:+.4f}  95% CI [{l:+.4f}, {h:+.4f}]  -> {verdict}")

    print("\nmodel calibration (decision-time band prob vs realized):")
    print(calib_table(df, "model_p", "outcome").to_string())
    print("\nmarket calibration (sanity — should be near-diagonal):")
    print(calib_table(df, "market_p", "outcome").to_string())

    pnl, traded = pnl_proxy(df)
    n_t = int(traded.sum())
    print(f"\nP&L proxy (edge>{EDGE_THR}, fee {FEE}): trades={n_t}/{len(df)}  "
          f"total={pnl[traded].sum():+.2f}  mean/trade={pnl[traded].mean() if n_t else 0:+.4f}")
    if n_t:
        wins = ((df['outcome'].to_numpy()[traded] == 1) & (pnl[traded] > 0)).sum()
        print(f"  gross win rate on trades: {(pnl[traded] > 0).mean():.3f}")

    print("\nCAVEAT: selection-biased subset (only bot-flagged near-money bands); a "
          "clean full-distribution test needs forward shadow logging (Phase 5).")


if __name__ == "__main__":
    main()
