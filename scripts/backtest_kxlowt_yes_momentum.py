#!/usr/bin/env python3
"""Backtest: KXLOWT YES price momentum strategy (v2 — entry execution sweep).

Hypothesis: YES bids on KXLOWT markets spike intraday even on days that settle NO.
Buy YES cheap in the morning (bid ≤ ENTRY_CAP), profit-take when it spikes, ignore
settlement outcome.

v1 finding: all combos negative EV when paying ask. Root cause: wide spreads on cheap
markets (bid=10¢, ask=25¢ typical). v2 sweeps the entry price model from 'bid'
(passive limit, optimistic) through 'mid' to 'ask' (market order, v1 behavior), and
also sweeps max_spread filters to find the breakeven point.

Entry strategies
----------------
  A — Pure price   : first morning candle where YES_bid ≤ ENTRY_CAP
  B — NO signal    : same price gate + raw_forecasts confirms NO edge ≥ 0.5°F
  C — Price dip    : YES was ≥ DIP_HIGH in first 2h, then drops to ≤ ENTRY_CAP

Exit strategies
---------------
  time_stop  : PT if bid ≥ PT_TARGET, else exit N hours before close at bid
  settle     : PT if bid ≥ PT_TARGET, else hold to settlement
  price_stop : PT if bid ≥ PT_TARGET, else hard stop if bid ≤ PRICE_STOP

Entry price models (new in v2)
------------------------------
  bid        YES_bid at entry — passive limit fill (optimistic)
  bid+1/2/3  Slightly above bid — limit order just inside spread
  mid        (bid + ask) / 2
  ask        YES_ask — market order (v1 behavior)

NOTE: 'bid' and 'bid+N' assume passive fill; fill probability unknown for illiquid
markets. Use these as upper bounds on achievable EV, not guaranteed performance.

Data
----
  data/candlesticks.db          — 962 settled KXLOWT tickers, hourly bid/ask OHLC
  data/db/opportunity_log.db    — raw_forecasts for strategy B

Run
---
  venv/bin/python scripts/backtest_kxlowt_yes_momentum.py
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from datetime import datetime, timezone

CANDLES_DB = "data/candlesticks.db"
TRADES_DB  = "data/db/opportunity_log.db"

# Entry window: 06:00–10:00 UTC on observation day
ENTRY_WINDOW_OFFSET = 16 * 3600
ENTRY_WINDOW_WIDTH  = 4  * 3600
DIP_WINDOW_WIDTH    = 2  * 3600

# Core parameter sweep (exit combos)
ENTRY_CAPS   = [10, 15, 20]
PT_TARGETS   = [25, 30, 35, 40]
TIME_STOPS   = [1, 2, 3]
PRICE_STOPS  = [3, 5]
DIP_HIGHS    = [20, 25]

# New v2 dimensions
ENTRY_MODELS = ["bid", "bid+1", "bid+2", "bid+3", "mid", "ask"]
MAX_SPREADS  = [3, 5, 8, 10, 15]

MIN_ENTRY_BID = 2


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_kxlowt_markets(con: sqlite3.Connection) -> list[dict]:
    rows = con.execute("""
        SELECT ticker, series, result, open_ts, close_ts
        FROM markets
        WHERE series LIKE 'KXLOWT%' AND result IN ('yes', 'no')
        ORDER BY open_ts
    """).fetchall()
    return [
        {"ticker": r[0], "series": r[1], "result": r[2],
         "open_ts": r[3], "close_ts": r[4],
         "settle_val": 100 if r[2] == "yes" else 0}
        for r in rows
    ]


def load_candles(con: sqlite3.Connection, ticker: str) -> list[dict]:
    rows = con.execute("""
        SELECT period_ts, bid_close, bid_open, ask_close, ask_open
        FROM candles WHERE ticker = ? ORDER BY period_ts
    """, (ticker,)).fetchall()
    out = []
    for r in rows:
        bid = r[1] if r[1] is not None else r[2]
        ask = r[3] if r[3] is not None else r[4]
        out.append({"ts": r[0], "bid": bid, "ask": ask})
    return out


def load_no_signals(trades_con: sqlite3.Connection) -> dict[str, set[str]]:
    rows = trades_con.execute("""
        SELECT DISTINCT ticker, date(logged_at)
        FROM raw_forecasts
        WHERE ticker LIKE 'KXLOWT%' AND edge >= 0.5
    """).fetchall()
    result: dict[str, set[str]] = defaultdict(set)
    for ticker, d in rows:
        result[d].add(ticker)
    return result


# ---------------------------------------------------------------------------
# Entry logic
# ---------------------------------------------------------------------------

def _entry_cost(bid: int, ask: int, model: str) -> int:
    if model == "bid":   return bid
    if model == "bid+1": return bid + 1
    if model == "bid+2": return bid + 2
    if model == "bid+3": return bid + 3
    if model == "mid":   return (bid + ask) // 2
    return ask  # 'ask'


def _find_entry(candles, open_ts, entry_cap, max_spread, entry_model):
    w_start = open_ts + ENTRY_WINDOW_OFFSET
    w_end   = w_start + ENTRY_WINDOW_WIDTH
    for c in candles:
        if c["ts"] < w_start: continue
        if c["ts"] > w_end:   break
        bid, ask = c["bid"], c["ask"]
        if bid is None or ask is None or bid < MIN_ENTRY_BID: continue
        if bid > entry_cap or (ask - bid) > max_spread:       continue
        return {"ts": c["ts"], "bid": bid, "cost": _entry_cost(bid, ask, entry_model)}
    return None


def _find_entry_dip(candles, open_ts, entry_cap, dip_high, max_spread, entry_model):
    dip_end = open_ts + DIP_WINDOW_WIDTH
    if not any(c["bid"] is not None and c["bid"] >= dip_high
               for c in candles if c["ts"] <= dip_end):
        return None
    return _find_entry(candles, open_ts, entry_cap, max_spread, entry_model)


# ---------------------------------------------------------------------------
# Exit simulation
# ---------------------------------------------------------------------------

def _simulate(candles, entry_ts, cost, pt_target, exit_mode, close_ts, settle_val,
              time_stop_h, price_stop):
    peak = cost
    for c in candles:
        if c["ts"] <= entry_ts: continue
        bid = c["bid"]
        if bid is None: continue
        peak = max(peak, bid)

        if bid >= pt_target:
            return {"reason": "pt", "pnl": bid - cost,
                    "hours": (c["ts"] - entry_ts) / 3600}

        if exit_mode == "time_stop" and c["ts"] >= close_ts - time_stop_h * 3600:
            return {"reason": "time_stop", "pnl": bid - cost, "hours": None}
        if exit_mode == "price_stop" and bid <= price_stop:
            return {"reason": "price_stop", "pnl": bid - cost, "hours": None}

    return {"reason": "settle", "pnl": settle_val - cost, "hours": None}


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_strategy(markets, candles_cache, strategy, no_signals, max_spread, entry_model):
    """Return results dict keyed by (entry_cap, pt_target, exit_mode, extra)."""
    results = {}

    for entry_cap in ENTRY_CAPS:
        entered: list[tuple[dict, dict]] = []

        for mkt in markets:
            ticker  = mkt["ticker"]
            candles = candles_cache.get(ticker)
            if not candles: continue

            obs_date = datetime.fromtimestamp(
                mkt["open_ts"] + ENTRY_WINDOW_OFFSET, tz=timezone.utc
            ).date().isoformat()

            if strategy == "A":
                e = _find_entry(candles, mkt["open_ts"], entry_cap, max_spread, entry_model)
                if e: entered.append((mkt, e))

            elif strategy == "B":
                if no_signals and ticker in no_signals.get(obs_date, set()):
                    e = _find_entry(candles, mkt["open_ts"], entry_cap, max_spread, entry_model)
                    if e: entered.append((mkt, e))

            else:  # C
                for dip_high in DIP_HIGHS:
                    e = _find_entry_dip(candles, mkt["open_ts"], entry_cap,
                                        dip_high, max_spread, entry_model)
                    if e:
                        entered.append((mkt, e))
                        break  # one entry per market (first dip_high that qualifies)

        for pt in PT_TARGETS:
            for mode in ("time_stop", "settle", "price_stop"):
                extras = TIME_STOPS if mode == "time_stop" else (
                    PRICE_STOPS if mode == "price_stop" else [None])
                for extra in extras:
                    key = (entry_cap, pt, mode, extra)
                    n = spike_n = 0
                    pnl_total = 0.0
                    hours_list = []
                    for mkt, e in entered:
                        sim = _simulate(
                            candles_cache[mkt["ticker"]], e["ts"], e["cost"],
                            pt, mode, mkt["close_ts"], mkt["settle_val"],
                            extra if mode == "time_stop" else 2,
                            extra if mode == "price_stop" else 5,
                        )
                        n += 1
                        pnl_total += sim["pnl"]
                        if sim["reason"] == "pt":
                            spike_n += 1
                            hours_list.append(sim["hours"])
                    results[key] = {"n": n, "spike_n": spike_n,
                                    "pnl": pnl_total, "hrs": hours_list}
    return results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def best_ev(results: dict) -> tuple[tuple, float]:
    best_key, best = None, float("-inf")
    for k, b in results.items():
        if b["n"] == 0: continue
        ev = b["pnl"] / b["n"]
        if ev > best:
            best, best_key = ev, k
    return best_key, best


def print_top_combos(results: dict, label: str, top_n: int = 8) -> None:
    ranked = sorted(
        [(k, b) for k, b in results.items() if b["n"] > 0],
        key=lambda x: x[1]["pnl"] / x[1]["n"],
        reverse=True,
    )[:top_n]
    print(f"\n  Top {top_n} combos — {label}")
    print(f"  {'EntCap':>7} {'PT':>4} {'ExitMode':>12} {'Param':>5} "
          f"{'N':>5} {'Spike%':>7} {'AvgHrs':>7} {'EV/tr':>7} {'Total($)':>9}")
    print("  " + "-" * 68)
    for k, b in ranked:
        entry_cap, pt, mode, extra = k
        ev = b["pnl"] / b["n"]
        sp = b["spike_n"] / b["n"]
        ah = sum(b["hrs"]) / len(b["hrs"]) if b["hrs"] else 0.0
        ps = (f"{extra}h" if mode == "time_stop" else
              f"{extra}¢" if mode == "price_stop" else "—")
        print(f"  {entry_cap:>6}¢ {pt:>3}¢ {mode:>12} {ps:>5} "
              f"{b['n']:>5} {sp:>6.1%} {ah:>7.1f} {ev:>+7.1f}¢ {b['pnl']/100:>+9.2f}")


def print_breakeven_table(
    all_results: dict,          # {(entry_model, max_spread): results_dict}
    strategy_label: str,
    fixed_entry_cap: int = 20,
    fixed_pt: int = 40,
    fixed_mode: str = "settle",
    fixed_extra=None,
) -> None:
    """Print EV by (entry_model × max_spread) for a fixed exit combo."""
    key = (fixed_entry_cap, fixed_pt, fixed_mode, fixed_extra)
    print(f"\n{'='*80}")
    print(f"BREAKEVEN TABLE — Strategy {strategy_label}")
    print(f"  (exit: {fixed_mode}, PT={fixed_pt}¢, entry≤{fixed_entry_cap}¢)")
    print(f"  NOTE: 'bid' and 'bid+N' assume passive limit order fills.")

    # Header
    spread_labels = [f"{ms}¢" for ms in MAX_SPREADS]
    print(f"\n  {'entry_model':>12}  " + "  ".join(f"{s:>7}" for s in spread_labels))
    print("  " + "-" * (14 + 9 * len(MAX_SPREADS)))

    for em in ENTRY_MODELS:
        row_vals = []
        for ms in MAX_SPREADS:
            b = all_results.get((em, ms), {}).get(key)
            if b and b["n"] > 0:
                ev = b["pnl"] / b["n"]
                flag = "+" if ev >= 0 else ""
                row_vals.append(f"{flag}{ev:.1f}¢")
            else:
                row_vals.append("  —  ")
        n_sample = None
        for ms in MAX_SPREADS:
            b = all_results.get((em, ms), {}).get(key)
            if b and b["n"] > 0:
                n_sample = b["n"]
                break
        n_str = f"(n≈{n_sample})" if n_sample else ""
        print(f"  {em:>12}  " + "  ".join(f"{v:>7}" for v in row_vals) + f"  {n_str}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    candles_con = sqlite3.connect(f"file:{CANDLES_DB}?mode=ro", uri=True)
    trades_con  = sqlite3.connect(f"file:{TRADES_DB}?mode=ro", uri=True)

    print("Loading markets and candles…", flush=True)
    markets = load_kxlowt_markets(candles_con)
    candles_cache = {m["ticker"]: load_candles(candles_con, m["ticker"]) for m in markets}
    print(f"  {len(markets)} markets, {len(candles_cache)} candle series")

    print("Loading NO signal index…", flush=True)
    no_signals = load_no_signals(trades_con)
    candles_con.close(); trades_con.close()

    # Collect all results across (strategy × entry_model × max_spread)
    all_by_strategy: dict[str, dict] = {"A": {}, "B": {}, "C": {}}

    for strategy in ("A", "B", "C"):
        label = {"A": "Pure Price", "B": "NO Signal", "C": "Price Dip"}[strategy]
        print(f"\nRunning Strategy {strategy} ({label})…", flush=True)

        for em in ENTRY_MODELS:
            for ms in MAX_SPREADS:
                r = run_strategy(markets, candles_cache, strategy,
                                 no_signals if strategy == "B" else None,
                                 ms, em)
                all_by_strategy[strategy][(em, ms)] = r

        print(f"  Done.", flush=True)

    # --- Per-strategy output ---
    for strategy in ("A", "B", "C"):
        label = {"A": "A — Pure Price", "B": "B — NO Signal (2026-05+)",
                 "C": "C — Price Dip"}[strategy]
        print(f"\n{'='*80}")
        print(f"STRATEGY {label}")
        print(f"{'='*80}")

        # Show top combos at each entry model (fixed max_spread=15 for comparability)
        for em in ENTRY_MODELS:
            results_15 = all_by_strategy[strategy].get((em, 15), {})
            if any(b["n"] > 0 for b in results_15.values()):
                print_top_combos(results_15, f"entry_model={em}, max_spread=15¢", top_n=3)

        # Breakeven table
        print_breakeven_table(
            all_by_strategy[strategy],
            label,
            fixed_entry_cap=20,
            fixed_pt=40,
            fixed_mode="settle",
            fixed_extra=None,
        )

    # --- Global summary ---
    print(f"\n{'='*80}")
    print("SUMMARY — Best EV/trade per (strategy × entry_model) at max_spread=15¢")
    print(f"{'='*80}")
    print(f"  {'Strategy':>20} {'EntModel':>8} {'Best combo':>35} "
          f"{'N':>5} {'Spike%':>7} {'EV/tr':>7}")
    print("  " + "-" * 90)
    for strategy in ("A", "B", "C"):
        slabel = {"A": "A-PurePrice", "B": "B-NOSignal", "C": "C-PriceDip"}[strategy]
        for em in ENTRY_MODELS:
            results = all_by_strategy[strategy].get((em, 15), {})
            bk, bev = best_ev(results)
            if bk and results[bk]["n"] > 0:
                b = results[bk]
                entry_cap, pt, mode, extra = bk
                ps = (f"{extra}h" if mode == "time_stop" else
                      f"{extra}¢" if mode == "price_stop" else "—")
                combo = f"cap={entry_cap}¢ PT={pt}¢ {mode}({ps})"
                sp = b["spike_n"] / b["n"]
                print(f"  {slabel:>20} {em:>8} {combo:>35} "
                      f"{b['n']:>5} {sp:>6.1%} {bev:>+7.1f}¢")


if __name__ == "__main__":
    main()
