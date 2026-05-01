"""Momentum / price-spike backtest on Kalshi historical candlestick data.

Reads candlestick data stored by fetch_candlestick_history.py and answers:

  Do Kalshi price spikes — a rapid move after a period of price stability —
  tend to continue far enough to hit a profit-take, or reverse immediately?

Strategy definition
-------------------
  1. Market is "stable" if the std-dev of bid_close over the last
     stability_window minutes is ≤ stability_thresh cents.
  2. A "spike" is detected when the next candle's bid_close moves ≥ spike_min
     cents from the previous candle.
  3. We simulate entry at ask_close of the spike candle (buying at ask).
  4. We then scan subsequent candles for exit:
       PT fires when bid_close ≥ entry × (1 + pt)
       SL fires when bid_close ≤ entry × (1 - sl)
       Market-end: use settlement result (won = +bid side, lost = -entry)
  5. No overlapping trades per (pt, sl) combo — skip past the exit candle.

Two directions are tested:
  - momentum: follow the spike (bid up → buy YES, bid down → buy NO)
  - fade: fade the spike (bid up → buy NO, bid down → buy YES)

Performance note
----------------
  The sweep loops over (sw, st, sm) combos (36 total). For each combo, all
  markets are scanned once and all (pt, sl) combinations (12 total) are
  evaluated in a single forward pass per spike. This avoids the O(864 × N)
  cost of re-scanning candles for every parameter combination.

Output
------
  Section 1  — Parameter sweep (top combos by avg P&L)
  Section 2  — Spike characterisation (size, duration, frequency)
  Section 3  — Momentum vs fade comparison at best parameters
  Section 4  — Time-of-day analysis (UTC hour)
  Section 5  — By market type (series prefix)
  Section 6  — Caveats

Usage
-----
  venv/bin/python scripts/backtest_momentum.py
  venv/bin/python scripts/backtest_momentum.py --db data/candlesticks_test.db
  venv/bin/python scripts/backtest_momentum.py \\
      --series KXHIGHTCHI KXBTCD --out momentum_backtest.txt
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).parent.parent))

DB_PATH = Path(__file__).parent.parent / "data" / "candlesticks.db"

# ---------------------------------------------------------------------------
# Parameter sweep grid
# ---------------------------------------------------------------------------

STABILITY_WINDOWS  = [5, 10, 20, 30]     # minutes of prior candles required
STABILITY_THRESHES = [1, 2, 3]           # ¢ — max std-dev for "stable"
SPIKE_MINS         = [3, 5, 8]           # ¢ — min move to qualify as spike
PT_THRESHES        = [0.08, 0.10, 0.15, 0.20]   # profit-take fraction
SL_THRESHES        = [0.05, 0.08, 0.10]          # stop-loss fraction

# How many top parameter combos to print in the summary table
TOP_N = 10

# All (pt, sl) pairs — evaluated simultaneously per spike
_PT_SL_COMBOS: list[tuple[float, float]] = [
    (pt, sl) for pt in PT_THRESHES for sl in SL_THRESHES
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_markets(con: sqlite3.Connection, series_filter: list[str] | None) -> list[dict]:
    where = ""
    params: list = []
    if series_filter:
        placeholders = ",".join("?" * len(series_filter))
        where = f"WHERE series IN ({placeholders})"
        params = list(series_filter)
    rows = con.execute(
        f"SELECT ticker, series, open_ts, close_ts, result FROM markets {where}",
        params,
    ).fetchall()
    return [
        {"ticker": r[0], "series": r[1], "open_ts": r[2],
         "close_ts": r[3], "result": r[4]}
        for r in rows
    ]


def load_candles(con: sqlite3.Connection, ticker: str) -> list[dict]:
    """Load all candles for one ticker, sorted by time."""
    rows = con.execute(
        """
        SELECT period_ts, bid_open, bid_close, bid_low, bid_high,
               ask_open, ask_close, price_close, volume
        FROM candles WHERE ticker=? ORDER BY period_ts
        """,
        (ticker,),
    ).fetchall()
    return [
        {
            "period_ts":   r[0],
            "bid_open":    r[1],
            "bid_close":   r[2],
            "bid_low":     r[3],
            "bid_high":    r[4],
            "ask_open":    r[5],
            "ask_close":   r[6],
            "price_close": r[7],
            "volume":      r[8],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Fast rolling std-dev (Welford / incremental)
# ---------------------------------------------------------------------------

def _std_of(vals: list) -> float:
    """Population std-dev of a list of numbers, skipping Nones. Returns 0 for <2 items."""
    clean = [v for v in vals if v is not None]
    n = len(clean)
    if n < 2:
        return 0.0
    m = sum(clean) / n
    return sqrt(sum((v - m) ** 2 for v in clean) / n)


# ---------------------------------------------------------------------------
# Core simulation — multi PT/SL evaluation in one forward pass per spike
# ---------------------------------------------------------------------------

def simulate_market_multi(
    candles: list[dict],
    market: dict,
    *,
    stability_window: int,
    stability_thresh: float,
    spike_min: float,
    direction: str,   # "momentum" | "fade"
) -> dict[tuple[float, float], list[dict]]:
    """
    Scan candles once for (sw, st, sm) spike detection.
    For each spike found, evaluate all (pt, sl) combos in one forward pass.

    Returns dict: (pt, sl) → list of simulated trade records.
    """
    n = len(candles)
    results: dict[tuple[float, float], list[dict]] = {k: [] for k in _PT_SL_COMBOS}
    # Track last-exit candle index per (pt, sl) to prevent overlapping trades
    last_exit: dict[tuple[float, float], int] = {k: -1 for k in _PT_SL_COMBOS}

    t = stability_window

    while t < n:
        # --- Stability check ---
        window_bids = [
            candles[i]["bid_close"]
            for i in range(t - stability_window, t)
        ]
        std_val = _std_of(window_bids)
        if std_val > stability_thresh:
            t += 1
            continue

        # --- Spike check ---
        prev_bid = candles[t - 1].get("bid_close")
        curr_bid = candles[t].get("bid_close")
        curr_ask = candles[t].get("ask_close")

        if prev_bid is None or curr_bid is None or curr_ask is None:
            t += 1
            continue

        move = curr_bid - prev_bid
        if abs(move) < spike_min:
            t += 1
            continue

        # Spike detected at candle t
        spike_up = move > 0

        if direction == "momentum":
            side = "yes" if spike_up else "no"
        else:
            side = "no" if spike_up else "yes"

        entry_price = curr_ask if side == "yes" else 100 - curr_bid
        if entry_price <= 0 or entry_price >= 100:
            t += 1
            continue

        # Determine which (pt, sl) combos are eligible (no overlap)
        active = {combo for combo in _PT_SL_COMBOS if t > last_exit[combo]}
        if not active:
            t += 1
            continue

        # Precompute PT/SL targets for active combos
        targets = {
            (pt, sl): (entry_price * (1.0 + pt), entry_price * (1.0 - sl))
            for (pt, sl) in active
        }

        # --- Single forward scan for all active (pt, sl) ---
        pending = set(active)  # combos still looking for an exit

        for s in range(t + 1, n):
            if not pending:
                break
            bid = candles[s].get("bid_close")
            if bid is None:
                continue

            current_exit = bid if side == "yes" else 100 - bid

            resolved = []
            for combo in pending:
                pt_target, sl_floor = targets[combo]
                if current_exit >= pt_target:
                    reason = "profit_take"
                elif current_exit <= sl_floor:
                    reason = "stop_loss"
                else:
                    continue
                pnl = current_exit - entry_price
                results[combo].append({
                    "ticker":      market["ticker"],
                    "series":      market["series"],
                    "spike_ts":    candles[t]["period_ts"],
                    "spike_size":  abs(move),
                    "spike_dir":   "up" if spike_up else "down",
                    "side":        side,
                    "entry_price": entry_price,
                    "exit_price":  current_exit,
                    "exit_reason": reason,
                    "pnl_cents":   pnl,
                    "hold_candles": s - t,
                    "std_before":  round(std_val, 2),
                })
                last_exit[combo] = s
                resolved.append(combo)
            for combo in resolved:
                pending.discard(combo)

        # Market-end exits for still-pending combos
        if pending:
            result = market.get("result", "")
            for combo in pending:
                if (side == "yes" and result == "yes") or (side == "no" and result == "no"):
                    exit_price  = 99
                    exit_reason = "settlement_win"
                elif result in ("yes", "no"):
                    exit_price  = 1
                    exit_reason = "settlement_loss"
                else:
                    exit_price  = entry_price
                    exit_reason = "market_end"
                results[combo].append({
                    "ticker":      market["ticker"],
                    "series":      market["series"],
                    "spike_ts":    candles[t]["period_ts"],
                    "spike_size":  abs(move),
                    "spike_dir":   "up" if spike_up else "down",
                    "side":        side,
                    "entry_price": entry_price,
                    "exit_price":  exit_price,
                    "exit_reason": exit_reason,
                    "pnl_cents":   exit_price - entry_price,
                    "hold_candles": n - 1 - t,
                    "std_before":  round(std_val, 2),
                })
                last_exit[combo] = n - 1

        t += 1

    return results


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _pct(n: int, d: int) -> str:
    return f"{n/d*100:.1f}%" if d > 0 else "  n/a"


def _bucket_spike_size(s: float) -> str:
    if s < 5:  return "3-4¢"
    if s < 8:  return "5-7¢"
    if s < 12: return "8-11¢"
    return "≥12¢"


def _series_prefix(series: str) -> str:
    if series.startswith("KXHIGHT"): return "KXHIGHT (temp-high new)"
    if series.startswith("KXHIGH"):  return "KXHIGH (temp-high old)"
    if series.startswith("KXLOWT"):  return "KXLOWT (temp-low)"
    if series.startswith("KXBTC"):   return "KXBTCD (bitcoin)"
    if series.startswith("KXETH"):   return "KXETH"
    if series.startswith("KXSOL"):   return "KXSOL"
    return series[:8]


def _utc_hour(ts: int) -> int:
    return datetime.fromtimestamp(ts, tz=timezone.utc).hour


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------

def run(args, out) -> None:
    con = sqlite3.connect(args.db)

    series_filter = args.series if args.series else None
    markets = load_markets(con, series_filter)

    if not markets:
        print("No markets found in database matching the filter.", file=out)
        return

    if args.from_date:
        cutoff_ts = int(datetime.fromisoformat(args.from_date).replace(
            tzinfo=timezone.utc).timestamp())
        markets = [m for m in markets if m["close_ts"] >= cutoff_ts]

    print("=" * 72, file=out)
    print("  MOMENTUM / PRICE-SPIKE BACKTEST", file=out)
    print("=" * 72, file=out)
    print(f"  Markets in DB : {len(markets)}", file=out)
    print(f"  Series filter : {series_filter or 'all'}", file=out)
    print(file=out)

    # Pre-load all candles into memory
    print("  Loading candles...", file=out, flush=True)
    candle_cache: dict[str, list[dict]] = {}
    for m in markets:
        candle_cache[m["ticker"]] = load_candles(con, m["ticker"])
    total_candles = sum(len(v) for v in candle_cache.values())
    print(f"  Loaded {total_candles:,} candles across {len(markets)} markets.", file=out)
    print(file=out)

    # -----------------------------------------------------------------
    # Section 1 — Parameter sweep
    # Outer loop: (sw, st, sm) = 36 combos
    # Inner: all markets scanned once; all (pt, sl) evaluated per spike
    # -----------------------------------------------------------------
    print("--- Section 1: Parameter Sweep (momentum direction) ---", file=out)
    print(f"  Sweeping {len(STABILITY_WINDOWS)*len(STABILITY_THRESHES)*len(SPIKE_MINS)} "
          f"detection combos × {len(PT_THRESHES)*len(SL_THRESHES)} exit combos ...", file=out, flush=True)

    sweep_results: list[dict] = []

    for sw in STABILITY_WINDOWS:
        for st in STABILITY_THRESHES:
            for sm in SPIKE_MINS:
                # Aggregate trades per (pt, sl) across all markets
                combo_trades: dict[tuple, list] = {k: [] for k in _PT_SL_COMBOS}
                for m in markets:
                    candles = candle_cache[m["ticker"]]
                    if len(candles) < sw + 2:
                        continue
                    per_market = simulate_market_multi(
                        candles, m,
                        stability_window=sw,
                        stability_thresh=st,
                        spike_min=sm,
                        direction="momentum",
                    )
                    for combo, trades in per_market.items():
                        combo_trades[combo].extend(trades)

                for (pt, sl), all_trades in combo_trades.items():
                    if not all_trades:
                        continue
                    pnls = [t["pnl_cents"] for t in all_trades]
                    n_pt  = sum(1 for t in all_trades if t["exit_reason"] == "profit_take")
                    n_sl  = sum(1 for t in all_trades if t["exit_reason"] == "stop_loss")
                    sweep_results.append({
                        "sw": sw, "st": st, "sm": sm, "pt": pt, "sl": sl,
                        "n":         len(all_trades),
                        "n_pt":      n_pt,
                        "n_sl":      n_sl,
                        "n_end":     len(all_trades) - n_pt - n_sl,
                        "avg_pnl":   mean(pnls),
                        "total_pnl": sum(pnls),
                    })

    sweep_results.sort(key=lambda x: x["avg_pnl"], reverse=True)

    hdr = (f"  {'win':>3} {'st':>2} {'sm':>3} {'PT':>5} {'SL':>5} "
           f"{'N':>5} {'PT%':>6} {'SL%':>6} {'AvgP&L':>8} {'TotalP&L':>10}")
    print(hdr, file=out)
    print("  " + "-" * (len(hdr) - 2), file=out)

    for r in sweep_results[:TOP_N]:
        print(
            f"  {r['sw']:>3} {r['st']:>2} {r['sm']:>3} "
            f"{r['pt']:>5.0%} {r['sl']:>5.0%} "
            f"{r['n']:>5} "
            f"{_pct(r['n_pt'], r['n']):>6} "
            f"{_pct(r['n_sl'], r['n']):>6} "
            f"{r['avg_pnl']:>+7.1f}¢ "
            f"{r['total_pnl']:>+9.0f}¢",
            file=out,
        )

    print(file=out)
    if not sweep_results:
        print("  No trades generated — check that the DB has enough candle data.", file=out)
        return

    best = sweep_results[0]
    print(
        f"  Best params: stability_window={best['sw']} stability_thresh={best['st']}¢"
        f"  spike_min={best['sm']}¢  PT={best['pt']:.0%}  SL={best['sl']:.0%}",
        file=out,
    )
    print(
        f"  → {best['n']} trades, PT rate={_pct(best['n_pt'], best['n'])},"
        f" SL rate={_pct(best['n_sl'], best['n'])}, avg P&L={best['avg_pnl']:+.1f}¢",
        file=out,
    )
    print(file=out)

    # -----------------------------------------------------------------
    # Section 2 — Spike characterisation (at best detection params)
    # -----------------------------------------------------------------
    print("--- Section 2: Spike Characterisation (best detection params) ---", file=out)

    # Use best (sw, st, sm) with a mid PT/SL to get representative spike set
    best_pt_sl = (best["pt"], best["sl"])
    all_spikes: list[dict] = []
    for m in markets:
        candles = candle_cache[m["ticker"]]
        if len(candles) < best["sw"] + 2:
            continue
        per_market = simulate_market_multi(
            candles, m,
            stability_window=best["sw"],
            stability_thresh=best["st"],
            spike_min=best["sm"],
            direction="momentum",
        )
        all_spikes.extend(per_market.get(best_pt_sl, []))

    # Spike size distribution
    size_buckets: dict[str, list[float]] = defaultdict(list)
    for t in all_spikes:
        size_buckets[_bucket_spike_size(t["spike_size"])].append(t["pnl_cents"])

    print("  Spike size distribution:", file=out)
    for bucket in ["3-4¢", "5-7¢", "8-11¢", "≥12¢"]:
        pnls = size_buckets.get(bucket, [])
        if not pnls:
            continue
        print(
            f"    {bucket:<8}  N={len(pnls):>4}  "
            f"avg_pnl={mean(pnls):+.1f}¢",
            file=out,
        )
    print(file=out)

    # Hold duration distribution
    hold_buckets: dict[str, list[float]] = defaultdict(list)
    for t in all_spikes:
        hc = t["hold_candles"]
        if hc <= 2:    k = "1-2 min"
        elif hc <= 5:  k = "3-5 min"
        elif hc <= 15: k = "6-15 min"
        elif hc <= 60: k = "16-60 min"
        else:          k = ">60 min"
        hold_buckets[k].append(t["pnl_cents"])

    print("  Hold duration until exit:", file=out)
    for bucket in ["1-2 min", "3-5 min", "6-15 min", "16-60 min", ">60 min"]:
        pnls = hold_buckets.get(bucket, [])
        if not pnls:
            continue
        print(
            f"    {bucket:<12}  N={len(pnls):>4}  "
            f"avg_pnl={mean(pnls):+.1f}¢",
            file=out,
        )
    print(file=out)

    # -----------------------------------------------------------------
    # Section 3 — Momentum vs fade
    # -----------------------------------------------------------------
    print("--- Section 3: Momentum vs Fade (best params) ---", file=out)

    for dirn in ("momentum", "fade"):
        dir_trades: list[dict] = []
        for m in markets:
            candles = candle_cache[m["ticker"]]
            if len(candles) < best["sw"] + 2:
                continue
            per_market = simulate_market_multi(
                candles, m,
                stability_window=best["sw"],
                stability_thresh=best["st"],
                spike_min=best["sm"],
                direction=dirn,
            )
            dir_trades.extend(per_market.get(best_pt_sl, []))

        if not dir_trades:
            print(f"  {dirn}: no trades", file=out)
            continue

        pnls = [t["pnl_cents"] for t in dir_trades]
        n_pt = sum(1 for t in dir_trades if t["exit_reason"] == "profit_take")
        n_sl = sum(1 for t in dir_trades if t["exit_reason"] == "stop_loss")
        print(
            f"  {dirn:<10}  N={len(dir_trades):>4}  "
            f"PT={_pct(n_pt, len(dir_trades)):>6}  "
            f"SL={_pct(n_sl, len(dir_trades)):>6}  "
            f"avg={mean(pnls):+.1f}¢  total={sum(pnls):+.0f}¢",
            file=out,
        )
    print(file=out)

    # -----------------------------------------------------------------
    # Section 4 — Time-of-day (UTC hour)
    # -----------------------------------------------------------------
    print("--- Section 4: Time-of-Day Analysis (UTC hour, momentum) ---", file=out)

    hour_trades: dict[int, list[float]] = defaultdict(list)
    for t in all_spikes:
        hour_trades[_utc_hour(t["spike_ts"])].append(t["pnl_cents"])

    hdr4 = f"  {'Hour':>5}  {'N':>4}  {'AvgP&L':>8}  {'TotalP&L':>10}"
    print(hdr4, file=out)
    print("  " + "-" * (len(hdr4) - 2), file=out)
    for hour in sorted(hour_trades):
        pnls = hour_trades[hour]
        print(
            f"  {hour:>4}h  {len(pnls):>4}  "
            f"{mean(pnls):>+7.1f}¢  {sum(pnls):>+9.0f}¢",
            file=out,
        )
    print(file=out)

    # -----------------------------------------------------------------
    # Section 5 — By market type
    # -----------------------------------------------------------------
    print("--- Section 5: By Market Type ---", file=out)

    prefix_trades: dict[str, list[dict]] = defaultdict(list)
    for t in all_spikes:
        prefix_trades[_series_prefix(t["series"])].append(t)

    hdr5 = (f"  {'Type':<26}  {'N_mkt':>5}  {'N_spikes':>8}  "
            f"{'spk/mkt':>7}  {'PT%':>6}  {'AvgP&L':>8}")
    print(hdr5, file=out)
    print("  " + "-" * (len(hdr5) - 2), file=out)

    prefix_mkt_count: dict[str, int] = defaultdict(int)
    for m in markets:
        prefix_mkt_count[_series_prefix(m["series"])] += 1

    for prefix in sorted(prefix_trades, key=lambda p: -len(prefix_trades[p])):
        ts = prefix_trades[prefix]
        pnls = [t["pnl_cents"] for t in ts]
        n_pt  = sum(1 for t in ts if t["exit_reason"] == "profit_take")
        n_mkt = prefix_mkt_count.get(prefix, 0)
        spk_per_mkt = len(ts) / n_mkt if n_mkt else 0
        print(
            f"  {prefix:<26}  {n_mkt:>5}  {len(ts):>8}  "
            f"{spk_per_mkt:>7.1f}  {_pct(n_pt, len(ts)):>6}  "
            f"{mean(pnls):>+7.1f}¢",
            file=out,
        )
    print(file=out)

    # -----------------------------------------------------------------
    # Section 6 — Caveats
    # -----------------------------------------------------------------
    print("=" * 72, file=out)
    print("  Caveats", file=out)
    print("=" * 72, file=out)
    print(
        "  1. Entry is simulated at ask_close of the spike candle. In live trading\n"
        "     we detect the spike at our next poll (≤60 s), so the real entry may\n"
        "     be up to 1 minute later — typically at a less favourable price.\n"
        "\n"
        "  2. No slippage or market-impact modelling. Assumes fill at ask_close;\n"
        "     thin markets may have no fills at that price.\n"
        "\n"
        "  3. The parameter sweep is entirely in-sample. A parameter combination\n"
        "     that looks best here may not generalise — treat as direction only.\n"
        "\n"
        "  4. We cannot identify *why* each spike happened. The behavioural\n"
        "     hypothesis (retail users reacting to data) is untested here; we\n"
        "     only measure whether spikes have exploitable momentum.\n"
        "\n"
        "  5. Settlement-end exits are approximated (99¢ win / 1¢ loss). Actual\n"
        "     settlement prices depend on the final market result.\n"
        "\n"
        "  6. Non-overlap is enforced per (pt, sl) combo independently. A spike\n"
        "     that exits quickly for one combo may overlap with the next spike\n"
        "     for a combo with a longer hold time.",
        file=out,
    )
    print(file=out)
    print("Backtest complete.", file=out)
    con.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Momentum/spike backtest on Kalshi candlestick data."
    )
    parser.add_argument(
        "--db", default=str(DB_PATH),
        help="Path to candlesticks.db (default: data/candlesticks.db)",
    )
    parser.add_argument(
        "--series", nargs="+", default=None, metavar="SERIES",
        help="Filter to specific series (default: all in DB)",
    )
    parser.add_argument(
        "--from-date", default=None, metavar="YYYY-MM-DD",
        help="Only include markets closing on or after this date",
    )
    parser.add_argument(
        "--out", default=None, metavar="FILE",
        help="Write output to FILE in addition to stdout",
    )
    args = parser.parse_args()

    if args.out:
        with open(args.out, "w") as f:
            run(args, f)
        with open(args.out) as f:
            sys.stdout.write(f.read())
    else:
        run(args, sys.stdout)


if __name__ == "__main__":
    main()
