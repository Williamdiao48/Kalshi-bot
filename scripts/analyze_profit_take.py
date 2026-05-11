#!/usr/bin/env python3
"""PT × SL grid search using full intraday candlestick data.

For each resolved trade, fetches candles from ENTRY to MARKET CLOSE (not just
to the actual exit), so stop-loss and profit-take exits can be re-simulated at
any threshold after the fact.

Settlement P&L is reconstructed from the market outcome, so the simulation
correctly handles trades that were exited early.

The 2D grid output sweeps every (PT threshold, SL threshold) combination and
shows total simulated P&L — the empirical optimum rather than single-variable
slices.

Usage:
    venv/bin/python scripts/analyze_profit_take.py
    venv/bin/python scripts/analyze_profit_take.py --kind forecast_no
    venv/bin/python scripts/analyze_profit_take.py --kind band_arb
    venv/bin/python scripts/analyze_profit_take.py --ids 37 40 83
"""

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.lib import DEFAULT_DB_PATH, DEFAULT_CANDLESTICK_DB_PATH, parse_iso_ts

# Grid axes — None means "disabled" (no PT or no SL)
PT_GRID = [None, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
SL_GRID = [None, 0.20, 0.30, 0.40, 0.50, 0.70]


def _pct_gain_series(
    candles: list[tuple[int, int | None, int | None]],
    side: str,
    entry_cost: int,
    entry_bid: int,
    entry_ask: int,
) -> list[tuple[int, float]]:
    """(period_ts, pct_gain) for every candle with a non-NULL bid_close.

    NO: gain = yes_bid_entry − yes_bid_now  (falling YES benefits NO holder)
    YES: gain = yes_bid_now  − yes_ask_entry (rising YES benefits YES holder)
    """
    result = []
    for ts, _ask_open, bid_close in candles:
        if bid_close is None:
            continue
        gain = (entry_bid - bid_close) if side == "no" else (bid_close - entry_ask)
        result.append((ts, gain / entry_cost if entry_cost > 0 else 0.0))
    return result


def _simulate(
    series: list[tuple[int, float]],
    pt: float | None,
    sl: float | None,
    settlement_pnl: float,
    entry_cost: int,
    count: int,
) -> float:
    """Simulate one trade under given PT and SL thresholds.

    Walks the full-lifetime series; first trigger wins.
    Falls through to settlement_pnl if neither fires.
    """
    for _ts, pct in series:
        if pt is not None and pct >= pt:
            return pt * entry_cost * count
        if sl is not None and pct <= -sl:
            return -sl * entry_cost * count
    return settlement_pnl


def _label(v: float | None) -> str:
    return "none" if v is None else f"{int(v * 100)}%"


def analyze(
    opp_conn: sqlite3.Connection,
    candle_conn: sqlite3.Connection,
    kind: str | None = None,
    trade_ids: list[int] | None = None,
    pt_grid: list[float | None] | None = None,
    sl_grid: list[float | None] | None = None,
) -> None:
    pt_grid = pt_grid if pt_grid is not None else PT_GRID
    sl_grid = sl_grid if sl_grid is not None else SL_GRID

    query = """
        SELECT id, ticker, opportunity_kind, side, yes_bid_entry, yes_ask_entry,
               limit_price, count, logged_at, exited_at, exit_reason, exit_pnl_cents, outcome
        FROM trades
        WHERE outcome IS NOT NULL
    """
    params: list = []
    if kind:
        query += " AND opportunity_kind = ?"
        params.append(kind)
    if trade_ids:
        query += f" AND id IN ({','.join('?' * len(trade_ids))})"
        params.extend(trade_ids)
    query += " ORDER BY id"

    trades = opp_conn.execute(query, params).fetchall()
    print(f"Analyzing {len(trades)} resolved trade(s)…\n")

    results = []
    no_candles = []

    for (trade_id, ticker, opp_kind, side, yes_bid_entry, yes_ask_entry,
         limit_price, count, logged_at, exited_at, exit_reason, exit_pnl_cents, outcome) in trades:

        entry_ts = parse_iso_ts(logged_at) if logged_at else 0

        # Always use market close_ts as the candle window end, so we see the
        # full price path regardless of when (or whether) we actually exited.
        mkt_row = candle_conn.execute(
            "SELECT close_ts FROM markets WHERE ticker = ?", (ticker,)
        ).fetchone()
        close_ts = int(mkt_row[0]) if mkt_row and mkt_row[0] else 0

        if not entry_ts or not close_ts:
            no_candles.append((trade_id, ticker, "no timestamps"))
            continue

        candles = candle_conn.execute(
            "SELECT period_ts, ask_open, bid_close FROM candles "
            "WHERE ticker = ? AND period_ts BETWEEN ? AND ? ORDER BY period_ts",
            (ticker, entry_ts, close_ts),
        ).fetchall()

        if not candles:
            no_candles.append((trade_id, ticker, "no candles in DB"))
            continue

        entry_bid  = yes_bid_entry or limit_price or 50
        entry_ask  = yes_ask_entry or limit_price or 50
        entry_cost = (100 - entry_bid) if side == "no" else entry_ask
        cnt        = count or 1
        lim        = limit_price or 50

        # Settlement P&L — what the market paid out at close, regardless of
        # our actual exit. Used as the fall-through in every simulation.
        if outcome == "won":
            settlement_pnl = float(((100 - lim) * cnt) if side == "yes" else (lim * cnt))
        elif outcome == "lost":
            settlement_pnl = float(((-lim) * cnt) if side == "yes" else (-(100 - lim) * cnt))
        else:
            settlement_pnl = 0.0

        # Actual P&L — what the bot recorded (may differ from settlement if
        # we exited early via profit_take or stop_loss).
        if exit_pnl_cents is not None:
            actual_pnl = float(exit_pnl_cents)
        else:
            actual_pnl = settlement_pnl

        series = _pct_gain_series(candles, side, entry_cost, entry_bid, entry_ask)
        if not series:
            no_candles.append((trade_id, ticker, "no bid_close values"))
            continue

        peak_pct   = max(pct for _, pct in series)
        trough_pct = min(pct for _, pct in series)

        # Find the candle index nearest to the actual bot exit (for per-trade display).
        actual_exit_ts = parse_iso_ts(exited_at) if exited_at else close_ts
        actual_exit_idx = len(series) - 1
        for i, (ts, _) in enumerate(series):
            if ts >= actual_exit_ts:
                actual_exit_idx = i
                break
        pct_at_exit = series[actual_exit_idx][1] if series else 0.0

        # Simulate all (PT, SL) combinations over the full-lifetime series.
        grid: dict[tuple, float] = {}
        for pt in pt_grid:
            for sl in sl_grid:
                grid[(pt, sl)] = _simulate(series, pt, sl, settlement_pnl, entry_cost, cnt)

        results.append({
            "id":             trade_id,
            "ticker":         ticker,
            "kind":           opp_kind,
            "side":           side,
            "exit_reason":    exit_reason or "settlement",
            "outcome":        outcome,
            "actual_pnl":     actual_pnl,
            "settlement_pnl": settlement_pnl,
            "entry_cost":     entry_cost,
            "count":          cnt,
            "peak_pct":       peak_pct,
            "trough_pct":     trough_pct,
            "pct_at_exit":    pct_at_exit,
            "grid":           grid,
        })

    # ── Per-trade detail ─────────────────────────────────────────────────────
    print("═" * 90)
    print("  PER-TRADE DETAIL  (full-lifetime candles; peak/trough over entire market window)")
    print("═" * 90)
    print(f"  {'#':<5}  {'ticker':<36}  {'exit':<18}  {'actual':>8}  {'settle':>8}  {'peak%':>7}  {'trough%':>8}  {'pct@exit':>9}")
    print("  " + "─" * 88)

    for r in results:
        print(
            f"  #{r['id']:<4d}  {r['ticker']:<36}  {r['exit_reason']:<18}"
            f"  {r['actual_pnl']/100:>+8.2f}  {r['settlement_pnl']/100:>+8.2f}"
            f"  {r['peak_pct']*100:>+6.1f}%  {r['trough_pct']*100:>+7.1f}%  {r['pct_at_exit']*100:>+8.1f}%"
        )

    if no_candles:
        print(f"\n  No candle data for {len(no_candles)} trade(s):")
        for tid, tkr, reason in no_candles:
            print(f"    #{tid}  {tkr}  ({reason})")

    if not results:
        return

    # ── 2D grid: PT × SL ────────────────────────────────────────────────────
    print(f"\n{'═' * 90}")
    print("  PT × SL GRID  (total simulated P&L in $, all trades with candle data)")
    print(f"  Actual total: ${sum(r['actual_pnl'] for r in results)/100:+.2f}   "
          f"Settlement total: ${sum(r['settlement_pnl'] for r in results)/100:+.2f}")
    print(f"{'═' * 90}")

    col_w = 8
    pt_labels = [_label(pt) for pt in pt_grid]
    hdr = f"  {'SL \\ PT':<10}" + "".join(f"{lbl:>{col_w}}" for lbl in pt_labels)
    print(hdr)
    print("  " + "─" * (10 + col_w * len(pt_grid)))

    best_total = max(
        sum(r["grid"][(pt, sl)] for r in results)
        for pt in pt_grid for sl in sl_grid
    )

    for sl in sl_grid:
        row_label = _label(sl)
        cells = []
        for pt in pt_grid:
            total = sum(r["grid"][(pt, sl)] for r in results) / 100.0
            marker = "*" if abs(sum(r["grid"][(pt, sl)] for r in results) - best_total) < 0.01 else " "
            cells.append(f"{total:>+{col_w-1}.2f}{marker}")
        print(f"  {row_label:<10}" + "".join(cells))

    print()
    print("  * = optimal combination")

    # ── Peak gain distribution ────────────────────────────────────────────────
    print(f"\n{'═' * 90}")
    print("  PEAK GAIN DISTRIBUTION  (full market window)")
    print(f"{'═' * 90}")
    buckets = [(-999, 0, "<0%"), (0, 10, "0–10%"), (10, 20, "10–20%"),
               (20, 30, "20–30%"), (30, 50, "30–50%"), (50, 100, "50–100%"), (100, 9999, ">100%")]
    for lo, hi, label in buckets:
        group = [r for r in results if lo <= r["peak_pct"] * 100 < hi]
        if not group:
            continue
        wins = sum(1 for r in group if r["outcome"] == "won")
        sl_  = sum(1 for r in group if r["exit_reason"] == "stop_loss")
        print(f"  {label:>10}  {'█' * len(group):<20}  {len(group):3d} trades  "
              f"win={wins}/{len(group)}  sl={sl_}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", default=None,
                        help="Filter by opportunity_kind (e.g. forecast_no, band_arb)")
    parser.add_argument("--ids", type=int, nargs="+", metavar="ID")
    parser.add_argument("--pt",  type=float, nargs="+", metavar="T",
                        help="PT thresholds to sweep, e.g. 0.20 0.30 0.40. Use 0 for 'none'.")
    parser.add_argument("--sl",  type=float, nargs="+", metavar="T",
                        help="SL thresholds to sweep, e.g. 0.30 0.50 0.70. Use 0 for 'none'.")
    parser.add_argument("--opp-db",    default=str(DEFAULT_DB_PATH))
    parser.add_argument("--candle-db", default=str(DEFAULT_CANDLESTICK_DB_PATH))
    args = parser.parse_args()

    def _parse_grid(vals: list[float] | None, default: list) -> list[float | None]:
        if not vals:
            return default
        return [None if v == 0 else v for v in vals]

    opp_conn    = sqlite3.connect(args.opp_db,    isolation_level=None)
    candle_conn = sqlite3.connect(args.candle_db, isolation_level=None)

    try:
        analyze(
            opp_conn, candle_conn,
            kind=args.kind,
            trade_ids=args.ids,
            pt_grid=_parse_grid(args.pt, PT_GRID),
            sl_grid=_parse_grid(args.sl, SL_GRID),
        )
    finally:
        opp_conn.close()
        candle_conn.close()


if __name__ == "__main__":
    main()
