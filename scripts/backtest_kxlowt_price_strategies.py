#!/usr/bin/env python3
"""Two backtests on KXLOWT band_arb NO price trajectory patterns.

PART 1 — Early Exit
====================
For each existing band_arb NO trade, check price_snapshots at hour 1/2/3.
If YES bid has risen X¢ above entry, exit early instead of holding.
Shows whether cutting losses early beats holding to settlement.

PART 2 — Late Confirmation Entry
=================================
Scan ALL KXLOWT candles for a "confirmed decline" pattern:
  - YES bid has fallen for N consecutive hours
  - YES bid is now at or below a threshold (e.g. ≤12¢)
  - Market still has ≥8h to close
Enter NO at that candle, simulate PT or settlement.
Measures whether the late-day downtrend is a reliable entry.

Usage:
  venv/bin/python scripts/backtest_kxlowt_price_strategies.py
"""

from __future__ import annotations

import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

FORECAST_DB = Path("data/db/opportunity_log.db")
CANDLES_DB  = Path("data/candlesticks.db")

_BAND_RE = re.compile(r"-B([\d.]+)$")


# ============================================================
# PART 1 — Early Exit Backtest
# ============================================================

def backtest_early_exit() -> None:
    print("=" * 60)
    print("PART 1: Early Exit Backtest")
    print("=" * 60)

    con = sqlite3.connect(FORECAST_DB)

    # Load all settled KXLOWT NO band_arb trades with snapshots
    trades = con.execute("""
        SELECT t.id, t.ticker, t.limit_price,
               COALESCE(t.exit_pnl_cents, t.settled_pnl_cents) AS baseline_pnl,
               t.settled_result
        FROM trades t
        WHERE t.source = 'band_arb'
          AND t.side = 'no'
          AND t.ticker LIKE 'KXLOWT%B%'
          AND COALESCE(t.exit_pnl_cents, t.settled_pnl_cents) IS NOT NULL
    """).fetchall()

    if not trades:
        print("No trades found.")
        return

    # Load snapshots keyed by trade_id
    snap_rows = con.execute("""
        SELECT ps.trade_id,
               ROUND((julianday(ps.snapshot_at) - julianday(t.logged_at)) * 24) AS hour,
               ps.yes_bid,
               ps.yes_ask,
               ps.unrealized_cents
        FROM price_snapshots ps
        JOIN trades t ON t.id = ps.trade_id
        WHERE t.source = 'band_arb'
          AND t.side = 'no'
          AND t.ticker LIKE 'KXLOWT%B%'
    """).fetchall()

    # Group snapshots by trade_id → {hour: [yes_bid, ...]}
    snaps: dict[int, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for trade_id, hour, yes_bid, yes_ask, unreal in snap_rows:
        if hour is not None and 0 <= hour <= 24:
            snaps[trade_id][int(hour)].append((yes_bid, yes_ask, unreal))

    baseline_pnl  = sum(t[3] for t in trades)
    baseline_wins = sum(1 for t in trades if t[3] > 0)
    n = len(trades)

    print(f"\nBaseline — {n} trades, {baseline_wins}W/{n-baseline_wins}L, "
          f"total P&L: {baseline_pnl/100:+.2f} USD\n")

    print(f"{'check_h':>8} {'threshold':>10} {'n_exited':>9} {'exit_pnl':>10} "
          f"{'hold_pnl':>10} {'total':>10} {'vs_base':>9} {'exit_wins':>10}")
    print("-" * 85)

    best = None
    for check_hour in [1, 2, 3]:
        for threshold in [2, 4, 6, 8, 10, 15]:
            exit_pnl  = 0.0
            hold_pnl  = 0.0
            n_exited  = 0
            exit_wins = 0

            for trade_id, ticker, limit_price, bl_pnl, result in trades:
                hour_snaps = snaps.get(trade_id, {}).get(check_hour, [])
                if not hour_snaps:
                    hold_pnl += bl_pnl
                    continue

                avg_yes_bid = sum(s[0] for s in hour_snaps if s[0] is not None) / len(hour_snaps)

                if avg_yes_bid > limit_price + threshold:
                    # Early exit: sell NO at current NO bid ≈ 100 - avg_yes_ask
                    yes_asks = [s[1] for s in hour_snaps if s[1] is not None]
                    avg_yes_ask = sum(yes_asks) / len(yes_asks) if yes_asks else avg_yes_bid + 2
                    no_bid_now = 100 - avg_yes_ask
                    # Entry cost = 100 - yes_ask_entry ≈ 100 - (limit_price + spread)
                    # Approximate: we paid 100 - (limit_price + 2) for NO
                    entry_cost = 100 - (limit_price + 2)
                    pnl = no_bid_now - entry_cost

                    # Scale to match baseline (which includes contract sizing)
                    # Use unrealized_cents if available for accurate scaling
                    unreals = [s[2] for s in hour_snaps if s[2] is not None]
                    if unreals:
                        pnl_scaled = sum(unreals) / len(unreals)
                    else:
                        pnl_scaled = pnl  # fallback

                    exit_pnl += pnl_scaled
                    n_exited += 1
                    if pnl_scaled > 0:
                        exit_wins += 1
                else:
                    hold_pnl += bl_pnl

            total = exit_pnl + hold_pnl
            vs_base = total - baseline_pnl

            if best is None or total > best[0]:
                best = (total, check_hour, threshold)

            print(f"{check_hour:>8}h {threshold:>+9}¢ {n_exited:>9} "
                  f"{exit_pnl/100:>+10.2f} {hold_pnl/100:>+10.2f} "
                  f"{total/100:>+10.2f} {vs_base/100:>+9.2f} "
                  f"{exit_wins:>6}/{n_exited}")

    if best:
        print(f"\n→ Best: check at hour {best[1]}, exit if YES bid >"
              f" entry+{best[2]}¢  (total {best[0]/100:+.2f} USD)")

    # Show what the exited trades looked like at baseline
    print(f"\n--- Breakdown: what did early-exit candidates eventually do? ---")
    for check_hour in [1, 2]:
        for threshold in [4, 8]:
            exited_results = []
            for trade_id, ticker, limit_price, bl_pnl, result in trades:
                hour_snaps = snaps.get(trade_id, {}).get(check_hour, [])
                if not hour_snaps:
                    continue
                avg_yes_bid = sum(s[0] for s in hour_snaps if s[0] is not None) / len(hour_snaps)
                if avg_yes_bid > limit_price + threshold:
                    exited_results.append((bl_pnl, result))
            if exited_results:
                wins = sum(1 for p, r in exited_results if p > 0)
                avg = sum(p for p, r in exited_results) / len(exited_results)
                print(f"  Hour {check_hour}, +{threshold}¢ threshold → "
                      f"{len(exited_results)} trades, {wins}W/{len(exited_results)-wins}L "
                      f"if held (avg P&L {avg:.0f}¢) — "
                      f"{'WORTH exiting' if wins < len(exited_results)*0.5 else 'NOT worth exiting'}")


# ============================================================
# PART 2 — Late Confirmation Entry Backtest
# ============================================================

def backtest_late_entry() -> None:
    print("\n" + "=" * 60)
    print("PART 2: Late Confirmation Entry Backtest")
    print("=" * 60)

    if not CANDLES_DB.exists():
        print(f"ERROR: {CANDLES_DB} not found")
        return

    con = sqlite3.connect(CANDLES_DB)

    markets = con.execute("""
        SELECT ticker, open_ts, close_ts, result
        FROM markets
        WHERE ticker LIKE 'KXLOWT%B%'
          AND result IN ('yes', 'no')
    """).fetchall()

    print(f"\n{len(markets)} settled KXLOWT B-band markets")

    # Load candles into memory per ticker
    all_candles: dict[str, list] = {}
    for ticker, open_ts, close_ts, result in markets:
        rows = con.execute("""
            SELECT period_ts, bid_close, ask_close
            FROM candles
            WHERE ticker = ?
            ORDER BY period_ts
        """, (ticker,)).fetchall()
        if rows:
            all_candles[ticker] = rows

    # ---------------------------------------------------------------
    # Corrected entry cost:
    #   Buying NO = paying the NO ask = 100 - YES bid (bid_close)
    #   NOT 100 - YES ask, which would be the NO bid (passive fill).
    # Spread filter: only enter when bid/ask spread is ≤ MAX_SPREAD
    # so entry cost is realistic and not distorted by wide markets.
    #
    # Also expand YES bid range: at ≤12¢ the risk/reward is terrible
    # (pay ~88¢ for NO, max gain 12¢). Need ≥20¢ YES bid for decent EV.
    # ---------------------------------------------------------------
    MAX_YES_BIDS   = [20, 25, 30, 35, 40]  # YES bid at entry (NO cost = 100-bid)
    MIN_YES_BIDS   = [5, 8, 10]             # also require YES bid > this (min decline)
    MIN_HOURS_LEFT = [4, 6, 8]
    DECLINE_HOURS  = [2, 3, 4]
    PT_YES_TARGETS = [3, 5, 8]             # PT when YES bid falls to ≤ this
    MAX_SPREAD     = 10                     # skip entries with bid-ask > this

    print(f"\nEntry cost = 100 - YES bid (corrected).  Max spread filter = {MAX_SPREAD}¢.")
    print(f"\n{'max_bid':>8} {'min_bid':>8} {'min_h':>7} {'dec_h':>7} {'pt':>5} "
          f"{'n':>6} {'win%':>7} {'avg¢':>8} {'total$':>9} {'EV_check':>10}")
    print("-" * 80)

    best_total = None
    results = []

    for max_yes_bid in MAX_YES_BIDS:
        for min_yes_bid in MIN_YES_BIDS:
            if min_yes_bid >= max_yes_bid:
                continue
            for min_hours_left in MIN_HOURS_LEFT:
                for decline_hours in DECLINE_HOURS:
                    for pt_yes_target in PT_YES_TARGETS:
                        trades_sim = []

                        for ticker, open_ts, close_ts, result in markets:
                            candles = all_candles.get(ticker, [])
                            if len(candles) < decline_hours + 2:
                                continue
                            if not _BAND_RE.search(ticker):
                                continue

                            for i in range(decline_hours, len(candles)):
                                ts, bid, ask = candles[i]
                                if bid is None or ask is None:
                                    continue

                                hours_to_close = (close_ts - ts) / 3600
                                if hours_to_close < min_hours_left:
                                    break

                                # Spread filter
                                if (ask - bid) > MAX_SPREAD:
                                    continue

                                # Entry condition
                                if bid > max_yes_bid or bid <= min_yes_bid:
                                    continue

                                # Confirmed decline: each earlier candle's bid ≥ next
                                prev_bids = [candles[i-j][1] for j in range(1, decline_hours+1)
                                             if candles[i-j][1] is not None]
                                if len(prev_bids) < decline_hours:
                                    continue
                                # prev_bids[0]=1h ago, prev_bids[1]=2h ago, ...
                                # Declining means: 2h ago ≥ 1h ago ≥ now
                                if not all(prev_bids[j] >= prev_bids[j-1]
                                           for j in range(1, len(prev_bids))):
                                    continue
                                # Require minimum total drop over the window
                                if prev_bids[-1] - bid < 3:  # must have dropped ≥3¢
                                    continue

                                # Correct entry cost: pay NO ask = 100 - YES bid
                                no_cost = 100 - bid
                                if no_cost <= 0 or no_cost >= 100:
                                    continue

                                # Simulate forward
                                pnl = None
                                exit_how = "settle"
                                for j in range(i+1, len(candles)):
                                    nts, nbid, nask = candles[j]
                                    if nbid is None:
                                        continue
                                    if nbid <= pt_yes_target:
                                        # Sell NO at NO bid = 100 - YES ask
                                        nask_use = nask if nask else nbid + 2
                                        exit_val = 100 - nask_use
                                        pnl = exit_val - no_cost
                                        exit_how = "PT"
                                        break

                                if pnl is None:
                                    settle_val = 100 if result == "no" else 0
                                    pnl = settle_val - no_cost

                                trades_sim.append((ticker, bid, no_cost, pnl,
                                                   result, exit_how))
                                break  # one entry per market

                        n = len(trades_sim)
                        if n < 10:
                            continue

                        wins = sum(1 for *_, p, r, h in trades_sim if p > 0)
                        avg_pnl = sum(p for *_, p, r, h in trades_sim) / n
                        total = avg_pnl * n / 100

                        # Theoretical break-even win rate check
                        avg_no_cost = sum(nc for _, _, nc, *_ in trades_sim) / n
                        avg_win_pnl = (sum(p for *_, p, r, h in trades_sim if p > 0)
                                       / max(1, wins))
                        avg_loss_pnl = (sum(p for *_, p, r, h in trades_sim if p <= 0)
                                        / max(1, n - wins))
                        be_wr = abs(avg_loss_pnl) / (avg_win_pnl + abs(avg_loss_pnl)) if wins > 0 else 1.0
                        ev_check = "✓" if (wins/n) > be_wr else "✗"

                        results.append((total, max_yes_bid, min_yes_bid,
                                        min_hours_left, decline_hours,
                                        pt_yes_target, n, wins, avg_pnl, be_wr, ev_check))

                        if best_total is None or total > best_total:
                            best_total = total

    results.sort(reverse=True)
    for row in results[:25]:
        total, mb, mnb, mh, dh, pt, n, wins, avg, be, ev = row
        print(f"{mb:>8} {mnb:>8} {mh:>7} {dh:>7} {pt:>5} "
              f"{n:>6} {wins/n:>6.0%} {avg:>+8.1f} {total:>+9.2f} "
              f"  be={be:.0%} {ev}")

    if results:
        total, mb, mnb, mh, dh, pt, n, wins, avg, be, ev = results[0]
        print(f"\n→ Best params: YES bid {mnb}–{mb}¢ (NO cost {100-mb}–{100-mnb}¢), "
              f"≥{mh}h left, {dh}h decline ≥3¢, PT when YES≤{pt}¢")
        print(f"  {n} trades, {wins}W/{n-wins}L ({wins/n:.0%}), "
              f"avg {avg:+.1f}¢, total ${total:+.2f}  break-even win rate: {be:.0%}")

        # Detail for best params
        detail = []
        for ticker, open_ts, close_ts, result in markets:
            candles = all_candles.get(ticker, [])
            if len(candles) < dh + 2 or not _BAND_RE.search(ticker):
                continue
            for i in range(dh, len(candles)):
                ts, bid, ask = candles[i]
                if bid is None or ask is None:
                    continue
                if (close_ts - ts) / 3600 < mh:
                    break
                if (ask - bid) > MAX_SPREAD:
                    continue
                if bid > mb or bid <= mnb:
                    continue
                prev_bids = [candles[i-j][1] for j in range(1, dh+1)
                             if candles[i-j][1] is not None]
                if len(prev_bids) < dh:
                    continue
                if not all(prev_bids[j] >= prev_bids[j-1] for j in range(1, len(prev_bids))):
                    continue
                if prev_bids[-1] - bid < 3:
                    continue
                no_cost = 100 - bid
                pnl, how = None, "settle"
                for j in range(i+1, len(candles)):
                    nts, nbid, nask = candles[j]
                    if nbid is None:
                        continue
                    if nbid <= pt:
                        exit_val = 100 - (nask if nask else nbid + 2)
                        pnl = exit_val - no_cost
                        how = "PT"
                        break
                if pnl is None:
                    pnl = (100 if result == "no" else 0) - no_cost
                detail.append((ticker, bid, no_cost, pnl, result, how))
                break

        print(f"\n{'ticker':45} {'yes_bid':>8} {'no_cost':>8} "
              f"{'pnl':>7} {'result':>7} {'how':>8}")
        print("-" * 90)
        for d in sorted(detail, key=lambda x: x[3])[:5]:
            t, yb, nc, p, r, how = d
            print(f"{t:45} {yb:>8} {nc:>8} {p:>+7.1f} {r:>7} {how:>8}")
        print("  …")
        for d in sorted(detail, key=lambda x: x[3])[-5:]:
            t, yb, nc, p, r, how = d
            print(f"{t:45} {yb:>8} {nc:>8} {p:>+7.1f} {r:>7} {how:>8}")


# ============================================================

def main() -> None:
    if not FORECAST_DB.exists():
        print(f"ERROR: {FORECAST_DB} not found")
        sys.exit(1)

    backtest_early_exit()
    backtest_late_entry()


if __name__ == "__main__":
    main()
