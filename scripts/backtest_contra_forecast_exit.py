#!/usr/bin/env python3
"""
Backtest: do contra-forecasts predict YES bid spikes in open KXLOWT NO positions?

Hypothesis: when a forecast source (NOAA, HRRR, GFS, NWS hourly) drops to within
CONTRA_BUFFER°F of the band ceiling during an open warm-NO position, the YES bid
spikes in the following hours - and this spike precedes the eventual loss.

If confirmed, the bot could use contra-forecast events as early exit triggers,
cutting losses before METAR observations move the market.

For each KXLOWT NO band_arb trade with price snapshot coverage:
  1. Find every "contra event": first time each forecast source reports
     a value ≤ band_ceil + CONTRA_BUFFER°F after entry.
  2. Measure YES bid at the event and at fixed lags (30m, 1h, 2h, 4h, 8h).
  3. Compare contra-event responses to baseline (no contra event).

Usage:
    venv/bin/python scripts/backtest_contra_forecast_exit.py [--buffer 2.0]

Output sections:
  A. Per-source summary: avg YES bid response by forecast source
  B. Margin buckets: does closeness to ceiling matter?
  C. Timing: how many minutes after the contra event does YES spike peak?
  D. Per-trade detail: first contra event for each trade
  E. Baseline: YES bid trajectory for trades with NO contra events
"""

import argparse
import sqlite3
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone

DB_PATH = "data/db/opportunity_log.db"

FORECAST_SOURCES = (
    "noaa", "hrrr", "nws_hourly",
    "open_meteo_gfs", "open_meteo",
    "open_meteo_ecmwf", "open_meteo_gem", "open_meteo_icon",
)
LAG_MINUTES = [30, 60, 120, 240, 480]


def parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def nearest_yes_bid(snap_times: list, target: datetime, window_s: int = 600) -> int | None:
    """Return yes_bid from the snapshot nearest to target within window_s seconds."""
    best_dt, best_bid, best_gap = None, None, float("inf")
    for t, bid in snap_times:
        gap = abs((t - target).total_seconds())
        if gap < best_gap and gap <= window_s:
            best_gap, best_bid = gap, bid
    return best_bid


def max_yes_bid_in_window(snap_times: list, start: datetime, end: datetime) -> int | None:
    vals = [b for t, b in snap_times if start <= t <= end and b is not None]
    return max(vals) if vals else None


def fmt(v) -> str:
    if v is None:
        return "   -"
    return f"{v:4d}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer", type=float, default=2.0,
                        help="°F above ceiling to flag as contra (default 2.0)")
    parser.add_argument("--since", default="2026-05-01",
                        help="Only include trades from this date onwards")
    args = parser.parse_args()
    CONTRA_BUFFER = args.buffer

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    trades = conn.execute("""
        SELECT t.id, t.ticker, t.logged_at, t.exited_at, t.outcome,
               t.settled_pnl_cents,
               CAST(json_extract(t.note, '$.band_ceil_f') AS REAL) AS band_ceil,
               json_extract(t.note, '$.metric') AS metric
        FROM trades t
        WHERE t.opportunity_kind = 'band_arb'
          AND t.ticker LIKE 'KXLOWT%'
          AND t.side = 'no'
          AND t.outcome IS NOT NULL
          AND date(t.logged_at) >= ?
          AND EXISTS (SELECT 1 FROM price_snapshots s WHERE s.trade_id = t.id)
          AND json_extract(t.note, '$.band_ceil_f') IS NOT NULL
          AND json_extract(t.note, '$.metric') IS NOT NULL
        ORDER BY t.logged_at
    """, (args.since,)).fetchall()

    print(f"Loaded {len(trades)} KXLOWT NO trades with snapshots since {args.since}")
    print(f"Contra buffer: ≤ ceiling + {CONTRA_BUFFER}°F")

    all_contra = []      # rows with at least one contra event
    all_baseline = []    # rows with zero contra events
    per_trade_first = {} # trade_id -> first contra event

    for trade in trades:
        trade_id   = trade["id"]
        ticker     = trade["ticker"]
        metric     = trade["metric"]
        band_ceil  = trade["band_ceil"]
        entry_dt   = parse_dt(trade["logged_at"])
        exit_dt    = parse_dt(trade["exited_at"])
        outcome    = trade["outcome"]
        pnl        = trade["settled_pnl_cents"]
        trade_date = trade["logged_at"][:10]

        if entry_dt is None or band_ceil is None:
            continue

        # Price snapshots
        snaps = conn.execute("""
            SELECT snapshot_at, yes_bid FROM price_snapshots
            WHERE trade_id = ? AND post_exit = 0
            ORDER BY snapshot_at
        """, (trade_id,)).fetchall()

        snap_times = [
            (parse_dt(s["snapshot_at"]), s["yes_bid"])
            for s in snaps
            if parse_dt(s["snapshot_at"]) is not None and s["yes_bid"] is not None
        ]
        if not snap_times:
            continue

        # Raw forecasts for this metric from entry onwards
        forecasts = conn.execute("""
            SELECT logged_at, source, data_value
            FROM raw_forecasts
            WHERE metric = ?
              AND date(logged_at) = ?
              AND logged_at >= ?
              AND source IN ('noaa','hrrr','nws_hourly',
                             'open_meteo_gfs','open_meteo',
                             'open_meteo_ecmwf','open_meteo_gem','open_meteo_icon')
            ORDER BY logged_at
        """, (metric, trade_date, trade["logged_at"])).fetchall()

        # Find first contra event per source
        first_contra_per_src: dict[str, dict] = {}
        for fc in forecasts:
            fc_dt = parse_dt(fc["logged_at"])
            if fc_dt is None:
                continue
            if exit_dt and fc_dt > exit_dt:
                continue
            src = fc["source"]
            val = float(fc["data_value"])
            margin = val - band_ceil  # positive = above ceiling (safe for NO)

            if margin <= CONTRA_BUFFER and src not in first_contra_per_src:
                first_contra_per_src[src] = {
                    "source":  src,
                    "dt":      fc_dt,
                    "value":   val,
                    "margin":  margin,
                }

        if not first_contra_per_src:
            # Baseline trade - no contra events
            # Record YES bid at fixed points from entry
            entry_bid = nearest_yes_bid(snap_times, entry_dt)
            all_baseline.append({
                "trade_id": trade_id,
                "ticker":   ticker,
                "outcome":  outcome,
                "pnl":      pnl,
                "entry_bid": entry_bid,
                "max_yes": max_yes_bid_in_window(snap_times, entry_dt,
                                                  exit_dt or entry_dt + timedelta(hours=24)),
            })
            continue

        # For each contra event, measure YES bid response
        for src, evt in first_contra_per_src.items():
            evt_dt  = evt["dt"]
            yes_now = nearest_yes_bid(snap_times, evt_dt)
            lag_bids = {
                lag: nearest_yes_bid(snap_times, evt_dt + timedelta(minutes=lag))
                for lag in LAG_MINUTES
            }
            max_4h = max_yes_bid_in_window(snap_times, evt_dt, evt_dt + timedelta(hours=4))
            max_8h = max_yes_bid_in_window(snap_times, evt_dt, evt_dt + timedelta(hours=8))

            # Minutes from entry to this contra event
            mins_after_entry = (evt_dt - entry_dt).total_seconds() / 60

            row = {
                "trade_id":         trade_id,
                "ticker":           ticker,
                "outcome":          outcome,
                "pnl":              pnl,
                "band_ceil":        band_ceil,
                "source":           src,
                "margin":           evt["margin"],
                "value":            evt["value"],
                "evt_dt":           evt_dt,
                "mins_after_entry": mins_after_entry,
                "yes_now":          yes_now,
                "lag_bids":         lag_bids,
                "max_4h":           max_4h,
                "max_8h":           max_8h,
            }
            all_contra.append(row)

            # Track first contra event across all sources per trade
            if (trade_id not in per_trade_first
                    or evt_dt < per_trade_first[trade_id]["evt_dt"]):
                per_trade_first[trade_id] = row

    conn.close()

    n_contra_trades  = len(per_trade_first)
    n_baseline_trades = len(all_baseline)
    print(f"Trades with ≥1 contra event : {n_contra_trades}")
    print(f"Trades with zero contra events: {n_baseline_trades}")

    # ── Section A: Per-source summary ───────────────────────────────────────
    print("\n" + "=" * 72)
    print("A. YES BID RESPONSE AFTER CONTRA EVENT, BY FORECAST SOURCE")
    print(f"   (contra = forecast ≤ ceiling + {CONTRA_BUFFER}°F;  first event per source per trade)")
    print("=" * 72)
    header = f"  {'Source':<22} {'N':>3}  {'Yes@Ev':>6}  {'Y+30m':>5}  {'Y+1h':>5}  {'Y+2h':>5}  {'Max4h':>5}  {'Won%':>5}"
    print(header)
    print("  " + "-" * 68)

    by_source = defaultdict(list)
    for r in all_contra:
        by_source[r["source"]].append(r)

    def avg(vals):
        clean = [v for v in vals if v is not None]
        return statistics.mean(clean) if clean else None

    def fmtf(v, w=5):
        return f"{v:{w}.1f}" if v is not None else " " * (w - 1) + "-"

    for src in sorted(by_source):
        rows = by_source[src]
        won  = sum(1 for r in rows if r["outcome"] == "won")
        print(
            f"  {src:<22} {len(rows):>3}  "
            f"{fmtf(avg(r['yes_now'] for r in rows)):>6}  "
            f"{fmtf(avg(r['lag_bids'].get(30) for r in rows)):>5}  "
            f"{fmtf(avg(r['lag_bids'].get(60) for r in rows)):>5}  "
            f"{fmtf(avg(r['lag_bids'].get(120) for r in rows)):>5}  "
            f"{fmtf(avg(r['max_4h'] for r in rows)):>5}  "
            f"{100*won/len(rows):>5.1f}%"
        )

    # ── Section B: Margin buckets ────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("B. YES BID RESPONSE BY MARGIN (forecast - ceiling)")
    print("   Smaller margin = forecast closer to / below ceiling = stronger contra")
    print("=" * 72)
    print(header)
    print("  " + "-" * 68)

    buckets = [
        ("≤0°F (at/below)",    lambda m: m <= 0.0),
        ("0–0.5°F",            lambda m: 0.0 < m <= 0.5),
        ("0.5–1°F",            lambda m: 0.5 < m <= 1.0),
        ("1–2°F",              lambda m: 1.0 < m <= 2.0),
    ]
    for label, fn in buckets:
        rows = [r for r in all_contra if fn(r["margin"])]
        if not rows:
            continue
        won = sum(1 for r in rows if r["outcome"] == "won")
        print(
            f"  {label:<22} {len(rows):>3}  "
            f"{fmtf(avg(r['yes_now'] for r in rows)):>6}  "
            f"{fmtf(avg(r['lag_bids'].get(30) for r in rows)):>5}  "
            f"{fmtf(avg(r['lag_bids'].get(60) for r in rows)):>5}  "
            f"{fmtf(avg(r['lag_bids'].get(120) for r in rows)):>5}  "
            f"{fmtf(avg(r['max_4h'] for r in rows)):>5}  "
            f"{100*won/len(rows):>5.1f}%"
        )

    # ── Section C: Timing ────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("C. WHEN DO CONTRA EVENTS OCCUR? (minutes after entry)")
    print("=" * 72)
    timing_buckets = [
        ("<1h",    lambda m: m < 60),
        ("1–3h",   lambda m: 60 <= m < 180),
        ("3–6h",   lambda m: 180 <= m < 360),
        ("6–12h",  lambda m: 360 <= m < 720),
        (">12h",   lambda m: m >= 720),
    ]
    print(f"  {'Bucket':<12} {'N':>3}  {'Yes@Ev':>6}  {'Max4h':>5}  {'Won%':>5}  {'AvgMargin':>9}")
    print("  " + "-" * 50)
    for label, fn in timing_buckets:
        rows = [r for r in per_trade_first.values() if fn(r["mins_after_entry"])]
        if not rows:
            continue
        won = sum(1 for r in rows if r["outcome"] == "won")
        print(
            f"  {label:<12} {len(rows):>3}  "
            f"{fmtf(avg(r['yes_now'] for r in rows)):>6}  "
            f"{fmtf(avg(r['max_4h'] for r in rows)):>5}  "
            f"{100*won/len(rows):>5.1f}%  "
            f"{fmtf(avg(r['margin'] for r in rows), 9):>9}"
        )

    # ── Section D: Per-trade detail ──────────────────────────────────────────
    print("\n" + "=" * 72)
    print("D. PER-TRADE: FIRST CONTRA EVENT")
    print("=" * 72)
    print(f"  {'Ticker':<32} {'Out':>3}  {'Src':<13} {'Mgn':>4}  {'Lag':>5}  {'Y@Ev':>4}  {'Y+1h':>4}  {'Max4h':>5}  {'PnL':>7}")
    print("  " + "-" * 85)
    for trade_id, r in sorted(per_trade_first.items(), key=lambda x: x[1]["evt_dt"]):
        lag_h = f"{r['mins_after_entry']/60:.1f}h"
        print(
            f"  {r['ticker']:<32} {r['outcome'][:3]:>3}  {r['source']:<13} "
            f"{r['margin']:>+4.1f}  {lag_h:>5}  "
            f"{fmt(r['yes_now']):>4}  "
            f"{fmt(r['lag_bids'].get(60)):>4}  "
            f"{fmt(r['max_4h']):>5}  "
            f"{int(r['pnl'] or 0):>7}"
        )

    # ── Section E: Baseline ───────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("E. BASELINE: TRADES WITH NO CONTRA EVENTS")
    print("=" * 72)
    if all_baseline:
        won = sum(1 for r in all_baseline if r["outcome"] == "won")
        max_yes_vals = [r["max_yes"] for r in all_baseline if r["max_yes"] is not None]
        entry_bids   = [r["entry_bid"] for r in all_baseline if r["entry_bid"] is not None]
        print(f"  N={len(all_baseline)}  Won={won} ({100*won/len(all_baseline):.1f}%)")
        print(f"  Avg YES bid at entry : {statistics.mean(entry_bids):.1f}" if entry_bids else "  No entry bids")
        print(f"  Avg max YES bid held : {statistics.mean(max_yes_vals):.1f}" if max_yes_vals else "  No max bids")
    else:
        print("  No baseline trades found.")

    # ── Section F: Win rate comparison ───────────────────────────────────────
    print("\n" + "=" * 72)
    print("F. WIN RATE SUMMARY")
    print("=" * 72)
    if per_trade_first:
        contra_rows = list(per_trade_first.values())
        won_c = sum(1 for r in contra_rows if r["outcome"] == "won")
        print(f"  Trades with contra event : {len(contra_rows):3d}  won={won_c}  ({100*won_c/len(contra_rows):.1f}%)")
    if all_baseline:
        won_b = sum(1 for r in all_baseline if r["outcome"] == "won")
        print(f"  Trades without contra    : {len(all_baseline):3d}  won={won_b}  ({100*won_b/len(all_baseline):.1f}%)")

    print()


if __name__ == "__main__":
    main()
