"""Backtest forecast_no strategy using live trade history.

Loads all forecast_no trades from the DB, classifies each by settlement
result (won/lost), and slices win rates and P&L across every relevant
feature — individually and in pairwise/triple combinations — to surface
which factors drive wins and which drive losses.

Ground truth: settled_result ('no' = win, 'yes' = loss).
P&L: settled_pnl_cents when available, else exit_pnl_cents.
Trades with neither are excluded (~2 pending today's markets).

Usage:
    venv/bin/python scripts/backtest_forecast_no_live.py
    venv/bin/python scripts/backtest_forecast_no_live.py --since 2026-05-14
    venv/bin/python scripts/backtest_forecast_no_live.py --min-n 5
    venv/bin/python scripts/backtest_forecast_no_live.py --out results/backtest.txt
"""

import argparse
import json
import re
import sqlite3
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "db" / "opportunity_log.db"

# ── feature extraction ────────────────────────────────────────────────────────

def parse_trades(conn, since: str | None = None) -> list[dict]:
    where = "source = 'forecast_no' AND settled_result IS NOT NULL"
    params: list = []
    if since:
        where += " AND logged_at >= ?"
        params.append(since)
    rows = conn.execute(f"""
        SELECT id, logged_at, ticker, limit_price, count,
               exit_pnl_cents, exit_reason, settled_result, settled_pnl_cents,
               p_estimate, note, yes_bid_entry, yes_ask_entry, market_p_entry,
               corroborating_sources
        FROM trades WHERE {where} ORDER BY logged_at
    """, params).fetchall()

    trades = []
    for r in rows:
        (tid, logged_at, ticker, entry, count,
         exit_pnl, exit_reason, settled_result, settled_pnl,
         p_est, note_str, yes_bid, yes_ask, market_p, _corr_sources) = r

        note: dict = {}
        if note_str:
            try:
                note = json.loads(note_str)
            except Exception:
                pass

        won = settled_result == "no"
        pnl = settled_pnl if settled_pnl is not None else exit_pnl

        # city / direction from ticker
        m = re.match(r"KX(?:HIGH[T]?|LOWT?)([A-Z]+)-", ticker)
        city = m.group(1) if m else "?"
        is_low = "LOWT" in ticker

        # note fields
        min_edge    = note.get("min_edge_f")
        max_edge    = note.get("max_edge_f")
        n_sources   = note.get("source_count")
        spread_f    = note.get("model_spread_f")
        hours       = note.get("hours_to_close")
        no_dir      = note.get("no_direction")
        sources_det = note.get("sources_detail", [])   # [[name, forecast, edge], ...]

        # derived
        edge_range  = (max_edge - min_edge) if (max_edge is not None and min_edge is not None) else None
        mkt_spread  = (yes_ask - yes_bid) if (yes_ask is not None and yes_bid is not None) else None
        # how far our entry is below what market implies (market was pricing YES higher → we got NO cheap)
        market_discount = ((market_p or 0) - (entry / 100.0)) if market_p else None
        # day of week from logged_at
        from datetime import datetime
        try:
            dt = datetime.fromisoformat(logged_at)
            dow = dt.weekday()   # 0=Mon 6=Sun
            local_hour = dt.hour
        except Exception:
            dow = -1
            local_hour = -1

        # which source families are present
        src_names = {s[0] for s in sources_det}
        has_hrrr   = "hrrr" in src_names
        has_ecmwf  = "open_meteo_ecmwf" in src_names
        has_icon   = "open_meteo_icon" in src_names
        has_gem    = "open_meteo_gem" in src_names
        has_noaa   = "noaa" in src_names
        has_nws_h  = "nws_hourly" in src_names

        trades.append(dict(
            id=tid, date=logged_at[:10], ticker=ticker,
            entry=entry, count=count, pnl=pnl,
            won=won, exit_reason=exit_reason or "no_exit",
            p_est=p_est,
            min_edge=min_edge, max_edge=max_edge, edge_range=edge_range,
            n_sources=n_sources, spread_f=spread_f, hours=hours,
            no_dir=no_dir, city=city, is_low=is_low,
            yes_bid=yes_bid, yes_ask=yes_ask, mkt_spread=mkt_spread,
            market_p=market_p, market_discount=market_discount,
            has_hrrr=has_hrrr, has_ecmwf=has_ecmwf, has_icon=has_icon,
            has_gem=has_gem, has_noaa=has_noaa, has_nws_h=has_nws_h,
            dow=dow, local_hour=local_hour,
        ))
    return trades


# ── bucketing helpers ─────────────────────────────────────────────────────────

def _stats(trades: list[dict]) -> dict:
    n   = len(trades)
    w   = sum(1 for t in trades if t["won"])
    pnl = sum(t["pnl"] for t in trades if t["pnl"] is not None)
    # limit_price stores the YES bid price; actual NO cost = 100 - yes_bid
    avg_yes     = sum(t["entry"] for t in trades) / n if n else 0
    avg_no_cost = 100 - avg_yes   # what the bot actually paid per contract
    return dict(n=n, w=w, l=n-w, wr=w/n if n else 0, pnl=pnl,
                avg_entry=avg_yes, avg_no_cost=avg_no_cost)

def _print_bucket_table(title: str, groups: dict[str, list[dict]], min_n: int = 3) -> None:
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")
    # NO_cost = what bot paid; YES_bid shown for reference; break-even = NO_cost/100
    print(f"{'Bucket':<22} {'N':>4} {'W':>4} {'L':>4} {'WR%':>6} {'P&L¢':>8}  {'NO_cost':>8}  {'YES_bid':>8}  {'BE%':>5}  Kelly_ok?")
    rows_out = []
    for label, subset in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        s = _stats(subset)
        if s["n"] < min_n:
            continue
        # Break-even win rate = NO_cost / 100 (what you paid / max payout)
        breakeven_p = s["avg_no_cost"] / 100.0
        kelly_ok = "✓" if s["wr"] > breakeven_p else "✗"
        rows_out.append((label, s, breakeven_p, kelly_ok))

    rows_out.sort(key=lambda x: -x[1]["pnl"])
    for label, s, be, kelly_ok in rows_out:
        print(f"  {label:<20} {s['n']:>4} {s['w']:>4} {s['l']:>4} "
              f"{s['wr']*100:>5.0f}%  {s['pnl']:>+8.0f}¢  "
              f"{s['avg_no_cost']:>7.0f}¢  {s['avg_entry']:>7.0f}¢  "
              f"{be*100:>4.0f}%  {kelly_ok}")


def bucket_by(trades: list[dict], key_fn, label_fn=None) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        v = key_fn(t)
        if v is None:
            continue
        label = label_fn(v) if label_fn else str(v)
        if label is not None:
            groups[label].append(t)
    return dict(groups)


# ── feature bucket definitions ────────────────────────────────────────────────

FEATURE_BUCKETS: list[tuple[str, callable]] = [
    # YES bid (limit_price) — what the market charged for YES at entry
    ("YES bid (limit_price)", lambda t: (
        "≤25¢"   if t["entry"] <= 25 else
        "26-35¢" if t["entry"] <= 35 else
        "36-50¢" if t["entry"] <= 50 else
        "≥51¢"
    )),
    # NO cost (100 - yes_bid) — what the bot actually paid per contract
    ("NO cost paid", lambda t: (
        "≤50¢ (YES≥50)"  if (100 - t["entry"]) <= 50 else
        "51-65¢ (YES35-49)" if (100 - t["entry"]) <= 65 else
        "66-75¢ (YES25-34)" if (100 - t["entry"]) <= 75 else
        "76-80¢ (YES20-24)"
    )),
    ("Hours to close", lambda t: None if t["hours"] is None else (
        "<12h"   if t["hours"] < 12 else
        "12-18h" if t["hours"] < 18 else
        "18-24h" if t["hours"] < 24 else
        "≥24h"
    )),
    ("Min edge (°F)", lambda t: None if t["min_edge"] is None else (
        "<2°F"   if t["min_edge"] < 2 else
        "2-3°F"  if t["min_edge"] < 3 else
        "3-4°F"  if t["min_edge"] < 4 else
        "4-5°F"  if t["min_edge"] < 5 else
        "5-7°F"  if t["min_edge"] < 7 else
        "≥7°F"
    )),
    ("N sources", lambda t: None if t["n_sources"] is None else (
        "2"   if t["n_sources"] == 2 else
        "3"   if t["n_sources"] == 3 else
        "4"   if t["n_sources"] == 4 else
        "5"   if t["n_sources"] == 5 else
        "6+"
    )),
    ("Model spread (°F)", lambda t: None if t["spread_f"] is None else (
        "<3°F"   if t["spread_f"] < 3 else
        "3-5°F"  if t["spread_f"] < 5 else
        "5-8°F"  if t["spread_f"] < 8 else
        "8-12°F" if t["spread_f"] < 12 else
        "≥12°F"
    )),
    ("Edge range (max-min)", lambda t: None if t["edge_range"] is None else (
        "<1°F"   if t["edge_range"] < 1 else
        "1-2°F"  if t["edge_range"] < 2 else
        "2-4°F"  if t["edge_range"] < 4 else
        "≥4°F"
    )),
    ("Market bid-ask spread", lambda t: None if t["mkt_spread"] is None else (
        "1¢"    if t["mkt_spread"] <= 1 else
        "2-3¢"  if t["mkt_spread"] <= 3 else
        "4-6¢"  if t["mkt_spread"] <= 6 else
        "7-10¢" if t["mkt_spread"] <= 10 else
        "≥11¢"
    )),
    ("Market discount (our_p - mkt_p)", lambda t: None if t["market_discount"] is None else (
        "<-0.10 (mkt already expensive)" if t["market_discount"] < -0.10 else
        "-0.10-0 (slight premium)"       if t["market_discount"] < 0    else
        "0-0.15 (mild discount)"         if t["market_discount"] < 0.15 else
        "0.15-0.30 (good discount)"      if t["market_discount"] < 0.30 else
        "≥0.30 (deep discount)"
    )),
    ("City", lambda t: t["city"]),
    ("HIGH vs LOW", lambda t: "LOW" if t["is_low"] else "HIGH"),
    ("NO direction", lambda t: t["no_dir"] or "over/under"),
    ("Has HRRR", lambda t: "HRRR yes" if t["has_hrrr"] else "HRRR no"),
    ("Has ECMWF", lambda t: "ECMWF yes" if t["has_ecmwf"] else "ECMWF no"),
    ("Has ICON", lambda t: "ICON yes" if t["has_icon"] else "ICON no"),
    ("Has GEM", lambda t: "GEM yes" if t["has_gem"] else "GEM no"),
    ("Day of week", lambda t: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][t["dow"]] if t["dow"] >= 0 else None),
    ("Entry hour (UTC)", lambda t: (
        "00-06h"  if 0 <= t["local_hour"] < 6  else
        "06-10h"  if t["local_hour"] < 10 else
        "10-14h"  if t["local_hour"] < 14 else
        "14-18h"  if t["local_hour"] < 18 else
        "18-24h"
    ) if t["local_hour"] >= 0 else None),
    ("Exit reason", lambda t: t["exit_reason"]),
]

# Pairs to always show (most informative cross-tabs)
KEY_PAIRS = [
    ("Hours to close", "NO cost paid"),
    ("Hours to close", "Min edge (°F)"),
    ("Hours to close", "City"),
    ("Hours to close", "Has HRRR"),
    ("NO cost paid", "Min edge (°F)"),
    ("NO cost paid", "N sources"),
    ("Min edge (°F)", "N sources"),
    ("Min edge (°F)", "Model spread (°F)"),
    ("Market bid-ask spread", "NO cost paid"),
    ("Market bid-ask spread", "Hours to close"),
    ("Has HRRR", "Min edge (°F)"),
    ("HIGH vs LOW", "Hours to close"),
    ("HIGH vs LOW", "Min edge (°F)"),
]

# ── threshold sweep helpers ───────────────────────────────────────────────────

class _Tee:
    """Write to stdout and an optional file simultaneously."""
    def __init__(self, *files):
        self._files = files
    def write(self, s: str) -> None:
        for f in self._files:
            f.write(s)
    def flush(self) -> None:
        for f in self._files:
            f.flush()


def _sweep_print_header(title: str) -> None:
    print(f"\n{'='*72}")
    print(f" {title}")
    print(f"{'='*72}")
    print(f"  {'Threshold':<26} {'N':>5} {'WR%':>6} {'P&L¢':>10}  {'NO_cost':>8}  {'BE%':>5}  Kelly")


def _sweep_row(label: str, subset: list[dict]) -> None:
    if not subset:
        print(f"  {label:<26} {'0':>5}")
        return
    s = _stats(subset)
    be = s["avg_no_cost"] / 100.0
    kelly = "✓" if s["wr"] > be else "✗"
    print(f"  {label:<26} {s['n']:>5} {s['wr']*100:>5.0f}%  {s['pnl']:>+10.0f}¢  "
          f"{s['avg_no_cost']:>7.0f}¢  {be*100:>4.0f}%  {kelly}")


def _print_threshold_sweeps(trades: list[dict]) -> None:
    """Simulate different gate thresholds and print P&L for each.

    Each sweep shows what would have happened if the bot only took trades
    that passed the given filter.  Rejected trades are simply not taken
    (no substitute band assumed), so P&L improvements here are a lower
    bound — if alternative bands existed they'd add more.
    """
    print(f"\n\n{'#'*72}")
    print("  THRESHOLD GATE SIMULATIONS")
    print("  (Rejected trades dropped — no replacement band assumed)")
    print("  Asterisk (*) marks the current live setting.")
    print(f"{'#'*72}")

    # ── Max NO cost gate (100 - entry <= thresh) ──
    _sweep_print_header("Gate: Max NO cost ≤ X¢  (lower NO cost = cheaper, cheaper = better BE%)")
    for thresh in [50, 55, 60, 65, 70, 75, 80]:
        label = f"≤{thresh}¢ NO cost"
        if thresh == 80:
            label += "  *"
        _sweep_row(label, [t for t in trades if (100 - t["entry"]) <= thresh])

    # ── Min YES bid gate (tighter from the YES side for clarity) ──
    _sweep_print_header("Gate: Min YES bid ≥ X¢  (higher YES bid = cheaper NO)")
    for thresh in [20, 25, 30, 35, 40, 45]:
        label = f"≥{thresh}¢ YES bid"
        if thresh == 20:
            label += "  *"
        _sweep_row(label, [t for t in trades if t["entry"] >= thresh])

    # ── Min edge gate ──
    _sweep_print_header("Gate: Min edge ≥ X°F  (minimum qualifying source edge)")
    for thresh in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        label = f"≥{thresh:.1f}°F edge"
        if thresh == 1.0:
            label += "  *"
        _sweep_row(label, [t for t in trades if t["min_edge"] is not None and t["min_edge"] >= thresh])

    # ── Max hours gate ──
    _sweep_print_header("Gate: Max hours to close ≤ X h  (reject far-future entries)")
    for thresh in [12, 15, 18, 20, 22, 24, 30, 999]:
        label = f"≤{thresh}h" if thresh < 999 else "no limit  *"
        _sweep_row(label, [t for t in trades if t["hours"] is not None and t["hours"] <= thresh])

    # ── Min n_sources gate ──
    _sweep_print_header("Gate: Min qualifying sources ≥ N")
    for thresh in [2, 3, 4, 5]:
        label = f"≥{thresh} sources"
        if thresh == 2:
            label += "  *"
        _sweep_row(label, [t for t in trades if t["n_sources"] is not None and t["n_sources"] >= thresh])

    # ── Entry hour gate (UTC) ──
    _sweep_print_header("Gate: Entry hour UTC  (which windows are profitable?)")
    windows = [
        ("All hours  *",    lambda _: True),
        ("10-18h only",     lambda t: 10 <= t["local_hour"] < 18),
        ("08-20h only",     lambda t: 8  <= t["local_hour"] < 20),
        ("12-22h only",     lambda t: 12 <= t["local_hour"] < 22),
        ("Excl 00-06h",     lambda t: t["local_hour"] >= 6),
        ("Excl 00-10h",     lambda t: t["local_hour"] >= 10),
    ]
    for label, fn in windows:
        _sweep_row(label, [t for t in trades if t["local_hour"] >= 0 and fn(t)])

    # ── Direction filter ──
    _sweep_print_header("Gate: NO direction  (exclude T-market over/under?)")
    _sweep_row("All directions  *",    trades)
    _sweep_row("Band only (no T-mkt)", [t for t in trades if t["no_dir"] not in (None, "over", "under")])
    _sweep_row("T-market only",        [t for t in trades if t["no_dir"] in ("over", "under")])

    # ── Combined best-guess gates ──
    _sweep_print_header("Combined gates  (stacking multiple filters)")
    combos = [
        ("Current (no gates)",
         lambda _: True),
        ("NO cost ≤65¢",
         lambda t: (100 - t["entry"]) <= 65),
        ("NO cost ≤65¢ + hours ≤22",
         lambda t: (100 - t["entry"]) <= 65 and t["hours"] is not None and t["hours"] <= 22),
        ("NO cost ≤65¢ + hours ≤22 + 10-18h",
         lambda t: (100 - t["entry"]) <= 65
                   and t["hours"] is not None and t["hours"] <= 22
                   and 10 <= t["local_hour"] < 18),
        ("NO cost ≤65¢ + band-only",
         lambda t: (100 - t["entry"]) <= 65
                   and t["no_dir"] not in (None, "over", "under")),
        ("NO cost ≤65¢ + band + hours≤22 + 10-18h",
         lambda t: (100 - t["entry"]) <= 65
                   and t["no_dir"] not in (None, "over", "under")
                   and t["hours"] is not None and t["hours"] <= 22
                   and 10 <= t["local_hour"] < 18),
        ("NO cost ≤65¢ + ≥3 src + band + hours≤22",
         lambda t: (100 - t["entry"]) <= 65
                   and t["n_sources"] is not None and t["n_sources"] >= 3
                   and t["no_dir"] not in (None, "over", "under")
                   and t["hours"] is not None and t["hours"] <= 22),
        ("Tightest: ≤60¢ + ≥3src + band + hrs≤20 + 10-18h",
         lambda t: (100 - t["entry"]) <= 60
                   and t["n_sources"] is not None and t["n_sources"] >= 3
                   and t["no_dir"] not in (None, "over", "under")
                   and t["hours"] is not None and t["hours"] <= 20
                   and 10 <= t["local_hour"] < 18),
    ]
    for label, fn in combos:
        _sweep_row(label, [t for t in trades if fn(t)])


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=None, help="ISO date filter, e.g. 2026-05-14")
    ap.add_argument("--min-n", type=int, default=4, help="Min trades per bucket to show")
    ap.add_argument("--pairs", action="store_true", default=True, help="Show pairwise combos")
    ap.add_argument("--no-pairs", dest="pairs", action="store_false")
    ap.add_argument("--triples", action="store_true", default=False, help="Show triple combos (slow)")
    ap.add_argument("--out", default=None, help="Also write output to this file path")
    args = ap.parse_args()

    out_file = None
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_file = open(out_path, "w", encoding="utf-8")
        sys.stdout = _Tee(sys.__stdout__, out_file)

    conn = sqlite3.connect(str(DB_PATH))
    trades = parse_trades(conn, since=args.since)
    conn.close()

    s = _stats(trades)
    date_range = f"{trades[0]['date']} → {trades[-1]['date']}" if trades else "?"
    print(f"\n{'#'*60}")
    print(f"  FORECAST-NO LIVE BACKTEST")
    print(f"  {len(trades)} trades  ({date_range})")
    if args.since:
        print(f"  (filtered: since {args.since})")
    print(f"{'#'*60}")
    print(f"  Win rate:    {s['wr']*100:.1f}%  ({s['w']}W / {s['l']}L)")
    print(f"  Total P&L:   {s['pnl']:+.0f}¢  (${s['pnl']/100:.2f})")
    print(f"  Avg YES bid: {s['avg_entry']:.0f}¢  (what market priced YES at entry)")
    print(f"  Avg NO cost: {s['avg_no_cost']:.0f}¢  (what bot actually paid per contract)")
    breakeven = s["avg_no_cost"] / 100.0
    print(f"  Break-even:  {breakeven*100:.0f}% win rate needed at avg NO cost")
    print(f"  Kelly:       {'POSITIVE' if s['wr'] > breakeven else 'NEGATIVE'}")

    # ── single-factor buckets ──
    feat_map = {name: fn for name, fn in FEATURE_BUCKETS}
    for feat_name, feat_fn in FEATURE_BUCKETS:
        groups = bucket_by(trades, feat_fn)
        _print_bucket_table(feat_name, groups, min_n=args.min_n)

    # ── threshold gate simulations (always shown) ──
    _print_threshold_sweeps(trades)

    if not args.pairs:
        if out_file:
            sys.stdout = sys.__stdout__
            out_file.close()
            print(f"Output written to {args.out}")
        return

    # ── pairwise cross-tabs ──
    print(f"\n\n{'#'*60}")
    print("  PAIRWISE COMBINATIONS")
    print(f"{'#'*60}")

    for f1_name, f2_name in KEY_PAIRS:
        f1_fn = feat_map[f1_name]
        f2_fn = feat_map[f2_name]
        groups: dict[str, list[dict]] = defaultdict(list)
        for t in trades:
            v1 = f1_fn(t)
            v2 = f2_fn(t)
            if v1 is None or v2 is None:
                continue
            groups[f"{v1}  ×  {v2}"].append(t)
        _print_bucket_table(f"{f1_name}  ×  {f2_name}", dict(groups), min_n=args.min_n)

    if args.triples:
        print(f"\n\n{'#'*60}")
        print("  TRIPLE COMBINATIONS (hours × NO cost × one more)")
        print(f"{'#'*60}")
        third_names  = ["Min edge (°F)", "N sources", "City", "Has HRRR", "HIGH vs LOW"]
        for t3 in third_names:
            f1_fn = feat_map["Hours to close"]
            f2_fn = feat_map["NO cost paid"]
            f3_fn = feat_map[t3]
            groups: dict[str, list[dict]] = defaultdict(list)
            for t in trades:
                v1, v2, v3 = f1_fn(t), f2_fn(t), f3_fn(t)
                if None in (v1, v2, v3):
                    continue
                groups[f"{v1}  ×  {v2}  ×  {v3}"].append(t)
            _print_bucket_table(
                f"Hours × Entry × {t3}", dict(groups), min_n=args.min_n
            )

    # ── threshold gate simulations ──
    _print_threshold_sweeps(trades)

    # ── worst and best 10 buckets (pairwise) ──
    print(f"\n\n{'#'*60}")
    print("  ALL-PAIRWISE: TOP/BOTTOM 10 by P&L  (min_n={})".format(args.min_n))
    print(f"{'#'*60}")

    all_pairs: list[tuple[str, dict]] = []
    # Exclude "Exit reason" from the exhaustive sweep — it's circular
    # (profit_take trades are wins by construction; stop_loss are losses).
    feat_names = [n for n, _ in FEATURE_BUCKETS if n != "Exit reason"]
    for f1_name, f2_name in combinations(feat_names, 2):
        f1_fn = feat_map[f1_name]
        f2_fn = feat_map[f2_name]
        groups: dict[str, list[dict]] = defaultdict(list)
        for t in trades:
            v1 = f1_fn(t)
            v2 = f2_fn(t)
            if v1 is None or v2 is None:
                continue
            groups[f"[{f1_name}={v1}] × [{f2_name}={v2}]"].append(t)
        for label, subset in groups.items():
            s2 = _stats(subset)
            if s2["n"] >= args.min_n:
                all_pairs.append((label, s2))

    all_pairs.sort(key=lambda x: x[1]["pnl"])

    print("\n  WORST 10 combinations:")
    print(f"  {'Bucket':<60} {'N':>4} {'WR%':>6} {'P&L¢':>8}")
    for label, s2 in all_pairs[:10]:
        print(f"  {label:<60} {s2['n']:>4} {s2['wr']*100:>5.0f}%  {s2['pnl']:>+8.0f}¢")

    print("\n  BEST 10 combinations:")
    print(f"  {'Bucket':<60} {'N':>4} {'WR%':>6} {'P&L¢':>8}")
    for label, s2 in reversed(all_pairs[-10:]):
        print(f"  {label:<60} {s2['n']:>4} {s2['wr']*100:>5.0f}%  {s2['pnl']:>+8.0f}¢")

    if out_file:
        sys.stdout = sys.__stdout__
        out_file.close()
        print(f"Output written to {args.out}")


if __name__ == "__main__":
    main()
