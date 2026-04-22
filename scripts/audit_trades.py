"""Trade audit script — compare losing vs winning non-band-arb trades.

Usage:
    # Full audit of all resolved non-band-arb trades
    venv/bin/python scripts/audit_trades.py

    # Drill into specific trade IDs
    venv/bin/python scripts/audit_trades.py --ids 89 100 103 108 116 120

    # Limit to a specific opportunity kind
    venv/bin/python scripts/audit_trades.py --kind forecast_no

    # Include band_arb in the analysis
    venv/bin/python scripts/audit_trades.py --all-kinds

    # Suppress raw per-trade detail, show summary only
    venv/bin/python scripts/audit_trades.py --summary-only

    # Write output to a file instead of stdout
    venv/bin/python scripts/audit_trades.py --out audit_report.txt

Output sections:
    1. Per-trade drill-down (targeted losses + comparable wins)
    2. Risk-factor frequency table (losses vs wins)
    3. City win-rate table
    4. Entry-hour win-rate table
    5. Source win-rate table (for numeric trades)
    6. Actionable flag checklist for each targeted loss

Risk flags checked per trade:
    - THIN_EDGE      : min qualifying edge < 3°F across all raw_forecasts sources
    - MODEL_DISAGREE : max(open_meteo) vs min(hrrr/nws_hourly) spread > 8°F
    - HRRR_DISSENT   : HRRR present and edge < 2°F (near or opposing YES side)
    - TERRAIN_CITY   : city in known terrain-volatile list (den, okc, dal)
    - LOW_T_MARKET   : KXLOWT "T" (over) direction — overnight-low "won't cool" bet
    - LATE_HIGH_ENTRY: HIGH market, entry local hour ≥ 15, METAR at/near strike
    - OVERNIGHT_ENTRY: LOW market, entry local hour < 6 (past midnight, before dawn lock)
    - MARGINAL_MKT_P : market_p_entry (YES bid) > 0.45 at entry (market uncertain)
    - WEAK_CORR      : numeric trade with 0-1 corroborating sources
    - SOLO_OPENMETEO : only qualifying source is open_meteo (high MAE, no near-term model)
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_PATH = Path(__file__).parent.parent / "opportunity_log.db"

# Cities with documented terrain-driven forecast difficulty
TERRAIN_CITIES = frozenset({"den", "okc", "dal", "slc"})

# Risk-flag thresholds
THIN_EDGE_F        = 3.0   # °F — qualifying edge below this is marginal
MODEL_DISAGREE_F   = 8.0   # °F — open_meteo vs HRRR/NWS spread above this is suspicious
HRRR_DISSENT_F     = 2.0   # °F — HRRR edge below this means HRRR is near or against NO
LATE_HIGH_HOUR     = 15    # local hour — HIGH entry at or past this is "late"
OVERNIGHT_LOW_HOUR = 6     # local hour — LOW entry before this is overnight window
MARGINAL_MKT_P     = 0.45  # YES bid — market giving YES ≥ this means genuine uncertainty

# City timezone abbreviations (local-hour computation uses UTC offset approximation)
# For exact local hours we'd need the full tz table; this is good enough for audit.
CITY_UTC_OFFSET: dict[str, int] = {
    "ny": -4, "bos": -4, "mia": -4, "phi": -4, "phx": -7, "phl": -4,
    "atl": -4, "dc": -4, "dca": -4,
    "chi": -5, "dal": -5, "dfw": -5, "hou": -5, "okc": -5, "sat": -5,
    "aus": -5, "nola": -5, "msy": -5,
    "den": -6,
    "lax": -7, "sfo": -7, "sea": -7, "las": -7, "lv": -7,
    "min": -5, "msp": -5,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def city_from_ticker(ticker: str) -> str:
    """Extract the city suffix from a Kalshi temperature ticker.

    KXHIGHTBOS-26APR20-B52.5  →  'bos'
    KXLOWTDEN-26APR19-T31     →  'den'
    Returns '' for non-temperature tickers.
    """
    for prefix in ("KXHIGHT", "KXHIGH", "KXLOWT", "KXLOW"):
        if ticker.startswith(prefix):
            rest = ticker[len(prefix):]
            city = rest.split("-")[0].lower()
            return city
    return ""


def market_type(ticker: str) -> str:
    """Return 'HIGH' / 'LOW' / 'OTHER'."""
    if "KXHIGH" in ticker:
        return "HIGH"
    if "KXLOWT" in ticker or "KXLOW" in ticker:
        return "LOW"
    return "OTHER"


def market_direction(ticker: str) -> str:
    """Return 'over' / 'under' / 'between' / '?' from the strike segment.

    KXHIGHTBOS-26APR20-B52.5  → 'under'   (B = below/under)
    KXLOWTDEN-26APR19-T31     → 'over'    (T = top/over)
    """
    parts = ticker.split("-")
    if len(parts) < 3:
        return "?"
    strike_seg = parts[2]
    if strike_seg.startswith("B") and "_" in strike_seg:
        return "between"
    if strike_seg.startswith("B"):
        return "under"
    if strike_seg.startswith("T"):
        return "over"
    return "?"


def local_hour(utc_iso: str, city: str) -> int | None:
    """Approximate local hour of entry for a given city."""
    try:
        dt = datetime.fromisoformat(utc_iso.replace("Z", "+00:00"))
        offset = CITY_UTC_OFFSET.get(city, 0)
        local_h = (dt.hour + offset) % 24
        return local_h
    except Exception:
        return None


def load_trades(conn: sqlite3.Connection, ids: list[int] | None,
                kinds: list[str] | None) -> list[dict]:
    cur = conn.cursor()
    query = """
        SELECT id, logged_at, ticker, side, count, limit_price,
               opportunity_kind, score, p_estimate, status,
               source, outcome, market_p_entry,
               yes_bid_entry, yes_ask_entry, signal_p_yes,
               corroborating_sources, fill_price_cents,
               exit_price_cents, exit_pnl_cents, exit_reason, peak_past
        FROM trades
        WHERE (outcome IS NOT NULL OR exit_pnl_cents IS NOT NULL)
    """
    params: list = []
    if ids:
        placeholders = ",".join("?" * len(ids))
        query += f" AND id IN ({placeholders})"
        params.extend(ids)
    if kinds:
        placeholders = ",".join("?" * len(kinds))
        query += f" AND opportunity_kind IN ({placeholders})"
        params.extend(kinds)
    query += " ORDER BY id"
    cur.execute(query, params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def load_forecasts(conn: sqlite3.Connection, ticker: str) -> list[dict]:
    cur = conn.cursor()
    cur.execute("""
        SELECT source, direction, data_value, strike, edge, logged_at
        FROM raw_forecasts WHERE ticker = ?
        ORDER BY logged_at, source
    """, (ticker,))
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def summarise_forecasts(rows: list[dict]) -> dict:
    """Aggregate raw_forecast rows into per-source stats."""
    by_src: dict[str, list[float]] = defaultdict(list)
    by_src_edge: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_src[r["source"]].append(r["data_value"])
        if r["edge"] is not None:
            by_src_edge[r["source"]].append(r["edge"])

    result: dict = {}
    for src in by_src:
        vals = by_src[src]
        edges = by_src_edge.get(src, [])
        result[src] = {
            "val_min": min(vals), "val_max": max(vals),
            "edge_min": min(edges) if edges else None,
            "edge_max": max(edges) if edges else None,
            "n": len(vals),
        }
    return result


def compute_flags(trade: dict, fc_summary: dict) -> list[str]:
    """Return list of risk flag names that fired for this trade."""
    flags: list[str] = []
    ticker   = trade["ticker"]
    city     = city_from_ticker(ticker)
    mtype    = market_type(ticker)
    mdir     = market_direction(ticker)
    kind     = trade["opportunity_kind"]
    src      = trade["source"] or ""
    corr     = trade["corroborating_sources"] or ""
    mkt_p    = trade["market_p_entry"]
    entry_ts = trade["logged_at"]
    lhour    = local_hour(entry_ts, city) if city else None

    # THIN_EDGE: minimum observed qualifying edge across all sources < threshold
    all_min_edges = [
        s["edge_min"] for s in fc_summary.values()
        if s["edge_min"] is not None and s["edge_min"] >= 0
    ]
    if all_min_edges and min(all_min_edges) < THIN_EDGE_F:
        flags.append("THIN_EDGE")
    elif not all_min_edges and kind == "forecast_no":
        flags.append("THIN_EDGE")   # no edge data at all

    # MODEL_DISAGREE: open_meteo vs HRRR/NWS spread
    om_max   = fc_summary.get("open_meteo", {}).get("val_max")
    hrrr_min = fc_summary.get("hrrr",      {}).get("val_min")
    nws_min  = fc_summary.get("nws_hourly",{}).get("val_min")
    near_min = min(x for x in [hrrr_min, nws_min] if x is not None) if (hrrr_min or nws_min) else None
    if om_max is not None and near_min is not None:
        if om_max - near_min > MODEL_DISAGREE_F:
            flags.append("MODEL_DISAGREE")

    # HRRR_DISSENT: HRRR present but barely supporting NO (edge < threshold)
    hrrr_info = fc_summary.get("hrrr")
    if hrrr_info and hrrr_info["edge_max"] is not None:
        if hrrr_info["edge_max"] < HRRR_DISSENT_F:
            flags.append("HRRR_DISSENT")

    # TERRAIN_CITY
    if city in TERRAIN_CITIES:
        flags.append("TERRAIN_CITY")

    # LOW_T_MARKET: KXLOWT "over" direction (overnight won't cool)
    if mtype == "LOW" and mdir == "over":
        flags.append("LOW_T_MARKET")

    # LATE_HIGH_ENTRY: HIGH market entered late with thin remaining time
    if mtype == "HIGH" and lhour is not None and lhour >= LATE_HIGH_HOUR:
        flags.append("LATE_HIGH_ENTRY")

    # OVERNIGHT_ENTRY: LOW market entered after midnight and before dawn lock
    if mtype == "LOW" and lhour is not None and lhour < OVERNIGHT_LOW_HOUR:
        flags.append("OVERNIGHT_ENTRY")

    # MARGINAL_MKT_P: market was already pricing significant YES probability
    if mkt_p is not None and mkt_p > MARGINAL_MKT_P:
        flags.append("MARGINAL_MKT_P")

    # WEAK_CORR: numeric trade with few corroborating sources
    if kind == "numeric":
        n_corr = len([x for x in corr.split(",") if x.strip()]) if corr else 0
        if n_corr <= 1:
            flags.append("WEAK_CORR")

    # SOLO_OPENMETEO: only open_meteo qualifies (no near-term model)
    if kind == "forecast_no":
        near_term = {"hrrr", "nws_hourly"}
        sources_present = set(fc_summary.keys())
        if not (sources_present & near_term) and "open_meteo" in sources_present:
            flags.append("SOLO_OPENMETEO")

    return flags


def pnl_cents(trade: dict) -> float | None:
    """Return net P&L in cents (positive = win)."""
    if trade["exit_pnl_cents"] is not None:
        return float(trade["exit_pnl_cents"])
    if trade["outcome"] == "won":
        # settled win: profit = (100 - limit_price) * count
        return (100 - trade["limit_price"]) * trade["count"]
    if trade["outcome"] == "lost":
        return -trade["limit_price"] * trade["count"]
    return None


def is_loss(trade: dict) -> bool:
    p = pnl_cents(trade)
    return p is not None and p < 0


def is_win(trade: dict) -> bool:
    p = pnl_cents(trade)
    return p is not None and p > 0


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def print_separator(out: TextIO, char: str = "─", width: int = 80) -> None:
    print(char * width, file=out)


def print_header(title: str, out: TextIO) -> None:
    print_separator(out, "═")
    print(f"  {title}", file=out)
    print_separator(out, "═")


def report_per_trade(trades: list[dict], fc_map: dict[str, dict],
                     targeted_ids: set[int], out: TextIO) -> None:
    print_header("PER-TRADE DRILL-DOWN", out)
    for t in trades:
        p = pnl_cents(t)
        result = "WIN " if is_win(t) else ("LOSS" if is_loss(t) else "???")
        targeted = " ◄ TARGETED" if t["id"] in targeted_ids else ""
        flags = compute_flags(t, fc_map.get(t["ticker"], {}))
        flag_str = "  [" + ", ".join(flags) + "]" if flags else ""

        print(f"\n  #{t['id']:<4d} {result}  {t['ticker']:<42s}{targeted}", file=out)
        print(f"         kind={t['opportunity_kind']:<12s} score={t['score']:.2f}"
              f"  p={t['p_estimate']:.4f}  src={t['source']}", file=out)
        if t["corroborating_sources"]:
            print(f"         corr={t['corroborating_sources']}", file=out)
        if t["market_p_entry"] is not None:
            print(f"         mkt_p_YES={t['market_p_entry']:.2f}"
                  f"  yes_bid={t['yes_bid_entry']}¢  yes_ask={t['yes_ask_entry']}¢", file=out)
        city = city_from_ticker(t["ticker"])
        lh = local_hour(t["logged_at"], city)
        print(f"         city={city:<6s} mtype={market_type(t['ticker'])}"
              f"  dir={market_direction(t['ticker']):<8s} local_entry_hour={lh}", file=out)
        print(f"         exit_reason={t['exit_reason']}  pnl={p:+.0f}¢" if p is not None
              else f"         exit_reason={t['exit_reason']}", file=out)
        if flag_str:
            print(f"         flags:{flag_str}", file=out)

        # Per-source forecast summary
        fc = fc_map.get(t["ticker"], {})
        if fc:
            print(f"         forecasts:", file=out)
            for src, info in sorted(fc.items(), key=lambda x: -(x[1]["edge_max"] or 0)):
                e_range = (f"edge=[{info['edge_min']:.1f},{info['edge_max']:.1f}]°F"
                           if info["edge_min"] is not None else "edge=N/A")
                print(f"           {src:<18s} val=[{info['val_min']:.1f},{info['val_max']:.1f}]"
                      f"  {e_range}  n={info['n']}", file=out)


def report_flag_frequencies(losses: list[dict], wins: list[dict],
                             fc_map: dict[str, dict], out: TextIO) -> None:
    print_header("RISK-FLAG FREQUENCY  (losses vs wins)", out)

    all_flags = [
        "THIN_EDGE", "MODEL_DISAGREE", "HRRR_DISSENT", "TERRAIN_CITY",
        "LOW_T_MARKET", "LATE_HIGH_ENTRY", "OVERNIGHT_ENTRY",
        "MARGINAL_MKT_P", "WEAK_CORR", "SOLO_OPENMETEO",
    ]

    loss_flags: dict[str, int] = defaultdict(int)
    win_flags:  dict[str, int] = defaultdict(int)

    for t in losses:
        for f in compute_flags(t, fc_map.get(t["ticker"], {})):
            loss_flags[f] += 1
    for t in wins:
        for f in compute_flags(t, fc_map.get(t["ticker"], {})):
            win_flags[f] += 1

    nl, nw = len(losses), len(wins)
    print(f"\n  {'Flag':<20s}  {'Loss rate':>10s}  {'Win rate':>10s}  {'Lift':>8s}", file=out)
    print_separator(out, "─", 60)
    for flag in all_flags:
        lc = loss_flags[flag]
        wc = win_flags[flag]
        lr = lc / nl if nl else 0
        wr = wc / nw if nw else 0
        lift = lr / wr if wr > 0 else float("inf")
        lift_str = f"{lift:.2f}×" if lift != float("inf") else "  ∞"
        print(f"  {flag:<20s}  {lc:3d}/{nl:<3d} {lr:5.0%}  "
              f"{wc:3d}/{nw:<3d} {wr:5.0%}  {lift_str:>8s}", file=out)
    print(file=out)


def report_city_winrate(losses: list[dict], wins: list[dict], out: TextIO) -> None:
    print_header("CITY WIN RATE", out)
    city_l: dict[str, int] = defaultdict(int)
    city_w: dict[str, int] = defaultdict(int)
    for t in losses:
        city_l[city_from_ticker(t["ticker"])] += 1
    for t in wins:
        city_w[city_from_ticker(t["ticker"])] += 1

    all_cities = sorted(set(city_l) | set(city_w))
    print(f"\n  {'City':<8s}  {'W':>4s}  {'L':>4s}  {'Total':>6s}  {'Win%':>6s}", file=out)
    print_separator(out, "─", 45)
    for city in all_cities:
        w = city_w[city]
        l = city_l[city]
        total = w + l
        pct = w / total if total else 0
        marker = "  ◄ terrain" if city in TERRAIN_CITIES else ""
        print(f"  {city:<8s}  {w:4d}  {l:4d}  {total:6d}  {pct:5.0%}{marker}", file=out)
    print(file=out)


def report_entry_hour_winrate(losses: list[dict], wins: list[dict], out: TextIO) -> None:
    print_header("ENTRY LOCAL HOUR WIN RATE  (binned by 3h)", out)
    bins = [(0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (15, 18), (18, 21), (21, 24)]

    def bucket(lh: int | None) -> str:
        if lh is None:
            return "??"
        for lo, hi in bins:
            if lo <= lh < hi:
                return f"{lo:02d}-{hi:02d}"
        return "??"

    bw: dict[str, int] = defaultdict(int)
    bl: dict[str, int] = defaultdict(int)
    for t in wins:
        city = city_from_ticker(t["ticker"])
        bw[bucket(local_hour(t["logged_at"], city))] += 1
    for t in losses:
        city = city_from_ticker(t["ticker"])
        bl[bucket(local_hour(t["logged_at"], city))] += 1

    all_b = sorted(set(bw) | set(bl))
    print(f"\n  {'Hour window':<12s}  {'W':>4s}  {'L':>4s}  {'Win%':>6s}", file=out)
    print_separator(out, "─", 40)
    for b in all_b:
        w = bw[b]; l = bl[b]
        total = w + l
        pct = w / total if total else 0
        marker = "  ◄ risky" if pct < 0.50 and total >= 3 else ""
        print(f"  {b:<12s}  {w:4d}  {l:4d}  {pct:5.0%}{marker}", file=out)
    print(file=out)


def report_source_winrate(losses: list[dict], wins: list[dict], out: TextIO) -> None:
    print_header("PRIMARY SOURCE WIN RATE  (numeric trades only)", out)
    numeric_l = [t for t in losses if t["opportunity_kind"] == "numeric"]
    numeric_w = [t for t in wins   if t["opportunity_kind"] == "numeric"]

    sw: dict[str, int] = defaultdict(int)
    sl: dict[str, int] = defaultdict(int)
    for t in numeric_w:
        sw[t["source"] or "?"] += 1
    for t in numeric_l:
        sl[t["source"] or "?"] += 1

    all_src = sorted(set(sw) | set(sl))
    print(f"\n  {'Source':<20s}  {'W':>4s}  {'L':>4s}  {'Win%':>6s}", file=out)
    print_separator(out, "─", 45)
    for src in all_src:
        w = sw[src]; l = sl[src]
        total = w + l
        pct = w / total if total else 0
        print(f"  {src:<20s}  {w:4d}  {l:4d}  {pct:5.0%}", file=out)
    print(file=out)


def report_flag_checklist(targeted: list[dict], fc_map: dict[str, dict],
                           out: TextIO) -> None:
    print_header("FLAG CHECKLIST FOR TARGETED LOSSES", out)
    for t in targeted:
        if not is_loss(t):
            continue
        flags = compute_flags(t, fc_map.get(t["ticker"], {}))
        p = pnl_cents(t)
        print(f"\n  #{t['id']}  {t['ticker']}  pnl={p:+.0f}¢", file=out)
        if flags:
            for f in flags:
                desc = FLAG_DESCRIPTIONS.get(f, "")
                print(f"    ✗ {f:<20s} {desc}", file=out)
        else:
            print("    (no flags — loss not explained by current flag set)", file=out)
    print(file=out)


FLAG_DESCRIPTIONS: dict[str, str] = {
    "THIN_EDGE":       "Qualifying edge < 3°F — marginal signal, small NWS rounding can flip outcome",
    "MODEL_DISAGREE":  "open_meteo vs HRRR/NWS spread > 8°F — models fundamentally disagree",
    "HRRR_DISSENT":    "HRRR max edge < 2°F — HRRR nearly on YES side (most accurate model)",
    "TERRAIN_CITY":    "Denver/OKC/DAL: terrain-driven reversals miss NWS hourly / open_meteo",
    "LOW_T_MARKET":    "KXLOWT 'over' direction: betting overnight low won't cool (high variance)",
    "LATE_HIGH_ENTRY": "HIGH market entered at or after 3 PM local — METAR may already be at peak",
    "OVERNIGHT_ENTRY": "LOW market entered after midnight before dawn — overnight low still falling",
    "MARGINAL_MKT_P":  "Market YES bid > 45¢ at entry — market already uncertain about outcome",
    "WEAK_CORR":       "Numeric: 0-1 corroborating sources — single-source conviction is risky",
    "SOLO_OPENMETEO":  "Forecast_no: only open_meteo qualifies, no HRRR/NWS (4-5°F MAE, no terrain)",
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ids", nargs="+", type=int, metavar="ID",
                        help="Specific trade IDs to include in per-trade drill-down")
    parser.add_argument("--kind", nargs="+", metavar="KIND",
                        help="Filter to specific opportunity_kinds (e.g. forecast_no numeric)")
    parser.add_argument("--all-kinds", action="store_true",
                        help="Include band_arb in analysis (excluded by default)")
    parser.add_argument("--summary-only", action="store_true",
                        help="Skip per-trade detail, show aggregated tables only")
    parser.add_argument("--out", metavar="FILE",
                        help="Write report to FILE instead of stdout")
    args = parser.parse_args()

    out: TextIO = open(args.out, "w") if args.out else sys.stdout

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    # Determine which kinds to load
    kinds: list[str] | None = args.kind
    if not args.all_kinds and kinds is None:
        kinds = None  # handled below with exclusion

    # Load all resolved trades
    cur = conn.cursor()
    query = """
        SELECT id, logged_at, ticker, side, count, limit_price,
               opportunity_kind, score, p_estimate, status,
               source, outcome, market_p_entry,
               yes_bid_entry, yes_ask_entry, signal_p_yes,
               corroborating_sources, fill_price_cents,
               exit_price_cents, exit_pnl_cents, exit_reason, peak_past
        FROM trades
        WHERE (outcome IS NOT NULL OR exit_pnl_cents IS NOT NULL)
    """
    params: list = []
    if args.kind:
        placeholders = ",".join("?" * len(args.kind))
        query += f" AND opportunity_kind IN ({placeholders})"
        params.extend(args.kind)
    elif not args.all_kinds:
        query += " AND opportunity_kind != 'band_arb'"
    query += " ORDER BY id"
    cur.execute(query, params)
    cols = [d[0] for d in cur.description]
    all_trades = [dict(zip(cols, row)) for row in cur.fetchall()]

    losses = [t for t in all_trades if is_loss(t)]
    wins   = [t for t in all_trades if is_win(t)]

    # Load raw_forecasts for all trades
    all_tickers = list({t["ticker"] for t in all_trades})
    fc_map: dict[str, dict] = {}
    for ticker in all_tickers:
        rows_raw = conn.execute(
            "SELECT source, direction, data_value, edge FROM raw_forecasts WHERE ticker = ?",
            (ticker,)
        ).fetchall()
        rows = [dict(zip(["source", "direction", "data_value", "edge"], r)) for r in rows_raw]
        fc_map[ticker] = summarise_forecasts(rows)

    # Targeted IDs for highlight and checklist
    targeted_ids: set[int] = set(args.ids) if args.ids else set()
    targeted_trades = [t for t in all_trades if t["id"] in targeted_ids]

    # Print report
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n{'='*80}", file=out)
    print(f"  TRADE AUDIT REPORT  —  generated {now_str}", file=out)
    print(f"  Losses: {len(losses)}   Wins: {len(wins)}   Total: {len(all_trades)}", file=out)
    if targeted_ids:
        print(f"  Targeted IDs: {sorted(targeted_ids)}", file=out)
    print(f"{'='*80}\n", file=out)

    if not args.summary_only:
        if targeted_ids:
            # Show only targeted trades + enough wins to compare
            display_trades = targeted_trades + [w for w in wins if w["id"] not in targeted_ids][:20]
            display_trades.sort(key=lambda t: t["id"])
        else:
            display_trades = all_trades
        report_per_trade(display_trades, fc_map, targeted_ids, out)

    report_flag_frequencies(losses, wins, fc_map, out)
    report_city_winrate(losses, wins, out)
    report_entry_hour_winrate(losses, wins, out)
    report_source_winrate(losses, wins, out)

    if targeted_trades:
        report_flag_checklist(targeted_trades, fc_map, out)

    conn.close()
    if args.out:
        out.close()
        print(f"Report written to {args.out}")


if __name__ == "__main__":
    main()
