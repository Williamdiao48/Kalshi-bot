#!/usr/bin/env python3
"""Backtest HRRR forecast_no signals suppressed by the edge_min gate.

Reads raw_forecasts from opportunity_log.db for HRRR signals with
edge in [--min-edge, --max-edge) (default 2.0–5.0°F, below the live
FORECAST_NO_MIN_EDGE_F threshold), fetches Mesonet actuals for those
city/dates, and computes theoretical win rate and P&L if those signals
had been taken at market price.

For price, we use the yes_bid logged in raw_forecasts when available;
otherwise we fall back to the median yes_bid from the opportunities table
for the same ticker on the same day, then to a configurable assumption.

Usage:
    venv/bin/python scripts/backtest_hrrr_edge.py
    venv/bin/python scripts/backtest_hrrr_edge.py --min-edge 3.0 --max-edge 5.0
    venv/bin/python scripts/backtest_hrrr_edge.py --db path/to/opportunity_log.db
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import aiohttp

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
DEFAULT_DB   = Path(__file__).parent.parent / "data" / "db" / "opportunity_log.db"
MESONET_URL  = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
MESONET_DELAY = 0.5  # seconds between requests

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Market-prefix → (city_key, station_id, is_low) mapping
# Built from KALSHI_STATION_IDS in noaa.py — low markets share the high station.
# ---------------------------------------------------------------------------
PREFIX_TO_CITY: dict[str, tuple[str, str, bool]] = {
    # prefix              city_key   station   is_low
    "kxhighaus":    ("aus", "KAUS", False),
    "kxhighchi":    ("chi", "KMDW", False),
    "kxhighden":    ("den", "KDEN", False),
    "kxhighlax":    ("lax", "KLAX", False),
    "kxhighmia":    ("mia", "KMIA", False),
    "kxhighny":     ("ny",  "KNYC", False),
    "kxhighphil":   ("phl", "KPHL", False),
    "kxhightatl":   ("atl", "KATL", False),
    "kxhightbos":   ("bos", "KBOS", False),
    "kxhightdal":   ("dfw", "KDFW", False),
    "kxhightdc":    ("dca", "KDCA", False),
    "kxhighthou":   ("hou", "KHOU", False),
    "kxhightlv":    ("las", "KLAS", False),
    "kxhightmin":   ("msp", "KMSP", False),
    "kxhightnola":  ("msy", "KMSY", False),
    "kxhightokc":   ("okc", "KOKC", False),
    "kxhightphx":   ("phx", "KPHX", False),
    "kxhightsatx":  ("sat", "KSAT", False),
    "kxhightsea":   ("sea", "KSEA", False),
    "kxhightsfo":   ("sfo", "KSFO", False),
    # low markets — same stations as corresponding high
    "kxlowtatl":    ("atl", "KATL", True),
    "kxlowtaus":    ("aus", "KAUS", True),
    "kxlowtbos":    ("bos", "KBOS", True),
    "kxlowtchi":    ("chi", "KMDW", True),
    "kxlowtdc":     ("dca", "KDCA", True),
    "kxlowtden":    ("den", "KDEN", True),
    "kxlowthou":    ("hou", "KHOU", True),
    "kxlowtlax":    ("lax", "KLAX", True),
    "kxlowtlv":     ("las", "KLAS", True),
    "kxlowtmia":    ("mia", "KMIA", True),
    "kxlowtmin":    ("msp", "KMSP", True),
    "kxlowtnola":   ("msy", "KMSY", True),
    "kxlowtnyc":    ("ny",  "KNYC", True),
    "kxlowtokc":    ("okc", "KOKC", True),
    "kxlowtphil":   ("phl", "KPHL", True),
    "kxlowtphx":    ("phx", "KPHX", True),
    "kxlowtsatx":   ("sat", "KSAT", True),
    "kxlowtsea":    ("sea", "KSEA", True),
    "kxlowtsfo":    ("sfo", "KSFO", True),
}

# Mesonet city timezone strings (for converting UTC obs to local date)
from zoneinfo import ZoneInfo
CITY_TZ: dict[str, ZoneInfo] = {
    "aus": ZoneInfo("America/Chicago"),
    "chi": ZoneInfo("America/Chicago"),
    "den": ZoneInfo("America/Denver"),
    "lax": ZoneInfo("America/Los_Angeles"),
    "mia": ZoneInfo("America/New_York"),
    "ny":  ZoneInfo("America/New_York"),
    "phl": ZoneInfo("America/New_York"),
    "atl": ZoneInfo("America/New_York"),
    "bos": ZoneInfo("America/New_York"),
    "dfw": ZoneInfo("America/Chicago"),
    "dca": ZoneInfo("America/New_York"),
    "hou": ZoneInfo("America/Chicago"),
    "las": ZoneInfo("America/Los_Angeles"),
    "msp": ZoneInfo("America/Chicago"),
    "msy": ZoneInfo("America/Chicago"),
    "okc": ZoneInfo("America/Chicago"),
    "phx": ZoneInfo("America/Phoenix"),
    "sat": ZoneInfo("America/Chicago"),
    "sea": ZoneInfo("America/Los_Angeles"),
    "sfo": ZoneInfo("America/Los_Angeles"),
}


# ---------------------------------------------------------------------------
# Ticker parsing
# ---------------------------------------------------------------------------

def parse_ticker(ticker: str) -> tuple[str, str, str, float | None, float | None] | None:
    """Return (prefix, date_str, direction, strike_lo, strike_hi) or None.

    For 'between' markets the band is ±0.5°F from the center encoded in the
    ticker (B78.5 → [78.0, 79.0]).  For 'over'/'under' markets strike_lo is
    the threshold and strike_hi is None.
    """
    try:
        dash26 = ticker.index("-26")
    except ValueError:
        return None

    prefix   = ticker[:dash26].lower()
    rest     = ticker[dash26 + 1:]          # e.g. "26MAY07-B78.5"
    parts    = rest.split("-", 1)
    if len(parts) < 2:
        return None
    date_part = parts[0]                    # "26MAY07"
    band_part = parts[1]                    # "B78.5" or "T81"

    # Parse date
    try:
        dt = datetime.strptime(date_part, "%y%b%d")
        date_str = dt.strftime("%Y-%m-%d")
    except ValueError:
        return None

    # Parse band / strike
    if band_part.startswith("B"):
        center    = float(band_part[1:])
        strike_lo = center - 0.5
        strike_hi = center + 0.5
        direction = "between"
    elif band_part.startswith("T"):
        strike_lo = float(band_part[1:])
        strike_hi = None
        direction = "threshold"   # resolved below per direction field
    else:
        return None

    return prefix, date_str, direction, strike_lo, strike_hi


# ---------------------------------------------------------------------------
# DB queries
# ---------------------------------------------------------------------------

def load_suppressed_signals(
    db: sqlite3.Connection,
    min_edge: float,
    max_edge: float,
    high_only: bool,
    source: str = "hrrr",
) -> list[dict]:
    """Load raw_forecast rows below the edge threshold for a given source,
    deduplicated per (ticker, date) by averaging across poll cycles."""
    where_market = "AND rf.ticker LIKE 'KXHIGH%'" if high_only else ""
    rows = db.execute(f"""
        SELECT
            rf.ticker,
            DATE(rf.logged_at)          AS date,
            rf.direction,
            AVG(rf.data_value)          AS hrrr_fc,
            rf.strike                   AS strike_threshold,
            AVG(rf.edge)                AS avg_edge,
            MAX(rf.edge)                AS max_edge,
            COUNT(*)                    AS n_polls,
            AVG(rf.yes_bid)             AS rf_yes_bid,
            MAX(o.yes_bid)              AS opp_yes_bid,
            MAX(o.yes_ask)              AS opp_yes_ask
        FROM raw_forecasts rf
        LEFT JOIN opportunities o
            ON  o.ticker       = rf.ticker
            AND DATE(o.logged_at) = DATE(rf.logged_at)
            AND o.kind         = 'numeric'
        WHERE rf.source  = ?
          AND rf.edge   >= ?
          AND rf.edge    < ?
          {where_market}
        GROUP BY rf.ticker, DATE(rf.logged_at), rf.direction
        HAVING n_polls >= 3
    """, (source, min_edge, max_edge)).fetchall()

    result = []
    for r in rows:
        parsed = parse_ticker(r[0])
        if parsed is None:
            continue
        prefix, date_str, _, strike_lo, strike_hi = parsed

        if prefix not in PREFIX_TO_CITY:
            continue
        city_key, station, is_low = PREFIX_TO_CITY[prefix]

        # Override direction with raw_forecasts direction (more reliable than ticker parse)
        direction = r[2]

        # For threshold markets, strike comes from the DB
        if direction in ("over", "under"):
            strike_lo = r[4]
            strike_hi = None

        # Price: prefer opp_yes_bid (came from live market), else rf_yes_bid
        yes_bid = r[9] or r[8]

        result.append({
            "ticker":    r[0],
            "date":      date_str,
            "prefix":    prefix,
            "city_key":  city_key,
            "station":   station,
            "is_low":    is_low,
            "direction": direction,
            "hrrr_fc":   r[3],
            "strike_lo": strike_lo,
            "strike_hi": strike_hi,
            "avg_edge":  r[5],
            "max_edge":  r[6],
            "n_polls":   r[7],
            "yes_bid":   yes_bid,
        })
    return result


def load_traded_outcomes(db: sqlite3.Connection) -> dict[str, str]:
    """Return {ticker: outcome} for all trades with resolved outcomes."""
    rows = db.execute(
        "SELECT ticker, outcome FROM trades WHERE outcome IS NOT NULL"
    ).fetchall()
    return {r[0]: r[1] for r in rows}


# ---------------------------------------------------------------------------
# Mesonet fetch
# ---------------------------------------------------------------------------

async def fetch_actuals_for_station(
    session: aiohttp.ClientSession,
    station: str,
    city_key: str,
    dates: list[str],
) -> dict[str, tuple[float | None, float | None]]:
    """Return {date_str: (daily_max_f, daily_min_f)} for given dates."""
    dates_sorted = sorted(dates)
    start = datetime.strptime(dates_sorted[0], "%Y-%m-%d")
    end   = datetime.strptime(dates_sorted[-1], "%Y-%m-%d")
    tz    = CITY_TZ.get(city_key, ZoneInfo("America/Chicago"))

    params = {
        "station":     station,
        "data":        "tmpf",
        "year1":       str(start.year),  "month1": str(start.month),  "day1": str(start.day),
        "year2":       str(end.year),    "month2": str(end.month),    "day2": str(end.day),
        "tz":          "UTC",
        "format":      "comma",
        "latlon":      "no",
        "missing":     "M",
        "trace":       "T",
        "direct":      "no",
        "report_type": "3,4",
    }
    try:
        async with session.get(MESONET_URL, params=params,
                               timeout=aiohttp.ClientTimeout(total=120)) as resp:
            resp.raise_for_status()
            text = await resp.text()
    except Exception as exc:
        log.warning("Mesonet failed for %s (%s): %s", city_key, station, exc)
        return {}

    # Group temps by local date
    temps_by_date: dict[str, list[float]] = defaultdict(list)
    for line in text.splitlines():
        if line.startswith("#") or line.startswith("station") or not line.strip():
            continue
        parts = line.split(",")
        if len(parts) < 3:
            continue
        try:
            utc_ts   = datetime.strptime(parts[1].strip(), "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
            temp_str = parts[2].strip()
            if temp_str in ("M", "T", ""):
                continue
            temp_f   = float(temp_str)
        except (ValueError, IndexError):
            continue
        local_date = utc_ts.astimezone(tz).strftime("%Y-%m-%d")
        temps_by_date[local_date].append(temp_f)

    result = {}
    for d, temps in temps_by_date.items():
        if len(temps) < 6:   # require at least 6 obs for a reliable daily stat
            continue
        result[d] = (max(temps), min(temps))
    return result


async def fetch_all_actuals(
    station_dates: dict[tuple[str, str], list[str]],
) -> dict[tuple[str, str], dict[str, tuple[float | None, float | None]]]:
    """Fetch Mesonet for all (station, city_key) pairs. Returns nested dict."""
    results: dict[tuple[str, str], dict[str, tuple[float | None, float | None]]] = {}
    async with aiohttp.ClientSession() as session:
        for (station, city_key), dates in sorted(station_dates.items()):
            print(f"  Fetching {station} ({city_key}) for {len(dates)} date(s)…", flush=True)
            actuals = await fetch_actuals_for_station(session, station, city_key, dates)
            results[(station, city_key)] = actuals
            await asyncio.sleep(MESONET_DELAY)
    return results


# ---------------------------------------------------------------------------
# Win/loss determination
# ---------------------------------------------------------------------------

def determine_outcome(
    signal: dict,
    actual_max: float | None,
    actual_min: float | None,
) -> str | None:
    """Return 'won', 'lost', or None (unknown)."""
    is_low   = signal["is_low"]
    actual   = actual_min if is_low else actual_max
    if actual is None:
        return None

    direction = signal["direction"]
    lo        = signal["strike_lo"]
    hi        = signal["strike_hi"]
    fc        = signal["hrrr_fc"]

    if direction == "between" and lo is not None and hi is not None:
        # HRRR predicts outside the band → NO signal
        if fc > hi:
            # NO_HIGH: actual must also be above hi to win
            return "won" if actual > hi else "lost"
        elif fc < lo:
            # NO_LOW: actual must also be below lo to win
            return "won" if actual < lo else "lost"
        else:
            # HRRR is inside the band — ambiguous direction, skip
            return None

    elif direction == "under" and lo is not None:
        # "will temp be under {lo}?" — NO wins if actual >= lo
        # HRRR forecast above lo → we expect NO
        return "won" if actual >= lo else "lost"

    elif direction == "over" and lo is not None:
        # "will temp be over {lo}?" — NO wins if actual <= lo
        # HRRR forecast below lo → we expect NO
        return "won" if actual <= lo else "lost"

    return None


# ---------------------------------------------------------------------------
# P&L calculation
# ---------------------------------------------------------------------------

DEFAULT_YES_BID = 65   # conservative fallback: NO costs 35¢, wins 65¢

def compute_pnl(outcome: str, yes_bid: float | None) -> float:
    """Return P&L in cents per contract.

    Buying NO: cost = (100 - yes_bid) ¢
    Win:  receive 100¢  → P&L = yes_bid ¢
    Lose: receive 0¢    → P&L = -(100 - yes_bid) ¢
    """
    bid = yes_bid if yes_bid is not None else DEFAULT_YES_BID
    if outcome == "won":
        return float(bid)
    return -(100.0 - bid)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(signals: list[dict], label: str) -> None:
    print(f"\n{'═'*72}")
    print(f"  {label}")
    print(f"{'═'*72}")

    if not signals:
        print("  No signals.\n")
        return

    # Partition by edge bucket
    buckets: dict[str, list[dict]] = {
        "2.0–2.5": [], "2.5–3.0": [], "3.0–3.5": [],
        "3.5–4.0": [], "4.0–4.5": [], "4.5–5.0": [],
    }
    def bucket(e: float) -> str:
        for lo in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]:
            if lo <= e < lo + 0.5:
                return f"{lo:.1f}–{lo+0.5:.1f}"
        return "other"

    for s in signals:
        b = bucket(s["avg_edge"])
        if b in buckets:
            buckets[b].append(s)

    # Header
    print(f"\n  {'Edge':>9}  {'N':>5}  {'Won':>5}  {'WinRate':>8}  "
          f"{'AvgPnL':>8}  {'TotPnL':>9}  {'AvgBid':>8}  {'PriceSrc':>9}")
    print(f"  {'-'*72}")

    grand = {"n": 0, "won": 0, "pnl": [], "priced": 0}

    for bname, sigs in buckets.items():
        resolved = [s for s in sigs if s.get("outcome") in ("won", "lost")]
        if not resolved:
            print(f"  {bname:>9}  {len(sigs):>5}  {'--':>5}  {'--':>8}  {'--':>8}  {'--':>9}  {'--':>8}")
            continue

        won    = sum(1 for s in resolved if s["outcome"] == "won")
        pnls   = [s["pnl"] for s in resolved if s.get("pnl") is not None]
        priced = sum(1 for s in resolved if s.get("yes_bid") is not None)
        avg_bid = mean(s["yes_bid"] for s in resolved if s.get("yes_bid") is not None) \
                  if priced else None

        win_rate = won / len(resolved)
        avg_pnl  = mean(pnls) if pnls else 0.0
        tot_pnl  = sum(pnls)  if pnls else 0.0
        price_src = f"{priced}/{len(resolved)}"

        bid_str = f"{avg_bid:>7.1f}¢" if avg_bid is not None else f"{'--':>8}"
        print(f"  {bname:>9}  {len(resolved):>5}  {won:>5}  "
              f"{win_rate:>7.1%}  {avg_pnl:>+8.1f}¢  {tot_pnl:>+9.1f}¢  "
              f"{bid_str}  {price_src:>9}")

        grand["n"]      += len(resolved)
        grand["won"]    += won
        grand["pnl"]    += pnls
        grand["priced"] += priced

    print(f"  {'-'*72}")
    if grand["n"]:
        gwr  = grand["won"] / grand["n"]
        gapnl = mean(grand["pnl"]) if grand["pnl"] else 0.0
        gtpnl = sum(grand["pnl"])
        print(f"  {'TOTAL':>9}  {grand['n']:>5}  {grand['won']:>5}  "
              f"{gwr:>7.1%}  {gapnl:>+8.1f}¢  {gtpnl:>+9.1f}¢")

    # Also note how many had no Mesonet actual
    n_unknown = sum(1 for s in signals if s.get("outcome") is None)
    if n_unknown:
        print(f"\n  ({n_unknown} signals had no Mesonet actual — excluded from totals)")

    print()


def report_by_direction(signals: list[dict]) -> None:
    dirs = sorted({s["direction"] for s in signals})
    for d in dirs:
        sigs = [s for s in signals if s["direction"] == d]
        report(sigs, f"Direction: {d}")


def report_vs_traded(signals: list[dict], traded_outcomes: dict[str, str]) -> None:
    """Show win rate for the same-ticker signals that WERE traded (reality check)."""
    matched = [(s, traded_outcomes[s["ticker"]]) for s in signals if s["ticker"] in traded_outcomes]
    if not matched:
        return
    print(f"\n  Cross-check: {len(matched)} suppressed tickers also appear in trades table")
    won = sum(1 for _, o in matched if o == "won")
    print(f"  Traded outcome for same tickers: {won}/{len(matched)} won ({won/len(matched):.1%})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest suppressed HRRR edge_min signals")
    p.add_argument("--db",        default=str(DEFAULT_DB), help="SQLite DB path")
    p.add_argument("--min-edge",  type=float, default=2.0,  help="Min HRRR edge (default 2.0)")
    p.add_argument("--max-edge",  type=float, default=5.0,  help="Max HRRR edge (default 5.0)")
    p.add_argument("--source",    default="hrrr",
                   help="Forecast source to backtest (default: hrrr)")
    p.add_argument("--high-only", action="store_true", help="Only KXHIGH markets (skip KXLOWT)")
    p.add_argument("--default-yes-bid", type=int, default=DEFAULT_YES_BID,
                   help=f"Fallback YES bid when market price unknown (default {DEFAULT_YES_BID})")
    return p.parse_args()


async def main_async(args: argparse.Namespace) -> None:
    global DEFAULT_YES_BID
    DEFAULT_YES_BID = args.default_yes_bid

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"DB not found: {db_path}")
        sys.exit(1)

    db = sqlite3.connect(db_path)
    traded_outcomes = load_traded_outcomes(db)

    print(f"Loading suppressed {args.source} signals (edge {args.min_edge}–{args.max_edge}°F)…")
    signals = load_suppressed_signals(db, args.min_edge, args.max_edge, args.high_only, args.source)
    print(f"Found {len(signals)} unique ticker-day signals across "
          f"{len({s['date'] for s in signals})} dates, "
          f"{len({s['city_key'] for s in signals})} cities.")

    if not signals:
        print("Nothing to backtest.")
        return

    # Build (station, city_key) → dates map for Mesonet fetches
    station_dates: dict[tuple[str, str], list[str]] = defaultdict(list)
    for s in signals:
        station_dates[(s["station"], s["city_key"])].append(s["date"])
    # Deduplicate dates per station
    station_dates = {k: sorted(set(v)) for k, v in station_dates.items()}

    print(f"\nFetching Mesonet actuals for {len(station_dates)} stations…")
    actuals_map = await fetch_all_actuals(station_dates)

    # Attach outcomes and P&L to each signal
    for s in signals:
        key = (s["station"], s["city_key"])
        day_actuals = actuals_map.get(key, {}).get(s["date"])
        actual_max = day_actuals[0] if day_actuals else None
        actual_min = day_actuals[1] if day_actuals else None

        s["actual_max"] = actual_max
        s["actual_min"] = actual_min
        s["outcome"]    = determine_outcome(s, actual_max, actual_min)
        s["pnl"]        = compute_pnl(s["outcome"], s["yes_bid"]) if s["outcome"] else None

    # Summary
    n_resolved = sum(1 for s in signals if s["outcome"] is not None)
    print(f"\nResolved {n_resolved}/{len(signals)} signals via Mesonet actuals.")

    # Main report: all signals by edge bucket
    report(signals, f"{args.source} edge_min backtest — all directions (edge {args.min_edge}–{args.max_edge}°F)")

    # Break down by market direction
    report_by_direction(signals)

    # Cross-check against traded outcomes
    report_vs_traded(signals, traded_outcomes)

    # Show best cases: high edge, won, no market price
    needs_price = [s for s in signals if s["outcome"] == "won" and s["yes_bid"] is None]
    if needs_price:
        print(f"  {len(needs_price)} winning signals used fallback price "
              f"(yes_bid={DEFAULT_YES_BID}¢). P&L may be under/over-stated.")

    db.close()


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
