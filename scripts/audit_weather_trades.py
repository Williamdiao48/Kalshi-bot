"""Weather trade audit script.

Two modes:
  Default  -- recent window across all matching weather trades (sections 0-5, 8)
  Deep-dive -- specific trade IDs with full source readings + price timeline (all sections)

Usage
-----
  # Last 7 days (default)
  venv/bin/python scripts/audit_weather_trades.py

  # Deep-dive on specific trades
  venv/bin/python scripts/audit_weather_trades.py --ids 152 149

  # Filter by source, recent 3 days
  venv/bin/python scripts/audit_weather_trades.py --days 3 --source hrrr

  # Save to file
  venv/bin/python scripts/audit_weather_trades.py --ids 152 --out audit_152.txt
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from typing import IO

from lib import (
    STARTING_CAPITAL_CENTS, DRAWDOWN_WINDOW_HOURS, DRAWDOWN_FULL_REDUCE,
    DRAWDOWN_MIN_FACTOR, TRADE_MIN_SCORE, SCORE_WEIGHT_MIN, ALL_CATS,
    bug_loss_col, bug_loss as _bug_loss, ticker_cat as _ticker_cat,
    note as _note, pnl as _pnl, outcome_label as _outcome_label,
    cost_cents as _cost_cents, win_prob as _win_prob,
    score_factor as _score_factor, kelly_count as _kelly_count,
    has_col,
)

_ALL_CATS = ALL_CATS

def _init_schema(db: sqlite3.Connection) -> None:
    pass  # schema detection now handled per-call via lib.has_col()


def _sep(out: IO, char: str = "─", width: int = 78) -> None:
    print(char * width, file=out)


def _header(out: IO, title: str) -> None:
    print(f"\n{'═'*78}", file=out)
    print(f"  {title}", file=out)
    print(f"{'═'*78}", file=out)


# ── fetch trades ───────────────────────────────────────────────────────────────

def _fetch_trades(db: sqlite3.Connection, *, ids: list[int] | None,
                  days: int, category: str | None, source: str | None) -> list[sqlite3.Row]:
    if ids:
        placeholders = ",".join("?" * len(ids))
        return db.execute(
            f"SELECT * FROM trades WHERE id IN ({placeholders}) ORDER BY id",
            ids,
        ).fetchall()

    params: list = [days]

    rows_all = db.execute(
        f"""
        SELECT * FROM trades
        WHERE mode = 'dry_run'
          AND logged_at >= datetime('now', '-' || ? || ' days')
        ORDER BY id DESC
        """,
        params,
    ).fetchall()

    result = []
    for r in rows_all:
        cat = _ticker_cat(r["ticker"])
        if category:
            if cat != category:
                continue
        else:
            # default: weather only
            if cat not in ("temp_high", "temp_low"):
                continue
        if source and r["source"] != source:
            continue
        result.append(r)

    return list(reversed(result))


def _fetch_all_closed(db: sqlite3.Connection) -> list[sqlite3.Row]:
    """All-time closed trades for balance/peak computation."""
    return db.execute(
        """
        SELECT * FROM trades
        WHERE mode = 'dry_run'
          AND (outcome IN ('won','lost') OR exited_at IS NOT NULL)
          AND outcome IS NOT 'void'
        ORDER BY logged_at ASC
        """
    ).fetchall()


# ── Section 0: Drawdown ────────────────────────────────────────────────────────

def print_drawdown(db: sqlite3.Connection, out: IO) -> None:
    _header(out, f"SECTION 0 — DRAWDOWN STATE  (48h window, bug_loss excluded)")

    rows_48h = db.execute(
        f"""
        SELECT ticker, exit_pnl_cents, outcome, side, count, limit_price
        FROM trades
        WHERE mode = 'dry_run'
          AND (outcome IN ('won','lost') OR exited_at IS NOT NULL)
          AND outcome IS NOT 'void'
          AND COALESCE({bug_loss_col(db)}, 0) = 0
          AND COALESCE(exited_at, logged_at) >= datetime('now', '-48 hours')
        ORDER BY logged_at ASC
        LIMIT 50
        """
    ).fetchall()

    buckets: dict[str, list[float]] = {c: [] for c in _ALL_CATS}
    for r in rows_48h:
        p = _pnl(r)
        if p is None:
            continue
        buckets[_ticker_cat(r["ticker"])].append(p)

    for cat, series in buckets.items():
        n = len(series)
        if n < 3:
            factor, dd_pct = 1.0, 0.0
        else:
            equity, peak = STARTING_CAPITAL_CENTS, STARTING_CAPITAL_CENTS
            for p in series:
                equity += p
                peak = max(peak, equity)
            if peak <= 0:
                factor, dd_pct = DRAWDOWN_MIN_FACTOR, 100.0
            else:
                dd = max(0.0, (peak - equity) / peak)
                dd_pct = dd * 100
                t = min(1.0, dd / DRAWDOWN_FULL_REDUCE)
                factor = max(DRAWDOWN_MIN_FACTOR, 1.0 - t * (1.0 - DRAWDOWN_MIN_FACTOR))

        net = sum(buckets[cat])
        if n < 3:
            note = f"< 3 trades in window → full sizing"
        else:
            note = f"net={net:+.0f}¢"
        print(f"  {cat:<12} factor={factor:.2f}  dd={dd_pct:.1f}%  ({n} trades, {note})", file=out)

    # All-time balance and peak
    all_closed = _fetch_all_closed(db)
    equity = STARTING_CAPITAL_CENTS
    peak   = STARTING_CAPITAL_CENTS
    wins = losses = 0
    for r in all_closed:
        p = _pnl(r)
        if p is None:
            continue
        equity += p
        peak = max(peak, equity)
        if p > 0:
            wins += 1
        else:
            losses += 1

    total = wins + losses
    wr = wins / total if total else 0
    dd_all = max(0.0, (peak - equity) / peak) * 100 if peak > 0 else 0

    open_trades = db.execute(
        "SELECT COUNT(*) FROM trades WHERE mode='dry_run' AND outcome IS NULL AND exited_at IS NULL"
    ).fetchone()[0]

    _sep(out)
    print(f"  Balance: ${equity/100:.2f}  Peak: ${peak/100:.2f}  "
          f"All-time dd: -{dd_all:.1f}%  "
          f"Win rate: {wr:.1%} ({wins}W/{losses}L)  Open: {open_trades}", file=out)


# ── Section 1: Trade table ─────────────────────────────────────────────────────

def print_trade_table(trades: list[sqlite3.Row], out: IO) -> None:
    _header(out, "SECTION 1 — TRADE TABLE")

    hdr = (f"{'ID':>4}  {'Date':11}  {'Ticker':35}  "
           f"{'S':1}  {'Ct':>3}  {'Cost':>5}  {'Outcome':8}  "
           f"{'PnL':>7}  {'Source':14}  {'Sc':>5}  {'p':>5}  "
           f"{'Edge':>5}  {'Dir':7}  {'Data':7}  {'Mkt_p':>5}  {'Exit':11}")
    print(hdr, file=out)
    _sep(out, "-")

    for t in trades:
        n    = _note(t)
        pnl  = _pnl(t)
        pnl_s = f"{pnl:+.0f}¢" if pnl is not None else "open"
        label = _outcome_label(t)
        date  = t["logged_at"][5:16]  # MM-DD HH:MM
        edge  = n.get("edge")
        edge_s = f"{edge:.1f}" if edge is not None else "—"
        data  = n.get("data_value")
        data_s = f"{data:.1f}°" if data is not None else "—"
        mkt_p = t["market_p_entry"]
        mkt_s = f"{mkt_p:.2f}" if mkt_p is not None else "—"
        p_est = t["p_estimate"]
        p_s   = f"{p_est:.2f}" if p_est is not None else "—"
        direction = (n.get("direction") or "—")[:7]
        exit_r = (t["exit_reason"] or "")[:11]
        # flag high-conviction losses or bug trades
        flag = ""
        if t["outcome"] == "lost" and p_est and p_est > 0.90:
            flag = " *"
        elif _bug_loss(t):
            flag = " [BUG]"

        side_s = "Y" if t["side"] == "yes" else "N"
        line = (f"{t['id']:>4}  {date:11}  {t['ticker']:35}  "
                f"{side_s:1}  {t['count']:>3}  {t['limit_price']:>4}¢  "
                f"{label:8}  {pnl_s:>7}  {(t['source'] or ''):14}  "
                f"{t['score']:>5.2f}  {p_s:>5}  "
                f"{edge_s:>5}  {direction:7}  {data_s:>7}  {mkt_s:>5}  {exit_r}")
        print(line + flag, file=out)

    _sep(out, "-")
    print(f"  {len(trades)} trades shown", file=out)


# ── Section 2: Kelly sizing ────────────────────────────────────────────────────

def print_kelly(trades: list[sqlite3.Row], out: IO) -> None:
    _header(out, "SECTION 2 — KELLY SIZING BREAKDOWN")

    for t in trades:
        win_prob = _win_prob(t)
        kf       = t["kelly_fraction"]
        cost     = _cost_cents(t)
        score    = t["score"]
        actual   = t["count"]

        if win_prob is None or kf is None or cost <= 0:
            continue

        raw_f = (win_prob - cost / 100.0) / (1.0 - cost / 100.0)
        sf    = _score_factor(score)

        # Show hypothetical final counts at several pos_max values.
        # When raw_f <= 0 Kelly returns 0, but the caller applies max(1,...) after
        # score weighting — so there is always at least 1 contract placed.
        hypotheticals = []
        for pos_max in (750, 1500, 3000, 5000):
            pre   = _kelly_count(win_prob, cost, kf, pos_max)
            final = max(1, math.floor(pre * sf))
            hypotheticals.append(f"{pos_max}¢→{final}ct")

        if raw_f <= 0:
            raw_f_s = f"{raw_f:.3f} (min 1ct via max(1,...))"
        else:
            raw_f_s = f"{raw_f:.3f}"

        print(f"\n  #{t['id']}  {t['ticker']}  {'YES' if t['side']=='yes' else 'NO'} @ {cost}¢"
              f"  (actual={actual}ct)", file=out)
        print(f"    win_prob={win_prob:.3f}  raw_f={raw_f_s}  kf={kf:.2f}"
              f"  score={score:.2f}  score_factor={sf:.2f}", file=out)
        print(f"    At pos_max: {' | '.join(hypotheticals)}", file=out)


# ── Section 3: Stop-loss analysis ─────────────────────────────────────────────

def print_stoploss(db: sqlite3.Connection, trades: list[sqlite3.Row], out: IO) -> None:
    sl_trades = [t for t in trades if t["exit_reason"] == "stop_loss"]
    if not sl_trades:
        return

    _header(out, "SECTION 3 — STOP-LOSS ANALYSIS")

    for t in sl_trades:
        n        = _note(t)
        entry_p  = t["yes_bid_entry"]   # YES bid at entry
        exit_p   = t["exit_price_cents"]
        mkt_p    = t["market_p_entry"]
        pnl      = _pnl(t)
        outcome  = t["outcome"]
        lim      = t["limit_price"]
        count    = t["count"]
        direction = n.get("direction", "?")
        data_val  = n.get("data_value")
        edge      = n.get("edge")

        # settlement counterfactual
        if outcome == "lost":
            settle_pnl = -(100 - lim) * count if t["side"] == "yes" else -lim * count
            saved = pnl - settle_pnl if pnl is not None else None
            if saved is not None and saved > 0:
                classification = f"[CORRECT EXIT — stop saved {saved:+.0f}¢ vs settlement]"
            else:
                classification = "[CORRECT EXIT]"
        elif outcome == "won":
            settle_pnl = (100 - lim) * count if t["side"] == "yes" else lim * count
            classification = f"[PREMATURE EXIT — would have won +{settle_pnl:.0f}¢ at settlement]"
        else:
            classification = "[UNKNOWN — not yet settled]"

        print(f"\n  #{t['id']}  {t['ticker']}  {t['side'].upper()}  "
              f"entry_bid={entry_p}¢  exit={exit_p}¢  mkt_p={mkt_p:.2f}"
              if mkt_p else f"\n  #{t['id']}  {t['ticker']}  {t['side'].upper()}  "
              f"entry_bid={entry_p}¢  exit={exit_p}¢", file=out)
        data_str = f"{data_val:.1f}°" if data_val is not None else "?"
        edge_str = f"{edge:.1f}°" if edge is not None else "?"
        print(f"    Signal: {t['source']}  data={data_str}  edge={edge_str}  direction={direction}"
              f"  → implied {'NO' if direction in ('between','below') else 'YES'}", file=out)
        print(f"    PnL: {pnl:+.0f}¢  {classification}", file=out)


# ── Section 4: Source accuracy ─────────────────────────────────────────────────

def print_source_accuracy(db: sqlite3.Connection, trades: list[sqlite3.Row], out: IO) -> None:
    _header(out, "SECTION 4 — SOURCE ACCURACY  (closed trades in filter)")

    stats: dict[str, dict] = {}
    for t in trades:
        src = t["source"] or "unknown"
        pnl = _pnl(t)
        if pnl is None:
            continue  # open
        if src not in stats:
            stats[src] = {"n": 0, "wins": 0, "net": 0.0}
        stats[src]["n"]    += 1
        stats[src]["net"]  += pnl
        if pnl > 0:
            stats[src]["wins"] += 1

    if not stats:
        print("  No closed trades in filter.", file=out)
        return

    print(f"  {'Source':<16}  {'Trades':>6}  {'Wins':>5}  {'Win%':>6}  {'Net PnL':>9}", file=out)
    _sep(out, "-", 55)
    for src, s in sorted(stats.items(), key=lambda kv: kv[1]["net"], reverse=True):
        wr = s["wins"] / s["n"] if s["n"] else 0
        print(f"  {src:<16}  {s['n']:>6}  {s['wins']:>5}  {wr:>5.0%}  {s['net']:>+8.0f}¢", file=out)


# ── Section 5: Bug candidates ─────────────────────────────────────────────────

def print_bug_candidates(db: sqlite3.Connection, trades: list[sqlite3.Row], out: IO) -> None:
    _header(out, "SECTION 5 — BUG CANDIDATES")

    marked   = [t for t in trades if _bug_loss(t)]
    unmarked = []

    for t in trades:
        if _bug_loss(t):
            continue
        flags = []
        p = t["p_estimate"]
        pnl = _pnl(t)
        n = _note(t)
        if p and p > 0.90 and t["outcome"] == "lost":
            flags.append("HIGH_CONF_LOSS")
        if t["exit_reason"] == "stop_loss" and t["outcome"] == "won":
            flags.append("PREMATURE_STOP")
        edge = n.get("edge")
        if t["source"] in ("noaa_observed", "metar") and edge is not None and edge < 1.0 \
                and t["outcome"] == "lost":
            flags.append("TINY_EDGE_LOCKED")
        if flags:
            unmarked.append((t, flags))

    if marked:
        print("  Already marked as bug_loss:", file=out)
        for t in marked:
            pnl = _pnl(t)
            pnl_s = f"{pnl:+.0f}¢" if pnl is not None else "open"
            print(f"    #{t['id']:>4}  {t['ticker']:40}  {t['side'].upper()}  "
                  f"{pnl_s:>8}  {t['source']}", file=out)
    else:
        print("  No bug-loss trades in filter.", file=out)

    if unmarked:
        print(file=out)
        print("  Unmarked candidates (heuristic flags):", file=out)
        for t, flags in unmarked:
            pnl = _pnl(t)
            pnl_s = f"{pnl:+.0f}¢" if pnl is not None else "open"
            flag_s = ", ".join(f"[{f}]" for f in flags)
            print(f"    #{t['id']:>4}  {t['ticker']:40}  {pnl_s:>8}  {flag_s}", file=out)
            print(f"           → consider: venv/bin/python scripts/mark_bug_loss.py --ids {t['id']}",
                  file=out)
    else:
        print("  No unmarked bug candidates in filter.", file=out)


# ── Section 6: Source readings ─────────────────────────────────────────────────

def print_source_readings(db: sqlite3.Connection, trade: sqlite3.Row, out: IO) -> None:
    ticker     = trade["ticker"]
    entry_time = trade["logged_at"]
    n          = _note(trade)
    strike_lo  = n.get("strike_lo")
    strike_hi  = n.get("strike_hi")

    rows = db.execute(
        """
        SELECT source, data_value, direction, edge, logged_at
        FROM raw_forecasts
        WHERE ticker = ?
          AND logged_at BETWEEN datetime(?, '-4 hours') AND datetime(?, '+4 hours')
        ORDER BY source, logged_at
        """,
        (ticker, entry_time, entry_time),
    ).fetchall()

    _header(out, f"SECTION 6 — SOURCE READINGS: #{trade['id']} {ticker}  (entry {entry_time[11:16]} UTC)")

    if strike_lo is not None and strike_hi is not None:
        print(f"  Window: ±4h around entry   Strike range: [{strike_lo:.1f}, {strike_hi:.1f}]°F",
              file=out)
    else:
        print(f"  Window: ±4h around entry", file=out)

    if not rows:
        print("  No raw_forecast readings found in this window.", file=out)
        return

    # group by source
    by_source: dict[str, list] = {}
    for r in rows:
        by_source.setdefault(r["source"], []).append(r)

    print(f"\n  {'Source':<16}  {'@Entry':>7}  {'Min':>6}  {'Max':>6}  {'Last':>6}  "
          f"{'Dir':7}  {'Edge':>5}  {'N':>3}", file=out)
    _sep(out, "-", 72)

    for src, readings in sorted(by_source.items()):
        vals  = [r["data_value"] for r in readings]
        edges = [r["edge"] for r in readings if r["edge"] is not None]
        dirs  = [r["direction"] for r in readings if r["direction"]]

        # reading closest to entry time
        closest = min(readings, key=lambda r: abs(
            _ts_delta(r["logged_at"], entry_time)
        ))
        at_entry = closest["data_value"]
        last_dir = dirs[-1] if dirs else "—"
        last_edge = edges[-1] if edges else None

        at_s    = f"{at_entry:.1f}°" if at_entry is not None else "—"
        min_s   = f"{min(vals):.1f}°"
        max_s   = f"{max(vals):.1f}°"
        last_s  = f"{vals[-1]:.1f}°"
        edge_s  = f"{last_edge:.1f}" if last_edge is not None else "—"

        # flag if any reading crossed the strike boundary
        flag = ""
        if strike_lo is not None and strike_hi is not None:
            if any(v >= strike_hi for v in vals):
                flag = " ← CROSSED UPPER"
            elif any(v >= strike_lo for v in vals):
                flag = " ← near lower"

        print(f"  {src:<16}  {at_s:>7}  {min_s:>6}  {max_s:>6}  {last_s:>6}  "
              f"{last_dir[:7]:7}  {edge_s:>5}  {len(readings):>3}{flag}", file=out)


def _ts_delta(a: str, b: str) -> float:
    """Rough timestamp delta in seconds (both ISO strings, no timezone)."""
    import datetime as dt
    try:
        ta = dt.datetime.fromisoformat(a.replace(" ", "T")[:19])
        tb = dt.datetime.fromisoformat(b.replace(" ", "T")[:19])
        return abs((ta - tb).total_seconds())
    except Exception:
        return 999999.0


# ── Section 7: Price timeline ──────────────────────────────────────────────────

def print_price_timeline(db: sqlite3.Connection, trade: sqlite3.Row, out: IO) -> None:
    ticker  = trade["ticker"]
    trade_id = trade["id"]
    side    = trade["side"]
    lim     = trade["limit_price"]
    count   = trade["count"]
    exited  = trade["exited_at"]
    outcome = trade["outcome"]
    exit_r  = trade["exit_reason"]

    rows = db.execute(
        """
        SELECT snapshot_at, yes_bid, yes_ask, pct_gain, post_exit
        FROM price_snapshots
        WHERE trade_id = ?
        ORDER BY snapshot_at
        """,
        (trade_id,),
    ).fetchall()

    _header(out, f"SECTION 7 — PRICE TIMELINE: #{trade_id} {ticker}")

    if not rows:
        print("  No price snapshots found.", file=out)
        return

    exit_label = ""
    if exited:
        exit_label = f"  {exit_r or 'exit'} @ {trade['exit_price_cents']}¢ YES_bid"

    print(f"  Entry: {trade['logged_at'][11:16]} UTC  cost={lim}¢/{side.upper()}{exit_label}", file=out)

    # settlement counterfactual
    if outcome == "won":
        settle_pnl = (100 - lim) * count if side == "yes" else lim * count
        settle_note = f"Settled WIN: +{settle_pnl:.0f}¢"
        if exited:
            actual_pnl = _pnl(trade) or 0
            settle_note += f"  (exited {actual_pnl:+.0f}¢, {'premature stop' if exit_r=='stop_loss' else 'early exit'})"
    elif outcome == "lost":
        settle_pnl = -lim * count if side == "yes" else -(100 - lim) * count
        settle_note = f"Settled LOSS: {settle_pnl:.0f}¢"
        if exited:
            actual_pnl = _pnl(trade) or 0
            saved = actual_pnl - settle_pnl
            settle_note += f"  (stop saved {saved:+.0f}¢)" if saved > 0 else ""
    else:
        settle_note = "Settlement unknown (market still open)"
    print(f"  {settle_note}", file=out)

    print(file=out)
    print(f"  {'Time':7}  {'YES_bid':>7}  {'YES_ask':>7}  {'pct_gain':>9}  note", file=out)
    _sep(out, "-", 55)

    for r in rows:
        time_s    = r["snapshot_at"][11:16] if r["snapshot_at"] else "?"
        bid_s     = f"{r['yes_bid']}¢" if r["yes_bid"] is not None else "—"
        ask_s     = f"{r['yes_ask']}¢" if r["yes_ask"] is not None else "—"
        gain_s    = f"{r['pct_gain']:+.0%}" if r["pct_gain"] is not None else "—"
        note_s    = "* post-exit" if r["post_exit"] else ""
        print(f"  {time_s:7}  {bid_s:>7}  {ask_s:>7}  {gain_s:>9}  {note_s}", file=out)


# ── Section 8: Open positions ──────────────────────────────────────────────────

def print_open_positions(db: sqlite3.Connection, out: IO) -> None:
    rows = db.execute(
        """
        SELECT *, (julianday('now') - julianday(logged_at)) * 24 AS age_hours
        FROM trades
        WHERE mode = 'dry_run'
          AND outcome IS NULL
          AND exited_at IS NULL
        ORDER BY logged_at ASC
        """
    ).fetchall()

    _header(out, f"SECTION 8 — OPEN POSITIONS  ({len(rows)})")

    if not rows:
        print("  No open positions.", file=out)
        return

    for r in rows:
        age   = r["age_hours"]
        side  = "YES" if r["side"] == "yes" else "NO"
        print(f"  #{r['id']:>4}  {r['ticker']:42}  {side} {r['count']}ct @ {r['limit_price']}¢  "
              f"age={age:.1f}h  {r['source']}", file=out)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Audit weather trades from opportunity_log.db")
    parser.add_argument("--days",     type=int, default=7,   help="Trades from last N days (default 7)")
    parser.add_argument("--ids",      type=int, nargs="+",   help="Deep-dive: specific trade IDs")
    parser.add_argument("--category", choices=list(_ALL_CATS), help="Filter by drawdown category")
    parser.add_argument("--source",   type=str,              help="Filter by source column")
    parser.add_argument("--db",       default="opportunity_log.db")
    parser.add_argument("--no-sizing", action="store_true",  help="Skip Kelly breakdown")
    parser.add_argument("--out",      type=str,              help="Write to file instead of stdout")
    args = parser.parse_args()

    db = sqlite3.connect(args.db)
    db.row_factory = sqlite3.Row
    _init_schema(db)

    out: IO = open(args.out, "w") if args.out else sys.stdout

    deep_dive = bool(args.ids)

    trades = _fetch_trades(db, ids=args.ids, days=args.days,
                           category=args.category, source=args.source)

    print_drawdown(db, out)
    print_trade_table(trades, out)
    if not args.no_sizing:
        print_kelly(trades, out)
    print_stoploss(db, trades, out)
    print_source_accuracy(db, trades, out)
    print_bug_candidates(db, trades, out)

    if deep_dive:
        for trade in trades:
            print_source_readings(db, trade, out)
            print_price_timeline(db, trade, out)

    print_open_positions(db, out)

    print(f"\n{'═'*78}", file=out)

    if args.out:
        out.close()
        print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
