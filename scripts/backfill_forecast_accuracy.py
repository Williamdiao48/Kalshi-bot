#!/usr/bin/env python3
"""Backfill forecast_accuracy from historical raw_forecasts data.

win_rate_tracker._record_forecast_accuracy() only runs at settlement time and
silently skips two classes of trade:
  (A) Trades without sources_detail in their note JSON (band_arb, single-source)
  (B) Trades where METAR ground truth was not in raw_forecasts at settlement time

This script retroactively processes all settled weather trades and inserts
forecast_accuracy rows for any that are missing.

Ground truth priority:
  1. raw_forecasts WHERE source='metar_6hr' (6-hour synoptic max — most reliable)
  2. raw_forecasts WHERE source='metar'     (running daily max)
  3. json_extract(trades.note, '$.observed_f') — stored in some trade notes

Forecast source priority:
  1. sources_detail from trade note JSON   (multi-source forecast_no trades)
  2. raw_forecasts_daily view              (avg_forecast per source per day)

Usage:
    venv/bin/python scripts/backfill_forecast_accuracy.py
    venv/bin/python scripts/backfill_forecast_accuracy.py --dry-run
    venv/bin/python scripts/backfill_forecast_accuracy.py --trade-ids 37 40 83
    venv/bin/python scripts/backfill_forecast_accuracy.py --since 2026-05-01
"""

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from kalshi_bot.market_parser import TICKER_TO_METRIC

DB_PATH = ROOT / "data" / "db" / "opportunity_log.db"

# Sources we treat as ground-truth observations (not forecasts)
_GROUND_TRUTH_SOURCES = {"metar_6hr", "metar", "metar_running_max",
                          "noaa_observed", "nws_climo"}
# Sources we include when falling back to raw_forecasts_daily
_FORECAST_SOURCES = {
    "noaa", "noaa_day2", "nws_hourly", "hrrr",
    "open_meteo", "open_meteo_gfs", "open_meteo_ecmwf",
    "open_meteo_icon", "open_meteo_gem", "weatherapi",
}


def _get_ground_truth(conn: sqlite3.Connection, metric: str, trade_date: str) -> tuple[float | None, str]:
    """Return (actual_f, source_label) using best available ground truth."""
    for source, label in [
        ("metar_6hr",    "metar_6hr"),
        ("metar",        "metar_running_max"),
        ("noaa_observed","noaa_observed"),
    ]:
        row = conn.execute(
            "SELECT MAX(data_value) FROM raw_forecasts "
            "WHERE source = ? AND metric = ? AND date(logged_at) = ?",
            (source, metric, trade_date),
        ).fetchone()
        if row and row[0] is not None:
            return float(row[0]), label
    return None, ""


def _get_forecasts_from_note(note_json: str | None) -> list[tuple[str, float]]:
    """Extract (source, forecast_f) pairs from trade note sources_detail."""
    if not note_json:
        return []
    try:
        note = json.loads(note_json)
    except (json.JSONDecodeError, TypeError):
        return []
    sources_detail = note.get("sources_detail", [])
    if not sources_detail:
        # Single-source trades store data_value directly
        val = note.get("data_value")
        return [] if val is None else []  # handled by raw_forecasts_daily fallback
    return [(src, float(val)) for src, val, *_ in sources_detail
            if src not in _GROUND_TRUTH_SOURCES]


def _get_forecasts_from_db(conn: sqlite3.Connection, ticker: str, trade_date: str) -> list[tuple[str, float]]:
    """Fall back to raw_forecasts_daily for trades without sources_detail."""
    rows = conn.execute(
        "SELECT source, avg_forecast FROM raw_forecasts_daily "
        "WHERE ticker = ? AND date = ? AND source NOT IN ({})".format(
            ",".join("?" * len(_GROUND_TRUTH_SOURCES))
        ),
        (ticker, trade_date, *_GROUND_TRUTH_SOURCES),
    ).fetchall()
    return [(src, float(avg)) for src, avg in rows
            if avg is not None and src in _FORECAST_SOURCES]


def _already_recorded(conn: sqlite3.Connection, trade_id: int, source: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM forecast_accuracy WHERE trade_id = ? AND source = ?",
        (trade_id, source),
    ).fetchone()
    return row is not None


def backfill(
    conn: sqlite3.Connection,
    dry_run: bool = False,
    trade_ids: list[int] | None = None,
    since: str | None = None,
) -> None:
    # Build query for settled weather trades
    query = """
        SELECT id, ticker, source, logged_at, note, outcome, opportunity_kind
        FROM trades
        WHERE outcome IS NOT NULL
          AND outcome != 'void'
          AND (ticker LIKE 'KXHIGH%' OR ticker LIKE 'KXLOWT%')
    """
    params: list = []
    if trade_ids:
        placeholders = ",".join("?" * len(trade_ids))
        query += f" AND id IN ({placeholders})"
        params.extend(trade_ids)
    if since:
        query += " AND logged_at >= ?"
        params.append(since)
    query += " ORDER BY id"

    trades = conn.execute(query, params).fetchall()
    print(f"Processing {len(trades)} settled weather trade(s)…\n")

    inserted = skipped_no_truth = skipped_no_sources = skipped_duplicate = 0

    now_str = datetime.now(timezone.utc).isoformat()

    for trade_id, ticker, trade_source, logged_at, note_json, outcome, opp_kind in trades:
        prefix = ticker.split("-")[0]
        metric = TICKER_TO_METRIC.get(prefix)
        if not metric or not metric.startswith(("temp_high_", "temp_low_")):
            continue
        city = metric.split("_")[-1]
        trade_date = (logged_at or "")[:10]
        if not trade_date:
            continue

        actual_f, actual_src = _get_ground_truth(conn, metric, trade_date)
        if actual_f is None:
            # Last-resort: check note for observed_f (stored by some signal types)
            try:
                note = json.loads(note_json or "{}")
                obs = note.get("observed_f") or note.get("obs_temp_f")
                if obs is not None:
                    actual_f, actual_src = float(obs), "note_observed_f"
            except (json.JSONDecodeError, TypeError):
                pass

        if actual_f is None:
            print(f"  #{trade_id:4d}  {ticker}  — SKIP: no ground truth for {metric} on {trade_date}")
            skipped_no_truth += 1
            continue

        # Try note sources_detail first; fall back to raw_forecasts_daily
        source_pairs = _get_forecasts_from_note(note_json)
        if not source_pairs:
            source_pairs = _get_forecasts_from_db(conn, ticker, trade_date)

        if not source_pairs:
            print(f"  #{trade_id:4d}  {ticker}  — SKIP: no forecast sources found ({opp_kind})")
            skipped_no_sources += 1
            continue

        note_obj = {}
        try:
            note_obj = json.loads(note_json or "{}")
        except (json.JSONDecodeError, TypeError):
            pass
        hours_to_close = note_obj.get("hours_to_close")

        trade_inserted = 0
        for src, forecast_f in source_pairs:
            if _already_recorded(conn, trade_id, src):
                skipped_duplicate += 1
                continue
            bias_f = round(forecast_f - actual_f, 3)
            if not dry_run:
                conn.execute(
                    """INSERT INTO forecast_accuracy
                       (settled_at, trade_id, source, metric, city,
                        forecast_f, actual_f, actual_src, bias_f, hours_to_close, outcome)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (now_str, trade_id, src, metric, city,
                     forecast_f, actual_f, actual_src, bias_f, hours_to_close, outcome),
                )
            inserted += 1
            trade_inserted += 1

        status = "DRY" if dry_run else "INS"
        print(
            f"  #{trade_id:4d}  {ticker:<35}  actual={actual_f:.1f}°F ({actual_src})"
            f"  sources={trade_inserted}  [{status}]"
        )

    print(f"\n{'─'*60}")
    print(f"  Inserted:           {inserted}")
    print(f"  Skipped (no truth): {skipped_no_truth}")
    print(f"  Skipped (no src):   {skipped_no_sources}")
    print(f"  Skipped (dup):      {skipped_duplicate}")

    if not dry_run and inserted > 0:
        # Verify final count
        total = conn.execute("SELECT COUNT(*) FROM forecast_accuracy").fetchone()[0]
        print(f"\n  forecast_accuracy total rows: {total}")

        print("\n  Per-source bias summary:")
        rows = conn.execute(
            "SELECT source, COUNT(*), ROUND(AVG(bias_f),2), ROUND(AVG(ABS(bias_f)),2) "
            "FROM forecast_accuracy GROUP BY source ORDER BY AVG(bias_f)"
        ).fetchall()
        print(f"  {'Source':<22} {'N':>5}  {'Avg bias':>9}  {'MAE':>6}")
        print(f"  {'─'*22}  {'─'*5}  {'─'*9}  {'─'*6}")
        for src, n, avg_bias, mae in rows:
            print(f"  {src:<22} {n:>5}  {avg_bias:>+9.2f}°F  {mae:>5.2f}°F")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be inserted without writing")
    parser.add_argument("--trade-ids", type=int, nargs="+", metavar="ID",
                        help="Only process these trade IDs")
    parser.add_argument("--since", metavar="YYYY-MM-DD",
                        help="Only process trades logged on or after this date")
    parser.add_argument("--db", default=str(DB_PATH),
                        help="Path to opportunity_log.db")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")

    if args.dry_run:
        print("DRY RUN — no rows will be written\n")

    try:
        backfill(conn, dry_run=args.dry_run, trade_ids=args.trade_ids, since=args.since)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
