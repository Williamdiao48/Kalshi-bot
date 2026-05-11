#!/usr/bin/env python3
"""Forecast bias vs. METAR actuals — per source, per hours-to-close.

Mines raw_forecasts for forecast model predictions and compares them to
METAR ground truth (metar_6hr daily MAX preferred, metar running max fallback).
Outputs a table showing mean bias (forecast − actual °F) per source broken
down by hours-to-close bucket.

Negative bias = model ran cold vs. actual temperature.

Usage:
    venv/bin/python scripts/backtest_forecast_bias.py
    venv/bin/python scripts/backtest_forecast_bias.py --cities ny bos --days 7
    venv/bin/python scripts/backtest_forecast_bias.py --csv /tmp/bias.csv
"""

import argparse
import csv
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.lib import (
    open_db, DEFAULT_DB_PATH,
    FORECAST_SOURCES,
    city_from_metric, build_market_close_utc,
)

# Preferred display columns (hours-to-close), shown right-to-left
HTB_DISPLAY = [24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Forecast bias vs METAR actuals")
    p.add_argument("--days", type=int, default=None,
                   help="Only use last N days of data (default: all)")
    p.add_argument("--cities", nargs="+", default=None,
                   help="Restrict to these city codes (e.g. ny bos chi)")
    p.add_argument("--min-n", type=int, default=3, dest="min_n",
                   help="Min city-date pairs for a source row to appear (default: 3)")
    p.add_argument("--csv", type=str, default=None,
                   help="Also write full detail rows to this CSV path")
    p.add_argument("--no-per-city", action="store_true",
                   help="Skip the per-city breakdown section")
    p.add_argument("--db", type=str, default=None,
                   help="Path to opportunity_log.db (default: auto-detect)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_ground_truths(
    db, since_date: str | None
) -> dict[tuple[str, str], tuple[float, str]]:
    """Return {(metric, date_str): (actual_f, actual_src)}.

    Ground truth is the highest data_value seen across all metar_6hr rows
    for that metric × date — each row represents one 6-hour synoptic period
    max, so MAX across all rows equals the true daily peak.  Falls back to
    the metar running max when metar_6hr has no data for that date.
    """
    result: dict[tuple[str, str], tuple[float, str]] = {}
    for source, label in [("metar_6hr", "metar_6hr"), ("metar", "metar_running_max")]:
        q = (
            "SELECT metric, date(logged_at) AS d, MAX(data_value) "
            "FROM raw_forecasts "
            "WHERE source = ? AND metric LIKE 'temp_%'"
        )
        params: list = [source]
        if since_date:
            q += " AND date(logged_at) >= ?"
            params.append(since_date)
        q += " GROUP BY metric, d"
        for row in db.execute(q, params).fetchall():
            metric, d, max_val = row[0], row[1], row[2]
            key = (metric, d)
            if key not in result and max_val is not None:
                result[key] = (max_val, label)
    return result


def get_hourly_forecasts(db, source: str, since_date: str | None) -> list[tuple]:
    """Return rows of (metric, date_utc, hour_utc 0-23, avg_forecast).

    Groups by UTC date and UTC hour to collapse per-minute logging noise.

    Note: readings that fall after midnight UTC but before local market close
    (0–7h HTC depending on city) are attributed to the next UTC date and will
    not be matched to that trade day.  This only affects the <7h HTC range
    which is the least analytically interesting window.
    """
    q = (
        "SELECT metric, "
        "       date(logged_at) AS d, "
        "       CAST(strftime('%H', substr(logged_at, 1, 19)) AS INTEGER) AS hr, "
        "       AVG(data_value) "
        "FROM raw_forecasts "
        "WHERE source = ? AND metric LIKE 'temp_%'"
    )
    params: list = [source]
    if since_date:
        q += " AND date(logged_at) >= ?"
        params.append(since_date)
    q += " GROUP BY metric, d, hr"
    return db.execute(q, params).fetchall()


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

# (source, city, metric, date_str, htc_bucket, avg_f, actual_f, actual_src, bias_f)
DetailRow = tuple


def _mean_or_none(vals: list[float]) -> float | None:
    return mean(vals) if vals else None


def _fmt(v: float | None, width: int = 7) -> str:
    if v is None:
        s = " -- "
    else:
        s = f"{v:+.1f}"
    return f"{s:>{width}}"


def print_summary_table(rows: list[DetailRow], min_n: int) -> None:
    bias_by: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    city_dates_by: dict[str, set] = defaultdict(set)

    for source, city, metric, d, htc_bucket, avg_f, actual_f, actual_src, bias_f in rows:
        bias_by[source][htc_bucket].append(bias_f)
        city_dates_by[source].add((metric, d))

    sources = [s for s in FORECAST_SOURCES if len(city_dates_by[s]) >= min_n]
    if not sources:
        print("No sources with enough data — try --min-n 1.")
        return

    all_buckets: set[int] = set()
    for s in sources:
        all_buckets.update(bias_by[s].keys())
    display_cols = [h for h in HTB_DISPLAY if h in all_buckets]
    if not display_cols:
        display_cols = sorted(all_buckets, reverse=True)

    col_w = 7
    hdr = f"{'Source':<22}" + "".join(f"{h:>{col_w-1}}h" for h in display_cols) + f"{'N':>6}"
    print("\n=== FORECAST BIAS vs ACTUAL (°F, negative = model ran cold) ===")
    print(hdr)
    print("-" * len(hdr))

    for source in sources:
        n = len(city_dates_by[source])
        cells = "".join(_fmt(_mean_or_none(bias_by[source].get(h, []))) for h in display_cols)
        print(f"{source:<22}{cells}{n:>6}")
    print()


def print_per_city(rows: list[DetailRow], min_n: int) -> None:
    by_city: dict[str, dict[str, dict[tuple, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    actual_src_by: dict[tuple, str] = {}

    for source, city, metric, d, htc_bucket, avg_f, actual_f, actual_src, bias_f in rows:
        by_city[city][source][(metric, d)].append(bias_f)
        actual_src_by[(metric, d)] = actual_src

    print("=== PER-CITY MEAN BIAS (collapsed across htc) ===")
    for city in sorted(by_city):
        print(f"\n── {city.upper()} {'─' * max(1, 38 - len(city))}")
        for source in FORECAST_SOURCES:
            if source not in by_city[city]:
                continue
            all_biases: list[float] = []
            for biases in by_city[city][source].values():
                all_biases.extend(biases)
            n = len(by_city[city][source])
            avg_bias = mean(all_biases)
            std_str = f" ±{stdev(all_biases):.1f}" if len(all_biases) >= 2 else ""
            sample_key = next(iter(by_city[city][source]))
            truth = actual_src_by.get(sample_key, "?")
            print(f"  {source:<24} {avg_bias:+.2f}°F{std_str:<8}  n={n}  truth={truth}")
    print()


def write_csv(rows: list[DetailRow], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "city", "metric", "trade_date", "htc_bucket",
                    "avg_forecast", "actual_f", "actual_src", "bias_f"])
        w.writerows(rows)
    print(f"Detail rows written to {path}  ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    db = open_db(args.db or DEFAULT_DB_PATH)

    since: str | None = None
    if args.days:
        since = (date.today() - timedelta(days=args.days)).isoformat()
        print(f"Using data since {since}  (--days {args.days})")

    print("Loading ground truth from METAR…")
    ground_truths = get_ground_truths(db, since)
    print(f"  {len(ground_truths)} (metric, date) pairs with ground truth")

    detail_rows: list[DetailRow] = []

    for source in FORECAST_SOURCES:
        print(f"  Processing {source}…", end=" ", flush=True)
        hourly = get_hourly_forecasts(db, source, since)
        added = 0
        for row in hourly:
            metric, d_str, hr, avg_f = row[0], row[1], row[2], row[3]
            key = (metric, d_str)
            if key not in ground_truths:
                continue
            actual_f, actual_src = ground_truths[key]
            city = city_from_metric(metric)
            if args.cities and city not in args.cities:
                continue

            trade_date = date.fromisoformat(d_str)
            close_utc = build_market_close_utc(city, trade_date)
            # Use start-of-hour as the reading timestamp (readings are averaged
            # across the hour so mid-hour would be +30 min; difference is <1 bucket)
            logged_utc = datetime(trade_date.year, trade_date.month, trade_date.day,
                                  hr, tzinfo=timezone.utc)
            htc = (close_utc - logged_utc).total_seconds() / 3600
            htc_bucket = int(htc)
            if not (0 <= htc_bucket <= 30):
                continue

            detail_rows.append((
                source, city, metric, d_str, htc_bucket,
                round(avg_f, 3), round(actual_f, 3), actual_src,
                round(avg_f - actual_f, 3),
            ))
            added += 1
        print(f"{added} records")

    if not detail_rows:
        print("No data matched ground truth. Check --days / --cities filters.")
        return

    print_summary_table(detail_rows, args.min_n)
    if not args.no_per_city:
        print_per_city(detail_rows, args.min_n)
    if args.csv:
        write_csv(detail_rows, args.csv)


if __name__ == "__main__":
    main()
