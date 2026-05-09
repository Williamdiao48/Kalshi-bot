#!/usr/bin/env python3
"""Analyze per-source bias stability across years using the local cache.

Reads data/bias_cache/*.json (populated by backtest_openmeteo_bias.py --years 10)
and computes mean bias per source × city × month for rolling year windows.
Helps identify when a model's bias "stabilized" so we can choose the right
lookback for the calibration table.

Usage:
    venv/bin/python scripts/analyze_bias_recency.py
    venv/bin/python scripts/analyze_bias_recency.py --source open_meteo_ecmwf
    venv/bin/python scripts/analyze_bias_recency.py --since 2022
    venv/bin/python scripts/analyze_bias_recency.py --compare-windows 2 3 5 10
    venv/bin/python scripts/analyze_bias_recency.py --emit-table --since 2023
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

CACHE_DIR = Path("data/bias_cache")
MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

SOURCE_ORDER = [
    "open_meteo", "open_meteo_gfs",
    "open_meteo_ecmwf", "open_meteo_icon", "open_meteo_gem",
]

def load_all_rows() -> list[tuple]:
    """Load all cached rows from every city JSON file."""
    rows = []
    for p in sorted(CACHE_DIR.glob("*.json")):
        try:
            data = json.loads(p.read_text())
            rows.extend(tuple(r) for r in data["rows"])
        except Exception as exc:
            print(f"  Warning: could not load {p.name}: {exc}", file=sys.stderr)
    return rows


def filter_rows(rows: list[tuple], since_year: int | None, until_year: int | None) -> list[tuple]:
    out = []
    for r in rows:
        yr = int(r[2][:4])
        if since_year and yr < since_year:
            continue
        if until_year and yr > until_year:
            continue
        out.append(r)
    return out


def bias_by_src_city_month(rows: list[tuple]) -> dict[str, dict[str, dict[int, list[float]]]]:
    """Group bias values: source → city → month → [bias_f]."""
    out: dict[str, dict[str, dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for source, city, date_str, month, fcast, actual, bias in rows:
        out[source][city][int(month)].append(float(bias))
    return out


def mean_bias_table(rows: list[tuple]) -> dict[tuple[str, str, int], float]:
    """Return {(source, city, month): mean_bias} for entries with n >= 10."""
    grouped = bias_by_src_city_month(rows)
    result = {}
    for src, cities in grouped.items():
        for city, months in cities.items():
            for month, vals in months.items():
                if len(vals) >= 10:
                    result[(src, city, month)] = mean(vals)
    return result


def fmt(v: float | None, w: int = 6) -> str:
    if v is None:
        return f"{'--':>{w}}"
    return f"{v:>+{w}.1f}"


# ---------------------------------------------------------------------------
# Command: show year-by-year global mean + rolling-window comparison
# ---------------------------------------------------------------------------

def cmd_compare_windows(rows: list[tuple], windows: list[int], source_filter: str | None) -> None:
    all_years = sorted({int(r[2][:4]) for r in rows})
    max_year  = max(all_years)

    sources = [s for s in SOURCE_ORDER if source_filter is None or s == source_filter]

    for src in sources:
        src_rows = [r for r in rows if r[0] == src]
        if not src_rows:
            continue

        print(f"\n{'═'*70}")
        print(f"  {src}")
        print(f"{'═'*70}")
        print(f"  {'Window':<14}  {'Jan':>5}{'Feb':>6}{'Mar':>6}{'Apr':>6}{'May':>6}{'Jun':>6}{'Jul':>6}{'Aug':>6}{'Sep':>6}{'Oct':>6}{'Nov':>6}{'Dec':>6}  {'Annual':>8}  {'N':>6}")
        print(f"  {'-'*110}")

        for w in sorted(windows, reverse=True):
            since = max_year - w + 1
            w_rows = [r for r in src_rows if int(r[2][:4]) >= since]
            if not w_rows:
                continue

            mo_means = []
            for m in range(1, 13):
                vals = [float(r[6]) for r in w_rows if int(r[3]) == m]
                mo_means.append(mean(vals) if vals else None)

            flat = [v for v in mo_means if v is not None]
            annual = mean(flat) if flat else None
            n = len(w_rows)

            label = f"last {w}yr ({since}+)"
            row = f"  {label:<14}  " + "".join(fmt(v) for v in mo_means)
            row += f"  {fmt(annual, 8)}  {n:>6}"
            print(row)

        # Also show year-by-year for context
        print(f"  {'─'*110}")
        for yr in all_years:
            yr_rows = [r for r in src_rows if int(r[2][:4]) == yr]
            if not yr_rows:
                continue
            mo_means = []
            for m in range(1, 13):
                vals = [float(r[6]) for r in yr_rows if int(r[3]) == m]
                mo_means.append(mean(vals) if vals else None)
            flat = [v for v in mo_means if v is not None]
            annual = mean(flat) if flat else None
            n = len(yr_rows)
            row = f"  {yr:<14}  " + "".join(fmt(v) for v in mo_means)
            row += f"  {fmt(annual, 8)}  {n:>6}"
            print(row)

    print()


# ---------------------------------------------------------------------------
# Command: emit a Python bias table for a given since_year cutoff
# ---------------------------------------------------------------------------

def cmd_emit_table(rows: list[tuple], since_year: int, threshold: float = 1.0) -> None:
    """Print a BIAS_F dict using data from since_year onwards, threshold |bias| >= threshold."""
    filtered = filter_rows(rows, since_year, None)
    table    = mean_bias_table(filtered)

    entries = {k: v for k, v in table.items() if abs(v) >= threshold}
    if not entries:
        print("No entries exceed the threshold.")
        return

    print(f"# Generated from cache: since {since_year}, |bias| >= {threshold}°F")
    print(f"# N qualifying entries: {len(entries)}")
    print(f"BIAS_F: dict[tuple[str, str, int], float] = {{")

    current_src = None
    for src in SOURCE_ORDER:
        src_entries = {(s, c, m): v for (s, c, m), v in entries.items() if s == src}
        if not src_entries:
            continue
        if current_src != src:
            print(f"\n    # {src}")
            current_src = src
        for (s, city, month), bias in sorted(src_entries.items(), key=lambda x: (x[0][1], x[0][2])):
            print(f'    ("{src}", "{city}", {month:2d}): {bias:+.1f},')

    print("}")


# ---------------------------------------------------------------------------
# Command: show per-city stability for a given source
# ---------------------------------------------------------------------------

def cmd_city_stability(rows: list[tuple], source: str) -> None:
    """For each city, show year-by-year annual mean bias to spot outlier cities."""
    src_rows = [r for r in rows if r[0] == source]
    if not src_rows:
        print(f"No data for source: {source}")
        return

    all_years = sorted({int(r[2][:4]) for r in src_rows})
    cities    = sorted({r[1] for r in src_rows})

    print(f"\n=== Per-city annual mean bias: {source} ===")
    print(f"  {'City':<6}" + "".join(f"{yr:>7}" for yr in all_years) + f"  {'σ_yrs':>7}  {'Δ':>7}")
    print("  " + "─" * (6 + 7 * len(all_years) + 16))

    for city in cities:
        city_rows = [r for r in src_rows if r[1] == city]
        yr_means  = []
        row = f"  {city.upper():<6}"
        for yr in all_years:
            vals = [float(r[6]) for r in city_rows if int(r[2][:4]) == yr]
            m    = mean(vals) if vals else None
            row += fmt(m)
            if m is not None:
                yr_means.append(m)
        sig = stdev(yr_means) if len(yr_means) >= 2 else None
        dlt = (yr_means[-1] - yr_means[0]) if len(yr_means) >= 2 else None
        row += f"  {fmt(sig, 7)}  {fmt(dlt, 7)}"
        print(row)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze bias cache for recency/stability")
    p.add_argument("--source",          type=str, default=None,
                   help="Filter to one source (e.g. open_meteo_ecmwf)")
    p.add_argument("--compare-windows", nargs="+", type=int, default=[2, 3, 5, 10],
                   help="Year windows to compare (default: 2 3 5 10)")
    p.add_argument("--city-stability",  action="store_true",
                   help="Show per-city year-by-year breakdown for --source")
    p.add_argument("--emit-table",      action="store_true",
                   help="Emit a BIAS_F Python dict for --since")
    p.add_argument("--since",           type=int, default=None,
                   help="Only use data from this year onwards")
    p.add_argument("--threshold",       type=float, default=1.0,
                   help="Min |bias| to include in emitted table (default: 1.0)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not CACHE_DIR.exists():
        print(f"Cache dir not found: {CACHE_DIR}")
        print("Run: venv/bin/python scripts/backtest_openmeteo_bias.py --years 10")
        sys.exit(1)

    rows = load_all_rows()
    if not rows:
        print("No cached rows found.")
        sys.exit(1)

    print(f"Loaded {len(rows):,} rows from {len(list(CACHE_DIR.glob('*.json')))} city cache files.")

    if args.since:
        rows = filter_rows(rows, args.since, None)
        print(f"Filtered to {len(rows):,} rows from {args.since} onwards.")

    if args.emit_table:
        since = args.since or (max(int(r[2][:4]) for r in rows) - 1)
        cmd_emit_table(rows, since, args.threshold)
        return

    if args.city_stability and args.source:
        cmd_city_stability(load_all_rows(), args.source)
        return

    cmd_compare_windows(rows, args.compare_windows, args.source)


if __name__ == "__main__":
    main()
