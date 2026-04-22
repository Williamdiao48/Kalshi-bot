"""Backtest for the band-arb YES intraday signal.

Joins three historical datasets to simulate the YES signal hour-by-hour and
evaluate how well the dynamic p_win formula predicts actual win rates.

Datasets required (run collection scripts first):
  data/mesonet_hourly.csv   — running daily max per city per local hour
  data/kxhigh_bands.csv     — resolved 'between' KXHIGH bands with outcomes
  data/nws_cli_actuals.csv  — NWS CLI official daily highs (sanity check)

The simulation mirrors the live signal logic:
  1. At each local hour, check if running_max is inside the band (with buffer)
  2. Skip if hours_to_close > MAX_HOURS_PRELOCK (pre-lock time gate)
  3. Compute dynamic p_win = BASE_P + clearance_factor + time_factor
  4. Record (p_win_formula, actual_won) for calibration analysis

Reports:
  1. Win rate by local hour
  2. Win rate by clearance from nearest band edge
  3. Win rate by band width
  4. P&L simulation under different filters
  5. Formula calibration errors + tuning recommendations

Usage:
  venv/bin/python scripts/backtest_band_arb_yes.py
  venv/bin/python scripts/backtest_band_arb_yes.py --buffer-f 1.5 --min-clearance 1.0
  venv/bin/python scripts/backtest_band_arb_yes.py --city chi bos ny --days 30
  venv/bin/python scripts/backtest_band_arb_yes.py --hour-start 10 --hour-end 16
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path(__file__).parent.parent / "data"

# ── Defaults matching the live signal config ────────────────────────────────
_DEFAULT_BASE_P      = 0.62
_DEFAULT_BUFFER_F    = 1.0
_DEFAULT_MAX_HTC     = 6.0   # BAND_ARB_YES_MAX_HOURS_PRELOCK
_DEFAULT_CLOSE_HOUR  = 23    # Kalshi markets close ~11 PM local


# ── Data loading ─────────────────────────────────────────────────────────────

def load_mesonet(path: Path) -> dict[tuple[str, str, int], float]:
    """Load mesonet_hourly.csv → {(metric, date, hour): running_max_f}."""
    data: dict[tuple[str, str, int], float] = {}
    if not path.exists():
        print(f"ERROR: {path} not found. Run fetch_mesonet_history.py first.")
        sys.exit(1)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (row["city_metric"], row["date"], int(row["local_hour"]))
            data[key] = float(row["running_max_f"])
    return data


def load_bands(path: Path, cutoff_date: str | None = None) -> list[dict]:
    """Load kxhigh_bands.csv → list of band rows."""
    if not path.exists():
        print(f"ERROR: {path} not found. Run fetch_kxhigh_history.py first.")
        sys.exit(1)
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if cutoff_date and row["date"] < cutoff_date:
                continue
            row["strike_lo"]  = float(row["strike_lo"])
            row["strike_hi"]  = float(row["strike_hi"])
            row["band_width"] = float(row["band_width"])
            rows.append(row)
    return rows


def load_actuals(path: Path) -> dict[tuple[str, str], float]:
    """Load nws_cli_actuals.csv → {(metric, date): actual_high_f}."""
    data: dict[tuple[str, str], float] = {}
    if path.exists():
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                data[(row["city_metric"], row["date"])] = float(row["actual_high_f"])
    return data


# ── Core simulation ───────────────────────────────────────────────────────────

def simulate(
    mesonet:    dict[tuple[str, str, int], float],
    bands:      list[dict],
    actuals:    dict[tuple[str, str], float],
    *,
    base_p:          float = _DEFAULT_BASE_P,
    buffer_f:        float = _DEFAULT_BUFFER_F,
    max_htc:         float = _DEFAULT_MAX_HTC,
    close_hour:      int   = _DEFAULT_CLOSE_HOUR,
    hour_start:      int   = 8,
    hour_end:        int   = 16,
    min_clearance:   float = 0.0,
    city_filter:     list[str] | None = None,
    min_yes_ask:     int   = 10,
    max_yes_ask:     int   = 85,
) -> list[dict]:
    """Run the hour-by-hour simulation and return list of observation dicts.

    Each observation represents one (city, date, band, hour) snapshot where
    the signal WOULD have fired — i.e. the running max was inside the band
    within the time gate.
    """
    records = []

    for band in bands:
        metric  = band["metric"]
        bdate   = band["date"]
        result  = band["result"]   # "yes" or "no"
        s_lo    = band["strike_lo"]
        s_hi    = band["strike_hi"]
        b_width = band["band_width"]

        # City filter
        if city_filter:
            suffix = metric.replace("temp_high_", "")
            if suffix not in city_filter:
                continue

        actual_high = actuals.get((metric, bdate))
        actual_won  = (result == "yes")

        for hour in range(hour_start, hour_end + 1):
            running_max = mesonet.get((metric, bdate, hour))
            if running_max is None:
                continue

            # hours to close (approximate: Kalshi closes at close_hour local)
            htc = max(0.0, close_hour - hour)

            # Pre-lock time gate: only fire within max_htc hours of close
            if htc > max_htc:
                continue

            # In-band check with NWS rounding buffer
            in_band = (s_lo + buffer_f) <= running_max <= (s_hi - buffer_f)
            if not in_band:
                continue

            # Clearance from nearest band edge
            min_clear = min(running_max - s_lo, s_hi - running_max)
            if min_clear < min_clearance:
                continue

            # Dynamic p_win formula (mirrors live signal)
            clearance_factor = min(0.20, min_clear / 5.0)
            time_factor      = 0.10 * (1.0 - htc / max(_DEFAULT_MAX_HTC, 0.001))
            p_win_formula    = min(0.92, base_p + clearance_factor + time_factor)

            records.append({
                "metric":          metric,
                "date":            bdate,
                "hour":            hour,
                "strike_lo":       s_lo,
                "strike_hi":       s_hi,
                "band_width":      b_width,
                "running_max":     running_max,
                "actual_high":     actual_high,
                "min_clearance":   round(min_clear, 2),
                "htc":             htc,
                "p_win_formula":   round(p_win_formula, 4),
                "actual_won":      actual_won,
            })

    return records


# ── Reporting helpers ─────────────────────────────────────────────────────────

def _bucket_stats(records: list[dict], key_fn) -> dict:
    """Aggregate records by a key function → {bucket: (n, wins, sum_p_win)}."""
    stats: dict = defaultdict(lambda: [0, 0, 0.0])
    for r in records:
        k = key_fn(r)
        stats[k][0] += 1
        stats[k][1] += int(r["actual_won"])
        stats[k][2] += r["p_win_formula"]
    return stats


def _print_table(title: str, rows: list[tuple]) -> None:
    print(f"\n{'='*60}")
    print(title)
    print("="*60)
    for row in rows:
        print("  " + "  ".join(str(c) for c in row))


def print_report(records: list[dict]) -> dict:
    """Print all reports and return tuning recommendations."""
    if not records:
        print("\nNo simulation records — check data files and filters.")
        return {}

    n_total = len(records)
    wins    = sum(1 for r in records if r["actual_won"])
    avg_p   = sum(r["p_win_formula"] for r in records) / n_total
    cal_err = avg_p - wins / n_total

    print(f"\n{'='*60}")
    print("BAND-ARB YES BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"  Total signal observations : {n_total}")
    print(f"  Unique (city, date, band) : "
          f"{len({(r['metric'], r['date'], r['strike_lo']) for r in records})}")
    print(f"  Overall win rate          : {wins/n_total*100:.1f}%")
    print(f"  Avg formula p_win         : {avg_p:.3f}")
    print(f"  Calibration error         : {cal_err:+.3f} "
          f"({'overestimates' if cal_err > 0 else 'underestimates'})")

    # ── Report 1: by hour ──────────────────────────────────────────────────
    h_stats = _bucket_stats(records, lambda r: r["hour"])
    rows = [("Hour", "N", "Win%", "Avg_clear°F", "Formula_p", "Cal_err")]
    for hour in sorted(h_stats):
        n, w, sp = h_stats[hour]
        avg_clear = sum(r["min_clearance"] for r in records if r["hour"] == hour) / n
        fp = sp / n
        wr = w / n
        rows.append((
            f"{hour:2d}h", n, f"{wr*100:.0f}%",
            f"{avg_clear:.1f}", f"{fp:.3f}", f"{fp-wr:+.3f}",
        ))
    _print_table("REPORT 1 — Win Rate by Hour of Day", rows)

    # ── Report 2: by clearance bucket ─────────────────────────────────────
    def clear_bucket(r):
        c = r["min_clearance"]
        if c < 1:   return "0–1°F"
        if c < 2:   return "1–2°F"
        if c < 3:   return "2–3°F"
        if c < 5:   return "3–5°F"
        return "5°F+"
    _CLEAR_ORDER = ["0–1°F", "1–2°F", "2–3°F", "3–5°F", "5°F+"]

    c_stats = _bucket_stats(records, clear_bucket)
    rows = [("Clearance", "N", "Win%", "Formula_p", "Cal_err")]
    for bucket in _CLEAR_ORDER:
        if bucket not in c_stats:
            continue
        n, w, sp = c_stats[bucket]
        fp = sp / n
        wr = w / n
        rows.append((bucket, n, f"{wr*100:.0f}%", f"{fp:.3f}", f"{fp-wr:+.3f}"))
    _print_table("REPORT 2 — Win Rate by Clearance from Band Edge", rows)

    # ── Report 3: by band width ────────────────────────────────────────────
    def width_bucket(r):
        w = r["band_width"]
        if w <= 1:   return "1°F"
        if w <= 2:   return "2°F"
        if w <= 3:   return "3°F"
        if w <= 4:   return "4°F"
        return "5°F+"

    w_stats = _bucket_stats(records, width_bucket)
    rows = [("Width", "N", "Win%", "Formula_p", "Cal_err")]
    for bucket in ["1°F", "2°F", "3°F", "4°F", "5°F+"]:
        if bucket not in w_stats:
            continue
        n, w, sp = w_stats[bucket]
        fp = sp / n
        wr = w / n
        rows.append((bucket, n, f"{wr*100:.0f}%", f"{fp:.3f}", f"{fp-wr:+.3f}"))
    _print_table("REPORT 3 — Win Rate by Band Width", rows)

    # ── Report 4: P&L simulation (assume 60¢ YES_ask as proxy) ───────────
    def pnl_bucket(name, subset):
        if not subset:
            return
        n = len(subset)
        wins = sum(1 for r in subset if r["actual_won"])
        # Hypothetical: pay YES_ask=60¢, collect 40¢ profit if win, lose 60¢ if loss
        avg_ask = 60  # proxy; real analysis would need live YES_ask at each hour
        net = sum((100 - avg_ask) / 100 if r["actual_won"] else -(avg_ask / 100)
                  for r in subset)
        print(f"  {name:<22s} N={n:4d}  Win%={wins/n*100:.0f}%  "
              f"Net_P&L=${net:+.2f}  (avg_ask={avg_ask}¢ proxy)")

    print(f"\n{'='*60}")
    print("REPORT 4 — Hypothetical P&L (60¢ YES_ask proxy)")
    print("="*60)
    pnl_bucket("All signals",          records)
    pnl_bucket("Clearance ≥ 2°F",      [r for r in records if r["min_clearance"] >= 2.0])
    pnl_bucket("Clearance ≥ 3°F",      [r for r in records if r["min_clearance"] >= 3.0])
    pnl_bucket("Hour ≥ 14 (post-2PM)", [r for r in records if r["hour"] >= 14])
    pnl_bucket("Band width ≥ 2°F",     [r for r in records if r["band_width"] >= 2.0])
    pnl_bucket("Clear≥2 + Hour≥14",    [r for r in records
                                         if r["min_clearance"] >= 2.0 and r["hour"] >= 14])

    # ── Report 5: per-city summary ─────────────────────────────────────────
    city_stats = _bucket_stats(records, lambda r: r["metric"].replace("temp_high_", ""))
    city_rows = [("City", "N", "Win%", "Avg_clear", "Cal_err")]
    for city in sorted(city_stats):
        n, w, sp = city_stats[city]
        avg_clear = sum(r["min_clearance"] for r in records
                        if r["metric"].replace("temp_high_", "") == city) / n
        fp = sp / n
        wr = w / n
        city_rows.append((city, n, f"{wr*100:.0f}%", f"{avg_clear:.1f}°F", f"{fp-wr:+.3f}"))
    _print_table("REPORT 5 — Per-City Summary", city_rows)

    # ── Report 6: formula recommendations ─────────────────────────────────
    print(f"\n{'='*60}")
    print("REPORT 6 — TUNING RECOMMENDATIONS")
    print("="*60)

    recs: dict[str, str] = {}

    overall_wr = wins / n_total
    if cal_err > 0.05:
        new_base = round(_DEFAULT_BASE_P - cal_err, 2)
        recs["BAND_ARB_YES_BASE_P"] = (
            f"{new_base} (was {_DEFAULT_BASE_P}) — formula overestimates by {cal_err:.2f}"
        )
    elif cal_err < -0.05:
        new_base = round(_DEFAULT_BASE_P - cal_err, 2)
        recs["BAND_ARB_YES_BASE_P"] = (
            f"{new_base} (was {_DEFAULT_BASE_P}) — formula underestimates by {-cal_err:.2f}"
        )
    else:
        print(f"  ✓ BASE_P={_DEFAULT_BASE_P} well-calibrated "
              f"(error={cal_err:+.3f}, within ±0.05 tolerance)")

    # Check if narrow bands (1°F) are dragging win rate down
    narrow = [r for r in records if r["band_width"] <= 1.0]
    if narrow:
        narrow_wr = sum(1 for r in narrow if r["actual_won"]) / len(narrow)
        if narrow_wr < 0.65:
            recs["BAND_ARB_YES_BUFFER_F"] = (
                f"1.5 (was 1.0) — 1°F bands win only {narrow_wr*100:.0f}%; "
                "raising buffer excludes them"
            )

    # Check if high-clearance trades outperform significantly
    high_clear = [r for r in records if r["min_clearance"] >= 2.0]
    if high_clear and len(records) > 0:
        hc_wr = sum(1 for r in high_clear if r["actual_won"]) / len(high_clear)
        if hc_wr - overall_wr > 0.08 and len(high_clear) >= 20:
            recs["BAND_ARB_YES_BUFFER_F"] = (
                f"2.0 (was 1.0) — clearance≥2°F wins {hc_wr*100:.0f}% vs "
                f"{overall_wr*100:.0f}% overall; tighter buffer improves quality"
            )

    if recs:
        for var, msg in recs.items():
            print(f"  ⚠ {var} = {msg}")
    else:
        print("  ✓ Current defaults look well-tuned for this dataset.")

    print()
    return recs


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    cutoff_date = (
        (date.today() - timedelta(days=args.days)).isoformat()
        if args.days else None
    )

    mesonet = load_mesonet(DATA_DIR / "mesonet_hourly.csv")
    bands   = load_bands(DATA_DIR / "kxhigh_bands.csv", cutoff_date)
    actuals = load_actuals(DATA_DIR / "nws_cli_actuals.csv")

    print(f"Loaded: {len(mesonet)} mesonet rows, {len(bands)} bands, "
          f"{len(actuals)} NWS actuals")

    records = simulate(
        mesonet, bands, actuals,
        base_p        = args.base_p,
        buffer_f      = args.buffer_f,
        max_htc       = args.max_htc,
        hour_start    = args.hour_start,
        hour_end      = args.hour_end,
        min_clearance = args.min_clearance,
        city_filter   = args.city or None,
    )

    print(f"Simulation produced {len(records)} signal observations.")

    print_report(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backtest the band-arb YES intraday signal."
    )
    parser.add_argument("--days",          type=int,   default=60,
                        help="Only include bands from the last N days (default: 60)")
    parser.add_argument("--buffer-f",      type=float, default=_DEFAULT_BUFFER_F,
                        help=f"Buffer from band edges in °F (default: {_DEFAULT_BUFFER_F})")
    parser.add_argument("--base-p",        type=float, default=_DEFAULT_BASE_P,
                        help=f"Simulate with a different BASE_P (default: {_DEFAULT_BASE_P})")
    parser.add_argument("--max-htc",       type=float, default=_DEFAULT_MAX_HTC,
                        help=f"Pre-lock window hours (default: {_DEFAULT_MAX_HTC})")
    parser.add_argument("--min-clearance", type=float, default=0.0,
                        help="Only include obs with clearance ≥ N°F (default: 0)")
    parser.add_argument("--hour-start",    type=int,   default=8,
                        help="Earliest local hour to simulate (default: 8)")
    parser.add_argument("--hour-end",      type=int,   default=16,
                        help="Latest pre-lock local hour (default: 16)")
    parser.add_argument("--city",          nargs="+",  default=None,
                        help="Filter to city suffixes e.g. --city chi bos ny")
    args = parser.parse_args()
    main(args)
