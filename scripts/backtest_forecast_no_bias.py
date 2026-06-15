#!/usr/bin/env python3
"""Backtest forecast_no with per-city, per-source bias correction.

Uses only live data from opportunity_log.db — no archived forecast CSVs.

Steps
-----
1. For every forecast source (hrrr, noaa, nws_hourly, open_meteo_*),
   compute per-city cold bias = mean(source_forecast - noaa_observed)
   using midday readings (11-13 UTC) from raw_forecasts.

2. Load every live forecast_no NO trade.  For each trade, parse
   sources_detail from the note JSON: list of [source, forecast_val, edge].

3. For each qualifying source in the trade, apply the city+source bias:
     corrected_edge = raw_edge + bias   (bias is negative → reduces edge)

4. Recompute bias-corrected min edge = min(corrected_edge across sources).

5. Sweep corrected-min-edge threshold and compare to baseline.

Usage
-----
  venv/bin/python scripts/backtest_forecast_no_bias.py
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

DB = Path("data/db/opportunity_log.db")

FORECAST_SOURCES = [
    "hrrr", "noaa", "nws_hourly",
    "open_meteo", "open_meteo_ecmwf", "open_meteo_gem",
    "open_meteo_gfs", "open_meteo_icon",
]

SERIES_TO_METRIC: dict[str, str] = {
    "KXHIGHTATL":  "temp_high_atl",
    "KXHIGHAUS":   "temp_high_aus",
    "KXHIGHTBOS":  "temp_high_bos",
    "KXHIGHTCHI":  "temp_high_chi",
    "KXHIGHTDAL":  "temp_high_dal",
    "KXHIGHTDC":   "temp_high_dca",
    "KXHIGHDEN":   "temp_high_den",
    "KXHIGHTHOU":  "temp_high_hou",
    "KXHIGHTLV":   "temp_high_las",
    "KXHIGHLAX":   "temp_high_lax",
    "KXHIGHMIA":   "temp_high_mia",
    "KXHIGHTMIN":  "temp_high_msp",
    "KXHIGHTNOLA": "temp_high_msy",
    "KXHIGHNY":    "temp_high_ny",
    "KXHIGHTOKC":  "temp_high_okc",
    "KXHIGHPHIL":  "temp_high_phl",
    "KXHIGHTPHX":  "temp_high_phx",
    "KXHIGHTSATX": "temp_high_sat",
    "KXHIGHTSEA":  "temp_high_sea",
    "KXHIGHTSFO":  "temp_high_sfo",
}

_SERIES_RE = re.compile(r"^(KXHIGH[A-Z]+)-")


def series_from_ticker(t: str) -> str | None:
    m = _SERIES_RE.match(t)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Step 1: per-source per-city bias
# ---------------------------------------------------------------------------

def compute_all_biases(con: sqlite3.Connection) -> dict[tuple[str, str], float]:
    """Return {(source, metric): bias_f} for all forecast sources."""
    obs = pd.read_sql("""
        SELECT metric, date(logged_at) AS obs_date, MAX(data_value) AS observed_high
        FROM raw_forecasts
        WHERE source='noaa_observed' AND metric LIKE 'temp_high%'
        GROUP BY metric, obs_date
    """, con)

    bias_map: dict[tuple[str, str], float] = {}
    rows_summary = []

    for src in FORECAST_SOURCES:
        df = pd.read_sql(f"""
            SELECT metric, date(logged_at) AS obs_date, AVG(data_value) AS forecast_f
            FROM raw_forecasts
            WHERE source='{src}' AND metric LIKE 'temp_high%'
              AND CAST(strftime('%H', logged_at) AS INTEGER) BETWEEN 11 AND 13
            GROUP BY metric, obs_date
            HAVING COUNT(*) >= 2
        """, con)
        if df.empty:
            continue
        merged = df.merge(obs, on=["metric", "obs_date"])
        merged["error"] = merged["forecast_f"] - merged["observed_high"]

        # Overall source stats
        rows_summary.append({
            "source": src,
            "n": len(merged),
            "bias": merged["error"].mean(),
            "mae":  merged["error"].abs().mean(),
            "w1":   (merged["error"].abs() <= 1).mean(),
            "w2":   (merged["error"].abs() <= 2).mean(),
        })

        # Per-city bias
        for metric, grp in merged.groupby("metric"):
            if len(grp) >= 5:
                bias_map[(src, metric)] = grp["error"].mean()

    # Print summary table
    print(f"\n{'source':22} {'n':>6} {'bias(°F)':>10} {'MAE(°F)':>9} "
          f"{'within1°':>10} {'within2°':>10}")
    print("-" * 72)
    for r in sorted(rows_summary, key=lambda x: x["bias"]):
        print(f"  {r['source']:22} {r['n']:>6} {r['bias']:>+10.2f} {r['mae']:>9.2f} "
              f"{r['w1']:>9.0%} {r['w2']:>9.0%}")

    return bias_map


# ---------------------------------------------------------------------------
# Step 2: load trades and parse sources_detail
# ---------------------------------------------------------------------------

def load_trades(con: sqlite3.Connection) -> pd.DataFrame:
    trades = pd.read_sql("""
        SELECT
            id, logged_at, ticker, side,
            limit_price AS yes_bid_entry,
            note,
            settled_result,
            COALESCE(exit_pnl_cents, settled_pnl_cents) AS pnl_cents,
            exit_reason
        FROM trades
        WHERE opportunity_kind='forecast_no'
          AND side='no'
          AND COALESCE(exit_pnl_cents, settled_pnl_cents) IS NOT NULL
    """, con)

    def _parse(row):
        try:
            n = json.loads(row["note"]) if row["note"] else {}
        except Exception:
            n = {}
        return pd.Series({
            "min_edge_f":     n.get("min_edge_f"),
            "model_spread_f": n.get("model_spread_f"),
            "source_count":   n.get("source_count"),
            "hours_to_close": n.get("hours_to_close"),
            "sources_detail": n.get("sources_detail") or [],
        })

    parsed = trades.apply(_parse, axis=1)
    trades = pd.concat([trades, parsed], axis=1)
    trades["ser"]    = trades["ticker"].map(series_from_ticker)
    trades["metric"] = trades["ser"].map(SERIES_TO_METRIC)
    trades["won"]    = trades["pnl_cents"] > 0
    return trades


# ---------------------------------------------------------------------------
# Step 3: apply bias correction to each source, recompute min edge
# ---------------------------------------------------------------------------

def apply_bias_correction(
    trades: pd.DataFrame,
    bias_map: dict[tuple[str, str], float],
) -> pd.DataFrame:
    """Add corrected_min_edge_f column to each trade."""

    corrected_mins = []
    raw_mins       = []
    sources_used   = []

    for _, row in trades.iterrows():
        metric   = row["metric"]
        details  = row["sources_detail"]
        if not details or not isinstance(details, list) or not metric:
            corrected_mins.append(None)
            raw_mins.append(row["min_edge_f"])
            sources_used.append(0)
            continue

        corrected_edges = []
        for entry in details:
            if not isinstance(entry, list) or len(entry) < 3:
                continue
            src, raw_edge = entry[0], entry[2]
            if raw_edge is None:
                continue
            # Look up per-city bias first, fall back to overall source bias
            key_city = (src, metric)
            bias = bias_map.get(key_city)
            if bias is None:
                # Fall back to mean bias across cities for this source
                fallbacks = [v for (s, _), v in bias_map.items() if s == src]
                bias = sum(fallbacks) / len(fallbacks) if fallbacks else 0.0
            corrected_edges.append(raw_edge + bias)  # bias negative → reduces edge

        if corrected_edges:
            corrected_mins.append(min(corrected_edges))
            raw_mins.append(min(e[2] for e in details if isinstance(e, list) and len(e) >= 3))
            sources_used.append(len(corrected_edges))
        else:
            corrected_mins.append(None)
            raw_mins.append(row["min_edge_f"])
            sources_used.append(0)

    trades = trades.copy()
    trades["corrected_min_edge_f"] = corrected_mins
    trades["raw_min_edge_parsed"]  = raw_mins
    trades["n_sources_corrected"]  = sources_used
    return trades


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def stats(df: pd.DataFrame, label: str) -> dict:
    n    = len(df)
    wins = int(df["won"].sum())
    pnl  = df["pnl_cents"].sum() / 100.0
    wr   = wins / n if n else 0
    return {"label": label, "n": n, "wins": wins, "wr": wr, "pnl_usd": pnl}


def print_stats(rows: list[dict]) -> None:
    print(f"\n  {'filter':50} {'n':>5} {'wins':>5} {'wr':>7} {'pnl_usd':>9}")
    print("  " + "-" * 78)
    for r in rows:
        print(f"  {r['label']:50} {r['n']:>5} {r['wins']:>5} "
              f"{r['wr']:>6.1%} {r['pnl_usd']:>+9.2f}")


def main() -> None:
    if not DB.exists():
        sys.exit(f"ERROR: {DB} not found")

    con = sqlite3.connect(DB)

    print("=== Step 1: per-source bias from live raw_forecasts ===")
    bias_map = compute_all_biases(con)
    print(f"\n  {len(bias_map)} per-city/source bias entries computed")

    print("\n=== Step 2: loading live forecast_no NO trades ===")
    trades = load_trades(con)
    print(f"  {len(trades)} trades total")

    print("\n=== Step 3: applying multi-source bias correction ===")
    trades = apply_bias_correction(trades, bias_map)
    has_corr = trades["corrected_min_edge_f"].notna()
    print(f"  {has_corr.sum()} trades with corrected min edge")

    # Baseline vs corrected edge sweep
    rows = [stats(trades, "Baseline (all 201 trades, no correction)")]

    with_corr = trades[has_corr]
    rows.append(stats(with_corr, "Has sources_detail (correctable)"))

    for thresh in [-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
        kept = with_corr[with_corr["corrected_min_edge_f"] >= thresh]
        rows.append(stats(kept, f"  corrected min edge ≥ {thresh:+.1f}°F"))

    print("\n=== Threshold sweep: bias-corrected min edge across all sources ===")
    print_stats(rows)

    # Compare: raw min edge vs corrected min edge
    print(f"\n=== Raw vs corrected min edge (where parseable) ===")
    vc = with_corr.dropna(subset=["raw_min_edge_parsed","corrected_min_edge_f"])
    print(f"  Raw min edge:       mean={vc['raw_min_edge_parsed'].mean():.2f}°F")
    print(f"  Corrected min edge: mean={vc['corrected_min_edge_f'].mean():.2f}°F")
    print(f"  Avg reduction:      {(vc['corrected_min_edge_f'] - vc['raw_min_edge_parsed']).mean():+.2f}°F")

    # Per-city breakdown after correction
    print(f"\n=== Per-city: raw edge vs corrected edge vs outcome ===")
    print(f"  {'city':12} {'n':>5} {'raw_min':>9} {'corr_min':>10} {'wr':>7} {'pnl':>8}")
    print("  " + "-" * 57)
    for metric, grp in with_corr.groupby("metric"):
        city = metric.replace("temp_high_","")
        rm   = grp["raw_min_edge_parsed"].mean()
        cm   = grp["corrected_min_edge_f"].mean()
        wr   = grp["won"].mean()
        pnl  = grp["pnl_cents"].sum() / 100
        n    = len(grp)
        print(f"  {city:12} {n:>5} {rm:>+9.2f} {cm:>+10.2f} {wr:>6.1%} {pnl:>+8.2f}")

    # How many trades does each threshold keep, and what's their performance?
    print(f"\n=== How much volume do we lose at each threshold? ===")
    total = len(with_corr)
    print(f"  {'threshold':20} {'kept':>6} {'pct_kept':>10} {'wr':>8} {'pnl':>9}")
    print("  " + "-" * 58)
    for thresh in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        kept = with_corr[with_corr["corrected_min_edge_f"] >= thresh]
        pct  = len(kept) / total if total else 0
        wr   = kept["won"].mean() if len(kept) else 0
        pnl  = kept["pnl_cents"].sum() / 100 if len(kept) else 0
        print(f"  corr_edge ≥ {thresh:.1f}°F      {len(kept):>6} {pct:>9.0%} {wr:>7.1%} {pnl:>+9.2f}")


if __name__ == "__main__":
    main()
