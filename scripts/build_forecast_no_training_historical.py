#!/usr/bin/env python3
"""Build historical forecast_no training data for general logistic regression model.

For each city × date (default 2022-01-01 → today-7d), fetches:
  - GFS, ECMWF, GEM, ICON daily high/low forecasts (Open-Meteo Historical Forecast API)
    *** NOT reanalysis — these are actual archived model run outputs ***
  - Actual daily high/low (IEM ASOS ground truth)

Emits ONE ROW per (city, date, direction) for every date that has an actual
temperature — NO minimum edge or source count filtering.  The general model
learns which per-model edge values predict wins; selection criteria are not
baked into the training data.

Features: continuous per-model edge vs GFS-proxy band (°F), model spread,
supporting source count, direction/market flags, month seasonality encoding.

HRRR: Open-Meteo's gfs_hrrr returns GFS values for historical dates (true HRRR
archive is unavailable). edge_hrrr is set to NaN for all historical rows;
the training script fills it to 0.0 (neutral/no signal).

Output: data/backtest/forecast_no_training_historical.csv

Usage:
    venv/bin/python scripts/build_forecast_no_training_historical.py
    venv/bin/python scripts/build_forecast_no_training_historical.py --start 2023-01-01
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
from datetime import date, timedelta
from pathlib import Path

import aiohttp
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from kalshi_bot.cities import CITIES, LOW_CITIES
from build_forecast_calibration import (  # noqa: E402
    IEM_STATIONS,
    OM_MODELS,
    fetch_hrrr_historical,
    fetch_iem_observed,
    fetch_om_historical,
)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

CACHE_FILE = Path(__file__).parent.parent / "data" / "backtest" / "hist_forecast_cache.json"
OUTPUT     = Path(__file__).parent.parent / "data" / "backtest" / "forecast_no_training_historical.csv"

# Filter mirrors the live bot's entry criterion so the model learns
# P(win | signal fired) rather than P(win | any day).
# HRRR excluded: returns GFS values historically, not a real independent source.
MIN_EDGE_THRESHOLD = 0.5   # °F — same as live FORECAST_NO_MIN_EDGE_F
MIN_SOURCES        = 2     # qualifying non-GFS sources

# OM model key → internal source name
_OM_TO_SOURCE = {om_key: src for om_key, src in OM_MODELS.items()}

# Non-GFS models used for qualifying count and edge features
_SIGNAL_SOURCES = ("open_meteo_ecmwf", "open_meteo_icon", "open_meteo_gem")


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_cache() -> dict:
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {"forecast": {}, "actual": {}}


def _save_cache(cache: dict) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)


# ---------------------------------------------------------------------------
# Per-city fetch (async)
# ---------------------------------------------------------------------------

async def _fetch_city(
    session: aiohttp.ClientSession,
    metric: str,
    lat: float,
    lon: float,
    is_high: bool,
    start: str,
    end: str,
    cache: dict,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Return (forecasts_by_source, actuals) for a city.

    forecasts_by_source: {internal_source_name: {date_str: temp_f}}
    actuals: {date_str: temp_f}

    Uses cache to avoid re-fetching data that's already stored.
    """
    fc_cache  = cache["forecast"].setdefault(metric, {})
    obs_cache = cache["actual"].setdefault(metric, {})

    # ---- Forecasts ----
    forecasts: dict[str, dict[str, float]] = {}

    # HRRR — treat empty dict as a cache miss (may have been caused by rate-limit)
    hrrr_key = f"hrrr_{start}_{end}"
    if not fc_cache.get(hrrr_key):
        fc_cache[hrrr_key] = await fetch_hrrr_historical(session, lat, lon, start, end, is_high)
        await asyncio.sleep(1.0)
    forecasts["hrrr"] = fc_cache[hrrr_key]

    # Multi-model OM — same empty-dict guard
    om_key = f"om_{start}_{end}"
    if not fc_cache.get(om_key):
        om_raw = await fetch_om_historical(session, lat, lon, start, end, is_high)
        fc_cache[om_key] = {_OM_TO_SOURCE[k]: v for k, v in om_raw.items()}
        await asyncio.sleep(1.0)
    for src, daily in fc_cache[om_key].items():
        forecasts[src] = daily

    # ---- Actuals (IEM ASOS) ----
    station_info = IEM_STATIONS.get(metric)
    if station_info is None:
        return forecasts, {}

    station, network = station_info
    start_year = int(start[:4])
    end_year   = int(end[:4])
    obs_key    = f"iem_{start_year}_{end_year}"
    if not obs_cache.get(obs_key):
        obs_cache[obs_key] = await fetch_iem_observed(session, station, network, start_year, end_year, is_high)
        await asyncio.sleep(0.5)
    actuals = obs_cache[obs_key]

    return forecasts, actuals


# ---------------------------------------------------------------------------
# Signal simulation
# ---------------------------------------------------------------------------

def _simulate_signals(
    metric: str,
    is_high: bool,
    forecasts: dict[str, dict[str, float]],
    actuals: dict[str, float],
    start: str,
    end: str,
) -> list[dict]:
    """Generate labeled training rows for one city.

    Emits one row per (date, direction) for every date that has an actual
    temperature AND a GFS value — no minimum edge or source count filter.
    The model learns which per-model edge values actually predict wins.

    Band proxy: floor(gfs_val)+1 = strike_hi, floor(gfs_val) = strike_lo.
    This mirrors how Kalshi sets the band near the early-morning GFS forecast.
    edge_gfs is excluded as a feature (always in (-1, 0] by construction).
    edge_hrrr is NaN for all historical rows (gfs_hrrr returns GFS values
    historically; training script fills to 0.0 = neutral/no signal).
    """
    rows: list[dict] = []

    d = date.fromisoformat(start)
    end_d = date.fromisoformat(end)
    city = metric.split("_", 2)[-1]  # e.g. "temp_high_mia" → "mia"

    while d <= end_d:
        ds = d.isoformat()
        month = d.month
        d += timedelta(days=1)

        actual = actuals.get(ds)
        if actual is None:
            continue

        model_vals: dict[str, float] = {}
        for src, daily in forecasts.items():
            v = daily.get(ds)
            if v is not None:
                model_vals[src] = v

        gfs_val = model_vals.get("open_meteo_gfs")
        if gfs_val is None:
            continue

        all_vals = list(model_vals.values())
        model_spread = round(max(all_vals) - min(all_vals), 2)

        strike_hi = math.floor(gfs_val) + 1
        strike_lo = math.floor(gfs_val)

        month_sin = round(math.sin(2 * math.pi * month / 12), 4)
        month_cos = round(math.cos(2 * math.pi * month / 12), 4)

        for direction in ("NO_HIGH", "NO_LOW"):
            strike = strike_hi if direction == "NO_HIGH" else strike_lo
            sign   = 1.0      if direction == "NO_HIGH" else -1.0

            def _edge(src: str, _s=strike, _sg=sign) -> float:
                v = model_vals.get(src)
                return round((v - _s) * _sg, 2) if v is not None else float("nan")

            # Apply the same entry filter as the live bot so the model learns
            # P(win | signal fired), not P(win | any day).
            n_qualifying = sum(
                1 for src in _SIGNAL_SOURCES
                if src in model_vals and (model_vals[src] - strike) * sign >= MIN_EDGE_THRESHOLD
            )
            if n_qualifying < MIN_SOURCES:
                continue

            n_supporting = sum(
                1 for src in _SIGNAL_SOURCES
                if src in model_vals and (model_vals[src] - strike) * sign > 0
            )

            won = (actual > strike_hi) if direction == "NO_HIGH" else (actual < strike_lo)

            rows.append({
                "city":           city,
                "date":           ds,
                "edge_ecmwf":     _edge("open_meteo_ecmwf"),
                "edge_icon":      _edge("open_meteo_icon"),
                "edge_gem":       _edge("open_meteo_gem"),
                "edge_hrrr":      float("nan"),  # unavailable historically
                "model_spread":   model_spread,
                "n_supporting":   n_supporting,
                "is_no_high":     int(direction == "NO_HIGH"),
                "is_high_market": int(is_high),
                "month_sin":      month_sin,
                "month_cos":      month_cos,
                "won":            int(won),
            })

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def _fetch_all(
    start: str,
    end: str,
    cache: dict,
) -> list[dict]:
    """Fetch all cities concurrently and simulate signals."""
    all_rows: list[dict] = []

    # Build city list: high + low markets
    city_jobs = []
    for metric, (_, lat, lon, _) in CITIES.items():
        city_jobs.append((metric, lat, lon, True))
    for metric, (_, lat, lon, _) in LOW_CITIES.items():
        city_jobs.append((metric, lat, lon, False))

    sem = asyncio.Semaphore(1)  # fully sequential — prevents 429s from Open-Meteo

    async def _worker(session: aiohttp.ClientSession, metric: str, lat: float, lon: float, is_high: bool) -> list[dict]:
        async with sem:
            print(f"  Fetching {metric}…", flush=True)
            forecasts, actuals = await _fetch_city(session, metric, lat, lon, is_high, start, end, cache)
            rows = _simulate_signals(metric, is_high, forecasts, actuals, start, end)
            print(f"    → {len(rows)} rows", flush=True)
            _save_cache(cache)  # persist after each city so partial runs survive
            return rows

    async with aiohttp.ClientSession() as session:
        tasks = [_worker(session, metric, lat, lon, is_high) for metric, lat, lon, is_high in city_jobs]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    done = 0
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            metric = city_jobs[i][0]
            print(f"  [warn] {metric}: {result}", file=sys.stderr)
        else:
            all_rows.extend(result)
            done += 1

    print(f"  Fetched {done}/{len(city_jobs)} cities successfully", flush=True)
    return all_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build historical forecast_no training CSV")
    parser.add_argument("--start", default="2022-01-01", help="Start date (YYYY-MM-DD)")
    end_default = (date.today() - timedelta(days=7)).isoformat()
    parser.add_argument("--end", default=end_default, help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    print(f"Building historical training data: {args.start} → {args.end}", flush=True)
    print(f"Loading cache from {CACHE_FILE}…", flush=True)
    cache = _load_cache()

    print("Fetching forecast + actual data for all cities…", flush=True)
    rows = asyncio.run(_fetch_all(args.start, args.end, cache))

    print(f"Saving updated cache…", flush=True)
    _save_cache(cache)

    if not rows:
        print("ERROR: no rows generated — check API connectivity", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)
    df.sort_values(["date", "city"], inplace=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)

    n  = len(df)
    wr = df["won"].mean()
    print(f"\nExported {n:,} rows to {OUTPUT}")
    print(f"Win rate: {wr:.1%}  ({int(df['won'].sum())} wins / {n} total)")
    print(f"\nFeature summary:")
    feat_cols = ["edge_ecmwf", "edge_icon", "edge_gem", "model_spread",
                 "n_supporting", "is_no_high", "is_high_market", "month_sin", "month_cos"]
    print(df[feat_cols].describe().round(2).to_string())
    print(f"\nTop cities by row count:")
    print(df["city"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
