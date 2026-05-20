#!/usr/bin/env python3
"""Comprehensive forecast_no historical backtest.

Replays the find_forecast_nos() signal logic against the raw_forecasts table
(4.5M rows, May 1–20 2026) to reconstruct every entry that would have fired
under any parameter combination, then simulates exit PnL using Kalshi fill
histories fetched from the trades API.

Data pipeline:
  A. Load raw_forecasts → build per-tick obs state (running daily max/min)
  B. Reconstruct per-(ticker, tick) signal snapshots with per-source edges
  C. Fetch Kalshi fill price series for all unique tickers (cached)
  D. Find first-entry per (ticker, min_edge, min_sources)
  E. Simulate hold-to-settlement and stop-loss/profit-take exits
  F. Parameter sweep + report

Cross-validation: "current policy" config (min_edge=3.0, min_sources=2,
require_om=True) should reproduce ~60-68% WR matching the 134 live dry-run trades.
"""

from __future__ import annotations

import asyncio
import bisect
import itertools
import json
import re
import sqlite3
import sys
import time
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import aiohttp
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from kalshi_bot.auth import generate_headers
from kalshi_bot.cities import CITY_TZ
from kalshi_bot.openmeteo_bias_table import BIAS_F

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

DB         = Path(__file__).parent.parent / "data" / "db" / "opportunity_log.db"
CACHE_FILE = Path(__file__).parent.parent / "data" / "forecast_no_hist_cache.json"

# Sources that contribute to forecast_no signal (mirrors _FORECAST_NO_SOURCES)
FORECAST_SOURCES = frozenset({
    "hrrr", "nws_hourly", "noaa",
    "open_meteo", "open_meteo_ecmwf", "open_meteo_gem", "open_meteo_icon",
})
OM_SOURCES = frozenset({
    "open_meteo", "open_meteo_ecmwf", "open_meteo_gem", "open_meteo_icon",
})

# Observed sources for running daily max / min
OBS_HIGH_SOURCES = frozenset({"metar", "nws_asos"})
OBS_LOW_SOURCES  = frozenset({"metar", "noaa_observed"})

MAX_RATE      = 2    # concurrent Kalshi API requests (4 caused constant 429s)
BIN_SECONDS   = 90  # poll cadence bucket size for fill time series

# Entry price gate (mirrors live bot defaults)
MIN_NO_ASK = 15   # ¢  — don't enter if NO too cheap (market nearly settled)
MAX_NO_ASK = 80   # ¢  — don't enter if NO too expensive (already priced in)

# Simulated contract count (PnL in cents, single contract)
CONTRACTS = 1

# HRRR terrain cities (require stronger HRRR confirmation)
HRRR_TERRAIN_CITIES = frozenset({"den"})
HRRR_TERRAIN_MIN_EDGE_F = 2.0

# ---------------------------------------------------------------------------
# Ticker parsing helpers
# ---------------------------------------------------------------------------

_MON = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
        "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}

# Map ticker prefix → canonical metric_suffix (city abbreviation)
_SERIES_TO_CITY: dict[str, str] = {
    "KXHIGHLAX": "lax", "KXHIGHDEN": "den", "KXHIGHCHI": "chi",
    "KXHIGHNY":  "ny",  "KXHIGHMIA": "mia", "KXHIGHDAL": "dal",
    "KXHIGHBOS": "bos", "KXHIGHAUS": "aus", "KXHIGHOU":  "hou",
    "KXHIGHTSFO":"sfo", "KXHIGHTSEA":"sea", "KXHIGHTBOS":"bos",
    "KXHIGHTPHX":"phx", "KXHIGHPHIL":"phl", "KXHIGHTATL":"atl",
    "KXHIGHTMIN":"msp", "KXHIGHTDC": "dca", "KXHIGHTLV": "las",
    "KXHIGHTOKC":"okc", "KXHIGHTDAL":"dfw", "KXHIGHTSATX":"sat",
    "KXHIGHTHOU":"hou", "KXHIGHTNOLA":"msy",
    # LOW series
    "KXLOWTLAX": "lax", "KXLOWTDEN": "den", "KXLOWTCHI": "chi",
    "KXLOWTNYC": "ny",  "KXLOWTMIA": "mia", "KXLOWTBOS": "bos",
    "KXLOWTPHX": "phx", "KXLOWTATL": "atl", "KXLOWTHOU": "hou",
    "KXLOWTAUS": "aus", "KXLOWTDC":  "dca", "KXLOWTLV":  "las",
    "KXLOWTMIN": "msp", "KXLOWTNOLA":"msy", "KXLOWTOKC": "okc",
    "KXLOWTPHIL":"phl", "KXLOWTSATX":"sat", "KXLOWTSEA": "sea",
    "KXLOWTSFO": "sfo",
}


def _parse_ticker(ticker: str) -> dict | None:
    """Return parsed fields dict or None if unrecognised."""
    # Find the series prefix (up to the date segment)
    parts = ticker.split("-")
    if len(parts) < 3:
        return None
    series  = parts[0]
    datestr = parts[1]
    suffix  = parts[2]

    m = re.fullmatch(r"(\d{2})([A-Z]{3})(\d{2})", datestr)
    if not m:
        return None
    yr, mon_str, dy = m.groups()
    mon = _MON.get(mon_str)
    if mon is None:
        return None
    try:
        mkt_date = date(2000 + int(yr), mon, int(dy))
    except ValueError:
        return None

    city_abbr = _SERIES_TO_CITY.get(series)
    if city_abbr is None:
        return None

    is_high = "KXHIGH" in series
    metric  = ("temp_high_" if is_high else "temp_low_") + city_abbr

    # Parse band type and center/strike
    bm = re.fullmatch(r"([BT])([\d.]+)", suffix)
    if not bm:
        return None
    band_type   = bm.group(1)           # "B" = between, "T" = threshold
    band_number = float(bm.group(2))

    if band_type == "B":
        direction  = "between"
        strike_lo  = band_number - 1.0
        strike_hi  = band_number + 1.0
    else:  # T-market
        # "T" markets: YES wins if temp > strike (over) for HIGH or temp < strike (under) for LOW
        direction  = "under" if is_high else "over"
        strike_lo  = None
        strike_hi  = None
        band_number = band_number  # = strike

    tz = CITY_TZ.get("temp_high_" + city_abbr)

    return {
        "series":      series,
        "city":        city_abbr,
        "metric":      metric,
        "mkt_date":    mkt_date,
        "band_type":   band_type,
        "band_number": band_number,
        "direction":   direction,
        "strike":      band_number if band_type == "T" else None,
        "strike_lo":   strike_lo,
        "strike_hi":   strike_hi,
        "is_high":     is_high,
        "tz":          tz,
    }


# ---------------------------------------------------------------------------
# Phase A — Load raw_forecasts
# ---------------------------------------------------------------------------

def load_raw_forecasts() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (df_fc, df_obs): forecast and observed rows from raw_forecasts."""
    print("A1: Loading raw_forecasts from SQLite…", flush=True)
    t0 = time.time()
    conn = sqlite3.connect(DB)

    fc_sources  = ",".join(f"'{s}'" for s in FORECAST_SOURCES)
    obs_sources = ",".join(f"'{s}'" for s in OBS_HIGH_SOURCES | OBS_LOW_SOURCES)

    df_fc = pd.read_sql(f"""
        SELECT logged_at, source, metric, ticker, data_value, edge, direction
        FROM raw_forecasts
        WHERE source IN ({fc_sources})
          AND (ticker LIKE 'KXHIGH%' OR ticker LIKE 'KXLOW%')
          AND edge IS NOT NULL
    """, conn)

    df_obs = pd.read_sql(f"""
        SELECT logged_at, source, metric, data_value
        FROM raw_forecasts
        WHERE source IN ({obs_sources})
          AND (metric LIKE 'temp_high_%' OR metric LIKE 'temp_low_%')
          AND data_value IS NOT NULL
    """, conn)

    conn.close()

    df_fc["logged_at"]  = pd.to_datetime(df_fc["logged_at"],  utc=True)
    df_obs["logged_at"] = pd.to_datetime(df_obs["logged_at"], utc=True)

    # Time bucket: floor to nearest minute
    df_fc["tick"]  = df_fc["logged_at"].dt.floor("min")
    df_obs["tick"] = df_obs["logged_at"].dt.floor("min")

    print(f"   Forecast rows: {len(df_fc):,}  Obs rows: {len(df_obs):,}  ({time.time()-t0:.0f}s)", flush=True)
    return df_fc, df_obs


# ---------------------------------------------------------------------------
# Phase A2 — Build running obs state per (metric, date)
# ---------------------------------------------------------------------------

def build_obs_state(df_obs: pd.DataFrame) -> dict[tuple[str, date], list[tuple[datetime, float]]]:
    """Return {(metric, date): [(tick_ts, running_max_or_min), ...]} sorted by tick.

    For HIGH metrics: uses metar/nws_asos → running max (cummax within each date).
    For LOW metrics:  uses metar/noaa_observed → running min (cummin within each date).
    """
    print("A2: Building running obs state…", flush=True)
    t0 = time.time()

    df_obs = df_obs.copy()
    df_obs["date"] = df_obs["logged_at"].dt.date
    df_obs["is_high"] = df_obs["metric"].str.startswith("temp_high_")

    obs_state: dict[tuple[str, date], list[tuple[datetime, float]]] = {}

    # HIGH: running max from metar/nws_asos
    df_high_obs = df_obs[
        df_obs["is_high"] & df_obs["source"].isin(OBS_HIGH_SOURCES)
    ].copy()
    if not df_high_obs.empty:
        df_high_obs.sort_values(["metric", "date", "tick"], inplace=True)
        df_high_obs["cum_obs"] = df_high_obs.groupby(["metric", "date"])["data_value"].cummax()
        for (metric, d), grp in df_high_obs.groupby(["metric", "date"]):
            obs_state[(metric, d)] = list(zip(grp["tick"].tolist(), grp["cum_obs"].tolist()))

    # LOW: running min from metar/noaa_observed
    df_low_obs = df_obs[
        ~df_obs["is_high"] & df_obs["source"].isin(OBS_LOW_SOURCES)
    ].copy()
    if not df_low_obs.empty:
        df_low_obs.sort_values(["metric", "date", "tick"], inplace=True)
        df_low_obs["cum_obs"] = df_low_obs.groupby(["metric", "date"])["data_value"].cummin()
        for (metric, d), grp in df_low_obs.groupby(["metric", "date"]):
            obs_state[(metric, d)] = list(zip(grp["tick"].tolist(), grp["cum_obs"].tolist()))

    print(f"   Built obs state for {len(obs_state):,} (metric, date) pairs ({time.time()-t0:.0f}s)", flush=True)
    return obs_state


def _obs_at(obs_timeline: list[tuple[datetime, float]], tick: datetime) -> float | None:
    """Binary search: latest obs value at or before tick."""
    if not obs_timeline:
        return None
    # obs_timeline sorted by tick
    ticks = [t for t, _ in obs_timeline]
    idx = bisect.bisect_right(ticks, tick) - 1
    return obs_timeline[idx][1] if idx >= 0 else None


# ---------------------------------------------------------------------------
# Phase A3+A4 — Fetch Kalshi fills + settlements (with cache)
# ---------------------------------------------------------------------------

async def _api_get(session: aiohttp.ClientSession, path: str, params: dict,
                   _retries: int = 0) -> dict:
    headers = generate_headers("GET", path)
    url = "https://api.elections.kalshi.com" + path
    async with session.get(url, headers=headers, params=params,
                           timeout=aiohttp.ClientTimeout(total=20)) as r:
        if r.status == 429:
            if _retries >= 10:
                raise RuntimeError(f"429 after 10 retries: {path}")
            wait = 2 * (2 ** min(_retries, 4))
            print(f"  [429] waiting {wait}s…", flush=True)
            await asyncio.sleep(wait)
            return await _api_get(session, path, params, _retries + 1)
        r.raise_for_status()
        return await r.json()


async def _fetch_fills(session: aiohttp.ClientSession, ticker: str) -> list[tuple[datetime, int]]:
    """Return (timestamp, yes_price_cents) sorted chronologically."""
    fills: list[tuple[datetime, int]] = []
    cursor = None
    while True:
        params: dict = {"ticker": ticker, "limit": 100}
        if cursor:
            params["cursor"] = cursor
        try:
            d = await _api_get(session, "/trade-api/v2/markets/trades", params)
        except Exception as exc:
            print(f"  [warn] fills {ticker}: {exc}", flush=True)
            break
        for t in d.get("trades", []):
            try:
                ts    = datetime.fromisoformat(t["created_time"].replace("Z", "+00:00"))
                price = round(float(t["yes_price_dollars"]) * 100)
                fills.append((ts, price))
            except Exception:
                continue
        cursor = d.get("cursor")
        if not cursor or not d.get("trades"):
            break
    fills.sort(key=lambda x: x[0])
    return fills


def _infer_settlement(fills: list[tuple[datetime, int]]) -> str | None:
    """Infer settlement from last fill price.

    In a settled Kalshi market, the final fill is always at 1 (NO wins) or 99
    (YES wins) — the closing auction removes residual spread.  We use a generous
    threshold of ≤ 5¢ = NO  /  ≥ 95¢ = YES to handle thin markets.
    """
    if not fills:
        return None
    last_price = fills[-1][1]
    if last_price <= 5:
        return "no"
    if last_price >= 95:
        return "yes"
    return None   # unsettled or no conclusive last fill


async def _fetch_all(tickers: list[str]) -> dict[str, dict]:
    """Return {ticker: {"fills": [...], "settlement": str|None}}.

    Settlement is inferred from the last fill price — no separate API call needed.
    """
    sem = asyncio.Semaphore(MAX_RATE)
    results: dict[str, dict] = {}
    done = 0

    async def worker(session: aiohttp.ClientSession, ticker: str) -> None:
        nonlocal done
        async with sem:
            fills = await _fetch_fills(session, ticker)
            sett  = _infer_settlement(fills)
            results[ticker] = {"fills": fills, "settlement": sett}
            done += 1
            if done % 100 == 0:
                print(f"  Fetched {done}/{len(tickers)} tickers…", flush=True)

    async with aiohttp.ClientSession() as session:
        await asyncio.gather(*[worker(session, t) for t in tickers])

    return results


def load_fills_and_settlements(tickers: list[str]) -> dict[str, dict]:
    """Load fills from cache, fetching only tickers not already cached (incremental)."""
    cached_raw: dict = {}
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            cached_raw = json.load(f)

    missing = [t for t in tickers if t not in cached_raw]

    if missing:
        print(f"A3+A4: Fetching {len(missing)} new tickers "
              f"({len(cached_raw)} already cached)…", flush=True)
        t0 = time.time()
        new_data = asyncio.run(_fetch_all(missing))
        print(f"  Done in {time.time()-t0:.0f}s", flush=True)

        # Merge new results into cache and save
        for ticker, d in new_data.items():
            cached_raw[ticker] = {
                "fills": [(ts.isoformat(), price) for ts, price in d["fills"]],
                "settlement": d["settlement"],
            }
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump(cached_raw, f)
        print(f"  Saved updated cache to {CACHE_FILE}", flush=True)
    else:
        print(f"A3+A4: All {len(tickers)} tickers already cached.", flush=True)

    # Deserialise and return only the requested tickers
    result = {}
    for ticker in tickers:
        d = cached_raw.get(ticker, {})
        result[ticker] = {
            "fills": [
                (datetime.fromisoformat(ts), price)
                for ts, price in d.get("fills", [])
            ],
            "settlement": d.get("settlement"),
        }
    return result


# ---------------------------------------------------------------------------
# Phase B — Build per-tick signal snapshots
# ---------------------------------------------------------------------------

def build_signal_snapshots(
    df_fc: pd.DataFrame,
    obs_state: dict,
    ticker_meta: dict[str, dict],
) -> pd.DataFrame:
    """Return a DataFrame with one row per (ticker, tick) on the ticker's settlement date.

    Columns:
      ticker, tick, metric, city, direction, is_high, band_type, no_direction,
      n_sources_total, model_spread,
      edge_{source} for each FORECAST_SOURCES member,
      has_om, hrrr_edge, hrrr_in_right_dir,
      obs_value, hours_to_close_approx
    """
    print("B: Building per-tick signal snapshots…", flush=True)
    t0 = time.time()

    df_fc = df_fc.copy()

    # --- B1: Vectorised OM bias correction ---
    # Build bias lookup table as a DataFrame for a fast merge
    print("  Applying OM bias correction (vectorised)…", flush=True)
    df_fc["city_abbr"] = (
        df_fc["metric"]
        .str.replace("temp_high_", "", regex=False)
        .str.replace("temp_low_",  "", regex=False)
    )
    df_fc["month"] = df_fc["logged_at"].dt.month

    if BIAS_F:
        bias_rows = [
            {"source": s, "city_abbr": c, "month": m, "_bias": b}
            for (s, c, m), b in BIAS_F.items()
        ]
        bias_df = pd.DataFrame(bias_rows)
        df_fc = df_fc.merge(bias_df, on=["source", "city_abbr", "month"], how="left")
        df_fc["_bias"] = df_fc["_bias"].fillna(0.0)
    else:
        df_fc["_bias"] = 0.0

    # Only apply bias to OM sources
    df_fc["edge_corr"] = df_fc["edge"].where(
        ~df_fc["source"].isin(OM_SOURCES),
        df_fc["edge"] - df_fc["_bias"],
    )

    # --- B2: Vectorised settlement-day filter ---
    # Attach mkt_date and tz to each row via the ticker lookup
    print("  Filtering to settlement-day ticks (vectorised)…", flush=True)
    meta_df = pd.DataFrame([
        {"ticker": t, "_mkt_date": m["mkt_date"], "_tz_str": str(m["tz"]), "_band_type": m["band_type"], "_band_num": m["band_number"]}
        for t, m in ticker_meta.items()
        if m["tz"] is not None
    ])
    df_fc = df_fc.merge(meta_df, on="ticker", how="inner")

    # Group by unique timezone and convert tick → local date
    day_masks = []
    for tz_str, grp in df_fc.groupby("_tz_str"):
        tz_obj = ZoneInfo(tz_str)
        local_dates = grp["tick"].dt.tz_convert(tz_obj).dt.date
        on_day = local_dates == grp["_mkt_date"]
        day_masks.append(on_day)

    if not day_masks:
        return pd.DataFrame()
    combined_mask = pd.concat(day_masks).sort_index()
    df_day = df_fc[combined_mask].copy()
    print(f"  Settlement-day rows: {len(df_day):,} (from {len(df_fc):,})", flush=True)

    if df_day.empty:
        return pd.DataFrame()

    # --- B3: no_direction for B-markets (vectorised) ---
    df_day["_no_direction"] = None
    b_mask = df_day["_band_type"] == "B"
    df_day.loc[b_mask & (df_day["data_value"] > df_day["_band_num"]), "_no_direction"] = "NO_HIGH"
    df_day.loc[b_mask & (df_day["data_value"] < df_day["_band_num"] - 1.0), "_no_direction"] = "NO_LOW"

    # --- B4: Pivot to (ticker, tick) rows ---
    print("  Pivoting per-source edges…", flush=True)
    pivot = (
        df_day.groupby(["ticker", "tick", "source"])["edge_corr"]
        .max()
        .unstack("source")
        .reset_index()
    )

    # Aggregate stats — use vectorised ops where possible
    agg = df_day.groupby(["ticker", "tick"]).agg(
        model_spread  =("data_value",   lambda x: x.max() - x.min()),
        n_sources_total=("source",       "nunique"),
        has_om        =("source",        lambda x: any(s in OM_SOURCES for s in x)),
        direction     =("direction",     "first"),
        metric        =("metric",        "first"),
        no_direction  =("_no_direction", lambda x: x.dropna().iloc[0] if x.dropna().size > 0 else None),
        city          =("city_abbr",     "first"),
        is_high       =("metric",        lambda x: x.iloc[0].startswith("temp_high_")),
        band_type     =("_band_type",    "first"),
        band_number   =("_band_num",     "first"),
        mkt_date      =("_mkt_date",     "first"),
        tz            =("_tz_str",        "first"),
    ).reset_index()

    snap = pivot.merge(agg, on=["ticker", "tick"], how="left")

    # --- B5: Join obs state via as-of merge ---
    print("  Joining obs state (as-of merge)…", flush=True)
    # Flatten obs_state into a DataFrame for merge_asof
    obs_rows = []
    for (metric, obs_date), timeline in obs_state.items():
        for obs_tick, obs_val in timeline:
            obs_rows.append({
                "metric":    metric,
                "obs_date":  obs_date,
                "obs_tick":  obs_tick,
                "obs_value": obs_val,
            })
    if obs_rows:
        obs_flat = pd.DataFrame(obs_rows)
        obs_flat["obs_tick"] = pd.to_datetime(obs_flat["obs_tick"], utc=True)
        obs_flat.sort_values("obs_tick", inplace=True)  # must be globally sorted for merge_asof

        snap["obs_date"] = snap["mkt_date"]
        snap_sorted = snap.sort_values("tick")

        # merge_asof requires same-type keys and sorted by key
        merged = pd.merge_asof(
            snap_sorted[["ticker", "tick", "metric", "obs_date"]],
            obs_flat[["metric", "obs_tick", "obs_date", "obs_value"]],
            left_on="tick",
            right_on="obs_tick",
            by=["metric", "obs_date"],
            direction="backward",
        )
        snap = snap.merge(
            merged[["ticker", "tick", "obs_value"]],
            on=["ticker", "tick"],
            how="left",
        )
    else:
        snap["obs_value"] = None

    # Rename source columns to edge_{source} for clarity
    for src in FORECAST_SOURCES:
        if src in snap.columns:
            snap.rename(columns={src: f"edge_{src}"}, inplace=True)

    print(f"  Snapshot rows: {len(snap):,}  ({time.time()-t0:.0f}s)", flush=True)
    return snap


# ---------------------------------------------------------------------------
# Phase D — Find first entries per (ticker, min_edge, min_sources)
# ---------------------------------------------------------------------------

def find_first_entries(
    snap: pd.DataFrame,
    fills_data: dict[str, dict],
    min_edge_f: float,
    min_sources: int,
    require_om: bool,
) -> pd.DataFrame:
    """For each ticker, return the first tick where the signal qualifies.

    Returns a DataFrame with one row per ticker (the first qualifying tick).
    Adds fill_price_at_entry (estimated YES ask) and hrrr_edge for veto checks.
    """
    # Count qualifying sources at this threshold
    edge_cols = [c for c in snap.columns if c.startswith("edge_")]
    snap = snap.copy()
    snap["n_qual"] = sum(
        (snap[c] >= min_edge_f).astype(int)
        for c in edge_cols
        if c != "edge_noaa_observed"  # obs sources don't count for n_qualifying
    )

    if require_om:
        snap = snap[snap["has_om"] == True].copy()  # noqa: E712

    # Must have at least min_sources qualifying
    snap = snap[snap["n_qual"] >= min_sources].copy()

    if snap.empty:
        return pd.DataFrame()

    # First qualifying tick per ticker
    snap.sort_values("tick", inplace=True)
    first = snap.groupby("ticker").first().reset_index()

    # Look up fill price at entry
    def _fill_at_tick(row: pd.Series) -> int | None:
        fills = fills_data.get(row["ticker"], {}).get("fills", [])
        if not fills:
            return None
        tick = row["tick"]
        # Find last fill at or before tick
        times  = [f[0] for f in fills]
        idx    = bisect.bisect_right(times, tick) - 1
        if idx < 0:
            # No fill before entry tick — use first available fill
            return fills[0][1] if fills else None
        return fills[idx][1]

    first["yes_fill_at_entry"] = first.apply(_fill_at_tick, axis=1)
    # Estimate no_ask = 100 - yes_fill (treating last fill ≈ YES bid)
    first["no_ask_entry"] = 100 - first["yes_fill_at_entry"].fillna(50)
    first["no_ask_entry"] = first["no_ask_entry"].clip(0, 100)

    # Filter by NO price gate
    first = first[
        (first["no_ask_entry"] >= MIN_NO_ASK) &
        (first["no_ask_entry"] <= MAX_NO_ASK)
    ].copy()

    # Add settlement
    first["settlement"] = first["ticker"].map(
        lambda t: fills_data.get(t, {}).get("settlement")
    )
    first = first[first["settlement"].isin(["yes", "no"])].copy()

    return first


# ---------------------------------------------------------------------------
# Phase E — Simulate exits
# ---------------------------------------------------------------------------

def simulate_exits(
    entries: pd.DataFrame,
    fills_data: dict[str, dict],
    sl_frac: float,
    pt_frac: float,
) -> pd.DataFrame:
    """Simulate SL/PT exit for each entry row.

    Returns entries with added columns:
      pnl_settlement, pnl_sl_pt, exit_reason_sl_pt
    """
    if entries.empty:
        return entries.assign(
            pnl_settlement=pd.Series(dtype=float),
            pnl_slpt=pd.Series(dtype=float),
            exit_reason=pd.Series(dtype=str),
        )

    pnl_sett  = []
    pnl_slpt  = []
    exit_rsns = []

    for _, row in entries.iterrows():
        ticker    = row["ticker"]
        entry_tick = row["tick"]
        no_ask    = float(row["no_ask_entry"])
        settlement = row["settlement"]  # "yes" | "no"
        fills     = fills_data.get(ticker, {}).get("fills", [])

        # Hold-to-settlement PnL
        if settlement == "no":
            pnl_sett.append((100 - no_ask) * CONTRACTS)   # NO won
        else:
            pnl_sett.append(-no_ask * CONTRACTS)           # NO lost

        # SL / PT exit simulation using fill series
        if not fills:
            pnl_slpt.append(pnl_sett[-1])
            exit_rsns.append("settlement_no_fills")
            continue

        # Find fills after entry
        entry_idx = bisect.bisect_right([f[0] for f in fills], entry_tick)
        post_fills = fills[entry_idx:]

        exited = False
        for _, yes_price in post_fills:
            no_bid = 100 - yes_price
            pnl_if_exit = (no_bid - no_ask) * CONTRACTS
            if pnl_if_exit <= -sl_frac * no_ask * CONTRACTS:
                pnl_slpt.append(pnl_if_exit)
                exit_rsns.append("stop_loss")
                exited = True
                break
            if pnl_if_exit >= pt_frac * no_ask * CONTRACTS:
                pnl_slpt.append(pnl_if_exit)
                exit_rsns.append("profit_take")
                exited = True
                break

        if not exited:
            pnl_slpt.append(pnl_sett[-1])
            exit_rsns.append("settlement")

    result = entries.copy()
    result["pnl_settlement"] = pnl_sett
    result["pnl_slpt"]       = pnl_slpt
    result["exit_reason"]    = exit_rsns
    return result


# ---------------------------------------------------------------------------
# Phase F — Parameter sweep
# ---------------------------------------------------------------------------

def sweep(
    snap: pd.DataFrame,
    fills_data: dict[str, dict],
) -> None:
    """Run full parameter sweep and print results."""

    # Pre-compute: for each (min_edge, min_sources, require_om), find first entries
    # then apply remaining gates during sweep

    EDGE_LEVELS   = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    SRC_LEVELS    = [2, 3, 4]
    REQUIRE_OM    = [True, False]
    SPREAD_MAX    = [4.0, 5.0, 6.0, 8.0, 999.0]
    SPREAD_MIN    = [0.0, 2.0, 3.0]
    HRRR_VETO     = [True, False]
    BLOCK_NO_HIGH = [True, False]
    DIRECTION     = ["all", "between_only", "T_only"]
    SL_FRACS      = [0.5, 0.7, 0.9]
    PT_FRACS      = [0.2, 0.3, 0.5]

    print("\n" + "=" * 78, flush=True)
    print("BUILDING ENTRY SETS (first-entry per ticker per config)…", flush=True)
    print("=" * 78, flush=True)

    # Cache entry sets keyed by (min_edge, min_sources, require_om)
    entry_cache: dict[tuple, pd.DataFrame] = {}
    for min_edge, min_src, req_om in itertools.product(EDGE_LEVELS, SRC_LEVELS, REQUIRE_OM):
        key = (min_edge, min_src, req_om)
        df = find_first_entries(snap, fills_data, min_edge, min_src, req_om)
        entry_cache[key] = df

    total_keys = len(entry_cache)
    nonempty = sum(1 for v in entry_cache.values() if not v.empty)
    print(f"  Built {total_keys} base entry sets, {nonempty} non-empty", flush=True)

    # For best (no SL/PT) exit: simulate settlement-only (already in pnl_settlement after
    # find_first_entries — no, we compute it in simulate_exits)
    # Pre-compute settlement PnL for each entry set
    print("  Pre-computing settlement PnL…", flush=True)
    for key, df in entry_cache.items():
        if df.empty:
            continue
        entry_cache[key] = simulate_exits(df, fills_data, sl_frac=999.0, pt_frac=999.0)

    # ----- Full sweep -----
    print("\n" + "=" * 78, flush=True)
    print("PARAMETER SWEEP", flush=True)
    print("=" * 78, flush=True)

    results = []
    total_combos = (
        len(EDGE_LEVELS) * len(SRC_LEVELS) * len(REQUIRE_OM) *
        len(SPREAD_MAX) * len(SPREAD_MIN) * len(HRRR_VETO) * len(BLOCK_NO_HIGH) * len(DIRECTION)
    )
    done = 0

    for (min_edge, min_src, req_om, spread_max, spread_min_f, hrrr_veto, block_noh, dir_filter) in itertools.product(
        EDGE_LEVELS, SRC_LEVELS, REQUIRE_OM, SPREAD_MAX, SPREAD_MIN, HRRR_VETO, BLOCK_NO_HIGH, DIRECTION
    ):
        done += 1
        if done % 1000 == 0:
            print(f"  Sweep {done}/{total_combos}…", flush=True)

        base_df = entry_cache.get((min_edge, min_src, req_om))
        if base_df is None or base_df.empty:
            continue

        df = base_df.copy()

        # Apply model spread gates
        if "model_spread" in df.columns and spread_max < 999:
            df = df[df["model_spread"] <= spread_max]
        if "model_spread" in df.columns and spread_min_f > 0:
            df = df[df["model_spread"] >= spread_min_f]

        # HRRR veto: if HRRR edge column exists and is ≤ 0 for "under" / HIGH markets, block
        if hrrr_veto and "edge_hrrr" in df.columns:
            hrrr_bad = (df["edge_hrrr"].notna()) & (df["edge_hrrr"] <= HRRR_TERRAIN_MIN_EDGE_F) & df["is_high"]
            df = df[~hrrr_bad]

        # Block NO_HIGH
        if block_noh:
            df = df[df["no_direction"] != "NO_HIGH"]

        # Direction filter
        if dir_filter == "between_only":
            df = df[df["band_type"] == "B"]
        elif dir_filter == "T_only":
            df = df[df["band_type"] == "T"]

        if len(df) < 8:
            continue

        # PnL from pre-computed settlement exit
        wins    = int((df["pnl_settlement"] > 0).sum())
        n       = len(df)
        total_p = float(df["pnl_settlement"].sum())
        avg_p   = total_p / n

        results.append({
            "min_edge":    min_edge,
            "min_src":     min_src,
            "req_om":      req_om,
            "spread_max":  spread_max,
            "spread_min":  spread_min_f,
            "hrrr_veto":   hrrr_veto,
            "block_noh":   block_noh,
            "dir":         dir_filter,
            "n":           n,
            "wins":        wins,
            "wr":          wins / n,
            "total_pnl":   total_p,
            "avg_pnl":     avg_p,
        })

    results.sort(key=lambda r: r["total_pnl"], reverse=True)

    # ---- Top 25 configs by total P&L ----
    print(f"\nTop 25 configs (hold-to-settlement exit, n ≥ 8):\n")
    hdr = (f"{'mE':>4} {'src':>3} {'om':>5} {'spMx':>5} {'hv':>4} "
           f"{'bNH':>4} {'dir':>12} | {'n':>4} {'wins':>5} {'WR':>6} "
           f"{'total_¢':>9} {'avg_¢':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in results[:25]:
        sp_str = f"{r['spread_max']:.0f}" if r["spread_max"] < 900 else "∞"
        print(
            f"{r['min_edge']:4.1f} {r['min_src']:3d} {str(r['req_om']):>5} {sp_str:>5}"
            f" {str(r['hrrr_veto']):>4} {str(r['block_noh']):>4} {r['dir']:>12}"
            f" | {r['n']:4d} {r['wins']:5d} {r['wr']:6.1%}"
            f" {r['total_pnl']:+9.0f}¢ {r['avg_pnl']:+8.1f}¢"
        )

    # ---- Top 15 by avg P&L (n ≥ 15) ----
    by_avg = sorted(
        [r for r in results if r["n"] >= 15],
        key=lambda r: r["avg_pnl"],
        reverse=True,
    )
    print(f"\nTop 15 configs by avg P&L (n ≥ 15):\n")
    print(hdr)
    print("-" * len(hdr))
    for r in by_avg[:15]:
        sp_str = f"{r['spread_max']:.0f}" if r["spread_max"] < 900 else "∞"
        print(
            f"{r['min_edge']:4.1f} {r['min_src']:3d} {str(r['req_om']):>5} {sp_str:>5}"
            f" {str(r['hrrr_veto']):>4} {str(r['block_noh']):>4} {r['dir']:>12}"
            f" | {r['n']:4d} {r['wins']:5d} {r['wr']:6.1%}"
            f" {r['total_pnl']:+9.0f}¢ {r['avg_pnl']:+8.1f}¢"
        )

    # ---- Source count comparison (between_only, spread≤6, spread_min=0, averaged across other params) ----
    print("\n--- Source count comparison (between_only, spread_max≤6, spread_min=0, averaged across other params) ---")
    print(f"  {'src':>4} | {'n_configs':>9}  {'avg_n_trades':>13}  {'avg_WR':>7}  {'avg_pnl':>8}")
    print(f"  {'-'*4}-+-{'-'*9}--{'-'*13}--{'-'*7}--{'-'*8}")
    for src in SRC_LEVELS:
        subset = [r for r in results
                  if r["min_src"] == src and r["dir"] == "between_only"
                  and r["spread_max"] <= 6 and r["spread_min"] == 0]
        if not subset:
            continue
        avg_n  = sum(r["n"]   for r in subset) / len(subset)
        avg_wr = sum(r["wr"]  for r in subset) / len(subset)
        avg_p  = sum(r["avg_pnl"] for r in subset) / len(subset)
        print(f"  {src:4d} | {len(subset):9d}  {avg_n:13.1f}  {avg_wr:7.1%}  {avg_p:+8.1f}¢")

    # ---- Min-edge sweep (between_only, src=4, spread≤6, spread_min=0, averaged across other params) ----
    print("\n--- Min-edge sweep (between_only, src=4, spread_max≤6, spread_min=0, averaged across other params) ---")
    print(f"  {'mE':>4} | {'n_configs':>9}  {'avg_n_trades':>13}  {'avg_WR':>7}  {'avg_pnl':>8}")
    print(f"  {'-'*4}-+-{'-'*9}--{'-'*13}--{'-'*7}--{'-'*8}")
    for me in EDGE_LEVELS:
        subset = [r for r in results
                  if r["min_edge"] == me and r["min_src"] == 4 and r["dir"] == "between_only"
                  and r["spread_max"] <= 6 and r["spread_min"] == 0]
        if not subset:
            continue
        avg_n  = sum(r["n"]   for r in subset) / len(subset)
        avg_wr = sum(r["wr"]  for r in subset) / len(subset)
        avg_p  = sum(r["avg_pnl"] for r in subset) / len(subset)
        print(f"  {me:4.1f} | {len(subset):9d}  {avg_n:13.1f}  {avg_wr:7.1%}  {avg_p:+8.1f}¢")

    # ---- Fixed-config edge comparison (exact trade counts, one pinned config) ----
    print("\n--- Fixed-config edge comparison (src=4, spread_max=4, spread_min=0, hv=False, bNH=False, between_only) ---")
    print(f"  {'mE':>4} | {'n':>6}  {'WR':>7}  {'total_¢':>9}  {'avg_¢':>8}")
    print(f"  {'-'*4}-+-{'-'*6}--{'-'*7}--{'-'*9}--{'-'*8}")
    for me in EDGE_LEVELS:
        match = next(
            (r for r in results
             if r["min_edge"] == me and r["min_src"] == 4 and not r["req_om"]
             and r["spread_max"] == 4.0 and r["spread_min"] == 0.0
             and not r["hrrr_veto"] and not r["block_noh"] and r["dir"] == "between_only"),
            None,
        )
        if match:
            print(f"  {me:4.1f} | {match['n']:6d}  {match['wr']:7.1%}  {match['total_pnl']:+9.0f}¢  {match['avg_pnl']:+8.1f}¢")
        else:
            print(f"  {me:4.1f} | {'(n<8)':>6}")

    # ---- Spread_min sweep (between_only, src=4, spread_max≤6, averaged across other params) ----
    print("\n--- Spread_min sweep (between_only, src=4, spread_max≤6, averaged across other params) ---")
    print(f"  {'spMn':>5} | {'n_configs':>9}  {'avg_n_trades':>13}  {'avg_WR':>7}  {'avg_pnl':>8}")
    print(f"  {'-'*5}-+-{'-'*9}--{'-'*13}--{'-'*7}--{'-'*8}")
    for smn in SPREAD_MIN:
        subset = [r for r in results
                  if r["spread_min"] == smn and r["min_src"] == 4
                  and r["dir"] == "between_only" and r["spread_max"] <= 6]
        if not subset:
            continue
        avg_n  = sum(r["n"]   for r in subset) / len(subset)
        avg_wr = sum(r["wr"]  for r in subset) / len(subset)
        avg_p  = sum(r["avg_pnl"] for r in subset) / len(subset)
        print(f"  {smn:5.1f} | {len(subset):9d}  {avg_n:13.1f}  {avg_wr:7.1%}  {avg_p:+8.1f}¢")

    # ---- Cross-validation: current policy config ----
    current_policy = next(
        (r for r in results
         if r["min_edge"] == 3.0 and r["min_src"] == 2 and r["req_om"]
         and r["spread_max"] >= 6.0 and not r["block_noh"] and r["dir"] == "all"),
        None,
    )
    print("\n--- Cross-validation vs live trades (should match ~60-68% WR) ---")
    if current_policy:
        print(
            f"  Current policy (min_edge=3.0, min_src=2, req_om=True):"
            f" n={current_policy['n']}  WR={current_policy['wr']:.1%}"
            f"  total={current_policy['total_pnl']:+.0f}¢  avg={current_policy['avg_pnl']:+.1f}¢"
        )
    else:
        print("  (no matching current-policy config found in results)")

    # ---- Best config SL/PT sensitivity ----
    if results:
        best = results[0]
        best_key = (best["min_edge"], best["min_src"], best["req_om"])
        best_df = entry_cache.get(best_key)
        if best_df is not None and not best_df.empty:
            # Apply same filters as best config
            df = best_df.copy()
            if best["spread_max"] < 900 and "model_spread" in df.columns:
                df = df[df["model_spread"] <= best["spread_max"]]
            if best["hrrr_veto"] and "edge_hrrr" in df.columns:
                hrrr_bad = (df["edge_hrrr"].notna()) & (df["edge_hrrr"] <= HRRR_TERRAIN_MIN_EDGE_F) & df["is_high"]
                df = df[~hrrr_bad]
            if best["block_noh"]:
                df = df[df["no_direction"] != "NO_HIGH"]
            if best["dir"] == "between_only":
                df = df[df["band_type"] == "B"]
            elif best["dir"] == "T_only":
                df = df[df["band_type"] == "T"]

            print(f"\n--- SL/PT sensitivity for best config ({best['min_edge']:.1f}¢ edge, "
                  f"n={len(df)}) ---")
            print(f"  {'exit':>20} | {'total_¢':>9} {'avg_¢':>8} {'WR':>6}")
            print(f"  {'-'*20}-+-{'-'*9}-{'-'*8}-{'-'*6}")
            for sl_frac in SL_FRACS:
                for pt_frac in PT_FRACS:
                    sim = simulate_exits(df, fills_data, sl_frac=sl_frac, pt_frac=pt_frac)
                    tp  = float(sim["pnl_slpt"].sum())
                    n_  = len(sim)
                    wins_ = int((sim["pnl_slpt"] > 0).sum())
                    wr_ = wins_ / n_ if n_ > 0 else 0.0
                    label = f"SL={sl_frac:.0%} PT={pt_frac:.0%}"
                    print(f"  {label:>20} | {tp:+9.0f}¢ {tp/max(n_,1):+8.1f}¢ {wr_:6.1%}")

    # ---- Breakdowns for best config ----
    if results and not entry_cache.get((results[0]["min_edge"], results[0]["min_src"], results[0]["req_om"]), pd.DataFrame()).empty:
        best = results[0]
        best_df = entry_cache[(best["min_edge"], best["min_src"], best["req_om"])].copy()

        # Apply the same filters used to produce the best config's n/WR
        if best["spread_max"] < 900 and "model_spread" in best_df.columns:
            best_df = best_df[best_df["model_spread"] <= best["spread_max"]]
        if best.get("spread_min", 0) > 0 and "model_spread" in best_df.columns:
            best_df = best_df[best_df["model_spread"] >= best["spread_min"]]
        if best["hrrr_veto"] and "edge_hrrr" in best_df.columns:
            hrrr_bad = (best_df["edge_hrrr"].notna()) & (best_df["edge_hrrr"] <= HRRR_TERRAIN_MIN_EDGE_F) & best_df["is_high"]
            best_df = best_df[~hrrr_bad]
        if best["block_noh"]:
            best_df = best_df[best_df["no_direction"] != "NO_HIGH"]
        if best["dir"] == "between_only":
            best_df = best_df[best_df["band_type"] == "B"]
        elif best["dir"] == "T_only":
            best_df = best_df[best_df["band_type"] == "T"]

        print(f"\n--- City breakdown (best config, n={len(best_df)}) ---")
        city_groups = best_df.groupby("city").apply(
            lambda g: pd.Series({
                "n": len(g),
                "wins": int((g["pnl_settlement"] > 0).sum()),
                "total_pnl": float(g["pnl_settlement"].sum()),
            })
        ).reset_index()
        city_groups["wr"] = city_groups["wins"] / city_groups["n"]
        city_groups.sort_values("total_pnl", inplace=True)
        for _, row in city_groups.iterrows():
            print(f"  {row['city']:6s}: n={row['n']:3.0f}  WR={row['wr']:5.1%}"
                  f"  total={row['total_pnl']:+7.0f}¢")

        print(f"\n--- Direction breakdown (best config, n={len(best_df)}) ---")
        dir_groups = best_df.groupby(["direction", "no_direction"]).apply(
            lambda g: pd.Series({
                "n": len(g),
                "wins": int((g["pnl_settlement"] > 0).sum()),
                "total_pnl": float(g["pnl_settlement"].sum()),
            })
        ).reset_index()
        dir_groups.sort_values("total_pnl", inplace=True)
        for _, row in dir_groups.iterrows():
            label = f"{row['direction']}/{row['no_direction'] or '-'}"
            print(f"  {label:20s}: n={row['n']:3.0f}  WR={row['wins']/row['n']:5.1%}"
                  f"  total={row['total_pnl']:+7.0f}¢")

        # ---- Source combination breakdown ----
        _SRC_SHORT = {
            "hrrr":             "HRRR",
            "nws_hourly":       "NWS",
            "noaa":             "NOAA",
            "open_meteo":       "OM",
            "open_meteo_ecmwf": "EC",
            "open_meteo_gem":   "GEM",
            "open_meteo_icon":  "IC",
        }
        edge_cols = [c for c in best_df.columns if c.startswith("edge_")
                     and c not in ("edge_noaa_observed",)]
        min_edge_threshold = best["min_edge"]

        def _qual_combo(row: pd.Series) -> str:
            srcs = sorted(
                _SRC_SHORT.get(c.replace("edge_", ""), c.replace("edge_", ""))
                for c in edge_cols
                if pd.notna(row[c]) and row[c] >= min_edge_threshold
            )
            return "+".join(srcs) if srcs else "(none)"

        best_df = best_df.copy()
        best_df["_src_combo"] = best_df.apply(_qual_combo, axis=1)
        src_groups = best_df.groupby("_src_combo").apply(
            lambda g: pd.Series({
                "n":         len(g),
                "wins":      int((g["pnl_settlement"] > 0).sum()),
                "total_pnl": float(g["pnl_settlement"].sum()),
            })
        ).reset_index()
        src_groups["wr"]      = src_groups["wins"] / src_groups["n"]
        src_groups["avg_pnl"] = src_groups["total_pnl"] / src_groups["n"]
        src_groups = src_groups[src_groups["n"] >= 2].sort_values("wr", ascending=False)

        print(f"\n--- Source combination breakdown (best config, min_edge={min_edge_threshold}) ---")
        print(f"  {'combo':40s} | {'n':>4}  {'WR':>6}  {'total_¢':>8}  {'avg_¢':>7}")
        print(f"  {'-'*40}-+-{'-'*4}--{'-'*6}--{'-'*8}--{'-'*7}")
        for _, row in src_groups.iterrows():
            print(f"  {row['_src_combo']:40s} | {row['n']:4.0f}  {row['wr']:6.1%}"
                  f"  {row['total_pnl']:+8.0f}¢  {row['avg_pnl']:+7.1f}¢")

        # ---- Model spread buckets ----
        print(f"\n--- Model spread at entry (best config, n={len(best_df)}) ---")
        print(f"  {'spread range':>14} | {'n':>4}  {'WR':>6}  {'total_¢':>9}  {'avg_¢':>8}")
        print(f"  {'-'*14}-+-{'-'*4}--{'-'*6}--{'-'*9}--{'-'*8}")
        for lo, hi in [(0, 2), (2, 4), (4, 6), (6, 8), (8, 999)]:
            grp = best_df[(best_df["model_spread"] >= lo) & (best_df["model_spread"] < hi)]
            if grp.empty:
                continue
            n_ = len(grp)
            wins_ = int((grp["pnl_settlement"] > 0).sum())
            total_ = float(grp["pnl_settlement"].sum())
            hi_str = f"{hi}°F" if hi < 999 else "∞"
            print(f"  {lo}–{hi_str:>5}          | {n_:4d}  {wins_/n_:6.1%}  {total_:+9.0f}¢  {total_/n_:+8.1f}¢")

        # ---- Entry NO price buckets ----
        best_df = best_df.copy()
        print(f"\n--- Entry NO price buckets (best config, n={len(best_df)}) ---")
        print(f"  {'no_ask range':>14} | {'n':>4}  {'WR':>6}  {'total_¢':>9}  {'avg_¢':>8}")
        print(f"  {'-'*14}-+-{'-'*4}--{'-'*6}--{'-'*9}--{'-'*8}")
        for lo, hi in [(15, 30), (30, 45), (45, 60), (60, 81)]:
            grp = best_df[(best_df["no_ask_entry"] >= lo) & (best_df["no_ask_entry"] < hi)]
            if grp.empty:
                continue
            n_ = len(grp)
            wins_ = int((grp["pnl_settlement"] > 0).sum())
            total_ = float(grp["pnl_settlement"].sum())
            print(f"  {lo:3d}–{hi-1:3d}¢         | {n_:4d}  {wins_/n_:6.1%}  {total_:+9.0f}¢  {total_/n_:+8.1f}¢")

        # ---- Hours before close at entry ----
        def _hours_to_close(row: pd.Series) -> float:
            tz_obj = ZoneInfo(row["tz"])
            md = row["mkt_date"]
            end_of_day = datetime(md.year, md.month, md.day, 23, 59, 59).replace(tzinfo=tz_obj)
            return max(0.0, (end_of_day - row["tick"].astimezone(tz_obj)).total_seconds() / 3600)

        best_df["_h2c"] = best_df.apply(_hours_to_close, axis=1)
        print(f"\n--- Hours before close at entry (best config, n={len(best_df)}) ---")
        print(f"  {'range':>10} | {'n':>4}  {'WR':>6}  {'total_¢':>9}  {'avg_¢':>8}")
        print(f"  {'-'*10}-+-{'-'*4}--{'-'*6}--{'-'*9}--{'-'*8}")
        for lo, hi in [(0, 4), (4, 8), (8, 12), (12, 18), (18, 25)]:
            grp = best_df[(best_df["_h2c"] >= lo) & (best_df["_h2c"] < hi)]
            if grp.empty:
                continue
            n_ = len(grp)
            wins_ = int((grp["pnl_settlement"] > 0).sum())
            total_ = float(grp["pnl_settlement"].sum())
            print(f"  {lo:2d}–{hi:2d}h      | {n_:4d}  {wins_/n_:6.1%}  {total_:+9.0f}¢  {total_/n_:+8.1f}¢")

        # ---- HIGH vs LOW markets ----
        print(f"\n--- HIGH vs LOW markets (best config, n={len(best_df)}) ---")
        for is_h, label in [(True, "KXHIGH (temp high)"), (False, "KXLOW  (temp low) ")]:
            grp = best_df[best_df["is_high"] == is_h]
            if grp.empty:
                continue
            n_ = len(grp)
            wins_ = int((grp["pnl_settlement"] > 0).sum())
            total_ = float(grp["pnl_settlement"].sum())
            print(f"  {label}: n={n_:3d}  WR={wins_/n_:5.1%}  total={total_:+7.0f}¢  avg={total_/n_:+6.1f}¢")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # --- Phase A: Load data ---
    df_fc, df_obs = load_raw_forecasts()

    # Parse all unique weather tickers from raw_forecasts
    print("Parsing ticker metadata…", flush=True)
    unique_tickers = df_fc["ticker"].unique().tolist()
    ticker_meta: dict[str, dict] = {}
    bad = 0
    for t in unique_tickers:
        m = _parse_ticker(t)
        if m is not None:
            ticker_meta[t] = m
        else:
            bad += 1
    print(f"  Parsed {len(ticker_meta):,} tickers ({bad} unrecognised)", flush=True)

    # Keep only rows for tickers we could parse
    df_fc = df_fc[df_fc["ticker"].isin(ticker_meta)].copy()
    print(f"  Rows after ticker filter: {len(df_fc):,}", flush=True)

    obs_state = build_obs_state(df_obs)

    # --- Pre-filter: only fetch tickers that showed ≥ 1 source with edge ≥ 2°F ---
    # Eliminates ~60-70% of tickers that never had a qualifying signal,
    # cutting API calls from ~4,500 to ~1,200-1,800.
    signal_tickers = set(
        df_fc.loc[df_fc["edge"] >= 0.5, "ticker"].unique().tolist()
    )
    all_tickers = [t for t in ticker_meta if t in signal_tickers]
    print(f"  Tickers with any edge ≥ 0.5°F: {len(all_tickers):,} "
          f"(filtered from {len(ticker_meta):,})", flush=True)

    # Also keep only snapshot rows for these tickers
    df_fc = df_fc[df_fc["ticker"].isin(signal_tickers)].copy()

    # --- Phase A3+A4: Fetch fills + settlements ---
    fills_data  = load_fills_and_settlements(all_tickers)

    settled_count = sum(
        1 for d in fills_data.values() if d.get("settlement") in ("yes", "no")
    )
    print(f"  Settled markets: {settled_count:,} / {len(all_tickers):,}", flush=True)

    # --- Phase B: Build signal snapshots ---
    snap = build_signal_snapshots(df_fc, obs_state, ticker_meta)

    if snap.empty:
        print("ERROR: No signal snapshots built — check data.", flush=True)
        return

    # --- Phase F: Sweep ---
    sweep(snap, fills_data)

    print("\n[Done]", flush=True)


if __name__ == "__main__":
    main()
