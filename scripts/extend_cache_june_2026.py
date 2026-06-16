"""
Extend band_arb_hist_cache.json with June 1–14 2026 data.

Fetches fresh data from IEM ASOS and Open-Meteo APIs and MERGES the new
dates into existing cache entries (does not create new top-level keys).
Existing dates are never overwritten — only truly missing dates are added.

Saves to a temp file first, then renames atomically to avoid partial writes.

Usage:
    venv/bin/python scripts/extend_cache_june_2026.py [--dry-run]
    venv/bin/python scripts/extend_cache_june_2026.py --start 2026-06-01 --end 2026-06-14
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.build_forecast_calibration import (  # noqa: E402
    IEM_STATIONS, OM_MODELS, HRRR_MODEL,
    fetch_hrrr_historical, fetch_om_historical, fetch_iem_observed,
)
from kalshi_bot.cities import CITIES, LOW_CITIES  # noqa: E402

CACHE_PATH = Path("data/backtest/band_arb_hist_cache.json")
TEMP_PATH  = CACHE_PATH.with_suffix(".json.tmp")

# Open-Meteo hourly API (same endpoint as historical-forecast)
_OM_HOURLY_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

# Semaphore — Open-Meteo free tier is strict about concurrency
_SEM = asyncio.Semaphore(1)
_IEM_SEM = asyncio.Semaphore(3)


def find_cache_key(cache: dict, prefix: str) -> str | None:
    for k in cache:
        if k.startswith(prefix):
            return k
    return None


def merge_dates(existing: dict, new_data: dict) -> int:
    """Add keys from new_data into existing dict; skip if date already present.
    Returns number of new dates added."""
    added = 0
    for d, v in new_data.items():
        if d not in existing:
            existing[d] = v
            added += 1
    return added


async def fetch_iem_hourly(
    session: aiohttp.ClientSession, station: str, start: str, end: str
) -> dict[str, dict[str, float]]:
    """Returns {date: {hour_str: temp_f}} from IEM ASOS METAR hourly."""
    url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
    params = {
        "station": station, "data": "tmpf",
        "year1": start[:4], "month1": start[5:7], "day1": start[8:10],
        "year2": end[:4],   "month2": end[5:7],   "day2": end[8:10],
        "tz": "UTC", "format": "onlycomma", "latlon": "no", "report_type": "3",
    }
    result: dict[str, dict[str, float]] = {}
    async with _IEM_SEM:
        try:
            async with session.get(url, params=params,
                                   timeout=aiohttp.ClientTimeout(total=60)) as resp:
                resp.raise_for_status()
                text = await resp.text()
        except Exception as e:
            print(f"  [IEM hourly] {station}: {e}", file=sys.stderr)
            return result
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("station") or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) < 3:
            continue
        valid_str, tmpf_str = parts[1].strip(), parts[2].strip()
        if tmpf_str in ("M", "", "None"):
            continue
        try:
            dt_date, dt_time = valid_str.split(" ")
            hour = str(int(dt_time.split(":")[0]))
            result.setdefault(dt_date, {})[hour] = float(tmpf_str)
        except (ValueError, IndexError):
            continue
    return result


async def fetch_om_hourly(
    session: aiohttp.ClientSession,
    lat: float, lon: float,
    start: str, end: str,
    model: str,   # "gfs_seamless" or "gfs_hrrr"
) -> dict[str, dict[str, float]]:
    """Returns {date: {hour_str: temp_f}} for one model's hourly forecasts."""
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "hourly": "temperature_2m",
        "temperature_unit": "fahrenheit",
        "timezone": "UTC",
        "models": model,
    }
    async with _SEM:
        try:
            async with session.get(_OM_HOURLY_URL, params=params,
                                   timeout=aiohttp.ClientTimeout(total=30)) as resp:
                resp.raise_for_status()
                data = await resp.json()
        except Exception as e:
            print(f"  [OM hourly {model}] ({lat},{lon}): {e}", file=sys.stderr)
            return {}
    hourly = data.get("hourly", {})
    times  = hourly.get("time", [])
    vals   = hourly.get("temperature_2m", [])
    result: dict[str, dict[str, float]] = {}
    for ts, v in zip(times, vals):
        if v is None:
            continue
        date_str = ts[:10]
        hour_str = str(int(ts[11:13]))
        result.setdefault(date_str, {})[hour_str] = float(v)
    return result


async def extend_cache(start: str, end: str, dry_run: bool) -> None:
    print(f"Extending cache: {start} → {end}  (dry_run={dry_run})")
    cache = json.loads(CACHE_PATH.read_text())
    print(f"Loaded cache: {len(cache)} keys, {CACHE_PATH.stat().st_size / 1e6:.1f} MB")

    totals: dict[str, int] = {}  # key-prefix → dates added

    async with aiohttp.ClientSession() as session:

        # ── 1. IEM hourly METAR obs ───────────────────────────────────────────
        print("\n[1/5] IEM hourly observations ...")
        # Get all unique METAR stations
        stations = sorted(set(v[0] for v in IEM_STATIONS.values()))
        tasks = {st: fetch_iem_hourly(session, st, start, end) for st in stations}
        results = {st: await coro for st, coro in tasks.items()}
        for station, new_data in results.items():
            key = find_cache_key(cache, f"hourly_{station}_")
            if not key:
                print(f"  WARN: no cache key for hourly_{station}_* — skipping")
                continue
            added = merge_dates(cache[key], new_data)
            totals[f"hourly_{station}"] = added
            print(f"  {station}: +{added} dates  (now {len(cache[key])} total)")

        # ── 2. HRRR daily high/low ────────────────────────────────────────────
        print("\n[2/5] HRRR daily forecasts ...")
        all_metrics = list(CITIES.keys()) + list(LOW_CITIES.keys())
        for metric, info in {**CITIES, **LOW_CITIES}.items():
            _, lat, lon, _ = info
            is_high = metric.startswith("temp_high_")
            city = metric.split("_", 2)[2]
            direction = "high" if is_high else "low"
            key = find_cache_key(cache, f"hrrr_temp_{direction}_{city}_")
            if not key:
                print(f"  WARN: no cache key for hrrr_temp_{direction}_{city}_*")
                continue
            async with _SEM:
                new_data = await fetch_hrrr_historical(session, lat, lon, start, end, is_high)
            added = merge_dates(cache[key], new_data)
            totals[f"hrrr_{metric}"] = added
            print(f"  {metric}: +{added} dates")

        # ── 3. Open-Meteo daily high/low (GFS/ECMWF/GEM/ICON) ────────────────
        print("\n[3/5] Open-Meteo daily forecasts ...")
        for metric, info in {**CITIES, **LOW_CITIES}.items():
            _, lat, lon, _ = info
            is_high = metric.startswith("temp_high_")
            city = metric.split("_", 2)[2]
            direction = "high" if is_high else "low"
            key = find_cache_key(cache, f"om_temp_{direction}_{city}_")
            if not key:
                print(f"  WARN: no cache key for om_temp_{direction}_{city}_*")
                continue
            async with _SEM:
                new_data = await fetch_om_historical(session, lat, lon, start, end, is_high)
            added_total = 0
            for model_name, model_dates in new_data.items():
                if model_name not in cache[key]:
                    cache[key][model_name] = {}
                added = merge_dates(cache[key][model_name], model_dates)
                added_total += added
            totals[f"om_{metric}"] = added_total
            print(f"  {metric}: +{added_total} model-date pairs")

        # ── 4. IEM daily actual high/low ─────────────────────────────────────
        print("\n[4/5] IEM daily actuals ...")
        for metric, (station, network) in IEM_STATIONS.items():
            is_high = metric.startswith("temp_high_")
            direction = "high" if is_high else "low"
            key = find_cache_key(cache, f"actual_{station}_")
            # match the specific direction key
            matching_keys = [k for k in cache if k.startswith(f"actual_{station}_")
                             and k.endswith(f"_{direction}")]
            if not matching_keys:
                print(f"  WARN: no cache key for actual_{station}_*_{direction}")
                continue
            key = matching_keys[0]
            # Only fetch if June dates are actually missing
            existing_dates = set(cache[key].keys())
            needed = [d for d in _date_range(start, end) if d not in existing_dates]
            if not needed:
                print(f"  {metric}: already up to date")
                continue
            async with _IEM_SEM:
                new_data = await fetch_iem_observed(session, station, network, 2026, 2026, is_high)
            # Filter to only the requested date range
            new_june = {d: v for d, v in new_data.items() if start <= d <= end}
            added = merge_dates(cache[key], new_june)
            totals[f"actual_{metric}"] = added
            print(f"  {metric}: +{added} dates")

        # ── 5. GFS + HRRR hourly forecasts — SKIPPED ────────────────────────
        # obs_vs_hrrr_h and obs_vs_gfs_h are hardcoded to 0.0 at inference
        # time (main.py _model_shadow_features), so June rows will naturally
        # have 0.0 — more consistent with runtime than forcing historical fetches.
        # The Open-Meteo hourly endpoint also stalls without triggering aiohttp
        # timeouts, causing indefinite hangs on large date-range requests.
        print("\n[5/5] Hourly FC — skipped (obs_vs_hrrr_h=0 matches runtime behaviour)")

    # ── Summary ───────────────────────────────────────────────────────────────
    total_added = sum(totals.values())
    print(f"\nTotal new data points added: {total_added:,}")

    if dry_run:
        print("DRY RUN — cache not saved.")
        return

    # Atomic write: temp file → rename
    TEMP_PATH.write_text(json.dumps(cache))
    TEMP_PATH.rename(CACHE_PATH)
    print(f"Cache saved → {CACHE_PATH}  ({CACHE_PATH.stat().st_size / 1e6:.1f} MB)")


def _date_range(start: str, end: str) -> list[str]:
    from datetime import date, timedelta
    d0 = date.fromisoformat(start)
    d1 = date.fromisoformat(end)
    out = []
    while d0 <= d1:
        out.append(d0.isoformat())
        d0 += timedelta(days=1)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",   default="2026-06-01")
    parser.add_argument("--end",     default="2026-06-14")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(extend_cache(args.start, args.end, args.dry_run))
