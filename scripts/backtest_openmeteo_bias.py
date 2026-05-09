#!/usr/bin/env python3
"""Historical Open-Meteo forecast bias vs. METAR actuals.

Fetches historical daily max temperature forecasts from the Open-Meteo
Historical Forecast API for all Kalshi city locations, compares to actual
daily highs from Iowa State Mesonet ASOS, and outputs a bias table
(forecast − actual °F) grouped by source × month and per city.

Negative bias = model ran cold (underestimated peak temperature).

Results are cached per city in --cache-dir (default: data/bias_cache/).
On subsequent runs, cached cities are loaded instantly — only new cities
or cities outside the cached date range hit the API.

Usage:
    venv/bin/python scripts/backtest_openmeteo_bias.py
    venv/bin/python scripts/backtest_openmeteo_bias.py --years 10
    venv/bin/python scripts/backtest_openmeteo_bias.py --cities ny chi bos --csv /tmp/bias.csv
    venv/bin/python scripts/backtest_openmeteo_bias.py --per-year          # year-over-year drift
    venv/bin/python scripts/backtest_openmeteo_bias.py --no-cache          # force re-fetch all
    venv/bin/python scripts/backtest_openmeteo_bias.py --no-cache --cities lax  # refresh one city
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, stdev, StatisticsError

import aiohttp

sys.path.insert(0, str(Path(__file__).parent.parent))
from kalshi_bot.news.noaa import CITIES, KALSHI_STATION_IDS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HIST_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
MESONET_URL       = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

# Open-Meteo model param → internal source name.
# None = no models param (default blended) → "open_meteo".
HIST_MODELS: dict[str | None, str] = {
    None:            "open_meteo",
    "gfs_seamless":  "open_meteo_gfs",
    "ecmwf_ifs":     "open_meteo_ecmwf",
    "icon_seamless": "open_meteo_icon",
    "gem_seamless":  "open_meteo_gem",
}

_MIN_OBS_PER_DAY = 8   # skip days with sparse Mesonet coverage
_MESONET_DELAY   = 0.6  # seconds between Mesonet requests (shared academic resource)
_OM_DELAY        = 0.3  # seconds between Open-Meteo requests

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

# ---------------------------------------------------------------------------
# Per-city cache helpers
# ---------------------------------------------------------------------------

def _cache_path(cache_dir: Path, city_code: str) -> Path:
    return cache_dir / f"{city_code}.json"


def load_cache(cache_dir: Path, city_code: str, start_date: date, end_date: date) -> list[tuple] | None:
    """Return cached bias rows for city if the cached date range covers start→end, else None."""
    p = _cache_path(cache_dir, city_code)
    if not p.exists():
        return None
    try:
        meta = json.loads(p.read_text())
        if (meta.get("start_date") == start_date.isoformat()
                and meta.get("end_date") == end_date.isoformat()):
            rows = [tuple(r) for r in meta["rows"]]
            log.info("[cache] %s: %d rows loaded", city_code.upper(), len(rows))
            return rows
        log.info("[cache] %s: date range mismatch — re-fetching", city_code.upper())
    except Exception as exc:
        log.warning("[cache] %s: load error (%s) — re-fetching", city_code.upper(), exc)
    return None


def save_cache(cache_dir: Path, city_code: str, start_date: date, end_date: date, rows: list[tuple]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = _cache_path(cache_dir, city_code)
    p.write_text(json.dumps({
        "city_code":  city_code,
        "start_date": start_date.isoformat(),
        "end_date":   end_date.isoformat(),
        "rows":       [list(r) for r in rows],
    }))
    log.debug("[cache] %s: saved %d rows", city_code.upper(), len(rows))


# ---------------------------------------------------------------------------
# Open-Meteo historical fetch
# ---------------------------------------------------------------------------

async def fetch_openmeteo_forecasts(
    session:    aiohttp.ClientSession,
    city_key:   str,
    lat:        float,
    lon:        float,
    tz_str:     str,
    start_date: date,
    end_date:   date,
) -> dict[str, dict[str, float]]:
    """Return {source_name: {date_str: forecast_max_f}} for all OM models.

    Makes two requests per city:
      1. No models param → default blended → "open_meteo"
      2. All named models in one call → "open_meteo_gfs" etc.
    """
    results: dict[str, dict[str, float]] = {}

    named_models = [m for m in HIST_MODELS if m is not None]

    for call_idx, model_param in enumerate([None, ",".join(named_models)]):
        params: dict = {
            "latitude":         f"{lat:.4f}",
            "longitude":        f"{lon:.4f}",
            "start_date":       start_date.isoformat(),
            "end_date":         end_date.isoformat(),
            "daily":            "temperature_2m_max",
            "temperature_unit": "fahrenheit",
            "timezone":         tz_str,
        }
        if model_param:
            params["models"] = model_param

        try:
            async with session.get(
                HIST_FORECAST_URL,
                params=params,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status == 429:
                    log.warning("OM rate-limited for %s — skipping this call", city_key)
                    await asyncio.sleep(5.0)
                    continue
                resp.raise_for_status()
                data = await resp.json()
        except Exception as exc:
            log.warning("OM fetch error for %s (call %d): %s", city_key, call_idx, exc)
            await asyncio.sleep(_OM_DELAY)
            continue

        daily  = data.get("daily", {})
        dates  = daily.get("time", [])

        if call_idx == 0:
            values = daily.get("temperature_2m_max", [])
            results[HIST_MODELS[None]] = {
                d: v for d, v in zip(dates, values) if v is not None
            }
        else:
            for api_model, src_name in HIST_MODELS.items():
                if api_model is None:
                    continue
                field  = f"temperature_2m_max_{api_model}"
                values = daily.get(field, [])
                if not values:
                    log.debug("No historical data for %s / %s", city_key, api_model)
                    continue
                results[src_name] = {
                    d: v for d, v in zip(dates, values) if v is not None
                }

        await asyncio.sleep(_OM_DELAY)

    return results


# ---------------------------------------------------------------------------
# Mesonet actuals fetch
# ---------------------------------------------------------------------------

async def fetch_mesonet_actuals(
    session:    aiohttp.ClientSession,
    city_key:   str,
    station:    str,
    city_tz,
    start_date: date,
    end_date:   date,
) -> dict[str, float]:
    """Return {date_str: actual_max_f} using Iowa State Mesonet ASOS data."""
    params = {
        "station":     station,
        "data":        "tmpf",
        "year1":       str(start_date.year),
        "month1":      str(start_date.month),
        "day1":        str(start_date.day),
        "year2":       str(end_date.year),
        "month2":      str(end_date.month),
        "day2":        str(end_date.day),
        "tz":          "UTC",
        "format":      "comma",
        "latlon":      "no",
        "missing":     "M",
        "trace":       "T",
        "direct":      "no",
        "report_type": "3,4",
    }

    try:
        async with session.get(
            MESONET_URL,
            params=params,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            resp.raise_for_status()
            text = await resp.text()
    except Exception as exc:
        log.error("Mesonet fetch failed for %s (%s): %s", city_key, station, exc)
        return {}

    obs_by_ordinal: dict[int, list[float]] = {}
    date_by_ordinal: dict[int, str] = {}

    for line in text.splitlines():
        if line.startswith("#") or line.startswith("station") or not line.strip():
            continue
        parts = line.split(",")
        if len(parts) < 3:
            continue
        try:
            utc_ts = datetime.strptime(parts[1].strip(), "%Y-%m-%d %H:%M").replace(
                tzinfo=timezone.utc
            )
            temp_str = parts[2].strip()
            if temp_str in ("M", "T", ""):
                continue
            temp_f = float(temp_str)
        except (ValueError, IndexError):
            continue

        local_ts = utc_ts.astimezone(city_tz)
        ordinal  = local_ts.toordinal()
        obs_by_ordinal.setdefault(ordinal, []).append(temp_f)
        date_by_ordinal[ordinal] = local_ts.strftime("%Y-%m-%d")

    result: dict[str, float] = {}
    for ordinal, temps in obs_by_ordinal.items():
        if len(temps) < _MIN_OBS_PER_DAY:
            continue
        result[date_by_ordinal[ordinal]] = max(temps)

    log.info("  Mesonet %s (%s): %d valid days", city_key, station, len(result))
    return result


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _fmt(v: float | None, width: int = 6) -> str:
    if v is None:
        return f"{'--':>{width}}"
    return f"{v:>+{width}.1f}"


def _mean_or_none(vals: list[float]) -> float | None:
    return mean(vals) if vals else None


def _std_or_none(vals: list[float]) -> float | None:
    try:
        return stdev(vals) if len(vals) >= 2 else None
    except StatisticsError:
        return None


def print_summary_table(bias_rows: list[tuple], month_filter: list[int] | None) -> None:
    """Print source × month mean bias grid (all cities combined)."""
    bias_by: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    n_by:    dict[str, set] = defaultdict(set)

    for source, city, date_str, month, fcast, actual, bias in bias_rows:
        bias_by[source][month].append(bias)
        n_by[source].add((city, date_str))

    sources = [s for s in HIST_MODELS.values() if s in bias_by]
    months  = month_filter or list(range(1, 13))
    months  = [m for m in months if any(m in bias_by[s] for s in sources)]

    col_w = 6
    hdr = (
        f"{'Source':<22}"
        + "".join(f"{MONTH_NAMES[m-1]:>{col_w}}" for m in months)
        + f"{'N':>6}"
    )
    print("\n=== OPEN-METEO HISTORICAL BIAS vs ACTUAL (°F, negative = model ran cold) ===")
    print(hdr)
    print("-" * len(hdr))

    for src in sources:
        n    = len(n_by[src])
        row  = f"{src:<22}"
        row += "".join(_fmt(_mean_or_none(bias_by[src].get(m, []))) for m in months)
        row += f"{n:>6}"
        print(row)
    print()


def print_per_city(bias_rows: list[tuple], verbose: bool, month_filter: list[int] | None) -> None:
    """Print per-city breakdown, optionally with per-month detail."""
    by_city: dict[str, dict[str, dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for source, city, date_str, month, fcast, actual, bias in bias_rows:
        by_city[city][source][month].append(bias)

    sources = list(HIST_MODELS.values())

    for city in sorted(by_city):
        print(f"\n── {city.upper()} {'─' * max(1, 38 - len(city))}")

        if verbose:
            months_present: set[int] = set()
            for src in by_city[city].values():
                months_present.update(src.keys())
            months = sorted(
                m for m in (month_filter or list(range(1, 13)))
                if m in months_present
            )
            hdr = f"  {'Source':<24}" + "".join(f"{MONTH_NAMES[m-1]:>6}" for m in months)
            print(hdr)
            print("  " + "-" * (len(hdr) - 2))

        for src in sources:
            if src not in by_city[city]:
                continue
            all_biases: list[float] = []
            for month_biases in by_city[city][src].values():
                all_biases.extend(month_biases)
            if not all_biases:
                continue
            n       = len(all_biases)
            avg     = mean(all_biases)
            std_str = f" ±{stdev(all_biases):.1f}" if n >= 2 else ""

            if verbose:
                month_cells = "".join(
                    _fmt(_mean_or_none(by_city[city][src].get(m, [])))
                    for m in months
                )
                print(f"  {src:<24}{month_cells}  (n={n})")
            else:
                print(f"  {src:<24} {avg:+.2f}°F{std_str:<8}  n={n}")
    print()


def print_per_year(bias_rows: list[tuple], month_filter: list[int] | None) -> None:
    """Print year-over-year bias breakdown to detect climate drift.

    For each source, prints a year × month grid of mean bias, plus σ across
    years per month (stability) and a simple first→last-year delta (trend).

    Only source×city×month combinations with data in at least 3 years are shown
    in the drift summary to avoid noise from short ECMWF/ICON/GEM archives.
    """
    # Group: source → year → month → [bias]
    by_src_yr_mo: dict[str, dict[int, dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in bias_rows:
        source, date_str, month, bias = row[0], row[2], row[3], row[6]
        year = int(date_str[:4])
        by_src_yr_mo[source][year][month].append(bias)

    sources = [s for s in HIST_MODELS.values() if s in by_src_yr_mo]
    months  = month_filter or list(range(1, 13))

    print("\n=== YEAR-OVER-YEAR BIAS (°F) — mean across all cities per source ===")
    print("(σ_yrs = std dev of annual means; Δ = last_year − first_year)\n")

    for src in sources:
        yr_data = by_src_yr_mo[src]
        years   = sorted(yr_data)
        if not years:
            continue

        # Filter to months that appear in at least half the years
        active_months = [
            m for m in months
            if sum(1 for yr in years if yr_data[yr].get(m)) >= max(2, len(years) // 2)
        ]
        if not active_months:
            continue

        col_w = 6
        hdr = (
            f"  {'Year':<6}"
            + "".join(f"{MONTH_NAMES[m-1]:>{col_w}}" for m in active_months)
            + f"  {'Annual':>8}"
        )
        print(f"── {src} {'─' * max(1, 60 - len(src))}")
        print(hdr)
        print("  " + "─" * (len(hdr) - 2))

        annual_means: list[tuple[int, float]] = []
        month_col_vals: dict[int, list[float]] = defaultdict(list)  # month → annual means

        for yr in years:
            mo_means = [_mean_or_none(yr_data[yr].get(m, [])) for m in active_months]
            flat = [v for v in mo_means if v is not None]
            annual_mean = mean(flat) if flat else None
            if annual_mean is not None:
                annual_means.append((yr, annual_mean))
            row = f"  {yr:<6}"
            row += "".join(_fmt(v) for v in mo_means)
            row += f"  {_fmt(annual_mean, 8) if annual_mean is not None else '      --'}"
            print(row)
            for m, v in zip(active_months, mo_means):
                if v is not None:
                    month_col_vals[m].append(v)

        # σ across years per month, and first→last delta
        sigma_row = f"  {'σ_yrs':<6}"
        delta_row = f"  {'Δ':<6}"
        sigma_vals = []
        delta_vals = []
        for m in active_months:
            col = month_col_vals[m]
            s = _std_or_none(col)
            sigma_row += _fmt(s)
            sigma_vals.append(s)
            if len(col) >= 2:
                delta = col[-1] - col[0]
                delta_row += _fmt(delta)
                delta_vals.append(delta)
            else:
                delta_row += f"{'--':>6}"
                delta_vals.append(None)

        # Annual σ and Δ
        ann_vals = [m for _, m in annual_means]
        ann_sigma = _std_or_none(ann_vals)
        ann_delta = (ann_vals[-1] - ann_vals[0]) if len(ann_vals) >= 2 else None
        sigma_row += f"  {_fmt(ann_sigma, 8) if ann_sigma is not None else '      --'}"
        delta_row += f"  {_fmt(ann_delta, 8) if ann_delta is not None else '      --'}"

        print("  " + "─" * (len(hdr) - 2))
        print(sigma_row)
        print(delta_row)

        # Flag notable drift: |Δ| > 1.0°F across the full window
        drifting = [
            f"{MONTH_NAMES[m-1]}: {d:+.1f}°F"
            for m, d in zip(active_months, delta_vals)
            if d is not None and abs(d) >= 1.0
        ]
        if drifting:
            print(f"  ⚑ Notable drift (|Δ|≥1°F): {', '.join(drifting)}")
        print()


def write_csv(bias_rows: list[tuple], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "city", "date", "month", "forecast_f", "actual_f", "bias_f"])
        w.writerows(bias_rows)
    print(f"Detail rows written to {path}  ({len(bias_rows)} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Historical Open-Meteo bias vs METAR actuals")
    p.add_argument("--years",      type=int, default=2,
                   help="Lookback in years (default: 2)")
    p.add_argument("--cities",     nargs="+", default=None,
                   help="City codes to include, e.g. ny chi bos (default: all)")
    p.add_argument("--months",     nargs="+", type=int, default=None,
                   help="Months to include, e.g. 4 5 6 7 8 (default: all)")
    p.add_argument("--csv",        type=str, default=None,
                   help="Write detail rows to this CSV path")
    p.add_argument("--cache-dir",  type=str, default="data/bias_cache",
                   help="Directory for per-city JSON cache (default: data/bias_cache)")
    p.add_argument("--no-cache",   action="store_true",
                   help="Ignore existing cache and re-fetch all cities from the API")
    p.add_argument("--no-per-city", action="store_true",
                   help="Skip per-city breakdown")
    p.add_argument("--verbose",    action="store_true",
                   help="Per-city table shows month-by-month columns")
    p.add_argument("--per-year",   action="store_true",
                   help="Print year-over-year bias breakdown to detect climate drift")
    return p.parse_args()


async def run(args: argparse.Namespace) -> None:
    end_date   = date.today() - timedelta(days=1)
    start_date = date(end_date.year - args.years, end_date.month, end_date.day)

    cache_dir = Path(args.cache_dir)

    # Build city list
    city_pairs: list[tuple] = []
    for metric, (city_name, lat, lon, city_tz) in CITIES.items():
        if not metric.startswith("temp_high_"):
            continue
        city_code = metric.replace("temp_high_", "")
        if args.cities and city_code not in args.cities:
            continue
        station = KALSHI_STATION_IDS.get(metric)
        if station is None:
            log.warning("No station ID for %s — skipping", metric)
            continue
        city_pairs.append((metric, city_code, lat, lon, city_tz, station, str(city_tz)))

    log.info(
        "Fetching %d cities  %s → %s  (%d years)",
        len(city_pairs), start_date, end_date, args.years,
    )

    all_bias_rows: list[tuple] = []

    async with aiohttp.ClientSession() as session:
        for i, (metric, city_code, lat, lon, city_tz, station, tz_str) in enumerate(city_pairs):
            log.info("[%d/%d] %s", i + 1, len(city_pairs), city_code.upper())

            # Try cache first
            if not args.no_cache:
                cached = load_cache(cache_dir, city_code, start_date, end_date)
                if cached is not None:
                    if args.months:
                        cached = [r for r in cached if r[3] in args.months]
                    all_bias_rows.extend(cached)
                    n_dates = len(set(r[2] for r in cached if r[1] == city_code))
                    log.info("  → %d city-date pairs (from cache)", n_dates)
                    continue

            # Fetch actuals
            actuals = await fetch_mesonet_actuals(
                session, city_code, station, city_tz, start_date, end_date
            )
            if not actuals:
                log.warning("  No Mesonet data for %s — skipping", city_code)
                await asyncio.sleep(_MESONET_DELAY)
                continue
            await asyncio.sleep(_MESONET_DELAY)

            # Fetch forecasts
            forecasts = await fetch_openmeteo_forecasts(
                session, city_code, lat, lon, tz_str, start_date, end_date
            )
            if not forecasts:
                log.warning("  No OM historical data for %s — skipping", city_code)
                continue

            # Build bias rows for this city (all months — month filter applied at read time)
            city_rows: list[tuple] = []
            for src_name, date_to_forecast in forecasts.items():
                for date_str, fcast_f in date_to_forecast.items():
                    actual_f = actuals.get(date_str)
                    if actual_f is None:
                        continue
                    month  = int(date_str[5:7])
                    bias_f = round(fcast_f - actual_f, 3)
                    city_rows.append(
                        (src_name, city_code, date_str, month,
                         round(fcast_f, 2), round(actual_f, 2), bias_f)
                    )

            # Save full city rows to cache (before month filter)
            if city_rows:
                save_cache(cache_dir, city_code, start_date, end_date, city_rows)

            # Apply month filter and extend global list
            if args.months:
                city_rows = [r for r in city_rows if r[3] in args.months]
            all_bias_rows.extend(city_rows)

            n_dates = len(set(r[2] for r in city_rows if r[1] == city_code))
            log.info("  → %d city-date pairs with bias data", n_dates)

    if not all_bias_rows:
        print("No data — check --cities / --years / --months filters.")
        return

    print_summary_table(all_bias_rows, args.months)
    if not args.no_per_city:
        print_per_city(all_bias_rows, args.verbose, args.months)
    if args.per_year:
        print_per_year(all_bias_rows, args.months)
    if args.csv:
        write_csv(all_bias_rows, args.csv)


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run(args))
