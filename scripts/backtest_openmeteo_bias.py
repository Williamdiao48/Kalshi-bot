#!/usr/bin/env python3
"""Historical Open-Meteo forecast bias vs. METAR actuals.

Fetches 2 years of historical daily max temperature forecasts from the
Open-Meteo Historical Forecast API for all Kalshi city locations, compares
to actual daily highs from Iowa State Mesonet ASOS, and outputs a bias
table (forecast − actual °F) grouped by source × month and per city.

Negative bias = model ran cold (underestimated peak temperature).

This covers ~730 city-date pairs per model vs. the ~10 days in the bot's
raw_forecasts DB, giving enough data to calibrate per-city/per-month spread gates.

Usage:
    venv/bin/python scripts/backtest_openmeteo_bias.py
    venv/bin/python scripts/backtest_openmeteo_bias.py --years 2 --months 4 5 6 7 8
    venv/bin/python scripts/backtest_openmeteo_bias.py --cities ny chi bos --csv /tmp/bias.csv
    venv/bin/python scripts/backtest_openmeteo_bias.py --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import math
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, stdev

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
            # Default blended — single temperature_2m_max field
            values = daily.get("temperature_2m_max", [])
            results[HIST_MODELS[None]] = {
                d: v for d, v in zip(dates, values) if v is not None
            }
        else:
            # Named models — fields named temperature_2m_max_{model}
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
    """Return {date_str: actual_max_f} using Iowa State Mesonet ASOS data.

    Adapted from backtest_peak_hour.py:_fetch_city_obs — we only need the
    daily max, not the time of peak.
    """
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

    # Group observations by local date ordinal
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


def print_summary_table(
    bias_rows: list[tuple],
    month_filter: list[int] | None,
) -> None:
    """Print source × month mean bias grid."""
    # bias_rows: (source, city, date_str, month, forecast_f, actual_f, bias_f)
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


def print_per_city(
    bias_rows: list[tuple],
    verbose: bool,
    month_filter: list[int] | None,
) -> None:
    """Print per-city breakdown, optionally with per-month detail."""
    # Group: city → source → month → [bias]
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
    p.add_argument("--years",    type=int, default=2,
                   help="Lookback in years (default: 2)")
    p.add_argument("--cities",   nargs="+", default=None,
                   help="City codes to include, e.g. ny chi bos (default: all)")
    p.add_argument("--months",   nargs="+", type=int, default=None,
                   help="Months to include, e.g. 4 5 6 7 8 (default: all)")
    p.add_argument("--csv",      type=str, default=None,
                   help="Write detail rows to this CSV path")
    p.add_argument("--no-per-city", action="store_true",
                   help="Skip per-city breakdown")
    p.add_argument("--verbose",  action="store_true",
                   help="Per-city table shows month-by-month columns")
    return p.parse_args()


async def run(args: argparse.Namespace) -> None:
    end_date   = date.today() - timedelta(days=1)  # yesterday (today not complete)
    start_date = date(end_date.year - args.years,
                      end_date.month, end_date.day)

    # Build city list: only high-temp cities (temp_high_*) since we want daily HIGH
    city_pairs: list[tuple[str, str, float, float, object, str]] = []
    for metric, (city_name, lat, lon, city_tz) in CITIES.items():
        city_code = metric.replace("temp_high_", "")
        if args.cities and city_code not in args.cities:
            continue
        station = KALSHI_STATION_IDS.get(metric)
        if station is None:
            log.warning("No station ID for %s — skipping", metric)
            continue
        tz_str = str(city_tz)
        city_pairs.append((metric, city_code, lat, lon, city_tz, station, tz_str))

    log.info(
        "Fetching %d cities  %s → %s  (%d years)",
        len(city_pairs), start_date, end_date, args.years,
    )

    all_bias_rows: list[tuple] = []

    async with aiohttp.ClientSession() as session:
        for i, (metric, city_code, lat, lon, city_tz, station, tz_str) in enumerate(city_pairs):
            log.info("[%d/%d] %s", i + 1, len(city_pairs), city_code.upper())

            # Fetch actuals from Mesonet
            actuals = await fetch_mesonet_actuals(
                session, city_code, station, city_tz, start_date, end_date
            )
            if not actuals:
                log.warning("  No Mesonet data for %s — skipping", city_code)
                await asyncio.sleep(_MESONET_DELAY)
                continue
            await asyncio.sleep(_MESONET_DELAY)

            # Fetch historical Open-Meteo forecasts
            forecasts = await fetch_openmeteo_forecasts(
                session, city_code, lat, lon, tz_str, start_date, end_date
            )
            if not forecasts:
                log.warning("  No OM historical data for %s — skipping", city_code)
                continue

            # Build bias rows
            for src_name, date_to_forecast in forecasts.items():
                for date_str, fcast_f in date_to_forecast.items():
                    actual_f = actuals.get(date_str)
                    if actual_f is None:
                        continue
                    month = int(date_str[5:7])
                    if args.months and month not in args.months:
                        continue
                    bias_f = round(fcast_f - actual_f, 3)
                    all_bias_rows.append(
                        (src_name, city_code, date_str, month,
                         round(fcast_f, 2), round(actual_f, 2), bias_f)
                    )

            n_dates = len(set(r[2] for r in all_bias_rows if r[1] == city_code))
            log.info("  → %d city-date pairs with bias data", n_dates)

    if not all_bias_rows:
        print("No data — check --cities / --years filters.")
        return

    print_summary_table(all_bias_rows, args.months)
    if not args.no_per_city:
        print_per_city(all_bias_rows, args.verbose, args.months)
    if args.csv:
        write_csv(all_bias_rows, args.csv)


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run(args))
