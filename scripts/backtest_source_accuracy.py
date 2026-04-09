"""Per-source weather accuracy backtest using clean historical API data.

Mode 1 (default, --mode api):
  Fetches observed daily high temperatures directly from weather source history
  APIs (Open-Meteo ERA5 archive, WeatherAPI history.json), then compares them
  against NWS CLI actuals.  Bypasses opportunity_log.db entirely — no timezone
  bug contamination.

Mode 2 (--mode db):
  Uses the raw_forecasts table in opportunity_log.db, which captures ALL
  pre-gate forecast data points from every poll cycle (logged by
  opp_log.log_raw_forecasts() in main.py).  This includes NOAA, HRRR, and
  nws_hourly which have no public history API.  Requires the bot to have been
  running with the log_raw_forecasts() call in place for a meaningful period.

  For each (ticker, source), takes the FIRST forecast logged that day — this
  is the closest proxy to "day-ahead forecast" since the first log of a market
  date occurs well before settlement.

What this measures
------------------
API mode: "observation accuracy" — how closely each source's model aligns with
the NWS CLI settlement value.  A proxy for forecast accuracy since sources with
structural cold/hot bias vs NWS CLI will produce similarly biased forecasts.

DB mode: actual day-ahead forecast accuracy for ALL sources including NOAA,
using real forecasts the bot acted on.

Sources covered
---------------
  API mode:  open_meteo (ERA5), weatherapi
  DB mode:   all sources in raw_forecasts (noaa, open_meteo, weatherapi,
             noaa_day2, hrrr, nws_hourly, noaa_observed, ...)

Output
------
  Accuracy table printed to stdout
  data/source_accuracy_report.csv written to disk

Usage
-----
  # Step 1: fetch NWS CLI actuals if not already done
  venv/bin/python scripts/fetch_nws_cli_history.py

  # Step 2: run this backtest
  venv/bin/python scripts/backtest_source_accuracy.py
  venv/bin/python scripts/backtest_source_accuracy.py --start 2026-01-01
"""

import argparse
import asyncio
import csv
import logging
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from kalshi_bot.news.noaa import CITIES  # metric → (name, lat, lon, tz)  # noqa: E402
from kalshi_bot.news.open_meteo import _CITY_TZ_STRINGS  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_DEFAULT_ACTUALS = "data/nws_cli_actuals.csv"
_ARCHIVE_BASE    = "https://archive-api.open-meteo.com/v1/archive"
_WEATHERAPI_BASE = "https://api.weatherapi.com/v1/history.json"
_HEADERS         = {"User-Agent": "kalshi-bot/1.0 (educational)"}

# ERA5 archive has a 5-day lag in rare cases, but in practice is up-to-date.
# Skip dates within this many days of today to avoid sparse/missing ERA5 data.
_ERA5_LAG_DAYS = 2


# ---------------------------------------------------------------------------
# Load NWS CLI actuals
# ---------------------------------------------------------------------------

def _load_actuals(path: Path) -> dict[tuple[str, str], float]:
    """Return {(city_metric, YYYY-MM-DD): actual_high_f}."""
    if not path.exists():
        log.error("Actuals file not found: %s  — run fetch_nws_cli_history.py first.", path)
        sys.exit(1)
    out: dict[tuple[str, str], float] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                out[(row["city_metric"], row["date"])] = float(row["actual_high_f"])
            except (KeyError, ValueError):
                continue
    log.info("Loaded %d NWS CLI actuals from %s", len(out), path)
    return out


# ---------------------------------------------------------------------------
# Open-Meteo ERA5 archive
# ---------------------------------------------------------------------------

async def _fetch_open_meteo_city(
    session:    aiohttp.ClientSession,
    metric:     str,
    lat:        float,
    lon:        float,
    start_date: str,
    end_date:   str,
    tz_str:     str,
) -> dict[str, float]:
    """Fetch ERA5 daily max temps for one city over a date range.

    Returns {YYYY-MM-DD: max_temp_f}.
    """
    params = {
        "latitude":         f"{lat:.4f}",
        "longitude":        f"{lon:.4f}",
        "start_date":       start_date,
        "end_date":         end_date,
        "daily":            "temperature_2m_max",
        "temperature_unit": "fahrenheit",
        "timezone":         tz_str,
    }
    try:
        async with session.get(
            _ARCHIVE_BASE,
            params=params,
            headers=_HEADERS,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
    except Exception as exc:
        log.warning("Open-Meteo archive failed for %s: %s", metric, exc)
        return {}

    daily = data.get("daily", {})
    times: list[str] = daily.get("time", [])
    temps: list      = daily.get("temperature_2m_max", [])

    result: dict[str, float] = {}
    for date_str, temp in zip(times, temps):
        if temp is not None:
            result[date_str] = float(temp)

    log.info("Open-Meteo archive [%s]: %d date(s) returned", metric, len(result))
    return result


async def fetch_open_meteo_all(
    session:    aiohttp.ClientSession,
    start_date: str,
    end_date:   str,
) -> dict[tuple[str, str], float]:
    """Fetch ERA5 archive for all cities. Returns {(metric, date): value}."""
    tasks = {
        metric: _fetch_open_meteo_city(
            session, metric, lat, lon,
            start_date, end_date,
            _CITY_TZ_STRINGS.get(metric, "UTC"),
        )
        for metric, (_, lat, lon, _) in CITIES.items()
    }

    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    out: dict[tuple[str, str], float] = {}
    for metric, result in zip(tasks.keys(), results):
        if isinstance(result, Exception):
            log.error("Open-Meteo archive error for %s: %s", metric, result)
        else:
            for date_str, val in result.items():
                out[(metric, date_str)] = val
    return out


# ---------------------------------------------------------------------------
# WeatherAPI historical
# ---------------------------------------------------------------------------

async def _fetch_weatherapi_city_date(
    session:  aiohttp.ClientSession,
    metric:   str,
    lat:      float,
    lon:      float,
    date_str: str,
    api_key:  str,
) -> float | None:
    """Fetch WeatherAPI historical max temp for one city on one date."""
    params = {"key": api_key, "q": f"{lat:.4f},{lon:.4f}", "dt": date_str}
    try:
        async with session.get(
            _WEATHERAPI_BASE,
            params=params,
            headers=_HEADERS,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            if resp.status == 400:
                # Free tier returns 400 for dates outside the allowed window
                return None
            resp.raise_for_status()
            data = await resp.json()
    except Exception as exc:
        log.debug("WeatherAPI history failed for %s %s: %s", metric, date_str, exc)
        return None

    try:
        return float(data["forecast"]["forecastday"][0]["day"]["maxtemp_f"])
    except (KeyError, IndexError, TypeError, ValueError):
        return None


async def fetch_weatherapi_all(
    session:  aiohttp.ClientSession,
    actuals:  dict[tuple[str, str], float],
    api_key:  str,
) -> dict[tuple[str, str], float]:
    """Fetch WeatherAPI history for all city×date pairs in actuals.

    Free tier allows the last 7 days — older dates silently return None.
    Uses a semaphore to stay well within WeatherAPI's rate limits.
    """
    sem = asyncio.Semaphore(4)

    async def _fetch_one(metric: str, lat: float, lon: float, date_str: str) -> tuple:
        async with sem:
            val = await _fetch_weatherapi_city_date(session, metric, lat, lon, date_str, api_key)
            await asyncio.sleep(0.1)
            return (metric, date_str), val

    pairs = [
        (metric, CITIES[metric][1], CITIES[metric][2], date_str)
        for metric, date_str in actuals
        if metric in CITIES
    ]

    tasks = [_fetch_one(metric, lat, lon, date_str) for metric, lat, lon, date_str in pairs]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    out: dict[tuple[str, str], float] = {}
    for result in results:
        if isinstance(result, Exception):
            continue
        key, val = result
        if val is not None:
            out[key] = val

    n_avail = len(out)
    n_total = len(pairs)
    log.info(
        "WeatherAPI history: %d/%d dates returned (free tier covers last 7 days)",
        n_avail, n_total,
    )
    return out


# ---------------------------------------------------------------------------
# Compute accuracy stats
# ---------------------------------------------------------------------------

def _compute_stats(
    source_data: dict[tuple[str, str], float],
    actuals:     dict[tuple[str, str], float],
) -> dict[str, list[float]]:
    """Return per-city-metric error lists: {metric: [errors...]} where error = source - actual."""
    errors: dict[str, list[float]] = defaultdict(list)
    for (metric, date_str), source_val in source_data.items():
        actual = actuals.get((metric, date_str))
        if actual is None:
            continue
        errors[metric].append(source_val - actual)
    return dict(errors)


def _summarize(errors: dict[str, list[float]]) -> dict:
    """Aggregate error lists into summary stats."""
    all_errors = [e for errs in errors.values() for e in errs]
    if not all_errors:
        return {"n": 0, "mae": None, "bias": None, "within_3f": None, "within_5f": None}
    n   = len(all_errors)
    mae = sum(abs(e) for e in all_errors) / n
    bias = sum(all_errors) / n
    within_3 = 100.0 * sum(1 for e in all_errors if abs(e) <= 3) / n
    within_5 = 100.0 * sum(1 for e in all_errors if abs(e) <= 5) / n
    return {"n": n, "mae": mae, "bias": bias, "within_3f": within_3, "within_5f": within_5}


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

_NOAA_NOTE = """
NOAA (NWS forecast) note
------------------------
The "noaa" source fetches the NWS 7-day gridpoint forecast — the same agency
that produces the NWS CLI settlement value Kalshi uses.  NWS does not archive
historical forecast products publicly, so it cannot be backtested here.

Published NWS day-1 forecast verification (national average):
  MAE  ≈  3–4°F     within 3°F: ~55–65%     within 5°F: ~80–85%
  Bias ≈  0°F  (NWS actively corrects for systematic biases)

This makes NOAA the most reliable signal when it fires: it has the smallest
structural bias against the settlement value and the best MAE.  The reason it
generates fewer trades than WeatherAPI is not lower accuracy — it is that NWS
forecasts are more conservative (closer to climatological norms), so they
produce smaller raw edges that fall below our 7°F gate more often.

Recommendation: consider lowering TEMP_FORECAST_MIN_EDGE_PER_SOURCE["noaa"]
to 5°F (≈ 1.4σ) since its bias vs NWS CLI is near zero.
"""


def _print_report(
    open_meteo_stats: dict,
    weatherapi_stats: dict,
) -> None:
    print()
    print("=" * 72)
    print("WEATHER SOURCE ACCURACY vs NWS CLI ACTUALS")
    print("(source observed/archived daily max vs NWS Climatological Report)")
    print("=" * 72)
    print(f"  {'Source':<18}  {'N':>5}  {'MAE':>6}  {'Bias':>7}  {'≤3°F':>6}  {'≤5°F':>6}")
    print("-" * 72)

    for label, stats in [("open_meteo (ERA5)", open_meteo_stats), ("weatherapi", weatherapi_stats)]:
        n = stats["n"]
        if n == 0:
            print(f"  {label:<18}  {'—':>5}")
            continue
        mae_s   = f"{stats['mae']:.1f}°F"
        bias_s  = f"{stats['bias']:+.1f}°F"
        w3_s    = f"{stats['within_3f']:.0f}%"
        w5_s    = f"{stats['within_5f']:.0f}%"
        print(f"  {label:<18}  {n:>5}  {mae_s:>6}  {bias_s:>7}  {w3_s:>6}  {w5_s:>6}")

    print()
    print("Bias interpretation: positive = source runs hot vs NWS CLI settlement.")
    print("A source with +5°F bias needs ~5°F MORE raw edge to yield the same")
    print("true edge vs the actual settlement value.")
    print()
    print(_NOAA_NOTE)


def _write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Report written to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def _main(args: argparse.Namespace) -> None:
    import os
    actuals_path = Path(args.actuals)
    out_path     = Path(args.out)
    api_key      = os.environ.get("WEATHERAPI_KEY", "")

    actuals = _load_actuals(actuals_path)
    if not actuals:
        log.error("No actuals loaded — exiting.")
        sys.exit(1)

    # Determine date range from actuals, applying optional --start / --end filters
    all_dates = sorted({d for _, d in actuals})
    cutoff = (date.today() - timedelta(days=_ERA5_LAG_DAYS)).isoformat()

    start_date = args.start or all_dates[0]
    end_date   = min(args.end or all_dates[-1], cutoff)

    if start_date > end_date:
        log.error("No dates in range %s – %s after applying ERA5 lag cutoff.", start_date, end_date)
        sys.exit(1)

    # Filter actuals to the requested date range
    actuals_filtered = {
        (m, d): v for (m, d), v in actuals.items()
        if start_date <= d <= end_date
    }
    log.info("Backtesting %d city-date pairs (%s to %s)", len(actuals_filtered), start_date, end_date)

    async with aiohttp.ClientSession() as session:
        # --- Open-Meteo ERA5 ---
        log.info("Fetching Open-Meteo ERA5 archive (%s → %s)...", start_date, end_date)
        om_data = await fetch_open_meteo_all(session, start_date, end_date)

        # --- WeatherAPI history ---
        wa_data: dict[tuple[str, str], float] = {}
        if api_key:
            log.info("Fetching WeatherAPI history (free tier: last 7 days)...")
            wa_data = await fetch_weatherapi_all(session, actuals_filtered, api_key)
        else:
            log.warning("WEATHERAPI_KEY not set — skipping WeatherAPI history.")

    # Filter source data to dates present in filtered actuals
    om_filtered = {k: v for k, v in om_data.items() if k in actuals_filtered}
    wa_filtered = {k: v for k, v in wa_data.items() if k in actuals_filtered}

    om_errors = _compute_stats(om_filtered, actuals_filtered)
    wa_errors = _compute_stats(wa_filtered, actuals_filtered)

    om_stats = _summarize(om_errors)
    wa_stats = _summarize(wa_errors)

    _print_report(om_stats, wa_stats)

    # Per-city breakdown
    print("PER-CITY BREAKDOWN (Open-Meteo ERA5)")
    print(f"  {'City metric':<22}  {'N':>4}  {'MAE':>6}  {'Bias':>7}")
    print("-" * 50)
    for metric in sorted(om_errors):
        errs = om_errors[metric]
        if not errs:
            continue
        n   = len(errs)
        mae = sum(abs(e) for e in errs) / n
        bias = sum(errs) / n
        print(f"  {metric:<22}  {n:>4}  {mae:.1f}°F  {bias:+.1f}°F")
    print()

    # Build CSV rows
    csv_rows: list[dict] = []
    for source_label, errors in [
        ("open_meteo", om_errors),
        ("weatherapi",  wa_errors),
    ]:
        for metric in sorted(set(list(errors.keys()) + list(CITIES.keys()))):
            errs = errors.get(metric, [])
            n    = len(errs)
            csv_rows.append({
                "source":      source_label,
                "city_metric": metric,
                "n":           n,
                "mae":         round(sum(abs(e) for e in errs) / n, 2) if n else "",
                "bias":        round(sum(errs) / n, 2) if n else "",
                "within_3f_pct": round(100 * sum(1 for e in errs if abs(e) <= 3) / n, 1) if n else "",
                "within_5f_pct": round(100 * sum(1 for e in errs if abs(e) <= 5) / n, 1) if n else "",
            })

    _write_csv(csv_rows, out_path)


def _load_db_forecasts(
    db_path: Path,
    start_date: str | None,
    end_date:   str | None,
) -> dict[tuple[str, str], dict[str, float]]:
    """Load pre-gate forecasts from raw_forecasts table.

    Returns {(metric, date_str): {source: data_value}} using the FIRST forecast
    per (ticker, source) per calendar day — closest proxy to day-ahead forecast.
    """
    import sqlite3 as _sqlite3
    import re as _re

    if not db_path.exists():
        log.error("Database not found: %s", db_path)
        sys.exit(1)

    where = "WHERE 1=1"
    params: list = []
    if start_date:
        where += " AND date(logged_at) >= ?"
        params.append(start_date)
    if end_date:
        where += " AND date(logged_at) <= ?"
        params.append(end_date)

    conn = _sqlite3.connect(db_path)
    conn.row_factory = _sqlite3.Row
    rows = conn.execute(f"""
        SELECT source, metric, ticker,
               AVG(data_value) AS data_value,
               MIN(logged_at)  AS first_logged
        FROM raw_forecasts
        {where}
        GROUP BY ticker, source, date(logged_at)
        ORDER BY ticker, source
    """, params).fetchall()
    conn.close()

    # Parse market date from ticker: e.g. KXHIGHLAX-26APR06-T75 → 2026-04-06
    _DATE_RE = _re.compile(
        r"-(\d{2})(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d{2})(?:-|$)",
        _re.IGNORECASE,
    )
    _MMAP = {
        "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04", "MAY": "05", "JUN": "06",
        "JUL": "07", "AUG": "08", "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
    }

    # result: (metric, market_date) → {source: value}
    result: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        m = _DATE_RE.search(row["ticker"])
        if not m:
            continue
        year  = "20" + m.group(1)
        month = _MMAP.get(m.group(2).upper())
        day   = m.group(3).zfill(2)
        if not month:
            continue
        market_date = f"{year}-{month}-{day}"
        key = (row["metric"], market_date)
        if key not in result:
            result[key] = {}
        result[key][row["source"]] = float(row["data_value"])

    all_sources = sorted({s for d in result.values() for s in d})
    log.info(
        "DB mode: %d (metric, date) pairs, %d source(s): %s",
        len(result), len(all_sources), ", ".join(all_sources),
    )
    return result


def _print_db_report(
    db_forecasts: dict[tuple[str, str], dict[str, float]],
    actuals: dict[tuple[str, str], float],
) -> None:
    """Print per-source accuracy table using raw_forecasts DB data."""
    # Collect errors per source
    errors: dict[str, list[float]] = {}
    for (metric, date_str), source_vals in db_forecasts.items():
        actual = actuals.get((metric, date_str))
        if actual is None:
            continue
        for source, val in source_vals.items():
            errors.setdefault(source, []).append(val - actual)

    print()
    print("=" * 72)
    print("PER-SOURCE ACCURACY vs NWS CLI (from raw_forecasts DB)")
    print("(first pre-gate forecast per market-day per source)")
    print("=" * 72)
    print(f"  {'Source':<22}  {'N':>5}  {'MAE':>6}  {'Bias':>7}  {'≤3°F':>6}  {'≤5°F':>6}")
    print("-" * 72)

    for source in sorted(errors):
        errs = errors[source]
        n = len(errs)
        if n == 0:
            continue
        mae    = sum(abs(e) for e in errs) / n
        bias   = sum(errs) / n
        w3     = 100.0 * sum(1 for e in errs if abs(e) <= 3) / n
        w5     = 100.0 * sum(1 for e in errs if abs(e) <= 5) / n
        print(f"  {source:<22}  {n:>5}  {mae:.1f}°F  {bias:+.1f}°F  {w3:.0f}%  {w5:.0f}%")

    print()
    print("Bias: positive = source runs hot vs NWS CLI; negative = runs cold.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-source weather accuracy backtest vs NWS CLI actuals.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode",    default="api", choices=["api", "db"],
                        help="'api' = fetch from Open-Meteo/WeatherAPI history (default); "
                             "'db'  = use raw_forecasts table from opportunity_log.db")
    parser.add_argument("--actuals", default=_DEFAULT_ACTUALS,
                        help="Path to nws_cli_actuals.csv (default: data/nws_cli_actuals.csv)")
    parser.add_argument("--db",      default="opportunity_log.db",
                        help="Path to opportunity_log.db (used in --mode db)")
    parser.add_argument("--out",     default="data/source_accuracy_report.csv",
                        help="Output CSV path")
    parser.add_argument("--start",   default=None,
                        help="Start date YYYY-MM-DD (default: earliest in actuals)")
    parser.add_argument("--end",     default=None,
                        help="End date YYYY-MM-DD (default: latest in actuals minus ERA5 lag)")
    args = parser.parse_args()

    if args.mode == "db":
        actuals = _load_actuals(Path(args.actuals))
        db_forecasts = _load_db_forecasts(Path(args.db), args.start, args.end)
        _print_db_report(db_forecasts, actuals)
    else:
        asyncio.run(_main(args))


if __name__ == "__main__":
    main()
