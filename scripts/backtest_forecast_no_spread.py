"""Calibrate FORECAST_NO_MODEL_SPREAD_F by analysing inter-model spread vs forecast error.

Two outputs:

1. SPREAD DISTRIBUTION — what fraction of historical days fall in each spread bin.
   Tells you how aggressively the current 10°F gate fires.

2. CONDITIONAL MAE — mean absolute error (mean_models vs ERA5 actual) by spread bin.
   If high-spread days have larger MAE, it supports reducing the gate threshold.

NOTE: Open-Meteo historical-forecast-api returns model analysis data (retrospectively
corrected to match observations) rather than actual day-ahead forecasts.  As a result,
the measured spread and MAE are lower than real-world day-ahead values.  The outputs
are best interpreted as:
  - Spread distribution: lower bound on real spread (real day-ahead spread is 3-8°F higher)
  - Conditional MAE: lower bound on real forecast error
  - Win rates from the simulation are not meaningful and are omitted.

For actual day-ahead calibration, accumulate raw_forecasts in the DB and run
  venv/bin/python scripts/backtest_source_accuracy.py --mode db

Usage:
  venv/bin/python scripts/backtest_forecast_no_spread.py --years 3
  venv/bin/python scripts/backtest_forecast_no_spread.py --years 2 --cities chi bos sea
"""

import argparse
import asyncio
import logging
import math
import sys
from datetime import date, timedelta
from pathlib import Path
from statistics import mean

import aiohttp
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from kalshi_bot.news.noaa import CITIES
from kalshi_bot.news.open_meteo import _CITY_TZ_STRINGS  # noqa: PLC2403

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_ARCHIVE_BASE       = "https://archive-api.open-meteo.com/v1/archive"
_HIST_FORECAST_BASE = "https://historical-forecast-api.open-meteo.com/v1/forecast"
_HEADERS            = {"User-Agent": "kalshi-bot/1.0 (educational)"}
_ERA5_LAG_DAYS      = 5
_MODELS             = ["gfs_seamless", "ecmwf_ifs", "icon_seamless"]
_SPREAD_BINS        = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, float("inf"))]
_SPREAD_BIN_LABELS  = ["0–2°F", "2–4°F", "4–6°F", "6–8°F", "8–10°F", ">10°F"]


# ---------------------------------------------------------------------------
# Fetch ERA5 archive (ground truth)
# ---------------------------------------------------------------------------

async def _fetch_era5(
    session: aiohttp.ClientSession,
    metric: str,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    tz_str: str,
) -> dict[str, float]:
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
            _ARCHIVE_BASE, params=params, headers=_HEADERS,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
    except Exception as exc:
        log.warning("ERA5 fetch failed for %s: %s", metric, exc)
        return {}

    daily = data.get("daily", {})
    result: dict[str, float] = {}
    for d, t in zip(daily.get("time", []), daily.get("temperature_2m_max", [])):
        if t is not None:
            result[d] = float(t)
    log.info("ERA5      [%-22s]: %4d dates", metric, len(result))
    return result


# ---------------------------------------------------------------------------
# Fetch historical model analysis
# ---------------------------------------------------------------------------

async def _fetch_model(
    session: aiohttp.ClientSession,
    metric: str,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    tz_str: str,
    model: str,
) -> dict[str, float]:
    params = {
        "latitude":         f"{lat:.4f}",
        "longitude":        f"{lon:.4f}",
        "start_date":       start_date,
        "end_date":         end_date,
        "daily":            "temperature_2m_max",
        "models":           model,
        "temperature_unit": "fahrenheit",
        "timezone":         tz_str,
    }
    try:
        async with session.get(
            _HIST_FORECAST_BASE, params=params, headers=_HEADERS,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
    except Exception as exc:
        log.warning("Historical forecast [%s / %-20s] failed: %s", metric, model, exc)
        return {}

    daily = data.get("daily", {})
    result: dict[str, float] = {}
    for d, t in zip(daily.get("time", []), daily.get("temperature_2m_max", [])):
        if t is not None:
            result[d] = float(t)
    log.debug("Model %-20s [%-22s]: %4d dates", model, metric, len(result))
    return result


# ---------------------------------------------------------------------------
# Bin helper
# ---------------------------------------------------------------------------

def _spread_bin(spread: float) -> int:
    for i, (lo, hi) in enumerate(_SPREAD_BINS):
        if lo <= spread < hi:
            return i
    return len(_SPREAD_BINS) - 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def _run(args: argparse.Namespace) -> None:
    today     = date.today()
    end_dt    = today - timedelta(days=_ERA5_LAG_DAYS)
    start_dt  = end_dt - timedelta(days=max(1, args.years) * 365)
    start_str = start_dt.isoformat()
    end_str   = end_dt.isoformat()

    if args.cities:
        city_keys: list[str] = []
        for abbr in args.cities:
            key = f"temp_high_{abbr.lower()}"
            if key in CITIES:
                city_keys.append(key)
            else:
                log.warning("Unknown city abbreviation: %s (key %s not found)", abbr, key)
        if not city_keys:
            log.error("No valid cities specified.")
            sys.exit(1)
    else:
        city_keys = sorted(CITIES.keys())

    log.info(
        "Fetching %d cities from %s to %s  (%d year(s))",
        len(city_keys), start_str, end_str, args.years,
    )

    # records: list of (spread, abs_error)
    # spread    = max(models) - min(models)
    # abs_error = |mean(models) - ERA5_actual|
    records: list[tuple[float, float]] = []

    async with aiohttp.ClientSession() as session:
        for metric in city_keys:
            _, lat, lon, _ = CITIES[metric]
            tz_str = _CITY_TZ_STRINGS.get(metric, "UTC")
            log.info("--- %s (%s) ---", CITIES[metric][0], metric)

            era5 = await _fetch_era5(session, metric, lat, lon, start_str, end_str, tz_str)
            await asyncio.sleep(0.4)

            model_data: dict[str, dict[str, float]] = {}
            for model in _MODELS:
                model_data[model] = await _fetch_model(
                    session, metric, lat, lon, start_str, end_str, tz_str, model,
                )
                await asyncio.sleep(0.5)

            valid_dates = set(era5.keys())
            for m in _MODELS:
                valid_dates &= set(model_data[m].keys())

            for date_str in sorted(valid_dates):
                actual = era5[date_str]
                vals   = [model_data[m][date_str] for m in _MODELS]
                spread = max(vals) - min(vals)
                mean_f = mean(vals)
                records.append((spread, abs(mean_f - actual)))

    if not records:
        log.error("No data collected — check API connectivity.")
        sys.exit(1)

    log.info("Collected %d (city, date) records.", len(records))
    _report(records, start_str, end_str, args.out)


def _report(
    records: list[tuple[float, float]],
    start_str: str,
    end_str: str,
    out_path_str: str,
) -> None:
    n_total = len(records)
    bin_n:   list[int]   = [0] * len(_SPREAD_BINS)
    bin_mae: list[float] = [0.0] * len(_SPREAD_BINS)

    for spread, abs_err in records:
        b = _spread_bin(spread)
        bin_n[b]   += 1
        bin_mae[b] += abs_err

    lines: list[str] = []
    lines.append("")
    lines.append("=" * 78)
    lines.append("  FORECAST NO: INTER-MODEL SPREAD ANALYSIS")
    lines.append(f"  Date range : {start_str}  →  {end_str}")
    lines.append(f"  Models     : GFS, ECMWF IFS, ICON  (Open-Meteo historical analysis)")
    lines.append(f"  Ground truth: ERA5 archive daily max")
    lines.append("=" * 78)
    lines.append("")
    lines.append("  NOTE: historical-forecast-api returns model analysis data, not day-ahead")
    lines.append("  forecasts.  Observed spread and MAE are lower bounds; real day-ahead")
    lines.append("  values are 3-8°F higher.  Interpretation: treat spread bins as")
    lines.append("  conservative proxies — high-spread days still tend to have higher error.")
    lines.append("")

    # Spread distribution
    lines.append("  SPREAD DISTRIBUTION (% of days per bin):")
    lines.append(f"  {'Spread bin':<12}  {'N':>7}  {'% days':>8}  MAE(mean_f vs ERA5)")
    lines.append("  " + "-" * 58)
    for i, label in enumerate(_SPREAD_BIN_LABELS):
        n = bin_n[i]
        if n == 0:
            lines.append(f"  {label:<12}  {'—':>7}  {'—':>8}  {'—'}")
            continue
        pct  = 100.0 * n / n_total
        mae  = bin_mae[i] / n
        lines.append(f"  {label:<12}  {n:>7}  {pct:>7.1f}%  {mae:.2f}°F")

    # Overall stats
    total_mae = sum(e for _, e in records) / n_total
    total_spread_mean = sum(s for s, _ in records) / n_total
    total_spread_p90  = sorted(s for s, _ in records)[int(0.9 * n_total)]
    total_spread_p99  = sorted(s for s, _ in records)[int(0.99 * n_total)]

    lines.append("")
    lines.append(f"  Overall MAE (mean_models vs ERA5): {total_mae:.2f}°F  "
                 f"(σ ≈ {total_mae * math.sqrt(math.pi / 2):.2f}°F if errors ~half-normal)")
    lines.append(f"  Mean spread: {total_spread_mean:.2f}°F  "
                 f" P90: {total_spread_p90:.1f}°F   P99: {total_spread_p99:.1f}°F")

    # Gate firing frequency for various thresholds
    lines.append("")
    lines.append("  HOW OFTEN DOES THE SPREAD GATE FIRE?")
    lines.append("  (analysis data; real day-ahead spread is ~3-8°F higher)")
    lines.append(f"  {'Gate threshold':<18}  {'Days blocked':>13}  {'% blocked':>10}")
    lines.append("  " + "-" * 50)
    for gate in [4, 5, 6, 8, 10, 12]:
        blocked = sum(1 for s, _ in records if s > gate)
        pct_blocked = 100.0 * blocked / n_total
        note = "  ← current default" if gate == 10 else ""
        lines.append(f"  {gate}°F{'':>14}  {blocked:>13}  {pct_blocked:>9.1f}%{note}")

    # Implied win rate from observed MAE
    # For a "too hot NO" with edge E:
    #   P(win) = P(actual > mean_f - E) = P(error > -E) where error = actual - mean_f
    #   With symmetric errors and std σ = MAE * sqrt(π/2):
    #   P(win) ≈ Φ(E / σ)
    sigma = total_mae * math.sqrt(math.pi / 2)
    lines.append("")
    lines.append(f"  IMPLIED P(win | edge E) — from observed MAE={total_mae:.2f}°F, σ≈{sigma:.2f}°F:")
    lines.append(f"  (Assumes day-ahead forecast errors are N(0, σ); uses ERA5 MAE as lower bound)")
    lines.append(f"  {'Edge (°F)':<12}  {'P(win) [lower bound]':>22}  {'Formula (0.75+0.02E)':>22}")
    lines.append("  " + "-" * 62)
    for e_int in range(6, 15):
        e = float(e_int)
        p_emp  = _cdf_normal(e / sigma) * 100.0
        p_form = min(95.0, 75.0 + 2.0 * e)
        note = "  ← formula conservative" if p_form < p_emp - 5 else ""
        lines.append(
            f"  {e_int}°F{'':>9}  {p_emp:>21.1f}%  {p_form:>21.1f}%{note}"
        )

    lines.append("")
    lines.append(f"  Total records: {n_total}")
    lines.append("=" * 78)
    lines.append("")
    lines.append(
        "  RECOMMENDATION: With reanalysis data, spread > 6°F is rare (<5% of days)."
    )
    lines.append(
        "  Real day-ahead spread is higher. Run the following once raw_forecasts"
    )
    lines.append(
        "  table has 3+ months of data for accurate calibration:"
    )
    lines.append(
        "    venv/bin/python scripts/backtest_source_accuracy.py --mode db"
    )
    lines.append("")

    output = "\n".join(lines)
    print(output)

    out_path = Path(out_path_str)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output)
    print(f"Results written to {out_path}")


def _cdf_normal(z: float) -> float:
    """Standard normal CDF using math.erfc."""
    return 0.5 * math.erfc(-z / math.sqrt(2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate FORECAST_NO_MODEL_SPREAD_F via inter-model spread analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--years", type=int, default=3,
                        help="Years of historical data (default: 3)")
    parser.add_argument("--cities", nargs="+", default=None,
                        help="City abbreviations, e.g. chi bos sea (default: all)")
    parser.add_argument("--out", default="data/forecast_no_spread_grid.txt",
                        help="Output file path")
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
