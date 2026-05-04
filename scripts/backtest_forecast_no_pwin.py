"""Calibrate the p_estimate formula for Forecast-NO signals.

Production formula: p_estimate = min(0.95, 0.75 + 0.02 * min_edge)

This script measures:
  1. The actual MAE of the model consensus (mean of GFS, ECMWF, ICON) vs ERA5 archive.
  2. The empirical P(win | edge E): derived from the observed error distribution.
  3. How well the production formula matches the empirical win rates.

Approach:
  - Fetches Open-Meteo historical model analysis (GFS, ECMWF, ICON) as forecast proxies.
  - Fetches ERA5 archive as settlement ground truth.
  - For each (city, date): records error = mean(models) - ERA5_actual.
  - Derives P(win | edge E) = P(actual > mean_f - E) = Φ(E / σ) where σ = MAE × √(π/2).

NOTE: historical-forecast-api returns model analysis data, not actual day-ahead forecasts.
The observed MAE (~1-2°F) is lower than real day-ahead MAE (~3-5°F), so the derived P(win)
is an UPPER BOUND. For actual calibration, collect real forecast data via raw_forecasts table.

Usage:
  venv/bin/python scripts/backtest_forecast_no_pwin.py --years 3
  venv/bin/python scripts/backtest_forecast_no_pwin.py --years 1 --cities chi bos sea
"""

import argparse
import asyncio
import logging
import math
import sys
from collections import defaultdict
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
_EDGE_BINS          = list(range(6, 17))  # 6..16


def _p_formula(edge: float) -> float:
    return min(0.95, 0.75 + 0.02 * edge)


def _cdf_normal(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2))


def _p_win_from_sigma(edge: float, sigma: float) -> float:
    """P(actual > mean_f - edge) = Φ(edge / sigma) for N(0, sigma) errors."""
    if sigma <= 0:
        return 1.0
    return _cdf_normal(edge / sigma)


# ---------------------------------------------------------------------------
# Fetch ERA5 archive
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
                log.warning("Unknown city: %s", abbr)
        if not city_keys:
            log.error("No valid cities.")
            sys.exit(1)
    else:
        city_keys = sorted(CITIES.keys())

    log.info(
        "Fetching %d cities from %s to %s (%d year(s))",
        len(city_keys), start_str, end_str, args.years,
    )

    # signed_errors[metric] = list of (mean_f - actual) signed errors
    signed_errors: dict[str, list[float]] = defaultdict(list)
    n_total = 0

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
                mean_f = mean(vals)
                signed_errors[metric].append(mean_f - actual)
                n_total += 1

    if n_total == 0:
        log.error("No data collected — check API connectivity.")
        sys.exit(1)

    log.info("Collected %d (city, date) records.", n_total)
    _report(signed_errors, n_total, start_str, end_str, args.out)


def _report(
    signed_errors: dict[str, list[float]],
    n_total: int,
    start_str: str,
    end_str: str,
    out_path_str: str,
) -> None:
    all_errors = [e for errs in signed_errors.values() for e in errs]
    mae_global  = sum(abs(e) for e in all_errors) / len(all_errors)
    bias_global = sum(all_errors) / len(all_errors)
    sigma_global = mae_global * math.sqrt(math.pi / 2)

    lines: list[str] = []
    lines.append("")
    lines.append("=" * 84)
    lines.append("  FORECAST NO: p_estimate CALIBRATION")
    lines.append(f"  Date range : {start_str}  →  {end_str}")
    lines.append(f"  Models     : GFS, ECMWF IFS, ICON  (Open-Meteo historical analysis)")
    lines.append(f"  Ground truth: ERA5 archive daily max")
    lines.append("=" * 84)
    lines.append("")
    lines.append("  NOTE: historical-forecast-api returns analysis data, not day-ahead")
    lines.append("  forecasts.  Real day-ahead MAE is ~3-5°F; these results (~1-2°F)")
    lines.append("  represent an upper bound on P(win).  Use as a calibration floor:")
    lines.append("  formula must yield P(win) <= the upper bound at each edge bin.")
    lines.append("")

    # Overall accuracy stats
    lines.append("  OVERALL FORECAST ACCURACY (mean_models vs ERA5):")
    lines.append(f"  MAE  = {mae_global:.2f}°F   Bias = {bias_global:+.2f}°F   σ ≈ {sigma_global:.2f}°F")
    lines.append("")

    # Per-city MAE
    lines.append("  PER-CITY MAE:")
    lines.append(f"  {'City metric':<22}  {'N':>5}  {'MAE':>7}  {'Bias':>7}  {'σ':>7}")
    lines.append("  " + "-" * 56)
    city_stats = []
    for metric in sorted(signed_errors):
        errs = signed_errors[metric]
        n = len(errs)
        if n == 0:
            continue
        mae  = sum(abs(e) for e in errs) / n
        bias = sum(errs) / n
        sig  = mae * math.sqrt(math.pi / 2)
        city_stats.append((metric, n, mae, bias, sig))
        lines.append(
            f"  {metric:<22}  {n:>5}  {mae:>6.2f}°F  {bias:>+6.2f}°F  {sig:>6.2f}°F"
        )

    # Implied P(win) table — using global σ
    lines.append("")
    lines.append("  IMPLIED P(win | edge E) — derived from observed σ:")
    lines.append(f"  σ (lower bound) = {sigma_global:.2f}°F   "
                 f"Real-world σ ≈ {sigma_global + 3:.1f}–{sigma_global + 5:.1f}°F")
    lines.append("")
    lines.append(
        f"  {'Edge':>6}  {'P(win) lb':>12}  "
        f"{'P(win) ub':>12}  {'Formula':>10}  Note"
    )
    lines.append(
        f"  {'(°F)':>6}  {'(σ_lower)':>12}  "
        f"{'(σ_upper)':>12}  {'':>10}"
    )
    lines.append("  " + "-" * 72)

    sigma_lower = sigma_global                       # lower bound (analysis data)
    sigma_upper = sigma_global + 4.0                 # rough upper bound (day-ahead)

    off_by_5pp: list[int] = []
    for e in _EDGE_BINS:
        p_lb   = 100.0 * _p_win_from_sigma(e, sigma_lower)
        p_ub   = 100.0 * _p_win_from_sigma(e, sigma_upper)
        p_form = 100.0 * _p_formula(e)
        note = ""
        if p_form < p_ub - 5:
            note = "  ← formula too low vs upper bound"
            off_by_5pp.append(e)
        elif p_form > p_lb + 5:
            note = "  ← formula exceeds lower bound"
        lines.append(
            f"  {e:>6}  {p_lb:>11.1f}%  {p_ub:>11.1f}%  "
            f"{p_form:>9.1f}%{note}"
        )

    # Error percentile table
    lines.append("")
    lines.append("  FORECAST ERROR PERCENTILES (|mean_models - ERA5|):")
    abs_errs_sorted = sorted(abs(e) for e in all_errors)
    n = len(abs_errs_sorted)
    pcts = [50, 75, 90, 95, 99]
    for pct in pcts:
        val = abs_errs_sorted[int(pct / 100.0 * n)]
        lines.append(f"  P{pct:>2} = {val:.2f}°F")

    # Recommendation
    lines.append("")
    lines.append("  RECOMMENDATION:")
    lines.append("  " + "-" * 72)

    if not off_by_5pp:
        lines.append(
            "  Formula p = min(0.95, 0.75 + 0.02 * edge) is consistent with"
        )
        lines.append(
            f"  the upper-bound P(win) range from observed analysis data."
        )
        lines.append(
            "  No adjustment needed — formula is appropriately conservative."
        )
    else:
        lines.append(
            f"  Formula underestimates P(win) by > 5pp at edges: "
            + ", ".join(f"{e}°F" for e in off_by_5pp)
        )
        lines.append(
            "  This means the formula is CONSERVATIVE — you are getting more wins"
        )
        lines.append(
            "  than the formula predicts, so Kelly sizing is under-allocated."
        )
        # Suggest fitted formula
        xs = [float(e) for e in _EDGE_BINS]
        ys = [_p_win_from_sigma(float(e), sigma_upper) for e in _EDGE_BINS]
        n_pts = len(xs)
        sx  = sum(xs)
        sy  = sum(ys)
        sxy = sum(x * y for x, y in zip(xs, ys))
        sxx = sum(x * x for x in xs)
        denom = n_pts * sxx - sx * sx
        if denom != 0:
            slope     = (n_pts * sxy - sx * sy) / denom
            intercept = (sy - slope * sx) / n_pts
            lines.append("")
            lines.append(
                f"  Conservative linear fit (based on σ_upper={sigma_upper:.1f}°F):"
            )
            lines.append(
                f"    p = min(0.95, {intercept:.3f} + {slope:.4f} * edge)"
            )
            lines.append(
                f"  Current: p = min(0.95, 0.75 + 0.02 * edge)"
            )

    lines.append("")
    lines.append(
        "  For precise calibration, run with 3+ months of raw_forecasts data:"
    )
    lines.append(
        "    venv/bin/python scripts/backtest_source_accuracy.py --mode db"
    )
    lines.append("")
    lines.append(f"  Total (city, date) records: {n_total}")
    lines.append("=" * 84)
    lines.append("")

    output = "\n".join(lines)
    print(output)

    out_path = Path(out_path_str)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output)
    print(f"Results written to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate p_estimate formula for Forecast-NO signals.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--years", type=int, default=3,
                        help="Years of historical data (default: 3)")
    parser.add_argument("--cities", nargs="+", default=None,
                        help="City abbreviations, e.g. chi bos sea (default: all)")
    parser.add_argument("--out", default="data/forecast_no_pwin_grid.txt",
                        help="Output file path")
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
