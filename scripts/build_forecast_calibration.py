"""
Historical weather model calibration for forecast_no strategy.

Answers: when GFS/ECMWF/GEM/ICON predicted daily high = T°F,
and the actual NWS observed high = O°F, what is the distribution of (T - O)?

Specifically per city × model:
  - Bias (mean error): does the model systematically over/underpredict?
  - RMSE / MAE: typical error magnitude
  - Win rate by edge bin: P(T - O > threshold | signal edge ≥ E°F)
    → "when model predicts 3°F above band, how often is actual below band?"

Data sources:
  Phase 1: Our own raw_forecasts table (May 1–16 2026, exact NOAA CLI values)
  Phase 2: Open-Meteo Historical Forecast API (GFS/ECMWF/GEM/ICON, 2024–2026)
            + IEM ASOS daily summaries (observed daily max, 2024–2026)

Run:
  venv/bin/python scripts/build_forecast_calibration.py
  venv/bin/python scripts/build_forecast_calibration.py --start 2025-01-01
  venv/bin/python scripts/build_forecast_calibration.py --phase1-only
"""

import argparse
import asyncio
import json
import math
import sqlite3
import sys
import os
from collections import defaultdict
from datetime import date, timedelta

import aiohttp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kalshi_bot.cities import CITIES, LOW_CITIES

TRADES_DB = "data/db/opportunity_log.db"

# Open-Meteo models → our raw_forecast source names
OM_MODELS = {
    "gfs_seamless":   "open_meteo_gfs",
    "ecmwf_ifs025":   "open_meteo_ecmwf",
    "gem_seamless":   "open_meteo_gem",
    "icon_seamless":  "open_meteo_icon",
}

# HRRR is fetched separately (single-model request, key not suffixed)
# Available from ~2022 onwards via historical-forecast API
HRRR_MODEL = "gfs_hrrr"
HRRR_SOURCE = "hrrr"

# metric key → (IEM station, IEM network, ASOS ICAO)
# IEM network codes: XX_ASOS where XX is state abbreviation
# Note: DCA (Reagan National) is physically in Virginia → VA_ASOS, not DC_ASOS
IEM_STATIONS: dict[str, tuple[str, str]] = {
    "temp_high_lax": ("LAX", "CA_ASOS"),
    "temp_high_den": ("DEN", "CO_ASOS"),
    "temp_high_chi": ("MDW", "IL_ASOS"),   # Midway — matches cities.py coords
    "temp_high_ny":  ("NYC", "NY_ASOS"),   # Central Park ASOS
    "temp_high_mia": ("MIA", "FL_ASOS"),
    "temp_high_aus": ("AUS", "TX_ASOS"),
    "temp_high_dal": ("DAL", "TX_ASOS"),   # Love Field
    "temp_high_bos": ("BOS", "MA_ASOS"),
    "temp_high_hou": ("HOU", "TX_ASOS"),   # Hobby
    "temp_high_dfw": ("DFW", "TX_ASOS"),
    "temp_high_sfo": ("SFO", "CA_ASOS"),
    "temp_high_sea": ("SEA", "WA_ASOS"),
    "temp_high_phx": ("PHX", "AZ_ASOS"),
    "temp_high_phl": ("PHL", "PA_ASOS"),
    "temp_high_atl": ("ATL", "GA_ASOS"),
    "temp_high_msp": ("MSP", "MN_ASOS"),
    "temp_high_dca": ("DCA", "VA_ASOS"),
    "temp_high_las": ("LAS", "NV_ASOS"),
    "temp_high_okc": ("OKC", "OK_ASOS"),
    "temp_high_sat": ("SAT", "TX_ASOS"),
    "temp_high_msy": ("MSY", "LA_ASOS"),
    # Low temp cities (same stations, different metric prefix)
    "temp_low_lax": ("LAX", "CA_ASOS"),
    "temp_low_den": ("DEN", "CO_ASOS"),
    "temp_low_chi": ("MDW", "IL_ASOS"),
    "temp_low_ny":  ("NYC", "NY_ASOS"),
    "temp_low_mia": ("MIA", "FL_ASOS"),
    "temp_low_aus": ("AUS", "TX_ASOS"),
    "temp_low_bos": ("BOS", "MA_ASOS"),
    "temp_low_hou": ("HOU", "TX_ASOS"),
    "temp_low_dfw": ("DFW", "TX_ASOS"),
    "temp_low_sfo": ("SFO", "CA_ASOS"),
    "temp_low_sea": ("SEA", "WA_ASOS"),
    "temp_low_phx": ("PHX", "AZ_ASOS"),
    "temp_low_phl": ("PHL", "PA_ASOS"),
    "temp_low_atl": ("ATL", "GA_ASOS"),
    "temp_low_msp": ("MSP", "MN_ASOS"),
    "temp_low_dca": ("DCA", "VA_ASOS"),
    "temp_low_las": ("LAS", "NV_ASOS"),
    "temp_low_okc": ("OKC", "OK_ASOS"),
    "temp_low_sat": ("SAT", "TX_ASOS"),
    "temp_low_msy": ("MSY", "LA_ASOS"),
}


# ──────────────────────────────────────────────
# Phase 1: Mine existing raw_forecasts table
# ──────────────────────────────────────────────

def phase1_from_raw_forecasts(db: str) -> dict[str, list[tuple[str, float, float]]]:
    """
    Returns: {model_source: [(metric, model_forecast, noaa_observed), ...]}
    Only includes rows where we have both a model forecast and noaa_observed
    for the exact same (metric, date bucket).
    """
    con = sqlite3.connect(db)

    # Get distinct (metric, date, source, avg_data_value) — use the first
    # logged_at for each (metric, date) to get the morning forecast snapshot.
    # We define "date" as the calendar date of logged_at UTC.
    rows = con.execute("""
        SELECT source,
               metric,
               date(logged_at) as obs_date,
               AVG(data_value)  as avg_val
        FROM raw_forecasts
        WHERE source IN ('hrrr', 'nws_hourly', 'noaa_observed',
                         'open_meteo', 'open_meteo_gfs', 'open_meteo_ecmwf',
                         'open_meteo_gem', 'open_meteo_icon')
          AND metric LIKE 'temp_%'
          AND data_value IS NOT NULL
        GROUP BY source, metric, obs_date
    """).fetchall()
    con.close()

    # Build observed map: (metric, date) → observed_value
    observed: dict[tuple[str, str], float] = {}
    model_rows: dict[str, list[tuple]] = defaultdict(list)

    for source, metric, obs_date, avg_val in rows:
        if source == "noaa_observed":
            observed[(metric, obs_date)] = avg_val
        else:
            model_rows[source].append((metric, obs_date, avg_val))

    result: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    for source, entries in model_rows.items():
        for metric, obs_date, model_val in entries:
            obs_val = observed.get((metric, obs_date))
            if obs_val is not None:
                result[source].append((metric, model_val, obs_val))

    return result


# ──────────────────────────────────────────────
# Phase 2: Historical API pulls
# ──────────────────────────────────────────────

async def fetch_hrrr_historical(
    session: aiohttp.ClientSession,
    lat: float,
    lon: float,
    start: str,
    end: str,
    is_high: bool,
) -> dict[str, float]:
    """
    Returns {date_str: daily_max_or_min_fahrenheit} for HRRR.
    HRRR is CONUS-only, available from ~2022. Single-model request
    returns unkeyed temperature_2m_max (no model suffix in key).
    """
    var = "temperature_2m_max" if is_high else "temperature_2m_min"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": var,
        "temperature_unit": "fahrenheit",
        "timezone": "UTC",
        "models": HRRR_MODEL,
    }
    try:
        async with session.get(
            "https://historical-forecast-api.open-meteo.com/v1/forecast",
            params=params,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
    except Exception as e:
        print(f"  [HRRR] error for ({lat},{lon}): {e}", file=sys.stderr)
        return {}

    daily = data.get("daily", {})
    dates = daily.get("time", [])
    vals  = daily.get(var, [])
    return {d: v for d, v in zip(dates, vals) if v is not None}


async def fetch_om_historical(
    session: aiohttp.ClientSession,
    lat: float,
    lon: float,
    start: str,
    end: str,
    is_high: bool,
) -> dict[str, dict[str, float]]:
    """
    Returns {model_key: {date_str: daily_max_or_min_fahrenheit}}
    model_key is the OM model name (gfs_seamless, etc.)
    """
    var = "temperature_2m_max" if is_high else "temperature_2m_min"
    models = ",".join(OM_MODELS.keys())

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": var,
        "temperature_unit": "fahrenheit",
        "timezone": "UTC",
        "models": models,
    }

    try:
        async with session.get(
            "https://historical-forecast-api.open-meteo.com/v1/forecast",
            params=params,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
    except Exception as e:
        print(f"  [OM forecast] error for ({lat},{lon}): {e}", file=sys.stderr)
        return {}

    daily = data.get("daily", {})
    dates = daily.get("time", [])
    result = {}
    for model in OM_MODELS:
        # Response keys are e.g. temperature_2m_max_gfs_seamless
        key = f"{var}_{model}"
        vals = daily.get(key, [])
        result[model] = {d: v for d, v in zip(dates, vals) if v is not None}
    return result


async def fetch_iem_observed(
    session: aiohttp.ClientSession,
    station: str,
    network: str,
    start_year: int,
    end_year: int,
    is_high: bool,
) -> dict[str, float]:
    """
    Returns {date_str: max_or_min_temp_fahrenheit} from IEM ASOS daily summaries.
    """
    field = "max_tmpf" if is_high else "min_tmpf"
    result = {}

    for year in range(start_year, end_year + 1):
        url = (
            f"https://mesonet.agron.iastate.edu/api/1/daily.json"
            f"?station={station}&network={network}&year={year}"
        )
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                resp.raise_for_status()
                data = await resp.json()
                for row in data.get("data", []):
                    d = row.get("date")
                    v = row.get(field)
                    if d and v is not None:
                        try:
                            result[d] = float(v)
                        except (TypeError, ValueError):
                            pass
        except Exception as e:
            print(f"  [IEM] {station}/{year}: {e}", file=sys.stderr)
        await asyncio.sleep(0.2)

    return result


# ──────────────────────────────────────────────
# Analysis / reporting
# ──────────────────────────────────────────────

def compute_stats(errors: list[float]) -> dict:
    if not errors:
        return {}
    n = len(errors)
    mean = sum(errors) / n
    rmse = math.sqrt(sum(e**2 for e in errors) / n)
    mae  = sum(abs(e) for e in errors) / n
    errors_s = sorted(errors)
    p25 = errors_s[n // 4]
    p75 = errors_s[3 * n // 4]
    return {"n": n, "bias": mean, "rmse": rmse, "mae": mae, "p25": p25, "p75": p75}


def win_rate_by_edge(
    pairs: list[tuple[float, float]],   # (model, observed)
    threshold: float = 1.0,
    edge_bins: list[float] = None,
) -> list[tuple[str, int, float]]:
    """
    For NO_HIGH: win = model > observed + threshold  (actual came in below model)
    For given edge bins, compute n and win_rate where edge = model - observed.
    Returns list of (bin_label, n, win_rate).
    """
    if edge_bins is None:
        edge_bins = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    results = []
    for e in edge_bins:
        subset = [(m, o) for m, o in pairs if m - o >= e]
        if not subset:
            results.append((f"≥{e:.0f}°F", 0, float("nan")))
            continue
        wins = sum(1 for m, o in subset if m - o > threshold)
        results.append((f"≥{e:.0f}°F", len(subset), wins / len(subset)))
    return results


def print_calibration(
    source_name: str,
    data: dict[str, list[tuple[str, float, float]]],  # source → [(metric, model, obs)]
    phase: str,
) -> None:
    print(f"\n{'='*80}")
    print(f"{phase} — Model: {source_name}")
    print(f"{'='*80}")

    # All cities combined
    all_pairs = [(m, o) for metric, m, o in data.get(source_name, [])]
    if not all_pairs:
        print("  No data.")
        return

    errors = [m - o for m, o in all_pairs]
    stats = compute_stats(errors)
    print(f"\n  ALL CITIES combined: n={stats['n']}, bias={stats['bias']:+.2f}°F, "
          f"RMSE={stats['rmse']:.2f}°F, MAE={stats['mae']:.2f}°F, "
          f"p25={stats['p25']:+.2f} p75={stats['p75']:+.2f}")

    # Win rate (NO_HIGH: actual came in below model - 1°F buffer)
    print(f"\n  Win rate (NO_HIGH): P(model - actual ≥ threshold | signal_edge ≥ E)")
    print(f"  {'Edge':>6}  {'n':>5}  {'WR(0°F)':>8}  {'WR(1°F)':>8}  {'WR(2°F)':>8}")
    bins = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    for edge_min in bins:
        subset = [(m, o) for m, o in all_pairs if m - o >= edge_min]
        if len(subset) < 5:
            continue
        wr0 = sum(1 for m, o in subset if m > o) / len(subset)
        wr1 = sum(1 for m, o in subset if m - o > 1) / len(subset)
        wr2 = sum(1 for m, o in subset if m - o > 2) / len(subset)
        print(f"  {f'≥{edge_min:.0f}°F':>6}  {len(subset):>5}  {wr0:>7.1%}  {wr1:>7.1%}  {wr2:>7.1%}")

    # Per city breakdown
    city_data: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for metric, m, o in data.get(source_name, []):
        city_key = metric.split("_")[-1]  # e.g. "lax"
        city_data[city_key].append((m, o))

    print(f"\n  Per-city stats (bias, RMSE, WR when model edge ≥ 3°F):")
    print(f"  {'City':>6}  {'n':>5}  {'bias':>7}  {'RMSE':>7}  {'WR≥3°':>7}  {'n≥3°':>5}")
    print("  " + "-"*50)
    for city in sorted(city_data):
        pairs = city_data[city]
        errs = [m - o for m, o in pairs]
        s = compute_stats(errs)
        subset_3 = [(m, o) for m, o in pairs if m - o >= 3]
        wr3 = sum(1 for m, o in subset_3 if m - o > 1) / len(subset_3) if subset_3 else float("nan")
        wr3_str = f"{wr3:.1%}" if not math.isnan(wr3) else "   —"
        print(f"  {city:>6}  {s['n']:>5}  {s['bias']:>+6.2f}°  {s['rmse']:>6.2f}°  {wr3_str:>7}  {len(subset_3):>5}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

async def main(start_date: str, phase1_only: bool) -> None:
    # ── Phase 1: raw_forecasts ──
    print("=== Phase 1: raw_forecasts calibration (May 1–16 2026) ===")
    p1_data = phase1_from_raw_forecasts(TRADES_DB)

    # Restructure: source → list of (metric, model, observed)
    p1_by_source: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    for source, triples in p1_data.items():
        for metric, model_val, obs_val in triples:
            p1_by_source[source].append((metric, model_val, obs_val))

    for source in sorted(p1_by_source):
        print_calibration(source, p1_by_source, "Phase 1")

    if phase1_only:
        return

    # ── Phase 2: historical API pull ──
    print(f"\n\n=== Phase 2: Historical calibration ({start_date} → today) ===")
    today = date.today().isoformat()

    # Only process high-temp cities for now (low-temp same stations, similar results)
    metrics_to_run = {k: v for k, v in CITIES.items() if k in IEM_STATIONS}

    start_year = int(start_date[:4])
    end_year = int(today[:4])

    # Accumulate: (om_model_name) → {metric → [(model_val, obs_val)]}
    p2_data: dict[str, dict[str, list[tuple[float, float]]]] = {
        m: defaultdict(list) for m in OM_MODELS
    }
    # HRRR stored separately (only goes back to ~2022)
    hrrr_start = max(start_date, "2022-01-01")
    hrrr_data: dict[str, list[tuple[float, float]]] = defaultdict(list)

    async with aiohttp.ClientSession() as session:
        for metric, (city_name, lat, lon, tz) in metrics_to_run.items():
            iem_station, iem_network = IEM_STATIONS[metric]
            is_high = metric.startswith("temp_high_")

            print(f"\n  Fetching {city_name} ({metric})...", end="", flush=True)

            # Fetch IEM observed, OM multi-model, and HRRR concurrently
            iem_task  = asyncio.create_task(
                fetch_iem_observed(session, iem_station, iem_network,
                                   start_year, end_year, is_high)
            )
            om_task   = asyncio.create_task(
                fetch_om_historical(session, lat, lon, start_date, today, is_high)
            )
            hrrr_task = asyncio.create_task(
                fetch_hrrr_historical(session, lat, lon, hrrr_start, today, is_high)
            )

            iem_obs, om_fc, hrrr_fc = await asyncio.gather(iem_task, om_task, hrrr_task)
            print(f" IEM={len(iem_obs)}d, OM={len(next(iter(om_fc.values()), {}))}d,"
                  f" HRRR={len(hrrr_fc)}d")

            if not iem_obs:
                print(f"    [warn] no IEM data for {iem_station}", file=sys.stderr)
                continue

            for om_model, date_vals in om_fc.items():
                for d, model_val in date_vals.items():
                    obs_val = iem_obs.get(d)
                    if obs_val is not None and model_val is not None:
                        p2_data[om_model][metric].append((model_val, obs_val))

            for d, model_val in hrrr_fc.items():
                obs_val = iem_obs.get(d)
                if obs_val is not None and model_val is not None:
                    hrrr_data[metric].append((model_val, obs_val))

            await asyncio.sleep(0.3)

    # Print Phase 2 results — HRRR first (most relevant to our strategy)
    hrrr_flat: dict[str, list[tuple[str, float, float]]] = {
        HRRR_SOURCE: [
            (metric, m, o)
            for metric, pairs in hrrr_data.items()
            for m, o in pairs
        ]
    }
    print_calibration(HRRR_SOURCE, hrrr_flat, "Phase 2 — HRRR (gfs_hrrr)")

    for om_model in OM_MODELS:
        source_label = OM_MODELS[om_model]
        flat: dict[str, list[tuple[str, float, float]]] = {
            source_label: [
                (metric, m, o)
                for metric, pairs in p2_data[om_model].items()
                for m, o in pairs
            ]
        }
        print_calibration(source_label, flat, f"Phase 2 — {om_model}")

    # Summary: all models including HRRR
    print(f"\n\n{'='*80}")
    print("SUMMARY: Model accuracy per city (Phase 2, IEM ASOS observed)")
    print(f"{'='*80}")
    print(f"  {'City':>6}  {'HRRR RMSE':>10}  {'HRRR bias':>10}  {'GFS RMSE':>9}  {'GFS bias':>9}  {'n(HRRR)':>8}")
    print("  " + "-"*65)

    all_cities = sorted(set(
        metric.split("_")[-1]
        for metric in hrrr_data
    ))
    for city in all_cities:
        metric = f"temp_high_{city}"
        h_pairs = hrrr_data.get(metric, [])
        g_pairs = p2_data["gfs_seamless"].get(metric, [])

        h_errs = [m - o for m, o in h_pairs]
        g_errs = [m - o for m, o in g_pairs]

        h_rmse = math.sqrt(sum(e**2 for e in h_errs) / len(h_errs)) if h_errs else float("nan")
        h_bias = sum(h_errs) / len(h_errs) if h_errs else float("nan")
        g_rmse = math.sqrt(sum(e**2 for e in g_errs) / len(g_errs)) if g_errs else float("nan")
        g_bias = sum(g_errs) / len(g_errs) if g_errs else float("nan")

        better = " ← HRRR better" if h_rmse < g_rmse else ""
        print(f"  {city:>6}  {h_rmse:>9.2f}°  {h_bias:>+9.2f}°  {g_rmse:>8.2f}°  {g_bias:>+8.2f}°  {len(h_pairs):>8}{better}")

    # Min-edge thresholds: for ~90% win probability on NO_HIGH (1°F band)
    # Win rate ≈ Φ((edge - bias)/σ) + 1 - Φ((edge + 1 - bias)/σ)
    # Solve for edge where P(NO loses) = P(error ∈ [edge, edge+1]) ≤ 10%
    import statistics
    print(f"\n  Implied min_edge_f for 90% win probability (NO_HIGH, 1°F band):")
    print(f"  {'City':>6}  {'HRRR σ':>8}  {'90% edge':>10}  {'current gate':>13}  {'assessment':>12}")
    print("  " + "-"*60)
    for city in all_cities:
        metric = f"temp_high_{city}"
        h_pairs = hrrr_data.get(metric, [])
        if len(h_pairs) < 30:
            continue
        h_errs = [m - o for m, o in h_pairs]
        sigma = statistics.stdev(h_errs)
        bias  = sum(h_errs) / len(h_errs)
        # We want P(bias ≤ error ≤ bias + sigma*z) ≤ 0.10 within a 1°F window
        # Approximate: edge where normal CDF in [edge, edge+1] = 0.10
        # → edge ≈ bias + 1.28*sigma (gives 10% of distribution beyond edge)
        edge_90 = bias + 1.28 * sigma
        current = 3.0
        ok = "OK" if edge_90 <= current else "RAISE to {:.1f}°".format(math.ceil(edge_90 * 2) / 2)
        print(f"  {city:>6}  {sigma:>7.2f}°  {edge_90:>9.1f}°  {current:>12.1f}°  {ok}")

    all_cities_set = set(
        metric.split("_")[-1]
        for om_model in OM_MODELS
        for metric in p2_data[om_model]
    )
    print(f"\n\n{'='*80}")
    print("LEGACY SUMMARY: Best OM model per city (lowest RMSE, Phase 2)")
    print(f"{'='*80}")
    print(f"  {'City':>6}  {'Best model':>14}  {'RMSE':>6}  {'Bias':>7}")
    print("  " + "-"*45)
    for city in sorted(all_cities_set):
        best_model, best_rmse, best_bias, best_wr = None, float("inf"), 0.0, 0.0
        for om_model in OM_MODELS:
            pairs = p2_data[om_model].get(f"temp_high_{city}", [])
            if len(pairs) < 30:
                continue
            errs = [m - o for m, o in pairs]
            rmse = math.sqrt(sum(e**2 for e in errs) / len(errs))
            bias = sum(errs) / len(errs)
            subset_3 = [(m, o) for m, o in pairs if m - o >= 3]
            wr3 = sum(1 for m, o in subset_3 if m - o > 1) / len(subset_3) if subset_3 else 0
            if rmse < best_rmse:
                best_rmse, best_bias, best_wr = rmse, bias, wr3
                best_model = OM_MODELS[om_model]
        if best_model:
            print(f"  {city:>6}  {best_model:>14}  {best_rmse:>5.2f}°  {best_bias:>+6.2f}°")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2024-01-01",
                        help="Start date for historical pull (default: 2024-01-01)")
    parser.add_argument("--phase1-only", action="store_true",
                        help="Only run Phase 1 (raw_forecasts, no API calls)")
    args = parser.parse_args()
    asyncio.run(main(args.start, args.phase1_only))
