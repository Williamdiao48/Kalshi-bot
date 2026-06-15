"""
Build historical training data for band_arb ML model.

For each city × date (default 2022-01-01 → yesterday):
  - IEM ASOS hourly observations → simulated running max/min at entry hour
  - Open-Meteo historical forecast API → GFS, HRRR, ECMWF, GEM, ICON daily forecasts
  - IEM ASOS daily actual → label (did NO win?)

Signal simulation:
  KXHIGH NO fires when round(running_max) > band_ceil.
  KXLOWT NO fires when round(running_min) > band_ceil.
  We simulate entry at ENTRY_HOUR_HIGH_UTC / ENTRY_HOUR_LOW_UTC.
  band_ceil = round(running_obs) - 1  (tightest band the signal fires on).
  NO wins if actual daily max/min > band_ceil (obs stayed outside the band).

Output: data/backtest/band_arb_training.csv
Cache:  data/backtest/band_arb_hist_cache.json  (raw API responses; re-runs skip fetched data)

Usage:
  venv/bin/python scripts/build_band_arb_training_data.py
  venv/bin/python scripts/build_band_arb_training_data.py --start 2023-01-01
  venv/bin/python scripts/build_band_arb_training_data.py --no-fetch
"""

import argparse
import asyncio
import csv
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path
from statistics import median

import aiohttp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kalshi_bot.cities import CITIES, LOW_CITIES
from scripts.build_forecast_calibration import (
    IEM_STATIONS,
    fetch_om_historical, fetch_hrrr_historical, fetch_iem_observed,
)

# ── Config ────────────────────────────────────────────────────────────────────

CACHE_PATH = Path("data/backtest/band_arb_hist_cache.json")
OUT_CSV    = Path("data/backtest/band_arb_training.csv")

# UTC entry hours by timezone offset (approximate local time targets).
# KXHIGH: target ~2pm local (daily max well-established).
# KXLOWT: target ~9am local (overnight low locked in for eastern cities).
# Keys match ZoneInfo zone names used in cities.py.
_ENTRY_HOURS: dict[str, tuple[int, int]] = {
    #                                     high  low
    "America/New_York":    (19, 14),   # 2pm / 9am ET
    "America/Chicago":     (20, 15),   # 2pm / 9am CT
    "America/Denver":      (21, 16),   # 2pm / 9am MT
    "America/Los_Angeles": (22, 17),   # 2pm / 9am PT
    "America/Phoenix":     (21, 16),   # 2pm / 9am MST (no DST)
}
_DEFAULT_ENTRY_HOURS = (19, 14)

# Minimum margin (running_obs - band_ceil) to emit a training row.
# Below this the signal is ambiguous (rounding could go either way).
MIN_MARGIN = 0.5

CSV_FIELDS = [
    "metric", "date", "is_high", "month", "day_of_year",
    "entry_hour_utc",
    "running_obs",          # METAR running max/min at entry hour (obs is inside the band)
    "band_lo",              # band lower boundary (floor of running_obs)
    "band_ceil",            # band upper boundary (band_lo + 1)
    "margin_to_ceil",       # band_ceil - running_obs (how close to overshooting)
    "margin_to_floor",      # running_obs - band_lo (how close to undershooting)
    "actual_f",             # IEM official daily max/min
    "gfs_f",                # GFS daily forecast
    "hrrr_f",               # HRRR daily forecast
    "ecmwf_f",              # ECMWF daily forecast
    "gem_f",                # GEM daily forecast
    "icon_f",               # ICON daily forecast
    "consensus_f",          # median of available model forecasts
    "model_spread",         # max - min across available models
    "n_models",             # number of model forecasts available
    "n_models_above_ceil",  # models predicting above band_ceil (overshoot risk)
    "n_models_below_floor", # models predicting below band_lo (undershoot risk)
    "gfs_vs_ceil",          # gfs_f - band_ceil (positive = GFS expects overshoot above)
    "hrrr_vs_ceil",         # hrrr_f - band_ceil
    "consensus_vs_ceil",    # consensus_f - band_ceil
    "gfs_vs_floor",         # gfs_f - band_lo (negative = GFS expects undershoot below)
    "hrrr_vs_floor",        # hrrr_f - band_lo
    "consensus_vs_floor",   # consensus_f - band_lo
    "won",                  # 1 = YES wins (actual landed in [band_lo, band_ceil])
]

# ── IEM hourly fetch ──────────────────────────────────────────────────────────

async def fetch_iem_hourly(
    session: aiohttp.ClientSession,
    station: str,
    start: str,
    end: str,
) -> dict[str, dict[int, float]]:
    """
    Returns {date_str: {hour_utc: temp_f}} from IEM ASOS hourly observations.
    Uses report_type=3 (METAR) with UTC timestamps.
    """
    y1, m1, d1 = start.split("-")
    y2, m2, d2 = end.split("-")
    url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
    params = {
        "station": station,
        "data": "tmpf",
        "year1": y1, "month1": m1, "day1": d1,
        "year2": y2, "month2": m2, "day2": d2,
        "tz": "UTC",
        "format": "onlycomma",
        "latlon": "no",
        "report_type": "3",
    }
    result: dict[str, dict[int, float]] = {}
    try:
        async with session.get(
            url, params=params,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            resp.raise_for_status()
            text = await resp.text()
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("station") or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue
            valid_str = parts[1].strip()   # e.g. "2022-01-01 14:56"
            tmpf_str  = parts[2].strip()
            if tmpf_str in ("M", "", "None"):
                continue
            try:
                dt_date, dt_time = valid_str.split(" ")
                hour = int(dt_time.split(":")[0])
                temp = float(tmpf_str)
                result.setdefault(dt_date, {})[hour] = temp
            except (ValueError, IndexError):
                continue
    except Exception as e:
        print(f"  [IEM hourly] {station}: {e}", file=sys.stderr)
    return result


async def fetch_om_hourly_forecast(
    session: aiohttp.ClientSession,
    lat: float,
    lon: float,
    start: str,
    end: str,
    model: str,  # e.g. "gfs_seamless" or "gfs_hrrr"
) -> dict[str, dict[int, float]]:
    """
    Fetch hourly temperature forecasts from Open-Meteo historical forecast API.
    Returns {date_str: {hour_utc: temp_f}}.

    This gives what the model *predicted* for each hour at forecast time —
    not ERA5 reanalysis. Used to compute obs_vs_model_h: how the observed
    temperature deviates from what the model expected at that specific hour.
    """
    params = {
        "latitude":         lat,
        "longitude":        lon,
        "start_date":       start,
        "end_date":         end,
        "hourly":           "temperature_2m",
        "temperature_unit": "fahrenheit",
        "timezone":         "UTC",
        "models":           model,
    }
    result: dict[str, dict[int, float]] = {}
    try:
        async with session.get(
            "https://historical-forecast-api.open-meteo.com/v1/forecast",
            params=params,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
        hourly = data.get("hourly", {})
        times  = hourly.get("time", [])
        # key may be "temperature_2m" or "temperature_2m_{model}"
        temp_key = f"temperature_2m_{model}" if f"temperature_2m_{model}" in hourly else "temperature_2m"
        temps  = hourly.get(temp_key, [])
        for ts, tf in zip(times, temps):
            if tf is None:
                continue
            # ts format: "2022-01-01T14:00"
            dt_date, dt_time = ts.split("T")
            hour = int(dt_time.split(":")[0])
            result.setdefault(dt_date, {})[hour] = float(tf)
    except Exception as e:
        print(f"  [OM hourly {model}] ({lat},{lon}): {e}", file=sys.stderr)
    return result


def running_obs_at_hour(
    hourly: dict[int, float],
    entry_hour: int,
    is_high: bool,
) -> float | None:
    """Running max (is_high) or min (not is_high) up to entry_hour inclusive."""
    temps = [v for h, v in hourly.items() if h <= entry_hour]
    if not temps:
        return None
    return max(temps) if is_high else min(temps)


# ── Core row builder ──────────────────────────────────────────────────────────

def build_rows(
    metric: str,
    is_high: bool,
    entry_hour: int,
    hourly_by_date: dict[str, dict[int, float]],
    actual_by_date: dict[str, float],
    om_by_model: dict[str, dict[str, float]],   # model_name → {date: forecast}
    hrrr_by_date: dict[str, float],
) -> list[dict]:
    """
    YES-signal training rows: at entry_hour, running_obs is inside a 1°F band.
    Label: did the actual daily max/min settle in that same band?

    band_lo  = floor(running_obs)
    band_ceil = band_lo + 1
    won = 1 if band_lo <= round(actual) <= band_ceil  (YES wins)
    """
    rows = []
    all_dates = set(actual_by_date) & set(hourly_by_date)

    for d in sorted(all_dates):
        hourly = hourly_by_date[d]
        actual = actual_by_date[d]

        obs = running_obs_at_hour(hourly, entry_hour, is_high)
        if obs is None:
            continue

        # Band the observation currently sits in.
        band_lo   = int(obs)       # floor — same as int() for positive temps
        band_ceil = band_lo + 1
        margin_to_ceil  = band_ceil - obs   # headroom before overshooting
        margin_to_floor = obs - band_lo     # headroom before undershooting

        # Skip only when the obs is within 0.05°F of the ceiling — that's the
        # ambiguous case where rounding could assign it to the next band up.
        # margin_to_floor is NOT filtered: METAR reads are whole numbers so
        # margin_to_floor is always 0.0, and that's a perfectly valid entry.
        if margin_to_ceil < 0.05:
            continue

        # Collect model forecasts
        gfs_f   = om_by_model.get("gfs_seamless",  {}).get(d)
        ecmwf_f = om_by_model.get("ecmwf_ifs025", {}).get(d)
        gem_f   = om_by_model.get("gem_seamless",  {}).get(d)
        icon_f  = om_by_model.get("icon_seamless", {}).get(d)
        hrrr_f  = hrrr_by_date.get(d)

        model_vals: list[float] = [v for v in (gfs_f, ecmwf_f, gem_f, icon_f, hrrr_f)
                                   if v is not None]
        if not model_vals:
            continue

        consensus = median(model_vals)
        spread    = max(model_vals) - min(model_vals)
        n_above   = sum(1 for v in model_vals if v > band_ceil)
        n_below   = sum(1 for v in model_vals if v < band_lo)

        dt = date.fromisoformat(d)

        # YES wins if settlement (rounded to nearest 1°F) is in [band_lo, band_ceil].
        actual_rounded = round(actual)
        won = 1 if (band_lo <= actual_rounded <= band_ceil) else 0

        rows.append({
            "metric":              metric,
            "date":                d,
            "is_high":             1 if is_high else 0,
            "month":               dt.month,
            "day_of_year":         dt.timetuple().tm_yday,
            "entry_hour_utc":      entry_hour,
            "running_obs":         round(obs, 2),
            "band_lo":             band_lo,
            "band_ceil":           band_ceil,
            "margin_to_ceil":      round(margin_to_ceil, 2),
            "margin_to_floor":     round(margin_to_floor, 2),
            "actual_f":            actual,
            "gfs_f":               gfs_f   if gfs_f   is not None else "",
            "hrrr_f":              hrrr_f  if hrrr_f  is not None else "",
            "ecmwf_f":             ecmwf_f if ecmwf_f is not None else "",
            "gem_f":               gem_f   if gem_f   is not None else "",
            "icon_f":              icon_f  if icon_f  is not None else "",
            "consensus_f":         round(consensus, 2),
            "model_spread":        round(spread, 2),
            "n_models":            len(model_vals),
            "n_models_above_ceil": n_above,
            "n_models_below_floor":n_below,
            "gfs_vs_ceil":         round(gfs_f   - band_ceil, 2) if gfs_f   is not None else "",
            "hrrr_vs_ceil":        round(hrrr_f  - band_ceil, 2) if hrrr_f  is not None else "",
            "consensus_vs_ceil":   round(consensus - band_ceil, 2),
            "gfs_vs_floor":        round(gfs_f   - band_lo, 2)  if gfs_f   is not None else "",
            "hrrr_vs_floor":       round(hrrr_f  - band_lo, 2)  if hrrr_f  is not None else "",
            "consensus_vs_floor":  round(consensus - band_lo, 2),
            "won":                 won,
        })

    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(start: str, end: str, no_fetch: bool) -> None:
    Path("data/backtest").mkdir(parents=True, exist_ok=True)

    # Load cache
    cache: dict = {}
    if CACHE_PATH.exists():
        try:
            cache = json.loads(CACHE_PATH.read_text())
        except Exception:
            cache = {}

    def save_cache() -> None:
        CACHE_PATH.write_text(json.dumps(cache, indent=2))

    all_rows: list[dict] = []
    hrrr_start = max(start, "2022-01-01")

    metrics: list[tuple[str, str, float, float, bool]] = []
    for metric, city_info in CITIES.items():
        lat, lon = city_info[1], city_info[2]
        if metric in IEM_STATIONS:
            metrics.append((metric, IEM_STATIONS[metric][0], lat, lon, True))
    for metric, city_info in LOW_CITIES.items():
        lat, lon = city_info[1], city_info[2]
        if metric in IEM_STATIONS:
            metrics.append((metric, IEM_STATIONS[metric][0], lat, lon, False))

    async with aiohttp.ClientSession() as session:
        for metric, station, lat, lon, is_high in metrics:
            city_info = (CITIES if is_high else LOW_CITIES)[metric]
            tz_name   = str(city_info[3])
            high_hr, low_hr = _ENTRY_HOURS.get(tz_name, _DEFAULT_ENTRY_HOURS)
            entry_hour = high_hr if is_high else low_hr
            iem_network = IEM_STATIONS[metric][1]
            start_year  = int(start[:4])
            end_year    = int(end[:4])

            print(f"\n{'KXHIGH' if is_high else 'KXLOWT'} {metric} ({station}):")

            # ── IEM hourly ──
            hourly_key = f"hourly_{station}_{start}_{end}"
            if hourly_key in cache:
                hourly_by_date: dict[str, dict[int, float]] = {
                    d: {int(h): v for h, v in hv.items()}
                    for d, hv in cache[hourly_key].items()
                }
                print(f"  hourly: {len(hourly_by_date)} days (cached)")
            elif no_fetch:
                print(f"  hourly: skip (--no-fetch)", file=sys.stderr)
                continue
            else:
                hourly_by_date = await fetch_iem_hourly(session, station, start, end)
                cache[hourly_key] = hourly_by_date
                save_cache()
                print(f"  hourly: {len(hourly_by_date)} days fetched")
                await asyncio.sleep(3.0)  # IEM rate limit: ~20 req/min

            # ── IEM daily actual (label) ──
            actual_key = f"actual_{station}_{start}_{end}_{'high' if is_high else 'low'}"
            if actual_key in cache:
                actual_by_date: dict[str, float] = cache[actual_key]
                print(f"  actual: {len(actual_by_date)} days (cached)")
            elif no_fetch:
                print(f"  actual: skip (--no-fetch)", file=sys.stderr)
                continue
            else:
                actual_by_date = await fetch_iem_observed(
                    session, station, iem_network, start_year, end_year, is_high
                )
                cache[actual_key] = actual_by_date
                save_cache()
                print(f"  actual: {len(actual_by_date)} days fetched")

            # ── Open-Meteo historical forecasts ──
            om_key = f"om_{metric}_{start}_{end}_{'high' if is_high else 'low'}"
            if om_key in cache:
                om_by_model: dict[str, dict[str, float]] = cache[om_key]
                n_om = len(next(iter(om_by_model.values()), {}))
                print(f"  OM forecasts: {n_om} days (cached)")
            elif no_fetch:
                print(f"  OM: skip (--no-fetch)", file=sys.stderr)
                om_by_model = {}
            else:
                om_by_model = await fetch_om_historical(session, lat, lon, start, end, is_high)
                cache[om_key] = om_by_model
                save_cache()
                n_om = len(next(iter(om_by_model.values()), {}))
                print(f"  OM forecasts: {n_om} days fetched")

            # ── HRRR historical forecasts ──
            hrrr_key = f"hrrr_{metric}_{hrrr_start}_{end}_{'high' if is_high else 'low'}"
            if hrrr_key in cache:
                hrrr_by_date: dict[str, float] = cache[hrrr_key]
                print(f"  HRRR: {len(hrrr_by_date)} days (cached)")
            elif no_fetch:
                print(f"  HRRR: skip (--no-fetch)", file=sys.stderr)
                hrrr_by_date = {}
            else:
                hrrr_by_date = await fetch_hrrr_historical(
                    session, lat, lon, hrrr_start, end, is_high
                )
                cache[hrrr_key] = hrrr_by_date
                save_cache()
                print(f"  HRRR: {len(hrrr_by_date)} days fetched")

            # ── Hourly HRRR forecast (obs_vs_hrrr_h feature) ──
            hrrr_hourly_key = f"hourly_fc_hrrr_{metric}_{start}_{end}"
            if hrrr_hourly_key in cache:
                print(f"  HRRR hourly fc: cached")
            elif no_fetch:
                print(f"  HRRR hourly fc: skip (--no-fetch)", file=sys.stderr)
            else:
                hrrr_h = await fetch_om_hourly_forecast(
                    session, lat, lon, hrrr_start, end, "gfs_hrrr"
                )
                cache[hrrr_hourly_key] = hrrr_h
                save_cache()
                print(f"  HRRR hourly fc: {len(hrrr_h)} days fetched")
                await asyncio.sleep(1.0)

            # ── Hourly GFS forecast (obs_vs_gfs_h feature) ──
            gfs_hourly_key = f"hourly_fc_gfs_{metric}_{start}_{end}"
            if gfs_hourly_key in cache:
                print(f"  GFS hourly fc: cached")
            elif no_fetch:
                print(f"  GFS hourly fc: skip (--no-fetch)", file=sys.stderr)
            else:
                gfs_h = await fetch_om_hourly_forecast(
                    session, lat, lon, start, end, "gfs_seamless"
                )
                cache[gfs_hourly_key] = gfs_h
                save_cache()
                print(f"  GFS hourly fc: {len(gfs_h)} days fetched")
                await asyncio.sleep(1.0)

            # ── Build rows ──
            rows = build_rows(
                metric, is_high, entry_hour,
                hourly_by_date, actual_by_date, om_by_model, hrrr_by_date,
            )
            won  = sum(r["won"] for r in rows)
            print(f"  rows: {len(rows)}  WR={100*won/len(rows):.1f}%" if rows else "  rows: 0")
            all_rows.extend(rows)

            await asyncio.sleep(1.0)

    if not all_rows:
        print("\nNo rows generated — check cache or re-run without --no-fetch.")
        return

    # Write CSV
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(all_rows)

    total_won = sum(r["won"] for r in all_rows)
    print(f"\nExported {len(all_rows)} rows → {OUT_CSV}")
    print(f"Overall simulated WR: {100*total_won/len(all_rows):.1f}%  ({total_won} won / {len(all_rows)-total_won} lost)")
    print(f"  KXHIGH: {sum(1 for r in all_rows if r['is_high'])}")
    print(f"  KXLOWT: {sum(1 for r in all_rows if not r['is_high'])}")


if __name__ == "__main__":
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",    default="2022-01-01",
                        help="Start date (default: 2022-01-01, HRRR unavailable before this)")
    parser.add_argument("--end",      default=yesterday,
                        help=f"End date (default: {yesterday})")
    parser.add_argument("--no-fetch", action="store_true",
                        help="Use cache only; skip any API calls")
    args = parser.parse_args()
    asyncio.run(main(args.start, args.end, args.no_fetch))
