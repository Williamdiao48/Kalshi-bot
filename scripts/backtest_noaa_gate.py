"""
Backtest: does METAR observed temp at p75 time exceeding the GFS morning forecast
predict continued heating for the rest of the day?

Hypothesis for band_arb YES gate:
  When observed temp at entry time > noaa morning forecast → temp keeps climbing →
  exits band top → YES loses.

Data sources:
  - GFS daily max forecast: Open-Meteo historical forecast API (2022–2025)
    Proxy for NWS morning forecast (NWS human-adjusts GFS output).
  - IEM ASOS hourly observations (2022–2025): compute running max at p75 time.

Key metrics:
  overshoot     = asos_running_max_at_p75 - gfs_daily_max_forecast
  continued     = actual_daily_high - asos_running_max_at_p75

Question: when overshoot > 0 (temp has beaten the forecast by p75),
is continued heating different from when overshoot <= 0?

Run:
  venv/bin/python scripts/backtest_noaa_gate.py [--start 2023-05-01] [--end 2025-09-30]
"""

import argparse
import asyncio
import csv
import io
import os
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import aiohttp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.peak_hour_p90 import P75_MINUTES
from kalshi_bot.cities import CITIES
from scripts.build_forecast_calibration import IEM_STATIONS

CACHE_DIR = "data/cache/noaa_gate_backtest"
os.makedirs(CACHE_DIR, exist_ok=True)

# Only high-temp metrics
HIGH_METRICS = {k: v for k, v in CITIES.items() if k.startswith("temp_high_") and k in IEM_STATIONS}


# ── Data fetching ──────────────────────────────────────────────────────────────

async def fetch_gfs_daily_max(
    session: aiohttp.ClientSession,
    lat: float,
    lon: float,
    start: str,
    end: str,
    cache_key: str,
) -> dict[str, float]:
    """GFS predicted daily max temperature. Returns {date_str: fahrenheit}."""
    import json
    cache_path = os.path.join(CACHE_DIR, f"gfs_{cache_key}_{start}_{end}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "daily": "temperature_2m_max",
        "temperature_unit": "fahrenheit",
        "timezone": "UTC",
        "models": "gfs_seamless",
    }
    for attempt in range(3):
        try:
            async with session.get(
                "https://historical-forecast-api.open-meteo.com/v1/forecast",
                params=params, timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
            break
        except Exception as e:
            print(f"  [GFS] {cache_key} attempt {attempt+1}: {e}", file=sys.stderr)
            await asyncio.sleep(2)
    else:
        return {}

    daily = data.get("daily", {})
    dates = daily.get("time", [])
    # Single-model requests may or may not suffix the key — try both
    vals = (daily.get("temperature_2m_max_gfs_seamless")
            or daily.get("temperature_2m_max") or [])
    result = {d: v for d, v in zip(dates, vals) if v is not None}

    with open(cache_path, "w") as f:
        json.dump(result, f)
    return result


async def fetch_iem_hourly(
    session: aiohttp.ClientSession,
    station: str,
    network: str,
    start: str,
    end: str,
    cache_key: str,
) -> dict[str, list[tuple[datetime, float]]]:
    """
    IEM ASOS hourly observations. Returns {date_str: [(utc_datetime, temp_f), ...]}.
    Data is in UTC.
    """
    cache_path = os.path.join(CACHE_DIR, f"iem_{cache_key}_{start}_{end}.csv")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            raw = f.read()
    else:
        s = datetime.fromisoformat(start)
        e = datetime.fromisoformat(end)
        url = (
            f"https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
            f"?station={station}&data=tmpf"
            f"&year1={s.year}&month1={s.month}&day1={s.day}"
            f"&year2={e.year}&month2={e.month}&day2={e.day}"
            f"&tz=UTC&format=comma&latlon=no&direct=yes&report_type=3"
        )
        for attempt in range(3):
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=180)) as resp:
                    resp.raise_for_status()
                    raw = await resp.text()
                break
            except Exception as ex:
                print(f"  [IEM hourly] {station} attempt {attempt+1}: {ex}", file=sys.stderr)
                await asyncio.sleep(5)
        else:
            return {}
        with open(cache_path, "w") as f:
            f.write(raw)

    # Parse CSV: station,valid,tmpf
    result: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
    reader = csv.reader(io.StringIO(raw))
    for row in reader:
        if len(row) < 3 or row[0].startswith("#") or row[0] == "station":
            continue
        try:
            dt = datetime.strptime(row[1].strip(), "%Y-%m-%d %H:%M").replace(tzinfo=ZoneInfo("UTC"))
            temp = float(row[2].strip())
        except (ValueError, IndexError):
            continue
        date_str = dt.date().isoformat()
        result[date_str].append((dt, temp))

    return dict(result)


# ── Core analysis ──────────────────────────────────────────────────────────────

def running_max_at_time(obs: list[tuple[datetime, float]], cutoff_utc: datetime) -> float | None:
    """Max temperature seen in obs up to and including cutoff_utc."""
    vals = [t for dt, t in obs if dt <= cutoff_utc]
    return max(vals) if vals else None


def p75_utc(metric: str, d: date, tz: ZoneInfo) -> datetime:
    """Convert p75 local minutes → UTC datetime for the given date."""
    month = d.month
    minutes = P75_MINUTES.get(metric, {}).get(month)
    if minutes is None:
        return None
    local_midnight = datetime(d.year, d.month, d.day, 0, 0, tzinfo=tz)
    local_p75 = local_midnight + timedelta(minutes=minutes)
    return local_p75.astimezone(ZoneInfo("UTC"))


def analyze(records: list[dict]) -> None:
    n = len(records)
    print(f"\nTotal (metric, date) records: {n}")
    if n == 0:
        return

    # ── Main split: overshot vs lagging at p75 ──
    above = [r for r in records if r["overshoot"] > 0]
    on    = [r for r in records if r["overshoot"] == 0]
    below = [r for r in records if r["overshoot"] < 0]

    def show(group, label):
        if not group:
            print(f"  {label}: n=0")
            return
        avg_c  = sum(r["continued"] for r in group) / len(group)
        avg_o  = sum(r["overshoot"] for r in group) / len(group)
        p0     = sum(1 for r in group if r["continued"] == 0) / len(group)
        p_ge1  = sum(1 for r in group if r["continued"] >= 1) / len(group)
        p_ge2  = sum(1 for r in group if r["continued"] >= 2) / len(group)
        p_ge3  = sum(1 for r in group if r["continued"] >= 3) / len(group)
        print(f"  {label}: n={len(group):>5}  "
              f"avg_overshoot={avg_o:+.1f}°  "
              f"avg_continued={avg_c:+.2f}°  "
              f"peaked={p0:.0%}  +≥1°={p_ge1:.0%}  +≥2°={p_ge2:.0%}  +≥3°={p_ge3:.0%}")

    print("\n=== Overshoot at p75 vs continued heating after p75 ===")
    print("  (continued = actual_daily_high - asos_running_max_at_p75)\n")
    show(above, "asos_p75 > GFS forecast (overshot)")
    show(on,    "asos_p75 = GFS forecast (on track)")
    show(below, "asos_p75 < GFS forecast (lagging) ")

    # ── Overshoot magnitude buckets ──
    print("\n=== Overshoot magnitude → continued heating ===")
    buckets: dict[str, list] = defaultdict(list)
    for r in records:
        o = r["overshoot"]
        if o <= -4:   k = "≤-4°"
        elif o == -3: k = "-3°"
        elif o == -2: k = "-2°"
        elif o == -1: k = "-1°"
        elif o == 0:  k = " 0°"
        elif o == 1:  k = "+1°"
        elif o == 2:  k = "+2°"
        elif o == 3:  k = "+3°"
        else:         k = "≥+4°"
        buckets[k].append(r)

    order = ["≤-4°", "-3°", "-2°", "-1°", " 0°", "+1°", "+2°", "+3°", "≥+4°"]
    print(f"  {'overshoot':>8}  {'n':>5}  {'avg_continued':>14}  {'peaked':>7}  {'≥+1°':>6}  {'≥+2°':>6}")
    for k in order:
        grp = buckets.get(k, [])
        if not grp: continue
        avg = sum(r["continued"] for r in grp) / len(grp)
        p0  = sum(1 for r in grp if r["continued"] == 0) / len(grp)
        p1  = sum(1 for r in grp if r["continued"] >= 1) / len(grp)
        p2  = sum(1 for r in grp if r["continued"] >= 2) / len(grp)
        print(f"  {k:>8}  {len(grp):>5}  {avg:>+13.2f}°  {p0:>6.0%}  {p1:>6.0%}  {p2:>6.0%}")

    # ── Per-city summary ──
    print("\n=== Per-city: avg continued heating when overshot vs lagging ===")
    city_records: dict[str, list] = defaultdict(list)
    for r in records:
        city_records[r["metric"]].append(r)
    print(f"  {'metric':<20}  {'n_over':>6}  {'avg_cont_over':>13}  {'n_lag':>5}  {'avg_cont_lag':>12}")
    for metric in sorted(city_records):
        grp = city_records[metric]
        ov  = [r for r in grp if r["overshoot"] > 0]
        lg  = [r for r in grp if r["overshoot"] < 0]
        avg_ov = (sum(r["continued"] for r in ov) / len(ov)) if ov else float("nan")
        avg_lg = (sum(r["continued"] for r in lg) / len(lg)) if lg else float("nan")
        print(f"  {metric:<20}  {len(ov):>6}  {avg_ov:>+12.2f}°  {len(lg):>5}  {avg_lg:>+11.2f}°")


# ── Main ───────────────────────────────────────────────────────────────────────

async def main(start: str, end: str) -> None:
    print(f"Fetching data for {len(HIGH_METRICS)} cities, {start} → {end}")
    print("(cached fetches skip download)\n")

    records: list[dict] = []

    async with aiohttp.ClientSession() as session:
        for metric, (city_name, lat, lon, tz) in HIGH_METRICS.items():
            station, network = IEM_STATIONS[metric]
            short = metric.replace("temp_high_", "")
            print(f"  {short:<6} ({station}) ...", end="", flush=True)

            # Sequential fetches with delay to avoid IEM rate limits
            gfs_data = await fetch_gfs_daily_max(session, lat, lon, start, end, short)
            await asyncio.sleep(1.0)
            iem_data = await fetch_iem_hourly(session, station, network, start, end, short)
            await asyncio.sleep(1.5)

            n_added = 0
            start_d = date.fromisoformat(start)
            end_d   = date.fromisoformat(end)
            d = start_d
            while d <= end_d:
                ds = d.isoformat()
                gfs_fcst = gfs_data.get(ds)
                obs = iem_data.get(ds)
                if gfs_fcst is None or not obs:
                    d += timedelta(days=1)
                    continue

                p75_dt = p75_utc(metric, d, tz)
                if p75_dt is None:
                    d += timedelta(days=1)
                    continue

                asos_p75 = running_max_at_time(obs, p75_dt)
                actual_hi = max(t for _, t in obs)

                if asos_p75 is None:
                    d += timedelta(days=1)
                    continue

                records.append({
                    "metric":    metric,
                    "date":      ds,
                    "gfs_fcst":  gfs_fcst,
                    "asos_p75":  asos_p75,
                    "actual_hi": actual_hi,
                    "overshoot": round(asos_p75 - gfs_fcst),   # integer degrees
                    "continued": round(actual_hi - asos_p75),  # integer degrees
                })
                n_added += 1
                d += timedelta(days=1)

            print(f" {n_added} days")
            await asyncio.sleep(0.3)

    analyze(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2022-05-01")
    parser.add_argument("--end",   default="2025-09-30")
    args = parser.parse_args()
    asyncio.run(main(args.start, args.end))
