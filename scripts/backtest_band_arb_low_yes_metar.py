"""
Band-arb LOW YES backtest using METAR T-group observations (0.1°C precision),
GFS daily-min forecasts, P75 trough anchors, and Kalshi overnight candlestick prices.

Strategy: At the P75 trough time, if round(running_min) maps to a KXLOWT band
[floor, ceil], buy YES — betting the final daily low stays in that band.
Win condition: final_daily_low >= floor - 0.5°F (rounding-aware).
Since the running min is monotonically non-increasing, the ONLY way to lose
is further cooling past the floor. This is fundamentally one-sided risk.

GFS dimension: cold_gap = gfs_daily_min - running_min
  Positive = GFS still expects more cooling (warmer than forecast at entry)
  Negative = already colder than GFS (trough may already be past GFS floor)
  Mirrors HIGH YES overshoot: cold_gap > 0 is the "lagging" zone, < 0 is "overshot".

Run:
  venv/bin/python scripts/backtest_band_arb_low_yes_metar.py
  venv/bin/python scripts/backtest_band_arb_low_yes_metar.py --refresh
  venv/bin/python scripts/backtest_band_arb_low_yes_metar.py --fetch-gfs
  venv/bin/python scripts/backtest_band_arb_low_yes_metar.py --fetch-candles
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import re
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import aiohttp

sys.path.insert(0, str(Path(__file__).parent.parent))

from kalshi_bot.auth import generate_headers
from kalshi_bot.cities import LOW_CITIES
from kalshi_bot.markets import KALSHI_API_BASE
from scripts.build_forecast_calibration import IEM_STATIONS

DATA_DIR   = Path(__file__).parent.parent / "data"
CACHE_DIR  = DATA_DIR / "cache" / "metar_low_historical"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
BANDS_CSV  = DATA_DIR / "kxlowt_bands.csv"
GFS_CACHE  = DATA_DIR / "cache" / "noaa_gate_low_backtest"
TROUGH_CSV = DATA_DIR / "overnight_low_analysis.csv"
CANDLE_JSON = DATA_DIR / "kxlowt_candle_cache.json"

BAND_START = "2026-03-15"
BAND_END   = "2026-05-21"

_UTC = ZoneInfo("UTC")

LOW_IEM = {k: v for k, v in IEM_STATIONS.items() if k.startswith("temp_low_")}

ENTRY_KEYS    = ["p75m2", "p75m1", "p75", "p75p1", "p75p2"]
ENTRY_OFFSETS = {"p75m2": -2, "p75m1": -1, "p75": 0, "p75p1": 1, "p75p2": 2}
ENTRY_LABELS  = {"p75m2": "P75-2h", "p75m1": "P75-1h", "p75": "P75",
                 "p75p1": "P75+1h", "p75p2": "P75+2h"}

_CANDLE_SEM = asyncio.Semaphore(4)

_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_parser.add_argument("--refresh",       action="store_true", help="Re-fetch METAR (ignores cache)")
_parser.add_argument("--fetch-gfs",     action="store_true", help="Fetch missing GFS daily-min from Open-Meteo")
_parser.add_argument("--fetch-candles", action="store_true", help="Fetch Kalshi overnight candlesticks for KXLOWT tickers")
_args = _parser.parse_args()


# ── Trough CSV ────────────────────────────────────────────────────────────────

def load_trough_minutes() -> dict[tuple[str, int], int]:
    result: dict[tuple[str, int], int] = {}
    with TROUGH_CSV.open() as f:
        for row in csv.DictReader(f):
            metric = row["city_key"]
            month  = int(row["month"])
            raw = row.get("p75", "").strip()
            if raw:
                hh, mm = raw.split(":")
                result[(metric, month)] = int(hh) * 60 + int(mm)
    return result


# ── METAR parsing ─────────────────────────────────────────────────────────────

_T_GROUP_RE   = re.compile(r'\bT([01])(\d{3})([01])(\d{3})\b')
_BODY_TEMP_RE = re.compile(r'(?<!\w)(M?\d{2})/(M?\d{2})(?!\w)')


def _parse_metar_temp_f(metar_str: str) -> float | None:
    m = _T_GROUP_RE.search(metar_str)
    if m:
        sign_t = -1 if m.group(1) == '1' else 1
        temp_c = sign_t * int(m.group(2)) / 10.0
        return round(temp_c * 1.8 + 32.0, 4)
    bm = _BODY_TEMP_RE.search(metar_str)
    if bm:
        raw = bm.group(1).replace('M', '-')
        try:
            return float(raw) * 1.8 + 32.0
        except ValueError:
            pass
    return None


# ── IEM METAR fetch ───────────────────────────────────────────────────────────

async def _fetch_metar(
    session: aiohttp.ClientSession,
    station: str,
    cache_key: str,
) -> dict[str, list[tuple[datetime, float]]]:
    cache_path = CACHE_DIR / f"metar_{cache_key}_{BAND_START}_{BAND_END}.csv"
    if cache_path.exists() and not _args.refresh:
        raw = cache_path.read_text()
    else:
        s = datetime.fromisoformat(BAND_START)
        e = datetime.fromisoformat(BAND_END) + timedelta(days=2)
        url = (
            "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
            f"?station={station}&data=metar"
            f"&year1={s.year}&month1={s.month}&day1={s.day}"
            f"&year2={e.year}&month2={e.month}&day2={e.day}"
            "&tz=UTC&format=comma&latlon=no&direct=yes&report_type=3"
        )
        for attempt in range(4):
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=300)) as resp:
                    resp.raise_for_status()
                    raw = await resp.text()
                break
            except Exception as ex:
                wait = 2 ** attempt
                print(f"  [IEM] {station} attempt {attempt+1}: {ex}", file=sys.stderr)
                await asyncio.sleep(wait)
        else:
            return {}
        cache_path.write_text(raw)

    result: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
    t_count = other_count = 0
    for row in csv.reader(io.StringIO(raw)):
        if len(row) < 3 or row[0].startswith('#') or row[0] == 'station':
            continue
        try:
            dt = datetime.strptime(row[1].strip(), "%Y-%m-%d %H:%M").replace(tzinfo=_UTC)
        except ValueError:
            continue
        metar_str = row[2].strip()
        has_tgroup = bool(_T_GROUP_RE.search(metar_str))
        t_count    += has_tgroup
        other_count += not has_tgroup
        temp_f = _parse_metar_temp_f(metar_str)
        if temp_f is not None:
            result[dt.date().isoformat()].append((dt, temp_f))

    for d in result:
        result[d].sort(key=lambda x: x[0])

    total = t_count + other_count
    print(f"  T-group: {t_count}/{total} ({100*t_count/total:.0f}%)" if total else "  no obs")
    return dict(result)


# ── GFS ───────────────────────────────────────────────────────────────────────

def _load_gfs_cache(city_short: str) -> dict[str, float]:
    merged: dict[str, float] = {}
    for f in sorted(GFS_CACHE.glob(f"gfs_{city_short}_20*.json")):
        try:
            merged.update(json.loads(f.read_text()))
        except Exception:
            continue
    return merged


async def _fetch_gfs_range(
    session: aiohttp.ClientSession,
    city_short: str, lat: float, lon: float,
    start: str, end: str,
) -> dict[str, float]:
    cache_path = GFS_CACHE / f"gfs_{city_short}_{start}_{end}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "daily": "temperature_2m_min",
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
            print(f"  [GFS] {city_short} attempt {attempt+1}: {e}", file=sys.stderr)
            await asyncio.sleep(2 ** attempt)
    else:
        return {}
    daily = data.get("daily", {})
    dates = daily.get("time", [])
    vals  = daily.get("temperature_2m_min_gfs_seamless") or daily.get("temperature_2m_min") or []
    result = {d: v for d, v in zip(dates, vals) if v is not None}
    cache_path.write_text(json.dumps(result))
    return result


# ── Kalshi overnight candlesticks ─────────────────────────────────────────────

async def _fetch_candles_for_ticker(
    session: aiohttp.ClientSession,
    ticker: str,
    local_date_str: str,
) -> list[dict]:
    """Fetch hourly KXLOWT candlesticks covering the overnight window.

    Window: 18:00 UTC on prior calendar day → 15:00 UTC on local_date.
    This covers ~6 PM to ~9 AM local for all US timezones.
    """
    local_d  = datetime.strptime(local_date_str, "%Y-%m-%d").date()
    start_ts = int(datetime(local_d.year, local_d.month, local_d.day,
                            0, 0, tzinfo=timezone.utc).timestamp()) - 6 * 3600  # prior 6PM UTC
    end_ts   = int(datetime(local_d.year, local_d.month, local_d.day,
                            15, 0, tzinfo=timezone.utc).timestamp())
    series   = ticker.rsplit("-", 2)[0]
    path     = f"/trade-api/v2/series/{series}/markets/{ticker}/candlesticks"
    headers  = generate_headers("GET", path)
    params   = {"period_interval": 60, "start_ts": start_ts, "end_ts": end_ts}

    async with _CANDLE_SEM:
        try:
            async with session.get(
                f"{KALSHI_API_BASE}/series/{series}/markets/{ticker}/candlesticks",
                params=params, headers=headers,
                timeout=aiohttp.ClientTimeout(total=20),
            ) as r:
                if r.status == 429:
                    await asyncio.sleep(3.0)
                    return []
                if r.status != 200:
                    return []
                data = await r.json()
        except Exception:
            return []
        await asyncio.sleep(0.15)
    return data.get("candlesticks", [])


def _candles_to_hourly_ask(candles: list[dict], city_tz: ZoneInfo) -> dict[int, float]:
    """Convert candlesticks → {local_hour: yes_ask_cents} using hourly close."""
    result: dict[int, float] = {}
    for c in candles:
        close_str = (c.get("yes_ask") or {}).get("close_dollars")
        if close_str is None:
            continue
        try:
            ask_cents = round(float(close_str) * 100)
        except (ValueError, TypeError):
            continue
        ts = datetime.fromtimestamp(c["end_period_ts"], tz=timezone.utc)
        local_hour = ts.astimezone(city_tz).hour
        result[local_hour] = ask_cents
    return result


def _ask_at_hour(hourly: dict[int, float], hour: int) -> float | None:
    for delta in [0, 1, -1, 2, -2]:
        v = hourly.get(hour + delta)
        if v is not None:
            return v
    return None


def load_candle_cache() -> dict[str, list[dict]]:
    if CANDLE_JSON.exists():
        return json.loads(CANDLE_JSON.read_text())
    return {}


# ── Band loading ───────────────────────────────────────────────────────────────

def load_bands() -> dict[tuple[str, str], list[dict]]:
    if not BANDS_CSV.exists():
        print(f"\nERROR: {BANDS_CSV} not found. Run: venv/bin/python scripts/build_kxlowt_bands.py")
        sys.exit(1)
    index: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with BANDS_CSV.open() as f:
        for row in csv.DictReader(f):
            if row.get("direction") != "between":
                continue
            row["strike_lo"] = float(row["strike_lo"])
            row["strike_hi"] = float(row["strike_hi"])
            utc_close  = date.fromisoformat(row["date"])
            local_date = (utc_close - timedelta(days=1)).isoformat()
            index[(row["metric"], local_date)].append(row)
    return dict(index)


# ── Core helpers ───────────────────────────────────────────────────────────────

def _lst(tz: ZoneInfo) -> timezone:
    return timezone(tz.utcoffset(datetime(2000, 1, 15)))


def _running_min_at(obs: list[tuple[datetime, float]], cutoff: datetime) -> float | None:
    vals = [t for dt, t in obs if dt <= cutoff]
    return min(vals) if vals else None


def _spot_temp_at(obs: list[tuple[datetime, float]], cutoff: datetime) -> float | None:
    recent = [t for dt, t in obs if dt <= cutoff]
    return recent[-1] if recent else None


def _final_daily_low(obs: list[tuple[datetime, float]], local_d: date, tz: ZoneInfo) -> float | None:
    lst = _lst(tz)
    day_start = datetime(local_d.year, local_d.month, local_d.day, 0, 0, tzinfo=lst).astimezone(_UTC)
    day_end   = (datetime(local_d.year, local_d.month, local_d.day, 0, 0, tzinfo=lst) + timedelta(days=1)).astimezone(_UTC)
    vals = [t for dt, t in obs if day_start <= dt < day_end]
    return min(vals) if vals else None


def _p75_utc(metric: str, d: date, tz: ZoneInfo, trough_mins: dict[tuple[str, int], int]) -> datetime | None:
    minutes = trough_mins.get((metric, d.month))
    if minutes is None:
        return None
    local_midnight = datetime(d.year, d.month, d.day, 0, 0, tzinfo=tz)
    return (local_midnight + timedelta(minutes=minutes)).astimezone(_UTC)


def _stats(group: list[dict]) -> dict:
    n = len(group)
    if n == 0:
        return {"n": 0, "wins": 0, "avg_cool": 0.0}
    wins     = sum(1 for r in group if r["yes_win"])
    cooled   = [r["continued_cooling"] for r in group if r["continued_cooling"] is not None]
    avg_cool = sum(cooled) / len(cooled) if cooled else 0.0
    return {"n": n, "wins": wins, "avg_cool": avg_cool}



# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    print(f"Loading trough times from {TROUGH_CSV.name}…")
    trough_mins = load_trough_minutes()
    print(f"  {len(trough_mins)} city-month entries")

    print(f"\nLoading bands from {BANDS_CSV.name}…")
    bands_index = load_bands()
    bands_flat: dict[tuple[str, str, int], dict] = {}
    for (metric, local_date), band_list in bands_index.items():
        for band in band_list:
            bands_flat[(metric, local_date, int(band["strike_lo"]))] = band
    print(f"  {len(bands_flat):,} band rows")

    # ── Fetch METAR ──────────────────────────────────────────────────────────
    print(f"\nFetching METAR ({BAND_START} → {BAND_END})…")
    metar_by_metric: dict[str, dict[str, list[tuple[datetime, float]]]] = {}
    async with aiohttp.ClientSession() as session:
        for metric, (station, _) in sorted(LOW_IEM.items()):
            short = metric.replace("temp_low_", "")
            print(f"  {short:<6} ({station}) …", end=" ", flush=True)
            metar_by_metric[metric] = await _fetch_metar(session, station, short)
            await asyncio.sleep(0.3)

    # ── GFS ──────────────────────────────────────────────────────────────────
    print("\nLoading GFS daily-min cache…")
    gfs_by_metric: dict[str, dict[str, float]] = {}
    async with aiohttp.ClientSession() as session:
        for metric, city_info in sorted(LOW_CITIES.items()):
            short = metric.replace("temp_low_", "")
            cache = _load_gfs_cache(short)
            gfs_by_metric[metric] = cache
            if not _args.fetch_gfs:
                print(f"  {short:<6}  {len(cache):>5} cached dates")
                continue
            needed  = {ld for (m, ld) in bands_index if m == metric}
            missing = sorted(d for d in needed if d not in cache)
            if not missing:
                print(f"  {short:<6}  {len(cache):>5} cached  (no gaps)")
                continue
            _, lat, lon, _ = city_info
            new = await _fetch_gfs_range(session, short, lat, lon, missing[0], missing[-1])
            cache.update(new)
            print(f"  {short:<6}  +{len(new)} fetched")
            await asyncio.sleep(0.5)

    # ── Build initial records (no candle prices yet) ──────────────────────────
    print("\nBuilding records…")
    records: list[dict] = []

    for metric, daily_obs in sorted(metar_by_metric.items()):
        city_info = LOW_CITIES.get(metric)
        if city_info is None:
            continue
        _, _, _, tz = city_info
        gfs_cache = gfs_by_metric.get(metric, {})

        for local_date in sorted(daily_obs):
            try:
                local_d = date.fromisoformat(local_date)
            except ValueError:
                continue

            next_utc   = (local_d + timedelta(days=1)).isoformat()
            merged_obs = sorted(
                daily_obs.get(local_date, []) + daily_obs.get(next_utc, []),
                key=lambda x: x[0],
            )
            if not merged_obs:
                continue

            p75_dt = _p75_utc(metric, local_d, tz, trough_mins)
            if p75_dt is None:
                continue

            gfs_daily_min = gfs_cache.get(local_date)
            final_low     = _final_daily_low(merged_obs, local_d, tz)
            if final_low is None:
                continue

            for key in ENTRY_KEYS:
                offset_h = ENTRY_OFFSETS[key]
                cutoff   = p75_dt + timedelta(hours=offset_h)
                rmin     = _running_min_at(merged_obs, cutoff)
                if rmin is None:
                    continue

                rounded = round(rmin)
                band    = bands_flat.get((metric, local_date, rounded))
                if band is None:
                    continue

                floor = band["strike_lo"]
                ceil  = band["strike_hi"]

                yes_win           = final_low >= (floor - 0.5)
                continued_cooling = rmin - final_low
                margin_floor      = rmin - floor
                # cold_gap mirrors HIGH YES overshoot: positive = GFS colder than current (riskier)
                cold_gap = (gfs_daily_min - rmin) if gfs_daily_min is not None else None

                local_hour    = cutoff.astimezone(tz).hour
                rmin_1h_ago   = _running_min_at(merged_obs, cutoff - timedelta(hours=1))
                spot_now      = _spot_temp_at(merged_obs, cutoff)
                spot_1h_ago   = _spot_temp_at(merged_obs, cutoff - timedelta(hours=1))
                still_cooling = rmin_1h_ago is not None and rmin < rmin_1h_ago
                temp_delta_1h = (
                    (spot_now - spot_1h_ago)
                    if spot_now is not None and spot_1h_ago is not None else None
                )

                records.append({
                    "metric":            metric,
                    "date":              local_date,
                    "ticker":            band["ticker"],
                    "entry_key":         key,
                    "local_hour":        local_hour,
                    "running_min":       rmin,
                    "band_floor":        floor,
                    "band_ceil":         ceil,
                    "margin_floor":      margin_floor,
                    "final_daily_low":   final_low,
                    "continued_cooling": continued_cooling,
                    "gfs_daily_min":     gfs_daily_min,
                    "cold_gap":          cold_gap,   # gfs_min - rmin; + = GFS colder; analog of HIGH YES overshoot
                    "yes_win":           yes_win,
                    "still_cooling":     still_cooling,
                    "temp_delta_1h":     temp_delta_1h,
                    "band_result":       band["result"],
                    "cents":             None,        # filled after candle fetch
                    "hourly_asks":       {},
                })

    total = len(records)
    print(f"Total records: {total:,}")
    if total == 0:
        print("No records — check kxlowt_bands.csv and METAR data.")
        return
    for key in ENTRY_KEYS:
        print(f"  {ENTRY_LABELS[key]}: {sum(1 for r in records if r['entry_key'] == key):>5}")

    # ── Fetch / load candles ──────────────────────────────────────────────────
    candle_cache = load_candle_cache()
    p75_recs_all = [r for r in records if r["entry_key"] == "p75"]
    tickers_needed = {(r["ticker"], r["date"]) for r in p75_recs_all}
    tickers_missing = {(t, d) for t, d in tickers_needed if t not in candle_cache}

    if _args.fetch_candles and tickers_missing:
        print(f"\nFetching candles for {len(tickers_missing)} KXLOWT tickers…")
        async with aiohttp.ClientSession() as session:
            tasks = {
                (t, d): asyncio.create_task(_fetch_candles_for_ticker(session, t, d))
                for t, d in tickers_missing
            }
            for i, ((t, _), task) in enumerate(tasks.items()):
                candles = await task
                if candles:
                    candle_cache[t] = candles
                if (i + 1) % 20 == 0:
                    print(f"  … {i+1}/{len(tasks)}")
        CANDLE_JSON.write_text(json.dumps(candle_cache))
        print(f"  Saved {len(candle_cache)} tickers to {CANDLE_JSON.name}")
    elif tickers_missing:
        print(f"\n  {len(tickers_missing)} tickers without candle data (run --fetch-candles to populate)")

    # Attach candle prices to all records
    for r in records:
        raw_candles = candle_cache.get(r["ticker"], [])
        city_info   = LOW_CITIES.get(r["metric"])
        if raw_candles and city_info:
            _, _, _, tz = city_info
            hourly = _candles_to_hourly_ask(raw_candles, tz)
            r["hourly_asks"] = hourly
            r["cents"]       = _ask_at_hour(hourly, r["local_hour"])

    # ── Analysis sections ─────────────────────────────────────────────────────
    p75_recs = [r for r in records if r["entry_key"] == "p75"]
    n_priced = sum(1 for r in p75_recs if r["cents"] is not None)

    # ── Section 1: Win rate by entry timing ───────────────────────────────────
    print()
    print("=" * 70)
    print("  1. WIN RATE BY ENTRY TIMING")
    print("     yes_win = final_daily_low >= band_floor − 0.5°F (rounding-aware)")
    print("     cold_gap = gfs_daily_min − running_min  (+ = GFS colder = more risk)")
    print("=" * 70)
    print(f"\n  {'Entry':>8}  {'n':>5} {'WR':>6} {'avg_cool':>8}  {'cold_gap>0 (riskier)':>22}  {'cold_gap<0 (safer)':>20}")
    print("  " + "-" * 72)

    for key in ENTRY_KEYS:
        grp    = [r for r in records if r["entry_key"] == key]
        grp_risk = [r for r in grp if r["cold_gap"] is not None and r["cold_gap"] > 0]
        grp_safe = [r for r in grp if r["cold_gap"] is not None and r["cold_gap"] <= 0]
        s = _stats(grp); sr = _stats(grp_risk); ss = _stats(grp_safe)
        wr   = f"{100*s['wins']/s['n']:.1f}%" if s['n'] else "—"
        wr_r = f"{100*sr['wins']/sr['n']:.1f}%" if sr['n'] else "—"
        wr_s = f"{100*ss['wins']/ss['n']:.1f}%" if ss['n'] else "—"
        print(f"  {ENTRY_LABELS[key]:>8}  {s['n']:>5} {wr:>6} {s['avg_cool']:>+7.2f}°F  "
              f"n={sr['n']:>4} WR={wr_r:>6}  n={ss['n']:>4} WR={wr_s:>6}")

    # ── Section 2: Continued cooling distribution at P75 ──────────────────────
    print()
    print("=" * 70)
    print("  2. CONTINUED COOLING DISTRIBUTION AT P75")
    print("=" * 70)

    cooled = [r["continued_cooling"] for r in p75_recs if r["continued_cooling"] is not None]
    peaked_pct = sum(1 for c in cooled if c <= 0) / len(cooled) * 100 if cooled else 0
    print(f"\n  n={len(p75_recs)}  peaked_pct={peaked_pct:.1f}%  avg_cool={sum(max(c,0) for c in cooled)/len(cooled):.2f}°F")

    cool_buckets = [
        ("0°F (peaked)",  0.0,  0.01),
        ("0–0.5°F",       0.01,  0.5),
        ("0.5–1°F",       0.5,   1.0),
        ("1–1.5°F",       1.0,   1.5),
        ("1.5–2°F",       1.5,   2.0),
        ("2–3°F",         2.0,   3.0),
        ("> 3°F",         3.0,  99.0),
    ]
    print(f"\n  {'Bucket':>14}  {'n':>5} {'%':>6}")
    print("  " + "-" * 28)
    for lbl, lo, hi in cool_buckets:
        n = sum(1 for c in cooled if lo <= c < hi)
        print(f"  {lbl:>14}  {n:>5} {100*n/len(cooled):>5.1f}%")

    print(f"\n  Split by cold_gap sign:")
    print(f"  {'Split':>22}  {'n':>5} {'peaked%':>8} {'avg_cool':>9}")
    print("  " + "-" * 50)
    for lbl, filt in [
        ("cold_gap > 0 (riskier)",  [r for r in p75_recs if r.get("cold_gap") is not None and r["cold_gap"] > 0]),
        ("cold_gap ≤ 0 (safer)",    [r for r in p75_recs if r.get("cold_gap") is not None and r["cold_gap"] <= 0]),
    ]:
        c = [r["continued_cooling"] for r in filt if r["continued_cooling"] is not None]
        if not c:
            print(f"  {lbl:>22}  {'—':>5}")
            continue
        pk  = sum(1 for x in c if x <= 0) / len(c) * 100
        avg = sum(max(x, 0) for x in c) / len(c)
        print(f"  {lbl:>22}  {len(filt):>5} {pk:>7.1f}%  {avg:>+8.2f}°F")

    # ── Section 3: Win rate by cold_gap bucket (mirrors HIGH YES overshoot) ───
    print()
    print("=" * 70)
    print("  3. WIN RATE BY COLD_GAP AT P75  (analog of HIGH YES overshoot)")
    print("     cold_gap = gfs_daily_min − running_min")
    print("     Positive = GFS still expects colder (more risk to floor)")
    print("     Negative = already past GFS minimum (trough may be set)")
    print("=" * 70)

    gap_buckets = [
        ("< −2°F",        -99, -2.0),
        ("−2 to −1°F",   -2.0, -1.0),
        ("−1 to −0.5°F", -1.0, -0.5),
        ("−0.5 to 0°F",  -0.5,  0.0),
        ("0 to +0.5°F",   0.0,  0.5),
        ("+0.5 to +1°F",  0.5,  1.0),
        ("+1 to +1.5°F",  1.0,  1.5),
        ("+1.5 to +2°F",  1.5,  2.0),
        ("> +2°F",        2.0, 99.0),
    ]
    p75_gfs = [r for r in p75_recs if r.get("cold_gap") is not None]
    print(f"\n  n with GFS: {len(p75_gfs)} / {len(p75_recs)}")
    print(f"\n  {'Bucket':>15}  {'n':>5} {'WR':>6} {'avg_cool':>9}")
    print("  " + "-" * 42)
    for lbl, lo, hi in gap_buckets:
        grp = [r for r in p75_gfs if lo <= r["cold_gap"] < hi]
        s   = _stats(grp)
        wr  = f"{100*s['wins']/s['n']:.1f}%" if s['n'] else "—"
        print(f"  {lbl:>15}  {s['n']:>5} {wr:>6} {s['avg_cool']:>+8.2f}°F")

    # ── Section 4: Win rate by trend ──────────────────────────────────────────
    print()
    print("=" * 70)
    print("  4. WIN RATE BY TEMPERATURE TREND AT P75")
    print("     temp_delta_1h = spot_now − spot_1h_ago")
    print("=" * 70)

    p75_trend = [r for r in p75_recs if r["temp_delta_1h"] is not None]
    delta_buckets = [
        ("< −1°F",    -99.0, -1.0),
        ("−1 to 0°F",  -1.0, -0.01),
        ("flat (0°F)", -0.01,  0.01),
        ("0 to +1°F",   0.01,  1.0),
        ("> +1°F",      1.0,  99.0),
    ]
    print(f"\n  n with delta: {len(p75_trend)} / {len(p75_recs)}")
    print(f"\n  {'Trend':>12}  {'n':>5} {'WR':>6} {'avg_cool':>9}")
    print("  " + "-" * 38)
    for lbl, lo, hi in delta_buckets:
        grp = [r for r in p75_trend if lo <= r["temp_delta_1h"] < hi]
        s   = _stats(grp)
        wr  = f"{100*s['wins']/s['n']:.1f}%" if s['n'] else "—"
        print(f"  {lbl:>12}  {s['n']:>5} {wr:>6} {s['avg_cool']:>+8.2f}°F")

    print(f"\n  still_cooling (rmin < rmin_1h_ago):")
    for lbl, filt in [
        ("still cooling",     [r for r in p75_recs if r["still_cooling"]]),
        ("stabilized/rising", [r for r in p75_recs if not r["still_cooling"]]),
    ]:
        s  = _stats(filt)
        wr = f"{100*s['wins']/s['n']:.1f}%" if s['n'] else "—"
        print(f"    {lbl:<20}  n={s['n']:>4} WR={wr:>6} avg_cool={s['avg_cool']:>+.2f}°F")

    # ── Section 5: Entry price breakdown (if candle data available) ───────────
    print()
    print("=" * 70)
    print("  5. WIN RATE BY YES ASK PRICE AT P75  (priced trades only)")
    print("=" * 70)
    print(f"\n  Priced: {n_priced} / {len(p75_recs)}")

    if n_priced > 0:
        price_buckets = [
            ("< 30¢",   0,  30),
            ("30–40¢",  30, 40),
            ("40–50¢",  40, 50),
            ("50–60¢",  50, 60),
            ("60–70¢",  60, 70),
            ("70–80¢",  70, 80),
            ("80–90¢",  80, 90),
            ("> 90¢",   90, 101),
        ]
        p75_priced = [r for r in p75_recs if r["cents"] is not None]
        print(f"\n  {'Price':>8}  {'n':>5} {'WR':>6} {'avg_cool':>9}  {'entry avg':>9}")
        print("  " + "-" * 46)
        for lbl, lo_p, hi_p in price_buckets:
            grp = [r for r in p75_priced if lo_p <= r["cents"] < hi_p]
            s   = _stats(grp)
            avg_e = sum(r["cents"] for r in grp) / len(grp) if grp else 0.0
            wr    = f"{100*s['wins']/s['n']:.1f}%" if s['n'] else "—"
            print(f"  {lbl:>8}  {s['n']:>5} {wr:>6} {s['avg_cool']:>+8.2f}°F  {avg_e:>8.1f}¢")
    else:
        print("\n  No candle data — run with --fetch-candles to populate.")

    # ── Section 6: Per-city breakdown ─────────────────────────────────────────
    print()
    print("=" * 70)
    print("  6. WIN RATE BY CITY AT P75")
    print("=" * 70)
    print(f"\n  {'City':>6}  {'n':>5} {'WR':>6} {'avg_cool':>9}  {'cold_gap>0':>12}  {'cold_gap≤0':>12}")
    print("  " + "-" * 60)
    for city in sorted(set(r["metric"].replace("temp_low_", "") for r in p75_recs)):
        grp    = [r for r in p75_recs if r["metric"] == f"temp_low_{city}"]
        grp_r  = [r for r in grp if r.get("cold_gap") is not None and r["cold_gap"] > 0]
        grp_s  = [r for r in grp if r.get("cold_gap") is not None and r["cold_gap"] <= 0]
        s = _stats(grp); sr = _stats(grp_r); ss = _stats(grp_s)
        wr   = f"{100*s['wins']/s['n']:.1f}%" if s['n'] else "—"
        wr_r = f"{100*sr['wins']/sr['n']:.1f}%" if sr['n'] else "—"
        wr_s = f"{100*ss['wins']/ss['n']:.1f}%" if ss['n'] else "—"
        print(f"  {city:>6}  {s['n']:>5} {wr:>6} {s['avg_cool']:>+8.2f}°F  "
              f"n={sr['n']:>3} {wr_r:>6}  n={ss['n']:>3} {wr_s:>6}")

    # ── Section 7: Combined factor grid ───────────────────────────────────────
    print()
    print("=" * 70)
    print("  7. COMBINED FACTOR GRID AT P75")
    print("=" * 70)

    def _safe(r):  return r.get("cold_gap") is not None and r["cold_gap"] <= 0
    def _risky(r): return r.get("cold_gap") is not None and r["cold_gap"] > 0
    def _stab(r):  return not r["still_cooling"]
    def _cool(r):  return r["still_cooling"]
    def _gap(r, lo, hi): return r.get("cold_gap") is not None and lo <= r["cold_gap"] < hi

    combos = [
        ("ALL baseline",                       lambda _: True),
        ("cold_gap ≤ 0 (already past GFS)",   _safe),
        ("cold_gap > 0 (GFS still colder)",   _risky),
        ("cold_gap > 0.5°F",                  lambda r: _gap(r, 0.5, 99)),
        ("cold_gap > 1°F",                    lambda r: _gap(r, 1.0, 99)),
        ("cold_gap > 1.5°F (skip zone?)",     lambda r: _gap(r, 1.5, 99)),
        ("cold_gap > 0 + stabilized",         lambda r: _risky(r) and _stab(r)),
        ("cold_gap > 0 + still cooling",      lambda r: _risky(r) and _cool(r)),
        ("cold_gap ≤ 0 + stabilized",         lambda r: _safe(r) and _stab(r)),
        ("stabilized",                        _stab),
        ("still cooling",                     _cool),
    ]

    print(f"\n  {'Filter':>38}  {'n':>5} {'WR':>6} {'avg_cool':>9}")
    print("  " + "-" * 62)
    for lbl, gate in combos:
        grp = [r for r in p75_recs if gate(r)]
        s   = _stats(grp)
        wr  = f"{100*s['wins']/s['n']:.1f}%" if s['n'] else "—"
        print(f"  {lbl:>38}  {s['n']:>5} {wr:>6} {s['avg_cool']:>+8.2f}°F")

    # ── Section 8: cold_gap sizer cross-tab (implied sizer table) ─────────────
    print()
    print("=" * 70)
    print("  8. COLD_GAP SIZER CROSS-TAB  (analog of HIGH YES overshoot × trend)")
    print("     Rows = cold_gap bucket; Cols = trend state")
    print("     cold_gap > 1.5°F may be a skip zone (mirrors HIGH YES skip < −1.5°F)")
    print("=" * 70)

    gap_sizer = [
        ("skip? (>+1.5°F)",  1.5, 99.0),
        ("risky (+0.5–1.5°F)", 0.5, 1.5),
        ("near  (0–+0.5°F)",  0.0,  0.5),
        ("safe  (≤ 0°F)",   -99.0,  0.0),
    ]

    print(f"\n  {'Bucket':>20}  {'── STABILIZED ───────────────':>30}  {'── STILL COOLING ──────────':>28}  {'── ALL ──────':>14}")
    print(f"  {'':>20}  {'n':>5} {'WR':>6} {'Laplace':>8}  {'n':>5} {'WR':>6} {'Laplace':>8}  {'n':>5} {'WR':>6}")
    print("  " + "-" * 100)

    for lbl, lo, hi in gap_sizer:
        grp  = [r for r in p75_gfs if lo <= r["cold_gap"] < hi]
        g_st = [r for r in grp if _stab(r)]
        g_co = [r for r in grp if _cool(r)]

        def _cell(g):
            n = len(g); w = sum(1 for r in g if r["yes_win"])
            wr = f"{100*w/n:.1f}%" if n else "—"
            lp = f"{(w+1)/(n+2):.2f}" if n else "—"
            return n, wr, lp

        ns, wrs, lps = _cell(g_st)
        nc, wrc, lpc = _cell(g_co)
        na, wra, _   = _cell(grp)
        print(f"  {lbl:>20}  {ns:>5} {wrs:>6} {lps:>8}  {nc:>5} {wrc:>6} {lpc:>8}  {na:>5} {wra:>6}")

    print()
    if n_priced == 0:
        print("  Run with --fetch-candles to add YES ask prices (Section 5 + price-stratified sizer).")
    print()


if __name__ == "__main__":
    asyncio.run(main())
