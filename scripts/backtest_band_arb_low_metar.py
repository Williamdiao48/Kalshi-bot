"""
Band-arb warm-NO backtest using METAR T-group observations (0.1°C precision),
GFS daily-min forecasts (near-analysis, no lead-time control — same limitation
as high backtest), and P50/P75/P90 trough anchors from overnight_low_analysis.csv.

Strategy: Buy NO when running_min > band_ceil. Bet the daily low stays above
the band. Win if Kalshi settles "no" (final low NOT in [floor, ceil)).

Data range: 2022-01-01 → 2026-05-21 (bands limited to Kalshi settled window).
NO side only. 12 entry anchors: P50/P75/P90 × +0h/+1h/+2h/+3h.

Run:
  venv/bin/python scripts/backtest_band_arb_low_metar.py
  venv/bin/python scripts/backtest_band_arb_low_metar.py --refresh     # re-fetch METAR
  venv/bin/python scripts/backtest_band_arb_low_metar.py --fetch-gfs   # fill GFS gaps
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

from kalshi_bot.cities import LOW_CITIES
from scripts.build_forecast_calibration import IEM_STATIONS

DATA_DIR  = Path(__file__).parent.parent / "data"
CACHE_DIR = DATA_DIR / "cache" / "metar_low_historical"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
BANDS_CSV  = DATA_DIR / "kxlowt_bands.csv"
GFS_CACHE  = DATA_DIR / "cache" / "noaa_gate_low_backtest"
TROUGH_CSV = DATA_DIR / "overnight_low_analysis.csv"

BAND_START = "2026-03-15"
BAND_END   = "2026-05-21"

_UTC = ZoneInfo("UTC")

LOW_IEM = {k: v for k, v in IEM_STATIONS.items() if k.startswith("temp_low_")}

ENTRY_ANCHORS = ["p50", "p75", "p90"]
ENTRY_OFFSETS = [0, 1, 2, 3]
ENTRY_KEYS    = [f"{a}+{n}h" for a in ENTRY_ANCHORS for n in ENTRY_OFFSETS]

_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_parser.add_argument("--refresh",   action="store_true", help="Re-fetch METAR (ignores cache)")
_parser.add_argument("--fetch-gfs", action="store_true", help="Fetch missing GFS data from Open-Meteo")
_args = _parser.parse_args()


# ── Trough CSV loading ─────────────────────────────────────────────────────────

def load_trough_minutes() -> dict[tuple[str, int], dict[str, int]]:
    """Returns {(metric, month): {"p50": minutes, "p75": minutes, "p90": minutes}}."""
    result: dict[tuple[str, int], dict[str, int]] = {}
    with TROUGH_CSV.open() as f:
        for row in csv.DictReader(f):
            metric = row["city_key"]
            month  = int(row["month"])
            p_vals: dict[str, int] = {}
            for p in ("p50", "p75", "p90"):
                raw = row.get(p, "").strip()
                if raw:
                    hh, mm = raw.split(":")
                    p_vals[p] = int(hh) * 60 + int(mm)
            if p_vals:
                result[(metric, month)] = p_vals
    return result


# ── METAR parsing (verbatim from backtest_band_arb_metar.py) ──────────────────

_T_GROUP_RE   = re.compile(r'\bT([01])(\d{3})([01])(\d{3})\b')
_BODY_TEMP_RE = re.compile(r'(?<!\w)(M?\d{2})/(M?\d{2})(?!\w)')


def _parse_metar_temp_f(metar_str: str) -> float | None:
    """0.1°C from T-group; fall back to whole-degree body temp."""
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
    """Returns {utc_date_str: [(utc_dt, temp_f), ...]} sorted by time.

    Fetch range extends +1 day past BAND_END so late-night UTC observations
    (local P90 trough at 22-23h local = next UTC day) are available for merge.
    """
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
                print(f"  [IEM] {station} attempt {attempt + 1}: {ex}", file=sys.stderr)
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


# ── GFS loading / fetching ─────────────────────────────────────────────────────

def _load_gfs_cache(city_short: str) -> dict[str, float]:
    """Merge all gfs_{city}_RANGE.json files → {date_str: fahrenheit}."""
    merged: dict[str, float] = {}
    for f in sorted(GFS_CACHE.glob(f"gfs_{city_short}_20*.json")):
        try:
            merged.update(json.loads(f.read_text()))
        except Exception:
            continue
    return merged


async def _fetch_gfs_range(
    session: aiohttp.ClientSession,
    city_short: str,
    lat: float,
    lon: float,
    start: str,
    end: str,
) -> dict[str, float]:
    """Fetch GFS daily-min for [start, end], cache, and return."""
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
            print(f"  [GFS] {city_short} attempt {attempt + 1}: {e}", file=sys.stderr)
            await asyncio.sleep(2 ** attempt)
    else:
        return {}

    daily = data.get("daily", {})
    dates = daily.get("time", [])
    vals  = (
        daily.get("temperature_2m_min_gfs_seamless")
        or daily.get("temperature_2m_min") or []
    )
    result = {d: v for d, v in zip(dates, vals) if v is not None}
    cache_path.write_text(json.dumps(result))
    return result


# ── Band loading ───────────────────────────────────────────────────────────────

def load_bands() -> dict[tuple[str, str], list[dict]]:
    """Returns {(metric, local_date): [band_row, ...]}

    local_date = utc_close_date - 1 day, same convention as kxhigh_bands backtest.
    KXLOWT markets close around midnight local = 4-8 AM UTC next day, so
    UTC close date = local_date + 1.
    """
    if not BANDS_CSV.exists():
        print(f"\nERROR: {BANDS_CSV} not found.")
        print("Run first: venv/bin/python scripts/build_kxlowt_bands.py")
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

def _running_min_at(obs: list[tuple[datetime, float]], cutoff: datetime) -> float | None:
    """Min temp in obs at or before cutoff (obs must be UTC-timestamped)."""
    vals = [t for dt, t in obs if dt <= cutoff]
    return min(vals) if vals else None


def _lst(tz: ZoneInfo) -> timezone:
    """Fixed-offset standard time for tz — no DST. Matches NWS settlement day boundary."""
    return timezone(tz.utcoffset(datetime(2000, 1, 15)))  # January = standard time


def _final_daily_low(
    obs: list[tuple[datetime, float]], local_d: date, tz: ZoneInfo
) -> float | None:
    """Min temp within the NWS Local Standard Time calendar day (midnight LST → midnight LST).

    NWS CLI uses LST year-round for daily extremes, so settlement day boundaries
    are midnight CST/MST/PST etc. regardless of DST.
    """
    lst = _lst(tz)
    day_start = datetime(local_d.year, local_d.month, local_d.day, 0, 0, tzinfo=lst).astimezone(_UTC)
    day_end   = (datetime(local_d.year, local_d.month, local_d.day, 0, 0, tzinfo=lst) + timedelta(days=1)).astimezone(_UTC)
    vals = [t for dt, t in obs if day_start <= dt < day_end]
    return min(vals) if vals else None


def _trough_utc(
    metric: str, d: date, tz: ZoneInfo,
    trough_mins: dict[tuple[str, int], dict[str, int]],
    anchor: str,
) -> datetime | None:
    """UTC datetime for anchor (p50/p75/p90) on local date d."""
    p_vals = trough_mins.get((metric, d.month), {})
    minutes = p_vals.get(anchor)
    if minutes is None:
        return None
    local_midnight = datetime(d.year, d.month, d.day, 0, 0, tzinfo=tz)
    return (local_midnight + timedelta(minutes=minutes)).astimezone(_UTC)


def _stats(group: list[dict]) -> dict:
    n    = len(group)
    wins = sum(1 for r in group if r["no_win"])
    return {"n": n, "wins": wins}


def _fmt(s: dict) -> str:
    if s["n"] == 0:
        return f"{'—':>5} {'—':>6}"
    return f"{s['n']:>5} {100 * s['wins'] / s['n']:>5.1f}%"



# ── Main ───────────────────────────────────────────────────────────────────────

async def main() -> None:
    print(f"Loading trough times from {TROUGH_CSV.name}…")
    trough_mins = load_trough_minutes()
    print(f"  {len(trough_mins)} city-month entries")

    print(f"\nLoading bands from {BANDS_CSV.name}…")
    bands_index = load_bands()
    total_bands = sum(len(v) for v in bands_index.values())
    print(f"  {total_bands:,} band rows across {len(bands_index):,} (metric, date) keys")

    # ── Fetch METAR ─────────────────────────────────────────────────────────
    print(f"\nFetching METAR ({BAND_START} → {BAND_END})…")
    metar_by_metric: dict[str, dict[str, list[tuple[datetime, float]]]] = {}

    async with aiohttp.ClientSession() as session:
        for metric, (station, _) in sorted(LOW_IEM.items()):
            short = metric.replace("temp_low_", "")
            print(f"  {short:<6} ({station}) …", end=" ", flush=True)
            data = await _fetch_metar(session, station, short)
            metar_by_metric[metric] = data
            await asyncio.sleep(0.3)

    # ── Load + optionally fill GFS ────────────────────────────────────────
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

            # Identify dates needed from the bands index
            needed = {
                local_date
                for (m, local_date) in bands_index
                if m == metric
            }
            missing = sorted(d for d in needed if d not in cache)
            if not missing:
                print(f"  {short:<6}  {len(cache):>5} cached  (no gaps)")
                continue

            _, lat, lon, _ = city_info
            new = await _fetch_gfs_range(
                session, short, lat, lon, missing[0], missing[-1]
            )
            cache.update(new)
            print(f"  {short:<6}  {len(cache):>5} cached  +{len(new)} fetched ({missing[0]}→{missing[-1]})")
            await asyncio.sleep(0.5)

    # ── Build records ─────────────────────────────────────────────────────
    print("\nBuilding records…")
    records: list[dict] = []

    for (metric, local_date), bands in sorted(bands_index.items()):
        city_info = LOW_CITIES.get(metric)
        if city_info is None:
            continue
        _, _, _, tz = city_info
        short = metric.replace("temp_low_", "")

        daily_obs = metar_by_metric.get(metric, {})
        gfs_cache = gfs_by_metric.get(metric, {})

        try:
            local_d = date.fromisoformat(local_date)
        except ValueError:
            continue

        # Merge UTC observations from local_date and local_date+1 to handle
        # late-night local observations that cross the UTC midnight boundary.
        next_utc = (local_d + timedelta(days=1)).isoformat()
        obs_today = daily_obs.get(local_date, [])
        obs_next  = daily_obs.get(next_utc, [])
        merged_obs = sorted(obs_today + obs_next, key=lambda x: x[0])

        if not merged_obs:
            continue

        gfs_daily_min = gfs_cache.get(local_date)
        final_low = _final_daily_low(merged_obs, local_d, tz)

        for anchor in ENTRY_ANCHORS:
            base_utc = _trough_utc(metric, local_d, tz, trough_mins, anchor)
            if base_utc is None:
                continue

            for offset_h in ENTRY_OFFSETS:
                entry_key = f"{anchor}+{offset_h}h"
                cutoff    = base_utc + timedelta(hours=offset_h)
                rmin      = _running_min_at(merged_obs, cutoff)
                if rmin is None:
                    continue

                local_hour = cutoff.astimezone(tz).hour
                overshoot  = (rmin - gfs_daily_min) if gfs_daily_min is not None else None

                # Pre-filter and sort qualifying bands ascending by ceiling.
                # band_pos 0 = lowest ceil (most margin/cheapest YES),
                # band_pos N-1 = highest ceil (least margin/most expensive YES).
                qualifying = sorted(
                    [b for b in bands if b["strike_hi"] <= rmin],
                    key=lambda b: b["strike_hi"],
                )
                n_qual = len(qualifying)

                for band_pos, band in enumerate(qualifying):
                    band_ceil  = band["strike_hi"]
                    band_floor = band["strike_lo"]
                    margin_f   = rmin - band_ceil
                    no_win     = band["result"] == "no"

                    records.append({
                        "metric":        metric,
                        "date":          local_date,
                        "entry_key":     entry_key,
                        "anchor":        anchor,
                        "offset_h":      offset_h,
                        "local_hour":    local_hour,
                        "running_min":   rmin,
                        "gfs_daily_min": gfs_daily_min,
                        "overshoot_gfs": overshoot,
                        "band_ceil":     band_ceil,
                        "band_floor":    band_floor,
                        "margin_f":      margin_f,
                        "final_daily_low": final_low,
                        "no_win":        no_win,
                        "ticker":        band["ticker"],
                        "band_result":   band["result"],
                        "band_pos":      band_pos,    # 0=deepest (most margin), N-1=shallowest
                        "n_qualifying":  n_qual,
                    })

    total = len(records)
    print(f"Total records: {total:,}")
    if total == 0:
        print("\nNo records — check that kxlowt_bands.csv has data and METAR fetched correctly.")
        return

    # ── Section 1: Continued Cooling Stats ────────────────────────────────
    print()
    print("=" * 70)
    print("  SECTION 1 — CONTINUED COOLING AFTER ENTRY")
    print("     further_cool = running_min − final_daily_low  (always ≥ 0)")
    print("     peaked_pct = fraction where final_daily_low ≥ running_min")
    print("=" * 70)

    for entry_key in ENTRY_KEYS:
        grp = [r for r in records if r["entry_key"] == entry_key and r["final_daily_low"] is not None]
        if not grp:
            continue
        deltas = [r["running_min"] - r["final_daily_low"] for r in grp]
        peaked = sum(1 for d in deltas if d <= 0)
        avg_cool = sum(max(d, 0) for d in deltas) / len(deltas)
        p25 = sorted(deltas)[len(deltas) // 4]
        p50 = sorted(deltas)[len(deltas) // 2]
        p75 = sorted(deltas)[3 * len(deltas) // 4]

        grp_ov = [r for r in grp if r["overshoot_gfs"] is not None and r["overshoot_gfs"] >= 0]
        grp_un = [r for r in grp if r["overshoot_gfs"] is not None and r["overshoot_gfs"] < 0]
        ov_peak = sum(1 for r in grp_ov if r["running_min"] - r["final_daily_low"] <= 0) / max(len(grp_ov), 1)
        un_peak = sum(1 for r in grp_un if r["running_min"] - r["final_daily_low"] <= 0) / max(len(grp_un), 1)

        print(f"\n  {entry_key:<9}  n={len(grp):>4}  peaked={100*peaked/len(grp):>4.0f}%"
              f"  avg_more_cool={avg_cool:>4.2f}°F  p25={p25:>+5.2f} p50={p50:>+5.2f} p75={p75:>+5.2f}")
        print(f"             overshoot≥0: peaked={100*ov_peak:.0f}% (n={len(grp_ov)})"
              f"   overshoot<0: peaked={100*un_peak:.0f}% (n={len(grp_un)})")

    # ── Section 2: WR by Entry Key ─────────────────────────────────────────
    print()
    print("=" * 70)
    print("  SECTION 2 — WIN RATE BY ENTRY KEY  (12 anchors)")
    print("     no_win = Kalshi settled 'no' (daily low NOT in [floor, ceil))")
    print("=" * 70)

    _ov_flag = lambda r: (r["overshoot_gfs"] is not None and r["overshoot_gfs"] >= 0)
    _un_flag = lambda r: (r["overshoot_gfs"] is not None and r["overshoot_gfs"] < 0)

    hdr = f"  {'Entry':>10}  {'─── ALL ─────':>14}  {'─── OV≥0 ────':>14}  {'─── OV<0 ────':>14}"
    sub = f"  {'':>10}  {'n':>5} {'WR':>6}  {'n':>5} {'WR':>6}  {'n':>5} {'WR':>6}"
    print(hdr)
    print(sub)
    print("  " + "-" * 52)
    for entry_key in ENTRY_KEYS:
        grp    = [r for r in records if r["entry_key"] == entry_key]
        grp_ov = [r for r in grp if _ov_flag(r)]
        grp_un = [r for r in grp if _un_flag(r)]
        s_all  = _stats(grp)
        s_ov   = _stats(grp_ov)
        s_un   = _stats(grp_un)
        print(f"  {entry_key:>10}  {_fmt(s_all)}  {_fmt(s_ov)}  {_fmt(s_un)}")

    # ── Section 3: WR by Absolute Local Hour ──────────────────────────────
    print()
    print("=" * 70)
    print("  SECTION 3 — WIN RATE BY LOCAL ENTRY HOUR  (all anchors pooled)")
    print("=" * 70)

    hour_buckets = list(range(0, 24))
    print(f"\n  {'Hour':>5}  {'─── ALL ─────':>14}  {'─── OV≥0 ────':>14}  {'─── OV<0 ────':>14}")
    print(f"  {'':>5}  {'n':>5} {'WR':>6}  {'n':>5} {'WR':>6}  {'n':>5} {'WR':>6}")
    print("  " + "-" * 52)
    for h in hour_buckets:
        grp = [r for r in records if r["local_hour"] == h]
        if not grp:
            continue
        grp_ov = [r for r in grp if _ov_flag(r)]
        grp_un = [r for r in grp if _un_flag(r)]
        print(f"  {h:>5}  {_fmt(_stats(grp))}  {_fmt(_stats(grp_ov))}  {_fmt(_stats(grp_un))}")

    # ── Section 4: WR by GFS Overshoot Bucket ─────────────────────────────
    print()
    print("=" * 70)
    print("  SECTION 4 — WIN RATE BY GFS OVERSHOOT BUCKET  (0.5°F bins)")
    print("     overshoot_gfs = running_min − gfs_daily_min")
    print("     POSITIVE = running_min warmer than GFS daily-min (GFS expects more cooling)")
    print("     NEGATIVE = running_min already colder than GFS daily-min")
    print("=" * 70)

    ov_buckets = [
        ("< -3°F",      -99, -3.0),
        ("-3 to -2.5",  -3.0, -2.5),
        ("-2.5 to -2",  -2.5, -2.0),
        ("-2 to -1.5",  -2.0, -1.5),
        ("-1.5 to -1",  -1.5, -1.0),
        ("-1 to -0.5",  -1.0, -0.5),
        ("-0.5 to 0",   -0.5,  0.0),
        ("0 to +0.5",    0.0,  0.5),
        ("+0.5 to +1",   0.5,  1.0),
        ("+1 to +1.5",   1.0,  1.5),
        ("+1.5 to +2",   1.5,  2.0),
        ("+2 to +3",     2.0,  3.0),
        ("> +3°F",       3.0, 99.0),
    ]

    for anchor in ENTRY_ANCHORS:
        anchor_recs = [r for r in records if r["anchor"] == anchor and r["offset_h"] == 0]
        ov_recs     = [r for r in anchor_recs if r["overshoot_gfs"] is not None]
        print(f"\n  {anchor.upper()}+0h  (n with GFS data: {len(ov_recs)} / {len(anchor_recs)})")
        print(f"  {'Bucket':>14}  {'n':>5} {'WR':>6}")
        print("  " + "-" * 30)
        for lbl, lo_v, hi_v in ov_buckets:
            grp = [r for r in ov_recs if lo_v <= r["overshoot_gfs"] < hi_v]
            if grp:
                print(f"  {lbl:>14}  {_fmt(_stats(grp))}")

    # Also print GFS clearance gate: gfs_daily_min vs band_ceil
    print()
    print("  GFS CLEARANCE GATE at P75+0h  (gfs_daily_min − band_ceil)")
    print("  Positive = GFS forecasts low stays above band ceiling (good for NO)")
    p75_recs = [r for r in records if r["entry_key"] == "p75+0h"]
    cl_buckets = [
        ("< -3",     -99, -3.0),
        ("-3 to -2", -3.0, -2.0),
        ("-2 to -1", -2.0, -1.0),
        ("-1 to 0",  -1.0,  0.0),
        ("0 to +1",   0.0,  1.0),
        ("+1 to +2",  1.0,  2.0),
        ("+2 to +3",  2.0,  3.0),
        ("> +3",      3.0, 99.0),
    ]
    cl_recs = [r for r in p75_recs if r["gfs_daily_min"] is not None]
    print(f"\n  n with GFS: {len(cl_recs)} / {len(p75_recs)}")
    print(f"  {'Clearance':>10}  {'n':>5} {'WR':>6}")
    print("  " + "-" * 26)
    for lbl, lo_v, hi_v in cl_buckets:
        grp = [r for r in cl_recs if lo_v <= (r["gfs_daily_min"] - r["band_ceil"]) < hi_v]
        if grp:
            print(f"  {lbl:>10}  {_fmt(_stats(grp))}")

    # ── Section 5: WR by Margin-to-Ceiling Bucket ─────────────────────────
    print()
    print("=" * 70)
    print("  SECTION 5 — WIN RATE BY MARGIN TO BAND CEILING  (0.5°F bins)")
    print("     margin_f = running_min − band_ceil  (always > 0 for warm-NO)")
    print("=" * 70)

    margin_buckets = [
        ("0–0.5",   0.0,  0.5),
        ("0.5–1",   0.5,  1.0),
        ("1–1.5",   1.0,  1.5),
        ("1.5–2",   1.5,  2.0),
        ("2–2.5",   2.0,  2.5),
        ("2.5–3",   2.5,  3.0),
        ("> 3",     3.0, 99.0),
    ]

    print(f"\n  {'Margin':>8}  ", end="")
    for anchor in ENTRY_ANCHORS:
        print(f"  {'─── ' + anchor.upper() + '+0h ────':>14}", end="")
    print()
    print(f"  {'':>8}  ", end="")
    for _ in ENTRY_ANCHORS:
        print(f"  {'n':>5} {'WR':>6}  ", end="")
    print()
    print("  " + "-" * 60)

    for lbl, lo_m, hi_m in margin_buckets:
        print(f"  {lbl:>8}  ", end="")
        for anchor in ENTRY_ANCHORS:
            grp = [
                r for r in records
                if r["anchor"] == anchor and r["offset_h"] == 0
                and lo_m <= r["margin_f"] < hi_m
            ]
            print(f"  {_fmt(_stats(grp))}  ", end="")
        print()

    print()
    print("  P75+0h margin × overshoot split:")
    print(f"\n  {'Margin':>8}  {'─── ALL ─────':>14}  {'─── OV≥0 ────':>14}  {'─── OV<0 ────':>14}")
    print(f"  {'':>8}  {'n':>5} {'WR':>6}  {'n':>5} {'WR':>6}  {'n':>5} {'WR':>6}")
    print("  " + "-" * 52)
    for lbl, lo_m, hi_m in margin_buckets:
        p75_m = [r for r in records if r["entry_key"] == "p75+0h" and lo_m <= r["margin_f"] < hi_m]
        ov = [r for r in p75_m if _ov_flag(r)]
        un = [r for r in p75_m if _un_flag(r)]
        print(f"  {lbl:>8}  {_fmt(_stats(p75_m))}  {_fmt(_stats(ov))}  {_fmt(_stats(un))}")

    # ── Section 6: Per-City Breakdown ─────────────────────────────────────
    print()
    print("=" * 70)
    print("  SECTION 6 — PER-CITY WIN RATE AT P75+0h")
    print("=" * 70)

    cities_seen = sorted(set(r["metric"].replace("temp_low_", "") for r in records))
    print(f"\n  {'City':>6}  {'─── ALL ─────':>14}  {'─── OV≥0 ────':>14}  {'─── OV<0 ────':>14}")
    print(f"  {'':>6}  {'n':>5} {'WR':>6}  {'n':>5} {'WR':>6}  {'n':>5} {'WR':>6}")
    print("  " + "-" * 54)
    for city in cities_seen:
        grp    = [r for r in records if r["entry_key"] == "p75+0h" and r["metric"] == f"temp_low_{city}"]
        grp_ov = [r for r in grp if _ov_flag(r)]
        grp_un = [r for r in grp if _un_flag(r)]
        s_all  = _stats(grp)
        if s_all["n"] == 0:
            continue
        flag = " ←" if s_all["n"] > 0 and s_all["wins"] / s_all["n"] < 0.75 else ""
        print(f"  {city:>6}  {_fmt(s_all)}  {_fmt(_stats(grp_ov))}  {_fmt(_stats(grp_un))}{flag}")

    # ── Section 7: Combined Factor Grid ───────────────────────────────────
    print()
    print("=" * 70)
    print("  SECTION 7 — COMBINED FACTOR GRID  (P75+0h, entry_key × margin × overshoot)")
    print("=" * 70)

    def _meets(r: dict, margin_lo: float, margin_hi: float,
               ov_lo: float, ov_hi: float) -> bool:
        if not (margin_lo <= r["margin_f"] < margin_hi):
            return False
        if r["overshoot_gfs"] is None:
            return False
        return ov_lo <= r["overshoot_gfs"] < ov_hi

    combo_margins = [(0.5, 99), (1.0, 99), (1.5, 99), (2.0, 99)]
    combo_ov = [(-99, 0), (0, 1), (0, 2), (0, 99), (-99, 99)]
    combo_ov_labels = ["OV<0", "OV∈[0,1)", "OV∈[0,2)", "OV≥0", "any OV"]

    p75_recs = [r for r in records if r["entry_key"] == "p75+0h"]
    print(f"\n  P75+0h total: {len(p75_recs):,} records")
    print(f"\n  {'Margin':>8}  {'OV bucket':>12}  {'n':>5} {'WR':>6}")
    print("  " + "-" * 40)
    for margin_lo, margin_hi in combo_margins:
        for (ov_lo, ov_hi), ov_lbl in zip(combo_ov, combo_ov_labels):
            grp = [r for r in p75_recs if _meets(r, margin_lo, margin_hi, ov_lo, ov_hi)]
            s = _stats(grp)
            if s["n"] < 5:
                continue
            wr = s["wins"] / s["n"]
            flag = " ★" if wr >= 0.85 and s["n"] >= 10 else ""
            m_lbl = f"≥{margin_lo}°F"
            print(f"  {m_lbl:>8}  {ov_lbl:>12}  {_fmt(s)}{flag}")

    print()
    print("  TOP COMBOS (WR ≥ 80%, n ≥ 10):")
    combos_ranked = []
    for margin_lo, margin_hi in [(0.5, 99), (1.0, 99), (1.5, 99)]:
        for (ov_lo, ov_hi), ov_lbl in zip(combo_ov, combo_ov_labels):
            grp = [r for r in p75_recs if _meets(r, margin_lo, margin_hi, ov_lo, ov_hi)]
            s = _stats(grp)
            if s["n"] >= 10:
                wr = s["wins"] / s["n"]
                combos_ranked.append((wr, s["n"], f"margin≥{margin_lo}°F + {ov_lbl}", s))

    combos_ranked.sort(reverse=True)
    print(f"\n  {'Filter':>35}  {'n':>5} {'WR':>6}")
    print("  " + "-" * 50)
    for wr, _, lbl, s in combos_ranked[:15]:
        if wr >= 0.80:
            print(f"  {lbl:>35}  {_fmt(s)}")

    # ── Section 8: WR by Band Position (pricing proxy) ────────────────────
    print()
    print("=" * 70)
    print("  SECTION 8 — WIN RATE BY BAND POSITION  (pricing proxy, P75+0h)")
    print("     band_pos 0 = deepest band (most margin, cheapest YES / costliest NO)")
    print("     band_pos 3 = shallowest band (least margin, priciest YES / cheapest NO)")
    print("=" * 70)

    pos_hour_buckets = [
        ("<8",   0, 8),
        ("8–10", 8, 10),
        ("10–12",10, 12),
        ("12–14",12, 14),
        ("14+",  14, 24),
    ]

    max_pos = 3
    print(f"\n  {'Pos':>4}  {'─── ALL ─────':>14}  {'─── margin≥1.5°F ──':>20}")
    print(f"  {'':>4}  {'n':>5} {'WR':>6}  {'n':>5} {'WR':>6}")
    print("  " + "-" * 40)
    for pos in range(max_pos + 1):
        lbl = f"pos={pos}"
        grp     = [r for r in p75_recs if r["band_pos"] == pos]
        grp_m15 = [r for r in grp if r["margin_f"] >= 1.5]
        flag = " ←" if grp and (_stats(grp)["wins"] / _stats(grp)["n"]) < 0.80 else ""
        print(f"  {lbl:>4}  {_fmt(_stats(grp))}  {_fmt(_stats(grp_m15))}{flag}")
    grp_3p = [r for r in p75_recs if r["band_pos"] >= 4]
    if grp_3p:
        print(f"  {'pos≥4':>4}  {_fmt(_stats(grp_3p))}")

    # ── Section 9: WR by Band Position × Entry Hour ───────────────────────
    print()
    print("=" * 70)
    print("  SECTION 9 — WIN RATE BY BAND POSITION × ENTRY HOUR  (P75+0h, margin≥1.5°F)")
    print("=" * 70)

    m15_recs = [r for r in p75_recs if r["margin_f"] >= 1.5]
    hdr = "  " + f"{'Pos':>4}  " + "  ".join(f"{'─'+b+'─':>9}" for b, *_ in pos_hour_buckets)
    print(f"\n{hdr}")
    sub = "  " + f"{'':>4}  " + "  ".join(f"{'n':>4} {'WR':>5}" for _ in pos_hour_buckets)
    print(sub)
    print("  " + "-" * 65)
    for pos in range(max_pos + 1):
        row_recs = [r for r in m15_recs if r["band_pos"] == pos]
        parts = []
        for _, h_lo, h_hi in pos_hour_buckets:
            g = [r for r in row_recs if h_lo <= r["local_hour"] < h_hi]
            s = _stats(g)
            if s["n"] == 0:
                parts.append(f"{'—':>4} {'—':>5}")
            else:
                parts.append(f"{s['n']:>4} {100*s['wins']/s['n']:>4.0f}%")
        print(f"  {'pos='+str(pos):>4}  " + "  ".join(parts))

    # ── Section 10: WR by Band Position × Margin ──────────────────────────
    print()
    print("=" * 70)
    print("  SECTION 10 — WIN RATE BY BAND POSITION × MARGIN  (P75+0h)")
    print("=" * 70)

    pos_margin_buckets = [
        ("0.5–1", 0.5, 1.0),
        ("1–1.5", 1.0, 1.5),
        ("1.5–2", 1.5, 2.0),
        ("2–3",   2.0, 3.0),
        (">3",    3.0, 99.0),
    ]
    hdr = "  " + f"{'Pos':>4}  " + "  ".join(f"{'─'+b+'─':>10}" for b, *_ in pos_margin_buckets)
    print(f"\n{hdr}")
    sub = "  " + f"{'':>4}  " + "  ".join(f"{'n':>5} {'WR':>5}" for _ in pos_margin_buckets)
    print(sub)
    print("  " + "-" * 70)
    for pos in range(max_pos + 1):
        row_recs = [r for r in p75_recs if r["band_pos"] == pos]
        parts = []
        for _, m_lo, m_hi in pos_margin_buckets:
            g = [r for r in row_recs if m_lo <= r["margin_f"] < m_hi]
            s = _stats(g)
            if s["n"] == 0:
                parts.append(f"{'—':>5} {'—':>5}")
            else:
                parts.append(f"{s['n']:>5} {100*s['wins']/s['n']:>4.0f}%")
        print(f"  {'pos='+str(pos):>4}  " + "  ".join(parts))

    # Summary: GFS gate recommendation
    print()
    print("=" * 70)
    print("  SUMMARY — GFS CLEARANCE GATE ANALYSIS AT P75+0h")
    print("     gfs_clearance = gfs_daily_min − band_ceil")
    print("     Current bot gate: gfs_daily_min > band_ceil + 2°F")
    print("=" * 70)

    gfs_gates = [(0, "gfs_daily_min > band_ceil"),
                 (1, "gfs_daily_min > band_ceil + 1°F"),
                 (2, "gfs_daily_min > band_ceil + 2°F"),
                 (3, "gfs_daily_min > band_ceil + 3°F")]

    cl_recs_p75 = [r for r in p75_recs if r["gfs_daily_min"] is not None]
    print(f"\n  n with GFS at P75+0h: {len(cl_recs_p75)}")
    print(f"\n  {'Gate':>38}  {'n':>5} {'WR':>6}  trades_kept")
    print("  " + "-" * 60)
    baseline = _stats(cl_recs_p75)
    print(f"  {'no gate (baseline)':>38}  {_fmt(baseline)}")
    for thresh, label in gfs_gates:
        grp = [r for r in cl_recs_p75 if (r["gfs_daily_min"] - r["band_ceil"]) >= thresh]
        s = _stats(grp)
        pct_kept = 100 * s["n"] / max(baseline["n"], 1)
        if s["n"] > 0:
            print(f"  {label:>38}  {_fmt(s)}  ({pct_kept:.0f}% of trades kept)")

    # GFS clearance × band_pos cross-tab (margin≥1.5°F, P75+0h)
    # Answers: does the GFS gate help more for affordable (pos=2/3) trades than
    # the expensive pos=0/1 trades the bot skips?
    print()
    print("=" * 70)
    print("  SECTION 11 — GFS CLEARANCE × BAND POSITION  (P75+0h, margin≥1.5°F)")
    print("     gfs_clearance = gfs_daily_min − band_ceil")
    print("     Rows = band_pos (pricing proxy); cols = clearance gate threshold")
    print("=" * 70)

    cl_m15 = [r for r in p75_recs
              if r["margin_f"] >= 1.5 and r["gfs_daily_min"] is not None]

    gfs_thresholds = [
        ("no gate",  None),
        ("≥0°F",      0),
        ("≥+1°F",     1),
        ("≥+2°F",     2),
        ("≥+3°F",     3),
    ]
    col_w = 14
    hdr11  = f"  {'Pos':>5}  " + "  ".join(f"{lbl:>{col_w}}" for lbl, _ in gfs_thresholds)
    sub11  = f"  {'':>5}  " + "  ".join(f"{'n':>5} {'WR':>5} {'%kpt':>4}" for _ in gfs_thresholds)
    print(f"\n{hdr11}")
    print(sub11)
    print("  " + "-" * 85)

    for pos in range(max_pos + 1):
        pos_recs = [r for r in cl_m15 if r["band_pos"] == pos]
        parts = []
        base_n = len(pos_recs)
        for _, thresh in gfs_thresholds:
            if thresh is None:
                g = pos_recs
            else:
                g = [r for r in pos_recs if (r["gfs_daily_min"] - r["band_ceil"]) >= thresh]
            s = _stats(g)
            pct = 100 * s["n"] / max(base_n, 1)
            if s["n"] == 0:
                parts.append(f"{'—':>5} {'—':>5} {'—':>4}")
            else:
                parts.append(f"{s['n']:>5} {100*s['wins']/s['n']:>4.0f}% {pct:>3.0f}%")
        print(f"  {'pos='+str(pos):>5}  " + "  ".join(parts))

    print()
    print("  Reading: for each band_pos, % of trades kept shrinks left→right as threshold rises.")
    print("  High WR lift per gate tightening shows how much that pos benefits from GFS filter.")

    # ── Section 12: Kelly sizing simulation ───────────────────────────────────
    # The backtest has no market prices, so we can't compute exact Kelly fractions.
    # Instead we show EV and sizing metrics at three assumed NO ask price levels
    # (60¢, 70¢, 80¢) representing the affordable range the bot actually enters.
    # For each (band_pos × margin) bucket at P75+0h, we report:
    #   WR   — win probability from backtest
    #   EV   — expected value per contract = WR*(100-ask) - (1-WR)*ask
    #   Kf   — half-Kelly fraction (half-Kelly is standard for correlated markets)
    #   EV$  — dollar EV per 10-contract trade at that Kelly stake
    # Positive EV requires WR > NO_ask (e.g. WR>70% for NO at 70¢).
    print()
    print()
    print("=" * 70)
    print("  SECTION 12 — KELLY SIZING SIMULATION  (P75+0h, assumed NO ask prices)")
    print("     EV/c = expected value per contract  (WR*(100-ask) - (1-WR)*ask)")
    print("     Kf   = half-Kelly fraction of bankroll to stake per trade")
    print("     EV$  = dollar EV for a 10-contract trade at half-Kelly stake")
    print("     Three price scenarios: ask=60¢  ask=70¢  ask=80¢")
    print("     bot's actual range: 20–88¢ NO ask; pos=2/3 likely 50–85¢")
    print("=" * 70)

    PRICE_SCENARIOS = [60, 70, 80]

    def _kelly_row(wr: float, asks: list[int]) -> list[str]:
        parts = []
        for ask in asks:
            payout = 100 - ask
            ev = wr * payout - (1 - wr) * ask          # EV per contract (cents)
            full_k = ev / payout if payout > 0 else 0  # full Kelly fraction
            half_k = max(0.0, full_k / 2)              # half-Kelly (safer)
            # Dollar EV for 10-contract trade: contracts * EV/contract / 100
            ev_dollar = 10 * ev / 100
            if ev <= 0:
                parts.append(f"  {'neg':>6} {'—':>5} {'—':>6}")
            else:
                parts.append(f"  {ev:>+5.1f}¢ {100*half_k:>4.1f}% {ev_dollar:>+5.2f}$")
        return parts

    col_hdr = "  ".join(f"{'ask='+str(a)+'¢':>20}" for a in PRICE_SCENARIOS)
    sub_hdr = "  ".join(f"{'EV/c':>6} {'Kf':>5} {'EV$':>6}" for _ in PRICE_SCENARIOS)

    for pos in range(max_pos + 1):
        print()
        print(f"  pos={pos}  {'─'*60}")
        print(f"  {'Margin':>10}   {'n':>5}   {'WR':>6}   {col_hdr}")
        print(f"  {'':>10}   {'':>5}   {'':>6}   {sub_hdr}")
        print("  " + "-" * 85)
        row_recs = [r for r in p75_recs if r["band_pos"] == pos]
        for label, m_lo, m_hi in pos_margin_buckets:
            g = [r for r in row_recs if m_lo <= r["margin_f"] < m_hi]
            s = _stats(g)
            if s["n"] < 5:
                continue
            wr = s["wins"] / s["n"]
            kelly_cols = "   ".join(_kelly_row(wr, PRICE_SCENARIOS))
            print(f"  {label:>10}   {s['n']:>5}   {100*wr:>5.1f}%   {kelly_cols}")

    print()
    print("  Note: WR from backtest temperature data only — actual market prices")
    print("  may differ. Use shadow trade avg NO ask per bucket to calibrate.")
    print()


if __name__ == "__main__":
    asyncio.run(main())
