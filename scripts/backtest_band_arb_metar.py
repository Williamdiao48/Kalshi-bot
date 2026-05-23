"""
band_arb YES backtest using historical raw METAR observations.

Improvements over _tmp_continued_heat.py:
  - 0.1°C precision via METAR T-group remarks (not whole-degree IEM tmpf)
  - Rounding-aware band membership: temp rounds to NWS whole-degree in [band_lo, band_hi]
  - Entry timing sweep: first band-crossing, P75-2h, P75-1h, P75, P75+1h
  - Feb-May 2026 only (actual band data period)

Run:
  venv/bin/python scripts/backtest_band_arb_metar.py
  venv/bin/python scripts/backtest_band_arb_metar.py --refresh   # re-fetch METAR
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

from data.peak_hour_p90 import P75_MINUTES
from kalshi_bot.cities import CITIES
from scripts.build_forecast_calibration import IEM_STATIONS

DATA_DIR    = Path(__file__).parent.parent / "data"
CACHE_DIR   = DATA_DIR / "cache" / "metar_historical"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
BANDS_CSV   = DATA_DIR / "kxhigh_bands.csv"
CANDLE_JSON = DATA_DIR / "band_arb_candle_cache.json"
GFS_CACHE   = DATA_DIR / "cache" / "continued_heat"  # re-use existing GFS cache

HIGH_IEM = {k: v for k, v in IEM_STATIONS.items() if k.startswith("temp_high_")}

BAND_START = "2026-02-01"
BAND_END   = "2026-05-21"

_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_parser.add_argument("--refresh", action="store_true", help="Re-fetch METAR from IEM (ignores cache)")
_args = _parser.parse_args()

# ── METAR parsing ─────────────────────────────────────────────────────────────

_T_GROUP_RE = re.compile(r'\bT([01])(\d{3})([01])(\d{3})\b')
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
    """Returns {date_str: [(utc_dt, temp_f), ...]} sorted by time."""
    cache_path = CACHE_DIR / f"metar_{cache_key}_{BAND_START}_{BAND_END}.csv"
    if cache_path.exists() and not _args.refresh:
        raw = cache_path.read_text()
    else:
        s = datetime.fromisoformat(BAND_START)
        e = datetime.fromisoformat(BAND_END)
        url = (
            "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
            f"?station={station}&data=metar"
            f"&year1={s.year}&month1={s.month}&day1={s.day}"
            f"&year2={e.year}&month2={e.month}&day2={e.day}"
            "&tz=UTC&format=comma&latlon=no&direct=yes&report_type=3"
        )
        for attempt in range(4):
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=180)) as resp:
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
            dt = datetime.strptime(row[1].strip(), "%Y-%m-%d %H:%M").replace(tzinfo=ZoneInfo("UTC"))
        except ValueError:
            continue
        metar_str = row[2].strip()
        has_tgroup = bool(_T_GROUP_RE.search(metar_str))
        t_count += has_tgroup
        other_count += not has_tgroup
        temp_f = _parse_metar_temp_f(metar_str)
        if temp_f is not None:
            result[dt.date().isoformat()].append((dt, temp_f))

    # sort each day's observations by time
    for d in result:
        result[d].sort(key=lambda x: x[0])

    total = t_count + other_count
    print(f"  T-group coverage: {t_count}/{total} ({100*t_count/total:.0f}%)" if total else "  no obs")
    return dict(result)


# ── GFS morning forecast ──────────────────────────────────────────────────────

def _load_gfs_cache(city_short: str) -> dict[str, float]:
    """Load GFS daily max from existing continued_heat cache → {date_str: daily_max_f}.
    Merges all matching files so later date ranges (e.g. _2026-05-15) are included."""
    merged: dict[str, float] = {}
    for f in sorted(GFS_CACHE.glob(f"gfs_{city_short}_*.json")):
        try:
            merged.update(json.loads(f.read_text()))
        except Exception:
            continue
    return merged


# ── Band / candle loaders ─────────────────────────────────────────────────────

_TICKER_PREFIX_MAP: dict[str, str] = {
    "kxhighaus":   "aus", "kxhightchi":  "chi", "kxhighchi":   "chi",
    "kxhighden":   "den", "kxhighlax":   "lax", "kxhighmia":   "mia",
    "kxhighny":    "ny",  "kxhightatl":  "atl",
    "kxhightbos":  "bos", "kxhightdal":  "dfw", "kxhightdc":   "dca",
    "kxhighthou":  "hou", "kxhightlv":   "las", "kxhightmin":  "msp",
    "kxhightmia":  "mia", "kxhightnola": "msy", "kxhightokc":  "okc",
    "kxhightphil": "phl", "kxhightphx":  "phx", "kxhightsatx": "sat",
    "kxhightsea":  "sea", "kxhightsfo":  "sfo", "kxhighphil":  "phl",
}


def _ticker_to_metric(ticker: str) -> str | None:
    tl = ticker.lower()
    for prefix, suffix in _TICKER_PREFIX_MAP.items():
        if tl.startswith(prefix):
            return f"temp_high_{suffix}"
    return None


def load_bands() -> dict[tuple[str, str], list[dict]]:
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


def load_candles() -> dict[str, dict[int, float]]:
    if not CANDLE_JSON.exists():
        return {}
    raw = json.loads(CANDLE_JSON.read_text())
    result: dict[str, dict[int, float]] = {}
    for ticker, candles in raw.items():
        if not candles:
            continue
        metric  = _ticker_to_metric(ticker)
        city_tz = CITIES[metric][3] if metric and metric in CITIES else None
        hourly: dict[int, float] = {}
        for c in candles:
            close_str = (c.get("yes_ask") or {}).get("close_dollars")
            if close_str is None:
                continue
            try:
                ask_cents = round(float(close_str) * 100)
            except (ValueError, TypeError):
                continue
            ts = datetime.fromtimestamp(c["end_period_ts"], tz=timezone.utc)
            local_hour = ts.astimezone(city_tz).hour if city_tz else ts.hour
            hourly[local_hour] = ask_cents
        if hourly:
            result[ticker] = hourly
    return result


# ── Core helpers ──────────────────────────────────────────────────────────────

def _running_max_at(obs: list[tuple[datetime, float]], cutoff: datetime) -> float | None:
    vals = [t for dt, t in obs if dt <= cutoff]
    return max(vals) if vals else None


def _spot_temp_at(obs: list[tuple[datetime, float]], cutoff: datetime) -> float | None:
    """Most recent METAR temperature observation at or before cutoff (obs must be time-sorted)."""
    recent = [t for dt, t in obs if dt <= cutoff]
    return recent[-1] if recent else None


def _p75_utc(metric: str, d: date, tz: ZoneInfo) -> datetime | None:
    minutes = P75_MINUTES.get(metric, {}).get(d.month)
    if minutes is None:
        return None
    local_midnight = datetime(d.year, d.month, d.day, 0, 0, tzinfo=tz)
    return (local_midnight + timedelta(minutes=minutes)).astimezone(ZoneInfo("UTC"))



def _ask_at_hour(hourly: dict[int, float], hour: int) -> float | None:
    for delta in [0, 1, -1, 2, -2]:
        v = hourly.get(hour + delta)
        if v is not None:
            return v
    return None


def simulate_exit(
    entry_cents: float,
    result: str,
    hourly_asks: dict[int, float],
    entry_hour: int,
    pt_cents: float | None,
    sl_frac: float | None,
) -> dict:
    sl_thresh = entry_cents * (1 - sl_frac) if sl_frac is not None else None
    for dh in range(1, 7):
        ask_h = _ask_at_hour(hourly_asks, entry_hour + dh)
        if ask_h is None:
            continue
        if pt_cents is not None and ask_h >= pt_cents:
            return {"pnl": ask_h - entry_cents, "reason": "PT"}
        if sl_thresh is not None and ask_h <= sl_thresh:
            return {"pnl": ask_h - entry_cents, "reason": "SL"}
    won = result == "yes"
    return {"pnl": (100 if won else 0) - entry_cents, "reason": "W" if won else "L"}


def grid_stats(group: list[dict], pt: float | None, sl: float | None) -> dict:
    sims = []
    for r in group:
        if r["cents"] is None:
            continue
        s = simulate_exit(r["cents"], r["result"], r["hourly_asks"], r["local_hour"], pt, sl)
        sims.append(s)
    if not sims:
        return {"n": 0}
    wins = sum(1 for s in sims if s["pnl"] > 0)
    pnls = [s["pnl"] for s in sims]
    return {"n": len(sims), "wr": wins / len(sims), "avg": sum(pnls) / len(pnls), "total": sum(pnls)}


def _fmt(s: dict) -> str:
    if s["n"] == 0:
        return f"{'n/a':>5} {'—':>6} {'—':>8} {'—':>9}"
    return f"{s['n']:>5} {100*s['wr']:>5.1f}% {s['avg']:>+7.1f}¢ ${s['total']/100:>+8.2f}"



# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    print("Loading bands and candles…")
    bands_index = load_bands()
    candles     = load_candles()
    print(f"  {sum(len(v) for v in bands_index.values()):,} band rows")
    print(f"  {len(candles):,} tickers with candle data")

    # Flat lookup: (metric, local_date, int(strike_lo)) → band row
    # One band per temperature rounded value — no overlap possible.
    bands_flat: dict[tuple[str, str, int], dict] = {}
    for (metric, local_date), band_list in bands_index.items():
        for band in band_list:
            bands_flat[(metric, local_date, int(band["strike_lo"]))] = band

    # ── Fetch METAR per city ─────────────────────────────────────────────────
    print(f"\nFetching METAR ({BAND_START} → {BAND_END})…")
    metar_by_metric: dict[str, dict[str, list[tuple[datetime, float]]]] = {}

    async with aiohttp.ClientSession() as session:
        for metric, (station, _) in sorted(HIGH_IEM.items()):
            short = metric.replace("temp_high_", "")
            print(f"  {short:<6} ({station}) …", end=" ", flush=True)
            data = await _fetch_metar(session, station, short)
            metar_by_metric[metric] = data
            await asyncio.sleep(0.5)

    # ── Build records ────────────────────────────────────────────────────────
    # For each (city, date, entry_timing), observe running_max, round to nearest °F,
    # find the ONE band where strike_lo == rounded. No double-counting across bands.

    ENTRY_KEYS    = ["p75m2", "p75m1", "p75", "p75p1", "p75p2"]
    ENTRY_OFFSETS = {"p75m2": -2, "p75m1": -1, "p75": 0, "p75p1": 1, "p75p2": 2}
    ENTRY_LABELS  = {"p75m2": "P75-2h", "p75m1": "P75-1h", "p75": "P75",
                     "p75p1": "P75+1h", "p75p2": "P75+2h"}

    records: list[dict] = []

    for metric, daily_obs in sorted(metar_by_metric.items()):
        city_entry = CITIES.get(metric)
        if city_entry is None:
            continue
        _, _, _, tz = city_entry
        short     = metric.replace("temp_high_", "")
        gfs_cache = _load_gfs_cache(short)

        for local_date, obs in sorted(daily_obs.items()):
            try:
                d = date.fromisoformat(local_date)
            except ValueError:
                continue

            p75_dt = _p75_utc(metric, d, tz)
            if p75_dt is None:
                continue

            gfs_fcst = gfs_cache.get(local_date)

            for key in ENTRY_KEYS:
                offset_h = ENTRY_OFFSETS[key]
                cutoff   = p75_dt + timedelta(hours=offset_h)
                rmax     = _running_max_at(obs, cutoff)
                if rmax is None:
                    continue

                rounded = round(rmax)
                band    = bands_flat.get((metric, local_date, rounded))
                if band is None:
                    continue

                lo          = band["strike_lo"]
                hi          = band["strike_hi"]
                result      = band["result"]
                ticker      = band["ticker"]
                hourly_asks = candles.get(ticker, {})
                local_hour  = cutoff.astimezone(tz).hour
                cents       = _ask_at_hour(hourly_asks, local_hour)
                margin_hi   = (hi + 0.5) - rmax
                overshoot   = (rmax - gfs_fcst) if gfs_fcst is not None else None

                rmax_1h_ago   = _running_max_at(obs, cutoff - timedelta(hours=1))
                rmax_2h_ago   = _running_max_at(obs, cutoff - timedelta(hours=2))
                spot_now      = _spot_temp_at(obs, cutoff)
                spot_1h_ago   = _spot_temp_at(obs, cutoff - timedelta(hours=1))
                plateau_1h    = rmax_1h_ago is not None and rmax == rmax_1h_ago
                plateau_2h    = rmax_2h_ago is not None and rmax == rmax_2h_ago
                temp_delta_1h = (spot_now - spot_1h_ago) if spot_now is not None and spot_1h_ago is not None else None

                records.append({
                    "metric":        metric,
                    "date":          local_date,
                    "entry_key":     key,
                    "rmax":          rmax,
                    "band_lo":       lo,
                    "band_hi":       hi,
                    "margin_hi":     margin_hi,
                    "result":        result,
                    "ticker":        ticker,
                    "local_hour":    local_hour,
                    "cents":         cents,
                    "overshoot":     overshoot,
                    "hourly_asks":   hourly_asks,
                    "plateau_1h":    plateau_1h,
                    "plateau_2h":    plateau_2h,
                    "temp_delta_1h": temp_delta_1h,
                })

    total = len(records)
    print(f"\nTotal records: {total:,}")
    for key in ENTRY_KEYS:
        grp    = [r for r in records if r["entry_key"] == key]
        priced = sum(1 for r in grp if r["cents"] is not None)
        print(f"  {ENTRY_LABELS[key]:<8}: {len(grp):>5} in-band  {priced:>4} priced")

    def _overshoot_flag(r: dict) -> bool | None:
        if r.get("overshoot") is None:
            return None
        return r["overshoot"] >= 0

    # ── Section 1: WR by entry timing ────────────────────────────────────────
    print()
    print("=" * 70)
    print("  1. WIN RATE AND PNL BY ENTRY TIMING  (hold to settle)")
    print("     Each row = independent entries at that timing; running_max rounded")
    print("     to nearest °F determines which single band qualifies.")
    print("=" * 70)

    hdr_cols = f"  {'n':>5} {'WR':>6} {'avg¢':>8} {'$total':>9}"
    print(f"\n  {'Entry':>8}  {'─── ALL ──────────────────────':>32}"
          f"  {'─── OVERSHOT ─────────────────':>32}"
          f"  {'─── LAGGING ──────────────────':>32}")
    print(f"  {'':>8}{hdr_cols}{hdr_cols}{hdr_cols}")
    print("  " + "-" * 105)

    for key in ENTRY_KEYS:
        grp    = [r for r in records if r["entry_key"] == key]
        grp_ov = [r for r in grp if _overshoot_flag(r) is True]
        grp_la = [r for r in grp if _overshoot_flag(r) is False]
        s_all  = grid_stats(grp, None, None)
        s_ov   = grid_stats(grp_ov, None, None)
        s_la   = grid_stats(grp_la, None, None)
        print(f"  {ENTRY_LABELS[key]:>8}  {_fmt(s_all)}  {_fmt(s_ov)}  {_fmt(s_la)}")

    # ── Section 2: WR by margin bucket at P75 ────────────────────────────────
    print()
    print("=" * 70)
    print("  2. WIN RATE BY MARGIN-TO-CEILING AT P75")
    print("     margin_hi = (band_hi + 0.5) - running_max  (rounding-aware ceiling)")
    print("=" * 70)

    margin_buckets = [
        ("<0.5",   0.0,  0.5),
        ("0.5-1",  0.5,  1.0),
        ("1-1.5",  1.0,  1.5),
        ("1.5-2",  1.5,  2.0),
        (">2",     2.0, 99.0),
    ]

    p75_recs = [r for r in records if r["entry_key"] == "p75"]
    print(f"\n  {'Margin':>8}  {'ALL':>32}  {'OVERSHOT':>32}  {'LAGGING':>32}")
    print(f"  {'':>8}{hdr_cols}{hdr_cols}{hdr_cols}")
    print("  " + "-" * 110)

    for lbl, lo_b, hi_b in margin_buckets:
        grp_all = [r for r in p75_recs if lo_b <= r["margin_hi"] < hi_b]
        grp_ov  = [r for r in grp_all if _overshoot_flag(r) is True]
        grp_la  = [r for r in grp_all if _overshoot_flag(r) is False]
        row     = "  ".join(_fmt(grid_stats(g, None, None)) for g in [grp_all, grp_ov, grp_la])
        print(f"  {lbl:>8}  {row}")

    # ── Section 3: PT/SL grid at P75 ─────────────────────────────────────────
    print()
    print("=" * 70)
    print("  3. PT/SL GRID AT P75  (priced trades only)")
    print("=" * 70)

    PT_VALS  = [80.0, 85.0, 90.0, 95.0, None]
    SL_FRACS = [0.3, 0.5, 0.7, None]
    pt_label = {80.0: "PT=80¢", 85.0: "PT=85¢", 90.0: "PT=90¢", 95.0: "PT=95¢", None: "hold  "}
    sl_label = {0.3: "SL=30%", 0.5: "SL=50%", 0.7: "SL=70%", None: "no-SL"}

    groups_3 = [
        ("ALL",      p75_recs),
        ("OVERSHOT", [r for r in p75_recs if _overshoot_flag(r) is True]),
        ("LAGGING",  [r for r in p75_recs if _overshoot_flag(r) is False]),
    ]
    for sub_label, sub_grp in groups_3:
        n_priced = sum(1 for r in sub_grp if r["cents"] is not None)
        print(f"\n  P75 — {sub_label}  (priced n={n_priced})")
        header = "  ".join(f"{sl_label[sl]:>36}" for sl in SL_FRACS)
        cols   = "  ".join(f"{'n':>5} {'WR':>6} {'avg¢':>7} {'$PnL':>8}" for _ in SL_FRACS)
        print(f"  {'':14}  {header}")
        print(f"  {'':14}  {cols}")
        print("  " + "-" * (14 + 2 + 36 * len(SL_FRACS) + 2 * (len(SL_FRACS) - 1)))
        for pt in PT_VALS:
            row_parts = [_fmt(grid_stats(sub_grp, pt, sl)) for sl in SL_FRACS]
            print(f"  {pt_label[pt]:14}  " + "  ".join(row_parts))

    # ── Section 4: METAR precision check ─────────────────────────────────────
    print()
    print("=" * 70)
    print("  4. METAR PRECISION CHECK  (margin_hi distribution at P75)")
    print("=" * 70)

    margins = [r["margin_hi"] for r in p75_recs]
    if margins:
        buckets_02: dict[float, int] = defaultdict(int)
        for m in margins:
            b = round(m * 5) / 5
            buckets_02[b] += 1
        print(f"\n  n={len(margins)}  mean={sum(margins)/len(margins):.3f}°F  "
              f"min={min(margins):.2f}  max={max(margins):.2f}")
        print(f"\n  margin_hi bucket (0.2°F wide):")
        for b in sorted(buckets_02):
            bar = "█" * (buckets_02[b] // max(1, len(margins) // 80))
            print(f"  {b:>6.2f}°F  {buckets_02[b]:>4}  {bar}")

    # ── Section 5: Overshoot magnitude ───────────────────────────────────────
    print()
    print("=" * 70)
    print("  5. OVERSHOOT MAGNITUDE AT P75  (hold to settle)")
    print("     overshoot = running_max_at_p75 - GFS_morning_forecast")
    print("=" * 70)

    ov_buckets = [
        ("< -3°F",      -99, -3.0),
        ("-3 to -2.5°F", -3.0, -2.5),
        ("-2.5 to -2°F", -2.5, -2.0),
        ("-2 to -1.5°F", -2.0, -1.5),
        ("-1.5 to -1°F", -1.5, -1.0),
        ("-1 to -0.5°F", -1.0, -0.5),
        ("-0.5 to  0°F", -0.5,  0.0),
        ("0 to +0.5°F",   0.0,  0.5),
        ("+0.5 to +1°F",  0.5,  1.0),
        ("+1 to +1.5°F",  1.0,  1.5),
        ("+1.5 to +2°F",  1.5,  2.0),
        ("> +2°F",        2.0, 99.0),
    ]
    p75_ov_recs = [r for r in p75_recs if r.get("overshoot") is not None]
    print(f"\n  n with GFS data: {len(p75_ov_recs)} / {len(p75_recs)}")
    print(f"\n  {'Bucket':>12}  {'n':>5} {'WR':>6} {'avg¢':>8} {'$total':>9}")
    print("  " + "-" * 47)
    for lbl, lo_v, hi_v in ov_buckets:
        grp = [r for r in p75_ov_recs if lo_v <= r["overshoot"] < hi_v]
        print(f"  {lbl:>12}  {_fmt(grp and grid_stats(grp, None, None) or {'n': 0})}")

    # Lagging margin = GFS - rmax (how much more heating is expected)
    # Combined with ceiling distance to assess bust risk
    lag_margin_buckets = [
        ("0–0.5°F",    0.0,  0.5),
        ("0.5–1°F",    0.5,  1.0),
        ("1–1.5°F",    1.0,  1.5),
        ("1.5–2°F",    1.5,  2.0),
        ("2–2.5°F",    2.0,  2.5),
        ("2.5–3°F",    2.5,  3.0),
        ("> 3°F",      3.0, 99.0),
    ]
    p75_lag = [r for r in p75_ov_recs if r["overshoot"] < 0]
    print(f"\n  LAGGING only (n={len(p75_lag)}):")
    print(f"  lag_margin = GFS_forecast - running_max  (expected remaining rise)")
    print(f"\n  {'Lag margin':>10}  {'n':>5} {'WR':>6} {'avg¢':>8} {'$total':>9}  {'ceiling dist':>12}")
    print("  " + "-" * 62)
    for lbl, lo_v, hi_v in lag_margin_buckets:
        grp = [r for r in p75_lag if lo_v <= -r["overshoot"] < hi_v]
        s   = grid_stats(grp, None, None)
        avg_ceil = sum(r["margin_hi"] for r in grp) / len(grp) if grp else 0.0
        print(f"  {lbl:>10}  {_fmt(s)}  {avg_ceil:>11.2f}°F")

    # ── Section 6: Entry price breakdown ─────────────────────────────────────
    print()
    print("=" * 70)
    print("  6. WIN RATE BY ENTRY PRICE AT P75  (hold to settle)")
    print("=" * 70)

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
    print(f"\n  n priced: {len(p75_priced)}")
    print(f"\n  {'Price':>8}  {'n':>5} {'WR':>6} {'avg¢':>8} {'$total':>9}  {'entry avg':>9}")
    print("  " + "-" * 58)
    for lbl, lo_p, hi_p in price_buckets:
        grp = [r for r in p75_priced if lo_p <= r["cents"] < hi_p]
        s   = grid_stats(grp, None, None)
        avg_e = sum(r["cents"] for r in grp) / len(grp) if grp else 0.0
        print(f"  {lbl:>8}  {_fmt(s)}  {avg_e:>8.1f}¢")

    # ── Section 7: City breakdown ─────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  7. WIN RATE BY CITY AT P75  (hold to settle, priced only)")
    print("=" * 70)

    def _sfmt(s: dict) -> str:
        if s["n"] == 0:
            return f"{'—':>4} {'—':>5} {'—':>7}"
        return f"{s['n']:>4} {100*s['wr']:>4.0f}% {s['avg']:>+6.1f}¢"

    print(f"\n  {'City':>6}  {'ALL':>18}  {'OVERSHOT':>18}  {'LAGGING':>18}")
    print(f"  {'':>6}  {'n':>4} {'WR':>5} {'avg¢':>7}" * 3)
    print("  " + "-" * 68)
    for city in sorted(set(r["metric"].replace("temp_high_", "") for r in p75_recs)):
        grp    = [r for r in p75_recs if r["metric"] == f"temp_high_{city}"]
        grp_ov = [r for r in grp if _overshoot_flag(r) is True]
        grp_la = [r for r in grp if _overshoot_flag(r) is False]
        print(f"  {city:>6}  {_sfmt(grid_stats(grp, None, None))}"
              f"  {_sfmt(grid_stats(grp_ov, None, None))}"
              f"  {_sfmt(grid_stats(grp_la, None, None))}")

    # ── Section 8: Plateau filter ─────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  8. PLATEAU FILTER AT P75")
    print("     plateau_Nh: running_max unchanged for N hours before P75")
    print("=" * 70)

    for p_key, p_lbl in [("plateau_1h", "1h flat"), ("plateau_2h", "2h flat")]:
        grp_plat    = [r for r in p75_recs if r[p_key]]
        grp_rise    = [r for r in p75_recs if not r[p_key]]
        grp_plat_ov = [r for r in grp_plat if _overshoot_flag(r) is True]
        grp_plat_la = [r for r in grp_plat if _overshoot_flag(r) is False]
        grp_rise_ov = [r for r in grp_rise if _overshoot_flag(r) is True]
        grp_rise_la = [r for r in grp_rise if _overshoot_flag(r) is False]
        print(f"\n  ── {p_lbl} ──")
        print(f"  {'Group':>20}  {'n':>5} {'WR':>6} {'avg¢':>8} {'$total':>9}")
        print("  " + "-" * 55)
        for lbl, g in [
            ("plateaued (all)",    grp_plat),
            ("  + overshot",       grp_plat_ov),
            ("  + lagging",        grp_plat_la),
            ("still rising (all)", grp_rise),
            ("  + overshot",       grp_rise_ov),
            ("  + lagging",        grp_rise_la),
        ]:
            print(f"  {lbl:>20}  {_fmt(grid_stats(g, None, None))}")

    # ── Section 9: Temperature trend into P75 ────────────────────────────────
    print()
    print("=" * 70)
    print("  9. TEMPERATURE TREND INTO P75  (spot temp delta over last 1h)")
    print("     Negative = cooling/peaked, positive = still rising")
    print("=" * 70)

    delta_buckets = [
        ("< -1°F",   -99.0, -1.0),
        ("-1–0°F",    -1.0, -0.01),
        ("flat(0°F)", -0.01,  0.01),
        ("0–+1°F",    0.01,  1.0),
        ("> +1°F",    1.0,  99.0),
    ]
    p75_delta = [r for r in p75_recs if r["temp_delta_1h"] is not None]
    print(f"\n  n with delta data: {len(p75_delta)} / {len(p75_recs)}")
    print(f"\n  {'Trend':>10}  {'─── ALL ──────────────────────':>32}"
          f"  {'─── OVERSHOT ─────────────────':>32}"
          f"  {'─── LAGGING ──────────────────':>32}")
    print(f"  {'':>10}{hdr_cols}{hdr_cols}{hdr_cols}")
    print("  " + "-" * 108)
    for lbl, lo_d, hi_d in delta_buckets:
        grp    = [r for r in p75_delta if lo_d <= r["temp_delta_1h"] < hi_d]
        grp_ov = [r for r in grp if _overshoot_flag(r) is True]
        grp_la = [r for r in grp if _overshoot_flag(r) is False]
        row    = "  ".join(_fmt(grid_stats(g, None, None)) for g in [grp, grp_ov, grp_la])
        print(f"  {lbl:>10}  {row}")

    # ── Section 10: Combined factor grid ─────────────────────────────────────
    print()
    print("=" * 70)
    print("  10. COMBINED FILTER GRID AT P75  (hold to settle, priced only)")
    print("=" * 70)

    def _gp(r: dict, lo: float, hi: float = 101) -> bool:
        return r["cents"] is not None and lo <= r["cents"] < hi

    def _gov(r: dict) -> bool:
        return _overshoot_flag(r) is True

    def _glag(r: dict, hi: float) -> bool:
        return r.get("overshoot") is not None and -hi < r["overshoot"] < 0

    def _gnear(r: dict, window: float) -> bool:
        return r.get("overshoot") is not None and r["overshoot"] > -window

    def _grising(r: dict) -> bool:
        return not r["plateau_1h"]

    combos: list[tuple[str, object]] = [
        ("ALL (baseline)",              lambda _: True),
        ("price ≥ 40¢",                lambda r: _gp(r, 40)),
        ("price ≥ 50¢",                lambda r: _gp(r, 50)),
        ("price ≥ 60¢",                lambda r: _gp(r, 60)),
        ("price 40–70¢",               lambda r: _gp(r, 40, 70)),
        ("overshot",                   lambda r: _gov(r)),
        ("overshot + price ≥ 40¢",     lambda r: _gov(r) and _gp(r, 40)),
        ("overshot + price ≥ 55¢",     lambda r: _gov(r) and _gp(r, 55)),
        ("overshot + rising",          lambda r: _gov(r) and _grising(r)),
        ("overshot + rising + p≥40",   lambda r: _gov(r) and _grising(r) and _gp(r, 40)),
        ("lag < 0.5°F",                lambda r: _glag(r, 0.5)),
        ("lag < 1°F",                  lambda r: _glag(r, 1.0)),
        ("lag < 1°F + price ≥ 40¢",    lambda r: _glag(r, 1.0) and _gp(r, 40)),
        ("near GFS (|ov| < 1°F)",      lambda r: _gnear(r, 1.0)),
        ("near GFS + price ≥ 40¢",     lambda r: _gnear(r, 1.0) and _gp(r, 40)),
        ("near GFS + price ≥ 40¢ + rising", lambda r: _gnear(r, 1.0) and _gp(r, 40) and _grising(r)),
        ("rising + price ≥ 40¢",       lambda r: _grising(r) and _gp(r, 40)),
        ("price ≥ 40¢ + delta > 0",    lambda r: _gp(r, 40) and r.get("temp_delta_1h") is not None and r["temp_delta_1h"] > 0),
        ("price ≥ 40¢ + delta ≤ 0",    lambda r: _gp(r, 40) and r.get("temp_delta_1h") is not None and r["temp_delta_1h"] <= 0),
    ]

    print(f"\n  {'Filter':>38}  {'n(all)':>6} {'n(prc)':>6} {'WR':>6} {'avg¢':>8} {'$total':>9}")
    print("  " + "-" * 82)
    for lbl, gate in combos:
        grp   = [r for r in p75_recs if gate(r)]
        s     = grid_stats(grp, None, None)
        n_all = len(grp)
        if s["n"] == 0:
            print(f"  {lbl:>38}  {n_all:>6} {'—':>6} {'—':>6} {'—':>8} {'—':>9}")
        else:
            print(f"  {lbl:>38}  {n_all:>6} {s['n']:>6} {100*s['wr']:>5.1f}% "
                  f"{s['avg']:>+7.1f}¢ ${s['total']/100:>+8.2f}")

    # ── Section 11: PT/SL for filtered groups ────────────────────────────────
    print()
    print("=" * 70)
    print("  11. PT/SL GRID FOR KEY FILTERED GROUPS  (P75 entry)")
    print("=" * 70)

    top_filters = [
        ("price ≥ 40¢",
         [r for r in p75_recs if _gp(r, 40)]),
        ("price ≥ 40¢ + overshot",
         [r for r in p75_recs if _gp(r, 40) and _gov(r)]),
        ("price ≥ 40¢ + near GFS (|ov|<1°F)",
         [r for r in p75_recs if _gp(r, 40) and _gnear(r, 1.0)]),
        ("overshot + rising",
         [r for r in p75_recs if _gov(r) and _grising(r)]),
        ("price ≥ 40¢ + rising",
         [r for r in p75_recs if _gp(r, 40) and _grising(r)]),
    ]

    for filter_name, filter_grp in top_filters:
        n_priced = sum(1 for r in filter_grp if r["cents"] is not None)
        print(f"\n  ── {filter_name}  (priced n={n_priced}) ──")
        header = "  ".join(f"{sl_label[sl]:>36}" for sl in SL_FRACS)
        cols   = "  ".join(f"{'n':>5} {'WR':>6} {'avg¢':>7} {'$PnL':>8}" for _ in SL_FRACS)
        print(f"  {'':14}  {header}")
        print(f"  {'':14}  {cols}")
        print("  " + "-" * (14 + 2 + 36 * len(SL_FRACS) + 2 * (len(SL_FRACS) - 1)))
        for pt in PT_VALS:
            row_parts = [_fmt(grid_stats(filter_grp, pt, sl)) for sl in SL_FRACS]
            print(f"  {pt_label[pt]:14}  " + "  ".join(row_parts))

    # ── Section 12: Overshoot × trend cross-tab ──────────────────────────────
    print()
    print("=" * 70)
    print("  12. OVERSHOOT × TREND CROSS-TAB AT P75  (sizer calibration)")
    print("      Rows = overshoot bucket; Cols = trend state")
    print("      Rising = not plateau_1h; Plateaued = plateau_1h at entry")
    print("=" * 70)

    # Sizer-aligned overshoot buckets (matching band_arb_sizer.py)
    sizer_buckets = [
        ("skip   (<-1.5°F)",  -99.0, -1.5),
        ("cautious(-1.5–0°F)", -1.5,  0.0),
        ("close   (0–+1°F)",    0.0,  1.0),
        ("overshot(>+1°F)",     1.0, 99.0),
    ]

    def _trend_lbl(r: dict) -> str:
        if r.get("temp_delta_1h") is None:
            return "unknown"
        return "rising" if r["temp_delta_1h"] > 0 else "plateaued"

    hdr12 = f"  {'n':>5} {'WR':>6} {'avg¢':>8} {'$total':>9}"

    print(f"\n  {'Bucket':>22}  {'── RISING ───────────────────────':>35}"
          f"  {'── PLATEAUED ────────────────────':>35}"
          f"  {'── ALL (with delta) ─────────────':>35}")
    print(f"  {'':>22}{hdr12}{hdr12}{hdr12}")
    print("  " + "-" * 115)

    for lbl, lo_v, hi_v in sizer_buckets:
        grp_ov = [r for r in p75_recs
                  if r.get("overshoot") is not None and lo_v <= r["overshoot"] < hi_v]
        rising    = [r for r in grp_ov if _trend_lbl(r) == "rising"]
        plateaued = [r for r in grp_ov if _trend_lbl(r) == "plateaued"]
        all_delta = [r for r in grp_ov if _trend_lbl(r) != "unknown"]
        row = "  ".join(_fmt(grid_stats(g, None, None)) for g in [rising, plateaued, all_delta])
        print(f"  {lbl:>22}  {row}")

    # No-GFS row (trend-only fallback)
    print("  " + "-" * 115)
    grp_nogfs  = [r for r in p75_recs if r.get("overshoot") is None]
    rising_ng  = [r for r in grp_nogfs if _trend_lbl(r) == "rising"]
    plat_ng    = [r for r in grp_nogfs if _trend_lbl(r) == "plateaued"]
    all_ng     = [r for r in grp_nogfs if _trend_lbl(r) != "unknown"]
    row_ng = "  ".join(_fmt(grid_stats(g, None, None)) for g in [rising_ng, plat_ng, all_ng])
    print(f"  {'no_gfs':>22}  {row_ng}")

    # Summary note: expected sizer table values
    print()
    print("  ── Implied sizer table (Laplace-smoothed win probs) ──")
    print(f"  {'Bucket':>22}  {'rising p(win)':>14}  {'plateaued p(win)':>16}  {'cells n (R / P)':>16}")
    print("  " + "-" * 76)
    for lbl, lo_v, hi_v in sizer_buckets:
        grp_ov = [r for r in p75_recs
                  if r.get("overshoot") is not None and lo_v <= r["overshoot"] < hi_v]
        rising    = [r for r in grp_ov if _trend_lbl(r) == "rising"]
        plateaued = [r for r in grp_ov if _trend_lbl(r) == "plateaued"]

        def _laplace(grp_: list[dict]) -> str:
            n_ = sum(1 for r in grp_ if r["cents"] is not None)
            if n_ == 0:
                return "  —"
            wins_ = sum(1 for r in grp_ if r["result"] == "yes" and r["cents"] is not None)
            p_ = (wins_ + 1) / (n_ + 2)
            return f"{p_:>5.1%}"

        sr = grid_stats(rising, None, None)
        sp = grid_stats(plateaued, None, None)
        print(f"  {lbl:>22}  {_laplace(rising):>14}  {_laplace(plateaued):>16}  "
              f"{sr['n']:>6} / {sp['n']:<6}")

    # No-GFS fallback
    rising_ng_p  = _laplace(rising_ng)
    plat_ng_p    = _laplace(plat_ng)
    sr_ng = grid_stats(rising_ng, None, None)
    sp_ng = grid_stats(plat_ng, None, None)
    print(f"  {'no_gfs':>22}  {rising_ng_p:>14}  {plat_ng_p:>16}  "
          f"{sr_ng['n']:>6} / {sp_ng['n']:<6}")

    print()


if __name__ == "__main__":
    asyncio.run(main())
