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

BAND_START    = "2026-02-01"
BAND_END      = "2026-05-21"
OMFC_CSV      = DATA_DIR / "openmeteo_forecasts.csv"  # gfs_hrrr daily max forecasts

_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_parser.add_argument("--refresh", action="store_true", help="Re-fetch METAR from IEM (ignores cache)")
_parser.add_argument("--require-price", action="store_true",
                     help="Only include observations that have candle price data")
_parser.add_argument("--out", default="data/backtest_band_arb_yes.txt",
                     help="Write full output to this file (default: data/backtest_band_arb_yes.txt)")
_args = _parser.parse_args()

# Tee stdout to file throughout the run
import io as _io
class _Tee:
    def __init__(self, *streams): self._streams = streams
    def write(self, s):
        for st in self._streams: st.write(s)
    def flush(self):
        for st in self._streams: st.flush()

_out_path = Path(_args.out)
_out_path.parent.mkdir(parents=True, exist_ok=True)
_out_file = open(_out_path, "w")
sys.stdout = _Tee(sys.__stdout__, _out_file)

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
    city_tz: ZoneInfo | None = None,
) -> dict[str, list[tuple[datetime, float]]]:
    """Returns {LOCAL_date_str: [(utc_dt, temp_f), ...]} sorted by time.

    Observations are grouped by LOCAL date (not UTC date) so the running max
    correctly reflects midnight-to-midnight local time — matching NWS settlement.
    """
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
            local_dt   = dt.astimezone(city_tz) if city_tz else dt
            local_date = local_dt.date().isoformat()
            result[local_date].append((dt, temp_f))

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


def load_hrrr_forecasts() -> dict[tuple[str, str], float]:
    """Load openmeteo_forecasts.csv (gfs_hrrr model) → {(metric, local_date): forecast_high_f}."""
    data: dict[tuple[str, str], float] = {}
    if not OMFC_CSV.exists():
        return data
    with OMFC_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("model") != "gfs_hrrr":
                continue
            data[(row["city_metric"], row["date"])] = float(row["forecast_high_f"])
    return data


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


def load_bands() -> tuple[
    dict[tuple[str, str], list[dict]],
    dict[tuple[str, str], dict],
    dict[tuple[str, str], dict],
]:
    """Return (between_index, t_upper, t_lower).

    between_index: (metric, local_date) → list of B-band rows
    t_upper:       (metric, local_date) → single "over" T-band row
    t_lower:       (metric, local_date) → single "under" T-band row
    """
    between_index: dict[tuple[str, str], list[dict]] = defaultdict(list)
    t_upper: dict[tuple[str, str], dict] = {}
    t_lower: dict[tuple[str, str], dict] = {}

    with BANDS_CSV.open() as f:
        for row in csv.DictReader(f):
            direction = row.get("direction", "between")
            utc_close  = date.fromisoformat(row["date"])
            local_date = (utc_close - timedelta(days=1)).isoformat()
            key = (row["metric"], local_date)

            if direction == "between":
                if not row.get("strike_lo") or not row.get("strike_hi"):
                    continue
                row["strike_lo"] = float(row["strike_lo"])
                row["strike_hi"] = float(row["strike_hi"])
                between_index[key].append(row)
            elif direction == "over":
                if not row.get("strike_lo"):
                    continue
                row["strike_lo"] = float(row["strike_lo"])
                row["strike_hi"] = None
                t_upper[key] = row
            elif direction == "under":
                if not row.get("strike_hi"):
                    continue
                row["strike_lo"] = None
                row["strike_hi"] = float(row["strike_hi"])
                t_lower[key] = row

    return dict(between_index), t_upper, t_lower


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
    bands_index, t_upper, t_lower = load_bands()
    candles       = load_candles()
    hrrr_forecast = load_hrrr_forecasts()
    n_between = sum(len(v) for v in bands_index.values())
    print(f"  {n_between:,} between-band rows")
    print(f"  {len(t_upper):,} upper T-band rows  {len(t_lower):,} lower T-band rows")
    print(f"  {len(candles):,} tickers with candle data")
    print(f"  {len(hrrr_forecast):,} HRRR city-day forecasts")

    # Flat lookup: (metric, local_date, rounded_temp) → B-band row
    # Index by BOTH strike_lo and strike_hi so entries in the upper half of a band
    # (where round(rmax) == strike_hi) are captured.  Bands skip values
    # (e.g. 66–67, 68–69) so strike_hi of one band never collides with
    # strike_lo of another.
    bands_flat: dict[tuple[str, str, int], dict] = {}
    for (metric, local_date), band_list in bands_index.items():
        for band in band_list:
            bands_flat[(metric, local_date, int(band["strike_lo"]))] = band
            bands_flat[(metric, local_date, int(band["strike_hi"]))] = band

    # ── Fetch METAR per city ─────────────────────────────────────────────────
    print(f"\nFetching METAR ({BAND_START} → {BAND_END})…")
    metar_by_metric: dict[str, dict[str, list[tuple[datetime, float]]]] = {}

    async with aiohttp.ClientSession() as session:
        for metric, (station, _) in sorted(HIGH_IEM.items()):
            short    = metric.replace("temp_high_", "")
            city_tz  = CITIES[metric][3] if metric in CITIES else None
            print(f"  {short:<6} ({station}) …", end=" ", flush=True)
            data = await _fetch_metar(session, station, short, city_tz=city_tz)
            metar_by_metric[metric] = data
            await asyncio.sleep(0.5)

    # ── Build records ────────────────────────────────────────────────────────
    # For each (city, date, entry_timing), observe running_max, round to nearest °F.
    # First try B-band (between) flat lookup; fall back to T-bands (over/under).

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

                rounded   = round(rmax)
                band      = bands_flat.get((metric, local_date, rounded))
                direction = "between"

                # T-band fallback when no B-band matches at this temperature
                if band is None:
                    city_key = (metric, local_date)
                    upper = t_upper.get(city_key)
                    lower = t_lower.get(city_key)
                    # Upper T-band: running max has EXCEEDED the floor threshold.
                    # round(rmax) == strike means the temp is in the highest B-band's
                    # territory, not the upper T-band. Need strictly > strike to be in
                    # T-band territory (i.e. round(rmax) >= strike + 1).
                    if upper is not None and rounded > int(upper["strike_lo"]):
                        band      = upper
                        direction = "over"
                    # Lower T-band: running max is still below the ceiling threshold
                    elif lower is not None and rounded < int(lower["strike_hi"]):
                        band      = lower
                        direction = "under"

                if band is None:
                    continue

                lo     = band["strike_lo"]   # None for "under" T-bands
                hi     = band["strike_hi"]   # None for "over" T-bands
                result = band["result"]
                ticker = band["ticker"]
                hourly_asks = candles.get(ticker, {})
                local_hour  = cutoff.astimezone(tz).hour
                cents       = _ask_at_hour(hourly_asks, local_hour)

                # margin_hi: distance from running max to band ceiling (None if no ceiling)
                margin_hi = ((hi + 0.5) - rmax) if hi is not None else None
                overshoot = (rmax - gfs_fcst) if gfs_fcst is not None else None

                rmax_1h_ago   = _running_max_at(obs, cutoff - timedelta(hours=1))
                rmax_2h_ago   = _running_max_at(obs, cutoff - timedelta(hours=2))
                spot_now      = _spot_temp_at(obs, cutoff)
                spot_1h_ago   = _spot_temp_at(obs, cutoff - timedelta(hours=1))
                plateau_1h    = rmax_1h_ago is not None and rmax == rmax_1h_ago
                plateau_2h    = rmax_2h_ago is not None and rmax == rmax_2h_ago
                temp_delta_1h = (spot_now - spot_1h_ago) if spot_now is not None and spot_1h_ago is not None else None

                hrrr_fc = hrrr_forecast.get((metric, local_date))
                # hrrr_vs_ceil: positive = HRRR above band ceiling (overshoot risk).
                # Only meaningful for between and under; None for over (no ceiling).
                hrrr_vs_ceil = (hrrr_fc - hi) if (hrrr_fc is not None and hi is not None) else None

                if _args.require_price and cents is None:
                    continue

                records.append({
                    "metric":        metric,
                    "date":          local_date,
                    "entry_key":     key,
                    "direction":     direction,
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
                    "hrrr_fc":       hrrr_fc,
                    "hrrr_vs_ceil":  hrrr_vs_ceil,
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
    print("  1. WIN RATE AND PNL BY ENTRY TIMING  (hold to settle, B-bands only)")
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
        grp    = [r for r in records if r["entry_key"] == key and r.get("direction", "between") == "between"]
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

    # B-bands only — T-band records have None margin_hi which breaks many sections below.
    # T-band analysis is in its own section at the end.
    p75_recs = [r for r in records if r["entry_key"] == "p75" and r.get("direction", "between") == "between"]
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
    print("  3. PT/SL GRID AT P75  (priced trades only, exit simulated on hourly candles)")
    print("=" * 70)

    PT_VALS  = [75.0, 80.0, 85.0, 88.0, 90.0, 92.0, 95.0, None]
    SL_FRACS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, None]
    pt_label = {75.0: "PT=75¢", 80.0: "PT=80¢", 85.0: "PT=85¢", 88.0: "PT=88¢",
                90.0: "PT=90¢", 92.0: "PT=92¢", 95.0: "PT=95¢", None: "hold  "}
    sl_label = {0.2: "SL=20%", 0.3: "SL=30%", 0.4: "SL=40%", 0.5: "SL=50%",
                0.6: "SL=60%", 0.7: "SL=70%", None: "no-SL"}

    groups_3 = [
        ("ALL",      p75_recs),
        ("TIGHT (<1F margin)",  [r for r in p75_recs if r["margin_hi"] < 1.0]),
        ("WIDE  (>=1F margin)", [r for r in p75_recs if r["margin_hi"] >= 1.0]),
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

        # Best combo for this group
        best_pnl, best_pt, best_sl = float("-inf"), None, None
        for pt in PT_VALS:
            for sl in SL_FRACS:
                s = grid_stats(sub_grp, pt, sl)
                if s["n"] > 0 and s["total"] > best_pnl:
                    best_pnl, best_pt, best_sl = s["total"], pt, sl
        if best_pt is not None or best_sl is not None:
            best_s = grid_stats(sub_grp, best_pt, best_sl)
            pt_str = f"PT={best_pt:.0f}¢" if best_pt else "hold"
            sl_str = f"SL={int(best_sl*100)}%" if best_sl else "no-SL"
            print(f"\n  ★ Best combo: {pt_str} + {sl_str}  →  "
                  f"WR={100*best_s['wr']:.1f}%  avg={best_s['avg']:+.1f}¢  net=${best_s['total']/100:+.2f}")

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

    def _ghrrr_safe(r: dict) -> bool:
        """HRRR forecasts below band ceiling (temp likely done rising)."""
        return r.get("hrrr_vs_ceil") is not None and r["hrrr_vs_ceil"] < 0

    def _ghrrr_risky(r: dict) -> bool:
        """HRRR forecasts at or above band ceiling (overshoot risk)."""
        return r.get("hrrr_vs_ceil") is not None and r["hrrr_vs_ceil"] >= 0

    def _ghrrr_done(r: dict) -> bool:
        """HRRR forecasts ≤ running max (model says temp already peaked)."""
        return r.get("hrrr_fc") is not None and r["hrrr_fc"] <= r["rmax"]

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
        # ── HRRR gate filters ──────────────────────────────────────────────────
        ("HRRR < ceil (safe)",         lambda r: _ghrrr_safe(r)),
        ("HRRR < ceil + p≥40",         lambda r: _ghrrr_safe(r) and _gp(r, 40)),
        ("HRRR < ceil + overshot",     lambda r: _ghrrr_safe(r) and _gov(r)),
        ("HRRR < ceil + ov + p≥40",    lambda r: _ghrrr_safe(r) and _gov(r) and _gp(r, 40)),
        ("HRRR ≥ ceil (risky)",        lambda r: _ghrrr_risky(r)),
        ("HRRR ≤ rmax (peak done)",    lambda r: _ghrrr_done(r)),
        ("HRRR ≤ rmax + p≥40",         lambda r: _ghrrr_done(r) and _gp(r, 40)),
        ("HRRR ≤ rmax + ov + p≥40",    lambda r: _ghrrr_done(r) and _gov(r) and _gp(r, 40)),
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

    # ── Section 13: HRRR forecast gate analysis ──────────────────────────────
    print()
    print("=" * 70)
    print("  13. HRRR FORECAST GATE AT P75  (hold to settle)")
    print("      hrrr_vs_ceil = HRRR_daily_high_forecast − band_ceiling")
    print("      Negative = HRRR says temp peaks below ceiling (safer YES)")
    print("      Positive = HRRR says temp will overshoot ceiling (risky YES)")
    print("=" * 70)

    p75_hrrr = [r for r in p75_recs if r.get("hrrr_vs_ceil") is not None]
    print(f"\n  n with HRRR data: {len(p75_hrrr)} / {len(p75_recs)}")

    hrrr_buckets = [
        ("HRRR ≤ ceil-2",  -99, -2),
        ("HRRR = ceil-1",   -2,  -1),
        ("HRRR = ceil",     -1,   0),
        ("HRRR = ceil+1",    0,   1),
        ("HRRR ≥ ceil+2",    1,  99),
    ]
    print(f"\n  {'HRRR vs ceil':>14}  {'n':>5} {'WR':>6} {'avg¢':>8} {'$total':>9}")
    print("  " + "-" * 50)
    for lbl, lo_h, hi_h in hrrr_buckets:
        grp = [r for r in p75_hrrr if lo_h <= r["hrrr_vs_ceil"] < hi_h]
        print(f"  {lbl:>14}  {_fmt(grid_stats(grp, None, None))}")

    # HRRR ≤ running max (model says temp already peaked)
    hrrr_done = [r for r in p75_hrrr if r["hrrr_fc"] <= r["rmax"]]
    hrrr_still= [r for r in p75_hrrr if r["hrrr_fc"] >  r["rmax"]]
    print(f"\n  {'Group':>22}  {'n':>5} {'WR':>6} {'avg¢':>8} {'$total':>9}")
    print("  " + "-" * 58)
    for lbl, grp in [("HRRR ≤ rmax (peaked)", hrrr_done), ("HRRR > rmax (rising)", hrrr_still)]:
        print(f"  {lbl:>22}  {_fmt(grid_stats(grp, None, None))}")

    # ── Section 14: T-band (terminal band) analysis ───────────────────────────
    print()
    print("=" * 70)
    print("  14. T-BAND (TERMINAL BAND) ANALYSIS  (P75 entry, hold to settle)")
    print("     Upper T-band (over): YES if daily high ≥ threshold — entered when")
    print("     running_max ≥ threshold at P75 (already locked in).")
    print("     Lower T-band (under): YES if daily high < threshold — entered when")
    print("     running_max < threshold at P75 (has not yet hit ceiling).")
    print("=" * 70)

    t_p75 = [r for r in records if r["entry_key"] == "p75" and r.get("direction") in ("over", "under")]
    t_over  = [r for r in t_p75 if r["direction"] == "over"]
    t_under = [r for r in t_p75 if r["direction"] == "under"]

    def _t_fmt(grp: list[dict]) -> str:
        if not grp:
            return f"{'n/a':>5} {'—':>6} {'—':>8} {'—':>9}"
        wins  = sum(1 for r in grp if r["result"] == "yes")
        wr    = wins / len(grp)
        priced = [r for r in grp if r["cents"] is not None]
        if not priced:
            return f"{len(grp):>5} {100*wr:>5.1f}% {'(no $)':>8} {'(no $)':>9}"
        total = sum((100 if r["result"] == "yes" else 0) - r["cents"] for r in priced)
        avg_c = total / len(priced)
        return f"{len(grp):>5} {100*wr:>5.1f}% {avg_c:>+7.1f}¢ ${total/100:>+8.2f}"

    print(f"\n  {'Band type':>16}  {'n':>5} {'WR':>6} {'avg¢':>8} {'$total':>9}")
    print("  " + "-" * 52)
    print(f"  {'upper (over)':>16}  {_t_fmt(t_over)}")
    print(f"  {'lower (under)':>16}  {_t_fmt(t_under)}")
    print(f"  {'combined':>16}  {_t_fmt(t_p75)}")

    # HRRR gate for T-bands
    t_over_safe  = [r for r in t_over  if r.get("hrrr_fc") is not None and r["hrrr_fc"] >= r["band_lo"]]
    t_over_risky = [r for r in t_over  if r.get("hrrr_fc") is not None and r["hrrr_fc"] <  r["band_lo"]]
    t_under_safe = [r for r in t_under if r.get("hrrr_vs_ceil") is not None and r["hrrr_vs_ceil"] < 0]
    t_under_risk = [r for r in t_under if r.get("hrrr_vs_ceil") is not None and r["hrrr_vs_ceil"] >= 0]

    # Price-gated upper T-band: only enter when market hasn't fully priced the win
    print(f"\n  Upper T-band price filter (ask < threshold):")
    print(f"  {'Price gate':>16}  {'n':>5} {'WR':>6} {'avg¢':>8} {'$total':>9}")
    print("  " + "-" * 52)
    for ceil_p in (95, 90, 85, 80, 75, 70):
        grp = [r for r in t_over if r["cents"] is not None and r["cents"] < ceil_p]
        print(f"  {'ask < '+str(ceil_p)+'¢':>16}  {_t_fmt(grp)}")

    print(f"\n  HRRR gate analysis:")
    print(f"  {'Group':>28}  {'n':>5} {'WR':>6} {'avg¢':>8} {'$total':>9}")
    print("  " + "-" * 62)
    print(f"  {'upper: HRRR ≥ threshold (locked)':>28}  {_t_fmt(t_over_safe)}")
    print(f"  {'upper: HRRR < threshold (risky)':>28}  {_t_fmt(t_over_risky)}")
    print(f"  {'lower: HRRR < ceiling  (safe)':>28}  {_t_fmt(t_under_safe)}")
    print(f"  {'lower: HRRR ≥ ceiling  (risky)':>28}  {_t_fmt(t_under_risk)}")

    # GFS lagging analysis for lower T-band
    # overshoot > 0 → running max already exceeded GFS morning forecast (peaked)
    # overshoot < 0 → running max still below GFS forecast (lagging, still rising)
    t_under_ov  = [r for r in t_under if r.get("overshoot") is not None and r["overshoot"] >= 0]
    t_under_lag = [r for r in t_under if r.get("overshoot") is not None and r["overshoot"] < 0]

    print(f"\n  GFS morning forecast vs running max (lower T-band):")
    print(f"  {'Group':>30}  {'n':>5} {'WR':>6} {'avg¢':>8} {'$total':>9}")
    print("  " + "-" * 64)
    print(f"  {'overshot GFS (peaked, safer)':>30}  {_t_fmt(t_under_ov)}")
    print(f"  {'lagging GFS (still rising, risky)':>30}  {_t_fmt(t_under_lag)}")

    # Combined HRRR + GFS gate for lower T-band
    t_under_both_safe = [r for r in t_under_safe if r.get("overshoot") is not None and r["overshoot"] >= 0]
    t_under_hrrr_safe_gfs_lag = [r for r in t_under_safe if r.get("overshoot") is not None and r["overshoot"] < 0]
    print(f"\n  Combined gate (HRRR-safe only):")
    print(f"  {'Group':>36}  {'n':>5} {'WR':>6} {'avg¢':>8} {'$total':>9}")
    print("  " + "-" * 70)
    print(f"  {'HRRR-safe + overshot GFS':>36}  {_t_fmt(t_under_both_safe)}")
    print(f"  {'HRRR-safe + lagging GFS':>36}  {_t_fmt(t_under_hrrr_safe_gfs_lag)}")

    # ── Section 15: Daily P&L chart ───────────────────────────────────────────
    print()
    print("=" * 70)
    print("  15. DAILY P&L  (P75 priced entries, hold to settle, B-bands only)")
    print("=" * 70)

    from collections import OrderedDict
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent.parent))
    from kalshi_bot.band_arb_sizer import win_prob as _sizer_win_prob, kelly_scale as _sizer_kelly_scale

    _BASE_KELLY = 0.75  # matches LOCKED_OBS_KELLY_FRACTION in trade_executor.py

    def _rec_pnl(r: dict) -> float:
        """Hold-to-settle P&L in cents for one flat ($1) contract."""
        return (100 - r["cents"]) if r["result"] == "yes" else -r["cents"]

    def _kelly_frac(r: dict) -> float:
        """Effective Kelly fraction (0–1) for this entry."""
        is_rising = not r.get("plateau_1h", False)
        overshoot = r.get("overshoot")
        p = _sizer_win_prob(overshoot, is_rising)
        if p is None:
            return 0.0
        cents = r["cents"]
        if not cents or cents >= 100:
            return 0.0
        f_raw = p - (1 - p) * (cents / (100 - cents))
        ks = _sizer_kelly_scale(overshoot, is_rising)
        return max(f_raw, 0.0) * _BASE_KELLY * ks

    daily: dict[str, float] = OrderedDict()
    daily_kelly: dict[str, float] = OrderedDict()
    for r in sorted(p75_priced, key=lambda r: r["date"]):
        d = r["date"]
        flat_pnl  = _rec_pnl(r)
        kelly_pnl = flat_pnl * _kelly_frac(r)
        daily[d]       = daily.get(d, 0.0) + flat_pnl
        daily_kelly[d] = daily_kelly.get(d, 0.0) + kelly_pnl

    cum_flat = cum_kelly = 0.0
    print(f"\n  {'Date':>12}  {'Flat P&L':>9}  {'Flat Cum':>9}  {'Kelly P&L':>10}  {'Kelly Cum':>10}  Trades")
    print("  " + "-" * 74)
    for d in daily:
        pnl_f  = daily[d]
        pnl_k  = daily_kelly.get(d, 0.0)
        cum_flat  += pnl_f
        cum_kelly += pnl_k
        n = sum(1 for r in p75_priced if r["date"] == d)
        print(f"  {d}  ${pnl_f/100:>+7.2f}  ${cum_flat/100:>+7.2f}  "
              f"${pnl_k/100:>+8.2f}  ${cum_kelly/100:>+8.2f}  n={n}")
    print(f"\n  Kelly sizing: base={_BASE_KELLY:.2f} × kelly_scale × raw_f*  "
          f"(win_prob from band_arb_sizer lookup table)")

    # Save chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        import numpy as np

        dates       = list(daily.keys())
        pnls        = [v / 100.0 for v in daily.values()]
        pnls_k      = [daily_kelly.get(d, 0.0) / 100.0 for d in dates]
        cumsum      = list(np.cumsum(pnls))
        cumsum_k    = list(np.cumsum(pnls_k))
        x           = range(len(dates))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        fig.suptitle("Band-Arb YES Backtest — Daily P&L (P75 priced, hold to settle)", fontsize=13)

        colors = ["#2ecc71" if p >= 0 else "#e74c3c" for p in pnls]
        ax1.bar(x, pnls, color=colors, width=0.7, label="Flat (1 contract)")
        ax1.plot(x, pnls_k, color="#e67e22", linewidth=1.5, marker=".", markersize=5,
                 label="Kelly-sized", zorder=3)
        ax1.axhline(0, color="black", linewidth=0.8)
        ax1.set_ylabel("Daily P&L ($)")
        ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2f"))
        ax1.set_title("Daily P&L  [bars = flat; orange line = Kelly-sized]")
        ax1.legend(fontsize=8)

        ax2.plot(x, cumsum,   color="#3498db", linewidth=2, marker="o", markersize=4,
                 label=f"Flat  total=${cumsum[-1]:+.2f}" if cumsum else "Flat")
        ax2.plot(x, cumsum_k, color="#e67e22", linewidth=2, marker="s", markersize=4,
                 linestyle="--",
                 label=f"Kelly total=${cumsum_k[-1]:+.2f}" if cumsum_k else "Kelly")
        ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax2.fill_between(x, cumsum, 0,
                         where=[c >= 0 for c in cumsum], alpha=0.12, color="#2ecc71")
        ax2.fill_between(x, cumsum, 0,
                         where=[c < 0 for c in cumsum], alpha=0.12, color="#e74c3c")
        ax2.set_ylabel("Cumulative P&L ($)")
        ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2f"))
        ax2.set_title("Cumulative P&L")
        ax2.legend(fontsize=9)

        step = max(1, len(dates) // 20)
        ax2.set_xticks(list(x)[::step])
        ax2.set_xticklabels(dates[::step], rotation=45, ha="right", fontsize=8)

        plt.tight_layout()
        chart_path = Path(__file__).parent.parent / "data" / "backtest_band_arb_daily_pnl.png"
        plt.savefig(chart_path, dpi=150)
        plt.close()
        print(f"\n  Chart saved → {chart_path}")
    except Exception as e:
        print(f"\n  (Chart generation failed: {e})")

    print()


if __name__ == "__main__":
    asyncio.run(main())
