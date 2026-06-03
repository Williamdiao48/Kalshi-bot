"""
Build forecast_no training data using ACTUAL Kalshi B-band markets.

Why this is better than the old script:
  OLD: band_ceil = round(running_obs) - 1 → margin always ~1°F, model never sees 5-10°F margins
  NEW: band_ceil from real Kalshi floor/cap_strike → realistic 1-15°F margins match live trading

Data sources (all from existing cache, no new API calls except Kalshi market list):
  - Kalshi API: all settled B-band KXHIGH/KXLOWT markets → real band positions + outcomes
  - band_arb_hist_cache.json: IEM hourly METAR, HRRR daily, OM daily, HRRR/GFS hourly fc

Output: data/backtest/forecast_no_training_data_kalshi.csv  (same columns as existing CSV)
"""

import asyncio, aiohttp, csv, json, re, sys, os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kalshi_bot.auth import generate_headers
from kalshi_bot.markets import KALSHI_API_BASE

CACHE_FILE   = Path("data/backtest/band_arb_hist_cache.json")
OUTPUT_CSV   = Path("data/backtest/forecast_no_training_data_kalshi.csv")
KALSHI_CACHE = Path("data/backtest/kalshi_markets_cache.json")
MIN_MARGIN   = 0.5
HOURS        = list(range(4, 23))

# series → (iem_station, metric_base, is_high)
SERIES_MAP = {
    "KXHIGHLAX":   ("LAX", "lax",  True),
    "KXHIGHDEN":   ("DEN", "den",  True),
    "KXHIGHCHI":   ("MDW", "chi",  True),
    "KXHIGHNY":    ("NYC", "ny",   True),
    "KXHIGHMIA":   ("MIA", "mia",  True),
    "KXHIGHDAL":   ("DAL", "dal",  True),
    "KXHIGHBOS":   ("BOS", "bos",  True),
    "KXHIGHAUS":   ("AUS", "aus",  True),
    "KXHIGHOU":    ("HOU", "hou",  True),
    "KXHIGHTSFO":  ("SFO", "sfo",  True),
    "KXHIGHTSEA":  ("SEA", "sea",  True),
    "KXHIGHTBOS":  ("BOS", "bos",  True),
    "KXHIGHTPHX":  ("PHX", "phx",  True),
    "KXHIGHTPHIL": ("PHL", "phl",  True),
    "KXHIGHTDC":   ("DCA", "dca",  True),
    "KXHIGHTLV":   ("LAS", "las",  True),
    "KXHIGHTOKC":  ("OKC", "okc",  True),
    "KXHIGHTDAL":  ("DFW", "dfw",  True),
    "KXHIGHTHOU":  ("HOU", "hou",  True),
    "KXHIGHTNOLA": ("MSY", "msy",  True),
    "KXHIGHTATL":  ("ATL", "atl",  True),
    "KXHIGHTMIN":  ("MSP", "msp",  True),
    "KXHIGHTDFW":  ("DFW", "dfw",  True),
    "KXHIGHTSATX": ("SAT", "sat",  True),
    "KXLOWTLAX":   ("LAX", "lax",  False),
    "KXLOWTDEN":   ("DEN", "den",  False),
    "KXLOWTCHI":   ("MDW", "chi",  False),
    "KXLOWTNYC":   ("NYC", "ny",   False),
    "KXLOWTMIA":   ("MIA", "mia",  False),
    "KXLOWTAUS":   ("AUS", "aus",  False),
    "KXLOWTBOS":   ("BOS", "bos",  False),
    "KXLOWTHOU":   ("HOU", "hou",  False),
    "KXLOWTDFW":   ("DFW", "dfw",  False),
    "KXLOWTSFO":   ("SFO", "sfo",  False),
    "KXLOWTSEA":   ("SEA", "sea",  False),
    "KXLOWTPHX":   ("PHX", "phx",  False),
    "KXLOWTPHIL":  ("PHL", "phl",  False),
    "KXLOWTATL":   ("ATL", "atl",  False),
    "KXLOWTMIN":   ("MSP", "msp",  False),
    "KXLOWTDC":    ("DCA", "dca",  False),
    "KXLOWTLV":    ("LAS", "las",  False),
    "KXLOWTOKC":   ("OKC", "okc",  False),
    "KXLOWTSATX":  ("SAT", "sat",  False),
    "KXLOWTNOLA":  ("MSY", "msy",  False),
}

CSV_FIELDS = [
    "metric", "date", "hour_utc", "is_high", "month", "city",
    "running_obs", "band_ceil", "margin_f",
    "delta_1h", "delta_2h", "hours_above_ceil", "hours_to_close",
    "obs_vs_hrrr_h", "obs_vs_gfs_h",
    "actual_f", "hrrr_vs_ceil", "gfs_vs_ceil",
    "consensus_vs_ceil", "model_spread", "n_models_above_ceil",
    "recent_hrrr_mae_7d",
    "clim_prob_exceed",   # P(further_drop > margin_f | city, month, hour) from 4yr METAR history
    "clim_drop_p50",      # median expected additional cooling for this city/month/hour
    "clim_drop_p75",      # 75th pct additional cooling
    "won",
]


def parse_ticker_date(ticker: str) -> str | None:
    m = re.match(r"[A-Z]+-(\d{2})([A-Z]{3})(\d{2})-[BT]", ticker)
    if not m:
        return None
    try:
        dt = datetime.strptime(f"20{m.group(1)}{m.group(2)}{m.group(3)}", "%Y%b%d")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return None


async def fetch_settled_markets(series_list: list[str]) -> dict[str, list[dict]]:
    """Fetch all settled B-band markets for each series. Returns {series: [market, ...]}."""
    if KALSHI_CACHE.exists():
        print(f"Loading Kalshi market cache from {KALSHI_CACHE}")
        return json.loads(KALSHI_CACHE.read_text())

    result = {}
    async with aiohttp.ClientSession() as session:
        for series in series_list:
            markets = []
            cursor = None
            while True:
                params = {"status": "settled", "limit": 200, "series_ticker": series}
                if cursor:
                    params["cursor"] = cursor
                headers = generate_headers("GET", "/trade-api/v2/markets")
                try:
                    async with session.get(
                        f"{KALSHI_API_BASE}/markets", headers=headers, params=params
                    ) as resp:
                        data = await resp.json()
                except Exception as e:
                    print(f"  {series}: fetch error {e}")
                    break
                batch = data.get("markets", [])
                b_bands = [m for m in batch if "-B" in m.get("ticker", "")]
                markets.extend(b_bands)
                cursor = data.get("cursor")
                if not cursor or not batch:
                    break
            result[series] = markets
            total = len(markets)
            if total:
                dates = sorted(m.get("close_time", "")[:10] for m in markets)
                print(f"  {series}: {total} B-bands  {dates[0]} → {dates[-1]}")
            else:
                print(f"  {series}: 0 B-bands")

    KALSHI_CACHE.write_text(json.dumps(result))
    print(f"Saved Kalshi market cache → {KALSHI_CACHE}")
    return result


def load_cache() -> dict:
    print(f"Loading data cache from {CACHE_FILE} ...")
    with CACHE_FILE.open() as f:
        return json.load(f)


def find_cache_key(cache: dict, prefix: str) -> str | None:
    for k in cache:
        if k.startswith(prefix):
            return k
    return None


def get_hourly_obs(cache: dict, station: str, date: str) -> dict[int, float]:
    """Returns {hour: temp_f} for hourly METAR obs on that date."""
    key = find_cache_key(cache, f"hourly_{station}_")
    if not key or date not in cache[key]:
        return {}
    return {int(h): float(v) for h, v in cache[key][date].items()}


def get_hrrr_daily(cache: dict, metric: str, is_high: bool, date: str) -> float | None:
    direction = "high" if is_high else "low"
    prefix = f"hrrr_temp_{direction}_{metric}_"
    key = find_cache_key(cache, prefix)
    if not key:
        return None
    return cache[key].get(date)


def get_om_models(cache: dict, metric: str, is_high: bool, date: str) -> dict[str, float]:
    """Returns {model_name: forecast_f} for all OM models (gfs_seamless, ecmwf, gem, icon)."""
    direction = "high" if is_high else "low"
    prefix = f"om_temp_{direction}_{metric}_"
    key = find_cache_key(cache, prefix)
    if not key:
        return {}
    result = {}
    for model_name, batch_val in cache[key].items():
        if isinstance(batch_val, dict) and date in batch_val:
            v = batch_val[date]
            if v is not None:
                result[model_name] = float(v)
    return result


def get_hourly_fc(cache: dict, model: str, metric: str, is_high: bool, date: str) -> dict[int, float]:
    direction = "high" if is_high else "low"
    prefix = f"hourly_fc_{model}_temp_{direction}_{metric}_"
    key = find_cache_key(cache, prefix)
    if not key or date not in cache[key]:
        return {}
    return {int(h): float(v) for h, v in cache[key][date].items()}


def get_actual(cache: dict, station: str, is_high: bool, date: str) -> float | None:
    direction = "high" if is_high else "low"
    key = find_cache_key(cache, f"actual_{station}_")
    if not key:
        return None
    # find the right direction key
    dir_key = f"actual_{station}_{key.split('_', 2)[2].rsplit('_', 1)[0]}_{direction}"
    dir_key_full = find_cache_key(cache, f"actual_{station}_" + key.split(f"actual_{station}_", 1)[1].rsplit("_", 1)[0] + f"_{direction}")
    # simpler: just look for actual_{station}_*_{direction}
    for k in cache:
        if k.startswith(f"actual_{station}_") and k.endswith(f"_{direction}"):
            val = cache[k].get(date)
            if val is not None:
                return float(val)
    return None


_STATION_TO_CITY = {
    "LAX":"lax","DEN":"den","MDW":"chi","NYC":"ny","MIA":"mia",
    "AUS":"aus","DAL":"dal","BOS":"bos","HOU":"hou","DFW":"dfw",
    "SFO":"sfo","SEA":"sea","PHX":"phx","PHL":"phl","ATL":"atl",
    "MSP":"msp","DCA":"dca","LAS":"las","OKC":"okc","SAT":"sat","MSY":"msy",
}


def build_further_drop_climatology(cache: dict) -> dict:
    """
    For each (city, month, hour): list of observed further_drop values.
    further_drop = running_min_at_hour - actual_daily_low  (how much more cooling happened)
    Built from 4 years of IEM hourly METAR data.
    """
    clim: dict[str, dict[int, dict[int, list]]] = {}
    for station, city in _STATION_TO_CITY.items():
        hourly_key = find_cache_key(cache, f"hourly_{station}_")
        low_key    = find_cache_key(cache, f"actual_{station}_")
        # find the _low key specifically
        low_key = next((k for k in cache if k.startswith(f"actual_{station}_") and k.endswith("_low")), None)
        if not hourly_key or not low_key:
            continue
        hourly_data = cache[hourly_key]
        low_data    = cache[low_key]
        clim[city]  = {}
        for date, obs in hourly_data.items():
            if not obs:
                continue
            actual_low = low_data.get(date)
            if actual_low is None:
                continue
            actual_low = float(actual_low)
            month      = int(date[5:7])
            running_min = None
            for h in sorted(int(x) for x in obs):
                t = obs.get(str(h))
                if t is None:
                    continue
                running_min = float(t) if running_min is None else min(running_min, float(t))
                drop = max(0.0, running_min - actual_low)
                clim[city].setdefault(month, {}).setdefault(h, []).append(drop)
    return clim


def clim_stats(clim: dict, city: str, month: int, hour: int, margin_f: float) -> tuple:
    """Return (prob_exceed, p50, p75) from the further-drop climatology."""
    drops = clim.get(city, {}).get(month, {}).get(hour, [])
    if not drops:
        return 0.15, 2.0, 3.0  # sensible defaults
    d   = sorted(drops)
    n   = len(d)
    p50 = d[n // 2]
    p75 = d[int(n * 0.75)]
    prob_exceed = sum(1 for x in d if x > margin_f) / n
    return round(prob_exceed, 4), round(p50, 2), round(p75, 2)


def compute_rolling_hrrr_mae(
    cache: dict, metric: str, is_high: bool, station: str, window: int = 7
) -> dict[str, float]:
    """Pre-compute {date: rolling_mae} — mean |HRRR_forecast - actual| over past `window` days."""
    from datetime import timedelta
    direction = "high" if is_high else "low"

    hrrr_key = find_cache_key(cache, f"hrrr_temp_{direction}_{metric}_")
    if not hrrr_key:
        return {}
    hrrr: dict[str, float] = cache[hrrr_key]

    actual: dict[str, float] = {}
    for k in cache:
        if k.startswith(f"actual_{station}_") and k.endswith(f"_{direction}"):
            actual = cache[k]
            break

    if not hrrr or not actual:
        return {}

    # Per-day absolute error
    day_errors: dict[str, float] = {}
    for d in hrrr:
        hf = hrrr.get(d)
        af = actual.get(d)
        if hf is not None and af is not None:
            try:
                day_errors[d] = abs(float(hf) - float(af))
            except (TypeError, ValueError):
                pass

    # Rolling window (past window days, not including current)
    rolling: dict[str, float] = {}
    for d in day_errors:
        dt = datetime.strptime(d, "%Y-%m-%d")
        past = [
            day_errors[(dt - timedelta(days=i)).strftime("%Y-%m-%d")]
            for i in range(1, window + 1)
            if (dt - timedelta(days=i)).strftime("%Y-%m-%d") in day_errors
        ]
        if past:
            rolling[d] = round(sum(past) / len(past), 2)
    return rolling


def build_rows(
    markets_by_series: dict[str, list[dict]],
    cache: dict,
) -> list[dict]:
    rows = []
    skipped_no_obs = 0
    skipped_no_actual = 0
    _mae_cache: dict[tuple, dict[str, float]] = {}  # (station, city_code, is_high) → rolling MAE
    print("Building further-drop climatology from 4yr METAR data...")
    _clim = build_further_drop_climatology(cache)
    print(f"  Climatology built for {len(_clim)} cities")

    for series, markets in markets_by_series.items():
        if series not in SERIES_MAP:
            continue
        station, city_code, is_high = SERIES_MAP[series]
        metric_full = f"temp_{'high' if is_high else 'low'}_{city_code}"
        fn = max if is_high else min

        mae_key = (station, city_code, is_high)
        if mae_key not in _mae_cache:
            _mae_cache[mae_key] = compute_rolling_hrrr_mae(cache, city_code, is_high, station)
        rolling_mae = _mae_cache[mae_key]

        for mkt in markets:
            ticker  = mkt.get("ticker", "")
            date    = parse_ticker_date(ticker)
            if not date:
                continue
            result  = mkt.get("result")
            if result not in ("yes", "no"):
                continue

            band_lo   = int(mkt.get("floor_strike", 0))
            band_ceil = int(mkt.get("cap_strike", band_lo + 1))
            won       = 1 if result == "no" else 0  # NO wins when temp is OUTSIDE band

            # Load obs data
            hourly_obs = get_hourly_obs(cache, station, date)
            if not hourly_obs:
                skipped_no_obs += 1
                continue

            actual_f = get_actual(cache, station, is_high, date)
            if actual_f is None:
                skipped_no_actual += 1
                continue

            # Forecasts — use all 5 models: HRRR + GFS + ECMWF + GEM + ICON
            hrrr_f    = get_hrrr_daily(cache, city_code, is_high, date)
            om_models = get_om_models(cache, city_code, is_high, date)
            gfs_f     = om_models.get("gfs_seamless")
            fc_vals   = ([hrrr_f] if hrrr_f is not None else []) + list(om_models.values())
            if not fc_vals:
                continue
            consensus  = median(fc_vals)
            spread     = max(fc_vals) - min(fc_vals) if len(fc_vals) > 1 else 0.0
            n_above    = sum(1 for v in fc_vals if v > band_ceil)
            hrrr_vc    = round(hrrr_f - band_ceil, 2) if hrrr_f is not None else 0.0
            gfs_vc     = round(gfs_f  - band_ceil, 2) if gfs_f  is not None else 0.0
            cons_vc    = round(consensus - band_ceil, 2)

            # Hourly forecasts for obs_vs_hrrr_h
            hrrr_hourly = get_hourly_fc(cache, "hrrr", city_code, is_high, date)
            gfs_hourly  = get_hourly_fc(cache, "gfs",  city_code, is_high, date)

            # Build running obs per hour
            running: dict[int, float] = {}
            cur = None
            for h in HOURS:
                obs_h = hourly_obs.get(h)
                if obs_h is not None:
                    cur = fn([cur, obs_h]) if cur is not None else obs_h
                if cur is not None:
                    running[h] = cur

            # Track consecutive hours above ceiling
            hours_above_counter = 0

            for h in HOURS:
                if h not in running:
                    hours_above_counter = 0
                    continue

                running_obs = running[h]
                margin_f    = round(running_obs - band_ceil, 2)

                if margin_f < MIN_MARGIN:
                    hours_above_counter = 0
                    continue

                hours_above_counter += 1

                r_prev1 = running.get(h - 1)
                r_prev2 = running.get(h - 2)
                delta_1h = round(running_obs - r_prev1, 2) if r_prev1 is not None else 0.0
                delta_2h = round(running_obs - r_prev2, 2) if r_prev2 is not None else 0.0

                hrrr_h_fc    = hrrr_hourly.get(h)
                gfs_h_fc     = gfs_hourly.get(h)
                obs_vs_hrrr  = round(running_obs - hrrr_h_fc, 2) if hrrr_h_fc else 0.0
                obs_vs_gfs   = round(running_obs - gfs_h_fc,  2) if gfs_h_fc  else 0.0

                prob_ex, dp50, dp75 = clim_stats(_clim, city_code, int(date[5:7]), h, margin_f)
                rows.append({
                    "metric":           metric_full,
                    "date":             date,
                    "hour_utc":         h,
                    "is_high":          1 if is_high else 0,
                    "month":            int(date[5:7]),
                    "city":             city_code,
                    "running_obs":      round(running_obs, 2),
                    "band_ceil":        band_ceil,
                    "margin_f":         margin_f,
                    "delta_1h":         delta_1h,
                    "delta_2h":         delta_2h,
                    "hours_above_ceil": hours_above_counter,
                    "hours_to_close":   max(0, 22 - h),
                    "obs_vs_hrrr_h":    obs_vs_hrrr,
                    "obs_vs_gfs_h":     obs_vs_gfs,
                    "actual_f":         actual_f,
                    "hrrr_vs_ceil":     hrrr_vc,
                    "gfs_vs_ceil":      gfs_vc,
                    "consensus_vs_ceil": cons_vc,
                    "model_spread":       round(spread, 2),
                    "n_models_above_ceil": n_above,
                    "recent_hrrr_mae_7d":  rolling_mae.get(date, 3.0),
                    "clim_prob_exceed":    prob_ex,
                    "clim_drop_p50":       dp50,
                    "clim_drop_p75":       dp75,
                    "won":               won,
                })

    print(f"Skipped: no_obs={skipped_no_obs}  no_actual={skipped_no_actual}")
    return rows


def main():
    series_list = list(SERIES_MAP.keys())
    print(f"Fetching settled B-band markets for {len(series_list)} series...")
    markets_by_series = asyncio.run(fetch_settled_markets(series_list))

    total_markets = sum(len(v) for v in markets_by_series.values())
    print(f"\nTotal B-band markets fetched: {total_markets:,}")

    cache = load_cache()
    print("Building training rows...")
    rows = build_rows(markets_by_series, cache)

    high_rows = [r for r in rows if r["is_high"] == 1]
    low_rows  = [r for r in rows if r["is_high"] == 0]
    wr_all  = sum(r["won"] for r in rows) / len(rows) if rows else 0
    wr_high = sum(r["won"] for r in high_rows) / len(high_rows) if high_rows else 0
    wr_low  = sum(r["won"] for r in low_rows)  / len(low_rows)  if low_rows  else 0

    print(f"\nRows: {len(rows):,}  (high={len(high_rows):,}  low={len(low_rows):,})")
    print(f"WR: overall={wr_all:.1%}  high={wr_high:.1%}  low={wr_low:.1%}")

    # Margin distribution
    from collections import Counter
    margin_buckets = Counter(int(r["margin_f"]) for r in rows)
    print("\nMargin distribution:")
    for m in sorted(margin_buckets)[:10]:
        n = margin_buckets[m]
        print(f"  margin~{m}°F  n={n:>7,}  ({100*n/len(rows):.1f}%)")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows):,} rows → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
