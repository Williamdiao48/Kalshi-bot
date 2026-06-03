"""
Backtest the forecast_no band model against actual live NO band_arb trades.

For each resolved NO band_arb trade, reconstructs features from:
  - note JSON  (margin_f, band_ceil_f, hrrr_val_f)
  - raw_forecasts (GFS, ECMWF, consensus, spread)
  - metar_obs_log (delta_1h where available, else 0)
  - logged_at (hour_utc, month)

Compares model P(NO wins) vs market-implied P vs actual outcome.
"""

import json, pickle, re, sqlite3, sys, os, warnings
warnings.filterwarnings("ignore")
from datetime import datetime
from pathlib import Path
from statistics import median
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DB         = "data/db/opportunity_log.db"
MODEL_HIGH = "data/models/forecast_no_band_model_high.pkl"
MODEL_LOW  = "data/models/forecast_no_band_model_low.pkl"
MODEL_COMB = "data/models/forecast_no_band_model.pkl"

def _load(path):
    p = Path(path)
    if p.exists():
        pl = pickle.loads(p.read_bytes())
        return pl["lgbm"], pl["isotonic"], pl["features"], pl.get("city_map", {})
    return None, None, None, {}

lgbm_h, iso_h, feat_h, city_map_h = _load("data/models/forecast_no_band_model_high.pkl")
lgbm_l, iso_l, feat_l, city_map_l = _load("data/models/forecast_no_band_model_low.pkl")
lgbm_c, iso_c, feat_c, city_map_c = _load(MODEL_COMB)

# Fall back to combined model if separate ones don't exist yet
if lgbm_h is None: lgbm_h, iso_h, feat_h, city_map_h = lgbm_c, iso_c, feat_c, city_map_c
if lgbm_l is None: lgbm_l, iso_l, feat_l, city_map_l = lgbm_c, iso_c, feat_c, city_map_c

print(f"High model: {'separate' if Path('data/models/forecast_no_band_model_high.pkl').exists() else 'combined fallback'}")
print(f"Low model:  {'separate' if Path('data/models/forecast_no_band_model_low.pkl').exists() else 'combined fallback'}")

conn = sqlite3.connect(DB)

trades = conn.execute("""
    SELECT ticker, limit_price, logged_at, outcome, settled_pnl_cents, count, note
    FROM trades
    WHERE opportunity_kind='band_arb' AND side='no' AND outcome IS NOT NULL
    ORDER BY logged_at
""").fetchall()

def _build_mae_lookups() -> tuple[dict, dict]:
    """Build HRRR and actual lookups from the historical cache (4yr) + DB (recent)."""
    import json as _json

    hrrr_lookup:   dict[tuple, float] = {}
    actual_lookup: dict[tuple, dict]  = {}

    # Load 4-year historical cache
    cache_path = Path("data/backtest/band_arb_hist_cache.json")
    if cache_path.exists():
        cache = _json.loads(cache_path.read_text())
        # station → metrics mapping (temp_high_* → high; station key = LAX, MDW, etc.)
        for key, day_map in cache.items():
            if not isinstance(day_map, dict):
                continue
            if key.startswith("hrrr_temp_high_"):
                city = key.split("hrrr_temp_high_")[1].split("_")[0]
                metric = f"temp_high_{city}"
                for d, v in day_map.items():
                    if v is not None:
                        hrrr_lookup[(metric, d)] = float(v)
            elif key.startswith("hrrr_temp_low_"):
                city = key.split("hrrr_temp_low_")[1].split("_")[0]
                metric = f"temp_low_{city}"
                for d, v in day_map.items():
                    if v is not None:
                        hrrr_lookup[(metric, d)] = float(v)
            elif key.startswith("actual_") and key.endswith("_high"):
                # actual_{STATION}_{dates}_high → use station to get city
                pass  # handled via metar obs below
            elif key.startswith("actual_") and key.endswith("_low"):
                pass

        # Build actual from cache: actual_{STATION}_*_high / _low
        _station_to_city = {
            "LAX":"lax","DEN":"den","MDW":"chi","NYC":"ny","MIA":"mia",
            "AUS":"aus","DAL":"dal","BOS":"bos","HOU":"hou","DFW":"dfw",
            "SFO":"sfo","SEA":"sea","PHX":"phx","PHL":"phl","ATL":"atl",
            "MSP":"msp","DCA":"dca","LAS":"las","OKC":"okc","SAT":"sat","MSY":"msy",
        }
        for station, city in _station_to_city.items():
            for direction in ("high", "low"):
                prefix = f"actual_{station}_"
                for key in cache:
                    if key.startswith(prefix) and key.endswith(f"_{direction}"):
                        obs_metric = f"temp_high_{city}"  # metar always stored as high_*
                        for d, v in cache[key].items():
                            if v is not None:
                                entry = actual_lookup.setdefault((obs_metric, d), {})
                                entry[direction] = float(v)
                        break

    # Supplement with live DB data (more recent)
    hrrr_rows = conn.execute("""
        SELECT metric, date(logged_at), AVG(data_value)
        FROM raw_forecasts WHERE source='hrrr' AND data_value IS NOT NULL
        GROUP BY metric, date(logged_at)
    """).fetchall()
    for m, d, v in hrrr_rows:
        hrrr_lookup[(m, d)] = v

    obs_rows = conn.execute("""
        SELECT metric, date(obs_at), MAX(temp_f), MIN(temp_f)
        FROM metar_obs_log GROUP BY metric, date(obs_at)
    """).fetchall()
    for metric, d, mx, mn in obs_rows:
        entry = actual_lookup.setdefault((metric, d), {})
        entry["high"] = mx
        entry["low"]  = mn

    print(f"MAE lookup: {len(hrrr_lookup):,} HRRR entries, {len(actual_lookup):,} actual entries")
    return hrrr_lookup, actual_lookup

_hrrr_lookup, _actual_lookup = _build_mae_lookups()


def _build_further_drop_climatology() -> dict:
    import json as _json
    cache_path = Path("data/backtest/band_arb_hist_cache.json")
    if not cache_path.exists():
        return {}
    cache = _json.loads(cache_path.read_text())
    STATION_TO_CITY = {
        "LAX":"lax","DEN":"den","MDW":"chi","NYC":"ny","MIA":"mia",
        "AUS":"aus","DAL":"dal","BOS":"bos","HOU":"hou","DFW":"dfw",
        "SFO":"sfo","SEA":"sea","PHX":"phx","PHL":"phl","ATL":"atl",
        "MSP":"msp","DCA":"dca","LAS":"las","OKC":"okc","SAT":"sat","MSY":"msy",
    }
    clim = {}
    for station, city in STATION_TO_CITY.items():
        hourly_key = next((k for k in cache if k.startswith(f"hourly_{station}_")), None)
        low_key    = next((k for k in cache if k.startswith(f"actual_{station}_") and k.endswith("_low")), None)
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
            actual_low  = float(actual_low)
            month       = int(date[5:7])
            running_min = None
            for h in sorted(int(x) for x in obs):
                t = obs.get(str(h))
                if t is None:
                    continue
                running_min = float(t) if running_min is None else min(running_min, float(t))
                drop = max(0.0, running_min - actual_low)
                clim[city].setdefault(month, {}).setdefault(h, []).append(drop)
    print(f"Climatology built for {len(clim)} cities")
    return clim


def get_clim_stats(city: str, month: int, hour: int, margin_f: float) -> tuple:
    drops = _clim.get(city, {}).get(month, {}).get(hour, [])
    if not drops:
        return 0.15, 2.0, 3.0
    d   = sorted(drops)
    n   = len(d)
    p50 = d[n // 2]
    p75 = d[int(n * 0.75)]
    prob_exceed = sum(1 for x in d if x > margin_f) / n
    return round(prob_exceed, 4), round(p50, 2), round(p75, 2)


_clim = _build_further_drop_climatology()

def get_recent_hrrr_mae(metric, date_str, window=7) -> float:
    """Rolling mean |HRRR_forecast - actual_daily| using pre-fetched lookups."""
    from datetime import timedelta
    obs_metric = metric.replace("temp_low_", "temp_high_")
    direction  = "high" if "high" in metric else "low"
    base = datetime.strptime(date_str, "%Y-%m-%d")
    errors = []
    for i in range(1, window + 1):
        d = (base - timedelta(days=i)).strftime("%Y-%m-%d")
        hrrr_f  = _hrrr_lookup.get((metric, d))
        act_day = _actual_lookup.get((obs_metric, d))
        actual_f = act_day.get(direction) if act_day else None
        if hrrr_f is not None and actual_f is not None:
            errors.append(abs(hrrr_f - actual_f))
    return round(sum(errors) / len(errors), 2) if errors else 3.0


def get_forecasts(metric, date_str):
    rows = conn.execute("""
        SELECT source, AVG(data_value)
        FROM raw_forecasts
        WHERE metric=? AND date(logged_at)=?
          AND source IN ('hrrr','open_meteo_gfs','open_meteo_ecmwf',
                         'open_meteo_gem','open_meteo_icon')
          AND data_value IS NOT NULL
        GROUP BY source
    """, (metric, date_str)).fetchall()
    return {s: v for s, v in rows}

def get_obs_features(metric, logged_at_str, band_ceil):
    """Running obs features from start of day: delta_1h, delta_2h, hours_above_ceil."""
    is_high = "high" in metric
    # metar_obs_log always stores temp_high_* regardless of market direction
    obs_metric = metric.replace("temp_low_", "temp_high_")
    day_start = logged_at_str[:10] + "T00:00:00"
    rows = conn.execute("""
        SELECT obs_at, temp_f FROM metar_obs_log
        WHERE metric=?
          AND obs_at <= ?
          AND obs_at >= ?
        ORDER BY obs_at ASC
    """, (obs_metric, logged_at_str, day_start)).fetchall()
    if len(rows) < 2:
        return 0.0, 0.0, 1.0
    fn = max if is_high else min
    by_hour = defaultdict(list)
    for obs_at, tf in rows:
        h = datetime.fromisoformat(obs_at).hour
        by_hour[h].append(tf)
    hours = sorted(by_hour)
    running = {}
    cur = None
    for h in hours:
        cur = fn([cur] + by_hour[h]) if cur is not None else fn(by_hour[h])
        running[h] = cur
    trade_hour = datetime.fromisoformat(logged_at_str).hour
    r_now = running.get(trade_hour, running.get(max(hours)))
    r_1h  = running.get(trade_hour - 1)
    r_2h  = running.get(trade_hour - 2)
    d1 = round(r_now - r_1h, 2) if r_1h is not None else 0.0
    d2 = round(r_now - r_2h, 2) if r_2h is not None else 0.0
    # Count consecutive hours where running obs > band_ceil.
    # Both KXHIGH and summer KXLOWT NO fire when running obs exceeds ceiling
    # (KXHIGH: daily max above band; KXLOWT: overnight low too warm to enter band).
    hours_above = 0
    for h in sorted(hours, reverse=True):
        if h > trade_hour:
            continue
        rv = running.get(h)
        if rv is None:
            break
        if rv > band_ceil:
            hours_above += 1
        else:
            break
    return d1, d2, float(max(1, hours_above))

# Metric lookup from ticker
_SERIES = {
    "KXHIGHLAX":"temp_high_lax","KXHIGHDEN":"temp_high_den","KXHIGHCHI":"temp_high_chi",
    "KXHIGHNY":"temp_high_ny","KXHIGHMIA":"temp_high_mia","KXHIGHDAL":"temp_high_dal",
    "KXHIGHBOS":"temp_high_bos","KXHIGHAUS":"temp_high_aus","KXHIGHOU":"temp_high_hou",
    "KXHIGHTSFO":"temp_high_sfo","KXHIGHTSEA":"temp_high_sea","KXHIGHTBOS":"temp_high_bos",
    "KXHIGHTPHX":"temp_high_phx","KXHIGHPHIL":"temp_high_phl","KXHIGHTDC":"temp_high_dca",
    "KXHIGHTLV":"temp_high_las","KXHIGHTOKC":"temp_high_okc","KXHIGHTDAL":"temp_high_dfw",
    "KXHIGHTSATX":"temp_high_sat","KXHIGHTHOU":"temp_high_hou","KXHIGHTNOLA":"temp_high_msy",
    "KXHIGHTATL":"temp_high_atl","KXHIGHTMIN":"temp_high_msp","KXHIGHTDFW":"temp_high_dfw",
    "KXLOWTLAX":"temp_low_lax","KXLOWTDEN":"temp_low_den","KXLOWTCHI":"temp_low_chi",
    "KXLOWTNYC":"temp_low_ny","KXLOWTMIA":"temp_low_mia","KXLOWTAUS":"temp_low_aus",
    "KXLOWTBOS":"temp_low_bos","KXLOWTHOU":"temp_low_hou","KXLOWTDFW":"temp_low_dfw",
    "KXLOWTSFO":"temp_low_sfo","KXLOWTSEA":"temp_low_sea","KXLOWTPHX":"temp_low_phx",
    "KXLOWTPHIL":"temp_low_phl","KXLOWTATL":"temp_low_atl","KXLOWTMIN":"temp_low_msp",
    "KXLOWTDC":"temp_low_dca","KXLOWTLV":"temp_low_las","KXLOWTOKC":"temp_low_okc",
    "KXLOWTSATX":"temp_low_sat","KXLOWTNOLA":"temp_low_msy",
}

import numpy as np

print(f"{'Ticker':<35} {'NO¢':>4} {'MktP':>5} {'ModP':>5} {'Edge':>5} {'Out':>5} {'PnL':>7}  margin  hrrr_vc  hour")
print("-" * 110)

results, skipped = [], 0

for ticker, limit_price, logged_at, outcome, pnl_cents, count, note_str in trades:
    # Parse ticker
    m = re.match(r"^([A-Z]+)-\d{2}[A-Z]{3}\d{2}-B(\d+\.?\d*)$", ticker)
    if not m: skipped += 1; continue
    series, mid = m.group(1), float(m.group(2))
    metric = _SERIES.get(series)
    if not metric: skipped += 1; continue

    band_lo   = int(mid - 0.5)
    band_ceil = band_lo + 1
    is_high   = 1.0 if "high" in metric else 0.0

    dt = datetime.fromisoformat(logged_at)
    date_str   = dt.date().isoformat()
    hour_utc   = dt.hour
    month      = dt.month

    # Extract margin_f and hrrr from note JSON
    note = {}
    try:
        if note_str: note = json.loads(note_str)
    except Exception: pass

    margin_f = note.get("margin_f", 1.0)
    note_hrrr = note.get("hrrr_val_f")
    # Use note band_ceil if available (more accurate)
    if "band_ceil_f" in note:
        band_ceil = int(note["band_ceil_f"])

    # Forecasts from raw_forecasts
    fc = get_forecasts(metric, date_str)
    if not fc: skipped += 1; continue

    vals = list(fc.values())
    if len(vals) < 2: skipped += 1; continue

    consensus = median(vals)
    spread    = max(vals) - min(vals)
    hrrr_f    = note_hrrr or fc.get("hrrr", consensus)
    gfs_f     = fc.get("open_meteo_gfs", consensus)
    ecmwf_f   = fc.get("open_meteo_ecmwf", consensus)
    n_above   = sum(1 for v in vals if v > band_ceil)

    delta_1h, delta_2h, hours_above_ceil = get_obs_features(metric, logged_at, band_ceil)
    recent_mae = get_recent_hrrr_mae(metric, date_str)
    city_for_clim = metric.split("_")[-1]
    clim_prob, clim_p50, clim_p75 = get_clim_stats(city_for_clim, month, hour_utc, float(margin_f))

    # Select model and feature list based on market type
    if is_high:
        _lgbm, _iso, _feat, _cmap = lgbm_h, iso_h, feat_h, city_map_h
    else:
        _lgbm, _iso, _feat, _cmap = lgbm_l, iso_l, feat_l, city_map_l

    city     = metric.split("_")[-1]
    city_enc = float(_cmap.get(city, 0))

    feat_map = {
        "margin_f":           float(margin_f),
        "delta_1h":           delta_1h,
        "delta_2h":           delta_2h,
        "hours_above_ceil":   hours_above_ceil,
        "hour_utc":           float(hour_utc),
        "hours_to_close":     float(max(0, 22 - hour_utc)),
        "obs_vs_hrrr_h":      0.0,   # not reconstructable from live DB
        "obs_vs_gfs_h":       0.0,
        "hrrr_vs_ceil":       hrrr_f  - band_ceil,
        "gfs_vs_ceil":        gfs_f   - band_ceil,
        "ecmwf_vs_ceil":      ecmwf_f - band_ceil,
        "consensus_vs_ceil":  consensus - band_ceil,
        "model_spread":       spread,
        "n_models_above_ceil":float(n_above),
        "recent_hrrr_mae_7d": recent_mae,
        "clim_prob_exceed":   clim_prob,
        "clim_drop_p50":      clim_p50,
        "clim_drop_p75":      clim_p75,
        "city_enc":           city_enc,
        "is_high":            is_high,
        "month":              float(month),
    }
    # Build feature vector matching this model's feature list
    X = np.array([[feat_map.get(f, 0.0) for f in _feat]])
    raw_p   = _lgbm.predict_proba(X)[0][1]
    model_p = float(_iso.predict([raw_p])[0])

    # Market implied P(NO wins) = 1 - yes_price/100 = (100 - limit_price)/100
    # limit_price is the YES price in cents
    mkt_no_p = (100 - limit_price) / 100.0
    edge     = model_p - mkt_no_p
    won      = outcome == "won"
    pnl      = (pnl_cents or 0) / 100

    hvc = round(hrrr_f - band_ceil, 1)
    print(f"{ticker:<35} {100-limit_price:>3}¢ {mkt_no_p:>4.0%} {model_p:>4.0%} {edge:>+4.0%} "
          f"{'WIN' if won else 'LOSS':>5} {pnl:>+6.2f}$  m={margin_f:.1f}  hvc={hvc:+.1f}  clim={clim_prob:.0%}  h={hour_utc}")
    results.append({"model_p": model_p, "mkt_p": mkt_no_p, "edge": edge,
                    "won": won, "pnl": pnl, "is_high": is_high})

conn.close()

n    = len(results)
wins = sum(1 for r in results if r["won"])
print(f"\n{'='*60}")
print(f"Trades: {n}  Skipped: {skipped}")
print(f"Actual WR:         {100*wins/n:.1f}%")
print(f"Avg model P(NO):   {100*sum(r['model_p'] for r in results)/n:.1f}%")
print(f"Avg market P(NO):  {100*sum(r['mkt_p']   for r in results)/n:.1f}%")
print(f"Avg edge:          {100*sum(r['edge']     for r in results)/n:+.1f}%")
print(f"Total PnL:         ${sum(r['pnl'] for r in results):+.2f}")

# Model P buckets
print(f"\nModel P bucket → actual WR:")
buckets = defaultdict(list)
for r in results:
    b = int(r["model_p"] * 10) / 10
    buckets[b].append(r["won"])
for b in sorted(buckets):
    grp = buckets[b]
    wr  = sum(grp)/len(grp)
    bar = "█"*sum(grp) + "░"*(len(grp)-sum(grp))
    print(f"  {b:.0%}–{b+.1:.0%}  n={len(grp):>3}  WR={100*wr:>5.1f}%  {bar}")

# Edge filter: what if we only traded when model edge > 0?
pos_edge = [r for r in results if r["edge"] > 0]
if pos_edge:
    pw = sum(1 for r in pos_edge if r["won"])
    print(f"\nIf only traded when model edge > 0%:")
    print(f"  n={len(pos_edge)}  WR={100*pw/len(pos_edge):.1f}%  "
          f"PnL=${sum(r['pnl'] for r in pos_edge):+.2f}")

neg_edge = [r for r in results if r["edge"] <= 0]
if neg_edge:
    nw = sum(1 for r in neg_edge if r["won"])
    print(f"If only traded when model edge ≤ 0% (model disagrees):")
    print(f"  n={len(neg_edge)}  WR={100*nw/len(neg_edge):.1f}%  "
          f"PnL=${sum(r['pnl'] for r in neg_edge):+.2f}")
