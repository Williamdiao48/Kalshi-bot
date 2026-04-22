"""Deep optimization of band-arb YES exit parameters using hourly candlestick data.

Builds on backtest_band_arb_yes_exits.py by using intraday HIGH prices (not just
close prices) to more accurately detect when a PT order would have triggered.
Also sweeps PT/SL thresholds jointly, tests trailing-stop vs fixed-PT, and
finds optimal entry filters by combining price + hour.

Reads cached candle data from data/band_arb_candle_cache.json (no new API calls).

Usage:
  venv/bin/python scripts/optimize_band_arb_exits.py
  venv/bin/python scripts/optimize_band_arb_exits.py --min-entry 50
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from kalshi_bot.markets import KALSHI_API_BASE  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR   = Path(__file__).parent.parent / "data"
CACHE_FILE = DATA_DIR / "band_arb_candle_cache.json"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_mesonet() -> dict[tuple[str, str, int], float]:
    path = DATA_DIR / "mesonet_hourly.csv"
    data: dict[tuple[str, str, int], float] = {}
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            data[(r["city_metric"], r["date"], int(r["local_hour"]))] = float(r["running_max_f"])
    return data


def load_bands() -> list[dict]:
    path = DATA_DIR / "kxhigh_bands.csv"
    rows = []
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            r["strike_lo"]  = float(r["strike_lo"])
            r["strike_hi"]  = float(r["strike_hi"])
            r["band_width"] = float(r["band_width"])
            rows.append(r)
    return rows


def load_cache() -> dict[str, list[dict]]:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}


def _city_tz(metric: str):
    from kalshi_bot.news.noaa import CITIES
    info = CITIES.get(metric.replace("temp_low_", "temp_high_"))
    return info[3] if info else timezone.utc


def candles_to_hourly(candles: list[dict], city_tz) -> dict[int, dict]:
    """Convert candlestick list to {local_hour: {close, high, low}} all in cents."""
    result: dict[int, dict] = {}
    for c in candles:
        ask = c.get("yes_ask", {})
        close_str = ask.get("close_dollars")
        high_str  = ask.get("high_dollars")
        low_str   = ask.get("low_dollars")
        if close_str is None:
            continue
        try:
            close_c = round(float(close_str) * 100)
            high_c  = round(float(high_str)  * 100) if high_str else close_c
            low_c   = round(float(low_str)   * 100) if low_str  else close_c
        except (ValueError, TypeError):
            continue
        ts = datetime.fromtimestamp(c["end_period_ts"], tz=timezone.utc)
        local_hour = ts.astimezone(city_tz).hour
        result[local_hour] = {"close": close_c, "high": high_c, "low": low_c}
    return result


# ── Core simulation ───────────────────────────────────────────────────────────

def simulate(
    band: dict,
    entry_hour: int,
    hourly: dict[int, dict],
    pt: float,
    sl: float,
    use_intraday_high: bool = True,
    trailing_stop_pct: float | None = None,
) -> dict | None:
    """Simulate one trade. Returns result dict or None if no entry price."""
    # Find entry price (try adjacent hours)
    entry_data = hourly.get(entry_hour)
    for delta in [0, 1, -1, 2, -2]:
        entry_data = hourly.get(entry_hour + delta)
        if entry_data:
            break
    if not entry_data or entry_data["close"] <= 0:
        return None

    entry_ask = entry_data["close"]
    pt_price  = entry_ask * (1 + pt)
    sl_price  = entry_ask * (1 - sl)

    exit_hour   = None
    exit_price  = None
    exit_reason = None
    peak_ask    = entry_ask

    future_hours = sorted(h for h in hourly if h > entry_hour)
    for hour in future_hours:
        d = hourly[hour]
        hour_high  = d["high"]  if use_intraday_high else d["close"]
        hour_low   = d["low"]   if use_intraday_high else d["close"]
        hour_close = d["close"]

        # Track peak
        if hour_high > peak_ask:
            peak_ask = hour_high

        # Trailing stop: exit if close is trailing_stop_pct below peak
        if trailing_stop_pct is not None:
            trail_floor = peak_ask * (1 - trailing_stop_pct)
            if hour_close <= trail_floor and peak_ask > entry_ask:
                # Only trigger trailing if peak exceeded a minimum gain
                exit_hour   = hour
                exit_price  = hour_close
                exit_reason = "trailing_stop"
                break

        # Fixed PT: check if intraday high reached PT level
        if hour_high >= pt_price:
            exit_hour   = hour
            exit_price  = min(hour_high, 100)  # cap at 100¢
            exit_reason = "profit_take"
            break

        # Stop-loss: check if intraday low fell to SL level
        if hour_low <= sl_price:
            exit_hour   = hour
            exit_price  = max(hour_low, 0)
            exit_reason = "stop_loss"
            break

    if exit_reason is None:
        exit_reason = "settlement"
        exit_price  = 100 if band["result"] == "yes" else 0
        exit_hour   = 99

    pnl = exit_price - entry_ask
    return {
        "ticker":      band["ticker"],
        "metric":      band["metric"],
        "date":        band["date"],
        "result":      band["result"],
        "entry_hour":  entry_hour,
        "entry_ask":   entry_ask,
        "peak_ask":    peak_ask,
        "exit_hour":   exit_hour,
        "exit_price":  exit_price,
        "exit_reason": exit_reason,
        "pnl_cents":   pnl,
        "hours_held":  (exit_hour - entry_hour) if exit_hour != 99 else None,
    }


def run_sim(
    triggered: list[tuple[dict, int]],
    cache: dict,
    pt: float,
    sl: float,
    use_intraday_high: bool = True,
    trailing: float | None = None,
    min_entry: int = 0,
    max_entry: int = 100,
    hour_start: int = 0,
    hour_end: int = 24,
) -> list[dict]:
    results = []
    for band, entry_hour in triggered:
        if entry_hour < hour_start or entry_hour > hour_end:
            continue
        candles = cache.get(band["ticker"], [])
        if not candles:
            continue
        tz = _city_tz(band["metric"])
        hourly = candles_to_hourly(candles, tz)
        res = simulate(band, entry_hour, hourly, pt, sl,
                       use_intraday_high=use_intraday_high,
                       trailing_stop_pct=trailing)
        if res is None:
            continue
        if res["entry_ask"] < min_entry or res["entry_ask"] > max_entry:
            continue
        results.append(res)
    return results


def avg_pnl(sims: list[dict]) -> float:
    if not sims:
        return 0.0
    return sum(s["pnl_cents"] for s in sims) / len(sims)


def stats(sims: list[dict]) -> str:
    if not sims:
        return "N=0"
    n = len(sims)
    pt  = sum(1 for s in sims if s["exit_reason"] == "profit_take")
    sl  = sum(1 for s in sims if s["exit_reason"] == "stop_loss")
    ts  = sum(1 for s in sims if s["exit_reason"] == "trailing_stop")
    se  = sum(1 for s in sims if s["exit_reason"] == "settlement")
    yes = sum(1 for s in sims if s["result"] == "yes")
    pnl = avg_pnl(sims)
    return (f"N={n:3d}  YES={yes/n*100:.0f}%  PT={pt/n*100:.0f}%  "
            f"SL={sl/n*100:.0f}%  TS={ts/n*100:.0f}%  "
            f"Set={se/n*100:.0f}%  P&L={pnl:+.1f}¢")


# ── Reports ───────────────────────────────────────────────────────────────────

def print_all_reports(
    triggered: list[tuple[dict, int]],
    cache: dict,
) -> None:

    # ── 1. Close-price vs intraday-high comparison ──────────────────────────
    print(f"\n{'='*70}")
    print("1. CLOSE PRICE vs INTRADAY HIGH  (PT=20%  SL=50%  all entries)")
    print(f"{'='*70}")
    base_close = run_sim(triggered, cache, pt=0.20, sl=0.50, use_intraday_high=False)
    base_high  = run_sim(triggered, cache, pt=0.20, sl=0.50, use_intraday_high=True)
    print(f"  Close-only:     {stats(base_close)}")
    print(f"  Intraday-high:  {stats(base_high)}")

    # ── 2. PT/SL joint sweep (intraday high, all entries) ──────────────────
    print(f"\n{'='*70}")
    print("2. PT × SL JOINT SWEEP  (intraday high, all entries)")
    print(f"{'='*70}")
    print(f"  {'PT':>5}  {'SL':>5}  {'N':>4}  {'PT%':>5}  {'SL%':>5}  {'P&L':>8}")
    best_pnl = -999; best_pt = best_sl = None
    for pt in [0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        for sl in [0.30, 0.40, 0.50, 0.60, 0.70]:
            sims = run_sim(triggered, cache, pt=pt, sl=sl, use_intraday_high=True)
            n = len(sims)
            if not n:
                continue
            pt_r = sum(1 for s in sims if s["exit_reason"] == "profit_take") / n
            sl_r = sum(1 for s in sims if s["exit_reason"] == "stop_loss") / n
            pnl  = avg_pnl(sims)
            marker = " ←best" if pnl > best_pnl else ""
            if pnl > best_pnl:
                best_pnl = pnl; best_pt = pt; best_sl = sl
            print(f"  PT={pt*100:.0f}%  SL={sl*100:.0f}%  "
                  f"{n:>4}  {pt_r*100:>5.0f}%  {sl_r*100:>5.0f}%  {pnl:>+7.1f}¢{marker}")

    # ── 3. Min-entry sweep with best PT/SL ─────────────────────────────────
    print(f"\n{'='*70}")
    print(f"3. MIN ENTRY PRICE SWEEP  (PT={best_pt*100:.0f}%  SL={best_sl*100:.0f}%  intraday high)")
    print(f"{'='*70}")
    print(f"  {'Min':>5}  {stats(run_sim(triggered, cache, pt=best_pt, sl=best_sl, use_intraday_high=True, min_entry=0))[:6]}...")
    print(f"  {'Min':>5}  {'N':>4}  {'YES%':>5}  {'PT%':>5}  {'SL%':>5}  {'P&L':>8}")
    for min_e in [0, 15, 20, 30, 40, 50, 60]:
        sims = run_sim(triggered, cache, pt=best_pt, sl=best_sl, use_intraday_high=True, min_entry=min_e)
        if not sims:
            continue
        n = len(sims)
        yes = sum(1 for s in sims if s["result"] == "yes") / n
        pt_r = sum(1 for s in sims if s["exit_reason"] == "profit_take") / n
        sl_r = sum(1 for s in sims if s["exit_reason"] == "stop_loss") / n
        pnl  = avg_pnl(sims)
        print(f"  ≥{min_e:>4}¢  {n:>4}  {yes*100:>5.0f}%  {pt_r*100:>5.0f}%  {sl_r*100:>5.0f}%  {pnl:>+7.1f}¢")

    # ── 4. Trailing stop vs fixed PT ────────────────────────────────────────
    print(f"\n{'='*70}")
    print("4. TRAILING STOP vs FIXED PT  (all entries, intraday high)")
    print(f"{'='*70}")
    print(f"  {'Strategy':>30}  {stats(run_sim(triggered, cache, pt=0.20, sl=0.50))[:6]}...")
    print(f"  {'Strategy':>30}  {'N':>4}  {'PT%':>5}  {'SL%':>5}  {'TS%':>5}  {'P&L':>8}")
    # Fixed PT strategies
    for pt_t in [0.15, 0.20, 0.25, 0.30]:
        sims = run_sim(triggered, cache, pt=pt_t, sl=best_sl, use_intraday_high=True)
        n = len(sims); pt_r = sum(1 for s in sims if s["exit_reason"] == "profit_take")/n
        sl_r = sum(1 for s in sims if s["exit_reason"] == "stop_loss")/n
        ts_r = 0
        print(f"  {'Fixed PT='+str(int(pt_t*100))+'%':>30}  {n:>4}  {pt_r*100:>5.0f}%  {sl_r*100:>5.0f}%  {ts_r*100:>5.0f}%  {avg_pnl(sims):>+7.1f}¢")
    # Trailing stop strategies
    for trail in [0.10, 0.15, 0.20, 0.25, 0.30]:
        sims = run_sim(triggered, cache, pt=999, sl=best_sl, trailing=trail, use_intraday_high=True)
        if not sims:
            continue
        n = len(sims); pt_r = sum(1 for s in sims if s["exit_reason"] == "profit_take")/n
        sl_r = sum(1 for s in sims if s["exit_reason"] == "stop_loss")/n
        ts_r = sum(1 for s in sims if s["exit_reason"] == "trailing_stop")/n
        print(f"  {'Trail '+str(int(trail*100))+'% from peak':>30}  {n:>4}  {pt_r*100:>5.0f}%  {sl_r*100:>5.0f}%  {ts_r*100:>5.0f}%  {avg_pnl(sims):>+7.1f}¢")
    # Combo: trailing + fixed PT as fallback
    for pt_t, trail in [(0.20, 0.15), (0.25, 0.15), (0.20, 0.20)]:
        sims = run_sim(triggered, cache, pt=pt_t, sl=best_sl, trailing=trail, use_intraday_high=True)
        if not sims:
            continue
        n = len(sims); pt_r = sum(1 for s in sims if s["exit_reason"] == "profit_take")/n
        sl_r = sum(1 for s in sims if s["exit_reason"] == "stop_loss")/n
        ts_r = sum(1 for s in sims if s["exit_reason"] == "trailing_stop")/n
        print(f"  {'PT='+str(int(pt_t*100))+'%+Trail='+str(int(trail*100))+'%':>30}  {n:>4}  {pt_r*100:>5.0f}%  {sl_r*100:>5.0f}%  {ts_r*100:>5.0f}%  {avg_pnl(sims):>+7.1f}¢")

    # ── 5. Peak analysis — what's the max theoretical P&L? ──────────────────
    print(f"\n{'='*70}")
    print("5. PEAK ASK ANALYSIS  (what was the max YES_ask reached post-entry?)")
    print(f"{'='*70}")
    all_sims = run_sim(triggered, cache, pt=999, sl=0.01, use_intraday_high=True)  # never PT, instant SL if any drop
    # Actually use no PT/SL — just track peak
    all_sims = []
    for band, entry_hour in triggered:
        candles = cache.get(band["ticker"], [])
        if not candles:
            continue
        tz = _city_tz(band["metric"])
        hourly = candles_to_hourly(candles, tz)
        entry_data = hourly.get(entry_hour)
        for delta in [0, 1, -1, 2, -2]:
            entry_data = hourly.get(entry_hour + delta)
            if entry_data:
                break
        if not entry_data or entry_data["close"] <= 0:
            continue
        entry_ask = entry_data["close"]
        future_highs = [hourly[h]["high"] for h in sorted(hourly) if h > entry_hour]
        peak = max(future_highs) if future_highs else entry_ask
        all_sims.append({
            "result":     band["result"],
            "entry_ask":  entry_ask,
            "peak_ask":   peak,
            "peak_gain":  peak - entry_ask,
            "peak_gain_pct": (peak - entry_ask) / entry_ask * 100,
        })

    if all_sims:
        yes_s = [s for s in all_sims if s["result"] == "yes"]
        no_s  = [s for s in all_sims if s["result"] == "no"]
        print(f"  All {len(all_sims)} triggered bands:")
        print(f"    Avg peak gain:  {sum(s['peak_gain'] for s in all_sims)/len(all_sims):+.1f}¢")
        print(f"    Avg peak gain%: {sum(s['peak_gain_pct'] for s in all_sims)/len(all_sims):+.0f}%")
        pct_reached = {t: sum(1 for s in all_sims if s["peak_gain_pct"] >= t)/len(all_sims)
                       for t in [10, 20, 30, 50, 75, 100]}
        print(f"    Peak ≥10%:  {pct_reached[10]*100:.0f}%  of trades")
        print(f"    Peak ≥20%:  {pct_reached[20]*100:.0f}%  of trades")
        print(f"    Peak ≥30%:  {pct_reached[30]*100:.0f}%  of trades")
        print(f"    Peak ≥50%:  {pct_reached[50]*100:.0f}%  of trades")
        print(f"    Peak ≥100%: {pct_reached[100]*100:.0f}%  of trades")
        if yes_s:
            print(f"\n  TRUE YES (N={len(yes_s)}):")
            print(f"    Avg peak gain:  {sum(s['peak_gain'] for s in yes_s)/len(yes_s):+.1f}¢")
            print(f"    % reaching 20%: {sum(1 for s in yes_s if s['peak_gain_pct']>=20)/len(yes_s)*100:.0f}%")
        if no_s:
            print(f"\n  TRUE NO (N={len(no_s)}):")
            print(f"    Avg peak gain:  {sum(s['peak_gain'] for s in no_s)/len(no_s):+.1f}¢")
            print(f"    % reaching 20%: {sum(1 for s in no_s if s['peak_gain_pct']>=20)/len(no_s)*100:.0f}%")

    # ── 6. Best combined filter: hour + min_entry + best PT/SL ──────────────
    print(f"\n{'='*70}")
    print(f"6. COMBINED FILTER SWEEP  (PT={best_pt*100:.0f}%  SL={best_sl*100:.0f}%  intraday high)")
    print(f"{'='*70}")
    print(f"  {'Filter':>30}  {'N':>4}  {'YES%':>5}  {'P&L':>8}")
    combos = [
        ("all",               dict()),
        ("≥10 entry",         dict(min_entry=10)),
        ("≥20 entry",         dict(min_entry=20)),
        ("≥30 entry",         dict(min_entry=30)),
        ("≥50 entry",         dict(min_entry=50)),
        ("hour 10-16",        dict(hour_start=10, hour_end=16)),
        ("≥20 + hour 10-16",  dict(min_entry=20, hour_start=10, hour_end=16)),
        ("≥30 + hour 10-16",  dict(min_entry=30, hour_start=10, hour_end=16)),
        ("≥50 + hour 10-16",  dict(min_entry=50, hour_start=10, hour_end=16)),
        ("≥50 + hour 12-16",  dict(min_entry=50, hour_start=12, hour_end=16)),
    ]
    for label, kw in combos:
        sims = run_sim(triggered, cache, pt=best_pt, sl=best_sl, use_intraday_high=True, **kw)
        if not sims:
            print(f"  {label:>30}  {'N=0':>4}")
            continue
        n = len(sims)
        yes = sum(1 for s in sims if s["result"] == "yes") / n
        pnl = avg_pnl(sims)
        print(f"  {label:>30}  {n:>4}  {yes*100:>5.0f}%  {pnl:>+7.1f}¢")

    # ── 7. By city with best params ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"7. BY CITY  (PT={best_pt*100:.0f}%  SL={best_sl*100:.0f}%  intraday high  all entries)")
    print(f"{'='*70}")
    sims_all = run_sim(triggered, cache, pt=best_pt, sl=best_sl, use_intraday_high=True)
    by_city: dict[str, list] = {}
    for s in sims_all:
        city = s["metric"].replace("temp_high_", "")
        by_city.setdefault(city, []).append(s)
    print(f"  {'City':>6}  {'N':>4}  {'YES%':>5}  {'PT%':>5}  {'SL%':>5}  {'AvgEntry':>9}  {'P&L':>8}")
    for city in sorted(by_city, key=lambda c: -avg_pnl(by_city[c])):
        g = by_city[city]
        n = len(g)
        yes = sum(1 for s in g if s["result"] == "yes") / n
        pt_r = sum(1 for s in g if s["exit_reason"] == "profit_take") / n
        sl_r = sum(1 for s in g if s["exit_reason"] == "stop_loss") / n
        ae   = sum(s["entry_ask"] for s in g) / n
        print(f"  {city:>6}  {n:>4}  {yes*100:>5.0f}%  {pt_r*100:>5.0f}%  "
              f"{sl_r*100:>5.0f}%  {ae:>8.0f}¢  {avg_pnl(g):>+7.1f}¢")

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print(f"{'='*70}")
    print(f"  Best PT threshold: {best_pt*100:.0f}%")
    print(f"  Best SL threshold: {best_sl*100:.0f}%")
    # Find optimal min_entry
    best_entry_pnl = -999; best_min_entry = 0
    for min_e in [0, 10, 15, 20, 25, 30, 40, 50]:
        sims = run_sim(triggered, cache, pt=best_pt, sl=best_sl, use_intraday_high=True, min_entry=min_e)
        if len(sims) >= 10:
            p = avg_pnl(sims)
            if p > best_entry_pnl:
                best_entry_pnl = p; best_min_entry = min_e
    print(f"  Best min entry:    ≥{best_min_entry}¢  (avg P&L: {best_entry_pnl:+.1f}¢/trade)")
    best_sims = run_sim(triggered, cache, pt=best_pt, sl=best_sl,
                        use_intraday_high=True, min_entry=best_min_entry)
    if best_sims:
        print(f"  Best config stats: {stats(best_sims)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    log.info("Loading data...")
    mesonet = load_mesonet()
    bands   = load_bands()
    cache   = load_cache()

    city_filter = set(args.city) if args.city else None

    # Find triggered bands (first hour temp enters band)
    triggered: list[tuple[dict, int]] = []
    for b in bands:
        metric = b["metric"]
        if city_filter:
            suffix = metric.replace("temp_high_", "")
            if suffix not in city_filter:
                continue
        s_lo = b["strike_lo"]
        s_hi = b["strike_hi"]
        for hour in range(args.hour_start, args.hour_end + 1):
            v = mesonet.get((metric, b["date"], hour))
            if v is not None and s_lo <= v <= s_hi:
                triggered.append((b, hour))
                break

    log.info("Triggered bands: %d / %d", len(triggered), len(bands))
    print_all_reports(triggered, cache)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize band-arb YES exit parameters.")
    parser.add_argument("--city", nargs="+", default=None)
    parser.add_argument("--hour-start", type=int, default=6)
    parser.add_argument("--hour-end",   type=int, default=20)
    parser.add_argument("--min-entry",  type=int, default=0)
    args = parser.parse_args()
    main(args)
