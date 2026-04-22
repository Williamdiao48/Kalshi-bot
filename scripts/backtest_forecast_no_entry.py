"""
Backtest: Pre-METAR NO entry — price trajectory analysis.

For every NO-settling band with hourly candle data, this script traces the
NO_ask price from candle open (~12-18h before settlement) through settlement,
and finds:
  - T_metar: first local hour where METAR running_max crosses band ceiling
  - T_entry: first candle where NO_ask <= each entry threshold
  - Whether the entry was pre- or post-METAR confirmation
  - Expected gain (hold to 100¢ settlement)

This answers: "how much alpha is available if we buy NO before METAR confirms?"

Also runs Report 5 on YES-settling bands to measure false-positive loss rate.

Usage:
    venv/bin/python scripts/backtest_forecast_no_entry.py
    venv/bin/python scripts/backtest_forecast_no_entry.py --max-entry 60 --city mia
"""
import argparse
import csv
import json
import datetime
import sys
from collections import defaultdict
from pathlib import Path

# --- paths ---
ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"

BANDS_CSV      = DATA / "kxhigh_bands.csv"
MESONET_CSV    = DATA / "mesonet_hourly.csv"
CANDLE_CACHE   = DATA / "band_arb_candle_cache.json"

ENTRY_THRESHOLDS = [30, 40, 50, 60, 70, 80]


def load_bands() -> list[dict]:
    with open(BANDS_CSV) as f:
        return list(csv.DictReader(f))


def load_mesonet() -> dict[tuple, float]:
    """Returns {(metric, date_str, local_hour_int): running_max_f}"""
    lookup: dict[tuple, float] = {}
    with open(MESONET_CSV) as f:
        for row in csv.DictReader(f):
            key = (row["city_metric"], row["date"], int(row["local_hour"]))
            lookup[key] = float(row["running_max_f"])
    return lookup


def load_candles() -> dict[str, list[dict]]:
    with open(CANDLE_CACHE) as f:
        return json.load(f)


def city_tz(metric: str):
    """Return pytz timezone for a temp_high_* metric."""
    import importlib
    noaa = importlib.import_module("kalshi_bot.news.noaa")
    info = noaa.CITIES.get(metric) or noaa.CITIES.get(metric.replace("temp_low_", "temp_high_"))
    if info is None:
        return None
    return info[3]


def candle_to_local_hour(ts: int, tz) -> tuple[str, int]:
    """Convert end_period_ts to (local_date_str, local_hour)."""
    dt_utc = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
    dt_local = dt_utc.astimezone(tz)
    return dt_local.strftime("%Y-%m-%d"), dt_local.hour


def trading_date_str(settlement_date_str: str) -> str:
    """
    The CSV 'date' column is the settlement/resolution date (e.g. 2026-03-29).
    Price action and METAR observations happen the day before (2026-03-28).
    All candles and mesonet lookups should use trading_date, not settlement_date.
    """
    d = datetime.date.fromisoformat(settlement_date_str)
    return (d - datetime.timedelta(days=1)).isoformat()


def find_metar_cross_hour(metric: str, tdate: str, strike_hi: float,
                          mesonet: dict) -> int | None:
    """First local hour where running_max_f >= strike_hi on the trading date."""
    for hour in range(0, 24):
        val = mesonet.get((metric, tdate, hour))
        if val is not None and val >= strike_hi:
            return hour
    return None


def trace_no_ask(ticker: str, metric: str, tdate: str, candles: list[dict],
                 tz) -> list[tuple[int, int]]:
    """
    Return list of (local_hour, no_ask_cents) for candles on the trading date.
    no_ask = round((1 - yes_bid.close) * 100)
    Settlement-day candles (100¢) are excluded.
    """
    result = []
    for c in candles:
        ts = c["end_period_ts"]
        cdate, chour = candle_to_local_hour(ts, tz)
        if cdate != tdate:
            continue
        yes_bid_close = float(c["yes_bid"]["close_dollars"])
        no_ask = round((1 - yes_bid_close) * 100)
        result.append((chour, no_ask))
    return sorted(result)


def find_entry_hour(hourly_no_ask: list[tuple[int, int]], threshold: int) -> tuple[int, int] | None:
    """First (hour, price) where NO_ask <= threshold."""
    for hour, price in hourly_no_ask:
        if price <= threshold:
            return hour, price
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-entry", type=int, default=None,
                        help="Only report this entry cap (e.g. 60)")
    parser.add_argument("--city", type=str, default=None,
                        help="Filter to one city suffix (e.g. mia)")
    args = parser.parse_args()

    # Add project root to path so kalshi_bot imports work
    sys.path.insert(0, str(ROOT))

    print("Loading data...", flush=True)
    bands    = load_bands()
    mesonet  = load_mesonet()
    candles  = load_candles()

    # Split into NO-settling and YES-settling
    no_bands  = [b for b in bands if b["result"] == "no" and b["ticker"] in candles and candles[b["ticker"]]]
    yes_bands = [b for b in bands if b["result"] == "yes" and b["ticker"] in candles and candles[b["ticker"]]]

    if args.city:
        suffix = args.city.lower()
        no_bands  = [b for b in no_bands  if b["metric"].endswith(suffix)]
        yes_bands = [b for b in yes_bands if b["metric"].endswith(suffix)]

    thresholds = [args.max_entry] if args.max_entry else ENTRY_THRESHOLDS

    # -------------------------------------------------------------------------
    # Report 1 — Entry threshold sweep for NO-settling bands
    # -------------------------------------------------------------------------
    print(f"\n{'='*68}")
    print("  REPORT 1 — Entry threshold sweep (NO-settling bands)")
    print(f"{'='*68}")
    print(f"  Total NO-settling bands with candles: {len(no_bands)}")
    print()

    # Per-threshold accumulators
    stats: dict[int, dict] = {t: {"n_signals": 0, "n_pre_metar": 0,
                                  "entries": [], "gains": [], "holds": []}
                               for t in thresholds}

    # Per-band data for other reports
    band_details = []

    for b in no_bands:
        ticker    = b["ticker"]
        metric    = b["metric"]
        date_str  = b["date"]
        tdate     = trading_date_str(date_str)  # day before settlement
        strike_hi = float(b["strike_hi"])

        tz = city_tz(metric)
        if tz is None:
            continue

        clist = candles[ticker]
        hourly = trace_no_ask(ticker, metric, tdate, clist, tz)
        if not hourly:
            continue

        t_metar = find_metar_cross_hour(metric, tdate, strike_hi, mesonet)
        city = metric.replace("temp_high_", "")

        detail = {
            "ticker": ticker,
            "city": city,
            "date": date_str,
            "strike_hi": strike_hi,
            "t_metar": t_metar,
            "opening_no_ask": hourly[0][1],
            "hourly": hourly,
            "entries": {},
        }

        for thresh in thresholds:
            entry = find_entry_hour(hourly, thresh)
            if entry is None:
                detail["entries"][thresh] = None
                continue
            t_entry, price = entry
            is_pre = t_metar is not None and t_entry < t_metar
            gain = 100 - price
            # hours held: from entry hour to end of settlement day (use last candle)
            last_hour = hourly[-1][0]
            hold_h = max(0, last_hour - t_entry)

            stats[thresh]["n_signals"] += 1
            if is_pre:
                stats[thresh]["n_pre_metar"] += 1
            stats[thresh]["entries"].append(price)
            stats[thresh]["gains"].append(gain)
            stats[thresh]["holds"].append(hold_h)

            detail["entries"][thresh] = {
                "t_entry": t_entry,
                "price": price,
                "gain": gain,
                "hold_h": hold_h,
                "is_pre_metar": is_pre,
            }

        band_details.append(detail)

    print(f"  {'Cap':>6}  {'N_sig':>7}  {'Pre-METAR':>9}  {'Hit%':>5}  "
          f"{'Avg_entry':>9}  {'Avg_gain':>8}  {'Avg_hold':>8}")
    print(f"  {'-'*6}  {'-'*7}  {'-'*9}  {'-'*5}  {'-'*9}  {'-'*8}  {'-'*8}")
    for thresh in thresholds:
        s = stats[thresh]
        n = s["n_signals"]
        if n == 0:
            print(f"  ≤{thresh:>4}¢  {'0':>7}  {'—':>9}  {'—':>5}  {'—':>9}  {'—':>8}  {'—':>8}")
            continue
        avg_entry = sum(s["entries"]) / n
        avg_gain  = sum(s["gains"]) / n
        avg_hold  = sum(s["holds"]) / n
        pre_pct   = 100 * s["n_pre_metar"] / n
        print(f"  ≤{thresh:>4}¢  {n:>7}  {s['n_pre_metar']:>9}  {pre_pct:>4.0f}%  "
              f"  {avg_entry:>7.1f}¢  {avg_gain:>7.1f}¢  {avg_hold:>7.1f}h")

    # -------------------------------------------------------------------------
    # Report 2 — How many NO-settling bands never reach each threshold?
    # -------------------------------------------------------------------------
    print(f"\n{'='*68}")
    print("  REPORT 2 — Bands that never dip below entry cap (always pre-priced)")
    print(f"{'='*68}")
    print()
    total_no = len(band_details)
    print(f"  {'Cap':>6}  {'Never_below':>11}  {'Pct_missed':>10}  {'Need_band_arb':>13}")
    print(f"  {'-'*6}  {'-'*11}  {'-'*10}  {'-'*13}")
    for thresh in thresholds:
        never = sum(1 for d in band_details if d["entries"].get(thresh) is None)
        pct = 100 * never / total_no if total_no else 0
        print(f"  ≤{thresh:>4}¢  {never:>11}  {pct:>9.0f}%  {never:>13}")

    # -------------------------------------------------------------------------
    # Report 3 — City breakdown
    # -------------------------------------------------------------------------
    print(f"\n{'='*68}")
    print("  REPORT 3 — City breakdown (entry cap = 60¢)")
    print(f"{'='*68}")
    print()
    city_data: dict[str, list] = defaultdict(list)
    for d in band_details:
        e = d["entries"].get(60)
        city_data[d["city"]].append(e)

    print(f"  {'City':>8}  {'N_bands':>7}  {'N_signals':>9}  {'Pre-METAR%':>10}  "
          f"{'Avg_entry':>9}  {'Avg_gain':>8}")
    print(f"  {'-'*8}  {'-'*7}  {'-'*9}  {'-'*10}  {'-'*9}  {'-'*8}")
    city_rows = []
    for city, entries in sorted(city_data.items()):
        n_bands = len(entries)
        hits = [e for e in entries if e is not None]
        n_sig = len(hits)
        if n_sig == 0:
            city_rows.append((city, n_bands, 0, 0, 0, 0))
            continue
        pre = sum(1 for e in hits if e["is_pre_metar"])
        avg_e = sum(e["price"] for e in hits) / n_sig
        avg_g = sum(e["gain"] for e in hits) / n_sig
        city_rows.append((city, n_bands, n_sig, 100*pre/n_sig, avg_e, avg_g))
    city_rows.sort(key=lambda r: -r[5])  # sort by avg_gain
    for city, n_bands, n_sig, pre_pct, avg_e, avg_g in city_rows:
        print(f"  {city:>8}  {n_bands:>7}  {n_sig:>9}  {pre_pct:>9.0f}%  "
              f"  {avg_e:>7.1f}¢  {avg_g:>7.1f}¢")

    # -------------------------------------------------------------------------
    # Report 4 — Lead time distribution (pre-METAR entries at 60¢ cap)
    # -------------------------------------------------------------------------
    print(f"\n{'='*68}")
    print("  REPORT 4 — Pre-METAR lead time distribution (entry cap = 60¢)")
    print(f"  (T_metar - T_entry for entries that fired before METAR)")
    print(f"{'='*68}")
    print()
    leads = []
    for d in band_details:
        e = d["entries"].get(60)
        if e and e["is_pre_metar"] and d["t_metar"] is not None:
            leads.append(d["t_metar"] - e["t_entry"])

    if leads:
        buckets = defaultdict(int)
        for l in leads:
            bucket = f"{l}h"
            buckets[l] += 1
        print(f"  Lead time  Count")
        print(f"  ---------  -----")
        for h in sorted(buckets):
            bar = "█" * buckets[h]
            print(f"  {h:>7}h   {buckets[h]:>3}  {bar}")
        avg_lead = sum(leads) / len(leads)
        med_lead = sorted(leads)[len(leads)//2]
        print(f"\n  Mean lead: {avg_lead:.1f}h  Median lead: {med_lead}h  "
              f"Max lead: {max(leads)}h  n={len(leads)}")
    else:
        print("  No pre-METAR entries found at 60¢ cap.")

    # -------------------------------------------------------------------------
    # Report 5 — False-positive rate on YES-settling bands
    # -------------------------------------------------------------------------
    print(f"\n{'='*68}")
    print("  REPORT 5 — False-positive rate (YES-settling bands)")
    print(f"  Would the strategy have entered on bands that settled YES?")
    print(f"  Loss = entry_price (100¢ paid, 0¢ received — wrong side)")
    print(f"{'='*68}")
    print()
    print(f"  Total YES-settling bands with candles: {len(yes_bands)}")
    print()

    fp_stats: dict[int, dict] = {t: {"n_fp": 0, "losses": []} for t in thresholds}

    for b in yes_bands:
        ticker   = b["ticker"]
        metric   = b["metric"]
        tdate    = trading_date_str(b["date"])

        tz = city_tz(metric)
        if tz is None:
            continue

        clist = candles[ticker]
        hourly = trace_no_ask(ticker, metric, tdate, clist, tz)
        if not hourly:
            continue

        for thresh in thresholds:
            entry = find_entry_hour(hourly, thresh)
            if entry:
                t_entry, price = entry
                # Loss: paid `price` cents, NO settles at 0¢
                fp_stats[thresh]["n_fp"] += 1
                fp_stats[thresh]["losses"].append(price)

    print(f"  {'Cap':>6}  {'N_FP':>6}  {'FP_rate':>7}  {'Avg_loss':>8}  "
          f"{'Expected_P&L_per_100_trades':>27}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*8}  {'-'*27}")
    for thresh in thresholds:
        s     = stats[thresh]
        fp    = fp_stats[thresh]
        n_tp  = s["n_signals"]
        n_fp  = fp["n_fp"]
        total = n_tp + n_fp
        if total == 0:
            continue
        fp_rate   = n_fp / total
        avg_loss  = sum(fp["losses"]) / n_fp if n_fp else 0
        avg_gain  = sum(s["gains"]) / n_tp if n_tp else 0
        # Expected P&L per trade = P(TP)*avg_gain - P(FP)*avg_loss
        exp_pnl   = (1 - fp_rate) * avg_gain - fp_rate * avg_loss
        print(f"  ≤{thresh:>4}¢  {n_fp:>6}  {100*fp_rate:>6.1f}%  {avg_loss:>7.1f}¢  "
              f"  {exp_pnl:>+.2f}¢ expected/trade")

    print()
    print("="*68)
    print("  Done.")
    print("="*68)


if __name__ == "__main__":
    main()
