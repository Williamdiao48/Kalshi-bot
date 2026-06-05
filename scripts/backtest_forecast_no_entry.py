"""
Backtest: HRRR-gated early NO entry — reworked to mirror forecast_yes logic.

Strategy being tested:
  - Entry window: 7-11 UTC (same as forecast_yes morning window)
  - Signal: HRRR forecast is ≥N°F OUTSIDE the band (clear NO)
  - Enter when NO ask (= 100 - YES bid) is in entry range
  - Exit at absolute NO bid PT (100 - YES ask) OR hold to settlement

Key insight from forecast_yes analysis:
  The morning HRRR signal creates intraday price discovery that reliably
  lifts/drops prices to exit levels. We want to capture the same mechanism
  but on the NO side of "clearly wrong" bands.

Reports:
  1. HRRR distance gate sweep — how does min HRRR distance affect signal quality?
  2. UTC hour entry window sweep — does restricting to 7-11 UTC help?
  3. Absolute NO bid PT sweep — at what PT level does PnL peak?
  4. Combined strategy (best distance + window + PT) vs. old approach
  5. False-positive rate on YES-settling bands (with HRRR gate)
  6. City breakdown

Usage:
    venv/bin/python scripts/backtest_forecast_no_entry.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT        = Path(__file__).parent.parent
DATA        = ROOT / "data"
BANDS_CSV   = DATA / "kxhigh_bands.csv"
MESONET     = DATA / "mesonet_hourly_combined.csv"
CANDLE_DB   = DATA / "candlesticks.db"
HIST_CACHE  = DATA / "backtest" / "band_arb_hist_cache.json"

# Strategy parameters to sweep
HRRR_DISTANCE_THRESHOLDS = [0, 1, 2, 3, 4]   # min °F HRRR must be outside band
ENTRY_NO_ASK_MAX         = 65                  # max NO ask cents (= YES bid ≥ 35¢)
ENTRY_NO_ASK_MIN         = 20                  # min NO ask cents (= YES bid ≤ 80¢)
UTC_ENTRY_HOURS          = set(range(7, 12))   # 7,8,9,10,11 UTC
NO_BID_PT_TARGETS        = [75, 80, 85, 88, 90]  # absolute NO bid PT levels

# ── Data loading ──────────────────────────────────────────────────────────────

def load_bands() -> list[dict]:
    """Load B-band market metadata from candlesticks.db markets table."""
    import sqlite3, re
    _MON = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
            "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}
    _RE = re.compile(r"^([A-Z]+)-(\d{2})([A-Z]{3})(\d{2})-B([\d.]+)$")
    _SERIES_TO_METRIC = {
        "KXHIGHLAX":"temp_high_lax","KXHIGHDEN":"temp_high_den","KXHIGHCHI":"temp_high_chi",
        "KXHIGHNY":"temp_high_ny","KXHIGHMIA":"temp_high_mia","KXHIGHDAL":"temp_high_dal",
        "KXHIGHBOS":"temp_high_bos","KXHIGHAUS":"temp_high_aus","KXHIGHOU":"temp_high_hou",
        "KXHIGHTSFO":"temp_high_sfo","KXHIGHTSEA":"temp_high_sea","KXHIGHTBOS":"temp_high_bos",
        "KXHIGHTPHX":"temp_high_phx","KXHIGHTPHIL":"temp_high_phl","KXHIGHTDC":"temp_high_dca",
        "KXHIGHTLV":"temp_high_las","KXHIGHTOKC":"temp_high_okc","KXHIGHTDAL":"temp_high_dfw",
        "KXHIGHTHOU":"temp_high_hou","KXHIGHTNOLA":"temp_high_msy","KXHIGHTATL":"temp_high_atl",
        "KXHIGHTMIN":"temp_high_msp","KXHIGHTDFW":"temp_high_dfw","KXHIGHTSATX":"temp_high_sat",
    }
    conn = sqlite3.connect(CANDLE_DB)
    rows_db = conn.execute(
        "SELECT ticker, result FROM markets WHERE ticker LIKE 'KXHIGH%B%'"
    ).fetchall()
    conn.close()
    out = []
    for ticker, result in rows_db:
        m = _RE.match(ticker)
        if not m:
            continue
        series, yy, mon, dd, mid = m.groups()
        metric = _SERIES_TO_METRIC.get(series)
        if not metric:
            continue
        try:
            settle_date = f"20{yy}-{_MON[mon]:02d}-{int(dd):02d}"
        except (KeyError, ValueError):
            continue
        mid_f = float(mid)
        band_lo = int(mid_f - 0.5)
        band_hi = band_lo + 1
        out.append({
            "ticker": ticker,
            "metric": metric,
            "date": settle_date,
            "strike_lo": str(float(band_lo)),
            "strike_hi": str(float(band_hi)),
            "result": result or "unknown",
        })
    return out


def load_hrrr() -> dict[str, dict[str, float]]:
    """Return {city: {date_str: forecast_high_f}}."""
    raw = json.loads(HIST_CACHE.read_text())
    out: dict[str, dict[str, float]] = {}
    for key, val in raw.items():
        if not key.startswith("hrrr_temp_high_"):
            continue
        city = key.replace("hrrr_temp_high_", "").split("_")[0]
        if isinstance(val, dict):
            out[city] = {k: float(v) for k, v in val.items() if v is not None}
    return out


def load_mesonet() -> dict[tuple, float]:
    """Return {(metric, date_str, local_hour): running_max_f}."""
    lookup: dict[tuple, float] = {}
    with open(MESONET) as f:
        for row in csv.DictReader(f):
            key = (row["city_metric"], row["date"], int(row["local_hour"]))
            lookup[key] = float(row["running_max_f"])
    return lookup


def load_candles() -> dict[str, list[tuple[int, int, int]]]:
    """Return {ticker: [(period_ts, bid_high, ask_open)]} from candlesticks.db."""
    import sqlite3
    conn = sqlite3.connect(CANDLE_DB)
    cur = conn.cursor()
    cur.execute(
        "SELECT ticker, period_ts, bid_high, ask_open "
        "FROM candles WHERE ticker LIKE 'KXHIGH%B%' ORDER BY ticker, period_ts"
    )
    out: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for ticker, ts, bh, ao in cur.fetchall():
        out[ticker].append((ts, bh or 0, ao or 0))
    conn.close()
    return dict(out)


# ── Candle helpers ────────────────────────────────────────────────────────────

def parse_candles(clist: list[tuple[int, int, int]]) -> list[tuple[int, int, int, int]]:
    """Return [(utc_hour, ts, no_ask_cents, no_bid_cents)] sorted by ts.

    From (period_ts, bid_high, ask_open) tuples from candlesticks.db:
      no_ask = 100 - bid_high   (cost to BUY NO = 100 minus YES bid high)
      no_bid = 100 - ask_open   (proceeds to SELL NO = 100 minus YES ask)
    bid_high is the highest YES bid in the period — best price to infer NO ask.
    ask_open is the YES ask at period open — conservative NO bid estimate.
    """
    out = []
    for ts, bid_high, ask_open in clist:
        if bid_high <= 0 and ask_open <= 0:
            continue
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        utc_hour = dt.hour
        no_ask = 100 - bid_high if bid_high > 0 else 100
        no_bid = 100 - ask_open if ask_open > 0 else 0
        out.append((utc_hour, ts, no_ask, no_bid))
    return out  # already sorted by ts from DB query


def find_entry(
    candles: list[tuple[int, int, int, int]],
    entry_date_ts_start: int,
    entry_date_ts_end: int,
    no_ask_max: int,
    no_ask_min: int,
    utc_hours: set[int],
) -> tuple[int, int] | None:
    """First (no_ask, ts) in the UTC window on entry_date within ask range."""
    for utc_hour, ts, no_ask, _ in candles:
        if ts < entry_date_ts_start or ts >= entry_date_ts_end:
            continue
        if utc_hour not in utc_hours:
            continue
        if no_ask_min <= no_ask <= no_ask_max:
            return no_ask, ts
    return None


def max_no_bid_after(
    candles: list[tuple[int, int, int, int]],
    entry_ts: int,
) -> int:
    """Highest NO bid seen strictly after entry_ts (through settlement)."""
    return max((c[3] for c in candles if c[1] > entry_ts), default=0)


# ── Settlement date → trading date ────────────────────────────────────────────

from datetime import date as _date, timedelta as _td

def trading_date(settlement_date_str: str) -> _date:
    # KXHIGH markets open ~midnight ET (4-5 UTC) on the settlement date and
    # close after the afternoon temperature peak on the same day.
    # The 7-11 UTC entry window is on the settlement date itself.
    return _date.fromisoformat(settlement_date_str)


def day_ts_bounds(d: _date) -> tuple[int, int]:
    start = int(datetime(d.year, d.month, d.day, 0, 0, tzinfo=timezone.utc).timestamp())
    return start, start + 86400


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading data...")
    bands   = load_bands()
    hrrr    = load_hrrr()
    candles = load_candles()

    no_bands  = [b for b in bands if b["result"] == "no"  and b["ticker"] in candles]
    yes_bands = [b for b in bands if b["result"] == "yes" and b["ticker"] in candles]
    print(f"  B-band NO markets with candles:  {len(no_bands)}")
    print(f"  B-band YES markets with candles: {len(yes_bands)}")

    # Pre-parse all candles once
    parsed: dict[str, list] = {}
    for b in no_bands + yes_bands:
        t = b["ticker"]
        if t not in parsed:
            parsed[t] = parse_candles(candles.get(t, []))

    # ── Report 1: HRRR distance gate sweep ───────────────────────────────────
    print(f"\n{'='*72}")
    print("  REPORT 1 — HRRR distance gate sweep")
    print(f"  (Entry: 7-11 UTC, NO ask {ENTRY_NO_ASK_MIN}-{ENTRY_NO_ASK_MAX}¢, PT=85¢ NO bid)")
    print(f"{'='*72}")

    PT = 85
    for min_dist in HRRR_DISTANCE_THRESHOLDS:
        trades = []
        fps = []
        for b in no_bands:
            city = b["metric"].replace("temp_high_", "")
            settle_date = b["date"]
            hrrr_fc = hrrr.get(city, {}).get(settle_date)
            if hrrr_fc is None:
                continue
            lo, hi = float(b["strike_lo"]), float(b["strike_hi"])
            # HRRR distance: how far outside the band is the forecast?
            if hrrr_fc < lo:
                dist = lo - hrrr_fc
            elif hrrr_fc > hi:
                dist = hrrr_fc - hi
            else:
                dist = 0.0
            if dist < min_dist:
                continue

            tdate = trading_date(settle_date)
            ts_start, ts_end = day_ts_bounds(tdate)
            ck = parsed[b["ticker"]]
            entry = find_entry(ck, ts_start, ts_end, ENTRY_NO_ASK_MAX, ENTRY_NO_ASK_MIN, UTC_ENTRY_HOURS)
            if entry is None:
                continue
            no_ask, entry_ts = entry
            peak_no_bid = max_no_bid_after(ck, entry_ts)
            pt_hit = peak_no_bid >= PT
            pnl = (PT - no_ask) if pt_hit else (100 - no_ask - 100 + (100 if b["result"]=="no" else 0))
            # simplified: PT hit → PT-ask gain; miss → settle NO = +gain, YES = -ask
            pnl_c = (PT - no_ask) if pt_hit else (100 - no_ask if b["result"]=="no" else -no_ask)
            trades.append({"no_ask": no_ask, "peak_no_bid": peak_no_bid, "pt_hit": pt_hit, "pnl": pnl_c})

        n = len(trades)
        if n == 0:
            print(f"  dist≥{min_dist}°F: 0 trades")
            continue
        pt_rate = sum(1 for t in trades if t["pt_hit"]) / n
        avg_ask = sum(t["no_ask"] for t in trades) / n
        avg_pnl = sum(t["pnl"] for t in trades) / n
        total_pnl = sum(t["pnl"] for t in trades) / 100
        print(f"  dist≥{min_dist}°F: {n:3d} trades  PT={pt_rate*100:5.1f}%  avg_ask={avg_ask:.1f}¢  "
              f"avg_pnl={avg_pnl:+.1f}¢  total={total_pnl:+.2f}")

    # ── Report 2: PT sweep at best HRRR distance (≥2°F) ─────────────────────
    BEST_DIST = 2
    print(f"\n{'='*72}")
    print(f"  REPORT 2 — Absolute NO bid PT sweep (HRRR dist≥{BEST_DIST}°F, 7-11 UTC)")
    print(f"{'='*72}")

    # Collect all trades at BEST_DIST once
    base_trades = []
    for b in no_bands:
        city = b["metric"].replace("temp_high_", "")
        settle_date = b["date"]
        hrrr_fc = hrrr.get(city, {}).get(settle_date)
        if hrrr_fc is None:
            continue
        lo, hi = float(b["strike_lo"]), float(b["strike_hi"])
        dist = max(0.0, lo - hrrr_fc) if hrrr_fc < lo else max(0.0, hrrr_fc - hi) if hrrr_fc > hi else 0.0
        if dist < BEST_DIST:
            continue
        tdate = trading_date(settle_date)
        ts_start, ts_end = day_ts_bounds(tdate)
        ck = parsed[b["ticker"]]
        entry = find_entry(ck, ts_start, ts_end, ENTRY_NO_ASK_MAX, ENTRY_NO_ASK_MIN, UTC_ENTRY_HOURS)
        if entry is None:
            continue
        no_ask, entry_ts = entry
        peak_no_bid = max_no_bid_after(ck, entry_ts)
        base_trades.append({"no_ask": no_ask, "peak_no_bid": peak_no_bid, "result": b["result"], "city": b["metric"].replace("temp_high_",""), "settle_date": b["date"]})

    n = len(base_trades)
    print(f"  Base trades (HRRR dist≥{BEST_DIST}°F, 7-11 UTC, NO ask {ENTRY_NO_ASK_MIN}-{ENTRY_NO_ASK_MAX}¢): {n}")
    print()
    for pt in NO_BID_PT_TARGETS:
        if n == 0:
            break
        wins   = [t for t in base_trades if t["peak_no_bid"] >= pt]
        losses = [t for t in base_trades if t["peak_no_bid"] < pt]
        win_pnl  = sum(pt - t["no_ask"] for t in wins) / 100
        loss_pnl = sum((100 - t["no_ask"] if t["result"]=="no" else -t["no_ask"]) for t in losses) / 100
        avg_pnl  = (win_pnl + loss_pnl) / n * 100
        print(f"  PT={pt}¢  hits={len(wins):3d}/{n} ({len(wins)/n*100:5.1f}%)  "
              f"avg_pnl={avg_pnl:+.1f}¢  total={win_pnl+loss_pnl:+.2f}")

    # ── Report 3: Entry ask range sensitivity ────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  REPORT 3 — Entry NO ask range sensitivity (dist≥{BEST_DIST}°F, 7-11 UTC, PT=85¢)")
    print(f"{'='*72}")
    PT = 85
    for max_ask in [50, 55, 60, 65, 70]:
        sub = [t for t in base_trades if t["no_ask"] <= max_ask]
        if not sub:
            continue
        wins = [t for t in sub if t["peak_no_bid"] >= PT]
        loss_pnl = sum((100 - t["no_ask"] if t["result"]=="no" else -t["no_ask"]) for t in sub if t["peak_no_bid"] < PT) / 100
        win_pnl  = sum(PT - t["no_ask"] for t in wins) / 100
        avg_pnl  = (win_pnl + loss_pnl) / len(sub) * 100
        print(f"  NO ask≤{max_ask}¢: {len(sub):3d} trades  PT={len(wins)/len(sub)*100:5.1f}%  "
              f"avg_pnl={avg_pnl:+.1f}¢  total={win_pnl+loss_pnl:+.2f}")

    # ── Report 4: UTC hour window sensitivity ────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  REPORT 4 — UTC entry hour window (dist≥{BEST_DIST}°F, NO ask {ENTRY_NO_ASK_MIN}-{ENTRY_NO_ASK_MAX}¢, PT=85¢)")
    print(f"{'='*72}")
    PT = 85
    for hour_range in [(7, 12), (7, 10), (7, 9), (6, 12), (0, 24)]:
        h_set = set(range(hour_range[0], hour_range[1]))
        h_trades = []
        for b in no_bands:
            city = b["metric"].replace("temp_high_", "")
            hrrr_fc = hrrr.get(city, {}).get(b["date"])
            if hrrr_fc is None:
                continue
            lo, hi = float(b["strike_lo"]), float(b["strike_hi"])
            dist = max(0.0, lo - hrrr_fc) if hrrr_fc < lo else max(0.0, hrrr_fc - hi) if hrrr_fc > hi else 0.0
            if dist < BEST_DIST:
                continue
            tdate = trading_date(b["date"])
            ts_start, ts_end = day_ts_bounds(tdate)
            ck = parsed[b["ticker"]]
            entry = find_entry(ck, ts_start, ts_end, ENTRY_NO_ASK_MAX, ENTRY_NO_ASK_MIN, h_set)
            if entry is None:
                continue
            no_ask, entry_ts = entry
            peak_no_bid = max_no_bid_after(ck, entry_ts)
            h_trades.append({"no_ask": no_ask, "peak_no_bid": peak_no_bid, "result": b["result"]})
        n_h = len(h_trades)
        if n_h == 0:
            print(f"  UTC {hour_range[0]:02d}-{hour_range[1]:02d}: 0 trades")
            continue
        wins = [t for t in h_trades if t["peak_no_bid"] >= PT]
        loss_pnl = sum((100 - t["no_ask"] if t["result"]=="no" else -t["no_ask"]) for t in h_trades if t["peak_no_bid"] < PT) / 100
        win_pnl  = sum(PT - t["no_ask"] for t in wins) / 100
        avg_pnl  = (win_pnl + loss_pnl) / n_h * 100
        print(f"  UTC {hour_range[0]:02d}-{hour_range[1]:02d}: {n_h:3d} trades  PT={len(wins)/n_h*100:5.1f}%  "
              f"avg_pnl={avg_pnl:+.1f}¢  total={win_pnl+loss_pnl:+.2f}")

    # ── Report 5: City breakdown ──────────────────────────────────────────────
    PT = 85
    print(f"\n{'='*72}")
    print(f"  REPORT 5 — City breakdown (dist≥{BEST_DIST}°F, 7-11 UTC, PT=85¢)")
    print(f"{'='*72}")
    city_trades: dict[str, list] = defaultdict(list)
    for t in base_trades:
        city_trades[t["city"]].append(t)
    print(f"  {'City':>6}  {'N':>4}  {'PT%':>5}  {'avg_ask':>7}  {'avg_pnl':>8}  {'total':>7}")
    for city, tlist in sorted(city_trades.items(), key=lambda x: -len(x[1])):
        wins = [t for t in tlist if t["peak_no_bid"] >= PT]
        asks = [t["no_ask"] for t in tlist]
        loss_pnl = sum((100 - t["no_ask"] if t["result"]=="no" else -t["no_ask"]) for t in tlist if t["peak_no_bid"] < PT) / 100
        win_pnl  = sum(PT - t["no_ask"] for t in wins) / 100
        avg_pnl  = (win_pnl + loss_pnl) / len(tlist) * 100
        print(f"  {city:>6}  {len(tlist):>4}  {len(wins)/len(tlist)*100:>4.0f}%  "
              f"{sum(asks)/len(asks):>7.1f}¢  {avg_pnl:>+7.1f}¢  {win_pnl+loss_pnl:>+6.2f}")

    # ── Report 6: False-positive rate on YES-settling bands ──────────────────
    PT = 85
    print(f"\n{'='*72}")
    print(f"  REPORT 6 — False-positive rate on YES-settling bands (dist≥{BEST_DIST}°F, 7-11 UTC)")
    print(f"  Loss = entry price (NO settles 0¢ — you bought the wrong side)")
    print(f"{'='*72}")
    fp_count = 0
    fp_losses = []
    for b in yes_bands:
        city = b["metric"].replace("temp_high_", "")
        hrrr_fc = hrrr.get(city, {}).get(b["date"])
        if hrrr_fc is None:
            continue
        lo, hi = float(b["strike_lo"]), float(b["strike_hi"])
        dist = max(0.0, lo - hrrr_fc) if hrrr_fc < lo else max(0.0, hrrr_fc - hi) if hrrr_fc > hi else 0.0
        if dist < BEST_DIST:
            continue
        tdate = trading_date(b["date"])
        ts_start, ts_end = day_ts_bounds(tdate)
        ck = parsed[b["ticker"]]
        entry = find_entry(ck, ts_start, ts_end, ENTRY_NO_ASK_MAX, ENTRY_NO_ASK_MIN, UTC_ENTRY_HOURS)
        if entry is None:
            continue
        no_ask, _ = entry
        fp_count += 1
        fp_losses.append(no_ask)

    n_yes = len(yes_bands)
    n_no  = len(base_trades)
    total = n_no + fp_count
    if total > 0:
        fp_rate = fp_count / total
        avg_fp_loss = sum(fp_losses) / fp_count if fp_count else 0
        avg_tp_gain = sum(t["no_ask"] for t in base_trades) / n_no if n_no else 0
        exp_pnl_settle = (1 - fp_rate) * (100 - avg_tp_gain) - fp_rate * avg_fp_loss
        print(f"  NO-settling (TPs): {n_no}")
        print(f"  YES-settling (FPs): {fp_count}  avg loss = {avg_fp_loss:.1f}¢")
        print(f"  FP rate: {fp_rate*100:.1f}%")
        print(f"  Expected P&L/trade (hold-to-settle): {exp_pnl_settle:+.1f}¢")
        print()
        wins_settle = n_no  # all NO-settling bands held to settlement are wins
        print(f"  Comparison to old forecast_no:")
        print(f"    Old: enter at 45-80¢ NO ask, 4 sources required, % PT")
        print(f"    New: enter at {ENTRY_NO_ASK_MIN}-{ENTRY_NO_ASK_MAX}¢ NO ask, HRRR dist≥{BEST_DIST}°F, 7-11 UTC, 85¢ abs PT")

    print(f"\n{'='*72}")
    print("  Done.")
    print(f"{'='*72}")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
