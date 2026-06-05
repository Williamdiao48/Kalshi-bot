"""
Backtest: does morning forecast consensus predict which KXHIGH B-band will settle YES?
And if so, does the YES price rise during the day allowing a profit-take?

Strategy:
  - Morning (7-11 UTC): forecast rounds to a specific °F → identifies the "target" band
  - Enter YES on that band if ask is in a reasonable range (e.g. 10-55¢)
  - Exit at a profit-take level OR hold to settlement

Sections:
  1. Per-source individual performance (each source as sole signal)
  2. Ensemble agreement threshold sweep (≥N sources must agree)
  3. Best PT per agreement level
  4. Exact agreement count breakdown
  5. HRRR + secondary source pairwise (intersection)

Data sources (no look-ahead, no simulation):
  - data/candlesticks.db         — real 1-min Kalshi YES bid/ask candles (Apr 4 – Jun 4)
  - data/db/opportunity_log.db   — bot's raw_forecasts (HRRR at 7-9 UTC only; other sources daily avg)
  - candlesticks.db markets table — actual settlement results

  NOTE: HRRR is loaded at 7-9 UTC only from raw_forecasts (May 2+).
  The hist_cache HRRR is no longer used — it reflected Open-Meteo retrospective
  daily-max values (look-ahead bias of ~17% band mismatches vs 7 UTC signal).

Run:
  venv/bin/python scripts/backtest_forecast_band_yes.py
"""

from __future__ import annotations

import json, re, sqlite3
from collections import defaultdict, Counter
from datetime import datetime, timezone
from pathlib import Path

CANDLE_DB   = Path("data/candlesticks.db")
HIST_CACHE  = Path("data/backtest/band_arb_hist_cache.json")
BOT_DB      = Path("data/db/opportunity_log.db")

ENTRY_HOURS_UTC = list(range(7, 12))
ENTRY_MIN_ASK = 10
ENTRY_MAX_ASK = 55
BEST_PT = 70
PT_TARGETS = [50, 60, 65, 70, 75, 80]

ENSEMBLE_SOURCES = [
    "hrrr",
    "noaa",
    "nws_hourly",
    "open_meteo",
    "open_meteo_ecmwf",
    "open_meteo_gfs",
    "open_meteo_gem",
    "open_meteo_icon",
]

_SERIES_TO_CITY = {
    "KXHIGHLAX":   "lax", "KXHIGHDEN":   "den", "KXHIGHCHI":   "chi",
    "KXHIGHNY":    "ny",  "KXHIGHMIA":   "mia", "KXHIGHDAL":   "dal",
    "KXHIGHBOS":   "bos", "KXHIGHAUS":   "aus", "KXHIGHOU":    "hou",
    "KXHIGHTSFO":  "sfo", "KXHIGHTSEA":  "sea", "KXHIGHTBOS":  "bos",
    "KXHIGHTPHX":  "phx", "KXHIGHTPHIL": "phl", "KXHIGHTDC":   "dca",
    "KXHIGHTLV":   "las", "KXHIGHTOKC":  "okc", "KXHIGHTDAL":  "dfw",
    "KXHIGHTHOU":  "hou", "KXHIGHTNOLA": "msy", "KXHIGHTATL":  "atl",
    "KXHIGHTMIN":  "msp", "KXHIGHTDFW":  "dfw", "KXHIGHTSATX": "sat",
}
_TICKER_RE = re.compile(r"^([A-Z]+)-(\d{2})([A-Z]{3})(\d{2})-B([\d.]+)$")
_MON = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
        "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}


def parse_ticker(ticker: str):
    m = _TICKER_RE.match(ticker)
    if not m:
        return None
    series, yy, mon, dd, mid = m.groups()
    city = _SERIES_TO_CITY.get(series)
    if not city:
        return None
    try:
        date_str = f"20{yy}-{_MON[mon]:02d}-{int(dd):02d}"
    except (KeyError, ValueError):
        return None
    mid_f   = float(mid)
    band_lo = int(mid_f - 0.5)
    band_hi = band_lo + 1
    return {"series": series, "city": city, "date": date_str,
            "band_lo": band_lo, "band_hi": band_hi, "mid": mid_f}


def load_all_forecasts() -> dict[tuple[str, str, str], float]:
    """Returns {(source, city, date): forecast_f} for all ensemble sources.

    HRRR is loaded exclusively from bot raw_forecasts at 7-9 UTC — the actual
    6z model run available at entry time.  Using the hist_cache HRRR (Open-Meteo
    historical API) introduced look-ahead: it reflects retrospective daily-max
    values that incorporate later model runs not available at 7 UTC.

    All other sources (open_meteo ensemble, noaa, nws_hourly) use daily AVG
    from raw_forecasts — their daily-max forecasts are stable through the day
    (initialized at 0z/6z and do not change materially by 7 UTC entry window).
    """
    forecasts: dict[tuple[str, str, str], float] = {}

    if BOT_DB.exists():
        conn = sqlite3.connect(BOT_DB)

        # HRRR: 7-9 UTC only (6z run available at entry time, no look-ahead)
        hrrr_rows = conn.execute("""
            SELECT metric, date(logged_at), AVG(data_value)
            FROM raw_forecasts
            WHERE source = 'hrrr'
              AND metric LIKE 'temp_high_%'
              AND data_value IS NOT NULL
              AND time(logged_at) BETWEEN '07:00' AND '09:00'
            GROUP BY metric, date(logged_at)
        """).fetchall()
        for metric, date_str, val in hrrr_rows:
            city = metric.replace("temp_high_", "")
            if val is not None:
                forecasts[("hrrr", city, date_str)] = float(val)

        # All other sources: daily AVG (stable forecasts, no timing bias)
        other_sources = [s for s in ENSEMBLE_SOURCES if s != "hrrr"]
        src_list = ",".join(f"'{s}'" for s in other_sources)
        rows = conn.execute(f"""
            SELECT source, metric, date(logged_at), AVG(data_value)
            FROM raw_forecasts
            WHERE source IN ({src_list})
              AND metric LIKE 'temp_high_%'
              AND data_value IS NOT NULL
            GROUP BY source, metric, date(logged_at)
        """).fetchall()
        conn.close()
        for source, metric, date_str, val in rows:
            city = metric.replace("temp_high_", "")
            k = (source, city, date_str)
            if k not in forecasts and val is not None:
                forecasts[k] = float(val)

    src_counts = Counter(k[0] for k in forecasts)
    print("Loaded forecasts per source (HRRR = 7-9 UTC only, others = daily avg):")
    for src in ENSEMBLE_SOURCES:
        print(f"  {src:25s}  {src_counts.get(src, 0):>5,} entries")

    return forecasts


def load_candles(conn: sqlite3.Connection) -> dict[str, list[tuple[int, int, int]]]:
    rows = conn.execute("""
        SELECT ticker, period_ts, ask_close, bid_close
        FROM candles
        WHERE ticker LIKE 'KXHIGH%-B%' AND ask_close IS NOT NULL
        ORDER BY ticker, period_ts
    """).fetchall()
    candles: dict[str, list] = defaultdict(list)
    for ticker, ts, ask, bid in rows:
        if ask and ask > 0:
            candles[ticker].append((ts, ask, bid or 0))
    return candles


def get_entry(candles_for_ticker: list, open_ts: int) -> tuple[int, int] | None:
    for ts, ask, bid in candles_for_ticker:
        if ts < open_ts:
            continue
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        if dt.hour not in ENTRY_HOURS_UTC:
            continue
        if ENTRY_MIN_ASK <= ask <= ENTRY_MAX_ASK:
            return (ts, ask)
    return None


def simulate(candles_for_ticker: list, entry_ts: int, entry_ask: int,
             result: str, pt_target: int) -> float:
    won = result == "yes"
    for ts, ask, bid in candles_for_ticker:
        if ts <= entry_ts:
            continue
        if ask >= pt_target:
            return ask - entry_ask
    return (100 if won else 0) - entry_ask


def simulate_timed(candles_for_ticker: list, entry_ts: int, entry_ask: int,
                   result: str, pt_target: int) -> tuple[float, int | None]:
    """Returns (pnl, pt_hit_ts) where pt_hit_ts is None if PT was never hit."""
    won = result == "yes"
    for ts, ask, bid in candles_for_ticker:
        if ts <= entry_ts:
            continue
        if ask >= pt_target:
            return ask - entry_ask, ts
    return (100 if won else 0) - entry_ask, None


def fmt(recs: list[dict], pt: int) -> str:
    if not recs:
        return "  n=0"
    pnls = [r["pnl_by_pt"][pt] for r in recs]
    n = len(recs)
    wins = sum(1 for p in pnls if p > 0)
    avg_entry = sum(r["entry_ask"] for r in recs) / n
    avg_pnl = sum(pnls) / n
    total = sum(pnls) / 100
    yes_rate = sum(1 for r in recs if r["won"]) / n
    return (f"{n:>4}  {100*wins/n:>5.1f}%  {avg_entry:>4.1f}¢"
            f"  {avg_pnl:>+6.1f}¢  {total:>+8.2f}$  YES={100*yes_rate:>4.1f}%")


def main():
    cconn = sqlite3.connect(CANDLE_DB)

    settlements: dict[str, str] = {}
    open_times:  dict[str, int] = {}
    for ticker, open_ts, close_ts, result in cconn.execute(
        "SELECT ticker, open_ts, close_ts, result FROM markets WHERE ticker LIKE 'KXHIGH%-B%'"
    ).fetchall():
        if result in ("yes", "no"):
            settlements[ticker] = result
            open_times[ticker]  = open_ts

    print(f"Markets with results: {len(settlements):,} "
          f"({sum(1 for r in settlements.values() if r=='yes')} YES)\n")

    all_forecasts = load_all_forecasts()
    candles = load_candles(cconn)
    cconn.close()
    print(f"Loaded candles for {len(candles):,} KXHIGH B-band tickers\n")

    # ── Build trade universe ──────────────────────────────────────────────────
    trade_recs: list[dict] = []

    for ticker, result in settlements.items():
        info = parse_ticker(ticker)
        if not info:
            continue

        city, date_str = info["city"], info["date"]
        band_lo, band_hi = info["band_lo"], info["band_hi"]

        ticker_candles = candles.get(ticker, [])
        if not ticker_candles:
            continue

        open_ts = open_times.get(ticker, 0)
        entry = get_entry(ticker_candles, open_ts)
        if entry is None:
            continue

        entry_ts, entry_ask = entry

        agreeing = [
            src for src in ENSEMBLE_SOURCES
            if (fc := all_forecasts.get((src, city, date_str))) is not None
            and band_lo <= round(fc) <= band_hi
        ]

        if not agreeing:
            continue

        entry_hour = datetime.fromtimestamp(entry_ts, tz=timezone.utc).hour

        pnl_by_pt:    dict[int, float] = {}
        pt_hit_ts:    dict[int, int | None] = {}
        for pt in PT_TARGETS:
            pnl, hit_ts = simulate_timed(ticker_candles, entry_ts, entry_ask, result, pt)
            pnl_by_pt[pt]   = pnl
            pt_hit_ts[pt]   = hit_ts

        fc_hrrr = all_forecasts.get(("hrrr", city, date_str))
        mid_f   = info["mid"]
        # fc_offset: how far HRRR is from band center (-1..+1); near 0 = robust, near ±1 = fragile
        fc_offset    = (fc_hrrr - mid_f) if fc_hrrr is not None else None
        # safe_margin: distance from nearest band outer boundary (higher = less likely rounding flip)
        safe_margin  = (1.0 - abs(fc_offset)) if fc_offset is not None else None

        trade_recs.append({
            "ticker":      ticker,
            "date":        date_str,
            "month":       date_str[5:7],
            "city":        city,
            "entry_ask":   entry_ask,
            "entry_ts":    entry_ts,
            "entry_hour":  entry_hour,
            "result":      result,
            "won":         result == "yes",
            "agreeing":    set(agreeing),
            "n_agree":     len(agreeing),
            "pnl_by_pt":   pnl_by_pt,
            "pt_hit_ts":   pt_hit_ts,
            "fc_hrrr":     fc_hrrr,
            "mid_f":       mid_f,
            "fc_offset":   fc_offset,
            "safe_margin": safe_margin,
        })

    print(f"Trade universe (≥1 source signals band, entry found): {len(trade_recs)}\n")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 1: Individual source performance
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"{'='*80}")
    print(f"  SECTION 1: Per-source performance (PT={BEST_PT}¢ — trades where that source agrees)")
    print(f"{'='*80}")
    print(f"  {'source':25s}  {'n':>4}  {'WR':>6}  {'entry':>5}  {'avg¢':>7}  {'total$':>8}  {'YES%':>7}")
    print(f"  {'-'*72}")

    for src in ENSEMBLE_SOURCES:
        recs = [r for r in trade_recs if src in r["agreeing"]]
        if not recs:
            print(f"  {src:25s}  — no trades")
            continue
        pnls = [r["pnl_by_pt"][BEST_PT] for r in recs]
        n = len(recs)
        wins = sum(1 for p in pnls if p > 0)
        avg_entry = sum(r["entry_ask"] for r in recs) / n
        avg_pnl = sum(pnls) / n
        total = sum(pnls) / 100
        yes_rate = sum(1 for r in recs if r["won"]) / n
        print(f"  {src:25s}  {n:>4}  {100*wins/n:>5.1f}%  {avg_entry:>4.1f}¢"
              f"  {avg_pnl:>+6.1f}¢  {total:>+8.2f}$  {100*yes_rate:>5.1f}%")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 2: Ensemble agreement threshold sweep
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  SECTION 2: Cumulative ensemble threshold — trade if ≥N sources agree (PT={BEST_PT}¢)")
    print(f"{'='*80}")
    print(f"  {'≥N':>4}  {'n':>4}  {'WR':>6}  {'entry':>5}  {'avg¢':>7}  {'total$':>8}  {'YES%':>7}")
    print(f"  {'-'*60}")

    for min_agree in range(1, len(ENSEMBLE_SOURCES) + 1):
        recs = [r for r in trade_recs if r["n_agree"] >= min_agree]
        if not recs:
            break
        pnls = [r["pnl_by_pt"][BEST_PT] for r in recs]
        n = len(recs)
        wins = sum(1 for p in pnls if p > 0)
        avg_entry = sum(r["entry_ask"] for r in recs) / n
        avg_pnl = sum(pnls) / n
        total = sum(pnls) / 100
        yes_rate = sum(1 for r in recs if r["won"]) / n
        print(f"  {min_agree:>4}  {n:>4}  {100*wins/n:>5.1f}%  {avg_entry:>4.1f}¢"
              f"  {avg_pnl:>+6.1f}¢  {total:>+8.2f}$  {100*yes_rate:>5.1f}%")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 3: Best PT per agreement level
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  SECTION 3: Best PT per cumulative agreement level")
    print(f"{'='*80}")
    print(f"  {'≥N':>4}  {'best_PT':>7}  {'n':>4}  {'WR':>6}  {'avg¢':>7}  {'total$':>8}")
    print(f"  {'-'*55}")

    for min_agree in range(1, len(ENSEMBLE_SOURCES) + 1):
        recs = [r for r in trade_recs if r["n_agree"] >= min_agree]
        if not recs:
            break
        best = max(PT_TARGETS, key=lambda pt: sum(r["pnl_by_pt"][pt] for r in recs))
        pnls = [r["pnl_by_pt"][best] for r in recs]
        n = len(recs)
        wins = sum(1 for p in pnls if p > 0)
        print(f"  {min_agree:>4}  {best:>5}¢    {n:>4}  {100*wins/n:>5.1f}%"
              f"  {sum(pnls)/n:>+6.1f}¢  {sum(pnls)/100:>+8.2f}$")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 4: Exact agreement count breakdown
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  SECTION 4: Exact agreement count (PT={BEST_PT}¢) — marginal value of each extra model")
    print(f"{'='*80}")
    print(f"  {'n_agree':>7}  {'n':>4}  {'WR':>6}  {'entry':>5}  {'avg¢':>7}  {'total$':>8}  {'YES%':>7}")
    print(f"  {'-'*65}")

    for n_agree in range(1, len(ENSEMBLE_SOURCES) + 1):
        recs = [r for r in trade_recs if r["n_agree"] == n_agree]
        if not recs:
            continue
        pnls = [r["pnl_by_pt"][BEST_PT] for r in recs]
        n = len(recs)
        wins = sum(1 for p in pnls if p > 0)
        avg_entry = sum(r["entry_ask"] for r in recs) / n
        avg_pnl = sum(pnls) / n
        total = sum(pnls) / 100
        yes_rate = sum(1 for r in recs if r["won"]) / n
        print(f"  {n_agree:>7}   {n:>4}  {100*wins/n:>5.1f}%  {avg_entry:>4.1f}¢"
              f"  {avg_pnl:>+6.1f}¢  {total:>+8.2f}$  {100*yes_rate:>5.1f}%")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 5: HRRR + secondary source pairwise (intersection)
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  SECTION 5: HRRR + secondary source intersection (PT={BEST_PT}¢)")
    print(f"{'='*80}")
    print(f"  {'combo':30s}  {'n':>4}  {'WR':>6}  {'avg¢':>7}  {'total$':>8}  {'YES%':>7}")
    print(f"  {'-'*70}")

    hrrr_recs = [r for r in trade_recs if "hrrr" in r["agreeing"]]
    pnls = [r["pnl_by_pt"][BEST_PT] for r in hrrr_recs]
    n = len(hrrr_recs)
    wins = sum(1 for p in pnls if p > 0)
    yes_rate = sum(1 for r in hrrr_recs if r["won"]) / n if n else 0
    print(f"  {'hrrr (baseline)':30s}  {n:>4}  {100*wins/n:>5.1f}%"
          f"  {sum(pnls)/n:>+6.1f}¢  {sum(pnls)/100:>+8.2f}$  {100*yes_rate:>5.1f}%")

    for src in ENSEMBLE_SOURCES:
        if src == "hrrr":
            continue
        recs = [r for r in trade_recs if "hrrr" in r["agreeing"] and src in r["agreeing"]]
        if not recs:
            print(f"  {'hrrr+'+src:30s}  — no overlap")
            continue
        pnls = [r["pnl_by_pt"][BEST_PT] for r in recs]
        n = len(recs)
        wins = sum(1 for p in pnls if p > 0)
        avg_pnl = sum(pnls) / n
        total = sum(pnls) / 100
        yes_rate = sum(1 for r in recs if r["won"]) / n
        print(f"  {'hrrr+'+src:30s}  {n:>4}  {100*wins/n:>5.1f}%"
              f"  {avg_pnl:>+6.1f}¢  {total:>+8.2f}$  {100*yes_rate:>5.1f}%")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 6: PT sweep for strongest ensemble tier
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  SECTION 6: Full PT sweep for ≥2 and ≥3 source agreement")
    print(f"{'='*80}")
    for min_agree in [2, 3]:
        recs = [r for r in trade_recs if r["n_agree"] >= min_agree]
        if not recs:
            continue
        print(f"\n  ≥{min_agree} sources agree (n={len(recs)}):")
        print(f"  {'PT':>5}  {'n':>4}  {'WR':>6}  {'avg¢':>7}  {'total$':>8}")
        print(f"  {'-'*45}")
        for pt in PT_TARGETS:
            pnls = [r["pnl_by_pt"][pt] for r in recs]
            n = len(recs)
            wins = sum(1 for p in pnls if p > 0)
            print(f"  {pt:>4}¢  {n:>4}  {100*wins/n:>5.1f}%  {sum(pnls)/n:>+6.1f}¢  {sum(pnls)/100:>+8.2f}$")


    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 7: Discriminating factors — what separates PT hits from full losses?
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  SECTION 7: Discriminating factors (HRRR-only, PT={BEST_PT}¢)")
    print(f"  All factors observable at entry — no look-ahead")
    print(f"{'='*80}")

    hrrr_recs = [r for r in trade_recs if "hrrr" in r["agreeing"]]

    def hit_stats(recs: list[dict], pt: int = BEST_PT) -> str:
        if not recs:
            return f"{'—':>4}  {'—':>6}  {'—':>6}  {'—':>7}"
        n      = len(recs)
        hits   = sum(1 for r in recs if r["pt_hit_ts"][pt] is not None)
        pnls   = [r["pnl_by_pt"][pt] for r in recs]
        yes    = sum(1 for r in recs if r["won"])
        return (f"{n:>4}  {100*hits/n:>5.1f}%  {100*yes/n:>5.1f}%  {sum(pnls)/n:>+6.1f}¢")

    hdr = f"  {'factor':35s}  {'n':>4}  {'PT%':>5}  {'YES%':>5}  {'avg¢':>7}"
    sep = f"  {'-'*62}"

    # ── Factor A: Entry ask ───────────────────────────────────────────────────
    print(f"\n  A. Entry ask — does market pricing predict HRRR accuracy?")
    print(f"     Intuition: higher ask = market already agrees; lower = market doubts HRRR")
    print(hdr); print(sep)
    ask_buckets = [(10,20),(20,30),(30,40),(40,50),(50,56)]
    for lo, hi in ask_buckets:
        grp = [r for r in hrrr_recs if lo <= r["entry_ask"] < hi]
        print(f"  {'ask '+str(lo)+'–'+str(hi-1)+'¢':35s}  {hit_stats(grp)}")

    # ── Factor B: HRRR safe margin ────────────────────────────────────────────
    print(f"\n  B. HRRR safe margin (1 - |fc - band_mid|)")
    print(f"     Intuition: near 0 = one rounding-flip away from missing band; near 1 = robust")
    print(hdr); print(sep)
    margin_buckets = [(0.0,0.25,"0.00–0.25 (fragile)"),(0.25,0.50,"0.25–0.50"),
                      (0.50,0.75,"0.50–0.75"),(0.75,1.01,"0.75–1.00 (robust)")]
    for lo, hi, label in margin_buckets:
        grp = [r for r in hrrr_recs
               if r["safe_margin"] is not None and lo <= r["safe_margin"] < hi]
        print(f"  {label:35s}  {hit_stats(grp)}")

    # ── Factor C: HRRR offset direction ──────────────────────────────────────
    print(f"\n  C. HRRR offset direction (fc vs band center)")
    print(f"     Intuition: HRRR cold-biased (neg) vs hot-biased (pos) in this city/season")
    print(hdr); print(sep)
    offset_buckets = [(-1.0,-0.5,"fc < mid−0.5 (cold end of band)"),
                      (-0.5, 0.0,"fc mid−0.5 to mid (lower half)"),
                      ( 0.0, 0.5,"fc mid to mid+0.5 (upper half)"),
                      ( 0.5, 1.1,"fc > mid+0.5 (hot end of band)")]
    for lo, hi, label in offset_buckets:
        grp = [r for r in hrrr_recs
               if r["fc_offset"] is not None and lo <= r["fc_offset"] < hi]
        print(f"  {label:35s}  {hit_stats(grp)}")

    # ── Factor D: City ────────────────────────────────────────────────────────
    print(f"\n  D. City — sorted by PT hit rate")
    print(f"     Intuition: HRRR accuracy varies by geography/terrain complexity")
    print(hdr); print(sep)
    cities = sorted(set(r["city"] for r in hrrr_recs))
    city_rows = []
    for city in cities:
        grp = [r for r in hrrr_recs if r["city"] == city]
        hits = sum(1 for r in grp if r["pt_hit_ts"][BEST_PT] is not None)
        city_rows.append((city, grp, hits / len(grp)))
    for city, grp, _ in sorted(city_rows, key=lambda x: -x[2]):
        print(f"  {city:35s}  {hit_stats(grp)}")

    # ── Factor E: Month ───────────────────────────────────────────────────────
    print(f"\n  E. Month — April vs May")
    print(f"     Intuition: spring temperature variance higher in April; May more settled")
    print(hdr); print(sep)
    for month, label in [("04","April"),("05","May")]:
        grp = [r for r in hrrr_recs if r["month"] == month]
        print(f"  {label:35s}  {hit_stats(grp)}")

    # ── Factor F: Multi-model agreement ──────────────────────────────────────
    print(f"\n  F. Multi-model agreement count")
    print(f"     Intuition: when multiple independent models agree, forecast uncertainty is lower")
    print(hdr); print(sep)
    for n_agree in sorted(set(r["n_agree"] for r in hrrr_recs)):
        grp = [r for r in hrrr_recs if r["n_agree"] == n_agree]
        label = f"{n_agree} model{'s' if n_agree>1 else ''} agree"
        print(f"  {label:35s}  {hit_stats(grp)}")

    # ── Combined: best combination of factors ────────────────────────────────
    print(f"\n  G. Combined gate: entry ask ≥30¢ AND safe_margin ≥0.5")
    print(f"     Intuition: market + HRRR both confident; forecast solidly inside band")
    print(hdr); print(sep)
    for label, grp in [
        ("all HRRR trades",      hrrr_recs),
        ("ask ≥30¢",             [r for r in hrrr_recs if r["entry_ask"] >= 30]),
        ("margin ≥0.5",          [r for r in hrrr_recs if r["safe_margin"] is not None and r["safe_margin"] >= 0.5]),
        ("ask ≥30 + margin ≥0.5",[r for r in hrrr_recs if r["entry_ask"] >= 30
                                   and r["safe_margin"] is not None and r["safe_margin"] >= 0.5]),
        ("ask ≥30 + margin ≥0.5\n  + exclude mia/sea/msy",
                                  [r for r in hrrr_recs if r["entry_ask"] >= 30
                                   and r["safe_margin"] is not None and r["safe_margin"] >= 0.5
                                   and r["city"] not in {"mia","sea","msy"}]),
    ]:
        print(f"  {label:35s}  {hit_stats(grp)}")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 8: PT hit rate and time-to-PT — how quickly does the price move?
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  SECTION 8: PT hit rate and time-to-PT (HRRR-only, PT={BEST_PT}¢)")
    print(f"  Confirms whether edge comes from intraday price rise before settlement")
    print(f"{'='*80}")

    hit_recs  = [r for r in hrrr_recs if r["pt_hit_ts"][BEST_PT] is not None]
    miss_recs = [r for r in hrrr_recs if r["pt_hit_ts"][BEST_PT] is None]

    n_total = len(hrrr_recs)
    n_hit   = len(hit_recs)
    n_miss  = len(miss_recs)

    hit_pnls  = [r["pnl_by_pt"][BEST_PT] for r in hit_recs]
    miss_pnls = [r["pnl_by_pt"][BEST_PT] for r in miss_recs]

    print(f"\n  PT hit:  {n_hit:>3}/{n_total} ({100*n_hit/n_total:.1f}%)"
          f"  avg pnl={sum(hit_pnls)/n_hit:>+.1f}¢  total={sum(hit_pnls)/100:>+.2f}$")
    print(f"  PT miss: {n_miss:>3}/{n_total} ({100*n_miss/n_total:.1f}%)"
          f"  avg pnl={sum(miss_pnls)/n_miss:>+.1f}¢  total={sum(miss_pnls)/100:>+.2f}$")
    miss_yes  = sum(1 for r in miss_recs if r["won"])
    miss_no   = sum(1 for r in miss_recs if not r["won"])
    print(f"    → PT-miss breakdown: {miss_yes} settled YES (+{100-sum(r['entry_ask'] for r in miss_recs if not r['won'] and False):.0f}¢ avg)"
          f", {miss_no} settled NO (full loss)")
    # cleaner miss breakdown
    miss_yes_recs = [r for r in miss_recs if r["won"]]
    miss_no_recs  = [r for r in miss_recs if not r["won"]]
    if miss_yes_recs:
        avg_yes_pnl = sum(r["pnl_by_pt"][BEST_PT] for r in miss_yes_recs) / len(miss_yes_recs)
        print(f"    → No-PT YES settlements: n={len(miss_yes_recs)}  avg pnl={avg_yes_pnl:>+.1f}¢")
    if miss_no_recs:
        avg_no_pnl = sum(r["pnl_by_pt"][BEST_PT] for r in miss_no_recs) / len(miss_no_recs)
        print(f"    → No-PT NO  settlements: n={len(miss_no_recs)}  avg pnl={avg_no_pnl:>+.1f}¢")

    # Time-to-PT distribution (minutes from entry)
    mins_to_pt = [
        (r["pt_hit_ts"][BEST_PT] - r["entry_ts"]) // 60
        for r in hit_recs
    ]
    print(f"\n  Time-to-PT distribution (minutes after entry):")
    buckets = [(0,30),(30,60),(60,120),(120,240),(240,480),(480,960),(960,99999)]
    labels  = ["0–30m","30–60m","1–2h","2–4h","4–8h","8–16h","16h+"]
    print(f"  {'bucket':>8}  {'n':>4}  {'%':>5}  {'avg exit¢':>10}  {'avg pnl¢':>9}")
    print(f"  {'-'*50}")
    for (lo, hi), label in zip(buckets, labels):
        grp = [r for r, m in zip(hit_recs, mins_to_pt) if lo <= m < hi]
        if not grp:
            continue
        pnls = [r["pnl_by_pt"][BEST_PT] for r in grp]
        # reconstruct exit price from pnl + entry
        avg_exit = sum(r["entry_ask"] + r["pnl_by_pt"][BEST_PT] for r in grp) / len(grp)
        print(f"  {label:>8}  {len(grp):>4}  {100*len(grp)/n_hit:>4.1f}%"
              f"  {avg_exit:>9.1f}¢  {sum(pnls)/len(grp):>+8.1f}¢")

    avg_mins = sum(mins_to_pt) / len(mins_to_pt) if mins_to_pt else 0
    med_mins = sorted(mins_to_pt)[len(mins_to_pt)//2] if mins_to_pt else 0
    print(f"\n  avg time-to-PT: {avg_mins:.0f} min  |  median: {med_mins} min")

    # PT hit rate across all PT targets
    print(f"\n  PT hit rate by target (HRRR-only, n={n_total}):")
    print(f"  {'PT':>5}  {'hit':>4}  {'hit%':>6}  {'avg_pnl(hit)':>13}  {'avg_pnl(miss)':>14}  {'overall_avg':>12}")
    print(f"  {'-'*65}")
    for pt in PT_TARGETS:
        h_recs = [r for r in hrrr_recs if r["pt_hit_ts"][pt] is not None]
        m_recs = [r for r in hrrr_recs if r["pt_hit_ts"][pt] is None]
        n_h = len(h_recs); n_m = len(m_recs)
        avg_h = sum(r["pnl_by_pt"][pt] for r in h_recs) / n_h if n_h else 0
        avg_m = sum(r["pnl_by_pt"][pt] for r in m_recs) / n_m if n_m else 0
        avg_o = sum(r["pnl_by_pt"][pt] for r in hrrr_recs) / n_total
        print(f"  {pt:>4}¢  {n_h:>4}  {100*n_h/n_total:>5.1f}%"
              f"  {avg_h:>+12.1f}¢  {avg_m:>+13.1f}¢  {avg_o:>+11.1f}¢")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 8: Entry hour breakdown — when during the morning is edge strongest?
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  SECTION 8: Entry hour (UTC) breakdown — HRRR-only trades")
    print(f"  Hypothesis: earlier entries are more profitable (less informed competition)")
    print(f"{'='*80}")
    print(f"  {'hour(UTC)':>9}  {'≈ET':>5}  {'n':>4}  {'WR':>6}  {'entry':>5}  {'avg¢':>7}  {'total$':>8}  {'YES%':>7}")
    print(f"  {'-'*70}")

    ET_OFFSET = {7: "2–3am", 8: "3–4am", 9: "4–5am", 10: "5–6am", 11: "6–7am"}

    hours_seen = sorted(set(r["entry_hour"] for r in hrrr_recs))
    for h in hours_seen:
        recs = [r for r in hrrr_recs if r["entry_hour"] == h]
        pnls = [r["pnl_by_pt"][BEST_PT] for r in recs]
        n = len(recs)
        wins = sum(1 for p in pnls if p > 0)
        avg_entry = sum(r["entry_ask"] for r in recs) / n
        avg_pnl = sum(pnls) / n
        total = sum(pnls) / 100
        yes_rate = sum(1 for r in recs if r["won"]) / n
        et = ET_OFFSET.get(h, "?")
        print(f"  {h:>9}  {et:>5}  {n:>4}  {100*wins/n:>5.1f}%  {avg_entry:>4.1f}¢"
              f"  {avg_pnl:>+6.1f}¢  {total:>+8.2f}$  {100*yes_rate:>5.1f}%")

    # Cumulative: ≤H (entries up to and including hour H)
    print(f"\n  Cumulative — entries at or before hour H:")
    print(f"  {'≤hour':>6}  {'n':>4}  {'WR':>6}  {'avg¢':>7}  {'total$':>8}  {'YES%':>7}")
    print(f"  {'-'*52}")
    for h in hours_seen:
        recs = [r for r in hrrr_recs if r["entry_hour"] <= h]
        pnls = [r["pnl_by_pt"][BEST_PT] for r in recs]
        n = len(recs)
        wins = sum(1 for p in pnls if p > 0)
        yes_rate = sum(1 for r in recs if r["won"]) / n
        print(f"  {h:>6}  {n:>4}  {100*wins/n:>5.1f}%  {sum(pnls)/n:>+6.1f}¢"
              f"  {sum(pnls)/100:>+8.2f}$  {100*yes_rate:>5.1f}%")

    # Full PT sweep by hour
    print(f"\n  Full PT sweep by entry hour (HRRR-only):")
    print(f"  {'hour':>5}  " + "  ".join(f"PT{pt:>2}¢(avg)" for pt in PT_TARGETS))
    print(f"  {'-'*75}")
    for h in hours_seen:
        recs = [r for r in hrrr_recs if r["entry_hour"] == h]
        row = f"  {h:>5}  "
        row += "  ".join(
            f"{sum(r['pnl_by_pt'][pt] for r in recs)/len(recs):>+8.1f}¢"
            for pt in PT_TARGETS
        )
        print(row)

    # Same for all-source union
    print(f"\n  Entry hour breakdown — all-source union (≥1 agree):")
    print(f"  {'hour(UTC)':>9}  {'≈ET':>5}  {'n':>4}  {'WR':>6}  {'entry':>5}  {'avg¢':>7}  {'total$':>8}  {'YES%':>7}")
    print(f"  {'-'*70}")
    hours_seen_all = sorted(set(r["entry_hour"] for r in trade_recs))
    for h in hours_seen_all:
        recs = [r for r in trade_recs if r["entry_hour"] == h]
        pnls = [r["pnl_by_pt"][BEST_PT] for r in recs]
        n = len(recs)
        wins = sum(1 for p in pnls if p > 0)
        avg_entry = sum(r["entry_ask"] for r in recs) / n
        avg_pnl = sum(pnls) / n
        total = sum(pnls) / 100
        yes_rate = sum(1 for r in recs if r["won"]) / n
        et = ET_OFFSET.get(h, "?")
        print(f"  {h:>9}  {et:>5}  {n:>4}  {100*wins/n:>5.1f}%  {avg_entry:>4.1f}¢"
              f"  {avg_pnl:>+6.1f}¢  {total:>+8.2f}$  {100*yes_rate:>5.1f}%")


if __name__ == "__main__":
    main()
