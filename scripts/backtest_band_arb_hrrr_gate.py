"""Backtest the HRRR gate for band-arb YES signals.

Classifies historical band-arb YES opportunities into two groups:

  Group A (dangerous): HRRR daily high >= band ceiling AND margin_hi > threshold
    Physical meaning: HRRR says warming continues past the ceiling while the
    running max is still meaningfully below it — a "late-afternoon breakthrough"
    setup. The bot entered at the bottom of the band but the peak is still
    coming and will push past the top.

  Group B (safe): HRRR daily high < band ceiling OR margin_hi <= threshold
    Physical meaning: either HRRR doesn't expect a ceiling breach, or the running
    max is already near the ceiling (quantization artifact, peak already happened).

Entry is simulated at the P75 peak-hour gate ± window (matching live bot behavior):
  P75_MINUTES from data/peak_hour_p90.py gives per-city per-month peak time.
  We look at the first qualifying hour within [P75 - window, P75 + window].

Date convention note:
  kxhigh_bands.csv stores the UTC market close date (= local measurement date + 1).
  mesonet_hourly.csv and openmeteo_forecasts.csv both use the LOCAL measurement date.
  All data lookups subtract 1 day from the band date.

Data dependencies (run first if stale):
  venv/bin/python scripts/fetch_mesonet_history.py --days 120
  venv/bin/python scripts/fetch_kxhigh_history.py --days 120
  venv/bin/python scripts/fetch_openmeteo_forecast_history.py --days 120 --models gfs_hrrr

Usage:
  venv/bin/python scripts/backtest_band_arb_hrrr_gate.py
  venv/bin/python scripts/backtest_band_arb_hrrr_gate.py --margin-threshold 0.5
  venv/bin/python scripts/backtest_band_arb_hrrr_gate.py --window 3
  venv/bin/python scripts/backtest_band_arb_hrrr_gate.py --cities dca chi hou msp
  venv/bin/python scripts/backtest_band_arb_hrrr_gate.py --sweep-thresholds
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from kalshi_bot.cities import CITIES  # noqa: E402

DATA_DIR = Path(__file__).parent.parent / "data"
NO_EDGE_THRESHOLD = 95  # skip markets where YES ask ≥ 95¢ (fully priced)
BOT_MAX_YES_ASK   = 85  # live bot only enters if YES ask ≤ this (BAND_ARB_YES_MAX_YES_ASK)


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_mesonet() -> dict[tuple[str, str, int], float]:
    """Load mesonet_hourly.csv → {(metric, LOCAL_date, local_hour): running_max_f}."""
    path = DATA_DIR / "mesonet_hourly.csv"
    if not path.exists():
        print(f"ERROR: {path} not found. Run fetch_mesonet_history.py first.")
        sys.exit(1)
    data: dict[tuple[str, str, int], float] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            data[(row["city_metric"], row["date"], int(row["local_hour"]))] = float(row["running_max_f"])
    return data


def load_bands() -> list[dict]:
    """Load kxhigh_bands.csv, filter to 'between' direction.

    NOTE: band["date"] is the UTC close date. The local measurement date is
    band["date"] - 1 day. All data lookups must use that offset.
    """
    path = DATA_DIR / "kxhigh_bands.csv"
    if not path.exists():
        print(f"ERROR: {path} not found. Run fetch_kxhigh_history.py first.")
        sys.exit(1)
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("direction", "between") != "between":
                continue
            row["strike_lo"] = float(row["strike_lo"])
            row["strike_hi"] = float(row["strike_hi"])
            # Pre-compute the local measurement date (one day before UTC close date)
            utc_close = date.fromisoformat(row["date"])
            row["local_date"] = (utc_close - timedelta(days=1)).isoformat()
            row["month"] = (utc_close - timedelta(days=1)).month
            rows.append(row)
    return rows


def load_hrrr_forecasts() -> dict[tuple[str, str], float]:
    """Load openmeteo_forecasts.csv (gfs_hrrr) → {(metric, LOCAL_date): forecast_high_f}."""
    path = DATA_DIR / "openmeteo_forecasts.csv"
    if not path.exists():
        print(f"ERROR: {path} not found. Run fetch_openmeteo_forecast_history.py first.")
        sys.exit(1)
    data: dict[tuple[str, str], float] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("model") != "gfs_hrrr":
                continue
            data[(row["city_metric"], row["date"])] = float(row["forecast_high_f"])
    return data


def load_candle_cache() -> dict[str, dict[int, float]]:
    """Load band_arb_candle_cache.json → {ticker: {local_hour: yes_ask_cents}}.

    Candles are converted to per-local-hour YES ask using the city timezone.
    Only tickers with actual candle data are included (empty → absent).
    """
    path = DATA_DIR / "band_arb_candle_cache.json"
    if not path.exists():
        return {}
    raw: dict[str, list[dict]] = json.loads(path.read_text())

    result: dict[str, dict[int, float]] = {}
    for ticker, candles in raw.items():
        if not candles:
            continue
        # Determine city timezone from ticker → metric
        # Ticker prefix maps to metric via CITIES keys. Try to infer metric.
        metric = _ticker_to_metric(ticker)
        city_entry = CITIES.get(metric) if metric else None
        city_tz    = city_entry[3] if city_entry else None

        hourly: dict[int, float] = {}
        for c in candles:
            ask_obj   = c.get("yes_ask") or {}
            close_str = ask_obj.get("close_dollars")
            if close_str is None:
                continue
            try:
                ask_cents = round(float(close_str) * 100)
            except (ValueError, TypeError):
                continue
            ts = datetime.fromtimestamp(c["end_period_ts"], tz=timezone.utc)
            if city_tz:
                local_hour = ts.astimezone(city_tz).hour
            else:
                local_hour = ts.hour
            hourly[local_hour] = ask_cents

        if hourly:
            result[ticker] = hourly
    return result


# City-key suffix → metric (temp_high_*)
_SUFFIX_TO_METRIC: dict[str, str] = {
    k.replace("temp_high_", ""): k for k in CITIES if k.startswith("temp_high_")
}

# Partial ticker prefix → metric suffix mapping (lower-case ticker prefix lookup)
_TICKER_PREFIX_MAP: dict[str, str] = {
    "kxhighaus":   "aus", "kxhightchi":  "chi", "kxhighchi":   "chi",
    "kxhighden":   "den", "kxhighlax":   "lax", "kxhighmia":   "mia",
    "kxhighny":    "ny",  "kxhightatl":  "atl", "kxhightatl":  "atl",
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


def _ask_at_hour(hourly: dict[int, float], hour: int) -> float | None:
    """Look up ask at given hour, falling back ±1 and ±2."""
    for delta in [0, 1, -1, 2, -2]:
        v = hourly.get(hour + delta)
        if v is not None:
            return v
    return None


def load_p75_minutes() -> dict[str, dict[int, int]]:
    """Load P75_MINUTES from data/peak_hour_p90.py → {metric: {month: minutes}}."""
    path = DATA_DIR / "peak_hour_p90.py"
    if not path.exists():
        print(f"ERROR: {path} not found. Run backtest_peak_hour.py --dict first.")
        sys.exit(1)
    spec = importlib.util.spec_from_file_location("peak_hour_p90", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return getattr(mod, "P75_MINUTES", {})


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate(
    mesonet:          dict[tuple[str, str, int], float],
    bands:            list[dict],
    hrrr:             dict[tuple[str, str], float],
    p75_minutes:      dict[str, dict[int, int]],
    candles:          dict[str, dict[int, float]],
    *,
    margin_threshold: float = 0.5,
    window:           int   = 2,
    min_offset:       int   = 0,      # only include entries at P75 + min_offset or later
    city_filter:      list[str] | None = None,
    max_yes_ask:      int | None = None,  # if set, skip bands where entry ask > this
    require_price:    bool = False,        # if True, skip bands with no candle price data
) -> list[dict]:
    """Return one record per qualifying band.

    For each resolved 'between' band:
      1. Compute P75 gate hour for that city × month.
      2. Search hours [p75 - window, p75 + window] for the first hour where the
         running max is inside the band [band_lo, band_hi].
      3. Classify using gfs_hrrr forecast: Group A if forecast >= band_hi AND
         margin_hi (= band_hi - running_max) > margin_threshold.

    Date offset: mesonet/hrrr are keyed by LOCAL date = band["local_date"].
    """
    records: list[dict] = []

    for band in bands:
        metric     = band["metric"]
        local_date = band["local_date"]
        month      = band["month"]
        band_lo    = band["strike_lo"]
        band_hi    = band["strike_hi"]
        result     = band["result"]
        ticker     = band["ticker"]

        if city_filter:
            suffix = metric.replace("temp_high_", "")
            if suffix not in city_filter:
                continue

        # P75 gate hour for this city × month
        monthly  = p75_minutes.get(metric, {})
        gate_min = monthly.get(month)
        if gate_min is None:
            continue
        p75_hour = gate_min // 60

        # Search [p75 + min_offset, p75 + window] for first in-band hour
        entry_hour: int | None = None
        entry_rm:   float | None = None
        for h in range(p75_hour + min_offset, p75_hour + window + 1):
            rm = mesonet.get((metric, local_date, h))
            if rm is not None and band_lo <= rm <= band_hi:
                entry_hour = h
                entry_rm   = rm
                break

        if entry_hour is None:
            continue

        offset    = entry_hour - p75_hour
        margin_hi = band_hi - entry_rm

        hrrr_fc        = hrrr.get((metric, local_date))
        hrrr_available = hrrr_fc is not None

        if hrrr_available:
            group_a = (hrrr_fc >= band_hi) and (margin_hi > margin_threshold)
        else:
            group_a = False

        # Price lookup from candle cache
        hourly_ask = candles.get(ticker, {})
        yes_ask    = _ask_at_hour(hourly_ask, entry_hour)
        has_price  = yes_ask is not None

        # Require price: skip bands with no candle data (makes WR = priced WR)
        if require_price and not has_price:
            continue

        # Entry-price filter: skip bands the live bot would not enter.
        # With max_yes_ask set (e.g. 85), only analyse bands where the market
        # hadn't already fully priced in the outcome at entry time.
        if max_yes_ask is not None and has_price and yes_ask > max_yes_ask:
            continue

        # P&L: buy YES at ask, collect 100 if YES resolves, 0 if NO
        won = result == "yes"
        if has_price and yes_ask < NO_EDGE_THRESHOLD and yes_ask > 0:
            pnl_cents = (100 - yes_ask) if won else (-yes_ask)
        else:
            pnl_cents = None  # no price → exclude from P&L totals

        city = metric.replace("temp_high_", "")

        records.append({
            "metric":         metric,
            "city":           city,
            "ticker":         ticker,
            "local_date":     local_date,
            "month":          month,
            "band_lo":        band_lo,
            "band_hi":        band_hi,
            "p75_hour":       p75_hour,
            "entry_hour":     entry_hour,
            "hours_offset":   offset,
            "running_max":    entry_rm,
            "margin_hi":      round(margin_hi, 2),
            "hrrr_fc":        hrrr_fc,
            "hrrr_available": hrrr_available,
            "group":          "A" if group_a else "B",
            "result":         result,
            "won":            won,
            "yes_ask":        yes_ask,
            "pnl_cents":      pnl_cents,
        })

    return records


# ── Statistics ────────────────────────────────────────────────────────────────

def chi2_pvalue(n_a: int, win_a: int, n_b: int, win_b: int) -> float | None:
    if n_a < 5 or n_b < 5:
        return None
    total  = n_a + n_b
    wins   = win_a + win_b
    losses = total - wins
    if wins == 0 or losses == 0:
        return None
    p_pool = wins / total

    def term(o: int, e: float) -> float:
        return (o - e) ** 2 / e if e > 1e-9 else 0.0

    chi2 = (
        term(win_a,        n_a * p_pool)
        + term(n_a - win_a, n_a * (1 - p_pool))
        + term(win_b,        n_b * p_pool)
        + term(n_b - win_b,  n_b * (1 - p_pool))
    )
    return 1.0 - math.erf(math.sqrt(chi2 / 2))


def _sig(p: float | None) -> str:
    if p is None:
        return "(n<5)"
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return f"(p={p:.2f})"


# ── Reporting helpers ─────────────────────────────────────────────────────────

def _stats(recs: list[dict]) -> dict:
    n    = len(recs)
    wins = sum(1 for r in recs if r["won"])
    wr   = wins / n if n else float("nan")
    avg_m = sum(r["margin_hi"] for r in recs) / n if n else float("nan")
    fc_vals = [r["hrrr_fc"] for r in recs if r["hrrr_fc"] is not None]
    avg_fc  = sum(fc_vals) / len(fc_vals) if fc_vals else float("nan")

    priced = [r for r in recs if r.get("pnl_cents") is not None]
    n_priced  = len(priced)
    avg_ask   = sum(r["yes_ask"] for r in priced) / n_priced if priced else float("nan")
    avg_pnl   = sum(r["pnl_cents"] for r in priced) / n_priced if priced else float("nan")
    total_pnl = sum(r["pnl_cents"] for r in priced)

    return {
        "n": n, "wins": wins, "losses": n - wins,
        "win_rate": wr, "avg_margin": avg_m, "avg_hrrr": avg_fc,
        "n_priced": n_priced, "avg_ask": avg_ask,
        "avg_pnl": avg_pnl, "total_pnl": total_pnl,
    }


def _pct(v: float) -> str:
    return "—" if v != v else f"{v*100:.1f}%"


def _f(v: float, decimals: int = 2) -> str:
    return "—" if v != v else f"{v:.{decimals}f}"


# ── Main report ───────────────────────────────────────────────────────────────

def _pnl_str(s: dict) -> str:
    if s["n_priced"] == 0:
        return "—"
    return f"{s['avg_pnl']:+.1f}¢"


def _total_pnl_str(s: dict) -> str:
    if s["n_priced"] == 0:
        return "—"
    return f"${s['total_pnl']/100:+.2f}"


def build_report(records: list[dict], margin_threshold: float, window: int,
                 min_offset: int) -> str:
    lines: list[str] = []

    ga = [r for r in records if r["group"] == "A"]
    gb = [r for r in records if r["group"] == "B"]
    sa = _stats(ga)
    sb = _stats(gb)
    sa_all = _stats(records)

    p = chi2_pvalue(sa["n"], sa["wins"], sb["n"], sb["wins"])

    n_priced_total = sa_all["n_priced"]
    offset_desc = f"P75+{min_offset}h to P75+{window}h" if min_offset >= 0 else f"P75±{window}h"

    lines += [
        "",
        "=" * 76,
        "  BAND-ARB YES: HRRR GATE BACKTEST",
        f"  margin_threshold={margin_threshold}°F  entry window={offset_desc}  "
        f"n={len(records)}  priced={n_priced_total}",
        "=" * 76,
        "",
        f"  {'':32}  {'Group A':>11}  {'Group B':>11}  {'All':>8}",
        f"  {'':32}  {'HRRR≥ceil&marg>thr':>11}  {'(else)':>11}  {'':>8}",
        "  " + "-" * 70,
        f"  {'Observations':32}  {sa['n']:>11}  {sb['n']:>11}  {len(records):>8}",
        f"  {'Wins':32}  {sa['wins']:>11}  {sb['wins']:>11}",
        f"  {'Losses':32}  {sa['losses']:>11}  {sb['losses']:>11}",
        f"  {'Win rate':32}  {_pct(sa['win_rate']):>11}  {_pct(sb['win_rate']):>11}  "
        f"{_pct(sa_all['win_rate']):>8}",
        f"  {'Avg YES ask (priced subset)':32}  {_f(sa['avg_ask'],1):>10}¢  "
        f"{_f(sb['avg_ask'],1):>10}¢  {_f(sa_all['avg_ask'],1):>7}¢",
        f"  {'Avg P&L per trade':32}  {_pnl_str(sa):>11}  {_pnl_str(sb):>11}  "
        f"{_pnl_str(sa_all):>8}",
        f"  {'Total P&L ($1/contract)':32}  {_total_pnl_str(sa):>11}  "
        f"{_total_pnl_str(sb):>11}  {_total_pnl_str(sa_all):>8}",
        f"  {'Avg margin to ceiling':32}  {_f(sa['avg_margin']):>10}°F  "
        f"{_f(sb['avg_margin']):>10}°F",
        "",
        f"  Chi-squared significance: {_sig(p)}",
    ]

    # ── By offset hour (the main table the user asked for) ────────────────────
    lines += ["", "=" * 76,
              "  BY HOURS OFFSET FROM P75 GATE",
              "  (0 = entered exactly at P75, +1 = one hour after P75, etc.)",
              "=" * 76,
              f"  {'Off':>4}  {'A n':>5}  {'A win%':>7}  {'A avg¢':>7}  {'A P&L':>7}  "
              f"  {'B n':>5}  {'B win%':>7}  {'B avg¢':>7}  {'B P&L':>7}  "
              f"  {'All n':>5}  {'All win%':>8}  {'Sig':>7}",
              "  " + "-" * 86]
    for offset in sorted({r["hours_offset"] for r in records}):
        hr = [r for r in records if r["hours_offset"] == offset]
        ha = [r for r in hr if r["group"] == "A"]
        hb = [r for r in hr if r["group"] == "B"]
        sa_h = _stats(ha); sb_h = _stats(hb); all_h = _stats(hr)
        p_h  = chi2_pvalue(sa_h["n"], sa_h["wins"], sb_h["n"], sb_h["wins"])
        lines.append(
            f"  {offset:>+4}  {sa_h['n']:>5}  {_pct(sa_h['win_rate']):>7}  "
            f"{_f(sa_h['avg_ask'],1):>7}  {_pnl_str(sa_h):>7}  "
            f"  {sb_h['n']:>5}  {_pct(sb_h['win_rate']):>7}  "
            f"{_f(sb_h['avg_ask'],1):>7}  {_pnl_str(sb_h):>7}  "
            f"  {all_h['n']:>5}  {_pct(all_h['win_rate']):>8}  {_sig(p_h):>7}"
        )

    # ── By city ──────────────────────────────────────────────────────────────
    lines += ["", "=" * 76, "  BY CITY", "=" * 76,
              f"  {'City':<6}  {'A n':>5}  {'A win%':>7}  {'A P&L':>7}  "
              f"  {'B n':>5}  {'B win%':>7}  {'B P&L':>7}  "
              f"  {'All':>5}  {'All win%':>8}  {'Sig':>7}",
              "  " + "-" * 74]
    for city in sorted({r["city"] for r in records}):
        cr = [r for r in records if r["city"] == city]
        ca = [r for r in cr if r["group"] == "A"]
        cb = [r for r in cr if r["group"] == "B"]
        sa_c = _stats(ca); sb_c = _stats(cb); all_c = _stats(cr)
        p_c  = chi2_pvalue(sa_c["n"], sa_c["wins"], sb_c["n"], sb_c["wins"])
        lines.append(
            f"  {city:<6}  {sa_c['n']:>5}  {_pct(sa_c['win_rate']):>7}  "
            f"{_pnl_str(sa_c):>7}  "
            f"  {sb_c['n']:>5}  {_pct(sb_c['win_rate']):>7}  "
            f"{_pnl_str(sb_c):>7}  "
            f"  {all_c['n']:>5}  {_pct(all_c['win_rate']):>8}  {_sig(p_c):>7}"
        )

    # ── By month ─────────────────────────────────────────────────────────────
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    lines += ["", "=" * 76, "  BY MONTH", "=" * 76,
              f"  {'Mo':<4}  {'A n':>5}  {'A win%':>7}  {'A P&L':>7}  "
              f"  {'B n':>5}  {'B win%':>7}  {'B P&L':>7}  "
              f"  {'All':>5}  {'All win%':>8}  {'Sig':>7}",
              "  " + "-" * 70]
    for month in sorted({r["month"] for r in records}):
        mr = [r for r in records if r["month"] == month]
        ma = [r for r in mr if r["group"] == "A"]
        mb = [r for r in mr if r["group"] == "B"]
        sa_m = _stats(ma); sb_m = _stats(mb); all_m = _stats(mr)
        p_m  = chi2_pvalue(sa_m["n"], sa_m["wins"], sb_m["n"], sb_m["wins"])
        lines.append(
            f"  {month_names.get(month, str(month)):<4}  {sa_m['n']:>5}  "
            f"{_pct(sa_m['win_rate']):>7}  {_pnl_str(sa_m):>7}  "
            f"  {sb_m['n']:>5}  {_pct(sb_m['win_rate']):>7}  {_pnl_str(sb_m):>7}  "
            f"  {all_m['n']:>5}  {_pct(all_m['win_rate']):>8}  {_sig(p_m):>7}"
        )

    lines.append("")
    return "\n".join(lines)


def sweep_thresholds(
    mesonet:     dict,
    bands:       list[dict],
    hrrr:        dict,
    p75_minutes: dict,
    candles:     dict,
    *,
    window:      int,
    min_offset:  int,
    city_filter: list[str] | None,
    max_yes_ask: int | None,
    require_price: bool = False,
) -> str:
    thresholds = [0.1, 0.3, 0.5, 0.9, 1.2, 1.5]
    lines = ["", "=" * 76,
             "  THRESHOLD SWEEP — win rate and P&L by margin_threshold", "=" * 76,
             f"  {'Threshold':>10}  {'A (n)':>7}  {'A win%':>8}  {'A P&L':>7}  "
             f"  {'B (n)':>7}  {'B win%':>8}  {'B P&L':>7}  {'Sig':>8}",
             "  " + "-" * 72]
    for thresh in thresholds:
        recs = simulate(mesonet, bands, hrrr, p75_minutes, candles,
                        margin_threshold=thresh, window=window,
                        min_offset=min_offset, city_filter=city_filter,
                        max_yes_ask=max_yes_ask, require_price=require_price)
        ga = [r for r in recs if r["group"] == "A"]
        gb = [r for r in recs if r["group"] == "B"]
        sa = _stats(ga); sb = _stats(gb)
        p  = chi2_pvalue(sa["n"], sa["wins"], sb["n"], sb["wins"])
        lines.append(
            f"  {thresh:>10.1f}  {sa['n']:>7}  {_pct(sa['win_rate']):>8}  "
            f"{_pnl_str(sa):>7}  "
            f"  {sb['n']:>7}  {_pct(sb['win_rate']):>8}  {_pnl_str(sb):>7}  "
            f"{_sig(p):>8}"
        )
    lines.append("")
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest HRRR gate for band-arb YES signals.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--margin-threshold", type=float, default=0.5)
    parser.add_argument("--window", type=int, default=2,
                        help="Max hours after P75 to search for entry (default: 2)")
    parser.add_argument("--min-offset", type=int, default=0,
                        help="Min hours after P75 for entry (default: 0 = P75 and later)")
    parser.add_argument("--cities", nargs="+", default=None)
    parser.add_argument("--sweep-thresholds", action="store_true")
    parser.add_argument(
        "--max-yes-ask", type=int, default=None,
        help=(
            f"Only include bands where entry YES ask ≤ this value (¢). "
            f"Default: no filter (uses NO_EDGE_THRESHOLD={NO_EDGE_THRESHOLD}). "
            f"Set to {BOT_MAX_YES_ASK} to match the live bot (BAND_ARB_YES_MAX_YES_ASK)."
        ),
    )
    parser.add_argument("--require-price", action="store_true",
                        help="Only include bands that have candle price data (WR = priced WR)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    max_yes_ask = args.max_yes_ask  # None = no extra filter beyond NO_EDGE_THRESHOLD

    print("Loading data…", flush=True)
    mesonet     = load_mesonet()
    bands       = load_bands()
    hrrr        = load_hrrr_forecasts()
    p75_minutes = load_p75_minutes()
    candles     = load_candle_cache()

    print(f"  Mesonet rows:    {len(mesonet):,}")
    print(f"  Band markets:    {len(bands):,}  (between only)")
    print(f"  HRRR city-days:  {len(hrrr):,}")
    print(f"  Tickers w/price: {len(candles):,}")
    if max_yes_ask is not None:
        print(f"  Entry filter:    YES ask ≤ {max_yes_ask}¢  (bot would not enter above this)")

    records = simulate(
        mesonet, bands, hrrr, p75_minutes, candles,
        margin_threshold=args.margin_threshold,
        window=args.window,
        min_offset=args.min_offset,
        city_filter=args.cities,
        max_yes_ask=max_yes_ask,
        require_price=args.require_price,
    )

    n_priced  = sum(1 for r in records if r["pnl_cents"] is not None)
    n_no_hrrr = sum(1 for r in records if not r["hrrr_available"])
    print(f"  Signal observations: {len(records):,}")
    print(f"  With price data:     {n_priced:,}")
    print(f"  No-HRRR entries:     {n_no_hrrr:,}  (treated as Group B)")

    report = build_report(records, args.margin_threshold, args.window, args.min_offset)

    if args.sweep_thresholds:
        report += sweep_thresholds(
            mesonet, bands, hrrr, p75_minutes, candles,
            window=args.window, min_offset=args.min_offset,
            city_filter=args.cities,
            max_yes_ask=max_yes_ask,
            require_price=args.require_price,
        )

    print(report)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report)
        print(f"Report written to {out}")


if __name__ == "__main__":
    main()
