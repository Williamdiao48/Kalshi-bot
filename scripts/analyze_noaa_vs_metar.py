"""Compare NWS observations API output vs METAR-precision readings.

Answers two questions:
  1. Does the NWS observations API (used by noaa_observed) return 5-minute
     synoptic readings in addition to :53 METAR readings?
  2. Do those synoptic readings use integer-°C rounding that inflates the
     apparent daily max vs. the true METAR precision value?

For each city, fetches the NWS observations API and classifies each reading:
  - "metar"   : timestamp minute NOT divisible by 5 (e.g. :53)
  - "synoptic": timestamp minute divisible by 5 (e.g. :50, :55, :00, :05)

Then computes per-day:
  - all_max_f     : max of ALL readings (what noaa_observed does today)
  - metar_max_f   : max of METAR-only readings
  - iem_max_f     : ground-truth integer-°F max from IEM ASOS
  - inflation_f   : all_max_f - metar_max_f

Also optionally pulls the local raw_forecasts DB to check whether
noaa_observed values in the bot's history changed at non-:53 times.

Usage:
  venv/bin/python scripts/analyze_noaa_vs_metar.py
  venv/bin/python scripts/analyze_noaa_vs_metar.py --cities dfw chi bos --days 14
  venv/bin/python scripts/analyze_noaa_vs_metar.py --db-only   # local DB analysis only
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sqlite3
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import aiohttp

sys.path.insert(0, str(Path(__file__).parent.parent))
from kalshi_bot.news.noaa import CITIES, KALSHI_STATION_IDS  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_NWS_OBS_URL = "https://api.weather.gov/stations/{station}/observations"
_IEM_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
_NWS_HEADERS = {"User-Agent": "kalshi-bot-research (github.com/kalshi-bot)"}
_FETCH_DELAY = 0.5  # seconds between requests — be polite


def _c_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def _is_metar_timestamp(ts: str) -> bool:
    """Return True if the observation timestamp looks like a METAR (:53 or non-5-min mark)."""
    try:
        dt = datetime.fromisoformat(ts)
        return dt.minute % 5 != 0
    except ValueError:
        return False


async def _fetch_nws_obs(
    session: aiohttp.ClientSession,
    station: str,
    start_utc: datetime,
    end_utc: datetime,
) -> list[dict]:
    """Fetch all NWS observations for a station between start and end UTC."""
    url = _NWS_OBS_URL.format(station=station)
    params = {
        "start": start_utc.isoformat(),
        "end":   end_utc.isoformat(),
        "limit": "200",
    }
    try:
        async with session.get(
            url, params=params, headers=_NWS_HEADERS,
            timeout=aiohttp.ClientTimeout(total=20),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data.get("features", [])
    except Exception as exc:
        log.warning("NWS obs fetch failed for %s: %s", station, exc)
        return []


async def _fetch_iem(
    session: aiohttp.ClientSession,
    station: str,
    day: date,
) -> float | None:
    """Fetch IEM ASOS daily high (°F integer) for one station and date."""
    params = {
        "station":     station,
        "data":        "tmpf",
        "year1":       str(day.year),
        "month1":      str(day.month),
        "day1":        str(day.day),
        "year2":       str(day.year),
        "month2":      str(day.month),
        "day2":        str(day.day),
        "tz":          "UTC",
        "format":      "comma",
        "latlon":      "no",
        "missing":     "M",
        "trace":       "T",
        "direct":      "no",
        "report_type": "3,4",
    }
    try:
        async with session.get(
            _IEM_URL, params=params,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            resp.raise_for_status()
            text = await resp.text()
    except Exception as exc:
        log.warning("IEM fetch failed for %s %s: %s", station, day, exc)
        return None

    max_f: float | None = None
    for line in text.splitlines():
        if line.startswith("#") or line.startswith("station") or not line.strip():
            continue
        parts = line.split(",")
        if len(parts) < 3:
            continue
        try:
            temp_str = parts[2].strip()
            if temp_str in ("M", "T", ""):
                continue
            t = float(temp_str)
            if max_f is None or t > max_f:
                max_f = t
        except (ValueError, IndexError):
            continue
    return max_f


def _analyze_nws_features(
    features: list[dict],
    day_start_utc: datetime,
    day_end_utc: datetime,
) -> dict:
    """
    Classify NWS observations for a single day window.
    Returns dict with all_max_f, metar_max_f, synoptic_max_f, n_metar, n_synoptic,
    and a list of (ts, temp_f, kind) for all readings.
    """
    readings: list[tuple[str, float, str]] = []

    for feature in features:
        props = feature.get("properties") or {}
        ts = props.get("timestamp", "")
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(ts)
        except ValueError:
            continue
        # Keep only readings within the day window
        if not (day_start_utc <= dt < day_end_utc):
            continue
        temp = props.get("temperature") or {}
        temp_c = temp.get("value")
        if temp_c is None:
            continue
        temp_f = _c_to_f(temp_c)
        kind = "metar" if _is_metar_timestamp(ts) else "synoptic"
        readings.append((ts, temp_f, kind))

    if not readings:
        return {"all_max_f": None, "metar_max_f": None, "synoptic_max_f": None,
                "n_metar": 0, "n_synoptic": 0, "readings": []}

    metar_temps   = [t for _, t, k in readings if k == "metar"]
    synoptic_temps = [t for _, t, k in readings if k == "synoptic"]

    return {
        "all_max_f":      max(t for _, t, _ in readings),
        "metar_max_f":    max(metar_temps)   if metar_temps   else None,
        "synoptic_max_f": max(synoptic_temps) if synoptic_temps else None,
        "n_metar":        len(metar_temps),
        "n_synoptic":     len(synoptic_temps),
        "readings":       sorted(readings),
    }


async def analyze_city(
    session: aiohttp.ClientSession,
    metric: str,
    station: str,
    city_tz: ZoneInfo,
    days: int,
    threshold_f: float,
) -> list[dict]:
    """Fetch and analyze NWS observations for one city over the past `days` days."""
    results = []
    today_local = datetime.now(city_tz).date()

    for delta in range(1, days + 1):
        day = today_local - timedelta(days=delta)

        # Compute day window in UTC using LST midnight (same logic as noaa_observed)
        local_midnight = datetime(day.year, day.month, day.day, tzinfo=city_tz)
        sample_dt = local_midnight + timedelta(hours=12)
        dst_offset = sample_dt.dst() or timedelta(0)
        lst_hour = 1 if dst_offset.total_seconds() > 0 else 0
        day_start_local = local_midnight.replace(hour=lst_hour)
        day_end_local   = day_start_local + timedelta(hours=24)
        day_start_utc   = day_start_local.astimezone(timezone.utc)
        day_end_utc     = day_end_local.astimezone(timezone.utc)

        features = await _fetch_nws_obs(session, station, day_start_utc, day_end_utc)
        await asyncio.sleep(_FETCH_DELAY)

        analysis = _analyze_nws_features(features, day_start_utc, day_end_utc)
        iem_max  = await _fetch_iem(session, station, day)
        await asyncio.sleep(_FETCH_DELAY)

        all_max    = analysis["all_max_f"]
        metar_max  = analysis["metar_max_f"]
        inflation  = (all_max - metar_max) if (all_max is not None and metar_max is not None) else None
        false_cross = (
            metar_max is not None
            and all_max is not None
            and all_max >= threshold_f
            and metar_max < threshold_f
        )

        results.append({
            "metric":      metric,
            "station":     station,
            "date":        day.isoformat(),
            "all_max_f":   round(all_max, 2)   if all_max   is not None else None,
            "metar_max_f": round(metar_max, 2) if metar_max is not None else None,
            "synoptic_max_f": round(analysis["synoptic_max_f"], 2) if analysis["synoptic_max_f"] is not None else None,
            "iem_max_f":   iem_max,
            "inflation_f": round(inflation, 2) if inflation is not None else None,
            "n_metar":     analysis["n_metar"],
            "n_synoptic":  analysis["n_synoptic"],
            "false_cross": false_cross,
        })

        log.info(
            "  %s %s  all=%.1f  metar=%.1f  iem=%.1f  inflation=%+.2f  %s",
            metric, day,
            all_max   or 0,
            metar_max or 0,
            iem_max   or 0,
            inflation or 0,
            "*** FALSE CROSS ***" if false_cross else "",
        )

    return results


def analyze_db(db_path: Path, metric: str) -> None:
    """
    Check whether noaa_observed values in raw_forecasts change at non-:53 times.

    For each (metric, date), pulls all noaa_observed logged_at / data_value pairs
    and reports any increase in data_value that happened at a non-:53 timestamp.
    """
    if not db_path.exists():
        log.error("DB not found: %s", db_path)
        return

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT logged_at, data_value
        FROM raw_forecasts
        WHERE source = 'noaa_observed'
          AND metric = ?
        ORDER BY logged_at
        """,
        (metric,),
    ).fetchall()
    conn.close()

    if not rows:
        log.warning("No noaa_observed rows found for %s", metric)
        return

    log.info("\n=== DB analysis: noaa_observed for %s ===", metric)
    prev_val: float | None = None
    prev_ts:  str          = ""
    changes_at_non53 = 0
    changes_at_53    = 0

    for ts_str, val in rows:
        if prev_val is not None and val > prev_val:
            try:
                dt = datetime.fromisoformat(ts_str)
                minute = dt.minute
                is_53  = (minute % 5 != 0)
                tag    = ":53-type" if is_53 else "5-min-synoptic"
                log.info(
                    "  VALUE INCREASE at %s (%s): %.2f → %.2f  (+%.2f°F)",
                    ts_str, tag, prev_val, val, val - prev_val,
                )
                if is_53:
                    changes_at_53 += 1
                else:
                    changes_at_non53 += 1
            except ValueError:
                pass
        prev_val = val
        prev_ts  = ts_str

    log.info(
        "  Summary: %d increases at :53-type timestamps, "
        "%d increases at 5-min synoptic timestamps",
        changes_at_53, changes_at_non53,
    )
    if changes_at_non53 > 0:
        log.info("  → CONFIRMED: noaa_observed picks up 5-minute synoptic readings")
    else:
        log.info("  → noaa_observed only changed at :53-type timestamps in this data")


async def main(
    city_suffixes: list[str],
    days: int,
    threshold_f: float,
    db_only: bool,
    db_path: Path,
) -> None:
    pairs = [
        (metric, station)
        for metric, station in KALSHI_STATION_IDS.items()
        if not city_suffixes or metric.replace("temp_high_", "") in city_suffixes
    ]

    if not pairs:
        log.error("No matching cities. Available: %s",
                  [m.replace("temp_high_", "") for m in KALSHI_STATION_IDS])
        return

    # --- Local DB analysis (always runs for the first city) ---
    for metric, _ in pairs[:3]:
        analyze_db(db_path, metric)

    if db_only:
        return

    # --- NWS API + IEM analysis ---
    log.info("\nFetching NWS observations + IEM ground truth for %d city(ies), %d days each",
             len(pairs), days)

    all_results: list[dict] = []
    async with aiohttp.ClientSession() as session:
        for i, (metric, station) in enumerate(pairs):
            city_entry = CITIES.get(metric)
            if city_entry is None:
                continue
            city_name, _, _, city_tz = city_entry
            log.info("[%d/%d] %s (%s)", i + 1, len(pairs), metric, station)
            results = await analyze_city(session, metric, station, city_tz, days, threshold_f)
            all_results.extend(results)
            if i < len(pairs) - 1:
                await asyncio.sleep(_FETCH_DELAY)

    # --- Summary ---
    valid = [r for r in all_results if r["inflation_f"] is not None]
    if not valid:
        log.warning("No valid results")
        return

    inflated  = [r for r in valid if (r["inflation_f"] or 0) > 0.01]
    false_crosses = [r for r in valid if r["false_cross"]]

    print("\n=== Summary ===")
    print(f"City-days analyzed : {len(valid)}")
    print(f"Days with synoptic inflation > 0.01°F : {len(inflated)} ({100*len(inflated)/len(valid):.0f}%)")
    print(f"Days with false threshold cross        : {len(false_crosses)}")
    if inflated:
        infl_vals = [r["inflation_f"] for r in inflated]
        print(f"Inflation range    : {min(infl_vals):.2f} – {max(infl_vals):.2f}°F")
        print(f"Inflation mean     : {sum(infl_vals)/len(infl_vals):.2f}°F")

    print("\n=== Days where synoptic > metar (inflation > 0.01°F) ===")
    print(f"{'City':<20} {'Date':<12} {'all_max':>8} {'metar_max':>10} {'iem_max':>8} {'inflation':>10} {'false_cross':>12}")
    for r in sorted(inflated, key=lambda x: -(x["inflation_f"] or 0)):
        print(
            f"{r['metric']:<20} {r['date']:<12} "
            f"{r['all_max_f']:>8.2f} {r['metar_max_f']:>10.2f} "
            f"{str(r['iem_max_f'] or '?'):>8} "
            f"{r['inflation_f']:>10.2f} "
            f"{'YES ***' if r['false_cross'] else '':>12}"
        )

    if false_crosses:
        print(f"\n*** {len(false_crosses)} false threshold crossing(s) detected ***")
        print(f"    Threshold: {threshold_f}°F")
        print("    These are cases where noaa_observed would trigger band_arb")
        print("    but the actual METAR max is below the trigger threshold.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare NWS observations API vs METAR precision for noaa_observed accuracy"
    )
    parser.add_argument("--cities", nargs="+", default=None,
                        help="City suffixes to analyze e.g. dfw chi bos (default: all)")
    parser.add_argument("--days", type=int, default=7,
                        help="Number of past days to analyze (default: 7)")
    parser.add_argument("--threshold", type=float, default=69.5,
                        help="Band-arb trigger threshold in °F (default: 69.5)")
    parser.add_argument("--db-only", action="store_true",
                        help="Only run local DB analysis, skip NWS API fetch")
    parser.add_argument("--db", default="opportunity_log.db",
                        help="Path to SQLite DB (default: opportunity_log.db)")
    args = parser.parse_args()

    asyncio.run(main(
        city_suffixes=args.cities or [],
        days=args.days,
        threshold_f=args.threshold,
        db_only=args.db_only,
        db_path=Path(args.db),
    ))
