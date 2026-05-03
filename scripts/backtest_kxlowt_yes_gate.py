"""Find optimal (hours_to_close X, floor_clearance Y) gate for KXLOWT between-YES signals.

Background
----------
A between-YES signal on KXLOWT fires when the METAR running daily minimum
has dropped into the target band [strike_lo-0.5, strike_hi+0.5].  The bot
bets YES = the final daily low stays inside the band.

The trade loses if the final daily low drops BELOW the floor (strike_lo - 0.5).
The only risk is downward: the running minimum can only fall, never rise.

The proposed gate blocks entries where:

    clearance_from_floor < Y   AND   hours_to_close > X

where clearance_from_floor = running_min_at_entry - (strike_lo - 0.5).

This script answers: for each (hours_remaining, clearance) pair, what fraction
of historical days saw the final daily min drop MORE than clearance below the
running min at that point?  That fraction is the empirical loss probability the
gate is designed to suppress.

Method
------
For each (city, day, entry_hour H) using 5 years of ASOS data:

  running_min[H]  = min of all observations from midnight LST through hour H
  final_daily_min = min of all observations for the full LST day
  additional_drop = running_min[H] - final_daily_min    (always >= 0)
  hours_remaining = hours from H until midnight LST      (= 24 - H for H>0)

For each clearance candidate c:
  P(loss | hours_remaining, c) = P(additional_drop > c)

The gate (X, Y) blocks when hours_remaining > X AND clearance < Y.
We want high P(loss) in the blocked region and low P(loss) in the allowed region.

Output
------
  data/kxlowt_yes_gate_grid.txt  — P(loss) grid over (hours_remaining × clearance)
  data/kxlowt_yes_gate.csv       — raw (city, month, entry_hour) summary stats

Usage
-----
  venv/bin/python scripts/backtest_kxlowt_yes_gate.py
  venv/bin/python scripts/backtest_kxlowt_yes_gate.py --years 3 --months 3 4 5 6
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import math
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).parent.parent))

from kalshi_bot.news.noaa import LOW_CITIES, KALSHI_STATION_IDS  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_MESONET_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
_FETCH_DELAY = 0.5
_MIN_OBS_PER_DAY = 10

# Entry hours to evaluate (LST, 0=midnight, 23=11 PM).
# Between-YES signals fire when the running min first enters the band — this
# typically happens in the afternoon/evening as temps cool.  We scan 12:00-23:00.
_ENTRY_HOURS = list(range(12, 24))

# Clearance thresholds to evaluate (°F above effective floor)
_CLEARANCE_THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

# Hours-remaining thresholds to evaluate
_HOURS_THRESHOLDS = [2, 4, 6, 8, 10, 12]


def _low_station(metric: str) -> str | None:
    return KALSHI_STATION_IDS.get(metric.replace("temp_low_", "temp_high_"))


def _lst_midnight_offset(city_tz) -> int:
    """Return LST hour (0–1) corresponding to NWS midnight in local clock time.

    During DST, NWS midnight (00:00 LST) = 01:00 local clock.
    During standard time they are the same.
    """
    # January = standard time; detect if DST is currently shifting the clock
    std_offset = city_tz.utcoffset(datetime(2000, 1, 15))
    now_offset = city_tz.utcoffset(datetime.now())
    dst_delta   = now_offset - std_offset  # 0 in winter, 1h in summer
    # midnight LST on the local clock = 0 + dst_delta.hours
    return int(dst_delta.total_seconds() / 3600)


def _percentile(vals: list[float], pct: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    idx = min(int(math.ceil(pct / 100 * len(s))) - 1, len(s) - 1)
    return s[max(idx, 0)]


async def _fetch_city_obs(
    session: aiohttp.ClientSession,
    metric: str,
    station: str,
    start_dt: date,
    end_dt: date,
) -> dict[int, list[tuple[int, float]]]:
    """Fetch ASOS observations → ordinal → sorted [(local_minute, temp_f)]."""
    _, _, _, city_tz = LOW_CITIES[metric]

    fetch_start = start_dt - timedelta(days=1)
    fetch_end   = end_dt   + timedelta(days=1)

    params = {
        "station": station, "data": "tmpf",
        "year1": str(fetch_start.year), "month1": str(fetch_start.month),
        "day1":  str(fetch_start.day),
        "year2": str(fetch_end.year),   "month2": str(fetch_end.month),
        "day2":  str(fetch_end.day),
        "tz": "UTC", "format": "comma", "latlon": "no",
        "missing": "M", "trace": "T", "direct": "no",
        "report_type": "3,4",
    }
    try:
        async with session.get(
            _MESONET_URL, params=params,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            resp.raise_for_status()
            text = await resp.text()
    except Exception as exc:
        log.error("Fetch failed for %s (%s): %s", metric, station, exc)
        return {}

    # Use LST (standard time, no DST) for day bucketing — same as NWS CLI
    std_offset = city_tz.utcoffset(datetime(2000, 1, 15))
    lst_tz = timezone(std_offset)

    obs_by_ordinal: dict[int, list[tuple[int, float]]] = {}
    for line in text.splitlines():
        if line.startswith("#") or line.startswith("station") or not line.strip():
            continue
        parts = line.split(",")
        if len(parts) < 3:
            continue
        try:
            utc_ts  = datetime.strptime(parts[1].strip(), "%Y-%m-%d %H:%M").replace(
                tzinfo=timezone.utc
            )
            temp_str = parts[2].strip()
            if temp_str in ("M", "T", ""):
                continue
            temp_f = float(temp_str)
        except (ValueError, IndexError):
            continue

        local_ts  = utc_ts.astimezone(lst_tz)
        ordinal   = local_ts.toordinal()
        local_min = local_ts.hour * 60 + local_ts.minute
        obs_by_ordinal.setdefault(ordinal, []).append((local_min, temp_f))

    for ordinal in obs_by_ordinal:
        obs_by_ordinal[ordinal].sort()

    return obs_by_ordinal


def _analyze_city(
    metric: str,
    obs_by_ordinal: dict[int, list[tuple[int, float]]],
    start_dt: date,
    end_dt: date,
    month_filter: set[int] | None,
) -> dict[tuple[int, int], list[float]]:
    """Return {(month, entry_hour) → [additional_drop values]}.

    additional_drop = running_min_at_entry_hour - final_daily_min  (>= 0)
    """
    results: dict[tuple[int, int], list[float]] = {}

    start_ord = start_dt.toordinal()
    end_ord   = end_dt.toordinal()

    for ordinal in range(start_ord, end_ord + 1):
        readings = obs_by_ordinal.get(ordinal, [])
        if len(readings) < _MIN_OBS_PER_DAY:
            continue

        month = datetime.fromordinal(ordinal).month
        if month_filter and month not in month_filter:
            continue

        final_daily_min = min(t for _, t in readings)

        for entry_hour in _ENTRY_HOURS:
            # running_min = min of all observations from midnight (00:00) to entry_hour
            cutoff_min = entry_hour * 60
            obs_so_far = [t for (m, t) in readings if m <= cutoff_min]
            if not obs_so_far:
                continue

            running_min     = min(obs_so_far)
            additional_drop = running_min - final_daily_min   # >= 0
            hours_remaining = 24 - entry_hour                 # hours until midnight LST

            key = (month, entry_hour)
            results.setdefault(key, []).append(additional_drop)

    return results


def _p_loss(drops: list[float], clearance: float) -> float:
    """Fraction of days where additional_drop exceeded clearance (= trade would lose)."""
    if not drops:
        return float("nan")
    return sum(1 for d in drops if d > clearance) / len(drops)


async def main(
    years: int,
    city_filter: set[str] | None,
    month_filter: set[int] | None,
    csv_path: Path,
    grid_path: Path,
) -> None:
    end_dt   = date.today()
    start_dt = date(end_dt.year - years, end_dt.month, end_dt.day)

    pairs: list[tuple[str, str]] = []
    for metric in LOW_CITIES:
        suffix = metric.replace("temp_low_", "")
        if city_filter and suffix not in city_filter:
            continue
        station = _low_station(metric)
        if station is None:
            continue
        pairs.append((metric, station))

    log.info("Fetching %d cities, %d years", len(pairs), years)

    # global_drops[(month, entry_hour)] = [additional_drop, ...]
    global_drops: dict[tuple[int, int], list[float]] = {}
    city_drops:   dict[str, dict[tuple[int, int], list[float]]] = {}

    async with aiohttp.ClientSession() as session:
        for i, (metric, station) in enumerate(pairs):
            if i > 0:
                await asyncio.sleep(_FETCH_DELAY)
            log.info("[%d/%d] %s (%s)", i + 1, len(pairs), metric, station)
            obs = await _fetch_city_obs(session, metric, station, start_dt, end_dt)
            if not obs:
                continue
            city_result = _analyze_city(metric, obs, start_dt, end_dt, month_filter)
            city_drops[metric] = city_result
            for key, vals in city_result.items():
                global_drops.setdefault(key, []).extend(vals)

    # ---- CSV: per (city, month, entry_hour) stats -------------------------
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "city_key", "city_name", "month", "entry_hour",
        "n_days", "mean_drop", "p50_drop", "p75_drop", "p90_drop",
    ] + [f"p_loss_c{str(c).replace('.','p')}" for c in _CLEARANCE_THRESHOLDS]

    rows: list[dict] = []
    for metric in sorted(city_drops):
        city_name = LOW_CITIES[metric][0]
        for (month, hour), vals in sorted(city_drops[metric].items()):
            if not vals:
                continue
            n    = len(vals)
            mean = sum(vals) / n
            row  = {
                "city_key":   metric,
                "city_name":  city_name,
                "month":      month,
                "entry_hour": f"{hour:02d}:00",
                "n_days":     n,
                "mean_drop":  f"{mean:.2f}",
                "p50_drop":   f"{_percentile(vals, 50):.2f}",
                "p75_drop":   f"{_percentile(vals, 75):.2f}",
                "p90_drop":   f"{_percentile(vals, 90):.2f}",
            }
            for c in _CLEARANCE_THRESHOLDS:
                row[f"p_loss_c{str(c).replace('.','p')}"] = f"{_p_loss(vals, c):.3f}"
            rows.append(row)

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote %d rows to %s", len(rows), csv_path)

    # ---- Grid: P(loss) by (hours_remaining_bucket × clearance) -----------
    # Bucket drops by hours_remaining
    drops_by_hours: dict[int, list[float]] = {x: [] for x in _HOURS_THRESHOLDS}
    for (month, hour), vals in global_drops.items():
        hours_rem = 24 - hour
        for x in _HOURS_THRESHOLDS:
            if hours_rem > x:
                drops_by_hours[x].extend(vals)

    lines: list[str] = []
    month_desc = (
        f"months {sorted(month_filter)}" if month_filter else "all months"
    )
    lines.append(
        f"KXLOWT between-YES gate: P(additional_drop > clearance)\n"
        f"Data: {years} years, {len(pairs)} cities ({month_desc})\n"
        f"\n"
        f"P(loss) = fraction of historical days where the final daily low\n"
        f"fell MORE than 'clearance' below the running min at entry time.\n"
        f"High P(loss) → the gate should block; low P(loss) → allow.\n"
    )

    # Header row
    hdr  = f"{'hours_remaining >':>18}"
    for c in _CLEARANCE_THRESHOLDS:
        hdr += f"  {f'c={c:.1f}°F':>10}"
    lines.append(hdr)
    lines.append("-" * len(hdr))

    for x in _HOURS_THRESHOLDS:
        drops = drops_by_hours[x]
        n     = len(drops)
        row_s = f"{'> ' + str(x) + 'h':>18}  (n={n:,})"
        for c in _CLEARANCE_THRESHOLDS:
            pl = _p_loss(drops, c)
            row_s += f"  {pl * 100:>9.1f}%"
        lines.append(row_s)

    lines.append("")
    lines.append(
        "Interpretation: gate (X, Y) blocks when hours_remaining > X and clearance < Y.\n"
        "Choose X and Y where P(loss) transitions from high (>50%) to low (<30%).\n"
    )

    # Also print P(loss) by entry hour for each clearance (gives the time-of-day curve)
    lines.append(f"\nP(loss) by entry hour (LST) across all clearance values:")
    hdr2 = f"{'entry_hour':>12}  {'n':>6}"
    for c in _CLEARANCE_THRESHOLDS:
        hdr2 += f"  {f'c={c:.1f}°F':>10}"
    lines.append(hdr2)
    lines.append("-" * len(hdr2))

    for hour in _ENTRY_HOURS:
        all_drops: list[float] = []
        for month in range(1, 13):
            all_drops.extend(global_drops.get((month, hour), []))
        if not all_drops:
            continue
        n     = len(all_drops)
        row_s = f"{hour:02d}:00 LST  {n:>6}"
        for c in _CLEARANCE_THRESHOLDS:
            pl = _p_loss(all_drops, c)
            row_s += f"  {pl * 100:>9.1f}%"
        lines.append(row_s)

    # Recommended gate
    lines.append("")
    lines.append("Gate recommendation (block if P(loss) > 50%):")
    lines.append(f"  {'clearance':>12}  {'min hours to block':>20}  {'P(loss) at that threshold':>26}")
    lines.append("-" * 64)
    for c in _CLEARANCE_THRESHOLDS:
        # Find the smallest X where P(loss) > 50%
        rec_x = None
        rec_p = None
        for x in sorted(_HOURS_THRESHOLDS):
            drops = drops_by_hours[x]
            pl = _p_loss(drops, c)
            if pl > 0.50:
                rec_x = x
                rec_p = pl
                break
        if rec_x is not None:
            lines.append(f"  {c:>10.1f}°F  {'block if h_remaining > ' + str(rec_x) + 'h':>20}  {rec_p * 100:>25.1f}%")
        else:
            lines.append(f"  {c:>10.1f}°F  {'never exceeds 50%':>20}")

    text = "\n".join(lines)
    print("\n" + text)
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    grid_path.write_text(text + "\n")
    log.info("Wrote grid to %s", grid_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find optimal KXLOWT between-YES entry gate (hours_to_close × floor_clearance)."
    )
    parser.add_argument("--years",  type=int, default=5, help="Years of history to fetch")
    parser.add_argument("--cities", nargs="+", default=None,
                        help="City suffixes to include (e.g. chi hou satx). Default: all.")
    parser.add_argument("--months", nargs="+", type=int, default=None,
                        help="Months to include (1-12). Default: all.")
    parser.add_argument("--csv",  default="data/kxlowt_yes_gate.csv")
    parser.add_argument("--grid", default="data/kxlowt_yes_gate_grid.txt")
    args = parser.parse_args()

    asyncio.run(main(
        years=args.years,
        city_filter=set(args.cities)  if args.cities  else None,
        month_filter=set(args.months) if args.months  else None,
        csv_path=Path(args.csv),
        grid_path=Path(args.grid),
    ))
