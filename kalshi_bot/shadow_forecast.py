"""Forecast shadow log: records every temperature forecast as it arrives.

Unlike raw_forecasts (which only logs DataPoints that matched an active Kalshi
market), this module writes *all* temperature DataPoints — including ECMWF,
ICON, GEM — regardless of whether a market exists.  Combined with the IEM
daily-actual backfill, this produces an unbiased D+1 vs actual error dataset
that can be used to train a calibration model free of look-ahead bias.

Table: forecast_shadow_log
  id              — autoincrement pk
  logged_at       — UTC ISO timestamp of when the forecast was recorded
  city            — e.g. "Los Angeles"
  date_target     — YYYY-MM-DD date the forecast is for
  is_high         — 1 = daily high (KXHIGH*), 0 = daily low (KXLOWT*)
  source          — e.g. "open_meteo_ecmwf", "noaa", "hrrr", "nws_hourly"
  forecast_f      — forecast temperature in °F
  actual_f        — filled in by backfill_iem_actuals() once IEM has the data
  actual_fetched_at — UTC ISO timestamp when actual_f was written

Deduplication: UNIQUE INDEX on (city, date_target, is_high, source,
strftime('%Y-%m-%dT%H', logged_at)) — 60 polls in the same clock-hour produce
exactly one row per source per city per date.
"""

import asyncio
import logging
import sqlite3
from datetime import datetime, timezone, timedelta, date
from typing import Any

import aiohttp

from .data import DataPoint

# IEM station → (station_id, network) for every city in cities.py
# Mapped to city name (the value field in CITIES/LOW_CITIES dicts).
_IEM_CITY_STATIONS: dict[str, tuple[str, str]] = {
    "Los Angeles":       ("LAX", "CA_ASOS"),
    "Denver":            ("DEN", "CO_ASOS"),
    "Chicago":           ("MDW", "IL_ASOS"),
    "New York":          ("NYC", "NY_ASOS"),
    "Miami":             ("MIA", "FL_ASOS"),
    "Austin":            ("AUS", "TX_ASOS"),
    "Dallas":            ("DAL", "TX_ASOS"),
    "Boston":            ("BOS", "MA_ASOS"),
    "Houston":           ("HOU", "TX_ASOS"),
    "Dallas/Fort Worth": ("DFW", "TX_ASOS"),
    "San Francisco":     ("SFO", "CA_ASOS"),
    "Seattle":           ("SEA", "WA_ASOS"),
    "Phoenix":           ("PHX", "AZ_ASOS"),
    "Philadelphia":      ("PHL", "PA_ASOS"),
    "Atlanta":           ("ATL", "GA_ASOS"),
    "Minneapolis":       ("MSP", "MN_ASOS"),
    "Washington DC":     ("DCA", "DC_ASOS"),
    "Las Vegas":         ("LAS", "NV_ASOS"),
    "Oklahoma City":     ("OKC", "OK_ASOS"),
    "San Antonio":       ("SAT", "TX_ASOS"),
    "New Orleans":       ("MSY", "LA_ASOS"),
}

# Minimum age before we attempt to fetch IEM actuals for a date.
# IEM daily summaries for a given date are typically finalized by 08:00 UTC
# the next day; using 2 days gives comfortable headroom.
_IEM_BACKFILL_MIN_AGE_DAYS = 2

_IEM_DAILY_URL = "https://mesonet.agron.iastate.edu/api/1/daily.json"


def log_shadow_forecasts(
    conn: sqlite3.Connection,
    data_points: list[DataPoint],
) -> None:
    """Write one row per temperature DataPoint into forecast_shadow_log.

    INSERT OR IGNORE: the unique hourly dedup index silently drops duplicate
    forecasts from the same source recorded in the same clock-hour.

    Only DataPoints with metric matching temp_high_* or temp_low_* are logged.
    The city name is taken from metadata["city"]; points without it are skipped.
    """
    now_utc = datetime.now(timezone.utc).isoformat()
    rows: list[tuple[Any, ...]] = []

    for dp in data_points:
        metric = dp.metric
        if not (metric.startswith("temp_high_") or metric.startswith("temp_low_")):
            continue
        city = dp.metadata.get("city")
        if not city:
            continue
        date_target = dp.metadata.get("forecast_date")
        if not date_target:
            # Fall back to extracting the date from as_of when forecast_date
            # is absent (e.g. NOAA observed DataPoints, NWS hourly).
            if dp.as_of:
                date_target = dp.as_of[:10]
            else:
                continue
        is_high = 1 if metric.startswith("temp_high_") else 0
        rows.append((now_utc, city, date_target, is_high, dp.source, float(dp.value)))

    if not rows:
        return

    try:
        conn.executemany(
            """
            INSERT OR IGNORE INTO forecast_shadow_log
                (logged_at, city, date_target, is_high, source, forecast_f)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        logging.debug("shadow_forecast: inserted/ignored %d temperature DataPoint(s).", len(rows))
    except Exception as exc:
        logging.warning("shadow_forecast: insert error: %s", exc)


async def backfill_iem_actuals(
    conn: sqlite3.Connection,
    session: aiohttp.ClientSession,
) -> None:
    """Fetch IEM daily actuals for any shadow log rows missing actual_f.

    For each (city, date_target, is_high) group that has NULL actual_f and
    whose date_target is at least _IEM_BACKFILL_MIN_AGE_DAYS old, fetch the
    IEM daily.json for that year and write max_tmpf / min_tmpf back.

    This is designed to be called once per calendar day (caller's responsibility
    to throttle).
    """
    cutoff = (date.today() - timedelta(days=_IEM_BACKFILL_MIN_AGE_DAYS)).isoformat()

    try:
        pending = conn.execute("""
            SELECT DISTINCT city, date_target, is_high
            FROM forecast_shadow_log
            WHERE actual_f IS NULL
              AND date_target <= ?
            ORDER BY date_target
        """, (cutoff,)).fetchall()
    except Exception as exc:
        logging.warning("shadow_forecast backfill: query error: %s", exc)
        return

    if not pending:
        logging.debug("shadow_forecast backfill: nothing pending.")
        return

    logging.info("shadow_forecast backfill: %d (city, date, is_high) groups to fill.", len(pending))

    # Batch by (city, year) so we make at most one IEM request per city per year.
    # Map: (city, year) → {date_str: {is_high: temp_f}}
    fetched: dict[tuple[str, int], dict[str, dict[int, float]]] = {}
    city_year_combos: set[tuple[str, int]] = set()
    for city, date_target, is_high in pending:
        year = int(date_target[:4])
        city_year_combos.add((city, year))

    for city, year in sorted(city_year_combos):
        station_info = _IEM_CITY_STATIONS.get(city)
        if not station_info:
            logging.debug("shadow_forecast backfill: no IEM station for city %r", city)
            continue
        station, network = station_info
        url = f"{_IEM_DAILY_URL}?station={station}&network={network}&year={year}"
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                resp.raise_for_status()
                data = await resp.json()
        except Exception as exc:
            logging.warning("shadow_forecast backfill: IEM fetch error %s/%d: %s", city, year, exc)
            await asyncio.sleep(0.3)
            continue

        day_map: dict[str, dict[int, float]] = {}
        for row in data.get("data", []):
            d = row.get("date")
            if not d:
                continue
            for is_h, field in ((1, "max_tmpf"), (0, "min_tmpf")):
                v = row.get(field)
                if v is not None:
                    try:
                        day_map.setdefault(d, {})[is_h] = float(v)
                    except (TypeError, ValueError):
                        pass
        fetched[(city, year)] = day_map
        await asyncio.sleep(0.3)

    # Write actuals back.
    now_utc = datetime.now(timezone.utc).isoformat()
    updates = 0
    for city, date_target, is_high in pending:
        year = int(date_target[:4])
        day_map = fetched.get((city, year), {})
        actual = day_map.get(date_target, {}).get(is_high)
        if actual is None:
            continue
        try:
            conn.execute(
                """
                UPDATE forecast_shadow_log
                SET actual_f = ?, actual_fetched_at = ?
                WHERE city = ? AND date_target = ? AND is_high = ? AND actual_f IS NULL
                """,
                (actual, now_utc, city, date_target, is_high),
            )
            updates += 1
        except Exception as exc:
            logging.warning("shadow_forecast backfill: update error: %s", exc)

    if updates:
        logging.info("shadow_forecast backfill: wrote actual_f for %d group(s).", updates)
