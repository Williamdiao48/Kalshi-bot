"""Backtest: METAR overnight min vs NWS API observed min vs NWS CLI official low.

For each day in the requested date range, compares three sources of a KXLOWT
city's overnight low temperature:

  1. METAR overnight min  — min(hourly+special METAR obs, local midnight–07:00)
                            via Iowa State Mesonet ASOS.  Mirrors the live bot's
                            obs_low_values running-minimum tracking.

  2. NWS API observed min — min(api.weather.gov observations, local midnight–07:00)
                            Mirrors what noaa_observed reports intraday before the
                            CLI is published.

  3. NWS CLI official low — the final quality-controlled value in the morning CLI
                            product; what Kalshi uses for settlement.

Output CSV per city: data/kxlowt_discrepancy_{city}.csv
Columns: date, metar_min_f, nws_api_min_f, cli_official_low_f,
         metar_vs_cli, nws_api_vs_cli

Usage:
  venv/bin/python scripts/backtest_kxlowt_metar.py --city den
  venv/bin/python scripts/backtest_kxlowt_metar.py --city den lax chi
  venv/bin/python scripts/backtest_kxlowt_metar.py --city all
  venv/bin/python scripts/backtest_kxlowt_metar.py --city den --days 60
  venv/bin/python scripts/backtest_kxlowt_metar.py --city den --start 2026-01-01 --end 2026-04-18
  venv/bin/python scripts/backtest_kxlowt_metar.py --city all --out data/kxlowt_all.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import aiohttp

sys.path.insert(0, str(Path(__file__).parent.parent))

from kalshi_bot.news.noaa import KALSHI_STATION_IDS, LOW_CITIES  # noqa: E402
from kalshi_bot.news.nws_climo import LOW_CLIMO_LOCATIONS, _parse_min_f  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# City resolution
# ---------------------------------------------------------------------------

# Short suffix → metric key (covers all LOW_CLIMO_LOCATIONS entries)
_SHORT_TO_METRIC: dict[str, str] = {
    metric.replace("temp_low_", ""): metric
    for metric in LOW_CLIMO_LOCATIONS
}

def _resolve_cities(city_args: list[str]) -> list[str]:
    """Expand city arguments into a list of metric keys.

    Accepts:
      "all"           → every city in LOW_CLIMO_LOCATIONS
      "den"           → "temp_low_den"
      "temp_low_den"  → passed through as-is
    """
    if len(city_args) == 1 and city_args[0].lower() == "all":
        return sorted(LOW_CLIMO_LOCATIONS.keys())

    metrics: list[str] = []
    for arg in city_args:
        arg_lower = arg.lower()
        if arg_lower in LOW_CLIMO_LOCATIONS:
            metrics.append(arg_lower)
        elif arg_lower in _SHORT_TO_METRIC:
            metrics.append(_SHORT_TO_METRIC[arg_lower])
        else:
            log.error(
                "Unknown city %r. Valid short codes: %s",
                arg, ", ".join(sorted(_SHORT_TO_METRIC)),
            )
            sys.exit(1)
    return metrics


def _metar_station(metric: str) -> str:
    """Return the METAR station ID (e.g. 'KDEN') for a temp_low_* metric.

    LOW_CITIES uses the same physical stations as the corresponding HIGH city.
    KALSHI_STATION_IDS is keyed by temp_high_* so we substitute the prefix.
    """
    high_metric = metric.replace("temp_low_", "temp_high_")
    station = KALSHI_STATION_IDS.get(high_metric)
    if station:
        return station
    # Fallback: capitalise the suffix (e.g. temp_low_den → KDEN)
    suffix = metric.replace("temp_low_", "").upper()
    return f"K{suffix}"


# ---------------------------------------------------------------------------
# Source A: Iowa State Mesonet ASOS (METAR overnight min)
# ---------------------------------------------------------------------------

_MESONET_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
_MESONET_SEM = asyncio.Semaphore(4)


def _c_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


async def _fetch_metar_mins(
    session: aiohttp.ClientSession,
    station: str,
    city_tz: ZoneInfo,
    start_date: date,
    end_date: date,
) -> dict[str, float]:
    """Return {date_str: metar_overnight_min_f} for the local midnight–07:00 window.

    Fetches all ASOS observations (hourly + special) for the station and date
    range, then groups by local calendar date and computes the minimum
    temperature observed between local midnight (00:00) and 07:00.
    """
    # Add one extra day on each end to handle timezone edge cases
    fetch_start = start_date - timedelta(days=1)
    fetch_end   = end_date   + timedelta(days=1)

    params = {
        "station":     station,
        "data":        "tmpf",
        "year1":       str(fetch_start.year),
        "month1":      str(fetch_start.month),
        "day1":        str(fetch_start.day),
        "year2":       str(fetch_end.year),
        "month2":      str(fetch_end.month),
        "day2":        str(fetch_end.day),
        "tz":          "UTC",
        "format":      "comma",
        "latlon":      "no",
        "missing":     "M",
        "trace":       "T",
        "direct":      "no",
        "report_type": "1,3",  # routine hourly + special obs
    }

    text: str | None = None
    for attempt in range(4):
        if attempt:
            delay = 2 ** attempt  # 2, 4, 8 seconds
            log.warning("Mesonet retry %d for %s in %ds…", attempt, station, delay)
            await asyncio.sleep(delay)
        async with _MESONET_SEM:
            try:
                async with session.get(
                    _MESONET_URL,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    if resp.status == 503:
                        log.warning("Mesonet 503 for %s (attempt %d)", station, attempt + 1)
                        continue
                    resp.raise_for_status()
                    text = await resp.text()
                    break
            except Exception as exc:
                log.warning("Mesonet error for %s (attempt %d): %s", station, attempt + 1, exc)

    if text is None:
        log.error("Mesonet fetch failed for %s after retries", station)
        return {}

    # Group by (local date, local hour) and collect minimum temperatures.
    # key: local date string, value: list of tmpf values in midnight-07:00 window
    buckets: dict[str, list[float]] = {}

    for line in text.splitlines():
        if line.startswith("#") or line.startswith("station") or not line.strip():
            continue
        parts = line.split(",")
        if len(parts) < 3:
            continue
        valid_str = parts[1].strip()   # "YYYY-MM-DD HH:MM" UTC
        tmpf_str  = parts[2].strip()
        if tmpf_str in ("M", "T", ""):
            continue
        try:
            tmpf = float(tmpf_str)
        except ValueError:
            continue
        try:
            obs_utc = datetime.strptime(valid_str, "%Y-%m-%d %H:%M").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            continue

        obs_local = obs_utc.astimezone(city_tz)
        local_date_str = obs_local.date().isoformat()
        local_hour = obs_local.hour

        # Only keep midnight through 06:59 (the overnight window before 7 AM)
        if local_hour >= 7:
            continue

        buckets.setdefault(local_date_str, []).append(tmpf)

    result: dict[str, float] = {}
    for d_str, temps in buckets.items():
        if start_date.isoformat() <= d_str <= end_date.isoformat():
            result[d_str] = min(temps)

    log.info("  METAR (%s): %d overnight mins fetched", station, len(result))
    return result


# ---------------------------------------------------------------------------
# Source B: NWS API station observations (noaa_observed overnight min)
# ---------------------------------------------------------------------------

_NWS_OBS_SEM = asyncio.Semaphore(3)
_NWS_HEADERS = {"User-Agent": "kalshi-bot-backtest/1.0 (educational)"}
_NWS_FETCH_DELAY = 0.3


async def _fetch_nws_api_mins(
    session: aiohttp.ClientSession,
    station: str,
    city_tz: ZoneInfo,
    start_date: date,
    end_date: date,
) -> dict[str, float]:
    """Return {date_str: nws_api_overnight_min_f} for the midnight–07:00 window.

    Queries api.weather.gov/stations/{station}/observations day-by-day so we
    can target exact UTC windows without fetching the entire month at once.
    Mirrors the live noaa.py observation fetch logic.
    """
    result: dict[str, float] = {}
    obs_url = f"https://api.weather.gov/stations/{station}/observations"

    current = start_date
    while current <= end_date:
        # Convert local midnight and 07:00 to UTC for the API query
        local_midnight = datetime(current.year, current.month, current.day,
                                  0, 0, 0, tzinfo=city_tz)
        local_7am      = datetime(current.year, current.month, current.day,
                                  7, 0, 0, tzinfo=city_tz)
        start_utc = local_midnight.astimezone(timezone.utc)
        end_utc   = local_7am.astimezone(timezone.utc)

        params = {
            "start": start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end":   end_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit": "100",
        }

        async with _NWS_OBS_SEM:
            await asyncio.sleep(_NWS_FETCH_DELAY)
            try:
                async with session.get(
                    obs_url,
                    params=params,
                    headers=_NWS_HEADERS,
                    timeout=aiohttp.ClientTimeout(total=20),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json(content_type=None)
            except Exception as exc:
                log.debug("NWS API obs failed for %s on %s: %s",
                          station, current.isoformat(), exc)
                current += timedelta(days=1)
                continue

        temps_f: list[float] = []
        for feature in data.get("features", []):
            props = feature.get("properties", {})
            temp_c_obj = props.get("temperature", {})
            temp_c = temp_c_obj.get("value") if isinstance(temp_c_obj, dict) else None
            if temp_c is None:
                continue
            try:
                temps_f.append(_c_to_f(float(temp_c)))
            except (TypeError, ValueError):
                continue

        if temps_f:
            result[current.isoformat()] = min(temps_f)

        current += timedelta(days=1)

    log.info("  NWS API (%s): %d overnight mins fetched", station, len(result))
    return result


# ---------------------------------------------------------------------------
# Source C: NWS CLI official low (settlement value)
# ---------------------------------------------------------------------------

_NWS_PRODUCTS_URL = "https://api.weather.gov/products"
_CLI_SEM = asyncio.Semaphore(5)
_CLI_FETCH_DELAY = 0.15

_MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def _extract_cli_date(text: str, issuance_dt: datetime, city_tz: ZoneInfo) -> str | None:
    m = re.search(
        r"\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+(\d{1,2})\s+(\d{4})\b",
        text, re.IGNORECASE,
    )
    if m:
        try:
            month = _MONTH_MAP[m.group(1).upper()]
            day   = int(m.group(2))
            year  = int(m.group(3))
            return date(year, month, day).isoformat()
        except (ValueError, KeyError):
            pass
    return issuance_dt.astimezone(city_tz).date().isoformat()


async def _fetch_cli_lows(
    session: aiohttp.ClientSession,
    cli_location: str,
    city_tz: ZoneInfo,
    start_date: date,
    end_date: date,
) -> dict[str, float]:
    """Return {date_str: cli_official_low_f} for all available CLI products."""
    async with _CLI_SEM:
        try:
            async with session.get(
                _NWS_PRODUCTS_URL,
                params={"type": "CLI", "location": cli_location, "limit": "500"},
                headers=_NWS_HEADERS,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json(content_type=None)
        except Exception as exc:
            log.error("CLI stubs fetch failed for %s: %s", cli_location, exc)
            return {}

    stubs = data.get("@graph", [])
    log.info("  CLI (%s): %d stubs available", cli_location, len(stubs))

    result: dict[str, float] = {}

    async def _parse_one(stub: dict) -> None:
        product_url = stub.get("@id") or stub.get("id")
        if not product_url:
            return
        issuance_str = stub.get("issuanceTime", "")
        try:
            issuance_dt = datetime.fromisoformat(issuance_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            issuance_dt = datetime.now(timezone.utc)

        # Quick date check: skip stubs clearly outside our range (+2 day buffer)
        local_date = issuance_dt.astimezone(city_tz).date()
        if not (start_date - timedelta(days=2) <= local_date <= end_date + timedelta(days=2)):
            return

        text = stub.get("productText")
        if not text:
            async with _CLI_SEM:
                await asyncio.sleep(_CLI_FETCH_DELAY)
                try:
                    async with session.get(
                        product_url,
                        headers=_NWS_HEADERS,
                        timeout=aiohttp.ClientTimeout(total=15),
                    ) as resp:
                        resp.raise_for_status()
                        prod_data = await resp.json(content_type=None)
                    text = prod_data.get("productText") or ""
                except Exception as exc:
                    log.debug("CLI text fetch failed %s: %s", product_url, exc)
                    return

        min_f = _parse_min_f(text)
        if min_f is None:
            return

        date_str = _extract_cli_date(text, issuance_dt, city_tz)
        if date_str is None:
            return
        if start_date.isoformat() <= date_str <= end_date.isoformat():
            # Keep the first (most-authoritative) value if multiple stubs map to same date
            if date_str not in result:
                result[date_str] = min_f

    await asyncio.gather(*[_parse_one(s) for s in stubs])

    log.info("  CLI (%s): %d official lows parsed", cli_location, len(result))
    return result


# ---------------------------------------------------------------------------
# Per-city analysis
# ---------------------------------------------------------------------------

async def _analyse_city(
    session: aiohttp.ClientSession,
    metric: str,
    start_date: date,
    end_date: date,
    out_dir: Path,
    combined_rows: list[dict],
) -> None:
    city_suffix    = metric.replace("temp_low_", "")
    city_name, _, _, city_tz = LOW_CITIES[metric]
    cli_location, _, _       = LOW_CLIMO_LOCATIONS[metric]
    station                  = _metar_station(metric)

    log.info("=== %s (%s / %s / CLI:%s) ===",
             city_name, metric, station, cli_location)

    metar_mins, nws_api_mins, cli_lows = await asyncio.gather(
        _fetch_metar_mins(session, station, city_tz, start_date, end_date),
        _fetch_nws_api_mins(session, station, city_tz, start_date, end_date),
        _fetch_cli_lows(session, cli_location, city_tz, start_date, end_date),
    )

    # Build per-day rows for dates where we have at least the CLI value
    rows: list[dict] = []
    for d_str in sorted(cli_lows):
        metar   = metar_mins.get(d_str)
        nws_api = nws_api_mins.get(d_str)
        cli     = cli_lows[d_str]

        row = {
            "city":               city_suffix,
            "date":               d_str,
            "metar_min_f":        f"{metar:.2f}"   if metar   is not None else "",
            "nws_api_min_f":      f"{nws_api:.2f}" if nws_api is not None else "",
            "cli_official_low_f": f"{cli:.1f}",
            "metar_vs_cli":       f"{metar - cli:+.2f}"   if metar   is not None else "",
            "nws_api_vs_cli":     f"{nws_api - cli:+.2f}" if nws_api is not None else "",
        }
        rows.append(row)
        combined_rows.append(row)

    # Write per-city CSV
    out_path = out_dir / f"kxlowt_discrepancy_{city_suffix}.csv"
    _write_csv(out_path, rows)

    # Print summary for this city
    _print_summary(city_suffix, rows)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        log.warning("No data to write for %s", path.name)
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    log.info("Wrote %d rows → %s", len(rows), path)


def _print_summary(city: str, rows: list[dict]) -> None:
    metar_deltas = [float(r["metar_vs_cli"]) for r in rows if r["metar_vs_cli"]]
    api_deltas   = [float(r["nws_api_vs_cli"]) for r in rows if r["nws_api_vs_cli"]]

    if not metar_deltas:
        print(f"\n[{city.upper()}] No overlapping data for summary.")
        return

    def _stats(deltas: list[float]) -> str:
        n = len(deltas)
        mean = sum(deltas) / n
        deltas_s = sorted(deltas)
        med  = deltas_s[n // 2]
        mn   = min(deltas)
        mx   = max(deltas)
        return f"n={n}  mean={mean:+.2f}  median={med:+.2f}  min={mn:+.2f}  max={mx:+.2f}"

    def _count_below(deltas: list[float], threshold: float) -> int:
        return sum(1 for d in deltas if d < -threshold)

    print(f"\n{'='*60}")
    print(f"  {city.upper()} — METAR midnight–07:00 min vs NWS CLI official low")
    print(f"{'='*60}")
    print(f"  METAR vs CLI:    {_stats(metar_deltas)}")
    if api_deltas:
        print(f"  NWS API vs CLI:  {_stats(api_deltas)}")

    print(f"\n  METAR more than X°F BELOW CLI (false NO signal risk):")
    for thresh in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0):
        cnt = _count_below(metar_deltas, thresh)
        pct = 100.0 * cnt / len(metar_deltas)
        bar = "#" * cnt
        print(f"    > {thresh:.1f}°F : {cnt:3d}/{len(metar_deltas)} ({pct:5.1f}%)  {bar}")

    # Flag the worst cases
    worst = sorted(
        [(r["date"], float(r["metar_vs_cli"]), float(r["cli_official_low_f"]),
          float(r["metar_min_f"]))
         for r in rows if r["metar_vs_cli"]],
        key=lambda x: x[1],
    )[:5]
    if worst:
        print(f"\n  Largest negative METAR-vs-CLI gaps (METAR coldest vs official):")
        for d, delta, cli, metar in worst:
            print(f"    {d}  METAR={metar:.1f}°F  CLI={cli:.1f}°F  delta={delta:+.2f}°F")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--city", nargs="+", required=True, metavar="CITY",
        help=(
            'City short code(s) or "all". '
            f'Valid codes: {", ".join(sorted(_SHORT_TO_METRIC))}. '
            'Use "all" for every KXLOWT city.'
        ),
    )
    p.add_argument("--days",  type=int, default=90,
                   help="Number of past days to analyse (default: 90)")
    p.add_argument("--start", help="Start date YYYY-MM-DD (overrides --days)")
    p.add_argument("--end",   help="End date YYYY-MM-DD (default: yesterday)")
    p.add_argument("--out",   default="", metavar="PATH",
                   help="Extra combined CSV output path (optional, for --city all)")
    p.add_argument("--outdir", default="data",
                   help="Directory for per-city CSVs (default: data/)")
    return p.parse_args()


async def _main() -> None:
    args = _parse_args()

    end_date = (
        date.fromisoformat(args.end)
        if args.end
        else date.today() - timedelta(days=1)
    )
    start_date = (
        date.fromisoformat(args.start)
        if args.start
        else end_date - timedelta(days=args.days - 1)
    )

    metrics = _resolve_cities(args.city)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Date range: %s → %s (%d days)",
             start_date, end_date, (end_date - start_date).days + 1)
    log.info("Cities: %s", ", ".join(m.replace("temp_low_", "") for m in metrics))

    combined_rows: list[dict] = []

    async with aiohttp.ClientSession() as session:
        # Run cities concurrently but with a small concurrency cap to avoid
        # hammering the NWS API (it asks for ≤1 req/s; the per-source
        # semaphores handle that)
        sem = asyncio.Semaphore(3)

        async def _run_city(metric: str) -> None:
            async with sem:
                await _analyse_city(
                    session, metric, start_date, end_date,
                    out_dir, combined_rows,
                )

        await asyncio.gather(*[_run_city(m) for m in metrics])

    # Optional combined CSV (useful when --city all)
    if args.out and combined_rows:
        _write_csv(Path(args.out), combined_rows)

    # All-city aggregate summary when multiple cities requested
    if len(metrics) > 1:
        metar_deltas = [float(r["metar_vs_cli"]) for r in combined_rows if r["metar_vs_cli"]]
        if metar_deltas:
            print(f"\n{'='*60}")
            print("  ALL CITIES COMBINED — METAR vs CLI discrepancy")
            print(f"{'='*60}")
            n = len(metar_deltas)
            mean = sum(metar_deltas) / n
            sorted_d = sorted(metar_deltas)
            med = sorted_d[n // 2]
            print(f"  n={n}  mean={mean:+.2f}°F  median={med:+.2f}°F"
                  f"  min={min(metar_deltas):+.2f}°F  max={max(metar_deltas):+.2f}°F")
            print(f"\n  METAR more than X°F BELOW CLI (across all cities):")
            for thresh in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0):
                cnt = sum(1 for d in metar_deltas if d < -thresh)
                pct = 100.0 * cnt / n
                print(f"    > {thresh:.1f}°F : {cnt:4d}/{n} ({pct:5.1f}%)")


if __name__ == "__main__":
    asyncio.run(_main())
