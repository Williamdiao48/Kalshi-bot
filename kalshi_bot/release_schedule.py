"""Scheduled public data release times for BLS, FRED, and EIA.

Used to gate trade execution: only permit ``bls_*``, ``fred_*``, and
``eia_*`` numeric opportunities within ``RELEASE_WINDOW_MINUTES`` of the
official release time.  Outside that window the data is already priced into
the market and no information edge exists.

The window opens AT the release time (never before) so we only trade on
published data, not pre-release speculation.

Metrics with no entry in ``_METRIC_RELEASE_FN`` (e.g. ``temp_high_*``,
``price_btc``) always pass — they are governed by their own quality gates.

Update annually
---------------
- ``_BLS_CPI_DATES``: BLS publishes the full calendar in late December.
  URL: https://www.bls.gov/schedule/news_release/cpi.htm
- ``_FOMC_DATES``: Federal Reserve publishes FOMC meeting dates in December.
  URL: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
- ``_dol_claims_releases``: computed algorithmically (every Thursday 08:30 ET),
  no manual update required.
- ``_ism_manufacturing_releases`` / ``_ism_services_releases``: computed
  algorithmically (1st / 3rd business day of each month at 10:00 ET),
  no manual update required.
"""

from __future__ import annotations

import datetime
from zoneinfo import ZoneInfo

_ET  = ZoneInfo("America/New_York")
_UTC = datetime.timezone.utc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _et(date: datetime.date, hour: int, minute: int = 0) -> datetime.datetime:
    """Build a UTC datetime from an ET date + hour/minute."""
    return datetime.datetime(
        date.year, date.month, date.day, hour, minute, tzinfo=_ET
    ).astimezone(_UTC)


def _first_friday(year: int, month: int) -> datetime.date:
    d = datetime.date(year, month, 1)
    return d + datetime.timedelta(days=(4 - d.weekday()) % 7)


def _all_weekday(year: int, weekday: int) -> list[datetime.date]:
    """All occurrences of ``weekday`` (0=Mon … 6=Sun) in a calendar year."""
    d = datetime.date(year, 1, 1)
    d += datetime.timedelta(days=(weekday - d.weekday()) % 7)
    dates: list[datetime.date] = []
    while d.year == year:
        dates.append(d)
        d += datetime.timedelta(days=7)
    return dates


def _nth_business_day(year: int, month: int, n: int) -> datetime.date:
    """Return the n-th business day (Mon–Fri) of a given month (1-indexed).

    No federal-holiday adjustment — ISM releases very rarely fall on a holiday
    and are simply rescheduled when they do.  The approximation is sufficient
    for the release-window gate since the window opens only after the data is
    published and the next closest release is always ≥ 1 week away.
    """
    d = datetime.date(year, month, 1)
    count = 0
    while True:
        if d.weekday() < 5:  # Mon–Fri
            count += 1
            if count == n:
                return d
        d += datetime.timedelta(days=1)


# ---------------------------------------------------------------------------
# BLS
# ---------------------------------------------------------------------------

def _bls_employment_releases(year: int) -> list[datetime.datetime]:
    """NFP + unemployment rate: first Friday of every month at 08:30 ET."""
    return [_et(_first_friday(year, m), 8, 30) for m in range(1, 13)]


# BLS CPI: variable mid-month date, 08:30 ET.  Update each December.
_BLS_CPI_DATES: dict[int, list[datetime.date]] = {
    2026: [
        datetime.date(2026,  1, 14),
        datetime.date(2026,  2, 12),
        datetime.date(2026,  3, 12),
        datetime.date(2026,  4, 10),
        datetime.date(2026,  5, 13),
        datetime.date(2026,  6, 11),
        datetime.date(2026,  7, 15),
        datetime.date(2026,  8, 12),
        datetime.date(2026,  9, 10),
        datetime.date(2026, 10, 15),
        datetime.date(2026, 11, 13),
        datetime.date(2026, 12, 11),
    ],
}


def _bls_cpi_releases(year: int) -> list[datetime.datetime]:
    return [_et(d, 8, 30) for d in _BLS_CPI_DATES.get(year, [])]


# BLS PPI: variable second-week date, 08:30 ET.  Update each December.
_BLS_PPI_DATES: dict[int, list[datetime.date]] = {
    2026: [
        datetime.date(2026,  1, 14),
        datetime.date(2026,  2, 13),
        datetime.date(2026,  3, 13),
        datetime.date(2026,  4, 11),
        datetime.date(2026,  5, 14),
        datetime.date(2026,  6, 12),
        datetime.date(2026,  7, 15),
        datetime.date(2026,  8, 13),
        datetime.date(2026,  9, 11),
        datetime.date(2026, 10, 15),
        datetime.date(2026, 11, 13),
        datetime.date(2026, 12, 11),
    ],
}


def _bls_ppi_releases(year: int) -> list[datetime.datetime]:
    return [_et(d, 8, 30) for d in _BLS_PPI_DATES.get(year, [])]


# ---------------------------------------------------------------------------
# FRED / FOMC
# ---------------------------------------------------------------------------

# FOMC statement release: announcement day at 14:00 ET.  Update each December.
_FOMC_DATES: dict[int, list[datetime.date]] = {
    2026: [
        datetime.date(2026,  1, 29),
        datetime.date(2026,  3, 18),
        datetime.date(2026,  5,  7),
        datetime.date(2026,  6, 18),
        datetime.date(2026,  7, 30),
        datetime.date(2026,  9, 17),
        datetime.date(2026, 10, 29),
        datetime.date(2026, 12, 17),
    ],
}


def _fomc_releases(year: int) -> list[datetime.datetime]:
    return [_et(d, 14, 0) for d in _FOMC_DATES.get(year, [])]


# ---------------------------------------------------------------------------
# EIA
# ---------------------------------------------------------------------------

def _eia_wti_releases(year: int) -> list[datetime.datetime]:
    """Petroleum Status Report: Wednesday 10:30 ET (EIA confirmed).

    Also includes Tuesday 16:30 ET — the API (American Petroleum Institute)
    weekly inventory report window.  This opens a pre-release trading window
    ~17 hours before the EIA Wednesday confirmation, allowing eia_inventory
    directional signals (based on the prior week's confirmed inventory change)
    to fire on Tuesday evening.
    """
    wednesdays = [_et(d, 10, 30) for d in _all_weekday(year, 2)]  # EIA: Wed 10:30
    tuesdays   = [_et(d, 16, 30) for d in _all_weekday(year, 1)]  # API: Tue 16:30
    return sorted(wednesdays + tuesdays)


def _eia_natgas_releases(year: int) -> list[datetime.datetime]:
    """Natural Gas Storage Report: Thursday 10:30 ET (EIA confirmed).

    Also includes Wednesday 10:30 ET — the AGA (American Gas Association)
    storage report window.  This opens a pre-release trading window ~24 hours
    before the EIA Thursday confirmation, allowing eia_inventory directional
    signals to fire on Wednesday morning.
    """
    thursdays  = [_et(d, 10, 30) for d in _all_weekday(year, 3)]  # EIA: Thu 10:30
    wednesdays = [_et(d, 10, 30) for d in _all_weekday(year, 2)]  # AGA: Wed 10:30
    return sorted(thursdays + wednesdays)


# ---------------------------------------------------------------------------
# ISM PMI
# ---------------------------------------------------------------------------

def _ism_manufacturing_releases(year: int) -> list[datetime.datetime]:
    """ISM Manufacturing PMI: 1st business day of each month at 10:00 ET."""
    return [_et(_nth_business_day(year, m, 1), 10, 0) for m in range(1, 13)]


def _ism_services_releases(year: int) -> list[datetime.datetime]:
    """ISM Services PMI: 3rd business day of each month at 10:00 ET."""
    return [_et(_nth_business_day(year, m, 3), 10, 0) for m in range(1, 13)]


# ---------------------------------------------------------------------------
# DOL / FRED — Initial Jobless Claims
# ---------------------------------------------------------------------------

def _dol_claims_releases(year: int) -> list[datetime.datetime]:
    """Initial jobless claims (ICSA): every Thursday at 08:30 ET.

    The Department of Labor releases the prior week's seasonally-adjusted
    initial claims every Thursday morning at 08:30 ET.  FRED updates the
    ICSA series within minutes.  Kalshi frequently lists weekly over/under
    markets tied to this release.
    """
    return [_et(d, 8, 30) for d in _all_weekday(year, 3)]


def _last_business_day(year: int, month: int) -> datetime.date:
    """Return the last business day (Mon–Fri) of a given month."""
    if month == 12:
        last = datetime.date(year, 12, 31)
    else:
        last = datetime.date(year, month + 1, 1) - datetime.timedelta(days=1)
    while last.weekday() >= 5:  # Sat=5, Sun=6
        last -= datetime.timedelta(days=1)
    return last


def _bea_pce_releases(year: int) -> list[datetime.datetime]:
    """PCE price index: last business day of each month at 08:30 ET."""
    return [_et(_last_business_day(year, m), 8, 30) for m in range(1, 13)]


# ---------------------------------------------------------------------------
# Dispatch table  metric-prefix → release-time generator
# ---------------------------------------------------------------------------

_METRIC_RELEASE_FN: dict[str, object] = {
    "bls_nfp":           _bls_employment_releases,
    "bls_unrate":        _bls_employment_releases,
    "bls_cpi":           _bls_cpi_releases,
    "bls_ppi_fd":        _bls_ppi_releases,
    "bls_ppi_core":      _bls_ppi_releases,
    "fred_fedfunds":     _fomc_releases,
    "fred_dgs10":        _fomc_releases,
    "fred_dgs2":         _fomc_releases,
    "fred_icsa":         _dol_claims_releases,
    "fred_pce":          _bea_pce_releases,
    "ism_manufacturing": _ism_manufacturing_releases,
    "ism_services":      _ism_services_releases,
    "eia_wti":           _eia_wti_releases,
    "eia_natgas":        _eia_natgas_releases,
}

# In-process cache: (metric_prefix, year) → sorted list of release datetimes
_cache: dict[tuple[str, int], list[datetime.datetime]] = {}


def _get_releases(prefix: str, year: int) -> list[datetime.datetime]:
    key = (prefix, year)
    if key not in _cache:
        fn = _METRIC_RELEASE_FN.get(prefix)
        _cache[key] = sorted(fn(year)) if fn else []  # type: ignore[call-arg]
    return _cache[key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_within_release_window(
    metric: str,
    now: datetime.datetime,
    window_minutes: int,
) -> bool:
    """Return True if ``now`` is within ``window_minutes`` after a release.

    Metrics not in ``_METRIC_RELEASE_FN`` (weather, crypto, forex) always
    return True — they are governed by their own quality gates elsewhere.

    Args:
        metric:         Full metric key, e.g. ``"bls_nfp"``.
        now:            Current UTC datetime (timezone-aware).
        window_minutes: How many minutes after a release to allow trading.

    Returns:
        True  → trade is allowed (within window, or metric has no schedule).
        False → trade is blocked (scheduled metric, outside all windows).
    """
    prefix = next((p for p in _METRIC_RELEASE_FN if metric.startswith(p)), None)
    if prefix is None:
        return True  # No schedule → always allow

    window = datetime.timedelta(minutes=window_minutes)

    # Check current year and adjacent year (in case of year boundary).
    year = now.year
    for y in (year - 1, year, year + 1):
        for release_time in _get_releases(prefix, y):
            if release_time <= now <= release_time + window:
                return True

    return False


def next_release(
    metric: str,
    now: datetime.datetime,
) -> datetime.datetime | None:
    """Return the next scheduled release time for a metric after ``now``.

    Returns None if the metric has no schedule or no future release is known.
    Useful for logging how far away the next window is.
    """
    prefix = next((p for p in _METRIC_RELEASE_FN if metric.startswith(p)), None)
    if prefix is None:
        return None

    year = now.year
    candidates: list[datetime.datetime] = []
    for y in (year, year + 1):
        candidates.extend(_get_releases(prefix, y))

    future = [r for r in candidates if r > now]
    return min(future) if future else None
