"""Real-time WTI front-month futures price from Yahoo Finance.

Unlike eia.py (1-business-day lag) this module fetches the live CME NYMEX
front-month settlement price within 1–3 minutes, giving genuine information
edge for KXWTI markets whose Kalshi settlement is derived from the same
CME WTI front-month contract.

Contract selection
------------------
Yahoo Finance's generic CL=F rolls to the NEXT contract on its own schedule,
which does NOT align with Kalshi's stated rollover window.  Kalshi's rule:
  "rollover occurs 2 calendar days prior to a futures contract's Last Trading
   Date (LTD)"
For example: May 2026 LTD = April 20 → Kalshi uses May contract through
April 17, then June contract from April 18.

This module computes the correct contract ticker (e.g., CLK26.NYM, CLM26.NYM)
from a hardcoded LTD table (updated annually) rather than using CL=F, which
can be $3–5 off from the actual settlement contract Kalshi uses.

Month codes (CME): F=Jan G=Feb H=Mar J=Apr K=May M=Jun N=Jul Q=Aug U=Sep
                   V=Oct X=Nov Z=Dec

Produces DataPoints with source="yahoo_wti_futures" and metric="eia_wti" —
the same metric key used by eia.py — so KXWTI/KXOIL markets match
automatically without any changes to market_parser.py.

Environment variables
---------------------
  WTI_FUTURES_ENABLED   "true" | "false". Default: true.
"""

import logging
import os
from datetime import date, datetime, timedelta, timezone

import aiohttp

from ..data import DataPoint

WTI_FUTURES_ENABLED: bool = (
    os.environ.get("WTI_FUTURES_ENABLED", "true").lower() != "false"
)

# Yahoo Finance base URL for a specific CME contract, e.g. CLK26.NYM
_BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

# CME month letter codes
_MONTH_CODES: dict[int, str] = {
    1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
    7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z",
}

# ---------------------------------------------------------------------------
# Known NYMEX WTI Last Trading Dates (LTD) — update annually.
# Source: CME Group WTI Crude Oil futures expiration calendar.
# Kalshi uses: rollover = LTD - 2 calendar days.
# ---------------------------------------------------------------------------
_LTD: dict[tuple[int, int], date] = {
    # (delivery_year, delivery_month): LTD
    (2026,  5): date(2026,  4, 20),   # confirmed by Kalshi disclaimer
    (2026,  6): date(2026,  5, 19),
    (2026,  7): date(2026,  6, 22),
    (2026,  8): date(2026,  7, 20),
    (2026,  9): date(2026,  8, 19),
    (2026, 10): date(2026,  9, 21),
    (2026, 11): date(2026, 10, 20),
    (2026, 12): date(2026, 11, 19),
    (2027,  1): date(2026, 12, 22),
    (2027,  2): date(2027,  1, 21),
    (2027,  3): date(2027,  2, 22),
    (2027,  4): date(2027,  3, 22),
    (2027,  5): date(2027,  4, 21),
    (2027,  6): date(2027,  5, 19),
}


def _ltd_formula(delivery_year: int, delivery_month: int) -> date:
    """Fallback formula: 3 business days back from the 25th of the preceding month,
    then subtract 2 to match Kalshi's observed rollover offset."""
    if delivery_month == 1:
        py, pm = delivery_year - 1, 12
    else:
        py, pm = delivery_year, delivery_month - 1
    anchor = date(py, pm, 25)
    bd, d = 0, anchor
    while bd < 3:
        d -= timedelta(days=1)
        if d.weekday() < 5:
            bd += 1
    # The formula tends to run 2 days ahead of CME's actual LTD; subtract to compensate.
    return d - timedelta(days=2)


def kalshi_active_contract(today: date | None = None) -> tuple[str, date]:
    """Return (yahoo_ticker, ltd) for the CME WTI contract Kalshi settles on *today*.

    Kalshi's rollover rule: switch to the next delivery month 2 calendar days
    before the current front-month LTD.  We use today < (ltd - 2) as the
    condition to stay on the current contract.
    """
    if today is None:
        today = datetime.now(timezone.utc).date()

    year, month = today.year, today.month
    for _ in range(10):
        ltd = _LTD.get((year, month)) or _ltd_formula(year, month)
        rollover = ltd - timedelta(days=2)   # day Kalshi transitions to next month
        if today < rollover:
            code = _MONTH_CODES[month]
            ticker = f"CL{code}{str(year)[-2:]}.NYM"
            return ticker, ltd
        # Advance to next delivery month
        month += 1
        if month > 12:
            month = 1
            year += 1

    raise RuntimeError("Could not determine active WTI contract for today")


async def fetch_futures(session: aiohttp.ClientSession) -> list[DataPoint]:
    """Fetch the current WTI front-month futures price using the contract
    that Kalshi will settle on today.

    Returns a single-element list on success, empty list on any error or if
    WTI_FUTURES_ENABLED is false.
    """
    if not WTI_FUTURES_ENABLED:
        return []

    symbol, ltd = kalshi_active_contract()
    url = _BASE_URL.format(symbol=symbol)

    params = {"interval": "1m", "range": "1d"}
    try:
        async with session.get(
            url,
            params=params,
            headers=_HEADERS,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
    except aiohttp.ClientResponseError as exc:
        logging.warning("WTI Futures HTTP %s for %s: %s", exc.status, symbol, exc.message)
        return []
    except aiohttp.ClientError as exc:
        logging.warning("WTI Futures request error for %s: %s", symbol, exc)
        return []

    try:
        result = data["chart"]["result"][0]
    except (KeyError, IndexError, TypeError):
        error = (data.get("chart") or {}).get("error") or {}
        logging.warning(
            "WTI Futures: bad response for %s — %s",
            symbol,
            error.get("description", "unknown error"),
        )
        return []

    meta          = result.get("meta", {})
    price: float | None = meta.get("regularMarketPrice")
    ts:    int   | None = meta.get("regularMarketTime")
    market_state: str   = meta.get("marketState", "UNKNOWN")

    if price is None:
        logging.warning("WTI Futures: no regularMarketPrice for %s", symbol)
        return []

    as_of = (
        datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        if ts else
        datetime.now(timezone.utc).isoformat()
    )

    days_to_ltd = (ltd - datetime.now(timezone.utc).date()).days
    if market_state == "CLOSED":
        logging.warning(
            "WTI Futures [%s]: market CLOSED — last close %.2f $/bbl (LTD in %d days)",
            symbol, price, days_to_ltd,
        )
    else:
        logging.info(
            "WTI Futures [%s]: %.2f $/bbl  state=%s  (Kalshi LTD=%s, %d days)",
            symbol, price, market_state, ltd, days_to_ltd,
        )

    return [DataPoint(
        source   = "yahoo_wti_futures",
        metric   = "eia_wti",
        value    = float(price),
        unit     = "$/bbl",
        as_of    = as_of,
        metadata = {"symbol": symbol, "market_state": market_state, "ltd": str(ltd)},
    )]
