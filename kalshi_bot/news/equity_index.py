"""Equity index price fetcher (S&P 500, Nasdaq Composite, Dow Jones).

Uses the Yahoo Finance chart API (free, no API key) to fetch real-time* index
prices for ^GSPC (S&P 500), ^IXIC (Nasdaq Composite), and ^DJI (Dow Jones).

  * Yahoo Finance enforces a ~15-minute delay for retail clients, but the JSON
    endpoint used here typically returns prices within 1–3 minutes of the tape
    during regular market hours — sufficient for Kalshi intraday markets.

Supports pre-market and post-market sessions via the ``includePrePost``
parameter, so signals are available before the regular session opens.

Market states
-------------
  REGULAR   — NYSE/Nasdaq session (9:30 AM – 4:00 PM ET).  Best signal quality.
  PRE       — Pre-market (4:00 AM – 9:30 AM ET).  Directional signal only.
  POST      — After-hours (4:00 PM – 8:00 PM ET).  Can confirm daily close.
  CLOSED    — Outside all sessions.  ``value`` is the most recent close;
              treat with caution (stale data relative to Kalshi market state).

DataPoints
----------
Two DataPoints are emitted per symbol:

  source="yahoo_finance"
      value = ``regularMarketPrice`` — the price as of ``regularMarketTime``.
      For direction-only ("up/down") Kalshi markets, metadata["pct_change"]
      gives the running daily change from the previous close.

  source="yahoo_finance_premarket"   (only during PRE state, if enabled)
      value = ``preMarketPrice`` — pre-market last trade.
      metadata["pct_change"] compares to the previous day's regular close.

Metrics and Kalshi tickers
---------------------------
  price_spx  ← KXSPX*, KXSPXD*      S&P 500 index level (points)
  price_ndx  ← KXNDX*, KXINXD*      Nasdaq Composite level (points)
  price_dow  ← KXDOW*               Dow Jones Industrial Average (points)

Edge scales (for probability model):
  price_spx: 50.0 points  (≈1% of ~5 200)
  price_ndx: 200.0 points (≈1.1% of ~18 000)
  price_dow: 300.0 points (≈0.7% of ~42 000)

Environment variables
---------------------
  EQUITY_INDEX_PREMARKET   "true" | "false"  — include pre/post-market prices
                            in the DataPoint stream.  Default: true.
                            Set to false to use only regular-session prices.
"""

import logging
import os
from datetime import datetime, timezone

import aiohttp

from ..data import DataPoint

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EQUITY_INDEX_PREMARKET: bool = (
    os.environ.get("EQUITY_INDEX_PREMARKET", "true").lower() != "false"
)

# Yahoo Finance ticker → (canonical metric key, display label)
_SYMBOLS: dict[str, tuple[str, str]] = {
    "^GSPC": ("price_spx", "S&P 500"),
    "^IXIC": ("price_ndx", "Nasdaq"),
    "^DJI":  ("price_dow", "Dow Jones"),
}

_BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"

# Yahoo Finance rejects requests without a realistic User-Agent.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}


# ---------------------------------------------------------------------------
# Per-symbol fetch
# ---------------------------------------------------------------------------

async def _fetch_symbol(
    session:     aiohttp.ClientSession,
    yf_ticker:   str,
    metric:      str,
    label:       str,
) -> list[DataPoint]:
    """Fetch price data for one symbol from Yahoo Finance.

    Returns 1–2 DataPoints on success, empty list on any error.
    """
    url = _BASE_URL.format(ticker=yf_ticker)
    params = {
        "interval":       "1m",
        "range":          "1d",
        "includePrePost": "true",
    }

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
        logging.warning(
            "Yahoo Finance HTTP %s for %s: %s", exc.status, yf_ticker, exc.message
        )
        return []
    except aiohttp.ClientError as exc:
        logging.warning("Yahoo Finance request error for %s: %s", yf_ticker, exc)
        return []

    try:
        result = data["chart"]["result"][0]
    except (KeyError, IndexError, TypeError):
        error = (data.get("chart") or {}).get("error") or {}
        logging.warning(
            "Yahoo Finance: bad response for %s — %s",
            yf_ticker, error.get("description", "unknown error"),
        )
        return []

    meta = result.get("meta", {})
    market_state:    str   = meta.get("marketState", "CLOSED")
    regular_price:   float | None = meta.get("regularMarketPrice")
    previous_close:  float | None = meta.get("chartPreviousClose")
    market_time_ts:  int   | None = meta.get("regularMarketTime")

    if regular_price is None:
        logging.warning("Yahoo Finance: no regularMarketPrice for %s", yf_ticker)
        return []

    # Build ISO-8601 as_of from the exchange timestamp, not wall-clock.
    if market_time_ts:
        as_of = datetime.fromtimestamp(market_time_ts, tz=timezone.utc).isoformat()
    else:
        as_of = datetime.now(timezone.utc).isoformat()

    pct_change: float | None = None
    if previous_close and previous_close != 0:
        pct_change = (regular_price - previous_close) / previous_close

    logging.info(
        "Yahoo Finance [%s]: %.2f  %+.2f%%  state=%s",
        label,
        regular_price,
        (pct_change or 0) * 100,
        market_state,
    )

    base_meta = {
        "symbol":        yf_ticker,
        "prev_close":    previous_close,
        "pct_change":    round(pct_change, 6) if pct_change is not None else None,
        "market_state":  market_state,
    }

    points: list[DataPoint] = [
        DataPoint(
            source   = "yahoo_finance",
            metric   = metric,
            value    = regular_price,
            unit     = "points",
            as_of    = as_of,
            metadata = base_meta,
        )
    ]

    # Pre-market DataPoint: only during PRE state and when enabled.
    if EQUITY_INDEX_PREMARKET and market_state == "PRE":
        pre_price: float | None = meta.get("preMarketPrice")
        pre_ts:    int   | None = meta.get("preMarketTime")
        if pre_price is not None:
            pre_pct: float | None = None
            if previous_close and previous_close != 0:
                pre_pct = (pre_price - previous_close) / previous_close

            pre_as_of = (
                datetime.fromtimestamp(pre_ts, tz=timezone.utc).isoformat()
                if pre_ts else as_of
            )
            logging.info(
                "Yahoo Finance [%s] PRE-MARKET: %.2f  %+.2f%%",
                label, pre_price, (pre_pct or 0) * 100,
            )
            points.append(DataPoint(
                source   = "yahoo_finance_premarket",
                metric   = metric,
                value    = pre_price,
                unit     = "points",
                as_of    = pre_as_of,
                metadata = {
                    **base_meta,
                    "pct_change":  round(pre_pct, 6) if pre_pct is not None else None,
                    "market_state": "PRE",
                },
            ))

    return points


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def fetch_prices(session: aiohttp.ClientSession) -> list[DataPoint]:
    """Fetch current prices for all tracked equity indices.

    Fetches sequentially (not concurrently) to avoid Yahoo Finance
    rate-limiting both requests simultaneously — the total latency is still
    only ~1–2 seconds for two symbols.

    Returns:
        List of DataPoints (1–2 per symbol depending on market state).
    """
    data_points: list[DataPoint] = []
    for yf_ticker, (metric, label) in _SYMBOLS.items():
        points = await _fetch_symbol(session, yf_ticker, metric, label)
        data_points.extend(points)
    return data_points
