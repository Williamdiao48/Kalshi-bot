"""Parse Kalshi market tickers and titles into structured numeric metadata.

Each market has a ticker like `KXHIGHLAX-26MAR05-T75` and a title like
"Will the high temp in LA be >75° on Mar 5, 2026?". This module extracts:
  - Which canonical metric the market is about (e.g. "temp_high_lax")
  - The direction of resolution ("over", "under", "between", "up", "down")
  - The strike value(s)

This structured form is then used by numeric_matcher to compare live data
against open market positions.
"""

import logging
import re
from dataclasses import dataclass
from datetime import date
from typing import Any


# ---------------------------------------------------------------------------
# Ticker prefix → canonical metric key
# ---------------------------------------------------------------------------

TICKER_TO_METRIC: dict[str, str] = {
    # Weather — daily high temperature by city
    "KXHIGHLAX": "temp_high_lax",
    "KXHIGHDEN": "temp_high_den",
    "KXHIGHCHI": "temp_high_chi",
    "KXHIGHNY":  "temp_high_ny",
    "KXHIGHMIA": "temp_high_mia",
    "KXHIGHDAL": "temp_high_dal",
    "KXHIGHBOS": "temp_high_bos",
    "KXHIGHAUS": "temp_high_aus",
    "KXHIGHOU":  "temp_high_hou",
    # New KXHIGHT* series (2026-04)
    "KXHIGHTSFO":  "temp_high_sfo",
    "KXHIGHTSEA":  "temp_high_sea",
    "KXHIGHTBOS":  "temp_high_bos",   # reuses existing metric key
    "KXHIGHTPHX":  "temp_high_phx",
    "KXHIGHPHIL":  "temp_high_phl",
    "KXHIGHTATL":  "temp_high_atl",
    "KXHIGHTMIN":  "temp_high_msp",
    "KXHIGHTDC":   "temp_high_dca",
    "KXHIGHTLV":   "temp_high_las",
    "KXHIGHTOKC":  "temp_high_okc",
    "KXHIGHTDAL":  "temp_high_dfw",   # DFW airport (not Love Field — confirmed by rules_primary)
    "KXHIGHTSATX": "temp_high_sat",
    "KXHIGHTHOU":  "temp_high_hou",   # reuses existing metric key
    "KXHIGHTNOLA": "temp_high_msy",
    # Weather — daily rain (binary: will it rain?)
    "KXRAINNYC":  "precip_total_ny",   # NYC daily rain — resolves YES if any measurable precip
    # Daily low temperature — KXLOWT* series (launched 2026-04)
    "KXLOWTLAX":  "temp_low_lax",
    "KXLOWTDEN":  "temp_low_den",
    "KXLOWTCHI":  "temp_low_chi",
    "KXLOWTNYC":  "temp_low_ny",    # "NYC" in ticker, normalized to "ny"
    "KXLOWTMIA":  "temp_low_mia",
    "KXLOWTAUS":  "temp_low_aus",
    "KXLOWTBOS":  "temp_low_bos",
    "KXLOWTHOU":  "temp_low_hou",
    "KXLOWTDFW":  "temp_low_dfw",
    "KXLOWTSFO":  "temp_low_sfo",
    "KXLOWTSEA":  "temp_low_sea",
    "KXLOWTPHX":  "temp_low_phx",
    "KXLOWTPHIL": "temp_low_phl",   # "PHIL" in ticker, normalized to "phl"
    "KXLOWTATL":  "temp_low_atl",
    "KXLOWTMIN":  "temp_low_msp",
    "KXLOWTDC":   "temp_low_dca",
    "KXLOWTLV":   "temp_low_las",
    "KXLOWTOKC":  "temp_low_okc",
    "KXLOWTSATX": "temp_low_sat",
    "KXLOWTNOLA": "temp_low_msy",
    # Crypto — price in USD
    "KXBTCD":    "price_btc_usd",   # daily close
    "KXBTC15M":  "price_btc_usd",   # 15-minute
    "KXETH15M":  "price_eth_usd",
    "KXSOL15M":  "price_sol_usd",
    "KXXRP15M":  "price_xrp_usd",
    "KXDOGE15M": "price_doge_usd",
    "KXDOGE":    "price_doge_usd",
    "KXADA15M":  "price_ada_usd",
    "KXADA":     "price_ada_usd",
    "KXAVAX15M": "price_avax_usd",
    "KXAVAX":    "price_avax_usd",
    "KXLINK15M": "price_link_usd",
    "KXLINK":    "price_link_usd",
    "KXBNB15M":  "price_bnb_usd",
    "KXBNB":     "price_bnb_usd",
    # Forex
    "KXEURUSD": "rate_eur_usd",
    "KXUSDJPY": "rate_usd_jpy",
    "KXGBPUSD": "rate_gbp_usd",
    # Economics (production markets)
    "KXCPI":     "bls_cpi_u",
    "KXNFP":     "bls_nfp",
    "KXADP":     "bls_nfp",          # ADP private payrolls (pre-signal for NFP)
    "KXUNRATE":  "bls_unrate",
    "KXJOBLESS": "fred_icsa",        # weekly initial jobless claims (DOL/FRED ICSA)
    "KXICSA":    "fred_icsa",        # alternate Kalshi ticker prefix for same series
    "KXPPI":     "bls_ppi_fd",       # PPI Final Demand
    "KXPCE":     "fred_pce",         # PCE price index (BEA via FRED PCEPI)
    "KXISM":     "ism_manufacturing", # ISM Manufacturing PMI (generic prefix)
    "KXISMMFG":  "ism_manufacturing", # ISM Manufacturing PMI (explicit)
    "KXISMSVC":  "ism_services",     # ISM Services PMI
    "KXGDP":     "fred_gdp_nowcast",  # Atlanta Fed GDPNow current-quarter estimate (FRED GDPNOW)
    # Interest rates (FRED)
    "KXFED":    "fred_fedfunds",
    "KXFFR":    "fred_fedfunds",
    "KXDGS10":  "fred_dgs10",
    "KXDGS2":   "fred_dgs2",
    # Energy (EIA)
    "KXWTI":    "eia_wti",
    "KXOIL":    "eia_wti",
    "KXNATGAS": "eia_natgas",
    "KXNG":     "eia_natgas",
    # Equity indices
    "KXSPX":    "price_spx",   # S&P 500 (daily close / intraday)
    "KXSPXD":   "price_spx",   # S&P 500 (alternate daily ticker)
    "KXNDX":    "price_ndx",   # Nasdaq Composite (daily close / intraday)
    "KXINXD":   "price_ndx",   # Nasdaq (alternate intraday ticker)
    "KXDOW":    "price_dow",   # Dow Jones Industrial Average
}


# ---------------------------------------------------------------------------
# Parsed result
# ---------------------------------------------------------------------------

@dataclass
class ParsedMarket:
    ticker: str
    title: str
    metric: str
    direction: str        # "over" | "under" | "between" | "up" | "down" | "unknown"
    strike: float | None = None      # single threshold (over / under)
    strike_lo: float | None = None   # lower bound (between)
    strike_hi: float | None = None   # upper bound (between)


# ---------------------------------------------------------------------------
# Regex patterns for extracting direction + strike from title text
# ---------------------------------------------------------------------------

# Match ">75", "> $95,000", "above 1.10", etc.
_OVER_RE = re.compile(
    r"(?:>|above|over|at least|≥)\s*\$?([\d,]+\.?\d*)", re.IGNORECASE
)
# Match "<68", "< $80,000", "below 1.05", "under 50", etc.
_UNDER_RE = re.compile(
    r"(?:<|below|under|at most|≤)\s*\$?([\d,]+\.?\d*)", re.IGNORECASE
)
# Match "74-75°", "$94,000-$96,000", "1.08-1.10", etc.
_BETWEEN_RE = re.compile(
    r"\$?([\d,]+\.?\d*)\s*[-–]\s*\$?([\d,]+\.?\d*)", re.IGNORECASE
)
_UP_RE = re.compile(r"\bup\b", re.IGNORECASE)
_DOWN_RE = re.compile(r"\bdown\b", re.IGNORECASE)

# Ticker-suffix fallback: when the market title contains no directional keyword
# (common for crypto/forex tickers like KXBTCD-26MAR1603-T73999.99 whose titles
# are just "Bitcoin price on Mar 16, 2026?"), extract direction and strike from
# the ticker suffix itself.
#
#   -T{N}   → "over"   (above threshold N)    e.g. KXBTCD-...-T73999.99
#   -B{N}   → "under"  (below threshold N)    e.g. KXBTCD-...-B65000
#
# Temperature tickers also use T/B suffixes but their titles always contain
# explicit directional language ("<60°", "47-48°", ">75°"), so the title-based
# patterns above will match first and this fallback is never reached for them.
_TICKER_T_RE = re.compile(r"-T([\d]+\.?\d*)$", re.IGNORECASE)
_TICKER_B_RE = re.compile(r"-B([\d]+\.?\d*)$", re.IGNORECASE)


def _to_float(s: str) -> float:
    return float(s.replace(",", ""))


def _metric_for_ticker(ticker: str) -> str | None:
    """Return the canonical metric key for a ticker, or None if unrecognised."""
    for prefix, metric in TICKER_TO_METRIC.items():
        if ticker.startswith(prefix):
            return metric
    return None


def parse_market(market: dict[str, Any]) -> ParsedMarket | None:
    """Parse a single Kalshi market dict into a structured ParsedMarket.

    Extracts direction, strike(s), and the canonical metric key from the
    market's ticker and title.  Direction is determined by scanning the title
    for directional language (">75", "above", "below", "74-75°", etc.) before
    falling back to the ticker suffix (`-T{N}` = over, `-B{N}` = under).

    Ticker format: ``KX{SERIES}{CITY}-{YYMMDD}-{B|T}{STRIKE}``
      - ``B`` suffix → ``direction="under"`` (market resolves YES if value < strike)
      - ``T`` suffix → ``direction="over"``  (market resolves YES if value > strike)
      - Decimal encoding: ``T695`` → 69.5, ``B65000`` → 65000.0

    Args:
        market: Raw market dict from the Kalshi API.  Expected keys: ``ticker``,
                ``title``.  Missing keys produce empty strings (safe to call on
                any dict).

    Returns:
        ``ParsedMarket`` if the ticker maps to a supported metric; ``None``
        if the ticker prefix is not in ``TICKER_TO_METRIC`` (unrecognised series).
    """
    ticker = market.get("ticker", "")
    title = market.get("title", "")

    metric = _metric_for_ticker(ticker)
    if metric is None:
        return None

    # Try each pattern in order of specificity
    if m := _OVER_RE.search(title):
        return ParsedMarket(ticker, title, metric, "over", strike=_to_float(m.group(1)))

    if m := _UNDER_RE.search(title):
        return ParsedMarket(ticker, title, metric, "under", strike=_to_float(m.group(1)))

    if m := _BETWEEN_RE.search(title):
        lo = _to_float(m.group(1))
        hi = _to_float(m.group(2))
        if lo > hi:
            lo, hi = hi, lo  # guard against reversed bounds in title text
        return ParsedMarket(ticker, title, metric, "between", strike_lo=lo, strike_hi=hi)

    if _UP_RE.search(title):
        return ParsedMarket(ticker, title, metric, "up")

    if _DOWN_RE.search(title):
        return ParsedMarket(ticker, title, metric, "down")

    # Ticker-suffix fallback: title had no directional language (e.g. crypto
    # titles like "Bitcoin price on Mar 16, 2026?").  Extract from the ticker.
    if m := _TICKER_T_RE.search(ticker):
        return ParsedMarket(ticker, title, metric, "over", strike=_to_float(m.group(1)))

    if m := _TICKER_B_RE.search(ticker):
        return ParsedMarket(ticker, title, metric, "under", strike=_to_float(m.group(1)))

    # Precipitation binary fallback: "Will it rain/snow in [city]?" with no numeric
    # strike in the title → treat as "over 0" (any measurable precipitation).
    # Uses precip_prob_* (forecast probability %) as the signal value.
    if metric.startswith(("precip_total", "precip_prob")):
        if re.search(r'\brain\b|\bsnow\b|\bprecipitation\b', title, re.IGNORECASE):
            return ParsedMarket(ticker, title, metric, "over", strike=0.0)

    return ParsedMarket(ticker, title, metric, "unknown")


def parse_all_markets(markets: list[dict[str, Any]]) -> list[ParsedMarket]:
    """Parse a list of Kalshi market dicts, silently dropping unrecognised tickers.

    Convenience wrapper around ``parse_market()`` for bulk processing.
    Unrecognised tickers (not in TICKER_TO_METRIC) are filtered out; the
    caller receives only markets we know how to score and trade.
    """
    parsed = []
    for m in markets:
        result = parse_market(m)
        if result is not None:
            parsed.append(result)
    return parsed


# ---------------------------------------------------------------------------
# Dynamic market discovery
# ---------------------------------------------------------------------------

# Ticker prefixes that indicate a numeric/data series we *might* want to track.
# Any ticker starting with one of these but absent from TICKER_TO_METRIC is
# surfaced as a candidate for manual addition.
_NUMERIC_PATTERN_PREFIXES: tuple[str, ...] = (
    "KXHIGH",                          # daily high temperature by city
    "KXLOWT",                          # daily low temperature by city
    "KXRAIN",                          # daily rain markets (ticker prefix TBD — discovered via scan_unknown_series)
    "KXSNOW",                          # daily snow markets (ticker prefix TBD)
    "KXPRECIP",                        # alternate precipitation prefix (candidate)
    "KXBTC", "KXETH", "KXSOL", "KXXRP",   # crypto prices (BTC/ETH/SOL/XRP)
    "KXDOGE", "KXADA", "KXAVAX", "KXLINK", "KXBNB",  # crypto prices
    "KXEUR", "KXUSD", "KXGBP", "KXJPY",  # forex
    "KXCPI", "KXNFP", "KXADP", "KXUNRATE", "KXPPI", "KXPCE",  # BLS / BEA economic data
    "KXGDP",                           # Real GDP growth rate (BEA via FRED)
    "KXJOBLESS", "KXICSA",            # DOL weekly jobless claims
    "KXISM",                           # ISM PMI indices
    "KXFED", "KXFFR", "KXDGS",        # interest rates / FRED
    "KXWTI", "KXOIL", "KXNATGAS", "KXNG",  # energy / EIA
    "KXSPX", "KXNDX", "KXDOW",        # equity indices
)


def scan_unknown_series(markets: list[dict[str, Any]]) -> None:
    """Log any ticker series that look numeric but aren't in TICKER_TO_METRIC.

    Called once each time the market cache refreshes.  Produces a single
    INFO log line per unrecognised series (with one example ticker), making
    it easy to add new Kalshi markets to TICKER_TO_METRIC without manually
    browsing the API.

    Only tickers whose first dash-delimited segment starts with a known
    numeric-pattern prefix are reported — purely political/text markets are
    not flagged since they don't belong in TICKER_TO_METRIC at all.
    """
    unknown: dict[str, str] = {}  # series_key → first example ticker

    for m in markets:
        ticker = m.get("ticker", "")
        if not ticker:
            continue
        if _metric_for_ticker(ticker) is not None:
            continue  # already tracked

        for prefix in _NUMERIC_PATTERN_PREFIXES:
            if ticker.startswith(prefix):
                # Series key = everything before the first date/strike segment
                series = ticker.split("-")[0]
                if series not in unknown:
                    unknown[series] = ticker
                break

    if unknown:
        lines = [
            f"Dynamic discovery: {len(unknown)} unrecognised numeric series "
            f"(consider adding to TICKER_TO_METRIC):"
        ]
        for series in sorted(unknown):
            lines.append(f"  {series:<18}  e.g. {unknown[series]}")
        logging.info("\n".join(lines))


# ---------------------------------------------------------------------------
# Ticker date parsing
# ---------------------------------------------------------------------------

_TICKER_DATE_RE = re.compile(r"^(\d{2})(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d{2})$")
_MONTH_NUM = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def ticker_date(ticker: str) -> date | None:
    """Parse the market date from a Kalshi ticker (e.g. KXHIGHAUS-26MAR10-T75 → date(2026,3,10))."""
    parts = ticker.split("-")
    if len(parts) < 2:
        return None
    m = _TICKER_DATE_RE.match(parts[1])
    if not m:
        return None
    try:
        return date(2000 + int(m.group(1)), _MONTH_NUM[m.group(2)], int(m.group(3)))
    except (ValueError, KeyError):
        return None
