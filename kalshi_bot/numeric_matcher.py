"""Match live numeric DataPoints against parsed Kalshi markets.

For each (DataPoint, ParsedMarket) pair sharing the same metric, we compute:
  - Whether the live value is currently on the YES or NO side of the strike
  - The edge: how far the value sits from the strike (larger = stronger signal)

A NumericOpportunity is emitted for every matching pair so the operator
can inspect current data vs. market pricing and decide whether to act.

Cross-exchange price confirmation
----------------------------------
When both ``binance`` and ``coinbase`` DataPoints are present for the same
metric, the matcher checks whether the two exchanges agree on the price.
Large inter-exchange divergence (> CROSS_EXCHANGE_DIVERGENCE_PCT) indicates
a data anomaly (API glitch, stale feed, or exchange-specific event) and is
penalised by multiplying the effective edge by CROSS_EXCHANGE_EDGE_PENALTY.

This means opportunities near the strike that only pass min_edge because of
a rogue Binance or Coinbase tick will be filtered out automatically.

Environment variables
---------------------
  CROSS_EXCHANGE_DIVERGENCE_PCT  Maximum acceptable price divergence between
                                 Binance and Coinbase, as a fraction (e.g.
                                 "0.005" = 0.5%).  Default: 0.005.
  CROSS_EXCHANGE_EDGE_PENALTY    Multiplier applied to edge when exchanges
                                 diverge beyond the threshold.  Default: 0.5
                                 (halve the effective edge).  Set to 0.0 to
                                 completely suppress divergent signals.
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .data import DataPoint
from .market_parser import ParsedMarket, parse_all_markets

# ---------------------------------------------------------------------------
# Metrics to skip entirely (zero-liquidity / no signal)
# ---------------------------------------------------------------------------

# Comma-separated metric names to exclude from opportunity generation.
# Default: "price_doge_usd" — KXDOGE has zero bid/ask depth and direction=unknown,
# generating thousands of useless UNKNOWN-outcome opportunities each cycle.
# Override via env: SKIP_METRICS="price_doge_usd,price_ada_usd"
_SKIP_METRICS_RAW = os.environ.get("SKIP_METRICS", "price_doge_usd")
SKIP_METRICS: frozenset[str] = frozenset(
    m.strip() for m in _SKIP_METRICS_RAW.split(",") if m.strip()
)

# ---------------------------------------------------------------------------
# Cross-exchange confirmation config
# ---------------------------------------------------------------------------

CROSS_EXCHANGE_DIVERGENCE_PCT: float = float(
    os.environ.get("CROSS_EXCHANGE_DIVERGENCE_PCT", "0.005")
)
CROSS_EXCHANGE_EDGE_PENALTY: float = float(
    os.environ.get("CROSS_EXCHANGE_EDGE_PENALTY", "0.5")
)

# Sources that participate in cross-exchange confirmation.
_CROSS_EXCHANGE_SOURCES: frozenset[str] = frozenset({"binance", "coinbase"})

# Edge multiplier applied to CoinGecko DataPoints when neither Binance nor
# Coinbase has data for that metric (i.e. CoinGecko is the sole crypto source).
# CoinGecko is a single-source, third-party aggregator with no cross-exchange
# validation — apply a conservative penalty to discount unconfirmed signals.
# Set to 1.0 to disable.  Default: 0.70 (30% discount on unvalidated edges).
COINGECKO_FALLBACK_PENALTY: float = float(
    os.environ.get("COINGECKO_FALLBACK_PENALTY", "0.70")
)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class NumericOpportunity:
    """A live data value matched against an open Kalshi market."""

    metric: str
    data_value: float
    unit: str
    source: str
    as_of: str
    market_ticker: str
    market_title: str
    current_market_price: Any   # cents or "N/A"
    direction: str              # "over" | "under" | "between" | "up" | "down"
    strike: float | None        # None for direction-only markets
    strike_lo: float | None
    strike_hi: float | None
    implied_outcome: str        # "YES" | "NO" | "UNKNOWN"
    edge: float                 # |value - strike|; 0.0 if strike is None
    # Cross-exchange divergence info (only set for binance/coinbase metrics)
    cross_exchange_divergence: float | None = None  # fractional divergence
    # Hours remaining until the market closes (from close_time / expiration_time).
    # Used by trade_executor to time-scale crypto uncertainty (σ ∝ √t).
    hours_to_close: float | None = None
    # Extra source-specific fields from DataPoint.metadata (e.g. ensemble_spread)
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Cross-exchange helpers
# ---------------------------------------------------------------------------

def _build_exchange_prices(
    data_points: list[DataPoint],
) -> dict[str, dict[str, float]]:
    """Return {metric: {source: price}} for all cross-exchange sources.

    Only sources in _CROSS_EXCHANGE_SOURCES are included.
    If multiple DataPoints share the same metric+source, the latest value wins
    (iteration order is preserved so the last one in the list wins).
    """
    prices: dict[str, dict[str, float]] = {}
    for dp in data_points:
        if dp.source in _CROSS_EXCHANGE_SOURCES:
            prices.setdefault(dp.metric, {})[dp.source] = dp.value
    return prices


def _divergence_penalty(
    source: str,
    metric: str,
    exchange_prices: dict[str, dict[str, float]],
) -> tuple[float, float | None]:
    """Return (edge_multiplier, divergence_fraction) for a DataPoint.

    If both Binance and Coinbase prices are available and they diverge beyond
    CROSS_EXCHANGE_DIVERGENCE_PCT, the edge multiplier is CROSS_EXCHANGE_EDGE_PENALTY.
    Otherwise the multiplier is 1.0 (no penalty).

    Returns:
        edge_multiplier:     1.0 or CROSS_EXCHANGE_EDGE_PENALTY
        divergence_fraction: Fractional price divergence, or None if only
                             one exchange price is available.
    """
    if source not in _CROSS_EXCHANGE_SOURCES:
        # CoinGecko fallback: when neither Binance nor Coinbase has data for
        # this metric, CoinGecko is unvalidated single-source.  Apply penalty.
        if source == "coingecko" and COINGECKO_FALLBACK_PENALTY < 1.0:
            by_source = exchange_prices.get(metric, {})
            if not by_source:  # no Binance or Coinbase data for this metric
                logging.debug(
                    "CoinGecko fallback penalty %.2f applied to %s"
                    " (no Binance/Coinbase data available)",
                    COINGECKO_FALLBACK_PENALTY, metric,
                )
                return COINGECKO_FALLBACK_PENALTY, None
        return 1.0, None

    by_source = exchange_prices.get(metric, {})
    binance_p  = by_source.get("binance")
    coinbase_p = by_source.get("coinbase")

    if binance_p is None or coinbase_p is None:
        return 1.0, None  # can't compare with only one source

    # Symmetric divergence: |a - b| / min(a, b)
    denom = min(binance_p, coinbase_p)
    if denom <= 0:
        return 1.0, None

    div = abs(binance_p - coinbase_p) / denom

    if div > CROSS_EXCHANGE_DIVERGENCE_PCT:
        logging.warning(
            "Cross-exchange divergence on %s: Binance=%.4f Coinbase=%.4f"
            " (%.3f%% > %.3f%%) — applying edge penalty %.2f",
            metric, binance_p, coinbase_p,
            div * 100, CROSS_EXCHANGE_DIVERGENCE_PCT * 100,
            CROSS_EXCHANGE_EDGE_PENALTY,
        )
        return CROSS_EXCHANGE_EDGE_PENALTY, div

    return 1.0, div


# ---------------------------------------------------------------------------
# Matching logic
# ---------------------------------------------------------------------------

def _implied_outcome(
    value: float,
    pm: ParsedMarket,
    pct_change: float | None = None,
) -> tuple[str, float]:
    """Return (implied_outcome, edge) for a data value against a parsed market.

    For direction-only (up/down) markets, ``pct_change`` provides the
    inter-cycle momentum signal (fractional price change since last poll).
    Edge is expressed as the absolute dollar move: |pct_change| × value.
    """
    if pm.direction == "over" and pm.strike is not None:
        edge = value - pm.strike
        # >= 0 so "at least N" resolves YES when value == strike (inclusive)
        return ("YES" if edge >= 0 else "NO"), abs(edge)

    if pm.direction == "under" and pm.strike is not None:
        edge = pm.strike - value
        # >= 0 so "at most N" resolves YES when value == strike (inclusive)
        return ("YES" if edge >= 0 else "NO"), abs(edge)

    if pm.direction == "between" and pm.strike_lo is not None and pm.strike_hi is not None:
        in_range = pm.strike_lo <= value <= pm.strike_hi
        if in_range:
            # Clearance: how far the value sits from the nearest boundary.
            # Large clearance = firmly inside range = strong YES signal.
            edge = min(value - pm.strike_lo, pm.strike_hi - value)
        else:
            # Distance from the nearest boundary for NO signals.
            edge = min(abs(value - pm.strike_lo), abs(value - pm.strike_hi))
        return ("YES" if in_range else "NO"), edge

    # Direction-only markets (up/down) — use inter-cycle momentum when available.
    # "up" market: YES if price is rising (pct_change > 0).
    # "down" market: YES if price is falling (pct_change < 0).
    if pm.direction in ("up", "down") and pct_change is not None and pct_change != 0.0:
        price_is_rising = pct_change > 0
        yes_outcome = (pm.direction == "up") == price_is_rising  # XOR-free equivalent
        outcome = "YES" if yes_outcome else "NO"
        edge = abs(pct_change) * value  # dollar magnitude of the move
        return outcome, edge

    return "UNKNOWN", 0.0


def find_numeric_opportunities(
    data_points: list[DataPoint],
    markets: list[dict[str, Any]],
    *,
    min_edge: float = 0.0,
) -> list[NumericOpportunity]:
    """Match a list of live DataPoints against open Kalshi markets.

    For crypto metrics, if both Binance and Coinbase prices are present and
    they diverge by more than CROSS_EXCHANGE_DIVERGENCE_PCT, the effective
    edge is multiplied by CROSS_EXCHANGE_EDGE_PENALTY before the min_edge
    filter is applied.

    Args:
        data_points: Live observations from NOAA, Binance, Coinbase, …
        markets:     Raw market dicts from the Kalshi API.
        min_edge:    Only emit opportunities where effective edge >= min_edge.

    Returns:
        List of NumericOpportunity objects, one per matching (data, market) pair.
    """
    parsed_markets = parse_all_markets(markets)

    # Index parsed markets by metric for fast lookup
    by_metric: dict[str, list[ParsedMarket]] = {}
    for pm in parsed_markets:
        by_metric.setdefault(pm.metric, []).append(pm)

    # Raw market dict lookup by ticker for price and close-time retrieval
    price_by_ticker: dict[str, Any] = {
        m.get("ticker", ""): m.get("last_price", "N/A") for m in markets
    }
    now_utc = datetime.now(timezone.utc)
    close_time_by_ticker: dict[str, float | None] = {}
    for m in markets:
        ticker = m.get("ticker", "")
        ct_str = m.get("close_time") or m.get("expiration_time")
        if ct_str:
            try:
                ct = datetime.fromisoformat(ct_str.replace("Z", "+00:00"))
                close_time_by_ticker[ticker] = max(0.0, (ct - now_utc).total_seconds() / 3600)
            except (ValueError, TypeError):
                close_time_by_ticker[ticker] = None
        else:
            close_time_by_ticker[ticker] = None

    # Build cross-exchange price map for divergence checking
    exchange_prices = _build_exchange_prices(data_points)

    opportunities: list[NumericOpportunity] = []

    for dp in data_points:
        # Skip metrics explicitly excluded from trading (e.g. zero-liquidity DOGE)
        if dp.metric in SKIP_METRICS:
            continue

        candidates = by_metric.get(dp.metric, [])

        # Compute edge multiplier once per DataPoint (same for all its markets)
        multiplier, div = _divergence_penalty(dp.source, dp.metric, exchange_prices)

        # Momentum signal from DataPoint metadata (set by binance.py inter-cycle tracking)
        pct_change: float | None = dp.metadata.get("pct_change") if dp.metadata else None

        for pm in candidates:
            outcome, raw_edge = _implied_outcome(dp.value, pm, pct_change)
            effective_edge = raw_edge * multiplier
            if effective_edge < min_edge:
                continue
            opportunities.append(
                NumericOpportunity(
                    metric=dp.metric,
                    data_value=dp.value,
                    unit=dp.unit,
                    source=dp.source,
                    as_of=dp.as_of,
                    market_ticker=pm.ticker,
                    market_title=pm.title,
                    current_market_price=price_by_ticker.get(pm.ticker, "N/A"),
                    direction=pm.direction,
                    strike=pm.strike,
                    strike_lo=pm.strike_lo,
                    strike_hi=pm.strike_hi,
                    implied_outcome=outcome,
                    edge=effective_edge,
                    cross_exchange_divergence=div,
                    hours_to_close=close_time_by_ticker.get(pm.ticker),
                    metadata=dp.metadata,
                )
            )

    return opportunities
