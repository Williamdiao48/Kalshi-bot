"""Shared data model for all numeric news sources."""

from dataclasses import dataclass, field


@dataclass
class DataPoint:
    """A single numeric observation from a news/data source.

    Attributes:
        source:   Which module produced this (e.g. "noaa", "coingecko").
        metric:   Canonical key used to match against Kalshi market tickers
                  (e.g. "temp_high_lax", "price_btc_usd", "rate_eur_usd").
        value:    The numeric value of the observation.
        unit:     Human-readable unit string (e.g. "°F", "USD", "USD/JPY").
        as_of:    ISO-8601 timestamp or date string for when the value applies.
        metadata: Optional extra fields (forecast details, series ID, etc.).
    """

    source: str
    metric: str
    value: float
    unit: str
    as_of: str
    metadata: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.metric}={self.value}{self.unit} ({self.source} @ {self.as_of})"
