"""Lookup-table Kelly sizer for band_arb YES entries.

Win probabilities derived from backtest_band_arb_metar.py over Feb–May 2026
(209 priced P75 entries). Laplace-smoothed: p = (wins + 1) / (n + 2).

Primary dimension: overshoot_f = running_max_at_entry - gfs_morning_forecast
  SKIP      lag > 1.5°F  → negative EV, do not trade
  CAUTIOUS  -1.5 to -0.5°F  → 60.7%  (Laplace: 34/56)
  CLOSE     -0.5 to  0.0°F  → 83.8%  (Laplace: 31/37)
  OVERSHOT  ≥ 0.0°F         → 82.7%  (Laplace: 67/81)
  NO_GFS    no forecast      → 70.0%  (conservative prior)

The bot's existing entry gate (yes_ask ∈ [50, 85]¢) already filters bad-price
trades, so price is not a separate dimension here.
"""

from __future__ import annotations

_TABLE: dict[str, float] = {
    "cautious": 0.607,
    "close":    0.838,
    "overshot": 0.827,
    "no_gfs":   0.700,
}


def _bucket(overshoot_f: float | None) -> str:
    if overshoot_f is None:
        return "no_gfs"
    if overshoot_f < -1.5:
        return "skip"
    if overshoot_f < -0.5:
        return "cautious"
    if overshoot_f < 0.0:
        return "close"
    return "overshot"


def win_prob(overshoot_f: float | None) -> float | None:
    """Return estimated win probability, or None if trade should be skipped.

    Args:
        overshoot_f: running_max - gfs_morning_forecast in °F.
                     Positive = overshot, negative = lagging. None if no GFS.
    Returns:
        float in (0, 1), or None to skip the trade.
    """
    b = _bucket(overshoot_f)
    if b == "skip":
        return None
    return _TABLE[b]
