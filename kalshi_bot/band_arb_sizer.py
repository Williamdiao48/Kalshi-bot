"""Lookup-table Kelly sizer for band_arb YES entries.

Win probabilities derived from backtest_band_arb_metar.py Section 12
(overshoot × trend cross-tab, P75 entries, Feb–May 2026). Laplace-smoothed.

Dimensions
----------
overshoot_f : running_max_at_entry − gfs_morning_forecast (°F)
  SKIP      < −1.5°F        → negative EV, do not trade
  CAUTIOUS  −1.5 to  0.0°F  → GFS still expects higher
  CLOSE      0.0 to +1.0°F  → near GFS forecast
  OVERSHOT  > +1.0°F        → already past GFS forecast

is_rising : bool | None
  True   = spot temp still at/near running max (last METAR reading ≈ daily high)
  False  = running max set earlier; spot has cooled off (plateaued)
  None   = trend unavailable; treated as rising (optimistic fallback)

Kelly scale override
--------------------
cautious + plateaued is marginal (62% win rate, thin EV above ~60¢ ask).
That cell receives 1/3× scale so effective Kelly = 0.75 × 1/3 = 0.25 (quarter Kelly).
no_gfs + plateaued gets the same treatment for consistency.
All other cells use full scale (1.0×).
"""

from __future__ import annotations

# Overshoot × trend probability table (Laplace-smoothed from backtest)
# Keys: bucket → {True: rising_p, False: plateaued_p}
_WIN_PROB: dict[str, dict[bool, float]] = {
    "cautious": {True: 0.860, False: 0.620},
    "close":    {True: 0.897, False: 0.806},
    "overshot": {True: 0.846, False: 0.929},
    "no_gfs":   {True: 0.880, False: 0.620},
}

# Kelly scale multiplier applied on top of LOCKED_OBS_KELLY_FRACTION (0.75)
# 1/3 → effective quarter Kelly for thin-edge cells
_KELLY_SCALE: dict[str, dict[bool, float]] = {
    "cautious": {True: 1.0, False: 1 / 3},
    "close":    {True: 1.0, False: 1.0},
    "overshot": {True: 1.0, False: 1.0},
    "no_gfs":   {True: 1.0, False: 1 / 3},
}


def _bucket(overshoot_f: float | None) -> str:
    if overshoot_f is None:
        return "no_gfs"
    if overshoot_f < -1.5:
        return "skip"
    if overshoot_f < 0.0:
        return "cautious"
    if overshoot_f < 1.0:
        return "close"
    return "overshot"


def _rising_key(is_rising: bool | None) -> bool:
    """Resolve None → True (optimistic fallback when trend is unavailable)."""
    return True if is_rising is None else is_rising


def win_prob(overshoot_f: float | None, is_rising: bool | None = None) -> float | None:
    """Return estimated win probability, or None if trade should be skipped.

    Args:
        overshoot_f: running_max − gfs_morning_forecast in °F.
                     Positive = overshot, negative = lagging. None if no GFS.
        is_rising:   True if last METAR spot is at/near the running max (still rising);
                     False if running max was set earlier (plateaued); None = unknown.
    Returns:
        float in (0, 1), or None to skip the trade.
    """
    b = _bucket(overshoot_f)
    if b == "skip":
        return None
    return _WIN_PROB[b][_rising_key(is_rising)]


def kelly_scale(overshoot_f: float | None, is_rising: bool | None = None) -> float:
    """Return Kelly fraction multiplier (0 < scale ≤ 1.0).

    Multiply the base LOCKED_OBS_KELLY_FRACTION by this value before sizing.
    Returns 1/3 for cautious+plateaued and no_gfs+plateaued (quarter Kelly),
    1.0 for all other cells.

    Args: same as win_prob.
    """
    b = _bucket(overshoot_f)
    if b == "skip":
        return 0.0
    return _KELLY_SCALE[b][_rising_key(is_rising)]
