"""Lookup-table Kelly sizer for band_arb_low warm-NO entries (KXLOWT).

Win probabilities derived from backtest_band_arb_low_metar.py (P75+0h,
Mar–May 2026).  Laplace-smoothed.

Dimensions
----------
margin_f : running_min − band_ceil (°F) — always positive for warm-NO
  skip        < 1.5°F  → marginal/negative EV; bot gates on this; sizer returns None
  thin     1.5 – 2.0°F → 87.4% WR (Section 5, n=183)
  medium   2.0 – 3.0°F → 89.6% WR (Section 5, n=470, weighted avg of 0.5°F bins)
  wide        > 3.0°F  → 94.9% WR (Section 5, n=1989)

gfs_clearance : gfs_daily_min − band_ceil (°F)
  skip      < 0.0°F → GFS forecasts low enters band; bot gates this; sizer returns None
  marginal  0 – +2°F → below current bot gate (+2°F); skip (returns None)
  mid      +2 – +3°F → Section 4: 95.5% overall (n=418); minor downward modifier for thin
  high        >+3°F  → Section 4: 93.5% overall (n=1363); Section 11 pos=3 drops to 86%
                       at this clearance level, so slight downward modifier for thin cells
  None              → no GFS data; treated as high (conservative fallback)

Interaction
-----------
Margin_f is the dominant signal.  Within the gate zone (clearance ≥+2°F),
clearance level shows only ~2pp aggregate WR difference, but Section 11
(clearance × band_pos) reveals that thin-margin (shallowest) bands degrade
more at higher clearance (91% → 86% for pos=3 as clearance rises ≥2→≥3°F).
Wide-margin (deepest) bands are unaffected (96% at both levels).

Kelly scale
-----------
Applied on top of LOCKED_OBS_KELLY_FRACTION (0.75).
  wide:   1.0× → effective 0.75 Kelly
  medium: 0.75× → effective 0.5625 Kelly
  thin:   0.5×  → effective 0.375 Kelly
"""

from __future__ import annotations

# Clearance bucket → margin bucket → win probability (Laplace-smoothed)
_WIN_PROB: dict[str, dict[str, float]] = {
    "mid": {
        "thin":   0.875,
        "medium": 0.905,
        "wide":   0.951,
    },
    "high": {
        # thin degrades at higher clearance (Section 11 pos=3 ≥3°F = 86%)
        "thin":   0.863,
        "medium": 0.896,
        "wide":   0.946,
    },
}

# Kelly scale multiplier applied on top of LOCKED_OBS_KELLY_FRACTION
_KELLY_SCALE: dict[str, float] = {
    "thin":   0.50,
    "medium": 0.75,
    "wide":   1.00,
}


def _clearance_bucket(gfs_clearance: float | None) -> str:
    if gfs_clearance is None:
        return "high"  # no GFS → conservative fallback
    if gfs_clearance < 2.0:
        return "skip"
    if gfs_clearance < 3.0:
        return "mid"
    return "high"


def _margin_bucket(margin_f: float) -> str:
    if margin_f < 1.5:
        return "skip"
    if margin_f < 2.0:
        return "thin"
    if margin_f < 3.0:
        return "medium"
    return "wide"


def win_prob(margin_f: float, gfs_clearance: float | None = None) -> float | None:
    """Return estimated win probability, or None if trade should be skipped.

    Args:
        margin_f:      running_min − band_ceil in °F.  Must be > 0 for warm-NO.
        gfs_clearance: gfs_daily_min − band_ceil in °F.  None if no GFS data.
    Returns:
        float in (0, 1), or None to skip the trade.
    """
    cb = _clearance_bucket(gfs_clearance)
    mb = _margin_bucket(margin_f)
    if cb == "skip" or mb == "skip":
        return None
    return _WIN_PROB[cb][mb]


def kelly_scale(margin_f: float) -> float:
    """Return Kelly fraction multiplier (0 < scale ≤ 1.0).

    Multiply the base LOCKED_OBS_KELLY_FRACTION by this value before sizing.
    Returns 0.0 if margin_f is below the skip threshold (< 1.5°F).

    Args:
        margin_f: running_min − band_ceil in °F.
    """
    mb = _margin_bucket(margin_f)
    if mb == "skip":
        return 0.0
    return _KELLY_SCALE[mb]
