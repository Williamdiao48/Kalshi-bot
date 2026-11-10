"""Kalshi trading-fee models for P&L accounting in evals.

Kalshi's taker fee is charged per ORDER and rounded UP to the next cent:

    fee = ceil_to_cent(0.07 * C * P * (1 - P))      C contracts at price P ($)

so it peaks at mid prices (1.75c/contract at 50c before rounding) and the
rounding dominates small orders (a 1-lot always pays at least 1c). The older
evals used a flat 1c/contract; FEE_SCHEDULES keeps that next to the real
schedule so results can be quoted under both.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

TAKER_RATE = 0.07
FLAT_FEE = 0.01


def kalshi_taker_fee(price, contracts: int = 1) -> np.ndarray:
    """Per-CONTRACT taker fee ($) for an order of `contracts` at `price` ($)."""
    p = np.asarray(price, dtype=float)
    raw_cents = TAKER_RATE * contracts * p * (1.0 - p) * 100.0
    # round() first so float noise (e.g. 1.0000000002) doesn't add a cent
    return np.ceil(np.round(raw_cents, 6)) / 100.0 / contracts


def flat_fee(price) -> np.ndarray:
    return np.full(np.shape(price), FLAT_FEE, dtype=float)


FEE_SCHEDULES: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "flat 1c": flat_fee,
    "kalshi taker, 1-lot": lambda p: kalshi_taker_fee(p, 1),
    "kalshi taker, 100-lot": lambda p: kalshi_taker_fee(p, 100),
}
