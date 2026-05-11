"""Shared utility helpers used across the kalshi_bot package."""

from __future__ import annotations

import os
from datetime import datetime


# ---------------------------------------------------------------------------
# ISO 8601 datetime parsing
# ---------------------------------------------------------------------------

def parse_iso_dt(s: str) -> datetime:
    """Parse an ISO 8601 datetime string, normalising the Z UTC suffix.

    Python < 3.11 does not accept 'Z' in fromisoformat; this wrapper
    centralises the workaround so upgrading to 3.11+ only needs one change.
    """
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


# ---------------------------------------------------------------------------
# Typed environment-variable accessors
# ---------------------------------------------------------------------------
# All helpers raise ValueError at import time if the value in .env cannot
# be parsed, so misconfiguration surfaces immediately on startup rather
# than as a cryptic error inside a running poll cycle.

def env_float(name: str, default: float) -> float:
    """Return the named env var as a float, or *default* if unset."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(
            f"Environment variable {name}={raw!r} cannot be parsed as float"
        ) from None


def env_int(name: str, default: int) -> int:
    """Return the named env var as an int, or *default* if unset."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(
            f"Environment variable {name}={raw!r} cannot be parsed as int"
        ) from None


def env_bool(name: str, default: bool) -> bool:
    """Return the named env var as a bool, or *default* if unset.

    Accepts 'true'/'false' (case-insensitive).  Raises ValueError for
    anything else so typos are caught at startup.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    norm = raw.strip().lower()
    if norm == "true":
        return True
    if norm == "false":
        return False
    raise ValueError(
        f"Environment variable {name}={raw!r} must be 'true' or 'false'"
    )
