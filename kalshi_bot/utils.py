"""Shared utility helpers used across the kalshi_bot package."""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import aiohttp


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


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

async def get_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: aiohttp.ClientTimeout | None = None,
    retries: int = 3,
    backoff: float = 2.0,
) -> Any:
    """GET *url*, retrying up to *retries* times on HTTP 429.

    Sleeps backoff * 2^attempt seconds before each retry.
    Raises aiohttp.ClientResponseError on non-429 HTTP errors or after all
    retries are exhausted.  Returns the parsed JSON body on success.
    """
    kwargs: dict[str, Any] = {}
    if params is not None:
        kwargs["params"] = params
    if headers is not None:
        kwargs["headers"] = headers
    if timeout is not None:
        kwargs["timeout"] = timeout

    for attempt in range(retries):
        async with session.get(url, **kwargs) as resp:
            if resp.status == 429 and attempt < retries - 1:
                await asyncio.sleep(backoff * (2 ** attempt))
                continue
            resp.raise_for_status()
            return await resp.json()

    # Should not reach here, but satisfies type checker
    raise aiohttp.ClientResponseError(None, (), status=429, message="Too Many Requests")  # type: ignore[arg-type]
