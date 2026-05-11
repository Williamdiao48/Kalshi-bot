"""Shared utilities for Kalshi-Bot analysis and backtest scripts.

Import pattern (scripts run from project root):
    from scripts.lib import pnl, outcome_label, init_schema, open_db
"""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import date, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants (mirror trade_executor / exit_manager defaults)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_DB_PATH            = PROJECT_ROOT / "data" / "db" / "opportunity_log.db"
DEFAULT_CANDLESTICK_DB_PATH = PROJECT_ROOT / "data" / "candlesticks.db"

# DST-adjusted UTC offsets (summer). Negative = hours behind UTC.
CITY_UTC_OFFSET: dict[str, int] = {
    "ny":   -4, "bos":  -4, "mia":  -4, "phi":  -4, "phl":  -4, "atl":  -4,
    "dc":   -4, "dca":  -4,                                                   # Eastern
    "chi":  -5, "dal":  -5, "dfw":  -5, "hou":  -5, "okc":  -5,
    "sat":  -5, "aus":  -5, "min":  -5, "msp":  -5,
    "msy":  -5, "nola": -5,                                                   # Central
    "den":  -6,                                                                # Mountain
    "lax":  -7, "sfo":  -7, "sea":  -7, "las":  -7, "lv":   -7, "phx": -7,  # Pacific
}
# UTC hour at which each city's market closes (midnight local → UTC).
CITY_CLOSE_UTC_HOUR: dict[str, int] = {k: abs(v) for k, v in CITY_UTC_OFFSET.items()}

# Weather observation sources (ground truth — not forecasts).
GROUND_TRUTH_SOURCES: frozenset[str] = frozenset({
    "metar_6hr", "metar", "metar_running_max", "noaa_observed", "nws_climo",
})

# Forecast model sources, ordered by typical near-term accuracy.
FORECAST_SOURCES: tuple[str, ...] = (
    "nws_hourly", "noaa", "noaa_day2", "hrrr",
    "open_meteo", "open_meteo_gfs", "open_meteo_ecmwf",
    "open_meteo_gem", "open_meteo_icon", "weatherapi",
)

STARTING_CAPITAL_CENTS: float = 10_000.0   # $100
DRAWDOWN_WINDOW_HOURS:  int   = 48
DRAWDOWN_FULL_REDUCE:   float = 0.20
DRAWDOWN_MIN_FACTOR:    float = 0.25
TRADE_MIN_SCORE:        float = 0.75
SCORE_WEIGHT_MIN:       float = 0.25

ALL_CATS = ("temp_high", "temp_low", "crypto", "forex", "other")

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def open_db(path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open opportunity_log.db (or a custom path) with row_factory set."""
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


# Schema detection — some columns were added in later migrations.
_schema_cache: dict[int, set[str]] = {}

def _cols(db: sqlite3.Connection) -> set[str]:
    key = id(db)
    if key not in _schema_cache:
        _schema_cache[key] = {row[1] for row in db.execute("PRAGMA table_info(trades)").fetchall()}
    return _schema_cache[key]


def has_col(db: sqlite3.Connection, col: str) -> bool:
    return col in _cols(db)


def bug_loss_col(db: sqlite3.Connection) -> str:
    """SQL expression for bug_loss safe across schema versions."""
    return "bug_loss" if has_col(db, "bug_loss") else "0"

# ---------------------------------------------------------------------------
# Trade row helpers
# ---------------------------------------------------------------------------

def bug_loss(trade: sqlite3.Row) -> bool:
    try:
        return bool(trade["bug_loss"])
    except Exception:
        return False


def ticker_cat(ticker: str) -> str:
    t = (ticker or "").upper()
    if t.startswith(("KXHIGHT", "KXHIGH")):
        return "temp_high"
    if t.startswith("KXLOWT"):
        return "temp_low"
    if t.startswith(("KXBTC", "KXETH", "KXSOL")):
        return "crypto"
    if t.startswith(("KXUSDJPY", "KXEURUSD", "KXGBPUSD")):
        return "forex"
    return "other"


def note(trade: sqlite3.Row) -> dict:
    try:
        return json.loads(trade["note"]) if trade["note"] else {}
    except Exception:
        return {}


def pnl(trade: sqlite3.Row) -> float | None:
    """Realized P&L in cents.  Uses exit_pnl_cents when present, else reconstructs from settlement."""
    if trade["exit_pnl_cents"] is not None:
        return float(trade["exit_pnl_cents"])
    outcome = trade["outcome"]
    side    = trade["side"]
    count   = trade["count"]
    lim     = trade["limit_price"]
    if outcome == "won":
        return float((100 - lim) * count if side == "yes" else lim * count)
    if outcome == "lost":
        return float(-lim * count if side == "yes" else -(100 - lim) * count)
    return None


def outcome_label(trade: sqlite3.Row) -> str:
    out    = trade["outcome"]
    reason = trade["exit_reason"]
    exited = trade["exited_at"]
    if out is None and exited is None:
        return "OPEN"
    if out == "void":
        return "VOID"
    if out == "won":
        return "PT(WIN)" if reason == "profit_take" else "WIN"
    if out == "lost":
        if reason == "stop_loss":
            return "SL(LOSS)"
        if reason == "profit_take":
            return "PT(LOSS)"
        return "LOSS"
    if exited:
        return f"EXIT({reason or '?'})"
    return "PENDING"


def cost_cents(trade: sqlite3.Row) -> int:
    """Cost per contract: YES pays limit_price, NO pays 100 - limit_price."""
    lim = trade["limit_price"]
    return lim if trade["side"] == "yes" else (100 - lim)


def win_prob(trade: sqlite3.Row) -> float | None:
    """P(our bet wins), accounting for two p_estimate conventions."""
    p = trade["p_estimate"]
    if p is None:
        return None
    sig = trade["signal_p_yes"]
    if sig is not None:
        return (1.0 - p) if trade["side"] == "no" else p
    return p


def score_factor(score: float) -> float:
    if TRADE_MIN_SCORE >= 1.0:
        return 1.0
    f = SCORE_WEIGHT_MIN + (1.0 - SCORE_WEIGHT_MIN) * (score - TRADE_MIN_SCORE) / (1.0 - TRADE_MIN_SCORE)
    return max(SCORE_WEIGHT_MIN, min(1.0, f))


def kelly_count(wp: float, cost: int, kf: float, pos_max: int) -> int:
    """Raw Kelly contract count before score weighting and max(1,...) floor."""
    raw_f = (wp - cost / 100.0) / (1.0 - cost / 100.0)
    if raw_f <= 0:
        return 0
    return math.floor(kf * raw_f * pos_max / cost)


# ---------------------------------------------------------------------------
# Timestamp / city utilities
# ---------------------------------------------------------------------------

def parse_iso_ts(iso: str) -> int:
    """ISO 8601 string → Unix seconds. Returns 0 on parse failure."""
    try:
        return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp())
    except (ValueError, AttributeError):
        return 0


def city_from_ticker(ticker: str) -> str:
    """Extract lowercase city code from a Kalshi temperature ticker.

    Examples: 'KXHIGHTBOS-26APR20-B52.5' → 'bos', 'KXLOWTDEN-26APR19-T31' → 'den'.
    Returns '' for non-temperature tickers.
    """
    for prefix in ("KXHIGHT", "KXHIGH", "KXLOWT", "KXLOW"):
        if ticker.upper().startswith(prefix):
            return ticker[len(prefix):].split("-")[0].lower()
    return ""


def city_from_metric(metric: str) -> str:
    """Extract city code from a metric string. 'temp_high_bos' → 'bos'."""
    return metric.replace("temp_high_", "").replace("temp_low_", "")


def local_hour_from_utc(utc_iso: str, city: str) -> int | None:
    """Approximate local hour (0–23) for a UTC timestamp and city code.

    Uses summer (DST) offsets from CITY_UTC_OFFSET. Returns None if the city
    is not in the table or the timestamp cannot be parsed.
    """
    offset = CITY_UTC_OFFSET.get(city)
    ts = parse_iso_ts(utc_iso)
    if offset is None or ts == 0:
        return None
    from datetime import timezone as _tz
    return (datetime.fromtimestamp(ts, tz=_tz.utc).hour + offset) % 24


def build_market_close_utc(city: str, trade_date: date) -> datetime:
    """UTC datetime when a city's temperature market closes (midnight local).

    Falls back to Central (UTC-5, close_hour=5) for unknown cities.
    """
    from datetime import timedelta, timezone as _tz
    next_day = trade_date + timedelta(days=1)
    close_hour = CITY_CLOSE_UTC_HOUR.get(city, 5)
    return datetime(next_day.year, next_day.month, next_day.day, close_hour, tzinfo=_tz.utc)


# ---------------------------------------------------------------------------
# Weather data helpers
# ---------------------------------------------------------------------------

def get_ground_truth(
    conn: sqlite3.Connection, metric: str, trade_date: str
) -> tuple[float | None, str]:
    """Best available observed temperature for metric + date.

    Priority: metar_6hr → metar (running max) → noaa_observed.
    Returns (actual_f, source_label) or (None, '') when no data found.
    """
    for source, label in [
        ("metar_6hr",     "metar_6hr"),
        ("metar",         "metar_running_max"),
        ("noaa_observed", "noaa_observed"),
    ]:
        row = conn.execute(
            "SELECT MAX(data_value) FROM raw_forecasts "
            "WHERE source = ? AND metric = ? AND date(logged_at) = ?",
            (source, metric, trade_date),
        ).fetchone()
        if row and row[0] is not None:
            return float(row[0]), label
    return None, ""


def load_resolved_trades(
    conn: sqlite3.Connection,
    *,
    kind: "str | list[str] | None" = None,
    exclude_kind: "str | list[str] | None" = None,
    ids: "list[int] | None" = None,
    since: "str | None" = None,
    exit_reason: "str | None" = None,
    columns: str = "*",
) -> "list[sqlite3.Row]":
    """Query resolved trades with composable filters, ordered by id ASC.

    'Resolved' means outcome IS NOT NULL OR exit_pnl_cents IS NOT NULL.

    Args:
        kind:         Filter to one or more opportunity_kind values.
        exclude_kind: Exclude one or more kinds (ignored when kind is set).
        ids:          Filter to specific trade IDs.
        since:        Filter logged_at >= this date string (YYYY-MM-DD or ISO).
        exit_reason:  Filter to a specific exit_reason value.
        columns:      SQL column list (default '*').
    """
    q = (
        f"SELECT {columns} FROM trades "
        "WHERE (outcome IS NOT NULL OR exit_pnl_cents IS NOT NULL)"
    )
    p: list = []

    if ids:
        q += f" AND id IN ({','.join('?' * len(ids))})"
        p.extend(ids)

    if kind is not None:
        kinds = [kind] if isinstance(kind, str) else list(kind)
        q += f" AND opportunity_kind IN ({','.join('?' * len(kinds))})"
        p.extend(kinds)
    elif exclude_kind is not None:
        eks = [exclude_kind] if isinstance(exclude_kind, str) else list(exclude_kind)
        q += f" AND opportunity_kind NOT IN ({','.join('?' * len(eks))})"
        p.extend(eks)

    if since:
        q += " AND logged_at >= ?"
        p.append(since)
    if exit_reason:
        q += " AND exit_reason = ?"
        p.append(exit_reason)

    return conn.execute(q + " ORDER BY id", p).fetchall()


def pnl_from_dict(trade: dict) -> "float | None":
    """Realized P&L in cents from a plain dict (e.g. from cursor.fetchall rows).

    Side-aware: NO wins earn limit_price per contract; YES wins earn 100-limit_price.
    Use lib.pnl() when the trade is a sqlite3.Row.
    """
    if trade.get("exit_pnl_cents") is not None:
        return float(trade["exit_pnl_cents"])
    outcome = trade.get("outcome")
    side    = trade.get("side", "no")
    count   = trade.get("count", 1)
    lim     = trade.get("limit_price", 0)
    if outcome == "won":
        return float((100 - lim) * count if side == "yes" else lim * count)
    if outcome == "lost":
        return float(-lim * count if side == "yes" else -(100 - lim) * count)
    return None
