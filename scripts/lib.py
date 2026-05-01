"""Shared utilities for Kalshi-Bot analysis and backtest scripts.

Import pattern (scripts run from project root):
    from scripts.lib import pnl, outcome_label, init_schema, open_db
"""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants (mirror trade_executor / exit_manager defaults)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "opportunity_log.db"

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
