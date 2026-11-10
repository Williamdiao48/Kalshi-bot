"""Band-NO classifier v3: point-in-time features shared by training and serving.

Why v3 exists (audit 2026-11-07). The v1/v2 classifier had three defects that
made its live record a verdict on a broken pipeline rather than on the model:

  1. Training look-ahead: its HRRR / Open-Meteo daily values came from Open-Meteo's
     historical-forecast API, which splices the freshest runs (for HRRR, close to
     an analysis), aggregated on UTC days.
  2. Train/serve skew: 9 of v2's 33 features were never computed live and were
     served as 0.0; trailing skill was a constant 3.0 once its static cache ran out.
  3. Date bug: the live loggers never checked the ticker's date, so most rows
     compared today's running max/min against TOMORROW's bands.

v3 fixes all three by construction. Every feature is computed by ONE function,
`band_features(day_state(rows, asof), ...)`, from rows of `forecast_shadow_log`
logged at or before the decision time `asof` for the contract's own date. The
training builder and the live logger both call it; `build_classifier_v3_data.py
--selftest` re-derives training rows through the live SQL path and asserts equality.

Decision times are the hourly METAR running-max/min rows the bot writes into
forecast_shadow_log, so `asof` is always a real logging instant, identically
reachable offline and live.
"""
from __future__ import annotations

import math
import pickle
import re
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

from .cities import CITIES, LOW_CITIES
from .market_parser import TICKER_TO_METRIC

MODEL_VERSION = "v3"
MODEL_DIR = Path("data/models")


def model_path(is_high: bool) -> Path:
    return MODEL_DIR / f"classifier_v3_{'high' if is_high else 'low'}.pkl"


# series -> (forecast_shadow_log city name, is_high). Derived from the bot's own
# series->metric map so serving and training agree on which station a series is.
_REG = {**CITIES, **LOW_CITIES}
SERIES_CITY: dict[str, tuple[str, bool]] = {
    s: (_REG[m][0], m.startswith("temp_high_"))
    for s, m in TICKER_TO_METRIC.items()
    if m.startswith(("temp_high_", "temp_low_")) and m in _REG
}

# STABLE city code, from the canonical registry (never dataset-relative).
CITY_CODE: dict[str, int] = {c: i for i, c in enumerate(sorted({v[0] for v in _REG.values()}))}
# Fixed-offset LOCAL STANDARD TIME per city: NWS settles midnight-to-midnight LST
# year-round, and the METAR running max/min resets on the same boundary.
_CITY_LST: dict[str, timezone] = {
    v[0]: timezone(v[3].utcoffset(datetime(2000, 1, 15))) for v in _REG.values()
}

OBS_SOURCE = "metar"   # running max (HIGH) / min (LOW) for the LST day, °F
FORECAST_SOURCES = (
    "hrrr", "noaa", "nws_hourly", "open_meteo", "open_meteo_gfs",
    "open_meteo_ecmwf", "weatherapi",   # last two are logged for HIGH only
)
SOURCES = (OBS_SOURCE, *FORECAST_SOURCES)
FORECAST_STALE_H = 12.0   # ignore a source's forecast older than this at asof

# Model domain (identical for training rows and live scoring).
#   HIGH: running max from 2°F below the band ceiling up to the point where it
#         rounds above the ceiling. At margin >= +0.5 the CLI high is already >
#         ceil, so the band has settled NO and there is nothing to predict.
#   LOW:  running min still above the ceiling (it can yet fall into the band).
GATE = {True: (-2.0, 0.5), False: (0.3, math.inf)}

FEATURES = [
    "margin_f", "local_hour", "delta_1h", "delta_3h",
    "consensus_vs_ceil", "obs_vs_consensus", "model_spread", "n_fc", "n_fc_above_ceil",
    *[f"{s}_vs_ceil" for s in FORECAST_SOURCES],
    "city_code",
]

_BAND_RE = re.compile(r"^([A-Z]+)-(\d{2}[A-Z]{3}\d{2})-B(\d+(?:\.\d+)?)$")


def parse_band_ticker(ticker: str) -> tuple[str, date, int, int] | None:
    """KXHIGHCHI-26SEP15-B71.5 -> ('KXHIGHCHI', date(2026,9,15), 71, 72)."""
    m = _BAND_RE.match(ticker)
    if not m or m.group(1) not in SERIES_CITY:
        return None
    d = datetime.strptime(m.group(2), "%y%b%d").date()
    band_lo = int(float(m.group(3)) - 0.5)
    return m.group(1), d, band_lo, band_lo + 1


def lst_midnight_utc(city: str, d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=_CITY_LST[city]).astimezone(timezone.utc)


def lst_date(city: str, t: datetime) -> date:
    return t.astimezone(_CITY_LST[city]).date()


def parse_ts(s: str) -> datetime:
    t = datetime.fromisoformat(s)
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


Row = tuple[datetime, str, float]   # (logged_at UTC, source, value °F)


def load_rows(conn: sqlite3.Connection, city: str, date_target: date | str,
              is_high: bool) -> list[Row]:
    """All feature-source rows for one city-day, sorted by logging time.

    This is the live path. The offline builder bulk-loads the same columns and
    groups them identically (see build_classifier_v3_data.load_all_rows).
    """
    q = (f"SELECT logged_at, source, forecast_f FROM forecast_shadow_log "
         f"WHERE city = ? AND date_target = ? AND is_high = ? "
         f"AND source IN ({','.join('?' * len(SOURCES))})")
    rows = conn.execute(q, (city, str(date_target), int(is_high), *SOURCES)).fetchall()
    return sort_rows([(parse_ts(t), s, float(v)) for t, s, v in rows if v is not None])


def sort_rows(rows: list[Row]) -> list[Row]:
    return sorted(rows, key=lambda r: (r[0], r[1]))


def obs_times(rows: list[Row]) -> list[datetime]:
    """Decision instants: every logged METAR running-obs row of the city-day."""
    return [t for t, s, _ in rows if s == OBS_SOURCE]


def day_state(rows: list[Row], asof: datetime, city: str, d: date) -> dict[str, Any] | None:
    """Everything knowable about the city-day at `asof`, band-independent."""
    obs = [(t, v) for t, s, v in rows if s == OBS_SOURCE and t <= asof]
    if not obs:
        return None

    def obs_before(hours: float) -> float:
        cut = asof - timedelta(hours=hours)
        prior = [v for t, v in obs if t <= cut]
        return prior[-1] if prior else math.nan

    stale = asof - timedelta(hours=FORECAST_STALE_H)
    fc: dict[str, float] = {}
    for t, s, v in rows:                      # sorted, so the latest value wins
        if s != OBS_SOURCE and stale <= t <= asof:
            fc[s] = v
    return {
        "obs_now": obs[-1][1],
        "obs_1h": obs_before(1.0),
        "obs_3h": obs_before(3.0),
        "local_hour": (asof - lst_midnight_utc(city, d)).total_seconds() / 3600.0,
        "fc": fc,
    }


def band_features(state: dict[str, Any], band_ceil: int, city: str) -> dict[str, float]:
    ceil = float(band_ceil)
    fc = state["fc"]
    vals = [fc[s] for s in FORECAST_SOURCES if s in fc]
    cons = median(vals) if vals else math.nan
    out = {
        "margin_f": state["obs_now"] - ceil,
        "local_hour": state["local_hour"],
        "delta_1h": state["obs_now"] - state["obs_1h"],
        "delta_3h": state["obs_now"] - state["obs_3h"],
        "consensus_vs_ceil": cons - ceil,
        "obs_vs_consensus": state["obs_now"] - cons,
        "model_spread": (max(vals) - min(vals)) if len(vals) >= 2 else math.nan,
        "n_fc": float(len(vals)),
        "n_fc_above_ceil": float(sum(v >= ceil + 0.5 for v in vals)),
        "city_code": float(CITY_CODE[city]),
    }
    for s in FORECAST_SOURCES:
        out[f"{s}_vs_ceil"] = fc[s] - ceil if s in fc else math.nan
    return out


def in_domain(margin_f: float, is_high: bool) -> bool:
    lo, hi = GATE[is_high]
    return lo <= margin_f < hi


# --- model bundle ---------------------------------------------------------------

def load_model(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("rb") as f:
        b = pickle.load(f)
    if b.get("version") != MODEL_VERSION or b.get("features") != FEATURES:
        raise ValueError(f"{path}: not a {MODEL_VERSION} bundle with the current FEATURES")
    return b


# Isotonic steps can hit exactly 0 or 1 at the edges of the calibration set; a
# hard 0/1 on a weather outcome is never warranted and makes log-loss infinite.
P_CLIP = (0.005, 0.995)


def predict_frame(bundle: dict[str, Any], X: "pd.DataFrame") -> "np.ndarray":
    """Calibrated, clipped P(NO wins) for a FEATURES frame (training eval + serving)."""
    import numpy as np

    raw = bundle["lgbm"].predict_proba(X[FEATURES].astype(float))[:, 1]
    return np.clip(bundle["iso"].predict(raw), *P_CLIP)


def predict(bundle: dict[str, Any], feats: list[dict[str, float]]) -> list[float]:
    """Live wrapper: one probability per feature dict."""
    import pandas as pd

    return [float(p) for p in predict_frame(bundle, pd.DataFrame(feats, columns=FEATURES))]


# --- live scoring -----------------------------------------------------------------

MAX_OBS_AGE = timedelta(minutes=75)


def features_at(conn: sqlite3.Connection, city: str, d: date, is_high: bool,
                bands: list[tuple[str, int]], now: datetime
                ) -> tuple[datetime, list[dict[str, Any]]] | None:
    """In-domain feature rows for one city-day at the latest METAR decision instant.

    `asof` is the newest logged METAR row at or before `now` (not `now` itself),
    so each feature row is exactly what the offline builder produces for that
    instant. Returns None unless `now` falls on the contract's own LST date and
    that METAR row is fresh.
    """
    if lst_date(city, now) != d:
        return None
    rows = load_rows(conn, city, d, is_high)
    times = [t for t in obs_times(rows) if t <= now]
    if not times or now - times[-1] > MAX_OBS_AGE:
        return None
    asof = times[-1]
    st = day_state(rows, asof, city, d)
    if st is None:
        return None
    out = []
    for ticker, band_ceil in bands:
        f = band_features(st, band_ceil, city)
        if in_domain(f["margin_f"], is_high):
            out.append({"ticker": ticker, "band_ceil": band_ceil, "feats": f})
    return asof, out
