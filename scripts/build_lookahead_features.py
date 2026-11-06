#!/usr/bin/env python3
"""Phase 2 — point-in-time feature builder for the ahead-of-time HIGH-market model.

Produces ONE leak-safe row per (city, date_target): the forecast state as it was
knowable at a fixed *morning* decision cutoff, BEFORE the afternoon high forms and
before any band breach. This is the deliberate opposite of the current
`forecast_no` model, which only fires post-breach (see project_high_model_degenerate).

The single most important invariant, enforced everywhere below:

    a forecast contributes to a (city, date_target) row ONLY if its `logged_at`
    (UTC) is <= that city-day's decision cutoff (09:00 local by default).

`forecast_shadow_log` is a MELTED table (one row per source per logged hour), so we
take, per source, the LATEST value at or before the cutoff. `logged_at` is UTC;
metar/obs `forecast_f` is the running observed max *as of* logged_at (verified: it
ramps through the day toward the actual), so it is a legitimate morning-trajectory
feature ONLY under the cutoff filter — never the final daily max.

Rolling per-source skill (MAE / bias) uses PRIOR city-days only (lagged by
LABEL_LAG_DAYS), so it never sees the current label.

LABEL (v2): the official NWS CLI daily high from `cli_daily` -- the exact product
Kalshi settles on. (v1 used IEM `max_tmpf` from hourly METAR, which disagreed with
the CLI on ~1/3 of city-days and fell outside the settled band ~1/4 of the time.)

TARGET (v2): the model predicts the CORRECTION to consensus, `actual_f -
consensus_median`, from features expressed relative to consensus. v1 predicted the
absolute temperature, and trees cannot extrapolate: trained on summer, its mu
floored near 65F in autumn (Seattle +8.5F bias). Absolute-level features (raw
forecasts, month, doy) are kept in the parquet for
diagnostics but are NOT model inputs -- see `feature_columns`.

LABEL AVAILABILITY: at the 09:00 decision the bot only has CLI labels through
date_target - 2 (the backfill stores finals >= 2 days old). Trailing skill and the
climate normal therefore lag by LABEL_LAG_DAYS in training too, so train == serve.

Output: a parquet of features + the label (`actual_f` = CLI daily high). The
model itself is Phase 3; this script builds and audits the training matrix only.

Usage:
    venv/bin/python scripts/build_lookahead_features.py            # HIGH, 09:00 local
    venv/bin/python scripts/build_lookahead_features.py --is-high 0 --cutoff 07:00

Outputs are written under data/ (gitignored) and must stay untracked.
"""
from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kalshi_bot.cities import CITIES, LOW_CITIES  # noqa: E402

# display name -> ZoneInfo (both high and low registries share settlement tz)
CITY_TZ = {v[0]: v[3] for v in CITIES.values()}
CITY_TZ.update({v[0]: v[3] for v in LOW_CITIES.values()})

# STABLE city -> integer code, shared by training and live serving. Must NOT be
# dataset-relative (pandas cat.codes) or serve-time single-city rows get code 0 and
# skew from training. Derived from the canonical registry, so it never shifts.
_ALL_CITIES = sorted({v[0] for v in CITIES.values()} | {v[0] for v in LOW_CITIES.values()})
CITY_CODE = {name: i for i, name in enumerate(_ALL_CITIES)}

# --- Source taxonomy (drives which columns become which kind of feature) --------
# Forecast models knowable at a morning cutoff. noaa_dayN are genuine day-ahead
# NWS forecasts (dayN logged ~N days before target); noaa is the same-day run.
CORE_MODELS = [
    "open_meteo", "nws_hourly", "weatherapi", "open_meteo_gfs", "hrrr",
    "noaa", "noaa_day2", "noaa_day3", "noaa_day4",
]
# International ensembles — the ONLY edge thesis (market may underweight these).
ENSEMBLE = ["open_meteo_ecmwf", "open_meteo_icon", "open_meteo_gem", "open_meteo_gfs"]
# Running observed max as-of cutoff = morning trajectory (leak-safe under filter).
OBS = ["metar", "noaa_observed", "nws_asos"]

ALL_SOURCES = sorted(set(CORE_MODELS) | set(ENSEMBLE) | set(OBS))

# Sources to attach trailing skill (MAE/bias) features to — the ones dense enough
# to have a meaningful rolling history without exploding feature width.
SKILL_SOURCES = ["open_meteo", "nws_hourly", "noaa", "open_meteo_ecmwf", "hrrr"]

# Days between a city-day and the newest CLI label available at its 09:00 cutoff.
LABEL_LAG_DAYS = 2

# Model inputs that are not per-source deviations / trailing skill. Everything
# here is relative to consensus or a count, so it is stationary across seasons.
STATIC_FEATURES = [
    "n_core", "model_spread", "model_range", "n_ens", "ens_vs_consensus",
    "morning_obs_vs_consensus", "consensus_anom", "city_code",
]
TARGET = "resid"          # actual_f - consensus_median


def feature_columns(df: pd.DataFrame) -> list[str]:
    """The model's inputs: deviations from consensus, trailing skill, statics."""
    dyn = [c for c in df.columns if c.startswith(("d_", "mae14_", "bias14_"))]
    return sorted(dyn) + [c for c in STATIC_FEATURES if c in df.columns]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default="data/db/opportunity_log.db")
    p.add_argument("--is-high", type=int, default=1, choices=(0, 1))
    p.add_argument("--cutoff", default="09:00",
                   help="local decision time HH:MM on the target date")
    p.add_argument("--mae-window", type=int, default=14,
                   help="trailing city-days for per-source MAE/bias")
    p.add_argument("--min-core", type=int, default=3,
                   help="drop a city-day with fewer than this many core forecasts")
    p.add_argument("--out", default=None)
    return p.parse_args()


def load_rows(db: str, is_high: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the melted shadow log for one direction (UTC-parsed) + CLI labels."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        df = pd.read_sql_query(
            "SELECT logged_at, city, date_target, source, forecast_f "
            "FROM forecast_shadow_log WHERE is_high = ?",
            con, params=(is_high,),
        )
        cli = load_cli(con, is_high)
    finally:
        con.close()
    df["logged_at"] = pd.to_datetime(df["logged_at"], utc=True, format="mixed")
    df["date_target"] = pd.to_datetime(df["date_target"]).dt.date
    return df, cli


def load_cli(con: sqlite3.Connection, is_high: int, city: str | None = None) -> pd.DataFrame:
    """Official CLI labels: (city, date_target, cli_f, normal_f)."""
    val, norm = ("high_f", "high_normal_f") if is_high else ("low_f", "low_normal_f")
    q = f"SELECT city, date AS date_target, {val} AS cli_f, {norm} AS normal_f FROM cli_daily"
    params: tuple = ()
    if city is not None:
        q += " WHERE city = ?"
        params = (city,)
    cli = pd.read_sql_query(q, con, params=params)
    cli["date_target"] = pd.to_datetime(cli["date_target"]).dt.date
    return cli


def cutoff_utc(city: str, d, hhmm: time) -> pd.Timestamp | None:
    """09:00 local on the target date, as a UTC timestamp (DST-correct)."""
    tz = CITY_TZ.get(city)
    if tz is None:
        return None
    local = datetime.combine(d, hhmm).replace(tzinfo=tz)
    return pd.Timestamp(local.astimezone(timezone.utc))


def latest_before_cutoff(df: pd.DataFrame, hhmm: time) -> pd.DataFrame:
    """Per (city, date_target, source): the latest forecast_f at/ before cutoff.

    This is where the leak-free invariant is enforced.
    """
    # one cutoff per (city, date_target)
    keys = df[["city", "date_target"]].drop_duplicates()
    keys["cutoff"] = [cutoff_utc(c, d, hhmm) for c, d in
                      zip(keys["city"], keys["date_target"])]
    df = df.merge(keys, on=["city", "date_target"], how="left")
    df = df[df["cutoff"].notna() & (df["logged_at"] <= df["cutoff"])]
    # latest row per source
    df = df.sort_values("logged_at")
    latest = df.groupby(["city", "date_target", "source"], as_index=False).last()
    return latest


def pivot_features(latest: pd.DataFrame, cli: pd.DataFrame) -> pd.DataFrame:
    """Melted latest-per-source -> wide one row per (city, date_target) + labels.

    `actual_f` is the CLI value for the city-day itself (the label; NaN until
    published). `normal_f` is the CLI climate normal LABEL_LAG_DAYS earlier --
    the newest one actually available at the cutoff (normals move <0.5F/day).
    """
    wide = latest.pivot_table(
        index=["city", "date_target"], columns="source",
        values="forecast_f", aggfunc="first",
    )
    wide.columns = [f"f_{c}" for c in wide.columns]
    wide = wide.reset_index()

    label = cli[["city", "date_target", "cli_f"]].rename(columns={"cli_f": "actual_f"})
    wide = wide.merge(label, on=["city", "date_target"], how="left")

    lagged = cli[["city", "date_target", "normal_f"]].copy()
    lagged["date_target"] = [d + timedelta(days=LABEL_LAG_DAYS) for d in lagged["date_target"]]
    wide = wide.merge(lagged, on=["city", "date_target"], how="left")
    return wide


def add_derived(df: pd.DataFrame, min_core: int) -> pd.DataFrame:
    core_cols = [f"f_{s}" for s in CORE_MODELS if f"f_{s}" in df.columns]
    ens_cols = [f"f_{s}" for s in ENSEMBLE if f"f_{s}" in df.columns]
    obs_cols = [f"f_{s}" for s in OBS if f"f_{s}" in df.columns]

    core = df[core_cols]
    df["n_core"] = core.notna().sum(axis=1)
    df["consensus_median"] = core.median(axis=1, skipna=True)
    df["consensus_mean"] = core.mean(axis=1, skipna=True)
    df["model_spread"] = core.std(axis=1, skipna=True)
    df["model_range"] = core.max(axis=1) - core.min(axis=1)

    if ens_cols:
        ens = df[ens_cols]
        df["n_ens"] = ens.notna().sum(axis=1)
        df["ens_median"] = ens.median(axis=1, skipna=True)
        # the edge signal: intl ensembles vs the broad consensus
        df["ens_vs_consensus"] = df["ens_median"] - df["consensus_median"]

    if obs_cols:
        obs = df[obs_cols]
        df["morning_obs_max"] = obs.max(axis=1, skipna=True)
        df["morning_obs_vs_consensus"] = df["morning_obs_max"] - df["consensus_median"]

    # every source as a deviation from consensus (seasonally stationary)
    for c in [c for c in df.columns if c.startswith("f_")]:
        df["d_" + c[2:]] = df[c] - df["consensus_median"]
    df["consensus_anom"] = df["consensus_median"] - df["normal_f"]
    df[TARGET] = df["actual_f"] - df["consensus_median"]

    # calendar
    dt = pd.to_datetime(df["date_target"])
    df["month"] = dt.dt.month
    df["doy"] = dt.dt.dayofyear
    df["city_code"] = df["city"].map(CITY_CODE).astype("int16")

    df = df[df["n_core"] >= min_core].copy()
    return df


def add_trailing_skill(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Per-source trailing MAE & signed bias vs the CLI label, per city.

    Calendar-day window over [date_target - LABEL_LAG_DAYS - window + 1,
    date_target - LABEL_LAG_DAYS]: only labels that exist at the 09:00 cutoff.
    """
    df = df.sort_values(["city", "date_target"]).reset_index(drop=True)
    err_cols = {}
    for s in SKILL_SOURCES + ["consensus_median"]:
        col = s if s == "consensus_median" else f"f_{s}"
        if col in df.columns:
            err_cols[s.replace("_median", "")] = df[col] - df["actual_f"]

    out = {f"{k}14_{name}": pd.Series(np.nan, index=df.index)
           for name in err_cols for k in ("mae", "bias")}
    for _, idx in df.groupby("city", sort=False).groups.items():
        dates = pd.to_datetime(df.loc[idx, "date_target"])
        for name, err in err_cols.items():
            daily = pd.Series(err.loc[idx].to_numpy(), index=dates).asfreq("D")
            lagged = daily.shift(LABEL_LAG_DAYS)
            mae = lagged.abs().rolling(window, min_periods=3).mean()
            bias = lagged.rolling(window, min_periods=3).mean()
            out[f"mae14_{name}"].loc[idx] = mae.reindex(dates).to_numpy()
            out[f"bias14_{name}"].loc[idx] = bias.reindex(dates).to_numpy()
    for k, v in out.items():
        df[k] = v
    return df


def audit(df: pd.DataFrame, labeled: pd.DataFrame) -> None:
    print("\n" + "=" * 68)
    print("PHASE 2 FEATURE BUILD — AUDIT")
    print("=" * 68)
    print(f"labeled city-days (feature matrix rows): {len(labeled):,}")
    print(f"date_target span: {labeled['date_target'].min()} -> {labeled['date_target'].max()}")
    print(f"cities: {labeled['city'].nunique()}")

    # --- leak canary: at a true morning cutoff the running obs max must sit
    #     BELOW the eventual daily high. A high fraction >= actual means leakage.
    if "morning_obs_max" in labeled.columns:
        m = labeled.dropna(subset=["morning_obs_max", "actual_f"])
        frac_ge = float((m["morning_obs_max"] >= m["actual_f"]).mean())
        gap = float((m["actual_f"] - m["morning_obs_max"]).mean())
        print(f"\nLEAK CANARY (morning_obs_max vs actual):")
        print(f"  frac obs_max >= actual : {frac_ge:6.3f}   (want ~0)")
        print(f"  mean (actual - obs_max): {gap:6.2f} F   (want clearly > 0)")

    # per-source availability
    print("\nper-source coverage (non-null share of rows):")
    for s in ALL_SOURCES:
        c = f"f_{s}"
        if c in labeled.columns:
            print(f"  {s:20s} {labeled[c].notna().mean():5.2f}")

    # naive forecast skill of the consensus (sanity, NOT edge): MAE vs actual
    for name in ("consensus_median", "ens_median"):
        if name in labeled.columns:
            e = (labeled[name] - labeled["actual_f"]).abs()
            print(f"\n{name} MAE vs actual: {e.mean():.2f} F  (n={e.notna().sum()})")

    print("\nmodel feature columns:", len(feature_columns(labeled)))


def main() -> None:
    a = parse_args()
    hh, mm = (int(x) for x in a.cutoff.split(":"))
    hhmm = time(hh, mm)
    kind = "high" if a.is_high else "low"
    out = a.out or f"data/lookahead/features_{kind}.parquet"

    print(f"loading {kind.upper()} rows from {a.db} ...")
    df, cli = load_rows(a.db, a.is_high)
    print(f"  {len(df):,} melted rows, {len(cli):,} CLI labels; applying {a.cutoff} local cutoff ...")

    latest = latest_before_cutoff(df, hhmm)
    wide = pivot_features(latest, cli)
    wide = add_derived(wide, a.min_core)
    wide = add_trailing_skill(wide, a.mae_window)

    labeled = wide.dropna(subset=["actual_f"]).reset_index(drop=True)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    labeled.to_parquet(out, index=False)
    audit(wide, labeled)
    print(f"\nwrote {len(labeled):,} rows -> {out}")


if __name__ == "__main__":
    main()
