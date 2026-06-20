"""
Build v2 training data by enriching the existing Kalshi CSV with three new
alpha features identified in scripts/analyze_feature_alpha.py:

  hrrr_skill_adj      = hrrr_vs_ceil / (recent_hrrr_mae_7d + 0.5)
      HRRR confidence scaled by recent forecast error.  A +3°F signal from a
      model with 1°F recent MAE is much stronger than the same signal when
      recent MAE is 4°F.

  min_model_vs_ceil   = min(hrrr_vs_ceil, gfs_vs_ceil)
      Conservative consensus floor: the *least* bullish of HRRR and GFS.
      If even the pessimistic model says +2°F above band, that's a stronger
      all-models-agree signal than one model at +5°F and the other at -1°F.

  margin_per_hour_left = margin_f / (hours_to_close + 1)
      Safety buffer per unit of remaining risk time.  A 3°F margin with 2h
      left is far safer than 3°F with 10h left.

All three showed partial_r > 0.09 and delta_AUC > 0.03 against margin_f
as the control in the feature alpha analysis.

Input:  data/backtest/forecast_no_training_data_kalshi.csv
Output: data/backtest/forecast_no_training_data_kalshi_v2.csv
"""

import csv
from pathlib import Path

SRC = Path("data/backtest/forecast_no_training_data_kalshi.csv")
DST = Path("data/backtest/forecast_no_training_data_kalshi_v2.csv")

NEW_FIELDS = [
    "hrrr_skill_adj", "min_model_vs_ceil", "margin_per_hour_left",
    "hour_utc_x_is_high",
    # tier-1: pure derivations from existing columns
    "day_of_year",          # 1-366, more precise than month
    "is_weekend",           # 0/1
    "hrrr_gfs_signed_diff", # hrrr_vs_ceil - gfs_vs_ceil (signed disagreement)
    "all_models_agree_no",  # 1 if hrrr, gfs, consensus all > 0
    "delta_accel",          # delta_1h - delta_2h (acceleration of cooling)
]


def derive(row: dict) -> dict:
    def f(k, default=0.0):
        v = row.get(k)
        try:
            return float(v) if v not in (None, "", "nan") else default
        except (ValueError, TypeError):
            return default

    hrrr_vc    = f("hrrr_vs_ceil")
    gfs_vc     = f("gfs_vs_ceil")
    cons_vc    = f("consensus_vs_ceil")
    mae_7d     = f("recent_hrrr_mae_7d", default=3.0)
    margin     = f("margin_f")
    hrs_left   = f("hours_to_close")
    hour_utc   = f("hour_utc")
    is_high    = f("is_high")
    delta_1h   = f("delta_1h")
    delta_2h   = f("delta_2h")
    date_str   = row.get("date", "")

    hrrr_skill_adj       = round(hrrr_vc / (mae_7d + 0.5), 4)
    min_model_vs_ceil    = round(min(hrrr_vc, gfs_vc), 4)
    margin_per_hour_left = round(margin / (hrs_left + 1), 4)
    hour_utc_x_is_high   = round(hour_utc * is_high, 4)

    # tier-1 features
    try:
        from datetime import date as _date
        d = _date.fromisoformat(date_str)
        day_of_year = d.timetuple().tm_yday
        is_weekend  = 1 if d.weekday() >= 5 else 0
    except (ValueError, AttributeError):
        day_of_year = 0
        is_weekend  = 0

    hrrr_gfs_signed_diff = round(hrrr_vc - gfs_vc, 4)
    all_models_agree_no  = 1 if (hrrr_vc > 0 and gfs_vc > 0 and cons_vc > 0) else 0
    delta_accel          = round(delta_1h - delta_2h, 4)

    return {
        "hrrr_skill_adj":       hrrr_skill_adj,
        "min_model_vs_ceil":    min_model_vs_ceil,
        "margin_per_hour_left": margin_per_hour_left,
        "hour_utc_x_is_high":  hour_utc_x_is_high,
        "day_of_year":          day_of_year,
        "is_weekend":           is_weekend,
        "hrrr_gfs_signed_diff": hrrr_gfs_signed_diff,
        "all_models_agree_no":  all_models_agree_no,
        "delta_accel":          delta_accel,
    }


def main():
    rows = list(csv.DictReader(SRC.open()))
    print(f"Loaded {len(rows):,} rows from {SRC}")

    # Existing fields + new ones
    fieldnames = list(rows[0].keys()) + NEW_FIELDS

    with DST.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row.update(derive(row))
            writer.writerow(row)

    print(f"Saved {len(rows):,} rows → {DST}")

    # Sanity check a few derived values
    sample = rows[:3]
    for r in sample:
        print(f"  hrrr_skill_adj={r['hrrr_skill_adj']}  "
              f"min_model_vs_ceil={r['min_model_vs_ceil']}  "
              f"margin_per_hour_left={r['margin_per_hour_left']}")


if __name__ == "__main__":
    main()
