"""Calendar "day factor": expected detrended RPV from calendar effects alone.

Two stages:

1. **Training** (``train_day_factor``): fit a multiplicative Prophet model
   (yearly + weekly + holiday seasonality) to detrended daily RPV, then fit a
   log-log OLS on the Prophet-derived weekday and holiday multipliers to obtain
   an interpretable interaction model::

       ln(RPV) = b0 + bW*ln(W) + bH*ln(H) + bWH*ln(W)*ln(H)

   The elasticity of RPV w.r.t. the weekday factor is ``bW + bWH*ln(H)``: on
   long holidays the day-of-week sensitivity is largely cancelled out.

2. **Inference** (``get_day_factor``): given a date, combine the fitted OLS
   coefficients, the Prophet weekly multiplier, a holiday-run multiplier and the
   Prophet yearly component into a single expected detrended RPV.

The fitted OLS coefficients and Prophet weekly multipliers below were produced
by ``train_day_factor`` on the historical revenue series; re-run training to
refresh them when new data arrives.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import EXTERNAL_DIR, PROCESSED_DIR

# --------------------------------------------------------------------------- #
# Fitted parameters (from train_day_factor on the historical series)
# --------------------------------------------------------------------------- #
# Log-log OLS coefficients (R^2 ~= 0.78, all p < 0.001).
OLS_INTERCEPT = 10.788757745111797
OLS_BETA_WEEKDAY = 1.0211359015678163
OLS_BETA_HOLIDAY = 0.8709493703867706
OLS_BETA_INTERACTION = -2.5627887775938483

# Prophet multiplicative weekly component (weekday 0=Mon .. 6=Sun), as multipliers.
WEEKLY_MULTIPLIER = {
    0: 0.930774, 1: 0.879771, 2: 0.868360, 3: 0.876122,
    4: 0.961106, 5: 1.250882, 6: 1.232985,
}

PROPHET_MODEL_PATH = PROCESSED_DIR / "prophet_model.json"
HOLIDAY_DATA_PATH = EXTERNAL_DIR / "holidays_2023_2025.feather"


def holiday_mult(run_length: int) -> float:
    """Multiplier for the length of a consecutive off-day (holiday) block."""
    if run_length >= 3:
        return 1.50
    if run_length >= 1:
        return 1.30
    return 1.00


def _load_prophet_model(path: Path = PROPHET_MODEL_PATH):
    from prophet.serialize import model_from_json

    with open(path, "r") as fin:
        return model_from_json(fin.read())


def get_annual_multiplier(date, model=None) -> float:
    """Prophet multiplicative yearly effect for a single date (as a multiplier)."""
    model = model or _load_prophet_model()
    seas = model.predict_seasonal_components(pd.DataFrame({"ds": [pd.to_datetime(date)]}))
    # In multiplicative mode Prophet returns (factor - 1).
    return float(seas["yearly"].iloc[0] + 1)


def _offday_run_for(date: pd.Timestamp, holiday_path: Path = HOLIDAY_DATA_PATH) -> int:
    """Length of the consecutive holiday block that ``date`` belongs to (else 0)."""
    df_hol = pd.read_feather(holiday_path)
    df_hol = df_hol.rename(columns={"일자": "date", "휴일명": "holiday_name"})
    df_hol["date"] = pd.to_datetime(df_hol["date"]).dt.normalize()

    cal = pd.DataFrame({"date": pd.date_range(df_hol["date"].min(), df_hol["date"].max())})
    cal["is_offday"] = cal["date"].isin(df_hol["date"])
    cal["block_id"] = (cal["is_offday"] != cal["is_offday"].shift(1)).cumsum()
    cal["offday_run"] = cal.groupby("block_id")["is_offday"].transform("sum")
    cal["offday_run"] = cal["offday_run"].where(cal["is_offday"], 0)

    match = cal.loc[cal["date"] == date, "offday_run"]
    return int(match.iloc[0]) if len(match) else 0


def day_factor_from_multipliers(weekday_factor: float, holiday_factor: float) -> float:
    """Base (pre-annual) detrended RPV from the log-log interaction model."""
    ln_w = np.log(weekday_factor)
    ln_h = np.log(holiday_factor)
    ln_y = (
        OLS_INTERCEPT
        + OLS_BETA_WEEKDAY * ln_w
        + OLS_BETA_HOLIDAY * ln_h
        + OLS_BETA_INTERACTION * ln_w * ln_h
    )
    return float(np.exp(ln_y))


def get_day_factor(input_date, model=None) -> float:
    """Predicted detrended RPV for a date: weekday x holiday x annual effects."""
    date = pd.to_datetime(input_date).normalize()
    run = _offday_run_for(date)
    base = day_factor_from_multipliers(
        WEEKLY_MULTIPLIER[date.weekday()], holiday_mult(run)
    )
    return base * get_annual_multiplier(date, model=model)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_day_factor(daily_summary: pd.DataFrame, save_prophet: bool = True) -> dict:
    """Fit Prophet + log-log OLS on a detrended daily-RPV series.

    Args:
        daily_summary: Must contain ``date``, ``detrended_rev_per_vehicle``,
            ``is_holiday`` (bool) and ``offday_run`` columns.
        save_prophet: Persist the fitted Prophet model to ``PROPHET_MODEL_PATH``.

    Returns:
        Dict with the OLS coefficients and the Prophet weekly multipliers.
    """
    import statsmodels.formula.api as smf
    from prophet import Prophet
    from prophet.serialize import model_to_json

    df = daily_summary.copy()
    df["date"] = pd.to_datetime(df["date"])

    # 1) Prophet with a +/-2 day window around each holiday.
    df_prophet = df.rename(columns={"date": "ds", "detrended_rev_per_vehicle": "y"})
    holidays = pd.DataFrame({
        "holiday": "custom_holiday",
        "ds": df_prophet.loc[df_prophet["is_holiday"], "ds"],
        "lower_window": -2,
        "upper_window": 2,
    })
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        holidays=holidays,
        seasonality_mode="multiplicative",
    )
    model.fit(df_prophet[["ds", "y"]])
    if save_prophet:
        PROPHET_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PROPHET_MODEL_PATH, "w") as f:
            f.write(model_to_json(model))

    # 2) Extract the weekly multipliers from the fitted model.
    week = pd.date_range(start="2025-01-05", periods=7, freq="D")
    df_week = pd.DataFrame({"ds": week})
    df_week["weekday"] = df_week["ds"].dt.weekday
    df_week["weekly_multiplier"] = model.predict_seasonal_components(df_week)["weekly"] + 1
    weekly = df_week.set_index("weekday")["weekly_multiplier"].to_dict()

    # 3) Log-log OLS with the weekday x holiday interaction.
    df["weekday_factor"] = df["date"].dt.weekday.map(weekly)
    df["holiday_factor"] = df["offday_run"].apply(holiday_mult)
    df["ln_rev"] = np.log(df["detrended_rev_per_vehicle"])
    df["ln_weekday"] = np.log(df["weekday_factor"])
    df["ln_holiday"] = np.log(df["holiday_factor"])
    ols = smf.ols("ln_rev ~ ln_weekday * ln_holiday", data=df).fit()

    return {
        "intercept": ols.params["Intercept"],
        "beta_weekday": ols.params["ln_weekday"],
        "beta_holiday": ols.params["ln_holiday"],
        "beta_interaction": ols.params["ln_weekday:ln_holiday"],
        "weekly_multiplier": weekly,
        "r_squared": ols.rsquared,
    }


if __name__ == "__main__":
    for d in ["2023-01-06", "2023-05-05", "2023-12-25"]:
        print(d, get_day_factor(d))
