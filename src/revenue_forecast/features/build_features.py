"""Feature-store construction: raw transactions -> modelling matrix.

Pipeline (per ``spot_id`` x ``vehicle_type`` x ``date``):

1. **Revenue allocation** – spread each rental's total billing evenly across its
   rental days, then aggregate to daily revenue, vehicle count and coupon count.
2. **Inventory estimation** – reconstruct per-spot vehicle inventory from rental
   intervals (padded +/-15 days and merged) as a cumulative +1/-1 event series.
3. **Calendar** – day-of-week / week / month / quarter, weekend & holiday flags,
   and the length of the current consecutive off-day block (``offday_run``).
4. **Lags & rolling** – 1/7/28-day revenue lags, 7-day rolling mean/std,
   7-day utilization, coupon lags.
5. **External joins** – weather (by region), population demographics, land price.
6. **Region aggregates** – region-level revenue / vehicle count / RPV and a
   region x vehicle-type mean-revenue interaction.
7. **Target transforms** – ``rev_per_vehicle`` (RPV) and its ``log1p``.

Raw column names in the source CSVs are Korean; the rename maps below document
that mapping. Converted from ``eda_new.ipynb``.

Note: this module documents the transformation logic. Concrete input paths /
schemas (company-internal revenue CSVs, spot metadata, population, land price)
are described in ``data/README.md``; the data itself is not distributed.
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd

from ..config import EXTERNAL_DIR, PROCESSED_DIR, RAW_DIR
from .weather_score import calculate_weather_score

# Source (Korean) -> analysis (English) column names.
REVENUE_RENAME = {
    "운행시작일": "rental_start",   # rental start date
    "운행종료일": "rental_end",     # rental end date
    "총청구요금": "total_revenue",  # total billed amount
    "차량번호": "vehicle_id",       # license plate
    "스팟ID": "spot_id",
    "차량유형": "vehicle_type",
}
SPOT_RENAME = {"스팟ID": "spot_id", "지역(시/도)": "region", "지역(구/군)": "district"}

# KMA weather-station code per high-level region.
REGION_TO_CODE = {
    "서울특별시": 108, "부산광역시": 159, "대구광역시": 143, "인천광역시": 112,
    "광주광역시": 156, "대전광역시": 133, "울산광역시": 152, "세종특별자치시": 239,
    "경기도": 119, "강원특별자치도": 101, "제주특별자치도": 184,
    "전라도": 146, "충청도": 131, "경상도": 143,
}

# Vehicle-type labels: Korean -> English.
VEHICLE_TYPE_MAP = {
    "준대형": "Upper Midsize", "준중형": "Compact", "중형SUV": "Midsize SUV",
    "중형": "Midsize", "소형SUV": "Subcompact SUV", "경형": "Microcar",
    "소형": "Subcompact", "전기": "Electric", "수입차": "Imported", "수입": "Imported",
    "준중형SUV": "Compact SUV", "대형SUV": "Fullsize SUV", "소형SUV_HEV": "Subcompact SUV HEV",
    "대형": "Fullsize", "승합": "Passenger Van", "경차": "Microcar", "수입차_1": "Imported",
    "중형_HEV": "Midsize HEV", "준중형_HEV": "Compact HEV", "준대형_HEV": "Upper Midsize HEV",
    "경형SUV": "Micro SUV", "준중형SUV_HEV": "Compact SUV HEV", "중형SUV_HEV": "Midsize SUV HEV",
    "SUV, RV": "SUV/RV", "소형트럭": "Small Truck",
}


def region_to_code(region: str) -> int | None:
    """Map a high-level region name to its KMA weather-station code."""
    return REGION_TO_CODE.get(region)


# --------------------------------------------------------------------------- #
# 1) Revenue -> daily summary
# --------------------------------------------------------------------------- #
def load_revenue(revenue_glob: str, bad_spot_ids: set[int] | None = None):
    """Load & concat revenue CSVs, allocate billing across rental days.

    Returns ``(all_data, start_date, end_date)`` where ``all_data`` has one row
    per rental with a per-day ``daily_rev``.
    """
    bad_spot_ids = bad_spot_ids or set()
    file_paths = glob.glob(revenue_glob)
    frames = []
    for fp in file_paths:
        df = pd.read_csv(fp, parse_dates=["운행시작일", "운행종료일"],
                         dtype={"총청구요금": str})
        df = df[df["BIZ구분"] == False]                     # noqa: E712 - personal (non-business) rentals
        df = df[~df["스팟ID"].isin(bad_spot_ids)]           # drop sparse/insufficient spots
        df = df[df["총청구요금"] != "0"]
        df = df[df["면책보험료"] != "0"]                    # non-zero insurance fee
        df["coupon_flag"] = df["쿠폰명(관리자)"].fillna("").astype(str).ne("")
        df = df.rename(columns=REVENUE_RENAME)
        df["total_revenue"] = df["total_revenue"].str.replace(",", "").astype(float)
        frames.append(df)

    all_data = pd.concat(frames, ignore_index=True).dropna(subset=["rental_start", "rental_end"])
    months = sorted(re.search(r"revenue_(\d{6})\.csv", fp).group(1) for fp in file_paths)
    start_date = pd.to_datetime(f"{months[0]}01") + pd.Timedelta(days=5)
    end_date = pd.to_datetime(f"{months[-1]}01") + MonthEnd(0)

    all_data["rental_days"] = (all_data["rental_end"] - all_data["rental_start"]).dt.days + 1
    all_data["daily_rev"] = all_data["total_revenue"] / all_data["rental_days"]
    return all_data, start_date, end_date


def daily_summary(all_data: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    """Explode rentals to daily rows and aggregate per (date, spot, vehicle_type)."""
    daily = (
        all_data.assign(date=lambda d: d.apply(
            lambda r: pd.date_range(r["rental_start"], r["rental_end"]), axis=1))
        .explode("date")
    )
    daily = daily[(daily["date"] >= start_date) & (daily["date"] <= end_date)]
    summary = (
        daily.groupby(["date", "spot_id", "vehicle_type"], as_index=False)
        .agg(total_rev=("daily_rev", "sum"),
             vehicle_count=("vehicle_id", "nunique"),
             coupon_count=("coupon_flag", "sum"))
        .sort_values(["date", "spot_id", "vehicle_type"])
    )
    return summary


# --------------------------------------------------------------------------- #
# 2) Inventory estimation from rental intervals
# --------------------------------------------------------------------------- #
def estimate_inventory_events(all_data: pd.DataFrame, start_date, end_date,
                              pad_days: int = 15) -> pd.DataFrame:
    """Per (spot, vehicle_type) daily inventory-change events (+1 / -1).

    Each vehicle's rentals are padded by ``pad_days`` and overlapping intervals
    merged, so a vehicle counts as "on the lot" between its (padded) rentals.
    """
    rentals = (all_data[["vehicle_id", "spot_id", "vehicle_type", "rental_start", "rental_end"]]
               .drop_duplicates().copy())
    rentals["start"] = rentals["rental_start"] - pd.Timedelta(days=pad_days)
    rentals["end"] = rentals["rental_end"] + pd.Timedelta(days=pad_days)

    merged = []
    for (_, sid, vtype), grp in rentals.groupby(["vehicle_id", "spot_id", "vehicle_type"]):
        intervals = sorted(zip(grp["start"], grp["end"]))
        cs, ce = intervals[0]
        for s, e in intervals[1:]:
            if s <= ce:
                ce = max(ce, e)
            else:
                merged.append((sid, vtype, cs, ce))
                cs, ce = s, e
        merged.append((sid, vtype, cs, ce))

    df_merged = pd.DataFrame(merged, columns=["spot_id", "vehicle_type", "start", "end"])
    df_merged["start"] = df_merged["start"].clip(lower=start_date)
    df_merged["end"] = df_merged["end"].clip(upper=end_date)

    ev_start = (df_merged[["spot_id", "vehicle_type", "start"]]
                .rename(columns={"start": "date"}).assign(change=1))
    ev_end = (df_merged[["spot_id", "vehicle_type", "end"]]
              .rename(columns={"end": "date"}).assign(change=-1))
    ev_end["date"] += pd.Timedelta(days=1)
    return (pd.concat([ev_start, ev_end], ignore_index=True)
            .groupby(["date", "spot_id", "vehicle_type"], as_index=False)["change"].sum())


# --------------------------------------------------------------------------- #
# 3) Calendar
# --------------------------------------------------------------------------- #
def build_calendar(start_date, end_date, holiday_path) -> pd.DataFrame:
    """Daily calendar with weekend/holiday flags and consecutive off-day runs."""
    df_hol = pd.read_feather(holiday_path).rename(columns={"일자": "date", "휴일명": "holiday_name"})
    df_hol["date"] = pd.to_datetime(df_hol["date"])

    cal = pd.DataFrame({"date": pd.date_range(start_date, end_date, freq="D")})
    cal["dow"] = cal["date"].dt.weekday
    cal["week"] = cal["date"].dt.isocalendar().week.astype(int)
    cal["month"] = cal["date"].dt.month
    cal["quarter"] = cal["date"].dt.quarter
    cal["is_holiday"] = cal["date"].isin(df_hol["date"])
    cal["is_weekend"] = cal["dow"].isin([5, 6])
    cal["is_offday"] = cal["is_weekend"] | cal["is_holiday"]
    cal["block_id"] = (cal["is_offday"] != cal["is_offday"].shift(1)).cumsum()
    cal["offday_run"] = cal.groupby("block_id")["is_offday"].transform("sum")
    cal["offday_run"] = cal["offday_run"].where(cal["is_offday"], 0)
    return cal.drop(columns=["block_id"])


# --------------------------------------------------------------------------- #
# 4-7) Full feature matrix
# --------------------------------------------------------------------------- #
def add_lag_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """Add revenue lags, rolling stats, utilization and coupon lags per series."""
    g = df.groupby(["spot_id", "vehicle_type"])["total_rev"]
    for lag in (1, 7, 28):
        df[f"lag_{lag}"] = g.shift(lag)
    df["roll_mean_7"] = g.shift(1).rolling(7, min_periods=1).mean()
    df["roll_std_7"] = g.shift(1).rolling(7, min_periods=1).std()

    df["has_inventory"] = (df["inventory_est"] > 0).astype(int)
    df["rent_flag"] = (df["total_rev"] > 0).astype(int)
    rent_count_7 = (df.groupby(["spot_id", "vehicle_type"])["rent_flag"]
                    .shift(1).rolling(7, min_periods=1).sum())
    df["utilization_7"] = rent_count_7 / df["inventory_est"].replace(0, np.nan)

    cg = df.groupby(["spot_id", "vehicle_type"])["coupon_count"]
    df["coupon_flag_prev1"] = cg.shift(1).gt(0).astype(int)
    df["coupon_count_lag7"] = cg.shift(7)
    return df


def add_region_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Region-level revenue / vehicle-count / RPV and region x vehicle interaction."""
    mean_rev = (df.groupby(["region_code", "vehicle_type"])["total_rev_log1p"].mean()
                .reset_index().rename(columns={"total_rev_log1p": "region_vehicle_mean_rev"}))
    df = df.merge(mean_rev, on=["region_code", "vehicle_type"], how="left")
    df["region_total_rev"] = df.groupby(["date", "region_code"])["total_rev"].transform("sum")
    df["region_vehicle_count"] = df.groupby(["date", "region_code"])["vehicle_count"].transform("sum")
    df["region_rev_per_vehicle"] = (
        df["region_total_rev"] / df["region_vehicle_count"].replace(0, np.nan))
    return df


def build_feature_store(
    revenue_glob: str = str(RAW_DIR / "revenue" / "revenue_*.csv"),
    spot_info_path: str = str(RAW_DIR / "spot_info.csv"),
    holiday_path: str = str(EXTERNAL_DIR / "holidays_2023_2025.feather"),
    weather_glob: str = str(EXTERNAL_DIR / "weather" / "weather_{code}.feather"),
    out_path: str = str(PROCESSED_DIR / "daily_features_per_all.feather"),
    bad_spot_ids: set[int] | None = None,
) -> pd.DataFrame:
    """Run the full pipeline and persist the modelling matrix to ``out_path``."""
    all_data, start_date, end_date = load_revenue(revenue_glob, bad_spot_ids)
    summary = daily_summary(all_data, start_date, end_date)
    events = estimate_inventory_events(all_data, start_date, end_date)
    cal = build_calendar(start_date, end_date, holiday_path)

    # Master grid: every (date x spot x vehicle_type) combination.
    spot_types = summary[["spot_id", "vehicle_type"]].drop_duplicates()
    grid = (pd.MultiIndex.from_product(
        [cal["date"], spot_types["spot_id"].unique(), spot_types["vehicle_type"].unique()],
        names=["date", "spot_id", "vehicle_type"]).to_frame(index=False)
        .merge(cal, on="date", how="left"))

    # Cumulative inventory from change events.
    grid = grid.merge(events, on=["date", "spot_id", "vehicle_type"], how="left")
    grid["change"] = grid["change"].fillna(0).astype(int)
    grid["inventory_est"] = (grid.groupby(["spot_id", "vehicle_type"])["change"]
                             .cumsum().clip(lower=0))
    grid = grid.drop(columns="change")

    df = grid.merge(summary, on=["date", "spot_id", "vehicle_type"], how="left")
    for col in ("total_rev", "vehicle_count", "coupon_count"):
        df[col] = df[col].fillna(0)

    df = add_lag_rolling(df)

    # Region code from spot metadata.
    df_spot = pd.read_csv(spot_info_path).rename(columns=SPOT_RENAME).dropna(subset=["spot_id"])
    df_spot["region_code"] = df_spot["region"].map(region_to_code)
    df = df.merge(df_spot[["spot_id", "region_code"]], on="spot_id", how="left")

    # Weather by (region_code, date).
    weather_frames = []
    for code in df["region_code"].dropna().unique():
        w = pd.read_feather(weather_glob.format(code=int(code)),
                            columns=["date", "T", "RH", "P", "W", "S"])
        w["date"] = pd.to_datetime(w["date"])
        w["region_code"] = code
        weather_frames.append(w)
    if weather_frames:
        df = df.merge(pd.concat(weather_frames, ignore_index=True),
                      on=["region_code", "date"], how="left")
        df["weather_score"] = [
            calculate_weather_score(t, rh, p, w, s)
            for t, rh, p, w, s in zip(df["T"], df["RH"], df["P"], df["W"], df["S"])
        ]
        df["rain_mm_lag1"] = df.groupby("region_code")["P"].shift(1)

    # Target transforms.
    df["total_rev_log1p"] = np.log1p(df["total_rev"])
    df["rev_per_vehicle"] = df["total_rev"] / df["inventory_est"].replace(0, np.nan)
    df["rev_per_vehicle_log1p"] = np.log1p(df["rev_per_vehicle"].fillna(0))

    df = add_region_aggregates(df)
    df["vehicle_type"] = df["vehicle_type"].map(VEHICLE_TYPE_MAP).fillna(df["vehicle_type"])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.reset_index(drop=True).to_feather(out_path)
    print(f"Saved feature store -> {out_path}  ({df.shape[0]} rows)")
    return df


if __name__ == "__main__":
    build_feature_store()
