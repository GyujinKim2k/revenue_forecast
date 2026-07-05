# Data

All data is **git-ignored** — this project was built on proprietary
carsharing-company transaction data plus public Korean open-data APIs. This file
documents the expected layout and schemas so the pipeline can be reproduced with
equivalent data.

```
data/
├── raw/          # company-internal source extracts (not distributed)
│   ├── revenue/revenue_YYYYMM.csv
│   ├── spot_info.csv
│   └── vehicle_info.csv
├── external/     # public / third-party sources
│   ├── holidays_2023_2025.feather
│   └── weather/weather_<station_code>.feather   # or .csv
├── interim/      # intermediate artifacts
└── processed/    # model-ready feature store + fitted calendar model
    ├── daily_features_per_all.feather
    ├── training_per_dataset.pkl   # pickled pytorch-forecasting TimeSeriesDataSet
    └── prophet_model.json
```

## Sources

| Source | Provider | Fields used |
|---|---|---|
| Rental transactions | Company internal | rental start/end, total billing, insurance fee, license plate, spot id, vehicle type, coupon name |
| Spot metadata | Company internal | spot id, region (시/도), district (구/군) |
| Vehicle info | Company internal | license plate → vehicle type / spec |
| Standard tariff | Company internal | standard daily rate by vehicle type |
| Daily weather (ASOS) | KMA via data.go.kr | temperature, humidity, precipitation, wind, sunshine |
| Land price | MOLIT V-World | individual published land price by parcel (PNU) |
| Population | MOIS | population distribution by legal dong / age band |

## Raw revenue CSV (`revenue_YYYYMM.csv`)

Original headers are Korean; see `REVENUE_RENAME` in
`src/revenue_forecast/features/build_features.py`:

| Korean | English | Notes |
|---|---|---|
| 운행시작일 | rental_start | parsed as datetime |
| 운행종료일 | rental_end | parsed as datetime |
| 총청구요금 | total_revenue | comma-stripped → float |
| 차량번호 | vehicle_id | license plate |
| 스팟ID | spot_id | |
| 차량유형 | vehicle_type | mapped to English via `VEHICLE_TYPE_MAP` |
| BIZ구분 | (filter) | keep personal (`False`) rentals |
| 면책보험료 | (filter) | drop zero insurance fee |
| 쿠폰명(관리자) | coupon_flag | non-empty → coupon used |

## Feature store (`daily_features_per_all.feather`)

One row per `date × spot_id × vehicle_type`. Feature groups are defined in
`src/revenue_forecast/config.py`:

- **Target**: `rev_per_vehicle` (RPV) and `rev_per_vehicle_log1p`.
- **Static categoricals**: `spot_id`, `vehicle_type`, `region_code`.
- **Known categoricals**: `dow`, `week`, `month`, `quarter`, `is_weekend`, `is_holiday`.
- **Known reals**: `day_to_offday`, `offday_run`, `rain_mm_lag1`, `RH`, `W`,
  `inventory_est`, `has_inventory`, `avg_vehicle_age_inv`, `avg_standard_rate_inv`.
- **Unknown reals**: revenue lags/rolling (`lag_1/7/28`, `roll_mean_7`, `roll_std_7`),
  `utilization_7`, coupon signals, and region-level aggregates.

## Credentials

API keys are read from environment variables — copy `.env.example` to `.env`:

- `KMA_SERVICE_KEY` — KMA weather API (data.go.kr)
- `VWORLD_API_KEY` — V-World land-price API
