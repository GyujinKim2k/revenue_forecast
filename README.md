# Carsharing Revenue-Per-Vehicle (RPV) Forecasting

A production-oriented machine-learning system that forecasts **Revenue Per
Vehicle (RPV)** for a carsharing fleet at the granularity of
`spot × vehicle_type`, at daily and weekly horizons, using **a compact ~0.42 M-parameter Temporal Fusion
Transformer (TFT) (~1.7 MB)** with quantile regression. The forecasts feed downstream
decisions: vehicle-allocation optimization, dynamic pricing / coupon strategy,
and "what-if" scenario analysis.

**Stack:** Python · PyTorch · PyTorch-Forecasting · PyTorch Lightning ·
LightGBM / CatBoost · Optuna · Prophet · statsmodels · pandas

> Built during an ML engagement at a carsharing company. The transaction data is
> proprietary and **not** included; this repository contains the code, the
> modelling methodology, and a summary of results
> (`reports/Review_of_Revenue_Forecasting_Model_Filtered.pdf`).

## Objective

Replace a legacy heuristic ("historical average of similar profiles + human
intuition", with limited features and no concrete error bounds) with a
**human-independent, high-precision** revenue model that:

- predicts RPV **with a margin of error** (quantile prediction intervals), not
  just a point estimate;
- assesses revenue impact via **multi-variable scenario analysis**;
- scales to new spots and vehicle types.

## What this project demonstrates

- **Deep time-series forecasting** — a Temporal Fusion Transformer with quantile
  regression for probabilistic, interpretable multi-horizon forecasts.
- **End-to-end ownership** — from raw transactions and public-API ingestion to a
  feature store, model training, and decision tooling.
- **Feature engineering at scale** — inventory reconstruction from rental
  intervals, calendar/holiday effects, lag & rolling statistics, weather, and
  region aggregates, per `spot × vehicle_type` series.
- **Rigorous model selection** — LightGBM feature ablation on WAPE, Optuna
  hyperparameter search, and time-series cross-validation (no future leakage).
- **From forecast to decision** — attention-based interpretability,
  counterfactual scenario analysis, and a greedy vehicle-allocation optimizer.

## Data

| Source | Provider | Signal |
|---|---|---|
| Rental transactions, vehicle & spot data, standard tariff | Company internal | revenue, inventory, coupons, spec |
| Daily weather (ASOS) | KMA (`data.go.kr`) | temperature, humidity, precipitation, wind, sunshine |
| Land price | MOLIT V-World | published land value by parcel |
| Population distribution | MOIS | demographics by legal dong |

Schemas and the expected `data/` layout are documented in
[`data/README.md`](data/README.md).

## Pipeline

```
raw transactions ─┐
weather (KMA) ────┼─▶ feature store ─▶ model training ─▶ analysis & optimization
land / population ┘   (build_features)   (TFT + baselines)   (forecasts, allocation)
```

1. **Data collection** — `src/revenue_forecast/data/` (KMA weather client,
   V-World land price).
2. **Feature engineering** — `features/build_features.py`: revenue allocation
   across rental days, **inventory estimation** from rental intervals, calendar &
   holiday features, revenue lags/rolling stats, utilization, coupon lags,
   weather join + amenity score, region aggregates. A separate **calendar
   day-factor** model (`features/day_factor.py`) fits Prophet + a log-log OLS to
   isolate weekday × holiday × annual effects.
3. **Feature validation** — `features/feature_selection.py`: LightGBM forward
   ablation tracking WAPE per feature block (+ Optuna tuning and a CatBoost
   reference). Land price and population had negative/low contribution and were
   dropped; weather was retained for contextual value.
4. **Modelling** — `models/`: the main **TFT** (`train_tft.py`) plus MLP
   (`train_mlp.py`), LSTM (`train_lstm.py`) and hierarchical
   (`experimental/hierarchical.py`) baselines.
5. **Analysis & optimization** — `models/analyze_tft.py` (back-testing, quantile
   intervals, counterfactuals, attention importance) and `models/allocation.py`
   (marginal-revenue curves + greedy vehicle rebalancing).

## Main model: Temporal Fusion Transformer

- **Target**: RPV per `spot × vehicle_type`; 60-day encoder, 20-day horizon.
- **Inputs** are split into static covariates, known-future inputs (calendar,
  weather, inventory, tariff) and observed inputs (lags, utilization, coupons,
  region aggregates).
- **Quantile regression** (`QuantileLoss`) predicts a distribution — median plus
  10/25/75/90% bands — so downstream users get best-/worst-case demand
  scenarios, not just a mean.
- **Interpretability**: the attention mechanism yields variable importance,
  answering not only *"how much"* but *"why"*.
- **Evaluation**: trained with quantile loss; back-tested with **WAPE** (weighted
  absolute percentage error) and per-series MAE/RMSE under time-series splits.
- Hyperparameters were selected by Optuna (`config.TFT_HPARAMS`).

## Selected findings

- **Feature importance (attention)**: `Vehicle Inventory > Holidays > Standard
  Tariff > Weather > Temporal context`. Contrary to expectation, **weather is
  marginal** while holidays and vehicle availability dominate revenue.
- **Weather**: same-day precipitation is ambiguous, but **1-day-lagged**
  precipitation shows a clear negative effect during heavy rain; peak RPV occurs
  at high (not "comfort-zone") humidity — consistent with users securing
  vehicles just before/after rain.
- **Vehicle age**: increasing age by a few years shifts RPV within the
  statistical margin of error; residential-heavy areas favor newer vehicles,
  high-footfall zones tolerate older ones.
- **Use cases**: weekly/daily revenue forecasts with intervals; RPV comparison
  by vehicle age and type; revenue-maximizing vehicle reallocation across Seoul.

## Repository layout

```
revenue_forecast/
├── src/revenue_forecast/
│   ├── config.py                 # paths, feature groups, hyperparameters
│   ├── data/                     # weather_api.py, land_price.py
│   ├── features/                 # weather_score, day_factor, build_features, feature_selection
│   └── models/                   # train_tft (main) · train_mlp · train_lstm
│                                 #   analyze_tft · allocation · _patches
│                                 #   experimental/hierarchical
├── scripts/                      # collect_weather, build_feature_store, train, run_eda
├── data/                         # git-ignored; schema in data/README.md
├── models/                       # git-ignored checkpoints
├── reports/                      # summary PDF including generated figures
├── results/                      # sample result images
├── pyproject.toml                # installable package (pip install -e .)
├── requirements.txt
└── .env.example                  # KMA_SERVICE_KEY, VWORLD_API_KEY
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # then fill in API keys
export PYTHONPATH=src           # or `pip install -e .`
```

## Usage

```bash
# 1) Collect weather (needs KMA_SERVICE_KEY)
python scripts/collect_weather.py --start 2023-01-01 --end 2023-12-31 \
    --station 108 --out data/external/weather/weather_108.csv

# 2) Build the feature store from raw sources
python scripts/build_feature_store.py

# 3) Train the Temporal Fusion Transformer
python scripts/train.py --max-epochs 1000

# 4) Regenerate EDA figures
python scripts/run_eda.py
```

Baselines and analysis are library modules, e.g.:

```python
from revenue_forecast.models import analyze_tft, allocation
tft, training = analyze_tft.load_model("models/tft.ckpt",
                                       "data/processed/training_per_dataset.pkl")
```

## Notes

- Model reference: Lim et al., *Temporal Fusion Transformers for interpretable
  multi-horizon time series forecasting*, International Journal of Forecasting (2021).
- No secrets are committed; API keys are read from the environment.
- Licensed under the MIT License — see [`LICENSE`](LICENSE).
