"""Central configuration: paths, feature groups, and model hyperparameters.

Keeping these in one place (instead of scattered across scripts) makes the
pipeline reproducible and the feature contract explicit.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# Repo root = two levels up from this file (src/revenue_forecast/config.py).
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("RF_DATA_DIR", ROOT_DIR / "data"))
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
MODELS_DIR = Path(os.environ.get("RF_MODELS_DIR", ROOT_DIR / "models"))

# Feature store produced by features.build_features and consumed by the models.
FEATURE_STORE = PROCESSED_DIR / "daily_features_per_all.feather"

# --------------------------------------------------------------------------- #
# Target
# --------------------------------------------------------------------------- #
TARGET = "rev_per_vehicle"  # Revenue Per Vehicle (RPV)
GROUP_IDS = ["spot_id", "vehicle_type"]

# --------------------------------------------------------------------------- #
# Feature groups for the Temporal Fusion Transformer
#   (see features/build_features.py for how each column is derived)
# --------------------------------------------------------------------------- #
STATIC_CATEGORICALS = ["spot_id", "vehicle_type", "region_code"]

TIME_VARYING_KNOWN_CATEGORICALS = [
    "dow",         # day of week
    "week",        # ISO week number
    "month",
    "quarter",
    "is_weekend",
    "is_holiday",
]

TIME_VARYING_KNOWN_REALS = [
    "day_to_offday",   # days remaining until the next off-day
    "offday_run",      # length of the current consecutive off-day block
    "rain_mm_lag1",    # 1-day lagged precipitation (mm)
    "RH",              # relative humidity (%)
    "W",               # wind speed (m/s)
    "inventory_est",
    "has_inventory",
    "avg_vehicle_age_inv",
    "avg_standard_rate_inv",
]

TIME_VARYING_UNKNOWN_REALS = [
    # lags & rolling statistics
    "lag_1", "lag_7", "lag_28",
    "roll_mean_7", "roll_std_7",
    # utilization
    "utilization_7",
    # coupon signals
    "vehicle_count", "coupon_count",
    "coupon_flag_prev1", "coupon_count_lag7", "coupon_per_vehicle",
    # region aggregates
    "region_total_rev", "region_vehicle_count", "region_rev_per_vehicle",
    # region x vehicle interaction
    "region_vehicle_mean_rev",
]

# --------------------------------------------------------------------------- #
# Windowing
# --------------------------------------------------------------------------- #
MAX_ENCODER_LENGTH = 60      # look-back window (days)
MAX_PREDICTION_LENGTH = 20   # forecast horizon (days)

# Only keep (spot, vehicle_type) groups whose positive-revenue days cover at
# least this fraction of the full period (filters out sparse series).
MIN_COVERAGE = 0.40
TRAIN_CUTOFF = "2024-12-31"  # inclusive upper bound on training dates

# --------------------------------------------------------------------------- #
# Temporal Fusion Transformer hyperparameters (from Optuna search)
# --------------------------------------------------------------------------- #
TFT_HPARAMS = dict(
    learning_rate=0.0012022644346174128,
    hidden_size=64,
    attention_head_size=6,
    dropout=0.2914981522194919,
    hidden_continuous_size=21,
    optimizer="ranger",
    reduce_on_plateau_patience=4,
)
GRADIENT_CLIP_VAL = 0.1561931652158813
BATCH_SIZE = 512
