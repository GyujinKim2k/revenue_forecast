"""Feature validation via gradient boosting (LightGBM / CatBoost).

Implements the "Feature Validation" step from the project review: quantify each
feature block's marginal contribution to accuracy (WAPE) so that low- or
negative-value blocks can be dropped. Key finding: weather, land price and
population density had the lowest importance; land price and population were
excluded, weather retained for its contextual value.

All CV uses ``TimeSeriesSplit`` (no future leakage) and WAPE on the
back-transformed (``expm1``) target. Converted from ``eda_new.ipynb``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "total_rev_log1p"

# Feature blocks added cumulatively in the forward ablation.
FEATURE_BLOCKS = {
    "calendar": ["dow", "week", "month", "quarter", "is_weekend", "is_holiday", "offday_run"],
    "lags": ["lag_1", "lag_7", "lag_28", "roll_mean_7", "roll_std_7"],
    "inventory": ["inventory_est", "has_inventory", "utilization_7"],
    "coupon": ["vehicle_count", "coupon_count", "coupon_flag_prev1", "coupon_count_lag7"],
    "weather": ["rain_mm_lag1", "RH", "W"],
    "region_agg": ["region_total_rev", "region_vehicle_count", "region_rev_per_vehicle"],
    "interaction": ["region_vehicle_mean_rev"],
}

# Final feature set retained after validation (land price & population dropped).
FINAL_FEATURES = [
    "dow", "week", "month", "quarter", "is_weekend", "is_holiday", "offday_run",
    "lag_1", "lag_7", "lag_28", "roll_mean_7", "roll_std_7",
    "inventory_est", "has_inventory", "utilization_7",
    "coupon_flag_prev1", "coupon_count_lag7",
    "rain_mm_lag1", "RH", "W",
    "region_total_rev", "region_vehicle_count", "region_rev_per_vehicle",
    "region_vehicle_mean_rev",
]


def _wape(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.abs(actual - pred).sum() / np.maximum(np.abs(actual).sum(), 1e-3))


def forward_ablation(df: pd.DataFrame, n_splits: int = 3) -> pd.DataFrame:
    """Add feature blocks cumulatively and track WAPE at each step."""
    import lightgbm as lgb
    from sklearn.model_selection import TimeSeriesSplit

    tscv = TimeSeriesSplit(n_splits=n_splits)
    params = {"objective": "regression", "metric": "mae",
              "learning_rate": 0.05, "num_leaves": 63, "verbose": -1}
    results, baseline = [], []
    for name, block in FEATURE_BLOCKS.items():
        feats = baseline + block
        wapes = []
        for tr, va in tscv.split(df):
            train, val = df.iloc[tr], df.iloc[va]
            model = lgb.train(
                params, lgb.Dataset(train[feats], label=train[TARGET]),
                num_boost_round=500,
                valid_sets=[lgb.Dataset(val[feats], label=val[TARGET])],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            pred = np.expm1(model.predict(val[feats], num_iteration=model.best_iteration))
            wapes.append(_wape(np.expm1(val[TARGET]), pred))
        results.append({"block": name, "n_features": len(block), "wape": float(np.mean(wapes))})
        baseline += block
    return pd.DataFrame(results)


def tune_lightgbm(df: pd.DataFrame, features=FINAL_FEATURES, n_trials: int = 60) -> dict:
    """Optuna search minimizing time-series-CV WAPE; returns best params."""
    import lightgbm as lgb
    import optuna
    from lightgbm import LGBMRegressor
    from sklearn.model_selection import TimeSeriesSplit

    tscv = TimeSeriesSplit(n_splits=3)

    def objective(trial):
        params = dict(
            num_leaves=trial.suggest_int("num_leaves", 31, 127),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 20, 300),
            feature_fraction=trial.suggest_float("feature_fraction", 0.6, 1.0),
            lambda_l1=trial.suggest_float("lambda_l1", 0, 5),
            lambda_l2=trial.suggest_float("lambda_l2", 0, 5),
            n_estimators=2000, objective="regression", metric="mae", verbosity=-1)
        wapes = []
        for tr, va in tscv.split(df):
            model = LGBMRegressor(**params)
            model.fit(df.iloc[tr][features], df.iloc[tr][TARGET],
                      eval_set=[(df.iloc[va][features], df.iloc[va][TARGET])],
                      eval_metric="mae",
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            pred = np.expm1(model.predict(df.iloc[va][features],
                                          num_iteration=model.best_iteration_))
            wapes.append(_wape(np.expm1(df.iloc[va][TARGET]), pred))
        return np.mean(wapes)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def catboost_holdout(df: pd.DataFrame, features=FINAL_FEATURES,
                     cat_cols=("spot_id", "vehicle_type", "region_code"),
                     test_days: int = 30) -> float:
    """CatBoost hold-out (last ``test_days``) WAPE reference."""
    from catboost import CatBoostRegressor, Pool

    df = df.copy()
    cat_cols = list(cat_cols)
    for c in cat_cols:
        df[c] = df[c].astype(str)
    cutoff = df["date"].max() - pd.Timedelta(days=test_days)
    train, val = df[df["date"] <= cutoff], df[df["date"] > cutoff]

    feats = features + cat_cols
    model = CatBoostRegressor(iterations=5000, learning_rate=0.05, depth=8,
                              loss_function="MAE", eval_metric="MAE",
                              early_stopping_rounds=150, verbose=False)
    model.fit(Pool(train[feats], train[TARGET], cat_features=cat_cols),
              eval_set=Pool(val[feats], val[TARGET], cat_features=cat_cols))
    wape = _wape(np.expm1(val[TARGET]), np.expm1(model.predict(val[feats])))
    print(f"Hold-out WAPE (CatBoost): {wape:.4f}")
    return wape


if __name__ == "__main__":
    from ..config import PROCESSED_DIR
    data = pd.read_feather(PROCESSED_DIR / "daily_features.feather")
    print(forward_ablation(data))
