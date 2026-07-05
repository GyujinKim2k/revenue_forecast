"""Inference & interpretation for the trained Temporal Fusion Transformer.

Loads a checkpoint and the fitted training ``TimeSeriesDataSet`` and provides:

* ``historical_predictions`` – 1-step sliding-window forecasts vs. actuals.
* ``group_errors`` – per (spot, vehicle_type) MAE / RMSE / WAPE.
* ``quantile_predictions`` – median + 10/25/75/90% bands (prediction intervals).
* ``counterfactual`` – "what-if" scenario analysis by perturbing a covariate
  (e.g. vehicle age +3 years) and measuring the RPV uplift.
* ``feature_importance`` – variable importance via the TFT attention mechanism.

Converted from ``analyzeTFT.ipynb``. Load order matters: patches are applied on
import so ``TemporalFusionTransformer`` behaves consistently.
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ._patches import apply_patches

apply_patches()

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet  # noqa: E402


def _to_2d(tensor):
    """Squeeze a trailing singleton dim from a prediction tensor."""
    return tensor.squeeze(-1) if tensor.ndim == 3 else tensor


def load_model(checkpoint_path: str, training_dataset_path: str):
    """Load a trained TFT and the ``TimeSeriesDataSet`` it was fit on."""
    tft = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)
    with open(training_dataset_path, "rb") as f:
        training = pickle.load(f)
    return tft, training


def _sliding_dataset(training, df: pd.DataFrame) -> TimeSeriesDataSet:
    """1-step sliding-window dataset for historical back-testing."""
    return TimeSeriesDataSet.from_dataset(
        training, df, predict=False, stop_randomization=True,
        min_prediction_length=1, max_prediction_length=1, allow_missing_timesteps=True,
    )


def historical_predictions(tft, training, df: pd.DataFrame, min_date) -> pd.DataFrame:
    """1-step-ahead forecasts vs. actuals as a long DataFrame."""
    loader = _sliding_dataset(training, df).to_dataloader(
        train=False, batch_size=256, num_workers=4)
    pred = tft.predict(loader, mode="prediction",
                       return_x=True, return_y=True, return_index=True)
    y_hat = _to_2d(pred.output).cpu().numpy().ravel()
    y_raw = pred.y[0] if isinstance(pred.y, (list, tuple)) else pred.y
    y_true = _to_2d(y_raw).cpu().numpy().ravel()
    out = pred.index.reset_index(drop=True).assign(
        y_hat=y_hat, y_true=y_true, time_idx=lambda d: d["time_idx"] + 1)
    out["date"] = out["time_idx"].apply(lambda x: min_date + pd.Timedelta(days=int(x)))
    return out


def group_errors(df_long: pd.DataFrame) -> pd.DataFrame:
    """Per (spot, vehicle_type) MAE, RMSE and WAPE."""
    def _metrics(g):
        wape = np.abs(g.y_true - g.y_hat).sum() / max(np.abs(g.y_true).sum(), 1e-3)
        return pd.Series({
            "MAE": mean_absolute_error(g.y_true, g.y_hat),
            "RMSE": np.sqrt(mean_squared_error(g.y_true, g.y_hat)),
            "WAPE": wape,
        })
    return (df_long.dropna(subset=["y_true"])
            .groupby(["spot_id", "vehicle_type"]).apply(_metrics).reset_index())


def quantile_predictions(tft, training, df: pd.DataFrame, min_date) -> pd.DataFrame:
    """Median + 10/25/75/90% quantile bands for each series (prediction intervals)."""
    loader = _sliding_dataset(training, df).to_dataloader(
        train=False, batch_size=256, num_workers=4)
    pred = tft.predict(loader, mode="quantiles", return_y=True, return_index=True)
    q = _to_2d(pred.output).cpu().numpy()  # (n_series, 1, n_quantiles)
    cols = {"y_q10": 1, "y_q25": 2, "y_q50": 3, "y_q75": 4, "y_q90": 5}
    assign = {name: q[:, 0, i] for name, i in cols.items()}
    y_raw = pred.y[0] if isinstance(pred.y, (list, tuple)) else pred.y
    out = pred.index.reset_index(drop=True).assign(
        y_true=_to_2d(y_raw).cpu().numpy().ravel(),
        time_idx=lambda d: d["time_idx"] + 1, **assign)
    out["date"] = out["time_idx"].apply(lambda x: min_date + pd.Timedelta(days=int(x)))
    return out


def counterfactual(tft, training, df: pd.DataFrame, column: str,
                   delta: float, min_date) -> pd.DataFrame:
    """RPV uplift from perturbing ``column`` by ``delta`` (e.g. vehicle age +3).

    Returns a long DataFrame with ``pred_orig``, ``pred_cf`` and ``uplift`` over
    the full forecast horizon.
    """
    def _predict(frame):
        ds = TimeSeriesDataSet.from_dataset(training, frame, predict=True,
                                            stop_randomization=True)
        return tft.predict(ds.to_dataloader(train=False, batch_size=512, num_workers=4),
                           mode="prediction", return_index=True)

    df_cf = df.copy()
    df_cf[column] = df_cf[column] + delta
    orig, cf = _predict(df), _predict(df_cf)

    idx = orig.index.reset_index(drop=True)
    pred_len, n_series = orig.output.shape[1], orig.output.shape[0]
    base = idx.loc[idx.index.repeat(pred_len)].reset_index(drop=True)
    base["dec_idx"] = np.tile(np.arange(pred_len), n_series)
    base["pred_orig"] = _to_2d(orig.output).cpu().numpy().ravel()
    base["pred_cf"] = _to_2d(cf.output).cpu().numpy().ravel()
    base["uplift"] = base.pred_cf - base.pred_orig
    base["time_idx"] = base.time_idx + base.dec_idx
    base["date"] = base.time_idx.apply(lambda x: min_date + pd.Timedelta(days=int(x)))
    return base


def feature_importance(tft, training, df: pd.DataFrame):
    """Attention-based variable importance (raw interpretation dict).

    Key finding: vehicle inventory and holidays dominate; weather is marginal
    (predictive priority Inventory > Holidays > Tariff > Weather > Temporal).
    """
    loader = _sliding_dataset(training, df).to_dataloader(
        train=False, batch_size=512, num_workers=4)
    pred = tft.predict(loader, mode="raw", return_x=True)
    interpretation = tft.interpret_output(pred.output, reduction="sum")
    return interpretation
