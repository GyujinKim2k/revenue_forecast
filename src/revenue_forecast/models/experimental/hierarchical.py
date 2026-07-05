"""Experimental: hierarchical / reconciled forecasting.

Explores forecasting RPV over the natural hierarchy
``region_code -> spot_id -> vehicle_type`` and reconciling the levels so that
child forecasts sum consistently to their parents. Two approaches:

* ``reconciled_statistical`` – classic base models (AutoARIMA/Naive) reconciled
  with BottomUp / TopDown / MiddleOut (``hierarchicalforecast``).
* ``hint_tft`` – a neural TFT wrapped in HINT with MinTrace-OLS reconciliation
  (``neuralforecast``).

Kept as a research branch — the production model is the flat per-series TFT in
``models.train_tft``. Converted from ``hierarchical.ipynb`` and
``TFT_hierarchical.ipynb``.
"""
from __future__ import annotations

import pandas as pd

from ...config import FEATURE_STORE, MAX_ENCODER_LENGTH, MAX_PREDICTION_LENGTH, TARGET

HIERARCHY_SPEC = [
    ["region_code"],                              # top level
    ["region_code", "spot_id"],                   # spot within region
    ["region_code", "spot_id", "vehicle_type"],   # bottom (full granularity)
]


def to_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """Rename to (unique_id, ds, y) long format keyed by the hierarchy path."""
    df = df.rename(columns={"date": "ds", TARGET: "y"}).copy()
    df["unique_id"] = (df["region_code"].astype(str) + "/"
                       + df["spot_id"].astype(str) + "/"
                       + df["vehicle_type"].astype(str))
    return df.dropna(subset=["region_code", "spot_id", "vehicle_type"])


def reconciled_statistical(df: pd.DataFrame, horizon: int = MAX_PREDICTION_LENGTH):
    """AutoARIMA/Naive base forecasts reconciled across the hierarchy."""
    from hierarchicalforecast.core import HierarchicalReconciliation
    from hierarchicalforecast.methods import BottomUp, MiddleOut, TopDown
    from hierarchicalforecast.utils import aggregate
    from statsforecast.core import StatsForecast
    from statsforecast.models import AutoARIMA, Naive

    long = to_long_format(df).drop_duplicates(subset=["unique_id", "ds"])
    y_hier, S, tags = aggregate(df=long, spec=HIERARCHY_SPEC)

    sf = StatsForecast(models=[AutoARIMA(season_length=7), Naive()], freq="D")
    base = sf.forecast(df=y_hier, h=horizon)

    reconcilers = [BottomUp(), TopDown(method="proportion_averages"),
                   MiddleOut(middle_level="region_code/spot_id",
                             top_down_method="proportion_averages")]
    hrec = HierarchicalReconciliation(reconcilers=reconcilers)
    return hrec.reconcile(Y_hat_df=base, S=S, tags=tags)


def hint_tft(df: pd.DataFrame, horizon: int = MAX_PREDICTION_LENGTH,
             input_size: int = MAX_ENCODER_LENGTH):
    """Neural TFT wrapped in HINT with MinTrace-OLS reconciliation."""
    from hierarchicalforecast.utils import aggregate
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import GMM
    from neuralforecast.models import HINT, TFT

    long = to_long_format(df).drop_duplicates(subset=["unique_id", "ds"])
    _, S, _ = aggregate(df=long, spec=HIERARCHY_SPEC)

    quantiles = [0.025, 0.5, 0.975]
    tft = TFT(h=horizon, input_size=input_size, hidden_size=64, dropout=0.1,
              learning_rate=1e-3, n_blocks=1, max_steps=30, random_state=42,
              loss=GMM(n_components=10, quantiles=quantiles))
    model = HINT(h=horizon, S=S, model=tft, reconciliation="MinTraceOLS")
    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(long)
    return nf.predict()


if __name__ == "__main__":
    data = pd.read_feather(FEATURE_STORE)
    print(reconciled_statistical(data).head())
