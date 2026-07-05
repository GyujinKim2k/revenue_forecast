"""Train the Temporal Fusion Transformer (main RPV model).

Loads the feature store, filters sparse series, builds the pytorch-forecasting
``TimeSeriesDataSet``, and trains a TFT with a ``QuantileLoss`` so that the model
predicts a full distribution (best-case / worst-case bands) rather than a point
estimate. Hyperparameters come from an Optuna search (see ``config.TFT_HPARAMS``).
"""
from __future__ import annotations

import warnings

import pandas as pd
import torch

from .. import config
from ._patches import apply_patches

warnings.filterwarnings("ignore")
apply_patches()  # must run before importing/constructing pytorch-forecasting objects

import lightning.pytorch as pl  # noqa: E402
from lightning.pytorch.callbacks import (  # noqa: E402
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger  # noqa: E402
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet  # noqa: E402
from pytorch_forecasting.data import GroupNormalizer  # noqa: E402
from pytorch_forecasting.data.encoders import NaNLabelEncoder  # noqa: E402
from pytorch_forecasting.metrics import QuantileLoss  # noqa: E402


def load_and_filter(feature_store=config.FEATURE_STORE) -> pd.DataFrame:
    """Load the feature store, keep positive-revenue rows and dense series."""
    df = pd.read_feather(feature_store)
    df = df[df["date"] <= pd.Timestamp(config.TRAIN_CUTOFF)]
    df = df[df[config.TARGET] > 0]

    # Coverage filter: drop (spot, vehicle_type) groups whose positive-revenue
    # days cover less than MIN_COVERAGE of the full observation window.
    total_days = (df["date"].max() - df["date"].min()).days + 1
    pos_unique_dates = df.groupby(config.GROUP_IDS)["date"].transform("nunique")
    coverage = pos_unique_dates / total_days
    df = df[coverage >= config.MIN_COVERAGE].reset_index(drop=True)

    print(f"Total period (days): {total_days}")
    print("Remaining groups:", df[config.GROUP_IDS].drop_duplicates().shape[0])
    print("Final row count:", df.shape[0])
    return df


def build_datasets(df: pd.DataFrame) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
    """Construct the training and validation ``TimeSeriesDataSet`` objects."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(config.GROUP_IDS + ["date"])
    # Continuous daily index is required by the TFT.
    df["time_idx"] = (df["date"] - df["date"].min()).dt.days

    # Categoricals must be strings for the encoders.
    for col in config.STATIC_CATEGORICALS + config.TIME_VARYING_KNOWN_CATEGORICALS:
        df[col] = df[col].astype(str)

    training_cutoff = df["time_idx"].max() - config.MAX_PREDICTION_LENGTH
    df_train = df[df["time_idx"] <= training_cutoff].copy()

    training = TimeSeriesDataSet(
        df_train,
        time_idx="time_idx",
        target=config.TARGET,
        group_ids=config.GROUP_IDS,
        min_encoder_length=config.MAX_ENCODER_LENGTH,
        max_encoder_length=config.MAX_ENCODER_LENGTH,
        min_prediction_length=config.MAX_PREDICTION_LENGTH,
        max_prediction_length=config.MAX_PREDICTION_LENGTH,
        categorical_encoders={
            "spot_id": NaNLabelEncoder(add_nan=True),
            "vehicle_type": NaNLabelEncoder(add_nan=True),
        },
        static_categoricals=config.STATIC_CATEGORICALS,
        time_varying_known_categoricals=config.TIME_VARYING_KNOWN_CATEGORICALS,
        time_varying_known_reals=config.TIME_VARYING_KNOWN_REALS,
        time_varying_unknown_reals=config.TIME_VARYING_UNKNOWN_REALS,
        target_normalizer=GroupNormalizer(groups=config.GROUP_IDS, method="standard"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
        randomize_length=False,
    )
    validation = TimeSeriesDataSet.from_dataset(
        training, df, predict=True, stop_randomization=True
    )
    print(f"train series: {len(training)}, val series: {len(validation)}")
    return training, validation


def main(feature_store=config.FEATURE_STORE, max_epochs: int = 1000) -> None:
    df = load_and_filter(feature_store)
    training, validation = build_datasets(df)

    batch_size = config.BATCH_SIZE
    train_dataloader = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=20,
        prefetch_factor=4, persistent_workers=True, pin_memory=True,
    )
    val_dataloader = validation.to_dataloader(
        train=False, batch_size=batch_size * 10, num_workers=4,
        prefetch_factor=4, persistent_workers=True, pin_memory=True,
    )

    torch.set_float32_matmul_precision("medium")

    tft = TemporalFusionTransformer.from_dataset(
        training,
        loss=QuantileLoss(),
        log_interval=10,
        **config.TFT_HPARAMS,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(config.MODELS_DIR),
        filename="tft-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3, monitor="val_loss", mode="min", save_last=True,
    )
    early_stop = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=100, verbose=True, mode="min",
    )
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        enable_model_summary=True,
        enable_progress_bar=True,
        gradient_clip_val=config.GRADIENT_CLIP_VAL,
        limit_train_batches=1.0,
        callbacks=[LearningRateMonitor(), early_stop, checkpoint_callback],
        logger=TensorBoardLogger("lightning_logs"),
    )
    trainer.fit(tft, train_dataloader, val_dataloaders=val_dataloader)


if __name__ == "__main__":
    main()
