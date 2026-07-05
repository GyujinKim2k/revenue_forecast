"""MLP baseline for RPV regression.

A residual multi-layer perceptron trained on the tabular daily/weekly feature
store with a log-transformed target, standard scaling, and K-fold cross
validation. Serves as a non-sequential baseline against the Temporal Fusion
Transformer. A LightGBM baseline (``lightgbm_baseline``) is included for a quick
gradient-boosting reference and feature-importance readout.

Distilled from ``mlp_training.ipynb`` and ``mlp_training_week.ipynb``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ComplexMLP(nn.Module):
    """4-block residual MLP (512 -> 256 -> 128 -> 64 -> 1)."""

    def __init__(self, in_dim: int, dropout: float = 0.3):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(in_dim, 512), nn.BatchNorm1d(512), nn.LeakyReLU(), nn.Dropout(dropout))
        self.block2 = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.LeakyReLU(), nn.Dropout(dropout))
        self.block3 = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.LeakyReLU(), nn.Dropout(dropout))
        self.block4 = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.LeakyReLU(), nn.Dropout(dropout))
        self.head = nn.Linear(64, 1)
        self.residual_proj = nn.Linear(512, 128)

    def forward(self, x):
        x = self.block1(x)
        x2 = self.block2(x)
        res = self.residual_proj(x)          # skip connection 512 -> 128
        x3 = self.block3(x2) + res
        x4 = self.block4(x3)
        return self.head(x4)


def build_xy(df: pd.DataFrame, features: list[str], target: str = "rev_per_vehicle"):
    """Scale features and log-target; return (X, y, scaler_y)."""
    x_raw = df[features].fillna(0).values.astype(np.float32)
    y_raw = np.log1p(df[target].values).reshape(-1, 1).astype(np.float32)
    x = StandardScaler().fit_transform(x_raw)
    scaler_y = StandardScaler()
    y = scaler_y.fit_transform(y_raw).flatten()
    return x, y, scaler_y


def kfold_train(
    x: np.ndarray, y: np.ndarray, scaler_y: StandardScaler,
    n_splits: int = 5, epochs: int = 1000, batch_size: int = 1024,
    lr: float = 1e-3, patience: int = 50,
) -> dict:
    """K-fold train/evaluate the MLP; return mean RMSE and R^2 (original scale)."""
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y).unsqueeze(1))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmse_list, r2_list = [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset), 1):
        train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=42)
        loaders = {
            name: DataLoader(Subset(dataset, idx), batch_size=batch_size,
                             shuffle=(name == "train"))
            for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]
        }
        model = ComplexMLP(x.shape[1]).to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5, min_lr=1e-6)
        loss_fn = nn.MSELoss()

        best_val, best_state, waited = np.inf, None, 0
        for _ in range(epochs):
            model.train()
            for xb, yb in loaders["train"]:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                val_loss = np.mean([
                    loss_fn(model(xb.to(DEVICE)), yb.to(DEVICE)).item()
                    for xb, yb in loaders["val"]
                ])
            sched.step(val_loss)
            if val_loss < best_val:
                best_val, best_state, waited = val_loss, model.state_dict(), 0
            else:
                waited += 1
                if waited >= patience:
                    break

        model.load_state_dict(best_state)
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in loaders["test"]:
                preds.append(model(xb.to(DEVICE)).cpu().numpy())
                trues.append(yb.numpy())
        # invert scaling + log to recover original RPV scale
        y_pred = np.expm1(scaler_y.inverse_transform(np.vstack(preds).astype(np.float32)))
        y_true = np.expm1(scaler_y.inverse_transform(np.vstack(trues).astype(np.float32)))
        y_pred = np.clip(y_pred, 0, None)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
        print(f"Fold {fold} - RMSE: {rmse:.2f}, R2: {r2:.3f}")
        rmse_list.append(rmse)
        r2_list.append(r2)

    result = {"rmse": float(np.mean(rmse_list)), "r2": float(np.mean(r2_list))}
    print(f"Average RMSE: {result['rmse']:.2f}, R2: {result['r2']:.3f}")
    return result


def lightgbm_baseline(df: pd.DataFrame, features: list[str], target: str = "rev_per_vehicle"):
    """Quick LightGBM reference; prints RMSE and returns the booster."""
    import lightgbm as lgb

    x_train, x_val, y_train, y_val = train_test_split(
        df[features], df[target], test_size=0.2, random_state=42)
    dtrain = lgb.Dataset(x_train, label=y_train)
    dval = lgb.Dataset(x_val, label=y_val, reference=dtrain)
    params = {"objective": "regression", "metric": "rmse", "learning_rate": 0.05,
              "num_leaves": 64, "min_data_in_leaf": 20, "feature_fraction": 0.8,
              "bagging_fraction": 0.8, "bagging_freq": 5, "seed": 42, "verbosity": -1}
    booster = lgb.train(
        params, dtrain, num_boost_round=10000, valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)],
    )
    rmse = float(np.sqrt(mean_squared_error(
        y_val, booster.predict(x_val, num_iteration=booster.best_iteration))))
    print(f"LightGBM RMSE: {rmse:.3f}")
    return booster


def main(feature_store: str, features: list[str] | None = None) -> None:
    df = pd.read_feather(feature_store)
    features = features or [c for c in df.columns
                            if c not in ("date", "spot_id", "vehicle_type", "rev_per_vehicle")]
    x, y, scaler_y = build_xy(df, features)
    kfold_train(x, y, scaler_y)


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "data/processed/weekly_seoul.feather")
