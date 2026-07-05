"""LSTM baseline for RPV regression.

Builds sliding windows over the daily feature store and trains a stacked LSTM to
capture temporal autocorrelation (revenue shows strong weekly seasonality — see
the ACF analysis). Serves as a sequential baseline against the Temporal Fusion
Transformer. Uses a log target, standard scaling, and K-fold cross validation.

Distilled from ``lstm_training.ipynb``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 60  # look-back window (days)


class SequenceDataset(Dataset):
    """Sliding windows of length ``seq_len`` predicting the next-step target."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor, seq_len: int):
        self.x, self.y, self.seq_len = x, y, seq_len

    def __len__(self):
        return self.x.shape[0] - self.seq_len

    def __getitem__(self, idx):
        return self.x[idx: idx + self.seq_len], self.y[idx + self.seq_len]


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))

    def forward(self, x):
        out, _ = self.lstm(x)      # (batch, seq_len, hidden)
        return self.fc(out[:, -1, :])  # use the last time step


def build_xy(df: pd.DataFrame, features: list[str], target: str = "rev_per_vehicle"):
    """Scale features and log-target; return (X, y, scaler_y)."""
    x_raw = df[features].fillna(0).values.astype(np.float32)
    y_raw = np.log(df[target].replace(0, np.nan).ffill().bfill().values
                   ).astype(np.float32).reshape(-1, 1)
    x = StandardScaler().fit_transform(x_raw)
    scaler_y = StandardScaler()
    y = scaler_y.fit_transform(y_raw).flatten()
    return x, y, scaler_y


def kfold_train(x, y, scaler_y, n_splits=5, epochs=50, batch_size=512, lr=1e-3) -> dict:
    """K-fold train/evaluate the LSTM; return mean RMSE and R^2 (original scale)."""
    dataset = SequenceDataset(
        torch.from_numpy(x).float(), torch.from_numpy(y).float(), SEQ_LEN)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmse_list, r2_list = [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset), 1):
        train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=42)
        loaders = {
            name: DataLoader(Subset(dataset, idx), batch_size=batch_size,
                             shuffle=(name == "train"))
            for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]
        }
        model = LSTMRegressor(x.shape[1]).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        loss_fn = nn.MSELoss()

        best_val, best_state = np.inf, None
        for _ in range(epochs):
            model.train()
            for seq_x, yb in loaders["train"]:
                seq_x, yb = seq_x.to(DEVICE), yb.unsqueeze(1).to(DEVICE)
                opt.zero_grad()
                loss_fn(model(seq_x), yb).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                val_loss = np.mean([
                    loss_fn(model(sx.to(DEVICE)), yb.unsqueeze(1).to(DEVICE)).item()
                    for sx, yb in loaders["val"]
                ])
            if val_loss < best_val:
                best_val, best_state = val_loss, model.state_dict()

        model.load_state_dict(best_state)
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for seq_x, yb in loaders["test"]:
                preds.append(model(seq_x.to(DEVICE)).cpu().numpy())
                trues.append(yb.unsqueeze(1).numpy())
        y_pred = np.exp(scaler_y.inverse_transform(np.vstack(preds).astype(np.float32)))
        y_true = np.exp(scaler_y.inverse_transform(np.vstack(trues).astype(np.float32)))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
        print(f"Fold {fold} - RMSE: {rmse:.2f}, R2: {r2:.3f}")
        rmse_list.append(rmse)
        r2_list.append(r2)

    result = {"rmse": float(np.mean(rmse_list)), "r2": float(np.mean(r2_list))}
    print(f"Average RMSE: {result['rmse']:.2f}, R2: {result['r2']:.3f}")
    return result


def main(feature_store: str, features: list[str] | None = None) -> None:
    df = pd.read_feather(feature_store).sort_values("date")
    features = features or [c for c in df.columns
                            if c not in ("date", "spot_id", "vehicle_type", "rev_per_vehicle")]
    x, y, scaler_y = build_xy(df, features)
    kfold_train(x, y, scaler_y)


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "data/processed/df_features_seoul.feather")
