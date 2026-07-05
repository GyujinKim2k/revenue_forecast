"""Descriptive statistics & EDA figures for the RPV feature store.

Regenerates the figures used in the project review:
* correlation matrix of numeric features,
* weekly average RPV trend,
* mean RPV heatmap by vehicle type x day-of-week,
plus a holiday-vs-weekday summary. Figures are written to ``reports/figures/``
(git-ignored). Converted from the EDA cells of ``analyzeTFT.ipynb`` /
``eda_new.ipynb``.

Usage:
    python scripts/run_eda.py [FEATURE_STORE.feather]
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Make the src package importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from revenue_forecast import config  # noqa: E402

TARGET = config.TARGET
NUMERIC_COLS = config.TIME_VARYING_KNOWN_REALS + config.TIME_VARYING_UNKNOWN_REALS + [TARGET]
FIG_DIR = config.ROOT_DIR / "reports" / "figures"


def run(feature_store: Path) -> None:
    df = pd.read_feather(feature_store)
    df["date"] = pd.to_datetime(df["date"])
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    numeric = [c for c in NUMERIC_COLS if c in df.columns]

    # 1) Correlation matrix.
    plt.figure(figsize=(12, 10))
    sns.heatmap(df[numeric].corr(method="pearson"), cmap="coolwarm", center=0, vmin=-1, vmax=1)
    plt.title("Correlation matrix (numeric features)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "corr_matrix.png", dpi=150)
    plt.close()

    # 2) Weekly average RPV trend.
    weekly = df.set_index("date").resample("W")[TARGET].mean()
    plt.figure(figsize=(12, 4))
    weekly.plot()
    plt.title("Weekly average revenue per vehicle")
    plt.ylabel("KRW / vehicle")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "weekly_trend.png", dpi=150)
    plt.close()

    # 3) Vehicle type x day-of-week heatmap.
    pivot = pd.pivot_table(df, values=TARGET, index="vehicle_type", columns="dow", aggfunc="mean")
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu")
    plt.title("Mean RPV by vehicle type and weekday (0=Mon)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "vehicle_dow_heatmap.png", dpi=150)
    plt.close()

    # 4) Holiday vs. weekday summary.
    print("Mean RPV by holiday flag:")
    print(df.groupby("is_holiday")[TARGET].mean())
    print(f"\nFigures written to {FIG_DIR}")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else config.FEATURE_STORE
    run(path)
