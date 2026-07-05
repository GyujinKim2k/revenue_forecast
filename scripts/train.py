"""CLI: train the Temporal Fusion Transformer (main RPV model).

Usage:
    python scripts/train.py [--feature-store PATH] [--max-epochs N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from revenue_forecast import config  # noqa: E402
from revenue_forecast.models.train_tft import main as train_tft  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature-store", default=str(config.FEATURE_STORE))
    p.add_argument("--max-epochs", type=int, default=1000)
    args = p.parse_args()
    train_tft(feature_store=args.feature_store, max_epochs=args.max_epochs)


if __name__ == "__main__":
    main()
