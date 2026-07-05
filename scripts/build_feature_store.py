"""CLI: build the modelling feature store from raw sources.

Reads the raw revenue CSVs, spot metadata, holiday calendar and weather, and
writes the daily feature matrix consumed by the models. Concrete input paths and
schemas are documented in ``data/README.md``.

Usage:
    python scripts/build_feature_store.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from revenue_forecast.features.build_features import build_feature_store  # noqa: E402

if __name__ == "__main__":
    build_feature_store()
