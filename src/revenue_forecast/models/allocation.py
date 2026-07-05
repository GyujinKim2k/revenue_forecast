"""Vehicle-allocation optimization on top of the TFT forecaster.

Downstream use case from the project review ("Allocation Optimization Algorithm
Design"): reallocate vehicles across spots to maximize total revenue, moving a
vehicle only when the marginal RPV premium clears an operational hurdle.

Two pieces:

1. ``marginal_revenue_curve`` – for one (spot, vehicle_type), sweep the inventory
   level and use the TFT to estimate the marginal revenue Δrev of adding each
   next vehicle (diminishing returns as a spot saturates).
2. ``rebalance_greedy`` – greedily move one vehicle at a time from the lowest-
   marginal-loss source to the highest-marginal-gain destination (same vehicle
   type, different spot) while the net gain exceeds ``min_gain``.

Converted from ``analyzeTFT.ipynb``.
"""
from __future__ import annotations

import time
from collections import defaultdict

import pandas as pd

from ._patches import apply_patches

apply_patches()

from pytorch_forecasting import TimeSeriesDataSet  # noqa: E402


def _mean_next_rpv(tft, training, frame: pd.DataFrame) -> float:
    """Mean forecast RPV over the horizon for a single-series frame."""
    ds = TimeSeriesDataSet.from_dataset(training, frame, predict=True, stop_randomization=True)
    pred = tft.predict(ds.to_dataloader(train=False, batch_size=64, num_workers=2),
                       mode="prediction", return_index=True)
    arr = pred.output
    arr = arr.squeeze(-1) if arr.ndim == 3 else arr
    return float(arr.cpu().numpy().ravel().mean())


def marginal_revenue_curve(tft, training, df: pd.DataFrame, spot_id, vehicle_type,
                           max_add: int = 2) -> list[tuple[int, float]]:
    """Marginal Δrev of adding each next vehicle at one (spot, vehicle_type).

    Returns ``[(inventory_level, delta_revenue), ...]`` where ``delta_revenue``
    is ``rev(level) - rev(level - 1)``.
    """
    base = df[(df.spot_id == spot_id) & (df.vehicle_type == vehicle_type)].copy()
    if base.empty:
        return []
    base_inv = int(base["inventory_est"].iloc[0])

    curve, prev_rev = [], None
    for inv in range(base_inv, base_inv + max_add + 1):
        frame = base.copy()
        frame["inventory_est"] = inv
        rev = _mean_next_rpv(tft, training, frame)
        if prev_rev is not None:
            curve.append((inv, rev - prev_rev))
        prev_rev = rev
    return curve


def all_marginal_curves(tft, training, df: pd.DataFrame, max_add: int = 2) -> dict:
    """Marginal-revenue curves for every (spot, vehicle_type), skipping failures."""
    sensitivity = defaultdict(list)
    for spot, veh in df.groupby(["spot_id", "vehicle_type"]).size().index:
        try:
            sensitivity[(spot, veh)] = marginal_revenue_curve(
                tft, training, df, spot, veh, max_add=max_add)
        except Exception as exc:  # noqa: BLE001 - skip series the model can't score
            print(f"skip ({spot}, {veh}): {exc}")
    return sensitivity


def current_allocation(df: pd.DataFrame) -> dict:
    """Current inventory per (spot, vehicle_type) from the feature frame."""
    dedup = df[["spot_id", "vehicle_type", "inventory_est"]].drop_duplicates(
        subset=["spot_id", "vehicle_type"])
    return {(r.spot_id, r.vehicle_type): int(r.inventory_est) for _, r in dedup.iterrows()}


def rebalance_greedy(sensitivity: dict, allocation: dict,
                     timeout_seconds: float = 600, min_gain: float = 5000):
    """Greedily relocate vehicles to maximize projected revenue.

    Args:
        sensitivity: ``{(spot, vehicle_type): [(level, delta_rev), ...]}``.
        allocation: current ``{(spot, vehicle_type): count}``.
        min_gain: only relocate when net gain (dest gain - src loss) exceeds this.

    Returns ``(new_allocation, moves, timed_out)`` where ``moves`` is a list of
    ``(vehicle_type, src_spot, dst_spot, net_gain)``.
    """
    start = time.time()
    alloc = allocation.copy()
    moves, timed_out = [], False

    while True:
        if time.time() - start > timeout_seconds:
            timed_out = True
            break

        # Loss of removing one vehicle at its current level.
        src = []
        for (s, v), curve in sensitivity.items():
            n = alloc.get((s, v), 0)
            if n <= 0:
                continue
            loss = next((m for lvl, m in curve if lvl == n), None)
            if loss is not None:
                src.append((loss, s, v))
        # Gain of adding one vehicle at the next level.
        dst = []
        for (s, v), curve in sensitivity.items():
            n = alloc.get((s, v), 0)
            gain = next((m for lvl, m in curve if lvl == n + 1), None)
            if gain is not None:
                dst.append((gain, s, v))

        best = (min_gain, None, None, None)  # (net_gain, src_spot, dst_spot, vehicle)
        for loss, s_src, v in src:
            for gain, s_dst, v2 in dst:
                if v != v2 or s_src == s_dst:
                    continue
                net = gain - loss
                if net > best[0]:
                    best = (net, s_src, s_dst, v)

        net, s_src, s_dst, v = best
        if s_src is None:
            break  # no move clears the hurdle
        alloc[(s_src, v)] -= 1
        alloc[(s_dst, v)] += 1
        moves.append((v, s_src, s_dst, net))

    return alloc, moves, timed_out


def projected_revenue(tft, training, df: pd.DataFrame, allocation: dict) -> float:
    """Total projected revenue = sum over series of forecast RPV x allocated count."""
    df2 = df.copy()
    df2["inventory_est"] = df2.apply(
        lambda r: allocation.get((r.spot_id, r.vehicle_type), r.inventory_est), axis=1)
    ds = TimeSeriesDataSet.from_dataset(training, df2, predict=True, stop_randomization=True)
    pred = tft.predict(ds.to_dataloader(train=False, batch_size=512, num_workers=4),
                       mode="prediction", return_index=True)
    arr = pred.output
    arr = arr.squeeze(-1) if arr.ndim == 3 else arr
    rpv = arr.cpu().numpy().mean(axis=1)
    tbl = pred.index.reset_index(drop=True).assign(rpv=rpv)
    tbl = tbl.groupby(["spot_id", "vehicle_type"])["rpv"].mean().reset_index()
    tbl["count"] = tbl.apply(
        lambda r: allocation.get((r.spot_id, r.vehicle_type), 0), axis=1)
    return float((tbl["rpv"] * tbl["count"]).sum())
