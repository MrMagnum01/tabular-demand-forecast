"""Leakage-safe feature construction, driven by eval/protocol.json.

Every feature for target week t uses only values available at
origin(t) = week_start_date(t): calendar lags read off the full
reindexed weekly grid, rolling means ending at t-1, and price/promo
PLANS keyed to t itself (never a later week's plan). No estimator code.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .calendar_utils import week_of_year
from .generator import GeneratedData
from .protocol import load_protocol

FEATURE_COLUMNS = [
    "lag1",
    "lag2",
    "lag4",
    "lag52",
    "roll_mean_4",
    "roll_mean_13",
    "price",
    "promo_flag",
    "week_of_year",
]

_LAG_KS = [1, 2, 4, 52]
_ROLL_SPECS = {"roll_mean_4": (4, 3), "roll_mean_13": (13, 10)}


class ObservationLookup:
    """O(1) (store, sku, week_index) -> y_obs lookup over the full grid.

    Returns NaN for a masked (missing) cell AND for a week outside the
    generated grid (week_index < 1); both are "no value available",
    never a nearest-available substitute.
    """

    def __init__(self, data: GeneratedData) -> None:
        obs = data.observed
        self._values: Dict[Tuple[str, str, int], float] = dict(
            zip(
                zip(obs["store"], obs["sku"], obs["week_index"]),
                obs["y_obs"],
            )
        )
        self._n_weeks = data.n_weeks

    def get(self, store: str, sku: str, week_index: int) -> float:
        if week_index < 1 or week_index > self._n_weeks:
            return float("nan")
        return self._values.get((store, sku, week_index), float("nan"))


class PlanLookup:
    """O(1) (store, sku, week_index) -> (price, promo_flag) lookup.

    A plan's join key is the target week itself; there is no code path
    here that can read a later week's plan for an earlier target.
    """

    def __init__(self, data: GeneratedData) -> None:
        plans = data.plans
        self._values: Dict[Tuple[str, str, int], Tuple[float, int]] = dict(
            zip(
                zip(plans["store"], plans["sku"], plans["week_index"]),
                zip(plans["price"], plans["promo_flag"]),
            )
        )

    def get(self, store: str, sku: str, week_index: int) -> Tuple[float, int]:
        return self._values.get((store, sku, week_index), (float("nan"), np.nan))


def _lag(obs: ObservationLookup, store: str, sku: str, target_week: int, k: int) -> float:
    return obs.get(store, sku, target_week - k)


def _roll_mean(
    obs: ObservationLookup,
    store: str,
    sku: str,
    target_week: int,
    window: int,
    min_observed: int,
) -> float:
    values = [
        obs.get(store, sku, target_week - offset) for offset in range(1, window + 1)
    ]
    observed_values = [v for v in values if not np.isnan(v)]
    if len(observed_values) < min_observed:
        return float("nan")
    return float(np.mean(observed_values))


def build_feature_frame(
    data: GeneratedData,
    stores: Iterable[str],
    target_weeks: Iterable[int],
) -> pd.DataFrame:
    """One row per (store, sku, target_week) in the given cross-product.

    Columns: store, sku, target_week, the fixed feature list, y_obs
    (the target, NaN if masked missing) and observed (bool).
    """
    obs = ObservationLookup(data)
    plans = PlanLookup(data)
    rows: List[dict] = []
    for store in stores:
        for sku in data.skus:
            for t in target_weeks:
                price, promo_flag = plans.get(store, sku, t)
                row = {
                    "store": store,
                    "sku": sku,
                    "target_week": t,
                    "lag1": _lag(obs, store, sku, t, 1),
                    "lag2": _lag(obs, store, sku, t, 2),
                    "lag4": _lag(obs, store, sku, t, 4),
                    "lag52": _lag(obs, store, sku, t, 52),
                    "roll_mean_4": _roll_mean(obs, store, sku, t, 4, 3),
                    "roll_mean_13": _roll_mean(obs, store, sku, t, 13, 10),
                    "price": price,
                    "promo_flag": promo_flag,
                    "week_of_year": week_of_year(t),
                    "y_obs": obs.get(store, sku, t),
                }
                row["observed"] = not np.isnan(row["y_obs"])
                rows.append(row)
    return pd.DataFrame(rows)


def _week_range(bounds: Tuple[int, int]) -> range:
    start, end = bounds
    return range(start, end + 1)


def partition_bounds() -> Dict[str, Tuple[int, int]]:
    p = load_protocol()["partitions"]["definition_by_store_group"]
    return {
        "train_AB": tuple(p["stores_A_B"]["train_target_weeks"]),
        "validation_AB": tuple(p["stores_A_B"]["validation_target_weeks"]),
        "final_refit_AB": tuple(p["stores_A_B"]["final_refit_target_weeks"]),
        "test_AB": tuple(p["stores_A_B"]["test_target_weeks"]),
        "test_C": tuple(p["store_C"]["test_target_weeks"]),
    }


def build_partition(data: GeneratedData, name: str) -> pd.DataFrame:
    bounds = partition_bounds()[name]
    stores = ["C"] if name == "test_C" else ["A", "B"]
    return build_feature_frame(data, stores, _week_range(bounds))
