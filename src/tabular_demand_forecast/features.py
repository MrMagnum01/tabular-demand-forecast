"""Leakage-safe feature construction, driven by eval/protocol.v2.json.

Every feature for target week t uses only values available at
origin(t) = week_start_date(t): calendar lags read off the full
reindexed weekly grid, rolling means ending at t-1, and price/promo
PLANS whose own available_at is <= origin(t) -- checked per row, never
inferred from a plan's week_index matching t alone, so a plan keyed to
t but delayed past origin(t) is rejected too. No estimator code.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .calendar_utils import origin as week_origin
from .calendar_utils import week_of_year
from .generator import GeneratedData
from .protocol import load_protocol


def _load_feature_spec() -> Tuple[List[str], List[int], Dict[str, Tuple[int, int]]]:
    """Derive the fixed feature list, lag K-values and rolling-window specs
    from eval/protocol.v2.json (the single source of truth), instead of
    hardcoding them a second time here. The derived column order is
    validated against features.fixed_list so the two can never silently
    drift apart."""
    feats = load_protocol()["features"]
    lag_ks = list(feats["lags"]["K_values"])
    roll_specs = {
        name: (feats["rolling_means"][name]["window_size"], feats["rolling_means"][name]["min_observed_required"])
        for name in ("roll_mean_4", "roll_mean_13")
    }
    derived_columns = (
        [f"lag{k}" for k in lag_ks]
        + list(roll_specs.keys())
        + ["price", "promo_flag", "week_of_year"]
    )
    if derived_columns != feats["fixed_list"]:
        raise ValueError(
            "features.py's derived feature column order "
            f"{derived_columns} no longer matches protocol features.fixed_list "
            f"{feats['fixed_list']}; update one to match the other explicitly."
        )
    return derived_columns, lag_ks, roll_specs


FEATURE_COLUMNS, _LAG_KS, _ROLL_SPECS = _load_feature_spec()


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
    here that can read a later week's plan for an earlier target. In
    addition, per protocol features.price_and_promo.availability_enforcement,
    a plan is usable for target week t only if ITS OWN available_at is
    <= origin(t) -- checked even when the plan's week_index equals t, so a
    same-target plan that was delayed past origin(t) is rejected, not
    just a different week's plan being looked up for the wrong target.
    """

    def __init__(self, data: GeneratedData) -> None:
        plans = data.plans
        self._values: Dict[Tuple[str, str, int], Tuple[float, float, object]] = dict(
            zip(
                zip(plans["store"], plans["sku"], plans["week_index"]),
                zip(plans["price"], plans["promo_flag"], plans["available_at"]),
            )
        )

    def get(self, store: str, sku: str, week_index: int, origin: pd.Timestamp) -> Tuple[float, float]:
        entry = self._values.get((store, sku, week_index))
        if entry is None:
            return (float("nan"), float("nan"))
        price, promo_flag, available_at = entry
        if pd.isna(available_at) or available_at > origin:
            return (float("nan"), float("nan"))
        return (float(price), float(promo_flag))


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
                origin_ts = pd.Timestamp(week_origin(t))
                price, promo_flag = plans.get(store, sku, t, origin_ts)
                row = {
                    "store": store,
                    "sku": sku,
                    "target_week": t,
                }
                for k in _LAG_KS:
                    row[f"lag{k}"] = _lag(obs, store, sku, t, k)
                for name, (window, min_observed) in _ROLL_SPECS.items():
                    row[name] = _roll_mean(obs, store, sku, t, window, min_observed)
                row["price"] = price
                row["promo_flag"] = promo_flag
                row["week_of_year"] = week_of_year(t)
                row["y_obs"] = obs.get(store, sku, t)
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
