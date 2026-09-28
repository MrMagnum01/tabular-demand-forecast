import numpy as np
import pandas as pd

from tabular_demand_forecast.calendar_utils import week_of_year
from tabular_demand_forecast.features import (
    ObservationLookup,
    _roll_mean,
    build_feature_frame,
    build_partition,
)
from tabular_demand_forecast.generator import GeneratedData, GeneratorSeeds, generate
from tabular_demand_forecast.protocol import load_protocol


def _fixture_data(observed_rows, n_weeks=110, stores=("A",), skus=("SKU001",)):
    """A minimal GeneratedData carrying only an observed table, for unit
    tests of lag/rolling logic that don't need the full random generator."""
    observed = pd.DataFrame(observed_rows, columns=["store", "sku", "week_index", "y_obs"])
    plans = pd.DataFrame(
        [
            {"store": r["store"], "sku": r["sku"], "week_index": r["week_index"], "price": 10.0, "promo_flag": 0}
            for r in observed_rows
        ]
    )
    return GeneratedData(
        stores=list(stores),
        skus=list(skus),
        n_weeks=n_weeks,
        latent_truth=None,
        mask=None,
        observed=observed,
        plans=plans,
        truth_meta={},
    )


def test_partition_row_counts_match_protocol():
    data = generate(GeneratorSeeds.frozen())
    counts = load_protocol()["partitions"]["effective_row_counts"]
    assert len(build_partition(data, "train_AB")) == counts["train_AB"]["rows"]
    assert len(build_partition(data, "validation_AB")) == counts["validation_AB"]["rows"]
    assert len(build_partition(data, "final_refit_AB")) == counts["final_refit_AB"]["rows"]
    assert len(build_partition(data, "test_AB")) == counts["test_AB"]["rows"]
    assert len(build_partition(data, "test_C")) == counts["test_C"]["rows"]


def test_lag1_reads_exact_prior_week():
    rows = [
        {"store": "A", "sku": "SKU001", "week_index": w, "y_obs": float(w)}
        for w in range(1, 60)
    ]
    data = _fixture_data(rows)
    frame = build_feature_frame(data, ["A"], [55])
    r = frame.iloc[0]
    assert r["lag1"] == 54.0
    assert r["lag2"] == 53.0
    assert r["lag4"] == 51.0
    assert r["lag52"] == 3.0


def test_lag_before_grid_start_is_nan():
    rows = [{"store": "A", "sku": "SKU001", "week_index": w, "y_obs": float(w)} for w in range(1, 10)]
    data = _fixture_data(rows)
    frame = build_feature_frame(data, ["A"], [2])
    r = frame.iloc[0]
    assert np.isnan(r["lag4"])  # week 2-4 = -2, no such week
    assert np.isnan(r["lag52"])


def test_roll_mean_requires_min_observed_count():
    obs = ObservationLookup(_fixture_data([
        {"store": "A", "sku": "SKU001", "week_index": 10, "y_obs": 10.0},
        {"store": "A", "sku": "SKU001", "week_index": 9, "y_obs": np.nan},
        {"store": "A", "sku": "SKU001", "week_index": 8, "y_obs": np.nan},
        {"store": "A", "sku": "SKU001", "week_index": 7, "y_obs": 7.0},
    ]))
    # target week 11: window is weeks 10,9,8,7 -> only 2 observed (<3) -> NaN
    assert np.isnan(_roll_mean(obs, "A", "SKU001", 11, 4, 3))


def test_roll_mean_computes_mean_of_observed_when_threshold_met():
    obs = ObservationLookup(_fixture_data([
        {"store": "A", "sku": "SKU001", "week_index": 10, "y_obs": 10.0},
        {"store": "A", "sku": "SKU001", "week_index": 9, "y_obs": 20.0},
        {"store": "A", "sku": "SKU001", "week_index": 8, "y_obs": np.nan},
        {"store": "A", "sku": "SKU001", "week_index": 7, "y_obs": 30.0},
    ]))
    # 3 of 4 observed (>=3 threshold): mean of 10,20,30
    assert _roll_mean(obs, "A", "SKU001", 11, 4, 3) == 20.0


def test_roll_mean_window_never_includes_target_week():
    rows = [{"store": "A", "sku": "SKU001", "week_index": w, "y_obs": 1000.0 if w == 11 else 1.0} for w in range(1, 12)]
    data = _fixture_data(rows)
    obs = ObservationLookup(data)
    # target week 11's own value (1000) must not leak into its own roll_mean_4
    assert _roll_mean(obs, "A", "SKU001", 11, 4, 3) == 1.0


def test_week_of_year_feature_matches_calendar_utils():
    data = generate(GeneratorSeeds.dev_fixture())
    frame = build_feature_frame(data, ["A"], [60])
    assert frame.iloc[0]["week_of_year"] == week_of_year(60)


def test_price_and_promo_feature_come_from_exact_target_week_plan():
    rows = [{"store": "A", "sku": "SKU001", "week_index": w, "y_obs": 1.0} for w in range(1, 60)]
    data = _fixture_data(rows)
    # Give week 55 and week 56 distinguishable plans.
    data.plans.loc[data.plans["week_index"] == 55, "price"] = 11.0
    data.plans.loc[data.plans["week_index"] == 55, "promo_flag"] = 1
    data.plans.loc[data.plans["week_index"] == 56, "price"] = 99.0
    data.plans.loc[data.plans["week_index"] == 56, "promo_flag"] = 0
    frame = build_feature_frame(data, ["A"], [55])
    r = frame.iloc[0]
    assert r["price"] == 11.0
    assert r["promo_flag"] == 1
