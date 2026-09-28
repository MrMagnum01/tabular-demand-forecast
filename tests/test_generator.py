import numpy as np
import pandas as pd

from tabular_demand_forecast.generator import GeneratorSeeds, generate
from tabular_demand_forecast.protocol import load_protocol


def test_full_grid_row_counts():
    data = generate(GeneratorSeeds.frozen())
    expected = load_protocol()["entities"]["full_grid_rows"]
    assert len(data.latent_truth) == expected
    assert len(data.mask) == expected
    assert len(data.observed) == expected
    assert len(data.plans) == expected


def test_frozen_seeds_are_deterministic():
    seeds = GeneratorSeeds.frozen()
    a = generate(seeds)
    b = generate(seeds)
    pd.testing.assert_frame_equal(a.latent_truth, b.latent_truth)
    pd.testing.assert_frame_equal(a.mask, b.mask)
    pd.testing.assert_frame_equal(a.observed, b.observed)
    pd.testing.assert_frame_equal(a.plans, b.plans)
    assert a.truth_meta["shift_sku_magnitude"] == b.truth_meta["shift_sku_magnitude"]
    assert a.truth_meta["realized_missing_fraction"] == b.truth_meta["realized_missing_fraction"]


def test_dev_fixture_seeds_are_a_distinct_dataset():
    frozen = generate(GeneratorSeeds.frozen())
    dev = generate(GeneratorSeeds.dev_fixture())
    assert not np.array_equal(
        frozen.latent_truth["y_latent"].to_numpy(),
        dev.latent_truth["y_latent"].to_numpy(),
    )


def test_demand_is_non_negative_integer():
    data = generate(GeneratorSeeds.dev_fixture())
    y = data.latent_truth["y_latent"].to_numpy()
    assert (y >= 0).all()
    assert np.array_equal(y, y.astype(int))


def test_missingness_is_bernoulli_not_exact_quota():
    p = load_protocol()["planted_effects"]["missingness"]
    assert p["mechanism"].startswith("Bernoulli")
    data = generate(GeneratorSeeds.frozen())
    fraction = data.truth_meta["realized_missing_fraction"]
    n = len(data.mask)
    exact_quota = round(p["probability"] * n) / n
    # A Bernoulli draw need not land on the exact quota fraction.
    assert fraction != exact_quota
    assert abs(fraction - p["probability"]) < 0.01
    assert fraction == float((~data.mask["observed"]).mean())


def test_missing_cells_produce_nan_y_obs_and_retain_latent_value():
    data = generate(GeneratorSeeds.frozen())
    merged = data.mask.merge(data.observed, on=["store", "sku", "week_index"]).merge(
        data.latent_truth[["store", "sku", "week_index", "y_latent"]],
        on=["store", "sku", "week_index"],
    )
    missing_rows = merged[~merged["observed"]]
    assert len(missing_rows) > 0
    assert missing_rows["y_obs"].isna().all()
    # Latent truth is retained even though y_obs is masked (evaluator-only visibility).
    assert missing_rows["y_latent"].notna().all()


def test_shift_effect_onset_and_magnitudes():
    data = generate(GeneratorSeeds.frozen())
    shift_cfg = load_protocol()["planted_effects"]["demand_shift"]
    onset = shift_cfg["onset_target_week"]
    shift_map = data.truth_meta["shift_sku_magnitude"]
    assert len(shift_map) == shift_cfg["n_skus"]
    assert set(shift_map.values()) == set(shift_cfg["magnitudes"])

    lt = data.latent_truth
    for sku, magnitude in shift_map.items():
        rows = lt[lt["sku"] == sku]
        before = rows[rows["week_index"] < onset]
        after = rows[rows["week_index"] >= onset]
        assert (before["shift_multiplier"] == 1.0).all()
        assert not before["is_shift_active"].any()
        assert np.allclose(after["shift_multiplier"], magnitude)
        assert after["is_shift_active"].all()

    non_shift = lt[~lt["sku"].isin(shift_map.keys())]
    assert (non_shift["shift_multiplier"] == 1.0).all()
    assert not non_shift["is_shift_active"].any()


def test_sku_base_level_and_reference_price_shapes():
    data = generate(GeneratorSeeds.frozen())
    n_skus = load_protocol()["entities"]["n_skus"]
    assert len(data.truth_meta["sku_base_level"]) == n_skus
    assert len(data.truth_meta["reference_price"]) == n_skus
    assert all(v > 0 for v in data.truth_meta["sku_base_level"].values())
    assert all(5.0 <= v <= 50.0 for v in data.truth_meta["reference_price"].values())


def test_price_formula_matches_protocol():
    data = generate(GeneratorSeeds.frozen())
    ref = data.truth_meta["reference_price"]
    row = data.plans.iloc[0]
    ref_price = ref[row["sku"]]
    ratio = row["price"] / (ref_price * (1.0 - 0.1 * row["promo_flag"]))
    assert 0.98 - 1e-9 <= ratio <= 1.02 + 1e-9


def test_seasonal_multiplier_matches_formula():
    data = generate(GeneratorSeeds.frozen())
    p = load_protocol()["planted_effects"]["seasonality"]
    row = data.latent_truth.iloc[0]
    from tabular_demand_forecast.calendar_utils import week_of_year

    woy = week_of_year(int(row["week_index"]))
    expected = 1.0 + p["amplitude"] * np.sin(2.0 * np.pi * (woy - 1) / 52.0)
    assert abs(row["seasonal_multiplier"] - expected) < 1e-12
