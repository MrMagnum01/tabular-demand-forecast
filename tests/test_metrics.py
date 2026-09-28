"""Metric formula and coverage tests on hand-built arrays only (no
generator, no frozen data). Includes the step-2 required fixtures
zero_target_wape_fixture and all_missing_stratum_fixture."""
import numpy as np
import pytest

from tabular_demand_forecast.metrics import NA, coverage, macro_store_average, point_metrics, stratum_report


def test_formulas():
    m = point_metrics([3.0, 1.0, 4.0], [1.0, 1.0, 2.0])
    assert m["N"] == 3
    assert m["MAE"] == pytest.approx(4.0 / 3)
    assert m["signed_bias"] == pytest.approx(4.0 / 3)
    assert m["WAPE"] == pytest.approx(4.0 / 4.0 * 100)
    assert m["WAPE_unit"] == "%"
    m2 = point_metrics([0.0, 2.0], [1.0, 3.0])
    assert m2["signed_bias"] == pytest.approx(-1.0)


def test_zero_target_wape_fixture():
    """id: zero_target_wape_fixture -- protocol worked example
    actual=[0], pred=[2] -> N=1, MAE=2, signed_bias=2, WAPE='N/A'."""
    m = point_metrics([2.0], [0.0])
    assert m["N"] == 1
    assert m["MAE"] == 2.0
    assert m["signed_bias"] == 2.0
    assert m["WAPE"] == NA
    # via a stratum report, counts are shown and the row is not dropped
    rep = stratum_report(np.array([0.0]), {"m": np.array([2.0])}, np.array([True]))
    own = rep["own_coverage"]["m"]
    assert own["eligible_rows"] == 1 and own["scored_rows"] == 1 and own["unavailable_rows"] == 0
    assert own["metrics"]["MAE"] == 2.0 and own["metrics"]["signed_bias"] == 2.0
    assert own["metrics"]["WAPE"] == NA


def test_all_missing_stratum_fixture():
    """id: all_missing_stratum_fixture -- a stratum with 0 eligible rows
    reports N=0 and all three metrics 'N/A' explicitly, with counts 0."""
    actual = np.array([np.nan, np.nan, 5.0])  # stratum = first two rows: both targets missing
    preds = {"a": np.array([1.0, 2.0, 3.0]), "b": np.array([np.nan, 2.0, 3.0])}
    stratum = np.array([True, True, False])
    rep = stratum_report(actual, preds, stratum)
    assert rep["rows_in_stratum"] == 2
    assert rep["eligible_rows"] == 0
    for name in ("a", "b"):
        own = rep["own_coverage"][name]
        assert own["eligible_rows"] == 0 and own["scored_rows"] == 0 and own["unavailable_rows"] == 0
        assert own["coverage"] == NA
        assert own["metrics"]["N"] == 0
        assert own["metrics"]["MAE"] == NA
        assert own["metrics"]["signed_bias"] == NA
        assert own["metrics"]["WAPE"] == NA
    assert rep["common_support"]["common_rows"] == 0
    assert set(rep["common_support"]["metrics"]) == {"a", "b"}  # not silently omitted


def test_empty_input_is_na_not_zero():
    m = point_metrics([], [])
    assert m["N"] == 0 and m["MAE"] == NA and m["signed_bias"] == NA and m["WAPE"] == NA


def test_non_finite_inputs_are_rejected_not_silently_scored():
    with pytest.raises(ValueError):
        point_metrics([np.nan], [1.0])
    with pytest.raises(ValueError):
        point_metrics([1.0], [np.nan])


def test_own_coverage_and_common_support_are_separate():
    actual = np.array([10.0, 10.0, 10.0, np.nan])
    preds = {"full": np.array([11.0, 12.0, 20.0, 1.0]), "partial": np.array([11.0, np.nan, np.nan, 1.0])}
    rep = stratum_report(actual, preds, np.ones(4, dtype=bool))
    full = rep["own_coverage"]["full"]
    part = rep["own_coverage"]["partial"]
    assert full["eligible_rows"] == 3 and full["scored_rows"] == 3 and full["coverage"] == 1.0
    assert part["scored_rows"] == 1 and part["unavailable_rows"] == 2 and part["coverage"] == pytest.approx(1 / 3)
    assert full["metrics"]["MAE"] == pytest.approx((1 + 2 + 10) / 3)
    cs = rep["common_support"]
    assert cs["common_rows"] == 1
    assert cs["metrics"]["full"]["MAE"] == pytest.approx(1.0)
    assert cs["metrics"]["full"]["MAE"] != full["metrics"]["MAE"]


def test_coverage_rejects_scored_outside_eligible():
    with pytest.raises(ValueError):
        coverage(np.array([False]), np.array([True]))


def test_macro_store_average_is_labelled_and_unweighted():
    actual = np.array([1.0] * 3 + [1.0])
    stores = {
        "store_A": stratum_report(actual, {"m": np.array([2.0, 2.0, 2.0, np.nan])}, np.array([1, 1, 1, 0], bool)),
        "store_B": stratum_report(actual, {"m": np.array([np.nan, np.nan, np.nan, 5.0])}, np.array([0, 0, 0, 1], bool)),
    }
    macro = macro_store_average(stores, "m")
    assert "macro store average" in macro["label"]
    assert macro["MAE"] == pytest.approx((1.0 + 4.0) / 2)  # row-weighted would be (3*1+4)/4
