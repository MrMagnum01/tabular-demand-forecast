"""Grid-search/selection tests. DEV FIXTURE ONLY (GeneratorSeeds.dev_fixture()
or hand-built frames): nothing here generates the frozen dataset, and no
test asserts on, prints or tunes to an accuracy value -- only on
structure, determinism and failure handling.

Implements protocol required_leakage_tests.deferred_to_step_2_with_estimator_code:
held_out_store_does_not_affect_fitting and failed_model_fixture.
"""
import dataclasses

import numpy as np
import pytest

from tabular_demand_forecast.evaluate import final_refit
from tabular_demand_forecast.features import ObservationLookup, build_partition
from tabular_demand_forecast.generator import GeneratorSeeds, generate
from tabular_demand_forecast.metrics import MethodFailure
from tabular_demand_forecast.model_selection import (
    ALL_FAILED_MESSAGE,
    GRID_KEYS,
    NO_ELIGIBLE_ROWS_MESSAGE,
    declared_grid,
    eligible_key_set_mask,
    gbm_available_mask,
    make_estimator,
    predict_gbm,
    run_grid_search,
)
from tabular_demand_forecast.protocol import load_protocol

SMALL = {"max_iter": 20, "learning_rate": 0.1, "max_leaf_nodes": 15, "min_samples_leaf": 20, "l2_regularization": 0.0}


@pytest.fixture(scope="module")
def dev():
    data = generate(GeneratorSeeds.dev_fixture())
    return {
        "data": data,
        "obs": ObservationLookup(data),
        "train": build_partition(data, "train_AB"),
        "val": build_partition(data, "validation_AB"),
    }


def test_declared_grid_matches_protocol_order():
    grid = declared_grid()
    spec = load_protocol()["models"]["gradient_boosting"]["pre_declared_grid"]
    assert len(grid) == 36 == spec["total_configurations"]
    # outer-to-inner nesting in literal list order
    assert grid[0] == {k: spec[k][0] for k in GRID_KEYS}
    assert grid[1]["l2_regularization"] == spec["l2_regularization"][1]
    assert grid[2]["max_leaf_nodes"] == spec["max_leaf_nodes"][1]
    assert grid[4]["learning_rate"] == spec["learning_rate"][1]
    assert grid[12]["max_iter"] == spec["max_iter"][1]
    assert grid[-1] == {k: spec[k][-1] for k in GRID_KEYS}


def test_estimator_settings_from_protocol():
    est = make_estimator(SMALL)
    spec = load_protocol()["models"]["gradient_boosting"]
    assert est.random_state == spec["random_state"] == 20260928
    assert est.early_stopping is False
    assert est.loss == "squared_error"


def test_failed_model_fixture(dev):
    """id: failed_model_fixture -- a degenerate configuration
    (max_leaf_nodes=1, rejected by sklearn) and a configuration whose
    estimator raises at fit are both recorded with a reason; the loop
    continues and still selects among the successes."""
    bad_param = dict(SMALL, max_leaf_nodes=1)
    boom = dict(SMALL, max_iter=21)

    def factory(cfg):
        if cfg["max_iter"] == 21:
            raise RuntimeError("engineered failure for fixture")
        return make_estimator(cfg)

    log = run_grid_search(dev["train"], dev["val"], dev["obs"], grid=[SMALL, bad_param, boom], estimator_factory=factory)
    statuses = [c["status"] for c in log["configurations"]]
    assert statuses == ["succeeded", "failed", "failed"]
    assert "max_leaf_nodes" in log["configurations"][1]["reason"]
    assert "engineered failure" in log["configurations"][2]["reason"]
    assert log["configurations"][1]["config"] == bad_param
    assert log["n_failed"] == 2 and log["n_succeeded"] == 1
    assert log["status"] == "selected" and log["selected"]["index"] == 0


def test_all_configs_failed_reports_failed_to_select(dev):
    bad = dict(SMALL, max_leaf_nodes=1)
    log = run_grid_search(dev["train"], dev["val"], dev["obs"], grid=[bad, bad])
    assert log["status"] == "all_failed"
    assert log["reason"] == ALL_FAILED_MESSAGE
    assert log["selected"] is None
    assert len(log["configurations"]) == 2


def test_no_eligible_rows_refuses_selection(dev):
    val = dev["val"].copy()
    val["y_obs"] = np.nan
    log = run_grid_search(dev["train"], val, dev["obs"], grid=[SMALL])
    assert log["status"] == "refused" and log["reason"] == NO_ELIGIBLE_ROWS_MESSAGE
    assert log["configurations"] == []


def test_same_mask_for_every_config_and_baseline(dev):
    mask = eligible_key_set_mask(dev["val"])
    log = run_grid_search(dev["train"], dev["val"], dev["obs"], grid=[SMALL, dict(SMALL, learning_rate=0.3)])
    n = int(mask.sum())
    assert n > 0 and log["eligible_key_set"]["eligible_key_set_rows"] == n
    for c in log["configurations"]:
        assert c["validation_metrics"]["N"] == n
    for b in log["baselines_on_eligible_key_set"].values():
        assert b["metrics"]["N"] == n


def test_tie_break_is_first_in_grid_order(dev):
    class Constant:
        def __init__(self, cfg):
            pass

        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.full(len(X), 5.0)

    grid = [dict(SMALL, max_iter=i) for i in (1, 2, 3)]
    log = run_grid_search(dev["train"], dev["val"], dev["obs"], grid=grid, estimator_factory=Constant)
    assert log["selected"]["index"] == 0
    assert log["selected"]["tied_indices"] == [0, 1, 2]


class _Broken:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)


def test_predict_gbm_raises_on_nonfinite_raw_for_available_row(dev):
    """Astra HOLD round-2 clause 3 (2026-09-28-astra-tabular-r2-review.md
    finding 3; probes nan_model / neg_inf_after_clip): predict_gbm must
    validate the RAW prediction's finiteness on rows the input-availability
    mask (gbm_available_mask) marks available, BEFORE any clipping. NaN or
    Inf/-Inf there is a failed prediction (MethodFailure), never a silent
    'unavailable' and never a value clip_non_negative can turn into 0."""
    frame = dev["val"]
    available = gbm_available_mask(frame)
    assert available.any()
    sub = frame.loc[available].iloc[:3]
    for bad_value in (np.nan, -np.inf, np.inf):
        with pytest.raises(MethodFailure):
            predict_gbm(_Broken(bad_value), sub)


def test_predict_gbm_genuinely_unavailable_row_is_not_a_failure(dev):
    """A row whose plan inputs are themselves unavailable must still come
    back as unavailable (available=False), never MethodFailure, even though
    the same broken estimator would raise if that row were available."""
    frame = dev["val"].iloc[:3].copy()
    frame["price"] = np.nan  # no plan for these rows -> gbm_available_mask is False
    frame["promo_flag"] = np.nan
    assert not gbm_available_mask(frame).any()
    out = predict_gbm(_Broken(np.nan), frame)
    assert not out["available"].any()


def _model_fingerprint(model):
    parts = [np.asarray(model._baseline_prediction).tobytes()]
    for iteration in model._predictors:
        for predictor in iteration:
            parts.append(predictor.nodes.tobytes())
    return b"".join(parts)


def test_held_out_store_does_not_affect_fitting(dev):
    """id: held_out_store_does_not_affect_fitting -- rewriting every one of
    store C's observations leaves the selection log and the refit model's
    fitted tree arrays byte-for-byte unchanged."""
    data = dev["data"]
    grid = [SMALL, dict(SMALL, learning_rate=0.3), dict(SMALL, max_leaf_nodes=31, l2_regularization=1.0)]

    def run(d):
        obs = ObservationLookup(d)
        log = run_grid_search(build_partition(d, "train_AB"), build_partition(d, "validation_AB"), obs, grid=grid)
        model, _ = final_refit(d, log["selected"]["config"])
        return log, _model_fingerprint(model)

    observed = data.observed.copy()
    c_rows = observed["store"] == "C"
    observed.loc[c_rows, "y_obs"] = observed.loc[c_rows, "y_obs"] * 7 + 3
    plans = data.plans.copy()
    plans.loc[plans["store"] == "C", "price"] *= 3.0
    changed = dataclasses.replace(data, observed=observed, plans=plans)
    assert not changed.observed.equals(data.observed)

    log_a, fp_a = run(data)
    log_b, fp_b = run(changed)
    assert log_a["selected"] == log_b["selected"]
    assert log_a["configurations"] == log_b["configurations"]
    assert fp_a == fp_b
