"""Evaluation/forecast/report tests on DEV FIXTURE or hand-built data only.

Implements protocol required fixture negative_prediction_policy_fixture:
a raw negative GBM prediction is clipped to max(0, raw) and BOTH values
are recoverable from the saved forecasts CSV bytes.
"""
import html

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from tabular_demand_forecast.baselines import baseline_predictions
from tabular_demand_forecast.calendar_utils import week_start_date
from tabular_demand_forecast.csv_safety import write_safe_csv
from tabular_demand_forecast.evaluate import (
    FUTURE_FORECAST_CAVEAT,
    build_forecasts_frame,
    future_forecast,
    method_predictions,
)
from tabular_demand_forecast.features import FEATURE_COLUMNS, ObservationLookup, build_feature_frame, build_partition
from tabular_demand_forecast.generator import GeneratedData, GeneratorSeeds, generate
from tabular_demand_forecast.model_selection import GBM_NAME, clip_non_negative
from tabular_demand_forecast.report import _e, render_report

PROV = {"model_hash": "m" * 64, "protocol_hash": "p" * 64, "manifest_hash": "f" * 64}


def _tiny_data(n_weeks=104):
    rows, plans = [], []
    for w in range(1, n_weeks + 1):
        rows.append({"store": "A", "sku": "SKU001", "week_index": w, "y_obs": float(w % 7)})
        plans.append({"store": "A", "sku": "SKU001", "week_index": w, "price": 10.0, "promo_flag": 0,
                      "available_at": pd.Timestamp(week_start_date(w))})
    return GeneratedData(stores=["A"], skus=["SKU001"], n_weeks=n_weeks, latent_truth=None, mask=None,
                         observed=pd.DataFrame(rows), plans=pd.DataFrame(plans), truth_meta={})


def _negative_model():
    X = pd.DataFrame(np.arange(40 * len(FEATURE_COLUMNS), dtype=float).reshape(40, -1), columns=FEATURE_COLUMNS)
    y = np.full(40, -10.0)  # fixture-only negative target => negative raw predictions
    return HistGradientBoostingRegressor(max_iter=5, early_stopping=False, random_state=0).fit(X, y)


def test_clip_non_negative():
    assert clip_non_negative(np.array([-3.0, 0.0, 2.5])).tolist() == [0.0, 0.0, 2.5]


def test_negative_prediction_policy_fixture(tmp_path):
    """id: negative_prediction_policy_fixture"""
    data = _tiny_data()
    obs = ObservationLookup(data)
    frame = build_feature_frame(data, ["A"], range(92, 96))
    model = _negative_model()
    preds = {"test_AB": method_predictions(model, obs, frame), "test_C": method_predictions(model, obs, frame.iloc[0:0])}
    frames = {"test_AB": frame, "test_C": frame.iloc[0:0]}
    out = build_forecasts_frame(frames, preds, PROV)
    path = tmp_path / "forecasts.csv"
    write_safe_csv(out, path)
    back = pd.read_csv(path)
    g = back[back["method"] == GBM_NAME]
    assert len(g) == 4
    assert (g["prediction_raw_preclip"] < 0).all()
    assert np.allclose(g["prediction"], np.maximum(0.0, g["prediction_raw_preclip"]))
    assert (g["prediction"] == 0.0).all()
    assert g["clipped"].all()
    # baselines are untouched by clipping and carry raw == prediction
    b = back[back["method"] != GBM_NAME].dropna(subset=["prediction"])
    assert (b["prediction"] == b["prediction_raw_preclip"]).all() and not b["clipped"].any()


def test_forecasts_frame_columns_and_timestamps():
    data = _tiny_data()
    obs = ObservationLookup(data)
    frame = build_feature_frame(data, ["A"], [92])
    model = _negative_model()
    preds = {"test_AB": method_predictions(model, obs, frame), "test_C": method_predictions(model, obs, frame.iloc[0:0])}
    out = build_forecasts_frame({"test_AB": frame, "test_C": frame.iloc[0:0]}, preds, PROV)
    for col in ("store", "sku", "target_week", "actual", "origin_utc", "availability_cutoff_utc", "status",
                "model_hash", "protocol_hash", "manifest_hash"):
        assert col in out.columns
    assert (out["origin_utc"] == "2023-10-02T00:00:00+00:00").all()
    assert (out["origin_utc"] == out["availability_cutoff_utc"]).all()


def test_gbm_unavailable_when_plan_missing():
    data = _tiny_data()
    plans = data.plans.copy()
    plans.loc[plans["week_index"] == 93, "available_at"] += pd.Timedelta(days=7)  # late plan
    import dataclasses
    data = dataclasses.replace(data, plans=plans)
    frame = build_feature_frame(data, ["A"], [92, 93])
    p = method_predictions(_negative_model(), ObservationLookup(data), frame)
    assert np.isfinite(p["predictions"][GBM_NAME][0]) and np.isnan(p["predictions"][GBM_NAME][1])


def test_baselines_equal_their_defining_features():
    data = generate(GeneratorSeeds.dev_fixture())
    frame = build_partition(data, "validation_AB")
    base = baseline_predictions(ObservationLookup(data), frame)
    for name, col in (("naive_last_week", "lag1"), ("seasonal_naive", "lag52"), ("moving_average_4", "roll_mean_4")):
        np.testing.assert_array_equal(base[name], frame[col].to_numpy(dtype=float))


def test_future_forecast_refuses_in_range_week_and_carries_caveat():
    data = _tiny_data()
    model = _negative_model()
    with pytest.raises(ValueError):
        future_forecast(model, data, [104], PROV)
    out = future_forecast(model, data, [105], PROV)
    assert len(out) == 1
    assert np.isnan(out["price"].iloc[0]) and np.isnan(out["promo_flag"].iloc[0])  # no fabricated plan
    assert out["lag1"].iloc[0] == 104 % 7
    assert FUTURE_FORECAST_CAVEAT in out["caveat"].iloc[0]
    assert "plan_unavailable" in out["status"].iloc[0]
    assert out["prediction"].iloc[0] == max(0.0, out["prediction_raw_preclip"].iloc[0])


def test_report_escapes_interpolated_text():
    evil = "<script>alert(1)</script>"
    assert "<script>" not in _e(evil)
    metrics = {
        "label": "on this synthetic set only",
        "methods": ["m"],
        "shifted_skus": ["SKU001"],
        "gbm_non_negative_clipping": {"note": evil},
        "groups": {
            "test_overall": {
                "strata": {"store_A": {"own_coverage": {"m": {"eligible_rows": 1, "scored_rows": 1, "unavailable_rows": 0,
                                                             "coverage": 1.0, "metrics": {"N": 1, "MAE": 1.0, "WAPE": 10.0, "signed_bias": 1.0}}},
                                       "common_support": {"eligible_rows": 1, "common_rows": 1,
                                                          "metrics": {"m": {"N": 1, "MAE": 1.0, "WAPE": 10.0, "signed_bias": 1.0}}}}},
                "macro_store_average": {},
            }
        },
    }
    fc = pd.DataFrame({"partition": ["test_AB", "test_C"], "store": ["A", "C"], "sku": ["SKU001"] * 2,
                       "target_week": [92, 92], "method": ["m", "m"], "actual": [1.0, 1.0], "prediction": [2.0, 2.0]})
    page = render_report(metrics, {"status": evil, "configurations": [], "reason": evil}, fc, {evil: evil})
    assert evil not in page
    assert html.escape(evil) in page
