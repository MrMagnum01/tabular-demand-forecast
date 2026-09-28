"""Final refit, the single locked-test scoring pass, and forecast frames.

Per eval/protocol.v2.json:
* partitions.refit_policy: the selected configuration is refit exactly once
  on final_refit_AB (A/B target weeks 53-91); nothing is re-selected after.
* metrics.reporting_rule: the test period (weeks 92-104) is scored once --
  test_AB and test_C separately, test_overall as a secondary summary -- for
  the three baselines and the refit GBM, strata overall / per_store /
  shifted_skus_only, each with own-coverage counts AND a separately
  reported common-support block.
* GBM predictions are clipped max(0, raw); both values are kept in the
  per-row output.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .baselines import BASELINE_NAMES, baseline_predictions
from .calendar_utils import origin as week_origin
from .features import ObservationLookup, build_feature_frame, build_partition
from .generator import GeneratedData
from .metrics import macro_store_average, stratum_report
from .model_selection import GBM_NAME, fit_model, make_estimator, predict_gbm

METHODS: Tuple[str, ...] = BASELINE_NAMES + (GBM_NAME,)
TEST_PARTITIONS = ("test_AB", "test_C")
LABEL = "on this synthetic set only"
FUTURE_FORECAST_CAVEAT = (
    "Any forecast issued for a week whose outcome does not yet exist carries no accuracy metric "
    "until that outcome is observed (protocol outputs_deferred_to_later_steps.real_future_forecast_caveat)."
)
NO_PLAN_NOTE = (
    "No price/promo plan exists for this week, so price and promo_flag are NaN and the model routes "
    "them via HistGradientBoostingRegressor native missing-value handling. Under the frozen protocol's "
    "plan_missing_policy such a row would be UNAVAILABLE for any scored set; this forecast is outside "
    "the protocol's scored contract."
)


def final_refit(data: GeneratedData, config: Mapping[str, object], estimator_factory=make_estimator):
    frame = build_partition(data, "final_refit_AB")
    model, n_fit = fit_model(frame, config, estimator_factory)
    return model, n_fit


def utc_iso(week: int) -> str:
    return pd.Timestamp(week_origin(int(week))).tz_localize("UTC").isoformat()


def method_predictions(model, obs: ObservationLookup, frame: pd.DataFrame) -> Dict[str, object]:
    """Per-method prediction arrays (NaN = unavailable) plus GBM raw/clipped."""
    preds: Dict[str, np.ndarray] = dict(baseline_predictions(obs, frame))
    g = predict_gbm(model, frame)
    gbm = np.where(g["available"], g["clipped"], np.nan)
    preds[GBM_NAME] = gbm
    return {
        "predictions": preds,
        "gbm_raw": np.where(g["available"], g["raw"], np.nan),
        "gbm_clipped": gbm,
    }


def _strata_masks(frame: pd.DataFrame, shift_skus: Sequence[str]) -> Dict[str, np.ndarray]:
    masks = {"overall": np.ones(len(frame), dtype=bool)}
    for store in sorted(frame["store"].unique()):
        masks[f"store_{store}"] = (frame["store"] == store).to_numpy()
    masks["shifted_skus_only"] = frame["sku"].isin(list(shift_skus)).to_numpy()
    return masks


def score_locked_test(model, data: GeneratedData, shift_skus: Sequence[str]) -> Tuple[Dict[str, object], Dict[str, pd.DataFrame], Dict[str, Dict[str, object]]]:
    """The single test scoring pass. Returns (metrics, frames, predictions)."""
    obs = ObservationLookup(data)
    frames = {name: build_partition(data, name) for name in TEST_PARTITIONS}
    frames["test_overall"] = pd.concat([frames["test_AB"], frames["test_C"]], ignore_index=True)
    preds = {name: method_predictions(model, obs, frame) for name, frame in frames.items()}

    groups: Dict[str, object] = {}
    for name, frame in frames.items():
        actual = frame["y_obs"].to_numpy(dtype=float)
        strata = {
            s: stratum_report(actual, preds[name]["predictions"], mask)
            for s, mask in _strata_masks(frame, shift_skus).items()
        }
        group: Dict[str, object] = {
            "rows": int(len(frame)),
            "target_weeks": [int(frame["target_week"].min()), int(frame["target_week"].max())],
            "stores": sorted(frame["store"].unique().tolist()),
            "strata": strata,
        }
        if name == "test_overall":
            group["role"] = "secondary summary (test_AB + test_C); primary reporting is test_AB and test_C separately"
            per_store = {k: v for k, v in strata.items() if k.startswith("store_")}
            group["macro_store_average"] = {m: macro_store_average(per_store, m) for m in METHODS}
        groups[name] = group

    raw_all = preds["test_overall"]["gbm_raw"]
    finite_raw = raw_all[np.isfinite(raw_all)]
    metrics = {
        "label": LABEL,
        "methods": list(METHODS),
        "shifted_skus": list(shift_skus),
        "groups": groups,
        "gbm_non_negative_clipping": {
            "form": "pred = max(0, raw_prediction)",
            "rows_with_raw_prediction": int(finite_raw.size),
            "rows_with_negative_raw": int(np.sum(finite_raw < 0)),
            "min_raw_prediction": float(finite_raw.min()) if finite_raw.size else None,
        },
    }
    return metrics, frames, preds


def build_forecasts_frame(
    frames: Mapping[str, pd.DataFrame],
    preds: Mapping[str, Mapping[str, object]],
    provenance: Mapping[str, str],
) -> pd.DataFrame:
    """Long format: one row per (partition, store, sku, target_week, method)."""
    parts: List[pd.DataFrame] = []
    for name in TEST_PARTITIONS:
        frame = frames[name]
        p = preds[name]
        actual = frame["y_obs"].to_numpy(dtype=float)
        target_ok = np.isfinite(actual)
        origin_iso = [utc_iso(t) for t in frame["target_week"]]
        for m in METHODS:
            pred = np.asarray(p["predictions"][m], dtype=float)
            raw = np.asarray(p["gbm_raw"], dtype=float) if m == GBM_NAME else pred
            pred_ok = np.isfinite(pred)
            unavailable_reason = "unavailable_plan" if m == GBM_NAME else "unavailable_history"
            status = []
            for po, to in zip(pred_ok, target_ok):
                reasons = []
                if not po:
                    reasons.append(unavailable_reason)
                if not to:
                    reasons.append("target_missing")
                status.append("scored" if not reasons else "+".join(reasons))
            parts.append(
                pd.DataFrame(
                    {
                        "partition": name,
                        "store": frame["store"].to_numpy(),
                        "sku": frame["sku"].to_numpy(),
                        "target_week": frame["target_week"].to_numpy(dtype=np.int64),
                        "method": m,
                        "actual": actual,
                        "prediction": pred,
                        "prediction_raw_preclip": raw,
                        "clipped": (m == GBM_NAME) & np.isfinite(raw) & (raw < 0),
                        "status": status,
                        "origin_utc": origin_iso,
                        "availability_cutoff_utc": origin_iso,
                        "model_hash": provenance["model_hash"] if m == GBM_NAME else "N/A (deterministic baseline; no fitted model)",
                        "protocol_hash": provenance["protocol_hash"],
                        "manifest_hash": provenance["manifest_hash"],
                    }
                )
            )
    return pd.concat(parts, ignore_index=True)


def future_forecast(
    model,
    data: GeneratedData,
    target_weeks: Sequence[int],
    provenance: Mapping[str, str],
) -> pd.DataFrame:
    """GBM forecast for target weeks beyond the generated grid. Features come
    from the existing observed history only; no plan is fabricated."""
    for t in target_weeks:
        if int(t) <= data.n_weeks:
            raise ValueError(f"target week {t} is inside the frozen range 1..{data.n_weeks}; forecast is for future weeks only")
    frame = build_feature_frame(data, data.stores, [int(t) for t in target_weeks])
    g = predict_gbm(model, frame)
    plan_ok = g["available"]
    origin_iso = [utc_iso(t) for t in frame["target_week"]]
    return pd.DataFrame(
        {
            "store": frame["store"].to_numpy(),
            "sku": frame["sku"].to_numpy(),
            "target_week": frame["target_week"].to_numpy(dtype=np.int64),
            "method": GBM_NAME,
            "prediction": g["clipped"],
            "prediction_raw_preclip": g["raw"],
            "clipped": g["raw"] < 0,
            "lag1": frame["lag1"].to_numpy(dtype=float),
            "lag52": frame["lag52"].to_numpy(dtype=float),
            "roll_mean_4": frame["roll_mean_4"].to_numpy(dtype=float),
            "price": frame["price"].to_numpy(dtype=float),
            "promo_flag": frame["promo_flag"].to_numpy(dtype=float),
            "status": np.where(plan_ok, "forecast_no_outcome_yet", "forecast_no_outcome_yet+plan_unavailable_native_nan_routing"),
            "origin_utc": origin_iso,
            "availability_cutoff_utc": origin_iso,
            "model_hash": provenance["model_hash"],
            "protocol_hash": provenance["protocol_hash"],
            "manifest_hash": provenance["manifest_hash"],
            "caveat": np.where(plan_ok, FUTURE_FORECAST_CAVEAT, FUTURE_FORECAST_CAVEAT + " " + NO_PLAN_NOTE),
        }
    )
