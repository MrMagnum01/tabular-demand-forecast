"""Gradient-boosting grid search and selection, per eval/protocol.v2.json
models.gradient_boosting and model_selection.

* Estimator: sklearn HistGradientBoostingRegressor, random_state/early_stopping/
  loss read from the protocol; native NaN pass-through, no imputer/scaler.
* Grid: models.gradient_boosting.pre_declared_grid, iterated max_iter (outer)
  x learning_rate x max_leaf_nodes x min_samples_leaf x l2_regularization
  (inner) in literal list order.
* Selection mask: model_selection.eligible_key_set -- validation_AB rows with
  observed y_obs(t) and finite lag1, lag52, roll_mean_4, price, promo_flag.
  Computed once, used unchanged for every configuration and every baseline.
* Every GBM prediction is clipped pred = max(0, raw) before scoring.
* Criterion: argmin validation MAE; ties -> first in declared grid order.
* A configuration that raises is recorded {status: failed, reason} and the
  loop continues. If all fail: "gradient boosting model failed to select".
"""
from __future__ import annotations

import itertools
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .baselines import BASELINE_NAMES, baseline_predictions
from .features import FEATURE_COLUMNS, ObservationLookup
from .metrics import MethodFailure, coverage, point_metrics
from .protocol import load_protocol

GBM_NAME = "gradient_boosting"
GRID_KEYS = ("max_iter", "learning_rate", "max_leaf_nodes", "min_samples_leaf", "l2_regularization")
ELIGIBLE_KEY_SET_COLUMNS = ("lag1", "lag52", "roll_mean_4", "price", "promo_flag")
GBM_PLAN_REQUIREMENTS = ("price", "promo_flag")
ALL_FAILED_MESSAGE = "gradient boosting model failed to select"
NO_ELIGIBLE_ROWS_MESSAGE = "no eligible rows for model selection"


def _gbm_spec() -> Mapping[str, object]:
    return load_protocol()["models"]["gradient_boosting"]


def declared_grid() -> List[Dict[str, object]]:
    grid = _gbm_spec()["pre_declared_grid"]
    configs = [dict(zip(GRID_KEYS, combo)) for combo in itertools.product(*(grid[k] for k in GRID_KEYS))]
    if len(configs) != grid["total_configurations"]:
        raise ValueError(f"grid size {len(configs)} != protocol total_configurations {grid['total_configurations']}")
    return configs


def make_estimator(config: Mapping[str, object]) -> HistGradientBoostingRegressor:
    spec = _gbm_spec()
    return HistGradientBoostingRegressor(
        loss=spec["objective_loss"],
        random_state=spec["random_state"],
        early_stopping=spec["early_stopping"],
        **dict(config),
    )


def clip_non_negative(raw: np.ndarray) -> np.ndarray:
    """protocol models.gradient_boosting.non_negative_prediction_clipping: pred = max(0, raw)."""
    return np.maximum(0.0, np.asarray(raw, dtype=float))


def feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[FEATURE_COLUMNS].astype(float)


def gbm_available_mask(frame: pd.DataFrame) -> np.ndarray:
    """GBM is unavailable where its plan inputs are missing/late
    (protocol features.price_and_promo.plan_missing_policy)."""
    mask = np.ones(len(frame), dtype=bool)
    for col in GBM_PLAN_REQUIREMENTS:
        mask &= np.isfinite(frame[col].to_numpy(dtype=float))
    return mask


def fit_rows_mask(frame: pd.DataFrame) -> np.ndarray:
    """Rows usable for fitting: observed target and GBM plan inputs available."""
    return np.isfinite(frame["y_obs"].to_numpy(dtype=float)) & gbm_available_mask(frame)


def eligible_key_set_mask(validation_frame: pd.DataFrame) -> np.ndarray:
    mask = np.isfinite(validation_frame["y_obs"].to_numpy(dtype=float))
    for col in ELIGIBLE_KEY_SET_COLUMNS:
        mask &= np.isfinite(validation_frame[col].to_numpy(dtype=float))
    return mask


def predict_gbm(model, frame: pd.DataFrame) -> Dict[str, np.ndarray]:
    """raw and clipped predictions for every row, plus the availability mask.
    Callers treat rows with available == False as unavailable (NaN).

    The raw prediction is checked for finiteness on AVAILABLE rows (per
    gbm_available_mask, the actual input-availability signal) BEFORE any
    clipping. A NaN/Inf/-Inf raw prediction on a row whose plan inputs are
    present is a failed prediction, not missing-input coverage: clipping
    would silently turn -inf into a plausible-looking 0, and passing NaN
    through would make it indistinguishable from a genuinely unavailable
    row. Both are refused here, loudly, before either can happen. Only rows
    where the inputs themselves are unavailable may be reported unavailable.
    """
    if len(frame) == 0:
        empty = np.zeros(0, dtype=float)
        return {"raw": empty, "clipped": empty, "available": np.zeros(0, dtype=bool)}
    raw = np.asarray(model.predict(feature_matrix(frame)), dtype=float)
    available = gbm_available_mask(frame)
    invalid = available & ~np.isfinite(raw)
    if invalid.any():
        raise MethodFailure(
            f"{GBM_NAME} produced {int(invalid.sum())} non-finite raw prediction(s) on "
            f"{int(available.sum())} row(s) with available plan inputs; this is a failed "
            "prediction, not missing-input coverage, and must not be clipped to 0 or masked "
            "as unavailable"
        )
    return {"raw": raw, "clipped": clip_non_negative(raw), "available": available}


def fit_model(frame: pd.DataFrame, config: Mapping[str, object],
              estimator_factory: Callable = make_estimator):
    rows = fit_rows_mask(frame)
    if not rows.any():
        raise ValueError("no fit rows (observed target with available plan inputs)")
    est = estimator_factory(config)
    est.fit(feature_matrix(frame.loc[rows]), frame.loc[rows, "y_obs"].to_numpy(dtype=float))
    return est, int(rows.sum())


def run_grid_search(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    obs: ObservationLookup,
    grid: Optional[Sequence[Mapping[str, object]]] = None,
    estimator_factory: Callable = make_estimator,
) -> Dict[str, object]:
    """Fit every configuration on train, score on the fixed eligible_key_set.

    Returns a JSON-serialisable log: every configuration's outcome, the
    baselines scored on the same mask, and the selection (or refusal)."""
    grid = declared_grid() if grid is None else [dict(c) for c in grid]
    mask = eligible_key_set_mask(validation_frame)
    actual = validation_frame["y_obs"].to_numpy(dtype=float)
    observed = np.isfinite(actual)
    log: Dict[str, object] = {
        "criterion": "minimum validation MAE on the fixed eligible_key_set; ties -> first in declared grid order",
        "eligible_key_set": {
            "definition": "validation_AB rows with observed y_obs(t) and finite " + ", ".join(ELIGIBLE_KEY_SET_COLUMNS),
            "validation_rows": int(len(validation_frame)),
            "validation_rows_with_observed_target": int(observed.sum()),
            "eligible_key_set_rows": int(mask.sum()),
        },
        "grid_size": len(grid),
        "configurations": [],
    }
    if int(mask.sum()) == 0:
        log["status"] = "refused"
        log["reason"] = NO_ELIGIBLE_ROWS_MESSAGE
        log["selected"] = None
        return log

    base = baseline_predictions(obs, validation_frame)
    log["baselines_on_eligible_key_set"] = {}
    for name in BASELINE_NAMES:
        p = base[name]
        own_scored = observed & np.isfinite(p)
        log["baselines_on_eligible_key_set"][name] = {
            "metrics": point_metrics(p[mask], actual[mask]),
            "own_coverage_over_observed_validation_rows": coverage(observed, own_scored),
        }

    fit_rows = fit_rows_mask(train_frame)
    configs_out: List[Dict[str, object]] = []
    best_idx: Optional[int] = None
    best_mae = float("inf")
    for idx, cfg in enumerate(grid):
        entry: Dict[str, object] = {"index": idx, "config": dict(cfg)}
        try:
            est, n_fit = fit_model(train_frame, cfg, estimator_factory)
            pred = predict_gbm(est, validation_frame.loc[mask])
            if not np.all(np.isfinite(pred["raw"])):
                raise ValueError("non-finite raw predictions on eligible_key_set")
            if not np.all(pred["available"]):
                raise ValueError("GBM unavailable on an eligible_key_set row (plan inputs missing)")
            m = point_metrics(pred["clipped"], actual[mask])
            entry.update(
                status="succeeded",
                fit_rows=n_fit,
                validation_MAE=m["MAE"],
                validation_metrics=m,
                n_raw_predictions_clipped=int(np.sum(pred["raw"] < 0)),
            )
            if m["MAE"] < best_mae:  # strict: first in grid order wins ties
                best_mae = float(m["MAE"])
                best_idx = idx
        except Exception as exc:  # noqa: BLE001 -- every failure is recorded, never dropped
            entry.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
        configs_out.append(entry)
    log["configurations"] = configs_out
    log["n_succeeded"] = sum(1 for c in configs_out if c["status"] == "succeeded")
    log["n_failed"] = sum(1 for c in configs_out if c["status"] == "failed")
    log["train_fit_rows"] = int(fit_rows.sum())
    if best_idx is None:
        log["status"] = "all_failed"
        log["reason"] = ALL_FAILED_MESSAGE
        log["selected"] = None
    else:
        ties = [c["index"] for c in configs_out if c["status"] == "succeeded" and c["validation_MAE"] == best_mae]
        log["status"] = "selected"
        log["selected"] = {
            "index": best_idx,
            "config": dict(grid[best_idx]),
            "validation_MAE": best_mae,
            "tied_indices": ties,
        }
    return log
