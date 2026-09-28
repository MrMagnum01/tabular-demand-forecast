"""Metric formulas and coverage reporting, per eval/protocol.v2.json metrics.

MAE          = sum(abs(pred - actual)) / N
signed_bias  = mean(pred - actual)
WAPE         = sum(abs(pred - actual)) / sum(abs(actual)) * 100, unit "%"

Zero/empty denominator (protocol.metrics.zero_or_empty_denominator):
  (1) N == 0                      -> MAE, signed_bias, WAPE all "N/A"
  (2) N > 0, sum(abs(actual)) == 0 -> WAPE alone "N/A"; MAE/bias computed.
No metric is ever reported as 0 in place of an undefined value.
"""
from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np

NA = "N/A"
WAPE_UNIT = "%"


class MethodFailure(RuntimeError):
    """A method produced a non-finite (Inf/-Inf) prediction on a row its own
    inputs declared eligible/available. This is a failed prediction, not
    missing-input coverage, and must never be silently excluded from the
    scored set to make coverage/metrics look better."""


def point_metrics(pred: Sequence[float], actual: Sequence[float]) -> Dict[str, object]:
    pred = np.asarray(pred, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if pred.shape != actual.shape:
        raise ValueError(f"pred shape {pred.shape} != actual shape {actual.shape}")
    if not (np.all(np.isfinite(pred)) and np.all(np.isfinite(actual))):
        raise ValueError(
            "non-finite value passed to point_metrics; unavailable predictions and "
            "missing targets must be excluded (and counted) before scoring"
        )
    n = int(pred.size)
    if n == 0:
        return {
            "N": 0,
            "MAE": NA,
            "signed_bias": NA,
            "WAPE": NA,
            "WAPE_unit": WAPE_UNIT,
            "na_reason": "N == 0 (no scored rows): all three metrics undefined",
        }
    err = pred - actual
    abs_err_sum = float(np.sum(np.abs(err)))
    denom = float(np.sum(np.abs(actual)))
    out: Dict[str, object] = {
        "N": n,
        "MAE": abs_err_sum / n,
        "signed_bias": float(np.mean(err)),
        "WAPE": NA if denom == 0.0 else abs_err_sum / denom * 100.0,
        "WAPE_unit": WAPE_UNIT,
    }
    if denom == 0.0:
        out["na_reason"] = "sum(abs(actual)) == 0: WAPE alone undefined; MAE and signed_bias defined"
    return out


def coverage(eligible: np.ndarray, scored: np.ndarray) -> Dict[str, object]:
    """eligible: rows with an observed target; scored: eligible rows this
    method produced a prediction for (must be a subset of eligible)."""
    eligible = np.asarray(eligible, dtype=bool)
    scored = np.asarray(scored, dtype=bool)
    if np.any(scored & ~eligible):
        raise ValueError("scored rows must be a subset of eligible rows")
    e = int(eligible.sum())
    s = int(scored.sum())
    return {
        "eligible_rows": e,
        "scored_rows": s,
        "unavailable_rows": e - s,
        "coverage": NA if e == 0 else s / e,
    }


def stratum_report(
    actual: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    stratum_mask: np.ndarray,
) -> Dict[str, object]:
    """Own-coverage metrics per method plus a separately reported
    common-support block (rows scored by ALL methods), for one stratum.

    predictions[m] is NaN wherever method m is unavailable (declared missing
    input, e.g. no history or no plan). Any OTHER non-finite value (Inf,
    -Inf) on an eligible row is not a missing-input signal: it means the
    method produced an invalid prediction, and stratum_report raises
    MethodFailure rather than quietly dropping that row out of coverage.
    """
    actual = np.asarray(actual, dtype=float)
    stratum_mask = np.asarray(stratum_mask, dtype=bool)
    eligible = stratum_mask & np.isfinite(actual)
    for m, p in predictions.items():
        p = np.asarray(p, dtype=float)
        invalid = eligible & ~np.isnan(p) & ~np.isfinite(p)
        if invalid.any():
            raise MethodFailure(
                f"method {m!r} produced {int(invalid.sum())} non-finite (e.g. Inf) prediction(s) on "
                f"{int(eligible.sum())} eligible row(s); a non-finite prediction is a failed method "
                "result, not missing-input coverage, and must not be excluded from scoring to improve "
                "its own coverage/metrics"
            )
    available = {m: np.isfinite(np.asarray(p, dtype=float)) for m, p in predictions.items()}

    own: Dict[str, object] = {}
    for m, p in predictions.items():
        scored = eligible & available[m]
        own[m] = {
            **coverage(eligible, scored),
            "metrics": point_metrics(np.asarray(p, dtype=float)[scored], actual[scored]),
        }

    common = eligible.copy()
    for m in predictions:
        common &= available[m]
    common_block = {
        "definition": "rows in this stratum with an observed target that EVERY compared method scored",
        "methods_compared": list(predictions.keys()),
        "eligible_rows": int(eligible.sum()),
        "common_rows": int(common.sum()),
        "metrics": {
            m: point_metrics(np.asarray(p, dtype=float)[common], actual[common])
            for m, p in predictions.items()
        },
    }
    return {
        "rows_in_stratum": int(stratum_mask.sum()),
        "eligible_rows": int(eligible.sum()),
        "own_coverage": own,
        "common_support": common_block,
    }


def macro_store_average(per_store: Mapping[str, Mapping[str, object]], method: str) -> Dict[str, object]:
    """Unweighted mean over stores of each store's own-coverage MAE/WAPE/bias.
    Labelled 'macro store average'; never the row-weighted overall figure."""
    out: Dict[str, object] = {"label": "macro store average (unweighted mean over stores; NOT the row-weighted overall)"}
    for key in ("MAE", "WAPE", "signed_bias"):
        vals = []
        used = []
        for store, rep in per_store.items():
            v = rep["own_coverage"][method]["metrics"][key]
            if v != NA:
                vals.append(float(v))
                used.append(store)
        out[key] = NA if not vals else float(np.mean(vals))
        out[f"{key}_stores_included"] = used
    out["WAPE_unit"] = WAPE_UNIT
    return out
