"""Input validation for observed data before any fit/score, per
eval/protocol.v2.json missing_data_and_input_validation_policy.
ingestion_time_deferred_to_step_2. Fails clearly (ValueError) on:
duplicate keys, non-numeric targets, negative targets, non-finite
targets other than the documented NaN-for-missing representation, and
unknown store/sku identifiers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .generator import GeneratedData, sku_ids
from .protocol import load_protocol


class InputValidationError(ValueError):
    pass


def validate_observed(observed: pd.DataFrame, mask: pd.DataFrame | None = None) -> None:
    p = load_protocol()
    known_stores = set(p["entities"]["stores"])
    known_skus = set(sku_ids(p["entities"]["n_skus"]))
    keys = ["store", "sku", "week_index"]
    for col in keys + ["y_obs"]:
        if col not in observed.columns:
            raise InputValidationError(f"observed table missing column {col!r}")
    dup = observed.duplicated(subset=keys, keep=False)
    if dup.any():
        raise InputValidationError(f"duplicate (store, sku, week_index) keys: {int(dup.sum())} rows")
    unknown_stores = set(observed["store"]) - known_stores
    if unknown_stores:
        raise InputValidationError(f"unknown store identifiers: {sorted(map(str, unknown_stores))}")
    unknown_skus = set(observed["sku"]) - known_skus
    if unknown_skus:
        raise InputValidationError(f"unknown sku identifiers: {sorted(map(str, unknown_skus))}")
    y = pd.to_numeric(observed["y_obs"], errors="coerce")
    malformed = y.isna() & observed["y_obs"].notna()
    if malformed.any():
        raise InputValidationError(f"malformed (non-numeric) target values: {int(malformed.sum())} rows")
    values = y.to_numpy(dtype=float)
    if np.any(np.isinf(values)):
        raise InputValidationError("non-finite (inf) observed target values")
    if np.any(values[np.isfinite(values)] < 0):
        raise InputValidationError("negative observed target values")
    if mask is not None:
        merged = observed[keys].assign(is_nan=np.isnan(values)).merge(mask, on=keys, how="left", validate="one_to_one")
        if merged["observed"].isna().any():
            raise InputValidationError("observed rows without a mask entry")
        bad = merged["is_nan"].to_numpy() == merged["observed"].to_numpy(dtype=bool)
        if bad.any():
            raise InputValidationError(
                f"NaN target not matching the documented missing-row mask: {int(bad.sum())} rows"
            )


def validate_generated(data: GeneratedData) -> None:
    validate_observed(data.observed, data.mask)
