"""Ingestion validation (protocol missing_data_and_input_validation_policy)
and provenance-gate tests, on hand-built tables only."""
import copy

import numpy as np
import pandas as pd
import pytest

from tabular_demand_forecast import provenance
from tabular_demand_forecast.validation import InputValidationError, validate_observed


def _obs(**override):
    rows = [
        {"store": "A", "sku": "SKU001", "week_index": 1, "y_obs": 3.0},
        {"store": "A", "sku": "SKU001", "week_index": 2, "y_obs": np.nan},
    ]
    df = pd.DataFrame(rows)
    for k, v in override.items():
        df.loc[0, k] = v
    return df


def test_valid_table_passes():
    mask = pd.DataFrame({"store": ["A", "A"], "sku": ["SKU001"] * 2, "week_index": [1, 2], "observed": [True, False]})
    validate_observed(_obs(), mask)


@pytest.mark.parametrize(
    "frame, fragment",
    [
        (pd.concat([_obs(), _obs().iloc[[0]]]), "duplicate"),
        (_obs(y_obs=-1.0), "negative"),
        (_obs(y_obs=np.inf), "inf"),
        (_obs(store="Z"), "unknown store"),
        (_obs(sku="SKU999"), "unknown sku"),
    ],
)
def test_malformed_inputs_fail_clearly(frame, fragment):
    with pytest.raises(InputValidationError, match=fragment):
        validate_observed(frame)


def test_non_numeric_target_fails():
    df = _obs().astype({"y_obs": object})
    df.loc[0, "y_obs"] = "twelve"
    with pytest.raises(InputValidationError, match="non-numeric"):
        validate_observed(df)


def test_nan_not_matching_mask_fails():
    mask = pd.DataFrame({"store": ["A", "A"], "sku": ["SKU001"] * 2, "week_index": [1, 2], "observed": [True, True]})
    with pytest.raises(InputValidationError, match="mask"):
        validate_observed(_obs(), mask)


def test_frozen_run_requires_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "MANIFEST_PATH", tmp_path / "missing.json")
    with pytest.raises(provenance.ProvenanceError, match="manifest"):
        provenance.verify_run_environment(require_manifest=True)
    assert provenance.verify_run_environment(require_manifest=False)["manifest_sha256"] is None


def test_frozen_source_hashes_match_protocol():
    provenance.check_frozen_sources()


def test_null_evaluator_hash_is_rejected_not_treated_as_wildcard(monkeypatch):
    """Astra HOLD clause 2a (2026-09-28-astra-tabular-step2-review.md finding
    2; probe null_evaluator_hash): a manifest whose recorded hash for an
    evaluator source file is null must NOT be accepted as a match for
    whatever that file's current hash happens to be. Every evaluator source
    file's exact hash is required, with no None wildcard."""
    real_manifest = copy.deepcopy(provenance.load_manifest())
    real_manifest["evaluator_source_hashes"]["src/tabular_demand_forecast/evaluate.py"] = None
    monkeypatch.setattr(provenance, "load_manifest", lambda path=provenance.MANIFEST_PATH: real_manifest)
    with pytest.raises(provenance.ProvenanceError, match="evaluator source"):
        provenance.verify_run_environment(require_manifest=True)


def test_parse_lock_format(tmp_path):
    p = tmp_path / "lock.txt"
    p.write_text("# comment\nfoo==1.2 \\\n    --hash=sha256:" + "a" * 64 + "\nbar==3 --hash=sha256:" + "b" * 64 + "\n")
    entries = provenance.parse_lock(p)
    assert entries == [{"name": "foo", "version": "1.2", "sha256": ["a" * 64]},
                       {"name": "bar", "version": "3", "sha256": ["b" * 64]}]
