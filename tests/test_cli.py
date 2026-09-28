"""CLI-level provenance and output-safety tests, using --seeds dev_fixture
(DEVELOPMENT ONLY data) and hand-built model fixtures -- no fit/selection
run and no touch of results/ (the frozen step-2 evidence)."""
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from tabular_demand_forecast import cli, provenance
from tabular_demand_forecast.features import FEATURE_COLUMNS
from tabular_demand_forecast.provenance import sha256_file


def _tiny_model():
    X = pd.DataFrame(np.arange(40 * len(FEATURE_COLUMNS), dtype=float).reshape(40, -1), columns=FEATURE_COLUMNS)
    y = np.arange(40, dtype=float)
    return HistGradientBoostingRegressor(max_iter=5, early_stopping=False, random_state=0).fit(X, y)


@pytest.fixture(autouse=True)
def _no_committed_manifest(tmp_path_factory, monkeypatch):
    """eval/manifest.json is bound (by evaluator_source_hashes) to the
    ORIGINAL evaluator source and is left byte-unchanged by this fix (no new
    manifest/re-selection is authorized here -- that is a separate, future
    step per Astra's HOLD finding 2). So a real run against the patched
    source now correctly refuses against that stale manifest. These tests
    exercise `forecast`'s OWN logic (env/model binding, no-overwrite), which
    is orthogonal to that manifest, so point MANIFEST_PATH at a missing file
    and take the documented 'no manifest -> tolerated' (dev_fixture) path.
    """
    monkeypatch.setattr(provenance, "MANIFEST_PATH", tmp_path_factory.mktemp("no-manifest") / "missing.json")


def _write_model_fixture(out_dir: Path, protocol_sha256, manifest_sha256, seeds="dev_fixture") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / cli.MODEL_FILE
    joblib.dump(_tiny_model(), model_path)
    meta = {
        "seeds": seeds,
        "estimator": "sklearn.ensemble.HistGradientBoostingRegressor",
        "selected_index": 0,
        "config": {},
        "feature_columns": FEATURE_COLUMNS,
        "final_refit_rows": 40,
        "model_file": cli.MODEL_FILE,
        "model_sha256": sha256_file(model_path),
        "protocol_sha256": protocol_sha256,
        "manifest_sha256": manifest_sha256,
        "python_version": provenance.python_version(),
        "package_versions": {},
        "created_utc": "2026-09-28T00:00:00+00:00",
    }
    (out_dir / cli.MODEL_META).write_text(json.dumps(meta), encoding="utf-8")


def test_forecast_refuses_foreign_model_environment_binding(tmp_path):
    """Astra HOLD clause 2b (finding 2; probe foreign_model_forecast_called):
    a model saved under an old protocol/manifest hash must be refused by
    `forecast`, not silently used just because it loads and its own file
    hash matches its own metadata. Mirrors the check `evaluate` already had."""
    out_dir = tmp_path / "run"
    _write_model_fixture(out_dir, protocol_sha256="OLD-PROTOCOL-HASH", manifest_sha256="OLD-MANIFEST-HASH")
    args = SimpleNamespace(out_dir=str(out_dir), seeds="dev_fixture", target_week=[105], output=None)
    with pytest.raises(SystemExit, match="protocol/manifest"):
        cli.cmd_forecast(args)
    # nothing was written
    assert not (out_dir / "future_forecast_week105.csv").exists()


def test_forecast_refuses_to_overwrite_existing_output(tmp_path):
    """Astra HOLD clause 4 (finding 4): forecast must never silently
    overwrite an existing output file, including a prior forecast or any
    other evidence file placed at that path."""
    out_dir = tmp_path / "run"
    env = provenance.verify_run_environment(require_manifest=False)
    _write_model_fixture(out_dir, protocol_sha256=env["protocol_sha256"], manifest_sha256=env["manifest_sha256"])
    target = out_dir / "future_forecast_week105.csv"
    target.write_text("PRE-EXISTING EVIDENCE, DO NOT TOUCH\n", encoding="utf-8")
    args = SimpleNamespace(out_dir=str(out_dir), seeds="dev_fixture", target_week=[105], output=None)
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        cli.cmd_forecast(args)
    assert target.read_text(encoding="utf-8") == "PRE-EXISTING EVIDENCE, DO NOT TOUCH\n"


def test_forecast_happy_path_writes_once_to_a_fresh_path(tmp_path):
    out_dir = tmp_path / "run"
    env = provenance.verify_run_environment(require_manifest=False)
    _write_model_fixture(out_dir, protocol_sha256=env["protocol_sha256"], manifest_sha256=env["manifest_sha256"])
    args = SimpleNamespace(out_dir=str(out_dir), seeds="dev_fixture", target_week=[105], output=None)
    rc = cli.cmd_forecast(args)
    assert rc == 0
    out_path = out_dir / "future_forecast_week105.csv"
    assert out_path.exists()
    frame = pd.read_csv(out_path)
    assert set(frame["status"]) <= {"forecast_no_outcome_yet", "unavailable_plan"}
    # week 105 has no plan anywhere in the dev_fixture data -> every row unavailable_plan, no numeric prediction
    assert (frame["status"] == "unavailable_plan").all()
    assert frame["prediction"].isna().all()
