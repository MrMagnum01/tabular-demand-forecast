"""Write eval/manifest.json -- the step-2 run manifest required by
protocol freeze_and_provenance.pre_fit_freeze.run_manifest_policy.

Aborts (non-zero, nothing written) if the protocol sha256 is not the
reviewed value or any frozen source file no longer matches the hash the
protocol records. The manifest never embeds its own hash.

Usage: .venv/bin/python scripts/write_manifest.py
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tabular_demand_forecast.model_selection import (  # noqa: E402
    ELIGIBLE_KEY_SET_COLUMNS,
    GRID_KEYS,
    declared_grid,
)
from tabular_demand_forecast.protocol import load_protocol  # noqa: E402
from tabular_demand_forecast.provenance import (  # noqa: E402
    LOCK_PATH,
    MANIFEST_PATH,
    evaluator_source_hashes_now,
    frozen_source_hashes_now,
    parse_lock,
    protocol_sha256,
    python_version,
    rel,
    sha256_file,
)

REVIEWED_PROTOCOL_SHA256 = "a9604f02542bedc616de8c4c53f479ecbd20eabecd9f25c7a134ebe84e06b9eb"


def main() -> int:
    proto_sha = protocol_sha256()
    if proto_sha != REVIEWED_PROTOCOL_SHA256:
        print(f"ABORT: protocol sha256 {proto_sha} != reviewed {REVIEWED_PROTOCOL_SHA256}", file=sys.stderr)
        return 1
    p = load_protocol()
    expected = p["freeze_and_provenance"]["pre_fit_freeze"]["source_hashes"]["files"]
    now = frozen_source_hashes_now()
    mismatch = {f: {"protocol": expected[f], "now": now[f]} for f in expected if expected[f] != now[f]}
    if mismatch:
        print(f"ABORT: frozen source hash mismatch {mismatch}", file=sys.stderr)
        return 1
    gbm = p["models"]["gradient_boosting"]
    grid = gbm["pre_declared_grid"]
    lock = parse_lock(LOCK_PATH)
    manifest = {
        "manifest_kind": "step-2 run manifest (protocol freeze_and_provenance.pre_fit_freeze.run_manifest_policy)",
        "protocol_version": p["protocol_version"],
        "protocol_path": "eval/protocol.v2.json",
        "protocol_sha256": proto_sha,
        "pre_fit_freeze_source_hashes": now,
        "pre_fit_freeze_source_hashes_match_protocol": True,
        "evaluator_source_hashes": evaluator_source_hashes_now(),
        "dependency_lock": {
            "path": rel(LOCK_PATH),
            "sha256": sha256_file(LOCK_PATH),
            "format": "pip hash-checking requirements (name==version + --hash=sha256 of the exact wheel, verified against installed RECORD)",
            "packages": [{"name": e["name"], "version": e["version"], "sha256": e["sha256"]} for e in lock],
        },
        "model_selection_rules": {
            "note": "Documentation only; the binding implementation is model_selection.py, bound by evaluator_source_hashes above.",
            "estimator": gbm["estimator"],
            "fixed_params": {"random_state": gbm["random_state"], "early_stopping": gbm["early_stopping"], "loss": gbm["objective_loss"]},
            "grid": {k: grid[k] for k in GRID_KEYS},
            "grid_order": "max_iter (outer) x learning_rate x max_leaf_nodes x min_samples_leaf x l2_regularization (inner), literal list order",
            "n_configurations": len(declared_grid()),
            "fit_partition": "train_AB (A/B target weeks 53-78), rows with observed target and available price/promo_flag",
            "criterion": "minimum validation MAE on validation_AB (A/B target weeks 79-91) over the fixed eligible_key_set",
            "eligible_key_set": "validation_AB rows with observed y_obs(t) and finite " + ", ".join(ELIGIBLE_KEY_SET_COLUMNS) + "; computed once, identical for all 36 configurations and all 3 baselines",
            "prediction_clipping": "pred = max(0, raw) before any scoring, GBM only",
            "tie_break": "first configuration in declared grid order achieving the minimum",
            "failure_handling": "each failing configuration recorded {status: failed, reason}; if all fail: 'gradient boosting model failed to select' (no new grid)",
            "refit": "selected configuration refit exactly once on final_refit_AB (A/B target weeks 53-91); no re-selection after",
            "test": "test_AB and test_C (weeks 92-104) scored exactly once per output directory; test_overall secondary",
        },
        "python_version": python_version(),
        "created_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "frozen_dataset_status": "NO fit, selection or score has been executed against the frozen (generator_seeds) dataset as of this manifest. Only dev_fixture_seeds and hand-built fixtures have been run.",
        "self_hash_note": "This file's own sha256 is never embedded in it; it is quoted externally alongside the commit that adds it.",
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {rel(MANIFEST_PATH)}; sha256 {sha256_file(MANIFEST_PATH)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
