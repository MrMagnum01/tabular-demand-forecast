"""Write eval/manifest.v2.json -- the SUCCESSOR run manifest that binds the
evaluator source as corrected by the 2026-09-28 Astra round-2 HOLD fix
(src/tabular_demand_forecast/{metrics,provenance,evaluate,cli,model_selection}.py).

eval/manifest.json (the step-2 historical record bound at commit 12bdc69,
matching the committed results/) is NEVER modified by this script or
anything else in this fix: it stays byte-unchanged, still describes the
code that produced results/, and a `--seeds frozen` run against it still
correctly refuses now that five evaluator files have changed. This second
manifest is a new, explicitly versioned binding for running the corrected
code (`--manifest eval/manifest.v2.json`); it authorizes no new fit,
selection or score against the frozen dataset, and results/ is untouched.
See eval/ERRATUM-2026-09-28.md for why this manifest exists.

Usage: .venv/bin/python scripts/write_manifest_v2.py
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
    load_manifest,
    parse_lock,
    protocol_sha256,
    python_version,
    rel,
    sha256_file,
)

MANIFEST_V2_PATH = REPO / "eval" / "manifest.v2.json"
REVIEWED_PROTOCOL_SHA256 = "a9604f02542bedc616de8c4c53f479ecbd20eabecd9f25c7a134ebe84e06b9eb"


def main() -> int:
    proto_sha = protocol_sha256()
    if proto_sha != REVIEWED_PROTOCOL_SHA256:
        print(f"ABORT: protocol sha256 {proto_sha} != reviewed {REVIEWED_PROTOCOL_SHA256}", file=sys.stderr)
        return 1
    p = load_protocol()
    expected = p["freeze_and_provenance"]["pre_fit_freeze"]["source_hashes"]["files"]
    now_frozen = frozen_source_hashes_now()
    mismatch = {f: {"protocol": expected[f], "now": now_frozen[f]} for f in expected if expected[f] != now_frozen[f]}
    if mismatch:
        print(f"ABORT: frozen source hash mismatch {mismatch}", file=sys.stderr)
        return 1

    original = load_manifest(MANIFEST_PATH)
    now_evaluator = evaluator_source_hashes_now()
    changed = sorted(f for f, h in now_evaluator.items() if original.get("evaluator_source_hashes", {}).get(f) != h)
    if not changed:
        print("ABORT: no evaluator source differs from eval/manifest.json; a v2 manifest is not needed", file=sys.stderr)
        return 1

    gbm = p["models"]["gradient_boosting"]
    grid = gbm["pre_declared_grid"]
    lock = parse_lock(LOCK_PATH)
    manifest = {
        "manifest_kind": "step-2 run manifest, SUCCESSOR binding v2 (protocol "
                          "freeze_and_provenance.pre_fit_freeze.run_manifest_policy)",
        "supersedes_for_runtime_use": {
            "path": rel(MANIFEST_PATH),
            "sha256": sha256_file(MANIFEST_PATH),
            "note": "the original manifest is retained byte-unchanged as the historical record for "
                    "results/; it is superseded ONLY for gating new runs of the corrected evaluator "
                    "source below, never edited, never used to relabel results/ as conforming.",
        },
        "changed_evaluator_files_since_original": changed,
        "erratum": "eval/ERRATUM-2026-09-28.md",
        "protocol_version": p["protocol_version"],
        "protocol_path": "eval/protocol.v2.json",
        "protocol_sha256": proto_sha,
        "pre_fit_freeze_source_hashes": now_frozen,
        "pre_fit_freeze_source_hashes_match_protocol": True,
        "evaluator_source_hashes": now_evaluator,
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
            "prediction_clipping": "pred = max(0, raw) before any scoring, GBM only, AFTER the raw-finiteness "
                                    "check on available rows (see eval/ERRATUM-2026-09-28.md clause 3)",
            "tie_break": "first configuration in declared grid order achieving the minimum",
            "failure_handling": "each failing configuration recorded {status: failed, reason}; if all fail: 'gradient boosting model failed to select' (no new grid)",
            "refit": "selected configuration refit exactly once on final_refit_AB (A/B target weeks 53-91); no re-selection after",
            "test": "test_AB and test_C (weeks 92-104) scored exactly once per output directory; test_overall secondary",
        },
        "python_version": python_version(),
        "created_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "frozen_dataset_status": "NO fit, selection or score has been executed against the frozen (generator_seeds) "
                                  "dataset under this v2 binding. results/ remains the step-2 evidence, produced "
                                  "and scored under the original manifest; it is not re-scored or relabelled here.",
        "self_hash_note": "This file's own sha256 is never embedded in it; it is quoted externally alongside the commit that adds it.",
    }
    MANIFEST_V2_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {rel(MANIFEST_V2_PATH)}; sha256 {sha256_file(MANIFEST_V2_PATH)}")
    print(f"changed evaluator files bound: {changed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
