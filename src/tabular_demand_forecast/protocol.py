"""Loader for the single frozen source of truth, eval/protocol.v2.json.

Every numeric parameter used by the generator or feature code (feature
list, lag K-values, rolling-window sizes/thresholds, planted-effect
distribution parameters) is read from here, never duplicated as a
separate hardcoded literal, and validated against this file where
practical (see features.FEATURE_COLUMNS). The functional forms that
combine those parameters -- e.g. the seasonal sine formula, the
negative-binomial demand construction -- remain Python code, since this
loader has no expression evaluator; those forms are documented in this
file's prose fields and bound by eval/protocol.v2.json's
freeze_and_provenance.pre_fit_freeze.source_hashes, not derived from JSON
at runtime.
"""
from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / "eval" / "protocol.v2.json"


@functools.lru_cache(maxsize=1)
def load_protocol() -> Dict[str, Any]:
    with open(PROTOCOL_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)
