"""Loader for the single frozen source of truth, eval/protocol.json.

No value from protocol.json is duplicated as a hardcoded literal elsewhere
in this package; every constant used by the generator or feature code is
read from here, so the frozen file and the code cannot silently drift.
"""
from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / "eval" / "protocol.json"


@functools.lru_cache(maxsize=1)
def load_protocol() -> Dict[str, Any]:
    with open(PROTOCOL_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)
