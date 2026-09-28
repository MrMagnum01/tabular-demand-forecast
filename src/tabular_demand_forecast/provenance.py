"""Run-manifest and environment checks (protocol
freeze_and_provenance.pre_fit_freeze.run_manifest_policy).

A frozen-seed run refuses to start unless eval/manifest.json exists and
the protocol sha256, the four frozen source hashes, every evaluator source
hash, the dependency-lock hash and the installed package versions all
match it. Drift fails loudly (ProvenanceError); nothing is auto-updated.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

from .protocol import PROTOCOL_PATH, REPO_ROOT, load_protocol

MANIFEST_PATH = REPO_ROOT / "eval" / "manifest.json"
LOCK_PATH = REPO_ROOT / "eval" / "dependency-lock.txt"
PACKAGE_DIR = "src/tabular_demand_forecast"
EVALUATOR_SOURCE_FILES = [
    f"{PACKAGE_DIR}/{name}"
    for name in (
        "__init__.py",
        "csv_safety.py",
        "metrics.py",
        "baselines.py",
        "model_selection.py",
        "validation.py",
        "provenance.py",
        "evaluate.py",
        "report.py",
        "cli.py",
    )
]
# The installer itself is locked but not a runtime dependency; it is not
# checked against the installed environment at run time.
RUNTIME_CHECK_EXCLUDE = {"pip"}


class ProvenanceError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def protocol_sha256() -> str:
    return sha256_file(PROTOCOL_PATH)


def frozen_source_hashes_now() -> Dict[str, str]:
    files = load_protocol()["freeze_and_provenance"]["pre_fit_freeze"]["source_hashes"]["files"]
    return {f: sha256_file(REPO_ROOT / f) for f in files}


def check_frozen_sources() -> Dict[str, str]:
    expected = load_protocol()["freeze_and_provenance"]["pre_fit_freeze"]["source_hashes"]["files"]
    now = frozen_source_hashes_now()
    bad = {f: (expected[f], now[f]) for f in expected if expected[f] != now[f]}
    if bad:
        raise ProvenanceError(f"frozen source file(s) changed since protocol freeze: {bad}")
    return now


def evaluator_source_hashes_now() -> Dict[str, str]:
    return {f: sha256_file(REPO_ROOT / f) for f in EVALUATOR_SOURCE_FILES}


_LOCK_LINE = re.compile(r"^([A-Za-z0-9_.\-]+)==([^\s\\]+)")
_HASH = re.compile(r"--hash=sha256:([0-9a-f]{64})")


def parse_lock(path: Path = LOCK_PATH) -> List[Dict[str, object]]:
    """Parse the pip hash-checking lock format: `name==ver \\` followed by
    one or more `--hash=sha256:<hex>` continuation lines."""
    entries: List[Dict[str, object]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        m = _LOCK_LINE.match(s)
        if m:
            entries.append({"name": m.group(1), "version": m.group(2), "sha256": []})
        for h in _HASH.findall(s):
            if not entries:
                raise ProvenanceError("hash line before any requirement in lock file")
            entries[-1]["sha256"].append(h)
    for e in entries:
        if not e["sha256"]:
            raise ProvenanceError(f"lock entry {e['name']} has no sha256")
    return entries


def check_installed_against_lock(entries: List[Dict[str, object]]) -> None:
    drift = []
    for e in entries:
        if e["name"].lower() in RUNTIME_CHECK_EXCLUDE:
            continue
        try:
            have = md.version(e["name"])
        except md.PackageNotFoundError:
            have = None
        if have != e["version"]:
            drift.append(f"{e['name']}: locked {e['version']}, installed {have}")
    pinned = load_protocol()["freeze_and_provenance"]["pre_fit_freeze"]["dependency_versions"]
    for dist, key in (("numpy", "numpy"), ("pandas", "pandas"), ("pytest", "pytest"), ("scikit-learn", "scikit_learn")):
        want = str(pinned[key]).split()[0]
        try:
            have = md.version(dist)
        except md.PackageNotFoundError:
            have = None
        if have != want:
            drift.append(f"{dist}: protocol pre_fit_freeze pins {want}, installed {have}")
    if drift:
        raise ProvenanceError("installed environment drifts from the frozen lock/protocol: " + "; ".join(drift))


def python_version() -> str:
    return ".".join(str(x) for x in sys.version_info[:3])


def load_manifest(path: Path = MANIFEST_PATH) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def verify_run_environment(require_manifest: bool) -> Dict[str, Optional[str]]:
    """Returns {protocol_sha256, manifest_sha256}. Raises ProvenanceError on
    any mismatch. With require_manifest=False (dev-fixture runs only) a
    missing manifest is tolerated and recorded as None."""
    proto = protocol_sha256()
    check_frozen_sources()
    if not MANIFEST_PATH.exists():
        if require_manifest:
            raise ProvenanceError(
                f"{rel(MANIFEST_PATH)} does not exist: a frozen-seed fit/score may not run before the manifest is committed"
            )
        return {"protocol_sha256": proto, "manifest_sha256": None}
    manifest = load_manifest()
    problems = []
    if manifest.get("protocol_sha256") != proto:
        problems.append("protocol sha256 differs from manifest")
    for f, h in frozen_source_hashes_now().items():
        if manifest.get("pre_fit_freeze_source_hashes", {}).get(f) != h:
            problems.append(f"frozen source {f} differs from manifest")
    recorded = manifest.get("evaluator_source_hashes", {})
    now = evaluator_source_hashes_now()
    if set(recorded) != set(now):
        problems.append(f"evaluator file set differs from manifest: {sorted(set(recorded) ^ set(now))}")
    for f, h in now.items():
        if recorded.get(f) != h:
            problems.append(f"evaluator source {f} differs from manifest")
    lock = manifest.get("dependency_lock", {})
    if lock.get("sha256") != sha256_file(LOCK_PATH):
        problems.append("dependency lock sha256 differs from manifest")
    mv = str(manifest.get("python_version", ""))
    if mv.split(".")[:2] != python_version().split(".")[:2]:
        problems.append(f"python {python_version()} vs manifest {mv} (major.minor must match)")
    if problems:
        raise ProvenanceError("run environment does not match eval/manifest.json: " + "; ".join(problems))
    check_installed_against_lock(parse_lock(LOCK_PATH))
    return {"protocol_sha256": proto, "manifest_sha256": sha256_file(MANIFEST_PATH)}
