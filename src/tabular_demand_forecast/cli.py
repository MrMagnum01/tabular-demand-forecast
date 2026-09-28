"""Command line: train / evaluate / forecast.

    python -m tabular_demand_forecast.cli train    --out-dir DIR
    python -m tabular_demand_forecast.cli evaluate --out-dir DIR
    python -m tabular_demand_forecast.cli forecast --out-dir DIR --target-week 105

--seeds frozen (default) is the one evaluation dataset and requires
eval/manifest.json plus a matching environment (provenance.py).
--seeds dev_fixture is for smoke runs only; its outputs are labelled
DEVELOPMENT ONLY and are never evaluation results.

train and evaluate refuse to overwrite their outputs in an existing
directory: selection happens once per directory, and the locked test is
scored once per directory.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata as md
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np

from .evaluate import (
    FUTURE_FORECAST_CAVEAT,
    LABEL,
    build_forecasts_frame,
    final_refit,
    future_forecast,
    score_locked_test,
)
from .csv_safety import write_safe_csv
from .features import FEATURE_COLUMNS, build_partition, ObservationLookup
from .generator import GeneratorSeeds, generate
from .model_selection import run_grid_search
from .provenance import ProvenanceError, python_version, sha256_file, verify_run_environment
from .report import markdown_results_table, render_report
from .validation import validate_generated

SEED_CHOICES = {"frozen": GeneratorSeeds.frozen, "dev_fixture": GeneratorSeeds.dev_fixture}
SEED_LABELS = {
    "frozen": "frozen (protocol generator_seeds) - the evaluation dataset",
    "dev_fixture": "dev_fixture - DEVELOPMENT ONLY, never an evaluation result",
}
MODEL_FILE = "model.joblib"
MODEL_META = "model_meta.json"
SELECTION_LOG = "selection_log.json"
METRICS_FILE = "metrics.json"
FORECASTS_FILE = "forecasts.csv"
REPORT_FILE = "report.html"
TABLE_FILE = "results_table.md"


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=_json_default, allow_nan=False) + "\n", encoding="utf-8")


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _env(seeds: str) -> Dict[str, Optional[str]]:
    return verify_run_environment(require_manifest=(seeds == "frozen"))


def _versions() -> Dict[str, str]:
    return {d: md.version(d) for d in ("numpy", "pandas", "scikit-learn", "scipy", "joblib", "matplotlib")}


def _refuse_existing(paths: List[Path]) -> None:
    existing = [str(p) for p in paths if p.exists()]
    if existing:
        raise SystemExit(f"refusing to overwrite existing output(s): {existing}; use a fresh --out-dir")


def _data(seeds: str):
    data = generate(SEED_CHOICES[seeds]())
    validate_generated(data)
    return data


def _load_model(out_dir: Path, seeds: str):
    meta = json.loads((out_dir / MODEL_META).read_text(encoding="utf-8"))
    if meta["seeds"] != seeds:
        raise SystemExit(f"model in {out_dir} was trained with seeds={meta['seeds']}, not {seeds}")
    model_path = out_dir / MODEL_FILE
    digest = sha256_file(model_path)
    if digest != meta["model_sha256"]:
        raise SystemExit(f"{model_path} sha256 {digest} does not match {MODEL_META}; refusing to load")
    # Only a file this repo's own `train` wrote, verified by hash above.
    return joblib.load(model_path), meta


def cmd_train(args) -> int:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _refuse_existing([out / MODEL_FILE, out / MODEL_META, out / SELECTION_LOG])
    env = _env(args.seeds)
    data = _data(args.seeds)
    obs = ObservationLookup(data)
    log = run_grid_search(build_partition(data, "train_AB"), build_partition(data, "validation_AB"), obs)
    log["seeds"] = SEED_LABELS[args.seeds]
    log["label"] = LABEL
    log["protocol_sha256"] = env["protocol_sha256"]
    log["manifest_sha256"] = env["manifest_sha256"]
    log["created_utc"] = _now()
    if log["status"] != "selected":
        _write_json(out / SELECTION_LOG, log)
        print(f"selection did not produce a model: {log.get('reason')}", file=sys.stderr)
        return 2
    config = log["selected"]["config"]
    model, n_fit = final_refit(data, config)
    joblib.dump(model, out / MODEL_FILE)
    model_sha = sha256_file(out / MODEL_FILE)
    log["final_refit"] = {"partition": "final_refit_AB", "fit_rows": n_fit, "config": config, "model_sha256": model_sha}
    _write_json(out / SELECTION_LOG, log)
    _write_json(out / MODEL_META, {
        "seeds": args.seeds,
        "estimator": "sklearn.ensemble.HistGradientBoostingRegressor",
        "selected_index": log["selected"]["index"],
        "config": config,
        "feature_columns": FEATURE_COLUMNS,
        "final_refit_rows": n_fit,
        "model_file": MODEL_FILE,
        "model_sha256": model_sha,
        "protocol_sha256": env["protocol_sha256"],
        "manifest_sha256": env["manifest_sha256"],
        "python_version": python_version(),
        "package_versions": _versions(),
        "created_utc": _now(),
    })
    print(f"selected config #{log['selected']['index']}: {config} "
          f"({log['n_succeeded']} succeeded, {log['n_failed']} failed); model sha256 {model_sha}")
    return 0


def _provenance(env, meta) -> Dict[str, str]:
    return {
        "model_hash": meta["model_sha256"],
        "protocol_hash": env["protocol_sha256"],
        "manifest_hash": env["manifest_sha256"] or "none (dev_fixture run without manifest)",
    }


def cmd_evaluate(args) -> int:
    out = Path(args.out_dir)
    targets = [out / f for f in (METRICS_FILE, FORECASTS_FILE, REPORT_FILE, TABLE_FILE)]
    _refuse_existing(targets)
    env = _env(args.seeds)
    model, meta = _load_model(out, args.seeds)
    if meta["protocol_sha256"] != env["protocol_sha256"] or meta["manifest_sha256"] != env["manifest_sha256"]:
        raise SystemExit("model was trained under a different protocol/manifest hash; refusing to score")
    data = _data(args.seeds)
    shift = data.truth_meta["shift_sku_magnitude"]
    metrics, frames, preds = score_locked_test(model, data, list(shift.keys()))
    prov = _provenance(env, meta)
    metrics.update({
        "seeds": SEED_LABELS[args.seeds],
        "selected_config": meta["config"],
        "selected_index": meta["selected_index"],
        "shift_sku_magnitude": shift,
        "realized_missing_fraction": data.truth_meta["realized_missing_fraction"],
        "provenance": prov,
        "created_utc": _now(),
    })
    forecasts = build_forecasts_frame(frames, preds, prov)
    selection_log = json.loads((out / SELECTION_LOG).read_text(encoding="utf-8"))
    report_meta = {
        "seeds": SEED_LABELS[args.seeds],
        "protocol sha256": prov["protocol_hash"],
        "manifest sha256": prov["manifest_hash"],
        "model sha256": prov["model_hash"],
        "selected config": json.dumps(meta["config"]),
        "realized missing fraction (truth_meta)": f"{data.truth_meta['realized_missing_fraction']:.6f}",
        "planted shift SKUs (magnitude)": json.dumps(shift),
        "python": python_version(),
        "packages": json.dumps(_versions()),
    }
    _write_json(out / METRICS_FILE, metrics)
    write_safe_csv(forecasts, out / FORECASTS_FILE)
    (out / REPORT_FILE).write_text(render_report(metrics, selection_log, forecasts, report_meta), encoding="utf-8")
    (out / TABLE_FILE).write_text(markdown_results_table(metrics), encoding="utf-8")
    print(f"locked test scored once; wrote {', '.join(str(t) for t in targets)}")
    return 0


def cmd_forecast(args) -> int:
    out = Path(args.out_dir)
    env = _env(args.seeds)
    model, meta = _load_model(out, args.seeds)
    data = _data(args.seeds)
    weeks = args.target_week or [data.n_weeks + 1]
    frame = future_forecast(model, data, weeks, _provenance(env, meta))
    path = Path(args.output) if args.output else out / f"future_forecast_week{'_'.join(str(w) for w in weeks)}.csv"
    write_safe_csv(frame, path)
    print(f"wrote {len(frame)} forecast rows to {path}")
    print(f"CAVEAT: {FUTURE_FORECAST_CAVEAT}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="tabular_demand_forecast.cli", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn in (("train", cmd_train), ("evaluate", cmd_evaluate), ("forecast", cmd_forecast)):
        p = sub.add_parser(name)
        p.add_argument("--out-dir", required=True, help="directory for model and result artifacts")
        p.add_argument("--seeds", choices=sorted(SEED_CHOICES), default="frozen")
        if name == "forecast":
            p.add_argument("--target-week", type=int, action="append",
                           help="future target week index > 104 (repeatable; default 105)")
            p.add_argument("--output", help="CSV path (default: OUT_DIR/future_forecast_week<N>.csv)")
        p.set_defaults(func=fn)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ProvenanceError as exc:
        print(f"PROVENANCE CHECK FAILED: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
