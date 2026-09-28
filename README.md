# tabular-demand-forecast

Synthetic portfolio demonstration, implemented with AI coding agents; independent review pending. No client data or client work.

**2026-09-28 erratum:** two round-2 review findings were fixed after `results/` was produced and frozen. See
[eval/ERRATUM-2026-09-28.md](eval/ERRATUM-2026-09-28.md). In short: (1) a non-finite raw model prediction on an
available input could previously escape detection as "unavailable" or a clipped 0 instead of failing loudly —
fixed, and (2) the quickstart below now runs against a new **successor** manifest,
[eval/manifest.v2.json](eval/manifest.v2.json), which binds the corrected evaluator source; the original
[eval/manifest.json](eval/manifest.json) is untouched and still describes exactly the code that produced
`results/`. Nothing in `results/` was re-fit, re-scored or relabelled.

Weekly demand forecasting on a fully synthetic, deterministic dataset (3 stores x 30 SKUs x 104 weeks), with a
protocol frozen before any model existed, three fixed baselines, a pre-declared gradient-boosting grid, and a
locked test period scored once.

**Headline, on this synthetic set only:** the selected gradient-boosting model (GBM) beat the seasonal-naive
baseline on the test set in every reported stratum: test_AB, test_C (the store unseen during fitting),
test_overall, each store, and the planted-shift SKUs. This holds on each method's own coverage and on the common
support (rows every method could score). It did **not** beat everything: on store C's two shifted SKUs,
naive_last_week has lower MAE (9.364 vs 11.972 own coverage; 9.364 vs 12.823 common support). GBM also has the
most negative signed bias of the four methods (it under-forecasts, -3.436 units overall). No significance test or
interval is computed, so none of these differences is claimed to be statistically meaningful.

## What is frozen, and in what order

| step | commit | what |
|---|---|---|
| 1 | `889534c` | `eval/protocol.v2.json` (sha256 `a9604f02...e06b9eb`), generator, features, leakage tests |
| 2a | `12bdc69` | `eval/manifest.json`, all evaluator/CLI source, dependency lock. **No fit or score on the frozen data yet** |
| 2b | `28b6687` | `results/`: selection log, refit model, locked-test metrics, forecasts, report |

`eval/manifest.json` records the protocol sha256, the four hash-bound step-1 files, the sha256 of every step-2
source file, and the dependency-lock hash. A `--seeds frozen` run refuses to start unless all of these, plus
the installed package versions, still match. The manifest does not embed its own hash. Its sha256 at `12bdc69` is
`7275e08eaf7ed88067f9a8ce63a597e92f6546b0be3987bbf8e1606ec37878cf`. It is the historical record for `results/`
below and is never edited — including by the 2026-09-28 fixes, which changed five evaluator source files and
therefore can no longer match it (that mismatch is the gate working correctly, not a defect: see
[eval/ERRATUM-2026-09-28.md](eval/ERRATUM-2026-09-28.md)).

Running the corrected code (this head) against the frozen dataset instead binds to the **successor manifest**
[eval/manifest.v2.json](eval/manifest.v2.json), passed explicitly with `--manifest`. It records the same
protocol, the same frozen generator/features/calendar hashes and the same dependency lock as `eval/manifest.json`
— only the five corrected evaluator file hashes differ. It authorizes no new fit, selection or score against
`results/`; `train`/`evaluate`/`forecast` below write into a fresh `runs/local`, never into `results/`.

## Quickstart (clean clone, Linux x86_64, CPython 3.13)

```bash
git clone https://github.com/MrMagnum01/tabular-demand-forecast.git
cd tabular-demand-forecast
python3.13 -m venv .venv
.venv/bin/python -m pip install --require-hashes --no-deps -r eval/dependency-lock.txt
.venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python -m tabular_demand_forecast.cli train    --out-dir runs/local --manifest eval/manifest.v2.json
PYTHONPATH=src .venv/bin/python -m tabular_demand_forecast.cli evaluate --out-dir runs/local --manifest eval/manifest.v2.json
PYTHONPATH=src .venv/bin/python -m tabular_demand_forecast.cli forecast --out-dir runs/local --manifest eval/manifest.v2.json --target-week 105
```

**Legacy reproduction** (the ORIGINAL, now-documented-as-defective code, matching `eval/manifest.json` exactly
with no `--manifest` flag needed): `git checkout e3e7b76` — the last commit before the 2026-09-28 HOLD fixes,
with evaluator source identical to `12bdc69`. See [eval/ERRATUM-2026-09-28.md](eval/ERRATUM-2026-09-28.md) for
the defects that commit still has (round-2 clause 3, and the round-1 clauses fixed at `fdad847`).

* `eval/dependency-lock.txt` pins all 23 installed packages, direct and transitive, each with the sha256 of its
  exact wheel. The hashes are for CPython 3.13 manylinux x86_64 wheels. On another platform, install
  `requirements.txt` instead; the frozen-run environment check will then report any version drift.
* `train` runs the 36-configuration grid on train_AB, selects on validation, and refits once on final_refit_AB.
  It takes about 30 s on 20 cores. `evaluate` scores the locked test. `forecast` writes a week-105 forecast.
* `train` and `evaluate` refuse to overwrite outputs in an existing `--out-dir`. The committed run lives in
  `results/`. Re-running into `runs/local` reproduces that same run: same data, same code, same selection. It is
  not a new selection or a second look at the test period.
* `--seeds dev_fixture` runs the pipeline on the development-only dataset. Those outputs are labelled DEVELOPMENT
  ONLY and are never evaluation results.

## Results, on this synthetic set only

Test period: target weeks 92-104, rolling one-week-ahead, scored once. Stores A/B were used for fitting; store C
was never used for training, validation or selection. Counts per method: eligible = rows with an observed target;
scored = eligible rows the method could predict; unavailable = eligible rows it could not (missing lag history).
WAPE is in %.

Own coverage (each method on the rows it could score):

| partition | stratum | method | eligible | scored | unavailable | MAE | WAPE % | signed bias |
|---|---|---|---:|---:|---:|---:|---:|---:|
| test_AB | overall | naive_last_week | 756 | 732 | 24 | 9.561 | 50.78 | -0.545 |
| test_AB | overall | seasonal_naive | 756 | 722 | 34 | 10.188 | 53.52 | 0.224 |
| test_AB | overall | moving_average_4 | 756 | 748 | 8 | 7.631 | 40.13 | -1.185 |
| test_AB | overall | gradient_boosting | 756 | 756 | 0 | 7.122 | 37.52 | -3.182 |
| test_C | overall | naive_last_week | 378 | 365 | 13 | 11.479 | 48.31 | -0.323 |
| test_C | overall | seasonal_naive | 378 | 366 | 12 | 12.246 | 52.09 | -0.836 |
| test_C | overall | moving_average_4 | 378 | 374 | 4 | 9.475 | 40.14 | -0.608 |
| test_C | overall | gradient_boosting | 378 | 378 | 0 | 8.817 | 37.06 | -3.946 |
| test_overall | overall | naive_last_week | 1134 | 1097 | 37 | 10.200 | 49.82 | -0.471 |
| test_overall | overall | seasonal_naive | 1134 | 1088 | 46 | 10.881 | 52.97 | -0.132 |
| test_overall | overall | moving_average_4 | 1134 | 1122 | 12 | 8.246 | 40.13 | -0.993 |
| test_overall | overall | gradient_boosting | 1134 | 1134 | 0 | 7.687 | 37.35 | -3.436 |
| test_AB | shifted_skus_only | naive_last_week | 50 | 48 | 2 | 7.938 | 54.35 | 0.396 |
| test_AB | shifted_skus_only | seasonal_naive | 50 | 46 | 4 | 9.609 | 60.30 | -1.826 |
| test_AB | shifted_skus_only | moving_average_4 | 50 | 50 | 0 | 7.058 | 44.73 | -1.038 |
| test_AB | shifted_skus_only | gradient_boosting | 50 | 50 | 0 | 6.932 | 43.93 | -2.659 |
| test_C | shifted_skus_only | naive_last_week | 24 | 22 | 2 | 9.364 | 36.08 | -1.909 |
| test_C | shifted_skus_only | seasonal_naive | 24 | 24 | 0 | 19.250 | 78.84 | -6.750 |
| test_C | shifted_skus_only | moving_average_4 | 24 | 24 | 0 | 12.674 | 51.91 | -2.701 |
| test_C | shifted_skus_only | gradient_boosting | 24 | 24 | 0 | 11.972 | 49.03 | -5.389 |
| test_overall | shifted_skus_only | naive_last_week | 74 | 70 | 4 | 8.386 | 46.15 | -0.329 |
| test_overall | shifted_skus_only | seasonal_naive | 74 | 70 | 4 | 12.914 | 68.54 | -3.514 |
| test_overall | shifted_skus_only | moving_average_4 | 74 | 74 | 0 | 8.880 | 47.79 | -1.578 |
| test_overall | shifted_skus_only | gradient_boosting | 74 | 74 | 0 | 8.567 | 46.11 | -3.544 |

Common-support MAE (only rows that all four methods scored), reported separately from own coverage:

| partition | stratum | common rows / eligible | naive_last_week | seasonal_naive | moving_average_4 | gradient_boosting |
|---|---|---|---:|---:|---:|---:|
| test_AB | overall | 695 / 756 | 9.594 | 10.232 | 7.657 | 7.117 |
| test_C | overall | 351 / 378 | 11.051 | 12.108 | 9.237 | 8.532 |
| test_overall | overall | 1046 / 1134 | 10.083 | 10.861 | 8.187 | 7.592 |
| test_AB | shifted_skus_only | 45 / 50 | 8.200 | 9.533 | 6.843 | 6.408 |
| test_C | shifted_skus_only | 22 / 24 | 9.364 | 20.591 | 13.523 | 12.823 |
| test_overall | shifted_skus_only | 67 / 74 | 8.582 | 13.164 | 9.036 | 8.515 |

The full tables, covering every stratum (per store, common support, and a separately labelled macro store
average), are in `results/metrics.json`, `results/results_table.md` and `results/report.html`.

**Did GBM beat seasonal-naive on the test set?** Yes, on this synthetic set only. It had lower MAE and lower WAPE
than seasonal_naive in every stratum listed above (test_AB, test_C, test_overall, stores A/B/C, shifted SKUs),
on both own coverage and common support. On store C's shifted SKUs it still lost to naive_last_week. Its
seasonal_naive comparison there (11.972 vs 19.250) is driven by seasonal_naive reading a pre-shift week.

### Model selection

* Grid: 36 pre-declared configurations of `HistGradientBoostingRegressor` (random_state 20260928,
  early_stopping False, squared_error). They were scored on one fixed validation mask (eligible_key_set, 703 of
  780 validation_AB rows), and the same mask was used for all three baselines.
* **Failed grid configurations: none.** 36 succeeded, 0 failed (see `results/selection_log.json`).
* Selected: #13, `max_iter=200, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=20,
  l2_regularization=1.0`, validation MAE 7.141. There were no ties. It was refit once on final_refit_AB (2274 fit
  rows).
* Non-negative clipping `max(0, raw)` is applied to every GBM prediction. On the test set, 0 of 1170 raw
  predictions were negative (minimum raw value 7.609), so clipping changed nothing there. Both the pre-clip and
  post-clip values are saved in `results/forecasts.csv`. The clipping path itself is exercised by
  `tests/test_evaluate.py::test_negative_prediction_policy_fixture`.

### Data facts (from the generator's truth_meta)

* Realized missing-target fraction: **0.030448717948717948** (285 of 9360 cells; Bernoulli p = 0.03, not a quota).
* Planted demand-shift SKUs, from test week 92 onward: SKU002 (x1.6) and SKU003 (x0.6).

### Forecast for a future week

**`results/future_forecast_week105.csv` is superseded, nonconforming output — kept for the record, not
current.** It was produced by the pre-fix `future_forecast`, before the 2026-09-28 round-2 fix (see
[eval/ERRATUM-2026-09-28.md](eval/ERRATUM-2026-09-28.md)): each row carries a numeric GBM prediction and
`status=forecast_no_outcome_yet+plan_unavailable_native_nan_routing` even though week 105 has no price/promo
plan. Under the current (and always-intended) `plan_missing_policy`, a row with no plan is unavailable for the
GBM — `status=unavailable_plan`, no numeric prediction, never a native-NaN-routed number. This file is retained
byte-unchanged as evidence of the pre-fix behavior; it is not deleted, not relabelled as conforming, and should
not be read as a current forecast.

For a conforming week-105 forecast, run `forecast --manifest eval/manifest.v2.json --target-week 105` (see
Quickstart above) into a fresh `--out-dir`; every row for week 105 will come back `unavailable_plan` with no
numeric prediction, since no plan exists for that week in the frozen dataset. **Any forecast for a week whose
outcome does not yet exist carries no accuracy metric until that outcome is observed.**

## Output files

* `results/forecasts.csv` has one row per (partition, store, sku, target_week, method). Columns: actual (empty if
  missing), prediction, prediction_raw_preclip, clipped, status (`scored` / `unavailable_history` /
  `unavailable_plan` / `target_missing`), origin_utc and availability_cutoff_utc (ISO 8601 UTC; equal by
  protocol), model_hash, protocol_hash, manifest_hash.
* CSV formula-injection mitigation: every string cell starting with `=`, `+`, `-`, `@`, TAB or CR is written with
  a leading apostrophe. Numeric cells are written as numbers and are not prefixed. CSV is not universally
  formula-safe; details are in `src/tabular_demand_forecast/csv_safety.py`.
* `results/model.joblib` is loaded only after its sha256 matches `results/model_meta.json`. Do not load joblib or
  pickle files from anywhere you do not trust.

## Licences and provenance

* [LICENSES.md](LICENSES.md) lists every direct and transitive dependency with its licence, read from installed
  package metadata. It includes matplotlib's own licence and its bundled third-party notices. Two packages are
  flagged for an owner decision: numpy (a CC0-1.0 component) and matplotlib (bundled fonts and components under
  non-OSI licences).
* [eval/manifest.json](eval/manifest.json) is the run manifest. [eval/protocol.v2.json](eval/protocol.v2.json)
  is the frozen protocol.
