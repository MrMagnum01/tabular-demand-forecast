"""The leakage guard tests listed in protocol.json's
required_leakage_tests.step_1_testable_now, implemented without any
estimator (none exists yet in this repo)."""
import dataclasses

import numpy as np
import pandas as pd

from tabular_demand_forecast.features import build_feature_frame, build_partition
from tabular_demand_forecast.generator import GeneratorSeeds, generate


def test_target_perturbation_no_backleak():
    """id: target_perturbation_no_backleak -- perturbing y_obs at or after
    an origin must leave every feature computed for an earlier origin
    unchanged."""
    data = generate(GeneratorSeeds.dev_fixture())
    perturb_week = 90
    stores = ["A", "B"]
    target_weeks = list(range(60, 104))  # spans well before and after perturb_week

    before = build_feature_frame(data, stores, target_weeks)

    perturbed_observed = data.observed.copy()
    row_mask = (perturbed_observed["store"] == "A") & (
        perturbed_observed["sku"] == "SKU001"
    ) & (perturbed_observed["week_index"] == perturb_week)
    perturbed_observed.loc[row_mask, "y_obs"] = 999999.0
    perturbed_data = data.__class__(
        stores=data.stores,
        skus=data.skus,
        n_weeks=data.n_weeks,
        latent_truth=data.latent_truth,
        mask=data.mask,
        observed=perturbed_observed,
        plans=data.plans,
        truth_meta=data.truth_meta,
    )
    after = build_feature_frame(perturbed_data, stores, target_weeks)

    key_cols = ["store", "sku", "target_week"]
    merged = before.merge(after, on=key_cols, suffixes=("_before", "_after"))
    affected_sku = merged["sku"] == "SKU001"

    # Earlier origin: any target_week strictly before perturb_week can never
    # reference week perturb_week in a lag/roll/target column.
    earlier = merged[affected_sku & (merged["target_week"] < perturb_week)]
    assert len(earlier) > 0
    for col in ["lag1", "lag2", "lag4", "lag52", "roll_mean_4", "roll_mean_13", "y_obs"]:
        before_vals = earlier[f"{col}_before"].to_numpy()
        after_vals = earlier[f"{col}_after"].to_numpy()
        both_nan = np.isnan(before_vals) & np.isnan(after_vals)
        assert np.all(both_nan | (before_vals == after_vals))

    # Sanity: the perturbation does propagate forward (test is not vacuous).
    later = merged[
        affected_sku & (merged["store"] == "A") & (merged["target_week"] == perturb_week + 1)
    ]
    assert len(later) > 0
    assert (later["lag1_before"] != later["lag1_after"]).all()


def test_future_plan_availability_cutoff():
    """id: future_plan_availability_cutoff -- a plan whose available_at
    exceeds a given origin can never be joined into that origin's feature
    row; the join key is the target week itself, never a later week."""
    data = generate(GeneratorSeeds.dev_fixture())
    tampered_plans = data.plans.copy()
    sentinel_week = 56
    tampered_plans.loc[
        (tampered_plans["store"] == "A")
        & (tampered_plans["sku"] == "SKU001")
        & (tampered_plans["week_index"] == sentinel_week),
        ["price", "promo_flag"],
    ] = [123456.0, 1]
    tampered_data = data.__class__(
        stores=data.stores,
        skus=data.skus,
        n_weeks=data.n_weeks,
        latent_truth=data.latent_truth,
        mask=data.mask,
        observed=data.observed,
        plans=tampered_plans,
        truth_meta=data.truth_meta,
    )
    frame = build_feature_frame(tampered_data, ["A"], [sentinel_week - 1])
    row = frame[(frame["sku"] == "SKU001")].iloc[0]
    assert row["price"] != 123456.0
    assert row["promo_flag"] != 1


def test_late_plan_for_same_target_is_not_available():
    """id: late_plan_same_target_week_is_unavailable -- Astra's 2026-09-28
    review probe (40-sessions/2026-09-28-astra-tabular-protocol-review-probe.py),
    wired in unchanged. A plan sharing its week_index with the target week
    itself, but delayed past origin(t), must be unavailable -- not just a
    different week's plan being looked up for the wrong target."""
    d = generate(GeneratorSeeds.dev_fixture())
    p = d.plans.copy()
    mask = (p.store == 'A') & (p.sku == 'SKU001') & (p.week_index == 56)
    p.loc[mask, ['price', 'promo_flag']] = [123456., 1]
    p.loc[mask, 'available_at'] += pd.Timedelta(days=7)
    f = build_feature_frame(dataclasses.replace(d, plans=p), ['A'], [56])
    r = f.loc[f.sku == 'SKU001'].iloc[0]
    assert pd.isna(r.price) and pd.isna(r.promo_flag), r.to_dict()


def test_missing_week_does_not_shift_lag52():
    """id: missing_week_does_not_shift_lag52 -- a missing (masked) week
    must not cause lag52 to silently read a different week; lag52(t) is
    always y_obs at the exact calendar week t-52, or NaN, never a
    nearest-available substitute."""
    from tabular_demand_forecast.features import ObservationLookup
    from tabular_demand_forecast.generator import GeneratedData

    target_week = 100
    exact_lag52_week = target_week - 52  # 48
    rows = [
        {"store": "A", "sku": "SKU001", "week_index": exact_lag52_week, "y_obs": np.nan},
        {"store": "A", "sku": "SKU001", "week_index": exact_lag52_week - 1, "y_obs": 111.0},
        {"store": "A", "sku": "SKU001", "week_index": exact_lag52_week + 1, "y_obs": 222.0},
    ]
    observed = pd.DataFrame(rows)
    fixture = GeneratedData(
        stores=["A"], skus=["SKU001"], n_weeks=110,
        latent_truth=None, mask=None, observed=observed, plans=None, truth_meta={},
    )
    obs = ObservationLookup(fixture)
    result = obs.get("A", "SKU001", exact_lag52_week)
    assert np.isnan(result)
    # Confirm it did not silently fall back to a neighboring week's value.
    assert result != 111.0 and result != 222.0


def test_no_cross_entity_rolling():
    """id: no_cross_entity_rolling -- per-(store,sku) rolling means never
    include another store's or SKU's observations."""
    from tabular_demand_forecast.features import ObservationLookup, _roll_mean
    from tabular_demand_forecast.generator import GeneratedData

    rows = []
    for w in range(1, 6):
        rows.append({"store": "A", "sku": "SKU001", "week_index": w, "y_obs": 1.0})
        rows.append({"store": "A", "sku": "SKU002", "week_index": w, "y_obs": 999999.0})
        rows.append({"store": "B", "sku": "SKU001", "week_index": w, "y_obs": 888888.0})
    observed = pd.DataFrame(rows)
    fixture = GeneratedData(
        stores=["A", "B"], skus=["SKU001", "SKU002"], n_weeks=10,
        latent_truth=None, mask=None, observed=observed, plans=None, truth_meta={},
    )
    obs = ObservationLookup(fixture)
    value = _roll_mean(obs, "A", "SKU001", 6, 4, 3)
    assert value == 1.0  # not contaminated by SKU002 or store B's huge values


def test_train_validation_test_key_disjointness():
    """id: train_validation_test_key_disjointness -- train_AB,
    validation_AB and test (AB+C) keys are disjoint by
    (store, sku, target_week); no key appears in two partitions."""
    data = generate(GeneratorSeeds.dev_fixture())

    def keys(name):
        frame = build_partition(data, name)
        return set(zip(frame["store"], frame["sku"], frame["target_week"]))

    train_ab = keys("train_AB")
    validation_ab = keys("validation_AB")
    test_ab = keys("test_AB")
    test_c = keys("test_C")

    assert train_ab.isdisjoint(validation_ab)
    assert train_ab.isdisjoint(test_ab)
    assert train_ab.isdisjoint(test_c)
    assert validation_ab.isdisjoint(test_ab)
    assert validation_ab.isdisjoint(test_c)
    assert test_ab.isdisjoint(test_c)  # disjoint by store (A/B vs C) too

    # final_refit_AB is a subset of train_AB + validation_AB by design, not
    # an additional disjoint split -- protocol.partitions.final_refit_AB note.
    final_refit = keys("final_refit_AB")
    assert final_refit == (train_ab | validation_ab)
