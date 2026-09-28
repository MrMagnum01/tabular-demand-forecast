"""Deterministic synthetic data generator.

Implements exactly the specification frozen in eval/protocol.json: the
full (store, sku, week) grid, planted seasonality / promo uplift / price
elasticity / demand shift / missingness effects, and the latent-truth,
mask and observed tables. No estimator, fitting or metric code lives
here -- step 1 scope only.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .calendar_utils import week_of_year
from .protocol import load_protocol


@dataclasses.dataclass(frozen=True)
class GeneratorSeeds:
    root_seed: int
    sku_base_level_seed: int
    price_reference_seed: int
    promo_schedule_seed: int
    price_noise_seed: int
    demand_noise_seed: int
    missingness_seed: int
    shift_selection_seed: int

    @classmethod
    def from_dict(cls, d: Dict[str, int]) -> "GeneratorSeeds":
        return cls(
            root_seed=d["root_seed"],
            sku_base_level_seed=d["sku_base_level_seed"],
            price_reference_seed=d["price_reference_seed"],
            promo_schedule_seed=d["promo_schedule_seed"],
            price_noise_seed=d["price_noise_seed"],
            demand_noise_seed=d["demand_noise_seed"],
            missingness_seed=d["missingness_seed"],
            shift_selection_seed=d["shift_selection_seed"],
        )

    @classmethod
    def frozen(cls) -> "GeneratorSeeds":
        """The seeds that produce the one frozen evaluation dataset."""
        return cls.from_dict(load_protocol()["generator_seeds"])

    @classmethod
    def dev_fixture(cls) -> "GeneratorSeeds":
        """Development-only seeds; never used for the frozen evaluation dataset."""
        return cls.from_dict(load_protocol()["dev_fixture_seeds"])


def sku_ids(n_skus: int) -> List[str]:
    return [f"SKU{i:03d}" for i in range(1, n_skus + 1)]


@dataclasses.dataclass(frozen=True)
class GeneratedData:
    stores: List[str]
    skus: List[str]
    n_weeks: int
    latent_truth: pd.DataFrame
    mask: pd.DataFrame
    observed: pd.DataFrame
    plans: pd.DataFrame
    truth_meta: Dict[str, object]


def generate(seeds: GeneratorSeeds) -> GeneratedData:
    p = load_protocol()
    entities = p["entities"]
    planted = p["planted_effects"]

    stores: List[str] = entities["stores"]
    n_skus: int = entities["n_skus"]
    n_weeks: int = p["calendar"]["n_weeks"]
    skus = sku_ids(n_skus)
    store_multiplier = entities["store_multiplier"]

    shape = (len(stores), n_skus, n_weeks)  # store outer, sku middle, week inner
    week_indices = np.arange(1, n_weeks + 1)

    # --- seasonality (calendar-derived, no randomness) ---
    amplitude = planted["seasonality"]["amplitude"]
    woy = np.array([week_of_year(int(w)) for w in week_indices], dtype=float)
    seasonal_multiplier_w = 1.0 + amplitude * np.sin(2.0 * np.pi * (woy - 1.0) / 52.0)

    # --- per-SKU draws ---
    rng_sku_base = np.random.default_rng(seeds.sku_base_level_seed)
    sku_base_level = rng_sku_base.lognormal(mean=np.log(20.0), sigma=0.4, size=n_skus)

    rng_price_ref = np.random.default_rng(seeds.price_reference_seed)
    reference_price = rng_price_ref.uniform(5.0, 50.0, size=n_skus)

    # --- per-(store,sku,week) draws, vectorised in fixed row-major order ---
    promo_probability = planted["promo_uplift"]["promo_probability"]
    rng_promo = np.random.default_rng(seeds.promo_schedule_seed)
    promo_flag = (rng_promo.random(size=shape) < promo_probability).astype(int)

    rng_price_noise = np.random.default_rng(seeds.price_noise_seed)
    price_noise = rng_price_noise.uniform(0.98, 1.02, size=shape)

    price = reference_price[None, :, None] * (1.0 - 0.1 * promo_flag) * price_noise

    elasticity = planted["price_elasticity"]["elasticity"]
    price_multiplier = (price / reference_price[None, :, None]) ** elasticity

    promo_uplift_factor = planted["promo_uplift"]["promo_uplift_factor"]
    promo_multiplier = 1.0 + promo_uplift_factor * promo_flag

    seasonal_multiplier = np.broadcast_to(seasonal_multiplier_w[None, None, :], shape)

    # --- demand shift: 2 SKUs, drawn without replacement, onset in the test period ---
    shift_cfg = planted["demand_shift"]
    n_shift = shift_cfg["n_skus"]
    onset_week = shift_cfg["onset_target_week"]
    magnitudes = shift_cfg["magnitudes"]
    rng_shift = np.random.default_rng(seeds.shift_selection_seed)
    shift_indices = rng_shift.choice(n_skus, size=n_shift, replace=False)
    shift_sku_magnitude = {
        skus[int(idx)]: magnitudes[i] for i, idx in enumerate(shift_indices)
    }

    shift_multiplier = np.ones(shape, dtype=float)
    is_shift_active = np.zeros(shape, dtype=bool)
    week_ge_onset = week_indices >= onset_week  # shape (n_weeks,)
    for sku_id, magnitude in shift_sku_magnitude.items():
        k = skus.index(sku_id)
        shift_multiplier[:, k, :] = np.where(week_ge_onset, magnitude, 1.0)
        is_shift_active[:, k, :] = week_ge_onset

    store_multiplier_arr = np.array([store_multiplier[s] for s in stores])

    mu = (
        sku_base_level[None, :, None]
        * store_multiplier_arr[:, None, None]
        * seasonal_multiplier
        * promo_multiplier
        * price_multiplier
        * shift_multiplier
    )

    # --- non-negative integer demand: Negative Binomial(n=r, p=r/(r+mu)) ---
    r = planted["non_negative_integer_construction"]["dispersion_r"]
    p_success = r / (r + mu)
    rng_demand = np.random.default_rng(seeds.demand_noise_seed)
    y_latent = rng_demand.negative_binomial(r, p_success)

    # --- missingness: Bernoulli(probability), independent of partition ---
    missing_probability = planted["missingness"]["probability"]
    rng_missing = np.random.default_rng(seeds.missingness_seed)
    missing_mask = rng_missing.random(size=shape) < missing_probability
    observed_mask = ~missing_mask
    y_obs = np.where(observed_mask, y_latent.astype(float), np.nan)

    # --- flatten to long form, row-major: store outer, sku middle, week inner ---
    store_col = np.repeat(stores, n_skus * n_weeks)
    sku_col = np.tile(np.repeat(skus, n_weeks), len(stores))
    week_col = np.tile(week_indices, len(stores) * n_skus)

    latent_truth = pd.DataFrame(
        {
            "store": store_col,
            "sku": sku_col,
            "week_index": week_col,
            "mu": mu.reshape(-1),
            "y_latent": y_latent.reshape(-1),
            "is_shift_active": is_shift_active.reshape(-1),
            "seasonal_multiplier": seasonal_multiplier.reshape(-1),
            "promo_multiplier": promo_multiplier.reshape(-1),
            "price_multiplier": price_multiplier.reshape(-1),
            "shift_multiplier": shift_multiplier.reshape(-1),
        }
    )

    mask = pd.DataFrame(
        {
            "store": store_col,
            "sku": sku_col,
            "week_index": week_col,
            "observed": observed_mask.reshape(-1),
        }
    )

    observed = pd.DataFrame(
        {
            "store": store_col,
            "sku": sku_col,
            "week_index": week_col,
            "y_obs": y_obs.reshape(-1),
        }
    )

    origin_dates = pd.to_datetime(
        [str(_week_start_date_iso(int(w))) for w in week_col]
    )
    plans = pd.DataFrame(
        {
            "store": store_col,
            "sku": sku_col,
            "week_index": week_col,
            "price": price.reshape(-1),
            "promo_flag": promo_flag.reshape(-1),
            "available_at": origin_dates,
        }
    )

    truth_meta = {
        "shift_sku_magnitude": shift_sku_magnitude,
        "shift_onset_week": onset_week,
        "realized_missing_fraction": float(missing_mask.mean()),
        "sku_base_level": dict(zip(skus, sku_base_level.tolist())),
        "reference_price": dict(zip(skus, reference_price.tolist())),
    }

    return GeneratedData(
        stores=list(stores),
        skus=skus,
        n_weeks=n_weeks,
        latent_truth=latent_truth,
        mask=mask,
        observed=observed,
        plans=plans,
        truth_meta=truth_meta,
    )


def _week_start_date_iso(week_index: int) -> str:
    from .calendar_utils import week_start_date

    return week_start_date(week_index).isoformat()
