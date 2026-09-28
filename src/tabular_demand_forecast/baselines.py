"""The three fixed baselines of eval/protocol.v2.json models.baselines.

All three read the exact same full reindexed weekly grid that
features.py uses (features.ObservationLookup), and moving_average_4 calls
features._roll_mean itself, so a baseline can never disagree with the
lag1 / lag52 / roll_mean_4 feature it is defined to equal. A NaN return
means UNAVAILABLE for that row (excluded from the scored set and counted),
never a substituted value.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .features import ObservationLookup, _roll_mean
from .protocol import load_protocol

BASELINE_NAMES = ("naive_last_week", "seasonal_naive", "moving_average_4")

NAIVE_LAG = 1
SEASONAL_LAG = 52


def _ma4_spec() -> tuple:
    p = load_protocol()
    ma = p["models"]["baselines"]["moving_average_4"]
    feat = p["features"]["rolling_means"]["roll_mean_4"]
    if ma["min_observed_required"] != feat["min_observed_required"]:
        raise ValueError("moving_average_4 threshold drifted from roll_mean_4 in protocol")
    ks = p["features"]["lags"]["K_values"]
    if NAIVE_LAG not in ks or SEASONAL_LAG not in ks:
        raise ValueError("baseline lags must be protocol lag K-values")
    return int(feat["window_size"]), int(ma["min_observed_required"])


MA4_WINDOW, MA4_MIN_OBSERVED = _ma4_spec()


def baseline_predictions(obs: ObservationLookup, frame: pd.DataFrame) -> Dict[str, np.ndarray]:
    """One prediction array per baseline, aligned with frame's rows."""
    keys = list(zip(frame["store"], frame["sku"], frame["target_week"]))
    naive = np.array([obs.get(s, k, int(t) - NAIVE_LAG) for s, k, t in keys], dtype=float)
    seasonal = np.array([obs.get(s, k, int(t) - SEASONAL_LAG) for s, k, t in keys], dtype=float)
    ma4 = np.array(
        [_roll_mean(obs, s, k, int(t), MA4_WINDOW, MA4_MIN_OBSERVED) for s, k, t in keys],
        dtype=float,
    )
    return {"naive_last_week": naive, "seasonal_naive": seasonal, "moving_average_4": ma4}
