"""Slot-of-week sanitary profiling, the G-saturation flag, and preset pickling.

The "profiled" sanitary fit used to run inside every prediction of the removed
``WholeFlowModel``; it is now :func:`sparsehydro.models.baseflow.profile_slot_levels`,
a pre-fit step. These tests port its behaviour checks.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from sparsehydro.models.abstraction import SoilGroundwaterStore
from sparsehydro.models.baseflow import CenteredWeeklyPattern, profile_slot_levels, slot_of_week
from sparsehydro.models.presets.wholeflow import PatternWholeFlowModel


def synthetic(n_days: int = 42, seed: int = 3):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2021-01-04", periods=n_days * 24, freq="h")   # starts on a Monday
    doy = t.dayofyear.to_numpy() + t.hour.to_numpy() / 24.0
    temp = 12.0 + 12.0 * np.sin(2.0 * np.pi * (doy - 105.0) / 365.0)
    rain = (rng.random(t.size) < 0.02) * rng.exponential(0.3, t.size)
    h = t.hour.to_numpy()
    san = 0.4 * (1.0 + 0.5 * np.sin(2.0 * np.pi * (h - 9) / 24.0))
    san = san * np.where(t.dayofweek.to_numpy() >= 5, 0.9, 1.0)
    other = 0.2 + 0.1 * np.sin(np.arange(t.size) / 50.0) ** 2   # stand-in for GWI + RDII
    df = pd.DataFrame({"datetime": t, "rainfall_in": rain, "temperature_c": temp})
    return df, san, other


def slot_cap(datetimes, values):
    slot, n = slot_of_week(datetimes, 1.0)
    return np.bincount(slot, weights=values, minlength=n) / np.bincount(slot, minlength=n)


def test_hourly_and_15min_slots():
    t = pd.date_range("2021-01-04", periods=24 * 14, freq="h")
    slot, n = slot_of_week(t.values, 1.0)
    assert n == 168
    np.testing.assert_array_equal(slot, t.dayofweek * 24 + t.hour)
    t = pd.date_range("2021-01-04", periods=96 * 7, freq="15min")
    slot, n = slot_of_week(t.values, 0.25)
    assert n == 7 * 96
    np.testing.assert_array_equal(slot, np.arange(7 * 96))


def test_profile_recovers_true_sanitary_when_cap_is_not_binding():
    df, san, other = synthetic()
    level, hits = profile_slot_levels(san + other, other, np.ones(len(df), bool), df.datetime, 1.0,
                                      cap_cfs=slot_cap(df.datetime, 1.3 * san))
    np.testing.assert_allclose(level, san, atol=1e-9)
    assert hits["at_cap"] == 0.0 and hits["empty_slots"] == 0


def test_profile_is_clipped_to_the_cap():
    df, san, other = synthetic()
    cap = slot_cap(df.datetime, 0.8 * san)
    level, hits = profile_slot_levels(san + other, other, np.ones(len(df), bool), df.datetime, 1.0,
                                      cap_cfs=cap)
    np.testing.assert_allclose(level, 0.8 * san, atol=1e-9)
    assert hits["at_cap"] == 1.0


def test_profile_uses_only_the_fit_mask():
    df, san, other = synthetic()
    mask = np.zeros(len(df), bool)
    mask[: len(df) // 2] = True
    obs = san + other
    bumped = obs.copy()
    bumped[~mask] += 5.0
    a, _ = profile_slot_levels(obs, other, mask, df.datetime, 1.0)
    b, _ = profile_slot_levels(bumped, other, mask, df.datetime, 1.0)
    np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(a, san, atol=1e-9)
    with pytest.raises(ValueError, match="aligned"):
        profile_slot_levels(obs, other[:-1], mask, df.datetime, 1.0)


def test_g_at_cap_flags_only_saturated_steps():
    df, _, _ = synthetic()
    store = SoilGroundwaterStore(100.0)
    store.initialize()
    store.prepare(df)
    store.get_scalar_parameter("g_max").value = 1e6                 # unbounded G never spills
    assert store.predict_arrays()["G_at_cap"].sum() == 0.0
    store.get_scalar_parameter("g_max").value = 0.01
    store.get_scalar_parameter("k_div").value = 0.05
    cap = store.predict_arrays()["G_at_cap"]
    assert cap.sum() > 0.0 and set(np.unique(cap)) <= {0.0, 1.0}


def test_prepared_preset_pickles():
    df, _, _ = synthetic()
    pattern = CenteredWeeklyPattern(tuple(0.05 * np.sin(np.arange(24) * np.pi / 12)), (0.0,) * 7)
    model = PatternWholeFlowModel(pattern, 100.0, 1.0, 0.5, 2.0)
    model.initialize()
    model.prepare(df)
    clone = pickle.loads(pickle.dumps(model))
    pd.testing.assert_frame_equal(model.predict(), clone.predict())
