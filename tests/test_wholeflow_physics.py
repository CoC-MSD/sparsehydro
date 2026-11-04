"""Regression checks for the rainfall partition shared by staged and joint fits."""

import numpy as np
import pytest

from sparsehydro.models.wholeflow import _gwi_B_core


@pytest.mark.parametrize("dt", [1 / 12, 0.25, 1.0])
@pytest.mark.parametrize("pulse", [0.0, 0.01, 0.5, 10.0])
def test_rainfall_limited_recharge_and_budget(dt, pulse):
    rain = np.zeros(720)
    rain[[0, 1, 120]] = pulse
    s, g, excess, et, sewer = _gwi_B_core(
        rain, np.full(len(rain), 10.0), dt,
        0.5, 5.0, 0.01, 0.01, 0.08, 10.0, 0.0, 0.005, 0.0005, 0.1,
    )
    assert np.all(excess >= 0), "Recharge must never exceed available rainfall"
    assert np.all(excess <= rain)
    assert np.all((s >= 0) & (s <= 0.5))
    assert np.all((g >= 0) & (g <= 0.1))
    assert np.all(et >= 0) and np.all(sewer >= 0)
    np.testing.assert_allclose(
        np.cumsum(rain), s + g + np.cumsum(excess + et + sewer),
        atol=1e-10, rtol=1e-10,
    )


@pytest.mark.parametrize("temperature", [-20.0, 0.0, 40.0])
@pytest.mark.parametrize("full", [False, True])
def test_initial_storage_capacity_spill_and_temperature(temperature, full):
    rain = np.array([0.01, 10.0, 0.0, 0.0])
    s0, g0 = (0.05, 0.01) if full else (0.0, 0.0)
    s, g, excess, et, sewer = _gwi_B_core(
        rain, np.full(4, temperature), 1.0,
        .05, 30.0, .2, .2, .2, 10.0, 0.0, .05, .02, .01, s0, g0,
    )
    if full:
        assert excess[0] == rain[0]
    if temperature < 0:
        assert np.all(et == 0)
    assert np.all((excess >= 0) & (excess <= rain))
    np.testing.assert_allclose(s + g + np.cumsum(excess + et + sewer),
                               s0 + g0 + np.cumsum(rain), atol=1e-10)
