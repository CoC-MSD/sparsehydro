"""Smoke tests for the multi-method bias-overlay plots."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("plotly")

from sparsehydro.visualization.bias_multi import MethodSeries, plot_cumulative_volume_multi, plot_residuals_multi


@pytest.fixture
def demo():
    rng = np.random.default_rng(0)
    n = 500
    dt = np.datetime64("2022-01-01") + np.arange(n).astype("timedelta64[h]")
    obs = np.abs(rng.normal(5, 2, n))
    methods = [
        MethodSeries("A·1a", obs + rng.normal(0, 0.5, n)),
        MethodSeries("A·1b", obs + rng.normal(0.2, 0.5, n)),
        MethodSeries("B", obs * 1.05 + rng.normal(0, 0.4, n)),
        MethodSeries("C", obs * 0.95 + rng.normal(0, 0.4, n)),
    ]
    mask = np.zeros(n, bool)
    mask[100:200] = True
    mask[300:360] = True
    return dt, obs, methods, mask


def test_cumulative_volume_multi_builds(demo):
    dt, obs, methods, mask = demo
    fig = plot_cumulative_volume_multi(dt, obs, methods, mask=mask)
    # obs + 4 method cum lines on row 1, 4 error lines on row 2 = 9 traces.
    assert len(fig.data) == 9
    names = [t.name for t in fig.data]
    assert "Observed" in names
    for m in methods:
        assert m.name in names


def test_residuals_multi_builds(demo):
    dt, obs, methods, mask = demo
    fig = plot_residuals_multi(obs, methods, mask=mask)
    # 4 scatter + 4 box + 1:1 line = 9 traces.
    assert len(fig.data) == 9


def test_mask_length_validated(demo):
    dt, obs, methods, _ = demo
    with pytest.raises(ValueError):
        plot_cumulative_volume_multi(dt, obs, methods, mask=np.zeros(3, bool))


def test_runs_without_mask(demo):
    dt, obs, methods, _ = demo
    fig = plot_residuals_multi(obs, methods)
    assert len(fig.data) == 9
