"""Tests for zonal SequentialFitter and GlobalSequentialFitter on real data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sparsehydro.calibration import WeightedRMSE
from sparsehydro.models.unithydrograph import (
    GammaUH, SequentialFitter, SequentialFitSummary, GlobalSequentialFitter,
)


def test_weighted_rmse_matches_manual():
    obs = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([1.5, 2.0, 2.0, 5.0])
    w = np.array([1.0, 1.0, 3.0, 1.0])
    e = obs - pred
    expected = float(np.sqrt(np.sum(w * e**2) / np.sum(w)))
    assert abs(WeightedRMSE(weights=w).compute(obs, pred) - expected) < 1e-9


def test_weighted_rmse_uniform_equals_rmse():
    obs = np.array([1.0, 2.0, 3.0])
    pred = np.array([1.0, 2.5, 2.0])
    plain = float(np.sqrt(np.mean((obs - pred) ** 2)))
    assert abs(WeightedRMSE().compute(obs, pred) - plain) < 1e-9


def test_sequential_fitter_zonal(muum019_window, muum019_hierarchy):
    _, subs, _ = muum019_hierarchy
    subs3 = subs[:3]
    fitter = SequentialFitter(lambda: GammaUH(), muum019_window, subs3)
    summary = fitter.fit(zonal_weights=True, savgol_options={"window_length": 7}, verbose=False)
    assert isinstance(summary, SequentialFitSummary)
    assert len(summary.events) == len(subs3)
    assert len(summary.global_predicted) == len(muum019_window)
    m = summary.metrics_summary()
    assert "wrmse" in m.columns  # zonal weighting active
    assert "nash_sutcliffe" in m.columns


def test_sequential_fitter_param_evolution(muum019_window, muum019_hierarchy):
    _, subs, _ = muum019_hierarchy
    fitter = SequentialFitter(lambda: GammaUH(), muum019_window, subs[:3])
    summary = fitter.fit(verbose=False)
    pe = summary.parameter_evolution()
    assert "A" in pe.columns and "tt" in pe.columns and "tp" in pe.columns
    assert len(pe) == len(summary.events)


def test_global_sequential_fitter(muum019_window, muum019_hierarchy):
    globals_, subs, _ = muum019_hierarchy
    globs2 = globals_[:2]
    fitter = GlobalSequentialFitter(lambda: GammaUH(), muum019_window, globs2, subs)
    summary = fitter.fit(verbose=False)
    assert isinstance(summary, SequentialFitSummary)
    assert len(summary.events) == len(globs2)
    assert len(summary.global_predicted) == len(muum019_window)
    ea = summary.effective_area_summary()
    assert "observed_ae" in ea.columns and "fitted_ae" in ea.columns


def test_global_fitter_predictions_finite(muum019_window, muum019_hierarchy):
    globals_, subs, _ = muum019_hierarchy
    fitter = GlobalSequentialFitter(lambda: GammaUH(), muum019_window, globals_[:2], subs)
    summary = fitter.fit(verbose=False)
    q = summary.global_predicted["Q_pred"].to_numpy()
    assert np.all(np.isfinite(q))
    assert np.all(q >= 0.0)
