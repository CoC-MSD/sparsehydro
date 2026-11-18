"""Tests for zonal SequentialFitter and GlobalSequentialFitter on real data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sparsehydro.calibration import WeightedRMSE
from sparsehydro.calibration import GlobalSequentialFitter, SequentialFitSummary, SequentialFitter
from sparsehydro.models.unithydrograph import (
    GammaUH,
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
    assert "A" in pe.columns and "alpha" in pe.columns and "theta" in pe.columns
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


def test_calibrated_areas_are_on_the_per_step_scale(muum019_window, muum019_hierarchy):
    """A is a per-rate amplitude; calibrated_areas() divides by dt for comparison."""
    _, subs, _ = muum019_hierarchy
    summary = SequentialFitter(lambda: GammaUH(), muum019_window, subs[:2]).fit(verbose=False)
    assert summary.dt_hours == pytest.approx(5.0 / 60.0)
    a = [dict(zip(c.param_names, c.pareto_X[0]))["A"] for c in summary.calibration_results]
    np.testing.assert_allclose(summary.calibrated_areas(), np.array(a) / summary.dt_hours)


def test_global_fitter_reports_progress_and_stops(muum019_window, muum019_hierarchy):
    globals_, subs, _ = muum019_hierarchy
    seen = []
    fitter = GlobalSequentialFitter(lambda: GammaUH(), muum019_window, globals_[:2], subs)
    fitter.fit(verbose=False, progress=lambda i, n, text: seen.append((i, n)))
    assert seen[0] == (0, 2) and seen[-1] == (2, 2) and len(seen) == 3
    calls = []

    def stop_after_first():
        calls.append(1)
        return len(calls) > 1

    with pytest.raises(InterruptedError):
        fitter.fit(verbose=False, should_stop=stop_after_first)


def test_independent_fits_start_from_the_defaults(muum019_window, muum019_hierarchy):
    globals_, subs, _ = muum019_hierarchy
    fitter = GlobalSequentialFitter(lambda: GammaUH(), muum019_window, globals_[:2], subs)
    warm = fitter.fit(verbose=False).parameter_evolution()
    cold = fitter.fit(verbose=False, warm_start=False).parameter_evolution()
    assert len(warm) == len(cold) == 2
    # the first event starts from the defaults either way
    np.testing.assert_allclose(warm.iloc[0][["A", "alpha", "theta"]].astype(float),
                               cold.iloc[0][["A", "alpha", "theta"]].astype(float))


def test_pooled_fit_is_one_parameter_set_scored_per_event(muum019_window, muum019_hierarchy):
    from sparsehydro.calibration import PooledEventFitter, PooledFitResult

    globals_, subs, _ = muum019_hierarchy
    fitter = PooledEventFitter(lambda: GammaUH(), muum019_window, globals_[:2], subs)
    w = fitter.weights(peak_weight=3.0, tail_weight=1.0)
    assert w.shape == (len(muum019_window),) and set(np.unique(w)) <= {0.0, 1.0, 3.0} and (w > 0).any()
    res = fitter.fit()
    assert isinstance(res, PooledFitResult)
    assert set(res.parameters) == {"A", "alpha", "theta"}
    assert len(res.predicted) == len(muum019_window) and np.all(np.isfinite(res.predicted["Q_pred"]))
    assert list(res.event_scores["event_id"]) == [g.global_id for g in sorted(
        globals_[:2], key=lambda g: g.window_start_datetime)]
    assert res.event_scores[["nse", "rmse"]].notna().all().all()


def test_pooled_fit_on_one_event_matches_the_per_event_fit(muum019_window, muum019_hierarchy):
    from sparsehydro.calibration import PooledEventFitter

    globals_, subs, _ = muum019_hierarchy
    one = globals_[:1]
    per_event = GlobalSequentialFitter(lambda: GammaUH(), muum019_window, one, subs).fit(verbose=False)
    pooled = PooledEventFitter(lambda: GammaUH(), muum019_window, one, subs).fit()
    a_event = dict(zip(per_event.calibration_results[0].param_names, per_event.calibration_results[0].pareto_X[0]))
    # same zones and objective; the pooled run sees the whole record, so allow a little difference
    assert pooled.parameters["theta"] == pytest.approx(a_event["theta"], rel=0.25)
