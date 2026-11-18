"""sparsehydro.metrics.functions: one implementation per metric."""

import math

import numpy as np
import pytest

from sparsehydro import metrics
from sparsehydro.metrics import functions as F
from sparsehydro.metrics.objectives import PBIAS, ConcordanceCorrelationCoefficient, NashSutcliffe

OBS = np.array([1.0, 2.0, 4.0, 3.0, 1.5])
PRED = np.array([1.2, 1.8, 3.5, 3.3, 1.4])


def test_objectives_wrap_the_functions():
    assert NashSutcliffe().evaluate(OBS, PRED) == F.nash_sutcliffe(OBS, PRED)
    assert ConcordanceCorrelationCoefficient().evaluate(OBS, PRED) == F.concordance_correlation(OBS, PRED)
    assert PBIAS().evaluate(OBS, PRED) == abs(F.pbias(OBS, PRED))


def test_pbias_is_signed_positive_for_over_prediction():
    assert F.pbias(OBS, OBS * 1.1) == pytest.approx(10.0)
    assert F.pbias(OBS, OBS * 0.9) == pytest.approx(-10.0)


def test_concordance_undefined_modes():
    flat = np.ones(4)
    with pytest.raises(ValueError, match="undefined"):
        F.concordance_correlation(flat, flat)
    assert math.isnan(F.concordance_correlation(flat, flat, undefined="nan"))
    assert math.isnan(F.concordance_correlation([1.0], [1.0], undefined="nan"))
    assert math.isnan(F.concordance_correlation([1.0, np.nan], [1.0, 2.0], undefined="nan"))
    with pytest.raises(ValueError, match="Shape mismatch"):
        F.concordance_correlation(OBS, PRED[:-1], undefined="nan")


def test_peak_helpers():
    np.testing.assert_array_equal(F.top_peak_indices([1.0, 5.0, 3.0, 5.0], 0.5), [1, 3])
    assert F.peak_regression_slope([1.0, 2.0], [2.0, 4.0]) == pytest.approx(2.0)
    assert math.isnan(F.peak_regression_slope([], []))


def test_fit_summary_drops_non_finite_pairs():
    s = F.fit_summary(np.r_[OBS, np.nan], np.r_[PRED, 1.0])
    assert s["n"] == len(OBS)
    assert s["nse"] == pytest.approx(F.nash_sutcliffe(OBS, PRED))
    assert s["pbias"] == pytest.approx(F.pbias(OBS, PRED))
    assert s["peak_error_pct"] == pytest.approx(100 * (3.5 - 4.0) / 4.0)
    assert math.isnan(F.fit_summary([1.0], [1.0])["nse"])


def test_package_exports():
    for name in ("nash_sutcliffe", "kge", "NashSutcliffe", "objective_registry", "per_event_scores"):
        assert hasattr(metrics, name)
