"""Warning when unit hydrograph bounds allow responses slower than the data record."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from sparsehydro.calibration import CalibrationProblem
from sparsehydro.calibration.diagnostics import slow_response_warnings
from sparsehydro.configuration import build_model
from sparsehydro.metrics import NashSutcliffe
from sparsehydro.models.rdii import RDIIModel
from sparsehydro.models.unithydrograph import GammaUH


def month(n: int = 8928) -> pd.DataFrame:
    rain = np.zeros(n)
    rain[100:130] = 0.05
    return pd.DataFrame({"datetime": pd.date_range("2022-03-01", periods=n, freq="5min"),
                         "rain": rain, "rainfall_in": rain, "flow": rain * 2.0})


def gamma(**upper) -> GammaUH:
    m = GammaUH()
    m.initialize()
    for name, value in upper.items():
        m.get_scalar_parameter(name).upper_bound = value
    assert m.validate()
    return m


def problem(model, predicted="Q_pred"):
    return CalibrationProblem(model, month(), objectives=[NashSutcliffe()],
                              column_map={"observed": "flow", "predicted": predicted})


def test_wide_default_bounds_warn_with_the_parameters_to_narrow():
    with pytest.warns(UserWarning, match=r"GammaUH; parameters alpha, theta\).*data cover 744 h"):
        problem(gamma())


def test_site_bounds_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        problem(gamma(alpha=10.0, theta=24.0))


def test_probe_restores_parameter_values_and_ignores_fixed_parameters():
    m = gamma(alpha=10.0, theta=24.0)
    before = {n: m.get_scalar_parameter(n).value for n in m.scalar_parameter_names}
    fast = m.slowest_median_hours()
    assert {n: m.get_scalar_parameter(n).value for n in m.scalar_parameter_names} == before
    m.get_scalar_parameter("theta").calibrate = False  # held at 0.5 h
    assert m.slowest_median_hours() < fast


def test_nested_unit_hydrographs_are_found_and_named():
    cfg = {"model": {"type": "composite", "components": {
        "fast": {"type": "nash-uh", "inputs": {"rain": "data.rain"}},
        "slow": {"type": "gamma-uh", "inputs": {"rain": "data.rain"}}},
        "outputs": {"q": {"sum": ["fast.Q_pred", "slow.Q_pred"]}}},
        "parameters": {"fast.n": {"upper_bound": 5.0}, "fast.K": {"upper_bound": 4.0}}}
    model = build_model(cfg)
    assert model.validate()  # pushes composite settings to the components
    messages = slow_response_warnings(model, 744.0)
    assert len(messages) == 1 and messages[0].startswith("slow (GammaUH")


def test_rdii_pushes_bounds_and_calibrate_flags_to_its_unit_hydrographs():
    r = RDIIModel()
    r.initialize()
    assert len(slow_response_warnings(r, 744.0)) == 3
    for i, t in enumerate((6.0, 24.0, 96.0), 1):
        r.get_scalar_parameter(f"uh{i}_T").upper_bound = t
        r.get_scalar_parameter(f"uh{i}_K").upper_bound = 4.0
    r.get_scalar_parameter("uh1_K").calibrate = False
    assert r.validate()
    child = r._uh_components[0]
    assert child.get_scalar_parameter("T").upper_bound == 6.0
    assert child.get_scalar_parameter("K").calibrate is False
    assert slow_response_warnings(r, 744.0) == []


def test_no_datetime_means_no_check():
    m = gamma()
    m.prepare(month())
    assert slow_response_warnings(m, 0.0) == []
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        CalibrationProblem(m, None, objectives=[NashSutcliffe()],
                           column_map={"observed": lambda _: month()["flow"].to_numpy(), "predicted": "Q_pred"})
