"""CompositeModel: bindings, outputs, shared parameters, overrides and order."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sparsehydro.calibration.objectives import NashSutcliffe
from sparsehydro.calibration.problem import CalibrationProblem
from sparsehydro.models import SeasonalityModel
from sparsehydro.models.abstraction import ExternalExcess
from sparsehydro.models.composite import CompositeModel
from sparsehydro.models.rdii import IAModel, RDIIModel, RTKTriangle
from sparsehydro.models.unithydrograph import NashUH, TriangleUH


def forcing(n: int = 480) -> pd.DataFrame:
    rain = np.zeros(n)
    rain[20:40] = 0.08
    rain[200:210] = 0.15
    return pd.DataFrame({
        "datetime": pd.date_range("2024-06-01", periods=n, freq="5min"),
        "rainfall_in": rain,
        "temperature_c": 18.0,
    })


def ia_rdii_composite(**kwargs) -> CompositeModel:
    kwargs.setdefault("outputs", {"rdii_cfs": "rdii.rdii_cfs", "excess": "ia.p_excess_in"})
    return CompositeModel(
        {"ia": IAModel(), "rdii": RDIIModel(ia_model=ExternalExcess())},
        bindings={"rdii.p_excess_in": "ia.p_excess_in"},
        **kwargs,
    )


def test_bound_components_reproduce_the_monolithic_model():
    data = forcing()
    plain = RDIIModel()
    plain.initialize()
    assert plain.validate()
    plain.prepare(data)
    expected = plain.predict_arrays()

    comp = ia_rdii_composite()
    comp.initialize()
    assert comp.validate()
    comp.prepare(data)
    got = comp.predict_arrays()

    assert comp.run_order == ["ia", "rdii"]
    np.testing.assert_array_equal(got["rdii_cfs"], expected["rdii_cfs"])
    np.testing.assert_array_equal(got["excess"], expected["p_excess_in"])
    assert list(comp.predict().columns) == ["datetime", "rdii_cfs", "excess"]


def test_default_prefixes_are_dotted_and_flat_api_works():
    comp = ia_rdii_composite()
    comp.initialize()
    names = comp.scalar_parameter_names
    assert "ia.ia_max" in names and "rdii.R_1" in names and "rdii.uh1_T" in names
    comp.apply_flat_parameter("rdii.R_1", 0.2)
    comp.validate()
    assert comp.component("rdii").get_scalar_parameter("R_1").value == 0.2
    assert comp.read_flat_parameter("rdii.R_1") == 0.2
    assert "rdii.sum_R_leq_1" in comp.inequality_constraint_names


def test_vector_parameters_forward_with_dotted_prefix():
    comp = CompositeModel({"seas": SeasonalityModel()}, outputs=None)
    comp.initialize()
    vec = comp.vector_parameter_names[0]
    assert vec.startswith("seas.")
    comp.apply_flat_parameter(f"{vec}[0]", 1.3)
    comp.validate()
    child_vec = comp.component("seas").get_vector_parameter(vec.split(".", 1)[1])
    assert child_vec.values[0] == 1.3


def test_sum_and_product_outputs():
    data = forcing()
    comp = CompositeModel(
        {"fast": NashUH(A=5.0, n=2, K=0.3), "slow": NashUH(A=2.0, n=3, K=3.0)},
        bindings={"fast.rain": "data.rainfall_in", "slow.rain": "data.rainfall_in"},
        outputs={"total": ["fast.Q_pred", "slow.Q_pred"],
                 "both": {"product": ["fast.Q_pred", "slow.Q_pred"]},
                 "fast_only": "fast.Q_pred"},
    )
    comp.initialize()
    comp.prepare(data)
    out = comp.predict_arrays()
    np.testing.assert_allclose(out["total"], out["fast_only"] + comp.component("slow").predict_flow())
    np.testing.assert_allclose(out["both"], out["fast_only"] * comp.component("slow").predict_flow())


def test_shared_parameter_drives_every_member():
    comp = CompositeModel(
        {"a": RDIIModel(ia_model=ExternalExcess(source_column="rainfall_in")),
         "b": RDIIModel(ia_model=ExternalExcess(source_column="rainfall_in"))},
        shared={"area_acres": ["a.area_acres", "b.area_acres"]},
        outputs={"total": ["a.rdii_cfs", "b.rdii_cfs"]},
    )
    comp.initialize()
    assert "area_acres" in comp.scalar_parameter_names
    assert "a.area_acres" not in comp.scalar_parameter_names
    comp.apply_flat_parameter("area_acres", 42.0)
    comp.validate()
    assert all(comp.component(n).get_scalar_parameter("area_acres").value == 42.0 for n in "ab")


def test_overrides_and_parameter_order():
    comp = ia_rdii_composite(
        parameter_overrides={"rdii.uh1_T": {"lower_bound": 0.5, "upper_bound": 3.0, "value": 1.0},
                             "ia.ia_max": {"calibrate": False}},
        parameter_order=["rdii.R_1", "rdii.uh1_T"],
    )
    comp.initialize()
    assert comp.scalar_parameter_names[:2] == ["rdii.R_1", "rdii.uh1_T"]
    p = comp.get_scalar_parameter("rdii.uh1_T")
    assert (p.lower_bound, p.upper_bound, p.value) == (0.5, 3.0, 1.0)
    child = comp.component("rdii").get_scalar_parameter("uh1_T")
    assert (child.lower_bound, child.upper_bound) == (0.5, 3.0)
    assert comp.get_scalar_parameter("ia.ia_max").calibrate is False
    with pytest.raises(ValueError, match="unknown parameter"):
        ia_rdii_composite(parameter_overrides={"nope": {"value": 1}}).initialize()


def test_wiring_errors_are_reported():
    with pytest.raises(ValueError, match="accepts inputs"):
        CompositeModel({"ia": IAModel(), "uh": NashUH()},
                       bindings={"uh.nonexistent": "ia.p_excess_in"}).initialize()
    with pytest.raises(ValueError, match="cycle"):
        CompositeModel({"a": NashUH(), "b": NashUH()},
                       bindings={"a.rain": "b.Q_pred", "b.rain": "a.Q_pred"}).initialize()
    with pytest.raises(ValueError, match="invalid component name"):
        CompositeModel({"data": NashUH()})
    with pytest.raises(ValueError, match="collision"):
        CompositeModel({"a": NashUH(), "b": NashUH()}, prefixes={"a": "", "b": ""}).initialize()


def test_unknown_output_field_names_the_available_ones():
    comp = ia_rdii_composite(outputs={"x": "rdii.not_a_field"})
    comp.initialize()
    comp.prepare(forcing())
    with pytest.raises(KeyError, match="available"):
        comp.predict_arrays()


def test_component_constraints_forward_and_require_feasible():
    comp = CompositeModel(
        {"tri": TriangleUH(A=1.0, Tp=1.0, Tb=3.0)},
        bindings={"tri.rain": "data.rainfall_in"},
        outputs={"q": "tri.Q_pred"}, require_feasible=True,
    )
    comp.initialize()
    comp.prepare(forcing())
    assert comp.inequality_constraint_names == ["tri.Tp_lt_Tb"]
    comp.predict_arrays()
    comp.apply_flat_parameter("tri.Tp", 4.0)
    assert comp.inequality_constraints() == [pytest.approx(1.0)]
    with pytest.raises(ValueError, match="violate"):
        comp.predict_arrays()


def test_composite_calibrates_through_calibration_problem():
    data = forcing()
    truth = ia_rdii_composite()
    truth.initialize()
    truth.prepare(data)
    obs = truth.predict_arrays()["rdii_cfs"]
    comp = ia_rdii_composite(parameter_order=["rdii.R_1"])
    comp.initialize()
    comp.validate()
    problem = CalibrationProblem(comp, data.assign(obs=obs), [NashSutcliffe()],
                                 column_map={"observed": "obs", "predicted": "rdii_cfs"})
    x = np.array([comp.read_flat_parameter(n) for n in problem.param_names])
    assert problem.evaluate(x)[0] == pytest.approx(-1.0)
    worker = problem.make_copy()
    assert worker.evaluate(x)[0] == pytest.approx(-1.0)


def test_rdii_component_outputs():
    m = RDIIModel(uh_components=[RTKTriangle(R=0.03, T=1.0, K=2.0), NashUH()], component_outputs=True)
    m.initialize()
    m.prepare(forcing())
    out = m.predict_arrays()
    np.testing.assert_allclose(out["rdii_component_1"] + out["rdii_component_2"], out["rdii_cfs"],
                               atol=1e-12)
