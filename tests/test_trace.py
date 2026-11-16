"""Trace fields: components and internal states as a secondary model output."""

from __future__ import annotations

import operator

import numpy as np
import pandas as pd
import pytest

from sparsehydro.configuration import build_model
from sparsehydro.models import EnsembleModel, IModel
from sparsehydro.models.abstraction import ExternalExcess
from sparsehydro.models.composite import AbstractionUHModel, CompositeModel
from sparsehydro.models.rdii import IAModel, RDIIModel, RTKTriangle
from sparsehydro.models.unithydrograph import GammaUH, MixtureUH, NashUH, PeakTailUH
from sparsehydro.parameters import FIELD_ROLES, FieldRecord
from sparsehydro.registry import registry


def forcing(n: int = 600) -> pd.DataFrame:
    rain = np.zeros(n)
    rain[20:40] = 0.08
    rain[200:212] = 0.15
    rain[400:403] = 0.4
    return pd.DataFrame({
        "datetime": pd.date_range("2024-06-01", periods=n, freq="5min"),
        "rain": rain, "rainfall_in": rain, "rainfall_mm": rain * 25.4, "temperature_c": 12.0,
    })


def run(model: IModel, data: pd.DataFrame | None = None) -> tuple[dict, dict]:
    model.initialize()
    model.validate()
    model.prepare(forcing() if data is None else data)
    return model.predict_arrays(), model.predict_trace()


def assert_components_sum(model: IModel, trace: dict) -> list[str]:
    """Every output with components equals their sum; returns the outputs checked."""
    groups: dict[str, list[str]] = {}
    for f in model.trace_fields:
        if f.role == "component":
            groups.setdefault(f.component_of, []).append(f.name)
    for owner, parts in groups.items():
        total = np.asarray(trace[owner], dtype=float)
        summed = np.sum([np.asarray(trace[p], dtype=float) for p in parts], axis=0)
        np.testing.assert_allclose(summed, total, rtol=1e-9, atol=1e-12 * max(1.0, np.abs(total).max()),
                                   err_msg=f"components of {owner} do not add up: {parts}")
    return sorted(groups)


# ---------------------------------------------------------------------------- contract


def _default_models():
    for name in registry.names():
        cls = registry.get(name)
        try:
            model = cls()
            model.initialize()
            model.validate()
            model.prepare(forcing())
        except Exception:  # needs configuration or bound inputs; covered by the explicit tests
            continue
        yield name, model


@pytest.mark.parametrize("name, model", list(_default_models()), ids=lambda x: x if isinstance(x, str) else "")
def test_every_runnable_registered_model_honours_the_trace_contract(name, model) -> None:
    outputs = model.predict_arrays()
    trace = model.predict_trace()
    for key, values in outputs.items():  # the primary output is unchanged
        np.testing.assert_array_equal(np.asarray(trace[key]), np.asarray(values))
    n = len(next(iter(outputs.values())))
    for f in model.trace_fields:
        assert f.role in FIELD_ROLES[1:]
        assert len(trace[f.name]) == n, f.name
        assert f.name not in model.output_field_names
    assert_components_sum(model, trace)


def test_field_record_roles_are_checked() -> None:
    with pytest.raises(ValueError):
        FieldRecord("x", role="volume")
    with pytest.raises(ValueError):
        FieldRecord("x", role="component")  # a component must say what it adds up to
    with pytest.raises(ValueError):
        FieldRecord("x", role="state", component_of="y")
    m = NashUH()
    m.initialize()
    with pytest.raises(ValueError):
        m.register_trace_field(FieldRecord("t", role="output"))
    with pytest.raises(ValueError):
        m.register_trace_field(FieldRecord("Q_pred", role="flux"))


def test_missing_trace_values_are_reported() -> None:
    m = NashUH()
    m.initialize()
    m.register_trace_field(FieldRecord("never", role="state"))
    m.prepare(forcing())
    with pytest.raises(RuntimeError, match="never"):
        m.predict_trace()


# ---------------------------------------------------------------------------- components


def test_rdii_reports_each_unit_hydrograph() -> None:
    model = RDIIModel(uh_components=[RTKTriangle(R=0.03, T=1.0, K=2.0), NashUH(), GammaUH()])
    outputs, trace = run(model)
    assert [f.name for f in model.trace_fields if f.role == "component"] == \
        ["rdii_component_1", "rdii_component_2", "rdii_component_3"]
    assert assert_components_sum(model, trace) == ["rdii_cfs"]
    assert all(trace[f"rdii_component_{i}"].max() > 0 for i in (1, 2, 3))
    assert "rdii_component_1" not in outputs  # calibration output is unchanged
    # the flag that returns components as outputs gives the same values
    flagged = RDIIModel(uh_components=[RTKTriangle(R=0.03, T=1.0, K=2.0), NashUH(), GammaUH()],
                        component_outputs=True)
    flagged_out, _ = run(flagged)
    np.testing.assert_allclose(flagged_out["rdii_component_2"], trace["rdii_component_2"])


@pytest.mark.parametrize("uh", [MixtureUH(), PeakTailUH(td=0.5),
                                MixtureUH([GammaUH(alpha=2, theta=0.2), NashUH(n=3, K=2.0), GammaUH(alpha=1.5, theta=9)],
                                          weights=[0.5, 0.3, 0.2])])
def test_mixture_parts_add_up_to_the_flow(uh) -> None:
    _, trace = run(uh)
    names = [f.name for f in uh.trace_fields]
    assert len(names) == len(uh.components) and all(n.endswith("Q_pred") for n in names)
    assert assert_components_sum(uh, trace) == ["Q_pred"]
    assert all(trace[n].max() > 0 for n in names)


def test_ensemble_sum_contributions() -> None:
    model = EnsembleModel([(NashUH(A=2.0), operator.itemgetter("Q_pred")),
                           (GammaUH(A=1.0), operator.itemgetter("Q_pred"))],
                          aliases=["fast", "slow"], output_name="total", normalize_weights=False)
    _, trace = run(model)
    assert assert_components_sum(model, trace) == ["total"]
    product = EnsembleModel([(NashUH(), "Q_pred"), (GammaUH(), "Q_pred")], mode="product")
    product.initialize()
    assert not product.trace_field_names  # factors are not additive components


def test_composite_exposes_children_and_nested_traces() -> None:
    model = build_model({"type": "composite", "components": {
        "ia": {"type": "initial-abstraction"},
        "rdii": {"type": "rdii", "ia_model": {"type": "external-excess"},
                 "routing": [{"type": "rtk-triangle"}, {"type": "mixture-uh"}],
                 "inputs": {"p_excess_in": "ia.p_excess_in"}},
        "base": {"type": "gamma-uh", "inputs": {"rain": "data.rainfall_in"}}},
        "outputs": {"total_cfs": {"sum": ["rdii.rdii_cfs", "base.Q_pred"]}}})
    model.validate()
    model.prepare(forcing())
    outputs = model.predict_arrays()
    trace = model.predict_trace()
    roles = {f.name: (f.role, f.component_of) for f in model.trace_fields}
    assert roles["rdii.rdii_cfs"] == ("component", "total_cfs")
    assert roles["base.Q_pred"] == ("component", "total_cfs")
    assert roles["rdii.rdii_component_2"] == ("component", "rdii.rdii_cfs")
    assert roles["ia.p_excess_in"][0] == "flux"
    checked = assert_components_sum(model, trace)
    assert checked == ["rdii.rdii_cfs", "total_cfs"]
    np.testing.assert_array_equal(trace["total_cfs"], outputs["total_cfs"])
    # a composite inside a composite keeps full addresses
    outer = CompositeModel({"inner": model}, outputs={"q": "inner.total_cfs"})
    outer.initialize()
    outer.validate()
    outer.prepare(forcing())
    deep = outer.predict_trace()
    np.testing.assert_allclose(deep["inner.rdii.rdii_component_1"], trace["rdii.rdii_component_1"])


def test_abstraction_uh_preset_traces_its_parts() -> None:
    model = AbstractionUHModel(uh=PeakTailUH())
    _, trace = run(model)
    assert {"uh.peak_Q_pred", "uh.tail_Q_pred", "abstraction.p_excess_in"} <= set(trace)
    assert "uh.Q_pred" in model.trace_field_names
    assert_components_sum(model, trace)


def test_tracing_does_not_change_later_predictions() -> None:
    model = RDIIModel(uh_components=[RTKTriangle(), NashUH()])
    first, _ = run(model)
    again = model.predict_arrays()
    for k in first:
        np.testing.assert_array_equal(np.asarray(first[k]), np.asarray(again[k]))


# ---------------------------------------------------------------------------- internal states


def test_initial_abstraction_capacity_is_traced_and_matches_the_reference() -> None:
    from sparsehydro.models.rdii import initial_abstraction as ia_mod

    model = IAModel()
    outputs, trace = run(model)
    ia = trace["ia_available"]
    ia_max = model.get_scalar_parameter("ia_max").value
    assert model.get_trace_field("ia_available").role == "state"
    assert np.all((ia >= 0.0) & (ia <= ia_max))
    rain = forcing()["rainfall_in"].to_numpy()
    assert ia[39] < ia[19] and ia[199] > ia[39]  # depleted by the first storm, recovered before the second
    # compiled and reference loops record the same series, bit for bit
    cold_then_warm = np.where(np.arange(len(rain)) < 300, -2.0, 8.0)  # snow, then melt
    kw = dict(rainfall_mm=rain, dt_hours=np.full(len(rain), 5 / 60), temperature=cold_then_warm,
              ia_max=0.2, k0=0.05, kT=0.02, theta=0.1, T_ref=20.0, k_dep=7.62, T_freeze=0.0,
              snow_ddf=0.1, snow_T=3.0)
    fast, slow = {}, {}
    np.testing.assert_array_equal(ia_mod._compute_excess_series_fast(record=fast, **kw),
                                  IAModel.compute_excess_series(record=slow, **kw))
    for key in ia_mod.IA_RECORD_KEYS:
        np.testing.assert_array_equal(fast[key], slow[key])
    assert fast["swe"].max() > 0 and fast["melt"].sum() > 0 and fast["snowfall"].sum() > 0
    # recording does not change the excess
    np.testing.assert_array_equal(ia_mod._compute_excess_series_fast(**kw), ia_mod._compute_excess_series_fast(
        record={}, **kw))


def test_snow_states_appear_only_with_snow() -> None:
    plain = IAModel()
    plain.initialize()
    assert plain.trace_field_names == ["ia_available"]
    snowy = IAModel(snow=True)
    data = forcing()
    data["temperature_c"] = -2.0
    _, trace = run(snowy, data)
    assert {"swe", "snowfall", "melt"} <= set(trace) and trace["swe"].max() > 0


@pytest.mark.parametrize("name", ["tank-constant", "tank-linear", "tank-sqrt"])
def test_tank_storage_and_gain(name) -> None:
    model = registry.create(name, V_tank=0.3, ae_min=2.0, ae_max=8.0, drain=0.02)
    outputs, trace = run(model)
    V, gain = trace["tank_storage"], trace["tank_gain"]
    assert np.all((V >= 0) & (V <= 0.3)) and V.max() > 0
    assert np.all((gain >= 2.0 - 1e-12) & (gain <= 8.0 + 1e-12))
    rain = forcing()["rainfall_in"].to_numpy()
    np.testing.assert_array_equal(outputs["p_excess_in"], np.maximum(rain * gain, 0.0))


def test_existing_store_outputs_carry_their_roles() -> None:
    from sparsehydro.models.abstraction import SoilGroundwaterStore
    from sparsehydro.models.forcing import DegreeDaySnow

    store = SoilGroundwaterStore()
    store.initialize()
    roles = {f.name: f.role for f in store.output_fields}
    assert roles["S_in"] == roles["G_in"] == "state"
    assert roles["recharge_in"] == roles["gwi_in"] == "flux" and roles["gwi_cfs"] == "output"
    snow = DegreeDaySnow()
    snow.initialize()
    assert {f.name: f.role for f in snow.output_fields}["SWE_in"] == "state"
    # inside a composite the states keep their role
    comp = build_model({"type": "composite", "components": {
        "snow": {"type": "degree-day-snow"},
        "store": {"type": "soil-groundwater-store", "inputs": {"liquid_input_in": "snow.liquid_input_in"}}},
        "outputs": {"gwi_cfs": "store.gwi_cfs"}})
    meta = {f.name: f.role for f in comp.trace_fields}
    assert meta["store.S_in"] == meta["snow.SWE_in"] == "state" and meta["store.recharge_in"] == "flux"


def test_rdii_forwards_abstraction_states() -> None:
    model = RDIIModel(ia_model=IAModel(snow=True))
    _, trace = run(model)
    assert {"ia.ia_available", "ia.swe", "ia.melt"} <= set(model.trace_field_names)
    assert model.get_trace_field("ia.swe").role == "state"
    assert len(trace["ia.ia_available"]) == len(trace["rdii_cfs"])


def test_metadata_table() -> None:
    model = RDIIModel(ia_model=ExternalExcess(source_column="rainfall_in"))
    model.initialize()
    table = model.trace_field_metadata()
    assert list(table.columns) == ["field", "role", "component_of", "units", "description"]
    assert set(table["role"]) == {"component"} and set(table["component_of"]) == {"rdii_cfs"}
    assert model.field_record("rdii_cfs").units == "CFS"
    assert model.field_record("rdii_component_1").role == "component"
    ia = IAModel()
    ia.initialize()
    assert ia.trace_field_metadata().empty or set(ia.trace_field_metadata()["role"]) <= {"state", "flux"}
