"""Building models, objectives and solvers from configuration data."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sparsehydro.calibration.objectives import ConcordanceCorrelationCoefficient, NashSutcliffe
from sparsehydro.calibration.solvers import PlatypusCmaEsSolver
from sparsehydro.configuration import build_model, build_objectives, build_solver, load_config
from sparsehydro.models.abstraction import ExternalExcess
from sparsehydro.models.composite import CompositeModel
from sparsehydro.models.rdii import IAModel, RDIIModel
from sparsehydro.models.unithydrograph import ClarkUH, NashUH

CONFIG = {
    "model": {
        "type": "composite",
        "components": {
            "ia": {"type": "initial-abstraction"},
            "rdii": {"type": "rdii", "excess": "ia.p_excess_in",
                     "routing": [{"type": "rtk-triangle", "R": 0.04, "T": 1.0, "K": 2.0},
                                 {"type": "nash-uh", "n": 3, "K": 2.0}]},
        },
        "outputs": {"rdii_cfs": "rdii.rdii_cfs"},
    },
    "parameters": {"rdii.R_2": {"value": 0.02, "upper_bound": 0.4},
                   "rdii.area_acres": {"value": 25.0, "calibrate": False}},
}


def forcing(n: int = 360) -> pd.DataFrame:
    rain = np.zeros(n)
    rain[30:50] = 0.1
    return pd.DataFrame({"datetime": pd.date_range("2024-05-01", periods=n, freq="5min"),
                         "rainfall_in": rain, "temperature_c": 16.0})


def test_config_builds_the_same_model_as_code():
    from sparsehydro.models.rdii import RTKTriangle

    built = build_model(CONFIG)
    assert isinstance(built, CompositeModel)
    assert built.get_scalar_parameter("rdii.R_2").upper_bound == 0.4
    assert built.get_scalar_parameter("rdii.area_acres").calibrate is False

    code = CompositeModel(
        {"ia": IAModel(),
         "rdii": RDIIModel(ia_model=ExternalExcess(),
                           uh_components=[RTKTriangle(R=0.04, T=1.0, K=2.0), NashUH(n=3, K=2.0)])},
        bindings={"rdii.p_excess_in": "ia.p_excess_in"},
        outputs={"rdii_cfs": "rdii.rdii_cfs"},
    )
    code.initialize()
    code.apply_flat_parameter("rdii.R_2", 0.02)
    code.apply_flat_parameter("rdii.area_acres", 25.0)

    data = forcing()
    for m in (built, code):
        assert m.validate()
        m.prepare(data)
    np.testing.assert_array_equal(built.predict_arrays()["rdii_cfs"], code.predict_arrays()["rdii_cfs"])


def test_component_inputs_and_plain_specs():
    cfg = {"type": "composite",
           "components": {"uh": {"type": "clark-uh", "A": 10.0, "Tc": 2.0, "Rs": 1.0,
                                 "inputs": {"rain": "data.rainfall_in"}}},
           "output": {"q": "uh.Q_pred"}}
    m = build_model(cfg)
    assert isinstance(m.component("uh"), ClarkUH)
    m.prepare(forcing())
    assert m.predict_arrays()["q"].max() > 0


def test_json_and_yaml_files_round_trip(tmp_path):
    path = tmp_path / "model.json"
    path.write_text(json.dumps(CONFIG), encoding="utf-8")
    assert load_config(path) == CONFIG
    yaml = pytest.importorskip("yaml")
    ypath = tmp_path / "model.yaml"
    ypath.write_text(yaml.safe_dump(CONFIG), encoding="utf-8")
    assert build_model(load_config(ypath)).scalar_parameter_names == build_model(CONFIG).scalar_parameter_names
    with pytest.raises(ValueError, match="Unsupported"):
        load_config(tmp_path / "model.toml")


def test_configuration_errors_are_specific():
    with pytest.raises(KeyError, match="Registered names"):
        build_model({"type": "no-such-model"})
    with pytest.raises(KeyError, match="Unknown parameter"):
        build_model({"model": {"type": "nash-uh"}, "parameters": {"nope": {"value": 1.0}}})
    with pytest.raises(ValueError, match="within its bounds"):
        build_model({"model": {"type": "nash-uh"}, "parameters": {"n": {"value": 1e6}}})
    with pytest.raises(ValueError, match="Unknown rdii configuration keys"):
        build_model({"type": "rdii", "routes": []})
    with pytest.raises(ValueError, match="Unknown composite configuration keys"):
        build_model({"type": "composite", "components": {"a": {"type": "nash-uh"}}, "outptus": {}})


def test_objectives_and_solvers_by_name():
    storm = np.array([True, False, True])
    objs = build_objectives(["nash_sutcliffe", {"type": "ccc", "mask": "storm"}], masks={"storm": storm})
    assert isinstance(objs[0], NashSutcliffe) and isinstance(objs[1], ConcordanceCorrelationCoefficient)
    np.testing.assert_array_equal(objs[1].mask, storm)
    with pytest.raises(KeyError, match="Unknown mask"):
        build_objectives([{"type": "ccc", "mask": "dry"}])
    with pytest.raises(KeyError, match="No objective named"):
        build_objectives(["not_an_objective"])
    solver = build_solver({"type": "cmaes", "n_trials": 10, "seed": 1})
    assert isinstance(solver, PlatypusCmaEsSolver)
    with pytest.raises(KeyError, match="No solver named"):
        build_solver("annealing")
