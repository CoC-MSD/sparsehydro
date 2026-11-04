import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sparsehydro.calibration.objectives import ConcordanceCorrelationCoefficient, NashSutcliffe
from sparsehydro.calibration.solvers.platypus_solver import PlatypusSolver
from sparsehydro.calibration.wholeflow_workflow import FeasibleCalibrationProblem
from sparsehydro.enums import ModelState
from sparsehydro.models import wholeflow_compact
from sparsehydro.models.rdii.initial_abstraction import SnowConfig
from sparsehydro.models.wholeflow_compact import CompactWholeFlowModel, SanitaryTemplate
from sparsehydro.models.wholeflow_pattern import CenteredWeeklyPattern, PatternWholeFlowModel


@pytest.fixture(params=["compact", "pattern"])
def make_model(request):
    def make(dt=1.0, snow=False):
        config = SnowConfig(initial_swe_in=.1) if snow else None
        if request.param == "compact":
            model = CompactWholeFlowModel(
                SanitaryTemplate((1.0,) * 24, (1.0,) * 7),
                20.26, dt, .6, 2.0, snow_config=config,
            )
        else:
            hourly = .1 * np.sin(np.arange(24) * np.pi / 12)
            model = PatternWholeFlowModel(
                CenteredWeeklyPattern(tuple(hourly), (0.0,) * 7),
                20.26, dt, .6, 2.0, snow_config=config,
            )
        model.initialize()
        return model
    return make


def forcing(dt=1.0):
    n = int(240 / dt)
    rain = np.zeros(n)
    rain[[1, int(72 / dt), int(192 / dt), n - 2]] = [.8, .3, 2.0, .6]
    elapsed = np.arange(n) * dt
    return pd.DataFrame({
        "datetime": pd.date_range("2025-01-01", periods=n,
                                 freq=pd.Timedelta(hours=dt)),
        "rainfall_in": rain,
        "temperature_c": np.where(elapsed < 48, -5.0, 8.0),
        "flow_cfs": .8 + .1 * np.sin(elapsed * np.pi / 12) + rain,
    })


@pytest.mark.parametrize("dt", [1 / 12, .25, 1.0])
@pytest.mark.parametrize("snow", [False, True])
@pytest.mark.parametrize("scenario", ["default", "short", "long", "spill", "dry"])
def test_lean_is_exact_and_full_diagnostics_remain_available(make_model, dt, snow, scenario):
    model = make_model(dt, snow)
    data = forcing(dt)
    if scenario in ("short", "long"):
        for i in (1, 2, 3):
            for name in (f"uh{i}_T", f"uh{i}_K"):
                parameter = model.get_scalar_parameter(name)
                parameter.value = (parameter.lower_bound if scenario == "short"
                                   else parameter.upper_bound)
    elif scenario == "spill":
        for name, value in (("g_max", .01), ("k_G", 1e-5), ("k_div", .05),
                            ("area_acres", 26.0), ("s_initial_fraction", .5),
                            ("g_initial_fraction", .9)):
            model.get_scalar_parameter(name).value = value
    elif scenario == "dry":
        data["rainfall_in"] = 0.0
    model.prepare(data)
    snow_forcing = model._snow_forcing.copy()
    full = model.predict()
    budget = model.budget.copy()
    lean = model.predict_total_cfs()
    assert lean.shape == (len(data),)
    assert lean.dtype == np.dtype("float64")
    np.testing.assert_array_equal(lean, full.total_cfs.to_numpy())
    assert model.budget == budget
    assert model._state == ModelState.PREDICTED
    pd.testing.assert_frame_equal(full, model.predict(), check_exact=True)
    pd.testing.assert_frame_equal(snow_forcing, model._snow_forcing, check_exact=True)
    lean[0] += 1
    np.testing.assert_array_equal(model.predict_total_cfs(), full.total_cfs.to_numpy())


def test_lean_does_not_construct_full_output(make_model, monkeypatch):
    model = make_model()
    model.prepare(forcing())
    expected = model.predict().total_cfs.to_numpy()

    def diagnostic_not_allowed(*args, **kwargs):
        raise AssertionError("Lean prediction must not construct diagnostic output")

    monkeypatch.setattr(wholeflow_compact, "pd", SimpleNamespace(DataFrame=diagnostic_not_allowed))
    monkeypatch.setattr(wholeflow_compact, "_g_at_cap", diagnostic_not_allowed)
    monkeypatch.setattr(CompactWholeFlowModel, "predict", diagnostic_not_allowed)
    np.testing.assert_array_equal(model.predict_total_cfs(), expected)


@pytest.mark.parametrize("method", ["predict", "predict_total_cfs"])
def test_both_paths_enforce_preparation_and_feasibility(make_model, method):
    model = make_model()
    with pytest.raises(ValueError, match="Prepare forcing"):
        getattr(model, method)()
    model.prepare(forcing())
    for name in ("R_1", "R_2", "R_3"):
        model.get_scalar_parameter(name).value = .6
    with pytest.raises(ValueError, match="feasible"):
        getattr(model, method)()


@pytest.mark.parametrize("method", ["predict", "predict_total_cfs"])
def test_both_paths_reject_injected_mass_balance_error(make_model, method, monkeypatch):
    model = make_model(snow=True)
    model.prepare(forcing())
    core = wholeflow_compact._gwi_B_core

    def broken_core(*args):
        s, groundwater, excess, et, gwi = core(*args)
        return s + .001, groundwater, excess, et, gwi

    monkeypatch.setattr(wholeflow_compact, "_gwi_B_core", broken_core)
    with pytest.raises(ArithmeticError, match="budget"):
        getattr(model, method)()


@pytest.mark.parametrize("method", ["predict", "predict_total_cfs"])
def test_both_paths_reject_negative_routing(make_model, method, monkeypatch):
    model = make_model()
    model.prepare(forcing())
    monkeypatch.setattr(wholeflow_compact, "fftconvolve",
                        lambda excess, weights: np.full(len(excess), -1.0))
    with pytest.raises(ArithmeticError, match="negative routed"):
        getattr(model, method)()


def problem_for(model, data, lean):
    train = np.arange(len(data)) < 200
    storm = train & (np.arange(len(data)) >= 60)
    model.get_scalar_parameter("area_acres").calibrate = True
    return FeasibleCalibrationProblem(
        model, data.assign(flow_cfs=data.flow_cfs.where(train)),
        objectives=[ConcordanceCorrelationCoefficient(mask=storm),
                    NashSutcliffe(mask=train)],
        column_map={"observed": "flow_cfs",
                    "predicted": (lambda values: values) if lean else "total_cfs"},
        predict_fn=CompactWholeFlowModel.predict_total_cfs if lean else None,
    )


def test_lean_callback_receives_worker_and_preserves_penalties(make_model, monkeypatch):
    model = make_model()
    problem = problem_for(model, forcing(), lean=True)
    seen = []

    def predict(worker):
        seen.append(worker)
        return worker.predict_total_cfs()

    problem._predict_fn = predict
    worker = problem.make_copy()
    assert isinstance(worker, FeasibleCalibrationProblem)
    assert worker._model is not model
    x = np.array([model.get_scalar_parameter(n).value for n in problem.param_names])
    x[problem.param_names.index("area_acres")] = 25.0

    def full_not_allowed(*args, **kwargs):
        raise AssertionError("Lean calibration must not call full predict")

    monkeypatch.setattr(CompactWholeFlowModel, "predict", full_not_allowed)
    assert np.isfinite(worker.evaluate(x)).all()
    assert len(seen) == 1 and seen[0] is worker._model
    assert model.get_scalar_parameter("area_acres").value == 20.26
    assert worker._model.get_scalar_parameter("area_acres").value == 25.0
    for name in ("R_1", "R_2", "R_3"):
        x[problem.param_names.index(name)] = .6
    assert (worker.evaluate(x) >= 1e6).all()
    assert (problem.evaluate(x) >= 1e6).all()
    assert len(seen) == 1


def test_prediction_callback_errors_are_not_swallowed(make_model):
    problem = problem_for(make_model(), forcing(), lean=True)
    x = np.array([problem._model.get_scalar_parameter(n).value for n in problem.param_names])

    def fail(worker):
        raise ArithmeticError("Injected prediction failure")

    problem._predict_fn = fail
    with pytest.raises(ArithmeticError, match="Injected prediction failure"):
        problem.evaluate(x)
    with pytest.raises(ArithmeticError, match="Injected prediction failure"):
        problem.make_copy().evaluate(x)


@pytest.mark.parametrize("snow", [False, True])
@pytest.mark.parametrize("n_workers", [1, 4])
@pytest.mark.parametrize("seed", [7, 42])
def test_seeded_optimizer_is_exact_for_full_and_lean(make_model, snow, n_workers, seed):
    import platypus

    model = make_model(snow=snow)
    full = problem_for(model, forcing(), lean=False)
    lean = problem_for(copy.deepcopy(model), forcing(), lean=True)
    x = np.array([model.get_scalar_parameter(n).value for n in full.param_names])
    np.testing.assert_array_equal(full.evaluate(x), lean.evaluate(x))
    assert full._model.budget == lean._model.budget
    a, b = [
        PlatypusSolver(platypus.NSGAII, population_size=12,
                       n_evaluations=48, seed=seed, n_workers=workers).solve(problem)
        for problem, workers in ((full, 1), (lean, n_workers))
    ]
    np.testing.assert_array_equal(a.pareto_X, b.pareto_X)
    np.testing.assert_array_equal(a.pareto_F, b.pareto_F)
    assert a.knee_index() == b.knee_index()
    assert len(a.history) == len(b.history) > 0
    for left, right in zip(a.history, b.history):
        np.testing.assert_array_equal(left.X, right.X)
        np.testing.assert_array_equal(left.F, right.F)
        assert left.n_pareto == right.n_pareto
    for problem, result in ((full, a), (lean, b)):
        for name, value in zip(result.param_names, result.pareto_X[result.knee_index()]):
            problem._model.apply_flat_parameter(name, float(value))
    pd.testing.assert_frame_equal(full._model.predict(), lean._model.predict(), check_exact=True)
