"""Regression coverage for scalar Platypus CMA-ES and bundled RDII routing."""

import copy
import importlib
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
from types import ModuleType

import numpy as np
import pytest

from sparsehydro.calibration import PlatypusCmaEsSolver
from sparsehydro.calibration.problem import CalibrationProblem


class QuadraticProblem(CalibrationProblem):
    def __init__(self):
        self._param_names = ["small", "large"]
        self._xl = np.array([0.0001, -1e6])
        self._xu = np.array([0.0002, 1e6])
        self.calls = 0

    @property
    def objective_names(self):
        return ["error", "skill"]

    @property
    def minimize_flags(self):
        return [True, False]

    def evaluate(self, x, penalty_weight=1e6):
        self.calls += 1
        z = (np.asarray(x) - self._xl) / (self._xu - self._xl)
        error = float(np.sum((z - np.array([0.3, 0.7])) ** 2))
        return np.array([error, error - 1])

    def make_copy(self):
        return copy.deepcopy(self)


def solver(**kwargs):
    return PlatypusCmaEsSolver(
        n_trials=kwargs.pop("n_trials", 53), offspring_size=8,
        progress_every=0, verbose=False, **kwargs,
    )


def test_real_cmaes_exact_budget_full_history_and_model_isolation(monkeypatch):
    import platypus

    original = platypus.CMAES
    instances = []

    def create(*args, **kwargs):
        algorithm = original(*args, **kwargs)
        instances.append(algorithm)
        return algorithm

    monkeypatch.setattr(platypus, "CMAES", create)
    problem = QuadraticProblem()
    fit = solver(seeds=[9, 3])
    result = fit.solve(problem)
    assert len(instances) == 2
    assert all(algorithm.problem.nobjs == 1 for algorithm in instances)
    assert problem.calls == 0
    assert result.pareto_X.shape == (1, 2)
    assert result.pareto_F.shape == (1, 2)
    assert len(result.history) == 106
    assert [r.generation for r in result.history] == list(range(1, 107))
    for record in result.history:
        assert np.all(record.X >= problem.bounds[0])
        assert np.all(record.X <= problem.bounds[1])
        np.testing.assert_allclose(problem.evaluate(record.X[0]), record.F[0])
    assert [entry["n_trials"] for entry in fit.seed_solutions_] == [53, 53]
    assert sum(entry["selected"] for entry in fit.seed_solutions_) == 1
    np.testing.assert_allclose(problem.evaluate(result.pareto_X[0]), result.pareto_F[0])


def test_converges_with_mixed_parameter_scales_and_maximized_objective():
    fit = solver(n_trials=600, objective="skill")
    result = fit.solve(QuadraticProblem())
    assert result.pareto_F[0, 0] < 1e-10
    assert result.objective_display_values()[0, 1] > 1 - 1e-10
    assert fit.objective_optimized_ == "skill"


def test_reproducibility_and_rng_states_are_preserved():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    fit = solver()
    first = fit.solve(QuadraticProblem())
    second = fit.solve(QuadraticProblem())
    np.testing.assert_array_equal(first.pareto_X, second.pareto_X)
    assert random.getstate() == python_state
    after = np.random.get_state()
    assert after[0] == numpy_state[0]
    np.testing.assert_array_equal(after[1], numpy_state[1])
    assert after[2:] == numpy_state[2:]


def test_initial_points_are_scored_and_count_toward_budget():
    problem = QuadraticProblem()
    optimum = problem.bounds[0] + np.array([0.3, 0.7]) * (problem.bounds[1] - problem.bounds[0])
    result = solver(initial_points=[optimum]).solve(problem)
    assert len(result.history) == 53
    np.testing.assert_allclose(result.pareto_X[0], optimum)
    assert result.pareto_F[0, 0] < 1e-28


def test_parallel_matches_sequential_including_seed_selection():
    pytest.importorskip("joblib")
    sequential = solver(seeds=[7, 8])
    parallel = solver(seeds=[7, 8], n_jobs=2)
    expected = sequential.solve(QuadraticProblem())
    actual = parallel.solve(QuadraticProblem())
    np.testing.assert_array_equal(expected.pareto_X, actual.pareto_X)
    np.testing.assert_array_equal(expected.pareto_F, actual.pareto_F)
    assert sequential.selected_seed_ == parallel.selected_seed_
    assert len(actual.history) == 106


@pytest.mark.parametrize("options", [
    {"seeds": []}, {"seeds": [2, 2]}, {"multi_start": 0}, {"n_jobs": 0},
    {"n_trials": 1}, {"objective_index": -1}, {"objective_index": 2},
    {"initial_points": [[1, 2]]}, {"unknown": 2},
])
def test_invalid_options_raise(options):
    with pytest.raises((TypeError, ValueError)):
        solver().solve(QuadraticProblem(), **options)


def test_worker_failure_is_not_hidden():
    class Broken(QuadraticProblem):
        def evaluate(self, x, penalty_weight=1e6):
            raise RuntimeError("model evaluation failed")

    with pytest.raises(RuntimeError, match="model evaluation failed"):
        solver().solve(Broken())


@pytest.fixture
def rdii(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "swmmrdii"))
    # Isolate solver routing from the bundled legacy model's removed factory API.
    model_module = ModuleType("swmmrdii.model")
    monkeypatch.setattr(model_module, "RdiiModel", FakeRdii, raising=False)
    monkeypatch.setitem(sys.modules, "swmmrdii.model", model_module)
    return importlib.import_module("swmmrdii.calibration")


class FakeTemplate:
    scalar_parameter_names = ["R_1", "scale"]

    def get_scalar_parameter(self, name):
        return SimpleNamespace(value={"R_1": 0.2, "scale": 2.0}[name])


class FakeRdii:
    name = "test"
    _param_meta = [("R_1", 0.01, 0.8, False), ("scale", 0.1, 10.0, True)]
    _templates = [FakeTemplate()]


def fake_optimizer(rdii, **kwargs):
    owner = rdii.CalOptimizer(FakeRdii(), objective=kwargs.pop("objective", "mse"),
                              seed_selection="objective", **kwargs)
    owner._r_sorted = ["R_1"]
    owner._obs_cal = np.array([1., 2., 3., 4.])
    owner._obs_cal_mean = owner._obs_cal.mean()
    owner._obs_cal_std = owner._obs_cal.std()
    owner._obs_cal_var = owner._obs_cal.var()
    owner._simulate_cal = lambda params, forcing: (
        owner._obs_cal * (params["R_1"] + params["scale"])
    )
    owner._store_selected_diagnostics = lambda forcing: None
    return owner


def test_rdii_aliases_retain_nsga2_and_reject_removed_sampler(rdii):
    for alias in ("NSGAII", "nsga-ii", "nsga2", "nsga_ii"):
        assert rdii._resolve_sampler(alias) == "sh_nsga2"
    with pytest.raises(ValueError, match="Unknown sampler"):
        rdii._resolve_sampler("TPE")


@pytest.mark.parametrize("sampler", ["CMA-ES", "NSGAII", "scipy", "sh_pso", "sh_platypus"])
def test_rdii_real_solver_routing_and_physical_result_units(rdii, sampler):
    import platypus

    owner = fake_optimizer(rdii, sampler=sampler, n_trials=24, n_startup=4)
    options = {"algorithm_class": platypus.NSGAII, "population_size": 4} if sampler == "sh_platypus" else {}
    if sampler == "scipy":
        options = {"popsize": 3, "polish": False}
    owner._optimize_sparsehydro([], owner._obs_cal, owner._obs_cal_mean,
                               owner._obs_cal_std, owner._obs_cal_var,
                               **options)
    result = owner.study
    assert np.isfinite(result.pareto_F).all()
    assert 0.01 <= owner.best_params["R_1"] <= 0.8
    assert 0.1 <= owner.best_params["scale"] <= 10
    for record in result.history:
        assert np.all((record.X[:, 1] >= 0.1) & (record.X[:, 1] <= 10))
    actual = np.mean((owner._simulate_cal(owner.best_params, []) - owner._obs_cal) ** 2)
    assert owner.cal_nse == pytest.approx(1 - actual / owner._obs_cal_var)


def test_rdii_warmstart_and_robust_seed_screening_are_used(rdii):
    owner = fake_optimizer(rdii, sampler="CMA-ES", n_trials=24, n_startup=4,
                           multi_start_seeds=[2, 3])
    owner.seed_selection = "robust"
    owner._warmstart_params = {"R_1": 0.25, "scale": 0.75}
    seen = []

    def screen(candidates, forcing):
        seen.extend(candidates)
        return 1, [{"seed": 3}], "screened calibration-only"

    owner._screen_seeds = screen
    owner._optimize_sparsehydro([], owner._obs_cal, owner._obs_cal_mean,
                               owner._obs_cal_std, owner._obs_cal_var)
    assert len(seen) == 2
    assert owner.best_seed == 3
    assert owner.seed_selection_note == "screened calibration-only"
    assert owner.study.pareto_F[0, 0] < 1e-28
    assert len(owner.study.history) == 48


@pytest.mark.parametrize("gate", ["watershed", "subcatchment"])
def test_rdii_rejects_an_entirely_infeasible_search(rdii, gate):
    owner = fake_optimizer(rdii, sampler="CMA-ES", n_trials=8, n_startup=4,
                           seed_initial=False)
    if gate == "watershed":
        owner.model._param_meta = [("R_1", 1.1, 1.3, False), ("scale", 0.1, 10., True)]
    else:
        owner._simulate_cal = lambda params, forcing: None
    with pytest.raises(RuntimeError, match="physically infeasible"):
        owner._optimize_sparsehydro([], owner._obs_cal, owner._obs_cal_mean,
                                   owner._obs_cal_std, owner._obs_cal_var)


def test_rdii_maximize_convergence_uses_display_values_and_running_best(rdii):
    owner = fake_optimizer(rdii, objective="ccc", sampler="CMA-ES", n_trials=24, n_startup=4)
    owner._optimize_sparsehydro([], owner._obs_cal, owner._obs_cal_mean,
                               owner._obs_cal_std, owner._obs_cal_var)
    with pytest.warns(UserWarning, match="importance plots"):
        plots = owner.plot_diagnostics(param_importances=True)
    convergence, _ = plots[0]
    expected = owner.study.to_pareto_dataframe().ccc.cummax().to_numpy()
    np.testing.assert_array_equal(convergence.data[0].y, expected)


def test_current_notebook_runs_cmaes_from_its_settings_page(tmp_path):
    from tests import test_configuration as notebook_support
    from tests.test_configuration import PATTERN, build, forcing, notebook_namespace
    from sparsehydro.calibration.msdgc_assessment import MSDGCSettings
    from sparsehydro.calibration.objectives import ConcordanceCorrelationCoefficient, NashSutcliffe, PBIAS
    from sparsehydro.configuration import GROUNDWATER_PARAMETERS, ModelSettings, StudySettings
    from sparsehydro.models.rdii.initial_abstraction import snow_partition

    path = (Path(__file__).resolve().parents[1] / "sparsehydro" / "notebooks"
            / "flowfinity_gwi_wholeflow_pattern_joint.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    sources = {c["id"]: "".join(c["source"]) for c in notebook["cells"]}
    ns = notebook_namespace(notebook)
    template = build(StudySettings(model=ModelSettings(n_triangles=4, seasonal_baseline="off")))
    overrides = {
        name: {"calibrate": name in ("sanitary_mean_cfs", "R_4")}
        for name in template.scalar_parameter_names if name not in GROUNDWATER_PARAMETERS
    }
    data = forcing()
    ns.update(
        GROUNDWATER_ENABLED=False, SEASONAL_BASELINE="off", N_RTK_TRIANGLES=4,
        SNOW_ENABLED=False, CALIBRATE_SNOW=False, PARAMETER_OVERRIDES=overrides,
        SOLVER="cmaes", SCORING_MODE="single", SINGLE_OBJECTIVE="nash_sutcliffe",
        CMA_ES_SETTINGS=dict(cma_n_trials=21, cma_multi_start=2, cma_n_jobs=1,
                             cma_seed=42, cma_offspring_size=4, cma_progress_every=0),
        EVENT_SOURCE="auto", odwf_cfs=.8, _dt_hours=1., model_df=data,
        sanitary_pattern=PATTERN, ARTIFACT_DIR=tmp_path,
        ConcordanceCorrelationCoefficient=ConcordanceCorrelationCoefficient,
        NashSutcliffe=NashSutcliffe, PBIAS=PBIAS,
        msdgc_settings=MSDGCSettings(minimum_minutes=(60., 20., 30.),
                                   maximum_minutes=(60., 240., 1440.)),
    )
    if hasattr(notebook_support, "build_model_state"):
        notebook_support.build_model_state(ns)
    else:
        for cell in ("pattern-joint-22", "pattern-joint-30", "pattern-joint-47"):
            exec(sources[cell], ns)
    data["flow_cfs"] = ns["joint_model"].predict_total_cfs() * 1.03
    data["stormflow"] = data.flow_cfs - .6
    for column, values in snow_partition(
            data.rainfall_in.to_numpy(), data.temperature_c.to_numpy(), 1., None).items():
        data[column] = values
    train = np.arange(len(data)) < 120
    import pandas as pd

    ns.update(
        train_mask=train, val_mask=~train, storm_mask=train.copy(),
        acceptance_events=pd.DataFrame({
            "event_id": ["train", "validation"], "start_idx": [48, 144],
            "end_idx": [72, 167], "duration_hours": [24., 23.], "split": ["train", "validation"],
        }),
    )
    if hasattr(notebook_support, "calibrate_state"):
        notebook_support.calibrate_state(ns)
    else:
        exec(sources["pattern-joint-49"], ns)
    assert isinstance(ns["solver2"], PlatypusCmaEsSolver)
    assert ns["joint_result"].pareto_X.shape == (1, 2)
    assert len(ns["joint_result"].history) == 42
    assert ns["calibration_config"].solver_metadata()["solver"] == "platypus.CMAES"
