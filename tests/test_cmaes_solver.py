"""Regression coverage for the scalar Platypus CMA-ES solver."""

import copy
import random

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


