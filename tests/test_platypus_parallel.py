import threading
import time

import numpy as np
import pandas as pd
import pytest

platypus = pytest.importorskip("platypus")

from sparsehydro.calibration.objectives import MSE, NashSutcliffe
from sparsehydro.calibration.problem import CalibrationProblem
from sparsehydro.calibration.solvers.platypus_solver import PlatypusSolver
from sparsehydro.enums import ModelState
from sparsehydro.models import IModel
from sparsehydro.parameters import ScalarParameter


class ThreadCheckedModel(IModel):
    model_name = "test-thread-checked"

    def initialize(self):
        self.register_scalar_parameter(ScalarParameter("slope", 1.0, .1, 4.0))
        self.owner = None
        self._state = ModelState.INITIALIZED

    def validate(self):
        return self.parameters_valid()

    def prepare(self, data):
        self.values = data["x"].to_numpy()
        self._state = ModelState.PREPARED

    def predict(self):
        if self.owner is None:
            self.owner = threading.get_ident()
        assert self.owner == threading.get_ident(), "Mutable model shared between threads"
        slope = self.get_scalar_parameter("slope").value
        time.sleep(.001 * (1 + int(slope * 10) % 3))
        assert self.get_scalar_parameter("slope").value == slope
        return pd.DataFrame({"prediction": slope * self.values})

    def finalize(self):
        self._state = ModelState.FINALIZED


def make_problem(predict_fn=None):
    data = pd.DataFrame({"x": np.arange(1, 33, dtype=float)})
    data["observed"] = data.x * 2.5 + .1 * np.sin(data.x)
    model = ThreadCheckedModel()
    model.initialize()
    return CalibrationProblem(
        model, data, objectives=[MSE(), NashSutcliffe()],
        column_map={"observed": "observed", "predicted": "prediction"},
        predict_fn=predict_fn,
    )


def assert_equal_results(left, right):
    np.testing.assert_array_equal(left.pareto_X, right.pareto_X)
    np.testing.assert_array_equal(left.pareto_F, right.pareto_F)
    assert left.knee_index() == right.knee_index()
    assert len(left.history) == len(right.history) > 0
    assert left.param_names == right.param_names
    assert left.objective_names == right.objective_names
    assert left.minimize_flags == right.minimize_flags
    for a, b in zip(left.history, right.history):
        assert a.generation == b.generation
        assert a.n_pareto == b.n_pareto
        np.testing.assert_array_equal(a.X, b.X)
        np.testing.assert_array_equal(a.F, b.F)


@pytest.mark.parametrize("workers", [2, 4])
@pytest.mark.parametrize("seed", [7, 42])
def test_ordered_workers_preserve_history_callbacks_and_original_model(workers, seed):
    problem = make_problem()
    serial_callbacks, parallel_callbacks = [], []
    serial = PlatypusSolver(
        platypus.NSGAII, population_size=12, n_evaluations=48, seed=seed,
        record_frequency=2, callback=serial_callbacks.append,
    ).solve(problem)
    barrier = threading.Barrier(workers, timeout=10)
    seen = {}

    def predict(model):
        if model.owner is None:
            seen[threading.get_ident()] = (model, threading.current_thread())
            barrier.wait()
        return model.predict()

    problem._predict_fn = predict
    main_thread = threading.get_ident()

    def callback(event):
        assert threading.get_ident() == main_thread
        parallel_callbacks.append(event)

    parallel = PlatypusSolver(
        platypus.NSGAII, population_size=12, n_evaluations=48, seed=seed,
        record_frequency=2, n_workers=workers, callback=callback,
    ).solve(problem)
    assert_equal_results(serial, parallel)
    assert len(seen) == workers
    assert len({id(model) for model, _ in seen.values()}) == workers
    assert all(model is not problem._model for model, _ in seen.values())
    assert all(not thread.is_alive() for _, thread in seen.values())
    assert problem._model.owner is None
    assert problem._model.get_scalar_parameter("slope").value == 1.0
    assert problem._model._state == ModelState.PREPARED
    assert len(serial_callbacks) == len(parallel_callbacks) > 0
    for left, right in zip(serial_callbacks, parallel_callbacks):
        np.testing.assert_array_equal(left.pop("pareto_F"), right.pop("pareto_F"))
        assert left == right


def test_per_call_worker_override_preserves_serial_path():
    threads = []

    def predict(model):
        threads.append(threading.current_thread())
        return model.predict()

    problem = make_problem(predict)
    solver = PlatypusSolver(platypus.NSGAII, population_size=8,
                            n_evaluations=16, n_workers=4)
    solver.solve(problem, n_workers=1)
    assert threads and all(thread is threading.current_thread() for thread in threads)


def test_process_workers_preserve_serial_results_and_original_model():
    problem = make_problem()
    options = dict(population_size=8, n_evaluations=16, seed=7)
    serial = PlatypusSolver(platypus.NSGAII, **options).solve(problem)
    parallel = PlatypusSolver(platypus.NSGAII, n_jobs=2, **options).solve(problem)
    assert_equal_results(serial, parallel)
    assert problem._model.owner is None
    assert problem._model.get_scalar_parameter("slope").value == 1.0
    assert problem._model._state == ModelState.PREPARED


@pytest.mark.parametrize("per_call", [False, True])
def test_process_and_thread_workers_cannot_both_be_enabled(per_call):
    solver = PlatypusSolver(platypus.NSGAII, n_jobs=2,
                            n_workers=1 if per_call else 2)
    with pytest.raises(ValueError, match="n_jobs and n_workers"):
        solver.solve(make_problem(), **({"n_workers": 2} if per_call else {}))


@pytest.mark.parametrize("workers", [0, -1, True, None, 1.5, "4"])
def test_invalid_worker_count_fails_before_evaluation(workers):
    with pytest.raises(ValueError, match="positive integer"):
        PlatypusSolver(platypus.NSGAII, n_workers=workers).solve(make_problem())
    with pytest.raises(ValueError, match="positive integer"):
        PlatypusSolver(platypus.NSGAII).solve(make_problem(), n_workers=workers)


def test_custom_evaluator_is_preserved_serially_but_rejected_with_threads():
    class CountingEvaluator(platypus.MapEvaluator):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def evaluate_all(self, jobs, **kwargs):
            self.calls += 1
            return super().evaluate_all(jobs, **kwargs)

    evaluator = CountingEvaluator()
    solver = PlatypusSolver(platypus.NSGAII, population_size=8,
                            n_evaluations=16, evaluator=evaluator)
    solver.solve(make_problem())
    assert evaluator.calls > 0
    with pytest.raises(ValueError, match="custom evaluator"):
        solver.solve(make_problem(), n_workers=4)


@pytest.mark.parametrize("failure", ["prediction", "callback"])
def test_errors_propagate_and_worker_threads_are_joined(failure):
    threads = set()
    lock = threading.Lock()

    def predict(model):
        with lock:
            threads.add(threading.current_thread())
        if failure == "prediction":
            raise ArithmeticError("Injected prediction failure")
        return model.predict()

    def callback(event):
        raise ArithmeticError("Injected callback failure")

    with pytest.raises(ArithmeticError, match=f"Injected {failure} failure"):
        PlatypusSolver(
            platypus.NSGAII, population_size=8, n_evaluations=16, n_workers=4,
            callback=callback,
        ).solve(make_problem(predict))
    assert threads and all(not thread.is_alive() for thread in threads)
