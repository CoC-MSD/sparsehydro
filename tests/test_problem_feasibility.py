"""CalibrationProblem: skipping infeasible vectors and copying for workers."""

import numpy as np
import pandas as pd
import pytest

from sparsehydro.metrics.objectives import MSE
from sparsehydro.calibration.problem import CalibrationProblem
from sparsehydro.enums import ModelState
from sparsehydro.models.base import IModel
from sparsehydro.parameters import ConstraintRecord, FieldRecord, ScalarParameter


class Scale(IModel):
    """y = a * x, with the constraint a <= 1; predict() refuses infeasible a."""

    model_name = "test-scale-feasibility"

    def __init__(self):
        super().__init__()
        self.calls = 0

    def initialize(self):
        self.register_scalar_parameter(ScalarParameter("a", 0.5, 0.0, 2.0))
        self.register_output_field(FieldRecord("y"))
        self.register_inequality_constraint(ConstraintRecord("a_leq_1"))
        self._state = ModelState.INITIALIZED

    def validate(self):
        return self.parameters_valid()

    def prepare(self, data):
        self._x = data["x"].to_numpy(float)
        self._state = ModelState.PREPARED

    def inequality_constraints(self):
        return [self.get_scalar_parameter("a").value - 1.0]

    def predict_arrays(self):
        self.calls += 1
        a = self.get_scalar_parameter("a").value
        if a > 1.0:
            raise ValueError("infeasible")
        return {"y": a * self._x}

    def predict(self):
        return pd.DataFrame(self.predict_arrays())

    def finalize(self):
        pass


def problem(**kwargs):
    model = Scale()
    model.initialize()
    data = pd.DataFrame({"x": np.arange(1.0, 6.0), "obs": 0.8 * np.arange(1.0, 6.0)})
    return CalibrationProblem(model, data, [MSE()],
                              column_map={"observed": "obs", "predicted": "y"}, **kwargs)


def test_infeasible_vectors_are_penalised_without_running_the_model():
    p = problem(skip_infeasible=True)
    calls = p._model.calls
    f = p.evaluate(np.array([1.5]), penalty_weight=10.0)
    np.testing.assert_allclose(f, [10.0 * (1 + 0.5 ** 2)])
    assert p.evaluate_single(np.array([1.5]), 0, penalty_weight=10.0) == pytest.approx(12.5)
    assert p._model.calls == calls
    assert p.evaluate(np.array([0.8]))[0] == pytest.approx(0.0)


def test_without_skip_infeasible_the_model_runs():
    with pytest.raises(ValueError, match="infeasible"):
        problem().evaluate(np.array([1.5]))


def test_make_copy_preserves_subclass_and_flag():
    class Custom(CalibrationProblem):
        pass

    model = Scale()
    model.initialize()
    data = pd.DataFrame({"x": [1.0, 2.0], "obs": [1.0, 2.0]})
    original = Custom(model, data, [MSE()], column_map={"observed": "obs", "predicted": "y"},
                      skip_infeasible=True)
    worker = original.make_copy()
    assert type(worker) is Custom
    assert worker._skip_infeasible
    assert worker._model is not original._model
