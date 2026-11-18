"""Whole-flow presets must reproduce the pinned whole-flow results.

``tests/golden/wholeflow.npz`` was written from the original inheritance-based
``CompactWholeFlowModel`` and ``PatternWholeFlowModel``. Parameter names, order,
bounds, calibrate flags and constraint names must match exactly; storage
states, flows and budgets to round-off (convolution branch and operation order
may differ).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.golden import wholeflow_cases as W

PINS = Path(__file__).resolve().parent / "golden" / "wholeflow.npz"
STATE_COLUMNS = {"S_in", "G_in", "excess_in", "et_in", "liquid_input_in", "SWE_in"}


@pytest.fixture(scope="module")
def pins():
    if not PINS.exists():
        pytest.skip("wholeflow.npz not generated")
    with np.load(PINS) as z:
        return {k: z[k] for k in z.files}


@pytest.fixture(scope="module")
def data():
    return W.forcing()


@pytest.mark.parametrize("case", W.CASES)
def test_parameter_and_constraint_tables_match(case, pins, data):
    model = W.build(case, data)
    names = model.scalar_parameter_names
    assert names == list(pins[f"{case}|param_names"])
    np.testing.assert_array_equal([model.get_scalar_parameter(n).lower_bound for n in names],
                                  pins[f"{case}|lower"])
    np.testing.assert_array_equal([model.get_scalar_parameter(n).upper_bound for n in names],
                                  pins[f"{case}|upper"])
    np.testing.assert_array_equal([model.get_scalar_parameter(n).calibrate for n in names],
                                  pins[f"{case}|calibrate"])
    assert model.inequality_constraint_names == list(pins[f"{case}|constraints"])


@pytest.mark.parametrize("vector", [0, 1])
@pytest.mark.parametrize("case", W.CASES)
def test_predictions_budgets_and_residuals_match(case, vector, pins, data):
    model = W.build(case, data)
    names = list(pins[f"{case}|param_names"])
    values = dict(zip(names, pins[f"{case}|x{vector}|values"]))
    prediction, budget, residuals = W.run(model, data, values)

    assert set(pins[f"{case}|x{vector}|columns"]) <= set(prediction.columns)
    np.testing.assert_allclose([budget[k] for k in W.BUDGET_KEYS[:-1]],
                               pins[f"{case}|x{vector}|budget"][:-1], rtol=1e-9, atol=1e-12)
    assert budget["max_abs_error_in"] < 1e-8
    np.testing.assert_allclose(residuals, pins[f"{case}|x{vector}|residuals"], rtol=1e-12, atol=1e-14)
    for col in W.PINNED_COLUMNS:
        key = f"{case}|x{vector}|{col}"
        if key not in pins:
            continue
        got = prediction[col].to_numpy(dtype=float)
        if col in STATE_COLUMNS:
            np.testing.assert_allclose(got, pins[key], rtol=1e-12, atol=1e-15, err_msg=key)
        else:
            np.testing.assert_allclose(got, pins[key], rtol=1e-9, atol=1e-10, err_msg=key)
