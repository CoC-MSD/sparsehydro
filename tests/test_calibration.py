"""Migrated workflow calibration regressions using unittest.TestCase.

Covers:
  - IObjective subclasses (MSE, RMSE, MAE, PeakWeightedMSE, NashSutcliffe, KGE)
  - CalibrationProblem (construction, evaluate, make_copy)
  - CalibrationResult (best_by, objective_display_values, to_pareto_dataframe)
  - _identify_pareto helper
  - NSGAIISolver (conditional on pymoo)
  - ScipySolver (conditional on scipy)
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from sparsehydro.calibration import IObjective, KGE, LogNSE, MAE, MSE, RMSE, CalibrationProblem, CalibrationResult, GenerationRecord, NashSutcliffe, PeakWeightedMSE
from sparsehydro.calibration.result import _identify_pareto
from sparsehydro.enums import ModelState
from sparsehydro.models import IModel
from sparsehydro.parameters import ScalarParameter

# ---------------------------------------------------------------------------
# Optional-dependency flags
# ---------------------------------------------------------------------------

try:
    import pymoo  # noqa: F401
    HAS_PYMOO = True
except ImportError:
    HAS_PYMOO = False

try:
    import scipy  # noqa: F401
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import platypus  # noqa: F401
    HAS_PLATYPUS = True
except ImportError:
    HAS_PLATYPUS = False


# ---------------------------------------------------------------------------
# Minimal test model (no RDII dependency)
# ---------------------------------------------------------------------------

class _LinearModel(IModel):
    """y = slope * x + intercept  — for testing CalibrationProblem."""

    model_name = "test_linear_calib"

    def initialize(self) -> None:
        self.register_scalar_parameter(
            ScalarParameter("slope", value=1.0, lower_bound=0.0, upper_bound=10.0)
        )
        self.register_scalar_parameter(
            ScalarParameter("intercept", value=0.0, lower_bound=-5.0, upper_bound=5.0)
        )
        self._state = ModelState.INITIALIZED

    def validate(self) -> bool:
        if self.parameters_valid():
            self._state = ModelState.VALIDATED
            return True
        return False

    def prepare(self, x: np.ndarray, **kwargs) -> None:
        self._x = np.asarray(x, dtype=float)
        self._state = ModelState.PREPARED

    def predict(self, **kwargs) -> pd.DataFrame:
        slope = self.get_scalar_parameter("slope").value
        intercept = self.get_scalar_parameter("intercept").value
        y = slope * self._x + intercept
        self._state = ModelState.PREDICTED
        return pd.DataFrame({"y": y})

    def finalize(self) -> None:
        self._state = ModelState.FINALIZED


def _make_linear_model_and_problem():
    """Return a CalibrationProblem around a prepared _LinearModel."""
    x = np.linspace(0, 10, 20)
    true_slope, true_intercept = 2.5, 1.0
    observed = true_slope * x + true_intercept + np.random.default_rng(0).normal(0, 0.1, len(x))

    model = _LinearModel()
    model.initialize()
    model.validate()
    model.prepare(x)

    _obs = observed  # capture for closure
    problem = CalibrationProblem(
        model=model,
        objectives=[MSE(), NashSutcliffe()],
        column_map={
            "observed":  lambda _: _obs,
            "predicted": lambda df: df["y"].to_numpy(),
        },
    )
    return problem, observed, true_slope, true_intercept


# ===========================================================================
# Test IObjective subclasses
# ===========================================================================

class TestObjectives(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(1)
        self.obs = rng.uniform(1, 10, 50)
        self.pred = rng.uniform(1, 10, 50)

    # --- shape mismatch ---

    def _assert_shape_mismatch(self, obj: IObjective):
        with self.assertRaises(Exception):
            obj.evaluate(np.array([1.0, 2.0]), np.array([1.0]))

    def test_mse_shape_mismatch(self):
        self._assert_shape_mismatch(MSE())

    def test_rmse_shape_mismatch(self):
        self._assert_shape_mismatch(RMSE())

    def test_mae_shape_mismatch(self):
        self._assert_shape_mismatch(MAE())

    # --- perfect predictor ---

    def test_mse_perfect(self):
        self.assertAlmostEqual(MSE().evaluate(self.obs, self.obs), 0.0, places=12)

    def test_rmse_perfect(self):
        self.assertAlmostEqual(RMSE().evaluate(self.obs, self.obs), 0.0, places=12)

    def test_mae_perfect(self):
        self.assertAlmostEqual(MAE().evaluate(self.obs, self.obs), 0.0, places=12)

    def test_nse_perfect(self):
        self.assertAlmostEqual(NashSutcliffe().evaluate(self.obs, self.obs), 1.0, places=12)

    def test_kge_perfect(self):
        self.assertAlmostEqual(KGE().evaluate(self.obs, self.obs), 1.0, places=10)

    # --- minimize flags ---

    def test_mse_minimize(self):
        self.assertTrue(MSE.minimize)

    def test_rmse_minimize(self):
        self.assertTrue(RMSE.minimize)

    def test_mae_minimize(self):
        self.assertTrue(MAE.minimize)

    def test_peak_weighted_mse_minimize(self):
        self.assertTrue(PeakWeightedMSE.minimize)

    def test_nse_maximize(self):
        self.assertFalse(NashSutcliffe.minimize)

    def test_kge_maximize(self):
        self.assertFalse(KGE.minimize)

    # --- MSE / RMSE relationship ---

    def test_rmse_is_sqrt_mse(self):
        mse_val = MSE().evaluate(self.obs, self.pred)
        rmse_val = RMSE().evaluate(self.obs, self.pred)
        self.assertAlmostEqual(rmse_val, mse_val ** 0.5, places=10)

    # --- NSE constant observed raises ---

    def test_nse_constant_obs_raises(self):
        obs = np.ones(10)
        pred = np.ones(10) * 1.5
        with self.assertRaises(ValueError):
            NashSutcliffe().evaluate(obs, pred)

    # --- KGE zero std / zero mean raises ---

    def test_kge_zero_std_raises(self):
        obs = np.ones(10)
        with self.assertRaises(ValueError):
            KGE().evaluate(obs, obs)

    # --- names ---

    def test_objective_names(self):
        self.assertEqual(MSE.name, "mse")
        self.assertEqual(RMSE.name, "rmse")
        self.assertEqual(MAE.name, "mae")
        self.assertEqual(NashSutcliffe.name, "nash_sutcliffe")
        self.assertEqual(KGE.name, "kge")
        self.assertEqual(PeakWeightedMSE.name, "peak_weighted_mse")

    # --- PeakWeightedMSE correct weighting ---

    def test_peak_weighted_mse_perfect(self):
        obs = np.array([1.0, 2.0, 5.0, 1.0])
        self.assertAlmostEqual(PeakWeightedMSE().evaluate(obs, obs), 0.0, places=12)

    def test_peak_weighted_mse_positive(self):
        obs = np.array([1.0, 3.0, 5.0, 2.0])
        pred = np.array([1.0, 2.0, 4.0, 2.5])
        self.assertGreater(PeakWeightedMSE().evaluate(obs, pred), 0.0)

    def test_peak_weighted_mse_power_default_matches_linear(self):
        from sparsehydro.calibration.objectives import _peak_weighted_mse
        obs = np.array([1.0, 3.0, 5.0, 2.0])
        pred = np.array([1.0, 2.0, 4.0, 2.5])
        self.assertAlmostEqual(
            PeakWeightedMSE().evaluate(obs, pred),
            _peak_weighted_mse(obs, pred, power=1.0),
            places=12,
        )

    def test_peak_weighted_mse_power_zero_is_plain_mse(self):
        obs = np.array([1.0, 3.0, 5.0, 2.0])
        pred = np.array([1.0, 2.0, 4.0, 2.5])
        self.assertAlmostEqual(
            PeakWeightedMSE(power=0.0).evaluate(obs, pred),
            MSE().evaluate(obs, pred),
            places=12,
        )

    def test_peak_weighted_mse_power_sharpens_peak_focus(self):
        # Same absolute error placed at the peak vs at the lowest flow:
        # raising the power must widen the penalty gap between the two cases.
        obs = np.array([1.0, 2.0, 3.0, 10.0])
        err_at_peak = obs + np.array([0.0, 0.0, 0.0, 1.0])
        err_at_low = obs + np.array([1.0, 0.0, 0.0, 0.0])
        for power in (1.0, 2.0, 3.0):
            ratio = (
                PeakWeightedMSE(power=power).evaluate(obs, err_at_peak)
                / PeakWeightedMSE(power=power).evaluate(obs, err_at_low)
            )
            self.assertGreater(ratio, 1.0)
            if power > 1.0:
                self.assertGreater(ratio, prev_ratio)
            prev_ratio = ratio

    def test_peak_weighted_mse_negative_power_raises(self):
        with self.assertRaises(ValueError):
            PeakWeightedMSE(power=-1.0)

    # --- compute(): masking + NaN handling ---

    def test_compute_no_mask_matches_evaluate(self):
        obj = MSE()
        self.assertAlmostEqual(
            obj.compute(self.obs, self.pred),
            obj.evaluate(self.obs, self.pred),
            places=12,
        )

    def test_compute_explicit_mask_selects_subset(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        pred = np.array([1.0, 9.0, 3.0, 9.0])
        mask = np.array([True, False, True, False])
        # Only positions 0 and 2 are kept, where pred == obs → MSE 0.
        self.assertAlmostEqual(MSE().compute(obs, pred, mask=mask), 0.0, places=12)

    def test_compute_always_drops_nan(self):
        obs = np.array([1.0, np.nan, 3.0, 4.0])
        pred = np.array([1.0, 100.0, 3.0, 4.0])
        # NaN position dropped automatically → remaining are perfect → MSE 0.
        self.assertAlmostEqual(MSE().compute(obs, pred), 0.0, places=12)

    def test_compute_drops_nan_in_predicted(self):
        obs = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.0, np.nan, 3.0])
        self.assertAlmostEqual(MSE().compute(obs, pred), 0.0, places=12)

    def test_compute_uses_objective_self_mask(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        pred = np.array([1.0, 9.0, 3.0, 9.0])
        mask = np.array([True, False, True, False])
        obj = MSE(mask=mask)
        self.assertAlmostEqual(obj.compute(obs, pred), 0.0, places=12)

    def test_compute_explicit_mask_overrides_self_mask(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        pred = np.array([1.0, 9.0, 3.0, 9.0])
        self_mask = np.array([False, True, False, True])   # would select bad points
        override = np.array([True, False, True, False])     # selects perfect points
        obj = MSE(mask=self_mask)
        self.assertAlmostEqual(obj.compute(obs, pred, mask=override), 0.0, places=12)

    def test_compute_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            MSE().compute(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))

    def test_compute_mask_shape_mismatch_raises(self):
        obs = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            MSE().compute(obs, pred, mask=np.array([True, False]))

    def test_compute_empty_selection_raises(self):
        obs = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            MSE().compute(obs, pred, mask=np.array([False, False, False]))

    def test_compute_all_nan_raises(self):
        obs = np.array([np.nan, np.nan])
        pred = np.array([1.0, 2.0])
        with self.assertRaises(ValueError):
            MSE().compute(obs, pred)

    def test_compute_str_mask_raises_typeerror(self):
        obs = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.0, 2.0, 3.0])
        with self.assertRaises(TypeError):
            MSE().compute(obs, pred, mask="is_storm")

    def test_compute_callable_mask_raises_typeerror(self):
        obs = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.0, 2.0, 3.0])
        with self.assertRaises(TypeError):
            MSE().compute(obs, pred, mask=lambda _: np.ones(3, dtype=bool))

    def test_peak_weighted_mse_accepts_mask(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        pred = np.array([1.0, 9.0, 3.0, 9.0])
        mask = np.array([True, False, True, False])
        obj = PeakWeightedMSE(power=2.0, mask=mask)
        self.assertIs(obj.mask, mask)
        self.assertAlmostEqual(obj.compute(obs, pred), 0.0, places=12)

    def test_log_nse_accepts_mask(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        mask = np.array([True, False, True, False])
        obj = LogNSE(epsilon=0.01, mask=mask)
        self.assertIs(obj.mask, mask)
        # Perfect prediction on the masked subset → LogNSE == 1.
        self.assertAlmostEqual(obj.compute(obs, obs), 1.0, places=10)


# ===========================================================================
# Test _identify_pareto
# ===========================================================================

class TestIdentifyPareto(unittest.TestCase):

    def test_single_solution_is_pareto(self):
        F = np.array([[1.0, 2.0]])
        mask = _identify_pareto(F)
        self.assertEqual(mask.tolist(), [True])

    def test_all_pareto_when_tradeoff(self):
        F = np.array([[0.0, 1.0], [1.0, 0.0]])
        mask = _identify_pareto(F)
        self.assertTrue(all(mask))

    def test_dominated_solution_excluded(self):
        F = np.array([
            [1.0, 1.0],
            [2.0, 2.0],  # dominated by first
            [0.5, 1.5],
        ])
        mask = _identify_pareto(F)
        self.assertFalse(mask[1])
        self.assertTrue(mask[0])
        self.assertTrue(mask[2])

    def test_all_identical_all_pareto(self):
        F = np.array([[1.0, 1.0], [1.0, 1.0]])
        mask = _identify_pareto(F)
        self.assertTrue(all(mask))


# ===========================================================================
# Test CalibrationResult
# ===========================================================================

class TestCalibrationResult(unittest.TestCase):

    def _make_result(self) -> CalibrationResult:
        """Two-objective result: obj0 minimised, obj1 maximised (stored negated)."""
        pareto_X = np.array([[1.0, 2.0], [3.0, 4.0]])
        pareto_F = np.array([[0.5, -0.8], [0.3, -0.9]])  # obj1 negated (maximised)
        rec = GenerationRecord(
            generation=1,
            X=pareto_X.copy(),
            F=pareto_F.copy(),
            n_pareto=2,
        )
        return CalibrationResult(
            history=[rec],
            pareto_X=pareto_X,
            pareto_F=pareto_F,
            param_names=["p0", "p1"],
            objective_names=["mse", "nse"],
            minimize_flags=[True, False],
        )

    def test_objective_display_values_unnegates_maximised(self):
        result = self._make_result()
        display = result.objective_display_values()
        # obj0 (minimise): no change → 0.5, 0.3
        np.testing.assert_allclose(display[:, 0], [0.5, 0.3])
        # obj1 (maximise): negated back → 0.8, 0.9
        np.testing.assert_allclose(display[:, 1], [0.8, 0.9])

    def test_best_by_minimised_objective(self):
        result = self._make_result()
        best = result.best_by("mse")
        # pareto_F col0: [0.5, 0.3] → argmin = 1 → pareto_X[1] = [3, 4]
        np.testing.assert_array_equal(best, [3.0, 4.0])

    def test_best_by_maximised_objective(self):
        result = self._make_result()
        best = result.best_by("nse")
        # pareto_F col1: [-0.8, -0.9] → argmin = 1 → pareto_X[1] = [3, 4]
        np.testing.assert_array_equal(best, [3.0, 4.0])

    def test_best_by_invalid_name_raises(self):
        result = self._make_result()
        with self.assertRaises(ValueError):
            result.best_by("nonexistent")

    def test_to_pareto_dataframe_columns(self):
        result = self._make_result()
        df = result.to_pareto_dataframe()
        expected = {"generation", "p0", "p1", "mse", "nse", "is_pareto"}
        self.assertTrue(expected.issubset(set(df.columns)))

    def test_to_pareto_dataframe_display_form(self):
        result = self._make_result()
        df = result.to_pareto_dataframe()
        pareto_rows = df[df["is_pareto"]]
        # NSE column should be un-negated: 0.8 and 0.9
        self.assertTrue((pareto_rows["nse"] > 0).all())

    def test_to_pareto_dataframe_is_pareto_only_in_final_gen(self):
        pareto_X = np.array([[1.0, 2.0]])
        pareto_F = np.array([[0.5, -0.8]])
        recs = [
            GenerationRecord(1, pareto_X.copy(), pareto_F.copy(), 1),
            GenerationRecord(2, pareto_X.copy(), pareto_F.copy(), 1),
        ]
        result = CalibrationResult(
            history=recs,
            pareto_X=pareto_X,
            pareto_F=pareto_F,
            param_names=["p0", "p1"],
            objective_names=["mse"],
            minimize_flags=[True],
        )
        df = result.to_pareto_dataframe()
        gen1_pareto = df[df["generation"] == 1]["is_pareto"].tolist()
        gen2_pareto = df[df["generation"] == 2]["is_pareto"].tolist()
        self.assertFalse(any(gen1_pareto))
        self.assertTrue(any(gen2_pareto))


# ===========================================================================
# Test CalibrationProblem
# ===========================================================================

class TestCalibrationProblem(unittest.TestCase):

    def setUp(self):
        self.problem, self.observed, self.slope, self.intercept = \
            _make_linear_model_and_problem()

    def test_n_params(self):
        self.assertEqual(self.problem.n_params, 2)

    def test_n_objectives(self):
        self.assertEqual(self.problem.n_objectives, 2)

    def test_param_names(self):
        self.assertEqual(self.problem.param_names, ["slope", "intercept"])

    def test_objective_names(self):
        self.assertEqual(self.problem.objective_names, ["mse", "nash_sutcliffe"])

    def test_minimize_flags(self):
        self.assertEqual(self.problem.minimize_flags, [True, False])

    def test_bounds_shape(self):
        xl, xu = self.problem.bounds
        self.assertEqual(xl.shape, (2,))
        self.assertEqual(xu.shape, (2,))

    def test_evaluate_returns_correct_length(self):
        x = np.array([self.slope, self.intercept])
        F = self.problem.evaluate(x)
        self.assertEqual(F.shape, (2,))

    def test_evaluate_mse_near_zero_for_true_params(self):
        x = np.array([self.slope, self.intercept])
        F = self.problem.evaluate(x)
        self.assertLess(F[0], 0.1)  # MSE ~ 0 for near-true params

    def test_evaluate_nse_negated(self):
        x = np.array([self.slope, self.intercept])
        F = self.problem.evaluate(x)
        # F[1] should be negative (NSE≈1 negated → ≈-1)
        self.assertLess(F[1], 0.0)

    def test_evaluate_updates_parameters(self):
        x = np.array([5.0, 2.5])
        self.problem.evaluate(x)
        self.assertAlmostEqual(
            self.problem._model.get_scalar_parameter("slope").value, 5.0
        )
        self.assertAlmostEqual(
            self.problem._model.get_scalar_parameter("intercept").value, 2.5
        )

    def test_make_copy_is_independent(self):
        copy = self.problem.make_copy()
        copy._model.get_scalar_parameter("slope").value = 99.0
        # Original unchanged
        self.assertNotAlmostEqual(
            self.problem._model.get_scalar_parameter("slope").value, 99.0
        )

    def test_make_copy_is_prepared(self):
        copy = self.problem.make_copy()
        self.assertTrue(copy._model.is_prepared())

    def test_model_not_prepared_raises(self):
        m = _LinearModel()
        m.initialize()
        m.validate()
        # not prepared, no data supplied → RuntimeError
        with self.assertRaises(RuntimeError):
            CalibrationProblem(
                model=m,
                objectives=[MSE()],
                column_map={
                    "observed":  lambda _: np.ones(5),
                    "predicted": lambda df: df["y"].to_numpy(),
                },
            )

    def test_empty_objectives_raises(self):
        x = np.linspace(0, 5, 10)
        m = _LinearModel()
        m.initialize()
        m.validate()
        m.prepare(x)
        with self.assertRaises(ValueError):
            CalibrationProblem(
                model=m,
                objectives=[],
                column_map={
                    "observed":  lambda _: np.ones(10),
                    "predicted": lambda df: df["y"].to_numpy(),
                },
            )


# ===========================================================================
# Test CalibrationProblem masking
# ===========================================================================

class TestCalibrationProblemMasking(unittest.TestCase):

    def _build(self, *, objectives, mask=None, column_map_extra=None, observed=None):
        """Build a prepared _LinearModel CalibrationProblem.

        ``predict()`` yields ``y = slope*x + intercept``; with the default
        params (slope=1, intercept=0) the prediction equals ``x`` exactly.
        """
        x = np.linspace(1.0, 10.0, 10)
        model = _LinearModel()
        model.initialize()
        model.validate()
        model.prepare(x)

        if observed is None:
            observed = x.copy()
        _obs = observed
        cmap = {
            "observed":  lambda _: _obs,
            "predicted": lambda df: df["y"].to_numpy(),
        }
        if column_map_extra:
            cmap.update(column_map_extra)
        problem = CalibrationProblem(
            model=model,
            objectives=objectives,
            column_map=cmap,
            mask=mask,
        )
        return problem, x

    def test_array_mask_changes_objective_value(self):
        # Corrupt the observed series only at masked-out positions; with the
        # mask applied the score must be unaffected (≈0) vs. large unmasked.
        x = np.linspace(1.0, 10.0, 10)
        observed = x.copy()
        observed[5:] += 50.0  # large mismatch in the second half
        keep = np.array([True] * 5 + [False] * 5)

        unmasked, _ = self._build(objectives=[MSE()], observed=observed)
        masked, _ = self._build(objectives=[MSE()], mask=keep, observed=observed)

        x0 = np.array([1.0, 0.0])  # slope=1, intercept=0 → pred == x
        F_unmasked = unmasked.evaluate(x0)
        F_masked = masked.evaluate(x0)
        self.assertGreater(F_unmasked[0], 1.0)
        self.assertAlmostEqual(F_masked[0], 0.0, places=10)

    def test_column_map_mask_name(self):
        # Provide the mask as a prepared-data column via column_map["mask"].
        x = np.linspace(1.0, 10.0, 10)
        observed = x.copy()
        observed[5:] += 50.0
        is_storm = np.array([True] * 5 + [False] * 5)

        model = _LinearModel()
        model.initialize()
        model.validate()
        model.prepare(x)

        prepared = pd.DataFrame({"is_storm": is_storm})
        problem = CalibrationProblem(
            model=model,
            objectives=[MSE()],
            column_map={
                "observed":  lambda _: observed,
                "predicted": lambda df: df["y"].to_numpy(),
                "mask":      lambda _: prepared["is_storm"].to_numpy(),
            },
        )
        F = problem.evaluate(np.array([1.0, 0.0]))
        self.assertAlmostEqual(F[0], 0.0, places=10)

    def test_callable_mask(self):
        x = np.linspace(1.0, 10.0, 10)
        observed = x.copy()
        observed[5:] += 50.0
        problem, _ = self._build(
            objectives=[MSE()],
            mask=lambda _: np.array([True] * 5 + [False] * 5),
            observed=observed,
        )
        F = problem.evaluate(np.array([1.0, 0.0]))
        self.assertAlmostEqual(F[0], 0.0, places=10)

    def test_explicit_mask_arg_beats_column_map(self):
        x = np.linspace(1.0, 10.0, 10)
        observed = x.copy()
        observed[5:] += 50.0
        model = _LinearModel()
        model.initialize()
        model.validate()
        model.prepare(x)
        # column_map["mask"] would keep the bad second half; explicit mask= wins.
        problem = CalibrationProblem(
            model=model,
            objectives=[MSE()],
            column_map={
                "observed":  lambda _: observed,
                "predicted": lambda df: df["y"].to_numpy(),
                "mask":      lambda _: np.array([False] * 5 + [True] * 5),
            },
            mask=np.array([True] * 5 + [False] * 5),
        )
        F = problem.evaluate(np.array([1.0, 0.0]))
        self.assertAlmostEqual(F[0], 0.0, places=10)

    def test_per_objective_override_beats_problem_default(self):
        # Problem default keeps the first half (good); the second objective
        # overrides to keep the second half (bad) → its value must be large.
        x = np.linspace(1.0, 10.0, 10)
        observed = x.copy()
        observed[5:] += 50.0
        good_half = np.array([True] * 5 + [False] * 5)
        bad_half = np.array([False] * 5 + [True] * 5)

        model = _LinearModel()
        model.initialize()
        model.validate()
        model.prepare(x)
        problem = CalibrationProblem(
            model=model,
            objectives=[MSE(), MSE(mask=bad_half)],
            column_map={
                "observed":  lambda _: observed,
                "predicted": lambda df: df["y"].to_numpy(),
            },
            mask=good_half,
        )
        F = problem.evaluate(np.array([1.0, 0.0]))
        self.assertAlmostEqual(F[0], 0.0, places=10)   # problem default
        self.assertGreater(F[1], 1.0)                  # per-objective override

    def test_mask_length_mismatch_raises(self):
        x = np.linspace(1.0, 10.0, 10)
        model = _LinearModel()
        model.initialize()
        model.validate()
        model.prepare(x)
        with self.assertRaises(ValueError):
            CalibrationProblem(
                model=model,
                objectives=[MSE()],
                column_map={
                    "observed":  lambda _: x.copy(),
                    "predicted": lambda df: df["y"].to_numpy(),
                },
                mask=np.array([True, False, True]),  # wrong length
            )

    def test_make_copy_preserves_masks(self):
        x = np.linspace(1.0, 10.0, 10)
        observed = x.copy()
        observed[5:] += 50.0
        problem, _ = self._build(
            objectives=[MSE()],
            mask=np.array([True] * 5 + [False] * 5),
            observed=observed,
        )
        cp = problem.make_copy()
        F = cp.evaluate(np.array([1.0, 0.0]))
        self.assertAlmostEqual(F[0], 0.0, places=10)
        # Independent arrays.
        self.assertIsNot(cp._mask, problem._mask)


# ===========================================================================
# Test NSGAIISolver (conditional)
# ===========================================================================

@unittest.skipUnless(HAS_PYMOO, "pymoo not installed")
class TestNSGAIISolver(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from sparsehydro.calibration.solvers.nsga2 import NSGAIISolver
        cls.NSGAIISolver = NSGAIISolver
        cls.problem, cls.observed, cls.slope, cls.intercept = \
            _make_linear_model_and_problem()

    def test_solve_returns_calibration_result(self):
        solver = self.NSGAIISolver(pop_size=20, n_gen=5, seed=0)
        result = solver.solve(self.problem)
        self.assertIsInstance(result, CalibrationResult)

    def test_result_has_history(self):
        solver = self.NSGAIISolver(pop_size=20, n_gen=5, seed=0)
        result = solver.solve(self.problem)
        self.assertEqual(len(result.history), 5)

    def test_result_pareto_X_shape(self):
        solver = self.NSGAIISolver(pop_size=20, n_gen=5, seed=0)
        result = solver.solve(self.problem)
        self.assertEqual(result.pareto_X.shape[1], 2)

    def test_result_objective_names_preserved(self):
        solver = self.NSGAIISolver(pop_size=20, n_gen=5, seed=0)
        result = solver.solve(self.problem)
        self.assertEqual(result.objective_names, ["mse", "nash_sutcliffe"])

    def test_result_minimize_flags_preserved(self):
        solver = self.NSGAIISolver(pop_size=20, n_gen=5, seed=0)
        result = solver.solve(self.problem)
        self.assertEqual(result.minimize_flags, [True, False])

    def test_best_by_mse_valid_shape(self):
        solver = self.NSGAIISolver(pop_size=20, n_gen=5, seed=0)
        result = solver.solve(self.problem)
        best = result.best_by("mse")
        self.assertEqual(best.shape, (2,))

    def test_convergence_direction(self):
        solver = self.NSGAIISolver(pop_size=30, n_gen=20, seed=42)
        result = solver.solve(self.problem)
        early_mse = result.history[0].F[:, 0].min()
        late_mse = result.history[-1].F[:, 0].min()
        self.assertLessEqual(late_mse, early_mse + 1e-6)

    def test_to_pareto_dataframe_has_all_objectives(self):
        solver = self.NSGAIISolver(pop_size=20, n_gen=5, seed=0)
        result = solver.solve(self.problem)
        df = result.to_pareto_dataframe()
        self.assertIn("mse", df.columns)
        self.assertIn("nash_sutcliffe", df.columns)


# ===========================================================================
# Test ScipySolver (conditional)
# ===========================================================================

@unittest.skipUnless(HAS_SCIPY, "scipy not installed")
class TestScipySolver(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from sparsehydro.calibration.solvers.scipy_solver import ScipySolver
        cls.ScipySolver = ScipySolver
        cls.problem, cls.observed, cls.slope, cls.intercept = \
            _make_linear_model_and_problem()

    def test_solve_returns_calibration_result(self):
        solver = self.ScipySolver(method="differential_evolution",
                                  objective_index=0, maxiter=50, seed=0)
        result = solver.solve(self.problem)
        self.assertIsInstance(result, CalibrationResult)

    def test_result_single_pareto_solution(self):
        solver = self.ScipySolver(objective_index=0, maxiter=50, seed=0)
        result = solver.solve(self.problem)
        self.assertEqual(result.pareto_X.shape[0], 1)

    def test_result_objective_names_preserved(self):
        solver = self.ScipySolver(objective_index=0, maxiter=50, seed=0)
        result = solver.solve(self.problem)
        self.assertEqual(result.objective_names, ["mse", "nash_sutcliffe"])

    def test_result_minimize_flags_preserved(self):
        solver = self.ScipySolver(objective_index=0, maxiter=50, seed=0)
        result = solver.solve(self.problem)
        self.assertEqual(result.minimize_flags, [True, False])

    def test_mse_converges_toward_zero(self):
        solver = self.ScipySolver(objective_index=0, maxiter=200, seed=42)
        result = solver.solve(self.problem)
        best = result.best_by("mse")
        # Evaluate MSE at best params
        F = self.problem.make_copy().evaluate(best)
        self.assertLess(F[0], 1.0)

    def test_scipy_minimize_method(self):
        solver = self.ScipySolver(method="L-BFGS-B", objective_index=0, maxiter=100)
        result = solver.solve(self.problem)
        self.assertIsInstance(result, CalibrationResult)
        self.assertEqual(result.pareto_X.shape[0], 1)


@unittest.skipUnless(HAS_PLATYPUS, "platypus not installed")
class TestPlatypusCmaEsSolver(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from sparsehydro.calibration.solvers.cmaes_solver import PlatypusCmaEsSolver
        cls.PlatypusCmaEsSolver = PlatypusCmaEsSolver
        cls.problem, cls.observed, cls.slope, cls.intercept = \
            _make_linear_model_and_problem()

    def _solver(self, **kwargs):
        opts = dict(n_trials=30, offspring_size=5, seed=0, n_jobs=1,
                    progress_every=0, verbose=False)
        opts.update(kwargs)
        return self.PlatypusCmaEsSolver(**opts)

    def test_solve_returns_calibration_result(self):
        result = self._solver(objective_index=0).solve(self.problem)
        self.assertIsInstance(result, CalibrationResult)

    def test_result_single_pareto_solution(self):
        result = self._solver(objective_index=0).solve(self.problem)
        self.assertEqual(result.pareto_X.shape[0], 1)

    def test_result_objective_names_preserved(self):
        result = self._solver(objective_index=0).solve(self.problem)
        self.assertEqual(result.objective_names, ["mse", "nash_sutcliffe"])
        self.assertEqual(result.minimize_flags, [True, False])

    def test_objective_selected_by_name(self):
        solver = self._solver(objective="nash_sutcliffe")
        result = solver.solve(self.problem)
        self.assertIsInstance(result, CalibrationResult)
        self.assertEqual(solver.objective_optimized_, "nash_sutcliffe")

    def test_unknown_objective_name_raises(self):
        with self.assertRaises(ValueError):
            self._solver(objective="not_an_objective").solve(self.problem)

    def test_multi_seed_populates_seed_solutions(self):
        solver = self._solver(objective="nash_sutcliffe", multi_start=2)
        solver.solve(self.problem)
        self.assertEqual(len(solver.seed_solutions_), 2)
        self.assertIn(solver.selected_seed_, [s["seed"] for s in solver.seed_solutions_])
        self.assertTrue(any(s["selected"] for s in solver.seed_solutions_))


if __name__ == "__main__":
    unittest.main()
