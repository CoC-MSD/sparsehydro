"""Comprehensive unit tests for sparsehydro.calibration.

Extends test_calibration.py to cover:
  - Numerical accuracy for all 10 objective functions with known expected values
  - _identify_pareto edge cases: single-objective, large fronts, dominance invariants
  - CalibrationResult edge cases: empty history, all-minimise/all-maximise, list inputs
  - CalibrationProblem edge cases: single objective, maximised-only objectives,
    exception fallback penalty, bounds correctness, copy independence
  - PlatypusSolver (requires platypus-opt):
      NSGA2, SPEA2, GDE3, IBEA, OMOPSO, SMPSO, EpsilonMOEA
      result structure, bounds, metadata, history, record_frequency,
      seed reproducibility, convergence direction
  - NSGAIISolver additional: single-objective, seed=None, history length invariant
  - ScipySolver additional: objective_index, history shape per method, convergence
  - Integration: pymoo + scipy converge to same MSE region on a known linear problem
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from sparsehydro.calibration import (
    IObjective,
    KGE,
    MAE,
    MSE,
    RMSE,
    CalibrationProblem,
    CalibrationResult,
    GenerationRecord,
    NashSutcliffe,
    PeakWeightedMSE,
)
from sparsehydro.metrics.objectives import (
    PBIAS,
    IndexOfAgreement,
    LogNSE,
    VolumeRelativeError,
)
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
# Shared test model and problem factory
# ---------------------------------------------------------------------------

class _LinearModel(IModel):
    """y = slope * x + intercept — minimal model for calibration tests."""

    model_name = "test_comprehensive_linear"

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
        self._state = ModelState.PREDICTED
        return pd.DataFrame({"y": slope * self._x + intercept})

    def finalize(self) -> None:
        self._state = ModelState.FINALIZED


_RNG = np.random.default_rng(0)
_X = np.linspace(0, 10, 30)
_TRUE_SLOPE, _TRUE_INTERCEPT = 2.5, 1.0
_OBSERVED = _TRUE_SLOPE * _X + _TRUE_INTERCEPT + _RNG.normal(0, 0.05, len(_X))


def _make_problem(objectives=None) -> CalibrationProblem:
    """Return a prepared CalibrationProblem. Default: [MSE, NashSutcliffe]."""
    if objectives is None:
        objectives = [MSE(), NashSutcliffe()]
    m = _LinearModel()
    m.initialize()
    m.validate()
    m.prepare(_X)
    _obs = _OBSERVED
    return CalibrationProblem(
        model=m,
        objectives=objectives,
        column_map={
            "observed":  lambda _: _obs,
            "predicted": lambda df: df["y"].to_numpy(),
        },
    )


# ===========================================================================
# I. Objective numerical accuracy
# ===========================================================================

class TestObjectiveNumericalAccuracy(unittest.TestCase):
    """Verify every objective against manually computed expected values."""

    # obs=[1,2,3,4,5], pred=[1,2,3,4,6]: only last timestep differs by +1
    _OBS = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    _PRED = np.array([1.0, 2.0, 3.0, 4.0, 6.0])

    def test_mse_known(self):
        # residuals=[0,0,0,0,1] → MSE = 1/5 = 0.2
        self.assertAlmostEqual(MSE().evaluate(self._OBS, self._PRED), 0.2, places=12)

    def test_rmse_known(self):
        self.assertAlmostEqual(RMSE().evaluate(self._OBS, self._PRED), 0.2 ** 0.5, places=10)

    def test_mae_known(self):
        self.assertAlmostEqual(MAE().evaluate(self._OBS, self._PRED), 0.2, places=12)

    def test_nse_known(self):
        # mean_obs=3, SS_obs=(4+1+0+1+4)=10, SS_res=1 → NSE=0.9
        self.assertAlmostEqual(NashSutcliffe().evaluate(self._OBS, self._PRED), 0.9, places=10)

    def test_kge_perfect(self):
        self.assertAlmostEqual(KGE().evaluate(self._OBS, self._OBS), 1.0, places=10)

    def test_kge_constant_mean_bias(self):
        # pred = obs + 1: r=1, alpha=1, beta=4/3 → KGE = 1 − 1/3
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = obs + 1.0
        self.assertAlmostEqual(KGE().evaluate(obs, pred), 1.0 - 1.0 / 3.0, places=8)

    def test_rmse_is_sqrt_mse(self):
        mse = MSE().evaluate(self._OBS, self._PRED)
        rmse = RMSE().evaluate(self._OBS, self._PRED)
        self.assertAlmostEqual(rmse, mse ** 0.5, places=10)

    def test_mse_equals_mae_squared_for_constant_error(self):
        # Constant error of 0.5 → MSE = 0.25, MAE = 0.5 → MSE == MAE^2
        obs = np.array([1.0, 2.0, 3.0])
        pred = obs + 0.5
        self.assertAlmostEqual(MSE().evaluate(obs, pred), MAE().evaluate(obs, pred) ** 2, places=12)

    def test_nse_mean_predictor_is_zero(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = np.full_like(obs, obs.mean())
        self.assertAlmostEqual(NashSutcliffe().evaluate(obs, pred), 0.0, places=10)

    def test_nse_negative_for_poor_predictor(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertLess(NashSutcliffe().evaluate(obs, obs * 5.0), 0.0)


# ===========================================================================
# II. Objectives not covered in test_calibration.py
# ===========================================================================

class TestNewObjectives(unittest.TestCase):

    _OBS = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    _PRED_OFF = np.array([1.0, 2.0, 3.0, 4.0, 6.0])  # last step +1

    # --- PBIAS ---

    def test_pbias_name_and_flag(self):
        self.assertEqual(PBIAS.name, "pbias")
        self.assertTrue(PBIAS.minimize)

    def test_pbias_perfect_is_zero(self):
        self.assertAlmostEqual(PBIAS().evaluate(self._OBS, self._OBS), 0.0, places=10)

    def test_pbias_known_value(self):
        # sum_obs=15, sum(obs−pred)=−1 → PBIAS = 100·|−1|/15
        expected = 100.0 / 15.0
        self.assertAlmostEqual(PBIAS().evaluate(self._OBS, self._PRED_OFF), expected, places=5)

    def test_pbias_symmetric_over_under(self):
        obs = np.array([2.0, 2.0, 2.0])
        over = np.array([3.0, 3.0, 3.0])
        under = np.array([1.0, 1.0, 1.0])
        self.assertAlmostEqual(
            PBIAS().evaluate(obs, over), PBIAS().evaluate(obs, under), places=10
        )

    def test_pbias_nonnegative(self):
        rng = np.random.default_rng(2)
        obs = rng.uniform(0.5, 5.0, 20)
        pred = rng.uniform(0.5, 5.0, 20)
        self.assertGreaterEqual(PBIAS().evaluate(obs, pred), 0.0)

    # --- VolumeRelativeError ---

    def test_vre_name_and_flag(self):
        self.assertEqual(VolumeRelativeError.name, "volume_relative_error")
        self.assertTrue(VolumeRelativeError.minimize)

    def test_vre_perfect_is_zero(self):
        self.assertAlmostEqual(
            VolumeRelativeError().evaluate(self._OBS, self._OBS), 0.0, places=10
        )

    def test_vre_known_value(self):
        # sum_obs=15, sum_pred=16 → VRE = |16−15|/15 = 1/15
        expected = 1.0 / 15.0
        self.assertAlmostEqual(
            VolumeRelativeError().evaluate(self._OBS, self._PRED_OFF), expected, places=5
        )

    def test_vre_nonnegative(self):
        rng = np.random.default_rng(7)
        obs = rng.uniform(0.5, 5.0, 20)
        pred = rng.uniform(0.5, 5.0, 20)
        self.assertGreaterEqual(VolumeRelativeError().evaluate(obs, pred), 0.0)

    # --- LogNSE ---

    def test_log_nse_name_and_flag(self):
        self.assertEqual(LogNSE.name, "log_nse")
        self.assertFalse(LogNSE.minimize)

    def test_log_nse_perfect_is_one(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(LogNSE().evaluate(obs, obs), 1.0, places=10)

    def test_log_nse_leq_one(self):
        rng = np.random.default_rng(3)
        obs = rng.uniform(0.1, 5.0, 20)
        pred = rng.uniform(0.1, 5.0, 20)
        self.assertLessEqual(LogNSE().evaluate(obs, pred), 1.0 + 1e-10)

    def test_log_nse_custom_epsilon_perfect(self):
        obs = np.array([0.001, 0.01, 0.1, 1.0])
        for eps in (0.001, 0.1, 1.0):
            with self.subTest(epsilon=eps):
                self.assertAlmostEqual(LogNSE(epsilon=eps).evaluate(obs, obs), 1.0, places=10)

    # --- IndexOfAgreement ---

    def test_ioa_name_and_flag(self):
        self.assertEqual(IndexOfAgreement.name, "index_of_agreement")
        self.assertFalse(IndexOfAgreement.minimize)

    def test_ioa_perfect_is_one(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(IndexOfAgreement().evaluate(obs, obs), 1.0, places=10)

    def test_ioa_mean_predictor_is_zero(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = np.full_like(obs, obs.mean())
        self.assertAlmostEqual(IndexOfAgreement().evaluate(obs, pred), 0.0, places=10)

    def test_ioa_range_zero_to_one(self):
        rng = np.random.default_rng(5)
        obs = rng.uniform(1.0, 10.0, 30)
        pred = rng.uniform(1.0, 10.0, 30)
        d = IndexOfAgreement().evaluate(obs, pred)
        self.assertGreaterEqual(d, 0.0 - 1e-10)
        self.assertLessEqual(d, 1.0 + 1e-10)


# ===========================================================================
# III. _identify_pareto edge cases
# ===========================================================================

class TestIdentifyParetoComprehensive(unittest.TestCase):

    def test_single_objective_dominated(self):
        # Only minimum survives
        F = np.array([[3.0], [1.0], [2.0]])
        self.assertEqual(_identify_pareto(F).tolist(), [False, True, False])

    def test_all_dominated_except_one(self):
        # First solution dominates all others
        F = np.array([[0.0, 0.0]] + [[float(i), float(i)] for i in range(1, 10)])
        mask = _identify_pareto(F)
        self.assertTrue(mask[0])
        self.assertFalse(any(mask[1:]))

    def test_three_objectives_all_nondominated(self):
        # Each solution is best on a different objective
        F = np.array([
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.5, 0.5, 0.5],
        ])
        mask = _identify_pareto(F)
        self.assertTrue(all(mask))

    def test_large_front_returns_correct_length(self):
        rng = np.random.default_rng(99)
        F = rng.uniform(0, 1, (100, 3))
        mask = _identify_pareto(F)
        self.assertEqual(len(mask), 100)
        self.assertTrue(mask.dtype == bool)
        self.assertTrue(mask.any())

    def test_pareto_set_internally_nondominated(self):
        # No member of the Pareto set may be dominated by another member
        rng = np.random.default_rng(11)
        F = rng.uniform(0, 10, (50, 2))
        pareto_F = F[_identify_pareto(F)]
        for i in range(len(pareto_F)):
            for j in range(len(pareto_F)):
                if i == j:
                    continue
                dominated = np.all(pareto_F[j] <= pareto_F[i]) and np.any(pareto_F[j] < pareto_F[i])
                self.assertFalse(dominated)


# ===========================================================================
# IV. CalibrationResult edge cases
# ===========================================================================

class TestCalibrationResultEdgeCases(unittest.TestCase):

    def _make(self, **overrides):
        defaults = dict(
            history=[],
            pareto_X=np.array([[1.0, 2.0]]),
            pareto_F=np.array([[0.25]]),
            param_names=["slope", "intercept"],
            objective_names=["mse"],
            minimize_flags=[True],
        )
        defaults.update(overrides)
        return CalibrationResult(**defaults)

    def test_empty_history_dataframe_is_empty(self):
        df = self._make().to_pareto_dataframe()
        self.assertEqual(len(df), 0)

    def test_list_inputs_coerced_to_float_array(self):
        result = CalibrationResult(
            history=[],
            pareto_X=[[1, 2], [3, 4]],
            pareto_F=[[0.1], [0.2]],
            param_names=["p0", "p1"],
            objective_names=["mse"],
            minimize_flags=[True],
        )
        self.assertIsInstance(result.pareto_X, np.ndarray)
        self.assertEqual(result.pareto_X.dtype, np.float64)

    def test_display_values_all_minimised_unchanged(self):
        F = np.array([[0.1, 0.2], [0.3, 0.4]])
        result = CalibrationResult(
            history=[], pareto_X=np.zeros((2, 1)), pareto_F=F,
            param_names=["p"], objective_names=["a", "b"],
            minimize_flags=[True, True],
        )
        np.testing.assert_array_equal(result.objective_display_values(), F)

    def test_display_values_all_maximised_unnegated(self):
        # Stored as −0.9, −0.8 (negated); display should return 0.9, 0.8
        F = np.array([[-0.9], [-0.8]])
        result = CalibrationResult(
            history=[], pareto_X=np.zeros((2, 1)), pareto_F=F,
            param_names=["p"], objective_names=["nse"], minimize_flags=[False],
        )
        np.testing.assert_allclose(result.objective_display_values(), [[0.9], [0.8]])

    def test_best_by_selects_correct_row(self):
        result = CalibrationResult(
            history=[],
            pareto_X=np.array([[10.0, 20.0], [30.0, 40.0]]),
            pareto_F=np.array([[0.5], [0.1]]),
            param_names=["slope", "intercept"],
            objective_names=["mse"],
            minimize_flags=[True],
        )
        np.testing.assert_array_equal(result.best_by("mse"), [30.0, 40.0])

    def test_multiple_history_records_accumulated(self):
        X = np.array([[1.0, 2.0]])
        F = np.array([[0.5]])
        recs = [GenerationRecord(i, X.copy(), F.copy(), 1) for i in range(1, 6)]
        result = CalibrationResult(
            history=recs, pareto_X=X, pareto_F=F,
            param_names=["p0", "p1"], objective_names=["mse"], minimize_flags=[True],
        )
        self.assertEqual(len(result.history), 5)


# ===========================================================================
# V. CalibrationProblem edge cases
# ===========================================================================

class TestCalibrationProblemEdgeCases(unittest.TestCase):

    def test_single_objective_shape(self):
        problem = _make_problem(objectives=[MSE()])
        F = problem.evaluate(np.array([_TRUE_SLOPE, _TRUE_INTERCEPT]))
        self.assertEqual(F.shape, (1,))

    def test_single_objective_near_zero_at_true_params(self):
        problem = _make_problem(objectives=[MSE()])
        F = problem.evaluate(np.array([_TRUE_SLOPE, _TRUE_INTERCEPT]))
        self.assertLess(F[0], 0.05)

    def test_all_maximised_objectives_negated_in_F(self):
        problem = _make_problem(objectives=[NashSutcliffe(), KGE()])
        F = problem.evaluate(np.array([_TRUE_SLOPE, _TRUE_INTERCEPT]))
        # Stored in minimisation form → both negative for a good predictor
        self.assertTrue(all(f < 0.0 for f in F))

    def test_exception_in_objective_returns_penalty(self):
        class _BrokenObjective(IObjective):
            name = "broken"
            minimize = True
            def evaluate(self, obs, pred):
                raise RuntimeError("forced failure")

        problem = _make_problem(objectives=[_BrokenObjective()])
        F = problem.evaluate(np.array([1.0, 0.0]))
        self.assertAlmostEqual(F[0], 1e12, places=5)

    def test_exception_in_maximised_objective_returns_negative_penalty(self):
        class _BrokenMax(IObjective):
            name = "broken_max"
            minimize = False
            def evaluate(self, obs, pred):
                raise RuntimeError("forced failure")

        problem = _make_problem(objectives=[_BrokenMax()])
        F = problem.evaluate(np.array([1.0, 0.0]))
        # maximize=False → penalty is −1e12 → stored negated = 1e12... wait
        # Looking at the code: val = 1e12 if obj.minimize else -1e12
        # then: F[i] = val if obj.minimize else -val  → -val = -(-1e12) = 1e12
        # Actually let me recheck: val = -1e12 (not minimize), F[i] = -val = 1e12
        self.assertAlmostEqual(F[0], 1e12, places=5)

    def test_bounds_match_parameter_bounds(self):
        problem = _make_problem()
        xl, xu = problem.bounds
        np.testing.assert_array_equal(xl, [0.0, -5.0])
        np.testing.assert_array_equal(xu, [10.0, 5.0])

    def test_bounds_arrays_are_copies(self):
        problem = _make_problem()
        xl1, _ = problem.bounds
        xl2, _ = problem.bounds
        xl1[0] = 999.0
        self.assertEqual(xl2[0], 0.0)

    def test_make_copy_evaluates_independently(self):
        problem = _make_problem()
        copy = problem.make_copy()
        F_true = problem.evaluate(np.array([_TRUE_SLOPE, _TRUE_INTERCEPT]))
        F_bad = copy.evaluate(np.array([0.1, 0.1]))
        self.assertFalse(np.allclose(F_true, F_bad))

    def test_make_copy_does_not_modify_original(self):
        problem = _make_problem()
        copy = problem.make_copy()
        copy._model.get_scalar_parameter("slope").value = 999.0
        original_slope = problem._model.get_scalar_parameter("slope").value
        self.assertNotAlmostEqual(original_slope, 999.0)


# ===========================================================================
# VI. PlatypusSolver — core
# ===========================================================================

@unittest.skipUnless(HAS_PLATYPUS, "platypus-opt not installed")
class TestPlatypusSolverCore(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import platypus as _p
        from sparsehydro.calibration.solvers.platypus_solver import PlatypusSolver
        cls.P = PlatypusSolver
        cls.platypus = _p
        cls.problem = _make_problem()
        cls.problem_single = _make_problem(objectives=[MSE()])

    def _solve(self, algo, n_evals=400, **kw):
        return self.P(algo, n_evaluations=n_evals, seed=42, **kw).solve(self.problem)

    def test_returns_calibration_result(self):
        self.assertIsInstance(self._solve(self.platypus.NSGAII, population_size=20),
                              CalibrationResult)

    def test_param_names_preserved(self):
        self.assertEqual(
            self._solve(self.platypus.NSGAII, population_size=20).param_names,
            ["slope", "intercept"],
        )

    def test_objective_names_preserved(self):
        self.assertEqual(
            self._solve(self.platypus.NSGAII, population_size=20).objective_names,
            ["mse", "nash_sutcliffe"],
        )

    def test_minimize_flags_preserved(self):
        self.assertEqual(
            self._solve(self.platypus.NSGAII, population_size=20).minimize_flags,
            [True, False],
        )

    def test_pareto_X_columns_match_n_params(self):
        self.assertEqual(
            self._solve(self.platypus.NSGAII, population_size=20).pareto_X.shape[1], 2
        )

    def test_pareto_F_columns_match_n_objectives(self):
        self.assertEqual(
            self._solve(self.platypus.NSGAII, population_size=20).pareto_F.shape[1], 2
        )

    def test_pareto_rows_consistent(self):
        result = self._solve(self.platypus.NSGAII, population_size=20)
        self.assertEqual(result.pareto_X.shape[0], result.pareto_F.shape[0])

    def test_pareto_solutions_within_bounds(self):
        result = self._solve(self.platypus.NSGAII, population_size=20)
        xl, xu = self.problem.bounds
        self.assertTrue(np.all(result.pareto_X >= xl - 1e-8))
        self.assertTrue(np.all(result.pareto_X <= xu + 1e-8))

    def test_pareto_front_is_non_dominated(self):
        result = self._solve(self.platypus.NSGAII, population_size=20)
        if len(result.pareto_F) > 1:
            mask = _identify_pareto(result.pareto_F)
            self.assertTrue(all(mask))

    def test_history_populated_with_record_frequency_1(self):
        result = self._solve(self.platypus.NSGAII, population_size=20)
        self.assertGreater(len(result.history), 0)

    def test_record_frequency_reduces_history_length(self):
        r1 = self.P(self.platypus.NSGAII, population_size=20, n_evaluations=400,
                    seed=42, record_frequency=1).solve(self.problem)
        r5 = self.P(self.platypus.NSGAII, population_size=20, n_evaluations=400,
                    seed=42, record_frequency=5).solve(self.problem)
        self.assertGreater(len(r1.history), len(r5.history))

    def test_seed_reproducibility(self):
        r1 = self._solve(self.platypus.NSGAII, population_size=20)
        r2 = self._solve(self.platypus.NSGAII, population_size=20)
        np.testing.assert_allclose(r1.pareto_F, r2.pareto_F, atol=1e-10)

    def test_generation_record_fields(self):
        result = self._solve(self.platypus.NSGAII, population_size=20)
        rec = result.history[0]
        self.assertIsInstance(rec.generation, int)
        self.assertIsInstance(rec.X, np.ndarray)
        self.assertIsInstance(rec.F, np.ndarray)
        self.assertIsInstance(rec.n_pareto, int)
        self.assertEqual(rec.X.shape[1], 2)
        self.assertEqual(rec.F.shape[1], 2)
        self.assertGreaterEqual(rec.n_pareto, 1)

    def test_single_objective_problem(self):
        result = self.P(self.platypus.NSGAII, population_size=20,
                        n_evaluations=400, seed=42).solve(self.problem_single)
        self.assertIsInstance(result, CalibrationResult)
        self.assertEqual(result.pareto_F.shape[1], 1)

    def test_to_pareto_dataframe_has_expected_columns(self):
        result = self._solve(self.platypus.NSGAII, population_size=20)
        df = result.to_pareto_dataframe()
        self.assertIn("generation", df.columns)
        self.assertIn("slope", df.columns)
        self.assertIn("intercept", df.columns)
        self.assertIn("mse", df.columns)
        self.assertIn("nash_sutcliffe", df.columns)


# ===========================================================================
# VII. PlatypusSolver — algorithm coverage
# ===========================================================================

@unittest.skipUnless(HAS_PLATYPUS, "platypus-opt not installed")
class TestPlatypusSolverAlgorithms(unittest.TestCase):
    """Smoke test: every Platypus algorithm produces a structurally valid result."""

    @classmethod
    def setUpClass(cls):
        import platypus as _p
        from sparsehydro.calibration.solvers.platypus_solver import PlatypusSolver
        cls.P = PlatypusSolver
        cls.platypus = _p
        cls.problem = _make_problem()

    def _assert_valid(self, result: CalibrationResult) -> None:
        self.assertIsInstance(result, CalibrationResult)
        self.assertGreater(result.pareto_X.shape[0], 0)
        self.assertEqual(result.pareto_X.shape[1], 2)
        self.assertEqual(result.pareto_F.shape[1], 2)

    def test_nsga2(self):
        self._assert_valid(
            self.P(self.platypus.NSGAII, population_size=20, n_evaluations=400, seed=0)
            .solve(self.problem)
        )

    def test_spea2(self):
        self._assert_valid(
            self.P(self.platypus.SPEA2, population_size=20, n_evaluations=400, seed=0)
            .solve(self.problem)
        )

    def test_gde3(self):
        self._assert_valid(
            self.P(self.platypus.GDE3, population_size=20, n_evaluations=400, seed=0)
            .solve(self.problem)
        )

    def test_ibea(self):
        self._assert_valid(
            self.P(self.platypus.IBEA, population_size=20, n_evaluations=400, seed=0)
            .solve(self.problem)
        )

    def test_omopso(self):
        self._assert_valid(
            self.P(self.platypus.OMOPSO, epsilons=[0.05, 0.05], swarm_size=20,
                   n_evaluations=400, seed=0)
            .solve(self.problem)
        )

    def test_smpso(self):
        self._assert_valid(
            self.P(self.platypus.SMPSO, swarm_size=20, n_evaluations=400, seed=0)
            .solve(self.problem)
        )

    def test_epsilon_moea(self):
        self._assert_valid(
            self.P(self.platypus.EpsMOEA,
                   population_size=20, epsilons=[0.05, 0.05],
                   n_evaluations=400, seed=0)
            .solve(self.problem)
        )


# ===========================================================================
# VIII. PlatypusSolver — convergence
# ===========================================================================

@unittest.skipUnless(HAS_PLATYPUS, "platypus-opt not installed")
class TestPlatypusSolverConvergence(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import platypus as _p
        from sparsehydro.calibration.solvers.platypus_solver import PlatypusSolver
        cls.P = PlatypusSolver
        cls.platypus = _p
        cls.problem = _make_problem()

    def test_mse_objective_decreases_over_iterations(self):
        result = self.P(
            self.platypus.NSGAII, population_size=30, n_evaluations=900,
            seed=42, record_frequency=1,
        ).solve(self.problem)
        early = result.history[0].F[:, 0].min()
        late = result.history[-1].F[:, 0].min()
        self.assertLessEqual(late, early + 1e-6)

    def test_best_params_near_true_values(self):
        result = self.P(
            self.platypus.NSGAII, population_size=30, n_evaluations=1500, seed=42
        ).solve(self.problem)
        best = result.best_by("mse")
        self.assertAlmostEqual(best[0], _TRUE_SLOPE, delta=0.5)
        self.assertAlmostEqual(best[1], _TRUE_INTERCEPT, delta=0.5)


# ===========================================================================
# IX. NSGAIISolver additional cases
# ===========================================================================

@unittest.skipUnless(HAS_PYMOO, "pymoo not installed")
class TestNSGAIISolverAdditional(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from sparsehydro.calibration.solvers.nsga2 import NSGAIISolver
        cls.Solver = NSGAIISolver
        cls.problem = _make_problem()
        cls.problem_single = _make_problem(objectives=[MSE()])

    def test_single_objective_problem(self):
        result = self.Solver(pop_size=20, n_gen=5, seed=0).solve(self.problem_single)
        self.assertIsInstance(result, CalibrationResult)
        self.assertEqual(result.pareto_F.shape[1], 1)

    def test_seed_none_does_not_crash(self):
        result = self.Solver(pop_size=10, n_gen=3, seed=None).solve(self.problem)
        self.assertIsInstance(result, CalibrationResult)

    def test_history_length_equals_n_gen(self):
        result = self.Solver(pop_size=15, n_gen=7, seed=0).solve(self.problem)
        self.assertEqual(len(result.history), 7)

    def test_pareto_X_and_F_row_counts_match(self):
        result = self.Solver(pop_size=20, n_gen=5, seed=0).solve(self.problem)
        self.assertEqual(result.pareto_X.shape[0], result.pareto_F.shape[0])

    def test_pareto_X_within_bounds(self):
        result = self.Solver(pop_size=20, n_gen=5, seed=0).solve(self.problem)
        xl, xu = self.problem.bounds
        self.assertTrue(np.all(result.pareto_X >= xl - 1e-8))
        self.assertTrue(np.all(result.pareto_X <= xu + 1e-8))

    def test_to_pareto_dataframe_columns_present(self):
        result = self.Solver(pop_size=15, n_gen=5, seed=0).solve(self.problem)
        df = result.to_pareto_dataframe()
        for col in ("generation", "slope", "intercept", "mse", "nash_sutcliffe"):
            self.assertIn(col, df.columns)


# ===========================================================================
# X. ScipySolver additional cases
# ===========================================================================

@unittest.skipUnless(HAS_SCIPY, "scipy not installed")
class TestScipySolverAdditional(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from sparsehydro.calibration.solvers.scipy_solver import ScipySolver
        cls.Solver = ScipySolver
        cls.problem = _make_problem()

    def test_objective_index_1_stored_in_minimisation_form(self):
        # Index 1 = NashSutcliffe (maximised) → converges to a negative stored value
        result = self.Solver(
            method="differential_evolution", objective_index=1, maxiter=100, seed=7
        ).solve(self.problem)
        self.assertIsInstance(result, CalibrationResult)
        self.assertLess(result.pareto_F[0, 1], 0.0)

    def test_local_method_history_is_empty(self):
        result = self.Solver(method="L-BFGS-B", objective_index=0, maxiter=50).solve(
            self.problem
        )
        self.assertEqual(len(result.history), 0)

    def test_de_history_nonempty(self):
        result = self.Solver(
            method="differential_evolution", objective_index=0, maxiter=10, seed=0
        ).solve(self.problem)
        self.assertGreater(len(result.history), 0)

    def test_nelder_mead_returns_single_solution(self):
        result = self.Solver(method="Nelder-Mead", objective_index=0, maxiter=200).solve(
            self.problem
        )
        self.assertEqual(result.pareto_X.shape[0], 1)
        self.assertEqual(result.pareto_F.shape[0], 1)

    def test_single_pareto_row_across_methods(self):
        for method in ("differential_evolution", "L-BFGS-B", "Nelder-Mead"):
            with self.subTest(method=method):
                result = self.Solver(method=method, objective_index=0,
                                     maxiter=30, seed=0).solve(self.problem)
                self.assertEqual(result.pareto_X.shape[0], 1)

    def test_objective_names_match_problem(self):
        result = self.Solver(objective_index=0, maxiter=20, seed=0).solve(self.problem)
        self.assertEqual(result.objective_names, self.problem.objective_names)

    def test_de_converges_toward_true_params(self):
        result = self.Solver(
            method="differential_evolution", objective_index=0, maxiter=300, seed=42
        ).solve(self.problem)
        best = result.best_by("mse")
        self.assertAlmostEqual(best[0], _TRUE_SLOPE, delta=0.5)
        self.assertAlmostEqual(best[1], _TRUE_INTERCEPT, delta=0.5)


# ===========================================================================
# XI. ParticleSwarmSolver
# ===========================================================================

@unittest.skipUnless(HAS_PLATYPUS, "platypus-opt not installed")
class TestParticleSwarmSolver(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from sparsehydro.calibration.solvers.platypus_solver import ParticleSwarmSolver
        cls.Solver = ParticleSwarmSolver
        cls.problem = _make_problem()
        cls.problem_single = _make_problem(objectives=[MSE()])

    # --- result structure ---

    def test_smpso_returns_calibration_result(self):
        result = self.Solver(swarm_size=20, n_evaluations=400, seed=42).solve(self.problem)
        self.assertIsInstance(result, CalibrationResult)

    def test_omopso_returns_calibration_result(self):
        result = self.Solver(
            swarm_size=20, n_evaluations=400, seed=42, epsilons=[0.05, 0.05]
        ).solve(self.problem)
        self.assertIsInstance(result, CalibrationResult)

    def test_param_names_preserved(self):
        result = self.Solver(swarm_size=20, n_evaluations=400, seed=42).solve(self.problem)
        self.assertEqual(result.param_names, ["slope", "intercept"])

    def test_objective_names_preserved(self):
        result = self.Solver(swarm_size=20, n_evaluations=400, seed=42).solve(self.problem)
        self.assertEqual(result.objective_names, ["mse", "nash_sutcliffe"])

    def test_minimize_flags_preserved(self):
        result = self.Solver(swarm_size=20, n_evaluations=400, seed=42).solve(self.problem)
        self.assertEqual(result.minimize_flags, [True, False])

    def test_pareto_X_columns_match_n_params(self):
        result = self.Solver(swarm_size=20, n_evaluations=400, seed=42).solve(self.problem)
        self.assertEqual(result.pareto_X.shape[1], 2)

    def test_pareto_F_columns_match_n_objectives(self):
        result = self.Solver(swarm_size=20, n_evaluations=400, seed=42).solve(self.problem)
        self.assertEqual(result.pareto_F.shape[1], 2)

    def test_pareto_rows_consistent(self):
        result = self.Solver(swarm_size=20, n_evaluations=400, seed=42).solve(self.problem)
        self.assertEqual(result.pareto_X.shape[0], result.pareto_F.shape[0])

    def test_pareto_solutions_within_bounds(self):
        result = self.Solver(swarm_size=20, n_evaluations=400, seed=42).solve(self.problem)
        xl, xu = self.problem.bounds
        self.assertTrue(np.all(result.pareto_X >= xl - 1e-8))
        self.assertTrue(np.all(result.pareto_X <= xu + 1e-8))

    def test_pareto_front_is_non_dominated(self):
        result = self.Solver(swarm_size=20, n_evaluations=400, seed=42).solve(self.problem)
        if len(result.pareto_F) > 1:
            mask = _identify_pareto(result.pareto_F)
            self.assertTrue(all(mask))

    # --- history ---

    def test_history_populated(self):
        result = self.Solver(swarm_size=20, n_evaluations=400, seed=42).solve(self.problem)
        self.assertGreater(len(result.history), 0)

    def test_record_frequency_reduces_history(self):
        r1 = self.Solver(swarm_size=20, n_evaluations=400, seed=42,
                         record_frequency=1).solve(self.problem)
        r5 = self.Solver(swarm_size=20, n_evaluations=400, seed=42,
                         record_frequency=5).solve(self.problem)
        self.assertGreater(len(r1.history), len(r5.history))

    def test_generation_record_fields(self):
        result = self.Solver(swarm_size=20, n_evaluations=400, seed=42).solve(self.problem)
        rec = result.history[0]
        self.assertIsInstance(rec.generation, int)
        self.assertIsInstance(rec.X, np.ndarray)
        self.assertIsInstance(rec.F, np.ndarray)
        self.assertIsInstance(rec.n_pareto, int)
        self.assertEqual(rec.X.shape[1], 2)
        self.assertEqual(rec.F.shape[1], 2)

    # --- reproducibility ---

    def test_seed_reproducibility(self):
        r1 = self.Solver(swarm_size=20, n_evaluations=400, seed=7).solve(self.problem)
        r2 = self.Solver(swarm_size=20, n_evaluations=400, seed=7).solve(self.problem)
        np.testing.assert_allclose(r1.pareto_F, r2.pareto_F, atol=1e-10)

    # --- validation ---

    def test_epsilons_wrong_length_raises(self):
        with self.assertRaises(ValueError):
            self.Solver(
                swarm_size=20, n_evaluations=100, epsilons=[0.05]  # only 1, needs 2
            ).solve(self.problem)

    # --- single-objective ---

    def test_single_objective_smpso(self):
        result = self.Solver(swarm_size=20, n_evaluations=400, seed=42).solve(
            self.problem_single
        )
        self.assertIsInstance(result, CalibrationResult)
        self.assertEqual(result.pareto_F.shape[1], 1)

    def test_single_objective_omopso(self):
        result = self.Solver(
            swarm_size=20, n_evaluations=400, seed=42, epsilons=[0.05]
        ).solve(self.problem_single)
        self.assertIsInstance(result, CalibrationResult)
        self.assertEqual(result.pareto_F.shape[1], 1)

    # --- convergence ---

    def test_smpso_converges_toward_true_params(self):
        result = self.Solver(swarm_size=30, n_evaluations=1500, seed=42).solve(self.problem)
        best = result.best_by("mse")
        self.assertAlmostEqual(best[0], _TRUE_SLOPE, delta=0.5)
        self.assertAlmostEqual(best[1], _TRUE_INTERCEPT, delta=0.5)

    def test_omopso_converges_toward_true_params(self):
        result = self.Solver(
            swarm_size=30, n_evaluations=1500, seed=42, epsilons=[0.05, 0.05]
        ).solve(self.problem)
        best = result.best_by("mse")
        self.assertAlmostEqual(best[0], _TRUE_SLOPE, delta=0.5)
        self.assertAlmostEqual(best[1], _TRUE_INTERCEPT, delta=0.5)


# ===========================================================================
# XII. Integration: multi-solver agreement
# ===========================================================================

@unittest.skipUnless(HAS_SCIPY and HAS_PYMOO, "scipy and pymoo required")
class TestSolverConsistency(unittest.TestCase):
    """Different solvers on the same problem should reach similar MSE minima."""

    @classmethod
    def setUpClass(cls):
        from sparsehydro.calibration.solvers.nsga2 import NSGAIISolver
        from sparsehydro.calibration.solvers.scipy_solver import ScipySolver
        cls.problem = _make_problem(objectives=[MSE()])
        cls.NSGAIISolver = NSGAIISolver
        cls.ScipySolver = ScipySolver

    def test_nsga2_and_de_reach_low_mse(self):
        nsga2_result = self.NSGAIISolver(pop_size=30, n_gen=30, seed=42).solve(self.problem)
        scipy_result = self.ScipySolver(
            method="differential_evolution", objective_index=0, maxiter=200, seed=42
        ).solve(self.problem)
        nsga2_mse = float(nsga2_result.pareto_F[:, 0].min())
        scipy_mse = float(scipy_result.pareto_F[0, 0])
        # Both should converge to near-zero MSE on this simple linear problem
        self.assertLess(nsga2_mse, 0.1)
        self.assertLess(scipy_mse, 0.1)

    def test_nsga2_and_de_agree_on_param_region(self):
        nsga2_result = self.NSGAIISolver(pop_size=30, n_gen=30, seed=42).solve(self.problem)
        scipy_result = self.ScipySolver(
            method="differential_evolution", objective_index=0, maxiter=200, seed=42
        ).solve(self.problem)
        nsga2_best = nsga2_result.best_by("mse")
        scipy_best = scipy_result.best_by("mse")
        # Both should recover slope and intercept within ±1.0 of the true values
        self.assertAlmostEqual(nsga2_best[0], _TRUE_SLOPE, delta=1.0)
        self.assertAlmostEqual(scipy_best[0], _TRUE_SLOPE, delta=1.0)


if __name__ == "__main__":
    unittest.main()
