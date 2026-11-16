"""NSGA-II multi-objective solver backed by pymoo (deprecated).

.. deprecated:: 1.0.0b2
   Use the ``"nsga2"`` solver (:class:`~sparsehydro.calibration.solvers.PlatypusSolver`
   with Platypus's NSGA-II). pymoo is no longer installed with sparsehydro and this
   solver will be removed in a later release.

Requires the optional ``pymoo`` dependency::

    pip install sparsehydro[pymoo]

The solver wraps any :class:`~sparsehydro.calibration.problem.CalibrationProblem`
inside a ``pymoo.ElementwiseProblem`` adapter and runs NSGA-II with SBX
crossover and polynomial mutation.  All objectives are passed to NSGA-II in
minimisation form (maximised objectives are negated by
:meth:`~sparsehydro.calibration.problem.CalibrationProblem.evaluate`).
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from .base import ISolver
from ..result import CalibrationResult, GenerationRecord, identify_pareto

if TYPE_CHECKING:
    from ..problem import CalibrationProblem

_DEPRECATION = ("NSGAIISolver (pymoo) is deprecated and will be removed in a later release; use the 'nsga2' "
                "solver (Platypus NSGA-II): solver_registry.create('nsga2', n_evaluations=pop_size * n_gen, ...)")
_INSTALL = "pymoo is required for the deprecated NSGAIISolver. Install with: pip install sparsehydro[pymoo]"


try:
    from pymoo.algorithms.moo.nsga2 import NSGA2  # type: ignore[import]
    from pymoo.core.callback import Callback  # type: ignore[import]
    from pymoo.core.problem import ElementwiseProblem  # type: ignore[import]
    from pymoo.operators.crossover.sbx import SBX  # type: ignore[import]
    from pymoo.operators.mutation.pm import PM  # type: ignore[import]
    from pymoo.operators.sampling.rnd import FloatRandomSampling  # type: ignore[import]
    from pymoo.optimize import minimize as pymoo_minimize  # type: ignore[import]

    class _PymooAdapter(ElementwiseProblem):
        """pymoo ElementwiseProblem backed by a CalibrationProblem."""

        def __init__(self, problem: "CalibrationProblem") -> None:
            xl, xu = problem.bounds
            super().__init__(
                n_var=problem.n_params,
                n_obj=problem.n_objectives,
                n_ieq_constr=problem.n_ieq_constr,
                xl=xl,
                xu=xu,
            )
            self._problem = problem

        def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> None:
            # penalty_weight=0: pymoo handles constraints natively via out["G"]
            out["F"] = self._problem.evaluate(x, penalty_weight=0.0)
            if self._problem.n_ieq_constr > 0:
                out["G"] = np.array(
                    self._problem._model.inequality_constraints(), dtype=float
                )

    class _GenerationCallback(Callback):
        """Appends a GenerationRecord to history and fires the user callback."""

        def __init__(self, user_callback=None, objective_names=None, minimize_flags=None) -> None:
            super().__init__()
            self.history: list[GenerationRecord] = []
            self._user_callback = user_callback
            self._objective_names = objective_names or []
            self._minimize_flags = minimize_flags or []

        def notify(self, algorithm) -> None:  # type: ignore[override]
            pop = algorithm.pop
            X = pop.get("X").copy()
            F = pop.get("F").copy()
            try:
                ranks = pop.get("rank")
                pareto_mask = ranks == 0
                n_pareto = int(np.sum(pareto_mask))
            except Exception:
                pareto_mask = identify_pareto(F)
                n_pareto = int(np.sum(pareto_mask))

            gen = int(algorithm.n_gen)
            self.history.append(GenerationRecord(generation=gen, X=X, F=F, n_pareto=n_pareto))

            if self._user_callback is not None:
                try:
                    n_evals = int(algorithm.evaluator.n_eval)
                except Exception:
                    n_evals = gen * len(X)
                self._user_callback({
                    "generation": gen,
                    "n_evals": n_evals,
                    "n_pareto": n_pareto,
                    "pareto_F": F[pareto_mask],
                    "objective_names": self._objective_names,
                    "minimize_flags": self._minimize_flags,
                })

    class NSGAIISolver(ISolver):
        """Multi-objective NSGA-II calibration solver.

        Works with any :class:`~sparsehydro.calibration.problem.CalibrationProblem`.
        For multi-objective problems the full Pareto front is returned in
        :attr:`~sparsehydro.calibration.result.CalibrationResult.pareto_X` and
        :attr:`~sparsehydro.calibration.result.CalibrationResult.pareto_F`.

        :param pop_size: Population size per generation.
        :type pop_size: int
        :param n_gen: Total number of generations.
        :type n_gen: int
        :param seed: Random seed for reproducibility (``None`` for random).
        :type seed: int or None
        :param verbose: Print pymoo progress output.
        :type verbose: bool
        :param n_jobs: Parallel workers via ``ProcessPoolExecutor``
            (``1`` = sequential).
        :type n_jobs: int
        :param callback: Optional progress callback invoked once per generation.
        :type callback: Callable or None
        """

        def __init__(
            self,
            pop_size: int = 100,
            n_gen: int = 200,
            seed: int | None = 42,
            verbose: bool = False,
            n_jobs: int = 1,
            callback=None,
        ) -> None:
            warnings.warn(_DEPRECATION, DeprecationWarning, stacklevel=2)
            self.pop_size = pop_size
            self.n_gen = n_gen
            self.seed = seed
            self.verbose = verbose
            self.n_jobs = n_jobs
            self.callback = callback

        def solve(self, problem: "CalibrationProblem", **kwargs) -> CalibrationResult:
            """Run NSGA-II and return a :class:`~sparsehydro.calibration.result.CalibrationResult`.

            :param problem: Calibration problem wrapping model + data + objectives.
            :type problem: CalibrationProblem
            :param kwargs: Per-call overrides: ``pop_size``, ``n_gen``, ``seed``,
                ``verbose``, ``n_jobs``, ``callback``.
            :returns: Result with per-generation history and final Pareto front.
            :rtype: CalibrationResult
            """
            pop_size = kwargs.get("pop_size", self.pop_size)
            n_gen    = kwargs.get("n_gen",    self.n_gen)
            seed     = kwargs.get("seed",     self.seed)
            verbose  = kwargs.get("verbose",  self.verbose)
            n_jobs   = kwargs.get("n_jobs",   self.n_jobs)
            callback = kwargs.get("callback", self.callback)

            worker_problem = problem.make_copy()
            pymoo_problem = _PymooAdapter(worker_problem)

            if n_jobs > 1:
                try:
                    from concurrent.futures import ProcessPoolExecutor
                    from pymoo.core.problem import (  # type: ignore[import]
                        StarmapParallelization,
                    )
                    executor = ProcessPoolExecutor(max_workers=n_jobs)
                    pymoo_problem.runner = StarmapParallelization(executor.map)
                except Exception:
                    pass  # fall back to sequential

            callback = _GenerationCallback(
                user_callback=callback,
                objective_names=problem.objective_names,
                minimize_flags=problem.minimize_flags,
            )
            algorithm = NSGA2(
                pop_size=pop_size,
                sampling=FloatRandomSampling(),
                crossover=SBX(prob=0.9, eta=15),
                mutation=PM(eta=20),
                eliminate_duplicates=True,
            )
            res = pymoo_minimize(
                pymoo_problem,
                algorithm,
                ("n_gen", n_gen),
                callback=callback,
                seed=seed,
                verbose=verbose,
            )
            n_params = worker_problem.n_params
            n_obj = worker_problem.n_objectives
            if res.X is not None and res.F is not None:
                pX = np.asarray(res.X)
                pF = np.asarray(res.F)
                # pymoo returns 1-D arrays for single-objective or single-solution results
                if pX.ndim == 1:
                    pX = pX.reshape(1, -1)
                if pF.ndim == 1:
                    # (n_solutions,) when n_obj==1, or (n_obj,) for a single solution
                    pF = pF.reshape(-1, 1) if n_obj == 1 else pF.reshape(1, -1)
            else:
                pX = np.empty((0, n_params), dtype=float)
                pF = np.empty((0, n_obj), dtype=float)

            return CalibrationResult(
                history=callback.history,
                pareto_X=pX,
                pareto_F=pF,
                param_names=problem.param_names,
                objective_names=problem.objective_names,
                minimize_flags=problem.minimize_flags,
            )

except ImportError:

    class _NSGAIISolverStub:
        """Placeholder — requires ``pymoo`` (``pip install sparsehydro[pymoo]``); deprecated, use ``"nsga2"``."""

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(_INSTALL)

        def solve(self, problem) -> None:
            raise ImportError(_INSTALL)

    NSGAIISolver = _NSGAIISolverStub  # type: ignore[assignment, misc]
