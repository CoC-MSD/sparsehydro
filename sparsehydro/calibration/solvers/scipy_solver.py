"""SciPy-backed single-objective calibration solver.

Requires the optional ``scipy`` dependency::

    pip install sparsehydro[rdii]

For multi-objective calibration problems, only one objective (selected by
:attr:`objective_index`) is optimised; the remaining objectives are evaluated
at the final solution and stored in the result for reference.

Two solver modes are supported:

* ``"differential_evolution"`` (default) — global stochastic optimiser; uses
  parameter bounds directly; recommended for non-convex problems.
* Any ``scipy.optimize.minimize`` method string (e.g. ``"L-BFGS-B"``) —
  gradient-based local optimiser; starts from the midpoint of each bound.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import ISolver
from ..result import CalibrationResult, GenerationRecord

if TYPE_CHECKING:
    from ..problem import CalibrationProblem


try:
    from scipy.optimize import differential_evolution  # type: ignore[import]
    from scipy.optimize import minimize as scipy_minimize  # type: ignore[import]

    class ScipySolver(ISolver):
        """Single-objective SciPy calibration solver.

        :param method: ``"differential_evolution"`` (default) or any
            ``scipy.optimize.minimize`` method string (e.g. ``"L-BFGS-B"``).
        :type method: str
        :param objective_index: Zero-based index of the objective to optimise
            when the problem has multiple objectives.
        :type objective_index: int
        :param maxiter: Maximum number of iterations / generations.
        :type maxiter: int
        :param seed: Random seed (used by ``differential_evolution``).
        :type seed: int or None
        :param verbose: Print solver progress.
        :type verbose: bool
        :param kwargs: Additional keyword arguments forwarded to the underlying
            SciPy function.
        :type kwargs: Any
        """

        def __init__(
            self,
            method: str = "differential_evolution",
            objective_index: int = 0,
            maxiter: int = 1000,
            seed: int | None = 42,
            verbose: bool = False,
            **kwargs,
        ) -> None:
            self.method = method
            self.objective_index = objective_index
            self.maxiter = maxiter
            self.seed = seed
            self.verbose = verbose
            self._kwargs = kwargs

        def solve(self, problem: "CalibrationProblem", **kwargs) -> CalibrationResult:
            """Run the SciPy solver and return a
            :class:`~sparsehydro.calibration.result.CalibrationResult`.

            The Pareto "front" contains exactly one solution — the optimum
            found for the selected objective.

            :param problem: Calibration problem wrapping model + data + objectives.
            :type problem: CalibrationProblem
            :param kwargs: Per-call overrides: ``method``, ``objective_index``,
                ``maxiter``, ``seed``, ``verbose``.
            :returns: Result with the single best solution.
            :rtype: CalibrationResult
            """
            method          = kwargs.get("method",          self.method)
            obj_idx         = kwargs.get("objective_index", self.objective_index)
            maxiter         = kwargs.get("maxiter",         self.maxiter)
            seed            = kwargs.get("seed",            self.seed)
            verbose         = kwargs.get("verbose",         self.verbose)

            worker = problem.make_copy()
            xl, xu = problem.bounds
            history: list[GenerationRecord] = []

            def scalar_fn(x: np.ndarray) -> float:
                # Only objective obj_idx is optimised; evaluate_single skips
                # computing the others.
                return worker.evaluate_single(x, obj_idx)

            if method == "differential_evolution":
                bounds = list(zip(xl.tolist(), xu.tolist()))
                iteration = [0]

                def _callback(xk, convergence):
                    iteration[0] += 1
                    val = scalar_fn(xk)
                    if verbose:
                        print(f"Iteration {iteration[0]}: f={val:.6g}")
                    F_arr = np.array([[val]])
                    history.append(
                        GenerationRecord(
                            generation=iteration[0],
                            X=xk.reshape(1, -1),
                            F=F_arr,
                            n_pareto=1,
                        )
                    )

                res = differential_evolution(
                    scalar_fn,
                    bounds=bounds,
                    maxiter=maxiter,
                    seed=seed,
                    callback=_callback,
                    **self._kwargs,
                )
                best_x = res.x

            else:
                from scipy.optimize import Bounds  # type: ignore[import]

                x0 = (xl + xu) / 2.0
                res = scipy_minimize(
                    scalar_fn,
                    x0,
                    method=method,
                    bounds=Bounds(lb=xl, ub=xu),
                    options={"maxiter": maxiter, "disp": verbose},
                    **self._kwargs,
                )
                best_x = res.x

            pareto_F = worker.evaluate(best_x).reshape(1, -1)
            return CalibrationResult(
                history=history,
                pareto_X=best_x.reshape(1, -1),
                pareto_F=pareto_F,
                param_names=problem.param_names,
                objective_names=problem.objective_names,
                minimize_flags=problem.minimize_flags,
            )

except ImportError:

    class ScipySolver:  # type: ignore[no-redef]
        """Placeholder — requires ``scipy``.

        Install with::

            pip install sparsehydro[rdii]
        """

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "scipy is required for ScipySolver. "
                "Install with: pip install sparsehydro[rdii]"
            )

        def solve(self, problem) -> None:  # type: ignore[misc]
            raise ImportError(
                "scipy is required for ScipySolver. "
                "Install with: pip install sparsehydro[rdii]"
            )
