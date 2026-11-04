"""Single-objective Platypus CMA-ES with independent, optionally parallel seeds.

All objectives are recorded in minimisation form, but only the selected objective
drives CMA-ES. Each seed has an exact model-evaluation budget, including any
initial points. Search coordinates are scaled to [0, 1] so disparate physical
parameter units do not distort the initial covariance.

Process-parallel seeds require ``sparsehydro[parallel]``. Sequential runs need
only the existing ``platypus-opt`` dependency.
"""

from __future__ import annotations

import random
from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np

from .base import ISolver
from ..result import CalibrationResult, GenerationRecord

if TYPE_CHECKING:
    from ..problem import CalibrationProblem


def _positive_integer(name: str, value: int, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _in_jupyter_kernel() -> bool:
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and type(shell).__name__ == "ZMQInteractiveShell"


def _cmaes_run_one_seed(
    problem: CalibrationProblem, objective_index: int, n_trials: int,
    offspring_size: int, seed: int, progress_every: int, verbose: bool,
    initial_points: np.ndarray,
) -> tuple[int, float, np.ndarray, list[GenerationRecord]]:
    import platypus

    worker = problem.make_copy()
    lower, upper = (np.asarray(b, dtype=float) for b in problem.bounds)
    span = upper - lower
    history: list[GenerationRecord] = []
    best_value = float("inf")
    best_x = None

    def evaluate(values):
        nonlocal best_value, best_x
        x = lower + np.asarray(values, dtype=float) * span
        f = np.asarray(worker.evaluate(x), dtype=float)
        if f.shape != (len(problem.objective_names),) or not np.isfinite(f).all():
            raise ValueError("CMA-ES objectives must be a finite vector matching objective_names")
        n = len(history) + 1
        history.append(GenerationRecord(
            generation=n, X=x.reshape(1, -1), F=f.reshape(1, -1), n_pareto=1,
        ))
        value = float(f[objective_index])
        if value < best_value:
            best_value, best_x = value, x.copy()
        if verbose and progress_every and n % progress_every == 0:
            display = best_value if problem.minimize_flags[objective_index] else -best_value
            print(f"  [seed {seed}] evaluation {n}/{n_trials}  "
                  f"best {problem.objective_names[objective_index]}={display:.4f}",
                  flush=True)
        return [value]

    rng_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        for point in initial_points:
            evaluate((point - lower) / span)
        size = min(offspring_size, n_trials - len(history))
        p = platypus.Problem(len(lower), 1)
        p.types[:] = [platypus.Real(0.0, 1.0) for _ in lower]
        p.directions[:] = [platypus.Problem.MINIMIZE]
        p.function = evaluate
        algorithm = platypus.CMAES(p, offspring_size=size, sigma=0.2)
        while len(history) + size <= n_trials:
            algorithm.step()
        # A partial final generation is sampled without updating the covariance:
        # no further generation follows, and the evaluation budget stays exact.
        remaining = n_trials - len(history)
        if remaining:
            for solution in algorithm.sample()[:remaining]:
                evaluate(solution.variables)
    finally:
        random.setstate(rng_state)
        np.random.set_state(numpy_state)
    if best_x is None:
        raise RuntimeError("CMA-ES completed without evaluating a solution")
    return seed, best_value, best_x, history


class PlatypusCmaEsSolver(ISolver):
    """Multi-start scalar CMA-ES, retaining all problem objectives in the result.

    ``n_trials`` is the exact number of evaluations per seed.
    ``offspring_size`` is the CMA-ES population size (at least two), not a
    random-startup budget. ``initial_points`` are physical parameter vectors
    evaluated before the evolutionary search and count against ``n_trials``.
    Explicit ``seeds`` override ``seed`` and ``multi_start``.

    ``n_jobs`` runs seeds in isolated joblib/loky processes. Sequential progress
    is printed every ``progress_every`` evaluations; parallel progress is
    reported by the parent when each seed completes. Set
    ``allow_parallel_in_kernel=False`` to disable seed processes in notebooks.
    """

    def __init__(
        self, objective: str | None = None, objective_index: int = 0,
        n_trials: int = 2000, offspring_size: int = 10, seed: int = 42,
        multi_start: int = 1, seeds: list[int] | None = None, n_jobs: int = 1,
        allow_parallel_in_kernel: bool = True, progress_every: int = 200,
        verbose: bool = True, initial_points=None,
    ) -> None:
        self.objective = objective
        self.objective_index = objective_index
        self.n_trials = n_trials
        self.offspring_size = offspring_size
        self.seed = seed
        self.multi_start = multi_start
        self.seeds = seeds
        self.n_jobs = n_jobs
        self.allow_parallel_in_kernel = allow_parallel_in_kernel
        self.progress_every = progress_every
        self.verbose = verbose
        self.initial_points = initial_points

    def solve(self, problem: CalibrationProblem, **kwargs) -> CalibrationResult:
        defaults = {name: value for name, value in vars(self).items()
                    if not name.endswith("_")}
        unknown = set(kwargs) - defaults.keys()
        if unknown:
            raise TypeError(f"Unknown CMA-ES options: {sorted(unknown)}")
        options = {name: kwargs.get(name, value) for name, value in defaults.items()}
        n_trials, size = options["n_trials"], options["offspring_size"]
        for name in ("n_trials", "offspring_size"):
            _positive_integer(name, options[name], minimum=2)
        for name in ("multi_start", "n_jobs"):
            _positive_integer(name, options[name])
        for name in ("seed", "progress_every"):
            _positive_integer(name, options[name], minimum=0)

        names = problem.objective_names
        index = options["objective_index"]
        if options["objective"] is not None:
            if options["objective"] not in names:
                raise ValueError(f"objective {options['objective']!r} not in {names}")
            index = names.index(options["objective"])
        _positive_integer("objective_index", index, minimum=0)
        if index >= len(names):
            raise ValueError(f"objective_index must be less than {len(names)}")
        lower, upper = (np.asarray(b, dtype=float) for b in problem.bounds)
        if (lower.ndim != 1 or lower.size < 2 or upper.shape != lower.shape
                or not np.isfinite([lower, upper]).all() or np.any(lower >= upper)):
            raise ValueError("CMA-ES requires at least 2 finite, strictly ordered parameter bounds")
        points = options["initial_points"]
        points = np.empty((0, len(lower))) if points is None else np.asarray(points, dtype=float)
        if (points.ndim != 2 or points.shape[1] != len(lower)
                or not np.isfinite(points).all()
                or np.any(points < lower) or np.any(points > upper)):
            raise ValueError("initial_points must be finite parameter vectors within bounds")
        if len(points) + 2 > n_trials:
            raise ValueError("n_trials must allow at least 2 CMA-ES evaluations after initial_points")

        seeds = options["seeds"]
        if seeds is None:
            seeds = [options["seed"] + i for i in range(options["multi_start"])]
        if not seeds:
            raise ValueError("seeds must not be empty")
        for seed in seeds:
            _positive_integer("seed", seed, minimum=0)
        if len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be unique")
        parallel = options["n_jobs"] > 1 and len(seeds) > 1
        verbose = options["verbose"]
        if parallel and not options["allow_parallel_in_kernel"] and _in_jupyter_kernel():
            parallel = False
            if verbose:
                print("  [multi-start] running sequential seeds in this Jupyter kernel",
                      flush=True)
        args = (problem, index, n_trials, size)
        tail = (options["progress_every"], verbose and not parallel, points)
        if parallel:
            try:
                from joblib import Parallel, delayed
            except ImportError as exc:
                raise ImportError(
                    "Parallel CMA-ES seeds require: pip install sparsehydro[parallel]"
                ) from exc
            with Parallel(n_jobs=min(options["n_jobs"], len(seeds)),
                          backend="loky", return_as="generator") as pool:
                completed = pool(delayed(_cmaes_run_one_seed)(*args, seed, *tail)
                                 for seed in seeds)
                results = []
                for result in completed:
                    results.append(result)
                    if verbose:
                        print(f"  [multi-start] seed {result[0]} completed "
                              f"({len(results)}/{len(seeds)})", flush=True)
        else:
            results = [_cmaes_run_one_seed(*args, seed, *tail) for seed in seeds]

        winner = min(results, key=lambda result: result[1])
        self.seed_solutions_ = []
        history: list[GenerationRecord] = []
        signs = np.where(problem.minimize_flags, 1.0, -1.0)
        for seed, value, x, records in results:
            best_record = min(records, key=lambda rec: rec.F[0, index])
            self.seed_solutions_.append({
                "seed": int(seed), "x": x.copy(),
                "objectives": dict(zip(names, best_record.F[0] * signs)),
                "opt_value": float(value * signs[index]), "n_trials": len(records),
                "selected": seed == winner[0],
            })
            for record in records:
                history.append(record._replace(generation=len(history) + 1))
        self.selected_seed_ = int(winner[0])
        self.objective_optimized_ = names[index]
        best_record = min(winner[3], key=lambda rec: rec.F[0, index])
        if verbose:
            print(f"CMA-ES selected seed {winner[0]}: best {names[index]}="
                  f"{winner[1] * signs[index]:.4f}", flush=True)
        return CalibrationResult(
            history=history, pareto_X=winner[2].reshape(1, -1),
            pareto_F=best_record.F.copy(), param_names=problem.param_names,
            objective_names=names, minimize_flags=problem.minimize_flags,
        )
