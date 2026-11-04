"""Generic Platypus multi-objective solver.

Wraps any :class:`platypus.Algorithm` subclass so that the full Platypus suite —
NSGA-II, NSGA-III, SPEA2, MOEA/D, GDE3, IBEA, OMOPSO, SMPSO, ε-MOEA, and more —
can calibrate any :class:`~sparsehydro.calibration.problem.CalibrationProblem`.

Requires the optional ``platypus`` dependency::

    pip install sparsehydro[platypus]

Example usage::

    import platypus
    from sparsehydro.calibration import PlatypusSolver

    # Any Platypus algorithm, any keyword arguments
    solver = PlatypusSolver(platypus.NSGAII, population_size=100, n_evaluations=10_000)
    solver = PlatypusSolver(platypus.SPEA2, population_size=50, n_evaluations=5_000)
    solver = PlatypusSolver(platypus.GDE3, population_size=80, n_evaluations=8_000)
    solver = PlatypusSolver(platypus.IBEA, population_size=100, n_evaluations=10_000)
"""

from __future__ import annotations

import multiprocessing as _mp
import pickle as _pickle
import random as _random
from concurrent.futures import ThreadPoolExecutor
from numbers import Integral
from threading import local
from typing import TYPE_CHECKING, Type

import numpy as np

from .base import ISolver
from ..result import CalibrationResult, GenerationRecord, _identify_pareto

if TYPE_CHECKING:
    from ..problem import CalibrationProblem


# ---------------------------------------------------------------------------
# Worker side of the optional process pool (``n_jobs``).
#
# These live at module scope, not inside solve(), because a closure pickles by
# reference to a name that does not exist at module scope and so cannot cross a
# process boundary.  The problem itself is shipped ONCE per worker by the pool
# initializer and cached here; each job then carries only the parameter vector.
# Pickling the problem per evaluation instead -- which is what Platypus's own
# ProcessPoolEvaluator does, since it serialises the whole Job -- would send the
# model and the full input record thousands of times and lose far more than the
# parallelism wins.
# ---------------------------------------------------------------------------

_WORKER_PROBLEM: "CalibrationProblem | None" = None

# What pickling a non-picklable problem actually raises.  A lambda fails with
# AttributeError ("Can't get local object ...<locals>.<lambda>"), not with a
# PicklingError, so the obvious single exception type is not enough.
_PICKLE_ERRORS = (AttributeError, TypeError, _pickle.PicklingError)


def _pool_init(problem: "CalibrationProblem") -> None:
    """Pool initializer: cache the problem and pay one-off costs up front.

    The warm-up evaluation forces numba JIT and FFT plan creation to happen
    during pool construction rather than inside the first generation, where it
    would serialise every worker behind the same compile.
    """
    global _WORKER_PROBLEM
    _WORKER_PROBLEM = problem
    lower, _upper = problem.bounds
    problem.evaluate(np.asarray(lower, dtype=float))


def _pool_evaluate(x) -> list[float]:
    """Evaluate one parameter vector in a worker process."""
    if _WORKER_PROBLEM is None:                      # pragma: no cover - guard
        raise RuntimeError("Worker pool was not initialised.")
    return _WORKER_PROBLEM.evaluate(np.asarray(x, dtype=float)).tolist()


def _resolve_n_jobs(n_jobs: int | None) -> int:
    """Normalise *n_jobs* to a worker count (1 means run serially).

    ``-1`` means "auto".  Auto caps at 8 rather than using every core: worker
    start-up costs about 3 s each (process spawn plus re-importing numpy,
    pandas and sparsehydro), and that is paid before the first generation runs.
    Past roughly 8 workers the extra start-up outweighs the faster search on a
    single calibration -- measured end to end, 24 workers finished a 120-
    generation run slower than 6.
    """
    if n_jobs is None:
        return 1
    if n_jobs == -1:
        return max(1, min(8, _mp.cpu_count()))
    if n_jobs < 1:
        raise ValueError(f"n_jobs must be >= 1, or -1 for auto; got {n_jobs}.")
    return int(n_jobs)


try:
    import platypus  # type: ignore[import]

    class _PoolEvaluator(platypus.evaluator.Evaluator):
        """Evaluate a generation across a worker pool, sending only the vectors.

        Platypus ships a ``ProcessPoolEvaluator``, but it pickles the entire Job
        -- and therefore the Solution, the Problem and the whole
        :class:`CalibrationProblem` behind it -- once per *evaluation*.  For a
        calibration that holds a multi-year input record that is megabytes on
        every one of tens of thousands of evaluations.  It also offers no
        ``initializer`` hook, so there is no way to seed the worker once.

        This evaluator keeps the problem resident in each worker (see
        :func:`_pool_init`) and sends only the decision vector, so the payload
        per evaluation is a few kilobytes.

        The body mirrors ``platypus.Problem.__call__``: decode the variables,
        evaluate, write the objectives back, mark the solution evaluated.
        ``CalibrationProblem`` declares no Platypus constraints -- bound
        violations are folded into the objectives as a penalty -- so every
        solution is feasible with zero violation, exactly as in the serial path.
        """

        def __init__(self, pool, chunksize: int = 1) -> None:
            super().__init__()
            self.pool = pool
            self.chunksize = chunksize

        def evaluate_all(self, jobs, **kwargs):
            jobs = list(jobs)
            if not jobs:
                return jobs

            vectors = []
            for job in jobs:
                solution = job.solution
                p = solution.problem
                vectors.append([
                    p.types[i].decode(solution.variables[i])
                    for i in range(p.nvars)
                ])

            objectives = self.pool.map(
                _pool_evaluate, vectors, chunksize=self.chunksize
            )

            for job, F in zip(jobs, objectives):
                solution = job.solution
                solution.objectives[:] = F
                solution.constraints[:] = []
                solution.constraint_violation = 0.0
                solution.feasible = True
                solution.evaluated = True
            return jobs

    class PlatypusSolver(ISolver):
        """Generic Platypus multi-objective solver.

        Wraps any :class:`platypus.Algorithm` subclass — NSGA-II, NSGA-III,
        SPEA2, MOEA/D, GDE3, IBEA, OMOPSO, SMPSO, ε-MOEA, etc. — in the
        :class:`~sparsehydro.calibration.solvers.ISolver` interface.

        The algorithm class and all of its constructor keyword arguments (except
        ``problem``, which is built internally) are passed directly through, so
        any algorithm-specific tuning is fully accessible.

        :param algorithm_class: A Platypus algorithm *class* (not instance), e.g.
            ``platypus.NSGA2``, ``platypus.SPEA2``, ``platypus.MOEAD``.
        :type algorithm_class: type
        :param n_evaluations: Total function-evaluation budget.  The solver stops
            when ``algorithm.nfe >= n_evaluations``.
        :type n_evaluations: int
        :param seed: Seed for :mod:`random` and :mod:`numpy.random` (``None``
            for non-deterministic runs).
        :type seed: int | None
        :param record_frequency: Append a :class:`~sparsehydro.calibration.result.GenerationRecord`
            to history every *record_frequency* algorithm iterations.  Set to a
            larger value to reduce memory usage on long runs.
        :type record_frequency: int
        :param n_jobs: Number of worker processes used to evaluate each
            generation.  ``None`` (the default) runs serially in this process.
            ``-1`` picks a sensible worker count automatically.  Results are
            unchanged either way: the algorithm's randomness is drawn in the
            parent process and evaluation order never feeds back into it, so a
            parallel run reproduces the serial Pareto front exactly.

            Parallel evaluation requires the problem to be picklable, which
            rules out ``lambda`` extractors — pass column names instead::

                EnsembleModel(components=[(rdii, "rdii_cfs"), ...])

            More workers is not automatically better.  Each one costs about
            three seconds to start (process spawn plus re-importing numpy,
            pandas and sparsehydro) and that is paid before the first
            generation; on a 24-core machine a 120-generation run finished
            faster with 6 workers than with 24.  Six to eight is a good default
            for a single calibration; go wider only when the pool amortises
            over a long run.
        :type n_jobs: int | None
        :param n_workers: Ordered evaluation threads, each with a private model
            copy. Defaults to 1; cannot be combined with parallel ``n_jobs`` or
            a custom evaluator. Callbacks/objectives must be thread-safe.
        :type n_workers: int
        :param algorithm_kwargs: All remaining keyword arguments are forwarded
            verbatim to ``algorithm_class(problem, **algorithm_kwargs)``.

        Examples::

            import platypus
            from sparsehydro.calibration import PlatypusSolver

            # NSGA-II with explicit population size
            solver = PlatypusSolver(platypus.NSGAII, population_size=100, n_evaluations=10_000)

            # The same, evaluated across six worker processes
            solver = PlatypusSolver(
                platypus.NSGAII,
                population_size=100,
                n_evaluations=10_000,
                n_jobs=6,
            )

            # SPEA2
            solver = PlatypusSolver(platypus.SPEA2, population_size=50, n_evaluations=5_000)

            # GDE3 (good for real-valued problems)
            solver = PlatypusSolver(platypus.GDE3, population_size=80, n_evaluations=8_000)

            # ε-MOEA with epsilon archive
            solver = PlatypusSolver(
                platypus.EpsMOEA,
                epsilons=[0.01, 0.01],
                population_size=100,
                n_evaluations=10_000,
            )

            result = solver.solve(problem)
        """

        def __init__(
            self,
            algorithm_class: Type[platypus.Algorithm],
            n_evaluations: int = 10_000,
            seed: int | None = 42,
            record_frequency: int = 1,
            callback=None,
            n_jobs: int | None = None,
            *,
            n_workers: int = 1,
            **algorithm_kwargs,
        ) -> None:
            self.algorithm_class = algorithm_class
            self.n_evaluations = n_evaluations
            self.seed = seed
            self.record_frequency = record_frequency
            self.callback = callback
            self.n_jobs = n_jobs
            self.n_workers = n_workers
            self.algorithm_kwargs = algorithm_kwargs

        def solve(self, problem: "CalibrationProblem", **kwargs) -> CalibrationResult:
            """Run the Platypus algorithm and return a :class:`~sparsehydro.calibration.result.CalibrationResult`.

            :param problem: Calibration problem wrapping model + data + objectives.
            :type problem: CalibrationProblem
            :param kwargs: Per-call overrides: ``n_evaluations``, ``seed``,
                ``record_frequency``, ``callback``, ``n_jobs``.  Additional keys are
                merged into ``algorithm_kwargs`` and forwarded to the algorithm
                constructor.
            :returns: Result with per-iteration history and final Pareto front.
            :rtype: CalibrationResult
            """
            n_evaluations    = kwargs.pop("n_evaluations",    self.n_evaluations)
            seed             = kwargs.pop("seed",             self.seed)
            record_frequency = kwargs.pop("record_frequency", self.record_frequency)
            user_callback    = kwargs.pop("callback",         self.callback)
            n_jobs           = kwargs.pop("n_jobs",           self.n_jobs)
            thread_workers   = kwargs.pop("n_workers",        self.n_workers)
            algorithm_kwargs = {**self.algorithm_kwargs, **kwargs}

            n_workers = _resolve_n_jobs(n_jobs)
            if isinstance(thread_workers, bool) or not isinstance(thread_workers, Integral) or thread_workers < 1:
                raise ValueError("n_workers must be a positive integer.")
            if n_workers > 1 and thread_workers > 1:
                raise ValueError("n_jobs and n_workers cannot both enable parallel evaluation.")
            if thread_workers > 1 and algorithm_kwargs.get("evaluator") is not None:
                raise ValueError("n_workers > 1 cannot be combined with a custom evaluator.")
            if n_workers > 1 and "evaluator" in algorithm_kwargs:
                raise ValueError(
                    "n_jobs and an explicit 'evaluator' cannot both be given; "
                    "drop one."
                )

            if seed is not None:
                _random.seed(seed)
                np.random.seed(seed)

            worker = problem.make_copy()
            xl, xu = worker.bounds
            n_params = worker.n_params
            n_obj = worker.n_objectives
            thread_state = local()

            def _initialize_thread() -> None:
                thread_state.problem = worker.make_copy()

            # Build the platypus Problem ------------------------------------------
            platypus_problem = platypus.Problem(n_params, n_obj)
            platypus_problem.types[:] = [
                platypus.Real(float(lo), float(hi)) for lo, hi in zip(xl, xu)
            ]
            platypus_problem.directions[:] = [platypus.Problem.MINIMIZE] * n_obj

            def _evaluate(vars_list: list) -> list:
                x = np.array(vars_list, dtype=float)
                evaluation_worker = worker if thread_workers == 1 else thread_state.problem
                return evaluation_worker.evaluate(x).tolist()

            platypus_problem.function = _evaluate

            pool = None
            thread_pool = None
            try:
                if thread_workers > 1:
                    thread_pool = ThreadPoolExecutor(
                        max_workers=int(thread_workers), initializer=_initialize_thread,
                        thread_name_prefix="sparsehydro-calibration",
                    )
                    algorithm_kwargs["evaluator"] = platypus.MapEvaluator(thread_pool.map)
                elif n_workers > 1:
                    # Check picklability here rather than letting the pool fail.
                    # Pool() spawns every worker before the first one reports a
                    # bad payload, so the failure arrives buried under EOFError
                    # tracebacks from the siblings.  One serialisation up front
                    # (milliseconds, once per solve) buys a clean message.
                    try:
                        _pickle.dumps(worker)
                    except _PICKLE_ERRORS as exc:
                        raise RuntimeError(
                            "Could not send the calibration problem to worker "
                            f"processes (n_jobs={n_jobs}): {exc}.  Every part of "
                            "the problem must be picklable, which lambdas and "
                            "other locally defined functions are not.  The usual "
                            "culprit is a lambda extractor -- pass a column name "
                            'instead, e.g. EnsembleModel(components=[(rdii, '
                            '"rdii_cfs"), ...]).  Omit n_jobs to run serially.'
                        ) from exc

                    # "spawn" rather than "fork": required on Windows, and the
                    # only context that is safe with the threaded BLAS and numba
                    # runtimes the models sit on.
                    ctx = _mp.get_context("spawn")
                    pool = ctx.Pool(
                        processes=n_workers,
                        initializer=_pool_init,
                        initargs=(worker,),
                    )
                    algorithm_kwargs["evaluator"] = _PoolEvaluator(pool)

                return self._run(
                    platypus_problem,
                    algorithm_kwargs,
                    problem=problem,
                    n_evaluations=n_evaluations,
                    record_frequency=record_frequency,
                    user_callback=user_callback,
                    n_params=n_params,
                    n_obj=n_obj,
                )
            finally:
                if thread_pool is not None:
                    thread_pool.shutdown(wait=True)
                if pool is not None:
                    pool.terminate()
                    pool.join()

        def _run(
            self,
            platypus_problem,
            algorithm_kwargs: dict,
            *,
            problem: "CalibrationProblem",
            n_evaluations: int,
            record_frequency: int,
            user_callback,
            n_params: int,
            n_obj: int,
        ) -> CalibrationResult:
            """Step the algorithm to the evaluation budget and collect results.

            Split out of :meth:`solve` only so the worker pool there has a
            single, obvious lifetime; the logic is identical either way.
            """
            # Build and step the algorithm ----------------------------------------
            algorithm = self.algorithm_class(platypus_problem, **algorithm_kwargs)

            history: list[GenerationRecord] = []
            iteration = 0

            # Use step() rather than initialize()/iterate() directly so that
            # algorithm.result is updated after each step (it is only written
            # inside step(), not inside the lower-level methods).
            while algorithm.nfe < n_evaluations:
                algorithm.step()
                iteration += 1

                if iteration % record_frequency == 0:
                    current = algorithm.result
                    if current:
                        X = np.array(
                            [list(sol.variables) for sol in current], dtype=float
                        )
                        F = np.array(
                            [list(sol.objectives) for sol in current], dtype=float
                        )
                        pareto_mask = _identify_pareto(F)
                        n_pareto = int(np.sum(pareto_mask))
                        history.append(
                            GenerationRecord(
                                generation=iteration,
                                X=X,
                                F=F,
                                n_pareto=n_pareto,
                            )
                        )
                        if user_callback is not None:
                            user_callback({
                                "generation": iteration,
                                "n_evals": int(algorithm.nfe),
                                "n_pareto": n_pareto,
                                "pareto_F": F[pareto_mask],
                                "objective_names": problem.objective_names,
                                "minimize_flags": problem.minimize_flags,
                            })

            # Final Pareto front --------------------------------------------------
            final = algorithm.result
            if final:
                all_X = np.array([list(sol.variables) for sol in final], dtype=float)
                all_F = np.array([list(sol.objectives) for sol in final], dtype=float)
                pareto_mask = _identify_pareto(all_F)
                pareto_X = all_X[pareto_mask]
                pareto_F = all_F[pareto_mask]
            else:
                pareto_X = np.empty((0, n_params), dtype=float)
                pareto_F = np.empty((0, n_obj), dtype=float)

            return CalibrationResult(
                history=history,
                pareto_X=pareto_X,
                pareto_F=pareto_F,
                param_names=problem.param_names,
                objective_names=problem.objective_names,
                minimize_flags=problem.minimize_flags,
            )

    class ParticleSwarmSolver(ISolver):
        """Multi-objective Particle Swarm solver backed by Platypus.

        Selects ``platypus.OMOPSO`` (ε-dominance archive) when *epsilons* are
        provided, otherwise uses ``platypus.SMPSO``.

        :param swarm_size: Number of particles in the swarm.
        :type swarm_size: int
        :param leader_size: Size of the leader archive.
        :type leader_size: int
        :param n_evaluations: Total function-evaluation budget.
        :type n_evaluations: int
        :param epsilons: Per-objective ε values for OMOPSO; ``None`` uses SMPSO.
            Length must equal the number of objectives.
        :type epsilons: list or None
        :param seed: Random seed (``None`` for non-deterministic).
        :type seed: int or None
        :param record_frequency: History snapshot interval (algorithm iterations).
        :type record_frequency: int
        :param n_jobs: Worker processes used to evaluate each generation;
            ``None`` runs serially.  See :class:`PlatypusSolver` for the
            picklability requirement and for why more workers is not always
            faster.
        :type n_jobs: int | None
        :param algorithm_kwargs: Additional keyword arguments forwarded to the
            Platypus algorithm constructor.
        :type algorithm_kwargs: Any
        """

        def __init__(
            self,
            swarm_size: int = 100,
            leader_size: int = 100,
            n_evaluations: int = 10_000,
            epsilons: list | None = None,
            seed: int | None = 42,
            record_frequency: int = 1,
            callback=None,
            n_jobs: int | None = None,
            **algorithm_kwargs,
        ) -> None:
            self.swarm_size = swarm_size
            self.leader_size = leader_size
            self.n_evaluations = n_evaluations
            self.epsilons = epsilons
            self.seed = seed
            self.record_frequency = record_frequency
            self.callback = callback
            self.n_jobs = n_jobs
            self.algorithm_kwargs = algorithm_kwargs

        def solve(self, problem: "CalibrationProblem", **kwargs) -> "CalibrationResult":
            """Run the PSO solver and return a :class:`~sparsehydro.calibration.result.CalibrationResult`.

            :param problem: Calibration problem wrapping model + data + objectives.
            :type problem: CalibrationProblem
            :param kwargs: Per-call overrides: ``swarm_size``, ``leader_size``,
                ``n_evaluations``, ``seed``, ``record_frequency``, ``epsilons``,
                ``callback``, ``n_jobs``.
            :returns: Result with per-iteration history and final Pareto front.
            :rtype: CalibrationResult
            """
            swarm_size       = kwargs.pop("swarm_size",       self.swarm_size)
            leader_size      = kwargs.pop("leader_size",      self.leader_size)
            n_evaluations    = kwargs.pop("n_evaluations",    self.n_evaluations)
            epsilons         = kwargs.pop("epsilons",         self.epsilons)
            seed             = kwargs.pop("seed",             self.seed)
            record_frequency = kwargs.pop("record_frequency", self.record_frequency)
            callback         = kwargs.pop("callback",         self.callback)
            n_jobs           = kwargs.pop("n_jobs",           self.n_jobs)

            if epsilons is not None and len(epsilons) != problem.n_objectives:
                raise ValueError(
                    f"epsilons length ({len(epsilons)}) must match the number of "
                    f"objectives ({problem.n_objectives})."
                )

            algo = platypus.OMOPSO if epsilons is not None else platypus.SMPSO
            extra = {"epsilons": epsilons} if epsilons is not None else {}

            return PlatypusSolver(
                algo,
                swarm_size=swarm_size,
                leader_size=leader_size,
                n_evaluations=n_evaluations,
                seed=seed,
                record_frequency=record_frequency,
                callback=callback,
                n_jobs=n_jobs,
                **extra,
                **{**self.algorithm_kwargs, **kwargs},
            ).solve(problem)

except ImportError:

    class _PlatypusSolverStub:
        """Placeholder — requires ``platypus-opt``.

        Install with::

            pip install sparsehydro[platypus]
        """

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "platypus-opt is required for PlatypusSolver. "
                "Install with: pip install sparsehydro[platypus]"
            )

        def solve(self, problem) -> None:  # type: ignore[return]
            raise ImportError(
                "platypus-opt is required for PlatypusSolver. "
                "Install with: pip install sparsehydro[platypus]"
            )

    PlatypusSolver = _PlatypusSolverStub  # type: ignore[assignment, misc]

    def ParticleSwarmSolver(*args, **kwargs):  # type: ignore[misc]
        raise ImportError(
            "platypus-opt is required for ParticleSwarmSolver. "
            "Install with: pip install sparsehydro[platypus]"
        )
