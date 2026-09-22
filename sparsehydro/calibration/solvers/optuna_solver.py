"""Optuna-backed single-objective CMA-ES calibration solver.

Requires the optional ``optuna`` dependency (and ``joblib`` for the process-
parallel multi-seed path)::

    pip install sparsehydro[optuna]

CMA-ES is a *single-objective* optimiser: for a multi-objective
:class:`~sparsehydro.calibration.problem.CalibrationProblem` it optimises exactly
one objective (selected by name via ``objective=`` or by ``objective_index``),
while the remaining objectives are still evaluated and recorded every trial so the
downstream Pareto / knee / scorecard machinery keeps working. The result carries a
single-solution Pareto "front".

The search is multi-seed and, optionally, process-parallel across seeds (CMA-ES is
basin-sensitive, so several independent seeds are run and the global best kept) —
a design ported from the swmmrdii CalVal template (``CalOptimizer.optimize``).
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import numpy as np

from .base import ISolver
from ..result import CalibrationResult, GenerationRecord

if TYPE_CHECKING:
    from ..problem import CalibrationProblem


try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Uses the `optuna` package directly (optuna.samplers.CmaEsSampler — the same
    # sampler the swmmrdii CalVal template drives under the hood), wrapped to
    # conform to the sparsehydro `ISolver` interface so it plugs into a
    # CalibrationProblem exactly like PlatypusSolver / ScipySolver. That optimizer
    # is bound to a SWMM RdiiModel and can't drive the sparsehydro EnsembleModel
    # here, so this is a from-scratch adapter — but the MULTI-SEED PARALLEL design
    # is ported faithfully from the template (swmmrdii.CalOptimizer.optimize):
    #
    #   • CMA-ES is basin-sensitive, so we run several independent seeds and keep
    #     the GLOBAL best rather than trusting any single trajectory.
    #   • Parallelism is ACROSS SEEDS, not across trials within a study. The hot
    #     path (pure-Python objective eval) holds the GIL, so threads give no
    #     speedup — we use PROCESSES via joblib/loky. n_jobs > 1 runs that many
    #     seeds in separate processes; each rebuilds its OWN isolated problem copy,
    #     so their mutable model state never collides.
    #   • Worker stdout can't reach a Jupyter cell, so parallel workers write
    #     best-so-far to a per-seed file that the parent polls and prints.
    #   • allow_parallel_in_kernel=False forces sequential seeds inside a notebook
    #     kernel (escape hatch); scripts/terminals always parallelise.
    #
    # CMA-ES is SINGLE-objective. The problem still declares BOTH KGE + CCC so the
    # §9-§11 Pareto / knee / best_by machinery keeps working; the solver optimises
    # exactly ONE of them (selected by name via `objective=`, here "ccc"), while the
    # other is still evaluated and recorded every trial.

    def _in_jupyter_kernel() -> bool:
        """True inside an IPython/Jupyter ZMQ kernel (where loky child-process spawn
        can occasionally destabilise an embedded kernel)."""
        try:
            from IPython import get_ipython
            ip = get_ipython()
            return ip is not None and ip.__class__.__name__ == "ZMQInteractiveShell"
        except Exception:
            return False


    def _cmaes_run_one_seed(problem, pnames, xl, xu, obj_idx, maximise, obj_label,
                            n_trials, n_startup_trials, seed, si, n_seeds,
                            progress_every, progress_file=None, verbose=True):
        """Run ONE CMA-ES seed to completion on an ISOLATED problem copy.

        Defined at cell scope (not a closure/bound method) so cloudpickle — which
        joblib/loky uses — can serialize it into a worker process. Returns
        ``(seed, best_val, best_x, history)`` with ``best_val`` in minimisation
        form. ``progress_file`` set -> parallel worker: write best-so-far to that
        file for the parent to poll. ``None`` -> sequential: print directly, which
        streams in every context (notebook / script / terminal).
        """
        # Workers are separate processes that don't inherit the parent's logging
        # config, so silence Optuna's per-trial INFO spam here too.

        worker = problem.make_copy()      # private prepared-model state per seed
        history: list[GenerationRecord] = []
        gen = 0

        def _objective(trial):
            nonlocal gen
            x = np.array(
                [trial.suggest_float(pn, float(lo), float(hi))
                 for pn, lo, hi in zip(pnames, xl, xu)],
                dtype=float,
            )
            F = np.asarray(worker.evaluate(x), dtype=float)   # minimisation form
            gen += 1
            history.append(GenerationRecord(
                generation=gen, X=x.reshape(1, -1), F=F.reshape(1, -1), n_pareto=1,
            ))
            return float(F[obj_idx])

        is_parallel = progress_file is not None

        def _cb(study, trial):
            n = trial.number + 1
            if not progress_every or n % progress_every != 0:
                return
            try:
                bv = study.best_value
            except ValueError:
                return   # no completed trial yet
            if is_parallel:
                try:
                    with open(progress_file, "w") as fh:
                        fh.write(f"{n} {bv:.6f}")
                except OSError:
                    pass
            elif verbose:
                disp = -bv if maximise else bv
                tag = f"seed {seed} ({si + 1}/{n_seeds})" if n_seeds > 1 else f"seed {seed}"
                print(f"  [{tag}] trial {n}/{n_trials}  best {obj_label}={disp:.4f}",
                      flush=True)

        if n_seeds > 1 and not is_parallel:
            print(f"  [multi-start {si + 1}/{n_seeds}]  seed={seed}", flush=True)

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.CmaEsSampler(
                n_startup_trials=n_startup_trials, seed=seed,
            ),
        )
        study.optimize(_objective, n_trials=n_trials, callbacks=[_cb])
        best_x = np.array([study.best_params[pn] for pn in pnames], dtype=float)
        return seed, float(study.best_value), best_x, history


    class OptunaCmaEsSolver(ISolver):
        """Single-objective CMA-ES calibration solver backed directly by Optuna,
        with multi-seed process-parallel search ported from the CalVal template.

        :param objective: Objective name (e.g. ``"ccc"``) to optimise. Overrides
            ``objective_index`` when given.
        :param objective_index: Zero-based objective to optimise when ``objective``
            is ``None`` and the problem declares several objectives.
        :param n_trials: CMA-ES trials PER SEED.
        :param n_startup_trials: Random startup trials before CMA-ES engages.
        :param seed: Base RNG seed; ``multi_start`` seeds run ``seed, seed+1, …``.
        :param multi_start: Number of independent CMA-ES seeds; the global best is
            kept. CMA-ES is basin-sensitive, so >1 makes the result robust.
        :param seeds: Explicit list of seeds; when given, ``multi_start``/``seed``
            are ignored and ``len(seeds)`` runs are launched.
        :param n_jobs: Number of seeds to run CONCURRENTLY as separate processes
            (joblib/loky). 1 -> sequential. Capped at the number of seeds.
        :param allow_parallel_in_kernel: When False, force sequential seeds inside a
            Jupyter kernel even if ``n_jobs > 1`` (escape hatch).
        :param progress_every: Report best-so-far every N trials (0 disables).
        :param verbose: Print per-seed / progress lines.
        """

        def __init__(self, objective=None, objective_index=0, n_trials=2000,
                     n_startup_trials=10, seed=42, multi_start=1, seeds=None,
                     n_jobs=1, allow_parallel_in_kernel=True,
                     progress_every=200, verbose=True, **study_kwargs):
            self.objective = objective
            self.objective_index = objective_index
            self.n_trials = n_trials
            self.n_startup_trials = n_startup_trials
            self.seed = seed
            self.multi_start = multi_start
            self.seeds = seeds
            self.n_jobs = n_jobs
            self.allow_parallel_in_kernel = allow_parallel_in_kernel
            self.progress_every = progress_every
            self.verbose = verbose
            self._study_kwargs = study_kwargs   # accepted for API parity; unused here

        def solve(self, problem, **kwargs):
            g = kwargs.get
            objective        = g("objective",        self.objective)
            obj_idx          = g("objective_index",  self.objective_index)
            n_trials         = g("n_trials",         self.n_trials)
            n_startup_trials = g("n_startup_trials", self.n_startup_trials)
            seed             = g("seed",             self.seed)
            multi_start      = g("multi_start",      self.multi_start)
            seeds            = g("seeds",            self.seeds)
            n_jobs           = g("n_jobs",           self.n_jobs)
            allow_in_kernel  = g("allow_parallel_in_kernel", self.allow_parallel_in_kernel)
            progress_every   = g("progress_every",   self.progress_every)
            verbose          = g("verbose",          self.verbose)

            names = problem.objective_names
            if objective is not None:
                if objective not in names:
                    raise ValueError(f"objective {objective!r} not in {names}")
                obj_idx = names.index(objective)
            maximise = not problem.minimize_flags[obj_idx]
            obj_label = names[obj_idx]

            xl, xu = problem.bounds
            pnames = problem.param_names
            if len(pnames) < 2:
                raise ValueError("CMA-ES requires at least 2 search dimensions.")

            # Seed list: explicit `seeds` wins; else `multi_start` seeds from `seed`.
            if seeds is not None:
                seed_list = [int(s) for s in seeds]
            else:
                seed_list = [int(seed) + i for i in range(max(1, int(multi_start)))]
            n_seeds = len(seed_list)

            # Parallelise ACROSS SEEDS only (needs >1 seed and n_jobs > 1).
            parallel_seeds = n_jobs > 1 and n_seeds > 1
            if parallel_seeds and _in_jupyter_kernel() and not allow_in_kernel:
                parallel_seeds = False
                if verbose:
                    print(f"  [multi-start] n_jobs={n_jobs} requested, but "
                          f"allow_parallel_in_kernel=False — running SEQUENTIAL seeds "
                          f"in this Jupyter kernel.", flush=True)

            t0 = time.time()
            common = (pnames, xl, xu, obj_idx, maximise, obj_label,
                      n_trials, n_startup_trials)

            if parallel_seeds:
                seed_results = self._run_parallel(
                    problem, seed_list, common, min(n_jobs, n_seeds),
                    progress_every, verbose,
                )
            else:
                seed_results = [
                    _cmaes_run_one_seed(problem, *common, s, si, n_seeds,
                                        progress_every, None, verbose)
                    for si, s in enumerate(seed_list)
                ]

            # Global best across seeds (best_val is in minimisation form => min()).
            best_seed, best_val, best_x, _ = min(seed_results, key=lambda r: r[1])

            # Concatenate per-seed histories, renumbering generations monotonically
            # so the §9 evolution plots read as one continuous search.
            history: list[GenerationRecord] = []
            gi = 0
            for _s, _v, _x, hist in seed_results:
                for rec in hist:
                    gi += 1
                    history.append(rec._replace(generation=gi))

            # ── Per-seed solutions (for the seed-comparison chart in §9) ─────────
            # Keep EVERY seed's best solution, not just the global winner, so a
            # later cell can chart the seed-to-seed spread (the payoff of the
            # multi-start). Objectives are stored in DISPLAY form (maximised
            # objectives un-negated), matching objective_display_values().
            self.seed_solutions_ = []
            for _s, _v, _x, _h in sorted(seed_results, key=lambda r: r[0]):
                _F_min = np.asarray(problem.make_copy().evaluate(_x), dtype=float)
                _F_disp = np.array(
                    [(-f if not mf else f)
                     for f, mf in zip(_F_min, problem.minimize_flags)],
                    dtype=float,
                )
                self.seed_solutions_.append({
                    "seed": int(_s),
                    "x": np.asarray(_x, dtype=float),
                    "objectives": dict(zip(names, _F_disp)),
                    "opt_value": float(-_v if maximise else _v),
                    "n_trials": len(_h),
                    "selected": int(_s) == int(best_seed),
                })
            self.selected_seed_ = int(best_seed)
            self.objective_optimized_ = obj_label

            if verbose and n_seeds > 1:
                scores = "  ".join(
                    f"s{s}={(-v if maximise else v):.4f}"
                    for s, v, _x, _h in sorted(seed_results, key=lambda r: r[0])
                )
                print(f"  [results]  {scores}", flush=True)
                print(f"  [selected] seed {best_seed} "
                      f"({obj_label}={(-best_val if maximise else best_val):.4f})",
                      flush=True)

            pareto_F = problem.make_copy().evaluate(best_x).reshape(1, -1)
            if verbose:
                disp = -best_val if maximise else best_val
                print(f"CMA-ES finished in {time.time() - t0:.1f} s — "
                      f"{n_seeds} seed(s) x {n_trials} trials"
                      + (f" across {min(n_jobs, n_seeds)} processes" if parallel_seeds else "")
                      + f", global best {obj_label}={disp:.4f}", flush=True)

            return CalibrationResult(
                history=history,
                pareto_X=best_x.reshape(1, -1),
                pareto_F=pareto_F,
                param_names=pnames,
                objective_names=names,
                minimize_flags=problem.minimize_flags,
            )

        # ------------------------------------------------------------------
        # Process-parallel seed dispatch (joblib/loky) with polled file progress
        # ------------------------------------------------------------------
        def _run_parallel(self, problem, seed_list, common, workers,
                          progress_every, verbose):
            """Run every seed in ``workers`` loky processes; poll per-seed progress
            files and print them from the parent. Ported from the template's
            optimize() parallel branch."""
            import tempfile, threading, shutil
            from joblib import Parallel, delayed

            n_seeds = len(seed_list)
            maximise = common[4]
            n_trials = common[6]
            if verbose:
                print(f"  [multi-start] running {n_seeds} seeds in {workers} processes",
                      flush=True)

            progress_dir = tempfile.mkdtemp(prefix="cmaes_progress_")
            holder: dict = {}

            def _dispatch():
                # Capture ANY failure so the parent can re-raise it instead of the
                # opaque KeyError('res') a dead thread would otherwise produce.
                try:
                    holder["res"] = Parallel(n_jobs=workers, backend="loky")(
                        delayed(_cmaes_run_one_seed)(
                            problem, *common, s, si, n_seeds, progress_every,
                            os.path.join(progress_dir, f"seed_{s}.txt"), verbose,
                        )
                        for si, s in enumerate(seed_list)
                    )
                except BaseException as exc:   # noqa: BLE001 — re-raised below
                    holder["err"] = exc

            worker_thread = threading.Thread(target=_dispatch)
            worker_thread.start()
            try:
                prev_snapshot = None
                while worker_thread.is_alive():
                    worker_thread.join(timeout=3.0)
                    snapshot = {}
                    for s in seed_list:
                        try:
                            with open(os.path.join(progress_dir, f"seed_{s}.txt")) as fh:
                                snapshot[s] = fh.read().strip()
                        except OSError:
                            snapshot[s] = ""   # not reported yet
                    if snapshot == prev_snapshot or not any(snapshot.values()):
                        continue
                    prev_snapshot = snapshot
                    trial_n_str = ""
                    cells = []
                    for s in seed_list:
                        txt = snapshot[s]
                        if txt:
                            tn, bv = txt.split()
                            trial_n_str = f"{int(tn)}/{n_trials}"
                            disp = -float(bv) if maximise else float(bv)
                            cells.append(f"s{s}={disp:.3f}")
                        else:
                            cells.append(f"s{s}=--")
                    if verbose:
                        print(f"  [progress]  {trial_n_str}  " + "  ".join(cells),
                              flush=True)
            finally:
                worker_thread.join()
                shutil.rmtree(progress_dir, ignore_errors=True)
                # loky keeps its pool warm for reuse, leaving a Jupyter cell showing
                # "running" after the result is in hand. Tear it down so the cell
                # settles cleanly (kill_workers=True can't stall on a wedged worker).
                try:
                    from joblib.externals.loky import get_reusable_executor
                    get_reusable_executor().shutdown(wait=True, kill_workers=True)
                except Exception:
                    pass
            if "err" in holder:
                raise RuntimeError(
                    "Parallel multi-start seed dispatch failed — see the chained "
                    "exception below for the underlying worker error."
                ) from holder["err"]
            return holder["res"]


except ImportError:

    class OptunaCmaEsSolver:  # type: ignore[no-redef]
        """Placeholder — requires ``optuna``.

        Install with::

            pip install sparsehydro[optuna]
        """

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "optuna is required for OptunaCmaEsSolver. "
                "Install with: pip install sparsehydro[optuna]"
            )

        def solve(self, problem, **kwargs) -> None:  # type: ignore[misc]
            raise ImportError(
                "optuna is required for OptunaCmaEsSolver. "
                "Install with: pip install sparsehydro[optuna]"
            )
