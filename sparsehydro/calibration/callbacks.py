"""Built-in callbacks for solver progress reporting.

A callback is any callable with the signature::

    def my_callback(info: dict) -> None: ...

The ``info`` dict contains:

``generation`` (int)
    Current generation / iteration (1-based).
``n_evals`` (int)
    Total function evaluations so far.
``n_pareto`` (int)
    Number of non-dominated solutions.
``pareto_F`` (ndarray)
    Pareto objectives in **minimisation form**, shape ``(n_pareto, n_obj)``.
``objective_names`` (list)
    Objective names (same order as columns).
``minimize_flags`` (list)
    ``True`` for minimised objectives.

Pass any callable to a solver via its ``callback`` parameter::

    from sparsehydro.calibration import NSGAIISolver, ProgressCallback

    cb = ProgressCallback(frequency=10)
    solver = NSGAIISolver(pop_size=100, n_gen=200, callback=cb)
    result = solver.solve(problem)
"""

from __future__ import annotations

import numpy as np


class ProgressCallback:
    """Prints a one-line progress summary after every *frequency* generations.

    Output format::

        gen=   10  evals=  1000  pareto=  12  mse=1234.5678  nash_sutcliffe=0.7231

    :param frequency: Print every *frequency* callback invocations (default 1).
    :type frequency: int
    """

    def __init__(self, frequency: int = 1) -> None:
        if frequency < 1:
            raise ValueError(f"frequency must be >= 1; got {frequency!r}")
        self.frequency = frequency
        self._call_count = 0

    def __call__(self, info: dict) -> None:
        """Print a progress line if the invocation count hits *frequency*.

        :param info: Progress dictionary supplied by the solver (see the module
            docstring for the expected keys).
        :type info: dict
        :returns: ``None``.
        :rtype: None
        """
        self._call_count += 1
        if self._call_count % self.frequency != 0:
            return

        gen      = info["generation"]
        n_evals  = info["n_evals"]
        n_pareto = info["n_pareto"]
        pareto_F = np.asarray(info["pareto_F"])
        obj_names     = info["objective_names"]
        minimize_flags = info["minimize_flags"]

        best_parts: list[str] = []
        if pareto_F.ndim == 2 and len(pareto_F) > 0:
            for j, (name, minimize) in enumerate(zip(obj_names, minimize_flags)):
                # pareto_F is in minimisation form; argmin is always "best"
                raw = float(pareto_F[:, j].min())
                display = raw if minimize else -raw
                best_parts.append(f"{name}={display:.4f}")
        else:
            best_parts = [f"{name}=N/A" for name in obj_names]

        obj_str = "  ".join(best_parts)
        print(f"gen={gen:5d}  evals={n_evals:7d}  pareto={n_pareto:4d}  {obj_str}")
