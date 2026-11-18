"""Calibration result containers — solver-agnostic.

:class:`CalibrationResult` stores the Pareto front and per-generation history
produced by any :class:`~sparsehydro.calibration.solvers.ISolver`.  Objective
values are kept in **minimisation form** internally: maximised objectives are
negated on the way in and un-negated on the way out via
:meth:`CalibrationResult.objective_display_values` and
:meth:`CalibrationResult.to_pareto_dataframe`.
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np
import pandas as pd


GenerationRecord = namedtuple(
    "GenerationRecord", ["generation", "X", "F", "n_pareto"]
)
"""Per-generation snapshot of the optimizer population.

:param generation: Generation index (1-based).
:type generation: int
:param X: Parameter matrix, shape ``(pop_size, n_params)``.
:type X: numpy.ndarray
:param F: Objective matrix in minimisation form, shape ``(pop_size, n_obj)``.
:type F: numpy.ndarray
:param n_pareto: Number of Pareto-dominant solutions in this generation.
:type n_pareto: int
"""


def identify_pareto(F: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated solutions (all objectives minimised).

    O(n²) dominance check — used when pymoo is unavailable or as a fallback.

    :param F: 2-D array of objective values (minimisation form),
        shape ``(n, n_obj)``.
    :returns: Boolean array of length ``n``.
    :rtype: numpy.ndarray
    """
    n = len(F)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                is_pareto[i] = False
                break
    return is_pareto


class CalibrationResult:
    """General-purpose container for calibration results.

    Produced by any :class:`~sparsehydro.calibration.solvers.ISolver`.
    Objective values are stored in **minimisation form** (maximised objectives
    are negated). Conversion back to display form is handled by
    :meth:`objective_display_values` and :meth:`to_pareto_dataframe`.

    :param history: One :class:`GenerationRecord` per generation.  May be
        empty for single-shot solvers.
    :type history: list[GenerationRecord]
    :param pareto_X: Parameter matrix for the final Pareto front,
        shape ``(m, n_params)``.
    :type pareto_X: numpy.ndarray
    :param pareto_F: Objective matrix in minimisation form,
        shape ``(m, n_obj)``.
    :type pareto_F: numpy.ndarray
    :param param_names: Ordered list of calibrated parameter names.
    :type param_names: list[str]
    :param objective_names: Names of each objective (matching objective order).
    :type objective_names: list[str]
    :param minimize_flags: ``True`` for minimised objectives; ``False`` for
        maximised objectives.
    :type minimize_flags: list[bool]
    """

    def __init__(
        self,
        history: list[GenerationRecord],
        pareto_X: np.ndarray,
        pareto_F: np.ndarray,
        param_names: list[str],
        objective_names: list[str],
        minimize_flags: list[bool],
    ) -> None:
        self.history = list(history)
        self.pareto_X = np.asarray(pareto_X, dtype=float)
        self.pareto_F = np.asarray(pareto_F, dtype=float)
        self.param_names = list(param_names)
        self.objective_names = list(objective_names)
        self.minimize_flags = list(minimize_flags)

    def objective_display_values(self) -> np.ndarray:
        """Return Pareto objective values in display form (un-negated).

        :returns: Array of shape ``(m, n_obj)`` with actual objective values.
        :rtype: numpy.ndarray
        """
        F = self.pareto_F.copy()
        for i, minimize in enumerate(self.minimize_flags):
            if not minimize:
                F[:, i] = -F[:, i]
        return F

    def best_by(self, objective_name: str) -> np.ndarray:
        """Return the parameter vector that is best for the named objective.

        "Best" means the lowest value for minimised objectives and the highest
        for maximised objectives.  Internally, :attr:`pareto_F` is always in
        minimisation form, so ``argmin`` always selects the best solution.

        :param objective_name: Name matching one of :attr:`objective_names`.
        :type objective_name: str
        :returns: 1-D parameter array.
        :rtype: numpy.ndarray
        :raises ValueError: If ``objective_name`` is not found.
        """
        if objective_name not in self.objective_names:
            raise ValueError(
                f"Objective {objective_name!r} not found. "
                f"Available: {self.objective_names}"
            )
        idx_obj = self.objective_names.index(objective_name)
        idx_sol = int(np.argmin(self.pareto_F[:, idx_obj]))
        return self.pareto_X[idx_sol]

    def knee_index(self) -> int:
        """Index of the knee (elbow) solution on the Pareto front.

        Each objective is min-max normalised over the front (minimisation
        form), and the knee is the solution closest — in Euclidean distance —
        to the ideal point (the origin of the normalised space).  This is the
        classic "compromise" solution: the point past which improving one
        objective costs disproportionately in the others.

        :returns: Row index into :attr:`pareto_X` / :attr:`pareto_F`.
        :rtype: int
        :raises ValueError: If the Pareto front is empty.
        """
        if len(self.pareto_F) == 0:
            raise ValueError("Pareto front is empty; no knee point exists.")
        F = self.pareto_F
        f_min = F.min(axis=0)
        span = F.max(axis=0) - f_min
        span[span == 0.0] = 1.0
        normalised = (F - f_min) / span
        return int(np.argmin(np.linalg.norm(normalised, axis=1)))

    def knee_point(self) -> tuple[np.ndarray, np.ndarray]:
        """Parameter vector and display-form objectives of the knee solution.

        :returns: Tuple ``(x, f)`` where ``x`` is the knee parameter vector and
            ``f`` the corresponding objective values in display form
            (maximised objectives un-negated).
        :rtype: tuple[numpy.ndarray, numpy.ndarray]
        """
        idx = self.knee_index()
        return self.pareto_X[idx].copy(), self.objective_display_values()[idx].copy()

    def pareto_objective_bounds(self) -> pd.DataFrame:
        """Bounds of the Pareto-optimal objective values, in display form.

        :returns: DataFrame indexed by objective name with ``min`` / ``max``
            columns spanning the final Pareto front.
        :rtype: pandas.DataFrame
        :raises ValueError: If the Pareto front is empty.
        """
        if len(self.pareto_F) == 0:
            raise ValueError("Pareto front is empty; no bounds exist.")
        F = self.objective_display_values()
        return pd.DataFrame(
            {"min": F.min(axis=0), "max": F.max(axis=0)},
            index=pd.Index(self.objective_names, name="objective"),
        )

    def to_pareto_dataframe(self) -> pd.DataFrame:
        """Return a tidy DataFrame spanning all generations.

        Columns: ``generation``, one column per parameter, one column per
        objective (display-form values), ``is_pareto`` (``True`` only for
        solutions on the final generation's Pareto front).

        :rtype: pandas.DataFrame
        """
        rows = []
        final_gen = self.history[-1].generation if self.history else -1
        for rec in self.history:
            is_pareto_mask = identify_pareto(rec.F)
            for j in range(len(rec.X)):
                row: dict = {"generation": rec.generation}
                for k, name in enumerate(self.param_names):
                    row[name] = float(rec.X[j, k])
                for k, (obj_name, minimize) in enumerate(
                    zip(self.objective_names, self.minimize_flags)
                ):
                    raw = float(rec.F[j, k])
                    row[obj_name] = raw if minimize else -raw
                row["is_pareto"] = bool(
                    is_pareto_mask[j] and rec.generation == final_gen
                )
                rows.append(row)
        return pd.DataFrame(rows)
