"""Choosing a final candidate from a calibration result.

* :func:`feasible_history` — a display copy of a result without penalised members.
* :func:`select_peak_filtered_pareto` — knee selection, optionally filtered or
  tie-broken by a training peak (Con15) error.
"""

from __future__ import annotations

import copy
import warnings

import numpy as np
import pandas as pd

from ..metrics.objectives import Con15
from ..models.base import IModel
from .result import CalibrationResult, GenerationRecord


def feasible_history(result: CalibrationResult, limit: float = 1e5) -> tuple[CalibrationResult, int]:
    """Display copy of ``result`` without penalised (infeasible) population members."""
    records, dropped = [], 0
    for record in result.history:
        keep = np.all(np.abs(record.F) < limit, axis=1)
        dropped += int((~keep).sum())
        if keep.any():
            records.append(GenerationRecord(record.generation, record.X[keep], record.F[keep], record.n_pareto))
    return CalibrationResult(records, result.pareto_X, result.pareto_F, result.param_names,
                             result.objective_names, result.minimize_flags), dropped


def select_peak_filtered_pareto(
    result: CalibrationResult,
    model: IModel,
    observed: np.ndarray,
    *,
    peak_filter: Con15 | None = None,
    max_error: float = 0.10,
    near_tie_allowance: float | None = None,
    predicted: str = "total_cfs",
) -> tuple[int, pd.DataFrame]:
    """Select a final-front candidate without modifying the front or source model.

    With no filter, preserve ``result.knee_index()`` exactly and do not simulate
    candidates. With a filter, rerun each physically feasible final candidate on
    a private prepared model and evaluate the supplied training-only Con15
    definition, independently of optimizer objectives/fractions. No solver or
    acceptance checker is called, and no validation observations are consulted.

    Choose the normalized active-objective compromise among candidates with
    ``abs(slope - 1) <= max_error`` (inclusive to 1e-12). Normalize on that subset,
    not on rejected candidates. If none pass, warn and select the lowest finite
    peak error, explicitly marked ``fallback_filter_failed``. Invalid models or
    non-finite objectives are never eligible for fallback. Unexpected simulation
    errors propagate; they are not silently discarded.

    With ``near_tie_allowance`` in [0, 1], replace that hard-filter policy with
    a bounded peak preference. Min-max normalize the active non-Con15 objectives
    over physically feasible final candidates, then score each by RMS distance
    to the ideal (0 = best, 1 = worst normalized corner). If Con15 is the only
    objective, use it as the primary score too. Shortlist candidates whose score
    is at most the best score plus the allowance (inclusive to 1e-12), then choose
    the smallest training peak error within that shortlist. Break equal peak
    errors by primary score, then original row order. The threshold is report
    only: a selected failure warns but NEVER expands the shortlist. Objective
    values, search behavior and the complete final front are unchanged.

    Candidates are simulated with ``model.predict_arrays()[predicted]``.

    The returned table keeps EVERY row of ``pareto_X``/``pareto_F`` in original
    order, including duplicates and rejected candidates, with stable indices,
    parameters, display-form objectives, feasibility/filter flags and selection
    status. Disabled filter scores are NaN with nullable pass flags, not passes.
    """
    count = len(result.pareto_X)
    if (not count or not result.objective_names
            or result.pareto_X.shape != (count, len(result.param_names))
            or result.pareto_F.shape != (count, len(result.objective_names))):
        raise ValueError("A nonempty, aligned final Pareto front is required")
    if peak_filter is not None and (not np.isfinite(max_error) or max_error < 0):
        raise ValueError("max_error must be finite and nonnegative")
    if near_tie_allowance is not None and (
            not np.isfinite(near_tie_allowance) or not 0 <= near_tie_allowance <= 1):
        raise ValueError("near_tie_allowance must be finite and in [0, 1]")
    if peak_filter is not None:
        # Validate the frozen target once, before evaluating any candidate.
        peak_filter.compute(observed, observed)
    table = pd.DataFrame(result.pareto_X, columns=result.param_names)
    table.insert(0, "pareto_index", np.arange(count))
    for i, name in enumerate(result.objective_names):
        table[name] = result.objective_display_values()[:, i]
    table["model_feasible"] = False
    table["filter_enabled"] = peak_filter is not None
    table["filter_basis"] = (getattr(peak_filter, "basis", "instantaneous_event_peak")
                             if peak_filter is not None else "disabled")
    table["filter_evaluated"] = False
    table["filter_peak_slope"] = np.nan
    table["filter_peak_error"] = np.nan
    table["filter_pass"] = pd.Series(pd.NA, index=table.index, dtype="boolean")
    table["candidate_status"] = "invalid_model_or_objectives"
    table["primary_score"] = np.nan
    table["primary_score_increase"] = np.nan
    table["primary_baseline"] = False
    table["near_tie_eligible"] = False
    table["selected"] = False
    table["selection_status"] = "not_selected"
    worker = copy.deepcopy(model)
    for i, x in enumerate(result.pareto_X):
        if not np.isfinite(x).all() or not np.isfinite(result.pareto_F[i]).all():
            continue
        for name, value in zip(result.param_names, x):
            worker.apply_flat_parameter(name, float(value))
        constraints = np.asarray(worker.inequality_constraints(), dtype=float)
        if not worker.parameters_valid() or not np.isfinite(constraints).all() or (constraints > 0).any():
            continue
        table.at[i, "model_feasible"] = True
        table.at[i, "candidate_status"] = "filter_disabled"
        if peak_filter is not None:
            slope = peak_filter.slope(observed, worker.predict_arrays()[predicted])
            error = abs(slope - 1.0)
            passed = error <= max_error + 1e-12
            table.loc[i, ["filter_evaluated", "filter_peak_slope", "filter_peak_error",
                          "filter_pass", "candidate_status"]] = [
                True, slope, error, passed, "filter_pass" if passed else "filter_fail",
            ]
    if peak_filter is None:
        selected = result.knee_index()
        if not table.at[selected, "model_feasible"]:
            raise RuntimeError("The unfiltered knee is infeasible; no valid selected model")
        status = "filter_disabled"
    elif near_tie_allowance is not None:
        evaluated = table.index[table["filter_evaluated"]].to_numpy()
        if not len(evaluated):
            raise RuntimeError("No feasible Pareto candidate has a defined peak-filter score")
        primary_columns = [i for i, name in enumerate(result.objective_names)
                           if name != "con15_error"]
        if not primary_columns:
            primary_columns = list(range(len(result.objective_names)))
        table.attrs["primary_objective_names"] = [
            result.objective_names[i] for i in primary_columns]
        primary_f = result.pareto_F[np.ix_(evaluated, primary_columns)]
        span = np.ptp(primary_f, axis=0)
        span[span == 0.0] = 1.0
        normalized = (primary_f - primary_f.min(axis=0)) / span
        scores = np.sqrt(np.mean(normalized ** 2, axis=1))
        increase = scores - scores.min()
        eligible = increase <= near_tie_allowance + 1e-12
        table.loc[evaluated, "primary_score"] = scores
        table.loc[evaluated, "primary_score_increase"] = increase
        table.loc[evaluated, "near_tie_eligible"] = eligible
        table.at[int(evaluated[np.argmin(scores)]), "primary_baseline"] = True
        shortlist = evaluated[eligible]
        errors = table.loc[shortlist, "filter_peak_error"].to_numpy(dtype=float)
        order = np.lexsort((shortlist, scores[eligible], errors))
        selected = int(shortlist[order[0]])
        passed = bool(table.at[selected, "filter_pass"])
        status = "near_tie_peak_pass" if passed else "near_tie_peak_failed"
        if not passed:
            warnings.warn(
                f"Selected near-tie candidate {selected} fails the training peak threshold "
                f"(error={table.at[selected, 'filter_peak_error']:.6g} > {max_error:g}). "
                f"Retaining it within the primary-score allowance {near_tie_allowance:g}; "
                "the threshold is report only and does not expand the shortlist.",
                RuntimeWarning, stacklevel=2,
            )
    else:
        passing = table.index[table["filter_pass"].fillna(False)].to_numpy()
        if len(passing):
            subset = CalibrationResult(
                [], result.pareto_X[passing], result.pareto_F[passing],
                result.param_names, result.objective_names, result.minimize_flags,
            )
            selected = int(passing[subset.knee_index()])
            status = "filter_pass_compromise"
        else:
            evaluated = table.index[table["filter_evaluated"]].to_numpy()
            if not len(evaluated):
                raise RuntimeError("No feasible Pareto candidate has a defined peak-filter score")
            selected = int(evaluated[np.argmin(
                table.loc[evaluated, "filter_peak_error"].to_numpy(dtype=float)
            )])
            status = "fallback_filter_failed"
            warnings.warn(
                f"No final Pareto candidate meets training peak error <= {max_error:g}. "
                f"Selecting lowest-error candidate {selected} "
                f"(error={table.at[selected, 'filter_peak_error']:.6g}) as a FAILING fallback.",
                RuntimeWarning, stacklevel=2,
            )
    table.at[selected, "selected"] = True
    table.at[selected, "selection_status"] = status
    return selected, table


__all__ = ["feasible_history", "select_peak_filtered_pareto"]
