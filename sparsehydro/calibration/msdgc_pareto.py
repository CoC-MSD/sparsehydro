"""Post-fit MSDGC comparisons; never changes the training-selected model."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from .msdgc_assessment import MSDGCAssessment
from .result import CalibrationResult
from ..models.wholeflow_compact import CompactWholeFlowModel


MSDGC_METRIC_ORDER = (
    "conveyance_ccc", "storage_ccc", "treatment_ccc",
    "three_zone_mean_ccc", "event_volume_ccc", "con15_slope",
)
MSDGC_REQUIRED_METRICS = (
    "conveyance_ccc", "storage_ccc", "treatment_ccc", "con15_slope",
)
MSDGC_REQUIRED_WINDOWS = ("validation", "whole")


def evaluate_msdgc_pareto(
    result: CalibrationResult,
    model: CompactWholeFlowModel,
    evaluators: dict[str, MSDGCAssessment],
) -> pd.DataFrame:
    """Simulate EVERY final-front vector, including when the training filter is off.

    The model must already have its full, continuous forcing prepared. Work on
    a private copy and retain original Pareto indices, bounds, counts, statuses
    and reasons. Invalid vectors remain explicit undefined rows; unexpected
    simulation errors propagate. No optimizer or training shortlist is used.
    """
    count = len(result.pareto_X)
    if (not count or not result.objective_names
            or result.pareto_X.shape != (count, len(result.param_names))
            or result.pareto_F.shape != (count, len(result.objective_names))):
        raise ValueError("A nonempty, aligned final Pareto front is required")
    if not set(MSDGC_REQUIRED_WINDOWS).issubset(evaluators):
        raise ValueError("MSDGC Pareto reporting requires validation and whole evaluators")
    if any(not isinstance(ev, MSDGCAssessment) for ev in evaluators.values()):
        raise ValueError("All evaluators must use MSDGC total-flow moving-volume rules")
    worker = copy.deepcopy(model)
    reports = []
    for index, vector in enumerate(result.pareto_X):
        reason = ""
        if not np.isfinite(vector).all() or not np.isfinite(result.pareto_F[index]).all():
            reason = "Non-finite candidate parameters or objectives"
        else:
            for name, value in zip(result.param_names, vector):
                worker.apply_flat_parameter(name, float(value))
            constraints = np.asarray(worker.inequality_constraints(), dtype=float)
            if (not worker.parameters_valid() or not np.isfinite(constraints).all()
                    or (constraints > 0).any()):
                reason = "Candidate violates model parameter bounds or physical constraints"
        prediction = None if reason else worker.predict_total_cfs()
        for window, evaluator in evaluators.items():
            if prediction is None:
                contract = evaluator.contract
                report = pd.DataFrame([
                    dict(metric=metric, value=np.nan,
                         lower=(1 - contract.con15_tolerance if metric == "con15_slope"
                                else contract.ccc_min),
                         upper=1 + contract.con15_tolerance if metric == "con15_slope" else 1.,
                         n=0, status="undefined", reason=reason)
                    for metric in MSDGC_METRIC_ORDER
                ])
            else:
                report = evaluator.metrics(prediction).set_index("metric").loc[
                    list(MSDGC_METRIC_ORDER)].reset_index()
            reports.append(report.assign(
                pareto_index=index, window=window, model_feasible=not reason))
    return pd.concat(reports, ignore_index=True)


def rank_msdgc_pareto(report: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    """Choose a diagnostic candidate by eight equally weighted threshold deficits.

    Require all three zone CCCs and Con15 in BOTH validation and whole. Sum
    max(lower-value, value-upper, 0) / 0.10 over these eight checks only.
    Passing checks (inclusive to 1e-12) contribute zero. Undefined/non-finite
    checks make the aggregate infinite, never a skip-NaN score. Training,
    three-zone mean and full-event-volume CCC do not influence ranking.
    Equal aggregates retain the lowest original Pareto index.

    Returns the diagnostic index and a wide comparison of every candidate.
    This is not a new calibrated selection or an official storm scorecard.
    """
    required_columns = {
        "pareto_index", "window", "metric", "value", "lower", "upper",
        "status", "reason", "model_feasible",
    }
    if report.empty or not required_columns.issubset(report):
        raise ValueError("A nonempty MSDGC Pareto metric report is required")
    keys = ["pareto_index", "window", "metric"]
    if report.duplicated(keys).any():
        raise ValueError("Each candidate/window/metric row must be unique")
    rows = []
    for index, candidate in report.groupby("pareto_index", sort=True):
        scores = candidate.set_index(["window", "metric"])
        required = pd.MultiIndex.from_product(
            [MSDGC_REQUIRED_WINDOWS, MSDGC_REQUIRED_METRICS],
            names=["window", "metric"],
        )
        if not required.isin(scores.index).all():
            raise ValueError(f"Candidate {index} is missing required MSDGC checks")
        checks = scores.loc[required]
        values = checks.value.to_numpy(dtype=float)
        lower, upper = checks.lower.to_numpy(float), checks.upper.to_numpy(float)
        if not np.isfinite([lower, upper]).all() or (lower > upper).any():
            raise ValueError("MSDGC checks require finite ordered acceptance bounds")
        defined = np.isfinite(values) & checks.status.isin(["pass", "fail"]).to_numpy()
        violation = np.maximum(np.maximum(lower - values, values - upper), 0.)
        violation[violation <= 1e-12] = 0.
        errors = np.where(defined, violation / 0.10, np.inf)
        feasible = bool(candidate.model_feasible.all())
        row = dict(
            pareto_index=int(index), model_feasible=feasible,
            checks_passed=int(np.count_nonzero(defined & (errors == 0))),
            checks_failed=int(np.count_nonzero(defined & (errors > 0))),
            checks_undefined=int(np.count_nonzero(~defined)),
            passes_all_msd=bool(feasible and defined.all() and (errors == 0).all()),
            aggregate_error=float(errors.sum()) if feasible else np.inf,
        )
        for (window, metric), score in scores.iterrows():
            row[f"{window}_{metric}"] = score["value"]
            row[f"{window}_{metric}_status"] = score["status"]
        for (window, metric), error in zip(required, errors):
            row[f"{window}_{metric}_normalized_violation"] = float(error)
        rows.append(row)
    summary = pd.DataFrame(rows)
    finite = summary.loc[np.isfinite(summary.aggregate_error)]
    if finite.empty:
        raise RuntimeError(
            "No physically feasible Pareto candidate has all eight defined MSDGC checks; "
            "inspect undefined statuses/reasons in the metric report")
    selected = int(finite.sort_values(
        ["aggregate_error", "pareto_index"], kind="stable").iloc[0].pareto_index)
    summary["diagnostic_selected"] = summary.pareto_index.eq(selected)
    summary["diagnostic_selection_status"] = np.where(
        summary.diagnostic_selected,
        np.where(summary.passes_all_msd, "passes_all_msd", "closest_threshold_shortfall"),
        "not_selected",
    )
    return selected, summary
