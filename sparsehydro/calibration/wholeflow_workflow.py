"""Reproducible local-data workflow for the compact joint-calibration notebook."""

from __future__ import annotations

import copy
import warnings

import numpy as np
import pandas as pd

from sparsehydro.calibration.objectives import Con15
from sparsehydro.calibration.problem import CalibrationProblem
from sparsehydro.calibration.result import CalibrationResult, GenerationRecord
from sparsehydro.calibration.wholeflow_acceptance import AcceptanceEvaluator, lin_ccc
from sparsehydro.events.msdgc import find_events
from sparsehydro.models.wholeflow import WholeFlowModel
from sparsehydro.models.rdii.initial_abstraction import SnowConfig, snow_partition
from sparsehydro.models.wholeflow_compact import CompactWholeFlowModel

MGD_TO_CFS = 1e6 / (7.480519 * 86400)
FORCING_COLUMNS = ["datetime", "rainfall_in", "temperature_c"]




def dry_weather_mask(data: pd.DataFrame, dt_hours: float, exclusion_hours: float) -> np.ndarray:
    """Exclude every positive rain pulse and an antecedent recession interval."""
    steps = int(np.ceil(exclusion_hours / dt_hours)) + 1
    wet = data["rainfall_in"].gt(0)
    return wet.rolling(steps, min_periods=steps).sum().eq(0).to_numpy()






def frozen_events(
    data: pd.DataFrame, train: np.ndarray, val: np.ndarray, reference: np.ndarray,
    *, detector_settings: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Apply the existing detector independently within contiguous split blocks.

    Never let validation stormflow extend/change a training event. Boundary events
    and the detector's omitted final rain span in each block are recorded, not
    assigned by majority. All-flow Qcritical is independent of these exclusions.
    """
    settings = dict(threshold_pct=5.0, interevent_hours=6.0, sliding_hours=12.0,
                    min_rain_in=0.2, min_duration_hours=3.0)
    if detector_settings is not None:
        unknown = set(detector_settings) - set(settings)
        if unknown:
            raise ValueError(f"Unknown storm detection settings: {sorted(unknown)}")
        settings.update(detector_settings)
    if (not np.isfinite(list(settings.values())).all()
            or not 0 <= settings["threshold_pct"] <= 100
            or settings["interevent_hours"] <= 0 or settings["sliding_hours"] <= 0
            or settings["min_rain_in"] < 0 or settings["min_duration_hours"] < 0):
        raise ValueError("Storm detection settings must be finite; threshold must be in [0, 100], "
                         "gap/window positive, and rainfall/duration nonnegative")
    labels = np.where(train, "train", np.where(val, "validation", "excluded"))
    edges = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(data)]
    wet = np.flatnonzero(data["rainfall_in"].to_numpy() > 0)
    times = pd.DatetimeIndex(data["datetime"]).as_unit("ns").asi8
    breaks = np.flatnonzero(
        np.diff(times[wet]) > pd.Timedelta(hours=settings["interevent_hours"]).value) + 1
    rain_spans = [(int(g[0]), int(g[-1])) for g in np.split(wet, breaks) if len(g)]
    rows = []
    for start, stop in zip(edges[:-1], edges[1:]):
        if labels[start] == "excluded":
            continue
        part = data.iloc[start:stop].copy().reset_index(drop=True)
        part["stormflow"] = np.maximum(part["flow_cfs"].to_numpy() - reference[start:stop], 0)
        official = find_events(part, rain_col="rainfall_in", close_trailing_event=False, **settings)
        inventory = find_events(part, rain_col="rainfall_in", close_trailing_event=True, **settings)
        official_pairs = {(e.start_idx, e.end_idx) for e in official}

        def boundary_reason(event) -> str:
            a, b = start + event.start_idx, start + event.end_idx
            if event.start_idx == 0 or event.end_idx >= len(part) - 1:
                return "touches_split_boundary"
            if any(s <= b and e >= a and (s < start or e >= stop) for s, e in rain_spans):
                return "rain_span_crosses_split_boundary"
            return ""

        for event in inventory:
            reason = ""
            if (event.start_idx, event.end_idx) not in official_pairs:
                reason = "trailing_span_omitted_or_merged_by_detector"
            reason = boundary_reason(event) or reason
            rows.append({
                "event_id": f"{start + event.start_idx}:{start + event.end_idx}", "start_idx": start + event.start_idx,
                "end_idx": start + event.end_idx, "start": event.start_datetime,
                "end": event.end_datetime, "split": labels[start],
                "duration_hours": event.duration_hours,
                "rain_in": event.total_rain, "eligible": not reason, "exclusion": reason,
            })
        # Official events can be altered by closing the trailing span in inventory.
        seen = {(row["start_idx"], row["end_idx"]) for row in rows}
        for event in official:
            pair = (start + event.start_idx, start + event.end_idx)
            if pair not in seen:
                reason = boundary_reason(event)
                rows.append({
                    "event_id": f"{pair[0]}:{pair[1]}", "start_idx": pair[0], "end_idx": pair[1],
                    "start": event.start_datetime, "end": event.end_datetime,
                    "duration_hours": event.duration_hours,
                    "split": labels[start], "rain_in": event.total_rain,
                    "eligible": not reason, "exclusion": reason,
                })
    columns = ["event_id", "start_idx", "end_idx", "start", "end", "split",
               "duration_hours", "rain_in", "eligible", "exclusion"]
    return pd.DataFrame(rows, columns=columns).sort_values(["start_idx", "end_idx"]).reset_index(drop=True)








def parameter_table(model: WholeFlowModel) -> pd.DataFrame:
    rows = []
    for name in model.scalar_parameter_names:
        p = model.get_scalar_parameter(name)
        span = p.upper_bound - p.lower_bound
        rows.append({
            "name": name, "value": p.value, "lower": p.lower_bound, "upper": p.upper_bound,
            "units": p.units, "calibrated": p.calibrate, "description": p.description,
            "near_bound": bool(p.calibrate and span > 0 and
                               min(p.value - p.lower_bound, p.upper_bound - p.value) / span < .01),
        })
    return pd.DataFrame(rows)




def storm_scoring_mask(
    events: pd.DataFrame, split_mask: np.ndarray, dt_hours: float, *, split: str = "train",
    pre_hours: float = 3.0, tail_hours: float = 48.0,
) -> np.ndarray:
    """The original notebook's storm CCC window, built from the complete events.

    Each complete ``split`` event is padded by a pre-event buffer and a recession
    tail, then restricted to ``split_mask`` so padding never spills into another
    split. These are objective windows only; acceptance still uses complete events.
    """
    split_mask = np.asarray(split_mask, dtype=bool)
    if dt_hours <= 0 or pre_hours < 0 or tail_hours < 0:
        raise ValueError("Timestep must be positive and padding non-negative")
    n = len(split_mask)
    pre = int(round(pre_hours / dt_hours))
    tail = int(round(tail_hours / dt_hours))
    mask = np.zeros(n, dtype=bool)
    chosen = events[events["eligible"].astype(bool) & events["split"].eq(split)]
    for event in chosen.itertuples():
        mask[max(0, event.start_idx - pre):min(n, event.end_idx + tail + 1)] = True
    return mask & split_mask


def snow_melt_envelope(
    precipitation_in: np.ndarray, temperature_c: np.ndarray, dt_hours: float, *,
    threshold_bounds: tuple[float, float], melt_factor_bounds: tuple[float, float],
    initial_swe_in: float = 0.0,
) -> np.ndarray:
    """Conservative possible-melt timesteps for every bounded snow candidate.

    The highest threshold and lowest melt factor give an upper SWE trajectory.
    Any candidate can melt only where that pack existed before the step and the
    temperature exceeds the lowest threshold. This is a wetness/scoring mask,
    NOT a water input. A zero lower melt bound can retain potential SWE forever.
    """
    bounds = np.asarray([threshold_bounds, melt_factor_bounds], dtype=float)
    if (bounds.shape != (2, 2) or not np.isfinite(bounds).all()
            or np.any(bounds[:, 0] > bounds[:, 1]) or bounds[1, 0] < 0):
        raise ValueError("Snow envelope bounds must be finite, ordered, with nonnegative melt")
    upper = snow_partition(
        precipitation_in, temperature_c, dt_hours,
        SnowConfig(threshold_c=float(bounds[0, 1]),
                   melt_factor_in_per_c_day=float(bounds[1, 0]),
                   initial_swe_in=initial_swe_in),
    )
    previous_swe = np.r_[initial_swe_in, upper["SWE_in"].to_numpy()[:-1]]
    return ((previous_swe > 0) & (np.asarray(temperature_c) > bounds[0, 0])
            & (bounds[1, 1] > 0))


def melt_scoring_mask(
    melt_in: np.ndarray, split_mask: np.ndarray, dt_hours: float, *,
    pre_hours: float = 3.0, tail_hours: float = 48.0,
) -> np.ndarray:
    """Pad every positive melt pulse, including rain-free thaw, then clip to split.

    Forcing-only windows supplement, but never redefine, MSD precipitation events.
    Antecedent melt may cross a split edge just as physical storage does; no
    observed flow is consulted in constructing these windows.
    """
    melt = np.asarray(melt_in, dtype=float)
    split = np.asarray(split_mask, dtype=bool)
    if (melt.ndim != 1 or split.shape != melt.shape or not np.isfinite(melt).all()
            or (melt < 0).any() or not np.isfinite([dt_hours, pre_hours, tail_hours]).all()
            or dt_hours <= 0 or pre_hours < 0 or tail_hours < 0):
        raise ValueError("Melt/mask must align; melt and padding nonnegative; timestep positive")
    pre, tail = int(round(pre_hours / dt_hours)), int(round(tail_hours / dt_hours))
    pulses = np.flatnonzero(melt > 0)
    changes = np.zeros(len(melt) + 1, dtype=np.int64)
    np.add.at(changes, np.maximum(0, pulses - pre), 1)
    np.add.at(changes, np.minimum(len(melt), pulses + tail + 1), -1)
    return (np.cumsum(changes[:-1]) > 0) & split


class FeasibleCalibrationProblem(CalibrationProblem):
    """Penalise constraint-violating vectors without calling ``predict()``.

    ``CompactWholeFlowModel.predict()`` refuses infeasible vectors (``sum R > 1``
    would create water), so they receive a dominated penalty instead of an error.
    """

    def _feasibility_penalty(self, x: np.ndarray, penalty_weight: float) -> float | None:
        for name, value in zip(self._param_names, x):
            self._model.apply_flat_parameter(name, float(value))
        violation = np.maximum(np.asarray(self._model.inequality_constraints(), float), 0.0)
        if violation.any():
            return penalty_weight * (1.0 + float(np.sum(violation ** 2)))
        return None

    def evaluate(self, x: np.ndarray, penalty_weight: float = 1e6) -> np.ndarray:
        penalty = self._feasibility_penalty(x, penalty_weight)
        if penalty is not None:
            return np.full(len(self._objectives), penalty)
        return super().evaluate(x, penalty_weight)

    def evaluate_single(self, x: np.ndarray, index: int, penalty_weight: float = 1e6) -> float:
        if not 0 <= index < len(self._objectives):
            raise IndexError(f"objective index {index} out of range")
        penalty = self._feasibility_penalty(x, penalty_weight)
        return penalty if penalty is not None else super().evaluate_single(x, index, penalty_weight)

    def make_copy(self) -> "FeasibleCalibrationProblem":
        copy = super().make_copy()
        copy.__class__ = type(self)
        return copy


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
    model: CompactWholeFlowModel,
    observed: np.ndarray,
    *,
    peak_filter: Con15 | None = None,
    max_error: float = 0.10,
    near_tie_allowance: float | None = None,
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
            slope = peak_filter.slope(observed, worker.predict_total_cfs())
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


def passing_pareto_metrics(
    result: CalibrationResult,
    model: CompactWholeFlowModel,
    pareto_solutions: pd.DataFrame,
    evaluators: dict[str, AcceptanceEvaluator],
) -> pd.DataFrame:
    """Report all filter-passing final candidates without selecting or refitting.

    Call only after selection when the training filter is enabled. Re-simulate
    each passing vector on a private model, then report each evaluator's declared
    metric set for each split, including all three zones for MSDGCAssessment.
    Validation/whole scores are diagnostic only. The filter's independently
    configured slope/error is included, not substituted for standard Con15.
    Zero passes returns a typed-column empty table without any simulations.
    """
    if (len(pareto_solutions) != len(result.pareto_X)
            or not np.array_equal(pareto_solutions["pareto_index"].to_numpy(),
                                  np.arange(len(result.pareto_X)))):
        raise ValueError("Pareto report rows must align with the complete final front")
    if not pareto_solutions["filter_enabled"].all():
        raise ValueError("Passing Pareto metrics require the training peak filter")
    metrics = next(iter(evaluators.values())).report_metric_names if evaluators else ()
    if any(evaluator.report_metric_names != metrics for evaluator in evaluators.values()):
        raise ValueError("Pareto report evaluators must use the same metric definitions")
    columns = ["pareto_index", "selected", "filter_peak_slope", "filter_peak_error"]
    for split in evaluators:
        for metric in metrics:
            columns.extend([f"{split}_{metric}", f"{split}_{metric}_status",
                            f"{split}_{metric}_n", f"{split}_{metric}_reason"])
    passing = pareto_solutions.loc[pareto_solutions["filter_pass"].fillna(False)]
    if passing.empty:
        return pd.DataFrame(columns=columns)
    if any(evaluator.contract.qcritical_cfs is not None for evaluator in evaluators.values()):
        raise ValueError("Passing Pareto reporting requires Qcritical-free report contracts")
    worker = copy.deepcopy(model)
    rows = []
    for candidate in passing.itertuples():
        index = int(candidate.pareto_index)
        for name, value in zip(result.param_names, result.pareto_X[index]):
            worker.apply_flat_parameter(name, float(value))
        prediction = worker.predict_total_cfs()
        row = dict(pareto_index=index, selected=bool(candidate.selected),
                   filter_peak_slope=float(candidate.filter_peak_slope),
                   filter_peak_error=float(candidate.filter_peak_error))
        for split, evaluator in evaluators.items():
            scores = evaluator.metrics(prediction).set_index("metric")
            for metric in metrics:
                score = scores.loc[metric]
                row.update({
                    f"{split}_{metric}": score["value"],
                    f"{split}_{metric}_status": score["status"],
                    f"{split}_{metric}_n": score["n"],
                    f"{split}_{metric}_reason": score["reason"],
                })
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)




def hydrograph_diagnostics(observed: np.ndarray, modeled: np.ndarray) -> dict:
    x, y = np.asarray(observed), np.asarray(modeled)
    if not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
        return {"ccc": np.nan, "nse": np.nan, "bias_pct": np.nan, "n": len(x)}
    denominator = float(np.sum((x - x.mean()) ** 2))
    return dict(ccc=lin_ccc(x, y), nse=1 - float(np.sum((y - x) ** 2)) / denominator if denominator > 0 else np.nan,
                bias_pct=100 * float((y - x).sum() / x.sum()) if x.sum() > 0 else np.nan, n=len(x))



