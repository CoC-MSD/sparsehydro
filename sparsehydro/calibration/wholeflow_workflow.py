"""Reproducible local-data workflow for the compact joint-calibration notebook."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from sparsehydro.calibration.result import CalibrationResult
from sparsehydro.calibration.wholeflow_acceptance import AcceptanceEvaluator
from sparsehydro.metrics.functions import concordance_correlation, nash_sutcliffe, pbias
from sparsehydro.events.msdgc import find_events
from sparsehydro.models.rdii.initial_abstraction import SnowConfig, snow_partition
from sparsehydro.models.wholeflow_compact import CompactWholeFlowModel
from sparsehydro.calibration.splits.masks import (  # noqa: F401 - used by study code
    dry_weather_mask, melt_scoring_mask, storm_scoring_mask,
)
from sparsehydro.calibration.selection import feasible_history, select_peak_filtered_pareto  # noqa: F401
from sparsehydro.calibration.diagnostics import parameter_table  # noqa: F401



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
    return dict(ccc=concordance_correlation(x, y, undefined="nan"),
                nse=nash_sutcliffe(x, y, undefined="nan"),
                bias_pct=pbias(x, y, undefined="nan"), n=len(x))


