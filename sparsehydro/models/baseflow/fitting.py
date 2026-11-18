"""Pre-fit estimators: everything a baseline learns from observed flow.

These run once, on training dry-weather data, before calibration. Their
results (:class:`~._shapes.CenteredWeeklyPattern`,
:class:`~._shapes.ObservedBackground`, slot levels) are passed to the
baseline components' ``prepare``; no component reads observed flow while
predicting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._shapes import CenteredWeeklyPattern, ObservedBackground, validate_background_options

__all__ = ["PatternEstimate", "estimate_observed_background", "estimate_sanitary_shapes",
           "profile_slot_levels", "slot_of_week"]


@dataclass
class PatternEstimate:
    pattern: CenteredWeeklyPattern
    detection_baseline_cfs: np.ndarray
    coverage: pd.DataFrame


def estimate_observed_background(
    data: pd.DataFrame, eligible: np.ndarray, pattern: CenteredWeeklyPattern, *,
    window_days: int = 21, minimum_days: int = 3, max_gap_days: int = 45,
    endpoint_hold_days: int = 0,
) -> ObservedBackground:
    """Smooth daily training-dry means after removing the frozen weekly anomalies.

    Anchors exist only on observed eligible days, not in empty rolling windows.
    Smoothing never bridges a source gap exceeding max_gap_days. The window and
    gap limit are explicit estimation assumptions, not a prescribed seasonal shape.
    """
    validate_background_options(window_days, minimum_days, max_gap_days, endpoint_hold_days)
    t = pd.DatetimeIndex(data["datetime"])
    mask = np.asarray(eligible)
    q = data["flow_cfs"].to_numpy(dtype=float)
    if (mask.dtype != bool or mask.shape != q.shape or not mask.any()
            or t.hasnans or t.has_duplicates or not t.is_monotonic_increasing):
        raise ValueError("Background fitting requires aligned boolean eligibility and ordered timestamps")
    if not np.isfinite(q[mask]).all() or np.any(q[mask] < 0):
        raise ValueError("Background fitting requires finite nonnegative training dry-weather flow")
    h, d = pattern.components(t)
    daily = pd.Series(np.where(mask, q - h - d, np.nan), index=t).resample("D").mean().dropna()
    segments = daily.index.to_series().diff().gt(pd.Timedelta(days=max_gap_days)).cumsum()
    slow = pd.concat([
        group.rolling(f"{window_days}D", center=True, min_periods=minimum_days).mean()
        for _, group in daily.groupby(segments)
    ]).dropna()
    if len(slow) < 2:
        raise ValueError("Insufficient supported training dry-weather days for observed background")
    source = t[mask]
    start = max(source[0], slow.index[0])
    end = min(source[-1], slow.index[-1] + pd.DateOffset(days=1) - pd.Timedelta(nanoseconds=1))
    return ObservedBackground(
        tuple(slow.index), tuple(slow.to_numpy()), start, end,
        window_days, minimum_days, max_gap_days, endpoint_hold_days,
    )


def estimate_sanitary_shapes(data: pd.DataFrame, eligible: np.ndarray) -> PatternEstimate:
    """Estimate shapes, not sanitary level, from training dry-weather rows only.

    A centered 21-day trend of eligible daily means removes slow variation before
    a weighted two-way hour/weekday fit. Its intercept is discarded. Neither the
    trend nor filtered observations are used in model prediction. This assumes
    slow GWI relative to the daily pattern; it does not identify sanitary flow.
    """
    t = pd.DatetimeIndex(data["datetime"])
    mask = np.asarray(eligible, dtype=bool)
    q = data["flow_cfs"].to_numpy(dtype=float)
    if (mask.shape != q.shape or len(t) < 2 or t.hasnans or t.has_duplicates
            or not t.is_monotonic_increasing):
        raise ValueError("Shape fitting requires aligned masks and ordered unique timestamps")
    if not mask.any() or not np.isfinite(q[mask]).all() or np.any(q[mask] < 0):
        raise ValueError("Shape fitting requires finite nonnegative training dry-weather flow")
    daily = pd.Series(np.where(mask, q, np.nan), index=t).resample("D").mean()
    slow = daily.rolling(21, center=True, min_periods=3).mean()
    trend = slow.reindex(t.normalize()).to_numpy()
    fit = mask & np.isfinite(trend) & (trend > 0)
    grouped = pd.DataFrame({
        "hour": t.hour[fit], "weekday": t.dayofweek[fit],
        "residual": q[fit] - trend[fit],
    }).groupby(["hour", "weekday"])["residual"].agg(["mean", "count"]).reset_index()
    if grouped["hour"].nunique() != 24 or grouped["weekday"].nunique() != 7:
        raise ValueError("Insufficient training dry-weather coverage for all hours/weekdays")
    design = np.column_stack((
        np.ones(len(grouped)),
        np.eye(24)[grouped["hour"].to_numpy(), 1:],
        np.eye(7)[grouped["weekday"].to_numpy(), 1:],
    ))
    weight = np.sqrt(grouped["count"].to_numpy())
    coefficients, _, rank, _ = np.linalg.lstsq(
        design * weight[:, None], grouped["mean"].to_numpy() * weight, rcond=None,
    )
    if rank != design.shape[1]:
        raise ValueError("Training hour/weekday coverage cannot identify both pattern shapes")
    hourly, weekday = np.r_[0.0, coefficients[1:24]], np.r_[0.0, coefficients[24:]]
    hourly -= hourly.mean()
    weekday -= weekday.mean()
    pattern = CenteredWeeklyPattern(tuple(hourly), tuple(weekday))
    anchors = slow.dropna()
    if len(anchors) < 2:
        raise ValueError("Insufficient training days for the event-detection reference")
    background = np.interp(t.as_unit("ns").asi8, anchors.index.as_unit("ns").asi8,
                           anchors.to_numpy())
    h, d = pattern.components(t)
    reference = background + h + d
    if not np.isfinite(reference).all() or np.any(reference <= 0):
        raise ValueError("Nonpositive event-detection reference; inspect training flow and shapes")
    coverage = pd.DataFrame({
        "month": t.month, "weekday": t.dayofweek, "hour": t.hour,
        "eligible": mask, "shape_fit": fit,
    }).groupby(["month", "weekday", "hour"]).sum().reset_index()
    return PatternEstimate(pattern, reference, coverage)


def slot_of_week(datetimes, dt_hours: float) -> tuple[np.ndarray, int]:
    """Slot-of-week index (weekday x step-of-day) per row, and the number of slots."""
    t = pd.DatetimeIndex(pd.to_datetime(np.asarray(datetimes)))
    step_s = max(1, int(round(dt_hours * 3600.0)))
    per_day = max(1, int(round(86400.0 / step_s)))
    tod_s = np.asarray(t.hour * 3600 + t.minute * 60 + t.second)
    slot = (tod_s // step_s) % per_day
    return (np.asarray(t.dayofweek) * per_day + slot).astype(np.int64), 7 * per_day


def profile_slot_levels(observed_cfs: np.ndarray, other_cfs: np.ndarray, fit_mask: np.ndarray,
                        datetimes, dt_hours: float, cap_cfs: np.ndarray | None = None
                        ) -> tuple[np.ndarray, dict[str, float]]:
    """Least-squares sanitary level per slot of the week.

    Each slot's level is the mean of ``observed - other`` over the rows in
    *fit_mask* (other = modelled GWI and RDII), clipped to ``[0, cap]``. Slots
    without fit rows take the cap (or 0 without one). This is the "profiled"
    sanitary fit of earlier versions, now a pre-fit step whose result can feed
    a baseline component, instead of a re-fit inside every prediction.

    :returns: ``(level_per_row_cfs, diagnostics)`` where diagnostics give the
        share of populated slots at 0 and at the cap, and the empty-slot count.
    """
    observed = np.asarray(observed_cfs, dtype=float)
    other = np.asarray(other_cfs, dtype=float)
    mask = np.asarray(fit_mask, dtype=bool)
    if not (observed.shape == other.shape == mask.shape):
        raise ValueError("observed, other and fit_mask must be aligned")
    slot, n_slots = slot_of_week(datetimes, dt_hours)
    cap = np.full(n_slots, np.inf) if cap_cfs is None else np.asarray(cap_cfs, dtype=float)
    if cap.shape != (n_slots,):
        raise ValueError(f"cap_cfs needs one value per slot ({n_slots})")
    fit = mask & np.isfinite(observed)
    counts = np.bincount(slot[fit], minlength=n_slots)
    totals = np.bincount(slot[fit], weights=(observed - other)[fit], minlength=n_slots)
    has = counts > 0
    fallback = np.where(np.isfinite(cap), cap, 0.0)
    level = np.where(has, totals / np.maximum(counts, 1), fallback)
    profile = np.clip(level, 0.0, cap)
    diagnostics = {
        "at_zero": float(np.mean(level[has] <= 0.0)) if has.any() else float("nan"),
        "at_cap": float(np.mean(level[has] >= cap[has])) if has.any() else float("nan"),
        "empty_slots": int((~has).sum()),
    }
    return profile[slot], diagnostics
