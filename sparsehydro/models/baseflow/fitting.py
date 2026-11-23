"""Pre-fit estimators: everything a baseline learns from observed flow.

These run once, on training dry-weather data, before calibration. Their
results (:class:`~._shapes.CenteredWeeklyPattern`,
:class:`~._shapes.SanitaryTemplate`, :class:`~._shapes.ObservedBackground`,
slot levels) are passed to the baseline components' ``prepare``; no
component reads observed flow while predicting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from ._shapes import (
    CenteredWeeklyPattern,
    ObservedBackground,
    SanitaryTemplate,
    validate_background_options,
)

__all__ = ["PatternEstimate", "estimate_observed_background", "estimate_sanitary_level",
           "estimate_sanitary_shapes", "estimate_sanitary_template", "profile_slot_levels",
           "slot_of_week"]


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


def _dry_rows(
    data: pd.DataFrame,
    eligible: np.ndarray,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """Check the inputs, then keep the eligible rows of days that are at least 90% eligible.

    This function does not choose the dry rows. Choose them first, for example
    with :func:`~sparsehydro.calibration.splits.masks.dry_weather_mask`, which
    marks each row that has no rain in the hours before it, and combine them
    with a known flow to make *eligible*, like below::

        eligible = dry_weather_mask(data, dt_hours, 72) & data["flow_cfs"].notna().to_numpy()

    This function then keeps a day only when at least 90% of its flow time
    steps are eligible, so a day that is dry for only some of its hours, or has
    too much missing flow, is left out. The estimators below need whole dry
    days, because they compare each day's mean flow with its night-time flow.
    """
    times = pd.DatetimeIndex(data["datetime"])
    eligible = np.asarray(eligible)
    flow = data["flow_cfs"].to_numpy(dtype=float)
    if (eligible.dtype != bool or eligible.shape != flow.shape or not eligible.any()
            or times.hasnans or times.has_duplicates or not times.is_monotonic_increasing):
        raise ValueError("Sanitary fitting needs an aligned boolean eligible mask "
                         "and ordered unique timestamps")
    if not np.isfinite(flow[eligible]).all() or np.any(flow[eligible] < 0):
        raise ValueError("Sanitary fitting needs finite nonnegative flow on eligible rows")
    # Keep whole dry days only: at least 90% of the day's time steps must be eligible
    row_day = times.normalize()
    steps_per_day = pd.Timedelta(days=1) / pd.Series(times).diff().median()
    eligible_share = pd.Series(eligible, index=times).groupby(row_day).sum() / steps_per_day
    whole_dry_day = eligible_share.reindex(row_day).to_numpy() >= 0.9
    return times, eligible & whole_dry_day, flow


def estimate_sanitary_template(
    data: pd.DataFrame,
    eligible: np.ndarray,
    *,
    window_days: int = 31,
) -> SanitaryTemplate:
    """Estimate hourly and daily multipliers for the dry-weather level, from whole dry days.

    The multipliers follow the dry-weather time patterns of SWMM: an hourly
    pattern, which means the diurnal pattern, with one multiplier for each hour
    of the day, and a daily pattern with one multiplier for each day of the
    week. Here each pattern is scaled to
    an average of 1, so the multipliers scale a level and the daily swing grows
    and shrinks with it, for :class:`~.components.SanitaryTemplateModel`.
    :func:`estimate_sanitary_shapes` gives amounts in cfs that are added to a
    level instead, for :class:`~.components.CenteredWeeklyPatternModel`.

    The multipliers come from the whole dry-weather flow, so they describe the
    base wastewater flow together with the normal groundwater infiltration.
    Hours and weekdays come from the timestamps, so pass local clock time.

    :param data: Frame with ``datetime`` and ``flow_cfs``.
    :param eligible: Boolean row mask of training dry-weather rows with a known
        flow (see :func:`_dry_rows`).
    :param window_days: Length of the rolling median for the daily multipliers [days].
    :returns: A :class:`~._shapes.SanitaryTemplate`.
    :raises ValueError: If the inputs are invalid or an hour or day of the week
        has no usable dry rows.
    """
    if isinstance(window_days, bool) or not isinstance(window_days, int) or window_days < 1:
        raise ValueError("window_days must be a positive integer")
    times, eligible, flow = _dry_rows(data, eligible)
    rows = pd.DataFrame({"day": times.normalize()[eligible], "hour": times.hour[eligible],
                         "flow": flow[eligible]})
    # Hourly multipliers (the diurnal pattern): how high each hour is compared
    # with the mean of its own day.
    # The mean flow of each dry day, written on every row of that day
    day_mean = rows.groupby("day")["flow"].transform("mean")
    # Each flow as a share of its day's mean (0.5 is half the mean, 1.5 is half above it)
    rows = rows.assign(share_of_day=rows["flow"] / day_mean)[day_mean > 0]
    # The median share for each hour over all dry days (for example 0.5 at 3 am)
    hourly = rows.groupby("hour")["share_of_day"].median().reindex(range(24))

    # Daily multipliers: how high each day of the week is compared with the
    # days around it.
    # The mean flow of each dry day
    dry_day_mean = rows.groupby("day")["flow"].mean()
    # The normal level around each day: the median of the day means in the window
    rolling_median = dry_day_mean.rolling(f"{window_days}D", center=True, min_periods=7).median()
    # Each day's mean as a share of that normal level (1.1 is 10% above it), so a
    # slow change over the season does not count as a weekday effect
    share_of_window = (dry_day_mean / rolling_median).dropna()
    # The mean share for each weekday over all dry days (0 is Monday)
    weekday = share_of_window.groupby(share_of_window.index.dayofweek).mean().reindex(range(7))
    if hourly.isna().any() or weekday.isna().any() or hourly.mean() <= 0 or weekday.mean() <= 0:
        raise ValueError("Too few training dry days to give a multiplier for every hour and day of the week")
    return SanitaryTemplate(tuple(hourly / hourly.mean()), tuple(weekday / weekday.mean()))


def estimate_sanitary_level(
    data: pd.DataFrame,
    eligible: np.ndarray,
    template: SanitaryTemplate,
    *,
    window_days: int = 31,
    night_hours: tuple[int, int] = (2, 5),
    minimum_days: int = 5,
    max_gap_days: int = 45,
    endpoint_hold_days: int = 0,
) -> ObservedBackground:
    """Estimate a daily dry-weather level from the rise between night-time and day-time flow.

    The level is the average dry-weather flow that the hourly and daily
    multipliers scale, as in the dry-weather inflow of SWMM, but here it can
    change from day to day. The rise from night to day is mainly base
    wastewater flow, while groundwater infiltration and storm water still
    draining stay about the same (steady flow), so subtracting the night-time
    flow cancels the steady flow. Dividing by the template's own rise turns the
    rise back into a full level, which puts the normal groundwater infiltration
    back in, while storm water still draining stays out. This suits sites where
    storm water keeps draining for weeks after a storm, such as some pump
    stations. For each whole dry day::

        level = (day mean flow - night flow) / (template day mean - template night)

    The night-time flow is the mean of the night before and the night after, so
    the next day must also be dry. The daily values are smoothed with a centred
    rolling median, never below 0, and returned as an
    :class:`~._shapes.ObservedBackground` for
    ``SanitaryTemplateModel(level_from_input=True)``. Pass local clock time.

    :param data: Frame with ``datetime`` and ``flow_cfs``.
    :param eligible: Boolean row mask of training dry-weather rows with a known
        flow (see :func:`_dry_rows`).
    :param template: Hourly and daily multipliers, for example from
        :func:`estimate_sanitary_template`.
    :param window_days: Length of the rolling median [days].
    :param night_hours: ``(start, end)`` clock hours of the night, end hour not included.
    :param minimum_days: Smallest number of daily values in a rolling window.
    :param max_gap_days: Longest gap between daily values that is bridged [days].
    :param endpoint_hold_days: Days the first and last values are held beyond
        the support (see :class:`~._shapes.ObservedBackground`).
    :returns: An :class:`~._shapes.ObservedBackground` of the daily level [cfs].
    :raises ValueError: If the inputs are invalid, the template has no
        day-night difference, or there are too few whole dry days.

    Example: a day mean of 1.0 cfs and a night flow of 0.6 cfs, with a
    template day mean of 1.0 and night of 0.2, give a level of
    (1.0 - 0.6) / (1.0 - 0.2) = 0.5 cfs.
    """
    validate_background_options(window_days, minimum_days, max_gap_days, endpoint_hold_days)
    night_start, night_end = night_hours
    if not (isinstance(night_start, int) and isinstance(night_end, int)
            and 0 <= night_start < night_end <= 24):
        raise ValueError("night_hours must be two whole clock hours with 0 <= start < end <= 24")
    if np.median(template.hourly[night_start:night_end]) >= 1.0 - 1e-9:
        raise ValueError("The template has no day-night difference: its night hours are not below its mean")
    times, eligible, flow = _dry_rows(data, eligible)
    rows = pd.DataFrame({"day": times.normalize()[eligible], "flow": flow[eligible],
                         "shape": template.predict(times[eligible]), "hour": times.hour[eligible]})
    # Day mean and night median of the flow and of the template, for each day
    day_mean = cast(pd.DataFrame, rows.groupby("day")[["flow", "shape"]].mean())
    at_night = rows["hour"].ge(night_start) & rows["hour"].lt(night_end)
    night_median: pd.DataFrame = rows[at_night].groupby("day")[["flow", "shape"]].median()
    # Night before: early on the same date. Night after: early on the next date
    night_before = night_median.reindex(day_mean.index)
    night_after = night_median.reindex(day_mean.index + pd.DateOffset(days=1)).set_axis(day_mean.index)
    # level = (day mean - night flow) / (template day mean - template night)
    day_night_rise = day_mean - (night_before + night_after) / 2
    template_rise = day_night_rise["shape"]
    daily_level = (day_night_rise["flow"] / template_rise).where(template_rise > 0).dropna()
    if len(daily_level) < minimum_days:
        raise ValueError(f"Too few eligible days with both nights for the sanitary level: "
                         f"{len(daily_level)} found, {minimum_days} needed")
    # Rolling median, never across a gap between days longer than max_gap_days
    gap_group = daily_level.index.to_series().diff().gt(pd.Timedelta(days=max_gap_days)).cumsum()
    level = pd.concat([
        days.rolling(f"{window_days}D", center=True, min_periods=minimum_days).median()
        for _, days in daily_level.groupby(gap_group)
    ]).dropna().clip(lower=0)
    if len(level) < 2:
        raise ValueError("Too few eligible days for a smoothed sanitary level; "
                         "supply more training dry data or lower minimum_days")
    dry_times = times[eligible]
    support_start = max(dry_times[0], level.index[0])
    support_end = min(dry_times[-1], level.index[-1] + pd.DateOffset(days=1) - pd.Timedelta(nanoseconds=1))
    return ObservedBackground(
        tuple(level.index), tuple(level.to_numpy()), support_start, support_end,
        window_days, minimum_days, max_gap_days, endpoint_hold_days,
    )


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
