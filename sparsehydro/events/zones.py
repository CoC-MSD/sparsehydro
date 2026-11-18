"""Complete event-zone definition: split rain sections into single-peak sub-events.

Ported (streamlined) from the Parsimonious Functions ``define_complete_event_zones``.
For each rain section it detects secondary crests, conservatively splits at valleys,
and assigns curvature-anchored ``rise_start`` / ``peak`` / ``tail_start`` / ``end``
boundaries to each resulting single-peak sub-event.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .bimodality import (
    BIMODALITY_SPLIT_MAX_VALLEY_FRAC,
    BIMODALITY_SPLIT_MIN_SEPARATION_HRS,
    calculate_peak_zone_bimodality,
    calculate_rain_bimodality,
)
from .records import SubEventRecord
from .smoothing import savgol_curvature, savgol_slope

DECLINE_RISE_RATIO = 2.0 / 3.0
END_SLOPE_WINDOW = 48
END_SLOPE_FRAC = 0.05
END_SLOPE_HOLD = 6


def _rain_onset_idx(rain: np.ndarray, peak_idx: int, lower: int, lead_steps: int) -> int:
    """Return the earliest wet-sample index within the lead window before *peak_idx*."""
    lo = max(lower, peak_idx - lead_steps)
    seg = rain[lo:peak_idx + 1]
    wet = np.flatnonzero(seg > 0)
    return int(lo + wet[0]) if wet.size else peak_idx


def _prev_curvature_knee(curv: np.ndarray, peak_idx: int, lower: int) -> int:
    """Return the curvature local-max index just before *peak_idx* (rise knee)."""
    lo = max(lower, 1)
    best = lower
    for i in range(lo, peak_idx):
        if curv[i] >= curv[i - 1] and curv[i] >= curv[min(i + 1, len(curv) - 1)] and curv[i] > 0:
            best = i
    return best


def _tail_start_idx(curv: np.ndarray, peak_idx: int, upper: int, rise_start: int) -> int:
    """Return the first post-peak curvature up-crossing (concave→convex inflection)."""
    for i in range(peak_idx + 1, min(upper, len(curv) - 1)):
        if curv[i - 1] <= 0.0 < curv[i]:
            return i
    fallback = peak_idx + int(round(DECLINE_RISE_RATIO * max(peak_idx - rise_start, 1)))
    return int(min(max(fallback, peak_idx + 1), upper))


def _recession_flatten_idx(slope: np.ndarray, peak_idx: int, hi: int, frac: float, hold: int) -> int | None:
    """Return the index where the recession slope settles to ~0 (or ``None``)."""
    if hi <= peak_idx + 1:
        return None
    seg = slope[peak_idx:hi + 1]
    if seg.size == 0:
        return None
    steep_local = int(np.argmin(seg))
    steep_val = float(seg[steep_local])
    if steep_val >= 0:
        return None
    thresh = frac * abs(steep_val)
    run = 0
    start_flat = None
    for i in range(peak_idx + steep_local, hi + 1):
        if abs(float(slope[i])) <= thresh:
            if start_flat is None:
                start_flat = i
            run += 1
            if run >= hold:
                return start_flat
        else:
            run = 0
            start_flat = None
    return None


def define_complete_event_zones(
    rain_stormflow: pd.DataFrame,
    sg_full,
    rain_sections: list[tuple[int, int]],
    *,
    dt_minutes: float = 5.0,
    rain_lead_hours: float = 3.0,
    decline_rise_ratio: float = DECLINE_RISE_RATIO,
    end_slope_window: int = END_SLOPE_WINDOW,
    end_slope_frac: float = END_SLOPE_FRAC,
    end_slope_hold: int = END_SLOPE_HOLD,
    split_max_valley_frac: float = BIMODALITY_SPLIT_MAX_VALLEY_FRAC,
    split_min_separation_hours: float = BIMODALITY_SPLIT_MIN_SEPARATION_HRS,
) -> list[SubEventRecord]:
    """Split rain sections into single-peak sub-events with zone boundaries.

    :param rain_stormflow: DataFrame with ``datetime``, ``rain``, ``stormflow``.
    :type rain_stormflow: pandas.DataFrame
    :param sg_full: Smoothed stormflow aligned to *rain_stormflow*.
    :type sg_full: array-like
    :param rain_sections: ``(start, end)`` index pairs from
        :func:`~sparsehydro.events.define_peaks`.
    :type rain_sections: list[tuple[int, int]]
    :param dt_minutes: Time step in minutes.
    :type dt_minutes: float
    :param rain_lead_hours: Lead window (hours) for the event-start rain onset.
    :type rain_lead_hours: float
    :param decline_rise_ratio: Fallback peak-zone-end fraction of the rise duration.
    :type decline_rise_ratio: float
    :param end_slope_window: Wide Savitzky-Golay window whose slope→0 ends the event.
    :type end_slope_window: int
    :param end_slope_frac: Flatten threshold as a fraction of the steepest descent.
    :type end_slope_frac: float
    :param end_slope_hold: Samples the slope must hold ~0 to count as settled.
    :type end_slope_hold: int
    :param split_max_valley_frac: Split only if the valley ≤ this × smaller crest.
    :type split_max_valley_frac: float
    :param split_min_separation_hours: Split only if peaks ≥ this many hours apart.
    :type split_min_separation_hours: float
    :returns: Single-peak sub-events (``global_id`` set later by the hierarchy).
    :rtype: list[SubEventRecord]
    """
    df = rain_stormflow.reset_index(drop=True).copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    times = df["datetime"].values
    rain = df["rain"].to_numpy(dtype=float)
    flow = np.maximum(df["stormflow"].to_numpy(dtype=float), 0.0)
    sg = np.asarray(sg_full, dtype=float)
    n = len(df)
    curv = savgol_curvature(sg)
    slope_wide = savgol_slope(flow, window=int(end_slope_window), polyorder=2)

    steps_per_hour = max(1, int(round(60.0 / float(dt_minutes)))) if dt_minutes else 12
    lead_steps = int(round(rain_lead_hours * steps_per_hour))
    min_sep_steps = int(round(split_min_separation_hours * steps_per_hour))

    sub_events: list[SubEventRecord] = []
    sub_id = 0

    for sec_i, (s_idx, e_idx) in enumerate(rain_sections):
        s_idx = int(max(0, s_idx))
        e_idx = int(min(n - 1, e_idx))
        if e_idx - s_idx < 2:
            continue
        max_end = rain_sections[sec_i + 1][0] - 1 if sec_i + 1 < len(rain_sections) else n - 1
        max_end = int(min(max_end, n - 1))

        primary_local = int(np.argmax(sg[s_idx:e_idx + 1]))
        primary_peak = s_idx + primary_local

        bim = calculate_peak_zone_bimodality(
            sg, flow, s_idx, primary_peak, e_idx, rain=rain, curv=curv, dt_minutes=dt_minutes,
        )
        peaks = sorted(set(bim["peak_zone_peak_list"]) | {primary_peak})

        # Conservative valley splitting between consecutive accepted peaks.
        split_points: list[int] = []
        for pa, pb in zip(peaks[:-1], peaks[1:]):
            if pb - pa < min_sep_steps:
                continue
            valley_local = int(np.argmin(sg[pa:pb + 1]))
            valley = pa + valley_local
            valley_val = float(sg[valley])
            smaller_crest = min(float(sg[pa]), float(sg[pb]))
            if smaller_crest > 0 and valley_val <= split_max_valley_frac * smaller_crest:
                split_points.append(valley)

        # Build segment boundaries at valleys; each segment holds exactly one peak.
        seg_bounds = [s_idx, *split_points, e_idx]
        segments = list(zip(seg_bounds[:-1], seg_bounds[1:]))

        rain_bi = calculate_rain_bimodality(rain, s_idx, e_idx, dt_minutes=dt_minutes)

        for seg_i, (seg_lo, seg_hi) in enumerate(segments):
            seg_lo = int(seg_lo)
            seg_hi = int(min(seg_hi, max_end))
            if seg_hi - seg_lo < 2:
                continue
            peak = seg_lo + int(np.argmax(sg[seg_lo:seg_hi + 1]))

            rise_knee = _prev_curvature_knee(curv, peak, seg_lo)
            rain_onset = _rain_onset_idx(rain, peak, seg_lo, lead_steps)
            rise_start = int(max(seg_lo, min(rise_knee, peak)))
            start = int(max(seg_lo, min(rise_start, rain_onset)))

            tail_start = _tail_start_idx(curv, peak, seg_hi, rise_start)

            flat = _recession_flatten_idx(slope_wide, peak, min(seg_hi, max_end), end_slope_frac, end_slope_hold)
            end = int(flat) if flat is not None else seg_hi
            end = int(min(max(end, peak + 1), max_end))
            if end <= start:
                continue

            total_rain = float(np.nansum(rain[start:end + 1]))
            total_flow = float(np.nansum(flow[start:end + 1]))
            effective_area = total_flow / total_rain if total_rain > 0 else 0.0
            peak_flow = float(np.nanmax(flow[start:end + 1]))

            sub_id += 1
            sub_events.append(SubEventRecord(
                sub_id=sub_id,
                global_id=0,
                start_datetime=pd.Timestamp(times[start]),
                end_datetime=pd.Timestamp(times[end]),
                peak_datetime=pd.Timestamp(times[peak]),
                rise_start_datetime=pd.Timestamp(times[rise_start]),
                tail_start_datetime=pd.Timestamp(times[tail_start]),
                total_rain=total_rain,
                total_flow=total_flow,
                effective_area=effective_area,
                peak_flow=peak_flow,
                start_idx=start,
                end_idx=end,
                peak_idx=peak,
                rise_start_idx=rise_start,
                tail_start_idx=tail_start,
                bimodality_index=float(bim["bimodality_index"]),
                rain_bimodality_index=float(rain_bi["rain_bimodality_index"]),
                is_bimodal=bool(bim["is_bimodal"]) or bool(rain_bi["rain_is_bimodal"]),
            ))

    return sub_events


__all__ = ["define_complete_event_zones"]
