"""Two-level storm event hierarchy detection.

Orchestrates :func:`~sparsehydro.events.define_peaks`,
:func:`~sparsehydro.events.variable_savgol_smooth`, and
:func:`~sparsehydro.events.define_complete_event_zones` into a single call that
returns global events (1-day clusters), single-peak sub-events, and the
variable-window smoothing result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .peaks import define_peaks
from .records import GlobalEvent, SubEventRecord
from .smoothing import (
    VariableSavgolResult,
    savgol_curvature,
    seed_curvature,
    variable_savgol_smooth,
)
from .zones import define_complete_event_zones

GLOBAL_EVENT_GAP_DAYS = 1.0
BASE_PAD_DAYS = 1.0


def _infer_dt_minutes(datetimes: pd.Series) -> float:
    """Infer the median time step in minutes from a datetime series."""
    if len(datetimes) < 2:
        return 5.0
    diffs = pd.to_datetime(datetimes).diff().dropna()
    if diffs.empty:
        return 5.0
    return float(pd.Timedelta(diffs.median()).total_seconds() / 60.0)


def _group_global_events(
    sub_events: list[SubEventRecord],
    times: np.ndarray,
    rain: np.ndarray,
    flow: np.ndarray,
    gap_days: float,
    pad_days: float,
) -> list[GlobalEvent]:
    """Cluster sub-events whose peaks are within *gap_days* into global events."""
    if not sub_events:
        return []
    order = sorted(range(len(sub_events)), key=lambda i: sub_events[i].peak_datetime)
    gap = pd.Timedelta(days=gap_days)
    pad = pd.Timedelta(days=pad_days)
    n = len(times)

    clusters: list[list[int]] = []
    current = [order[0]]
    for idx in order[1:]:
        if sub_events[idx].peak_datetime - sub_events[current[-1]].peak_datetime <= gap:
            current.append(idx)
        else:
            clusters.append(current)
            current = [idx]
    clusters.append(current)

    global_events: list[GlobalEvent] = []
    for gid, members in enumerate(clusters, start=1):
        members_sorted = sorted(members, key=lambda i: sub_events[i].start_datetime)
        subs = [sub_events[i] for i in members_sorted]
        for s in subs:
            s.global_id = gid
        start_dt = min(s.start_datetime for s in subs)
        end_dt = max(s.end_datetime for s in subs)
        lo_idx = min(s.start_idx for s in subs)
        hi_idx = max(s.end_idx for s in subs)
        peak_sub = max(subs, key=lambda s: s.peak_flow)
        total_rain = float(np.nansum(rain[lo_idx:hi_idx + 1]))
        total_flow = float(np.nansum(flow[lo_idx:hi_idx + 1]))
        global_events.append(GlobalEvent(
            global_id=gid,
            sub_ids=[s.sub_id for s in subs],
            start_datetime=start_dt,
            end_datetime=end_dt,
            window_start_datetime=pd.Timestamp(start_dt) - pad,
            window_end_datetime=pd.Timestamp(end_dt) + pad,
            peak_datetime=peak_sub.peak_datetime,
            total_rain=total_rain,
            total_flow=total_flow,
            effective_area=(total_flow / total_rain) if total_rain > 0 else 0.0,
        ))

    # Clamp padded windows so adjacent global events do not overlap.
    for i, ge in enumerate(global_events):
        if i > 0:
            prev_end = global_events[i - 1].end_datetime
            if ge.window_start_datetime < prev_end:
                ge.window_start_datetime = pd.Timestamp(prev_end)
        if i + 1 < len(global_events):
            nxt_start = global_events[i + 1].start_datetime
            if ge.window_end_datetime > nxt_start:
                ge.window_end_datetime = pd.Timestamp(nxt_start)
    return global_events


def detect_event_hierarchy(
    rain_stormflow: pd.DataFrame,
    *,
    rain_col: str = "rain",
    flow_col: str = "stormflow",
    dt_minutes: float | None = None,
    global_event_gap_days: float = GLOBAL_EVENT_GAP_DAYS,
    base_pad_days: float = BASE_PAD_DAYS,
    peak_kwargs: dict | None = None,
    savgol_kwargs: dict | None = None,
    zone_kwargs: dict | None = None,
) -> tuple[list[GlobalEvent], list[SubEventRecord], VariableSavgolResult]:
    """Detect global events, single-peak sub-events, and the smoothing result.

    Pipeline: rain-driven peak detection → variable-window Savitzky-Golay
    smoothing → curvature-anchored sub-event zones → 1-day global clustering.

    :param rain_stormflow: DataFrame with ``datetime``, *rain_col*, *flow_col*.
    :type rain_stormflow: pandas.DataFrame
    :param rain_col: Rainfall column name.
    :type rain_col: str
    :param flow_col: Stormflow column name.
    :type flow_col: str
    :param dt_minutes: Time step in minutes; inferred from ``datetime`` if ``None``.
    :type dt_minutes: float | None
    :param global_event_gap_days: Max peak separation (days) within one global event.
    :type global_event_gap_days: float
    :param base_pad_days: Padding (days) added around each global-event window.
    :type base_pad_days: float
    :param peak_kwargs: Extra keyword arguments for :func:`define_peaks`.
    :type peak_kwargs: dict | None
    :param savgol_kwargs: Extra keyword arguments for :func:`variable_savgol_smooth`.
    :type savgol_kwargs: dict | None
    :param zone_kwargs: Extra keyword arguments for :func:`define_complete_event_zones`.
    :type zone_kwargs: dict | None
    :returns: ``(global_events, sub_events, savgol_result)``.
    :rtype: tuple[list[GlobalEvent], list[SubEventRecord], VariableSavgolResult]
    """
    df = rain_stormflow.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    if dt_minutes is None:
        dt_minutes = _infer_dt_minutes(df["datetime"])

    peak_kwargs = dict(peak_kwargs or {})
    savgol_kwargs = dict(savgol_kwargs or {})
    zone_kwargs = dict(zone_kwargs or {})

    df_peaks, peak_idxs, rain_sections = define_peaks(
        df, rain_col=rain_col, flow_col=flow_col, **peak_kwargs
    )

    flow = np.maximum(df_peaks[flow_col].to_numpy(dtype=float), 0.0)
    smoothed, windows = variable_savgol_smooth(
        flow, peak_idxs, segments=rain_sections, return_windows=True, **savgol_kwargs
    )

    zone_input = pd.DataFrame({
        "datetime": df_peaks["datetime"].values,
        "rain": df_peaks[rain_col].to_numpy(dtype=float),
        "stormflow": flow,
    })
    sub_events = define_complete_event_zones(
        zone_input, smoothed, rain_sections, dt_minutes=dt_minutes, **zone_kwargs
    )

    times = df_peaks["datetime"].values
    rain = df_peaks[rain_col].to_numpy(dtype=float)
    global_events = _group_global_events(
        sub_events, times, rain, flow, global_event_gap_days, base_pad_days
    )

    sg_win = int(savgol_kwargs.get("win_max", 24))
    seed_win = int(savgol_kwargs.get("seed_window", 24))
    poly = int(savgol_kwargs.get("polyorder", 3))
    result = VariableSavgolResult(
        datetime=times,
        raw_flow=flow,
        smoothed=smoothed,
        windows=windows,
        seed_curvature=seed_curvature(flow, seed_window=seed_win, win_max=sg_win, polyorder=poly),
        curvature=savgol_curvature(smoothed),
        peak_idxs=np.asarray(peak_idxs, dtype=int),
    )
    return global_events, sub_events, result


__all__ = ["detect_event_hierarchy", "GLOBAL_EVENT_GAP_DAYS", "BASE_PAD_DAYS"]
