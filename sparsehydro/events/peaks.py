"""Rain-driven storm peak detection.

Ported from the Parsimonious Functions ``define_peaks``.  Identifies rain
sections from cumulative rainfall, then within each section returns the smoothed
flow's global maximum plus any prominent local maxima (multi-burst storms), each
snapped to the nearest raw-flow apex.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

_SNAP_WINDOW = 12  # ±12 samples ≈ ±1 h at 5-min cadence


def define_peaks(
    df: pd.DataFrame,
    *,
    rain_col: str = "rain",
    flow_col: str = "stormflow",
    rain_window_hours: int = 12,
    total_rain_min: float = 0.05,
    dry_gap_hours: int = 6,
    sg_window: int = 31,
    polyorder: int = 2,
    neg_deriv_hours: int = 3,
    recession_hours: int = 48,
    prominence_frac: float = 0.10,
    min_peak_dist: int = 12,
) -> tuple[pd.DataFrame, np.ndarray, list[tuple[int, int]]]:
    """Detect rain-driven storm peaks and rain-section zones.

    :param df: DataFrame with ``datetime``, *rain_col*, and *flow_col* columns.
    :type df: pandas.DataFrame
    :param rain_col: Rainfall column name.
    :type rain_col: str
    :param flow_col: Stormflow column name.
    :type flow_col: str
    :param rain_window_hours: Rolling window (samples) for cumulative rainfall.
    :type rain_window_hours: int
    :param total_rain_min: Minimum cumulative rainfall to flag a rain section.
    :type total_rain_min: float
    :param dry_gap_hours: Dry gap (hours) that splits adjacent rain sections.
    :type dry_gap_hours: int
    :param sg_window: Savitzky-Golay window for the smoothed flow.
    :type sg_window: int
    :param polyorder: Savitzky-Golay polynomial order.
    :type polyorder: int
    :param neg_deriv_hours: Sustained-negative-slope length (samples) confirming
        recession onset.
    :type neg_deriv_hours: int
    :param recession_hours: Maximum recession extension (hours) for a zone end.
    :type recession_hours: int
    :param prominence_frac: Minimum sub-peak prominence as a fraction of the
        section's smoothed maximum.
    :type prominence_frac: float
    :param min_peak_dist: Minimum samples between detected peaks.
    :type min_peak_dist: int
    :returns: ``(df_with_sg, peak_idxs, rain_sections)`` — the input frame with an
        added ``sg`` column, sorted deduplicated peak indices, and one
        ``(start, end)`` zone per rain section.
    :rtype: tuple[pandas.DataFrame, numpy.ndarray, list[tuple[int, int]]]
    """
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    rain = df[rain_col].to_numpy(dtype=float)
    flow = df[flow_col].to_numpy(dtype=float)
    dt = df["datetime"].values
    n = len(df)

    win = sg_window if sg_window < n else n - 1
    if win % 2 == 0:
        win -= 1
    win = max(win, polyorder + 2)
    sg = savgol_filter(flow, win, polyorder) if win <= n and win > polyorder else flow.copy()

    roll = df[rain_col].rolling(rain_window_hours, min_periods=1).sum().to_numpy()
    rain_flag = roll >= total_rain_min

    rain_sections: list[tuple[int, int]] = []
    in_section = False
    start_idx: int | None = None
    last_rain_time = None
    dry_gap = np.timedelta64(int(dry_gap_hours), "h")
    for i in range(n):
        if rain_flag[i]:
            if not in_section:
                in_section = True
                start_idx = i
            last_rain_time = dt[i]
        elif in_section and last_rain_time is not None and dt[i] - last_rain_time >= dry_gap:
            rain_sections.append((int(start_idx), i - 1))
            in_section = False
            start_idx = None
            last_rain_time = None
    if in_section and start_idx is not None:
        rain_sections.append((int(start_idx), n - 1))

    def _recession_end(peak_idx: int, end_idx: int) -> int:
        dsg = np.diff(sg)
        m = neg_deriv_hours
        for i in range(peak_idx, min(end_idx, n - m)):
            if np.all(dsg[i:i + m] < 0):
                min_idx = i
                for j in range(i, peak_idx, -1):
                    if flow[j] > flow[j - 1]:
                        break
                    min_idx = j
                return min_idx
        return end_idx

    peak_idxs: list[int] = []
    event_zones: list[tuple[int, int]] = []
    for (s, e) in rain_sections:
        sg_slice = sg[s:e + 1]
        if len(sg_slice) == 0:
            continue
        section_max = float(sg_slice.max())
        global_max_local = int(np.argmax(sg_slice))
        if section_max > 0 and len(sg_slice) >= 3:
            locals_, _ = find_peaks(sg_slice, prominence=prominence_frac * section_max, distance=min_peak_dist)
        else:
            locals_ = np.array([], dtype=int)
        section_locals = sorted({global_max_local, *(int(p) for p in locals_)})
        section_globals = [s + p for p in section_locals]

        snapped = []
        for p_global in section_globals:
            lo = max(0, p_global - _SNAP_WINDOW)
            hi = min(len(flow), p_global + _SNAP_WINDOW + 1)
            snapped.append(int(lo + np.argmax(flow[lo:hi])) if lo < hi else p_global)
        section_globals = sorted(set(snapped))
        peak_idxs.extend(section_globals)

        global_peak_idx = s + global_max_local
        zone_end_time = dt[e] + np.timedelta64(int(recession_hours), "h")
        zone_end = int(np.searchsorted(dt, zone_end_time, side="right")) - 1
        zone_end = min(zone_end, n - 1)
        zone_end = _recession_end(global_peak_idx, zone_end)
        event_zones.append((s, zone_end))

    df["sg"] = sg
    return df, np.array(sorted(set(peak_idxs)), dtype=int), event_zones


__all__ = ["define_peaks"]
