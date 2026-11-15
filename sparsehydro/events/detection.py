"""Savitzky-Golay derivative event detection (:func:`detect_events`).

Segments a stormflow series into discrete storm events from the filtered
signal and its derivatives (see :mod:`sparsehydro.filters`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

from ..filters import FilterResult, apply_savgol_filter, compute_thresholds
from .records import EventRecord


def _find_peaks_with_thresholds(
    sg_0: np.ndarray,
    height_perc: float,
    prom_perc: float,
    width_perc: float,
    distance: int,
) -> np.ndarray:
    """Return peak indices passing percentile-based thresholds.

    :param sg_0: Smoothed stormflow signal (sg_0).
    :type sg_0: numpy.ndarray
    :param height_perc: Percentile (0-100) of peak heights used as a threshold.
    :type height_perc: float
    :param prom_perc: Percentile (0-100) of peak prominences used as a threshold.
    :type prom_perc: float
    :param width_perc: Percentile (0-100) of peak widths used as a threshold.
    :type width_perc: float
    :param distance: Minimum number of samples between accepted peaks.
    :type distance: int
    :returns: Indices of accepted peaks (empty array if none).
    :rtype: numpy.ndarray
    """
    idx_all, props_all = find_peaks(sg_0, height=0, prominence=0, width=1)
    if len(idx_all) == 0:
        return np.array([], dtype=int)

    height_th = np.percentile(props_all["peak_heights"], height_perc)
    prom_th = np.percentile(props_all["prominences"], prom_perc)
    width_th = np.percentile(props_all["widths"], width_perc)

    peaks_idx, _ = find_peaks(
        sg_0, height=height_th, prominence=prom_th, width=width_th, distance=distance,
    )
    return peaks_idx


def _calc_window(window_size, sg_0_win, sg_1_win, sg_0_th, sg_1_th):
    """Adapt the boundary-walk window based on local signal magnitudes.

    :param window_size: Base window size.
    :type window_size: int
    :param sg_0_win: sg_0 values in the current window.
    :type sg_0_win: numpy.ndarray
    :param sg_1_win: sg_1 values in the current window.
    :type sg_1_win: numpy.ndarray
    :param sg_0_th: sg_0 steady-state threshold.
    :type sg_0_th: float
    :param sg_1_th: sg_1 steady-state threshold.
    :type sg_1_th: float
    :returns: Adapted window size, clamped to ``[2, window_size]``.
    :rtype: int
    """
    mean_sg0 = float(np.mean(sg_0_win))
    mean_abs_sg1 = float(np.mean(np.abs(sg_1_win)))
    if mean_abs_sg1 > 1e-6 and mean_sg0 > 1e-6:
        calc = int(window_size * (sg_0_th / mean_sg0) * (sg_1_th / mean_abs_sg1))
    else:
        calc = window_size
    return max(2, min(calc, window_size))


def _backward_walk(sg_0, sg_1, sg_2, peak, window_size, sg_0_th, sg_1_th, sg_2_th):
    """Walk backward from a peak to locate the event start index.

    :param sg_0: Smoothed signal (zeroth derivative).
    :type sg_0: numpy.ndarray
    :param sg_1: First derivative signal.
    :type sg_1: numpy.ndarray
    :param sg_2: Second derivative signal.
    :type sg_2: numpy.ndarray
    :param peak: Peak index to walk back from.
    :type peak: int
    :param window_size: Base look-back window size.
    :type window_size: int
    :param sg_0_th: sg_0 steady-state threshold.
    :type sg_0_th: float
    :param sg_1_th: sg_1 steady-state threshold.
    :type sg_1_th: float
    :param sg_2_th: sg_2 steady-state threshold.
    :type sg_2_th: float
    :returns: Tuple ``(start_idx, b2b_start)`` where *b2b_start* flags a
        back-to-back trough boundary.
    :rtype: tuple[int, bool]
    """
    lower_bound = min(window_size, peak)
    start_idx = peak
    min_val_bwd = sg_0[peak]
    min_idx_bwd = peak
    current_window_size = window_size

    for i in range(peak - 1, lower_bound - 1, -1):
        curr_val = sg_0[i]
        if curr_val < min_val_bwd:
            min_val_bwd = curr_val
            min_idx_bwd = i

        lookback_start = max(0, i - (current_window_size - 1))
        win_idx = range(lookback_start, i + 1)
        sg0_w, sg1_w, sg2_w = sg_0[win_idx], sg_1[win_idx], sg_2[win_idx]

        sscond = (
            (sg0_w < sg_0_th).all()
            and (np.abs(sg1_w) < sg_1_th).all()
            and (np.abs(sg2_w) < sg_2_th).all()
        )
        if sscond:
            return i, False

        bbcond = (sg0_w > (min_val_bwd + sg_0_th * 2)).all() and (sg1_w < -sg_1_th).all()
        if bbcond:
            return min_idx_bwd, True

        start_idx = i
        current_window_size = _calc_window(window_size, sg0_w, sg1_w, sg_0_th, sg_1_th)

    return start_idx, False


def _forward_walk(sg_0, sg_1, sg_2, peak, window_size, sg_0_th, sg_1_th, sg_2_th):
    """Walk forward from a peak to locate the event end index.

    :param sg_0: Smoothed signal (zeroth derivative).
    :type sg_0: numpy.ndarray
    :param sg_1: First derivative signal.
    :type sg_1: numpy.ndarray
    :param sg_2: Second derivative signal.
    :type sg_2: numpy.ndarray
    :param peak: Peak index to walk forward from.
    :type peak: int
    :param window_size: Base look-ahead window size.
    :type window_size: int
    :param sg_0_th: sg_0 steady-state threshold.
    :type sg_0_th: float
    :param sg_1_th: sg_1 steady-state threshold.
    :type sg_1_th: float
    :param sg_2_th: sg_2 steady-state threshold.
    :type sg_2_th: float
    :returns: Tuple ``(end_idx, b2b_end)`` where *b2b_end* flags a
        back-to-back trough boundary.
    :rtype: tuple[int, bool]
    """
    n = len(sg_0)
    upper_bound = max(peak + 1, n - window_size)
    end_idx = peak
    min_val_fwd = sg_0[peak]
    min_idx_fwd = peak
    current_window_size = window_size

    for i in range(peak + 1, upper_bound):
        curr_val = sg_0[i]
        if curr_val < min_val_fwd:
            min_val_fwd = curr_val
            min_idx_fwd = i

        lookforward_end = min(upper_bound, i + current_window_size)
        win_idx = range(i, lookforward_end)
        sg0_w, sg1_w, sg2_w = sg_0[win_idx], sg_1[win_idx], sg_2[win_idx]

        sscond = (
            (sg0_w < sg_0_th).all()
            and (np.abs(sg1_w) < sg_1_th).all()
            and (np.abs(sg2_w) < sg_2_th).all()
        )
        if sscond:
            return i, False

        bbcond = (sg0_w > (min_val_fwd + sg_0_th * 2)).all() and (sg1_w > sg_1_th).all()
        if bbcond:
            return min_idx_fwd, True

        end_idx = i
        current_window_size = _calc_window(window_size, sg0_w, sg1_w, sg_0_th, sg_1_th)

    return end_idx, False


def _resolve_collisions(
    events_raw, peak, start_idx, end_idx, b2b_start, b2b_end,
    sg_0, sg_0_th, simp_th, back_to_back,
):
    """Merge or split the current event against already-detected events.

    :param events_raw: Previously accepted raw event dicts (mutated in place).
    :type events_raw: list[dict]
    :param peak: Peak index of the current candidate event.
    :type peak: int
    :param start_idx: Start index of the current candidate event.
    :type start_idx: int
    :param end_idx: End index of the current candidate event.
    :type end_idx: int
    :param b2b_start: Whether the candidate starts at a back-to-back trough.
    :type b2b_start: bool
    :param b2b_end: Whether the candidate ends at a back-to-back trough.
    :type b2b_end: bool
    :param sg_0: Smoothed signal (zeroth derivative).
    :type sg_0: numpy.ndarray
    :param sg_0_th: sg_0 steady-state threshold.
    :type sg_0_th: float
    :param simp_th: Height-ratio threshold for merging near-equal peaks (0-1).
    :type simp_th: float
    :param back_to_back: Whether to keep distinct events separated at troughs.
    :type back_to_back: bool
    :returns: Updated ``(events_raw, peak, start_idx, end_idx, b2b_start,
        b2b_end)``.
    :rtype: tuple[list[dict], int, int, int, bool, bool]
    """
    while events_raw:
        prev = events_raw[-1]
        if not (start_idx <= prev["end_idx"] or b2b_start or prev["b2b_end"]):
            break

        if not back_to_back:
            start_idx = min(prev["start_idx"], start_idx)
            end_idx = max(prev["end_idx"], end_idx)
            b2b_start = b2b_end = False
            if sg_0[prev["peak_idx"]] > sg_0[peak]:
                peak = prev["peak_idx"]
            events_raw.pop()
            continue

        p1, p2 = prev["peak_idx"], peak
        min_idx = int(np.argmin(sg_0[p1: p2 + 1])) + p1
        val_trough = sg_0[min_idx]
        h1 = max(1e-6, float(sg_0[p1]) - val_trough)
        h2 = max(1e-6, float(sg_0[p2]) - val_trough)

        if min(h1, h2) < simp_th * max(h1, h2):
            if h1 < h2:
                start_idx = min(prev["start_idx"], start_idx)
                b2b_start = prev["b2b_start"]
            else:
                start_idx = min(prev["start_idx"], start_idx)
                end_idx = max(prev["end_idx"], end_idx)
                b2b_start = prev["b2b_start"]
                peak = p1
            events_raw.pop()
            continue

        prev["end_idx"] = min_idx
        prev["b2b_end"] = True
        if start_idx < min_idx:
            start_idx = min_idx
            b2b_start = True
        break

    return events_raw, peak, start_idx, end_idx, b2b_start, b2b_end


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_events(
    rain_stormflow: pd.DataFrame,
    time_range: tuple[str, str] | None = None,
    *,
    back_to_back: bool = True,
    height_perc: float = 25.0,
    prom_perc: float = 25.0,
    width_perc: float = 10.0,
    distance: int = 48,
    window_length: int = 48,
    polyorder: int = 2,
    deriv_factor: float = 1.2,
    sg_0_per: float = 0.15,
    sg_1_per: float = 0.25,
    sg_2_per: float = 0.25,
    window_size: int = 72,
    simp_th: float = 0.5,
    verbose: bool = False,
) -> tuple[list[EventRecord], FilterResult]:
    """Detect storm events using Savitzky-Golay derivative segmentation.

    :param rain_stormflow: Must contain ``datetime``, ``rain``, ``stormflow``.
    :type rain_stormflow: pandas.DataFrame
    :param time_range: Restrict detection to this ``(start, end)`` date range.
    :type time_range: tuple[str, str] | None
    :param back_to_back: When ``True``, keep distinct events separated at troughs.
    :type back_to_back: bool
    :param height_perc: Percentile threshold for peak height (0-100).
    :type height_perc: float
    :param prom_perc: Percentile threshold for peak prominence (0-100).
    :type prom_perc: float
    :param width_perc: Percentile threshold for peak width (0-100).
    :type width_perc: float
    :param distance: Minimum samples between peaks.
    :type distance: int
    :param window_length: Base Savitzky-Golay window for sg_0.
    :type window_length: int
    :param polyorder: Savitzky-Golay polynomial order.
    :type polyorder: int
    :param deriv_factor: Window multiplier for each successive derivative pass.
    :type deriv_factor: float
    :param sg_0_per: Threshold multiplier for the sg_0 boundary signal.
    :type sg_0_per: float
    :param sg_1_per: Threshold multiplier for the sg_1 boundary signal.
    :type sg_1_per: float
    :param sg_2_per: Threshold multiplier for the sg_2 boundary signal.
    :type sg_2_per: float
    :param window_size: Look-ahead/look-back window for boundary walks.
    :type window_size: int
    :param simp_th: Height-ratio threshold for merging near-equal adjacent peaks
        (0-1).
    :type simp_th: float
    :param verbose: Print progress diagnostics.
    :type verbose: bool
    :returns: Tuple ``(events, filter_result)`` where *events* are the detected
        :class:`EventRecord` objects (``event_id`` starts at 1) and
        *filter_result* holds the sg signals and computed thresholds for
        diagnostics.
    :rtype: tuple[list[EventRecord], FilterResult]
    """
    df = rain_stormflow.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["stormflow"] = np.maximum(df["stormflow"].values, 0.0)

    if time_range is not None:
        s, e = pd.to_datetime(time_range[0]), pd.to_datetime(time_range[1])
        df = df[(df["datetime"] >= s) & (df["datetime"] <= e)].reset_index(drop=True)

    if len(df) < 5:
        fr = apply_savgol_filter(df, window_length, polyorder, deriv_factor)
        return [], compute_thresholds(fr, sg_0_per, sg_1_per, sg_2_per)

    n = len(df)
    safe_w = max(3 if 3 % 2 != 0 else 5, min(5, n if n % 2 != 0 else n - 1))
    sg_peak = np.maximum(
        savgol_filter(
            np.maximum(df["stormflow"].values, 0.0),
            window_length=safe_w,
            polyorder=min(polyorder, safe_w - 1),
            mode="interp",
        ),
        0.0,
    )
    peaks_idx = _find_peaks_with_thresholds(sg_peak, height_perc, prom_perc, width_perc, distance)

    if verbose:
        print(f"Candidate peaks: {len(peaks_idx)}")

    fr = apply_savgol_filter(df, window_length, polyorder, deriv_factor)
    fr = compute_thresholds(fr, sg_0_per, sg_1_per, sg_2_per)

    sg_0, sg_1, sg_2 = fr.sg_0, fr.sg_1, fr.sg_2
    sg_0_th = fr.thresholds["sg_0_th"]
    sg_1_th = fr.thresholds["sg_1_th"]
    sg_2_th = fr.thresholds["sg_2_th"]

    events_raw: list[dict] = []
    for idx, peak in enumerate(peaks_idx):
        start_idx, b2b_start = _backward_walk(sg_0, sg_1, sg_2, peak, window_size, sg_0_th, sg_1_th, sg_2_th)
        end_idx, b2b_end = _forward_walk(sg_0, sg_1, sg_2, peak, window_size, sg_0_th, sg_1_th, sg_2_th)

        if verbose:
            print(f"Peak {idx+1}/{len(peaks_idx)} idx={peak} → [{start_idx}, {end_idx}] b2b=({b2b_start},{b2b_end})")

        events_raw, peak, start_idx, end_idx, b2b_start, b2b_end = _resolve_collisions(
            events_raw, peak, start_idx, end_idx, b2b_start, b2b_end,
            sg_0, sg_0_th, simp_th, back_to_back,
        )

        if end_idx > start_idx:
            events_raw.append({"peak_idx": peak, "start_idx": start_idx, "end_idx": end_idx,
                                "b2b_start": b2b_start, "b2b_end": b2b_end})

    events: list[EventRecord] = []
    for i, raw in enumerate(events_raw):
        s_idx, e_idx = raw["start_idx"], raw["end_idx"]
        slice_df = df.iloc[s_idx: e_idx + 1]
        total_rain = float(slice_df["rain"].sum())
        total_flow = float(np.maximum(slice_df["stormflow"].values, 0.0).sum())
        ae = total_flow / total_rain if total_rain > 0 else 0.0
        events.append(
            EventRecord(
                event_id=i + 1,
                start_datetime=pd.Timestamp(slice_df["datetime"].iloc[0]),
                end_datetime=pd.Timestamp(slice_df["datetime"].iloc[-1]),
                peak_datetime=pd.Timestamp(df["datetime"].iloc[raw["peak_idx"]]),
                total_rain=total_rain,
                total_flow=total_flow,
                effective_area=ae,
                peak_flow=float(slice_df["stormflow"].max()),
                b2b_start=bool(raw["b2b_start"]),
                b2b_end=bool(raw["b2b_end"]),
            )
        )

    if verbose:
        print(f"Detected {len(events)} events.")

    return events, fr


__all__ = ["detect_events"]
