"""Variable-window Savitzky-Golay smoothing and derivative helpers.

Ported from the Parsimonious Functions ``data_processing.py``.  The per-point SG
window tracks the seed-pass curvature ``|sg_2|``: SMALL windows where curvature is
large (sharp peaks) and LARGE windows on flat recession tails.  Detected peaks are
pinned to the smallest window and a final tapered, raw-capped lift forces the
smoothed curve to pass exactly through each raw peak.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d
from scipy.signal import savgol_filter


@dataclass
class VariableSavgolResult:
    """Output of :func:`variable_savgol_smooth` bundled for plotting and fitting.

    :ivar datetime: Timestamps aligned to the smoothed series.
    :vartype datetime: numpy.ndarray
    :ivar raw_flow: Original (clipped ≥ 0) stormflow.
    :vartype raw_flow: numpy.ndarray
    :ivar smoothed: Variable-window smoothed stormflow.
    :vartype smoothed: numpy.ndarray
    :ivar windows: Per-point SG window size (samples) used for each point.
    :vartype windows: numpy.ndarray
    :ivar seed_curvature: Fixed-window seed second derivative that drove the
        window mapping.
    :vartype seed_curvature: numpy.ndarray
    :ivar curvature: Second derivative of the smoothed curve (inflection signal).
    :vartype curvature: numpy.ndarray
    :ivar peak_idxs: Peak indices pinned during smoothing.
    :vartype peak_idxs: numpy.ndarray
    """

    datetime: np.ndarray
    raw_flow: np.ndarray
    smoothed: np.ndarray
    windows: np.ndarray
    seed_curvature: np.ndarray
    curvature: np.ndarray
    peak_idxs: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))

    def to_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame with the smoothing signals.

        :returns: Columns ``datetime``, ``raw_flow``, ``smoothed``, ``windows``,
            ``seed_curvature``, ``curvature``.
        :rtype: pandas.DataFrame
        """
        return pd.DataFrame({
            "datetime": self.datetime,
            "raw_flow": self.raw_flow,
            "smoothed": self.smoothed,
            "windows": self.windows,
            "seed_curvature": self.seed_curvature,
            "curvature": self.curvature,
        })


def seed_curvature(flow, seed_window: int = 24, win_max: int = 24, polyorder: int = 2) -> np.ndarray:
    """Fixed-window Savitzky-Golay second derivative of the raw flow.

    This "seed" curvature drives the per-point window sizing in
    :func:`variable_savgol_smooth`.  The window is clamped identically to that
    function (capped at *win_max* and the series length, forced odd, floored at
    ``polyorder + 2``).

    :param flow: Raw flow values.
    :type flow: array-like
    :param seed_window: Fixed window for the seed derivative.
    :type seed_window: int
    :param win_max: Upper window cap shared with the smoother.
    :type win_max: int
    :param polyorder: Savitzky-Golay polynomial order.
    :type polyorder: int
    :returns: Signed second derivative (all zeros if the series is too short).
    :rtype: numpy.ndarray
    """
    arr = np.asarray(flow, dtype=float)
    n = arr.size
    if n == 0:
        return arr.copy()
    hi = int(min(win_max, n))
    if hi % 2 == 0:
        hi -= 1
    seed_w = int(seed_window)
    if seed_w % 2 == 0:
        seed_w += 1
    seed_w = min(max(seed_w, polyorder + 2), hi)
    if seed_w % 2 == 0:
        seed_w -= 1
    if seed_w < polyorder + 2 or seed_w > n:
        return np.zeros_like(arr)
    return savgol_filter(arr, seed_w, polyorder, deriv=2)


def savgol_curvature(values, window: int = 31, polyorder: int = 2) -> np.ndarray:
    """Second derivative (curvature) of an already-smoothed signal.

    Zero-crossings mark hydrograph inflection points (used for zone boundaries).

    :param values: Signal to differentiate (typically the smoothed flow).
    :type values: array-like
    :param window: Savitzky-Golay window (clamped odd and valid for short series).
    :type window: int
    :param polyorder: Savitzky-Golay polynomial order.
    :type polyorder: int
    :returns: Second derivative (all zeros if too short to fit a window).
    :rtype: numpy.ndarray
    """
    arr = np.asarray(values, dtype=float)
    n = arr.size
    window = int(window)
    if window % 2 == 0:
        window += 1
    if window > n:
        window = max(n - 1, polyorder + 2)
        if window % 2 == 0:
            window -= 1
    if window < polyorder + 2:
        window = polyorder + 2
        if window % 2 == 0:
            window += 1
        if window > n:
            return np.zeros_like(arr)
    return savgol_filter(arr, window, polyorder, deriv=2)


def savgol_slope(values, window: int = 31, polyorder: int = 2) -> np.ndarray:
    """First derivative (slope) of a signal, in value-units per sample.

    A wide window yields the slope of a wide smoothing pass, which returns to ~0
    once the recession has settled — the basis for event-end detection.

    :param values: Signal to differentiate.
    :type values: array-like
    :param window: Savitzky-Golay window (clamped odd and valid for short series).
    :type window: int
    :param polyorder: Savitzky-Golay polynomial order.
    :type polyorder: int
    :returns: First derivative (all zeros if too short to fit a window).
    :rtype: numpy.ndarray
    """
    arr = np.asarray(values, dtype=float)
    n = arr.size
    window = int(window)
    if window % 2 == 0:
        window += 1
    if window > n:
        window = max(n - 1, polyorder + 2)
        if window % 2 == 0:
            window -= 1
    if window < polyorder + 2:
        window = polyorder + 2
        if window % 2 == 0:
            window += 1
        if window > n:
            return np.zeros_like(arr)
    return savgol_filter(arr, window, polyorder, deriv=1)


def apply_peak_lift(values, flow, peak_idxs, lift_pad: int = 5) -> np.ndarray:
    """Force a smoothed curve to pass exactly through each raw peak.

    For every peak the ``(raw - smoothed)`` residual is added back, cosine-tapered
    to zero over ``±lift_pad`` samples and capped at the raw envelope so a tapered
    shoulder cannot overshoot.  The taper half-width is clamped to half the
    distance to the nearest neighbouring peak.

    :param values: Smoothed values to lift.
    :type values: array-like
    :param flow: Raw flow (the lift target and cap).
    :type flow: array-like
    :param peak_idxs: Indices of peak apexes to hit exactly.
    :type peak_idxs: iterable of int
    :param lift_pad: Cosine-taper half-width in samples.
    :type lift_pad: int
    :returns: New array equal to the smoothed curve lifted through each peak.
    :rtype: numpy.ndarray
    """
    out = np.asarray(values, dtype=float).copy()
    flow = np.asarray(flow, dtype=float)
    n = out.size
    peaks = np.asarray(sorted({int(p) for p in peak_idxs if 0 <= p < n}), dtype=int)
    for k, p in enumerate(peaks):
        pad = lift_pad
        if k > 0:
            pad = min(pad, (p - peaks[k - 1]) // 2)
        if k < peaks.size - 1:
            pad = min(pad, (peaks[k + 1] - p) // 2)
        pad = max(int(pad), 0)
        residual = float(flow[p] - out[p])
        j = np.arange(-pad, pad + 1)
        j = j[(p + j >= 0) & (p + j < n)]
        weight = 0.5 * (1.0 + np.cos(np.pi * np.abs(j) / (pad + 1)))
        out[p + j] = np.minimum(out[p + j] + residual * weight, flow[p + j])
    return out


def variable_savgol_smooth(
    flow,
    peak_idxs,
    segments: list[tuple[int, int]] | None = None,
    *,
    seed_window: int = 24,
    win_min: int = 5,
    win_max: int = 24,
    polyorder: int = 3,
    peak_pad: int = 2,
    lift_pad: int = 5,
    curv_env_window: int = 48,
    anchor_pct: float = 90.0,
    return_windows: bool = False,
):
    """Curvature-driven variable-window Savitzky-Golay smoother.

    Small windows track sharp curvature near peaks; large windows smooth flat
    tails.  Detected peaks are pinned to *win_min* and a tapered raw-capped lift
    forces the curve through each peak.

    :param flow: Raw flow values.
    :type flow: array-like
    :param peak_idxs: Indices of detected peak apexes to hit exactly.
    :type peak_idxs: iterable of int
    :param segments: ``(start, end)`` index pairs for per-event curvature
        normalisation; ``None`` uses a single global anchor.
    :type segments: list[tuple[int, int]] | None
    :param seed_window: Fixed window for the seed second derivative.
    :type seed_window: int
    :param win_min: Minimum (peak) window (odd; even rounds down).
    :type win_min: int
    :param win_max: Maximum (tail) window (odd; even rounds down).
    :type win_max: int
    :param polyorder: Savitzky-Golay polynomial order.
    :type polyorder: int
    :param peak_pad: Samples each side of a peak held at *win_min*.
    :type peak_pad: int
    :param lift_pad: Cosine-taper half-width of the peak lift.
    :type lift_pad: int
    :param curv_env_window: Rolling-max envelope width applied to ``|sg_2|``
        before window mapping (bridges curvature nulls at limb inflections).
    :type curv_env_window: int
    :param anchor_pct: Percentile of ``|sg_2|`` used as the per-segment curvature
        anchor (100 = max).
    :type anchor_pct: float
    :param return_windows: When ``True`` also return the per-point window array.
    :type return_windows: bool
    :returns: Smoothed flow, or ``(smoothed, windows)`` if *return_windows*.
    :rtype: numpy.ndarray | tuple[numpy.ndarray, numpy.ndarray]
    """
    flow = np.asarray(flow, dtype=float)
    n = flow.size
    if n == 0:
        return (flow.copy(), np.zeros(0, dtype=int)) if return_windows else flow.copy()

    lo = max(int(win_min), polyorder + 2)
    if lo % 2 == 0:
        lo += 1
    hi = int(min(win_max, n))
    if hi % 2 == 0:
        hi -= 1
    if hi < lo or hi < polyorder + 2 or hi > n:
        windows = np.full(n, hi if hi > 0 else int(win_max), dtype=int)
        return (flow.copy(), windows) if return_windows else flow.copy()

    abs_sg2 = np.abs(seed_curvature(flow, seed_window=seed_window, win_max=win_max, polyorder=polyorder))

    env_w = int(curv_env_window) if curv_env_window else 0
    if env_w > 1 and abs_sg2.size:
        abs_sg2 = maximum_filter1d(abs_sg2, size=env_w, mode="nearest")

    pct = float(anchor_pct)
    global_M = float(abs_sg2.max()) if abs_sg2.size else 0.0
    anchor = np.full(n, global_M, dtype=float)
    if segments:
        for s, e in segments:
            s = max(0, int(s))
            e = min(n - 1, int(e))
            if s <= e:
                seg_curv = abs_sg2[s:e + 1]
                anchor[s:e + 1] = float(np.percentile(seg_curv, pct) if pct < 100 else seg_curv.max())

    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(anchor > 0, np.clip(abs_sg2 / anchor, 0.0, 1.0), 0.0)
    windows = win_max + (win_min - win_max) * frac
    windows = np.rint(windows).astype(int)
    windows = np.where(windows % 2 == 0, windows - 1, windows)
    windows = np.clip(windows, lo, hi)
    windows[anchor <= 0] = hi

    if segments:
        in_seg = np.zeros(n, dtype=bool)
        for s, e in segments:
            s = max(0, int(s))
            e = min(n - 1, int(e))
            if s <= e:
                in_seg[s:e + 1] = True
        if in_seg.any() and not in_seg.all():
            idx = np.arange(n)
            left = np.where(in_seg, idx, -1)
            np.maximum.accumulate(left, out=left)
            right = np.where(in_seg, idx, n)
            right = np.minimum.accumulate(right[::-1])[::-1]
            dist_l = np.where(left >= 0, idx - left, n)
            dist_r = np.where(right < n, right - idx, n)
            dist = np.minimum(dist_l, dist_r)
            ramp = win_min + 2 * (dist // 2)
            ramp = np.clip(ramp, lo, hi)
            windows = np.where(in_seg, windows, ramp)

    peaks = np.asarray(sorted({int(p) for p in peak_idxs if 0 <= p < n}), dtype=int)
    for p in peaks:
        windows[max(0, p - peak_pad):min(n, p + peak_pad + 1)] = lo

    out = np.zeros(n, dtype=float)
    for w in np.unique(windows):
        w = int(w)
        if w < polyorder + 2 or w > n:
            continue
        sel = windows == w
        out[sel] = savgol_filter(flow, w, polyorder)[sel]

    out = apply_peak_lift(out, flow, peaks, lift_pad=lift_pad)

    if return_windows:
        return out, windows
    return out


__all__ = [
    "VariableSavgolResult",
    "seed_curvature",
    "savgol_curvature",
    "savgol_slope",
    "apply_peak_lift",
    "variable_savgol_smooth",
]
