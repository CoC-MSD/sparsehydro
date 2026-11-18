"""Savitzky-Golay smoothing and derivative signals for event detection."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


@dataclass
class FilterResult:
    """Output of :func:`apply_savgol_filter`.

    :ivar datetime: Timestamps aligned to the stormflow input.
    :vartype datetime: pandas.Series
    :ivar raw_flow: Original stormflow values (clipped to ≥ 0).
    :vartype raw_flow: numpy.ndarray
    :ivar sg_0: Smoothed signal (Savitzky-Golay 0th-order output).
    :vartype sg_0: numpy.ndarray
    :ivar sg_1: First derivative of the smoothed signal (slope).
    :vartype sg_1: numpy.ndarray
    :ivar sg_2: Second derivative of the smoothed signal (curvature).
    :vartype sg_2: numpy.ndarray
    :ivar thresholds: Populated by :func:`compute_thresholds`.  Keys:
        ``"sg_0_th"``, ``"sg_1_th"``, ``"sg_2_th"``.
    :vartype thresholds: dict[str, float]
    """

    datetime: pd.Series
    raw_flow: np.ndarray
    sg_0: np.ndarray
    sg_1: np.ndarray
    sg_2: np.ndarray
    thresholds: dict[str, float] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame with columns datetime, raw_flow, sg_0, sg_1, sg_2.

        :returns: Tidy frame combining the timestamps and signal arrays.
        :rtype: pandas.DataFrame
        """
        return pd.DataFrame(
            {
                "datetime": self.datetime,
                "raw_flow": self.raw_flow,
                "sg_0": self.sg_0,
                "sg_1": self.sg_1,
                "sg_2": self.sg_2,
            }
        )


def _get_safe_window(window: float, n: int) -> int:
    """Round *window* to nearest odd integer, capped at *n*, minimum 3.

    :param window: Desired window length (may be fractional).
    :type window: float
    :param n: Number of available samples (upper bound for the window).
    :type n: int
    :returns: Odd window length in ``[3, n]``.
    :rtype: int
    """
    val = int(round(window))
    val = min(val, n)
    val = max(val, 3)
    if val % 2 == 0:
        val = max(3, val - 1)
    return val


def apply_savgol_filter(
    rain_stormflow: pd.DataFrame,
    window_length: int = 48,
    polyorder: int = 2,
    deriv_factor: float = 1.2,
) -> FilterResult:
    """Apply Savitzky-Golay smoothing and compute derivative signals.

    :param rain_stormflow: Must contain columns ``datetime``, ``rain``,
        ``stormflow``.
    :type rain_stormflow: pandas.DataFrame
    :param window_length: Base window for the sg_0 smoother.  Windows for sg_1
        and sg_2 are scaled by successive powers of *deriv_factor*.
    :type window_length: int
    :param polyorder: Polynomial order for all filter passes.
    :type polyorder: int
    :param deriv_factor: Multiplier applied to the window length for each
        derivative pass.
    :type deriv_factor: float
    :returns: Arrays of the same length as *rain_stormflow*; ``thresholds`` is
        empty until :func:`compute_thresholds` is called.
    :rtype: FilterResult
    """
    df = rain_stormflow.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    stormflow = df["stormflow"].values.astype(float)
    stormflow = np.maximum(stormflow, 0.0)
    n = len(stormflow)

    min_allowed = polyorder + 1 if (polyorder + 1) % 2 != 0 else polyorder + 2

    if n < min_allowed:
        return FilterResult(
            datetime=df["datetime"].reset_index(drop=True),
            raw_flow=stormflow,
            sg_0=stormflow.copy(),
            sg_1=np.zeros(n),
            sg_2=np.zeros(n),
        )

    w0 = _get_safe_window(window_length, n)
    w1 = _get_safe_window(window_length * deriv_factor, n)
    w2 = _get_safe_window(window_length * (deriv_factor**2), n)

    sg_0 = savgol_filter(stormflow, window_length=w0, polyorder=polyorder, mode="interp")
    sg_0 = np.maximum(sg_0, 0.0)
    sg_1 = savgol_filter(sg_0, window_length=w1, polyorder=polyorder, deriv=1, mode="interp")
    sg_2 = savgol_filter(sg_0, window_length=w2, polyorder=polyorder, deriv=2, mode="interp")

    return FilterResult(
        datetime=df["datetime"].reset_index(drop=True),
        raw_flow=stormflow,
        sg_0=sg_0,
        sg_1=sg_1,
        sg_2=sg_2,
    )


def compute_thresholds(
    result: FilterResult,
    sg_0_per: float = 0.15,
    sg_1_per: float = 0.25,
    sg_2_per: float = 0.25,
) -> FilterResult:
    """Attach percentile-based detection thresholds to *result*.

    :param result: Output of :func:`apply_savgol_filter`.
    :type result: FilterResult
    :param sg_0_per: ``sg_0_th = 95th-percentile(sg_0[sg_0 > 0]) * sg_0_per``.
    :type sg_0_per: float
    :param sg_1_per: ``sg_1_th = std(sg_1) * sg_1_per``.
    :type sg_1_per: float
    :param sg_2_per: ``sg_2_th = std(sg_2) * sg_2_per``.
    :type sg_2_per: float
    :returns: New :class:`FilterResult` with ``thresholds`` populated (immutable
        update — the original is not modified).
    :rtype: FilterResult
    """
    sg_0_positive = result.sg_0[result.sg_0 > 0]
    if len(sg_0_positive) == 0:
        sg_0_th = 0.0
    else:
        sg_0_th = float(np.percentile(sg_0_positive, 95) * sg_0_per)

    sg_1_th = float(np.std(result.sg_1) * sg_1_per)
    sg_2_th = float(np.std(result.sg_2) * sg_2_per)

    return FilterResult(
        datetime=result.datetime,
        raw_flow=result.raw_flow,
        sg_0=result.sg_0,
        sg_1=result.sg_1,
        sg_2=result.sg_2,
        thresholds={
            "sg_0_th": sg_0_th,
            "sg_1_th": sg_1_th,
            "sg_2_th": sg_2_th,
        },
    )


__all__ = ["FilterResult", "apply_savgol_filter", "compute_thresholds"]
