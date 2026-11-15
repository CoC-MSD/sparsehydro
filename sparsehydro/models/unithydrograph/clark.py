"""Clark and ModClark unit hydrographs: translation plus linear-reservoir storage.

Clark separates **translation** (a time-area curve: the fraction of the basin
contributing by travel time ``tau``, scaled to the time of concentration ``Tc``)
from **attenuation** (routing through one linear reservoir with storage
coefficient ``Rs``).

With ``A(t)`` the cumulative time-area curve and ``a(t) = A'(t)``, the routed
cumulative response is::

    F(t) = A(t) - I(t),    I(t) = integral_0^min(t, Tc) a(s) exp(-(t - s)/Rs) ds

The time-area curve is piecewise linear, so ``I(t)`` has a closed form on each
segment and ``F`` is evaluated exactly at any time (no sub-stepping). As
``Rs -> 0`` the response tends to the time-area curve itself. The mean response
time is the time-area mean plus ``Rs``.

**ModClark** (HEC-HMS's distributed variant) assigns each grid cell a travel
time ``Tc * d_cell / d_max``. With rainfall lumped over the basin, as here, it
reduces to Clark with the cells' area-weighted travel-time histogram as the
time-area curve; :class:`ModClarkUH` builds that histogram from cell data.
Spatially distributed rainfall is not modelled.
"""

from __future__ import annotations

from typing import ClassVar, Sequence

import numpy as np

from ...parameters import ScalarParameter
from ...registry import registry
from .base import UnitHydrographBase

__all__ = ["ClarkUH", "ModClarkUH", "hec_time_area"]

#: Number of equal segments used to represent the HEC-HMS synthetic curve.
_HEC_SEGMENTS = 64


def hec_time_area(n_segments: int = _HEC_SEGMENTS) -> tuple[np.ndarray, np.ndarray]:
    """HEC-HMS default dimensionless time-area curve.

    ``A = 1.414 tau^1.5`` for ``tau <= 0.5`` and ``1 - 1.414 (1 - tau)^1.5``
    above, on ``n_segments`` equal steps of ``tau`` in ``[0, 1]``.

    :returns: ``(tau, cumulative_area)``, both starting at 0 and ending at 1.
    """
    tau = np.linspace(0.0, 1.0, n_segments + 1)
    cum = np.where(tau <= 0.5, 1.414 * tau ** 1.5, 1.0 - 1.414 * (1.0 - tau) ** 1.5)
    cum[0], cum[-1] = 0.0, 1.0
    return tau, np.clip(cum, 0.0, 1.0)


def _validated_curve(tau: Sequence[float], cum: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    tau = np.asarray(tau, dtype=float)
    cum = np.asarray(cum, dtype=float)
    if tau.ndim != 1 or tau.shape != cum.shape or tau.size < 2:
        raise ValueError("time-area curve needs matching 1-D tau and area arrays (>= 2 points)")
    if not (np.all(np.isfinite(tau)) and np.all(np.isfinite(cum))):
        raise ValueError("time-area curve must be finite")
    if tau[0] != 0.0 or tau[-1] != 1.0 or np.any(np.diff(tau) <= 0.0):
        raise ValueError("time-area tau must increase strictly from 0 to 1")
    if cum[0] != 0.0 or cum[-1] != 1.0 or np.any(np.diff(cum) < 0.0):
        raise ValueError("cumulative time-area must be non-decreasing from 0 to 1")
    return tau, cum


@registry.register
class ClarkUH(UnitHydrographBase):
    """Clark unit hydrograph: time-area translation routed through one reservoir.

    :param A: Amplitude.
    :param Tc: Time of concentration [hr]. Bounds [0.05, 240].
    :param Rs: Storage coefficient of the linear reservoir [hr]. Bounds [0.001, 480].
    :param time_area: Optional dimensionless curve ``(tau, cumulative_area)`` with
        ``tau`` from 0 to 1 and area from 0 to 1. Defaults to the HEC-HMS curve.
    :param delay: Optional delay ``td`` [hr].
    """

    model_name: ClassVar[str] = "clark-uh"

    def __init__(self, A: float = 1.0, Tc: float = 1.0, Rs: float = 1.0, *,
                 time_area: tuple[Sequence[float], Sequence[float]] | None = None,
                 delay: float | None = None) -> None:
        super().__init__(A, delay=delay)
        self._Tc_init, self._Rs_init = float(Tc), float(Rs)
        self._tau, self._cum = (hec_time_area() if time_area is None
                                else _validated_curve(*time_area))

    @property
    def time_area(self) -> tuple[np.ndarray, np.ndarray]:
        """Dimensionless ``(tau, cumulative_area)`` curve (copies)."""
        return self._tau.copy(), self._cum.copy()

    def _register_shape_parameters(self) -> None:
        self.register_scalar_parameter(ScalarParameter(
            "Tc", value=self._Tc_init, lower_bound=0.05, upper_bound=240.0, units="hr",
            description="Time of concentration"))
        self.register_scalar_parameter(ScalarParameter(
            "Rs", value=self._Rs_init, lower_bound=1e-3, upper_bound=480.0, units="hr",
            description="Clark storage coefficient"))

    def _support_hours(self) -> float:
        # Beyond Tc the response is an exponential recession: F = 1 - I(Tc) e^{-(t-Tc)/Rs}.
        tc, rs = self._p("Tc"), max(self._p("Rs"), 1e-12)
        return min(tc + rs * np.log(1.0 / self.tail_eps), self.max_duration_hours)

    def _cdf(self, t):
        t = np.asarray(t, dtype=float)
        tc, rs = self._p("Tc"), max(self._p("Rs"), 1e-12)
        s = self._tau * tc                      # segment edges [hr]
        area = np.interp(np.minimum(t, tc), s, self._cum)
        density = np.diff(self._cum) / np.diff(s)  # per segment [1/hr]
        I = np.zeros_like(t)

        inside = t < tc
        if inside.any():
            ti = t[inside][:, None]
            lo = s[:-1][None, :]
            hi = np.minimum(s[1:][None, :], ti)
            active = ti > lo
            with np.errstate(over="ignore", invalid="ignore"):
                part = rs * (np.exp(-(ti - hi) / rs) - np.exp(-(ti - lo) / rs))
            I[inside] = np.where(active, density[None, :] * part, 0.0).sum(axis=1)

        after = ~inside
        if after.any():
            ends = s[1:]
            i_tc = float(np.sum(density * rs * (np.exp(-(tc - ends) / rs)
                                                - np.exp(-(tc - s[:-1]) / rs))))
            I[after] = i_tc * np.exp(-(t[after] - tc) / rs)

        return np.clip(area - I, 0.0, 1.0)


@registry.register
class ModClarkUH(ClarkUH):
    """ModClark with lumped rainfall: Clark with a time-area curve from cell data.

    Each cell's travel time is ``Tc * travel_index / max(travel_index)``, where
    ``travel_index`` is the cell's travel distance (or travel time) to the
    outlet. The time-area curve is the area-weighted histogram of those travel
    times over ``n_bins`` equal bins.

    :param A: Amplitude.
    :param Tc: Time of concentration [hr].
    :param Rs: Storage coefficient [hr].
    :param travel_index: Per-cell travel distance or time (any positive units).
        ``None`` falls back to the HEC-HMS curve.
    :param weights: Per-cell area weights (default equal).
    :param n_bins: Histogram bins for the time-area curve.
    :param delay: Optional delay ``td`` [hr].
    """

    model_name: ClassVar[str] = "modclark-uh"

    def __init__(self, A: float = 1.0, Tc: float = 1.0, Rs: float = 1.0, *,
                 travel_index: Sequence[float] | None = None,
                 weights: Sequence[float] | None = None, n_bins: int = 50,
                 delay: float | None = None) -> None:
        time_area = None
        if travel_index is not None:
            time_area = self.time_area_from_cells(travel_index, weights, n_bins)
        super().__init__(A, Tc, Rs, time_area=time_area, delay=delay)

    @staticmethod
    def time_area_from_cells(travel_index: Sequence[float], weights: Sequence[float] | None = None,
                             n_bins: int = 50) -> tuple[np.ndarray, np.ndarray]:
        """Build a dimensionless time-area curve from per-cell travel indices.

        :raises ValueError: On empty, negative, non-finite or zero-weight input.
        """
        d = np.asarray(travel_index, dtype=float)
        w = np.ones_like(d) if weights is None else np.asarray(weights, dtype=float)
        if d.ndim != 1 or d.size == 0 or w.shape != d.shape:
            raise ValueError("travel_index and weights must be matching non-empty 1-D arrays")
        if not (np.all(np.isfinite(d)) and np.all(np.isfinite(w))) or d.min() < 0 or w.min() < 0:
            raise ValueError("travel_index and weights must be finite and non-negative")
        if d.max() <= 0.0 or w.sum() <= 0.0:
            raise ValueError("travel_index needs a positive maximum and weights a positive sum")
        if int(n_bins) < 1:
            raise ValueError("n_bins must be >= 1")
        hist, edges = np.histogram(d / d.max(), bins=int(n_bins), range=(0.0, 1.0), weights=w)
        cum = np.concatenate([[0.0], np.cumsum(hist) / hist.sum()])
        cum[-1] = 1.0
        return edges, cum
