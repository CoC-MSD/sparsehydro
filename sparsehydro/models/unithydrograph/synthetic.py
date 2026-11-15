"""Synthetic unit hydrographs: SCS/NRCS dimensionless and Snyder.

Both are represented by a gamma-shaped response whose mode is the time to peak
``Tp`` and whose peak ordinate matches the method's peak-rate factor.

Peak-rate factor (PRF)
    In US units a unit hydrograph peak is ``qp = PRF * A_mi2 * Q_in / Tp``
    [cfs]. Converting to a unit-area response [1/hr], ``qp * Tp = PRF / 645.33``
    (645.33 = cubic feet in one square-mile-inch / 3600). The standard SCS value
    is 484, i.e. a dimensionless peak of 0.75.

Gamma equivalent
    A gamma density with shape ``m + 1`` and scale ``Tp / m`` peaks at ``Tp``
    with ``qp * Tp = m^(m+1) e^(-m) / Gamma(m + 1)``. Solving this for ``m``
    gives the gamma that reproduces a PRF; PRF 484 gives ``m ~ 3.7``, the
    well-known gamma approximation of the SCS dimensionless unit hydrograph.

Snyder
    Snyder's peak is ``qp = 640 * Cp * A_mi2 / tL`` [cfs/in], i.e. a PRF of
    ``640 * Cp`` with the peak at the basin lag ``tL``. This module matches that
    peak time and peak rate with the gamma shape above. Snyder's W50/W75 widths
    are not matched separately.
"""

from __future__ import annotations

from functools import lru_cache
from typing import ClassVar

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammainc, gammaincinv, gammaln

from ...parameters import ScalarParameter
from ...registry import registry
from .base import UnitHydrographBase

__all__ = ["SCSUH", "SnyderUH", "gamma_shape_for_prf", "PRF_DIMENSIONLESS_DIVISOR"]

#: ``PRF / PRF_DIMENSIONLESS_DIVISOR`` is the dimensionless peak ``qp * Tp``.
PRF_DIMENSIONLESS_DIVISOR: float = 640.0 * 3630.0 / 3600.0  # = 645.33...


@lru_cache(maxsize=256)
def gamma_shape_for_prf(prf: float) -> float:
    """Return ``m`` such that a gamma with shape ``m + 1``, mode ``Tp`` has the PRF.

    :param prf: Peak-rate factor (484 is the standard SCS value).
    :returns: ``m > 0``.
    :raises ValueError: If *prf* is outside the solvable range.
    """
    target = np.log(prf / PRF_DIMENSIONLESS_DIVISOR)

    def g(m: float) -> float:
        return (m + 1.0) * np.log(m) - m - gammaln(m + 1.0) - target

    lo, hi = 1e-6, 1e6
    if not (g(lo) < 0.0 < g(hi)):
        raise ValueError(f"peak-rate factor {prf!r} is outside the solvable range")
    return float(brentq(g, lo, hi, xtol=1e-12, rtol=1e-12, maxiter=200))


class _PeakRateGammaUH(UnitHydrographBase):
    """Gamma response with mode ``Tp`` and a given peak-rate factor."""

    def _shape_scale(self) -> tuple[float, float]:
        m = gamma_shape_for_prf(float(self._prf()))
        return m + 1.0, self._tp() / m

    def _tp(self) -> float:
        raise NotImplementedError

    def _prf(self) -> float:
        raise NotImplementedError

    def _cdf(self, t):
        shape, scale = self._shape_scale()
        return gammainc(shape, np.asarray(t, dtype=float) / scale)

    def _ppf(self, q):
        shape, scale = self._shape_scale()
        return float(gammaincinv(shape, q) * scale)


@registry.register
class SCSUH(_PeakRateGammaUH):
    """SCS/NRCS dimensionless unit hydrograph (gamma form).

    ``Tp`` is the time to peak of the instantaneous response. For a basin with
    time of concentration ``Tc``, the usual estimate is ``Tp ~ 0.6 Tc``.

    :param A: Amplitude.
    :param Tp: Time to peak [hr]. Bounds [0.05, 240].
    :param PRF: Peak-rate factor. Default 484 and not calibrated; flat or swampy
        basins use lower values (e.g. 300), steep basins up to about 600.
    :param delay: Optional delay ``td`` [hr].
    """

    model_name: ClassVar[str] = "scs-uh"

    def __init__(self, A: float = 1.0, Tp: float = 1.0, PRF: float = 484.0, *,
                 delay: float | None = None) -> None:
        super().__init__(A, delay=delay)
        self._Tp_init, self._PRF_init = float(Tp), float(PRF)

    def _register_shape_parameters(self) -> None:
        self.register_scalar_parameter(ScalarParameter(
            "Tp", value=self._Tp_init, lower_bound=0.05, upper_bound=240.0, units="hr",
            description="Time to peak"))
        self.register_scalar_parameter(ScalarParameter(
            "PRF", value=self._PRF_init, lower_bound=100.0, upper_bound=600.0,
            description="Peak-rate factor (484 standard)", calibrate=False))

    def _tp(self) -> float:
        return self._p("Tp")

    def _prf(self) -> float:
        return self._p("PRF")


@registry.register
class SnyderUH(_PeakRateGammaUH):
    """Snyder synthetic unit hydrograph (gamma form matched to peak time and rate).

    :param A: Amplitude.
    :param tL: Basin lag, used as the time to peak [hr]. Bounds [0.05, 240].
    :param Cp: Peaking coefficient [-]. Bounds [0.3, 0.9375] (PRF = 640 Cp <= 600).
    :param delay: Optional delay ``td`` [hr].
    """

    model_name: ClassVar[str] = "snyder-uh"

    def __init__(self, A: float = 1.0, tL: float = 1.0, Cp: float = 0.6, *,
                 delay: float | None = None) -> None:
        super().__init__(A, delay=delay)
        self._tL_init, self._Cp_init = float(tL), float(Cp)

    def _register_shape_parameters(self) -> None:
        self.register_scalar_parameter(ScalarParameter(
            "tL", value=self._tL_init, lower_bound=0.05, upper_bound=240.0, units="hr",
            description="Basin lag (time to peak)"))
        self.register_scalar_parameter(ScalarParameter(
            "Cp", value=self._Cp_init, lower_bound=0.3, upper_bound=0.9375,
            description="Snyder peaking coefficient"))

    def _tp(self) -> float:
        return self._p("tL")

    def _prf(self) -> float:
        return 640.0 * self._p("Cp")
