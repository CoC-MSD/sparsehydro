"""Classic unit hydrograph shapes: gamma / Nash, triangle, rectangle, decay.

All time parameters are in hours and kernels are exact bin averages of the
continuous response (see :mod:`.base`). Each class is registered by name.

* ``NashUH`` (``nash-uh``): ``F = P(n, t/K)``, the regularised lower gamma.
* ``GammaUH`` (``gamma-uh``): ``F = P(alpha, t/theta)``.
* ``GammaDelayUH`` (``gamma-delay-uh``): ``GammaUH`` shifted by ``td``.
* ``TriangleUH`` (``triangle-uh``): piecewise quadratic ``F``, peak at ``Tp``,
  ending at ``Tb``.
* ``RectangleUH`` (``rectangle-uh``): ``F = min(t/Tr, 1)``.
* ``DecayUH`` (``decay-uh``): ``F = 1 - exp(-t/K)``, one linear reservoir.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from scipy.special import gammainc, gammaincinv

from ...parameters import ConstraintRecord, ScalarParameter
from ...registry import registry
from .base import UnitHydrographBase, reject_step_kwargs
from .kernels import (  # noqa: F401 - re-exported for backwards compatibility
    MAX_STEPS as _MAX_STEPS,
    infer_dt_hours as _infer_dt_hours,
    normalize_kernel as _normalize_kernel,
    trim_pad as _trim_pad,
)

__all__ = ["NashUH", "GammaUH", "GammaDelayUH", "TriangleUH", "RectangleUH", "DecayUH"]

#: Bounds shared by time scales [hr].
TIME_SCALE_BOUNDS: tuple[float, float] = (0.01, 240.0)
#: Bounds shared by gamma-type shape parameters.
SHAPE_BOUNDS: tuple[float, float] = (0.1, 50.0)


def _gamma_cdf(t: np.ndarray, shape: float, scale: float) -> np.ndarray:
    return gammainc(shape, np.asarray(t, dtype=float) / scale)


def _gamma_ppf(q: float, shape: float, scale: float) -> float:
    return float(gammaincinv(shape, q) * scale)


# ---------------------------------------------------------------------------
# Gamma family
# ---------------------------------------------------------------------------

@registry.register
class NashUH(UnitHydrographBase):
    """Nash cascade: ``n`` equal linear reservoirs with storage constant ``K``.

    The instantaneous response is the gamma density with shape ``n`` and scale
    ``K``; ``n`` may be non-integer. Mean ``n K``, variance ``n K^2``, peak at
    ``(n - 1) K`` for ``n >= 1``.

    :param A: Amplitude (see :class:`~.base.UnitHydrographBase`).
    :param n: Number of reservoirs [-]. Bounds [0.1, 50].
    :param K: Storage constant of each reservoir [hr]. Bounds [0.01, 240].
    :param delay: Optional delay ``td`` [hr].
    """

    model_name: ClassVar[str] = "nash-uh"

    def __init__(self, A: float = 1.0, n: float = 3.0, K: float = 0.5, *,
                 delay: float | None = None, **legacy) -> None:
        reject_step_kwargs("NashUH", legacy, {"k": "K [hr] (= k_steps * dt_hours)"})
        super().__init__(A, delay=delay)
        self._n_init, self._K_init = float(n), float(K)

    def _register_shape_parameters(self) -> None:
        self.register_scalar_parameter(ScalarParameter(
            "n", value=self._n_init, lower_bound=SHAPE_BOUNDS[0], upper_bound=SHAPE_BOUNDS[1],
            description="Number of linear reservoirs"))
        self.register_scalar_parameter(ScalarParameter(
            "K", value=self._K_init, lower_bound=TIME_SCALE_BOUNDS[0],
            upper_bound=TIME_SCALE_BOUNDS[1], units="hr", description="Reservoir storage constant"))

    def _cdf(self, t):
        return _gamma_cdf(t, self._p("n"), self._p("K"))

    def _ppf(self, q):
        return _gamma_ppf(q, self._p("n"), self._p("K"))


@registry.register
class GammaUH(UnitHydrographBase):
    """Gamma-distribution unit hydrograph (shape ``alpha``, scale ``theta``).

    The same response as :class:`NashUH` with ``n = alpha`` and ``K = theta``;
    provided under the statistical parameter names. Peak at
    ``(alpha - 1) theta`` for ``alpha >= 1``.

    :param A: Amplitude.
    :param alpha: Shape [-]. Bounds [0.1, 50].
    :param theta: Scale [hr]. Bounds [0.01, 240].
    :param delay: Optional delay ``td`` [hr].
    """

    model_name: ClassVar[str] = "gamma-uh"

    def __init__(self, A: float = 1.0, alpha: float = 3.0, theta: float = 0.5, *,
                 delay: float | None = None, **legacy) -> None:
        reject_step_kwargs("GammaUH", legacy, {
            "tt": "alpha (= tt + 1)", "tp": "theta [hr] (= tp_steps * dt_hours)"})
        super().__init__(A, delay=delay)
        self._alpha_init, self._theta_init = float(alpha), float(theta)

    def _register_shape_parameters(self) -> None:
        self.register_scalar_parameter(ScalarParameter(
            "alpha", value=self._alpha_init, lower_bound=SHAPE_BOUNDS[0],
            upper_bound=SHAPE_BOUNDS[1], description="Gamma shape"))
        self.register_scalar_parameter(ScalarParameter(
            "theta", value=self._theta_init, lower_bound=TIME_SCALE_BOUNDS[0],
            upper_bound=TIME_SCALE_BOUNDS[1], units="hr", description="Gamma scale"))

    def _cdf(self, t):
        return _gamma_cdf(t, self._p("alpha"), self._p("theta"))

    def _ppf(self, q):
        return _gamma_ppf(q, self._p("alpha"), self._p("theta"))


@registry.register
class GammaDelayUH(GammaUH):
    """:class:`GammaUH` with a pure delay ``td`` [hr] registered by default."""

    model_name: ClassVar[str] = "gamma-delay-uh"

    def __init__(self, A: float = 1.0, alpha: float = 3.0, theta: float = 0.5,
                 td: float = 0.5, **legacy) -> None:
        reject_step_kwargs("GammaDelayUH", legacy, {
            "tt": "alpha (= tt + 1)", "tp": "theta [hr] (= tp_steps * dt_hours)"})
        super().__init__(A, alpha, theta, delay=td)


# ---------------------------------------------------------------------------
# Triangle, rectangle, decay
# ---------------------------------------------------------------------------

@registry.register
class TriangleUH(UnitHydrographBase):
    """Triangular unit hydrograph rising to a peak at ``Tp`` and ending at ``Tb``.

    Constraint ``Tp_lt_Tb``: ``Tp - Tb <= 0``.

    :param A: Amplitude.
    :param Tp: Time to peak [hr]. Bounds [0.05, 240].
    :param Tb: Base time [hr]. Bounds [0.1, 960].
    :param delay: Optional delay ``td`` [hr].
    """

    model_name: ClassVar[str] = "triangle-uh"
    _finite_support: ClassVar[bool] = True

    def __init__(self, A: float = 1.0, Tp: float = 1.0, Tb: float = 3.0, *,
                 delay: float | None = None, **legacy) -> None:
        reject_step_kwargs("TriangleUH", legacy, {
            "tp": "Tp [hr]", "tt": "Tb [hr] (total duration)"})
        super().__init__(A, delay=delay)
        self._Tp_init, self._Tb_init = float(Tp), float(Tb)

    def _register_shape_parameters(self) -> None:
        self.register_scalar_parameter(ScalarParameter(
            "Tp", value=self._Tp_init, lower_bound=0.05, upper_bound=240.0, units="hr",
            description="Time to peak"))
        self.register_scalar_parameter(ScalarParameter(
            "Tb", value=self._Tb_init, lower_bound=0.1, upper_bound=960.0, units="hr",
            description="Base time (total duration)"))
        self.register_inequality_constraint(ConstraintRecord("Tp_lt_Tb", "Tp - Tb <= 0"))

    def _constraint_residuals(self) -> list[float]:
        return [self._p("Tp") - self._p("Tb")]

    def _shape(self) -> tuple[float, float]:
        tb = max(self._p("Tb"), 1e-9)
        tp = min(max(self._p("Tp"), 0.0), tb)
        return tp, tb

    def _support_hours(self) -> float:
        return min(self._shape()[1], self.max_duration_hours)

    def _cdf(self, t):
        tp, tb = self._shape()
        t = np.clip(np.asarray(t, dtype=float), 0.0, tb)
        F = np.empty_like(t)
        rising = t <= tp
        F[rising] = t[rising] ** 2 / (tb * tp) if tp > 0.0 else 1.0
        fall = ~rising
        F[fall] = 1.0 - (tb - t[fall]) ** 2 / (tb * (tb - tp)) if tb > tp else 1.0
        return F


@registry.register
class RectangleUH(UnitHydrographBase):
    """Uniform response over ``0 < t <= Tr``.

    :param A: Amplitude.
    :param Tr: Duration [hr]. Bounds [0.01, 480].
    :param delay: Optional delay ``td`` [hr].
    """

    model_name: ClassVar[str] = "rectangle-uh"
    _finite_support: ClassVar[bool] = True

    def __init__(self, A: float = 1.0, Tr: float = 1.0, *, delay: float | None = None,
                 **legacy) -> None:
        reject_step_kwargs("RectangleUH", legacy, {"tr": "Tr [hr]"})
        super().__init__(A, delay=delay)
        self._Tr_init = float(Tr)

    def _register_shape_parameters(self) -> None:
        self.register_scalar_parameter(ScalarParameter(
            "Tr", value=self._Tr_init, lower_bound=0.01, upper_bound=480.0, units="hr",
            description="Response duration"))

    def _support_hours(self) -> float:
        return min(max(self._p("Tr"), 1e-9), self.max_duration_hours)

    def _cdf(self, t):
        return np.clip(np.asarray(t, dtype=float) / max(self._p("Tr"), 1e-9), 0.0, 1.0)


@registry.register
class DecayUH(UnitHydrographBase):
    """Exponential decay: a single linear reservoir with storage constant ``K``.

    Equal to :class:`NashUH` with ``n = 1``. The step-based ``alpha`` of earlier
    versions maps to ``K = -dt_hours / ln(alpha)``.

    :param A: Amplitude.
    :param K: Storage constant [hr]. Bounds [0.001, 480].
    :param delay: Optional delay ``td`` [hr].
    """

    model_name: ClassVar[str] = "decay-uh"

    def __init__(self, A: float = 1.0, K: float = 0.25, *, delay: float | None = None,
                 **legacy) -> None:
        reject_step_kwargs("DecayUH", legacy, {"alpha": "K [hr] (= -dt_hours / ln(alpha))"})
        super().__init__(A, delay=delay)
        self._K_init = float(K)

    def _register_shape_parameters(self) -> None:
        self.register_scalar_parameter(ScalarParameter(
            "K", value=self._K_init, lower_bound=1e-3, upper_bound=480.0, units="hr",
            description="Reservoir storage constant"))

    def _cdf(self, t):
        return -np.expm1(-np.asarray(t, dtype=float) / self._p("K"))

    def _ppf(self, q):
        return float(-self._p("K") * np.log1p(-q))
