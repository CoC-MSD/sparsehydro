"""Distribution-shaped unit hydrographs: Weibull, lognormal, generalized gamma.

Each response is a probability density on ``t > 0`` [hr]; the kernel is the
exact bin average of its CDF (see :mod:`.base`).

* :class:`WeibullUH` -- similar flexibility to the gamma with a different
  (stretched-exponential) tail.
* :class:`LognormalUH` -- strongly right-skewed responses with long recessions.
* :class:`GeneralizedGammaUH` -- three parameters; contains the gamma
  (``c = 1``) and Weibull (``a = 1``) as special cases.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from scipy.special import gammainc, gammaincinv, ndtr, ndtri

from ...parameters import ScalarParameter
from ...registry import registry
from .base import UnitHydrographBase
from .models import SHAPE_BOUNDS, TIME_SCALE_BOUNDS

__all__ = ["WeibullUH", "LognormalUH", "GeneralizedGammaUH"]


@registry.register
class WeibullUH(UnitHydrographBase):
    """Weibull response: ``F(t) = 1 - exp(-(t/lam)^k)``.

    ``k = 1`` is a single linear reservoir; ``k > 1`` has a rising limb.

    :param A: Amplitude.
    :param k: Shape [-]. Bounds [0.2, 10].
    :param lam: Scale [hr]. Bounds [0.01, 240].
    :param delay: Optional delay ``td`` [hr].
    """

    model_name: ClassVar[str] = "weibull-uh"

    def __init__(self, A: float = 1.0, k: float = 1.5, lam: float = 1.0, *,
                 delay: float | None = None) -> None:
        super().__init__(A, delay=delay)
        self._k_init, self._lam_init = float(k), float(lam)

    def _register_shape_parameters(self) -> None:
        self.register_scalar_parameter(ScalarParameter(
            "k", value=self._k_init, lower_bound=0.2, upper_bound=10.0, description="Weibull shape"))
        self.register_scalar_parameter(ScalarParameter(
            "lam", value=self._lam_init, lower_bound=TIME_SCALE_BOUNDS[0],
            upper_bound=TIME_SCALE_BOUNDS[1], units="hr", description="Weibull scale"))

    def _cdf(self, t):
        x = np.asarray(t, dtype=float) / self._p("lam")
        return -np.expm1(-(x ** self._p("k")))

    def _ppf(self, q):
        return float(self._p("lam") * (-np.log1p(-q)) ** (1.0 / self._p("k")))


@registry.register
class LognormalUH(UnitHydrographBase):
    """Lognormal response: ``ln t ~ Normal(mu, sigma)``.

    Median ``exp(mu)`` [hr]; mode ``exp(mu - sigma^2)``. Heavy tails with
    ``sigma`` near its upper bound are truncated at ``MAX_DURATION_HOURS``.

    :param A: Amplitude.
    :param mu: Mean of ``ln t`` (``t`` in hours). Bounds [-4.6, 5.5].
    :param sigma: Standard deviation of ``ln t``. Bounds [0.05, 2].
    :param delay: Optional delay ``td`` [hr].
    """

    model_name: ClassVar[str] = "lognormal-uh"

    def __init__(self, A: float = 1.0, mu: float = 0.0, sigma: float = 0.5, *,
                 delay: float | None = None) -> None:
        super().__init__(A, delay=delay)
        self._mu_init, self._sigma_init = float(mu), float(sigma)

    def _register_shape_parameters(self) -> None:
        self.register_scalar_parameter(ScalarParameter(
            "mu", value=self._mu_init, lower_bound=-4.6, upper_bound=5.5, units="ln(hr)",
            description="Mean of log time"))
        self.register_scalar_parameter(ScalarParameter(
            "sigma", value=self._sigma_init, lower_bound=0.05, upper_bound=2.0,
            description="Standard deviation of log time"))

    def _cdf(self, t):
        t = np.asarray(t, dtype=float)
        with np.errstate(divide="ignore"):
            z = (np.log(t) - self._p("mu")) / self._p("sigma")
        return ndtr(z)

    def _ppf(self, q):
        return float(np.exp(self._p("mu") + self._p("sigma") * ndtri(q)))


@registry.register
class GeneralizedGammaUH(UnitHydrographBase):
    """Generalized gamma (Stacy): ``F(t) = P(a, (t/scale)^c)``.

    ``c = 1`` gives :class:`~.models.GammaUH` (``alpha = a``, ``theta = scale``);
    ``a = 1`` gives :class:`WeibullUH` (``k = c``, ``lam = scale``). Matches
    ``scipy.stats.gengamma(a, c, scale=scale)``.

    :param A: Amplitude.
    :param a: First shape [-]. Bounds [0.1, 50].
    :param c: Power shape [-]. Bounds [0.2, 10].
    :param scale: Scale [hr]. Bounds [0.01, 240].
    :param delay: Optional delay ``td`` [hr].
    """

    model_name: ClassVar[str] = "gengamma-uh"

    def __init__(self, A: float = 1.0, a: float = 2.0, c: float = 1.0, scale: float = 0.5, *,
                 delay: float | None = None) -> None:
        super().__init__(A, delay=delay)
        self._a_init, self._c_init, self._scale_init = float(a), float(c), float(scale)

    def _register_shape_parameters(self) -> None:
        self.register_scalar_parameter(ScalarParameter(
            "a", value=self._a_init, lower_bound=SHAPE_BOUNDS[0], upper_bound=SHAPE_BOUNDS[1],
            description="Generalized gamma shape a"))
        self.register_scalar_parameter(ScalarParameter(
            "c", value=self._c_init, lower_bound=0.2, upper_bound=10.0,
            description="Generalized gamma power c"))
        self.register_scalar_parameter(ScalarParameter(
            "scale", value=self._scale_init, lower_bound=TIME_SCALE_BOUNDS[0],
            upper_bound=TIME_SCALE_BOUNDS[1], units="hr", description="Scale"))

    def _cdf(self, t):
        x = np.asarray(t, dtype=float) / self._p("scale")
        return gammainc(self._p("a"), x ** self._p("c"))

    def _ppf(self, q):
        return float(self._p("scale") * gammaincinv(self._p("a"), q) ** (1.0 / self._p("c")))
