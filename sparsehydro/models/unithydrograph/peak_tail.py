"""Peak + tail unit hydrograph: a triangular peak blended with a gamma tail.

The Parsimonious Functions "ensemble" unit hydrograph as one component::

    u(t) = A [ (1 - w) triangle(t - td) + w gamma(t - td) ]

Both parts have unit area, so ``w`` is the fraction of the response volume in
the recession tail. It is a :class:`~.mixture.MixtureUH` of a
:class:`~.models.TriangleUH` (prefix ``peak_``) and a
:class:`~.models.GammaUH` (prefix ``tail_``) sharing one delay ``td``.
"""

from __future__ import annotations

from typing import ClassVar

from ...registry import registry
from .base import reject_step_kwargs
from .mixture import MixtureUH
from .models import GammaUH, TriangleUH

__all__ = ["PeakTailUH"]


@registry.register
class PeakTailUH(MixtureUH):
    """Blended triangular peak and gamma tail.

    Parameters (all times in hours): ``A``, ``w`` (tail fraction, [0, 1]),
    ``td`` (shared delay), ``peak_Tp``, ``peak_Tb`` (triangle; constraint
    ``peak_Tp_lt_Tb``), ``tail_alpha``, ``tail_theta`` (gamma).

    :param A: Amplitude.
    :param w: Tail fraction of the response volume.
    :param td: Shared delay [hr].
    :param peak_Tp: Triangle time to peak [hr].
    :param peak_Tb: Triangle base time [hr].
    :param tail_alpha: Gamma shape.
    :param tail_theta: Gamma scale [hr].
    """

    model_name: ClassVar[str] = "peak-tail-uh"

    def __init__(self, A: float = 1.0, w: float = 0.5, td: float = 0.25,
                 peak_Tp: float = 1.0, peak_Tb: float = 3.0,
                 tail_alpha: float = 2.0, tail_theta: float = 2.0, **legacy) -> None:
        reject_step_kwargs("PeakTailUH", legacy, {
            "peak_tp": "peak_Tp [hr]", "peak_tt": "peak_Tb [hr]",
            "tail_tt": "tail_alpha (= tail_tt + 1)", "tail_tp": "tail_theta [hr]"})
        super().__init__(
            [TriangleUH(Tp=peak_Tp, Tb=peak_Tb), GammaUH(alpha=tail_alpha, theta=tail_theta)],
            weights=[1.0 - float(w), float(w)], A=A,
            prefixes=["peak_", "tail_"], weight_names=["w"], delay=td)
