"""sparsehydro.models.unithydrograph -- unit hydrograph library.

Every unit hydrograph is defined by its cumulative unit response ``F(t)`` with
``t`` in **hours**; kernels are exact bin averages, so results do not depend on
the model time step. All classes are registered by name (``registry.create``):

* :class:`NashUH` (``nash-uh``): ``n``, ``K`` [hr]
* :class:`GammaUH` (``gamma-uh``): ``alpha``, ``theta`` [hr]
* :class:`GammaDelayUH` (``gamma-delay-uh``): ``alpha``, ``theta``, ``td`` [hr]
* :class:`SCSUH` (``scs-uh``): ``Tp`` [hr], ``PRF`` (fixed at 484)
* :class:`SnyderUH` (``snyder-uh``): ``tL`` [hr], ``Cp``
* :class:`ClarkUH` (``clark-uh``): ``Tc`` [hr], ``Rs`` [hr]
* :class:`ModClarkUH` (``modclark-uh``): ``Tc``, ``Rs`` plus cell travel data
* :class:`WeibullUH` (``weibull-uh``): ``k``, ``lam`` [hr]
* :class:`LognormalUH` (``lognormal-uh``): ``mu``, ``sigma``
* :class:`GeneralizedGammaUH` (``gengamma-uh``): ``a``, ``c``, ``scale`` [hr]
* :class:`MixtureUH` (``mixture-uh``): weights plus prefixed component parameters
* :class:`PeakTailUH` (``peak-tail-uh``): ``w``, ``peak_Tp``, ``peak_Tb``,
  ``tail_alpha``, ``tail_theta``
* :class:`TriangleUH` (``triangle-uh``): ``Tp``, ``Tb`` [hr]
* :class:`RectangleUH` (``rectangle-uh``): ``Tr`` [hr]
* :class:`DecayUH` (``decay-uh``): ``K`` [hr]
* :class:`RTKTriangle` (``rtk-triangle``): ``R`` (fraction), ``T`` [hr], ``K``

Every class except ``RTKTriangle`` also has an amplitude ``A`` and accepts an
optional delay ``td`` (``delay=`` in the constructor).

Standalone prediction is ``Q = A * convolve(rain_depth_per_step, kernel)``.
Inside :class:`~sparsehydro.models.rdii.RDIIModel`, each component's kernel is
scaled by that model's ``R_i`` instead of ``A``.

Quick start::

    from sparsehydro.models.unithydrograph import NashUH
    uh = NashUH(A=20.0, n=3, K=0.5)
    uh.initialize()
    uh.prepare(df)            # columns: datetime, rain
    q = uh.predict()["Q_pred"]
"""

from .base import UnitHydrographBase
from .clark import ClarkUH, ModClarkUH
from .distributions import GeneralizedGammaUH, LognormalUH, WeibullUH
from .mixture import MixtureUH
from .models import DecayUH, GammaDelayUH, GammaUH, NashUH, RectangleUH, TriangleUH
from .peak_tail import PeakTailUH
from .rtk import RTKTriangle, triangular_uh
from .synthetic import SCSUH, SnyderUH

__all__ = [
    "UnitHydrographBase",
    "NashUH",
    "GammaUH",
    "GammaDelayUH",
    "SCSUH",
    "SnyderUH",
    "ClarkUH",
    "ModClarkUH",
    "WeibullUH",
    "LognormalUH",
    "GeneralizedGammaUH",
    "MixtureUH",
    "PeakTailUH",
    "TriangleUH",
    "RectangleUH",
    "DecayUH",
    "RTKTriangle",
    "triangular_uh",
]
