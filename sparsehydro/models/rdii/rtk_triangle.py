"""Triangular RTK unit hydrograph and RTK ensemble model.

:class:`RTKTriangle` defines a single triangular RTK unit hydrograph component.
:class:`RTKEnsembleModel` builds an additive ensemble of :class:`RDIIModel` instances,
one per RDII pathway (fast / medium / slow), each with its own :class:`IAModel`.

RTK triangle shape
------------------
- **R** — fraction of rainfall excess entering the sewer (dimensionless).
- **T** — time to peak [hours].
- **K** — ratio of recession time to time-to-peak (must be > 1).

The unit hydrograph is piecewise-linear:

- Rising limb: linear from 0 to peak over [0, T].
- Falling limb: linear from peak to 0 over [T, T*(1+K)].
- Peak ordinate = 2 / (T * (1+K)), ensuring area under curve equals 1.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..unithydrograph.rtk import (  # noqa: F401 - re-exported for existing imports
    RTKTriangle,
    _integrate_triangle,
    _triangle_cumint,
    _triangle_value,
    triangular_uh,
)

#: Deprecated.  Convolution algorithm selection moved to
#: :mod:`sparsehydro.models.convolution`, which switches on the cost ``n * m``
#: rather than ``max(n, m)``.  Retained so existing imports keep resolving.
_FFT_THRESHOLD = 500


# ---------------------------------------------------------------------------
# RTK Ensemble helpers
# ---------------------------------------------------------------------------

_FAST_MEDIUM_SLOW = ["fast", "medium", "slow"]


def default_rtk_params(
    n_models: int,
    n_triangles: int = 1,
    T_min: float = 0.5,
    T_max: float = 120.0,
    R_total: float = 0.10,
    K_min: float = 1.5,
    K_max: float = 3.0,
) -> list[list[tuple[float, float, float]]]:
    """Generate initial ``(R, T, K)`` tuples grouped by model.

    Returns a list of *n_models* sub-lists, each containing *n_triangles*
    ``(R, T, K)`` tuples distributed across the parameter space:

    * **T** — log-spaced between *T_min* and *T_max*.
    * **K** — linearly interpolated from *K_min* (fastest) to *K_max* (slowest).
    * **R** — ``R_total / (n_models × n_triangles)`` for every triangle.

    :param n_models: Number of RDIIModel instances (>= 1).
    :type n_models: int
    :param n_triangles: RTK triangles per model (>= 1).
    :type n_triangles: int
    :param T_min: Time-to-peak for the fastest triangle [hr].
    :type T_min: float
    :param T_max: Time-to-peak for the slowest triangle [hr].
    :type T_max: float
    :param R_total: Total runoff fraction divided equally across all triangles.
    :type R_total: float
    :param K_min: Recession ratio for the fastest triangle.
    :type K_min: float
    :param K_max: Recession ratio for the slowest triangle.
    :type K_max: float
    :returns: Nested list ``[[( R, T, K), ...], ...]``.
    :rtype: list[list[tuple[float, float, float]]]
    :raises ValueError: If any count or bound argument is invalid.
    """
    if n_models < 1:
        raise ValueError(f"n_models must be >= 1; got {n_models!r}")
    if n_triangles < 1:
        raise ValueError(f"n_triangles must be >= 1; got {n_triangles!r}")
    if T_min <= 0.0:
        raise ValueError(f"T_min must be > 0; got {T_min!r}")
    if T_max <= T_min:
        raise ValueError(f"T_max must be > T_min; got T_min={T_min!r}, T_max={T_max!r}")
    if R_total <= 0.0:
        raise ValueError(f"R_total must be > 0; got {R_total!r}")

    n_total = n_models * n_triangles
    R_each = R_total / n_total

    if n_total == 1:
        T_vals = [math.sqrt(T_min * T_max)]
        K_vals = [(K_min + K_max) / 2.0]
    else:
        T_vals = list(np.logspace(math.log10(T_min), math.log10(T_max), n_total))
        K_vals = list(np.linspace(K_min, K_max, n_total))

    flat = [(R_each, float(T), float(K)) for T, K in zip(T_vals, K_vals)]
    return [flat[i * n_triangles:(i + 1) * n_triangles] for i in range(n_models)]


def _make_aliases(n_models: int) -> list[str]:
    if n_models == 3:
        return list(_FAST_MEDIUM_SLOW)
    return [f"rtk_{i}" for i in range(1, n_models + 1)]


class RTKEnsembleModel:
    """Additive ensemble of :class:`RDIIModel` instances — one per RDII pathway.

    In SWMM each RTK pathway (fast / medium / slow) has its own independent
    initial-abstraction model.  ``RTKEnsembleModel.create()`` wires up
    *n_models* :class:`~sparsehydro.models.rdii.RDIIModel` instances — each
    carrying its own :class:`~sparsehydro.models.rdii.IAModel` and *n_triangles*
    :class:`RTKTriangle` objects — into a single additive
    :class:`~sparsehydro.models.EnsembleModel`.

    Use :meth:`create` rather than the constructor::

        ensemble = RTKEnsembleModel.create(
            n_models=3,
            n_triangles=3,
            units="imperial",
            area_acres=500.0,
        )
        ensemble.validate()
        ensemble.prepare(df)
        result = ensemble.predict()
        ensemble.finalize()
    """

    model_name = "rtk-ensemble"

    @classmethod
    def create(
        cls,
        n_models: int = 3,
        n_triangles: int = 1,
        units: str = "imperial",
        area_acres: float = 100.0,
        ia_defaults: dict | None = None,
        rtk_defaults: list[list[tuple[float, float, float]]] | None = None,
        aliases: list[str] | None = None,
        output_name: str = "rdii_cfs",
    ) -> "Any":  # returns EnsembleModel
        """Build an additive RTK ensemble.

        :param n_models: Number of RDIIModel instances (>= 1).  Defaults to 3.
        :type n_models: int
        :param n_triangles: RTK triangles inside each RDIIModel (>= 1).  Defaults to 1.
        :type n_triangles: int
        :param units: Unit system — ``"imperial"`` or ``"metric"``.
        :type units: str
        :param area_acres: Drainage area shared by all components [acres].
            Frozen and excluded from calibration.
        :type area_acres: float
        :param ia_defaults: Keyword arguments forwarded to every
            :class:`~sparsehydro.models.rdii.IAModel` constructor.
        :type ia_defaults: dict | None
        :param rtk_defaults: Nested initial ``(R, T, K)`` values.
            If omitted, :func:`default_rtk_params` is called.
        :type rtk_defaults: list[list[tuple[float, float, float]]] | None
        :param aliases: Label per component.  Defaults to ``["fast", "medium", "slow"]``
            for n_models=3, else ``["rtk_1", …, "rtk_N"]``.
        :type aliases: list[str] | None
        :param output_name: Column name for the combined RDII signal.
        :type output_name: str
        :returns: Fully initialised :class:`~sparsehydro.models.EnsembleModel`.
        :rtype: EnsembleModel
        :raises ValueError: If counts or default lengths are inconsistent.
        """
        from .model import RDIIModel
        from ..ensemble import EnsembleModel

        if n_models < 1:
            raise ValueError(f"n_models must be >= 1; got {n_models!r}")
        if n_triangles < 1:
            raise ValueError(f"n_triangles must be >= 1; got {n_triangles!r}")

        if rtk_defaults is None:
            rtk_defaults = default_rtk_params(n_models, n_triangles)
        if len(rtk_defaults) != n_models:
            raise ValueError(
                f"rtk_defaults length ({len(rtk_defaults)}) must equal n_models ({n_models})"
            )
        for i, group in enumerate(rtk_defaults):
            if len(group) != n_triangles:
                raise ValueError(
                    f"rtk_defaults[{i}] length ({len(group)}) must equal "
                    f"n_triangles ({n_triangles})"
                )

        if aliases is None:
            aliases = _make_aliases(n_models)
        if len(aliases) != n_models:
            raise ValueError(
                f"aliases length ({len(aliases)}) must equal n_models ({n_models})"
            )

        ia_kwargs: dict = ia_defaults or {}
        children: list[RDIIModel] = []

        for model_rtk in rtk_defaults:
            child = RDIIModel(
                ia_model=IAModel(units=units, **ia_kwargs),
                uh_components=[RTKTriangle(R=R, T=T, K=K) for R, T, K in model_rtk],
                units=units,
            )
            child.initialize()
            child.get_scalar_parameter("area_acres").update(
                value=area_acres, calibrate=False
            )
            children.append(child)

        extractor = lambda df: df["rdii_cfs"].to_numpy()  # noqa: E731
        ensemble = EnsembleModel(
            components=[(child, extractor) for child in children],
            mode="sum",
            aliases=aliases,
            output_name=output_name,
            normalize_weights=False,
        )
        ensemble.initialize()

        for i in range(1, n_models + 1):
            ensemble.set_parameter(f"w_{i}", value=1.0, calibrate=False)

        return ensemble


# Lazy import to avoid circular at module level
from .initial_abstraction import IAModel  # noqa: E402
