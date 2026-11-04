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
import pandas as pd

from ...enums import ModelState
from ..base import IUnitHydroComponent
from ...parameters import FieldRecord, ScalarParameter
from ..convolution import convolve_causal

#: Deprecated.  Convolution algorithm selection moved to
#: :mod:`sparsehydro.models.convolution`, which switches on the cost ``n * m``
#: rather than ``max(n, m)``.  Retained so existing imports keep resolving.
_FFT_THRESHOLD = 500


class RTKTriangle(IUnitHydroComponent):
    """Triangular RTK unit hydrograph component.

    :param R: Fraction of rainfall excess entering the sewer [0, 1].
    :type R: float
    :param T: Time to peak [hr] (must be > 0).
    :type T: float
    :param K: Ratio of recession time to time-to-peak (must be > 1).
    :type K: float
    """

    model_name = "rtk-triangle"
    _amplitude_param_name = "R"

    def __init__(
        self,
        R: float = 0.05,
        T: float = 1.0,
        K: float = 1.5,
    ) -> None:
        super().__init__()
        if float(R) < 0.0 or float(R) > 1.0:
            raise ValueError(f"R must be in [0, 1]; got {R!r}")
        if float(T) <= 0.0:
            raise ValueError(f"T must be > 0; got {T!r}")
        if float(K) <= 1.0:
            raise ValueError(f"K must be > 1.0; got {K!r}")
        self.R = float(R)
        self.T = float(T)
        self.K = float(K)
        self._prepared_df: pd.DataFrame | None = None
        self._dt_hours: float = 1.0

    # ------------------------------------------------------------------
    # Physics properties
    # ------------------------------------------------------------------

    @property
    def base_duration(self) -> float:
        """Total triangle duration in hours: T * (1 + K).

        :returns: Base duration ``T * (1 + K)`` [hr].
        :rtype: float
        """
        return self.T * (1.0 + self.K)

    @property
    def peak_ordinate(self) -> float:
        """Peak ordinate value 2 / (T * (1 + K)), ensuring unit area.

        :returns: Peak ordinate ``2 / (T * (1 + K))`` [1/hr].
        :rtype: float
        """
        return 2.0 / (self.T * (1.0 + self.K))

    # ------------------------------------------------------------------
    # IModel lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Register R, T, K as scalar parameters and advance to INITIALIZED.

        :returns: Nothing.
        :rtype: None
        """
        self.register_scalar_parameter(ScalarParameter(
            "R", value=self.R, lower_bound=0.0, upper_bound=1.0,
            units="-", description="Fraction of rainfall excess entering the sewer",
        ))
        self.register_scalar_parameter(ScalarParameter(
            "T", value=self.T, lower_bound=0.1, upper_bound=240.0,
            units="hr", description="Time to peak",
        ))
        self.register_scalar_parameter(ScalarParameter(
            "K", value=self.K, lower_bound=1.001, upper_bound=10.0,
            units="-", description="Recession-to-peak ratio",
        ))
        self.register_output_field(FieldRecord(name="datetime", description="Simulation time step"))
        self.register_output_field(FieldRecord(
            name="rdii_mm",
            units="mm",
            description="RDII depth contribution from this RTK triangle component",
            calibratable=False,
        ))
        self._state = ModelState.INITIALIZED

    def validate(self) -> bool:
        """Validate parameter bounds and advance to VALIDATED.

        :returns: ``True`` if all parameters are within bounds.
        :rtype: bool
        """
        if not self.parameters_valid():
            return False
        self._state = ModelState.VALIDATED
        return True

    def prepare(self, data: pd.DataFrame) -> None:
        """Load rainfall-excess forcing data and infer time step.

        :param data: DataFrame with columns ``datetime`` and ``p_excess_mm``.
        :type data: pandas.DataFrame
        :returns: Nothing.
        :rtype: None
        :raises ValueError: If required columns are absent.
        """
        required = {"datetime", "p_excess_mm"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(
                f"prepare() data is missing required columns: {sorted(missing)}"
            )
        df = data.sort_values("datetime").reset_index(drop=True).copy()
        diffs = df["datetime"].diff().dropna()
        self._dt_hours = pd.Timedelta(diffs.median()).total_seconds() / 3600.0
        self._prepared_df = df
        self._sync_from_params()
        self._state = ModelState.PREPARED

    def predict(self) -> pd.DataFrame:
        """Convolve rainfall excess with this triangle's UH kernel and apply R-scaling.

        :returns: DataFrame with columns ``datetime`` and ``rdii_mm``.
        :rtype: pandas.DataFrame
        :raises RuntimeError: If :meth:`prepare` has not been called.
        """
        if self._prepared_df is None:
            raise RuntimeError("Call prepare(data) before predict().")

        self._sync_from_params()
        p_excess = self._prepared_df["p_excess_mm"].to_numpy(dtype=float)
        kernel = triangular_uh(self, self._dt_hours)
        n = len(p_excess)

        conv = convolve_causal(p_excess, kernel, n_out=n)

        rdii = np.clip(self.R * conv, 0.0, None)
        result = pd.DataFrame({
            "datetime": self._prepared_df["datetime"].values,
            "rdii_mm": rdii,
        })
        self._state = ModelState.PREDICTED
        return result

    def finalize(self) -> None:
        """Release stored forcing data and advance to FINALIZED.

        :returns: Nothing.
        :rtype: None
        """
        self._prepared_df = None
        self._state = ModelState.FINALIZED

    def get_kernel(
        self,
        dt_hours: float,
        n_steps: int | None = None,
    ) -> np.ndarray:
        """Return the normalized triangular UH ordinate array.

        The R-scaling is intentionally **not** applied here; the caller
        (:class:`~sparsehydro.models.rdii.RDIIModel`) applies its own ``R_i`` fraction.

        :param dt_hours: Time-step size [hr].
        :type dt_hours: float
        :param n_steps: Number of output steps; defaults to the natural support.
        :type n_steps: int | None
        :returns: 1-D array of normalized UH ordinates [1/hr].
        :rtype: numpy.ndarray
        """
        self._sync_from_params()
        return triangular_uh(self, dt_hours, n_steps)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sync_from_params(self) -> None:
        """Update instance attributes from the scalar-parameter registry.

        :returns: Nothing.
        :rtype: None
        """
        self.R = self.get_scalar_parameter("R").value
        self.T = self.get_scalar_parameter("T").value
        self.K = self.get_scalar_parameter("K").value

    def __repr__(self) -> str:
        return f"RTKTriangle(R={self.R!r}, T={self.T!r}, K={self.K!r})"


# ---------------------------------------------------------------------------
# Triangle math helpers
# ---------------------------------------------------------------------------

def _triangle_value(t: float, T: float, K: float) -> float:
    """Evaluate the triangular UH function at continuous time t.

    Scalar reference for the piecewise-linear ordinate; retained for clarity
    and testing.  The hot path (:func:`triangular_uh`) uses the vectorized
    closed-form integral in :func:`_triangle_cumint` instead.

    :param t: Continuous time [hr].
    :type t: float
    :param T: Time to peak [hr].
    :type T: float
    :param K: Recession-to-peak ratio.
    :type K: float
    :returns: UH ordinate value at time *t* [1/hr].
    :rtype: float
    """
    base = T * (1.0 + K)
    peak = 2.0 / base
    if t <= 0.0:
        return 0.0
    if t <= T:
        return peak * (t / T)
    if t <= base:
        return peak * (base - t) / (base - T)
    return 0.0


def _integrate_triangle(t0: float, t1: float, T: float, K: float) -> float:
    """Integrate the triangular UH exactly over [t0, t1], handling breakpoints.

    Scalar reference (trapezoid split at the ``T`` / ``base`` breakpoints).
    Equivalent to differencing :func:`_triangle_cumint`; retained for clarity
    and testing.

    :param t0: Interval start time [hr].
    :type t0: float
    :param t1: Interval end time [hr].
    :type t1: float
    :param T: Time to peak [hr].
    :type T: float
    :param K: Recession-to-peak ratio.
    :type K: float
    :returns: Integral of the triangle over ``[t0, t1]``.
    :rtype: float
    """
    base = T * (1.0 + K)
    breaks = [t0]
    for bp in (T, base):
        if t0 < bp < t1:
            breaks.append(bp)
    breaks.append(t1)
    breaks.sort()

    total = 0.0
    for a, b in zip(breaks[:-1], breaks[1:]):
        fa = _triangle_value(a, T, K)
        fb = _triangle_value(b, T, K)
        total += 0.5 * (fa + fb) * (b - a)
    return total


def _triangle_cumint(t: np.ndarray, T: float, K: float) -> np.ndarray:
    """Cumulative integral ``F(t) = ∫₀ᵗ f(s) ds`` of the unit-area triangle.

    ``f`` is the piecewise-linear triangle with peak ``2/base`` at ``t = T`` and
    zero outside ``[0, base]`` (``base = T·(1+K)``).  Its running integral is a
    piecewise **quadratic** with the closed form::

        t ≤ 0        : 0
        0 < t ≤ T    : t² / (base·T)
        T < t ≤ base : T/base + (2/base)/(base−T) · (base·(t−T) − (t²−T²)/2)
        t > base     : 1

    Differencing ``F`` at adjacent bin edges gives each bin's exact integral, so
    this is analytically identical to the trapezoid-with-breakpoints reference
    (:func:`_integrate_triangle`) to floating-point rounding, but evaluated in a
    single vectorized pass.  Inputs are assumed pre-clipped to ``[0, base]``.

    ``base - T = T·K`` is never zero: :class:`RTKTriangle` rejects ``K <= 1.0``
    and ``T <= 0`` at construction.

    :param t: Times [hr], clipped to ``[0, base]``.
    :type t: numpy.ndarray
    :param T: Time to peak [hr].
    :type T: float
    :param K: Recession-to-peak ratio.
    :type K: float
    :returns: ``F(t)`` evaluated elementwise.
    :rtype: numpy.ndarray
    """
    base = T * (1.0 + K)
    F = np.empty_like(t)
    rising = t <= T
    tr = t[rising]
    F[rising] = tr * tr / (base * T)
    falling = ~rising
    tf = t[falling]
    F[falling] = T / base + (2.0 / base) / (base - T) * (
        base * (tf - T) - (tf * tf - T * T) / 2.0
    )
    return F


def triangular_uh(
    triangle: RTKTriangle,
    dt_hours: float,
    n_steps: int | None = None,
) -> np.ndarray:
    """Compute the unit hydrograph ordinate array for one RTK triangle.

    Each element represents the average flow rate [1/hr] over its interval,
    computed by analytically integrating the piecewise-linear triangle.  The
    per-bin integrals are obtained by differencing the closed-form cumulative
    integral (:func:`_triangle_cumint`) at the bin edges — a single vectorized
    numpy pass rather than a Python loop over every step, which matters because
    the calibrator rebuilds this kernel on every ``predict()`` and a slow
    seasonal triangle (large ``T·(1+K)``) can span hundreds of thousands of
    steps.

    The array satisfies::

        np.sum(ordinates) * dt_hours ≈ 1.0   (within 1e-10 relative error)

    :param triangle: RTK triangle parameters.
    :type triangle: RTKTriangle
    :param dt_hours: Time-step size [hr].
    :type dt_hours: float
    :param n_steps: Number of output steps.  Defaults to natural support.
    :type n_steps: int | None
    :returns: 1-D array of UH ordinates [1/hr].
    :rtype: numpy.ndarray
    :raises ValueError: If ``dt_hours <= 0``.
    """
    if dt_hours <= 0.0:
        raise ValueError(f"dt_hours must be > 0; got {dt_hours!r}")

    base = triangle.base_duration
    if n_steps is None:
        n_steps = math.ceil(base / dt_hours) + 1

    # Bin edges 0, dt, 2·dt, …, n_steps·dt, clipped to the triangle's support so
    # bins beyond `base` contribute exactly zero (F(base) − F(base) = 0), matching
    # the reference loop's early `break`.
    edges = np.clip(np.arange(n_steps + 1, dtype=float) * dt_hours, 0.0, base)
    F = _triangle_cumint(edges, triangle.T, triangle.K)
    return (F[1:] - F[:-1]) / dt_hours


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
