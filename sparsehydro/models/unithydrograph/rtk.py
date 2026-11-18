"""Triangular RTK unit hydrograph.

- **R** -- fraction of rainfall excess entering the sewer (dimensionless).
- **T** -- time to peak [hours].
- **K** -- ratio of recession time to time-to-peak (must be > 1).

The response rises linearly to a peak of ``2 / (T (1 + K))`` at ``T`` and falls
linearly to zero at ``T (1 + K)``; its area is 1. Kernels are exact bin
integrals of the triangle (:func:`triangular_uh`), so they do not depend on the
time step.

:class:`~sparsehydro.models.rdii.RDIIModel` applies its own ``R_i`` to
:meth:`RTKTriangle.get_kernel`, which is unscaled.
"""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
import pandas as pd

from ...enums import ModelState
from ...parameters import FieldRecord, ScalarParameter
from ...registry import registry
from ..convolution import convolve_causal
from .base import UnitHydrographBase


@registry.register
class RTKTriangle(UnitHydrographBase):
    """Triangular RTK unit hydrograph component.

    :param R: Fraction of rainfall excess entering the sewer [0, 1].
    :type R: float
    :param T: Time to peak [hr] (must be > 0).
    :type T: float
    :param K: Ratio of recession time to time-to-peak (must be > 1).
    :type K: float
    """

    model_name: ClassVar[str] = "rtk-triangle"
    _amplitude_param_name: ClassVar[str | None] = "R"
    _amplitude_is_fraction: ClassVar[bool] = True
    _rain_column: ClassVar[str] = "p_excess_mm"
    _output_column: ClassVar[str] = "rdii_mm"
    _finite_support: ClassVar[bool] = True

    def __init__(
        self,
        R: float = 0.05,
        T: float = 1.0,
        K: float = 1.5,
    ) -> None:
        super().__init__(A=1.0)
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
        self._reset_registries()
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

    def predict_arrays(self) -> dict[str, np.ndarray]:
        """Convolve rainfall excess with this triangle's kernel and apply R.

        :returns: ``{"datetime": ..., "rdii_mm": ...}``.
        :raises RuntimeError: If :meth:`prepare` has not been called.
        """
        if self._prepared_df is None:
            raise RuntimeError("Call prepare(data) before predict().")

        self._sync_from_params()
        p_excess = self._prepared_df["p_excess_mm"].to_numpy(dtype=float)
        kernel = triangular_uh(self, self._dt_hours)
        conv = convolve_causal(p_excess, kernel, n_out=len(p_excess))
        rdii = np.clip(self.R * conv, 0.0, None)
        self._state = ModelState.PREDICTED
        return {"datetime": self._prepared_df["datetime"].values, "rdii_mm": rdii}

    def predict(self) -> pd.DataFrame:
        """Return ``pd.DataFrame(self.predict_arrays())``."""
        return pd.DataFrame(self.predict_arrays())

    def set_forcing(self, rain: np.ndarray) -> None:
        """Replace the rainfall-excess forcing (``p_excess_mm``) after :meth:`prepare`."""
        if self._prepared_df is None:
            raise RuntimeError("Call prepare() before set_forcing().")
        arr = np.asarray(rain, dtype=float)
        if arr.shape[0] != len(self._prepared_df):
            raise ValueError("set_forcing() length does not match the prepared series")
        self._prepared_df = self._prepared_df.assign(p_excess_mm=arr)
        self._state = ModelState.PREPARED

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

    # Continuous response (lets an RTK triangle sit inside a MixtureUH).
    def _register_shape_parameters(self) -> None:  # pragma: no cover - initialize() is overridden
        raise NotImplementedError

    def _support_hours(self) -> float:
        self._sync_from_params()
        return min(self.base_duration, self.max_duration_hours)

    def _cdf(self, t):
        self._sync_from_params()
        base = self.base_duration
        return _triangle_cumint(np.clip(np.asarray(t, dtype=float), 0.0, base), self.T, self.K)

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


__all__ = ["RTKTriangle", "triangular_uh"]
