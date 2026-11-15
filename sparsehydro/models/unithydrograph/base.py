"""Shared machinery for unit hydrograph (UH) components.

A unit hydrograph here is defined by one function: the cumulative distribution
``F(t)`` of its unit-area instantaneous response, with ``t`` in **hours**
(:meth:`UnitHydrographBase._cdf`). Everything else -- discretisation,
truncation, an optional delay, parameter registration, the prepare/predict
lifecycle and convolution -- lives here once.

**Kernels do not depend on the time step.** The ordinate for step ``j`` is the
exact bin average ``(F((j+1)dt) - F(j dt)) / dt`` [1/hr], so ``sum(k) * dt == 1``
and a model gives the same physical response whether it is run at 5 minutes or
1 hour. Convolving rainfall depth per step with the kernel gives the
instantaneous outflow at the end of each step for rain falling uniformly within
each step, which is why a 5-minute run sampled at hour boundaries equals the
hourly run.

**Prediction**::

    Q = A * convolve(rain_depth_per_step, kernel)

``Q`` is in (rain depth units) per hour times ``A``. With rain in inches and
``A`` in cfs per (in/hr), ``Q`` is in cfs and ``A`` is about 1.0083 x the
contributing area in acres. The flow volume ``sum(Q) * dt`` equals
``A * sum(rain)`` for any ``dt``.

**The predict inversion.** ``predict_flow()`` is the numeric implementation,
``predict_arrays()`` wraps it in a column mapping and ``predict()`` builds a
DataFrame from that, so the array fast path used in calibration cannot drift
from the DataFrame path.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from ...enums import ModelState
from ...parameters import FieldRecord, ScalarParameter
from ..base import IUnitHydroComponent
from ..convolution import convolve_causal
from .kernels import (
    MAX_DURATION_HOURS,
    MAX_ORDINATES,
    TAIL_EPS,
    infer_dt_hours_from_values,
    trim_pad,
)

#: Default bounds for a response delay ``td`` [hr].
DELAY_BOUNDS: tuple[float, float] = (0.0, 48.0)

#: Default bounds for a generic amplitude ``A``.
AMPLITUDE_BOUNDS: tuple[float, float] = (0.0, 1e4)


def reject_step_kwargs(cls_name: str, legacy: dict[str, Any], renames: dict[str, str]) -> None:
    """Raise a helpful ``TypeError`` for keyword arguments from the step-based API.

    Earlier unit hydrographs took their time parameters in time steps under
    different names. Passing an old name now fails loudly instead of being
    silently read in hours.

    :param cls_name: Class name for the message.
    :param legacy: Unexpected keyword arguments received by the constructor.
    :param renames: Old name -> description of the replacement.
    :raises TypeError: Always, when *legacy* is not empty.
    """
    if not legacy:
        return
    parts = []
    for name in legacy:
        hint = renames.get(name)
        parts.append(f"{name!r} -> {hint}" if hint else repr(name))
    raise TypeError(
        f"{cls_name}() got unexpected keyword argument(s): {', '.join(parts)}. "
        "Unit hydrograph time parameters are now in hours, not time steps."
    )


class UnitHydrographBase(IUnitHydroComponent, ABC):
    """Base class for unit hydrographs defined by a cumulative response ``F(t)``.

    Subclasses implement:

    * :meth:`_cdf` -- the cumulative unit response for ``t >= 0`` [hr];
    * :meth:`_register_shape_parameters` -- register the shape parameters;
    * optionally :meth:`_support_hours` (finite support or a fast quantile),
      :meth:`_constraint_residuals` with matching ``ConstraintRecord`` s
      registered in :meth:`_register_shape_parameters`, and :meth:`_extra_valid`.

    :param A: Amplitude: flow per unit rainfall rate. ``Q = A * conv(rain, k)``.
    :param delay: Optional pure delay ``td`` [hr]. ``None`` (default) means no
        ``td`` parameter is registered.
    :param tail_eps: Tail mass left out when truncating an infinite response.
    :param max_duration_hours: Longest response kept [hr].

    :cvar _amplitude_param_name: Name of the amplitude parameter.
    :cvar _amplitude_is_fraction: ``True`` when the amplitude is a fraction in
        ``[0, 1]`` (``RTKTriangle.R``) rather than a flow scale.
    :cvar _rain_column: Forcing column read by :meth:`prepare`.
    :cvar _output_column: Name of the predicted-flow column.
    :cvar _finite_support: ``True`` when ``F`` reaches exactly 1 at
        :meth:`_support_hours`, so no renormalisation is needed.
    """

    _amplitude_param_name: ClassVar[str | None] = "A"
    _amplitude_is_fraction: ClassVar[bool] = False
    _rain_column: ClassVar[str] = "rain"
    _output_column: ClassVar[str] = "Q_pred"
    _finite_support: ClassVar[bool] = False
    supports_array_predict: ClassVar[bool] = True

    def __init__(
        self,
        A: float = 1.0,
        *,
        delay: float | None = None,
        tail_eps: float = TAIL_EPS,
        max_duration_hours: float = MAX_DURATION_HOURS,
    ) -> None:
        super().__init__()
        if not 0.0 < tail_eps < 1.0:
            raise ValueError(f"tail_eps must be in (0, 1); got {tail_eps!r}")
        if not max_duration_hours > 0.0:
            raise ValueError(f"max_duration_hours must be > 0; got {max_duration_hours!r}")
        self._A_init = float(A)
        self._delay_init = None if delay is None else float(delay)
        self.tail_eps = float(tail_eps)
        self.max_duration_hours = float(max_duration_hours)
        self._rain: np.ndarray | None = None
        self._datetimes: np.ndarray | None = None
        self._dt_hours: float = 1.0 / 12.0
        self._n: int = 0

    # ------------------------------------------------------------------
    # Shape definition (subclass hooks)
    # ------------------------------------------------------------------

    @abstractmethod
    def _cdf(self, t: np.ndarray) -> np.ndarray:
        """Cumulative unit response at times ``t >= 0`` [hr], undelayed.

        Must be non-decreasing with ``F(0) = 0``. It need not reach exactly 1
        for infinite-support shapes; the base class truncates and renormalises.
        """

    @abstractmethod
    def _register_shape_parameters(self) -> None:
        """Register shape parameters (and any constraint records)."""

    def _ppf(self, q: float) -> float:
        """Quantile of the undelayed response; override for a closed form.

        The default searches numerically with :meth:`_cdf`.
        """
        target = q
        hi = 1.0
        while hi < self.max_duration_hours and float(self._cdf(np.array([hi]))[0]) < target:
            hi *= 2.0
        if hi >= self.max_duration_hours:
            return self.max_duration_hours
        lo = hi / 2.0 if hi > 1.0 else 0.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if float(self._cdf(np.array([mid]))[0]) < target:
                lo = mid
            else:
                hi = mid
        return hi

    def _support_hours(self) -> float:
        """End of the retained undelayed response [hr]."""
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            t = self._ppf(1.0 - self.tail_eps)
        if not np.isfinite(t) or t <= 0.0:
            return self.max_duration_hours
        return min(float(t), self.max_duration_hours)

    def _constraint_residuals(self) -> list[float]:
        """Inequality residuals ``g`` (feasible when ``g <= 0``); default none."""
        return []

    def _extra_valid(self) -> bool:
        """Structural checks beyond bounds and constraints; default passes."""
        return True

    # ------------------------------------------------------------------
    # Parameter access
    # ------------------------------------------------------------------

    def _p(self, name: str) -> float:
        """Return a registered scalar parameter's value (no membership check)."""
        return self._scalar_parameters[name].value

    @property
    def delay_hours(self) -> float:
        """Current delay ``td`` [hr]; 0 when no delay parameter is registered."""
        param = self._scalar_parameters.get("td")
        return max(float(param.value), 0.0) if param is not None else 0.0

    def _register_amplitude(self) -> None:
        lo, hi = AMPLITUDE_BOUNDS
        self.register_scalar_parameter(ScalarParameter(
            "A", value=self._A_init, lower_bound=lo, upper_bound=hi, units="flow/(depth/hr)",
            description="Amplitude: flow per unit rainfall rate (cfs per in/hr ~ 1.0083 x acres)",
        ))

    def _register_delay(self) -> None:
        lo, hi = DELAY_BOUNDS
        self.register_scalar_parameter(ScalarParameter(
            "td", value=self._delay_init, lower_bound=lo, upper_bound=hi, units="hr",
            description="Pure delay before the response starts",
        ))

    # ------------------------------------------------------------------
    # Continuous response (also used by MixtureUH)
    # ------------------------------------------------------------------

    def support_hours(self) -> float:
        """End of the retained response including any delay [hr]."""
        return self.delay_hours + self._support_hours()

    def cdf(self, t: np.ndarray) -> np.ndarray:
        """Cumulative retained response at times ``t`` [hr], including the delay.

        Equals 0 before the delay and exactly 1 at :meth:`support_hours`.
        """
        t = np.asarray(t, dtype=float)
        T = self._support_hours()
        x = np.clip(t - self.delay_hours, 0.0, T)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            F = np.asarray(self._cdf(x), dtype=float)
            if not self._finite_support:
                total = float(self._cdf(np.array([T]))[0])
                if total > 0.0 and np.isfinite(total):
                    F = F / total
        return F

    # ------------------------------------------------------------------
    # Kernel
    # ------------------------------------------------------------------

    def natural_length(self, dt_hours: float) -> int:
        """Number of ordinates covering the retained response at *dt_hours*."""
        n = math.ceil(self.support_hours() / dt_hours) + 1
        return int(min(max(n, 1), MAX_ORDINATES))

    def _ordinates(self, dt_hours: float, limit: int | None) -> np.ndarray:
        """Kernel ordinates over the natural support, cut at *limit* if given."""
        if not dt_hours > 0.0:
            raise ValueError(f"dt_hours must be > 0; got {dt_hours!r}")
        n = self.natural_length(dt_hours)
        if limit is not None:
            n = max(min(n, int(limit)), 0)
        edges = np.arange(n + 1, dtype=float) * dt_hours
        F = self.cdf(edges)
        k = (F[1:] - F[:-1]) / dt_hours
        if not np.isfinite(k.sum()):
            k = np.nan_to_num(k, nan=0.0, posinf=0.0, neginf=0.0)
        if k.size and k.min() < 0.0:
            k = np.maximum(k, 0.0)
        return k

    def get_kernel(self, dt_hours: float, n_steps: int | None = None) -> np.ndarray:
        """Return unit-area kernel ordinates [1/hr] for time step *dt_hours*.

        :param dt_hours: Time step [hr].
        :param n_steps: Output length (trimmed or zero-padded); defaults to the
            natural support.
        :returns: Ordinates with ``sum * dt_hours == 1`` over the natural support.
        """
        return trim_pad(self._ordinates(dt_hours, n_steps), n_steps)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Register amplitude, shape parameters, optional delay and outputs."""
        self._reset_registries()
        if self._amplitude_param_name == "A":
            self._register_amplitude()
        self._register_shape_parameters()
        if self._delay_init is not None:
            self._register_delay()
        self.register_output_field(FieldRecord("datetime", description="Simulation time step"))
        self.register_output_field(FieldRecord(
            self._output_column, description="Predicted flow", calibratable=True,
        ))
        self._state = ModelState.INITIALIZED

    def inequality_constraints(self) -> list[float]:
        """Residuals of the registered constraints (feasible when ``<= 0``)."""
        return list(self._constraint_residuals())

    def validate(self) -> bool:
        """Check bounds, constraints and structural validity."""
        ok = (self.parameters_valid()
              and all(g <= 0.0 for g in self._constraint_residuals())
              and self._extra_valid())
        if ok:
            self._state = ModelState.VALIDATED
        return ok

    def prepare(self, data: pd.DataFrame, *, dt_hours: float | None = None) -> None:
        """Cache the forcing series as contiguous arrays.

        :param data: DataFrame with ``datetime`` and the rainfall column.
        :param dt_hours: Known time step [hr]; inferred from ``datetime`` when
            omitted.
        :raises KeyError: If the rainfall column is absent.
        """
        rain = np.ascontiguousarray(data[self._rain_column].to_numpy(dtype=float))
        datetimes = pd.to_datetime(data["datetime"]).to_numpy()
        self.prepare_arrays(rain, datetimes, dt_hours=dt_hours)

    def prepare_arrays(
        self,
        rain: np.ndarray,
        datetimes: np.ndarray | None = None,
        *,
        dt_hours: float | None = None,
    ) -> None:
        """Cache forcing arrays directly, bypassing pandas.

        The time step is re-inferred unless *dt_hours* is given or *datetimes*
        is the very same array object as the cached one.
        """
        self._rain = np.ascontiguousarray(rain, dtype=float)
        self._n = int(self._rain.shape[0])
        if dt_hours is not None:
            self._dt_hours = float(dt_hours)
        if datetimes is None:
            self._datetimes = None
        else:
            arr = np.asarray(datetimes)
            same_object = self._datetimes is arr
            self._datetimes = arr
            if dt_hours is None and not same_object:
                self._dt_hours = infer_dt_hours_from_values(arr)
        self._state = ModelState.PREPARED

    def set_forcing(self, rain: np.ndarray) -> None:
        """Replace the rainfall forcing, keeping the cached time base and ``dt``.

        :raises RuntimeError: If called before :meth:`prepare`.
        :raises ValueError: If *rain* has a different length.
        """
        if self._rain is None:
            raise RuntimeError("Call prepare() before set_forcing().")
        arr = np.ascontiguousarray(rain, dtype=float)
        if arr.shape[0] != self._n:
            raise ValueError(
                f"set_forcing() length {arr.shape[0]} does not match the "
                f"prepared series length {self._n}."
            )
        self._rain = arr
        self._state = ModelState.PREPARED

    @property
    def input_field_names(self) -> list[str]:
        """The forcing column, which a composite can bind to an upstream output."""
        return [self._rain_column]

    def set_input(self, name: str, values: np.ndarray) -> None:
        """Replace the forcing (the only bindable input) after :meth:`prepare`."""
        if name != self._rain_column:
            raise KeyError(f"{type(self).__name__} input is {self._rain_column!r}; got {name!r}")
        self.set_forcing(values)

    def amplitude(self) -> float:
        """Current amplitude (1.0 when the class has none)."""
        return self._p(self._amplitude_param_name) if self._amplitude_param_name else 1.0

    def predict_flow(self) -> np.ndarray:
        """Return ``A * convolve(rain, kernel)`` over the prepared series.

        :raises RuntimeError: If :meth:`prepare` has not been called.
        """
        if self._rain is None:
            raise RuntimeError("Call prepare() before predict().")
        kernel = self._ordinates(self._dt_hours, self._n)
        q = self.amplitude() * convolve_causal(self._rain, kernel, n_out=self._n)
        self._state = ModelState.PREDICTED
        return q

    def predict_arrays(self) -> dict[str, np.ndarray]:
        """Return the prediction as plain arrays (``datetime`` when cached)."""
        q = self.predict_flow()
        if self._datetimes is None:
            return {self._output_column: q}
        return {"datetime": self._datetimes, self._output_column: q}

    def predict(self) -> pd.DataFrame:
        """Return ``pd.DataFrame(self.predict_arrays())``."""
        return pd.DataFrame(self.predict_arrays())

    def finalize(self) -> None:
        """Release cached forcing arrays."""
        self._rain = None
        self._datetimes = None
        self._n = 0
        self._state = ModelState.FINALIZED

    # ------------------------------------------------------------------
    # Backwards compatibility
    # ------------------------------------------------------------------

    @property
    def _data(self) -> pd.DataFrame | None:
        """Forcing frame rebuilt on access (for code that read ``model._data``)."""
        if self._rain is None:
            return None
        cols: dict[str, Any] = {}
        if self._datetimes is not None:
            cols["datetime"] = self._datetimes
        cols[self._rain_column] = self._rain
        return pd.DataFrame(cols)


__all__ = ["UnitHydrographBase", "reject_step_kwargs", "DELAY_BOUNDS", "AMPLITUDE_BOUNDS"]
