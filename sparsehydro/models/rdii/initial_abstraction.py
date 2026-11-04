"""Physics-based Initial Abstraction (IA) recovery and depletion model.

Tracks available soil/vadose-zone storage capacity over time.  During dry
intervals the capacity recovers toward ``ia_max`` via an exponential approach
whose rate is temperature-dependent (capturing ET and drainage).  During
rainfall the capacity is depleted exponentially per mm of input.  Recovery is
suppressed entirely below ``T_freeze``, producing emergent seasonal RDII
behaviour from a single parameter set.

Key equations
-------------
Recovery (dry interval Δt, temperature T)::

    k_rec(T) = k0 + kT * exp(θ * (T - T_ref))   if T >= T_freeze
             = 0                                   if T < T_freeze

    IA_avail(t+Δt) = IA_max - (IA_max - IA_avail(t)) * exp(-k_rec(T) * Δt)

Depletion and rainfall excess (rainfall pulse ΔP mm, uniform within the step):

The wet step integrates the depletion ODE ``d(IA)/dp = -k_dep * IA`` exactly
over the step (p = cumulative rain), so results are invariant to sub-step
refinement — equivalent to uniformly disaggregating the step to an arbitrarily
fine timestep::

    IA(p) = IA_avail(t) * exp(-k_dep * p)

    instantaneous excess rate = max(0, 1 - k_dep * IA(p))

    IA_avail(t+Δt) = IA_avail(t) * exp(-k_dep * ΔP)

Optional degree-day snow model (``snow=True``)::

    T <= snow_T :  SWE += ΔP ;  liquid input = 0          (snowfall)
    T >  snow_T :  melt = min(SWE, snow_ddf * (T - snow_T) * Δt_days)
                   SWE -= melt ;  liquid input = ΔP + melt (rain-on-snow adds)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ...enums import ModelState
from ..base import IModel
from ...parameters import FieldRecord, ScalarParameter

# Optional numba acceleration for the hot excess loop.  When numba is importable
# the stateful recurrence in ``compute_excess_series`` is JIT-compiled; otherwise
# the pure-Python reference below runs unchanged.  ``fastmath`` is kept OFF so the
# compiled exp/log stay bit-identical to the reference.
try:
    from numba import njit  # type: ignore[import]

    _HAS_NUMBA = True
except Exception:  # pragma: no cover - depends on optional dependency
    _HAS_NUMBA = False

    def njit(*args, **kwargs):  # no-op fallback: return function undecorated
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]                      # bare @njit

        def _wrap(fn):
            return fn

        return _wrap                            # @njit(...) with options


def _wet_step_excess(ia0: float, k_dep: float, P: float) -> float:
    """Exact rainfall excess for one wet step (uniform intra-step rainfall).

    :param ia0: Available abstraction capacity at the start of the step.
    :type ia0: float
    :param k_dep: Depletion rate constant.
    :type k_dep: float
    :param P: Rainfall depth over the step.
    :type P: float
    :returns: Rainfall excess depth produced during the step.
    :rtype: float
    """
    if P <= 0.0:
        return 0.0
    if k_dep * ia0 <= 1.0:
        return P - ia0 * (1.0 - math.exp(-k_dep * P))
    p_star = math.log(k_dep * ia0) / k_dep
    if p_star >= P:
        return 0.0
    return (P - p_star) - (1.0 / k_dep - ia0 * math.exp(-k_dep * P))


@njit(cache=True, fastmath=False)
def _wet_step_excess_nb(ia0, k_dep, P):
    """numba twin of :func:`_wet_step_excess` (identical arithmetic)."""
    if P <= 0.0:
        return 0.0
    if k_dep * ia0 <= 1.0:
        return P - ia0 * (1.0 - math.exp(-k_dep * P))
    p_star = math.log(k_dep * ia0) / k_dep
    if p_star >= P:
        return 0.0
    return (P - p_star) - (1.0 / k_dep - ia0 * math.exp(-k_dep * P))


@njit(cache=True, fastmath=False)
def _compute_excess_core_nb(rainfall_mm, dt_hours, temperature, use_temp,
                            ia_max, k0, kT, theta, T_ref, k_dep, T_freeze,
                            snow_on, snow_ddf, snow_T):
    """Compiled transcription of the :meth:`IAModel.compute_excess_series` loop.

    ``None`` sentinels (temperature / snow_T) are passed as typed flags
    (``use_temp`` / ``snow_on``) so numba compiles one concrete signature rather
    than specializing on optional types.

    The arithmetic is a line-for-line copy of the pure-Python reference and
    ``fastmath`` is off, so the output is bit-identical to it.
    """
    n = rainfall_mm.shape[0]
    excess = np.zeros(n, dtype=np.float64)
    ia_avail = ia_max
    swe = 0.0
    for i in range(n):
        P = rainfall_mm[i]
        dt = dt_hours[i]
        T = temperature[i] if use_temp else T_ref
        if snow_on:
            if T <= snow_T:
                swe += P
                P = 0.0
            elif swe > 0.0:
                melt = min(swe, snow_ddf * (T - snow_T) * dt / 24.0)
                swe -= melt
                P += melt
        if P <= 0.0:
            if T >= T_freeze:
                k_rec = k0 + kT * math.exp(theta * (T - T_ref))
            else:
                k_rec = 0.0
            deficit = ia_max - ia_avail
            ia_avail = ia_max - deficit * math.exp(-k_rec * dt)
        else:
            excess[i] = _wet_step_excess_nb(ia_avail, k_dep, P)
            ia_avail = ia_avail * math.exp(-k_dep * P)
        if ia_avail < 0.0:
            ia_avail = 0.0
        elif ia_avail > ia_max:
            ia_avail = ia_max
    return excess


def _compute_excess_series_fast(
    rainfall_mm,
    dt_hours,
    temperature,
    ia_max,
    k0,
    kT,
    theta,
    T_ref,
    k_dep,
    T_freeze,
    snow_ddf=0.0,
    snow_T=None,
):
    """numba-backed twin of :meth:`IAModel.compute_excess_series`.

    Same signature and semantics; ``temperature`` / ``snow_T`` may be ``None``.
    Only the array marshalling lives here — the recurrence itself is
    :func:`_compute_excess_core_nb`.

    :returns: Rainfall excess depth series, same length as *rainfall_mm*.
    :rtype: numpy.ndarray
    """
    rainfall_mm = np.ascontiguousarray(rainfall_mm, dtype=np.float64)
    dt_hours = np.ascontiguousarray(dt_hours, dtype=np.float64)
    use_temp = temperature is not None
    temp_arr = (np.ascontiguousarray(temperature, dtype=np.float64)
                if use_temp else np.empty(0, dtype=np.float64))
    snow_on = snow_T is not None
    return _compute_excess_core_nb(
        rainfall_mm, dt_hours, temp_arr, use_temp,
        float(ia_max), float(k0), float(kT), float(theta), float(T_ref),
        float(k_dep), float(T_freeze), snow_on, float(snow_ddf),
        float(snow_T) if snow_on else 0.0,
    )


class IAModel(IModel):
    """Stateful initial-abstraction model implementing the :class:`~sparsehydro.models.IModel` lifecycle.

    Constructor arguments seed the :meth:`initialize` parameter registry.
    After :meth:`initialize`, calibratable values live in the scalar-parameter
    registry and are kept in sync with the instance attributes used by the
    physics methods.

    :param ia_max: Maximum abstraction capacity.  Defaults to 0.2 in (imperial)
        or 5.0 mm (metric).  Pass ``None`` to use the unit-appropriate default.
    :type ia_max: float | None
    :param k0: Base recovery rate constant [1/hr].
    :type k0: float
    :param kT: Temperature-dependent recovery coefficient [1/hr].
    :type kT: float
    :param theta: Temperature sensitivity exponent [1/°C].
    :type theta: float
    :param T_ref: Reference temperature [°C].
    :type T_ref: float
    :param k_dep: Depletion rate constant.  Defaults to 7.62 /in (imperial)
        or 0.3 /mm (metric).  Pass ``None`` to use the unit-appropriate default.
    :type k_dep: float | None
    :param T_freeze: Temperature below which recovery is suppressed [°C].
    :type T_freeze: float
    :param units: Unit system — ``"imperial"`` (default, inches) or ``"metric"`` (mm).
    :type units: str
    :param snow: Enable the degree-day snow model.
    :type snow: bool
    :param snow_T: Initial rain/snow threshold & melt base [°C].
    :type snow_T: float
    :param snow_ddf: Initial degree-day factor.
    :type snow_ddf: float | None
    """

    model_name = "initial-abstraction"
    supports_array_predict = True

    def __init__(
        self,
        ia_max: float | None = None,
        k0: float = 0.05,
        kT: float = 0.02,
        theta: float = 0.1,
        T_ref: float = 20.0,
        k_dep: float | None = None,
        T_freeze: float = 0.0,
        units: str = "imperial",
        snow: bool = False,
        snow_T: float = 1.0,
        snow_ddf: float | None = None,
    ) -> None:
        super().__init__()
        if units not in ("imperial", "metric"):
            raise ValueError(f"units must be 'imperial' or 'metric'; got {units!r}")
        self._units = units
        self._snow = bool(snow)
        self.snow_T = float(snow_T)
        if snow_ddf is not None:
            self.snow_ddf = float(snow_ddf)
        else:
            self.snow_ddf = 0.12 if units == "imperial" else 3.0
        self._rainfall_col = "rainfall_in" if units == "imperial" else "rainfall_mm"
        self._excess_col = "p_excess_in" if units == "imperial" else "p_excess_mm"

        if units == "imperial":
            self.ia_max = float(ia_max) if ia_max is not None else 0.2
            self.k_dep = float(k_dep) if k_dep is not None else 7.62
        else:
            self.ia_max = float(ia_max) if ia_max is not None else 5.0
            self.k_dep = float(k_dep) if k_dep is not None else 0.3

        self.k0 = float(k0)
        self.kT = float(kT)
        self.theta = float(theta)
        self.T_ref = float(T_ref)
        self.T_freeze = float(T_freeze)
        self.ia_avail: float = self.ia_max
        self._prepared_df: pd.DataFrame | None = None
        # Forcing arrays and timestep cached by prepare(); see predict_arrays().
        self._dt_hours: float = 1.0 / 12.0
        self._dt_arr: np.ndarray | None = None
        self._rain_arr: np.ndarray | None = None
        self._temp_arr: np.ndarray | None = None
        self._datetime_arr: np.ndarray | None = None

    # ------------------------------------------------------------------
    # IModel lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Register all IA scalar parameters and advance to INITIALIZED.

        :returns: Nothing.
        :rtype: None
        """
        if self._units == "imperial":
            ia_max_lb, ia_max_ub, ia_max_units = 0.004, 2.0, "in"
            ia_kdep_lb, ia_kdep_ub, ia_kdep_units = 0.25, 127.0, "1/in"
            ia_kdep_desc = "Depletion rate per inch of rainfall"
        else:
            ia_max_lb, ia_max_ub, ia_max_units = 0.1, 50.0, "mm"
            ia_kdep_lb, ia_kdep_ub, ia_kdep_units = 0.01, 5.0, "1/mm"
            ia_kdep_desc = "Depletion rate per mm of rainfall"

        self.register_scalar_parameter(ScalarParameter(
            "ia_max", value=self.ia_max, lower_bound=ia_max_lb, upper_bound=ia_max_ub,
            units=ia_max_units, description="Maximum initial abstraction capacity",
        ))
        self.register_scalar_parameter(ScalarParameter(
            "ia_k0", value=self.k0, lower_bound=0.001, upper_bound=1.0,
            units="1/hr", description="Base recovery rate (gravity drainage)",
        ))
        self.register_scalar_parameter(ScalarParameter(
            "ia_kT", value=self.kT, lower_bound=0.0, upper_bound=0.5,
            units="1/hr", description="Temperature-sensitive recovery coefficient",
        ))
        self.register_scalar_parameter(ScalarParameter(
            "ia_theta", value=self.theta, lower_bound=0.0, upper_bound=0.5,
            units="1/degC", description="Temperature sensitivity exponent",
        ))
        self.register_scalar_parameter(ScalarParameter(
            "ia_T_ref", value=self.T_ref, lower_bound=0.0, upper_bound=30.0,
            units="degC", description="Reference temperature for ET scaling",
        ))
        self.register_scalar_parameter(ScalarParameter(
            "ia_k_dep", value=self.k_dep, lower_bound=ia_kdep_lb, upper_bound=ia_kdep_ub,
            units=ia_kdep_units, description=ia_kdep_desc,
        ))
        self.register_scalar_parameter(ScalarParameter(
            "ia_T_freeze", value=self.T_freeze, lower_bound=-5.0, upper_bound=5.0,
            units="degC", description="Recovery suppressed below this temperature",
        ))

        if self._snow:
            if self._units == "imperial":
                ddf_ub, ddf_units = 0.5, "in/degC/day"
            else:
                ddf_ub, ddf_units = 12.0, "mm/degC/day"
            self.register_scalar_parameter(ScalarParameter(
                "snow_T", value=self.snow_T, lower_bound=-2.0, upper_bound=4.0,
                units="degC", description="Rain/snow partition threshold and melt base",
            ))
            self.register_scalar_parameter(ScalarParameter(
                "snow_ddf", value=self.snow_ddf, lower_bound=0.0, upper_bound=ddf_ub,
                units=ddf_units, description="Degree-day snowmelt factor",
            ))

        depth_units = "in" if self._units == "imperial" else "mm"
        self.register_output_field(FieldRecord(
            name="datetime",
            description="Simulation time step",
        ))
        self.register_output_field(FieldRecord(
            name=self._excess_col,
            units=depth_units,
            description="Rainfall excess depth after initial abstraction",
            calibratable=False,
        ))

        self._state = ModelState.INITIALIZED

    def validate(self) -> bool:
        """Validate parameter bounds and physical constraint T_freeze < T_ref.

        :returns: ``True`` if all parameters are within bounds and
            ``T_freeze < T_ref``.
        :rtype: bool
        """
        if not self.parameters_valid():
            return False
        T_freeze = self.get_scalar_parameter("ia_T_freeze").value
        T_ref = self.get_scalar_parameter("ia_T_ref").value
        if T_freeze >= T_ref:
            return False
        self._state = ModelState.VALIDATED
        return True

    def prepare(self, data: pd.DataFrame) -> None:
        """Load forcing data and sync parameters from the registry.

        :param data: DataFrame with columns ``datetime`` and the unit-specific
            rainfall column (``rainfall_in`` or ``rainfall_mm``); an optional
            ``temperature_c`` column defaults to ``T_ref`` when absent.
        :type data: pandas.DataFrame
        :returns: Nothing.
        :rtype: None
        :raises ValueError: If required columns are absent.
        """
        required = {"datetime", self._rainfall_col}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(
                f"prepare() data is missing required columns: {sorted(missing)}"
            )

        df = data.sort_values("datetime").reset_index(drop=True).copy()
        df[self._rainfall_col] = df[self._rainfall_col].fillna(0.0).clip(lower=0.0)

        T_ref = self.get_scalar_parameter("ia_T_ref").value
        if "temperature_c" not in df.columns:
            df["temperature_c"] = T_ref
        else:
            df["temperature_c"] = df["temperature_c"].fillna(T_ref)

        self._prepared_df = df
        # Cache the arrays predict() needs.  These are fixed for the prepared
        # window, but predict() used to re-derive the timestep with a pandas
        # diff/median and reallocate dt_arr on every call -- once per candidate
        # parameter set, throughout a calibration.
        diffs = df["datetime"].diff().dropna()
        self._dt_hours = pd.Timedelta(diffs.median()).total_seconds() / 3600.0
        self._dt_arr = np.full(len(df), self._dt_hours)
        self._rain_arr = df[self._rainfall_col].to_numpy(dtype=float)
        self._temp_arr = df["temperature_c"].to_numpy(dtype=float)
        self._datetime_arr = df["datetime"].values
        self._sync_from_params()
        self.reset()
        self._state = ModelState.PREPARED

    def predict_arrays(self) -> dict[str, np.ndarray]:
        """Compute rainfall excess, returning plain arrays.

        :returns: Mapping with ``datetime`` and the unit-specific excess column.
        :rtype: dict[str, numpy.ndarray]
        :raises RuntimeError: If :meth:`prepare` has not been called.
        """
        if self._prepared_df is None:
            raise RuntimeError("Call prepare(data) before predict().")

        self._sync_from_params()

        # The compiled core is a bit-identical transcription of the reference
        # (fastmath off), so this swaps speed in without changing results.
        _engine = (_compute_excess_series_fast if _HAS_NUMBA
                   else IAModel.compute_excess_series)
        excess = _engine(
            rainfall_mm=self._rain_arr,
            dt_hours=self._dt_arr,
            temperature=self._temp_arr,
            ia_max=self.ia_max,
            k0=self.k0,
            kT=self.kT,
            theta=self.theta,
            T_ref=self.T_ref,
            k_dep=self.k_dep,
            T_freeze=self.T_freeze,
            snow_ddf=self.snow_ddf if self._snow else 0.0,
            snow_T=self.snow_T if self._snow else None,
        )

        self._state = ModelState.PREDICTED
        return {"datetime": self._datetime_arr, self._excess_col: excess}

    def predict(self) -> pd.DataFrame:
        """Compute rainfall excess for the prepared time series.

        :returns: DataFrame with columns ``datetime`` and the unit-specific
            excess column (``p_excess_in`` or ``p_excess_mm``).
        :rtype: pandas.DataFrame
        :raises RuntimeError: If :meth:`prepare` has not been called.
        """
        return pd.DataFrame(self.predict_arrays())

    def finalize(self) -> None:
        """Release stored forcing data and advance to FINALIZED.

        :returns: Nothing.
        :rtype: None
        """
        self._prepared_df = None
        self._dt_arr = None
        self._rain_arr = None
        self._temp_arr = None
        self._datetime_arr = None
        self._state = ModelState.FINALIZED

    # ------------------------------------------------------------------
    # Physics methods (stateful, single time-step)
    # ------------------------------------------------------------------

    def recovery_rate(self, temperature: float | None = None) -> float:
        """Compute k_rec(T) = k0 + kT * exp(θ*(T - T_ref)), zeroed below T_freeze.

        :param temperature: Air temperature [°C]; defaults to ``T_ref``.
        :type temperature: float | None
        :returns: Recovery rate constant [1/hr] (``0.0`` below ``T_freeze``).
        :rtype: float
        """
        T = self.T_ref if temperature is None else float(temperature)
        if T < self.T_freeze:
            return 0.0
        return self.k0 + self.kT * math.exp(self.theta * (T - self.T_ref))

    def step_dry(self, dt_hours: float, temperature: float | None = None) -> float:
        """Advance ``ia_avail`` over a dry interval of ``dt_hours``.

        :param dt_hours: Length of the dry interval [hr].
        :type dt_hours: float
        :param temperature: Air temperature [°C]; defaults to ``T_ref``.
        :type temperature: float | None
        :returns: Updated available abstraction capacity ``ia_avail``.
        :rtype: float
        """
        k = self.recovery_rate(temperature)
        deficit = self.ia_max - self.ia_avail
        self.ia_avail = self.ia_max - deficit * math.exp(-k * dt_hours)
        self.ia_avail = max(0.0, min(self.ia_max, self.ia_avail))
        return self.ia_avail

    def step_wet(self, delta_precip_mm: float) -> float:
        """Deplete ``ia_avail`` for a rainfall pulse of ``delta_precip_mm`` mm.

        :param delta_precip_mm: Rainfall depth applied this step.
        :type delta_precip_mm: float
        :returns: Updated available abstraction capacity ``ia_avail``.
        :rtype: float
        """
        if delta_precip_mm <= 0.0:
            return self.ia_avail
        self.ia_avail = self.ia_avail * math.exp(-self.k_dep * delta_precip_mm)
        self.ia_avail = max(0.0, min(self.ia_max, self.ia_avail))
        return self.ia_avail

    def compute_excess(self, rainfall_mm: float) -> float:
        """Return rainfall excess and deplete ``ia_avail`` accordingly.

        :param rainfall_mm: Rainfall depth applied this step.
        :type rainfall_mm: float
        :returns: Rainfall excess depth for the step.
        :rtype: float
        """
        if rainfall_mm <= 0.0:
            return 0.0
        excess = _wet_step_excess(self.ia_avail, self.k_dep, rainfall_mm)
        self.step_wet(rainfall_mm)
        return excess

    def reset(self) -> None:
        """Reset ``ia_avail`` to ``ia_max`` (fully recovered state).

        :returns: Nothing.
        :rtype: None
        """
        self.ia_avail = self.ia_max

    # ------------------------------------------------------------------
    # Static vectorized method — hot path for the optimizer
    # ------------------------------------------------------------------

    @staticmethod
    def compute_excess_series(
        rainfall_mm: np.ndarray,
        dt_hours: np.ndarray,
        temperature: np.ndarray | None,
        ia_max: float,
        k0: float,
        kT: float,
        theta: float,
        T_ref: float,
        k_dep: float,
        T_freeze: float,
        snow_ddf: float = 0.0,
        snow_T: float | None = None,
    ) -> np.ndarray:
        """Compute rainfall excess array for a full time series.

        Pure function (no instance state) — safe for parallel optimizer use.
        ``ia_avail`` starts at ``ia_max`` (fully saturated capacity).

        This is the readable reference implementation.  When numba is installed,
        :meth:`predict_arrays` routes through the compiled twin
        (:func:`_compute_excess_series_fast`) instead, which produces
        bit-identical output; this method stays the definition of correct.

        :param rainfall_mm: Per-step rainfall depth series.
        :type rainfall_mm: numpy.ndarray
        :param dt_hours: Per-step interval length series [hr].
        :type dt_hours: numpy.ndarray
        :param temperature: Per-step temperature series [°C], or ``None`` to use
            ``T_ref`` everywhere.
        :type temperature: numpy.ndarray | None
        :param ia_max: Maximum abstraction capacity.
        :type ia_max: float
        :param k0: Base recovery rate constant [1/hr].
        :type k0: float
        :param kT: Temperature-dependent recovery coefficient [1/hr].
        :type kT: float
        :param theta: Temperature sensitivity exponent [1/°C].
        :type theta: float
        :param T_ref: Reference temperature [°C].
        :type T_ref: float
        :param k_dep: Depletion rate constant.
        :type k_dep: float
        :param T_freeze: Temperature below which recovery is suppressed [°C].
        :type T_freeze: float
        :param snow_ddf: Degree-day snowmelt factor (``0.0`` disables melt).
        :type snow_ddf: float
        :param snow_T: Rain/snow threshold [°C], or ``None`` to disable snow.
        :type snow_T: float | None
        :returns: Rainfall excess depth series, same length as *rainfall_mm*.
        :rtype: numpy.ndarray
        """
        n = len(rainfall_mm)
        excess = np.zeros(n, dtype=float)
        ia_avail = float(ia_max)

        snow_on = snow_T is not None
        swe = 0.0
        use_temp = temperature is not None

        for i in range(n):
            P = float(rainfall_mm[i])
            dt = float(dt_hours[i])
            T = float(temperature[i]) if use_temp else T_ref

            if snow_on:
                if T <= snow_T:
                    swe += P
                    P = 0.0
                elif swe > 0.0:
                    melt = min(swe, snow_ddf * (T - snow_T) * dt / 24.0)
                    swe -= melt
                    P += melt

            if P <= 0.0:
                if T >= T_freeze:
                    k_rec = k0 + kT * math.exp(theta * (T - T_ref))
                else:
                    k_rec = 0.0
                deficit = ia_max - ia_avail
                ia_avail = ia_max - deficit * math.exp(-k_rec * dt)
            else:
                excess[i] = _wet_step_excess(ia_avail, k_dep, P)
                ia_avail = ia_avail * math.exp(-k_dep * P)

            ia_avail = max(0.0, min(ia_max, ia_avail))

        return excess

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sync_from_params(self) -> None:
        """Update instance attributes from the scalar-parameter registry.

        :returns: Nothing.
        :rtype: None
        """
        self.ia_max = self.get_scalar_parameter("ia_max").value
        self.k0 = self.get_scalar_parameter("ia_k0").value
        self.kT = self.get_scalar_parameter("ia_kT").value
        self.theta = self.get_scalar_parameter("ia_theta").value
        self.T_ref = self.get_scalar_parameter("ia_T_ref").value
        self.k_dep = self.get_scalar_parameter("ia_k_dep").value
        self.T_freeze = self.get_scalar_parameter("ia_T_freeze").value
        if self._snow:
            self.snow_T = self.get_scalar_parameter("snow_T").value
            self.snow_ddf = self.get_scalar_parameter("snow_ddf").value


def _snow_step(precipitation, temperature, dt_hours, swe, snow_T, snow_ddf):
    """Partition water once, before either IA implementation receives it."""
    snowfall = 0.0
    melt = 0.0
    if temperature <= snow_T:
        snowfall = precipitation
        swe += precipitation
        liquid = 0.0
    else:
        if swe > 0.0:
            melt = min(swe, snow_ddf * (temperature - snow_T) * dt_hours / 24.0)
            swe -= melt
        liquid = precipitation + melt
    return swe, liquid, snowfall, melt


_snow_step_nb = njit(cache=True, fastmath=False)(_snow_step)


@dataclass(frozen=True)
class SnowConfig:
    """Immutable imperial degree-day settings; SWE is water, not snow depth."""

    threshold_c: float = 1.0
    melt_factor_in_per_c_day: float = 0.12
    initial_swe_in: float = 0.0

    def __post_init__(self) -> None:
        if (not np.isfinite(self.threshold_c)
                or not np.isfinite(self.melt_factor_in_per_c_day)
                or self.melt_factor_in_per_c_day < 0
                or not np.isfinite(self.initial_swe_in) or self.initial_swe_in < 0):
            raise ValueError("Snow settings must be finite; melt factor and initial SWE nonnegative")


@njit(cache=True, fastmath=False, nogil=True)
def _snow_partition_core(precipitation, temperature, dt_hours, threshold, factor, initial):
    n = len(precipitation)
    swe = np.zeros(n)
    liquid = np.zeros(n)
    snowfall = np.zeros(n)
    melt = np.zeros(n)
    storage = initial
    for i in range(n):
        storage, liquid[i], snowfall[i], melt[i] = _snow_step_nb(
            precipitation[i], temperature[i], dt_hours, storage, threshold, factor)
        swe[i] = storage
    return swe, liquid, snowfall, melt


def snow_partition(
    precipitation_in: np.ndarray, temperature_c: np.ndarray, dt_hours: float,
    config: SnowConfig | None = None,
) -> pd.DataFrame:
    """Deterministic forcing-only snow ledger; ``None`` passes precipitation through.

    Run on the full forcing record including warmup, never separately by split.
    ``rainfall_in`` at the caller remains measured total precipitation; only
    ``liquid_input_in`` may enter the soil/GWI/RTK physics.
    """
    rain = np.ascontiguousarray(precipitation_in, dtype=float)
    temp = np.ascontiguousarray(temperature_c, dtype=float)
    if (rain.ndim != 1 or rain.size == 0 or temp.shape != rain.shape
            or not np.isfinite(rain).all() or (rain < 0).any()
            or not np.isfinite(temp).all() or not np.isfinite(dt_hours) or dt_hours <= 0):
        raise ValueError("Snow forcing must be aligned finite vectors, nonnegative rain, positive timestep")
    if config is None:
        swe, liquid, snowfall, melt = np.zeros_like(rain), rain.copy(), np.zeros_like(rain), np.zeros_like(rain)
    else:
        swe, liquid, snowfall, melt = _snow_partition_core(
            rain, temp, dt_hours, config.threshold_c,
            config.melt_factor_in_per_c_day, config.initial_swe_in)
    return pd.DataFrame({
        "snowfall_in": snowfall, "rain_liquid_in": rain - snowfall,
        "melt_in": melt, "liquid_input_in": liquid, "SWE_in": swe,
    })
