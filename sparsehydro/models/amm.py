"""AMMModel — reparameterized Antecedent Moisture Model (AMM).

Implements the reparameterized AMM equations of Edgren, Czachorski & Gonwa
(2024, *Journal of Water Management Modeling* 32: C525,
https://doi.org/10.14796/JWMM.C525) as a concrete
:class:`~sparsehydro.models.IModel`.

A single configurable :class:`AMMModel` supports two component types:

- ``"standard"`` — the full three-level wet-weather component (Equations 1–11),
  used for rainfall runoff or rain-derived infiltration/inflow (RDII).
- ``"baseflow"`` — the two-level baseflow component (Equations 1–3, 12–16),
  which omits Level 2 and drives the capture fraction directly from
  temperature.  Suitable for base flow / groundwater infiltration.

Multiple components (e.g. fast inflow + slow infiltration + baseflow,
Figure 5 of the paper) are combined by summing several :class:`AMMModel`
instances via :class:`~sparsehydro.models.EnsembleModel`.

Equations (standard component)
------------------------------
Level 1 — flow (Eq. 1)::

    Q_t = A·(RD + (RW_t + RW_{t-1})/2)·MAP_t·(1 - SF)/Δt + SF·Q_{t-1}

Shape factor (Eq. 2) and antecedent-moisture retention factor (Eq. 6)::

    SF   = 0.5 ** (Δt / HHL)
    AMRF = 0.5 ** (Δt / AMHL)

Moving-average precipitation (Eq. 3, start-of-interval, lagged one step)::

    MAP_t = mean(P_{t-1}, …, P_{t-N}),   N = PAT/Δt + 1

Level 2 — additional wet-weather capture fraction (Eq. 5)::

    RW_t = (AMRF - 1)/ln(AMRF)·SHCF_t·MAP_t + AMRF·RW_{t-1}

Level 3 — seasonal hydrologic condition factor (Eqs. 7–11)::

    L      = 1.2·(Cold_SHCF - Hot_SHCF)
    k      = 4.7964 / (Cold_Temp - Hot_Temp)
    x0     = (Cold_Temp + Hot_Temp)/2
    SHCF_t = L/(1 + exp(-k·(MATemp_t - x0))) + Cold_SHCF - (11/12)·L
    MATemp_t = mean(Temp_t, …, Temp_{t-(TAT/Δt)})

Baseflow component (Eqs. 12–16) replaces Level 2/3 with a temperature-driven
total capture fraction ``R_t`` (Eq. 13) and drops ``RD``::

    Q_t = A·((R_t + R_{t-1})/2)·MAP_t·(1 - SF)/Δt + SF·Q_{t-1}
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from ..enums import ModelState
from ..parameters import ConstraintRecord, FieldRecord, ScalarParameter
from ..registry import registry
from .base import IModel

# Catchment-depth (acre·depth/hr) → CFS conversion factors.
#   acres → ft²            : × 43560
#   depth → ft             : ÷ 12 (inches) or ÷ 304.8 (mm)
#   per-hour → per-second  : ÷ 3600
from ..units import IN_AC_PER_HR_TO_CFS as _IN_AC_PER_HR_TO_CFS, MM_AC_PER_HR_TO_CFS as _MM_AC_PER_HR_TO_CFS

# Sigmoid steepness constant from the paper (Eq. 9 / 15): places Point 1 and
# Point 2 at 11/12 and 1/12 of the SHCF range, respectively.
_SIGMOID_K_CONST = 4.7964


def _moving_average_precip(values: np.ndarray, n_steps: int) -> np.ndarray:
    """Lagged moving-average precipitation per Eq. 3 (start-of-interval).

    ``MAP_t = mean(P_{t-1}, …, P_{t-n_steps})`` with zero-padding before the
    start of the series (the denominator is always ``n_steps``).  For
    ``n_steps == 1`` this reduces to a one-step lag ``MAP_t = P_{t-1}``.

    :param values: Incremental precipitation series.
    :type values: numpy.ndarray
    :param n_steps: Window length ``PAT/Δt + 1`` (≥ 1).
    :type n_steps: int
    :returns: Moving-average precipitation, same length as *values*.
    :rtype: numpy.ndarray
    """
    n = max(int(n_steps), 1)
    csum = np.concatenate(([0.0], np.cumsum(values, dtype=float)))
    t = np.arange(len(values))
    lo = np.maximum(t - n, 0)
    return (csum[t] - csum[lo]) / n


def _moving_average_temp(values: np.ndarray, n_steps: int) -> np.ndarray:
    """Trailing moving-average temperature per Eq. 11 (includes current step).

    ``MATemp_t = mean(Temp_t, …, Temp_{t-(n_steps-1)})``.  Near the start of the
    series the window shrinks (divides by the number of available samples)
    rather than zero-padding, since temperature is not zero before the record.

    :param values: Temperature series.
    :type values: numpy.ndarray
    :param n_steps: Window length ``TAT/Δt + 1`` (≥ 1).
    :type n_steps: int
    :returns: Moving-average temperature, same length as *values*.
    :rtype: numpy.ndarray
    """
    n = max(int(n_steps), 1)
    csum = np.concatenate(([0.0], np.cumsum(values, dtype=float)))
    t = np.arange(len(values))
    lo = np.maximum(t - n + 1, 0)
    cnt = (t + 1) - lo
    return (csum[t + 1] - csum[lo]) / cnt


def _shcf_sigmoid(
    matemp: np.ndarray,
    cold_temp: float,
    hot_temp: float,
    cold_value: float,
    hot_value: float,
) -> np.ndarray:
    """Seasonal hydrologic condition factor sigmoid (Eqs. 7–10 / 13–16).

    Evaluates the temperature → capture-rate sigmoid defined by the two
    calibration points ``(Cold_Temp, Cold_value)`` and ``(Hot_Temp, Hot_value)``.
    Used for both the Level-3 ``SHCF`` (standard component) and the
    temperature-driven total capture fraction ``R`` (baseflow component).

    :param matemp: Moving-average temperature series.
    :type matemp: numpy.ndarray
    :param cold_temp: Cold-season temperature (Point 1).
    :type cold_temp: float
    :param hot_temp: Hot-season temperature (Point 2).
    :type hot_temp: float
    :param cold_value: Factor value at the cold point.
    :type cold_value: float
    :param hot_value: Factor value at the hot point.
    :type hot_value: float
    :returns: Sigmoid factor series, same length as *matemp*.
    :rtype: numpy.ndarray
    """
    span = 1.2 * (cold_value - hot_value)
    k = _SIGMOID_K_CONST / (cold_temp - hot_temp)
    x0 = (cold_temp + hot_temp) / 2.0
    return span / (1.0 + np.exp(-k * (matemp - x0))) + cold_value - (11.0 / 12.0) * span


@registry.register
class AMMModel(IModel):
    """Reparameterized Antecedent Moisture Model component.

    :param component_type: ``"standard"`` for the three-level wet-weather
        component (Eqs. 1–11) or ``"baseflow"`` for the two-level baseflow
        component (Eqs. 1–3, 12–16).
    :type component_type: str
    :param units: Unit system — ``"imperial"`` (inches, acres, CFS; default)
        or ``"metric"`` (mm).  Temperature units are unconstrained but
        ``cold_temp``/``hot_temp`` must match the supplied temperature series.
    :type units: str
    :param output_name: Column name for the flow output of :meth:`predict`.
    :type output_name: str
    :param area_acres: Catchment area [acres].
    :type area_acres: float
    :param pat_hours: Precipitation averaging time ``PAT`` [hr]; ``TP = PAT + Δt``.
    :type pat_hours: float
    :param hhl_hours: Hydrograph half-life ``HHL`` [hr].
    :type hhl_hours: float
    :param amhl_hours: Antecedent-moisture half-life ``AMHL`` [hr] (standard only).
    :type amhl_hours: float
    :param rd: Dry-weather capture fraction ``RD`` (standard only).
    :type rd: float
    :param cold_temp: Cold-season temperature (Point 1).
    :type cold_temp: float
    :param hot_temp: Hot-season temperature (Point 2).
    :type hot_temp: float
    :param cold_value: ``Cold_SHCF`` (standard) or ``Cold_R`` (baseflow) at Point 1.
    :type cold_value: float | None
    :param hot_value: ``Hot_SHCF`` (standard) or ``Hot_R`` (baseflow) at Point 2.
    :type hot_value: float | None
    :param tat_hours: Temperature averaging time ``TAT`` [hr].
    :type tat_hours: float

    Usage::

        from sparsehydro.models.amm import AMMModel

        model = AMMModel(component_type="standard", units="imperial")
        model.initialize()
        model.validate()
        model.prepare(df)              # columns: datetime, rainfall_in, temperature
        result = model.predict()       # datetime, amm_cfs, capture_fraction, rw, shcf, map, matemp
        model.finalize()
    """

    model_name: ClassVar[str] = "amm"

    def __init__(
        self,
        component_type: str = "standard",
        units: str = "imperial",
        output_name: str = "amm_cfs",
        *,
        area_acres: float = 100.0,
        pat_hours: float = 0.0,
        hhl_hours: float = 2.0,
        amhl_hours: float = 8.0,
        rd: float = 0.01,
        cold_temp: float = 30.0,
        hot_temp: float = 70.0,
        cold_value: float | None = None,
        hot_value: float | None = None,
        tat_hours: float = 0.0,
    ) -> None:
        super().__init__()
        if component_type not in ("standard", "baseflow"):
            raise ValueError(
                f"component_type must be 'standard' or 'baseflow'; got {component_type!r}"
            )
        if units not in ("imperial", "metric"):
            raise ValueError(f"units must be 'imperial' or 'metric'; got {units!r}")

        self._component_type = component_type
        self._units = units
        self._output_name = output_name

        self._rainfall_col = "rainfall_in" if units == "imperial" else "rainfall_mm"
        self._depth_units = "in" if units == "imperial" else "mm"
        self._shcf_units = "1/in" if units == "imperial" else "1/mm"
        self._depth_to_cfs = (
            _IN_AC_PER_HR_TO_CFS if units == "imperial" else _MM_AC_PER_HR_TO_CFS
        )

        # Seed values for the parameter registry.
        self._seed_area = float(area_acres)
        self._seed_pat = float(pat_hours)
        self._seed_hhl = float(hhl_hours)
        self._seed_amhl = float(amhl_hours)
        self._seed_rd = float(rd)
        self._seed_cold_temp = float(cold_temp)
        self._seed_hot_temp = float(hot_temp)
        self._seed_tat = float(tat_hours)
        if cold_value is None:
            cold_value = 0.07 if component_type == "standard" else 0.05
        if hot_value is None:
            hot_value = 0.03 if component_type == "standard" else 0.01
        self._seed_cold_value = float(cold_value)
        self._seed_hot_value = float(hot_value)

        self._prepared_df: pd.DataFrame | None = None
        self._dt_hours: float = 1.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def component_type(self) -> str:
        """Component type — ``"standard"`` or ``"baseflow"``."""
        return self._component_type

    @property
    def is_standard(self) -> bool:
        """``True`` for the three-level standard wet-weather component."""
        return self._component_type == "standard"

    @property
    def time_to_peak_hours(self) -> float:
        """Derived time to peak ``TP = PAT + Δt`` [hr] (Eq. 4)."""
        return self.get_scalar_parameter("PAT").value + self._dt_hours

    # ------------------------------------------------------------------
    # IModel lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Register parameters, output fields, and constraints."""
        self._reset_registries()
        self.register_scalar_parameter(ScalarParameter(
            "area_acres", value=self._seed_area, lower_bound=0.01, upper_bound=100_000.0,
            units="acres", description="Catchment area — converts capture depth to flow (CFS)",
        ))
        self.register_scalar_parameter(ScalarParameter(
            "PAT", value=self._seed_pat, lower_bound=0.0, upper_bound=720.0,
            units="hr", description="Precipitation averaging time (time to peak = PAT + dt)",
        ))
        self.register_scalar_parameter(ScalarParameter(
            "HHL", value=self._seed_hhl, lower_bound=0.01, upper_bound=2000.0,
            units="hr", description="Hydrograph half-life (recession time)",
        ))
        self.register_scalar_parameter(ScalarParameter(
            "TAT", value=self._seed_tat, lower_bound=0.0, upper_bound=10_000.0,
            units="hr", description="Temperature averaging time",
        ))
        self.register_scalar_parameter(ScalarParameter(
            "Cold_Temp", value=self._seed_cold_temp, lower_bound=-100.0, upper_bound=200.0,
            units="temp", description="Cold-season temperature (sigmoid Point 1)",
        ))
        self.register_scalar_parameter(ScalarParameter(
            "Hot_Temp", value=self._seed_hot_temp, lower_bound=-100.0, upper_bound=200.0,
            units="temp", description="Hot-season temperature (sigmoid Point 2)",
        ))

        if self.is_standard:
            self.register_scalar_parameter(ScalarParameter(
                "RD", value=self._seed_rd, lower_bound=0.0, upper_bound=1.0,
                units="-", description="Dry-weather (minimum) rainfall capture fraction",
            ))
            self.register_scalar_parameter(ScalarParameter(
                "AMHL", value=self._seed_amhl, lower_bound=0.01, upper_bound=20_000.0,
                units="hr", description="Antecedent-moisture half-life (drying time)",
            ))
            self.register_scalar_parameter(ScalarParameter(
                "Cold_SHCF", value=self._seed_cold_value, lower_bound=0.0, upper_bound=100.0,
                units=self._shcf_units,
                description="Seasonal hydrologic condition factor at the cold point",
            ))
            self.register_scalar_parameter(ScalarParameter(
                "Hot_SHCF", value=self._seed_hot_value, lower_bound=0.0, upper_bound=100.0,
                units=self._shcf_units,
                description="Seasonal hydrologic condition factor at the hot point",
            ))
        else:
            self.register_scalar_parameter(ScalarParameter(
                "Cold_R", value=self._seed_cold_value, lower_bound=0.0, upper_bound=1.0,
                units="-", description="Total rainfall capture fraction at the cold point",
            ))
            self.register_scalar_parameter(ScalarParameter(
                "Hot_R", value=self._seed_hot_value, lower_bound=0.0, upper_bound=1.0,
                units="-", description="Total rainfall capture fraction at the hot point",
            ))

        self.register_inequality_constraint(ConstraintRecord(
            name="cold_temp_lt_hot_temp",
            description="Cold_Temp < Hot_Temp (sigmoid points must be distinct and ordered)",
        ))

        self._register_output_fields()
        self._state = ModelState.INITIALIZED

    def _register_output_fields(self) -> None:
        """Register the :class:`FieldRecord` metadata for the output columns."""
        self.register_output_field(FieldRecord(
            name="datetime", description="Simulation time step",
        ))
        self.register_output_field(FieldRecord(
            name=self._output_name, units="CFS",
            description="AMM flow rate (Level 1)", calibratable=True,
        ))
        self.register_output_field(FieldRecord(
            name="capture_fraction", units="-",
            description="Total rainfall capture fraction (RD + RW, or R for baseflow)",
            calibratable=False,
        ))
        if self.is_standard:
            self.register_output_field(FieldRecord(
                name="rw", units="-",
                description="Additional wet-weather capture fraction (Level 2)",
                calibratable=False,
            ))
            self.register_output_field(FieldRecord(
                name="shcf", units=self._shcf_units,
                description="Seasonal hydrologic condition factor (Level 3)",
                calibratable=False,
            ))
        self.register_output_field(FieldRecord(
            name="map", units=self._depth_units,
            description="Moving-average incremental precipitation",
            calibratable=False,
        ))
        self.register_output_field(FieldRecord(
            name="matemp", units="temp",
            description="Moving-average temperature",
            calibratable=False,
        ))

    def validate(self) -> bool:
        """Validate parameter bounds and the temperature-ordering constraint.

        :returns: ``True`` if all parameters are within bounds and
            ``Cold_Temp < Hot_Temp``.
        :rtype: bool
        """
        if not self.parameters_valid():
            return False
        if self.get_scalar_parameter("Cold_Temp").value >= self.get_scalar_parameter("Hot_Temp").value:
            return False
        self._state = ModelState.VALIDATED
        return True

    def prepare(self, data: pd.DataFrame, **kwargs: Any) -> None:
        """Load forcing data, infer the time step, and fill temperature.

        :param data: DataFrame with columns ``datetime`` and ``rainfall_in``
            (imperial) or ``rainfall_mm`` (metric).  An optional
            ``temperature`` (or ``temperature_c``) column drives the seasonal
            sigmoid; when absent it falls back to the midpoint of
            ``Cold_Temp`` and ``Hot_Temp``.
        :type data: pandas.DataFrame
        :raises ValueError: If required columns are absent or the time step
            cannot be inferred.
        """
        required = {"datetime", self._rainfall_col}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(
                f"prepare() data is missing required columns: {sorted(missing)}"
            )

        df = data.sort_values("datetime").reset_index(drop=True).copy()
        df[self._rainfall_col] = df[self._rainfall_col].fillna(0.0).clip(lower=0.0)

        diffs = df["datetime"].diff().dropna()
        if len(diffs) == 0:
            raise ValueError("Cannot infer dt_hours: DataFrame has fewer than 2 rows.")
        self._dt_hours = pd.Timedelta(diffs.median()).total_seconds() / 3600.0
        if self._dt_hours <= 0.0:
            raise ValueError(
                f"Inferred dt_hours = {self._dt_hours:.4f} is not positive. "
                "Ensure the datetime column is sorted with a uniform step."
            )

        temp_col = None
        if "temperature" in df.columns:
            temp_col = "temperature"
        elif "temperature_c" in df.columns:
            temp_col = "temperature_c"
        midpoint = 0.5 * (
            self.get_scalar_parameter("Cold_Temp").value
            + self.get_scalar_parameter("Hot_Temp").value
        )
        if temp_col is None:
            df["temperature"] = midpoint
        else:
            df["temperature"] = df[temp_col].fillna(midpoint)

        self._prepared_df = df
        self._state = ModelState.PREPARED

    def predict(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Run the AMM recursion and return the output time series.

        :returns: DataFrame with columns ``datetime``, ``{output_name}``,
            ``capture_fraction``, ``map``, ``matemp`` and — for the standard
            component — ``rw`` and ``shcf``.
        :rtype: pandas.DataFrame
        :raises RuntimeError: If :meth:`prepare` has not been called.
        """
        if self._prepared_df is None:
            raise RuntimeError("Call prepare(data) before predict().")

        dt = self._dt_hours
        precip = self._prepared_df[self._rainfall_col].to_numpy(dtype=float)
        temp = self._prepared_df["temperature"].to_numpy(dtype=float)
        n = len(precip)

        area = self.get_scalar_parameter("area_acres").value
        pat = self.get_scalar_parameter("PAT").value
        hhl = self.get_scalar_parameter("HHL").value
        tat = self.get_scalar_parameter("TAT").value
        cold_temp = self.get_scalar_parameter("Cold_Temp").value
        hot_temp = self.get_scalar_parameter("Hot_Temp").value

        sf = 0.5 ** (dt / hhl)
        n_precip = int(round(pat / dt)) + 1
        n_temp = int(round(tat / dt)) + 1

        map_series = _moving_average_precip(precip, n_precip)
        matemp = _moving_average_temp(temp, n_temp)

        flow_scale = area * self._depth_to_cfs * (1.0 - sf) / dt

        if self.is_standard:
            rd = self.get_scalar_parameter("RD").value
            amhl = self.get_scalar_parameter("AMHL").value
            cold_shcf = self.get_scalar_parameter("Cold_SHCF").value
            hot_shcf = self.get_scalar_parameter("Hot_SHCF").value

            shcf = _shcf_sigmoid(matemp, cold_temp, hot_temp, cold_shcf, hot_shcf)
            amrf = 0.5 ** (dt / amhl)
            # (AMRF - 1) / ln(AMRF): time-step correction factor (Eq. 5).
            # Guard the removable singularities so optimisers can probe the
            # bounds of AMHL without a math-domain crash: for a small AMHL
            # (relative to dt) ``amrf`` underflows to 0 and the factor tends to
            # 0; for a large AMHL ``amrf`` tends to 1 and the factor tends to 1.
            if amrf <= 0.0:
                rw_gain = 0.0
            elif amrf >= 1.0:
                rw_gain = 1.0
            else:
                rw_gain = (amrf - 1.0) / math.log(amrf)

            rw = np.zeros(n, dtype=float)
            for t in range(1, n):
                rw[t] = rw_gain * shcf[t] * map_series[t] + amrf * rw[t - 1]
            cf = rw
            rd_const = rd
        else:
            cold_r = self.get_scalar_parameter("Cold_R").value
            hot_r = self.get_scalar_parameter("Hot_R").value
            cf = _shcf_sigmoid(matemp, cold_temp, hot_temp, cold_r, hot_r)
            shcf = None
            rw = None
            rd_const = 0.0

        flow = np.zeros(n, dtype=float)
        for t in range(1, n):
            cf_avg = 0.5 * (cf[t] + cf[t - 1])
            flow[t] = flow_scale * (rd_const + cf_avg) * map_series[t] + sf * flow[t - 1]

        capture_fraction = rd_const + cf

        result = {
            "datetime": self._prepared_df["datetime"].values,
            self._output_name: flow,
            "capture_fraction": capture_fraction,
        }
        if self.is_standard:
            result["rw"] = rw
            result["shcf"] = shcf
        result["map"] = map_series
        result["matemp"] = matemp

        self._state = ModelState.PREDICTED
        return pd.DataFrame(result)

    def finalize(self) -> None:
        """Release cached forcing data and advance to FINALIZED."""
        self._prepared_df = None
        self._state = ModelState.FINALIZED

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    def inequality_constraints(self) -> list[float]:
        """Inequality constraint residuals (``g ≤ 0`` is feasible).

        Returns ``[Cold_Temp - Hot_Temp]`` enforcing ``Cold_Temp < Hot_Temp``.

        :returns: Single-element list ``[Cold_Temp - Hot_Temp]``.
        :rtype: list[float]
        """
        return [
            self.get_scalar_parameter("Cold_Temp").value
            - self.get_scalar_parameter("Hot_Temp").value
        ]

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"AMMModel(component_type={self._component_type!r}, "
            f"units={self._units!r}, output_name={self._output_name!r})"
        )
