"""Observation-free sanitary template + explicit shared GWI/three-RTK model.

The inherited parameter registry and triangle bounds match WholeFlowModel.
The inherited G-cap spill is retained and reported, not a new groundwater source.
Only the sanitary level is jointly fitted; the normalized template is frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, overload

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve

from sparsehydro.enums import ModelState
from sparsehydro.parameters import FieldRecord, ScalarParameter
from sparsehydro.models.rdii.initial_abstraction import SnowConfig, snow_partition
from sparsehydro.models.wholeflow import J_INIT_RTK, WholeFlowModel, _g_at_cap, _gwi_B_core


@dataclass(frozen=True)
class SanitaryTemplate:
    """24 hourly and 7 weekday factors; 29 independent normalized shape values."""

    hourly: tuple[float, ...]
    weekday: tuple[float, ...]

    def __post_init__(self) -> None:
        for values, size in ((self.hourly, 24), (self.weekday, 7)):
            a = np.asarray(values)
            if a.shape != (size,) or not np.all(np.isfinite(a)) or np.any(a < 0):
                raise ValueError("Template factors must be finite, nonnegative and complete")
            if not np.isclose(a.mean(), 1.0):
                raise ValueError("Each template dimension must have mean one")

    def predict(self, datetimes: pd.Series | pd.DatetimeIndex) -> np.ndarray:
        t = pd.DatetimeIndex(datetimes)
        return np.asarray(self.hourly)[t.hour] * np.asarray(self.weekday)[t.dayofweek]




class _TemplateTerm:
    def __init__(self, template: SanitaryTemplate, level: float, upper: float):
        self.template = template
        self.level = level
        self.upper = upper
        self.values: np.ndarray | None = None

    def register(self, model: WholeFlowModel) -> None:
        model.register_scalar_parameter(ScalarParameter(
            "sanitary_level_cfs", self.level, 0.0, self.upper, units="cfs",
            description="Jointly calibrated mean sanitary level over a reference week",
        ))

    def prepare(self, datetimes: pd.DatetimeIndex) -> None:
        self.values = self.template.predict(datetimes)

    def evaluate(self, model: WholeFlowModel) -> np.ndarray:
        if self.values is None:
            raise RuntimeError("Prepare the sanitary template before prediction")
        return float(model.get_scalar_parameter("sanitary_level_cfs").value) * self.values


class CompactWholeFlowModel(WholeFlowModel):
    """One vector, no observation-derived fitting inside prepare/predict.

    Fixed ia_k_dep=5 and ia_theta=0.08 reduce two free coefficients. kT remains
    calibrated for the initial temperature-aware model; a training-only kT=0
    comparison tests simpler recovery. S0 and G0 are bounded fractions of capacity.
    Snow is fixed/off by default; calibrate_snow adds bounded snow_T/snow_ddf.
    """

    model_name = "wholeflow-compact"

    def __init__(
        self, template: SanitaryTemplate, area_acres: float, dt_hours: float,
        level_cfs: float, level_upper_cfs: float, *, snow_config: SnowConfig | None = None,
        calibrate_snow: bool = False, groundwater_enabled: bool = True,
        init_rtk: tuple[tuple[float, float, float], ...] | None = None,
    ):
        if not (np.isfinite(area_acres) and area_acres > 0 and np.isfinite(dt_hours) and dt_hours > 0):
            raise ValueError("Area and timestep must be positive and finite")
        if not (0 <= level_cfs <= level_upper_cfs and np.isfinite(level_upper_cfs)):
            raise ValueError("Invalid sanitary level bounds")
        if calibrate_snow and snow_config is None:
            raise ValueError("Snow calibration requires an enabled SnowConfig")
        super().__init__(np.zeros(1), area_acres, dt_hours,
                         J_INIT_RTK if init_rtk is None else init_rtk,
                         sanitary_mode="offset", groundwater_enabled=groundwater_enabled,
                         sanitary_term=_TemplateTerm(template, level_cfs, level_upper_cfs))
        self._snow_config = snow_config
        self._calibrate_snow = calibrate_snow
        self._snow_forcing: pd.DataFrame | None = None
        self._snow_forcing_config: SnowConfig | None = None
        self.budget: dict[str, float] = {}

    @property
    def snow_config(self) -> SnowConfig | None:
        """Effective snow settings, including the current calibrated coefficients."""
        config = self._snow_config
        if config is None or not self._calibrate_snow or "snow_T" not in self.scalar_parameter_names:
            return config
        return SnowConfig(
            threshold_c=float(self.get_scalar_parameter("snow_T").value),
            melt_factor_in_per_c_day=float(self.get_scalar_parameter("snow_ddf").value),
            initial_swe_in=config.initial_swe_in,
        )

    def initialize(self) -> None:
        super().initialize()
        if self._calibrate_snow:
            config = self._snow_config
            if config is None:
                raise ValueError("Snow calibration requires an enabled SnowConfig")
            for name, value, lower, upper, units, description in (
                ("snow_T", config.threshold_c, -2.0, 3.0, "degC",
                 "Joint rain/snow partition threshold and degree-day melt base"),
                ("snow_ddf", config.melt_factor_in_per_c_day, 0.005, 0.5, "in/degC/day",
                 "Degree-day melt factor; initial SWE remains fixed"),
            ):
                if not lower <= value <= upper:
                    raise ValueError(f"{name} initial value must lie within [{lower}, {upper}]")
                self.register_scalar_parameter(ScalarParameter(
                    name, value, lower, upper, units=units, description=description,
                ))
        for name in ("ia_k_dep", "ia_theta"):
            self.get_scalar_parameter(name).calibrate = False
        for name in ("s_initial_fraction", "g_initial_fraction"):
            self.register_scalar_parameter(ScalarParameter(
                name, 0.0 if name == "g_initial_fraction" and not self.groundwater_enabled else 0.5,
                0.0, 1.0, units="-",
                description="Initial storage / capacity; explicit antecedent-water assumption",
                calibrate=name != "g_initial_fraction" or self.groundwater_enabled,
            ))
        for name, units in (
            ("et_in", "in"), ("recharge_in", "in"), ("gwi_release_in", "in"),
            ("gwi_spill_in", "in"), ("unrouted_excess_in", "in"),
            ("routing_storage_in", "in"), ("budget_error_in", "in"),
            ("snowfall_in", "in"), ("rain_liquid_in", "in"), ("melt_in", "in"),
            ("liquid_input_in", "in"), ("SWE_in", "in"),
        ):
            self.register_output_field(FieldRecord(name=name, units=units))
        for i in range(1, len(self._tri) + 1):
            self.register_output_field(FieldRecord(name=f"rtk{i}_cfs", units="cfs"))

    def validate(self) -> bool:
        return (super().validate()
                and (self.groundwater_enabled
                     or self.get_scalar_parameter("g_initial_fraction").value == 0)
                and all(x <= 0 for x in self.inequality_constraints()))

    def prepare(self, data: pd.DataFrame) -> None:
        required = ["datetime", self._rain_col, self._temp_col]
        if not set(required).issubset(data):
            raise ValueError(f"Prediction requires only {required}")
        df = data[required].reset_index(drop=True).copy()
        t = pd.DatetimeIndex(df["datetime"])
        if len(t) < 2 or t.hasnans or not t.is_monotonic_increasing or t.has_duplicates:
            raise ValueError("Forcing must have unique ascending timestamps")
        if not np.allclose(np.diff(t.as_unit("ns").asi8) / 3.6e12, self._dt_hours):
            raise ValueError("Forcing must be on the configured regular timestep")
        rain, temp = df[self._rain_col].to_numpy(float), df[self._temp_col].to_numpy(float)
        if not np.all(np.isfinite(rain)) or np.any(rain < 0) or not np.all(np.isfinite(temp)):
            raise ValueError("Rain and temperature must be complete; rain must be nonnegative")
        self._df, self._rain, self._temp = df, rain, temp
        self._snow_forcing = None
        self._refresh_snow_forcing()
        self.sanitary.prepare(t)
        self._state = ModelState.PREPARED

    def _refresh_snow_forcing(self) -> None:
        config = self.snow_config
        if self._snow_forcing is None or config != self._snow_forcing_config:
            self._snow_forcing = snow_partition(self._rain, self._temp, self._dt_hours, config)
            self._snow_forcing_config = config

    def predict(self) -> pd.DataFrame:
        """Return the complete diagnostic prediction."""
        return self._predict(diagnostics=True)

    def predict_total_cfs(self) -> np.ndarray:
        """Return total flow only, retaining identical physics and budget checks.

        The float64 array follows the prepared forcing order, including warmup.
        No diagnostic dataframe or per-component cfs arrays are constructed.
        ``budget`` and model state are updated just as in ``predict()``.
        """
        return self._predict(diagnostics=False)

    @overload
    def _predict(self, *, diagnostics: Literal[True]) -> pd.DataFrame: ...

    @overload
    def _predict(self, *, diagnostics: Literal[False]) -> np.ndarray: ...

    def _predict(self, *, diagnostics: bool) -> pd.DataFrame | np.ndarray:
        if self._df is None or self._snow_forcing is None or not self.validate():
            raise ValueError("Prepare forcing and provide physically feasible parameters first")
        self._refresh_snow_forcing()
        g = lambda name: float(self.get_scalar_parameter(name).value)
        s0 = g("s_initial_fraction") * g("s_max")
        g0 = g("g_initial_fraction") * g("g_max")
        snow0 = self.snow_config.initial_swe_in if self.snow_config is not None else 0.0
        liquid = self._snow_forcing["liquid_input_in"].to_numpy()
        swe = self._snow_forcing["SWE_in"].to_numpy()
        s, groundwater, excess, et, gwi = _gwi_B_core(
            liquid, self._temp, self._dt_hours,
            g("s_max"), g("ia_k_dep"), g("ia_k0"), g("ia_kT"), g("ia_theta"),
            g("ia_T_ref"), g("ia_T_freeze"), g("k_div"), g("k_G"), g("g_max"),
            s0, g0,
        )
        step_cfs = g("area_acres") * 3630.0 / (self._dt_hours * 3600)
        n = len(excess)
        routed = np.zeros(n)
        components = {}
        for i, tri in enumerate(self._tri, 1):
            tri.get_scalar_parameter("T").value = g(f"uh{i}_T")
            tri.get_scalar_parameter("K").value = g(f"uh{i}_K")
            weights = tri.get_kernel(self._dt_hours) * self._dt_hours
            full = fftconvolve(excess, weights) * g(f"R_{i}")
            tolerance = 1e-12 * max(1.0, float(excess.sum()))
            if full.min() < -tolerance:
                raise ArithmeticError("Material negative routed rainfall")
            full = np.maximum(full, 0.0)
            component = full[:n]
            routed += component
            if diagnostics:
                components[f"rtk{i}_cfs"] = component * step_cfs
        fraction = sum(g(f"R_{i}") for i in range(1, len(self._tri) + 1))
        routing_storage = np.cumsum(excess * fraction - routed)
        unrouted = excess * (1 - fraction)
        error = s0 + g0 + snow0 + np.cumsum(self._rain - et - gwi - routed - unrouted)
        error -= s + groundwater + routing_storage + swe
        sanitary = self.sanitary.evaluate(self)
        total_cfs = sanitary + (gwi + routed) * step_cfs
        self.budget = {
            "initial_storage_in": s0 + g0 + snow0, "rain_in": float(self._rain.sum()),
            "initial_snow_in": snow0, "final_snow_in": float(swe[-1]),
            "snowfall_in": float(self._snow_forcing["snowfall_in"].sum()),
            "melt_in": float(self._snow_forcing["melt_in"].sum()),
            "liquid_input_in": float(liquid.sum()),
            "et_in": float(et.sum()), "gwi_in": float(gwi.sum()),
            "rdii_in": float(routed.sum()), "unrouted_in": float(unrouted.sum()),
            "final_soil_in": float(s[-1]), "final_gwi_in": float(groundwater[-1]),
            "final_routing_in": float(routing_storage[-1]),
            "max_abs_error_in": float(np.abs(error).max()),
        }
        if self.budget["max_abs_error_in"] > 1e-8 or routing_storage.min() < -1e-8:
            raise ArithmeticError("Rainfall subsystem budget did not close within 1e-8 inch")
        out: pd.DataFrame | np.ndarray
        if diagnostics:
            release = groundwater * np.expm1(g("k_G") * self._dt_hours)
            spill = gwi - release
            out = pd.DataFrame({
                "datetime": self._df["datetime"], "S_in": s, "G_in": groundwater,
                "G_at_cap": _g_at_cap(groundwater, g("g_max"), g("k_G"), self._dt_hours),
                "excess_in": excess, "recharge_in": liquid - excess, "et_in": et,
                "gwi_release_in": release, "gwi_spill_in": spill,
                "unrouted_excess_in": unrouted, "routing_storage_in": routing_storage,
                "budget_error_in": error, "sanitary_cfs": sanitary, "gwi_cfs": gwi * step_cfs,
                "rdii_cfs": routed * step_cfs, "baseflow_cfs": sanitary + gwi * step_cfs,
                "total_cfs": total_cfs, **components,
                **{name: self._snow_forcing[name].to_numpy() for name in self._snow_forcing},
            })
        else:
            out = total_cfs
        self._state = ModelState.PREDICTED
        return out
