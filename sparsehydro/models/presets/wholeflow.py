"""Whole-flow presets: snow -> soil/groundwater store -> RDII routing + sanitary baseline.

``total_cfs = sanitary + GWI + RDII``. A whole-flow model is a
:class:`~sparsehydro.models.composite.CompositeModel` of registered components:

=============  ==========================================  ======================
Component      Class                                       Provides
=============  ==========================================  ======================
``snow``       :class:`~sparsehydro.models.forcing.DegreeDaySnow`  liquid input, SWE
``store``      :class:`~sparsehydro.models.abstraction.SoilGroundwaterStore`  excess, GWI
``rdii``       :class:`~sparsehydro.models.rdii.RDIIModel` with RTK triangles  RDII
``baseline``   a sanitary baseline component               sanitary flow
=============  ==========================================  ======================

:class:`CompactWholeFlowModel` uses a fixed sanitary template;
:class:`PatternWholeFlowModel` uses
fitted weekly patterns (optionally with an observed background). Both keep
the parameter names, order, bounds and outputs of the original classes, check
the forcing strictly, and close the whole-system water budget every
prediction (``budget``).

The same model can be written as configuration; these classes are presets.

:class:`PatternWholeFlowModel` uses a
:class:`~sparsehydro.models.baseflow.CenteredWeeklyPatternModel` baseline: a
calibrated mean (or a frozen observed background) plus scaled hourly and
weekday anomalies estimated beforehand on training dry weather; IA
coefficients are calibrated and initial storage is fixed at zero.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..abstraction import ExternalExcess, SoilGroundwaterStore
from ..baseflow import (
    CenteredWeeklyPattern,
    CenteredWeeklyPatternModel,
    ObservedBackground,
    ObservedBackgroundModel,
    SanitaryTemplate,
    SanitaryTemplateModel,
)
from ..base import IModel
from ..composite import CompositeModel
from ..forcing import DegreeDaySnow, SnowConfig
from ..rdii import RDIIModel, default_rtk_params
from ..unithydrograph.rtk import RTKTriangle
from ...registry import registry

__all__ = ["WholeFlowPreset", "CompactWholeFlowModel", "PatternWholeFlowModel", "pattern_init_rtk",
           "J_INIT_RTK", "J_T_BOUNDS", "J_K_BOUNDS", "SUPER_SLOW_RTK", "SUPER_SLOW_T_BOUNDS"]

# Initial (R, T[hr], K) and search bounds for fast / medium / slow / very slow pathways.
J_INIT_RTK = [(0.03, 1.0, 2.0), (0.02, 8.0, 2.5), (0.01, 48.0, 3.0)]
J_T_BOUNDS = [(0.1, 6.0), (2.0, 48.0), (12.0, 120.0)]
J_K_BOUNDS = (1.05, 4.0)
SUPER_SLOW_RTK = (0.005, 168.0, 3.0)
SUPER_SLOW_T_BOUNDS = (96.0, 720.0)

_STORE_PARAMETERS = ("s_max", "ia_k_dep", "ia_k0", "ia_kT", "ia_theta", "ia_T_ref", "ia_T_freeze",
                     "k_div", "k_G", "g_max", "area_acres")
_SNOW_OUTPUTS = ("snowfall_in", "rain_liquid_in", "melt_in", "liquid_input_in", "SWE_in")


class WholeFlowPreset(CompositeModel):
    """Common machinery of the whole-flow presets (not registered).

    :param baseline: Sanitary baseline component.
    :param area_acres: Contributing area.
    :param dt_hours: Model time step; forcing must be on this regular grid.
    :param snow_config: Snow settings (``None``: no snow).
    :param calibrate_snow: Calibrate ``snow_T`` and ``snow_ddf``.
    :param groundwater_enabled: Use the deep groundwater store.
    :param init_rtk: ``(R, T, K)`` per RTK pathway.
    :param background: Optional component providing ``background_cfs`` to the
        baseline's ``level_cfs`` input.
    :param store_overrides: Extra parameter overrides for store parameters.
    """

    def __init__(self, baseline: IModel, area_acres: float, dt_hours: float, *,
                 snow_config: SnowConfig | None, calibrate_snow: bool, groundwater_enabled: bool,
                 init_rtk, background: IModel | None = None,
                 store_overrides: Mapping[str, Mapping[str, Any]] | None = None) -> None:
        if not (np.isfinite(area_acres) and area_acres > 0 and np.isfinite(dt_hours) and dt_hours > 0):
            raise ValueError("Area and timestep must be positive and finite")
        if calibrate_snow and snow_config is None:
            raise ValueError("Snow calibration requires an enabled SnowConfig")
        if not isinstance(groundwater_enabled, bool):
            raise ValueError("groundwater_enabled must be a boolean")
        init_rtk = list(init_rtk)
        if not init_rtk:
            raise ValueError("Whole-flow models require at least one RTK triangle")
        self._dt_hours = float(dt_hours)
        self.groundwater_enabled = groundwater_enabled
        self._calibrate_snow = calibrate_snow
        self._snow_config = snow_config
        self._init = init_rtk
        self.budget: dict[str, float] = {}

        snow = DegreeDaySnow(snow_config, calibrate=calibrate_snow,
                             threshold_bounds=(-2.0, 3.0), melt_factor_bounds=(0.005, 0.5))
        store = SoilGroundwaterStore(area_acres, groundwater=groundwater_enabled)
        rdii = RDIIModel(ia_model=ExternalExcess(),
                         uh_components=[RTKTriangle(R=R, T=T, K=K) for R, T, K in init_rtk],
                         component_outputs=True)
        components: dict[str, IModel] = {"snow": snow, "store": store, "rdii": rdii}
        bindings = {"store.liquid_input_in": "snow.liquid_input_in",
                    "rdii.p_excess_in": "store.p_excess_in"}
        if background is not None:
            components["background"] = background
            bindings["baseline.level_cfs"] = "background.background_cfs"
        components["baseline"] = baseline

        t_bounds = (*J_T_BOUNDS, SUPER_SLOW_T_BOUNDS)
        overrides: dict[str, dict[str, Any]] = {}
        for i in range(1, len(init_rtk) + 1):
            lo, hi = t_bounds[min(i - 1, len(t_bounds) - 1)]
            overrides[f"uh{i}_T"] = {"lower_bound": lo, "upper_bound": hi}
            overrides[f"uh{i}_K"] = {"lower_bound": J_K_BOUNDS[0], "upper_bound": J_K_BOUNDS[1]}
        for name, opts in (store_overrides or {}).items():
            overrides.setdefault(name, {}).update(opts)

        self._baseline_parameters = self._baseline_parameter_names(baseline)
        order = [*_STORE_PARAMETERS, *self._baseline_parameters]
        for i in range(1, len(init_rtk) + 1):
            order += [f"R_{i}", f"uh{i}_T", f"uh{i}_K"]
        if calibrate_snow:
            order += ["snow_T", "snow_ddf"]
        order += ["s_initial_fraction", "g_initial_fraction"]

        outputs: dict[str, Any] = {
            "S_in": "store.S_in", "G_in": "store.G_in", "G_at_cap": "store.G_at_cap",
            "excess_in": "store.p_excess_in", "recharge_in": "store.recharge_in",
            "et_in": "store.et_in", "gwi_release_in": "store.gwi_release_in",
            "gwi_spill_in": "store.gwi_spill_in", "sanitary_cfs": "baseline.sanitary_cfs",
            "gwi_cfs": "store.gwi_cfs", "rdii_cfs": "rdii.rdii_cfs", "rdii_in": "rdii.rdii_in",
            "gwi_in": "store.gwi_in",
            "baseflow_cfs": ["baseline.sanitary_cfs", "store.gwi_cfs"],
            "total_cfs": ["baseline.sanitary_cfs", "store.gwi_cfs", "rdii.rdii_cfs"],
            **{f"rtk{i}_cfs": f"rdii.rdii_component_{i}" for i in range(1, len(init_rtk) + 1)},
            **{name: f"snow.{name}" for name in _SNOW_OUTPUTS},
        }
        super().__init__(components, bindings=bindings, outputs=outputs,
                         prefixes={name: "" for name in components},
                         shared={"area_acres": ["store.area_acres", "rdii.area_acres"]},
                         parameter_overrides=overrides, parameter_order=order)

    @staticmethod
    def _baseline_parameter_names(baseline: IModel) -> list[str]:
        baseline.initialize()
        return list(baseline.scalar_parameter_names)

    # -- compatibility properties ------------------------------------------
    @property
    def n_triangles(self) -> int:
        return len(self._init)

    @property
    def snow_config(self) -> SnowConfig | None:
        """Effective snow settings, including calibrated coefficients."""
        return self.component("snow").effective_config

    @property
    def _snow_forcing(self) -> pd.DataFrame | None:
        snow = self.component("snow")
        if snow._rain is None:
            return None
        return pd.DataFrame(snow.partition())

    # -- lifecycle ----------------------------------------------------------
    def validate(self) -> bool:
        return super().validate() and all(g <= 0 for g in self.inequality_constraints())

    def prepare(self, data: pd.DataFrame, **child_kwargs: Mapping[str, Any]) -> None:
        """Check the forcing strictly, then prepare the components.

        :raises ValueError: Unless ``datetime``, ``rainfall_in`` and
            ``temperature_c`` are complete, rain is nonnegative, and timestamps
            are unique, ascending and on the configured time step.
        """
        required = ["datetime", "rainfall_in", "temperature_c"]
        if not set(required).issubset(data):
            raise ValueError(f"Prediction requires only {required}")
        keep = required + [c for c in ("background_required",) if c in data]
        df = data[keep].reset_index(drop=True).copy()
        t = pd.DatetimeIndex(df["datetime"])
        if len(t) < 2 or t.hasnans or not t.is_monotonic_increasing or t.has_duplicates:
            raise ValueError("Forcing must have unique ascending timestamps")
        if not np.allclose(np.diff(t.as_unit("ns").asi8) / 3.6e12, self._dt_hours):
            raise ValueError("Forcing must be on the configured regular timestep")
        rain = df["rainfall_in"].to_numpy(float)
        temp = df["temperature_c"].to_numpy(float)
        if not np.all(np.isfinite(rain)) or np.any(rain < 0) or not np.all(np.isfinite(temp)):
            raise ValueError("Rain and temperature must be complete; rain must be nonnegative")
        self._rain = rain
        super().prepare(df, **child_kwargs)

    def predict_arrays(self) -> dict[str, np.ndarray]:
        """All components plus the whole-system budget columns.

        :raises ValueError: Before :meth:`prepare` or with infeasible parameters.
        :raises ArithmeticError: If the water budget does not close within 1e-8 in.
        """
        if self._datetime is None or not self.validate():
            raise ValueError("Prepare forcing and provide physically feasible parameters first")
        out = super().predict_arrays()
        store, snow = self.component("store"), self.component("snow")
        s0, g0 = store.initial_storage
        config = snow.effective_config
        snow0 = config.initial_swe_in if config is not None else 0.0
        excess, routed = out["excess_in"], out.pop("rdii_in")
        fraction = sum(self._scalar_parameters[f"R_{i}"].value for i in range(1, self.n_triangles + 1))
        routing_storage = np.cumsum(excess * fraction - routed)
        unrouted = excess * (1 - fraction)
        gwi = out.pop("gwi_in")
        error = s0 + g0 + snow0 + np.cumsum(self._rain - out["et_in"] - gwi - routed - unrouted)
        error -= out["S_in"] + out["G_in"] + routing_storage + out["SWE_in"]
        self.budget = {
            "initial_storage_in": s0 + g0 + snow0, "rain_in": float(self._rain.sum()),
            "initial_snow_in": snow0, "final_snow_in": float(out["SWE_in"][-1]),
            "snowfall_in": float(out["snowfall_in"].sum()), "melt_in": float(out["melt_in"].sum()),
            "liquid_input_in": float(out["liquid_input_in"].sum()),
            "et_in": float(out["et_in"].sum()), "gwi_in": float(gwi.sum()),
            "rdii_in": float(routed.sum()), "unrouted_in": float(unrouted.sum()),
            "final_soil_in": float(out["S_in"][-1]), "final_gwi_in": float(out["G_in"][-1]),
            "final_routing_in": float(routing_storage[-1]),
            "max_abs_error_in": float(np.abs(error).max()),
        }
        if self.budget["max_abs_error_in"] > 1e-8 or routing_storage.min() < -1e-8:
            raise ArithmeticError("Rainfall subsystem budget did not close within 1e-8 inch")
        ordered = {name: out[name] for name in (
            "datetime", "S_in", "G_in", "G_at_cap", "excess_in", "recharge_in", "et_in",
            "gwi_release_in", "gwi_spill_in")}
        ordered.update(unrouted_excess_in=unrouted, routing_storage_in=routing_storage,
                       budget_error_in=error)
        ordered.update({name: out[name] for name in out if name not in ordered})
        return ordered

    def predict_total_cfs(self) -> np.ndarray:
        """Total flow only (the same computation and budget check as :meth:`predict`)."""
        return self.predict_arrays()["total_cfs"]


@registry.register
class CompactWholeFlowModel(WholeFlowPreset):
    """Whole-flow model with a fixed sanitary template and a calibrated level.

    ``ia_k_dep`` and ``ia_theta`` are fixed; initial S and G are calibrated
    fractions of capacity (G fixed at 0 without groundwater).

    :param template: Normalised hourly x weekday sanitary template.
    :param area_acres: Contributing area.
    :param dt_hours: Model time step [hr].
    :param level_cfs: Initial sanitary level [cfs].
    :param level_upper_cfs: Upper bound of the sanitary level [cfs].
    """

    model_name = "wholeflow-compact"

    def __init__(self, template: SanitaryTemplate, area_acres: float, dt_hours: float,
                 level_cfs: float, level_upper_cfs: float, *, snow_config: SnowConfig | None = None,
                 calibrate_snow: bool = False, groundwater_enabled: bool = True,
                 init_rtk: tuple[tuple[float, float, float], ...] | None = None):
        if not (0 <= level_cfs <= level_upper_cfs and np.isfinite(level_upper_cfs)):
            raise ValueError("Invalid sanitary level bounds")
        self.template = template
        store_overrides = {
            "ia_k_dep": {"calibrate": False}, "ia_theta": {"calibrate": False},
            "s_initial_fraction": {"value": 0.5, "calibrate": True},
            "g_initial_fraction": ({"value": 0.5, "calibrate": True} if groundwater_enabled
                                   else {"value": 0.0, "calibrate": False}),
        }
        super().__init__(SanitaryTemplateModel(template, level_cfs, level_upper_cfs),
                         area_acres, dt_hours, snow_config=snow_config,
                         calibrate_snow=calibrate_snow, groundwater_enabled=groundwater_enabled,
                         init_rtk=J_INIT_RTK if init_rtk is None else init_rtk,
                         store_overrides=store_overrides)


def pattern_init_rtk(n_triangles: int) -> tuple[tuple[float, float, float], ...]:
    """Initial ``(R, T, K)`` for *n_triangles* pathways (fast, medium, slow, very slow, ...)."""
    if isinstance(n_triangles, bool) or not isinstance(n_triangles, int) or n_triangles < 1:
        raise ValueError("n_triangles must be a positive integer")
    init_rtk = tuple((*J_INIT_RTK, SUPER_SLOW_RTK)[:n_triangles])
    if n_triangles > len(init_rtk):
        extra_rtk = default_rtk_params(
            1, n_triangles - len(init_rtk),
            T_min=SUPER_SLOW_RTK[1], T_max=SUPER_SLOW_T_BOUNDS[1],
            R_total=SUPER_SLOW_RTK[0], K_min=SUPER_SLOW_RTK[2], K_max=SUPER_SLOW_RTK[2],
        )[0]
        # Keep log-spaced endpoints inside their bounds despite floating-point rounding.
        init_rtk += tuple((R, float(np.clip(T, *SUPER_SLOW_T_BOUNDS)), K) for R, T, K in extra_rtk)
    return init_rtk


@registry.register
class PatternWholeFlowModel(WholeFlowPreset):
    """Centered weekly sanitary pattern + snow + soil/groundwater store + RTK RDII.

    :param pattern: Fitted hourly/weekday anomalies.
    :param area_acres: Contributing area.
    :param dt_hours: Model time step [hr].
    :param sanitary_mean_cfs: Initial sanitary mean [cfs].
    :param sanitary_upper_cfs: Upper bound of the sanitary mean [cfs].
    :param n_triangles: Number of RTK pathways.
    :param observed_background: A frozen :class:`ObservedBackground` that
        replaces the calibrated mean (requires ``groundwater_enabled=False``).
        ``sanitary_cfs`` then includes unidentified slow inflow.
    """

    model_name = "wholeflow-pattern-joint"

    def __init__(
        self, pattern: CenteredWeeklyPattern, area_acres: float, dt_hours: float,
        sanitary_mean_cfs: float, sanitary_upper_cfs: float, *, snow_config: SnowConfig | None = None,
        calibrate_snow: bool = False,
        groundwater_enabled: bool = True, n_triangles: int = 3,
        observed_background: ObservedBackground | None = None,
        annual_seasonality: bool = False,
    ):
        if annual_seasonality is not False:
            raise ValueError("Calendar annual_seasonality has been removed; use a training-derived observed_background")
        init_rtk = pattern_init_rtk(n_triangles)
        if observed_background is not None:
            if not isinstance(observed_background, ObservedBackground):
                raise ValueError("observed_background must be an ObservedBackground")
            if groundwater_enabled:
                raise ValueError("Observed background and physical GWI are mutually exclusive")
        if not (0 <= sanitary_mean_cfs <= sanitary_upper_cfs and np.isfinite(sanitary_upper_cfs)):
            raise ValueError("Invalid sanitary level bounds")
        self.pattern = pattern
        self.observed_background = observed_background
        background = (ObservedBackgroundModel(observed_background)
                      if observed_background is not None else None)
        baseline = CenteredWeeklyPatternModel(
            pattern, sanitary_mean_cfs, sanitary_upper_cfs,
            level_from_input=observed_background is not None,
            level_floor_cfs=(min(observed_background.flow_cfs)
                             if observed_background is not None else None))
        super().__init__(baseline, area_acres, dt_hours, snow_config=snow_config,
                         calibrate_snow=calibrate_snow, groundwater_enabled=groundwater_enabled,
                         init_rtk=init_rtk, background=background)
        self._background_component = background
        if observed_background is not None:
            self._outputs_spec.update({
                "slow_background_cfs": "background.background_cfs",
                "background_supported": "background.background_supported",
                "background_endpoint_held": "background.background_endpoint_held",
            })

    def prepare(self, data, **child_kwargs) -> None:
        """Check observed-background support first, then prepare as usual."""
        if self._background_component is not None:
            self._background_component.check_support(data)
        super().prepare(data, **child_kwargs)
