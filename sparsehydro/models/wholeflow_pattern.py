"""Whole-flow preset with fitted weekly sanitary patterns.

:class:`PatternWholeFlowModel` is a
:class:`~sparsehydro.models.wholeflow_compact.WholeFlowPreset` whose baseline is
a :class:`~sparsehydro.models.baseflow.CenteredWeeklyPatternModel`: a calibrated
mean (or a frozen observed background) plus scaled hourly and weekday anomalies
estimated beforehand on training dry weather. IA coefficients are calibrated and
initial storage is fixed at zero (for forcing-only spin-up).

The pattern dataclasses and estimators live in :mod:`sparsehydro.models.baseflow`
and are re-exported here for existing imports.
"""

from __future__ import annotations

import numpy as np

from sparsehydro.models.baseflow import (  # noqa: F401 - re-exported
    CenteredWeeklyPattern,
    CenteredWeeklyPatternModel,
    ObservedBackground,
    ObservedBackgroundModel,
    PatternEstimate,
    SanitaryTemplate,
    estimate_observed_background,
    estimate_sanitary_shapes,
    validate_background_options,
)
from sparsehydro.models.forcing import SnowConfig
from sparsehydro.models.rdii import default_rtk_params
from sparsehydro.models.wholeflow_compact import (
    J_INIT_RTK, SUPER_SLOW_RTK, SUPER_SLOW_T_BOUNDS, WholeFlowPreset,
)
from sparsehydro.registry import registry

__all__ = ["PatternWholeFlowModel", "CenteredWeeklyPattern", "ObservedBackground", "PatternEstimate",
           "estimate_observed_background", "estimate_sanitary_shapes", "validate_background_options"]


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
