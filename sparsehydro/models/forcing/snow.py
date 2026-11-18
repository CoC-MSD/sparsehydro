"""Degree-day snow: the single snow implementation in sparsehydro.

Precipitation at or below the threshold temperature accumulates as snow water
equivalent (SWE); above it, the pack melts at ``ddf * (T - threshold) * dt / 24``
(capped at the SWE) and adds to the liquid input. :func:`snow_step` is the
one-step rule; :func:`snow_partition` applies it to a series, and
:class:`~sparsehydro.models.rdii.IAModel` calls the same step inside its
recurrence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd

from ..._numba import njit
from ...enums import ModelState
from ...parameters import FieldRecord, ScalarParameter
from ...registry import registry
from ..base import IModel
from ..unithydrograph.kernels import infer_dt_hours_from_values

__all__ = ["SnowConfig", "snow_step", "snow_partition", "DegreeDaySnow"]


def snow_step(precipitation, temperature, dt_hours, swe, snow_T, snow_ddf):
    """One degree-day step: return ``(swe, liquid, snowfall, melt)``."""
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


snow_step_nb = njit(cache=True, fastmath=False)(snow_step)


@dataclass(frozen=True)
class SnowConfig:
    """Immutable degree-day settings; SWE is water depth, not snow depth.

    :param threshold_c: Rain/snow threshold and melt base [degC].
    :param melt_factor_in_per_c_day: Degree-day melt factor [depth/degC/day].
    :param initial_swe_in: Initial snow water equivalent [depth].
    """

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
        storage, liquid[i], snowfall[i], melt[i] = snow_step_nb(
            precipitation[i], temperature[i], dt_hours, storage, threshold, factor)
        swe[i] = storage
    return swe, liquid, snowfall, melt


def snow_partition(
    precipitation_in: np.ndarray, temperature_c: np.ndarray, dt_hours: float,
    config: SnowConfig | None = None,
) -> pd.DataFrame:
    """Partition precipitation into liquid input and SWE; ``None`` passes it through.

    Run on the full forcing record including warmup, never separately by split.
    Only ``liquid_input_in`` should enter the soil/runoff physics.

    :returns: Columns ``snowfall_in``, ``rain_liquid_in``, ``melt_in``,
        ``liquid_input_in``, ``SWE_in``.
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


@registry.register
class DegreeDaySnow(IModel):
    """Degree-day snow component producing the liquid input for downstream stores.

    :param config: Snow settings; ``None`` disables snow (precipitation passes
        straight through as liquid input).
    :param calibrate: Register ``snow_T`` and ``snow_ddf`` as calibratable
        parameters (initial values from *config*).
    :param rainfall_col: Precipitation column read in :meth:`prepare`.
    :param temperature_col: Temperature column read in :meth:`prepare`.
    :param threshold_bounds: Bounds for ``snow_T`` [degC].
    :param melt_factor_bounds: Bounds for ``snow_ddf`` [depth/degC/day].
    """

    model_name: ClassVar[str] = "degree-day-snow"
    supports_array_predict: ClassVar[bool] = True
    OUTPUTS: ClassVar[tuple[str, ...]] = (
        "snowfall_in", "rain_liquid_in", "melt_in", "liquid_input_in", "SWE_in")

    def __init__(self, config: SnowConfig | None = SnowConfig(), *, calibrate: bool = False,
                 rainfall_col: str = "rainfall_in", temperature_col: str = "temperature_c",
                 threshold_bounds: tuple[float, float] = (-2.0, 4.0),
                 melt_factor_bounds: tuple[float, float] = (0.0, 0.5)) -> None:
        super().__init__()
        if calibrate and config is None:
            raise ValueError("Snow calibration requires an enabled SnowConfig")
        self.config = config
        self.calibrate = bool(calibrate)
        self.rainfall_col = rainfall_col
        self.temperature_col = temperature_col
        self.threshold_bounds = tuple(threshold_bounds)
        self.melt_factor_bounds = tuple(melt_factor_bounds)
        self._rain: np.ndarray | None = None
        self._temp: np.ndarray | None = None
        self._datetime: np.ndarray | None = None
        self._dt_hours = 1.0
        self._cache: tuple[SnowConfig | None, dict[str, np.ndarray]] | None = None

    @property
    def effective_config(self) -> SnowConfig | None:
        """Snow settings including the current calibrated coefficients."""
        if self.config is None or not self.calibrate or "snow_T" not in self._scalar_parameters:
            return self.config
        return SnowConfig(
            threshold_c=float(self._scalar_parameters["snow_T"].value),
            melt_factor_in_per_c_day=float(self._scalar_parameters["snow_ddf"].value),
            initial_swe_in=self.config.initial_swe_in,
        )

    def initialize(self) -> None:
        self._reset_registries()
        if self.calibrate:
            for name, value, (lower, upper), units, description in (
                ("snow_T", self.config.threshold_c, self.threshold_bounds, "degC",
                 "Rain/snow partition threshold and degree-day melt base"),
                ("snow_ddf", self.config.melt_factor_in_per_c_day, self.melt_factor_bounds,
                 "in/degC/day", "Degree-day melt factor; initial SWE remains fixed"),
            ):
                if not lower <= value <= upper:
                    raise ValueError(f"{name} initial value must lie within [{lower}, {upper}]")
                self.register_scalar_parameter(ScalarParameter(
                    name, value, lower, upper, units=units, description=description))
        self.register_output_field(FieldRecord("datetime", description="Simulation time step"))
        roles = {"SWE_in": "state", "liquid_input_in": "output"}  # the rest are fluxes
        for name in self.OUTPUTS:
            self.register_output_field(FieldRecord(name, units="in", role=roles.get(name, "flux")))
        self._cache = None
        self._state = ModelState.INITIALIZED

    def validate(self) -> bool:
        ok = self.parameters_valid()
        if ok:
            self._state = ModelState.VALIDATED
        return ok

    def prepare(self, data: pd.DataFrame, **_: object) -> None:
        self._datetime = pd.to_datetime(data["datetime"]).to_numpy()
        self._rain = data[self.rainfall_col].to_numpy(dtype=float)
        self._temp = data[self.temperature_col].to_numpy(dtype=float)
        self._dt_hours = infer_dt_hours_from_values(self._datetime)
        self._cache = None
        self._state = ModelState.PREPARED

    def partition(self) -> dict[str, np.ndarray]:
        """Current partition (recomputed only when the settings change)."""
        if self._rain is None:
            raise RuntimeError("Call prepare() before predict().")
        config = self.effective_config
        if self._cache is None or self._cache[0] != config:
            frame = snow_partition(self._rain, self._temp, self._dt_hours, config)
            self._cache = (config, {c: frame[c].to_numpy() for c in frame.columns})
        return self._cache[1]

    def predict_arrays(self) -> dict[str, np.ndarray]:
        out = {"datetime": self._datetime, **self.partition()}
        self._state = ModelState.PREDICTED
        return out

    def predict(self) -> pd.DataFrame:
        return pd.DataFrame(self.predict_arrays())

    def finalize(self) -> None:
        self._rain = self._temp = None
        self._cache = None
        self._state = ModelState.FINALIZED
