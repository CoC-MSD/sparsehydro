"""Baseline (sanitary / dry-weather) flow components.

* :class:`SanitaryTemplateModel` -- a fixed normalised hour x weekday template
  times a calibrated level. With ``level_from_input=True`` the level is an
  input bound from :class:`ObservedBackgroundModel` (for example a daily level
  from :func:`~.fitting.estimate_sanitary_level`) times a calibrated scale.
* :class:`CenteredWeeklyPatternModel` -- a level plus scaled hourly and weekday
  anomalies estimated beforehand (see :mod:`.fitting`). The level is either a
  calibrated mean or an input bound from :class:`ObservedBackgroundModel`.
* :class:`ObservedBackgroundModel` -- a frozen training-dry background series,
  for retrospective analysis without a groundwater store.

Anything learned from observed flow is passed in (constructor or ``prepare``);
none of these components read observations while predicting.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd

from ...enums import ModelState
from ...parameters import ConstraintRecord, FieldRecord, ScalarParameter
from ...registry import registry
from ..base import IModel
from ._shapes import CenteredWeeklyPattern, ObservedBackground, SanitaryTemplate

__all__ = ["SanitaryTemplateModel", "CenteredWeeklyPatternModel", "ObservedBackgroundModel"]


class _BaselineBase(IModel):
    supports_array_predict: ClassVar[bool] = True

    def __init__(self) -> None:
        super().__init__()
        self._datetime: pd.DatetimeIndex | None = None

    def validate(self) -> bool:
        ok = self.parameters_valid()
        if ok:
            self._state = ModelState.VALIDATED
        return ok

    def _p(self, name: str) -> float:
        return float(self._scalar_parameters[name].value)

    def predict(self) -> pd.DataFrame:
        return pd.DataFrame(self.predict_arrays())

    def finalize(self) -> None:
        self._datetime = None
        self._state = ModelState.FINALIZED


@registry.register
class SanitaryTemplateModel(_BaselineBase):
    """``sanitary_cfs = sanitary_level_cfs * template(hour, weekday)``.

    With ``level_from_input=True`` the daily dry-weather level comes from the
    bound input ``level_cfs``, and one calibrated scale multiplies it:
    ``sanitary_cfs = sanitary_level_scale * level_cfs * template(hour, weekday)``.

    :param template: Normalised 24 hourly and 7 weekday factors (each mean 1).
    :param level_cfs: Initial level [cfs] (not used when the level is an input).
    :param level_upper_cfs: Upper bound of the level [cfs] (not used when the
        level is an input).
    :param level_from_input: Take the level from the bound input ``level_cfs``
        (for example an :class:`ObservedBackgroundModel` built from
        :func:`~.fitting.estimate_sanitary_level`) and calibrate
        ``sanitary_level_scale`` (start 1, bounds 0.5 to 1.5) instead of
        ``sanitary_level_cfs``.

    Example: a level input of 0.2 cfs, a scale of 1.1 and a template
    multiplier of 1.5 give 1.1 x 0.2 x 1.5 = 0.33 cfs.
    """

    model_name: ClassVar[str] = "sanitary-template"

    def __init__(
        self,
        template: SanitaryTemplate | None = None,
        level_cfs: float = 0.5,
        level_upper_cfs: float = 5.0,
        *,
        level_from_input: bool = False,
    ) -> None:
        super().__init__()
        self.template = template if template is not None else SanitaryTemplate((1.0,) * 24, (1.0,) * 7)
        if not (0 <= level_cfs <= level_upper_cfs and np.isfinite(level_upper_cfs)):
            raise ValueError("Invalid sanitary level bounds")
        self._level, self._upper = float(level_cfs), float(level_upper_cfs)
        self.level_from_input = bool(level_from_input)
        self._values: np.ndarray | None = None
        self._level_input: np.ndarray | None = None

    @property
    def input_field_names(self) -> list[str]:
        return ["level_cfs"] if self.level_from_input else []

    def initialize(self) -> None:
        self._reset_registries()
        if self.level_from_input:
            self.register_scalar_parameter(ScalarParameter(
                "sanitary_level_scale", 1.0, 0.5, 1.5, units="-",
                description="Scale on the input daily dry-weather level"))
        else:
            self.register_scalar_parameter(ScalarParameter(
                "sanitary_level_cfs", self._level, 0.0, self._upper, units="cfs",
                description="Mean sanitary level over a reference week"))
        self.register_output_field(FieldRecord("datetime"))
        self.register_output_field(FieldRecord("sanitary_cfs", units="cfs", description="Sanitary flow"))
        self._state = ModelState.INITIALIZED

    def prepare(self, data: pd.DataFrame, **_: object) -> None:
        self._datetime = pd.DatetimeIndex(data["datetime"])
        self._values = self.template.predict(self._datetime)
        self._level_input = np.zeros(len(data)) if self.level_from_input else None
        self._state = ModelState.PREPARED

    def set_input(self, name: str, values: np.ndarray) -> None:
        if name not in self.input_field_names:
            raise KeyError(f"{type(self).__name__} inputs are {self.input_field_names}; got {name!r}")
        self._level_input = np.asarray(values, dtype=float)

    def predict_arrays(self) -> dict[str, np.ndarray]:
        if self._values is None:
            raise RuntimeError("Prepare the sanitary template before prediction")
        if self.level_from_input:
            if self._level_input is None:
                raise RuntimeError("Prepare the sanitary template before prediction")
            level = self._p("sanitary_level_scale") * self._level_input
        else:
            level = self._p("sanitary_level_cfs")
        self._state = ModelState.PREDICTED
        return {"datetime": self._datetime.to_numpy(), "sanitary_cfs": level * self._values}


@registry.register
class CenteredWeeklyPatternModel(_BaselineBase):
    """``sanitary_cfs = level + hour_scale * hourly + weekday_scale * weekday``.

    The anomalies come from a :class:`CenteredWeeklyPattern` estimated on
    training dry weather (:func:`~.fitting.estimate_sanitary_shapes`).

    :param pattern: Fitted pattern (may instead be passed to :meth:`prepare`).
    :param mean_cfs: Initial calibrated mean level [cfs].
    :param mean_upper_cfs: Upper bound of the mean [cfs].
    :param level_from_input: Take the level from the bound input ``level_cfs``
        (e.g. an observed background) instead of a ``sanitary_mean_cfs``
        parameter.
    :param level_floor_cfs: Smallest input level, used by the
        ``sanitary_nonnegative`` constraint when the level is an input.

    Constraint ``sanitary_nonnegative``: the smallest level plus the smallest
    scaled anomalies must be >= 0 (no clipping is applied).
    """

    model_name: ClassVar[str] = "centered-weekly-pattern"

    def __init__(self, pattern: CenteredWeeklyPattern | None = None, mean_cfs: float = 0.5,
                 mean_upper_cfs: float = 5.0, *, level_from_input: bool = False,
                 level_floor_cfs: float | None = None) -> None:
        super().__init__()
        self.pattern = pattern
        self._mean, self._upper = float(mean_cfs), float(mean_upper_cfs)
        self.level_from_input = bool(level_from_input)
        self.level_floor_cfs = level_floor_cfs
        self._hour: np.ndarray | None = None
        self._weekday: np.ndarray | None = None
        self._level_input: np.ndarray | None = None

    @property
    def input_field_names(self) -> list[str]:
        return ["level_cfs"] if self.level_from_input else []

    def initialize(self) -> None:
        self._reset_registries()
        if not self.level_from_input:
            self.register_scalar_parameter(ScalarParameter(
                "sanitary_mean_cfs", self._mean, 0.0, self._upper, units="cfs",
                description="Sanitary mean over an equally weighted reference week"))
        for name, description in (("sanitary_hour_scale", "Amplitude of the hourly anomaly"),
                                  ("sanitary_weekday_scale", "Amplitude of the weekday anomaly")):
            self.register_scalar_parameter(ScalarParameter(
                name, 1.0, 0.0, 3.0, units="-", description=description))
        self.register_inequality_constraint(ConstraintRecord(
            "sanitary_nonnegative", "Minimum level plus weekly anomalies >= 0, without clipping"))
        self.register_output_field(FieldRecord("datetime"))
        self.register_output_field(FieldRecord("sanitary_cfs", units="cfs", description="Sanitary flow"))
        self._state = ModelState.INITIALIZED

    def prepare(self, data: pd.DataFrame, *, pattern: CenteredWeeklyPattern | None = None,
                level_floor_cfs: float | None = None, **_: object) -> None:
        if pattern is not None:
            self.pattern = pattern
        if level_floor_cfs is not None:
            self.level_floor_cfs = float(level_floor_cfs)
        if self.pattern is None:
            raise ValueError("CenteredWeeklyPatternModel needs a fitted pattern")
        self._datetime = pd.DatetimeIndex(data["datetime"])
        self._hour, self._weekday = self.pattern.components(self._datetime)
        self._level_input = np.zeros(len(data)) if self.level_from_input else None
        self._state = ModelState.PREPARED

    def set_input(self, name: str, values: np.ndarray) -> None:
        if name not in self.input_field_names:
            raise KeyError(f"{type(self).__name__} inputs are {self.input_field_names}; got {name!r}")
        self._level_input = np.asarray(values, dtype=float)

    def inequality_constraints(self) -> list[float]:
        if self.pattern is None:
            return [0.0]
        if self.level_from_input:
            if self.level_floor_cfs is None:
                raise ValueError("level_floor_cfs is required when the level is an input")
            level = float(self.level_floor_cfs)
        else:
            level = self._p("sanitary_mean_cfs")
        minimum = (level + self._p("sanitary_hour_scale") * min(self.pattern.hourly)
                   + self._p("sanitary_weekday_scale") * min(self.pattern.weekday))
        return [-minimum]

    def predict_arrays(self) -> dict[str, np.ndarray]:
        if self._hour is None:
            raise RuntimeError("Prepare the sanitary pattern before prediction")
        level = self._level_input if self.level_from_input else self._p("sanitary_mean_cfs")
        baseline = (level + self._p("sanitary_hour_scale") * self._hour
                    + self._p("sanitary_weekday_scale") * self._weekday)
        self._state = ModelState.PREDICTED
        return {"datetime": self._datetime.to_numpy(), "sanitary_cfs": baseline}


@registry.register
class ObservedBackgroundModel(_BaselineBase):
    """A frozen observed background (retrospective interpolation, not forecasting).

    :param background: Fitted :class:`ObservedBackground` (may instead be passed
        to :meth:`prepare`).

    In :meth:`prepare`, rows flagged in an optional boolean ``background_required``
    column must have background support; otherwise a ``ValueError`` explains
    where support is missing.
    """

    model_name: ClassVar[str] = "observed-background"

    def __init__(self, background: ObservedBackground | None = None) -> None:
        super().__init__()
        if background is not None and not isinstance(background, ObservedBackground):
            raise ValueError("background must be an ObservedBackground")
        self.background = background
        self._values: np.ndarray | None = None

    def initialize(self) -> None:
        self._reset_registries()
        self.register_output_field(FieldRecord("datetime"))
        self.register_output_field(FieldRecord(
            "background_cfs", units="cfs", description="Frozen training-dry background"))
        self.register_output_field(FieldRecord(
            "background_supported", units="-", description="Interpolation support or bounded hold"))
        self.register_output_field(FieldRecord(
            "background_endpoint_held", units="-", description="Constant source-boundary value"))
        self._state = ModelState.INITIALIZED

    def prepare(self, data: pd.DataFrame, *, background: ObservedBackground | None = None,
                **_: object) -> None:
        if background is not None:
            self.background = background
        values = self.check_support(data)
        t = pd.DatetimeIndex(data["datetime"])
        self._datetime = t
        self._values = values
        self._held = self.background.endpoint_hold_mask(t)
        self._state = ModelState.PREPARED

    def check_support(self, data: pd.DataFrame) -> np.ndarray:
        """Evaluate the background on *data* and check rows flagged as required.

        :raises ValueError: Explaining where required rows lack support.
        """
        if self.background is None:
            raise ValueError("ObservedBackgroundModel needs a fitted ObservedBackground")
        t = pd.DatetimeIndex(data["datetime"])
        values = self.background.evaluate(t)
        required = (data["background_required"].to_numpy()
                    if "background_required" in data else np.ones(len(data), dtype=bool))
        if required.dtype != bool or required.shape != values.shape:
            raise ValueError("background_required must be an aligned boolean column")
        missing = required & ~np.isfinite(values)
        if missing.any():
            first = data.loc[missing, "datetime"].iloc[0]
            before = int((missing & (t < self.background.start)).sum())
            after = int((missing & (t > self.background.end)).sum())
            interior = int(missing.sum()) - before - after
            raise ValueError(
                f"Observed background lacks support for {int(missing.sum())} required rows, first {first}. "
                f"Before source support: {before}; interior gaps: {interior}; after source support: {after}. "
                f"Endpoint hold limit: {self.background.endpoint_hold_days} days. "
                "Restrict the scoring period to supported interpolation, supply more training dry data, "
                "or use physical GWI with seasonal_baseline='off'; no extrapolation beyond the endpoint "
                "hold limit or filling of long interior gaps is performed."
            )
        return values

    def predict_arrays(self) -> dict[str, np.ndarray]:
        if self._values is None:
            raise RuntimeError("Prepare the observed background before prediction")
        self._state = ModelState.PREDICTED
        return {"datetime": self._datetime.to_numpy(), "background_cfs": self._values,
                "background_supported": np.isfinite(self._values),
                "background_endpoint_held": self._held}
