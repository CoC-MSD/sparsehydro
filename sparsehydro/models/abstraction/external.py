"""Rainfall excess supplied from outside the model.

:class:`ExternalExcess` is an abstraction stage with no parameters: its excess
series is either read from a data column or handed in by another model through
:meth:`set_input`. It lets :class:`~sparsehydro.models.rdii.RDIIModel` route
excess produced by a sibling component inside a
:class:`~sparsehydro.models.composite.CompositeModel` (for example a
soil/groundwater store that also produces groundwater infiltration).
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd

from ...enums import ModelState
from ...parameters import FieldRecord
from ...registry import registry
from ..base import IModel


@registry.register
class ExternalExcess(IModel):
    """Pass-through rainfall excess.

    :param units: ``"imperial"`` (``p_excess_in``) or ``"metric"`` (``p_excess_mm``).
    :param source_column: Optional data column to read in :meth:`prepare`. When
        omitted the excess starts at zero and is expected via :meth:`set_input`.
    """

    model_name: ClassVar[str] = "external-excess"
    supports_array_predict: ClassVar[bool] = True

    def __init__(self, units: str = "imperial", source_column: str | None = None) -> None:
        super().__init__()
        if units not in ("imperial", "metric"):
            raise ValueError(f"units must be 'imperial' or 'metric'; got {units!r}")
        self.units = units
        self.excess_column = "p_excess_in" if units == "imperial" else "p_excess_mm"
        self.source_column = source_column
        self._excess: np.ndarray | None = None
        self._datetime: np.ndarray | None = None

    @property
    def input_field_names(self) -> list[str]:
        return [self.excess_column]

    def initialize(self) -> None:
        self._reset_registries()
        self.register_output_field(FieldRecord("datetime", description="Simulation time step"))
        self.register_output_field(FieldRecord(
            self.excess_column, units="in" if self.units == "imperial" else "mm",
            description="Rainfall excess supplied from outside the model"))
        self._state = ModelState.INITIALIZED

    def validate(self) -> bool:
        self._state = ModelState.VALIDATED
        return True

    def prepare(self, data: pd.DataFrame, **_: object) -> None:
        self._datetime = pd.to_datetime(data["datetime"]).to_numpy()
        if self.source_column is not None:
            self._excess = data[self.source_column].to_numpy(dtype=float).copy()
        else:
            self._excess = np.zeros(len(data), dtype=float)
        self._state = ModelState.PREPARED

    def set_input(self, name: str, values: np.ndarray) -> None:
        if name != self.excess_column:
            raise KeyError(f"ExternalExcess input is {self.excess_column!r}; got {name!r}")
        if self._excess is None:
            raise RuntimeError("Call prepare() before set_input().")
        arr = np.asarray(values, dtype=float)
        if arr.shape != self._excess.shape:
            raise ValueError(f"excess length {arr.shape} does not match the prepared {self._excess.shape}")
        self._excess = arr

    def predict_arrays(self) -> dict[str, np.ndarray]:
        if self._excess is None:
            raise RuntimeError("Call prepare() before predict().")
        self._state = ModelState.PREDICTED
        return {"datetime": self._datetime, self.excess_column: self._excess}

    def predict(self) -> pd.DataFrame:
        return pd.DataFrame(self.predict_arrays())

    def finalize(self) -> None:
        self._excess = None
        self._state = ModelState.FINALIZED


__all__ = ["ExternalExcess"]
