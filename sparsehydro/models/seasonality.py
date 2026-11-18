"""SeasonalityModel — discrete peaking-factor seasonal flow estimator.

Models cyclic flow patterns using discrete peaking factors for three
time dimensions: hour-of-day (24 bins), day-of-week (7 bins), and
month-of-year (12 bins).

Formula
-------

For each active dimension *d* with *N_d* discrete categories, the model
computes a normalized peaking factor vector and a dimension weight:

.. math::

    pf_{d,\\text{norm}} = pf_d / \\text{mean}(pf_d)

    \\hat{q}[t] = \\text{baseline} \\cdot \\sum_d w_d \\cdot pf_{d,\\text{norm}}[\\text{category}(t)]

where the weights are constrained to sum to 1.

The mean-normalization of each peaking factor vector ensures that the
model output equals ``baseline`` whenever all peaking factors are uniform.
This decouples the scale (``baseline``) from the pattern (``pf_d``), making
calibration well-conditioned.

Usage::

    from sparsehydro.models import SeasonalityModel

    model = SeasonalityModel(
        include_hour=True,
        include_dow=True,
        include_month=True,
        output_name="sanitary_cfs",
    )
    model.initialize()
    model.validate()
    model.prepare(df)         # df must have a "datetime" column
    result = model.predict()  # DataFrame: datetime, sanitary_cfs
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..enums import ModelState
from .base import IModel
from ..parameters import ConstraintRecord, FieldRecord, ScalarParameter, VectorParameter
from ..registry import registry


@registry.register
class SeasonalityModel(IModel):
    """Discrete peaking-factor seasonal flow model.

    Estimates cyclic flow patterns using calibratable peaking factors for
    hour-of-day, day-of-week, and/or month-of-year.  Each dimension's
    peaking factors are auto-normalized to have mean = 1 before use, and
    dimension weights are constrained to sum to 1.

    :param include_hour: Include hour-of-day (24 bins) dimension.
    :type include_hour: bool
    :param include_dow: Include day-of-week (7 bins, Monday=0) dimension.
    :type include_dow: bool
    :param include_month: Include month-of-year (12 bins, January=0) dimension.
    :type include_month: bool
    :param output_name: Column name for model output in :meth:`predict` results.
    :type output_name: str

    Total calibratable parameters:
        - 1 baseline
        - N_active dimension weights (constrained to sum = 1)
        - 24 hourly peaking factors (if include_hour)
        - 7 daily peaking factors (if include_dow)
        - 12 monthly peaking factors (if include_month)

    Example::

        model = SeasonalityModel(include_hour=True, include_dow=False, include_month=True)
        model.initialize()
        model.validate()
        model.prepare(df)
        out = model.predict()   # columns: datetime, flow
    """

    model_name = "seasonality"
    supports_array_predict = True

    def __init__(
        self,
        include_hour: bool = True,
        include_dow: bool = True,
        include_month: bool = True,
        output_name: str = "flow",
    ) -> None:
        super().__init__()
        if not any([include_hour, include_dow, include_month]):
            raise ValueError(
                "At least one dimension (include_hour, include_dow, include_month) "
                "must be active."
            )
        self._include_hour = include_hour
        self._include_dow = include_dow
        self._include_month = include_month
        self._output_name = output_name

        self._datetime: np.ndarray | None = None
        self._h: np.ndarray | None = None    # hour index 0-23
        self._dow: np.ndarray | None = None  # day-of-week 0-6
        self._month: np.ndarray | None = None  # month 0-11

    # ------------------------------------------------------------------
    # IModel lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Register baseline, dimension weights, and peaking factor parameters."""
        self._reset_registries()
        self.register_scalar_parameter(ScalarParameter(
            name="baseline",
            value=1.0,
            lower_bound=0.0,
            upper_bound=1e9,
            units="",
            description="Base flow value — average flow before peaking factor adjustment",
        ))

        active_dims = self._active_dims()

        for dim in active_dims:
            self.register_scalar_parameter(ScalarParameter(
                name=f"w_{dim}",
                value=1.0 / len(active_dims),
                lower_bound=0.0,
                upper_bound=1.0,
                units="-",
                description=f"Mixing weight for {self._dim_label(dim)} dimension",
            ))

        if self._include_hour:
            self.register_vector_parameter(VectorParameter(
                name="pf_hour",
                values=np.ones(24),
                lower_bounds=np.full(24, 0.01),
                upper_bounds=np.full(24, 100.0),
                units="-",
                description="Hourly peaking factors (auto-normalized to mean=1)",
            ))

        if self._include_dow:
            self.register_vector_parameter(VectorParameter(
                name="pf_dow",
                values=np.ones(7),
                lower_bounds=np.full(7, 0.01),
                upper_bounds=np.full(7, 100.0),
                units="-",
                description="Day-of-week peaking factors, Mon=0 (auto-normalized to mean=1)",
            ))

        if self._include_month:
            self.register_vector_parameter(VectorParameter(
                name="pf_month",
                values=np.ones(12),
                lower_bounds=np.full(12, 0.01),
                upper_bounds=np.full(12, 100.0),
                units="-",
                description="Monthly peaking factors, Jan=0 (auto-normalized to mean=1)",
            ))

        # Weight equality constraint: sum(w_d) = 1 expressed as two inequalities
        w_label = " + ".join(f"w_{d}" for d in active_dims)
        self.register_inequality_constraint(ConstraintRecord(
            name="sum_w_leq_1",
            description=f"Σ weights ({w_label}) ≤ 1.0",
        ))
        self.register_inequality_constraint(ConstraintRecord(
            name="sum_w_geq_1",
            description=f"1.0 ≤ Σ weights ({w_label})  [equality enforced via two inequalities]",
        ))

        self.register_output_field(FieldRecord(name="datetime", description="Simulation time step"))
        self.register_output_field(FieldRecord(
            name=self._output_name,
            units="",
            description=(
                f"Peaking-factor seasonal flow estimate — "
                f"active dimensions: {', '.join(self._active_dims())}"
            ),
            calibratable=True,
        ))

        self._state = ModelState.INITIALIZED

    def validate(self) -> bool:
        """Validate parameter bounds and advance to VALIDATED."""
        if not self.parameters_valid():
            return False
        self._state = ModelState.VALIDATED
        return True

    def prepare(self, data: Any, **kwargs: Any) -> None:
        """Extract and cache time indices from the datetime column.

        :param data: DataFrame containing a ``datetime`` column.
        :type data: pandas.DataFrame
        :raises KeyError: If the ``datetime`` column is missing.
        """
        df = data
        dt = pd.to_datetime(df["datetime"])
        self._datetime = dt.to_numpy()
        self._h = dt.dt.hour.to_numpy(dtype=int)
        self._dow = dt.dt.dayofweek.to_numpy(dtype=int)
        self._month = (dt.dt.month - 1).to_numpy(dtype=int)
        self._state = ModelState.PREPARED

    def predict_arrays(self) -> dict[str, np.ndarray]:
        """Apply peaking factors, returning plain arrays.

        :returns: Mapping with ``datetime`` and ``{output_name}``.
        :rtype: dict[str, numpy.ndarray]
        """
        baseline = self.get_scalar_parameter("baseline").value
        active_dims = self._active_dims()

        # Collect and normalize dimension weights
        raw_weights = {d: self.get_scalar_parameter(f"w_{d}").value for d in active_dims}
        w_sum = sum(raw_weights.values())
        weights = {d: v / w_sum for d, v in raw_weights.items()} if w_sum > 0 else raw_weights

        output = np.zeros(len(self._datetime), dtype=float)

        if self._include_hour and "hour" in active_dims:
            pf = self.get_vector_parameter("pf_hour").values
            pf_norm = pf / pf.mean()
            output += weights["hour"] * pf_norm[self._h]

        if self._include_dow and "dow" in active_dims:
            pf = self.get_vector_parameter("pf_dow").values
            pf_norm = pf / pf.mean()
            output += weights["dow"] * pf_norm[self._dow]

        if self._include_month and "month" in active_dims:
            pf = self.get_vector_parameter("pf_month").values
            pf_norm = pf / pf.mean()
            output += weights["month"] * pf_norm[self._month]

        output = baseline * output

        self._state = ModelState.PREDICTED
        return {"datetime": self._datetime, self._output_name: output}

    def predict(self) -> pd.DataFrame:
        """Apply peaking factors and return the seasonal flow estimate.

        :returns: DataFrame with columns ``datetime`` and ``{output_name}``.
        :rtype: pandas.DataFrame
        """
        return pd.DataFrame(self.predict_arrays())

    def finalize(self) -> None:
        """Release cached data and advance to FINALIZED."""
        self._datetime = None
        self._h = None
        self._dow = None
        self._month = None
        self._state = ModelState.FINALIZED

    def inequality_constraints(self) -> list[float]:
        """Return weight equality constraints as two inequalities.

        Returns ``[sum(w) - 1.0, 1.0 - sum(w)]``.
        Both ≤ 0 enforces sum(w) == 1.

        :returns: Two-element list ``[Σw - 1, 1 - Σw]``.
        :rtype: list[float]
        """
        active_dims = self._active_dims()
        w_sum = sum(self.get_scalar_parameter(f"w_{d}").value for d in active_dims)
        return [w_sum - 1.0, 1.0 - w_sum]

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def output_name(self) -> str:
        """Column name for the combined output in :meth:`predict` results."""
        return self._output_name

    @property
    def include_hour(self) -> bool:
        """Whether the hour-of-day dimension is active."""
        return self._include_hour

    @property
    def include_dow(self) -> bool:
        """Whether the day-of-week dimension is active."""
        return self._include_dow

    @property
    def include_month(self) -> bool:
        """Whether the month-of-year dimension is active."""
        return self._include_month

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _active_dims(self) -> list[str]:
        """Return list of active dimension keys in canonical order.

        :returns: Active dimension keys among ``"hour"``, ``"dow"``, ``"month"``.
        :rtype: list[str]
        """
        dims = []
        if self._include_hour:
            dims.append("hour")
        if self._include_dow:
            dims.append("dow")
        if self._include_month:
            dims.append("month")
        return dims

    @staticmethod
    def _dim_label(dim: str) -> str:
        return {"hour": "hour-of-day", "dow": "day-of-week", "month": "month-of-year"}.get(dim, dim)
