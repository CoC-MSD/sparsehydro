"""Storage-based rainfall-abstraction (tank) models.

Ported from Parsimonious Functions ``tank_models.py``.  A tank accumulates a
wetness proxy from rainfall and drains it over time.  The tank fill fraction
``w = V / V_tank`` linearly interpolates an *effective-area* multiplier between
``ae_max`` (empty tank) and ``ae_min`` (full tank); effective rainfall is
``Ie = R · ae(w)``.

Because ``ae`` is an effective-area gain (typically > 1), the tank carries the
rainfall→runoff volume scaling.  When chaining a tank into
:class:`~sparsehydro.models.AbstractionUHModel`, fix the unit-hydrograph
amplitude ``A = 1`` so the gain is not double-counted.

The tank state is advanced with sub-stepping (``n_substeps`` per time step) for
numerical stability of the drainage integration. The three built-in drainage
laws run in a compiled kernel (with the ``speed`` extra); a subclass that
defines its own :meth:`~TankAbstractionModel._drainage_full` runs the same
loop in Python.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar

import numpy as np
import pandas as pd

from ..._numba import njit
from ...enums import ModelState
from ...parameters import ConstraintRecord, FieldRecord, ScalarParameter
from ...registry import registry
from ..base import IModel


#: Drainage-law codes understood by :func:`_tank_kernel`.
_LAW_CONSTANT, _LAW_LINEAR, _LAW_SQRT = 0, 1, 2


@njit(cache=True)
def _tank_kernel(rain, V_tank, ae_min, ae_max, drain, n_substeps, law, record=False):  # pragma: no cover - compiled
    """Advance the tank; same operation order as the Python loop (bit-identical).

    Returns ``(excess, storage, gain)``: tank storage at the end of each step
    and the effective-area multiplier applied in it when *record* is true,
    empty arrays otherwise.
    """
    n = rain.shape[0]
    excess = np.zeros(n)
    m = n if record else 0
    rec_V = np.zeros(m)
    rec_ae = np.zeros(m)
    V = 0.0
    for i in range(n):
        R = rain[i]
        w = min(max(V / V_tank, 0.0), 1.0)
        ae = ae_max - w * (ae_max - ae_min)
        excess[i] = max(R * ae, 0.0)
        r_sub = R / n_substeps
        for _ in range(n_substeps):
            V += r_sub
            if law == _LAW_CONSTANT:
                full = min(drain, V)
            elif law == _LAW_LINEAR:
                full = min(drain * V, V)
            else:
                full = min(drain * np.sqrt(max(V, 0.0)), V)
            V -= full / n_substeps
            V = min(max(V, 0.0), V_tank)
        if record:
            rec_V[i] = V
            rec_ae[i] = ae
    return excess, rec_V, rec_ae


class TankAbstractionModel(IModel):
    """Abstract base for storage-based abstraction models.

    Subclasses register a drainage parameter and implement
    :meth:`_drainage_full` (the full-step drainage volume for a given storage).

    :param V_tank: Tank capacity (wetness-proxy storage).  Bounds [0.1, 1e6].
    :type V_tank: float
    :param ae_min: Effective-area multiplier at a full tank.  Bounds [0, 1e4].
    :type ae_min: float
    :param ae_max: Effective-area multiplier at an empty tank.  Bounds [0, 1e4].
        Must be ``>= ae_min``.
    :type ae_max: float
    :param drain: Drainage parameter value (constant ``Qd`` or coefficient ``k``
        depending on the subclass).
    :type drain: float
    :param units: Unit system — ``"imperial"`` (inches, default) or ``"metric"`` (mm).
    :type units: str
    :param n_substeps: Sub-steps per time step for drainage integration.
    :type n_substeps: int
    """

    supports_array_predict: ClassVar[bool] = True

    #: Name of the subclass drainage parameter (e.g. ``"tank_qd"`` or ``"tank_k"``).
    _drain_param_name: ClassVar[str] = "tank_drain"
    _drain_lower: ClassVar[float] = 1e-6
    _drain_upper: ClassVar[float] = 1000.0
    _drain_units: ClassVar[str] = ""
    _drain_desc: ClassVar[str] = "Drainage parameter"
    #: Kernel drainage-law code; ``None`` runs :meth:`_drainage_full` in Python.
    _drain_law: ClassVar[int | None] = None

    def __init__(
        self,
        V_tank: float = 1000.0,
        ae_min: float = 5.0,
        ae_max: float = 10.0,
        drain: float = 0.1,
        units: str = "imperial",
        n_substeps: int = 5,
    ) -> None:
        super().__init__()
        if units not in ("imperial", "metric"):
            raise ValueError(f"units must be 'imperial' or 'metric'; got {units!r}")
        self._units = units
        self._rainfall_col = "rainfall_in" if units == "imperial" else "rainfall_mm"
        self._excess_col = "p_excess_in" if units == "imperial" else "p_excess_mm"
        self._V_tank_init = float(V_tank)
        self._ae_min_init = float(ae_min)
        self._ae_max_init = float(ae_max)
        self._drain_init = float(drain)
        self._n_substeps = max(int(n_substeps), 1)
        self._prepared_df: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # IModel lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Register the tank parameters and advance to INITIALIZED.

        :returns: Nothing.
        :rtype: None
        """
        self._reset_registries()
        self.register_scalar_parameter(ScalarParameter(
            "tank_V", value=self._V_tank_init, lower_bound=0.1, upper_bound=1e6,
            description="Tank capacity (wetness-proxy storage)",
        ))
        self.register_scalar_parameter(ScalarParameter(
            "tank_ae_min", value=self._ae_min_init, lower_bound=0.0, upper_bound=1e4,
            description="Effective-area multiplier at full tank",
        ))
        self.register_scalar_parameter(ScalarParameter(
            "tank_ae_max", value=self._ae_max_init, lower_bound=0.0, upper_bound=1e4,
            description="Effective-area multiplier at empty tank",
        ))
        self.register_scalar_parameter(ScalarParameter(
            self._drain_param_name, value=self._drain_init,
            lower_bound=self._drain_lower, upper_bound=self._drain_upper,
            units=self._drain_units, description=self._drain_desc,
        ))
        self.register_inequality_constraint(ConstraintRecord(
            "ae_min_le_ae_max", "Gain at a full tank no larger than at an empty one (ae_min - ae_max <= 0)"))
        depth_units = "in" if self._units == "imperial" else "mm"
        self.register_output_field(FieldRecord(name="datetime", description="Simulation time step"))
        self.register_output_field(FieldRecord(
            name=self._excess_col, units=depth_units,
            description="Effective rainfall depth after tank abstraction",
            calibratable=False,
        ))
        self.register_trace_field(FieldRecord(
            "tank_storage", units=depth_units, role="state",
            description="Tank storage at the end of the step"))
        self.register_trace_field(FieldRecord(
            "tank_gain", role="state",
            description="Effective-area multiplier applied in the step (ae_max when empty, ae_min when full)"))
        self._state = ModelState.INITIALIZED

    def inequality_constraints(self) -> list[float]:
        """``[ae_min - ae_max]``; feasible when ``<= 0``."""
        return [self.get_scalar_parameter("tank_ae_min").value - self.get_scalar_parameter("tank_ae_max").value]

    def validate(self) -> bool:
        """Validate parameter bounds and the ``ae_min <= ae_max`` ordering.

        :returns: ``True`` if all parameters are within bounds and ``ae_min <= ae_max``.
        :rtype: bool
        """
        ok = self.parameters_valid()
        if ok and self.get_scalar_parameter("tank_ae_min").value > self.get_scalar_parameter("tank_ae_max").value:
            ok = False
        if ok:
            self._state = ModelState.VALIDATED
        return ok

    def prepare(self, data: pd.DataFrame) -> None:
        """Load forcing data.

        :param data: DataFrame with ``datetime`` and the unit-specific rainfall
            column (``rainfall_in`` or ``rainfall_mm``).
        :type data: pandas.DataFrame
        :returns: Nothing.
        :rtype: None
        :raises ValueError: If required columns are absent.
        """
        required = {"datetime", self._rainfall_col}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"prepare() data is missing required columns: {sorted(missing)}")
        df = data.sort_values("datetime").reset_index(drop=True).copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        df[self._rainfall_col] = df[self._rainfall_col].fillna(0.0).clip(lower=0.0)
        self._prepared_df = df
        self._rain = np.ascontiguousarray(df[self._rainfall_col].to_numpy(dtype=float))
        self._datetime = df["datetime"].to_numpy()
        self._state = ModelState.PREPARED

    def predict_arrays(self) -> dict[str, np.ndarray]:
        """Advance the tank over the series, returning plain arrays.

        :returns: Mapping with ``datetime`` and the unit-specific excess column.
        :rtype: dict[str, numpy.ndarray]
        :raises RuntimeError: If :meth:`prepare` has not been called.
        """
        if self._prepared_df is None:
            raise RuntimeError("Call prepare(data) before predict().")
        excess, _, _ = self._advance(record=False)
        self._state = ModelState.PREDICTED
        return {"datetime": self._datetime, self._excess_col: excess}

    def _advance(self, record: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rain = self._rain
        V_tank = max(self.get_scalar_parameter("tank_V").value, 1e-9)
        ae_min = self.get_scalar_parameter("tank_ae_min").value
        ae_max = self.get_scalar_parameter("tank_ae_max").value
        drain = self.get_scalar_parameter(self._drain_param_name).value
        n = self._n_substeps
        if self._drain_law is not None:
            return _tank_kernel(rain, float(V_tank), float(ae_min), float(ae_max), float(drain), int(n),
                                int(self._drain_law), record)
        return self._advance_python(rain, V_tank, ae_min, ae_max, drain, n, record)

    def _trace_arrays(self, outputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Tank storage and the effective-area multiplier per step."""
        _, storage, gain = self._advance(record=True)
        return {"tank_storage": storage, "tank_gain": gain}

    def _advance_python(self, rain, V_tank, ae_min, ae_max, drain, n, record=False):
        """Reference loop using :meth:`_drainage_full` (custom drainage laws).

        Returns ``(excess, storage, gain)`` like :func:`_tank_kernel`.
        """
        excess = np.zeros(len(rain))
        m = len(rain) if record else 0
        rec_V, rec_ae = np.zeros(m), np.zeros(m)
        V = 0.0
        for i, R in enumerate(rain):
            w = min(max(V / V_tank, 0.0), 1.0)
            ae = ae_max - w * (ae_max - ae_min)
            excess[i] = max(R * ae, 0.0)
            r_sub = R / n
            for _ in range(n):
                V += r_sub
                V -= self._drainage_full(V, drain) / n
                V = min(max(V, 0.0), V_tank)
            if record:
                rec_V[i] = V
                rec_ae[i] = ae
        return excess, rec_V, rec_ae

    def predict(self) -> pd.DataFrame:
        """Advance the tank over the series and return effective rainfall.

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
        self._state = ModelState.FINALIZED

    @abstractmethod
    def _drainage_full(self, V: float, drain: float) -> float:
        """Return the full-step drainage volume for the current storage *V*.

        :param V: Current tank storage.
        :type V: float
        :param drain: Subclass drainage parameter value.
        :type drain: float
        :returns: Volume drained over a full time step.
        :rtype: float
        """
        raise NotImplementedError


@registry.register
class ConstantDrainTank(TankAbstractionModel):
    """``V_C`` tank — constant drainage per time step.

    Drainage law: ``Qd`` (independent of storage).
    """

    model_name: ClassVar[str] = "tank-constant"
    _drain_law: ClassVar[int | None] = _LAW_CONSTANT
    _drain_param_name: ClassVar[str] = "tank_qd"
    _drain_lower: ClassVar[float] = 1e-6
    _drain_upper: ClassVar[float] = 1000.0
    _drain_desc: ClassVar[str] = "Constant drainage per step"

    def _drainage_full(self, V: float, drain: float) -> float:
        """Return constant drainage ``Qd`` clipped to available storage.

        :param V: Current tank storage.
        :type V: float
        :param drain: Constant drainage ``Qd`` per step.
        :type drain: float
        :returns: Drainage volume over a full step.
        :rtype: float
        """
        return min(drain, V)


@registry.register
class LinearDrainTank(TankAbstractionModel):
    """``V_lin`` tank — linear-reservoir drainage.

    Drainage law: ``k · V`` (proportional to storage).
    """

    model_name: ClassVar[str] = "tank-linear"
    _drain_law: ClassVar[int | None] = _LAW_LINEAR
    _drain_param_name: ClassVar[str] = "tank_k"
    _drain_lower: ClassVar[float] = 1e-6
    _drain_upper: ClassVar[float] = 1.0
    _drain_desc: ClassVar[str] = "Linear drainage coefficient (k·V)"

    def __init__(self, V_tank: float = 1000.0, ae_min: float = 5.0, ae_max: float = 10.0,
                 drain: float = 0.1, units: str = "imperial", n_substeps: int = 5) -> None:
        super().__init__(V_tank=V_tank, ae_min=ae_min, ae_max=ae_max, drain=drain, units=units, n_substeps=n_substeps)

    def _drainage_full(self, V: float, drain: float) -> float:
        """Return storage-proportional drainage ``k·V``.

        :param V: Current tank storage.
        :type V: float
        :param drain: Linear drainage coefficient ``k``.
        :type drain: float
        :returns: Drainage volume over a full step.
        :rtype: float
        """
        return min(drain * V, V)


@registry.register
class SqrtDrainTank(TankAbstractionModel):
    """``V_sroot`` tank — square-root drainage.

    Drainage law: ``k · sqrt(V)`` (intermediate between constant and linear).
    """

    model_name: ClassVar[str] = "tank-sqrt"
    _drain_law: ClassVar[int | None] = _LAW_SQRT
    _drain_param_name: ClassVar[str] = "tank_k"
    _drain_lower: ClassVar[float] = 1e-6
    _drain_upper: ClassVar[float] = 10.0
    _drain_desc: ClassVar[str] = "Square-root drainage coefficient (k·sqrt(V))"

    def __init__(self, V_tank: float = 1000.0, ae_min: float = 5.0, ae_max: float = 10.0,
                 drain: float = 0.1, units: str = "imperial", n_substeps: int = 5) -> None:
        super().__init__(V_tank=V_tank, ae_min=ae_min, ae_max=ae_max, drain=drain, units=units, n_substeps=n_substeps)

    def _drainage_full(self, V: float, drain: float) -> float:
        """Return square-root drainage ``k·sqrt(V)``.

        :param V: Current tank storage.
        :type V: float
        :param drain: Square-root drainage coefficient ``k``.
        :type drain: float
        :returns: Drainage volume over a full step.
        :rtype: float
        """
        return min(drain * float(np.sqrt(max(V, 0.0))), V)


__all__ = [
    "TankAbstractionModel",
    "ConstantDrainTank",
    "LinearDrainTank",
    "SqrtDrainTank",
]
