"""RDIIModel — configurable IA model + any mix of UH components.

Combines one :class:`~sparsehydro.models.rdii.IAModel` with *N* configurable
:class:`~sparsehydro.models.IUnitHydroComponent` objects (RTK triangles, Nash/Gamma
adapters, or custom shapes).

Parameter naming convention
---------------------------
- Own: ``area_acres``, ``R_1 … R_N``
- From IA model: original parameter names (e.g. ``ia_max``, ``ia_k0`` …)
- From UH component *i* (1-indexed): ``uh{i}_{name}`` for each shape
  parameter, skipping the component's amplitude/scaling parameter which is
  subsumed by ``R_{i}``.

``prepare()`` input DataFrame columns
--------------------------------------
+---------------------+----------+---------------------------+
| Column              | Required | Notes                     |
+=====================+==========+===========================+
| ``datetime``        | Yes      | Any pandas DatetimeLike   |
| ``rainfall_in``     | imperial | Depth per step [in]       |
| ``rainfall_mm``     | metric   | Depth per step [mm]       |
| ``flow_cfs``        | No       | Observed flow (optimizer) |
| ``temperature_c``   | No       | Falls back to ``ia_T_ref``|
+---------------------+----------+---------------------------+
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd

from ...enums import ModelState
from ..base import IModel, IUnitHydroComponent
from ..convolution import convolve_causal
from ...parameters import ConstraintRecord, FieldRecord, ScalarParameter
from ...registry import registry
from .initial_abstraction import IAModel
from .rtk_triangle import RTKTriangle

_DEFAULT_RTK = [
    (0.05, 1.0, 1.5),   # fast
    (0.03, 12.0, 2.0),  # medium
    (0.02, 72.0, 3.0),  # slow
]

#: Deprecated.  Convolution algorithm selection moved to
#: :mod:`sparsehydro.models.convolution`, which switches on the cost ``n * m``
#: rather than ``max(n, m)``.  Retained so existing imports keep resolving.
_FFT_THRESHOLD = 500
_MM_AC_PER_HR_TO_CFS = 43560.0 / (304.8 * 3600.0)
_IN_AC_PER_HR_TO_CFS = 43560.0 / (12.0 * 3600.0)


@registry.register
class RDIIModel(IModel):
    """Rainfall-Derived Inflow and Infiltration model.

    Combines one :class:`IAModel` with *N* configurable
    :class:`~sparsehydro.models.IUnitHydroComponent` instances.  Defaults to the
    classic three-pathway (fast / medium / slow) RTK parameterization.

    :param ia_model: Initial-abstraction model.  Its ``predict()`` must return
        a DataFrame containing a ``p_excess_mm`` or ``p_excess_in`` column
        (depending on *units*).  Defaults to :class:`IAModel`.
    :type ia_model: IModel, optional
    :param uh_components: Unit hydrograph components.  Each must implement
        :class:`~sparsehydro.models.IUnitHydroComponent`.  Defaults to three
        :class:`RTKTriangle` instances (fast / medium / slow).
    :type uh_components: list[IUnitHydroComponent], optional
    :param units: Unit system — ``"imperial"`` (inches, default) or ``"metric"`` (mm).
    :type units: str

    Usage::

        from sparsehydro.models.rdii import RDIIModel, IAModel, RTKTriangle

        model = RDIIModel()           # defaults: 1 IAModel + 3 RTKTriangles
        model.initialize()
        model.validate()
        model.prepare(df)
        result = model.predict()      # datetime, rdii_cfs, rdii_mm, p_excess_mm
        model.finalize()

        # Mix RTK and Nash UH:
        from sparsehydro.models.unithydrograph import NashUH
        model2 = RDIIModel(
            ia_model=IAModel(),
            uh_components=[RTKTriangle(R=0.05, T=1.0, K=1.5), NashUH()],
        )
    """

    model_name: ClassVar[str] = "rdii"

    def __init__(
        self,
        ia_model: IModel | None = None,
        uh_components: list[IUnitHydroComponent] | None = None,
        units: str = "imperial",
    ) -> None:
        super().__init__()
        if units not in ("imperial", "metric"):
            raise ValueError(f"units must be 'imperial' or 'metric'; got {units!r}")
        self._units = units
        self._rainfall_col = "rainfall_in" if units == "imperial" else "rainfall_mm"
        self._excess_col = "p_excess_in" if units == "imperial" else "p_excess_mm"
        self._depth_col = "rdii_in" if units == "imperial" else "rdii_mm"
        self._depth_to_cfs = _IN_AC_PER_HR_TO_CFS if units == "imperial" else _MM_AC_PER_HR_TO_CFS

        self._ia_model: IModel = ia_model if ia_model is not None else IAModel(units=units)
        if uh_components is None:
            self._uh_components: list[IUnitHydroComponent] = [
                RTKTriangle(R=R, T=T, K=K) for R, T, K in _DEFAULT_RTK
            ]
        else:
            self._uh_components = list(uh_components)

        self._prepared_df: pd.DataFrame | None = None
        self._dt_hours: float = 1.0

        self._ia_param_name_map: dict[str, str] = {}
        self._uh_param_maps: list[dict[str, str]] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_components(self) -> int:
        """Number of unit hydrograph components.

        :returns: Count of registered UH components.
        :rtype: int
        """
        return len(self._uh_components)

    @property
    def ia_model(self) -> IModel:
        """The initial-abstraction sub-model."""
        return self._ia_model

    @property
    def uh_components(self) -> list[IUnitHydroComponent]:
        """The unit hydrograph components, in registration order."""
        return list(self._uh_components)

    # ------------------------------------------------------------------
    # IModel lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Initialize sub-models and register all parameters.

        Registers ``area_acres``, ``R_1 … R_N`` (composite fractions), all
        IA model parameters (original names), and all UH shape parameters
        (``uh{i}_{name}`` prefix, excluding each component's amplitude param).

        :returns: Nothing.
        :rtype: None
        """
        self._ia_model.initialize()
        for uh in self._uh_components:
            uh.initialize()

        self.register_scalar_parameter(ScalarParameter(
            "area_acres", value=100.0, lower_bound=0.01, upper_bound=100_000.0,
            units="acres", description="Drainage area — converts rdii depth to rdii_cfs flow",
        ))
        for i, uh in enumerate(self._uh_components, 1):
            R_default = _amplitude_default(uh)
            self.register_scalar_parameter(ScalarParameter(
                f"R_{i}", value=R_default, lower_bound=0.0, upper_bound=1.0,
                units="-", description=f"Component {i}: fraction of P_excess routed through this UH",
            ))

        self._ia_param_name_map = {
            name: name for name in self._ia_model.scalar_parameter_names
        }
        for name in self._ia_param_name_map:
            p = self._ia_model.get_scalar_parameter(name)
            self.register_scalar_parameter(ScalarParameter(
                p.name, p.value, p.lower_bound, p.upper_bound,
                p.units, p.description, p.calibrate,
            ))

        self._uh_param_maps = []
        for i, uh in enumerate(self._uh_components, 1):
            amp = type(uh)._amplitude_param_name
            mapping: dict[str, str] = {}
            for name in uh.scalar_parameter_names:
                if name == amp:
                    continue
                composite_name = f"uh{i}_{name}"
                p = uh.get_scalar_parameter(name)
                self.register_scalar_parameter(ScalarParameter(
                    composite_name, p.value, p.lower_bound, p.upper_bound,
                    p.units, p.description, p.calibrate,
                ))
                mapping[name] = composite_name
            self._uh_param_maps.append(mapping)

        self.register_inequality_constraint(ConstraintRecord(
            name="sum_R_leq_1",
            description=(
                f"R_1 + … + R_{len(self._uh_components)} ≤ 1.0  "
                "(total runoff fraction must not exceed 100 %)"
            ),
        ))
        self.register_inequality_constraint(ConstraintRecord(
            name="T_freeze_lt_T_ref",
            description=(
                "ia_T_freeze < ia_T_ref  "
                "(freeze threshold must be below the reference temperature)"
            ),
        ))

        depth_units = "in" if self._units == "imperial" else "mm"
        self.register_output_field(FieldRecord(
            name="datetime",
            description="Simulation time step",
        ))
        self.register_output_field(FieldRecord(
            name="rdii_cfs",
            units="CFS",
            description="RDII flow rate — all UH components combined",
            calibratable=True,
        ))
        self.register_output_field(FieldRecord(
            name=self._depth_col,
            units=depth_units,
            description="RDII depth per time step",
            calibratable=False,
        ))
        self.register_output_field(FieldRecord(
            name=self._excess_col,
            units=depth_units,
            description="Rainfall excess depth after initial abstraction",
            calibratable=False,
        ))

        self._state = ModelState.INITIALIZED

    def validate(self) -> bool:
        """Validate all parameters and physical constraints.

        Checks ``ia_T_freeze < ia_T_ref`` when both are present.

        :returns: ``True`` if all constraints are satisfied.
        :rtype: bool
        """
        if not self.parameters_valid():
            return False

        if (
            "ia_T_freeze" in self._scalar_parameters
            and "ia_T_ref" in self._scalar_parameters
        ):
            T_freeze = self.get_scalar_parameter("ia_T_freeze").value
            T_ref = self.get_scalar_parameter("ia_T_ref").value
            if T_freeze >= T_ref:
                return False

        self._state = ModelState.VALIDATED
        return True

    def prepare(self, data: pd.DataFrame, **kwargs: Any) -> None:
        """Load input data, infer dt, fill missing temperature, prepare the IA model.

        :param data: DataFrame with columns ``datetime``, ``rainfall_in`` or
            ``rainfall_mm`` (depending on *units*), and optionally ``flow_cfs``
            and ``temperature_c``.
        :type data: pandas.DataFrame
        :raises ValueError: If required columns are absent or dt cannot be inferred.
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
            raise ValueError(
                "Cannot infer dt_hours: DataFrame has fewer than 2 rows."
            )
        median_td = diffs.median()
        self._dt_hours = median_td.total_seconds() / 3600.0
        if self._dt_hours <= 0.0:
            raise ValueError(
                f"Inferred dt_hours = {self._dt_hours:.4f} is not positive. "
                "Ensure the datetime column is sorted and has a uniform step."
            )

        if "ia_T_ref" in self._scalar_parameters:
            T_ref = self.get_scalar_parameter("ia_T_ref").value
        else:
            T_ref = 20.0
        if "temperature_c" not in df.columns:
            df["temperature_c"] = T_ref
        else:
            df["temperature_c"] = df["temperature_c"].fillna(T_ref)

        self._prepared_df = df

        self._sync_to_submodels()
        self._ia_model.prepare(df)

        self._state = ModelState.PREPARED

    def predict_arrays(self) -> dict[str, np.ndarray]:
        """Compute RDII by convolving P_excess with each UH component.

        Syncs parameter values to sub-models on every call so parameter changes
        made between optimizer iterations are picked up automatically.

        :returns: Mapping with ``datetime``, ``rdii_cfs``, ``rdii_mm`` (or
            ``rdii_in``), ``p_excess_mm`` (or ``p_excess_in``).
        :rtype: dict[str, numpy.ndarray]
        :raises RuntimeError: If ``prepare()`` has not been called.
        """
        if self._prepared_df is None:
            raise RuntimeError("Call prepare(data) before predict().")

        self._sync_to_submodels()

        ia_result = self._ia_model.predict_arrays()
        if self._excess_col not in ia_result:
            raise RuntimeError(
                f"ia_model.predict_arrays() must provide a "
                f"'{self._excess_col}' entry; got keys: {list(ia_result)}"
            )
        p_excess = np.asarray(ia_result[self._excess_col], dtype=float)
        n = len(p_excess)
        rdii = np.zeros(n, dtype=float)

        for i, uh in enumerate(self._uh_components, 1):
            R_i = self.get_scalar_parameter(f"R_{i}").value
            kernel = uh.get_kernel(self._dt_hours)
            rdii += R_i * convolve_causal(p_excess, kernel, n_out=n)

        rdii = np.clip(rdii, 0.0, None)
        area_acres = self.get_scalar_parameter("area_acres").value
        rdii_cfs = rdii * area_acres * self._depth_to_cfs
        rdii_depth = rdii * self._dt_hours

        self._state = ModelState.PREDICTED
        return {
            "datetime": self._prepared_df["datetime"].values,
            "rdii_cfs": rdii_cfs,
            self._depth_col: rdii_depth,
            self._excess_col: p_excess.copy(),
        }

    def predict(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Compute RDII by convolving P_excess with each UH component.

        :returns: DataFrame with columns ``datetime``, ``rdii_cfs``,
            ``rdii_mm`` (or ``rdii_in``), ``p_excess_mm`` (or ``p_excess_in``).
        :rtype: pandas.DataFrame
        :raises RuntimeError: If ``prepare()`` has not been called.
        """
        return pd.DataFrame(self.predict_arrays())

    def predict_components(self) -> pd.DataFrame:
        """Compute the RDII contribution of each UH component separately.

        Runs the same IA + convolution pipeline as :meth:`predict` but keeps
        each component's flow contribution in its own column, ready for
        :func:`~sparsehydro.visualization.plot_rdii_components`.

        :returns: DataFrame with columns ``datetime``, the rainfall column,
            the excess column, ``rdii_component_1 … rdii_component_N`` (flow
            per component [CFS]), and ``rdii_cfs`` (their sum).
        :rtype: pandas.DataFrame
        :raises RuntimeError: If ``prepare()`` has not been called.
        """
        if self._prepared_df is None:
            raise RuntimeError("Call prepare(data) before predict_components().")

        self._sync_to_submodels()

        ia_result = self._ia_model.predict()
        p_excess = ia_result[self._excess_col].to_numpy(dtype=float)
        n = len(p_excess)
        area_acres = self.get_scalar_parameter("area_acres").value

        out: dict[str, Any] = {
            "datetime": self._prepared_df["datetime"].values,
            self._rainfall_col: self._prepared_df[self._rainfall_col].to_numpy(dtype=float),
            self._excess_col: p_excess.copy(),
        }

        total = np.zeros(n, dtype=float)
        for i, uh in enumerate(self._uh_components, 1):
            R_i = self.get_scalar_parameter(f"R_{i}").value
            kernel = uh.get_kernel(self._dt_hours)
            conv = convolve_causal(p_excess, kernel, n_out=n)
            component_cfs = np.clip(R_i * conv, 0.0, None) * area_acres * self._depth_to_cfs
            out[f"rdii_component_{i}"] = component_cfs
            total += component_cfs

        out["rdii_cfs"] = total
        return pd.DataFrame(out)

    def finalize(self) -> None:
        """Release stored data and advance to FINALIZED.

        :returns: Nothing.
        :rtype: None
        """
        self._prepared_df = None
        self._state = ModelState.FINALIZED

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    def inequality_constraints(self) -> list[float]:
        """Inequality constraint residuals for the optimizer.

        Returns ``[Σ R_i - 1.0, ia_T_freeze - ia_T_ref]``.
        A value ≤ 0 means feasible.

        :returns: Two-element list ``[Σ R_i - 1, ia_T_freeze - ia_T_ref]``.
        :rtype: list[float]
        """
        R_sum = sum(
            self.get_scalar_parameter(f"R_{i}").value
            for i in range(1, self.n_components + 1)
        )
        T_freeze = self.get_scalar_parameter("ia_T_freeze").value
        T_ref = self.get_scalar_parameter("ia_T_ref").value
        return [R_sum - 1.0, T_freeze - T_ref]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sync_to_submodels(self) -> None:
        """Push composite registry values to sub-model registries.

        :returns: Nothing.
        :rtype: None
        """
        for ia_name, composite_name in self._ia_param_name_map.items():
            if composite_name in self._scalar_parameters:
                try:
                    self._ia_model.get_scalar_parameter(ia_name).value = (
                        self.get_scalar_parameter(composite_name).value
                    )
                except KeyError:
                    pass

        for i, (uh, mapping) in enumerate(
            zip(self._uh_components, self._uh_param_maps), 1
        ):
            for orig_name, composite_name in mapping.items():
                try:
                    uh.get_scalar_parameter(orig_name).value = (
                        self.get_scalar_parameter(composite_name).value
                    )
                except KeyError:
                    pass

            amp = type(uh)._amplitude_param_name
            if amp:
                try:
                    uh.get_scalar_parameter(amp).value = (
                        self.get_scalar_parameter(f"R_{i}").value
                    )
                except KeyError:
                    pass

    def rename_scalar_parameter(self, old_name: str, new_name: str) -> None:
        """Rename a scalar parameter and keep internal sync maps up to date.

        :param old_name: Current name of the parameter.
        :type old_name: str
        :param new_name: New name to assign to the parameter.
        :type new_name: str
        :returns: Nothing.
        :rtype: None
        :raises ValueError: If a parameter named *new_name* already exists.
        :raises KeyError: If no parameter named *old_name* is registered.
        """
        super().rename_scalar_parameter(old_name, new_name)
        for ia_name, composite in self._ia_param_name_map.items():
            if composite == old_name:
                self._ia_param_name_map[ia_name] = new_name
                break
        for mapping in self._uh_param_maps:
            for orig, composite in list(mapping.items()):
                if composite == old_name:
                    mapping[orig] = new_name
                    break


def _amplitude_default(uh: IUnitHydroComponent) -> float:
    """Return the default R value from the component's amplitude parameter, else 0.05.

    :param uh: Unit hydrograph component to inspect.
    :type uh: IUnitHydroComponent
    :returns: The component's amplitude parameter value, or ``0.05`` if absent.
    :rtype: float
    """
    amp = type(uh)._amplitude_param_name
    if amp:
        try:
            return float(uh.get_scalar_parameter(amp).value)
        except KeyError:
            pass
    return 0.05
