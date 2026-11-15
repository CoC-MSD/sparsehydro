"""AbstractionUHModel — chained abstraction → unit hydrograph (+ optional seasonality).

Composes three native SparseHydro building blocks into a single
:class:`~sparsehydro.models.IModel` suitable for calibration and sequential
fitting:

1. **Abstraction** — an :class:`~sparsehydro.models.rdii.IAModel`
   (temperature-driven, the default) or a
   :class:`~sparsehydro.models.abstraction.TankAbstractionModel`.  Transforms
   rainfall into *effective rainfall*.
2. **Unit hydrograph** — any
   :class:`~sparsehydro.models.IUnitHydroComponent` (default
   :class:`~sparsehydro.models.unithydrograph.PeakTailUH`).  Convolves effective
   rainfall into predicted flow ``Q_pred``.
3. **Seasonality** *(optional)* — a
   :class:`~sparsehydro.models.SeasonalityModel` applied as a **multiplicative**
   peaking factor on ``Q_pred`` (baseline fixed at 1).

Parameter naming
----------------
Child parameters are re-registered on the composite with a prefix so a single
flat calibration vector drives every sub-model:

- Abstraction: original names (already ``ia_*`` or ``tank_*``).
- Unit hydrograph: ``uh_{name}`` (e.g. ``uh_A``, ``uh_tp``).
- Seasonality: ``seas_{name}`` (e.g. ``seas_pf_month``); ``seas_baseline`` is
  fixed (``calibrate=False``).

.. note::
   Seasonality (hour/day/month peaking factors) captures **between-event**
   variation and is near-constant within a single storm.  Enable it for
   whole-series or multi-event predictions; for strict per-event sequential
   fitting prefer the temperature-driven :class:`IAModel` for seasonal signal.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd

from ..enums import ModelState
from ..parameters import ConstraintRecord, FieldRecord, ScalarParameter, VectorParameter
from ..registry import registry
from .base import IModel, IUnitHydroComponent
from .rdii import IAModel
from .seasonality import SeasonalityModel
from .unithydrograph import PeakTailUH


@registry.register
class AbstractionUHModel(IModel):
    """Chained abstraction → unit hydrograph (+ optional seasonality) model.

    :param abstraction: Abstraction sub-model producing an effective-rainfall
        column.  Defaults to :class:`~sparsehydro.models.rdii.IAModel`.
    :type abstraction: IModel, optional
    :param uh: Unit hydrograph component convolving effective rainfall.
        Defaults to :class:`~sparsehydro.models.unithydrograph.PeakTailUH`.
    :type uh: IUnitHydroComponent, optional
    :param seasonality: Optional multiplicative seasonal modifier.  When
        omitted, no seasonal adjustment is applied.
    :type seasonality: SeasonalityModel, optional
    :param units: Unit system — ``"imperial"`` (default) or ``"metric"``.
    :type units: str
    """

    model_name: ClassVar[str] = "abstraction-uh"
    supports_array_predict: ClassVar[bool] = True

    def __init__(
        self,
        abstraction: IModel | None = None,
        uh: IUnitHydroComponent | None = None,
        seasonality: SeasonalityModel | None = None,
        units: str = "imperial",
    ) -> None:
        super().__init__()
        if units not in ("imperial", "metric"):
            raise ValueError(f"units must be 'imperial' or 'metric'; got {units!r}")
        self._units = units
        self._rainfall_col = "rainfall_in" if units == "imperial" else "rainfall_mm"
        self._abstraction: IModel = abstraction if abstraction is not None else IAModel(units=units)
        self._uh: IUnitHydroComponent = uh if uh is not None else PeakTailUH()
        self._seasonality: SeasonalityModel | None = seasonality

        # child -> prefix, built in initialize()
        self._children: list[tuple[IModel, str]] = []
        # composite param name -> (child, child_param_name, kind)
        self._param_map: dict[str, tuple[IModel, str, str]] = {}
        self._season_output: str | None = None
        self._abs_input: pd.DataFrame | None = None
        self._season_input: pd.DataFrame | None = None
        self._datetime: np.ndarray | None = None
        #: Abstraction output column, resolved once in prepare().
        self._exc_col: str | None = None
        #: The UH's set_forcing, when it supports the array path; None means
        #: fall back to rebuilding a DataFrame per predict().
        self._uh_set_forcing = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Initialize child models and forward their parameters with prefixes.

        :returns: Nothing.
        :rtype: None
        """
        self._reset_registries()
        self._param_map = {}

        self._children = [(self._abstraction, ""), (self._uh, "uh_")]
        if self._seasonality is not None:
            self._children.append((self._seasonality, "seas_"))

        for child, prefix in self._children:
            child.initialize()
            for sname in child.scalar_parameter_names:
                p = child.get_scalar_parameter(sname)
                cname = f"{prefix}{sname}"
                calibrate = p.calibrate
                value = p.value
                # Fix the seasonality baseline so it acts as a pure shape modifier.
                if prefix == "seas_" and sname == "baseline":
                    calibrate = False
                    value = 1.0
                self.register_scalar_parameter(ScalarParameter(
                    cname, value=value, lower_bound=p.lower_bound, upper_bound=p.upper_bound,
                    units=p.units, description=p.description, calibrate=calibrate,
                ))
                self._param_map[cname] = (child, sname, "scalar")
            for vname in child.vector_parameter_names:
                vp = child.get_vector_parameter(vname)
                cname = f"{prefix}{vname}"
                self.register_vector_parameter(VectorParameter(
                    cname, values=np.array(vp.values, dtype=float).copy(),
                    lower_bounds=np.array(vp.lower_bounds, dtype=float).copy(),
                    upper_bounds=np.array(vp.upper_bounds, dtype=float).copy(),
                    units=vp.units, description=vp.description, calibrate=vp.calibrate,
                ))
                self._param_map[cname] = (child, vname, "vector")
            for cn, cd in zip(child.inequality_constraint_names, child.inequality_constraint_descriptions):
                self.register_inequality_constraint(ConstraintRecord(name=f"{prefix}{cn}", description=cd))

        if self._seasonality is not None:
            season_outputs = [n for n in self._seasonality.output_field_names if n != "datetime"]
            self._season_output = season_outputs[0] if season_outputs else None

        self.register_output_field(FieldRecord(name="datetime", description="Simulation time step"))
        self.register_output_field(FieldRecord(
            name="Q_pred", units="", description="Predicted stormflow", calibratable=True,
        ))
        self._state = ModelState.INITIALIZED

    def _sync_children(self) -> None:
        """Push composite parameter values into the child models."""
        for cname, (child, child_name, kind) in self._param_map.items():
            if kind == "scalar":
                child.get_scalar_parameter(child_name).value = self.get_scalar_parameter(cname).value
            else:
                child.get_vector_parameter(child_name).values[:] = self.get_vector_parameter(cname).values

    def validate(self) -> bool:
        """Validate the composite and every child model.

        :returns: ``True`` if all composite and child parameters are valid.
        :rtype: bool
        """
        if not self.parameters_valid():
            return False
        self._sync_children()
        for child, _ in self._children:
            if not child.validate():
                return False
        self._state = ModelState.VALIDATED
        return True

    def prepare(self, data: pd.DataFrame) -> None:
        """Cache forcing data and build child inputs.

        :param data: DataFrame with ``datetime`` and a rainfall column
            (``rain``, ``rainfall_in``, or ``rainfall_mm``); an optional
            ``temperature_c`` column feeds a temperature-driven abstraction.
        :type data: pandas.DataFrame
        :returns: Nothing.
        :rtype: None
        :raises ValueError: If no recognized rainfall column is present.
        """
        df = data.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])

        rain_source = None
        for cand in ("rain", self._rainfall_col, "rainfall_in", "rainfall_mm"):
            if cand in df.columns:
                rain_source = cand
                break
        if rain_source is None:
            raise ValueError(
                "prepare() data needs a rainfall column "
                "('rain', 'rainfall_in', or 'rainfall_mm')."
            )

        abs_input = pd.DataFrame({
            "datetime": df["datetime"].values,
            self._rainfall_col: df[rain_source].to_numpy(dtype=float),
        })
        if "temperature_c" in df.columns:
            abs_input["temperature_c"] = df["temperature_c"].to_numpy(dtype=float)

        self._abs_input = abs_input
        self._season_input = pd.DataFrame({"datetime": df["datetime"].values})
        self._datetime = df["datetime"].values

        # Prepare the children once, here, rather than on every predict().
        #
        # This is safe because both abstraction models are pure over
        # (prepared data, parameter values): IAModel.predict() re-syncs its
        # parameters at its own top and routes through the *static*
        # compute_excess_series() with all state passed explicitly, so its
        # reset() has no bearing on predict; TankAbstractionModel.predict()
        # likewise reads only _prepared_df and parameters.  The golden
        # composite fixture pins that this holds.
        self._abstraction.prepare(self._abs_input)
        exc_names = [n for n in self._abstraction.output_field_names if n != "datetime"]
        self._exc_col = exc_names[0] if exc_names else None

        # Give the UH its time base once; predict() then swaps only the rainfall
        # via set_forcing(), so no DataFrame is built per evaluation.
        self._uh_set_forcing = getattr(self._uh, "set_forcing", None)
        if self._uh_set_forcing is not None and hasattr(self._uh, "prepare_arrays"):
            self._uh.prepare_arrays(
                np.zeros(len(self._datetime)), self._datetime, dt_hours=None
            )
        else:
            self._uh_set_forcing = None

        if self._seasonality is not None:
            self._seasonality.prepare(self._season_input)

        self._state = ModelState.PREPARED

    def predict_arrays(self) -> dict[str, np.ndarray]:
        """Run abstraction → UH (→ seasonality), returning plain arrays.

        The children are prepared in :meth:`prepare`; this only pushes the
        current parameter values down and re-runs them, so a calibration
        evaluation builds no DataFrames at all.

        :returns: Mapping with ``datetime`` and ``Q_pred``.
        :rtype: dict[str, numpy.ndarray]
        :raises RuntimeError: If :meth:`prepare` has not been called.
        """
        if self._abs_input is None:
            raise RuntimeError("Call prepare(data) before predict().")
        self._sync_children()

        exc_arrays = self._abstraction.predict_arrays()
        exc_col = self._exc_col
        if exc_col is None or exc_col not in exc_arrays:
            exc_col = next(c for c in exc_arrays if c != "datetime")
        effective = np.asarray(exc_arrays[exc_col], dtype=float)

        if self._uh_set_forcing is not None:
            self._uh_set_forcing(effective)
        else:
            self._uh.prepare(
                pd.DataFrame({"datetime": self._datetime, "rain": effective})
            )
        q = np.asarray(self._uh.predict_arrays()["Q_pred"], dtype=float)

        if self._seasonality is not None and self._season_output is not None:
            pf = np.asarray(
                self._seasonality.predict_arrays()[self._season_output], dtype=float
            )
            q = q * pf

        self._state = ModelState.PREDICTED
        return {"datetime": self._datetime, "Q_pred": q}

    def predict(self) -> pd.DataFrame:
        """Run abstraction → UH (→ seasonality) and return predicted flow.

        :returns: DataFrame with columns ``datetime`` and ``Q_pred``.
        :rtype: pandas.DataFrame
        :raises RuntimeError: If :meth:`prepare` has not been called.
        """
        return pd.DataFrame(self.predict_arrays())

    def finalize(self) -> None:
        """Finalize all child models and advance to FINALIZED.

        :returns: Nothing.
        :rtype: None
        """
        for child, _ in self._children:
            try:
                child.finalize()
            except Exception:
                pass
        self._abs_input = None
        self._season_input = None
        self._state = ModelState.FINALIZED

    def inequality_constraints(self) -> list[float]:
        """Return concatenated child inequality-constraint residuals.

        :returns: Residuals (``g <= 0`` feasible) from every child model in
            registration order.
        :rtype: list[float]
        """
        self._sync_children()
        residuals: list[float] = []
        for child, _ in self._children:
            residuals.extend(child.inequality_constraints())
        return residuals


__all__ = ["AbstractionUHModel"]
