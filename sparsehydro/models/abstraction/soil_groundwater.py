"""Soil store S feeding rainfall excess and a deep groundwater store G.

Each step, rain (or liquid input after snow) recharges S with the
initial-abstraction rule ``min(P, room, room * (1 - exp(-k_dep P)))``; the
rest is **rainfall excess**. S drains continuously by temperature-dependent
evapotranspiration (``k0 + kT exp(theta (T - T_ref))`` above ``T_freeze``) and
by diversion ``k_div`` to G. G is capped at ``g_max`` (overflow spills to the
sewer) and releases slowly at ``k_G``; release plus spill is **groundwater
infiltration** (GWI). Both stores conserve mass, which :class:`SoilGroundwaterStore`
checks every prediction.

The store emits ``p_excess_in`` (to route through an RDII model) and
``gwi_in`` / ``gwi_cfs`` (to add to baseflow), so one component drives both
pathways inside a :class:`~sparsehydro.models.composite.CompositeModel`.
"""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
import pandas as pd

from ..._numba import HAS_NUMBA, njit  # noqa: F401 - HAS_NUMBA re-exported
from ...enums import ModelState
from ...parameters import ConstraintRecord, FieldRecord, ScalarParameter
from ...registry import registry
from ...units import ACRE_INCH_TO_FT3
from ..base import IModel
from ..unithydrograph.kernels import infer_dt_hours_from_values

__all__ = ["SoilGroundwaterStore", "soil_groundwater_core", "g_at_cap"]

#: Largest acceptable mass-balance error [in].
BUDGET_TOLERANCE_IN = 1e-8


def g_at_cap(G, g_max, k_G, dt_hours):
    """1.0 where G is saturated (it spilled this step), else 0.0.

    After a spill G is clamped to ``g_max`` and decays one step, so it sits at
    ``g_max * exp(-k_G dt)``; without a spill it is strictly below.
    """
    return (G >= g_max * math.exp(-k_G * dt_hours) * (1.0 - 1e-9)).astype(float)


@njit(fastmath=False, cache=True, nogil=True)
def soil_groundwater_core(rain, temp, dt, s_max, k_dep, k0, kT, theta, T_ref, T_freeze,
                          k_div, k_G, g_max, s_initial=0.0, g_initial=0.0):
    """Run the S/G recurrence; return ``(S, G, excess, et, gwi)`` depth arrays."""
    n = rain.shape[0]
    S = np.zeros(n)
    G = np.zeros(n)
    excess = np.zeros(n)
    et = np.zeros(n)
    gwi = np.zeros(n)
    if not (dt > 0.0 and 0.0 <= s_initial <= s_max
            and 0.0 <= g_initial <= g_max):
        raise ValueError("Invalid timestep or initial store depth")
    s = s_initial
    gg = g_initial
    for i in range(n):
        P = rain[i]
        T = temp[i]
        if P > 0.0:                                  # recharge S (initial abstraction)
            capacity = max(0.0, s_max - s)
            infil = min(P, capacity, capacity * (-math.expm1(-k_dep * P)))
            s += infil
            excess[i] = P - infil
        if T >= T_freeze:                            # ET/recovery drain rate on S
            k_rec = k0 + kT * math.exp(theta * (T - T_ref))
        else:
            k_rec = 0.0
        k_tot = k_rec + k_div                         # ET and diversion act together
        to_G = 0.0
        if k_tot > 0.0 and s > 0.0:
            drained = s * (1.0 - math.exp(-k_tot * dt))
            to_G = drained * (k_div / k_tot)
            et[i] = drained * (k_rec / k_tot)
            s = s * math.exp(-k_tot * dt)
        if s < 0.0:
            s = 0.0
        elif s > s_max:
            s = s_max
        gg += to_G                                    # deep store: recharge, cap, release
        spill = 0.0
        if gg > g_max:
            spill = gg - g_max
            gg = g_max
        release = 0.0
        if gg > 0.0 and k_G > 0.0:
            release = gg * (1.0 - math.exp(-k_G * dt))
            gg = gg * math.exp(-k_G * dt)
        gwi[i] = release + spill
        S[i] = s
        G[i] = gg
    return S, G, excess, et, gwi


@registry.register
class SoilGroundwaterStore(IModel):
    """Soil store S with rainfall excess and an optional deep groundwater store G.

    Depths in inches, area in acres, flow in cfs.

    :param area_acres: Contributing area (converts GWI depth to flow).
    :param groundwater: ``False`` disables diversion to G (``k_div = 0``) and
        fixes ``k_div``, ``k_G``, ``g_max`` and ``g_initial_fraction``.
    :param rainfall_col: Data column used when ``liquid_input_in`` is not bound.
    :param temperature_col: Temperature column [degC].

    Inputs: ``liquid_input_in`` (e.g. from :class:`~sparsehydro.models.forcing.DegreeDaySnow`).
    Constraints (with groundwater): ``g_max_lt_s_max``, ``k_G_lt_k0``.
    """

    model_name: ClassVar[str] = "soil-groundwater-store"
    supports_array_predict: ClassVar[bool] = True

    def __init__(self, area_acres: float = 100.0, *, groundwater: bool = True,
                 rainfall_col: str = "rainfall_in", temperature_col: str = "temperature_c") -> None:
        super().__init__()
        if not isinstance(groundwater, bool):
            raise ValueError("groundwater must be a boolean")
        self._area_init = float(area_acres)
        self.groundwater = groundwater
        self.rainfall_col = rainfall_col
        self.temperature_col = temperature_col
        self._liquid: np.ndarray | None = None
        self._temp: np.ndarray | None = None
        self._datetime: np.ndarray | None = None
        self._dt_hours = 1.0
        #: Initial (S, G) depths of the last prediction [in].
        self.initial_storage: tuple[float, float] = (0.0, 0.0)

    @property
    def input_field_names(self) -> list[str]:
        return ["liquid_input_in"]

    def initialize(self) -> None:
        self._reset_registries()
        for name, value, lo, hi, units, description, calibrate in (
            ("s_max", 0.5, 0.05, 1.0, "in", "Soil store S capacity", True),
            ("ia_k_dep", 5.0, 1.0, 30.0, "1/in", "Abstraction (recharge) rate per inch of rain", True),
            ("ia_k0", 0.01, 1e-4, 0.2, "1/hr", "Base ET/recovery drain rate on S", True),
            ("ia_kT", 0.01, 0.0, 0.2, "1/hr", "Temperature-sensitive recovery coefficient", True),
            ("ia_theta", 0.08, 0.0, 0.2, "1/degC", "Temperature sensitivity exponent", True),
            ("ia_T_ref", 10.0, 0.0, 30.0, "degC", "Reference temperature", False),
            ("ia_T_freeze", 0.0, -5.0, 5.0, "degC", "Recovery suppressed below this temperature", False),
            ("k_div", 5e-3, 1e-4, 5e-2, "1/hr", "S -> G diversion rate", True),
            ("k_G", 5e-4, 1e-5, 2e-2, "1/hr", "G -> sewer slow GWI release rate", True),
            ("g_max", 0.1, 1e-2, 0.3, "in", "Deep store G capacity (< s_max)", True),
            ("area_acres", self._area_init, 1.0, 1e5, "acre", "Contributing area (depth -> cfs)", False),
            ("s_initial_fraction", 0.0, 0.0, 1.0, "-", "Initial S / s_max", False),
            ("g_initial_fraction", 0.0, 0.0, 1.0, "-", "Initial G / g_max", False),
        ):
            self.register_scalar_parameter(ScalarParameter(
                name, value=value, lower_bound=lo, upper_bound=hi, units=units,
                description=description, calibrate=calibrate))
        if not self.groundwater:
            self.get_scalar_parameter("k_div").update(value=0.0, lower_bound=0.0)
            for name in ("k_div", "k_G", "g_max"):
                self.get_scalar_parameter(name).calibrate = False
        else:
            self.register_inequality_constraint(ConstraintRecord(
                "g_max_lt_s_max", "Deep store smaller than S (g_max - s_max <= 0)"))
            self.register_inequality_constraint(ConstraintRecord(
                "k_G_lt_k0", "Deep store slower than S (k_G - k0 <= 0)"))
        self.register_output_field(FieldRecord("datetime", description="Simulation time step"))
        for name, units, description, role in (
            ("S_in", "in", "Soil store depth", "state"), ("G_in", "in", "Deep store depth", "state"),
            ("p_excess_in", "in", "Rainfall excess", "flux"),
            ("G_at_cap", "-", "1 where G is saturated", "state"),
            ("et_in", "in", "Evapotranspiration from S", "flux"),
            ("recharge_in", "in", "Liquid input retained by S", "flux"),
            ("gwi_in", "in", "Groundwater infiltration (release + spill)", "flux"),
            ("gwi_release_in", "in", "Slow release from G", "flux"),
            ("gwi_spill_in", "in", "Overflow from a full G", "flux"),
            ("gwi_cfs", "cfs", "Groundwater infiltration flow", "output"),
            ("budget_error_in", "in", "Cumulative store mass-balance error", "state"),
        ):
            self.register_output_field(FieldRecord(name, units=units, description=description, role=role))
        self._state = ModelState.INITIALIZED

    def inequality_constraints(self) -> list[float]:
        if not self.groundwater:
            return []
        g = self._p
        return [g("g_max") - g("s_max"), g("k_G") - g("ia_k0")]

    def _p(self, name: str) -> float:
        return float(self._scalar_parameters[name].value)

    def validate(self) -> bool:
        g = self._p
        ok = (self.parameters_valid() and g("ia_T_freeze") < g("ia_T_ref")
              and (not self.groundwater or (g("g_max") <= g("s_max") and g("k_G") <= g("ia_k0")))
              and (self.groundwater or (g("k_div") == 0 and g("g_initial_fraction") == 0)))
        if ok:
            self._state = ModelState.VALIDATED
        return ok

    def prepare(self, data: pd.DataFrame, **_: object) -> None:
        self._datetime = pd.to_datetime(data["datetime"]).to_numpy()
        self._dt_hours = infer_dt_hours_from_values(self._datetime)
        self._liquid = data[self.rainfall_col].to_numpy(dtype=float)
        self._temp = data[self.temperature_col].to_numpy(dtype=float)
        if not (np.isfinite(self._liquid).all() and np.isfinite(self._temp).all()):
            raise ValueError("Store forcing (rain and temperature) must be complete")
        self._state = ModelState.PREPARED

    def set_input(self, name: str, values: np.ndarray) -> None:
        if name != "liquid_input_in":
            raise KeyError(f"SoilGroundwaterStore input is 'liquid_input_in'; got {name!r}")
        arr = np.asarray(values, dtype=float)
        if self._liquid is None or arr.shape != self._liquid.shape:
            raise ValueError("liquid_input_in must match the prepared series")
        self._liquid = arr

    def predict_arrays(self) -> dict[str, np.ndarray]:
        if self._liquid is None:
            raise RuntimeError("Call prepare() before predict().")
        g = self._p
        dt = self._dt_hours
        s0 = g("s_initial_fraction") * g("s_max")
        g0 = g("g_initial_fraction") * g("g_max")
        self.initial_storage = (s0, g0)
        liquid = np.ascontiguousarray(self._liquid, dtype=float)
        S, G, excess, et, gwi = soil_groundwater_core(
            liquid, np.ascontiguousarray(self._temp), dt,
            g("s_max"), g("ia_k_dep"), g("ia_k0"), g("ia_kT"), g("ia_theta"),
            g("ia_T_ref"), g("ia_T_freeze"), g("k_div"), g("k_G"), g("g_max"), s0, g0)
        error = s0 + g0 + np.cumsum(liquid - et - gwi - excess) - (S + G)
        if np.abs(error).max() > BUDGET_TOLERANCE_IN:
            raise ArithmeticError("Soil/groundwater store mass balance did not close within 1e-8 inch")
        release = G * np.expm1(g("k_G") * dt)
        step_cfs = g("area_acres") * ACRE_INCH_TO_FT3 / (dt * 3600)
        self._state = ModelState.PREDICTED
        return {
            "datetime": self._datetime, "S_in": S, "G_in": G, "p_excess_in": excess,
            "G_at_cap": g_at_cap(G, g("g_max"), g("k_G"), dt), "et_in": et,
            "recharge_in": liquid - excess, "gwi_in": gwi, "gwi_release_in": release,
            "gwi_spill_in": gwi - release, "gwi_cfs": gwi * step_cfs, "budget_error_in": error,
        }

    def predict(self) -> pd.DataFrame:
        return pd.DataFrame(self.predict_arrays())

    def finalize(self) -> None:
        self._liquid = self._temp = None
        self._state = ModelState.FINALIZED
