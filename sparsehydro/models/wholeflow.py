"""Joint whole-flow RDII model: shared IA store drives BOTH storm RTK and GWI.

This module hosts the ``WholeFlowModel`` (and its numba store core ``_gwi_B_core``)
that ``flowfinity_gwi_wholeflow_joint.ipynb`` calibrates, plus the two *staged*
models — ``GwiStoreModelB`` (Stage-1 GWI stores) and ``RTKOnFrozenExcess`` (Stage-2
RTK on the frozen excess) — that ``flowfinity_gwi_wholeflow_eventmask.ipynb`` and the
Stage-1 cell of the joint notebook use.  They all live in the package —
rather than being defined in a notebook cell — for one concrete reason: the
:class:`~sparsehydro.calibration.solvers.optuna_solver.OptunaCmaEsSolver` multi-seed
path runs each CMA-ES seed in a separate ``loky`` process, and ``cloudpickle`` can
only serialise the problem (hence the model) into those workers when the model class
is importable by qualified name.  A notebook-``__main__`` class cannot be pickled, so
the joint calibration was pinned to ``n_jobs=1`` (sequential seeds).  Importing
``WholeFlowModel`` from here unlocks ``n_jobs > 1`` — the seeds run concurrently.

The store core is a ``@njit`` loop identical to the Method-B GWI store (see
``flowfinity_gwi_secondary_store.ipynb``); because it now lives in a real source file
it can use ``cache=True`` (a cell-defined njit could not — it had no file to cache to).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve

from sparsehydro.models import IModel
from sparsehydro.models.rdii import RTKTriangle
from sparsehydro.parameters import ScalarParameter, FieldRecord, ConstraintRecord
from sparsehydro.enums import ModelState


try:
    from numba import njit

    _HAS_NUMBA = True
except Exception:                       # numba not installed -> pure-Python fallback
    _HAS_NUMBA = False

    def njit(*a, **k):                  # noqa: D401 - no-op decorator shim
        if len(a) == 1 and callable(a[0]) and not k:
            return a[0]

        def _w(f):
            return f

        return _w


# in.acre/hr -> cfs (same base the two-stage RTK path used).
_IN_AC_PER_HR_TO_CFS = 43560.0 / (12.0 * 3600.0)

# Initial (R, T[hr], K) + search bounds for fast / medium / slow triangles.
J_INIT_RTK = [(0.03, 1.0, 2.0), (0.02, 8.0, 2.5), (0.01, 48.0, 3.0)]
J_T_BOUNDS = [(0.1, 6.0), (2.0, 48.0), (12.0, 120.0)]
J_K_BOUNDS = (1.05, 4.0)

# The staged (two-stage) notebook uses the same RTK init/bounds under its own names.
INIT_RTK = J_INIT_RTK
RTK_T_BOUNDS = J_T_BOUNDS
RTK_K_BOUNDS = J_K_BOUNDS

# GWI/IA knobs shared with GwiStoreModelB (order matters for warm-start copy).
GWI_CAL = ("s_max", "ia_k_dep", "ia_k0", "ia_kT", "ia_theta", "k_div", "k_G", "g_max")


@njit(fastmath=False, cache=True)   # real source file -> cache=True is valid & cheap
def _gwi_B_core(rain, temp, dt, s_max, k_dep, k0, kT, theta, T_ref, T_freeze,
                k_div, k_G, g_max):
    # S: rain recharges; ET (k_rec) + diversion (k_div) drain it EVERY step.
    # G: filled by the S->G diversion (capped at g_max; overflow spills to sewer);
    #    releases GWI slowly at k_G. Both stores mass-conserving.
    n = rain.shape[0]
    S = np.zeros(n)
    G = np.zeros(n)
    excess = np.zeros(n)
    et = np.zeros(n)
    gwi = np.zeros(n)
    s = 0.0
    gg = 0.0
    for i in range(n):
        P = rain[i]
        T = temp[i]
        if P > 0.0:                                  # recharge S (IAModel abstraction)
            infil = (s_max - s) * (1.0 - math.exp(-k_dep * P))
            if infil < 0.0:
                infil = 0.0
            s += infil
            excess[i] = P - infil
        if T >= T_freeze:                            # ET/recovery drain rate on S
            k_rec = k0 + kT * math.exp(theta * (T - T_ref))
        else:
            k_rec = 0.0
        k_tot = k_rec + k_div                         # ET + diversion act simultaneously
        to_G = 0.0
        if k_tot > 0.0 and s > 0.0:
            drained = s * (1.0 - math.exp(-k_tot * dt))
            to_G = drained * (k_div / k_tot)          # diversion share -> G
            et[i] = drained * (k_rec / k_tot)         # ET share -> atmosphere
            s = s * math.exp(-k_tot * dt)
        if s < 0.0:
            s = 0.0
        elif s > s_max:
            s = s_max
        # Deep store G: recharge, cap (overflow spills to sewer), then slow release.
        gg += to_G
        spill = 0.0
        if gg > g_max:
            spill = gg - g_max
            gg = g_max
        release = 0.0
        if gg > 0.0 and k_G > 0.0:
            release = gg * (1.0 - math.exp(-k_G * dt))
            gg = gg * math.exp(-k_G * dt)
        gwi[i] = release + spill                       # both reach the sewer as GWI
        S[i] = s
        G[i] = gg
    return S, G, excess, et, gwi


class WholeFlowModel(IModel):
    """Joint GWI-stores + RTK model on the SHARED, live initial-abstraction store.

    Reuses :func:`_gwi_B_core` for the S/G stores; convolves the SAME-step excess
    through 3 RTK triangles. Nothing is frozen: every IA knob change re-derives both
    the excess (-> RDII) and the GWI, so a single objective can trade the two off.
    """

    model_name = "wholeflow-joint"

    def __init__(self, floor_cfs, area_acres, dt_hours, init_rtk,
                 rainfall_col="rainfall_in", temp_col="temperature_c"):
        super().__init__()
        self._floor = np.asarray(floor_cfs, dtype=float)
        self._area_acres = float(area_acres)
        self._dt_hours = float(dt_hours)
        self._init = list(init_rtk)
        self._rain_col = rainfall_col
        self._temp_col = temp_col
        self._df = None
        self._rain = None
        self._temp = None
        self._tri = [RTKTriangle(R=R, T=T, K=K) for R, T, K in self._init]

    def initialize(self):
        # --- GWI / IA store parameters (identical to GwiStoreModelB) ----------
        self.register_scalar_parameter(ScalarParameter(
            "s_max", value=0.5, lower_bound=0.05, upper_bound=1.0,
            units="in", description="IA store S capacity"))
        self.register_scalar_parameter(ScalarParameter(
            "ia_k_dep", value=5.0, lower_bound=1.0, upper_bound=30.0,
            units="1/in", description="Abstraction (recharge) rate per inch of rain"))
        self.register_scalar_parameter(ScalarParameter(
            "ia_k0", value=0.01, lower_bound=1e-4, upper_bound=0.2,
            units="1/hr", description="Base ET/recovery drain rate on S"))
        self.register_scalar_parameter(ScalarParameter(
            "ia_kT", value=0.01, lower_bound=0.0, upper_bound=0.2,
            units="1/hr", description="Temperature-sensitive recovery coefficient"))
        self.register_scalar_parameter(ScalarParameter(
            "ia_theta", value=0.08, lower_bound=0.0, upper_bound=0.2,
            units="1/degC", description="Temperature sensitivity exponent"))
        self.register_scalar_parameter(ScalarParameter(
            "ia_T_ref", value=10.0, lower_bound=0.0, upper_bound=30.0,
            units="degC", description="Reference temperature"))
        self.register_scalar_parameter(ScalarParameter(
            "ia_T_freeze", value=0.0, lower_bound=-5.0, upper_bound=5.0,
            units="degC", description="Recovery suppressed below this temperature"))
        self.register_scalar_parameter(ScalarParameter(
            "k_div", value=5e-3, lower_bound=1e-4, upper_bound=5e-2,
            units="1/hr", description="S -> G diversion rate"))
        self.register_scalar_parameter(ScalarParameter(
            "k_G", value=5e-4, lower_bound=1e-5, upper_bound=2e-2,
            units="1/hr", description="G -> sewer slow GWI release rate"))
        self.register_scalar_parameter(ScalarParameter(
            "g_max", value=0.1, lower_bound=1e-2, upper_bound=0.3,
            units="in", description="Deep store G capacity (< s_max)"))
        self.register_scalar_parameter(ScalarParameter(
            "area_acres", value=self._area_acres, lower_bound=1.0, upper_bound=1e5,
            units="acre", description="Contributing area (depth->cfs scale; fixed)"))
        for nm in ("area_acres", "ia_T_ref", "ia_T_freeze"):
            self.get_scalar_parameter(nm).calibrate = False

        # --- RTK triangle parameters (fast / medium / slow) -------------------
        for t in self._tri:
            t.initialize()
        for i, (R, T, K) in enumerate(self._init, 1):
            self.register_scalar_parameter(ScalarParameter(
                f"R_{i}", value=R, lower_bound=0.0, upper_bound=1.0, units="-",
                description=f"Pathway {i}: fraction of live excess routed through this UH"))
            lo, hi = J_T_BOUNDS[i - 1]
            self.register_scalar_parameter(ScalarParameter(
                f"uh{i}_T", value=T, lower_bound=lo, upper_bound=hi, units="hr",
                description=f"Pathway {i}: time to peak"))
            self.register_scalar_parameter(ScalarParameter(
                f"uh{i}_K", value=K, lower_bound=J_K_BOUNDS[0], upper_bound=J_K_BOUNDS[1],
                units="-", description=f"Pathway {i}: recession-to-peak ratio"))

        # --- Constraints (residual <= 0 feasible; penalised by the solver) ----
        self.register_inequality_constraint(ConstraintRecord(
            name="g_max_lt_s_max", description="Deep store smaller than S (g_max - s_max <= 0)"))
        self.register_inequality_constraint(ConstraintRecord(
            name="k_G_lt_k0", description="Deep store slower than S (k_G - k0 <= 0)"))
        self.register_inequality_constraint(ConstraintRecord(
            name="sum_R_leq_1", description="R_1+...+R_N <= 1 (runoff fraction <= 100%)"))

        for nm, u, d in (
            ("S_in", "in", "IA store filled depth S"),
            ("G_in", "in", "Deep store filled depth G"),
            ("excess_in", "in", "Live rainfall excess feeding the RTK triangles"),
            ("gwi_cfs", "cfs", "GWI released from G, converted to flow"),
            ("baseflow_cfs", "cfs", "Sanitary floor + GWI"),
            ("rdii_cfs", "cfs", "RTK RDII convolved from the live excess"),
            ("total_cfs", "cfs", "Whole flow = floor + GWI + RDII"),
        ):
            self.register_output_field(FieldRecord(name=nm, units=u, description=d,
                                                   calibratable=(nm in ("baseflow_cfs",
                                                                        "rdii_cfs", "total_cfs"))))
        self._state = ModelState.INITIALIZED

    def validate(self):
        if not self.parameters_valid():
            return False
        g = self.get_scalar_parameter
        if g("ia_T_freeze").value >= g("ia_T_ref").value:
            return False
        if g("g_max").value > g("s_max").value or g("k_G").value > g("ia_k0").value:
            return False
        self._state = ModelState.VALIDATED
        return True

    def inequality_constraints(self):
        g = self.get_scalar_parameter
        return [
            g("g_max").value - g("s_max").value,                       # g_max <= s_max
            g("k_G").value - g("ia_k0").value,                         # k_G   <= k0
            sum(g(f"R_{i}").value for i in (1, 2, 3)) - 1.0,           # sum R <= 1
        ]

    def prepare(self, data):
        df = data.sort_values("datetime").reset_index(drop=True).copy()
        self._df = df
        self._rain = df[self._rain_col].fillna(0.0).clip(lower=0.0).to_numpy(float)
        _tref = self.get_scalar_parameter("ia_T_ref").value
        if self._temp_col in df.columns:
            self._temp = df[self._temp_col].fillna(_tref).to_numpy(float)
        else:
            self._temp = np.full(len(df), _tref, dtype=float)
        self._state = ModelState.PREPARED

    def predict(self):
        g = lambda nm: float(self.get_scalar_parameter(nm).value)
        # 1) Shared store: excess (-> RTK) AND gwi (-> baseflow), one live pass.
        S, G, excess, et, gwi = _gwi_B_core(
            self._rain, self._temp, self._dt_hours,
            g("s_max"), g("ia_k_dep"), g("ia_k0"), g("ia_kT"), g("ia_theta"),
            g("ia_T_ref"), g("ia_T_freeze"), g("k_div"), g("k_G"), g("g_max"))
        step_cfs = g("area_acres") * 3630.0 / (self._dt_hours * 3600.0)
        gwi_cfs = gwi * step_cfs
        # 2) Route the LIVE excess through the 3 triangles.
        n = len(excess)
        rdii = np.zeros(n)
        for i, tri in enumerate(self._tri, 1):
            tri.get_scalar_parameter("T").value = g(f"uh{i}_T")
            tri.get_scalar_parameter("K").value = g(f"uh{i}_K")
            kernel = tri.get_kernel(self._dt_hours)     # normalized [1/hr]; R applied below
            m = len(kernel)
            if max(n, m) > 500:
                conv = fftconvolve(excess, kernel, mode="full")[:n]
            else:
                conv = np.convolve(excess, kernel, mode="full")[:n]
            rdii += g(f"R_{i}") * conv
        rdii_cfs = np.clip(rdii, 0.0, None) * g("area_acres") * _IN_AC_PER_HR_TO_CFS
        baseflow_cfs = self._floor + gwi_cfs
        out = pd.DataFrame({
            "datetime": self._df["datetime"].values,
            "S_in": S, "G_in": G, "excess_in": excess,
            "gwi_cfs": gwi_cfs, "baseflow_cfs": baseflow_cfs,
            "rdii_cfs": rdii_cfs, "total_cfs": baseflow_cfs + rdii_cfs,
        })
        self._state = ModelState.PREDICTED
        return out

    def finalize(self):
        self._df = None
        self._state = ModelState.FINALIZED


class GwiStoreModelB(IModel):
    """Stage-1 GWI model: GWI released from a secondary deep store G fed by S.

    Reuses the module-level :func:`_gwi_B_core` (so it shares the numba build with
    :class:`WholeFlowModel`).  Lives in the package — rather than in a notebook cell
    — so ``cloudpickle`` can serialise the Stage-1 ``CalibrationProblem`` into loky
    workers, which is what lets the multi-seed CMA-ES run with ``n_jobs > 1``.

    ``s_max_upper`` is exposed because the two staged notebooks disagree on the S
    capacity ceiling (eventmask=1.0, joint=2.0); each passes its own value so the
    calibration search space is preserved bit-for-bit.
    """

    model_name = "gwi-store-B"

    def __init__(self, floor_cfs, area_acres, dt_hours,
                 rainfall_col="rainfall_in", temp_col="temperature_c",
                 s_max_upper=1.0):
        super().__init__()
        self._floor = np.asarray(floor_cfs, dtype=float)
        self._area_acres = float(area_acres)
        self._dt_hours = float(dt_hours)
        self._rain_col = rainfall_col
        self._temp_col = temp_col
        self._s_max_upper = float(s_max_upper)
        self._df = None
        self._rain = None
        self._temp = None

    def initialize(self):
        self.register_scalar_parameter(ScalarParameter(
            "s_max", value=0.5, lower_bound=0.05, upper_bound=self._s_max_upper,
            units="in", description="IA store S capacity"))
        self.register_scalar_parameter(ScalarParameter(
            "ia_k_dep", value=5.0, lower_bound=1.0, upper_bound=30.0,
            units="1/in", description="Abstraction (recharge) rate per inch of rain"))
        self.register_scalar_parameter(ScalarParameter(
            "ia_k0", value=0.01, lower_bound=1e-4, upper_bound=0.2,
            units="1/hr", description="Base ET/recovery drain rate on S"))
        self.register_scalar_parameter(ScalarParameter(
            "ia_kT", value=0.01, lower_bound=0.0, upper_bound=0.2,
            units="1/hr", description="Temperature-sensitive recovery coefficient"))
        self.register_scalar_parameter(ScalarParameter(
            "ia_theta", value=0.08, lower_bound=0.0, upper_bound=0.2,
            units="1/degC", description="Temperature sensitivity exponent"))
        self.register_scalar_parameter(ScalarParameter(
            "ia_T_ref", value=10.0, lower_bound=0.0, upper_bound=30.0,
            units="degC", description="Reference temperature"))
        self.register_scalar_parameter(ScalarParameter(
            "ia_T_freeze", value=0.0, lower_bound=-5.0, upper_bound=5.0,
            units="degC", description="Recovery suppressed below this temperature"))
        self.register_scalar_parameter(ScalarParameter(
            "k_div", value=5e-3, lower_bound=1e-4, upper_bound=5e-2,
            units="1/hr", description="S -> G diversion rate"))
        self.register_scalar_parameter(ScalarParameter(
            "k_G", value=5e-4, lower_bound=1e-5, upper_bound=2e-2,
            units="1/hr", description="G -> sewer slow GWI release rate"))
        self.register_scalar_parameter(ScalarParameter(
            "g_max", value=0.1, lower_bound=1e-2, upper_bound=0.3,
            units="in", description="Deep store G capacity (< s_max)"))
        self.register_scalar_parameter(ScalarParameter(
            "area_acres", value=self._area_acres, lower_bound=1.0, upper_bound=1e5,
            units="acre", description="Contributing area (depth->cfs scale; fixed)"))
        for nm in ("area_acres", "ia_T_ref", "ia_T_freeze"):
            self.get_scalar_parameter(nm).calibrate = False
        # Named inequality constraints (residual <= 0 is feasible; penalized by solver).
        self.register_inequality_constraint(ConstraintRecord(
            name="g_max_lt_s_max", description="Deep store smaller than S (g_max - s_max <= 0)"))
        self.register_inequality_constraint(ConstraintRecord(
            name="k_G_lt_k0", description="Deep store slower than S (k_G - k0 <= 0)"))
        for nm, u, d in (
            ("S_in", "in", "IA store filled depth S"),
            ("G_in", "in", "Deep store filled depth G"),
            ("excess_in", "in", "Rainfall excess (would feed the RTK; unused here)"),
            ("et_loss_in", "in", "ET/recovery loss to atmosphere"),
            ("gwi_in", "in", "GWI release (+ overflow spill) to the sewer"),
            ("gwi_cfs", "cfs", "GWI converted to flow"),
            ("model_baseflow_cfs", "cfs", "Sanitary floor + GWI"),
        ):
            self.register_output_field(FieldRecord(name=nm, units=u, description=d,
                                                   calibratable=False))
        self._state = ModelState.INITIALIZED

    def validate(self):
        if not self.parameters_valid():
            return False
        g = self.get_scalar_parameter
        if g("ia_T_freeze").value >= g("ia_T_ref").value:
            return False
        # Defaults must be feasible so a fresh model validates.
        if g("g_max").value > g("s_max").value or g("k_G").value > g("ia_k0").value:
            return False
        self._state = ModelState.VALIDATED
        return True

    def inequality_constraints(self):
        g = self.get_scalar_parameter
        return [
            g("g_max").value - g("s_max").value,   # g_max <= s_max
            g("k_G").value - g("ia_k0").value,     # k_G   <= k0
        ]

    def prepare(self, data):
        df = data.sort_values("datetime").reset_index(drop=True).copy()
        self._df = df
        self._rain = df[self._rain_col].fillna(0.0).clip(lower=0.0).to_numpy(float)
        _tref = self.get_scalar_parameter("ia_T_ref").value
        if self._temp_col in df.columns:
            self._temp = df[self._temp_col].fillna(_tref).to_numpy(float)
        else:
            self._temp = np.full(len(df), _tref, dtype=float)
        self._state = ModelState.PREPARED

    def predict(self):
        g = lambda nm: float(self.get_scalar_parameter(nm).value)
        S, G, excess, et, gwi = _gwi_B_core(
            self._rain, self._temp, self._dt_hours,
            g("s_max"), g("ia_k_dep"), g("ia_k0"), g("ia_kT"), g("ia_theta"),
            g("ia_T_ref"), g("ia_T_freeze"), g("k_div"), g("k_G"), g("g_max"))
        step_cfs = g("area_acres") * 3630.0 / (self._dt_hours * 3600.0)
        gwi_cfs = gwi * step_cfs
        out = pd.DataFrame({
            "datetime": self._df["datetime"].values,
            "S_in": S, "G_in": G, "excess_in": excess, "et_loss_in": et, "gwi_in": gwi,
            "gwi_cfs": gwi_cfs, "model_baseflow_cfs": self._floor + gwi_cfs,
        })
        self._state = ModelState.PREDICTED
        return out

    def finalize(self):
        self._df = None
        self._state = ModelState.FINALIZED


class RTKOnFrozenExcess(IModel):
    """RTK triangles convolving a FIXED excess series (Stage-2 whole-flow).

    The excess is frozen by the Stage-1 GWI fit, so this model only routes it -
    it never re-abstracts rain.  predict() returns rdii_cfs and the whole-flow
    total_cfs = frozen (floor + GWI) + RDII.

    Lives in the package (rather than a notebook cell) so the Stage-2
    ``CalibrationProblem`` can be pickled into loky workers -- the prerequisite
    for running the multi-seed CMA-ES seeds in parallel.
    """

    model_name = "rtk-on-frozen-excess"

    def __init__(self, excess_in, base_cfs, area_acres, dt_hours, init_rtk):
        super().__init__()
        self._excess = np.asarray(excess_in, dtype=float)     # frozen RTK input
        self._base = np.asarray(base_cfs, dtype=float)        # frozen floor + GWI
        self._area = float(area_acres)
        self._dt = float(dt_hours)
        self._init = list(init_rtk)
        self._tri = [RTKTriangle(R=R, T=T, K=K) for R, T, K in self._init]
        self._dtcol = None

    def initialize(self):
        for t in self._tri:
            t.initialize()
        for i, (R, T, K) in enumerate(self._init, 1):
            self.register_scalar_parameter(ScalarParameter(
                f"R_{i}", value=R, lower_bound=0.0, upper_bound=1.0, units="-",
                description=f"Pathway {i}: fraction of frozen excess routed through this UH"))
            lo, hi = RTK_T_BOUNDS[i - 1]
            self.register_scalar_parameter(ScalarParameter(
                f"uh{i}_T", value=T, lower_bound=lo, upper_bound=hi, units="hr",
                description=f"Pathway {i}: time to peak"))
            self.register_scalar_parameter(ScalarParameter(
                f"uh{i}_K", value=K, lower_bound=RTK_K_BOUNDS[0], upper_bound=RTK_K_BOUNDS[1],
                units="-", description=f"Pathway {i}: recession-to-peak ratio"))
        self.register_inequality_constraint(ConstraintRecord(
            name="sum_R_leq_1", description="R_1+...+R_N <= 1 (runoff fraction <= 100%)"))
        for nm, u, d in (
            ("rdii_cfs", "cfs", "RTK RDII convolved from the frozen excess"),
            ("total_cfs", "cfs", "Whole flow = sanitary floor + GWI + RDII"),
        ):
            self.register_output_field(FieldRecord(name=nm, units=u, description=d,
                                                   calibratable=False))
        self._state = ModelState.INITIALIZED

    def validate(self):
        if not self.parameters_valid():
            return False
        self._state = ModelState.VALIDATED
        return True

    def inequality_constraints(self):
        g = self.get_scalar_parameter
        return [sum(g(f"R_{i}").value for i in range(1, len(self._tri) + 1)) - 1.0]

    def prepare(self, data):
        self._dtcol = (data.sort_values("datetime").reset_index(drop=True)
                       ["datetime"].values)
        self._state = ModelState.PREPARED

    def predict(self):
        g = self.get_scalar_parameter
        n = len(self._excess)
        rdii = np.zeros(n)
        for i, tri in enumerate(self._tri, 1):
            tri.get_scalar_parameter("T").value = g(f"uh{i}_T").value
            tri.get_scalar_parameter("K").value = g(f"uh{i}_K").value
            kernel = tri.get_kernel(self._dt)                 # normalized [1/hr]; R NOT applied
            m = len(kernel)
            if max(n, m) > 500:
                conv = fftconvolve(self._excess, kernel, mode="full")[:n]
            else:
                conv = np.convolve(self._excess, kernel, mode="full")[:n]
            rdii += g(f"R_{i}").value * conv
        rdii_cfs = np.clip(rdii, 0.0, None) * self._area * _IN_AC_PER_HR_TO_CFS
        out = pd.DataFrame({"datetime": self._dtcol, "rdii_cfs": rdii_cfs,
                            "total_cfs": self._base + rdii_cfs})
        self._state = ModelState.PREDICTED
        return out

    def finalize(self):
        self._state = ModelState.FINALIZED
