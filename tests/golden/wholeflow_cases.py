"""Whole-flow model cases pinned in ``wholeflow.npz``.

The pins were first written from the original inheritance-based
``CompactWholeFlowModel`` / ``PatternWholeFlowModel`` and guard their
replacement by component presets. Each case is pinned at its default
parameters and at one random feasible vector.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sparsehydro.models.forcing import SnowConfig

DT = 1.0
DAYS = 14

#: Prediction columns pinned for every case (when the model produces them).
PINNED_COLUMNS = (
    "total_cfs", "sanitary_cfs", "gwi_cfs", "rdii_cfs", "baseflow_cfs", "S_in", "G_in",
    "excess_in", "et_in", "liquid_input_in", "SWE_in", "rtk1_cfs", "routing_storage_in",
)

#: Budget entries, in a fixed order.
BUDGET_KEYS = (
    "initial_storage_in", "rain_in", "initial_snow_in", "final_snow_in", "snowfall_in",
    "melt_in", "liquid_input_in", "et_in", "gwi_in", "rdii_in", "unrouted_in",
    "final_soil_in", "final_gwi_in", "final_routing_in", "max_abs_error_in",
)


def forcing() -> pd.DataFrame:
    """Two weeks of hourly rain with temperatures crossing freezing."""
    t = pd.date_range("2024-02-01", periods=int(DAYS * 24 / DT), freq=pd.Timedelta(hours=DT))
    n = len(t)
    rain = np.zeros(n)
    for start, depth in ((10, 0.08), (60, 0.05), (130, 0.12), (200, 0.04), (260, 0.1)):
        rain[start:start + 6] = depth
    temp = 2.0 + 8.0 * np.sin(np.arange(n) * 2 * np.pi / (24 * 5)) - 3.0 * np.cos(t.hour * np.pi / 12)
    flow = 0.7 + 0.1 * np.sin(t.hour.to_numpy() * np.pi / 12)
    return pd.DataFrame({"datetime": t, "rainfall_in": rain, "temperature_c": temp, "flow_cfs": flow})


def _pattern():
    from sparsehydro.models.baseflow import CenteredWeeklyPattern

    h = 0.1 * np.sin(np.arange(24) * np.pi / 12)
    d = np.array([0.02, 0.02, 0.02, 0.02, 0.02, -0.05, -0.05])
    return CenteredWeeklyPattern(tuple(h - h.mean()), tuple(d - d.mean()))


def _background(data):
    from sparsehydro.models.baseflow import estimate_observed_background

    return estimate_observed_background(data, np.ones(len(data), bool), _pattern(),
                                        window_days=3, minimum_days=1, max_gap_days=10)


SNOW = SnowConfig(threshold_c=0.5, melt_factor_in_per_c_day=0.05, initial_swe_in=0.2)


def build(case: str, data: pd.DataFrame):
    """Return an initialized model for *case*."""
    from sparsehydro.models.presets.wholeflow import CompactWholeFlowModel
    from sparsehydro.models.baseflow import SanitaryTemplate
    from sparsehydro.models.presets.wholeflow import PatternWholeFlowModel

    hourly = 1.0 + 0.2 * np.sin(np.arange(24) * np.pi / 12)
    template = SanitaryTemplate(tuple(hourly / hourly.mean()), (1.0,) * 7)
    p = _pattern()
    factories = {
        "pattern_gw": lambda: PatternWholeFlowModel(p, 20.0, DT, 0.6, 2.0),
        "pattern_gw_snowfixed": lambda: PatternWholeFlowModel(p, 20.0, DT, 0.6, 2.0, snow_config=SNOW),
        "pattern_gw_snowcal": lambda: PatternWholeFlowModel(p, 20.0, DT, 0.6, 2.0, snow_config=SNOW,
                                                            calibrate_snow=True),
        "pattern_nogw": lambda: PatternWholeFlowModel(p, 20.0, DT, 0.6, 2.0, groundwater_enabled=False),
        "pattern_tri1": lambda: PatternWholeFlowModel(p, 20.0, DT, 0.6, 2.0, n_triangles=1),
        "pattern_tri4": lambda: PatternWholeFlowModel(p, 20.0, DT, 0.6, 2.0, n_triangles=4),
        "pattern_tri5": lambda: PatternWholeFlowModel(p, 20.0, DT, 0.6, 2.0, n_triangles=5),
        "pattern_bg": lambda: PatternWholeFlowModel(p, 20.0, DT, 0.6, 2.0, groundwater_enabled=False,
                                                    observed_background=_background(data)),
        "compact_default": lambda: CompactWholeFlowModel(template, 20.0, DT, 0.6, 2.0),
        "compact_snowcal": lambda: CompactWholeFlowModel(template, 20.0, DT, 0.6, 2.0,
                                                         snow_config=SNOW, calibrate_snow=True),
        "compact_nogw": lambda: CompactWholeFlowModel(template, 20.0, DT, 0.6, 2.0,
                                                      groundwater_enabled=False),
    }
    model = factories[case]()
    model.initialize()
    return model


CASES = (
    "pattern_gw", "pattern_gw_snowfixed", "pattern_gw_snowcal", "pattern_nogw", "pattern_tri1",
    "pattern_tri4", "pattern_tri5", "pattern_bg", "compact_default", "compact_snowcal",
    "compact_nogw",
)


def parameter_vectors(model, seed: int) -> list[dict[str, float]]:
    """Defaults plus one random feasible vector (calibratable parameters only)."""
    defaults = {n: float(model.get_scalar_parameter(n).value) for n in model.scalar_parameter_names}
    rng = np.random.default_rng(seed)
    free = [n for n in model.scalar_parameter_names if model.get_scalar_parameter(n).calibrate]
    for _ in range(500):
        trial = dict(defaults)
        for n in free:
            p = model.get_scalar_parameter(n)
            lo, hi = p.lower_bound, p.upper_bound
            band = (0.01, 0.12) if n.startswith("R_") else (0.15, 0.85)
            trial[n] = float(lo + (band[0] + (band[1] - band[0]) * rng.random()) * (hi - lo))
        for n, v in trial.items():
            model.apply_flat_parameter(n, v)
        if model.validate() and all(g <= 0 for g in model.inequality_constraints()):
            break
    else:  # pragma: no cover - generator safeguard
        raise RuntimeError("no feasible random vector found")
    for n, v in defaults.items():
        model.apply_flat_parameter(n, v)
    return [defaults, trial]


def run(model, data, values: dict[str, float]) -> tuple[pd.DataFrame, dict, list[float]]:
    """Apply *values*, prepare and predict; return (prediction, budget, residuals)."""
    for n, v in values.items():
        model.apply_flat_parameter(n, v)
    model.prepare(data)
    prediction = model.predict()
    return prediction, dict(model.budget), list(model.inequality_constraints())


def build_pins() -> dict[str, np.ndarray]:
    """All arrays for ``wholeflow.npz``."""
    data = forcing()
    out: dict[str, np.ndarray] = {}
    for seed, case in enumerate(CASES):
        model = build(case, data)
        names = model.scalar_parameter_names
        out[f"{case}|param_names"] = np.array(names)
        out[f"{case}|lower"] = np.array([model.get_scalar_parameter(n).lower_bound for n in names])
        out[f"{case}|upper"] = np.array([model.get_scalar_parameter(n).upper_bound for n in names])
        out[f"{case}|calibrate"] = np.array([model.get_scalar_parameter(n).calibrate for n in names])
        out[f"{case}|constraints"] = np.array(model.inequality_constraint_names)
        for i, values in enumerate(parameter_vectors(model, seed)):
            prediction, budget, residuals = run(model, data, values)
            out[f"{case}|x{i}|values"] = np.array([values[n] for n in names])
            out[f"{case}|x{i}|columns"] = np.array(list(prediction.columns))
            out[f"{case}|x{i}|budget"] = np.array([budget[k] for k in BUDGET_KEYS])
            out[f"{case}|x{i}|residuals"] = np.array(residuals)
            for col in PINNED_COLUMNS:
                if col in prediction:
                    out[f"{case}|x{i}|{col}"] = prediction[col].to_numpy(dtype=float)
    return out
