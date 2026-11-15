"""Compose a model from registered components, entirely from configuration.

An initial-abstraction store takes the first part of each storm (more when
the ground has had time to dry out); the rain that gets past it is routed by two
unit hydrographs in parallel, a fast and a slow response, and their sum is the
predicted stormflow::

    rain --> ia --> excess --+--> fast (Nash UH) --+
                             |                     +--> stormflow_cfs
                             +--> slow (gamma UH) -+

No new class is written: the wiring is data. The same dictionary could live in
a JSON or YAML file and be read with ``load_config``.

Run from the repository root::

    python examples/03_composite_from_config.py
"""

import json
import tempfile
from pathlib import Path

from sparsehydro.calibration import CalibrationProblem
from sparsehydro.configuration import build_model, build_objectives, build_solver, load_config
from sparsehydro.metrics import fit_summary
from sparsehydro.sample_data import load_sample

CONFIG = {
    "model": {
        "type": "composite",
        "components": {
            # Each component is any registered model, by name, with constructor
            # arguments. "inputs" binds a component input to another output.
            "ia": {"type": "initial-abstraction"},
            "fast": {"type": "nash-uh", "n": 2.0, "K": 0.5,
                     "inputs": {"rain": "ia.p_excess_in"}},
            "slow": {"type": "gamma-uh", "alpha": 1.0, "theta": 8.0,
                     "inputs": {"rain": "ia.p_excess_in"}},
        },
        # What the composite returns: here the sum of the two routed flows.
        "outputs": {"stormflow_cfs": {"sum": ["fast.Q_pred", "slow.Q_pred"]}},
    },
    # Parameter names are "<component>.<parameter>". Set values, bounds, or
    # calibrate=False to hold a parameter fixed.
    "parameters": {
        # The sample has no air temperature, so hold the temperature terms of
        # the abstraction store fixed and calibrate only its capacity and
        # recovery rate.
        "ia.ia_kT": {"calibrate": False},
        "ia.ia_theta": {"calibrate": False},
        "ia.ia_T_ref": {"calibrate": False},
        "ia.ia_k_dep": {"calibrate": False},
        "ia.ia_T_freeze": {"calibrate": False},
        "ia.ia_max": {"upper_bound": 0.5},
        "fast.A": {"value": 5.0, "upper_bound": 50.0},
        "fast.n": {"upper_bound": 5.0},
        "fast.K": {"upper_bound": 4.0},
        "slow.A": {"value": 5.0, "upper_bound": 50.0},
        "slow.alpha": {"upper_bound": 5.0},
        "slow.theta": {"lower_bound": 2.0, "upper_bound": 48.0},
    },
    "objectives": ["nash_sutcliffe"],
    "solver": {"type": "scipy", "maxiter": 40, "seed": 1},
}

# Round-trip through a JSON file to show the file-based workflow (a temporary
# directory, so nothing is written into the repository).
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "model.json"
    path.write_text(json.dumps(CONFIG, indent=2), encoding="utf-8")
    config = load_config(path)

model = build_model(config)
print("Components run in order:", " -> ".join(model.run_order))
print("Calibrated parameters:", ", ".join(
    p for p in model.scalar_parameter_names if model.get_scalar_parameter(p).calibrate))

data = load_sample("tc-nb-001")
data["stormflow"] = data["stormflow"].clip(lower=0.0)  # see example 01
# The abstraction store reads "rainfall_in" (and "temperature_c" when present;
# without it the store uses its reference temperature, hence the fixed
# temperature terms above).
data = data.rename(columns={"rain": "rainfall_in"})

problem = CalibrationProblem(
    model, data,
    objectives=build_objectives(config["objectives"]),
    column_map={"observed": "stormflow", "predicted": "stormflow_cfs"},
)
result = build_solver(config["solver"]).solve(problem)

for name, value in zip(problem.param_names, result.best_by("nash_sutcliffe")):
    model.apply_flat_parameter(name, value)
    print(f"  {name:>14} = {value:.4g}")
model.validate()
model.prepare(data)
outputs = model.predict_arrays()

summary = fit_summary(data["stormflow"].to_numpy(), outputs["stormflow_cfs"])
print(f"NSE {summary['nse']:.3f} | percent bias {summary['pbias']:+.1f}% | "
      f"peak error {summary['peak_error_pct']:+.1f}%")
