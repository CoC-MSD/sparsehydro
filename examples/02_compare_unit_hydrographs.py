"""Fit several unit hydrograph formulations and compare them.

Every unit hydrograph is registered by name, so trying another formulation is
a one-word change. This fits each one the same way and tabulates the fit.

Run from the repository root::

    python examples/02_compare_unit_hydrographs.py
"""

import pandas as pd

from sparsehydro.calibration import CalibrationProblem, ScipySolver
from sparsehydro.configuration import build_model
from sparsehydro.metrics import KGE, NashSutcliffe, fit_summary, kge
from sparsehydro.sample_data import load_sample

data = load_sample("tc-nb-001")
data["stormflow"] = data["stormflow"].clip(lower=0.0)  # see example 01
observed = data["stormflow"].to_numpy()

# Registered names, each with upper bounds suited to a small sewershed (the
# library defaults allow time scales up to 240 h; see example 01). Parameter
# names differ by formulation; docs/unit_hydrographs.md lists them all.
FORMULATIONS = {
    "nash-uh": {"n": 10.0, "K": 24.0},
    "gamma-uh": {"alpha": 10.0, "theta": 24.0},
    "scs-uh": {"Tp": 24.0},
    "weibull-uh": {"lam": 48.0},
    "lognormal-uh": {"mu": 4.0},
}

rows = []
for name, upper in FORMULATIONS.items():
    # A config dict builds and initializes the model; "parameters" overrides
    # any registered parameter's value, bounds or calibrate flag.
    settings = {"A": {"upper_bound": 50.0}}
    settings.update({param: {"upper_bound": bound} for param, bound in upper.items()})
    model = build_model({"model": {"type": name}, "parameters": settings})
    problem = CalibrationProblem(
        model, data,
        objectives=[NashSutcliffe(), KGE()],
        column_map={"observed": "stormflow", "predicted": "Q_pred"},
    )
    # With two objectives ScipySolver optimises the first (NSE); KGE is reported.
    result = ScipySolver(maxiter=30, seed=1).solve(problem)
    for param, value in zip(problem.param_names, result.best_by("nash_sutcliffe")):
        model.apply_flat_parameter(param, value)
    model.validate()
    model.prepare(data)
    predicted = model.predict_arrays()["Q_pred"]

    summary = fit_summary(observed, predicted)
    rows.append({
        "formulation": name,
        "parameters": len(problem.param_names),
        "NSE": summary["nse"],
        "KGE": kge(observed, predicted),
        "bias %": summary["pbias"],
        "peak error %": summary["peak_error_pct"],
    })
    print(f"fitted {name}")

table = pd.DataFrame(rows).set_index("formulation").sort_values("NSE", ascending=False)
print()
print(table.round(3).to_string())
print("\nA Nash cascade with a non-integer number of reservoirs is the gamma "
      "distribution, so those two rows should match.")
