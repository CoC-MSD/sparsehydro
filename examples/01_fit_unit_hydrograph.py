"""Fit one unit hydrograph to a month of rain and stormflow.

The bundled sample is March 2022 at flow meter TC-NB-001: 5-minute rainfall
[in] and the separated wet-weather stormflow [cfs].

Run from the repository root::

    python examples/01_fit_unit_hydrograph.py          # prints the fit
    python examples/01_fit_unit_hydrograph.py --plot   # also opens a plot (needs plotly)
"""

import sys

from sparsehydro.calibration import CalibrationProblem, ScipySolver
from sparsehydro.metrics import NashSutcliffe, fit_summary
from sparsehydro.models.unithydrograph import GammaUH
from sparsehydro.sample_data import load_sample

# 1. Data: a DataFrame with datetime, rain and stormflow columns. The meter's
#    separated stormflow dips slightly below zero in dry weather; those
#    negatives add up over a month and distort volume comparisons, so clip them.
data = load_sample("tc-nb-001")
data["stormflow"] = data["stormflow"].clip(lower=0.0)

# 2. Model: a gamma unit hydrograph. It reads the "rain" column and returns
#    "Q_pred". Its parameters (A = effective area, alpha = shape,
#    theta = scale in hours) start at their defaults.
model = GammaUH()
model.initialize()

#    The default bounds cover any catchment (time scales up to 240 h). Narrow
#    them to what is plausible for this small sewershed: most of the default
#    range gives a response far longer than the month of data, and the
#    optimiser can get lost there.
model.get_scalar_parameter("A").upper_bound = 50.0       # effective area
model.get_scalar_parameter("alpha").upper_bound = 10.0   # shape
model.get_scalar_parameter("theta").upper_bound = 24.0   # scale [h]
model.validate()

# 3. Problem: which column is observed, which model output is compared, and
#    what "good" means (Nash-Sutcliffe efficiency, maximised).
problem = CalibrationProblem(
    model,
    data,
    objectives=[NashSutcliffe()],
    column_map={"observed": "stormflow", "predicted": "Q_pred"},
)
print("Calibrating:", ", ".join(problem.param_names))

# 4. Solve: SciPy's differential evolution. Raise maxiter for a closer fit.
result = ScipySolver(method="differential_evolution", maxiter=40, seed=1).solve(problem)

# 5. Put the best parameters back into the model and predict.
best = result.best_by("nash_sutcliffe")
for name, value in zip(problem.param_names, best):
    model.apply_flat_parameter(name, value)
    print(f"  {name:>6} = {value:.4g}")
model.validate()
model.prepare(data)
predicted = model.predict_arrays()["Q_pred"]

summary = fit_summary(data["stormflow"].to_numpy(), predicted)
print(f"NSE {summary['nse']:.3f} | percent bias {summary['pbias']:+.1f}% | "
      f"peak error {summary['peak_error_pct']:+.1f}%")

if "--plot" in sys.argv:
    from sparsehydro.visualization import plot_timeseries

    plot_timeseries(data["datetime"], data["rain"].to_numpy(), data["stormflow"], predicted,
                    title="Gamma UH fit", rainfall_label="Rain [in]", flow_label="Stormflow [cfs]").show()
