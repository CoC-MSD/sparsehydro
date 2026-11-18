"""Find storm events, fit a unit hydrograph, and score the fit storm by storm.

A single NSE over a month hides how well individual storms are reproduced.
This detects the storms in the sample, then reports metrics for each one.

Run from the repository root::

    python examples/04_events_and_event_scores.py
"""

import numpy as np

from sparsehydro.calibration import CalibrationProblem, ScipySolver
from sparsehydro.events import detect_events, events_to_dataframe, merge_by_interevent, screen_small_rainfall
from sparsehydro.metrics import KGE, NashSutcliffe, per_event_scores, summarize_event_scores
from sparsehydro.models.unithydrograph import GammaUH
from sparsehydro.sample_data import load_sample

data = load_sample("tc-nb-001")
data["stormflow"] = data["stormflow"].clip(lower=0.0)  # see example 01

# 1. Events: Savitzky-Golay smoothing plus peak finding on the stormflow.
#    The percentile thresholds (0-100) control how small a peak still counts,
#    so detection also picks up dry-weather wiggles. Screening sorts them out:
#    merge peaks less than 6 h apart into one storm, then keep storms with at
#    least 0.1 in of rain.
candidates, _ = detect_events(data, height_perc=25.0, prom_perc=25.0, width_perc=10.0)
merged = merge_by_interevent(candidates, data, interevent_hours=6.0)
events = [e for e in screen_small_rainfall(merged, min_rain_in=0.1) if e.qualifying]
table = events_to_dataframe(events)
print(f"{len(candidates)} candidate peaks -> {len(merged)} after merging -> {len(events)} storms")
print(table[["event_id", "start_datetime", "end_datetime", "total_rain", "peak_flow"]].round(3).to_string(index=False))

# A label per time step: the event id inside an event, -1 elsewhere.
event_ids = np.full(len(data), -1)
for event in events:
    inside = (data["datetime"] >= event.start_datetime) & (data["datetime"] <= event.end_datetime)
    event_ids[inside.to_numpy()] = event.event_id

# 2. Fit a gamma unit hydrograph to the whole month (as in example 01).
model = GammaUH()
model.initialize()
model.get_scalar_parameter("A").upper_bound = 50.0
model.get_scalar_parameter("alpha").upper_bound = 10.0
model.get_scalar_parameter("theta").upper_bound = 24.0
model.validate()
problem = CalibrationProblem(model, data, objectives=[NashSutcliffe()],
                             column_map={"observed": "stormflow", "predicted": "Q_pred"})
result = ScipySolver(maxiter=40, seed=1).solve(problem)
for name, value in zip(problem.param_names, result.best_by("nash_sutcliffe")):
    model.apply_flat_parameter(name, value)
model.validate()
model.prepare(data)
predicted = model.predict_arrays()["Q_pred"]

# 3. Score every event on its own.
scores = per_event_scores(
    data["stormflow"].to_numpy(), predicted, event_ids,
    objectives={"nse": NashSutcliffe(), "kge": KGE()},
    datetime=data["datetime"],
)
print("\nPer-event scores")
print(scores.round(3).to_string())

summary = summarize_event_scores(scores, metrics=("nse", "kge"))
print(f"\nMedian event NSE {summary['nse_median']:.3f} "
      f"(interquartile range {summary['nse_iqr']:.3f}, {summary['nse_n']:.0f} events)")
print("A fit to the whole month is dominated by the largest storm; small storms "
      "can still be poor. Fitting storm by storm (calibration.SequentialFitter) "
      "or weighting objectives with event masks (calibration.splits) addresses this.")
