"""Regression checks for the reference notebook's corrected volume formulas."""

import copy

import numpy as np
import pandas as pd
import pytest

from sparsehydro.calibration import CalibrationProblem, ConcordanceCorrelationCoefficient, NashSutcliffe
from sparsehydro.calibration.msdgc_assessment import (
    CFS_H_TO_MG, MSDGCAssessment, MSDGCCon15, MSDGCSettings, _zone_window,
)
from sparsehydro.models import SeasonalityModel


@pytest.mark.parametrize("duration,minutes,counts", [
    (3, [15, 45, 135], [3, 9, 27]),
    (4, [15, 60, 180], [3, 12, 36]),
    (6, [15.12, 90, 270], [3, 18, 54]),
    (8, [20.16, 120, 360], [4, 24, 72]),
    (12, [30, 180, 540], [6, 36, 108]),
    (48, [30, 240, 1440], [6, 48, 288]),
])
def test_reference_zone_rules(duration, minutes, counts):
    settings = MSDGCSettings()
    actual = [settings.window(duration, 1 / 12, zone) for zone in settings.zones]
    np.testing.assert_allclose([row[0] for row in actual], minutes)
    assert [row[1] for row in actual] == counts


def test_midpoint_to_even_not_ceiling():
    settings = MSDGCSettings(minimum_minutes=(12.5, 17.5, 22.5),
                             maximum_minutes=(12.5, 17.5, 22.5))
    assert [settings.window(4, 1 / 12, zone)[1] for zone in settings.zones] == [2, 4, 4]


def test_native_scan_keeps_first_tie_and_includes_final_endpoint():
    q = np.array([1., 9, 2, 8, 7, 0, 0]) / (CFS_H_TO_MG / 12)
    row = _zone_window(q, 0, 6, 4, 1 / 12, MSDGCSettings(), "conveyance")
    assert row["volume_mg"] == 19
    assert (row["window_start_idx"], row["window_end_idx"]) == (1, 3)
    q[-3:] = 20 / (CFS_H_TO_MG / 12)
    row = _zone_window(q, 0, 6, 4, 1 / 12, MSDGCSettings(), "conveyance")
    assert row["volume_mg"] == 60 and row["window_end_idx"] == 6


@pytest.fixture
def case():
    observed = np.repeat([1., 2., 4., 8.], 49)
    events = pd.DataFrame(dict(
        event_id=["1", "2", "3", "4"], start_idx=[0, 49, 98, 147],
        end_idx=[48, 97, 146, 195], duration_hours=4.,
    ))
    mask = np.ones(len(observed), bool)
    return observed, events, mask


def objective_for(events, mask=None, fraction=.15):
    return MSDGCCon15(events[["start_idx", "end_idx"]].to_numpy(),
                      events.duration_hours.to_numpy(), 1 / 12, fraction, mask)


def test_independent_windows_and_mg_rounding(case):
    observed, events, mask = case
    observed[:] = 0
    modeled = observed.copy()
    for i, event in enumerate(events.itertuples(), 1):
        observed[event.start_idx + 5:event.start_idx + 8] = i
        modeled[event.start_idx + 20:event.start_idx + 23] = i
    evaluator = MSDGCAssessment(observed, events, mask, 1 / 12)
    report = evaluator.zone_report(modeled)
    np.testing.assert_array_equal(report.observed_volume_mg, report.modeled_volume_mg)
    np.testing.assert_array_equal(report.observed_volume_mg, report.observed_volume_mg.round(4))
    conveyance = report.query("zone == 'conveyance'")
    assert (conveyance.modeled_window_start_idx - conveyance.observed_window_start_idx).eq(15).all()
    scores = evaluator.metrics(modeled).set_index("metric")
    assert scores.loc["conveyance_ccc", "value"] == pytest.approx(1)
    assert scores.loc["con15_slope", "value"] == objective_for(events).slope(observed, modeled)
    assert not scores.official_storm_scorecard.any()
    assert scores.flow_basis.eq("total_flow").all()


def test_con15_selects_broad_volume_crest_not_instantaneous_peak():
    observed = np.zeros(98)
    observed[5] = 10
    observed[60:63] = 5
    modeled = observed.copy()
    modeled[60:63] *= 2
    objective = MSDGCCon15(np.array([[0, 48], [49, 97]]), np.array([4., 4.]), 1 / 12, .5)
    assert objective.selected_event_indices(observed).tolist() == [1]
    expected = round(30 * CFS_H_TO_MG / 12, 4) / round(15 * CFS_H_TO_MG / 12, 4) - 1
    assert objective.compute(observed, modeled) == pytest.approx(expected)
    assert copy.deepcopy(objective).compute(observed, modeled) == objective.compute(observed, modeled)
    assert objective.name == "con15_error" and objective.minimize


def test_complete_training_events_ranked_before_fraction_and_no_missing_sample_drop(case):
    observed, events, mask = case
    mask[98:] = False
    objective = objective_for(events, mask, .5)
    assert objective.selected_event_indices(observed).tolist() == [1]
    changed = observed.copy()
    changed[98:] = np.nan
    assert objective.compute(changed, observed) == 0
    changed[50] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        objective.compute(changed, observed)
    mask[90:98] = False
    assert objective_for(events, mask, .5).selected_event_indices(observed).tolist() == [0]


def test_calibration_fast_path_matches_full_axis_and_preserves_default_metrics(case):
    observed, events, mask = case
    mask[:49] = False
    mask[147:] = False
    model = SeasonalityModel(include_hour=True, include_dow=False, include_month=False,
                             output_name="sanitary_cfs")
    model.initialize()
    assert model.validate()
    data = pd.DataFrame(dict(datetime=pd.date_range("2025-01-01", periods=len(mask), freq="5min"),
                             observed=observed))
    con15 = objective_for(events, mask)
    problem = CalibrationProblem(
        model=model, data=data,
        objectives=[NashSutcliffe(), ConcordanceCorrelationCoefficient(), con15],
        column_map={"observed": "observed", "predicted": "sanitary_cfs"}, mask=mask,
    )
    lower, upper = problem.bounds
    x = (lower + upper) / 2
    scores = problem.evaluate(x, penalty_weight=0)
    predicted = model.predict()["sanitary_cfs"].to_numpy()
    assert scores[0] == pytest.approx(-NashSutcliffe().compute(observed, predicted, mask))
    assert scores[1] == pytest.approx(-ConcordanceCorrelationCoefficient().compute(observed, predicted, mask))
    assert scores[2] == pytest.approx(con15.compute(observed, predicted))
    assert problem._obj_ctx_ok[2]
    assert problem.make_copy().evaluate(x, penalty_weight=0)[2] == pytest.approx(scores[2])
    fallback = CalibrationProblem(
        model=model, data=data, objectives=[objective_for(events)],
        column_map={"observed": "observed", "predicted": "sanitary_cfs"}, mask=mask,
    )
    assert fallback.evaluate(x, penalty_weight=0)[0] == pytest.approx(scores[2])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -99.])
def test_bad_event_invalidates_zone_aggregate_without_dropping_storm(case, bad):
    observed, events, mask = case
    model = observed.copy()
    model[1] = bad
    scores = MSDGCAssessment(observed, events, mask, 1 / 12).metrics(model).set_index("metric")
    assert scores.loc["conveyance_ccc", "status"] == "undefined"
    assert scores.loc["conveyance_ccc", "n"] == 4
    assert scores.loc["conveyance_ccc", "n_defined"] == 3
    assert scores.loc["three_zone_mean_ccc", "status"] == "undefined"


def test_degenerate_empty_and_unsupported_reports(case):
    observed, events, mask = case
    evaluator = MSDGCAssessment(observed, events, mask, 1 / 12)
    assert evaluator.metrics(np.ones_like(observed)).query(
        "metric == 'conveyance_ccc'").status.eq("undefined").all()
    empty = MSDGCAssessment(observed, events.iloc[:0], mask, 1 / 12)
    assert empty.zone_report(observed).empty
    assert empty.metrics(observed).status.eq("undefined").all()
    hourly = MSDGCAssessment(observed, events.iloc[:1], mask, 1)
    assert hourly.zone_report(observed).iloc[0].reason == "Requested window rounds to zero native samples"
    short = events.iloc[:1].assign(end_idx=1)
    with pytest.raises(ValueError, match="exceeds event"):
        objective_for(short).compute(observed, observed)


def test_split_crossing_event_is_rejected(case):
    observed, events, mask = case
    mask[48] = False
    with pytest.raises(ValueError, match="own split"):
        MSDGCAssessment(observed, events, mask, 1 / 12)


@pytest.mark.parametrize("fraction", [0, -1, 1.01, np.nan, np.inf])
def test_bad_objective_fractions_raise(case, fraction):
    _, events, _ = case
    with pytest.raises(ValueError, match="top_fraction"):
        objective_for(events, fraction=fraction)
