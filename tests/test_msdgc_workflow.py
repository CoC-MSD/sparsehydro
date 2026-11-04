import copy

import numpy as np
import pandas as pd
import pytest

from sparsehydro.calibration.msdgc_assessment import CFS_H_TO_MG, MSDGCAssessment, MSDGCCon15, MSDGCSettings, _zone_window, assessment_event_inventory
from sparsehydro.calibration.objectives import Con15
from sparsehydro.calibration.wholeflow_acceptance import AcceptanceContract, lin_ccc


@pytest.mark.parametrize("duration,minutes,counts", [
    (3, [15, 45, 135], [3, 9, 27]),
    (4, [15, 60, 180], [3, 12, 36]),
    (6, [15.12, 90, 270], [3, 18, 54]),
    (8, [20.16, 120, 360], [4, 24, 72]),
    (12, [30, 180, 540], [6, 36, 108]),
    (48, [30, 240, 1440], [6, 48, 288]),
])
def test_saved_tool_zone_rules(duration, minutes, counts):
    settings = MSDGCSettings()
    actual = [settings.window(duration, 1 / 12, zone) for zone in settings.zones]
    np.testing.assert_allclose([row[0] for row in actual], minutes)
    assert [row[1] for row in actual] == counts


def test_native_sample_counts_use_midpoint_to_even_not_ceiling():
    settings = MSDGCSettings(minimum_minutes=(12.5, 17.5, 22.5),
                             maximum_minutes=(12.5, 17.5, 22.5))
    assert [settings.window(4, 1 / 12, zone)[1] for zone in settings.zones] == [2, 4, 4]
    assert settings.window(4.004, 1 / 12, "conveyance")[0] == 12.5


def test_moving_window_keeps_first_tie_and_includes_final_endpoint():
    q = np.array([1., 9, 2, 8, 7, 0, 0]) / (CFS_H_TO_MG / 12)
    row = _zone_window(q, 0, 6, 4, 1 / 12, MSDGCSettings(), "conveyance")
    assert row["volume_mg"] == 19
    assert (row["window_start_idx"], row["window_end_idx"]) == (1, 3)
    q[-3:] = 20 / (CFS_H_TO_MG / 12)
    row = _zone_window(q, 0, 6, 4, 1 / 12, MSDGCSettings(), "conveyance")
    assert row["volume_mg"] == 60 and row["window_end_idx"] == 6


def make_case():
    times = pd.date_range("2025-01-01", periods=4 * 49, freq="5min")
    observed = np.repeat([1., 2., 4., 8.], 49)
    events = pd.DataFrame(dict(
        event_id=["1", "2", "3", "4"], start_idx=[0, 49, 98, 147],
        end_idx=[48, 97, 146, 195], duration_hours=4., split="train",
    ))
    mask = np.ones(len(times), bool)
    evaluator = MSDGCAssessment(observed, events, mask, mask, 1 / 12, AcceptanceContract())
    return observed, events, mask, evaluator


def test_independent_windows_do_not_penalize_a_pure_in_event_time_shift():
    observed, events, mask, _ = make_case()
    observed[:] = 0
    modeled = observed.copy()
    for i, event in enumerate(events.itertuples(), 1):
        observed[event.start_idx + 5:event.start_idx + 8] = i
        modeled[event.start_idx + 20:event.start_idx + 23] = i
    evaluator = MSDGCAssessment(observed, events, mask, mask, 1 / 12, AcceptanceContract())
    report = evaluator.zone_report(modeled)
    np.testing.assert_array_equal(report.observed_volume_mg, report.modeled_volume_mg)
    conv = report.query("zone == 'conveyance'")
    assert (conv.modeled_window_start_idx - conv.observed_window_start_idx).eq(15).all()
    scores = evaluator.metrics(modeled).set_index("metric")
    assert scores.loc["conveyance_ccc", "value"] == pytest.approx(1)


def test_volumes_are_rounded_in_mg_before_ccc_and_slope():
    observed, events, mask, evaluator = make_case()
    model = observed * np.repeat([1.2, .8, 1.1, .9], 49)
    report = evaluator.zone_report(model)
    scores = evaluator.metrics(model).set_index("metric")
    for zone in MSDGCSettings.zones:
        pairs = report.query("zone == @zone")
        x, y = pairs.observed_volume_mg.to_numpy(), pairs.modeled_volume_mg.to_numpy()
        np.testing.assert_allclose(x, np.round(x, 4))
        assert scores.loc[f"{zone}_ccc", "value"] == pytest.approx(lin_ccc(x, y))
    assert scores.loc["three_zone_mean_ccc", "value"] == pytest.approx(
        scores.loc[["conveyance_ccc", "storage_ccc", "treatment_ccc"], "value"].mean())
    definition = MSDGCCon15(events[["start_idx", "end_idx"]].to_numpy(),
                             events.duration_hours.to_numpy(), 1 / 12, mask=mask)
    assert scores.loc["con15_slope", "value"] == definition.slope(observed, model)
    assert scores.flow_basis.eq("total_flow").all()
    assert not scores.official_storm_scorecard.any()


def test_volume_objective_ranks_broad_crest_not_highest_instantaneous_peak():
    observed = np.zeros(98)
    observed[5] = 10
    observed[60:63] = 5
    modeled = observed.copy()
    modeled[60:63] *= 2
    windows = np.array([[0, 48], [49, 97]])
    objective = MSDGCCon15(windows, np.array([4., 4.]), 1 / 12, top_fraction=.5)
    assert objective.selected_event_indices(observed).tolist() == [1]
    assert Con15(windows, top_fraction=.5).selected_event_indices(observed).tolist() == [0]
    expected = round(30 * CFS_H_TO_MG / 12, 4) / round(15 * CFS_H_TO_MG / 12, 4) - 1
    assert objective.compute(observed, modeled) == pytest.approx(expected)
    assert objective.name == "con15_error" and objective.minimize
    assert copy.deepcopy(objective).compute(observed, modeled) == objective.compute(observed, modeled)


def test_objective_filters_complete_training_events_before_ranking():
    observed, events, mask, _ = make_case()
    mask[98:] = False
    objective = MSDGCCon15(events[["start_idx", "end_idx"]].to_numpy(),
                           events.duration_hours.to_numpy(), 1 / 12, .5, mask=mask)
    assert objective.selected_event_indices(observed).tolist() == [1]
    changed = observed.copy()
    changed[98:] = np.nan
    assert objective.compute(changed, observed) == 0
    changed[50] = np.nan
    with pytest.raises(ValueError, match="finite samples"):
        objective.compute(changed, observed)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -99.])
def test_bad_event_scan_invalidates_zone_aggregate_without_dropping_storm(bad):
    observed, _, _, evaluator = make_case()
    model = observed.copy()
    model[1] = bad
    scores = evaluator.metrics(model).set_index("metric")
    assert scores.loc["conveyance_ccc", "status"] == "undefined"
    assert scores.loc["conveyance_ccc", "n"] == 4
    assert scores.loc["conveyance_ccc", "n_defined"] == 3
    assert scores.loc["three_zone_mean_ccc", "status"] == "undefined"


def test_constant_modeled_zone_is_undefined_like_tool_pearson_expression():
    observed, _, _, evaluator = make_case()
    scores = evaluator.metrics(np.ones_like(observed)).set_index("metric")
    assert scores.loc["conveyance_ccc", "status"] == "undefined"


def test_empty_report_has_explicit_shapes_and_undefined_metrics():
    observed, events, mask, _ = make_case()
    evaluator = MSDGCAssessment(observed, events.iloc[:0], mask, mask, 1 / 12,
                                AcceptanceContract())
    assert evaluator.zone_report(observed).empty
    assert evaluator.conveyance_report(observed).empty
    assert evaluator.event_report(observed).empty
    assert evaluator.metrics(observed).status.eq("undefined").all()


def test_missing_observed_band_does_not_export_partial_con15_ranking():
    observed, events, mask, _ = make_case()
    observed[1] = np.nan
    evaluator = MSDGCAssessment(observed, events, mask, mask, 1 / 12, AcceptanceContract())
    report = evaluator.event_report(np.ones_like(observed))
    assert not report.con15_selected.any()
    assert report.con15_basis.eq(MSDGCCon15.basis).all()
    assert evaluator.metrics(np.ones_like(observed)).query(
        "metric == 'con15_slope'").status.eq("undefined").all()


def test_unsupported_resolution_and_short_event_are_explicit_not_clipped():
    observed, events, mask, _ = make_case()
    evaluator = MSDGCAssessment(observed, events.iloc[:1], mask, mask, 1,
                                AcceptanceContract())
    assert evaluator.zone_report(observed).iloc[0].reason == "Requested window rounds to zero native samples"
    short = events.iloc[:1].assign(end_idx=1)
    objective = MSDGCCon15(short[["start_idx", "end_idx"]].to_numpy(),
                           short.duration_hours.to_numpy(), 1 / 12)
    with pytest.raises(ValueError, match="exceeds event"):
        objective.compute(observed, observed)


def test_assessment_exclusion_87_and_evt_end_lookup_do_not_change_broad_masks():
    times = pd.date_range("2025-01-01", periods=100, freq="5min")
    train = np.arange(100) < 60
    validation = ~train
    approved = pd.DataFrame(dict(
        event_id=["87", "88"], start=[times[0], times[65]],
        end=[times[48] + pd.Timedelta(seconds=12), times[95]],
        start_idx=[0, 65], end_idx=[48, 95], duration_hours=[4.0033, 2.5],
        split=["train", "validation"], eligible=True, exclusion="",
    ))
    original = approved.copy(deep=True)
    before_train = train.copy()
    full = approved.copy()
    full.loc[0, "end"] = times[48]
    result = assessment_event_inventory(times, approved, train, validation, full_inventory=full)
    assert not result.loc[0, "eligible"]
    assert result.loc[0, "exclusion"] == "assessment_criteria_exclusion"
    assert result.loc[0, "end_idx"] == 48
    assert result.loc[1, "eligible"] and result.loc[1, "split"] == "validation"
    assert result.assessment_window_source.eq("full_evt").all()
    pd.testing.assert_frame_equal(approved, original)
    np.testing.assert_array_equal(train, before_train)
    result = assessment_event_inventory(times, approved, train, validation, excluded_event_ids=())
    assert result.loc[0, "end_idx"] == 49
    train[49] = False
    validation[49] = True
    result = assessment_event_inventory(times, approved, train, validation, excluded_event_ids=())
    assert not result.loc[0, "eligible"]
    assert "crosses_split_or_unscored" in result.loc[0, "exclusion"]


@pytest.mark.parametrize("kwargs", [
    {"fractions": (0., .25, .75)}, {"minimum_minutes": (30., 20., 30.),
                                   "maximum_minutes": (15., 240., 1440.)},
    {"fractions": (float("nan"), .25, .75)},
])
def test_invalid_rules_raise(kwargs):
    with pytest.raises(ValueError, match="zone fractions"):
        MSDGCSettings(**kwargs)
