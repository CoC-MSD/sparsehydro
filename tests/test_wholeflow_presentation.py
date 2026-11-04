import numpy as np
import pandas as pd
import pytest

from sparsehydro.visualization.display import thin_for_display
from sparsehydro.visualization.wholeflow_presentation import plot_wholeflow_presentation


@pytest.fixture
def inputs():
    times = pd.date_range("2025-01-01", periods=12, freq="15min")
    warmup = np.arange(12) < 2
    train = np.isin(np.arange(12), [2, 3, 4, 8, 9])
    val = np.isin(np.arange(12), [5, 6, 11])
    return dict(
        data=pd.DataFrame(dict(datetime=times, flow_cfs=np.arange(12) + 10.,
                               rainfall_in=np.arange(12) / 100)),
        prediction=pd.DataFrame(dict(datetime=times, total_cfs=np.arange(12) + 3.,
                                     baseflow_cfs=np.full(12, 3.), rdii_cfs=np.arange(12))),
        warmup_mask=warmup, train_mask=train, val_mask=val,
        event_source="auto",
        events=pd.DataFrame(dict(start_idx=[3, 5, 12], end_idx=[3, 6, 15],
                                 eligible=[True, True, False])),
        calibration_events=pd.DataFrame(dict(start_idx=[8], end_idx=[9], eligible=[True])),
        event_mask=train | val, data_title="Test site", candidate_label="Diagnostic Pareto 7",
        selection_status="closest_threshold_shortfall", spinup_mode="synthetic wrap",
    )


def test_figures_use_supplied_candidate_and_preserve_inputs(inputs):
    original = inputs["data"].copy(deep=True)
    prediction = inputs["prediction"].copy(deep=True)
    figures = plot_wholeflow_presentation(**inputs)
    assert len(figures) == 4
    best, components, zones, storms = figures.values()
    np.testing.assert_array_equal(best.data[2].y, prediction.total_cfs[2:])
    np.testing.assert_array_equal(components.data[1].y, prediction.total_cfs[2:])
    np.testing.assert_array_equal(components.data[2].y, prediction.baseflow_cfs[2:])
    np.testing.assert_array_equal(components.data[3].y, prediction.rdii_cfs[2:])
    assert components.data[1].yaxis == components.data[3].yaxis == "y2"
    assert components.data[2].yaxis == "y3"
    assert "own vertical scale" in components.layout.annotations[0].text
    assert np.isnan(np.asarray(best.data[1].y)[5])  # excluded row 7 is not observed
    assert all("Pareto band" not in trace.name for fig in figures.values() for trace in fig.data)
    assert "Diagnostic Pareto 7" in best.layout.title.text
    assert "validation-informed" in best.layout.annotations[0].text
    for fig in figures.values():
        assert fig.layout.width / fig.layout.height == 16 / 9
        assert fig.layout.yaxis.autorange == "reversed"
        assert fig.layout.yaxis.title.text == "Rain [in/15 min]"
        assert all(isinstance(value, str) for value in fig.layout.xaxis.range)
        assert fig.layout.title.yanchor == "top"
    pd.testing.assert_frame_equal(inputs["data"], original)
    pd.testing.assert_frame_equal(inputs["prediction"], prediction)


def test_exact_split_boundaries_including_single_step_and_final_run(inputs):
    zones = plot_wholeflow_presentation(**inputs)["03_warmup_training_validation"]
    times = inputs["data"].datetime
    expected = [(0, 2), (2, 5), (8, 10), (5, 7), (11, 12), (7, 8), (10, 11)]
    assert len(zones.layout.shapes) == len(expected)
    for shape, (start, stop) in zip(zones.layout.shapes, expected):
        assert isinstance(shape.x0, str) and isinstance(shape.x1, str)
        assert pd.Timestamp(shape.x0) == times.iloc[start]
        assert pd.Timestamp(shape.x1) == times.iloc[stop - 1] + pd.Timedelta(minutes=15)
        assert shape.yref == "paper"  # rainfall and flow panels
        assert shape.layer == "below"
    assert len(zones.data[0].x) == 12  # includes the forcing-only warmup
    assert np.isnan(zones.data[1].y[:2]).all()


@pytest.mark.parametrize("source", ["auto", "file", "hybrid"])
def test_selected_event_source_and_scoring_coverage(inputs, source):
    inputs["event_source"] = source
    storms = plot_wholeflow_presentation(**inputs)["04_selected_storm_windows"]
    times = inputs["data"].datetime
    coverage = [s for s in storms.layout.shapes if s.name == "Storm scoring coverage"]
    assert len(coverage) == 3
    raw_name = {"auto": "Automatic detection", "file": "Imported approved events",
                "hybrid": "Calibration EVT windows"}[source]
    raw = [s for s in storms.layout.shapes if s.name == raw_name]
    if source == "hybrid":
        assert len(raw) == 1
        assert pd.Timestamp(raw[0].x0) == times.iloc[8]
        approved = [s for s in storms.layout.shapes if s.name == "Approved event windows"]
        assert len(approved) == 2
        assert pd.Timestamp(approved[0].x0) == times.iloc[3]
    else:
        assert len(raw) == 2
        assert pd.Timestamp(raw[0].x0) == times.iloc[3]
        assert pd.Timestamp(raw[0].x1) == times.iloc[4]  # one-step event is visible


def test_empty_events_no_validation_or_warmup(inputs):
    inputs["warmup_mask"][:] = False
    inputs["train_mask"][:] = True
    inputs["val_mask"][:] = False
    inputs["events"] = inputs["events"].iloc[:0]
    inputs["event_mask"][:] = False
    figures = plot_wholeflow_presentation(**inputs)
    assert len(figures["01_best_solution_hydrograph"].data[0].x) == 12
    assert not figures["04_selected_storm_windows"].layout.shapes
    assert "0 eligible of 0" in figures["04_selected_storm_windows"].layout.title.text
    assert len(figures["03_warmup_training_validation"].layout.shapes) == 1


def test_thinning_does_not_change_full_resolution_figures(inputs):
    figures = plot_wholeflow_presentation(**inputs)
    components = figures["02_modeled_flow_components"]
    thin = thin_for_display(components, 8)
    assert len(components.data[1].y) == 10
    assert max(thin.data[1].y) == max(components.data[1].y)
    np.testing.assert_allclose(
        np.asarray(thin.data[1].y), np.asarray(thin.data[2].y) + np.asarray(thin.data[3].y))


@pytest.mark.parametrize("problem, message", [
    ("source", "event_source"), ("overlap", "must not overlap"),
    ("alignment", "timestamps must match"), ("components", "baseflow \\+ RDII"),
    ("event_bounds", "Eligible event indices"), ("mask", "boolean array"),
    ("coverage", "inside training/validation"), ("rain", "rain must be nonnegative"),
])
def test_invalid_inputs_are_explicit(inputs, problem, message):
    if problem == "source":
        inputs["event_source"] = "other"
    elif problem == "overlap":
        inputs["train_mask"][0] = True
    elif problem == "alignment":
        inputs["prediction"]["datetime"] += pd.Timedelta(minutes=15)
    elif problem == "components":
        inputs["prediction"]["total_cfs"] += 1
    elif problem == "event_bounds":
        inputs["events"].loc[0, "end_idx"] = 12
    elif problem == "mask":
        inputs["val_mask"] = inputs["val_mask"].astype(int)
    elif problem == "coverage":
        inputs["event_mask"][0] = True
    elif problem == "rain":
        inputs["data"].loc[0, "rainfall_in"] = -1
    with pytest.raises(ValueError, match=message):
        plot_wholeflow_presentation(**inputs)
