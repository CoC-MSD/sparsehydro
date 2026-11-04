from tests.pattern_workflow_support import (
    configure_state, build_model_state, prepare_data_state, event_coverage_state,
    objective_state, calibrate_state, report_state, assessment_state,
)

import ast
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pandas as pd
import pytest

from sparsehydro.calibration.objectives import Con15, ConcordanceCorrelationCoefficient
from sparsehydro.calibration.wholeflow_workflow import storm_scoring_mask, melt_scoring_mask
from sparsehydro.events.imported import imported_event_inventory, load_event_windows


@pytest.fixture
def notebook_sources():
    path = (Path(__file__).parents[1] / "sparsehydro" / "notebooks"
            / "flowfinity_gwi_wholeflow_pattern_joint.ipynb")
    sources = {c["id"]: "".join(c["source"]) for c in json.loads(
        path.read_text(encoding="utf-8"))["cells"]}
    sources["pattern-joint-00"] = "\n".join(
        source for cell_id, source in sources.items()
        if cell_id == "pattern-joint-00" or (
            cell_id.startswith("pattern-settings-") and cell_id != "pattern-settings-index"))
    return sources


def summary_file(path, rows):
    frame = pd.DataFrame(rows, columns=["Event", "Group", "Date", "Duration (h)"])
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame([["Event summary"]]).to_excel(writer, header=False, index=False)
        frame.to_excel(writer, startrow=2, index=False)
    return path


def event_csv(path, windows):
    pd.DataFrame(windows, columns=["event_id", "start_date", "end_date"]).to_csv(path, index=False)
    return path


def model_data():
    return pd.DataFrame(dict(
        datetime=pd.date_range("2025-01-01", periods=144, freq="1h"),
        rainfall_in=np.zeros(144), flow_cfs=1 + np.arange(144) / 144,
        possible_melt=np.ones(144, bool), dry_eligible=np.zeros(144, bool),
    ))


def test_summary_title_excel_datetimes_and_rounded_duration(tmp_path):
    start = pd.Timestamp("2024-11-27 19:00")
    path = summary_file(tmp_path / "events.xlsx", [
        [12, "Qualifying Event", start, 11.92],
        [55, "Qualifying Event", pd.Timestamp("2024-11-28 06:55"), 24.0],
    ])
    events = load_event_windows(path)
    assert events.event_id.tolist() == ["12", "55"]
    assert events.source_row.tolist() == [4, 5]
    assert events.start.iloc[0] == start
    assert events.end.iloc[0] == pd.Timestamp("2024-11-28 06:55:12")
    assert events.duration_hours.iloc[0] == 11.92
    assert events.overlap_previous_hours.tolist() == pytest.approx([0, 12 / 3600])


@pytest.mark.parametrize("suffix", [".csv", ".xlsx"])
def test_explicit_windows_are_sorted_and_not_redetected(tmp_path, suffix):
    frame = pd.DataFrame({
        "event_id": ["B", "A"],
        "start_date": ["2025-01-02 12:00", "2025-01-01 10:00"],
        "end_date": ["2025-01-02 16:00", "2025-01-01 14:00"],
    })
    path = tmp_path / ("events" + suffix)
    if suffix == ".xlsx":
        frame.to_excel(path, index=False)
    else:
        frame.to_csv(path, index=False)
    events = load_event_windows(path)
    assert events.event_id.tolist() == ["A", "B"]
    assert events.source_row.tolist() == [3, 2]
    assert events.duration_hours.tolist() == [4.0, 4.0]


@pytest.mark.parametrize("start,end,message", [
    ("not-a-date", "2025-01-01", "invalid start"),
    ("2025-01-01", None, "end must be a date/time"),
    ("2025-01-02", "2025-01-01", "end must be after start"),
    ("2025-01-01", "2025-01-01", "end must be after start"),
    ("2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z", "timezone-naive"),
    (45623, 45624, "invalid start timestamp"),
])
def test_invalid_timestamps_raise_with_source_row(tmp_path, start, end, message):
    path = event_csv(tmp_path / "bad.csv", [["A", start, end]])
    with pytest.raises(ValueError, match=message) as error:
        load_event_windows(path)
    assert "row 2" in str(error.value)


@pytest.mark.parametrize("duration", [None, "bad", 0, -1, np.inf])
def test_invalid_summary_durations_raise(tmp_path, duration):
    path = summary_file(tmp_path / "bad.xlsx", [
        [12, "Qualifying Event", pd.Timestamp("2025-01-01"), duration],
    ])
    with pytest.raises(ValueError, match="duration"):
        load_event_windows(path)


def test_missing_empty_unsupported_duplicate_and_bad_headers(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_event_windows(tmp_path / "missing.xlsx")
    unsupported = tmp_path / "events.txt"
    unsupported.write_text("not supported")
    with pytest.raises(ValueError, match=".csv or .xlsx"):
        load_event_windows(unsupported)
    path = event_csv(tmp_path / "empty.csv", [])
    with pytest.raises(ValueError, match="no approved event rows"):
        load_event_windows(path)
    event_csv(path, [["A", "2025-01-01", "2025-01-02"], ["A", "2025-01-03", "2025-01-04"]])
    with pytest.raises(ValueError, match="duplicate event IDs"):
        load_event_windows(path)
    path.write_text("event_id,start,end,start_date\nA,2025-01-01,2025-01-02,2025-01-01\n")
    with pytest.raises(ValueError, match="ambiguous"):
        load_event_windows(path)
    path.write_text("unrecognized,columns\n1,2\n")
    with pytest.raises(ValueError, match="expected"):
        load_event_windows(path)
    path = summary_file(tmp_path / "missing_engine.xlsx", [])

    def missing_engine(*args, **kwargs):
        raise ImportError("No openpyxl")

    monkeypatch.setattr(pd, "read_excel", missing_engine)
    with pytest.raises(ImportError, match=r"sparsehydro\[excel\]"):
        load_event_windows(path)


@pytest.mark.parametrize("minutes", [5, 15, 60])
def test_exact_inclusive_grid_mapping_and_explicit_exclusions(tmp_path, minutes):
    times = pd.date_range("2025-01-01", periods=72 * 60 // minutes,
                          freq=pd.Timedelta(minutes=minutes))
    data = pd.DataFrame(dict(datetime=times, rainfall_in=0.1))
    train = (times >= "2025-01-01 06:00") & (times < "2025-01-02 12:00")
    validation = times >= "2025-01-02 12:00"
    path = event_csv(tmp_path / "events.csv", [
        ["outside", "2024-12-31 22:00", "2025-01-01 01:00"],
        ["warmup", "2025-01-01 02:00", "2025-01-01 04:00"],
        ["training", "2025-01-01 08:02", "2025-01-01 11:58"],
        ["crossing", "2025-01-02 11:00", "2025-01-02 13:00"],
        ["validation", "2025-01-02 15:00", "2025-01-02 18:00"],
        ["tiny", "2025-01-02 19:01", "2025-01-02 19:02"],
        ["after", "2025-01-04", "2025-01-05"],
    ])
    events = imported_event_inventory(data, train, validation, path).set_index("event_id")
    assert events.loc["outside", "exclusion"] == "outside_model_record"
    assert events.loc["after", "exclusion"] == "outside_model_record"
    assert events.loc["warmup", "exclusion"] == "warmup_or_unscored"
    assert events.loc["crossing", "exclusion"] == "crosses_split_boundary"
    assert events.loc["tiny", "exclusion"] == "insufficient_model_samples"
    assert events.index[events.eligible].tolist() == ["training", "validation"]
    mask = storm_scoring_mask(events.reset_index(), train, minutes / 60,
                              pre_hours=0, tail_hours=0)
    expected = (times >= "2025-01-01 08:02") & (times <= "2025-01-01 11:58")
    np.testing.assert_array_equal(mask, expected)
    for event in events[events.eligible].itertuples():
        assert times[event.start_idx] >= event.start
        assert times[event.end_idx] <= event.end
    changed = data.assign(flow_cfs=np.arange(len(data)) * 1000, rainfall_in=0)
    other = imported_event_inventory(changed, train, validation, path).set_index("event_id")
    pd.testing.assert_frame_equal(events.drop(columns="rain_in"), other.drop(columns="rain_in"))


def event_namespace(notebook_sources, tmp_path):
    data = model_data()
    train = np.arange(len(data)) < 72
    ns = dict(np=np, pd=pd, Path=Path, time=time, phase_seconds={},
              display=lambda value: None, model_df=data, train_mask=train, val_mask=~train,
              dry_eligible=data.dry_eligible.to_numpy(), _dt_hours=1.0,
              DOWNLOAD_DIR=tmp_path, shape_fit=SimpleNamespace(detection_baseline_cfs=np.ones(len(data))))
    exec(notebook_sources["pattern-joint-00"], ns)
    source = notebook_sources["pattern-joint-45"]
    return ns, source




def test_notebook_auto_mode_retains_padding_melt_and_ignores_filename(
    notebook_sources, tmp_path, monkeypatch,
):
    import sparsehydro.calibration.pattern_workflow as workflow

    ns, source = event_namespace(notebook_sources, tmp_path)
    events = pd.DataFrame(dict(event_id=["auto"], start_idx=[20], end_idx=[24],
                               start=[ns["model_df"].datetime.iloc[20]],
                               end=[ns["model_df"].datetime.iloc[24]], duration_hours=[4.],
                               split=["train"], eligible=[True], exclusion=[""]))
    detector_calls = []

    def detected(*args, **kwargs):
        detector_calls.append(kwargs)
        return events.copy()

    monkeypatch.setattr(workflow, "frozen_events", detected)

    def forbidden(*args, **kwargs):
        raise AssertionError("Auto mode must not read EVENT_FILE")

    monkeypatch.setattr(workflow, "imported_event_inventory", forbidden)
    ns.update(EVENT_SOURCE="auto", EVENT_FILE="does-not-exist.xlsx")
    event_coverage_state(ns)
    assert detector_calls == [{"detector_settings": ns["study_settings"].workflow["events"]["STORM_DETECTION_SETTINGS"]}]
    assert detector_calls[0]["detector_settings"]["min_rain_in"] == 0.2
    assert ns["_event_file"] is None
    expected = storm_scoring_mask(events, ns["train_mask"], 1)
    np.testing.assert_array_equal(ns["rain_storm_mask"], expected)
    assert expected[17:72].all() and not expected[:17].any()
    expected |= melt_scoring_mask(ns["model_df"].possible_melt, ns["train_mask"], 1)
    np.testing.assert_array_equal(ns["storm_mask"], expected)
    assert ns["melt_mask"].any()


@pytest.mark.parametrize("mode,filename,message", [
    ("invalid", "ignored.xlsx", "EVENT_SOURCE"),
    ("file", "", "EVENT_FILE"),
    ("file", "missing.xlsx", "does not exist"),
])
def test_notebook_invalid_mode_or_file_has_no_auto_fallback(
    notebook_sources, tmp_path, mode, filename, message,
):
    ns, source = event_namespace(notebook_sources, tmp_path)
    ns.update(EVENT_SOURCE=mode, EVENT_FILE=filename)
    with pytest.raises((ValueError, FileNotFoundError), match=message):
        event_coverage_state(ns)


def test_notebook_no_training_events_reports_exclusions_and_fails(notebook_sources, tmp_path, capsys):
    ns, source = event_namespace(notebook_sources, tmp_path)
    path = event_csv(tmp_path / "events.csv", [["outside", "2026-01-01", "2026-01-02"]])
    ns.update(EVENT_SOURCE="file", EVENT_FILE=str(path))
    with pytest.raises(ValueError, match="No usable training event windows"):
        event_coverage_state(ns)
    assert "outside_model_record" in capsys.readouterr().out






def test_con15_nested_overlap_does_not_hide_out_of_domain_event():
    objective = Con15(np.array([[0, 60], [12, 24]]), allow_overlapping_events=True)
    with pytest.raises(ValueError, match="outside the observation domain"):
        objective.compute(np.ones(48), np.ones(48))
    with pytest.raises(ValueError, match="ordered"):
        Con15(np.array([[12, 24], [0, 20]]), allow_overlapping_events=True)


def test_notebook_report_accepts_workbook_style_overlap_in_all_splits(notebook_sources, tmp_path):
    path = summary_file(tmp_path / "approved.xlsx", [
        [54, "Qualifying Event", pd.Timestamp("2025-04-03 14:50"), 22.67],
        [55, "Qualifying Event", pd.Timestamp("2025-04-04 13:30"), 48.08],
        [80, "Qualifying Event", pd.Timestamp("2025-04-08 06:00"), 4.0],
        [81, "Qualifying Event", pd.Timestamp("2025-04-09 06:00"), 8.0],
    ])
    times = pd.date_range("2025-04-03", periods=10 * 24 * 12, freq="5min")
    observed = 2 + np.sin(np.arange(len(times)) * np.pi / (12 * 8))
    data = pd.DataFrame(dict(datetime=times, rainfall_in=0.0, flow_cfs=observed,
                              possible_melt=False, dry_eligible=True))
    ns, event_source = event_namespace(notebook_sources, tmp_path)
    ns.update(model_df=data, train_mask=np.asarray(times < "2025-04-07"),
              val_mask=np.asarray(times >= "2025-04-07"), scored_mask=np.ones(len(data), bool),
              dry_eligible=np.ones(len(data), bool), _dt_hours=1 / 12,
              EVENT_SOURCE="file", EVENT_FILE=str(path))
    event_coverage_state(ns)
    events = ns["acceptance_events"].copy(deep=True)
    assert events.event_id.tolist() == ["54", "55", "80", "81"]
    assert events.end_idx.iloc[0] == events.start_idx.iloc[1] + 1
    assert ns["events_df"].end_idx.iloc[0] == ns["events_df"].start_idx.iloc[1]
    assert events.overlap_previous_hours.iloc[1] == pytest.approx(12 / 3600)
    ns.update(Con15=Con15, ConcordanceCorrelationCoefficient=ConcordanceCorrelationCoefficient,
              MULTIOBJECTIVE_OBJECTIVES=["con15_error", "ccc"], USE_TRAINING_PEAK_FILTER=True)
    objective_state(ns)
    from sparsehydro.calibration.msdgc_assessment import MSDGCAssessment, MSDGCCon15
    assert isinstance(ns["peak_objective"], MSDGCCon15)
    assert isinstance(ns["peak_filter"], MSDGCCon15)
    definition = ns["peak_objective"]
    x = definition.event_values(observed)[definition.selected_event_indices(observed)]
    y = definition.event_values(observed * 1.05)[definition.selected_event_indices(observed)]
    expected = abs(float(x @ y / (x @ x)) - 1)
    assert definition.compute(observed, observed * 1.05) == expected
    assert ns["peak_filter"].compute(observed, observed * 1.05) == expected
    ns.update(obs_flow=observed, q_total=observed * 1.05)
    assessment_state(ns)
    pd.testing.assert_frame_equal(ns["acceptance_events"], events)
    for split, expected_ids in (("train", ["54", "55"]), ("validation", ["80", "81"]),
                                 ("whole", ["54", "55", "80", "81"])):
        evaluator = ns["report_evaluators"][split]
        assert evaluator.events.event_id.tolist() == expected_ids
        report = evaluator.event_report(ns["q_total"])
        expected = [observed[e.start_idx:e.end_idx + 1].sum() / 12
                    for e in evaluator.events.itertuples()]
        np.testing.assert_allclose(report.observed_volume_cfs_h, expected)
        assert set(ns["acceptance_tbl"].query("window == @split").metric) == set(
            MSDGCAssessment.report_metric_names)
    assert len(ns["events_eval"]) == 8
    assert len(ns["conveyance_tbl"]) == 4
    assert len(ns["zone_volumes_tbl"]) == 12
    assert ns["event_mask"].sum() < (
        events.end_idx - events.start_idx + 1).sum()
