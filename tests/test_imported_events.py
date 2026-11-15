

import numpy as np
import pandas as pd
import pytest

from sparsehydro.metrics.objectives import Con15
from sparsehydro.calibration.splits.masks import storm_scoring_mask
from sparsehydro.events.io import imported_event_inventory, load_event_windows


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


def test_con15_nested_overlap_does_not_hide_out_of_domain_event():
    objective = Con15(np.array([[0, 60], [12, 24]]), allow_overlapping_events=True)
    with pytest.raises(ValueError, match="outside the observation domain"):
        objective.compute(np.ones(48), np.ones(48))
    with pytest.raises(ValueError, match="ordered"):
        Con15(np.array([[12, 24], [0, 20]]), allow_overlapping_events=True)


