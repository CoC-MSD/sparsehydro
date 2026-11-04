from tests.pattern_workflow_support import (
    configure_state, build_model_state, prepare_data_state, event_coverage_state,
    objective_state, calibrate_state, report_state, assessment_state,
)

import ast
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sparsehydro.calibration.objectives import Con15, ConcordanceCorrelationCoefficient, NashSutcliffe
from sparsehydro.calibration.splits import SplitConfig, build_masks, prepend_spinup
from sparsehydro.calibration.wholeflow_workflow import dry_weather_mask
from sparsehydro.events.imported import load_pcswmm_events, prepare_hybrid_events
from sparsehydro.models.rdii.initial_abstraction import snow_partition


@pytest.fixture
def sources():
    path = (Path(__file__).parents[1] / "sparsehydro" / "notebooks"
            / "flowfinity_gwi_wholeflow_pattern_joint.ipynb")
    sources = {c["id"]: "".join(c["source"]) for c in json.loads(
        path.read_text(encoding="utf-8"))["cells"]}
    sources["pattern-joint-00"] = "\n".join(
        source for cell_id, source in sources.items()
        if cell_id == "pattern-joint-00" or (
            cell_id.startswith("pattern-settings-") and cell_id != "pattern-settings-index"))
    return sources


def write_evt(path, rows):
    lines = ["; PCSWMM events file", "; Please do not edit."]
    epoch = pd.Timestamp("1899-12-30")
    for i, (name, start, end, group) in enumerate(rows):
        serials = [(pd.Timestamp(value) - epoch) / pd.Timedelta(days=1)
                   for value in (start, end)]
        lines += [f"[Event{i}]", f"Name ={name}", f"Start ={serials[0]:.10f}",
                  f"End ={serials[1]:.10f}", f"Group ={group}", ""]
    lines += ["[Group0]", "Name =Qualifying Event", "Color =123"]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def case(tmp_path):
    data = pd.DataFrame(dict(
        datetime=pd.date_range("2025-01-01", periods=30 * 24, freq="1h"),
        rainfall_in=0.0, flow_cfs=1.0, temperature_c=10.0, possible_melt=False,
        dry_eligible=True,
    ))
    all_file = write_evt(tmp_path / "all.evt", [
        ["bad", "2025-01-12", "2025-01-12 12:00", "Excluded;Data Issues"],
        ["small", "2025-01-04", "2025-01-04 04:00", "Excluded;Small Rainfall"],
        ["winter", "2025-01-07", "2025-01-07 04:00", "Excluded;Winter Conditions"],
        ["approved-train", "2025-01-10", "2025-01-10 06:00", "Qualifying Event"],
        ["no-response", "2025-01-14", "2025-01-14 08:00", "Excluded;Little or No WWF Reaction"],
        ["cross", "2025-01-20 23:00", "2025-01-21 02:00", "Excluded;Small Rainfall"],
        ["approved-val", "2025-01-23", "2025-01-23 06:00", "Qualifying Event"],
    ])
    approved_file = tmp_path / "approved.csv"
    pd.DataFrame([
        ["approved-train", "2025-01-10", "2025-01-10 06:00"],
        ["approved-val", "2025-01-23", "2025-01-23 06:00"],
    ], columns=["event_id", "start_date", "end_date"]).to_csv(approved_file, index=False)
    train = np.asarray(data.datetime < "2025-01-21")
    return data, train, ~train, all_file, approved_file


def test_evt_ole_dates_groups_precision_and_metadata(tmp_path):
    path = write_evt(tmp_path / "all.evt", [
        ["54", "2025-04-03 14:50", "2025-04-04 13:30", "Qualifying Event"],
        ["55", "2025-04-04 13:30", "2025-04-06 13:35", "Qualifying Event"],
    ])
    events = load_pcswmm_events(path)
    assert events.event_id.tolist() == ["54", "55"]
    assert events.source_section.tolist() == ["Event0", "Event1"]
    assert events.start.iloc[0] == pd.Timestamp("2025-04-03 14:50")
    assert events.end.iloc[0] == events.start.iloc[1]
    assert events.duration_hours.iloc[0] == pytest.approx(22 + 40 / 60)
    assert events.group.eq("Qualifying Event").all()


@pytest.mark.parametrize("text,message", [
    ("[Event0]\nName=1\nStart=abc\nEnd=45623\nGroup=Qualifying Event", "OLE Automation"),
    ("[Event0]\nName=1\nStart=nan\nEnd=45623\nGroup=Qualifying Event", "OLE Automation"),
    ("[Event0]\nName=1\nStart=45623\nEnd=45624", "required"),
    ("[Event0]\nName=1\nStart=45624\nEnd=45623\nGroup=Qualifying Event", "after start"),
    ("[Group0]\nName=metadata", "no PCSWMM event"),
    ("[Event0]\nName=1\nName=2", "invalid PCSWMM"),
])
def test_evt_rejects_malformed_records(tmp_path, text, message):
    path = tmp_path / "bad.evt"
    path.write_text(text)
    with pytest.raises(ValueError, match=message):
        load_pcswmm_events(path)


def test_hybrid_quality_and_approved_inventory_are_independent(tmp_path):
    data, train, val, all_file, approved_file = case(tmp_path)
    before = data.copy(deep=True)
    full, approved, quality = prepare_hybrid_events(data, train, val, all_file, approved_file)
    pd.testing.assert_frame_equal(data, before)
    assert len(full) == 7 and len(approved) == 2
    expected_bad = data.datetime.between("2025-01-12", "2025-01-12 12:00").to_numpy()
    np.testing.assert_array_equal(~quality, expected_bad)
    indexed = full.set_index("event_id")
    assert indexed.loc["bad", "exclusion"] == "data_issues"
    assert indexed.loc["cross", "exclusion"] == "crosses_split_boundary"
    assert indexed.loc[["small", "winter", "no-response"], "eligible"].all()
    assert approved.eligible.all()
    assert approved.event_id.tolist() == ["approved-train", "approved-val"]
    changed = data.copy()
    changed.loc[val | ~quality, "flow_cfs"] = 1e6
    other = prepare_hybrid_events(changed, train, val, all_file, approved_file)
    pd.testing.assert_frame_equal(full, other[0])
    pd.testing.assert_frame_equal(approved, other[1])
    np.testing.assert_array_equal(quality, other[2])


def test_hybrid_rejects_unmatched_approved_ids_and_unknown_groups(tmp_path):
    data, train, val, all_file, approved_file = case(tmp_path)
    approved_file.write_text("event_id,start_date,end_date\nwrong,2025-01-01,2025-01-02\n")
    with pytest.raises(ValueError, match="Approved IDs must match"):
        prepare_hybrid_events(data, train, val, all_file, approved_file)
    text = all_file.read_text().replace("Excluded;Small Rainfall", "Unreviewed Group")
    all_file.write_text(text)
    with pytest.raises(ValueError, match="Unrecognized EVT groups"):
        prepare_hybrid_events(data, train, val, all_file, approved_file)


def test_hybrid_overlap_with_bad_interval_excludes_whole_approved_event(tmp_path):
    data, train, val, _, approved_file = case(tmp_path)
    path = write_evt(tmp_path / "overlap.evt", [
        ["bad", "2025-01-10 02:00", "2025-01-10 03:00", "Excluded;Data Issues"],
        ["approved-train", "2025-01-10", "2025-01-10 06:00", "Qualifying Event"],
        ["approved-val", "2025-01-23", "2025-01-23 06:00", "Qualifying Event"],
    ])
    full, approved, _ = prepare_hybrid_events(data, train, val, path, approved_file)
    assert approved.exclusion.tolist() == ["overlaps_data_issues", ""]
    assert not approved.eligible.iloc[0]
    assert not full.loc[full.event_id.eq("approved-train"), "eligible"].any()


def notebook_case(sources, tmp_path):
    data, _, _, all_file, approved_file = case(tmp_path)
    data.loc[270, "possible_melt"] = True
    ns = dict(np=np, pd=pd, Path=Path, time=time, phase_seconds={}, display=lambda _value: None,
              dry_weather_mask=dry_weather_mask, prepend_spinup=prepend_spinup,
              build_masks=build_masks, snow_partition=snow_partition, model_df=data.copy())
    exec(sources["pattern-joint-00"], ns)
    ns.update(
        EVENT_SOURCE="hybrid", EVENT_FILE=approved_file.name,
        APPROVED_EVENTS_FILE=all_file.name, DOWNLOAD_DIR=tmp_path,
        SNOW_ENABLED=False, CALIBRATE_SNOW=False, DT_HOURS=1.0,
        SPLIT_CFG=SplitConfig(dt_hours=1, warmup_days=0, spinup_mode="none",
                              split_mode="continuous", train_frac=2 / 3,
                              snap_edges_to_dry=False),
    )
    prepare_data_state(ns)
    ns["dry_eligible"] = ns["model_df"].dry_eligible.to_numpy()
    # A melt pulse inside a flagged interval must not reintroduce bad observations.
    ns["model_df"].loc[270, "possible_melt"] = True
    event_coverage_state(ns)
    return ns, data


def test_notebook_hybrid_restores_broad_windows_but_not_bad_observations(sources, tmp_path):
    ns, original = notebook_case(sources, tmp_path)
    assert ns["EVENT_SOURCE"] == "hybrid"
    assert ns["_all_event_file"].name == "all.evt"
    assert ns["_event_file"].name == "approved.csv"
    bad = ~ns["observation_eligible"]
    assert bad.sum() == 13
    assert not ns["train_mask"][bad].any()
    assert not ns["val_mask"][bad].any()
    assert not ns["scored_mask"][bad].any()
    assert not ns["model_df"].dry_eligible[bad].any()
    for name in ("storm_mask", "val_storm_mask", "melt_mask", "val_melt_mask",
                 "rain_storm_mask", "approved_storm_mask"):
        assert not ns[name][bad].any(), name
    assert ns["storm_mask"].sum() > ns["approved_storm_mask"].sum()
    assert ns["storm_mask"][3 * 24]  # Small-rainfall event retained.
    assert not ns["approved_storm_mask"][3 * 24]
    assert ns["storm_mask"][3 * 24 - 3]  # Lead-in.
    assert ns["storm_mask"][3 * 24 + 4 + 48]  # Inclusive tail endpoint.
    assert not np.any(ns["storm_mask"] & ns["val_mask"])
    assert not np.any(ns["val_storm_mask"] & ns["train_mask"])
    assert [event.event_id for event in ns["_events"]] == ["approved-train", "approved-val"]
    for col in ("rainfall_in", "flow_cfs", "temperature_c", "datetime"):
        pd.testing.assert_series_equal(ns["model_df"][col], original[col])


def test_notebook_hybrid_peak_setup_uses_only_approved_training_events(sources, tmp_path):
    from sparsehydro.calibration.msdgc_assessment import MSDGCCon15, MSDGCSettings

    ns, _ = notebook_case(sources, tmp_path)
    ns.update(USE_TRAINING_PEAK_FILTER=True,
              msdgc_settings=MSDGCSettings(minimum_minutes=(60., 20., 30.),
                                           maximum_minutes=(60., 240., 1440.)))
    ns.update(Con15=Con15, ConcordanceCorrelationCoefficient=ConcordanceCorrelationCoefficient,
              NashSutcliffe=NashSutcliffe, MULTIOBJECTIVE_OBJECTIVES=["ccc", "nash_sutcliffe"])
    objective_state(ns)
    assert ns["USE_TRAINING_PEAK_FILTER"] is True
    assert ns["peak_objective"] is None
    assert isinstance(ns["peak_filter"], MSDGCCon15)
    assert ns["training_peak_events"].event_id.tolist() == ["approved-train"]
    expected = ns["acceptance_events"].query("split == 'train'")[["start_idx", "end_idx"]]
    np.testing.assert_array_equal(ns["peak_filter"].event_windows, expected.to_numpy())
    assert ns["peak_filter"].top_fraction == ns["CON15_FILTER_FRACTION"]
    assert ns["CON15_NEAR_TIE_ALLOWANCE"] == .02
    before = ns["peak_objective_events"].copy()
    ns["model_df"].loc[ns["val_mask"], "flow_cfs"] = 1e8
    objective_state(ns)
    pd.testing.assert_frame_equal(before, ns["peak_objective_events"])
    for objective, mask in zip(ns["calibration_objectives"], [ns["storm_mask"], ns["train_mask"]]):
        np.testing.assert_array_equal(objective.mask, mask)


def test_notebook_87_excluded_from_all_assessment_controls_not_broad_fit(sources, tmp_path):
    from sparsehydro.calibration.msdgc_assessment import MSDGCAssessment, MSDGCSettings
    from sparsehydro.calibration.wholeflow_acceptance import AcceptanceContract

    ns, original = notebook_case(sources, tmp_path)
    full = load_pcswmm_events(ns["_all_event_file"])
    rows = [[("87" if e.event_id == "approved-train" else e.event_id),
             e.start, e.end, e.group] for e in full.itertuples()]
    rows.append(["88", "2025-01-16", "2025-01-16 06:00", "Qualifying Event"])
    write_evt(ns["_all_event_file"], rows)
    approved = pd.read_csv(ns["_event_file"])
    approved["event_id"] = approved.event_id.replace({"approved-train": "87"})
    approved.loc[len(approved)] = ["88", "2025-01-16", "2025-01-16 06:00"]
    approved.to_csv(ns["_event_file"], index=False)
    ns["model_df"] = original.copy()
    prepare_data_state(ns)
    ns["dry_eligible"] = ns["model_df"].dry_eligible.to_numpy()
    ns["model_df"].loc[270, "possible_melt"] = True
    ns["ASSESSMENT_EXCLUDED_EVENT_IDS"] = ()
    event_coverage_state(ns)
    original_masks = {name: ns[name].copy() for name in (
        "train_mask", "val_mask", "storm_mask", "val_storm_mask", "observation_eligible")}
    ns["ASSESSMENT_EXCLUDED_EVENT_IDS"] = ("87",)
    event_coverage_state(ns)
    for name, mask in original_masks.items():
        np.testing.assert_array_equal(ns[name], mask, err_msg=name)
    assert ns["events_df"].query("event_id == '87'").eligible.all()
    assert ns["calibration_events_df"].query("event_id == '87'").eligible.all()
    excluded = ns["assessment_events_df"].query("event_id == '87'").iloc[0]
    assert not excluded.eligible and excluded.exclusion == "assessment_criteria_exclusion"
    assert "87" not in set(ns["acceptance_events"].event_id)
    assert ns["storm_mask"][excluded.start_idx:excluded.end_idx + 1].all()
    assert not ns["assessment_storm_mask"][excluded.start_idx:excluded.end_idx + 1].any()

    ns.update(
        msdgc_settings=MSDGCSettings(minimum_minutes=(60., 20., 30.),
                                     maximum_minutes=(60., 240., 1440.)),
        USE_TRAINING_PEAK_FILTER=True,
        MULTIOBJECTIVE_OBJECTIVES=["ccc", "nash_sutcliffe", "con15_error"],
        ConcordanceCorrelationCoefficient=ConcordanceCorrelationCoefficient,
        NashSutcliffe=NashSutcliffe,
    )
    objective_state(ns)
    assert ns["peak_objective_events"].event_id.tolist() == ["88"]
    windows = ns["training_peak_events"][["start_idx", "end_idx"]].to_numpy(dtype=int)
    for definition in (ns["peak_objective"], ns["peak_filter"]):
        np.testing.assert_array_equal(definition.event_windows, windows)
        assert definition.compute(ns["model_df"].flow_cfs.to_numpy(),
                                  ns["model_df"].flow_cfs.to_numpy()) == 0
    observed = ns["model_df"].flow_cfs.to_numpy()
    report = MSDGCAssessment(
        observed, ns["acceptance_events"], ns["scored_mask"], ns["dry_eligible"],
        ns["_dt_hours"], AcceptanceContract(), settings=ns["msdgc_settings"],
        allow_overlapping_events=True,
    ).zone_report(observed)
    assert len(report) == 6 and "87" not in set(report.event_id)


def test_hybrid_shape_fit_cannot_use_bad_or_held_out_flow(sources, tmp_path):
    ns, _ = notebook_case(sources, tmp_path)
    ns["plot_sanitary_shapes"] = lambda pattern: SimpleNamespace(show=lambda: None)
    exec(sources["pattern-joint-27"], ns)
    original = ns["shape_fit"]
    mean = ns["odwf_cfs"]
    excluded = ~ns["observation_eligible"] | ns["val_mask"]
    ns["model_df"].loc[excluded, "flow_cfs"] = 1e6
    exec(sources["pattern-joint-27"], ns)
    assert ns["odwf_cfs"] == mean
    assert ns["shape_fit"].pattern == original.pattern
    np.testing.assert_array_equal(ns["shape_fit"].detection_baseline_cfs,
                                  original.detection_baseline_cfs)
    assert not np.any(ns["sanitary_fit_mask"] & excluded)
