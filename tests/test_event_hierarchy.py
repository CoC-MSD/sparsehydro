"""Tests for the global + sub-event detection hierarchy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sparsehydro.events import (
    GlobalEvent, SubEventRecord, VariableSavgolResult,
    detect_event_hierarchy, define_peaks, variable_savgol_smooth,
    define_complete_event_zones, calculate_rain_bimodality,
    calculate_peak_zone_bimodality, sub_events_to_dataframe, global_events_to_dataframe,
)


def test_hierarchy_return_types(muum019_hierarchy):
    globals_, subs, savgol = muum019_hierarchy
    assert isinstance(globals_, list) and all(isinstance(g, GlobalEvent) for g in globals_)
    assert isinstance(subs, list) and all(isinstance(s, SubEventRecord) for s in subs)
    assert isinstance(savgol, VariableSavgolResult)


def test_hierarchy_nonempty(muum019_hierarchy):
    globals_, subs, _ = muum019_hierarchy
    assert len(globals_) >= 1
    assert len(subs) >= 1


def test_sub_event_boundary_ordering(muum019_hierarchy):
    _, subs, _ = muum019_hierarchy
    for s in subs:
        assert s.start_idx <= s.peak_idx <= s.end_idx
        assert s.rise_start_idx <= s.peak_idx <= s.tail_start_idx
        assert s.start_datetime <= s.peak_datetime <= s.end_datetime


def test_sub_events_reference_globals(muum019_hierarchy):
    globals_, subs, _ = muum019_hierarchy
    gids = {g.global_id for g in globals_}
    for s in subs:
        assert s.global_id in gids
    # every global lists only real sub ids
    all_sub_ids = {s.sub_id for s in subs}
    for g in globals_:
        assert set(g.sub_ids).issubset(all_sub_ids)


def test_effective_area_consistency(muum019_hierarchy):
    _, subs, _ = muum019_hierarchy
    for s in subs:
        if s.total_rain > 0:
            assert abs(s.effective_area - s.total_flow / s.total_rain) < 1e-6


def test_savgol_windows_bounded(muum019_hierarchy):
    _, _, savgol = muum019_hierarchy
    assert len(savgol.smoothed) == len(savgol.raw_flow)
    assert savgol.windows.min() >= 5
    assert savgol.windows.max() <= 24


def test_dataframes(muum019_hierarchy):
    globals_, subs, _ = muum019_hierarchy
    sdf = sub_events_to_dataframe(subs)
    gdf = global_events_to_dataframe(globals_)
    assert len(sdf) == len(subs)
    assert len(gdf) == len(globals_)
    assert "effective_area" in sdf.columns
    assert "n_sub_events" in gdf.columns


def test_define_peaks(synthetic_rain_stormflow):
    df_sg, peaks, sections = define_peaks(synthetic_rain_stormflow)
    assert "sg" in df_sg.columns
    assert len(peaks) >= 1
    assert all(0 <= s <= e < len(df_sg) for s, e in sections)


def test_variable_savgol_smooth(synthetic_rain_stormflow):
    flow = synthetic_rain_stormflow["stormflow"].to_numpy()
    smoothed, windows = variable_savgol_smooth(flow, [500], return_windows=True)
    assert len(smoothed) == len(flow)
    assert windows.min() >= 5 and windows.max() <= 24


def test_bimodality_two_bursts():
    rain = np.zeros(400)
    rain[20:30] = 0.1
    rain[200:210] = 0.1
    res = calculate_rain_bimodality(rain, 0, 399)
    assert res["rain_burst_count"] == 2
    assert res["rain_bimodality_index"] > 0.0


def test_peak_zone_bimodality_keys(synthetic_rain_stormflow):
    flow = synthetic_rain_stormflow["stormflow"].to_numpy()
    res = calculate_peak_zone_bimodality(flow, flow, 0, int(np.argmax(flow)), len(flow) - 1)
    for k in ("bimodality_index", "is_bimodal", "secondary_peak_list", "rain_bimodality_index"):
        assert k in res


def test_zones_single_peak_per_segment(muum019_window):
    df_sg, peaks, sections = define_peaks(muum019_window)
    flow = np.maximum(muum019_window["stormflow"].to_numpy(), 0.0)
    smoothed = variable_savgol_smooth(flow, peaks, segments=sections)
    zi = pd.DataFrame({"datetime": muum019_window["datetime"], "rain": muum019_window["rain"], "stormflow": flow})
    subs = define_complete_event_zones(zi, smoothed, sections)
    assert all(isinstance(s, SubEventRecord) for s in subs)
    for s in subs:
        assert s.end_idx > s.start_idx
