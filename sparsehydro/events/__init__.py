"""sparsehydro.events — Storm event detection and record structures.

Provides derivative-based segmentation of stormflow timeseries into discrete
storm events using Savitzky-Golay signals from :mod:`sparsehydro.filters`.

Public API
----------
- :class:`EventRecord` — dataclass for a single storm event
- :func:`detect_events` — automatic event detection from rain/stormflow data
- :func:`events_to_dataframe` — convert a list of EventRecord to a DataFrame
- :func:`load_events_from_csv` — load pre-defined events from a CSV file

Quick start::

    from sparsehydro.events import detect_events, load_events_from_csv

    events, filter_result = detect_events(rain_stormflow_df, verbose=True)
    # or
    events = load_events_from_csv("events_list.csv")
"""

from .bimodality import calculate_peak_zone_bimodality, calculate_rain_bimodality
from .detection import detect_events
from .hierarchy import detect_event_hierarchy
from .io import (
    imported_event_inventory,
    load_event_windows,
    load_events_from_csv,
    load_pcswmm_events,
    map_event_windows,
)
from .peaks import define_peaks
from .records import (
    EventRecord,
    GlobalEvent,
    SubEventRecord,
    events_to_dataframe,
    events_to_mask,
    global_events_to_dataframe,
    sub_events_to_dataframe,
)
from .screening import (
    build_global_events, flag_flow_response, merge_by_interevent, refine_event_durations,
    screen_flow_response, screen_min_duration, screen_small_rainfall, screening_summary,
    split_by_interevent, split_top_peak_events,
)
from .smoothing import (
    VariableSavgolResult,
    apply_peak_lift,
    savgol_curvature,
    savgol_slope,
    seed_curvature,
    variable_savgol_smooth,
)
from .zones import define_complete_event_zones

__all__ = [
    "EventRecord",
    "detect_events",
    "events_to_dataframe",
    "events_to_mask",
    "build_global_events",
    "flag_flow_response",
    "merge_by_interevent",
    "refine_event_durations",
    "screen_flow_response",
    "screen_min_duration",
    "screen_small_rainfall",
    "screening_summary",
    "split_by_interevent",
    "split_top_peak_events",
    "load_events_from_csv",
    "load_pcswmm_events",
    "load_event_windows",
    "imported_event_inventory",
    "map_event_windows",
    # Hierarchical detection
    "SubEventRecord",
    "GlobalEvent",
    "sub_events_to_dataframe",
    "global_events_to_dataframe",
    "VariableSavgolResult",
    "variable_savgol_smooth",
    "seed_curvature",
    "savgol_curvature",
    "savgol_slope",
    "apply_peak_lift",
    "define_peaks",
    "calculate_rain_bimodality",
    "calculate_peak_zone_bimodality",
    "define_complete_event_zones",
    "detect_event_hierarchy",
]
