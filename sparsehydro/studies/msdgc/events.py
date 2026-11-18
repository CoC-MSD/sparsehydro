"""MSDGC study events: frozen detection per split, hybrid EVT inventories, melt windows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sparsehydro.events.io import load_event_windows, load_pcswmm_events, map_event_windows
from sparsehydro.studies.msdgc.detector import find_events
from sparsehydro.models.forcing import SnowConfig, snow_partition


def frozen_events(
    data: pd.DataFrame, train: np.ndarray, val: np.ndarray, reference: np.ndarray,
    *, detector_settings: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Apply the existing detector independently within contiguous split blocks.

    Never let validation stormflow extend/change a training event. Boundary events
    and the detector's omitted final rain span in each block are recorded, not
    assigned by majority. All-flow Qcritical is independent of these exclusions.
    """
    settings = dict(threshold_pct=5.0, interevent_hours=6.0, sliding_hours=12.0,
                    min_rain_in=0.2, min_duration_hours=3.0)
    if detector_settings is not None:
        unknown = set(detector_settings) - set(settings)
        if unknown:
            raise ValueError(f"Unknown storm detection settings: {sorted(unknown)}")
        settings.update(detector_settings)
    if (not np.isfinite(list(settings.values())).all()
            or not 0 <= settings["threshold_pct"] <= 100
            or settings["interevent_hours"] <= 0 or settings["sliding_hours"] <= 0
            or settings["min_rain_in"] < 0 or settings["min_duration_hours"] < 0):
        raise ValueError("Storm detection settings must be finite; threshold must be in [0, 100], "
                         "gap/window positive, and rainfall/duration nonnegative")
    labels = np.where(train, "train", np.where(val, "validation", "excluded"))
    edges = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(data)]
    wet = np.flatnonzero(data["rainfall_in"].to_numpy() > 0)
    times = pd.DatetimeIndex(data["datetime"]).as_unit("ns").asi8
    breaks = np.flatnonzero(
        np.diff(times[wet]) > pd.Timedelta(hours=settings["interevent_hours"]).value) + 1
    rain_spans = [(int(g[0]), int(g[-1])) for g in np.split(wet, breaks) if len(g)]
    rows = []
    for start, stop in zip(edges[:-1], edges[1:]):
        if labels[start] == "excluded":
            continue
        part = data.iloc[start:stop].copy().reset_index(drop=True)
        part["stormflow"] = np.maximum(part["flow_cfs"].to_numpy() - reference[start:stop], 0)
        official = find_events(part, rain_col="rainfall_in", close_trailing_event=False, **settings)
        inventory = find_events(part, rain_col="rainfall_in", close_trailing_event=True, **settings)
        official_pairs = {(e.start_idx, e.end_idx) for e in official}

        def boundary_reason(event) -> str:
            a, b = start + event.start_idx, start + event.end_idx
            if event.start_idx == 0 or event.end_idx >= len(part) - 1:
                return "touches_split_boundary"
            if any(s <= b and e >= a and (s < start or e >= stop) for s, e in rain_spans):
                return "rain_span_crosses_split_boundary"
            return ""

        for event in inventory:
            reason = ""
            if (event.start_idx, event.end_idx) not in official_pairs:
                reason = "trailing_span_omitted_or_merged_by_detector"
            reason = boundary_reason(event) or reason
            rows.append({
                "event_id": f"{start + event.start_idx}:{start + event.end_idx}", "start_idx": start + event.start_idx,
                "end_idx": start + event.end_idx, "start": event.start_datetime,
                "end": event.end_datetime, "split": labels[start],
                "duration_hours": event.duration_hours,
                "rain_in": event.total_rain, "eligible": not reason, "exclusion": reason,
            })
        # Official events can be altered by closing the trailing span in inventory.
        seen = {(row["start_idx"], row["end_idx"]) for row in rows}
        for event in official:
            pair = (start + event.start_idx, start + event.end_idx)
            if pair not in seen:
                reason = boundary_reason(event)
                rows.append({
                    "event_id": f"{pair[0]}:{pair[1]}", "start_idx": pair[0], "end_idx": pair[1],
                    "start": event.start_datetime, "end": event.end_datetime,
                    "duration_hours": event.duration_hours,
                    "split": labels[start], "rain_in": event.total_rain,
                    "eligible": not reason, "exclusion": reason,
                })
    columns = ["event_id", "start_idx", "end_idx", "start", "end", "split",
               "duration_hours", "rain_in", "eligible", "exclusion"]
    return pd.DataFrame(rows, columns=columns).sort_values(["start_idx", "end_idx"]).reset_index(drop=True)


def snow_melt_envelope(
    precipitation_in: np.ndarray, temperature_c: np.ndarray, dt_hours: float, *,
    threshold_bounds: tuple[float, float], melt_factor_bounds: tuple[float, float],
    initial_swe_in: float = 0.0,
) -> np.ndarray:
    """Conservative possible-melt timesteps for every bounded snow candidate.

    The highest threshold and lowest melt factor give an upper SWE trajectory.
    Any candidate can melt only where that pack existed before the step and the
    temperature exceeds the lowest threshold. This is a wetness/scoring mask,
    NOT a water input. A zero lower melt bound can retain potential SWE forever.
    """
    bounds = np.asarray([threshold_bounds, melt_factor_bounds], dtype=float)
    if (bounds.shape != (2, 2) or not np.isfinite(bounds).all()
            or np.any(bounds[:, 0] > bounds[:, 1]) or bounds[1, 0] < 0):
        raise ValueError("Snow envelope bounds must be finite, ordered, with nonnegative melt")
    upper = snow_partition(
        precipitation_in, temperature_c, dt_hours,
        SnowConfig(threshold_c=float(bounds[0, 1]),
                   melt_factor_in_per_c_day=float(bounds[1, 0]),
                   initial_swe_in=initial_swe_in),
    )
    previous_swe = np.r_[initial_swe_in, upper["SWE_in"].to_numpy()[:-1]]
    return ((previous_swe > 0) & (np.asarray(temperature_c) > bounds[0, 0])
            & (bounds[1, 1] > 0))


def assessment_event_inventory(
    times: pd.DatetimeIndex, approved: pd.DataFrame,
    train: np.ndarray, validation: np.ndarray, *,
    excluded_event_ids: tuple[str, ...] = ("87",),
    full_inventory: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Preserve source approval and audit exclusions without changing fitting masks.

    Hybrid assessment uses full EVT timestamps for approved IDs. Like MSDGC's
    lookup, the scan includes the first native sample on or after the end.
    Whole-event split/quality checks include that sample; no clipping or leakage.
    """
    times = pd.DatetimeIndex(times)
    train, validation = np.asarray(train, bool), np.asarray(validation, bool)
    if (len(times) < 2 or times.hasnans or times.has_duplicates
            or not times.is_monotonic_increasing or times.tz is not None
            or train.shape != (len(times),) or validation.shape != train.shape
            or (train & validation).any()):
        raise ValueError("Assessment requires ordered native timestamps and aligned disjoint split masks")
    source = approved.copy()
    if source.event_id.astype(str).duplicated().any():
        raise ValueError("Assessment event IDs must be unique")
    source["source_start"], source["source_end"] = source["start"], source["end"]
    source["assessment_window_source"] = "approved"
    if full_inventory is not None:
        full = full_inventory.copy()
        full.index = full.event_id.astype(str)
        if full.index.has_duplicates or not source.event_id.astype(str).isin(full.index).all():
            raise ValueError("Full EVT inventory must contain each approved event ID exactly once")
        for column in ("start", "end"):
            source[column] = source.event_id.astype(str).map(full[column])
        source["assessment_window_source"] = "full_evt"
    excluded = set(map(str, excluded_event_ids))
    for index, event in source.iterrows():
        start = int(times.searchsorted(event["start"], side="left"))
        end = int(times.searchsorted(event["end"], side="left"))
        reason = str(event["exclusion"]) if not event["eligible"] else ""
        if str(event["event_id"]) in excluded:
            reason = ";".join(filter(None, [reason, "assessment_criteria_exclusion"]))
        split = "excluded"
        if event["start"] < times[0] or event["end"] > times[-1]:
            reason = reason or "outside_model_record"
        elif end - start + 1 < 2:
            reason = reason or "insufficient_model_samples"
        elif train[start:end + 1].all():
            split = "train"
        elif validation[start:end + 1].all():
            split = "validation"
        else:
            reason = reason or "assessment_endpoint_crosses_split_or_unscored"
        source.loc[index, ["start_idx", "end_idx", "split", "eligible", "exclusion"]] = [
            start, end, split, not reason, reason,
        ]
    source["duration_hours"] = (
        (pd.to_datetime(source["end"]) - pd.to_datetime(source["start"])).dt.total_seconds() / 3600
    ).round(2)
    return source


def prepare_hybrid_events(
    data: pd.DataFrame, train: np.ndarray, validation: np.ndarray,
    all_event_file: str | Path, approved_event_file: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Separate reliable calibration coverage from approved reporting windows.

    Exact Data Issues intervals are excluded from every observation-based fit,
    including padding/melt/NSE and dry-weather fitting. Forcing is untouched.
    Return full/approved inventories and a fixed observation-quality mask.
    """
    full = load_pcswmm_events(all_event_file)
    approved = load_event_windows(approved_event_file)
    groups = full["group"].str.casefold()
    known = {
        "qualifying event", "excluded;data issues", "excluded;small rainfall",
        "excluded;winter conditions", "excluded;little or no wwf reaction",
    }
    unknown = set(groups) - known
    if unknown:
        raise ValueError(f"Unrecognized EVT groups; review calibration eligibility: {sorted(unknown)}")
    qualifying = full.loc[groups.eq("qualifying event")]
    if set(approved["event_id"]) != set(qualifying["event_id"]):
        raise ValueError(
            "Approved IDs must match EVT Qualifying Event IDs; "
            f"missing from approved={sorted(set(qualifying.event_id) - set(approved.event_id))}, "
            f"not qualifying in EVT={sorted(set(approved.event_id) - set(qualifying.event_id))}"
        )
    times = pd.DatetimeIndex(data["datetime"])
    quality = np.ones(len(data), bool)
    issues = full.loc[groups.eq("excluded;data issues")]
    for event in issues.itertuples():
        a = int(times.searchsorted(event.start, side="left"))
        b = int(times.searchsorted(event.end, side="right"))
        quality[a:b] = False
    train, validation = np.asarray(train, bool), np.asarray(validation, bool)
    calibration = map_event_windows(data, train & quality, validation & quality, full)
    reports = map_event_windows(data, train & quality, validation & quality, approved)
    for inventory in (calibration, reports):
        for i, event in inventory.iterrows():
            a, b = max(0, int(event["start_idx"])), min(len(data), int(event["end_idx"]) + 1)
            if not quality[a:b].all():
                inventory.loc[i, ["eligible", "exclusion"]] = [False, "overlaps_data_issues"]
    flagged = calibration["group"].str.casefold().eq("excluded;data issues")
    calibration.loc[flagged, ["eligible", "exclusion"]] = [False, "data_issues"]
    return calibration, reports, quality
