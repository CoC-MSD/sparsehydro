"""Approved event windows, independent of automatic rain/flow detection."""

from __future__ import annotations

import configparser
from pathlib import Path
import re

import numpy as np
import pandas as pd


def _column_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def load_pcswmm_events(path: str | Path) -> pd.DataFrame:
    """Read PCSWMM EVT sections with OLE Automation dates and group labels.

    Serial-day floating-point timestamps are rounded to the nearest second,
    not to the model grid. Group metadata sections are not event records.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Event file does not exist: {path}")
    parser = configparser.ConfigParser(interpolation=None)
    try:
        with path.open(encoding="utf-8-sig") as stream:
            parser.read_file(stream)
    except configparser.Error as exc:
        raise ValueError(f"{path}: invalid PCSWMM event file: {exc}") from exc
    rows = []
    for section in parser.sections():
        if not re.fullmatch(r"Event\d+", section):
            continue
        record = parser[section]
        if any(not record.get(key, "").strip() for key in ("Name", "Start", "End", "Group")):
            raise ValueError(f"{path}, {section}: Name, Start, End and Group are required")
        try:
            serials = [float(record[key]) for key in ("Start", "End")]
            if not np.isfinite(serials).all():
                raise ValueError("nonfinite serial date")
            start, end = pd.to_datetime(serials, unit="D", origin="1899-12-30").round("s")
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"{path}, {section}: invalid OLE Automation dates") from exc
        if end <= start:
            raise ValueError(f"{path}, {section}: end must be after start")
        rows.append(dict(
            event_id=record["Name"].strip(), start=start, end=end,
            duration_hours=(end - start).total_seconds() / 3600,
            group=record["Group"].strip(), source_section=section,
        ))
    if not rows:
        raise ValueError(f"{path}: no PCSWMM event sections")
    return _validate_event_windows(pd.DataFrame(rows), path)


def load_event_windows(path: str | Path) -> pd.DataFrame:
    """Read approved CSV/XLSX windows without screening, merging or padding.

    Accept ``event_id, start_date, end_date`` or an event-summary spreadsheet
    with ``Event, Date, Duration (h)`` (title/blank rows before its header are OK).
    ``Date`` is the start; duration is elapsed hours, not a number of samples.
    All input rows must already be approved events. Times must use the same
    timezone-naive clock as the model. The first Excel worksheet is used.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Qualifying-event file does not exist: {path}")
    if path.suffix.lower() == ".evt":
        return load_pcswmm_events(path)
    if path.suffix.lower() == ".xlsx":
        try:
            raw = pd.read_excel(path, header=None, engine="openpyxl")
        except ImportError as exc:
            raise ImportError(
                'Reading XLSX events requires openpyxl; install sparsehydro[excel] '
                "in the notebook's Python environment."
            ) from exc
    elif path.suffix.lower() == ".csv":
        raw = pd.read_csv(path, header=None)
    else:
        raise ValueError(f"Qualifying events must be a .csv or .xlsx file, or PCSWMM .evt: {path}")

    aliases = {
        "eventid": "event_id", "event": "event_id",
        "start": "start", "startdate": "start", "startdatetime": "start", "date": "start",
        "end": "end", "enddate": "end", "enddatetime": "end",
        "durationh": "duration_hours", "durationhours": "duration_hours",
    }
    header = None
    columns = []
    for i, row in raw.head(25).iterrows():
        names = [aliases.get(_column_name(v), _column_name(v)) for v in row]
        if {"event_id", "start"} <= set(names) and ({"end", "duration_hours"} & set(names)):
            header, columns = i, names
            break
    if header is None:
        raise ValueError(
            f"{path}: expected event_id/start_date/end_date or Event/Date/Duration (h) headers"
        )
    required = {"event_id", "start", "end", "duration_hours"}
    if any(columns.count(name) > 1 for name in required):
        raise ValueError(f"{path}: ambiguous duplicate event columns")
    frame = raw.iloc[header + 1:].dropna(how="all").copy()
    frame.columns = columns
    if frame.empty:
        raise ValueError(f"{path}: no approved event rows")

    rows = []
    for index, row in frame.iterrows():
        location = f"{path}, row {index + 1}"
        if pd.isna(row["event_id"]) or not str(row["event_id"]).strip():
            raise ValueError(f"{location}: event ID is required")
        times = {}
        for field in ("start", "end") if "end" in frame else ("start",):
            value = row[field]
            if pd.isna(value) or isinstance(value, (int, float, np.number)):
                raise ValueError(f"{location}: {field} must be a date/time, not a blank or number")
            try:
                stamp = pd.Timestamp(value)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{location}: invalid {field} timestamp {value!r}") from exc
            if pd.isna(stamp) or stamp.tzinfo is not None:
                raise ValueError(f"{location}: {field} must be a timezone-naive timestamp")
            times[field] = stamp
        if "end" in times:
            duration = (times["end"] - times["start"]).total_seconds() / 3600
        else:
            try:
                duration = float(row["duration_hours"])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{location}: duration must be numeric hours") from exc
            if not np.isfinite(duration) or duration <= 0:
                raise ValueError(f"{location}: duration must be finite and positive")
            times["end"] = times["start"] + pd.Timedelta(hours=duration)
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError(f"{location}: end must be after start")
        rows.append(dict(event_id=str(row["event_id"]).strip(), **times,
                         duration_hours=duration, source_row=index + 1))
    return _validate_event_windows(pd.DataFrame(rows), path)


def _validate_event_windows(events: pd.DataFrame, path: Path) -> pd.DataFrame:
    events = events.sort_values(["start", "end"]).reset_index(drop=True)
    if events["event_id"].duplicated().any():
        raise ValueError(f"{path}: duplicate event IDs")
    events["overlap_previous_hours"] = (
        (events["end"].cummax().shift() - events["start"]).dt.total_seconds() / 3600
    ).clip(lower=0).fillna(0.0)
    return events


def imported_event_inventory(
    data: pd.DataFrame, train: np.ndarray, validation: np.ndarray, path: str | Path,
) -> pd.DataFrame:
    """Map exact approved windows to inclusive model indices and whole splits.

    Only samples inside each supplied interval are used. Never expand, merge,
    redetect, truncate into eligibility, or assign crossing events by majority.
    Every source row remains in the inventory with an explicit exclusion reason
    when outside the record, unscored, too short, or crossing train/validation.
    """
    return _map_event_windows(data, train, validation, load_event_windows(path))


def _map_event_windows(
    data: pd.DataFrame, train: np.ndarray, validation: np.ndarray, events: pd.DataFrame,
) -> pd.DataFrame:
    times = pd.DatetimeIndex(data["datetime"])
    if (len(times) < 2 or times.hasnans or times.has_duplicates
            or not times.is_monotonic_increasing or times.tz is not None):
        raise ValueError("Model timestamps must be ordered, unique, timezone-naive and nonempty")
    train, validation = np.asarray(train, bool), np.asarray(validation, bool)
    if (train.shape != (len(data),) or validation.shape != train.shape
            or (train & validation).any()):
        raise ValueError("Split masks must align and cannot overlap")
    rows = []
    for event in events.to_dict("records"):
        start = int(times.searchsorted(event["start"], side="left"))
        end = int(times.searchsorted(event["end"], side="right")) - 1
        split, reason = "excluded", ""
        if event["start"] < times[0] or event["end"] > times[-1]:
            reason = "outside_model_record"
        elif end - start + 1 < 2:
            reason = "insufficient_model_samples"
        elif train[start:end + 1].all():
            split = "train"
        elif validation[start:end + 1].all():
            split = "validation"
        elif (train | validation)[start:end + 1].all():
            split, reason = "mixed", "crosses_split_boundary"
        else:
            reason = "warmup_or_unscored"
        rain = data["rainfall_in"].iloc[start:end + 1]
        rows.append(dict(
            **event, start_idx=start, end_idx=end, split=split,
            rain_in=float(rain.sum(min_count=1)), eligible=not reason, exclusion=reason,
        ))
    return pd.DataFrame(rows)


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
    calibration = _map_event_windows(data, train & quality, validation & quality, full)
    reports = _map_event_windows(data, train & quality, validation & quality, approved)
    for inventory in (calibration, reports):
        for i, event in inventory.iterrows():
            a, b = max(0, int(event["start_idx"])), min(len(data), int(event["end_idx"]) + 1)
            if not quality[a:b].all():
                inventory.loc[i, ["eligible", "exclusion"]] = [False, "overlaps_data_issues"]
    flagged = calibration["group"].str.casefold().eq("excluded;data issues")
    calibration.loc[flagged, ["eligible", "exclusion"]] = [False, "data_issues"]
    return calibration, reports, quality
