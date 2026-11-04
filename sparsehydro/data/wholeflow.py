"""Strict local-input preparation used by whole-flow study clients."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import importlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from .services import DownloadResult, FlowFinityClient
from .utilities import read_tsf


MGD_TO_CFS = 1e6 / (7.480519 * 86400.0)
TIMESTEPS = {"5min": ("5min", 5.0 / 60.0), "15min": ("15min", .25),
             "60min": ("1h", 1.0), "1h": ("1h", 1.0)}


@dataclass
class FlowFinityInputs:
    """Downloaded inputs and discovery results, never authentication secrets."""

    title: str
    directory: Path
    records: pd.DataFrame
    results: dict[str, DownloadResult]


def select_flowfinity_record(
    records: pd.DataFrame, *, title: str | None, index: int,
) -> str:
    """Select an exact discovered title, or validate the zero-based record index."""
    if title is not None:
        if title not in records["DataTitle"].values:
            raise ValueError("FLOWFINITY_RECORD_TITLE is not in the discovered records")
        return title
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(records):
        raise ValueError("FLOWFINITY_RECORD_INDEX must identify a discovered record")
    return records["DataTitle"].iloc[index]


def download_flowfinity_inputs(
    download_root: Path, *, auth_dir: Path, record_title: str | None, record_index: int,
    name_filter: str | None, types: Sequence[str], start: str | None = None, end: str | None = None,
) -> FlowFinityInputs:
    """Reload local credentials, discover records, and download the selected study.

    An absent authentication module explicitly falls back to the client's
    environment-variable authentication. Missing imports inside that module
    propagate rather than being misreported as missing credentials.
    """
    if str(auth_dir) not in sys.path:
        sys.path.insert(0, str(auth_dir))
    username, password = None, None
    try:
        auth = importlib.reload(importlib.import_module("flowfinity_auth"))
    except ModuleNotFoundError as error:
        if error.name != "flowfinity_auth":
            raise
        print("flowfinity_auth.py not found - falling back to environment variables.")
    else:
        username = getattr(auth, "FLOWFINITY_USERNAME", None)
        password = getattr(auth, "FLOWFINITY_PASSWORD", None)
        print("Loaded credentials from flowfinity_auth.py")
    client = FlowFinityClient(username=username, password=password)
    print("Client ready ->", client.base_url)
    resources = client.list_resources(name_filter)
    records = pd.DataFrame(
        [{"DataTitle": record.get("DataTitle")} for record in resources]
    ).sort_values("DataTitle", ignore_index=True)
    print(f"{len(records)} record(s) found" + (f" matching {name_filter!r}" if name_filter else ""))
    title = select_flowfinity_record(records, title=record_title, index=record_index)
    print(f"Selected record: {title!r}")
    print("Available data types:", client.available_types(title))
    directory = Path(download_root) / title.replace(" ", "_")
    results = client.download(title, types=list(types), output_dir=directory, parse=True, start=start, end=end)
    describe_results(results)
    return FlowFinityInputs(title, directory, records, results)


def modeling_timestep(timestep: str) -> tuple[str, float]:
    """Return the resampling frequency and hours per model step."""
    if timestep not in TIMESTEPS:
        raise ValueError(f"TIMESTEP must be one of {sorted(TIMESTEPS)}, got {timestep!r}")
    return TIMESTEPS[timestep]


def read_local_results(
    directory: Path, files: Mapping[str, tuple[str | None, str | int]],
    *, start: str | None = None, end: str | None = None,
) -> dict[str, DownloadResult]:
    """Read one selected value column per TSF/CSV, retaining native timestamps."""
    for required in ("Rain", "Flow"):
        if required not in files or not files[required][0]:
            raise ValueError(f"{required} file must be configured")
    results = {}
    for data_type, (file, column) in files.items():
        if file is None or file == "":
            print(f"{data_type:10s}: no file configured; skipped")
            continue
        path = Path(directory) / file
        if not path.exists():
            raise FileNotFoundError(f"{data_type} file not found: {path}")
        if path.suffix.lower() == ".tsf":
            data = read_tsf(path, start=start, end=end)
        else:
            data = pd.read_csv(path, index_col=0, parse_dates=True).sort_index().loc[start:end]
        data = data[[column]] if isinstance(column, str) else data.iloc[:, [column]]
        results[data_type] = DownloadResult(data_type=data_type, path=path, dataframe=data)
    describe_results(results)
    return results


def describe_results(results: Mapping[str, DownloadResult]) -> None:
    """Print the input period and columns without displaying observations."""
    for data_type, result in results.items():
        data = result.dataframe
        if data is not None and len(data):
            columns = ", ".join(map(str, data.columns))
            print(f"{data_type:10s}: {len(data):>6d} rows ({data.index.min()} -> "
                  f"{data.index.max()}) [{columns}] -> {result.path}")
        else:
            print(f"{data_type:10s}: no parsed data -> {result.path}")


def assemble_modeling_data(results: Mapping[str, DownloadResult], timestep: str) -> pd.DataFrame:
    """Average flow and sum rainfall only over complete common native bins."""
    frequency, _ = modeling_timestep(timestep)
    resampled = {}
    for data_type, name in (("Flow", "flow_mgd"), ("Rain", "rainfall_in")):
        frame = results[data_type].dataframe
        if frame is None or frame.empty:
            raise ValueError(f"{data_type}: nonempty parsed input is required")
        series = frame.iloc[:, 0]
        if series.index.has_duplicates or series.index.hasnans or not series.index.is_monotonic_increasing:
            raise ValueError(f"{name}: timestamps must be unique and ordered")
        cadence = series.index.to_series().diff().dropna()
        if cadence.empty or cadence.nunique() != 1:
            raise ValueError(f"{name}: irregular or missing native timestamps")
        native = cadence.iloc[0]
        if pd.Timedelta(frequency) < native or pd.Timedelta(frequency) % native:
            raise ValueError(f"TIMESTEP must be a multiple of the native {name} cadence: {native}")
        if not np.isfinite(series).all() or (series < 0).any():
            raise ValueError(f"{name}: missing or negative input; no silent filling")
        expected = int(pd.Timedelta(frequency) / native)
        groups = series.resample(frequency)
        values = groups.mean() if name == "flow_mgd" else groups.sum(min_count=expected)
        resampled[name] = values.where(groups.count() == expected)
    if any(series.first_valid_index() is None for series in resampled.values()):
        raise ValueError("No complete flow/rainfall resampling bins")
    start = max(series.first_valid_index() for series in resampled.values())
    end = min(series.last_valid_index() for series in resampled.values())
    data = pd.DataFrame(resampled).loc[start:end].copy()
    if data.empty or data.isna().any().any():
        raise ValueError("Incomplete flow/rainfall coverage within the common modeling period")
    data["flow_cfs"] = data["flow_mgd"] * MGD_TO_CFS
    return data.rename_axis("datetime").reset_index()


def align_air_temperature(
    data: pd.DataFrame, candidates: Sequence[Path], max_gap_hours: float,
) -> tuple[pd.DataFrame, Path]:
    """Interpolate UTC temperature to fixed EST without filling large gaps/edges."""
    path = next((Path(p) for p in candidates if Path(p).is_file()), None)
    if path is None:
        raise FileNotFoundError(f"No air_temperature.csv found in {list(candidates)}")
    air = pd.read_csv(path, comment="#")
    time_column = next(c for c in air if "date" in c.lower() or "time" in c.lower())
    value_column = next(c for c in air if "temp" in c.lower())
    times = pd.to_datetime(air[time_column], utc=True, errors="raise")
    values = pd.to_numeric(air[value_column], errors="raise")
    temperature = pd.Series(
        values.to_numpy(), index=times.dt.tz_convert("Etc/GMT+5").dt.tz_localize(None)).sort_index()
    if (temperature.groupby(level=0).nunique(dropna=False) > 1).any():
        raise ValueError("Conflicting duplicate temperature timestamps")
    print(f"Temperature source {path}; missing source rows removed: {temperature.isna().sum()}")
    temperature = temperature[~temperature.index.duplicated(keep="first")].dropna()
    if temperature.empty:
        raise ValueError("No finite temperature observations are available")
    keep = data["datetime"].between(temperature.index[0], temperature.index[-1])
    print(f"Rows outside temperature coverage removed: {int((~keep).sum())}")
    data = data.loc[keep].reset_index(drop=True).copy()
    if data.empty:
        raise ValueError("No common flow/rain/temperature coverage")
    grid = pd.DatetimeIndex(data["datetime"])
    margin = pd.Timedelta(hours=max_gap_hours)
    temperature = temperature.loc[grid[0] - margin:grid[-1] + margin]
    gaps = temperature.index.to_series().diff().dropna()
    if gaps.empty or gaps.max() > margin:
        raise ValueError("Temperature gap exceeds TEMPERATURE_MAX_GAP_HOURS")
    aligned = temperature.reindex(temperature.index.union(grid)).interpolate("time", limit_area="inside")
    data["temperature_c"] = aligned.reindex(grid).to_numpy()
    if not np.isfinite(data["temperature_c"]).all():
        raise ValueError("Incomplete temperature coverage; no endpoint filling")
    print("Time convention: fixed EST as in the original notebook; source TSF timezone unverified.")
    return data, path
