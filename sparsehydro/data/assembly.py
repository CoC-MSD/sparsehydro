"""Assemble a model-ready forcing frame from monitoring time series.

Reads one value column per local TSF/CSV file and resamples flow (mean) and
rainfall (sum) onto a regular model time step, using only complete native
bins; nothing is filled silently.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from ..units import MGD_TO_CFS
from .services import DownloadResult
from .utilities import read_tsf

logger = logging.getLogger(__name__)

#: Supported model time steps: name -> (pandas frequency, hours per step).
TIMESTEPS = {"5min": ("5min", 5.0 / 60.0), "15min": ("15min", .25),
             "60min": ("1h", 1.0), "1h": ("1h", 1.0)}


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
            logger.info("%-10s: no file configured; skipped", data_type)
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
    """Log the input period and columns without displaying observations."""
    for data_type, result in results.items():
        data = result.dataframe
        if data is not None and len(data):
            columns = ", ".join(map(str, data.columns))
            logger.info("%-10s: %6d rows (%s -> %s) [%s] -> %s", data_type, len(data),
                        data.index.min(), data.index.max(), columns, result.path)
        else:
            logger.info("%-10s: no parsed data -> %s", data_type, result.path)


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


__all__ = ["TIMESTEPS", "modeling_timestep", "read_local_results", "describe_results",
           "assemble_modeling_data"]
