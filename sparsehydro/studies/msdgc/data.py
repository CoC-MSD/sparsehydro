"""Strict local-input preparation used by whole-flow study clients."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sparsehydro.data.assembly import describe_results
from sparsehydro.data.services import DownloadResult, FlowFinityClient




logger = logging.getLogger(__name__)


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
    download_root: Path, *, record_title: str | None, record_index: int,
    name_filter: str | None, types: Sequence[str], start: str | None = None, end: str | None = None,
    username: str | None = None, password: str | None = None,
    client: FlowFinityClient | None = None,
) -> FlowFinityInputs:
    """Discover FlowFinity records and download the selected study inputs.

    Credentials are passed explicitly or, when omitted, read by
    :class:`~sparsehydro.data.services.FlowFinityClient` from the
    ``FLOWFINITY_USERNAME`` / ``FLOWFINITY_PASSWORD`` environment variables.
    Keep them out of notebooks and source files.

    :param client: An already configured client (credentials are then ignored).
    """
    if client is None:
        client = FlowFinityClient(username=username, password=password)
    logger.info("Client ready -> %s", client.base_url)
    resources = client.list_resources(name_filter)
    records = pd.DataFrame(
        [{"DataTitle": record.get("DataTitle")} for record in resources]
    ).sort_values("DataTitle", ignore_index=True)
    logger.info("%d record(s) found%s", len(records),
                f" matching {name_filter!r}" if name_filter else "")
    title = select_flowfinity_record(records, title=record_title, index=record_index)
    logger.info("Selected record: %r", title)
    logger.info("Available data types: %s", client.available_types(title))
    directory = Path(download_root) / title.replace(" ", "_")
    results = client.download(title, types=list(types), output_dir=directory, parse=True, start=start, end=end)
    describe_results(results)
    return FlowFinityInputs(title, directory, records, results)


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
