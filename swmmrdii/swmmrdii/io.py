"""
SWMM and TSF time-series export utilities for swmmrdii.

After calibration, each subcatchment's predicted RDII is written to a .tsf
file in PCSWMM's tab-delimited multi-block format.  Two companion text
snippets are also generated:

    [INFLOWS]     — assigns each TSF time series to a sewer node (junction)
    [TIMESERIES]  — registers each .tsf file with a unique time-series name

These snippets can be pasted directly into a SWMM .inp file or appended
programmatically via a SWMM editing API.
"""

from __future__ import annotations

import os

import pandas as pd

# Prefix for sparsehydro-generated series/file names (time-series names in
# [TIMESERIES]/[INFLOWS] exports, .tsf files, and .pkl bundles).  Shared
# with swmm_runner, which uses it to recognize and drop STALE sparsehydro
# static-export [INFLOWS] rows when building the routing-only INP — rows
# exported with a custom prefix are NOT recognized there, so prefer the
# default unless you have a reason not to.
SH_SERIES_PREFIX = "SH_"


def write_tsf(
    pred_df: pd.DataFrame,
    sc_name: str,
    out_path: str,
    sim_start: str | pd.Timestamp,
    sim_end: str | pd.Timestamp,
    flow_col: str = "rdii_cfs",
    flow_unit: str = "cfs",
    tsf_name: str | None = None,
) -> str:
    """
    Write a single subcatchment's predicted RDII to a PCSWMM .tsf file.

    The file uses PCSWMM's tab-delimited multi-block TSF format, readable by
    ``load_tsf()`` and importable directly into PCSWMM/SWMM:

        IDs:\\t<tsf_name>
        Date/Time\\t<tsf_name>
        M/d/yyyy\\t<unit>
        1/1/2008 12:00:00 AM\\t45.3200
        ...

    Parameters
    ----------
    pred_df : pd.DataFrame
        Output of ``model.predict()``.  Must contain a ``datetime`` column and
        the column named by *flow_col*.
    sc_name : str
        Subcatchment name; used in the default *tsf_name*.
    out_path : str
        Full path to the output .tsf file.
    sim_start, sim_end : str or Timestamp
        Inclusive bounds of the export window (warm-up period is excluded).
    flow_col : str
        Name of the flow column in *pred_df* (default ``'rdii_cfs'``).
    flow_unit : str
        Unit label written in the header row (default ``'cfs'``).
    tsf_name : str or None
        Time-series label used as the series identifier in the TSF header and
        in the SWMM [INFLOWS] / [TIMESERIES] blocks.  Defaults to
        ``'SH_{sc_name}'``.

    Returns
    -------
    str
        The time-series name (``'SH_{sc_name}'`` or the provided *tsf_name*),
        used when building [INFLOWS] and [TIMESERIES] blocks.
    """
    if tsf_name is None:
        tsf_name = f"{SH_SERIES_PREFIX}{sc_name}"

    # Trim to the export window (excludes warm-up rows).
    mask = (pred_df["datetime"] >= sim_start) & (pred_df["datetime"] <= sim_end)
    pred_out = pred_df.loc[mask].reset_index(drop=True)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        # PCSWMM TSF header (3 rows before data).
        f.write(f"IDs:\t{tsf_name}\n")
        f.write(f"Date/Time\t{tsf_name}\n")
        f.write(f"M/d/yyyy\t{flow_unit}\n")
        for _, row in pred_out.iterrows():
            dt = pd.Timestamp(row["datetime"])
            # PCSWMM 12-hour datetime format: M/d/yyyy h:mm:ss AM/PM (no zero-padding)
            date_str = f"{dt.month}/{dt.day}/{dt.year} {dt.strftime('%I:%M:%S %p')}"
            f.write(f"{date_str}\t{row[flow_col]:.4f}\n")

    return tsf_name


def write_tsf_multi(
    series_dict: dict[str, pd.Series],
    out_path: str,
    sim_start: str | pd.Timestamp,
    sim_end: str | pd.Timestamp,
    flow_unit: str = "cfs",
    name_prefix: str = SH_SERIES_PREFIX,
) -> dict[str, str]:
    """
    Write multiple junction RDII series into a single PCSWMM .tsf file.

    Each series becomes one block in the file, separated by a blank line.
    SWMM's ``[TIMESERIES] FILE`` directive can reference the same file
    multiple times with different series names — SWMM locates each block
    by matching the name in the ``IDs:`` header.

    Parameters
    ----------
    series_dict : dict
        ``{junction_id: pd.Series(rdii_cfs, index=DatetimeIndex)}``.
        Keys become the series names (prefixed with *name_prefix*).
    out_path : str
        Full path to the combined output .tsf file.
    sim_start, sim_end : str or Timestamp
        Inclusive export window (warm-up excluded).
    flow_unit : str
        Unit label in each block header (default ``'cfs'``).
    name_prefix : str
        Prepended to each junction ID to form the series name
        (default ``'SH_'``).

    Returns
    -------
    dict
        ``{junction_id: tsf_series_name}`` — the names to use in
        ``[INFLOWS]`` and ``[TIMESERIES]`` blocks.
    """
    sim_start = pd.Timestamp(sim_start)
    sim_end   = pd.Timestamp(sim_end)
    name_map  = {}   # {junction_id: tsf_series_name}

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for i, (jn_id, series) in enumerate(sorted(series_dict.items())):
            tsf_name = f"{name_prefix}{jn_id}"
            name_map[jn_id] = tsf_name

            # Trim to export window.
            clipped = series[(series.index >= sim_start) & (series.index <= sim_end)]

            if i > 0:
                f.write("\n")  # blank line between blocks

            f.write(f"IDs:\t{tsf_name}\n")
            f.write(f"Date/Time\t{tsf_name}\n")
            f.write(f"M/d/yyyy\t{flow_unit}\n")
            for dt, val in clipped.items():
                ts = pd.Timestamp(dt)
                date_str = f"{ts.month}/{ts.day}/{ts.year} {ts.strftime('%I:%M:%S %p')}"
                f.write(f"{date_str}\t{val:.4f}\n")

    return name_map


def write_swmm_inflows(
    sc_records: list[dict],
    tsf_map: dict[str, str],
    out_path: str,
    constituent: str = "FLOW",
    mfactor: float = 1.0,
    sfactor: float = 1.0,
    baseline: float = 0.0,
) -> None:
    """
    Write a SWMM ``[INFLOWS]`` block assigning each TSF to its outlet node.

    Parameters
    ----------
    sc_records : list of dicts
        Each dict must have keys ``'name'`` (subcatchment name) and
        ``'outlet'`` (the SWMM junction node that receives the inflow).
    tsf_map : dict
        Mapping of subcatchment name → time-series name, as returned by
        ``write_tsf()``.
    out_path : str
        File path for the output text snippet (e.g. ``artifacts/swmm_inflows.txt``).
    constituent : str
        SWMM constituent type (almost always ``'FLOW'``).
    mfactor, sfactor, baseline : float
        SWMM multiplier, scale factor, and baseline flow (defaults match SWMM
        no-modification behaviour).
    """
    lines = [
        "[INFLOWS]",
        ";;Node                Constituent  Time Series           Type    Mfactor Sfactor Baseline",
        ";;--------------------  -----------  --------------------  ------  ------- ------- --------",
    ]
    for sc in sc_records:
        # Each subcatchment's RDII is injected at its outlet junction.
        lines.append(
            f"{sc['outlet']:<22s}  {constituent:<11s}  {tsf_map[sc['name']]:<22s}  "
            f"FLOW    {mfactor:<7.1f} {sfactor:<7g} {baseline}"
        )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_swmm_timeseries(
    tsf_map: dict[str, str],
    tsf_dir: str,
    out_path: str,
) -> None:
    """
    Write a SWMM ``[TIMESERIES]`` FILE block referencing each .dat file.

    Parameters
    ----------
    tsf_map : dict
        Mapping of subcatchment name → time-series name (``'SH_...'``).
    tsf_dir : str
        Directory that contains the .dat files written by ``write_tsf()``.
        Absolute paths are used so the SWMM model can be run from any
        working directory.
    out_path : str
        File path for the output text snippet.
    """
    tsf_abs = os.path.abspath(tsf_dir)

    lines = [
        "[TIMESERIES]",
        ";;Name                    Date  Time  Value",
        ";;------------------------",
    ]
    for _sc_name, ts_name in tsf_map.items():
        # SWMM FILE directive: Name  FILE  "absolute/path/to/file.tsf"
        tsf_path = os.path.join(tsf_abs, f"{ts_name}.tsf")
        lines.append(f'{ts_name:<26s}  FILE  "{tsf_path}"')

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
