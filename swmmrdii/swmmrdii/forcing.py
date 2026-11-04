"""
Time-series forcing data loaders for swmmrdii.

SWMM and PCSWMM use two common plain-text time-series formats:

    .dat (space-delimited)
        Typically used for continuous forcing data such as PRISM precipitation
        or temperature grids that have been extracted to gage locations.
        Column layout: gage  year  month  day  hour  minute  value

    .tsf (tab-delimited, multi-block)
        PCSWMM's time-series file format.  A single file can contain multiple
        named blocks (one per gage or station), each starting with an ``IDs:``
        header row.  Used for both observed flow and model output.

Both formats are generic SWMM constructs — neither loader is specific to
PRISM or USGS data.  The callers (``RdiiModel.load_forcing`` /
``RdiiModel.load_observed``) choose which loader to call based on the file
extension, but the loaders themselves can read any conforming .dat or .tsf
file.

Public API
----------
load_dat(path, value_col)
    Parse a space-delimited .dat file into a datetime-indexed DataFrame.

load_tsf(path, series_id, value_col, datetime_fmt)
    Parse a single named block from a multi-block .tsf file.

resolve_conversion(conversion)
    Resolve a string preset or callable to a conversion function (or None).
"""

import os
from io import StringIO

import pandas as pd

# ---------------------------------------------------------------------------
# Built-in unit-conversion presets
# ---------------------------------------------------------------------------
# Keys are lowercase strings accepted by load_forcing(temp_conversion=...) and
# load_forcing(ppt_conversion=...).  Each value is a single-argument callable
# that receives a pandas Series and returns a converted Series.
#
# Extend this dict to add new presets without touching the callers.

CONVERSIONS: dict[str, object] = {
    # Temperature
    "degf_to_degc":     lambda x: (x - 32.0) * 5.0 / 9.0,
    "degc_to_degf":     lambda x: x * 9.0 / 5.0 + 32.0,
    "kelvin_to_degc":   lambda x: x - 273.15,
    "degc_to_kelvin":   lambda x: x + 273.15,
    # Precipitation
    "mm_to_in":         lambda x: x / 25.4,
    "in_to_mm":         lambda x: x * 25.4,
    "cm_to_in":         lambda x: x / 2.54,
    "in_to_cm":         lambda x: x * 2.54,
    # Flow
    "cms_to_cfs":       lambda x: x * 35.3147,
    "cfs_to_cms":       lambda x: x / 35.3147,
}


# ---------------------------------------------------------------------------
# Temperature-unit → conversion-preset mapping
# ---------------------------------------------------------------------------
# Lets callers pass a raw unit string (what the data is in) instead of naming
# the conversion direction.  sparsehydro expects °C, so each entry maps the
# source unit to the preset that converts it to °C (None = already °C).

_TEMP_UNIT_TO_PRESET: dict[str, object] = {
    "degf":      "degf_to_degc",
    "f":         "degf_to_degc",
    "fahrenheit": "degf_to_degc",
    "degc":      None,
    "c":         None,
    "celsius":   None,
    "k":         "kelvin_to_degc",
    "kelvin":    "kelvin_to_degc",
}


def resolve_temp_unit(unit):
    """
    Resolve a temperature *unit* string to the conversion needed to reach °C.

    sparsehydro expects temperature in degrees Celsius.  This helper lets the
    user state what unit their data is *in* (e.g. ``'degF'``) rather than
    naming the conversion direction.

    Parameters
    ----------
    unit : str, callable, or None
        Source temperature unit.  Accepted strings (case-insensitive):
        ``'degF'`` / ``'F'`` / ``'Fahrenheit'``, ``'degC'`` / ``'C'`` /
        ``'Celsius'``, ``'K'`` / ``'Kelvin'``.  A callable is returned
        unchanged (treated as a custom conversion).  ``None`` returns ``None``.

    Returns
    -------
    callable or None
        A conversion function (or ``None`` when no conversion is needed).

    Raises
    ------
    KeyError
        If *unit* is an unrecognised string.
    """
    if unit is None:
        return None
    if callable(unit):
        return unit
    if isinstance(unit, str):
        key = unit.strip().lower().replace("°", "").replace(" ", "")
        if key.startswith("deg"):
            key = key  # already e.g. "degf"
        if key not in _TEMP_UNIT_TO_PRESET:
            available = ", ".join(sorted({u for u in _TEMP_UNIT_TO_PRESET}))
            raise KeyError(
                f"Unknown temperature unit {unit!r}. "
                f"Accepted units: {available}. "
                "Pass a callable for custom conversions."
            )
        return resolve_conversion(_TEMP_UNIT_TO_PRESET[key])
    raise TypeError(
        f"unit must be a string, callable, or None; got {type(unit).__name__!r}"
    )


# ---------------------------------------------------------------------------
# Precipitation-unit → conversion-preset mapping
# ---------------------------------------------------------------------------
# sparsehydro expects precipitation in inches.  Each entry maps the source
# unit to the preset that converts it to inches (None = already inches).

_PPT_UNIT_TO_PRESET: dict[str, object] = {
    "in":         None,
    "inch":       None,
    "inches":     None,
    "mm":         "mm_to_in",
    "millimeter": "mm_to_in",
    "millimeters": "mm_to_in",
    "cm":         "cm_to_in",
    "centimeter": "cm_to_in",
    "centimeters": "cm_to_in",
}


def resolve_ppt_unit(unit):
    """
    Resolve a precipitation *unit* string to the conversion needed to reach
    inches.

    sparsehydro expects precipitation in inches.  This helper lets the user
    state what unit their data is *in* (e.g. ``'mm'``) rather than naming the
    conversion direction.

    Parameters
    ----------
    unit : str, callable, or None
        Source precipitation unit.  Accepted strings (case-insensitive):
        ``'in'`` / ``'inch'`` / ``'inches'``, ``'mm'`` / ``'millimeters'``,
        ``'cm'`` / ``'centimeters'``.  A callable is returned unchanged
        (treated as a custom conversion).  ``None`` returns ``None``.

    Returns
    -------
    callable or None
        A conversion function (or ``None`` when no conversion is needed).

    Raises
    ------
    KeyError
        If *unit* is an unrecognised string.
    """
    if unit is None:
        return None
    if callable(unit):
        return unit
    if isinstance(unit, str):
        key = unit.strip().lower().replace('"', "").replace(" ", "")
        if key not in _PPT_UNIT_TO_PRESET:
            available = ", ".join(sorted({u for u in _PPT_UNIT_TO_PRESET}))
            raise KeyError(
                f"Unknown precipitation unit {unit!r}. "
                f"Accepted units: {available}. "
                "Pass a callable for custom conversions."
            )
        return resolve_conversion(_PPT_UNIT_TO_PRESET[key])
    raise TypeError(
        f"unit must be a string, callable, or None; got {type(unit).__name__!r}"
    )


def resolve_conversion(conversion):
    """
    Resolve a conversion argument to a callable (or None).

    Accepts:

    * ``None``      — no conversion; returns ``None``.
    * ``str``       — looked up in :data:`CONVERSIONS` (case-insensitive).
      Raises ``KeyError`` with a helpful message if the key is unknown.
    * ``callable``  — returned unchanged.

    Parameters
    ----------
    conversion : str, callable, or None

    Returns
    -------
    callable or None

    Examples
    --------
    >>> fn = resolve_conversion('degF_to_degC')
    >>> fn = resolve_conversion(lambda x: (x - 32) * 5/9)
    >>> fn = resolve_conversion(None)   # no-op
    """
    if conversion is None:
        return None
    if callable(conversion):
        return conversion
    if isinstance(conversion, str):
        key = conversion.lower().replace(" ", "_").replace("-", "_")
        if key not in CONVERSIONS:
            available = ", ".join(sorted(CONVERSIONS))
            raise KeyError(
                f"Unknown conversion preset {conversion!r}. "
                f"Available presets: {available}. "
                "Pass a callable for custom conversions."
            )
        return CONVERSIONS[key]
    raise TypeError(
        f"conversion must be a string preset, callable, or None; got {type(conversion).__name__!r}"
    )


# Per-file index of the [TIMESERIES] section: series name -> list of
# contiguous [byte_start, byte_end) ranges holding that series' rows.
# Keyed by (abspath, mtime_ns, size) so edits invalidate naturally.  The
# index itself is tiny (a few tuples per series), so caching a handful of
# files is safe even when the INPs are hundreds of MB.
_TS_INDEX_CACHE: dict[tuple, "dict[str, list[list[int]]] | None"] = {}
_TS_INDEX_CACHE_MAX = 4


def _timeseries_index(inp_path: str) -> "dict[str, list[list[int]]] | None":
    """
    Build (or fetch) the byte-range index of a file's ``[TIMESERIES]`` section.

    One pass over the file records, for each series name, the byte ranges of
    its data rows (adjacent rows are merged into one range, so a typical
    contiguous series block is a single seek + read).  Returns ``None`` when
    the file has no ``[TIMESERIES]`` section.
    """
    st = os.stat(inp_path)
    key = (os.path.abspath(inp_path), st.st_mtime_ns, st.st_size)
    if key in _TS_INDEX_CACHE:
        return _TS_INDEX_CACHE[key]

    index: dict[str, list[list[int]]] = {}
    in_ts = found = False
    offset = 0
    with open(inp_path, "rb") as f:
        for line in f:
            line_start = offset
            offset += len(line)
            stripped = line.strip()
            if stripped.startswith(b"["):
                if in_ts:
                    break   # next section — done
                in_ts = stripped.upper() == b"[TIMESERIES]"
                found = found or in_ts
                continue
            if not in_ts or not stripped or stripped.startswith(b";"):
                continue
            name = stripped.split(None, 1)[0].decode("utf-8", errors="replace")
            ranges = index.setdefault(name, [])
            if ranges and ranges[-1][1] == line_start:
                ranges[-1][1] = offset          # extend the contiguous range
            else:
                ranges.append([line_start, offset])

    result = index if found else None
    if len(_TS_INDEX_CACHE) >= _TS_INDEX_CACHE_MAX:
        _TS_INDEX_CACHE.pop(next(iter(_TS_INDEX_CACHE)))
    _TS_INDEX_CACHE[key] = result
    return result


def parse_swmm_timeseries(
    inp_path: str,
    series_name: str,
    value_col: str = "value",
) -> pd.DataFrame:
    """
    Read a named series from a SWMM INP ``[TIMESERIES]`` section.

    SWMM time series come in three forms; this loader handles all three:

    1. **Inline, dated** — rows of ``Name  MM/DD/YYYY  HH:MM:SS  Value``.
    2. **Inline, time-only** — rows of ``Name  HH:MM  Value`` (hours from the
       simulation start).  Returned with a ``RangeIndex`` of hours rather than
       a datetime index, since no absolute dates are given.
    3. **External file** — a single ``Name FILE "path"`` line.  The referenced
       file is read by extension: ``.tsf`` via :func:`load_tsf` (block matched
       by the series name), anything else via :func:`load_dat` (SWMM ``.dat``
       layout).  PCSWMM binary ``.tsb`` files are not readable and raise
       ``NotImplementedError``.

    The continuation convention (a row that repeats the series name but omits
    the date, carrying the previous row's date) is supported for the dated
    inline form.

    Parameters
    ----------
    inp_path : str
        Path to the SWMM .inp file.
    series_name : str
        Name of the series in the ``[TIMESERIES]`` section.
    value_col : str
        Name to assign to the returned value column.

    Returns
    -------
    pd.DataFrame
        Datetime-indexed (dated form) or hour-indexed (time-only form) frame
        with one column named *value_col*.

    Raises
    ------
    ValueError
        If *series_name* is not found in the ``[TIMESERIES]`` section.
    """
    # Read ONLY this series' rows via the per-file byte-range index.  A
    # naive scan re-reads the whole INP for every series, which is
    # prohibitive for PCSWMM exports that embed hundreds of series inline
    # (a 258 MB INP x 12 series = ~3 minutes); the index is built in one
    # pass and cached, so each series load is a seek + small read.
    index = _timeseries_index(inp_path)
    if index is None:
        raise ValueError(f"No [TIMESERIES] section found in {inp_path!r}")
    ranges = index.get(series_name)
    if ranges is None:
        raise ValueError(
            f"Series {series_name!r} not found (or empty) in [TIMESERIES] of {inp_path!r}"
        )
    chunks = []
    with open(inp_path, "rb") as f:
        for r_start, r_end in ranges:
            f.seek(r_start)
            chunks.append(f.read(r_end - r_start))
    lines = b"".join(chunks).decode("utf-8", errors="replace").splitlines()

    # Collect rows belonging to this series.
    file_path: str | None = None
    records: list[tuple] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith(";"):
            continue
        # Split on whitespace but keep quoted paths intact.
        tokens = _split_keep_quotes(line)
        if not tokens or tokens[0] != series_name:
            continue
        rest = tokens[1:]
        if not rest:
            continue
        # FILE form: Name FILE "path"
        if rest[0].upper() == "FILE":
            file_path = rest[1].strip('"') if len(rest) > 1 else None
            break
        records.append(tuple(rest))

    if file_path is not None:
        # Delegate to the loader matching the referenced file's format.
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".tsf":
            # PCSWMM tab-delimited multi-block file; match block by the
            # SWMM series name.
            return load_tsf(file_path, series_id=series_name, value_col=value_col)
        if ext == ".tsb":
            raise NotImplementedError(
                f"Series {series_name!r} references a PCSWMM binary .tsb file "
                f"({file_path!r}), which swmmrdii cannot read. Export the "
                "series as .tsf (text) or .dat and repoint the reference."
            )
        return load_dat(file_path, value_col=value_col)

    if not records:
        raise ValueError(
            f"Series {series_name!r} not found (or empty) in [TIMESERIES] of {inp_path!r}"
        )

    # Decide inline form: dated (>=3 fields: date time value) vs time-only.
    # A dated row looks like ('MM/DD/YYYY', 'HH:MM:SS', 'value').
    # A continuation row looks like ('HH:MM:SS', 'value') carrying prior date.
    datetimes: list = []
    values: list[float] = []
    hours: list[float] = []
    last_date: str | None = None
    is_dated = "/" in records[0][0]

    if is_dated:
        for rec in records:
            if "/" in rec[0]:
                # date time value
                date_s, time_s, val_s = rec[0], rec[1], rec[2]
                last_date = date_s
            else:
                # continuation: time value (reuse last_date)
                time_s, val_s = rec[0], rec[1]
                date_s = last_date
            datetimes.append(f"{date_s} {time_s}")
            values.append(float(val_s))
        idx = pd.to_datetime(pd.Series(datetimes), errors="coerce")
        df = pd.DataFrame({value_col: values})
        df.index = idx
        df.index.name = "datetime"
        # Collapse sub-daily to the date if all times are midnight (PRISM daily).
        return df[df.index.notna()]
    else:
        # Time-only form: HH:MM value → hours-from-start index.
        for rec in records:
            t, val_s = rec[0], rec[1]
            if ":" in t:
                h, _, mn = t.partition(":")
                hours.append(float(h) + float(mn or 0) / 60.0)
            else:
                hours.append(float(t))
            values.append(float(val_s))
        df = pd.DataFrame({value_col: values}, index=pd.Index(hours, name="hours"))
        return df


def _split_keep_quotes(line: str) -> list[str]:
    """Whitespace-split a line but keep ``\"quoted strings\"`` as one token."""
    out: list[str] = []
    buf = ""
    in_quote = False
    for ch in line:
        if ch == '"':
            in_quote = not in_quote
            buf += ch
        elif ch.isspace() and not in_quote:
            if buf:
                out.append(buf)
                buf = ""
        else:
            buf += ch
    if buf:
        out.append(buf)
    return out


def load_dat(path: str, value_col: str = "value") -> pd.DataFrame:
    """
    Parse a SWMM-style space-delimited .dat file.

    Expected column layout (space-separated, no header row):
        gage  year  month  day  hour  minute  value

    This format is produced by PRISM raster extraction tools and by PCSWMM's
    "Export to .dat" function for any continuous time series.

    Parameters
    ----------
    path : str
        Path to the .dat file.
    value_col : str
        Name to assign to the value column in the returned DataFrame.
        Use descriptive names such as ``'rainfall_in'``, ``'temperature_c'``,
        or ``'flow_cfs'`` so downstream code can identify the column by role.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by ``datetime`` with a single column named
        *value_col*.  The gage / hour / minute columns are consumed during
        parsing and discarded.

    Examples
    --------
    >>> ppt = load_dat('PRISM_RG_022.dat', value_col='rainfall_in')
    >>> tmean = load_dat('PRISM_RG_022_tmean.dat', value_col='temperature_c')
    >>> # Convert PRISM TMEAN (delivered in degrees F) to Celsius:
    >>> tmean['temperature_c'] = (tmean['temperature_c'] - 32.0) * 5.0 / 9.0
    """
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["gage", "year", "month", "day", "hour", "minute", "value"],
        comment=";",  # skip SWMM-style header/comment lines (e.g. ";Rainfall (F)")
    )
    df["datetime"] = pd.to_datetime(df[["year", "month", "day"]])
    return df.set_index("datetime")[["value"]].rename(columns={"value": value_col})


def _tsf_header_names(line: str) -> list[str]:
    """
    Split a TSF ``IDs:`` header line into its series names, exactly.

    Cells are tab-delimited; the leading ``IDs:`` tag and PCSWMM's
    ``" (obs)"`` suffix are stripped, mirroring the column normalization in
    ``load_tsf``.  Names must then match by string equality — substring
    matching is unsafe because one series id can be a prefix of another
    (e.g. ``'0314548'`` vs ``'03145483'``).
    """
    names = []
    for cell in line.strip().split("\t"):
        name = cell.strip()
        if name.startswith("IDs:"):
            name = name[len("IDs:"):].strip()
        name = name.replace(" (obs)", "").strip()
        if name:
            names.append(name)
    return names


def load_tsf(
    path: str,
    series_id: str,
    value_col: str = "value",
    datetime_fmt: str = "%m/%d/%Y %I:%M:%S %p",
) -> pd.DataFrame:
    """
    Read a single named time-series block from a PCSWMM/SWMM .tsf file.

    .tsf files are tab-delimited and may contain multiple blocks, one per
    gage or model element.  Each block starts with a header line beginning
    with ``IDs:``; the function locates the block whose header contains
    *series_id* and returns its data as a DataFrame.

    Parameters
    ----------
    path : str
        Path to the .tsf file.
    series_id : str
        Identifier string that appears in the block's ``IDs:`` header line.
        For USGS gage files this is typically the 8-digit gage number
        (e.g. ``'03145483'``).  For model-output TSF files it matches the
        subcatchment or node name.
    value_col : str
        Name to assign to the data column in the returned DataFrame
        (default ``'value'``).  Use ``'flow_cfs'`` for streamflow,
        ``'rainfall_in'`` for precipitation, etc.
    datetime_fmt : str
        ``strptime`` format string for the datetime column.  PCSWMM exports
        use ``'%m/%d/%Y %I:%M:%S %p'`` (12-hour clock with AM/PM) by default.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by ``datetime`` with a single column named
        *value_col*.

    Raises
    ------
    ValueError
        If no block containing *series_id* is found in the file.

    Examples
    --------
    >>> flow = load_tsf('USGS_DV_Discharge_24hr_Shift.tsf',
    ...                 series_id='03145483', value_col='flow_cfs')
    >>> rdii = load_tsf('SH_SubcatchmentA.dat',
    ...                 series_id='SubcatchmentA', value_col='rdii_cfs',
    ...                 datetime_fmt='%m/%d/%Y %H:%M:%S')
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    # Locate all block start lines (each block begins with an IDs: header).
    block_starts = [i for i, line in enumerate(lines) if line.strip().startswith("IDs:")]

    target_idx = next(
        (
            idx for idx, start in enumerate(block_starts)
            if series_id in _tsf_header_names(lines[start])
        ),
        None,
    )
    if target_idx is None:
        available = sorted(
            {n for start in block_starts for n in _tsf_header_names(lines[start])}
        )
        raise ValueError(
            f"Series {series_id!r} not found in any block header in {path!r}. "
            f"Available series: {available}"
        )

    header_line = block_starts[target_idx]
    # Data rows start 3 lines after the IDs: header (header, units, blank).
    data_start = header_line + 3
    data_end = (
        block_starts[target_idx + 1]
        if target_idx + 1 < len(block_starts)
        else len(lines)
    )

    # Parse the header line to get column names.
    raw_cols = lines[header_line].strip().split("\t")
    col_names = [
        c.replace("IDs:", "datetime").replace(" (obs)", "").strip() for c in raw_cols
    ]

    df = pd.read_csv(
        StringIO("".join(lines[data_start:data_end])),
        sep="\t",
        header=None,
        names=col_names,
    )
    df["datetime"] = pd.to_datetime(df["datetime"], format=datetime_fmt, errors="coerce")
    df = df.set_index("datetime").loc[lambda d: d.index.notna()]
    df = df.apply(pd.to_numeric, errors="coerce")

    # Find the data column — exact name match (col_names above were already
    # stripped of the "IDs:" tag and " (obs)" suffix, same as
    # _tsf_header_names).  Substring matching would mis-pick when one series
    # id is a prefix of another in a multi-ID block.
    data_col = next((c for c in df.columns if c.strip() == series_id), None)
    if data_col is None:
        # Fall back to the first non-datetime column if no name match.
        data_col = df.columns[0]

    return df[[data_col]].rename(columns={data_col: value_col})


# ---------------------------------------------------------------------------
# Backwards-compatible aliases
# (The original notebook code called load_prism_dat / load_tsf_gage.
#  These thin wrappers preserve that interface so existing notebooks
#  continue to work without modification.)
# ---------------------------------------------------------------------------

def load_prism_dat(path: str, value_col: str = "rainfall_in") -> pd.DataFrame:
    """
    Alias for ``load_dat()``.

    Retained for backward compatibility with existing notebook code that
    calls ``load_prism_dat()``.  New code should call ``load_dat()`` directly.
    """
    return load_dat(path, value_col=value_col)


def load_tsf_gage(
    path: str,
    gage_id: str,
    datetime_fmt: str = "%m/%d/%Y %I:%M:%S %p",
) -> pd.DataFrame:
    """
    Alias for ``load_tsf()`` that returns the data column as ``'flow_cfs'``.

    Retained for backward compatibility with existing notebook code that
    calls ``load_tsf_gage()``.  New code should call ``load_tsf()`` directly.
    """
    return load_tsf(path, series_id=gage_id, value_col="flow_cfs", datetime_fmt=datetime_fmt)
