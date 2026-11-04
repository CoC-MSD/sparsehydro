"""Shared data utilities for MSDGC flow-monitoring datasets.

This module collects helpers that operate on the time-series data produced by
the service clients in :mod:`data.services`:

* :func:`read_tsf` parses FlowFinity ``.tsf`` time-series files into pandas
  DataFrames.
* :func:`write_tsf` writes a modeled series with the FlowFinity/PCSWMM header.
* :func:`add_time_features` augments a datetime-indexed (or datetime-columned)
  DataFrame with decimal time-of-week / time-of-day / time-of-year features
  commonly used for seasonality modelling.

Example
-------
::

    from sparsehydro.data.utilities import read_tsf, add_time_features

    df = read_tsf("downloads/lstStormTs.tsf")
    df = read_tsf("downloads/lstStormTs.tsf", usecols=["StormFlow"])
    df = add_time_features(df, hour_of_day=True, day_of_week=True)
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["read_tsf", "write_tsf", "add_time_features"]

UseCols = Sequence[str] | Sequence[int] | Callable[[str], bool]


def _resolve_usecols(
    header: Sequence[str],
    usecols: UseCols | None,
) -> list[str] | None:
    """Resolve a pandas-style *usecols* spec against a ``.tsf`` header.

    Mirrors :func:`pandas.read_csv`: *usecols* may be a list-like of column
    names, a list-like of integer positions, or a callable evaluated against
    each column name.  The returned order follows the file, not *usecols*.

    The ``Date/Time`` column (``header[0]``) is always retained because
    :func:`read_tsf` uses it as the index.

    :param header: Column names parsed from the second line of the file.
    :param usecols: The spec to resolve, or ``None`` for every column.
    :returns: The column names to keep, or ``None`` when *usecols* is ``None``.
    :raises ValueError: If *usecols* is not one of the accepted forms, or names
        / positions it requests are absent from *header*.
    """
    if usecols is None:
        return None

    time_col = header[0]

    if callable(usecols):
        selected = [c for c in header if usecols(c)]
    elif isinstance(usecols, (str, bytes)):
        raise ValueError(
            "usecols must be list-like of all strings, all integers, or a "
            "callable; got a bare string.  Pass e.g. usecols=['StormFlow']."
        )
    else:
        items = list(usecols)
        is_int = [
            isinstance(i, (int, np.integer)) and not isinstance(i, (bool, np.bool_))
            for i in items
        ]
        is_str = [isinstance(i, str) for i in items]
        if items and all(is_int):
            bad = [i for i in items if not -len(header) <= int(i) < len(header)]
            if bad:
                raise ValueError(
                    f"usecols positions {bad} are out of bounds for a .tsf file "
                    f"with {len(header)} column(s)."
                )
            wanted = {header[int(i)] for i in items}
        elif items and all(is_str):
            missing = [c for c in items if c not in header]
            if missing:
                raise ValueError(
                    f"usecols columns {missing} not found in the .tsf header; "
                    f"available columns are {list(header)}."
                )
            wanted = set(items)
        elif not items:
            wanted = set()
        else:
            raise ValueError(
                "usecols must be list-like of all strings, all integers, or a "
                "callable; got a mix of types."
            )
        selected = [c for c in header if c in wanted]

    if time_col not in selected:
        selected = [time_col, *selected]
    return selected


def read_tsf(
    source: str | bytes | os.PathLike[str],
    *,
    usecols: UseCols | None = None,
    start: Any = None,
    end: Any = None,
) -> pd.DataFrame:
    """Parse a FlowFinity ``.tsf`` time-series file into a DataFrame.

    The ``.tsf`` layout is::

        IDs:\t<title>
        Date/Time\t<Variable>
        <date-format>\t<unit>
        <datetime>\t<value>
        ...

    The first (``IDs:``) and third (units) lines are skipped; the second
    line provides column names.  The ``Date/Time`` column is parsed to
    datetimes and used as the index.

    *usecols* behaves like the :func:`pandas.read_csv` argument of the same
    name — a list of column names, a list of integer positions, or a callable
    evaluated against each column name::

        read_tsf(path, usecols=["StormFlow"])
        read_tsf(path, usecols=[1, 2])
        read_tsf(path, usecols=lambda c: c.endswith("Flow"))

    As in pandas, the result keeps the file's column order rather than the
    order given in *usecols*.  The ``Date/Time`` column is always retained
    because it becomes the index, so it need not be listed.

    :param source: A filesystem path, the raw file ``bytes``, or the decoded
        text content of a ``.tsf`` file.
    :param usecols: Optional subset of value columns to read.  ``None`` (the
        default) reads every column.
    :param start: Optional inclusive lower datetime bound.
    :param end: Optional inclusive upper datetime bound.
    :returns: DataFrame indexed by ``Date/Time`` with the value column(s).
    :raises ValueError: If the content does not contain the expected header, or
        if *usecols* is malformed or names columns absent from the file.
    """
    if isinstance(source, (str, os.PathLike)) and not (
        isinstance(source, str) and "\n" in source
    ):
        text = Path(source).read_text(encoding="utf-8-sig")
    elif isinstance(source, bytes):
        text = source.decode("utf-8-sig")
    else:
        text = str(source)

    lines = text.splitlines()
    if len(lines) < 3:
        raise ValueError("Not a valid .tsf file: fewer than 3 header lines.")

    # Line 0 = "IDs:\t<title>", line 1 = headers, line 2 = units, data follows.
    header = lines[1].split("\t")
    data_text = "\n".join(lines[3:])

    frame = pd.read_csv(
        StringIO(data_text),
        sep="\t",
        names=header,
        usecols=_resolve_usecols(header, usecols),
        engine="python",
    )

    time_col = header[0]
    raw_time = frame[time_col]
    nonempty = raw_time.notna()
    # FlowFinity .tsf timestamps are US-format (e.g. "1/25/2022 12:40:00 PM").
    # Try known explicit formats first: they parse vectorised in C (orders of
    # magnitude faster than the per-element dateutil fallback) and avoid pandas'
    # "Could not infer format" warning. Fall back to a generic parse only when no
    # explicit format covers every populated row.
    parsed = None
    for fmt in ("%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y"):
        candidate = pd.to_datetime(raw_time, format=fmt, errors="coerce")
        if (candidate.notna() | ~nonempty).all():
            parsed = candidate
            break
    if parsed is None:
        parsed = pd.to_datetime(raw_time, errors="coerce")
    frame[time_col] = parsed
    frame = frame.dropna(subset=[time_col]).set_index(time_col).sort_index()

    if start is not None:
        frame = frame[frame.index >= pd.to_datetime(start)]
    if end is not None:
        frame = frame[frame.index <= pd.to_datetime(end)]

    return frame


def write_tsf(
    series: pd.Series,
    path: str | os.PathLike[str],
    *,
    title: str,
    variable: str,
    unit: str,
) -> None:
    """Write one series using the three-line FlowFinity/PCSWMM TSF header.

    Values are written as supplied; ``unit`` labels them but does not convert
    them. Callers must select the export period and convert units beforehand.
    Timestamps must be ordered, unique, timezone-naive and second-aligned.
    The destination's parent directory must already exist.

    :raises ValueError: If the series is empty, timestamps are invalid, values
        are not finite numbers, or header fields contain tabs/newlines.
    """
    for label, value in (("title", title), ("variable", variable), ("unit", unit)):
        if not value.strip() or any(char in value for char in "\t\r\n"):
            raise ValueError(f"TSF {label} must be nonempty and contain no tabs/newlines")
    index = series.index
    if (series.empty or not isinstance(index, pd.DatetimeIndex)
            or index.hasnans or index.has_duplicates or not index.is_monotonic_increasing):
        raise ValueError("TSF requires a nonempty series with unique ascending timestamps")
    if index.tz is not None or not index.equals(index.floor("s")):
        raise ValueError("TSF timestamps must be timezone-naive and second-aligned")
    values = pd.to_numeric(series, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("TSF values must all be finite")
    if variable == "Date/Time":
        raise ValueError("TSF variable must differ from Date/Time")
    output = pd.DataFrame({
        "Date/Time": index.strftime("%m/%d/%Y %H:%M:%S"),
        variable: values,
    })
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        stream.write(f"IDs:\t{title}\nDate/Time\t{variable}\nM/d/yyyy\t{unit}\n")
        output.to_csv(stream, sep="\t", index=False, header=False,
                      float_format="%.12g", lineterminator="\n")


def add_time_features(
    df: pd.DataFrame,
    *,
    hour_of_day: bool = False,
    day_of_week: bool = False,
    day_of_year: bool = False,
    datetime_col: str | None = None,
) -> pd.DataFrame:
    """Add decimal time-feature columns to a DataFrame.

    Each feature is opt-in via its boolean flag.  Timestamps are taken from
    *datetime_col* when given, otherwise from the DataFrame's
    :class:`~pandas.DatetimeIndex`, otherwise from a column named
    ``"datetime"``.

    The decimal features are:

    * ``hour_of_day`` — hour of the day in ``[0, 24)``, computed as
      ``hour + minute / 60 + second / 3600``.
    * ``day_of_week`` — day of the week in ``[0, 7)`` with Monday as ``0``,
      computed as ``dayofweek + hour_of_day / 24``.
    * ``day_of_year`` — day of the year (``1``-based) plus the fractional day,
      computed as ``dayofyear + hour_of_day / 24``.

    :param df: Source DataFrame.  It is not modified; a copy is returned.
    :param hour_of_day: When ``True``, add a ``"hour_of_day"`` column.
    :param day_of_week: When ``True``, add a ``"day_of_week"`` column.
    :param day_of_year: When ``True``, add a ``"day_of_year"`` column.
    :param datetime_col: Name of the column holding timestamps.  When ``None``,
        the DatetimeIndex is used if present, else a ``"datetime"`` column.
    :returns: A copy of *df* with the requested feature column(s) appended.
    :raises KeyError: If *datetime_col* is given but absent, or if no datetime
        source can be located.
    :raises TypeError: If the located source cannot be parsed as datetimes.
    """
    result = df.copy()

    if datetime_col is not None:
        if datetime_col not in result.columns:
            raise KeyError(f"Column {datetime_col!r} not found in DataFrame.")
        dt = pd.to_datetime(result[datetime_col])
    elif isinstance(result.index, pd.DatetimeIndex):
        dt = result.index.to_series()
    elif "datetime" in result.columns:
        dt = pd.to_datetime(result["datetime"])
    else:
        raise KeyError(
            "No datetime source found: pass datetime_col=, provide a "
            "DatetimeIndex, or include a 'datetime' column."
        )

    accessor = dt.dt
    decimal_hour = accessor.hour + accessor.minute / 60.0 + accessor.second / 3600.0
    decimal_hour = pd.Series(decimal_hour.to_numpy(), index=result.index)

    if hour_of_day:
        result["hour_of_day"] = decimal_hour
    if day_of_week:
        dow = pd.Series(accessor.dayofweek.to_numpy(), index=result.index)
        result["day_of_week"] = dow + decimal_hour / 24.0
    if day_of_year:
        doy = pd.Series(accessor.dayofyear.to_numpy(), index=result.index)
        result["day_of_year"] = doy + decimal_hour / 24.0

    return result
