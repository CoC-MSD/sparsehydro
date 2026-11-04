"""Display helpers that keep notebook plot outputs small.

A plotly figure embedded in a notebook stores every point as JSON, and datetime
x values as ~25-byte strings.  A multi-year 5- or 15-minute record therefore
produces several MB *per trace*, and a calibration notebook with a dozen
time-series panels grows to tens of MB.

:func:`thin_for_display` returns a lightweight **copy** of a figure for inline
display.  The original is untouched, so ``fig.write_html(...)`` artifacts stay
full-resolution.  Thinning is a per-bin **min/max envelope** (first, min, max
and last point of each bin, plus the first NaN to keep gaps), not a stride, so
storm peaks and dry-weather troughs survive.  Traces sharing an x axis are
thinned at the *same* indices, which keeps stacked / filled traces aligned.

:func:`add_val_vrects` shades the validation runs of a train / validation
split (see :mod:`sparsehydro.calibration.splits`).
"""

from __future__ import annotations

import base64
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

#: Trace types that :func:`thin_for_display` thins; all others are left as-is.
THIN_TRACE_TYPES = ("scatter", "scattergl")

#: Per-point trace attributes subsampled together with ``x`` / ``y``.
_POINT_ATTRS = ("text", "hovertext", "customdata", "ids")
_MARKER_POINT_ATTRS = ("color", "size", "symbol", "opacity")

VAL_FILL = "rgba(255,127,14,0.07)"


def _as_array(values: Any) -> np.ndarray | None:
    """Trace data as a 1-D numpy array (``None`` when absent or scalar).

    Plotly typed-array specs (``{"dtype": "f8", "bdata": <base64>}``, how plotly
    serialises numpy data, e.g. in a figure re-loaded from notebook JSON) are
    decoded.
    """
    if values is None or isinstance(values, (str, bytes)):
        return None
    if isinstance(values, dict):
        if "bdata" not in values or "dtype" not in values or values.get("shape"):
            return None
        return np.frombuffer(base64.b64decode(values["bdata"]),
                             dtype=np.dtype(values["dtype"]).newbyteorder("<"))
    arr = np.asarray(values)
    return arr if arr.ndim == 1 else None


def _x_key(x: np.ndarray | None, n: int) -> tuple:
    """Grouping key: traces with the same key share their x values."""
    if x is None or x.size == 0:
        return ("index", n)
    return ("x", n, str(x[0]), str(x[-1]))


def _envelope_indices(y: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """First, min, max and last index of every bin, plus each bin's first NaN.

    :param y: Values of one trace.
    :type y: numpy.ndarray
    :param bins: Non-decreasing bin id per point (same length as *y*).
    :type bins: numpy.ndarray
    :returns: Sorted unique indices.
    :rtype: numpy.ndarray
    """
    n = y.size
    idx = np.arange(n)
    starts = np.flatnonzero(np.r_[True, bins[1:] != bins[:-1]])
    ends = np.r_[starts[1:] - 1, n - 1]
    keep = [starts, ends]
    try:
        yf = y.astype(float)
    except (TypeError, ValueError):
        return np.unique(np.concatenate(keep))
    nan = np.isnan(yf)
    # lexsort sorts by the LAST key first: bin, then value.  NaN is pushed to the
    # far end of each bin so it never wins min / max unless the bin is all-NaN.
    order_min = np.lexsort((np.where(nan, np.inf, yf), bins))
    order_max = np.lexsort((np.where(nan, -np.inf, yf), bins))
    keep.append(order_min[starts])   # smallest value of each bin
    keep.append(order_max[ends])     # largest value of each bin
    if nan.any():
        nan_idx = idx[nan]
        _, first = np.unique(bins[nan_idx], return_index=True)
        keep.append(nan_idx[first])  # keeps line gaps visible
    return np.unique(np.concatenate(keep))


def _subsample_trace(tr: Any, keep: np.ndarray, n: int) -> None:
    """Subsample every per-point attribute of *tr* (in place) to *keep*."""
    for attr in ("x", "y") + _POINT_ATTRS:
        arr = _as_array(getattr(tr, attr, None))
        if arr is not None and arr.shape[0] == n:
            setattr(tr, attr, arr[keep])
    marker = getattr(tr, "marker", None)
    if marker is not None:
        for attr in _MARKER_POINT_ATTRS:
            arr = _as_array(getattr(marker, attr, None))
            if arr is not None and arr.shape[0] == n:
                setattr(marker, attr, arr[keep])


def thin_for_display(fig: go.Figure, max_points: int | None = 5000) -> go.Figure:
    """Return a thinned copy of *fig* for inline notebook display.

    Every ``scatter`` / ``scattergl`` trace with more than *max_points* points
    is reduced to a per-bin min/max envelope.  Traces with identical x (same
    length, first and last value) form a group and keep the **union** of their
    envelope indices, so they stay aligned (stacked ``fill="tonexty"``,
    observed vs. model on one axis).  Bars, heatmaps, parcoords and short
    traces are copied unchanged, as are layout, shapes and annotations.

    :param fig: Figure to thin.  It is **not** modified.
    :type fig: plotly.graph_objects.Figure
    :param max_points: Target ceiling on points per thinned trace.  The bin
        count is chosen so the union over a group stays at or below it (a few
        more when NaN gaps are kept).  ``None`` disables thinning and returns
        an unthinned copy.
    :type max_points: int or None
    :returns: A new figure.
    :rtype: plotly.graph_objects.Figure
    """
    if max_points is None:
        return go.Figure(fig)
    if max_points < 4:
        raise ValueError(f"max_points must be >= 4, got {max_points!r}")
    out = go.Figure(fig)

    groups: dict[tuple, list[tuple[Any, np.ndarray]]] = {}
    for tr in out.data:
        if tr.type not in THIN_TRACE_TYPES:
            continue
        y = _as_array(tr.y)
        if y is None or y.size <= max_points:
            continue
        x = _as_array(tr.x)
        if x is not None and x.shape[0] != y.size:
            continue
        groups.setdefault(_x_key(x, y.size), []).append((tr, y))

    for (_, n, *_rest), members in groups.items():
        # Each bin contributes its shared first/last point plus at most a min
        # and a max per trace.
        n_bins = max(1, max_points // (2 * len(members) + 2))
        bins = (np.arange(n) * n_bins) // n
        keep = np.unique(np.concatenate([_envelope_indices(y, bins) for _, y in members]))
        for tr, _ in members:
            _subsample_trace(tr, keep, n)
    return out


def add_val_vrects(fig: go.Figure, t: Any, mask: Any, **vrect_kw: Any) -> go.Figure:
    """Shade every contiguous ``True`` run of *mask* on a plotly figure.

    One shaded tail in continuous-split mode; the interleaved blocks in
    stratified mode.  Default styling is a faint orange band below the traces.

    :param fig: Figure to annotate (modified in place and returned).
    :type fig: plotly.graph_objects.Figure
    :param t: Timestamps, same length as *mask*.
    :type t: array-like
    :param mask: Validation mask (e.g. ``SplitMasks.val``).
    :type mask: array-like of bool
    :param vrect_kw: Passed to ``fig.add_vrect`` (e.g. ``row=2, col=1``) and
        override the default ``fillcolor`` / ``line_width`` / ``layer``.
    :returns: *fig*.
    :rtype: plotly.graph_objects.Figure
    """
    m = np.asarray(mask, dtype=bool)
    t = pd.Series(pd.to_datetime(t)).reset_index(drop=True)
    if len(t) != m.size:
        raise ValueError(f"t has {len(t):,} values, mask {m.size:,}")
    style = {"fillcolor": VAL_FILL, "line_width": 0, "layer": "below", **vrect_kw}
    edges = np.diff(np.concatenate(([0], m.astype(int), [0])))
    for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1):
        fig.add_vrect(x0=t.iloc[s], x1=t.iloc[e], **style)
    return fig


__all__ = ["THIN_TRACE_TYPES", "VAL_FILL", "thin_for_display", "add_val_vrects"]


def use_webgl(fig: go.Figure) -> go.Figure:
    """Convert scatter traces and animation frames to WebGL in place."""
    payload = fig.to_plotly_json()
    trace_groups = [payload["data"]]
    trace_groups.extend(frame.get("data", []) for frame in payload.get("frames", []))
    for traces in trace_groups:
        for trace in traces:
            if trace["type"] == "scatter":
                trace["type"] = "scattergl"
    converted = go.Figure(payload)
    fig.data = ()
    fig.add_traces(converted.data)
    fig.frames = converted.frames
    return fig
