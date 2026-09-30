"""Streamlit-free helpers for the SparseHydro viewer.

Runs the parsimonious workflow -- :func:`~sparsehydro.events.detect_event_hierarchy`
followed by :class:`~sparsehydro.models.unithydrograph.GlobalSequentialFitter`
(one unit hydrograph per global event) -- and turns the result into the tables
and plot windows the app needs.  Nothing here imports Streamlit, so it can be
unit-tested on its own.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from ..events import (
    GlobalEvent,
    SubEventRecord,
    VariableSavgolResult,
    detect_event_hierarchy,
    sub_events_to_dataframe,
)
from ..events.hierarchy import GLOBAL_EVENT_GAP_DAYS
from ..models.base import IModel
from ..models.unithydrograph import GlobalSequentialFitter, SequentialFitSummary

REQUIRED_COLUMNS = ("datetime", "rain", "stormflow")

_SEASONS = {12: "Winter", 1: "Winter", 2: "Winter",
            3: "Spring", 4: "Spring", 5: "Spring",
            6: "Summer", 7: "Summer", 8: "Summer",
            9: "Fall", 10: "Fall", 11: "Fall"}

SEASON_ORDER = ["Winter", "Spring", "Summer", "Fall"]


@dataclass
class ViewerResult:
    """Everything the viewer displays for one dataset + settings.

    :ivar data: Input frame (``datetime``, ``rain``, ``stormflow``).
    :vartype data: pandas.DataFrame
    :ivar global_events: Detected global events (all, fitted or not).
    :vartype global_events: list[GlobalEvent]
    :ivar sub_events: Detected single-peak sub-events.
    :vartype sub_events: list[SubEventRecord]
    :ivar savgol: Variable-window smoothing result.
    :vartype savgol: VariableSavgolResult
    :ivar summary: Global sequential fit; ``summary.events`` are the fitted
        :class:`~sparsehydro.events.GlobalEvent` objects.
    :vartype summary: SequentialFitSummary
    :ivar dt_hours: Median time step of *data*, in hours.
    :vartype dt_hours: float
    """

    data: pd.DataFrame
    global_events: "list[GlobalEvent]"
    sub_events: "list[SubEventRecord]"
    savgol: VariableSavgolResult
    summary: SequentialFitSummary
    dt_hours: float


def read_rain_stormflow(source: "str | Path | bytes") -> pd.DataFrame:
    """Read a ``rain_stormflow.csv``-shaped file.

    :param source: A path to a CSV, or the raw bytes of one (e.g. from an upload).
    :type source: str | pathlib.Path | bytes
    :returns: DataFrame with parsed ``datetime`` plus ``rain`` and ``stormflow``.
    :rtype: pandas.DataFrame
    :raises ValueError: If a required column is missing.
    """
    df = pd.read_csv(io.BytesIO(source) if isinstance(source, bytes) else source)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s) {missing}; expected {list(REQUIRED_COLUMNS)}, "
            f"found {list(df.columns)}."
        )
    df = df[list(REQUIRED_COLUMNS)].copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


def content_hash(data: bytes) -> str:
    """Stable hash of file content, used as a cache key.

    :param data: File bytes.
    :type data: bytes
    :returns: Hex SHA-256 digest.
    :rtype: str
    """
    return hashlib.sha256(data).hexdigest()


def season_of(ts: pd.Timestamp) -> str:
    """Meteorological season name for a timestamp.

    :param ts: Timestamp.
    :type ts: pandas.Timestamp
    :returns: ``"Winter"``, ``"Spring"``, ``"Summer"`` or ``"Fall"``.
    :rtype: str
    """
    return _SEASONS[pd.Timestamp(ts).month]


def run_analysis(
    data: pd.DataFrame,
    model_factory: Callable[[], IModel],
    *,
    global_event_gap_days: float = GLOBAL_EVENT_GAP_DAYS,
    peak_weight: float = 3.0,
    tail_weight: float = 1.0,
) -> ViewerResult:
    """Detect the event hierarchy and fit one UH per global event.

    :param data: Frame with ``datetime``, ``rain``, ``stormflow``.
    :type data: pandas.DataFrame
    :param model_factory: Zero-argument callable returning a fresh UH model.
    :type model_factory: Callable[[], IModel]
    :param global_event_gap_days: Max peak separation within one global event.
    :type global_event_gap_days: float
    :param peak_weight: Fit weight inside sub-event peak zones.
    :type peak_weight: float
    :param tail_weight: Fit weight elsewhere inside sub-events.
    :type tail_weight: float
    :returns: Detection and fit results.
    :rtype: ViewerResult
    """
    df = data.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    global_events, sub_events, savgol = detect_event_hierarchy(
        df, global_event_gap_days=global_event_gap_days
    )
    summary = GlobalSequentialFitter(model_factory, df, global_events, sub_events).fit(
        peak_weight=peak_weight, tail_weight=tail_weight, verbose=False
    )
    dt_hours = float(df["datetime"].diff().median().total_seconds() / 3600.0) if len(df) > 1 else 1.0 / 12.0
    return ViewerResult(df, global_events, sub_events, savgol, summary, dt_hours)


def span_nse(result: ViewerResult, start: pd.Timestamp, end: pd.Timestamp) -> float:
    """Nash-Sutcliffe efficiency of the fitted prediction over ``[start, end]``.

    :param result: Viewer result.
    :type result: ViewerResult
    :param start: Window start.
    :type start: pandas.Timestamp
    :param end: Window end.
    :type end: pandas.Timestamp
    :returns: NSE, or NaN if observed flow is constant over the window.
    :rtype: float
    """
    obs = result.summary.global_observed
    pred = result.summary.global_predicted
    sel = (obs["datetime"] >= start) & (obs["datetime"] <= end)
    o = obs.loc[sel, "stormflow"].to_numpy(dtype=float)
    p = pred.loc[sel.to_numpy(), "Q_pred"].to_numpy(dtype=float)
    denom = float(np.sum((o - o.mean()) ** 2)) if o.size else 0.0
    return 1.0 - float(np.sum((o - p) ** 2)) / denom if denom > 0 else float("nan")


def global_event_table(result: ViewerResult) -> pd.DataFrame:
    """One row per global event, with fields for filtering and sorting.

    Columns: ``global_id``, ``start``, ``end``, ``window_start``,
    ``window_end``, ``peak``, ``n_sub``, ``fitted``, ``total_rain``,
    ``peak_flow``, ``nse`` (over the member span), ``season``.

    :param result: Viewer result.
    :type result: ViewerResult
    :returns: Global-event table in chronological order.
    :rtype: pandas.DataFrame
    """
    cols = ["global_id", "start", "end", "window_start", "window_end", "peak", "n_sub",
            "fitted", "total_rain", "peak_flow", "nse", "season"]
    if not result.global_events:
        return pd.DataFrame(columns=cols)
    fitted_ids = {g.global_id for g in result.summary.events}
    subs_by_global: dict[int, list[SubEventRecord]] = {}
    for s in result.sub_events:
        subs_by_global.setdefault(s.global_id, []).append(s)

    rows = []
    for g in sorted(result.global_events, key=lambda g: g.start_datetime):
        subs = subs_by_global.get(g.global_id, [])
        fitted = g.global_id in fitted_ids
        rows.append({
            "global_id": int(g.global_id),
            "start": pd.Timestamp(g.start_datetime),
            "end": pd.Timestamp(g.end_datetime),
            "window_start": pd.Timestamp(g.window_start_datetime),
            "window_end": pd.Timestamp(g.window_end_datetime),
            "peak": pd.Timestamp(g.peak_datetime),
            "n_sub": len(subs),
            "fitted": fitted,
            "total_rain": float(g.total_rain),
            "peak_flow": float(max((s.peak_flow for s in subs), default=np.nan)),
            "nse": span_nse(result, g.start_datetime, g.end_datetime) if fitted else np.nan,
            "season": season_of(g.start_datetime),
        })
    return pd.DataFrame(rows, columns=cols)


def filter_sort_events(table: pd.DataFrame, *, seasons: "list[str] | None" = None,
                       min_rain: float = 0.0, sort_by: str = "start",
                       descending: bool = False) -> pd.DataFrame:
    """Filter and sort a :func:`global_event_table`.

    :param table: Output of :func:`global_event_table`.
    :type table: pandas.DataFrame
    :param seasons: Keep only these seasons (``None`` or empty keeps all).
    :type seasons: list[str] | None
    :param min_rain: Minimum ``total_rain``.
    :type min_rain: float
    :param sort_by: Column to sort on.
    :type sort_by: str
    :param descending: Sort descending.
    :type descending: bool
    :returns: Filtered, sorted copy (NaNs sort last).
    :rtype: pandas.DataFrame
    """
    out = table
    if seasons:
        out = out[out["season"].isin(seasons)]
    out = out[out["total_rain"] >= min_rain]
    return out.sort_values(sort_by, ascending=not descending, na_position="last",
                           kind="stable").reset_index(drop=True)


def fit_index(result: ViewerResult, global_id: int) -> "int | None":
    """Position of a global event in ``result.summary.events`` (``None`` if not fitted).

    :param result: Viewer result.
    :type result: ViewerResult
    :param global_id: Global event identifier.
    :type global_id: int
    :returns: Index into the fit summary, or ``None``.
    :rtype: int | None
    """
    return next((i for i, g in enumerate(result.summary.events) if g.global_id == global_id), None)


def fitted_model(summary: SequentialFitSummary, index: int, model_factory: Callable[[], IModel]) -> IModel:
    """Rebuild the fitted model for one event of a fit summary.

    The summary stores parameter vectors rather than model objects; this
    applies ``calibration_results[index].pareto_X[0]`` to a fresh model so its
    kernel can be plotted.

    :param summary: Fit summary.
    :type summary: SequentialFitSummary
    :param index: Position of the event in ``summary.events``.
    :type index: int
    :param model_factory: The factory used for fitting.
    :type model_factory: Callable[[], IModel]
    :returns: Initialized, validated model carrying the fitted parameters.
    :rtype: IModel
    """
    cal = summary.calibration_results[index]
    model = model_factory()
    model.initialize()
    for name, val in zip(cal.param_names, cal.pareto_X[0]):
        model.apply_flat_parameter(name, float(val))
    model.validate()
    return model


def slice_savgol(savgol: VariableSavgolResult, start: pd.Timestamp, end: pd.Timestamp) -> VariableSavgolResult:
    """Restrict a smoothing result to ``[start, end]`` (peak indices re-based).

    :param savgol: Full-record smoothing result.
    :type savgol: VariableSavgolResult
    :param start: Window start.
    :type start: pandas.Timestamp
    :param end: Window end.
    :type end: pandas.Timestamp
    :returns: A new result covering only the window.
    :rtype: VariableSavgolResult
    """
    t = np.asarray(savgol.datetime, dtype="datetime64[ns]")
    i0 = int(np.searchsorted(t, np.datetime64(pd.Timestamp(start))))
    i1 = int(np.searchsorted(t, np.datetime64(pd.Timestamp(end)), side="right"))
    peaks = np.asarray(savgol.peak_idxs, dtype=int)
    peaks = peaks[(peaks >= i0) & (peaks < i1)] - i0
    return replace(
        savgol,
        datetime=savgol.datetime[i0:i1], raw_flow=savgol.raw_flow[i0:i1],
        smoothed=savgol.smoothed[i0:i1], windows=savgol.windows[i0:i1],
        seed_curvature=savgol.seed_curvature[i0:i1], curvature=savgol.curvature[i0:i1],
        peak_idxs=peaks,
    )


def events_in_window(events: list, start: pd.Timestamp, end: pd.Timestamp) -> list:
    """Events whose ``[start_datetime, end_datetime]`` overlaps ``[start, end]``.

    :param events: Global or sub-events.
    :type events: list
    :param start: Window start.
    :type start: pandas.Timestamp
    :param end: Window end.
    :type end: pandas.Timestamp
    :returns: Overlapping events.
    :rtype: list
    """
    return [e for e in events if e.end_datetime >= start and e.start_datetime <= end]


def member_table(result: ViewerResult, global_id: int) -> pd.DataFrame:
    """Sub-event details for one global event.

    :param result: Viewer result.
    :type result: ViewerResult
    :param global_id: Global event identifier.
    :type global_id: int
    :returns: One row per member sub-event.
    :rtype: pandas.DataFrame
    """
    return sub_events_to_dataframe([s for s in result.sub_events if s.global_id == global_id])


def fit_details(result: ViewerResult, global_id: int) -> pd.DataFrame:
    """Fitted parameters and objectives for one global event (one row; empty if unfitted).

    :param result: Viewer result.
    :type result: ViewerResult
    :param global_id: Global event identifier.
    :type global_id: int
    :returns: One-row frame of objectives then parameters.
    :rtype: pandas.DataFrame
    """
    summary = result.summary
    if fit_index(result, global_id) is None:
        return pd.DataFrame()
    metrics = summary.metrics_summary()
    params = summary.parameter_evolution().drop(columns=["start_datetime", "end_datetime"], errors="ignore")
    out = metrics.merge(params, on="event_id", how="left")
    return out[out["event_id"] == global_id].rename(columns={"event_id": "global_id"}).reset_index(drop=True)


def kernel_display_steps(kernels: "list[np.ndarray]", mass: float = 0.995, min_steps: int = 12) -> int:
    """Number of kernel steps to plot so every kernel shows ``mass`` of its area.

    :param kernels: UH ordinate arrays.
    :type kernels: list[numpy.ndarray]
    :param mass: Fraction of each kernel's area that must be visible.
    :type mass: float
    :param min_steps: Lower bound on the returned length.
    :type min_steps: int
    :returns: Step count covering ``mass`` of the slowest kernel.
    :rtype: int
    """
    steps = min_steps
    for k in kernels:
        total = float(np.sum(k))
        if total > 0:
            steps = max(steps, int(np.searchsorted(np.cumsum(k) / total, mass)) + 1)
    return steps


__all__ = [
    "REQUIRED_COLUMNS",
    "SEASON_ORDER",
    "ViewerResult",
    "content_hash",
    "events_in_window",
    "filter_sort_events",
    "fit_details",
    "fit_index",
    "fitted_model",
    "global_event_table",
    "kernel_display_steps",
    "member_table",
    "read_rain_stormflow",
    "run_analysis",
    "season_of",
    "slice_savgol",
    "span_nse",
]
