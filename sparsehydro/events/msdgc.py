"""sparsehydro.events.msdgc — Faithful port of the MSDGC storm-event detector.

This module reproduces the "Generate Storm Events" algorithm from the
*Observed vs. Model Flow* tool of the MSDGC *Model Application Tools* software
(``frmHdrSegr.findEvents``), decompiled from the shipped .NET assembly.  It is
deliberately **self-contained**: it imports nothing from the rest of
:mod:`sparsehydro.events` and shares no state with
:func:`sparsehydro.events.detect_events`.  The two detectors can therefore run
side by side on the same data with zero cross-contamination, which is the whole
point — you feed each its own clean frame and compare the outputs.

The algorithm, in brief
------------------------
1. **Seed candidates.**  If rain data is present, candidate timesteps are those
   with ``rain > 0``.  If rain is absent, candidates are timesteps whose storm
   flow is ``>= threshold_pct%`` of the global peak storm flow.
2. **Group into spans.**  Consecutive candidates are split into separate events
   wherever the gap between them exceeds ``interevent_hours``.
3. **Extend the end.**  From each span's end, walk forward while tracking a
   *running* peak storm flow; the event ends the first timestep the storm flow
   falls to ``<= threshold_pct%`` of that running peak.
4. **Validate.**  An event survives if its storm flow is non-zero and (rain
   mode) the maximum rain accumulated over any ``sliding_hours`` window within
   the event reaches ``min_rain_in``.
5. **Merge overlaps**, then drop events shorter than ``min_duration_hours``.

Shipped configuration defaults (``MSDGC Model Application Tools.exe.config``)::

    gStormEvent_EvtThreshold    = 5     -> threshold_pct
    gStormEvent_EvtTimeGap      = 6     -> interevent_hours
    gStormEvent_EvtSlidingHr    = 12    -> sliding_hours
    gStormEvent_EvtTotalRain    = 0.2   -> min_rain_in
    gStormEvent_MinEvtDuration  = 3     -> min_duration_hours

Quick start::

    from sparsehydro.events.msdgc import find_events, compare_observed_vs_model

    events = find_events(win)                    # win: datetime / rain / stormflow
    report = compare_observed_vs_model(events, win)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Conversion for reporting cfs·hr accumulations as million gallons (MG).
_CF_PER_GALLON = 7.480519
_SECONDS_PER_HOUR = 3600.0


@dataclass
class MsdgcEvent:
    """A single storm event detected by the MSDGC algorithm.

    :ivar event_id: Sequential identifier starting at 1 (assigned after
        invalid events are dropped).
    :vartype event_id: int
    :ivar start_datetime: Event start timestamp.
    :vartype start_datetime: pandas.Timestamp
    :ivar end_datetime: Event end timestamp (running-peak recession crossing).
    :vartype end_datetime: pandas.Timestamp
    :ivar peak_datetime: Timestamp of the maximum storm flow in the window.
    :vartype peak_datetime: pandas.Timestamp
    :ivar start_idx: Row index of *start_datetime* in the source frame.
    :vartype start_idx: int
    :ivar end_idx: Row index of *end_datetime* in the source frame.
    :vartype end_idx: int
    :ivar duration_hours: ``end_datetime - start_datetime`` in hours.
    :vartype duration_hours: float
    :ivar total_rain: Cumulative rain depth over ``[start, end]``.
    :vartype total_rain: float
    :ivar peak_stormflow: Maximum storm flow value over ``[start, end]``.
    :vartype peak_stormflow: float
    """

    event_id: int
    start_datetime: pd.Timestamp
    end_datetime: pd.Timestamp
    peak_datetime: pd.Timestamp
    start_idx: int
    end_idx: int
    duration_hours: float
    total_rain: float
    peak_stormflow: float

    def to_dict(self) -> dict:
        """Serialize to a plain dict.

        :returns: Mapping of every event field to its value.
        :rtype: dict
        """
        return {
            "event_id": self.event_id,
            "start_datetime": self.start_datetime,
            "end_datetime": self.end_datetime,
            "peak_datetime": self.peak_datetime,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "duration_hours": self.duration_hours,
            "total_rain": self.total_rain,
            "peak_stormflow": self.peak_stormflow,
        }


@dataclass
class _WorkEvent:
    """Mutable scratch event used during detection (not part of the public API)."""

    event_id: int
    start_idx: int
    end_idx: int = -1
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    duration: float = 0.0
    valid: bool = True


def events_to_dataframe(events: list[MsdgcEvent]) -> pd.DataFrame:
    """Convert a list of :class:`MsdgcEvent` objects to a DataFrame.

    :param events: Events to convert (may be empty).
    :type events: list[MsdgcEvent]
    :returns: One row per event with all event fields as columns.
    :rtype: pandas.DataFrame
    """
    columns = [
        "event_id", "start_datetime", "end_datetime", "peak_datetime",
        "start_idx", "end_idx", "duration_hours", "total_rain", "peak_stormflow",
    ]
    if not events:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame([e.to_dict() for e in events], columns=columns)


def _sliding_time_window_max(
    times: list[pd.Timestamp], values: np.ndarray, hrs_window: float
) -> float:
    """Return the maximum rain accumulated over any window ``<= hrs_window``.

    Mirrors ``clsUtlts.slidingTimeWindowMax`` exactly, including its quirk that
    the final sample is never part of any window sum (the loop runs ``j`` over
    ``1..n-1`` and sums the half-open slice ``values[i:j]``).

    :param times: Per-sample timestamps, ascending.
    :type times: list[pandas.Timestamp]
    :param values: Per-sample rain depths aligned with *times*.
    :type values: numpy.ndarray
    :param hrs_window: Sliding window length in hours.
    :type hrs_window: float
    :returns: Largest windowed rain sum found (``0.0`` if fewer than 2 samples).
    :rtype: float
    """
    max_sum = 0.0
    i = 0
    for j in range(1, len(times)):
        while (times[j] - times[i]).total_seconds() / _SECONDS_PER_HOUR > hrs_window:
            i += 1
        s = float(np.sum(values[i:j]))  # Skip(i).Take(j - i) == values[i:j]
        if s > max_sum:
            max_sum = s
    return max_sum


def find_events(
    df: pd.DataFrame,
    *,
    datetime_col: str = "datetime",
    rain_col: str = "rain",
    stormflow_col: str = "stormflow",
    threshold_pct: float = 5.0,
    interevent_hours: float = 6.0,
    sliding_hours: float = 12.0,
    min_rain_in: float = 0.2,
    min_duration_hours: float = 3.0,
    use_rain: bool | None = None,
    close_trailing_event: bool = False,
) -> list[MsdgcEvent]:
    """Detect storm events exactly the way the MSDGC tool's ``findEvents`` does.

    The input frame must carry a datetime column, a per-step rain depth column,
    and a base-flow-removed storm-flow column, all on the **same** row axis
    (unlike the original tool, which keeps rain and flow as separate series and
    index-matches between them — sharing one axis makes that bookkeeping moot).

    :param df: Source frame with *datetime_col*, *rain_col*, *stormflow_col*.
    :type df: pandas.DataFrame
    :param datetime_col: Name of the timestamp column.
    :type datetime_col: str
    :param rain_col: Name of the per-step rain-depth column (e.g. inches/step).
    :type rain_col: str
    :param stormflow_col: Name of the base-flow-removed storm-flow column.
    :type stormflow_col: str
    :param threshold_pct: Recession/seed threshold as a percent of running peak
        storm flow (``gStormEvent_EvtThreshold``, default 5).
    :type threshold_pct: float
    :param interevent_hours: Split candidates into separate events when the gap
        between them exceeds this (``gStormEvent_EvtTimeGap``, default 6).
    :type interevent_hours: float
    :param sliding_hours: Rain-accumulation window used for validation
        (``gStormEvent_EvtSlidingHr``, default 12).
    :type sliding_hours: float
    :param min_rain_in: Minimum rain over the sliding window for an event to be
        valid (``gStormEvent_EvtTotalRain``, default 0.2).
    :type min_rain_in: float
    :param min_duration_hours: Drop events shorter than this
        (``gStormEvent_MinEvtDuration``, default 3).
    :type min_duration_hours: float
    :param use_rain: Force the rain-seeded branch (``True``) or the flow-seeded
        branch (``False``).  ``None`` auto-selects: rain-seeded when any rain is
        present, else flow-seeded (matching ``clsHydSeg1.isRainDataEmpty``).
    :type use_rain: bool | None
    :param close_trailing_event: When ``False`` (default) reproduce the tool
        bug-for-bug — the final open candidate span is *not* emitted because the
        original only closes a span when it sees the next gap.  Set ``True`` to
        also emit that trailing span.
    :type close_trailing_event: bool
    :returns: Valid detected events, renumbered from 1, in chronological order.
    :rtype: list[MsdgcEvent]
    """
    d = df.reset_index(drop=True)
    times = pd.to_datetime(d[datetime_col]).to_list()
    storm = d[stormflow_col].to_numpy(dtype=float)
    rain = d[rain_col].to_numpy(dtype=float)
    n = len(d)
    if n == 0:
        return []

    if use_rain is None:
        use_rain = float(np.nansum(rain)) > 0.0

    pct = threshold_pct / 100.0

    # 1) Candidate indices -------------------------------------------------
    if use_rain:
        candidates = [i for i in range(n) if rain[i] > 0.0]
    else:
        global_peak = float(np.nanmax(storm))
        seed_threshold = pct * global_peak
        candidates = [i for i in range(n) if storm[i] >= seed_threshold]
    if not candidates:
        return []

    # 2) Split candidates into spans on gaps > interevent_hours ------------
    spans: list[tuple[int, int]] = []
    group_start = 0
    for m in range(1, len(candidates)):
        gap = (times[candidates[m]] - times[candidates[m - 1]]).total_seconds() / _SECONDS_PER_HOUR
        if gap > interevent_hours:
            spans.append((candidates[group_start], candidates[m - 1]))
            group_start = m
    if close_trailing_event:  # the original tool omits this final span
        spans.append((candidates[group_start], candidates[-1]))

    # 3) Build events; extend END on the running-peak recession rule -------
    events: list[_WorkEvent] = []
    next_id = 1
    for s_idx, e_idx in spans:
        # Flow-seeded spans begin one hour before the first candidate; the
        # rain-seeded branch begins right on it (per findEvents' two branches).
        if use_rain:
            start_idx = s_idx
        else:
            start_idx = max(0, s_idx - 1)
        if times[start_idx] < times[0]:
            continue

        ev = _WorkEvent(event_id=next_id, start_idx=start_idx, start=times[start_idx])
        peak = float(np.max(storm[s_idx:e_idx + 1])) if e_idx >= s_idx else storm[s_idx]
        for j in range(e_idx, n):
            peak = max(peak, storm[j])
            if storm[j] <= pct * peak:
                ev.end_idx = j
                ev.end = times[j]
                ev.duration = round(
                    (times[j] - ev.start).total_seconds() / _SECONDS_PER_HOUR, 2
                )
                break
        if ev.end is None:
            # Recession never crossed before the series ended: the tool leaves
            # the end unset (duration 0) so the min-duration filter drops it.
            ev.valid = False
            ev.end_idx = n - 1
            ev.end = times[n - 1]
        events.append(ev)
        next_id += 1

    # 4) Validity: non-zero storm flow + enough rain in the sliding window -
    for ev in events:
        if not ev.valid or ev.start_idx < 0 or ev.end_idx < 0:
            ev.valid = False
            continue
        seg = storm[ev.start_idx:ev.end_idx]  # half-open [start, end), per the exe
        if seg.size and float(np.max(seg)) <= 0.0:
            ev.valid = False
            continue
        if use_rain:
            r_times = times[ev.start_idx:ev.end_idx]
            r_vals = rain[ev.start_idx:ev.end_idx]
            if _sliding_time_window_max(r_times, r_vals, sliding_hours) < min_rain_in:
                ev.valid = False

    # 5) Merge overlapping events -----------------------------------------
    for a in range(len(events) - 1):
        if not events[a].valid:
            continue
        for b in range(a + 1, len(events)):
            if not events[b].valid:
                continue
            if not (events[a].end >= events[b].start):
                break
            events[b].valid = False
            if events[a].end < events[b].end:
                events[a].end = events[b].end
                events[a].end_idx = events[b].end_idx
                events[a].duration = round(
                    (events[a].end - events[a].start).total_seconds() / _SECONDS_PER_HOUR, 2
                )

    # 6) Min-duration filter, keep valid, renumber ------------------------
    for ev in events:
        if ev.duration < min_duration_hours:
            ev.valid = False
    kept = [ev for ev in events if ev.valid]

    result: list[MsdgcEvent] = []
    for new_id, ev in enumerate(kept, start=1):
        window = slice(ev.start_idx, ev.end_idx + 1)
        seg_storm = storm[window]
        peak_offset = int(np.argmax(seg_storm)) if seg_storm.size else 0
        result.append(
            MsdgcEvent(
                event_id=new_id,
                start_datetime=pd.Timestamp(ev.start),
                end_datetime=pd.Timestamp(ev.end),
                peak_datetime=pd.Timestamp(times[ev.start_idx + peak_offset]),
                start_idx=ev.start_idx,
                end_idx=ev.end_idx,
                duration_hours=ev.duration,
                total_rain=float(np.sum(rain[window])),
                peak_stormflow=float(np.max(seg_storm)) if seg_storm.size else 0.0,
            )
        )
    return result


def _accumulation_to_mg(flow_cfs_sum: float, step_hours: float) -> float:
    """Convert a summed cfs series to million gallons.

    :param flow_cfs_sum: Sum of a flow series in cubic-feet-per-second.
    :type flow_cfs_sum: float
    :param step_hours: Timestep length in hours.
    :type step_hours: float
    :returns: Volume in million gallons.
    :rtype: float
    """
    cubic_feet = flow_cfs_sum * step_hours * _SECONDS_PER_HOUR
    return cubic_feet * _CF_PER_GALLON / 1e6


def compare_observed_vs_model(
    events: list[MsdgcEvent],
    df: pd.DataFrame,
    *,
    datetime_col: str = "datetime",
    obs_col: str = "stormflow",
    model_col: str = "model_stormflow",
    rain_col: str = "rain",
) -> pd.DataFrame:
    """Score observed vs. modeled flow within each detected event window.

    Reproduces the columns of the tool's *Observed vs. Model Flow* event report:
    per-event volumes (in MG, assuming cfs flow columns), peak flows, total
    rain, and percent errors of model relative to observed.

    :param events: Events from :func:`find_events`.
    :type events: list[MsdgcEvent]
    :param df: Frame the events were detected on, plus a modeled-flow column.
    :type df: pandas.DataFrame
    :param datetime_col: Name of the timestamp column.
    :type datetime_col: str
    :param obs_col: Observed flow column (cfs) to integrate/peak.
    :type obs_col: str
    :param model_col: Modeled flow column (cfs) to integrate/peak.
    :type model_col: str
    :param rain_col: Per-step rain-depth column.
    :type rain_col: str
    :returns: One row per event with observed/model volumes, peaks, and errors.
    :rtype: pandas.DataFrame
    """
    columns = [
        "Event ID", "Start Date", "End Date", "Duration (hr)", "Total Rain (in)",
        "Obs Event Volume (MG)", "Model Event Volume (MG)", "Volume % Error",
        "Peak Obs Flow (cfs)", "Peak Model Flow (cfs)", "Peak % Error",
    ]
    if not events:
        return pd.DataFrame(columns=columns)

    d = df.reset_index(drop=True)
    t = pd.to_datetime(d[datetime_col])
    if len(t) > 1:
        step_hours = float(np.median(np.diff(t.values)) / np.timedelta64(1, "h"))
    else:
        step_hours = 1.0
    obs = d[obs_col].to_numpy(dtype=float)
    mdl = d[model_col].to_numpy(dtype=float)
    rain = d[rain_col].to_numpy(dtype=float)

    rows = []
    for e in events:
        window = slice(e.start_idx, e.end_idx + 1)
        obs_vol = _accumulation_to_mg(float(np.sum(obs[window])), step_hours)
        mdl_vol = _accumulation_to_mg(float(np.sum(mdl[window])), step_hours)
        obs_peak = float(np.max(obs[window]))
        mdl_peak = float(np.max(mdl[window]))
        rows.append({
            "Event ID": e.event_id,
            "Start Date": e.start_datetime,
            "End Date": e.end_datetime,
            "Duration (hr)": e.duration_hours,
            "Total Rain (in)": round(float(np.sum(rain[window])), 3),
            "Obs Event Volume (MG)": round(obs_vol, 4),
            "Model Event Volume (MG)": round(mdl_vol, 4),
            "Volume % Error": round(100.0 * (mdl_vol - obs_vol) / obs_vol, 1) if obs_vol else np.nan,
            "Peak Obs Flow (cfs)": round(obs_peak, 3),
            "Peak Model Flow (cfs)": round(mdl_peak, 3),
            "Peak % Error": round(100.0 * (mdl_peak - obs_peak) / obs_peak, 1) if obs_peak else np.nan,
        })
    return pd.DataFrame(rows, columns=columns)


__all__ = [
    "MsdgcEvent",
    "find_events",
    "events_to_dataframe",
    "compare_observed_vs_model",
]
