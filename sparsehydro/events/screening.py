"""sparsehydro.events.screening — MSDGC global wet-weather-flow event rules.

Layers the MSDGC *Modeling Guidelines* Vol. II §6.2 "Global Event Definition"
procedure on top of the derivative-based detector in
:mod:`sparsehydro.events`.  The detector answers *where are the storm events*;
this module answers *which of them qualify and how far should each one extend*.

Implemented rules (rain/flow-derivable only — GARR, winter-conditions and
no-data screening are intentionally out of scope and left as manual review):

- **6-hour interevent period** — merge events separated by less than the
  interevent gap (:func:`merge_by_interevent`) and split any single event that
  straddles an interior dry gap of at least the interevent period
  (:func:`split_by_interevent`), so each event tracks one rain cluster.
- **Duration refinement** — start 0.25 hr before the first recorded rainfall;
  end two-thirds of the way from the end of rainfall to the return of the
  system to base flow, capped at 48 hr past the end of rainfall and never
  overlapping the following event (:func:`refine_event_durations`).
- **Top-10 peak split** — split any event that contains two or more of the ten
  largest flow peaks when a meaningful trough separates them
  (:func:`split_top_peak_events`).
- **Small-rainfall screening** — flag events with less than 0.25 in of rainfall
  as excluded "small rainfall" and the remainder as *qualifying*
  (:func:`screen_small_rainfall`).
- **Flow-response check** — flag events with no discernible flow rise following
  the rainfall (:func:`flag_flow_response`).

The orchestrator :func:`build_global_events` runs the full pipeline and returns
the qualifying :class:`~sparsehydro.events.EventRecord` objects together with a
``summary`` DataFrame categorising every candidate event.

Because the categorisation is attached to each :class:`EventRecord` as the
attributes ``qualifying``, ``exclusion_reason`` and ``flow_response`` (set at
runtime), this module is fully usable without modifying ``EventRecord`` itself.

Quick start::

    from sparsehydro.events.screening import build_global_events

    global_events, summary = build_global_events(win, savgol, events=global_events)
    plot_event_detection(savgol, global_events).show()
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from ..filters import FilterResult
from . import EventRecord, detect_events, events_to_dataframe

# Categorisation labels attached to every candidate EventRecord.
_QUALIFYING = "qualifying"
_SMALL_RAINFALL = "small rainfall"
_SHORT_DURATION = "short duration"
_NO_FLOW_RESPONSE = "no flow response"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _hours_per_sample(df: pd.DataFrame) -> float:
    """Return the median sampling interval of *df* in hours.

    :param df: Frame with a ``datetime`` column.
    :type df: pandas.DataFrame
    :returns: Median timestep in hours (falls back to ``5 min`` if undefined).
    :rtype: float
    """
    dt = pd.to_datetime(df["datetime"])
    diffs = dt.diff().dropna().dt.total_seconds().to_numpy()
    if len(diffs) == 0:
        return 5.0 / 60.0
    return float(np.median(diffs)) / 3600.0


def _samples(hours: float, hours_per_sample: float) -> int:
    """Convert a duration in hours to a whole number of samples (minimum 1).

    :param hours: Duration in hours.
    :type hours: float
    :param hours_per_sample: Sampling interval in hours.
    :type hours_per_sample: float
    :returns: Number of samples spanning *hours* (>= 1).
    :rtype: int
    """
    return max(1, int(round(hours / hours_per_sample)))


def _times(df: pd.DataFrame) -> np.ndarray:
    """Return the ``datetime`` column as a sorted ``datetime64[ns]`` array.

    :param df: Frame with a ``datetime`` column.
    :type df: pandas.DataFrame
    :returns: Timestamps as ``numpy.datetime64[ns]``.
    :rtype: numpy.ndarray
    """
    return pd.to_datetime(df["datetime"]).to_numpy().astype("datetime64[ns]")


def _idx_of(t: np.ndarray, when: pd.Timestamp) -> int:
    """Return the index in *t* nearest to timestamp *when*.

    :param t: Sorted ``datetime64[ns]`` array.
    :type t: numpy.ndarray
    :param when: Target timestamp.
    :type when: pandas.Timestamp
    :returns: Clamped index into *t*.
    :rtype: int
    """
    i = int(np.searchsorted(t, np.datetime64(pd.Timestamp(when), "ns")))
    i = max(0, min(i, len(t) - 1))
    if i > 0:
        prev = t[i - 1]
        cur = t[i]
        if abs(np.datetime64(pd.Timestamp(when), "ns") - prev) < abs(cur - np.datetime64(pd.Timestamp(when), "ns")):
            return i - 1
    return i


def _record_from_span(
    event_id: int,
    i0: int,
    i1: int,
    df: pd.DataFrame,
    peak_idx: int | None = None,
    *,
    b2b_start: bool = False,
    b2b_end: bool = False,
) -> EventRecord:
    """Build an :class:`EventRecord` from an index span, recomputing totals.

    :param event_id: Identifier to assign.
    :type event_id: int
    :param i0: Inclusive start index into *df*.
    :type i0: int
    :param i1: Inclusive end index into *df*.
    :type i1: int
    :param df: Source ``datetime``/``rain``/``stormflow`` frame.
    :type df: pandas.DataFrame
    :param peak_idx: Optional peak index; recomputed from the span when ``None``.
    :type peak_idx: int | None
    :param b2b_start: Back-to-back start flag to preserve.
    :type b2b_start: bool
    :param b2b_end: Back-to-back end flag to preserve.
    :type b2b_end: bool
    :returns: Event with rain/flow totals summed over ``[i0, i1]``.
    :rtype: EventRecord
    """
    i0 = max(0, min(i0, len(df) - 1))
    i1 = max(i0, min(i1, len(df) - 1))
    sl = df.iloc[i0: i1 + 1]
    flow = np.maximum(sl["stormflow"].to_numpy(dtype=float), 0.0)
    total_rain = float(sl["rain"].sum())
    total_flow = float(flow.sum())
    ae = total_flow / total_rain if total_rain > 0 else 0.0

    if peak_idx is None or not (i0 <= peak_idx <= i1):
        local = int(np.argmax(flow)) if len(flow) else 0
        peak_idx = i0 + local
    peak_flow = float(np.maximum(df["stormflow"].iloc[peak_idx], 0.0))

    return EventRecord(
        event_id=event_id,
        start_datetime=pd.Timestamp(df["datetime"].iloc[i0]),
        end_datetime=pd.Timestamp(df["datetime"].iloc[i1]),
        peak_datetime=pd.Timestamp(df["datetime"].iloc[peak_idx]),
        total_rain=total_rain,
        total_flow=total_flow,
        effective_area=ae,
        peak_flow=peak_flow,
        b2b_start=b2b_start,
        b2b_end=b2b_end,
    )


def _first_last_rain(df: pd.DataFrame, i0: int, i1: int) -> tuple[int | None, int | None]:
    """Return the first and last indices with positive rain in ``[i0, i1]``.

    :param df: Source frame with a ``rain`` column.
    :type df: pandas.DataFrame
    :param i0: Inclusive window start index.
    :type i0: int
    :param i1: Inclusive window end index.
    :type i1: int
    :returns: ``(first_rain_idx, last_rain_idx)`` or ``(None, None)`` if the
        window contains no rainfall.
    :rtype: tuple[int | None, int | None]
    """
    rain = df["rain"].to_numpy(dtype=float)[i0: i1 + 1]
    wet = np.flatnonzero(rain > 0.0)
    if len(wet) == 0:
        return None, None
    return i0 + int(wet[0]), i0 + int(wet[-1])


def _renumber(events: list[EventRecord]) -> list[EventRecord]:
    """Sort events by start time and reassign ``event_id`` from 1.

    :param events: Events to order.
    :type events: list[EventRecord]
    :returns: The same list objects, sorted and renumbered in place.
    :rtype: list[EventRecord]
    """
    events.sort(key=lambda e: e.start_datetime)
    for i, e in enumerate(events, start=1):
        e.event_id = i
    return events


# ---------------------------------------------------------------------------
# Public API — individual rules
# ---------------------------------------------------------------------------

def merge_by_interevent(
    events: list[EventRecord],
    rain_stormflow: pd.DataFrame,
    interevent_hours: float = 6.0,
) -> list[EventRecord]:
    """Merge events separated by less than the interevent period.

    Implements the MSDGC global 6-hour interevent period (the Auto-Select
    Events / Generate Storm Events analog): consecutive events whose dry gap is
    shorter than *interevent_hours* are combined into one.

    :param events: Detected events (any order).
    :type events: list[EventRecord]
    :param rain_stormflow: Source ``datetime``/``rain``/``stormflow`` frame.
    :type rain_stormflow: pandas.DataFrame
    :param interevent_hours: Minimum dry gap between distinct events, in hours.
    :type interevent_hours: float
    :returns: Merged events, renumbered from 1.
    :rtype: list[EventRecord]
    """
    if not events:
        return []
    df = rain_stormflow
    t = _times(df)
    gap = pd.Timedelta(hours=interevent_hours)

    ordered = sorted(events, key=lambda e: e.start_datetime)
    merged: list[EventRecord] = []
    cur_i0 = _idx_of(t, ordered[0].start_datetime)
    cur_i1 = _idx_of(t, ordered[0].end_datetime)
    cur_peak = _idx_of(t, ordered[0].peak_datetime)

    for e in ordered[1:]:
        e_i0 = _idx_of(t, e.start_datetime)
        e_i1 = _idx_of(t, e.end_datetime)
        e_peak = _idx_of(t, e.peak_datetime)
        if pd.Timestamp(df["datetime"].iloc[e_i0]) - pd.Timestamp(df["datetime"].iloc[cur_i1]) < gap:
            cur_i1 = max(cur_i1, e_i1)
            if float(df["stormflow"].iloc[e_peak]) > float(df["stormflow"].iloc[cur_peak]):
                cur_peak = e_peak
        else:
            merged.append(_record_from_span(len(merged) + 1, cur_i0, cur_i1, df, cur_peak))
            cur_i0, cur_i1, cur_peak = e_i0, e_i1, e_peak

    merged.append(_record_from_span(len(merged) + 1, cur_i0, cur_i1, df, cur_peak))
    return _renumber(merged)


def split_by_interevent(
    events: list[EventRecord],
    rain_stormflow: pd.DataFrame,
    interevent_hours: float = 6.0,
) -> list[EventRecord]:
    """Split events that straddle an interior dry gap of >= the interevent period.

    The complement of :func:`merge_by_interevent`.  The derivative-based
    detector can chain a whole wet spell into one multi-week "event" whose
    interior contains long rain-free stretches.  MSDGC §6.2 defines a global
    event by its rainfall: a dry gap of at least *interevent_hours* between
    successive rain pulses starts a new event.  This function enforces that by
    cutting each event wherever two consecutive rain pulses are *interevent_hours*
    or more apart, so every resulting event tracks a single rain cluster.

    The recession tail between two clusters is assigned to the earlier event
    (its span ends just before the next cluster's first rain) and is then
    trimmed back by :func:`refine_event_durations`.  Events containing fewer
    than two rain pulses cannot straddle an interior gap and pass through
    unchanged.

    :param events: Events to inspect (any order).
    :type events: list[EventRecord]
    :param rain_stormflow: Source ``datetime``/``rain``/``stormflow`` frame.
    :type rain_stormflow: pandas.DataFrame
    :param interevent_hours: Minimum dry gap between distinct events, in hours.
    :type interevent_hours: float
    :returns: Events split at interior interevent gaps, renumbered from 1.
    :rtype: list[EventRecord]
    """
    if not events:
        return []
    df = rain_stormflow
    t = _times(df)
    rain = df["rain"].to_numpy(dtype=float)
    hps = _hours_per_sample(df)
    gap_samples = _samples(interevent_hours, hps)

    out: list[EventRecord] = []
    for e in sorted(events, key=lambda x: x.start_datetime):
        i0 = _idx_of(t, e.start_datetime)
        i1 = _idx_of(t, e.end_datetime)
        wet = i0 + np.flatnonzero(rain[i0: i1 + 1] > 0.0)

        if len(wet) < 2:
            out.append(_record_from_span(len(out) + 1, i0, i1, df, _idx_of(t, e.peak_datetime)))
            continue

        # Cut wherever two successive rain pulses are >= the interevent apart.
        seg_start = i0
        for a, b in zip(wet[:-1], wet[1:]):
            if (b - a) >= gap_samples:
                out.append(_record_from_span(len(out) + 1, seg_start, int(b) - 1, df))
                seg_start = int(b)
        out.append(_record_from_span(len(out) + 1, seg_start, i1, df))

    return _renumber(out)


def split_top_peak_events(
    events: list[EventRecord],
    rain_stormflow: pd.DataFrame,
    filter_result: FilterResult,
    n_top: int = 10,
    sep_frac: float = 0.5,
) -> list[EventRecord]:
    """Split events that contain two or more of the top-*n* flow peaks.

    The ten (by default) largest peaks of the smoothed ``sg_0`` signal are
    identified globally.  Any event spanning two or more of them is split at the
    intervening trough, provided the trough is a *meaningful* separation: the
    trough must fall below *sep_frac* of the smaller flanking peak.

    :param events: Events to inspect.
    :type events: list[EventRecord]
    :param rain_stormflow: Source ``datetime``/``rain``/``stormflow`` frame.
    :type rain_stormflow: pandas.DataFrame
    :param filter_result: Filter output supplying the smoothed ``sg_0`` signal.
    :type filter_result: FilterResult
    :param n_top: Number of largest peaks to consider.
    :type n_top: int
    :param sep_frac: Trough-depth ratio below which two peaks count as
        meaningfully separated (0-1).
    :type sep_frac: float
    :returns: Events with multi-peak spans split, renumbered from 1.
    :rtype: list[EventRecord]
    """
    if not events:
        return []
    df = rain_stormflow
    t = _times(df)
    sg0 = np.asarray(filter_result.sg_0, dtype=float)

    peak_idx, props = find_peaks(sg0, height=0)
    if len(peak_idx) == 0:
        return _renumber(list(events))
    order = np.argsort(props["peak_heights"])[::-1][:n_top]
    top = np.sort(peak_idx[order])

    out: list[EventRecord] = []
    for e in sorted(events, key=lambda x: x.start_datetime):
        i0 = _idx_of(t, e.start_datetime)
        i1 = _idx_of(t, e.end_datetime)
        inside = [int(p) for p in top if i0 <= p <= i1]

        if len(inside) < 2:
            out.append(_record_from_span(len(out) + 1, i0, i1, df, _idx_of(t, e.peak_datetime)))
            continue

        # Walk the in-event top peaks, cutting at meaningful troughs.
        seg_start = i0
        prev_peak = inside[0]
        for nxt_peak in inside[1:]:
            trough = prev_peak + int(np.argmin(sg0[prev_peak: nxt_peak + 1]))
            val_trough = float(sg0[trough])
            smaller = min(float(sg0[prev_peak]), float(sg0[nxt_peak]))
            if val_trough < sep_frac * smaller:
                out.append(_record_from_span(len(out) + 1, seg_start, trough, df, prev_peak,
                                             b2b_end=True))
                seg_start = trough
            prev_peak = nxt_peak
        out.append(_record_from_span(len(out) + 1, seg_start, i1, df, prev_peak,
                                     b2b_start=(seg_start != i0)))

    return _renumber(out)


def refine_event_durations(
    events: list[EventRecord],
    rain_stormflow: pd.DataFrame,
    filter_result: FilterResult,
    *,
    pre_rain_lead_hours: float = 0.25,
    max_post_rain_hours: float = 48.0,
    baseflow_frac: float = 2.0 / 3.0,
    baseflow_threshold: float | None = None,
    min_rain_in: float = 0.25,
) -> list[EventRecord]:
    """Recompute each event's start and end per the MSDGC duration rules.

    For every event:

    - the **start** is moved to *pre_rain_lead_hours* before the first recorded
      rainfall in the event window;
    - the **end** is set two-thirds of the way from the end of rainfall to the
      point where the smoothed flow returns to base flow, then capped at
      *max_post_rain_hours* past the end of rainfall and clamped so it never
      reaches the **next qualifying** event's start.

    The clamp deliberately ignores non-qualifying events (those with less than
    *min_rain_in* of rainfall): a trailing sprinkle that will itself be screened
    out must not amputate a real event's recession.  This mirrors the MSDGC
    wording — the event ends at the start of the next *qualifying* event.

    "Return to base flow" is the first sample after the end of rainfall whose
    ``sg_0`` value falls below *baseflow_threshold* (default: the filter's
    ``sg_0_th`` steady-state threshold), since ``stormflow`` is already
    baseflow-separated.

    Events with no rainfall in their window are returned unchanged.

    :param events: Events to refine.
    :type events: list[EventRecord]
    :param rain_stormflow: Source ``datetime``/``rain``/``stormflow`` frame.
    :type rain_stormflow: pandas.DataFrame
    :param filter_result: Filter output supplying ``sg_0`` and thresholds.
    :type filter_result: FilterResult
    :param pre_rain_lead_hours: Lead-in before first rain, in hours.
    :type pre_rain_lead_hours: float
    :param max_post_rain_hours: Hard cap on extension past end of rainfall.
    :type max_post_rain_hours: float
    :param baseflow_frac: Fraction of the rain-to-baseflow interval to retain.
    :type baseflow_frac: float
    :param baseflow_threshold: Override for the return-to-baseflow level.
    :type baseflow_threshold: float | None
    :param min_rain_in: Rainfall (inches) at or above which an event counts as
        *qualifying* for the end-of-event clamp.
    :type min_rain_in: float
    :returns: Refined events, renumbered from 1.
    :rtype: list[EventRecord]
    """
    if not events:
        return []
    df = rain_stormflow
    t = _times(df)
    n = len(df)
    hps = _hours_per_sample(df)
    lead = _samples(pre_rain_lead_hours, hps)
    cap = _samples(max_post_rain_hours, hps)
    sg0 = np.asarray(filter_result.sg_0, dtype=float)
    bf_th = baseflow_threshold
    if bf_th is None:
        bf_th = float(filter_result.thresholds.get("sg_0_th", 0.0))

    ordered = sorted(events, key=lambda e: e.start_datetime)

    # First pass: resolve each event's new start and provisional end.
    spans: list[dict] = []
    for e in ordered:
        i0 = _idx_of(t, e.start_datetime)
        i1 = _idx_of(t, e.end_datetime)
        peak = _idx_of(t, e.peak_datetime)
        first_rain, last_rain = _first_last_rain(df, i0, i1)

        if first_rain is None:
            spans.append({"start": i0, "end": i1, "peak": peak, "qual": False})
            continue

        new_start = max(0, first_rain - lead)

        # Return to base flow: first post-rain sample below threshold.
        search_end = min(n - 1, last_rain + cap)
        bf_idx = search_end
        for j in range(last_rain + 1, search_end + 1):
            if sg0[j] <= bf_th:
                bf_idx = j
                break

        target_end = last_rain + int(round(baseflow_frac * (bf_idx - last_rain)))
        new_end = min(target_end, last_rain + cap, n - 1)
        new_end = max(new_end, peak)  # never end before the peak
        rain_tot = float(df["rain"].iloc[i0: i1 + 1].sum())
        spans.append({"start": new_start, "end": new_end, "peak": peak,
                      "qual": rain_tot >= min_rain_in})

    # Second pass: clamp each end to the start of the next *qualifying* event
    # (non-qualifying sprinkles must not truncate a real event's recession).
    # Non-qualifying events are clamped to the immediately following event so
    # their reported spans stay tidy without constraining the qualifying ones.
    n_spans = len(spans)
    for k in range(n_spans):
        if spans[k]["qual"]:
            nxt = next((m for m in range(k + 1, n_spans) if spans[m]["qual"]), None)
        else:
            nxt = k + 1 if k + 1 < n_spans else None
        if nxt is None:
            continue
        nxt_start = spans[nxt]["start"]
        if spans[k]["end"] >= nxt_start:
            spans[k]["end"] = max(spans[k]["start"], nxt_start - 1)

    refined = [
        _record_from_span(k + 1, s["start"], s["end"], df, s["peak"])
        for k, s in enumerate(spans)
    ]
    return _renumber(refined)


def screen_small_rainfall(
    events: list[EventRecord],
    min_rain_in: float = 0.25,
) -> list[EventRecord]:
    """Tag events by the < 0.25 in "small rainfall" screening criterion.

    Sets ``qualifying`` and ``exclusion_reason`` on every event in place.  No
    events are dropped — the caller decides what to keep.

    :param events: Events to categorise.
    :type events: list[EventRecord]
    :param min_rain_in: Minimum total rainfall (inches) to qualify.
    :type min_rain_in: float
    :returns: The same events, annotated in place.
    :rtype: list[EventRecord]
    """
    for e in events:
        if e.total_rain < min_rain_in:
            e.qualifying = False
            e.exclusion_reason = _SMALL_RAINFALL
        else:
            e.qualifying = True
            e.exclusion_reason = None
    return events


def flag_flow_response(
    events: list[EventRecord],
    rain_stormflow: pd.DataFrame,
    filter_result: FilterResult,
    *,
    response_threshold: float | None = None,
) -> list[EventRecord]:
    """Flag events with no discernible flow rise after rainfall.

    Since ``stormflow`` is baseflow-separated, a discernible response is taken
    as a peak exceeding *response_threshold* (default: the filter's ``sg_0_th``
    steady-state threshold).  Events failing the test have ``flow_response`` set
    to ``False`` — they are flagged, not excluded (the spec says they "may be
    considered for exclusion").

    :param events: Events to flag.
    :type events: list[EventRecord]
    :param rain_stormflow: Source frame (unused placeholder for symmetry/future
        pre-event baseline logic).
    :type rain_stormflow: pandas.DataFrame
    :param filter_result: Filter output supplying the response threshold.
    :type filter_result: FilterResult
    :param response_threshold: Override for the minimum discernible peak.
    :type response_threshold: float | None
    :returns: The same events, annotated in place.
    :rtype: list[EventRecord]
    """
    th = response_threshold
    if th is None:
        th = float(filter_result.thresholds.get("sg_0_th", 0.0))
    for e in events:
        e.flow_response = bool(e.peak_flow > th)
    return events


def screen_min_duration(
    events: list[EventRecord],
    min_duration_hours: float = 3.0,
) -> list[EventRecord]:
    """Exclude events shorter than *min_duration_hours* (downgrade only).

    Mirrors the MSDGC *Generate Storm Events* tool's own ``MinEvtDuration``
    setting (default 3 hr): degenerate few-sample windows that survive the
    duration refinement are marked non-qualifying so they never reach
    calibration or scoring, where their tiny sample counts make variance-based
    metrics (KGE) explode.  Only events that are still *qualifying* are
    downgraded, so an event already excluded by an earlier rule keeps its
    original ``exclusion_reason``.

    Sets ``qualifying`` and ``exclusion_reason`` in place; no events are
    dropped.

    :param events: Events to categorise (already screened for rainfall).
    :type events: list[EventRecord]
    :param min_duration_hours: Minimum event duration, in hours, to qualify.
    :type min_duration_hours: float
    :returns: The same events, annotated in place.
    :rtype: list[EventRecord]
    """
    for e in events:
        if getattr(e, "qualifying", True) and e.duration_hours() < min_duration_hours:
            e.qualifying = False
            e.exclusion_reason = _SHORT_DURATION
    return events


def screen_flow_response(events: list[EventRecord]) -> list[EventRecord]:
    """Exclude events flagged with no discernible flow response (downgrade only).

    Enforces the §6.2.2 flow-response check as a hard screen: events whose
    ``flow_response`` attribute is ``False`` (set by :func:`flag_flow_response`)
    are marked non-qualifying.  Only events that are still *qualifying* are
    downgraded, so an event already excluded for small rainfall or short
    duration keeps its original ``exclusion_reason``.  Run
    :func:`flag_flow_response` first.

    Sets ``qualifying`` and ``exclusion_reason`` in place; no events are
    dropped.

    :param events: Events to categorise (already flagged by
        :func:`flag_flow_response`).
    :type events: list[EventRecord]
    :returns: The same events, annotated in place.
    :rtype: list[EventRecord]
    """
    for e in events:
        if getattr(e, "qualifying", True) and not getattr(e, "flow_response", True):
            e.qualifying = False
            e.exclusion_reason = _NO_FLOW_RESPONSE
    return events


# ---------------------------------------------------------------------------
# Public API — orchestrator
# ---------------------------------------------------------------------------

def screening_summary(events: list[EventRecord]) -> pd.DataFrame:
    """Return a per-event categorisation table for all candidate events.

    :param events: Candidate events (already annotated by the screening rules).
    :type events: list[EventRecord]
    :returns: One row per event with the standard :class:`EventRecord` fields
        plus ``qualifying``, ``exclusion_reason`` and ``flow_response``.
    :rtype: pandas.DataFrame
    """
    df = events_to_dataframe(events)
    if df.empty:
        df["qualifying"] = []
        df["exclusion_reason"] = []
        df["flow_response"] = []
        return df
    df["qualifying"] = [getattr(e, "qualifying", True) for e in events]
    df["exclusion_reason"] = [getattr(e, "exclusion_reason", None) for e in events]
    df["flow_response"] = [getattr(e, "flow_response", True) for e in events]
    return df


def build_global_events(
    rain_stormflow: pd.DataFrame,
    filter_result: FilterResult | None = None,
    events: list[EventRecord] | None = None,
    *,
    interevent_hours: float = 6.0,
    pre_rain_lead_hours: float = 0.25,
    max_post_rain_hours: float = 48.0,
    baseflow_frac: float = 2.0 / 3.0,
    min_rain_in: float = 0.25,
    min_duration_hours: float = 3.0,
    drop_no_flow_response: bool = True,
    n_top_peaks: int = 10,
    detect_kwargs: dict | None = None,
) -> tuple[list[EventRecord], pd.DataFrame]:
    """Run the full MSDGC §6.2 global-event pipeline.

    Pipeline order: interevent merge -> interevent split -> top-10 peak split
    -> duration refinement -> small-rainfall screening -> flow-response
    flagging -> minimum-duration screening -> flow-response screening
    (optional) -> renumber.  When *events* / *filter_result* are omitted,
    :func:`detect_events` is called first.

    :param rain_stormflow: Source ``datetime``/``rain``/``stormflow`` frame.
    :type rain_stormflow: pandas.DataFrame
    :param filter_result: Existing filter output; recomputed if ``None``.
    :type filter_result: FilterResult | None
    :param events: Pre-detected events; detected here if ``None``.
    :type events: list[EventRecord] | None
    :param interevent_hours: Global interevent period, in hours.
    :type interevent_hours: float
    :param pre_rain_lead_hours: Lead-in before first rain, in hours.
    :type pre_rain_lead_hours: float
    :param max_post_rain_hours: Hard cap on extension past end of rainfall.
    :type max_post_rain_hours: float
    :param baseflow_frac: Fraction of the rain-to-baseflow interval to retain.
    :type baseflow_frac: float
    :param min_rain_in: Minimum total rainfall (inches) to qualify.
    :type min_rain_in: float
    :param min_duration_hours: Minimum refined event duration (hours) to
        qualify; events shorter than this are excluded "short duration"
        (matches the MSDGC tool's ``MinEvtDuration``).  Set to ``0`` to disable.
    :type min_duration_hours: float
    :param drop_no_flow_response: When ``True`` (default), events flagged with
        no discernible flow response are excluded from the qualifying set (a
        hard §6.2.2 screen).  When ``False``, they are only flagged.
    :type drop_no_flow_response: bool
    :param n_top_peaks: Number of largest peaks used by the split rule.
    :type n_top_peaks: int
    :param detect_kwargs: Extra keyword args forwarded to :func:`detect_events`
        when it is called internally.
    :type detect_kwargs: dict | None
    :returns: ``(qualifying_events, summary)`` where *qualifying_events* are the
        renumbered events that passed screening and *summary* categorises every
        candidate event (including excluded ones).
    :rtype: tuple[list[EventRecord], pandas.DataFrame]
    """
    if events is None or filter_result is None:
        events, filter_result = detect_events(rain_stormflow, **(detect_kwargs or {}))

    ev = merge_by_interevent(events, rain_stormflow, interevent_hours)
    ev = split_by_interevent(ev, rain_stormflow, interevent_hours)
    ev = split_top_peak_events(ev, rain_stormflow, filter_result, n_top_peaks)
    ev = refine_event_durations(
        ev, rain_stormflow, filter_result,
        pre_rain_lead_hours=pre_rain_lead_hours,
        max_post_rain_hours=max_post_rain_hours,
        baseflow_frac=baseflow_frac,
        min_rain_in=min_rain_in,
    )
    screen_small_rainfall(ev, min_rain_in)
    flag_flow_response(ev, rain_stormflow, filter_result)
    screen_min_duration(ev, min_duration_hours)
    if drop_no_flow_response:
        screen_flow_response(ev)
    _renumber(ev)

    summary = screening_summary(ev)
    qualifying = _renumber([e for e in ev if getattr(e, "qualifying", True)])
    return qualifying, summary


__all__ = [
    "merge_by_interevent",
    "split_by_interevent",
    "split_top_peak_events",
    "refine_event_durations",
    "screen_small_rainfall",
    "flag_flow_response",
    "screen_min_duration",
    "screen_flow_response",
    "screening_summary",
    "build_global_events",
]
