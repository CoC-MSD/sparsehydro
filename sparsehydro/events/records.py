"""Hierarchical event records: sub-events and global events.

These complement the flat :class:`~sparsehydro.events.EventRecord` with the
two-level hierarchy produced by :func:`~sparsehydro.events.detect_event_hierarchy`:

- :class:`SubEventRecord` — a single-peak storm response with curvature-anchored
  rise/peak/tail zone boundaries and bimodality diagnostics.  Exposes
  ``event_id``/``start_datetime``/``end_datetime``/``effective_area`` so it is
  drop-in compatible with :class:`~sparsehydro.models.unithydrograph.SequentialFitter`.
- :class:`GlobalEvent` — a cluster of nearby sub-events (storms driven by one
  weather system) with a padded fitting window.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class EventRecord:
    """A single detected or imported storm event.

    :ivar event_id: Sequential identifier starting at 1.
    :vartype event_id: int
    :ivar start_datetime: Event start timestamp.
    :vartype start_datetime: pandas.Timestamp
    :ivar end_datetime: Event end timestamp.
    :vartype end_datetime: pandas.Timestamp
    :ivar peak_datetime: Timestamp of the sg_0 peak within the event window.
    :vartype peak_datetime: pandas.Timestamp
    :ivar total_rain: Cumulative rain depth over [start, end].
    :vartype total_rain: float
    :ivar total_flow: Cumulative stormflow over [start, end] (clipped to ≥ 0).
    :vartype total_flow: float
    :ivar effective_area: ``total_flow / total_rain``.  Zero when
        ``total_rain == 0``.
    :vartype effective_area: float
    :ivar peak_flow: Maximum stormflow value within the event window.
    :vartype peak_flow: float
    :ivar b2b_start: ``True`` if the event starts at a back-to-back trough.
    :vartype b2b_start: bool
    :ivar b2b_end: ``True`` if the event ends at a back-to-back trough.
    :vartype b2b_end: bool
    """

    event_id: int
    start_datetime: pd.Timestamp
    end_datetime: pd.Timestamp
    peak_datetime: pd.Timestamp
    total_rain: float
    total_flow: float
    effective_area: float
    peak_flow: float
    b2b_start: bool = False
    b2b_end: bool = False
    qualifying: bool = True
    exclusion_reason: str | None = None
    flow_response: bool = True

    def duration_hours(self) -> float:
        """Return event duration in hours.

        :returns: Duration between *start_datetime* and *end_datetime* in hours.
        :rtype: float
        """
        return (self.end_datetime - self.start_datetime).total_seconds() / 3600.0

    def to_dict(self) -> dict:
        """Serialize to plain dict.

        :returns: Mapping of every event field to its value.
        :rtype: dict
        """
        return {
            "event_id": self.event_id,
            "start_datetime": self.start_datetime,
            "end_datetime": self.end_datetime,
            "peak_datetime": self.peak_datetime,
            "total_rain": self.total_rain,
            "total_flow": self.total_flow,
            "effective_area": self.effective_area,
            "peak_flow": self.peak_flow,
            "b2b_start": self.b2b_start,
            "b2b_end": self.b2b_end,
            "qualifying": self.qualifying,
            "exclusion_reason": self.exclusion_reason,
            "flow_response": self.flow_response,
        }


def events_to_dataframe(events: list[EventRecord]) -> pd.DataFrame:
    """Convert a list of :class:`EventRecord` objects to a DataFrame.

    :param events: Events to convert (may be empty).
    :type events: list[EventRecord]
    :returns: One row per event with all event fields as columns.
    :rtype: pandas.DataFrame
    """
    if not events:
        return pd.DataFrame(
            columns=[
                "event_id", "start_datetime", "end_datetime", "peak_datetime",
                "total_rain", "total_flow", "effective_area", "peak_flow",
                "b2b_start", "b2b_end", "qualifying", "exclusion_reason", "flow_response",
            ]
        )
    return pd.DataFrame([e.to_dict() for e in events])


def events_to_mask(
    events: list[EventRecord], datetimes, *, qualifying_only: bool = False,
) -> np.ndarray:
    """Mark timestamps inside inclusive event windows, optionally after screening."""
    dt = pd.to_datetime(pd.Series(datetimes)).to_numpy(dtype="datetime64[ns]")
    mask = np.zeros(dt.shape[0], dtype=bool)
    for event in events:
        if qualifying_only and not event.qualifying:
            continue
        start = np.datetime64(pd.Timestamp(event.start_datetime), "ns")
        end = np.datetime64(pd.Timestamp(event.end_datetime), "ns")
        mask |= (dt >= start) & (dt <= end)
    return mask


@dataclass
class SubEventRecord:
    """A single-peak storm sub-event with zone boundaries and bimodality metrics.

    :ivar sub_id: Sequential sub-event identifier (1-based).  Aliased as
        ``event_id`` for sequential-fitter compatibility.
    :vartype sub_id: int
    :ivar global_id: Identifier of the owning :class:`GlobalEvent`.
    :vartype global_id: int
    :ivar start_datetime: Event start (rise-in, may include a baseflow lead).
    :vartype start_datetime: pandas.Timestamp
    :ivar end_datetime: Event end (recession settled to baseline).
    :vartype end_datetime: pandas.Timestamp
    :ivar peak_datetime: Timestamp of the sub-event peak.
    :vartype peak_datetime: pandas.Timestamp
    :ivar rise_start_datetime: Start of the peak zone (curvature knee / rain onset).
    :vartype rise_start_datetime: pandas.Timestamp
    :ivar tail_start_datetime: Start of the recession zone (post-peak inflection).
    :vartype tail_start_datetime: pandas.Timestamp
    :ivar total_rain: Cumulative rain over ``[start, end]``.
    :vartype total_rain: float
    :ivar total_flow: Cumulative stormflow over ``[start, end]`` (clipped ≥ 0).
    :vartype total_flow: float
    :ivar effective_area: ``total_flow / total_rain`` (0 when ``total_rain == 0``).
    :vartype effective_area: float
    :ivar peak_flow: Maximum stormflow within the event window.
    :vartype peak_flow: float
    :ivar start_idx: Integer index of *start_datetime* into the source series.
    :vartype start_idx: int
    :ivar end_idx: Integer index of *end_datetime* into the source series.
    :vartype end_idx: int
    :ivar peak_idx: Integer index of *peak_datetime* into the source series.
    :vartype peak_idx: int
    :ivar rise_start_idx: Integer index of the peak-zone start.
    :vartype rise_start_idx: int
    :ivar tail_start_idx: Integer index of the recession-zone start.
    :vartype tail_start_idx: int
    :ivar bimodality_index: Flow bimodality score in ``[0, 1]``.
    :vartype bimodality_index: float
    :ivar rain_bimodality_index: Rain-burst bimodality score in ``[0, 1]``.
    :vartype rain_bimodality_index: float
    :ivar is_bimodal: ``True`` if the parent event was flagged bimodal.
    :vartype is_bimodal: bool
    """

    sub_id: int
    global_id: int
    start_datetime: pd.Timestamp
    end_datetime: pd.Timestamp
    peak_datetime: pd.Timestamp
    rise_start_datetime: pd.Timestamp
    tail_start_datetime: pd.Timestamp
    total_rain: float
    total_flow: float
    effective_area: float
    peak_flow: float
    start_idx: int
    end_idx: int
    peak_idx: int
    rise_start_idx: int
    tail_start_idx: int
    bimodality_index: float = 0.0
    rain_bimodality_index: float = 0.0
    is_bimodal: bool = False

    @property
    def event_id(self) -> int:
        """Alias of :attr:`sub_id` for sequential-fitter compatibility.

        :returns: The sub-event identifier.
        :rtype: int
        """
        return self.sub_id

    def duration_hours(self) -> float:
        """Return the event duration in hours.

        :returns: ``(end_datetime - start_datetime)`` in hours.
        :rtype: float
        """
        return (self.end_datetime - self.start_datetime).total_seconds() / 3600.0

    def to_dict(self) -> dict:
        """Serialize to a plain dict.

        :returns: Mapping of every field (plus ``duration_hours``) to its value.
        :rtype: dict
        """
        return {
            "sub_id": self.sub_id,
            "global_id": self.global_id,
            "start_datetime": self.start_datetime,
            "end_datetime": self.end_datetime,
            "peak_datetime": self.peak_datetime,
            "rise_start_datetime": self.rise_start_datetime,
            "tail_start_datetime": self.tail_start_datetime,
            "total_rain": self.total_rain,
            "total_flow": self.total_flow,
            "effective_area": self.effective_area,
            "peak_flow": self.peak_flow,
            "bimodality_index": self.bimodality_index,
            "rain_bimodality_index": self.rain_bimodality_index,
            "is_bimodal": self.is_bimodal,
            "duration_hours": self.duration_hours(),
        }


@dataclass
class GlobalEvent:
    """A cluster of nearby sub-events with a padded fitting window.

    :ivar global_id: Sequential global-event identifier (1-based).
    :vartype global_id: int
    :ivar sub_ids: Member sub-event identifiers in chronological order.
    :vartype sub_ids: list[int]
    :ivar start_datetime: Earliest member sub-event start.
    :vartype start_datetime: pandas.Timestamp
    :ivar end_datetime: Latest member sub-event end.
    :vartype end_datetime: pandas.Timestamp
    :ivar window_start_datetime: Padded window start (baseline lead-in).
    :vartype window_start_datetime: pandas.Timestamp
    :ivar window_end_datetime: Padded window end (full recession).
    :vartype window_end_datetime: pandas.Timestamp
    :ivar peak_datetime: Timestamp of the largest member peak.
    :vartype peak_datetime: pandas.Timestamp
    :ivar total_rain: Cumulative rain over the member span.
    :vartype total_rain: float
    :ivar total_flow: Cumulative stormflow over the member span.
    :vartype total_flow: float
    :ivar effective_area: ``total_flow / total_rain`` over the member span.
    :vartype effective_area: float
    """

    global_id: int
    sub_ids: list[int]
    start_datetime: pd.Timestamp
    end_datetime: pd.Timestamp
    window_start_datetime: pd.Timestamp
    window_end_datetime: pd.Timestamp
    peak_datetime: pd.Timestamp
    total_rain: float
    total_flow: float
    effective_area: float
    b2b_members: bool = field(default=False)

    @property
    def event_id(self) -> int:
        """Alias of :attr:`global_id` for sequential-fitter compatibility.

        :returns: The global-event identifier.
        :rtype: int
        """
        return self.global_id

    def duration_hours(self) -> float:
        """Return the member span duration in hours.

        :returns: ``(end_datetime - start_datetime)`` in hours.
        :rtype: float
        """
        return (self.end_datetime - self.start_datetime).total_seconds() / 3600.0

    def to_dict(self) -> dict:
        """Serialize to a plain dict.

        :returns: Mapping of every field to its value.
        :rtype: dict
        """
        return {
            "global_id": self.global_id,
            "sub_ids": list(self.sub_ids),
            "n_sub_events": len(self.sub_ids),
            "start_datetime": self.start_datetime,
            "end_datetime": self.end_datetime,
            "window_start_datetime": self.window_start_datetime,
            "window_end_datetime": self.window_end_datetime,
            "peak_datetime": self.peak_datetime,
            "total_rain": self.total_rain,
            "total_flow": self.total_flow,
            "effective_area": self.effective_area,
        }


def sub_events_to_dataframe(sub_events: list[SubEventRecord]) -> pd.DataFrame:
    """Convert a list of :class:`SubEventRecord` to a tidy DataFrame.

    :param sub_events: Sub-events to convert (may be empty).
    :type sub_events: list[SubEventRecord]
    :returns: One row per sub-event with all public fields as columns.
    :rtype: pandas.DataFrame
    """
    cols = [
        "sub_id", "global_id", "start_datetime", "end_datetime", "peak_datetime",
        "rise_start_datetime", "tail_start_datetime", "total_rain", "total_flow",
        "effective_area", "peak_flow", "bimodality_index", "rain_bimodality_index",
        "is_bimodal", "duration_hours",
    ]
    if not sub_events:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame([e.to_dict() for e in sub_events])[cols]


def global_events_to_dataframe(global_events: list[GlobalEvent]) -> pd.DataFrame:
    """Convert a list of :class:`GlobalEvent` to a tidy DataFrame.

    :param global_events: Global events to convert (may be empty).
    :type global_events: list[GlobalEvent]
    :returns: One row per global event with all public fields as columns.
    :rtype: pandas.DataFrame
    """
    cols = [
        "global_id", "sub_ids", "n_sub_events", "start_datetime", "end_datetime",
        "window_start_datetime", "window_end_datetime", "peak_datetime",
        "total_rain", "total_flow", "effective_area",
    ]
    if not global_events:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame([e.to_dict() for e in global_events])[cols]


__all__ = [
    "SubEventRecord",
    "GlobalEvent",
    "sub_events_to_dataframe",
    "global_events_to_dataframe",
]
