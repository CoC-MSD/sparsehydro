"""Tests for :func:`sparsehydro.events.events_to_mask`.

Exercises the bridge from :class:`EventRecord` spans to the boolean-mask API
used by the calibration objectives / problem: inclusive endpoints, union across
multiple events, the ``qualifying_only`` filter, and acceptance of assorted
datetime container types.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sparsehydro.events import EventRecord, events_to_mask


def _event(start, end, *, event_id=1, qualifying=True) -> EventRecord:
    """Build a minimal EventRecord spanning ``[start, end]``."""
    return EventRecord(
        event_id=event_id,
        start_datetime=pd.Timestamp(start),
        end_datetime=pd.Timestamp(end),
        peak_datetime=pd.Timestamp(start),
        total_rain=0.0,
        total_flow=0.0,
        effective_area=0.0,
        peak_flow=0.0,
        qualifying=qualifying,
    )


def _hourly(start="2022-01-01", periods=24):
    return pd.date_range(start=start, periods=periods, freq="1h")


def test_marks_inclusive_span():
    dts = _hourly(periods=10)  # 00:00 .. 09:00
    events = [_event("2022-01-01 02:00", "2022-01-01 05:00")]
    mask = events_to_mask(events, dts)

    assert mask.dtype == bool
    assert mask.shape == (10,)
    # Endpoints are inclusive: hours 2,3,4,5 True; everything else False.
    expected = np.array([False, False, True, True, True, True,
                         False, False, False, False])
    assert np.array_equal(mask, expected)


def test_union_across_multiple_events():
    dts = _hourly(periods=12)
    events = [
        _event("2022-01-01 01:00", "2022-01-01 02:00", event_id=1),
        _event("2022-01-01 08:00", "2022-01-01 09:00", event_id=2),
    ]
    mask = events_to_mask(events, dts)
    assert set(np.flatnonzero(mask)) == {1, 2, 8, 9}


def test_empty_events_all_false():
    dts = _hourly(periods=5)
    mask = events_to_mask([], dts)
    assert mask.shape == (5,)
    assert not mask.any()


def test_qualifying_only_filters_excluded_events():
    dts = _hourly(periods=12)
    events = [
        _event("2022-01-01 01:00", "2022-01-01 02:00", event_id=1, qualifying=True),
        _event("2022-01-01 08:00", "2022-01-01 09:00", event_id=2, qualifying=False),
    ]
    all_mask = events_to_mask(events, dts)
    qual_mask = events_to_mask(events, dts, qualifying_only=True)

    assert set(np.flatnonzero(all_mask)) == {1, 2, 8, 9}
    # The non-qualifying event (hours 8,9) is dropped.
    assert set(np.flatnonzero(qual_mask)) == {1, 2}


def test_accepts_series_index_and_list():
    idx = _hourly(periods=6)
    events = [_event("2022-01-01 02:00", "2022-01-01 03:00")]
    expected = np.array([False, False, True, True, False, False])

    from_index = events_to_mask(events, idx)
    from_series = events_to_mask(events, pd.Series(idx))
    from_list = events_to_mask(events, list(idx))

    assert np.array_equal(from_index, expected)
    assert np.array_equal(from_series, expected)
    assert np.array_equal(from_list, expected)
