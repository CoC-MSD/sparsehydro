"""Tests for the MSDGC global-event screening rules.

Synthetic 5-minute rain/stormflow series with injected, well-separated storm
pulses are used to exercise each rule in
:mod:`sparsehydro.events.screening` independently, plus the
:func:`build_global_events` orchestrator end to end.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sparsehydro.events import EventRecord
from sparsehydro.filters import apply_savgol_filter, compute_thresholds
from sparsehydro.events.screening import build_global_events, flag_flow_response, merge_by_interevent, refine_event_durations, screen_flow_response, screen_min_duration, screen_small_rainfall, split_by_interevent, split_top_peak_events

STEP = "5min"
PPH = 12  # samples per hour at 5-minute cadence


def _triangle(length: int, peak: float) -> np.ndarray:
    """Return a symmetric triangular pulse of *length* samples peaking at *peak*."""
    half = length // 2
    up = np.linspace(0.0, peak, half, endpoint=False)
    down = np.linspace(peak, 0.0, length - half)
    return np.concatenate([up, down])


def _series(pulses, n_hours=240, start="2023-03-01"):
    """Build a rain/stormflow frame.

    Parameters
    ----------
    pulses : list[dict]
        Each dict has ``at`` (hour offset), ``rain_hours``, ``rain_rate``,
        ``flow_hours`` and ``flow_peak``.
    """
    periods = n_hours * PPH
    idx = pd.date_range(start=start, periods=periods, freq=STEP)
    rain = np.zeros(periods)
    flow = np.zeros(periods)
    for p in pulses:
        s = p["at"] * PPH
        rlen = int(p["rain_hours"] * PPH)
        rain[s: s + rlen] += p["rain_rate"]
        flen = int(p["flow_hours"] * PPH)
        flow[s: s + flen] += _triangle(flen, p["flow_peak"])
    return pd.DataFrame({"datetime": idx, "rain": rain, "stormflow": flow})


def _fr(df):
    fr = apply_savgol_filter(df, window_length=24, polyorder=2)
    return compute_thresholds(fr)


def _event(df, start_h, end_h, peak_h):
    """Construct an EventRecord spanning the given hour offsets."""
    t0 = df["datetime"].iloc[0]
    return EventRecord(
        event_id=1,
        start_datetime=t0 + pd.Timedelta(hours=start_h),
        end_datetime=t0 + pd.Timedelta(hours=end_h),
        peak_datetime=t0 + pd.Timedelta(hours=peak_h),
        total_rain=0.0,
        total_flow=0.0,
        effective_area=0.0,
        peak_flow=0.0,
    )


# ---------------------------------------------------------------------------
# screen_small_rainfall
# ---------------------------------------------------------------------------

def test_small_rainfall_excluded():
    small = _event(_series([]), 0, 5, 2)
    small.total_rain = 0.10
    big = _event(_series([]), 10, 15, 12)
    big.total_rain = 0.80

    screen_small_rainfall([small, big], min_rain_in=0.25)

    assert small.qualifying is False
    assert small.exclusion_reason == "small rainfall"
    assert big.qualifying is True
    assert big.exclusion_reason is None


# ---------------------------------------------------------------------------
# merge_by_interevent
# ---------------------------------------------------------------------------

def test_merge_within_interevent():
    df = _series([
        {"at": 20, "rain_hours": 2, "rain_rate": 0.2, "flow_hours": 8, "flow_peak": 5.0},
        {"at": 32, "rain_hours": 2, "rain_rate": 0.2, "flow_hours": 8, "flow_peak": 5.0},
    ])
    # Two events 4 h apart (gap < 6 h interevent) should merge.
    e1 = _event(df, 20, 26, 22)
    e2 = _event(df, 30, 38, 34)
    merged = merge_by_interevent([e1, e2], df, interevent_hours=6.0)
    assert len(merged) == 1

    # Same events with a 6 h+ gap should stay separate.
    kept = merge_by_interevent([e1, e2], df, interevent_hours=3.0)
    assert len(kept) == 2


# ---------------------------------------------------------------------------
# split_by_interevent
# ---------------------------------------------------------------------------

def test_split_at_interior_dry_gap():
    # One over-broad event straddling two rain bursts 48 h apart (a dry gap far
    # larger than the 6 h interevent) must split into two events.
    df = _series([
        {"at": 24, "rain_hours": 3, "rain_rate": 0.3, "flow_hours": 10, "flow_peak": 6.0},
        {"at": 75, "rain_hours": 3, "rain_rate": 0.3, "flow_hours": 10, "flow_peak": 6.0},
    ], n_hours=140)
    blob = _event(df, 23, 110, 27)
    out = split_by_interevent([blob], df, interevent_hours=6.0)
    assert len(out) == 2
    # The cut lands before the second burst's first rain (hour 75).
    t0 = df["datetime"].iloc[0]
    assert out[0].end_datetime < t0 + pd.Timedelta(hours=75)
    assert out[1].start_datetime >= t0 + pd.Timedelta(hours=75) - pd.Timedelta(hours=1)


def test_no_split_for_continuous_burst():
    # A single continuous rain burst has no interior >= 6 h dry gap and must be
    # left intact.
    df = _series([
        {"at": 24, "rain_hours": 5, "rain_rate": 0.3, "flow_hours": 12, "flow_peak": 6.0},
    ], n_hours=120)
    blob = _event(df, 23, 60, 27)
    out = split_by_interevent([blob], df, interevent_hours=6.0)
    assert len(out) == 1


# ---------------------------------------------------------------------------
# refine_event_durations
# ---------------------------------------------------------------------------

def test_duration_starts_before_first_rain():
    df = _series([
        {"at": 24, "rain_hours": 3, "rain_rate": 0.3, "flow_hours": 20, "flow_peak": 6.0},
    ])
    fr = _fr(df)
    # Detected window overshoots on both sides; refine should pull start to
    # 0.25 h before the first rain at hour 24.
    e = _event(df, 20, 80, 30)
    e.total_rain = 0.9
    refined = refine_event_durations([e], df, fr)[0]

    first_rain = df["datetime"].iloc[0] + pd.Timedelta(hours=24)
    expected_start = first_rain - pd.Timedelta(hours=0.25)
    assert abs((refined.start_datetime - expected_start).total_seconds()) <= 300


def test_duration_capped_at_48h_past_rain():
    df = _series([
        {"at": 24, "rain_hours": 3, "rain_rate": 0.3, "flow_hours": 3, "flow_peak": 6.0},
    ], n_hours=140)
    fr = _fr(df)
    # Flow never returns to baseflow within the window -> 48 h cap governs.
    e = _event(df, 23, 130, 26)
    e.total_rain = 0.9
    refined = refine_event_durations([e], df, fr, baseflow_frac=1.0)[0]

    end_of_rain = df["datetime"].iloc[0] + pd.Timedelta(hours=27)
    max_end = end_of_rain + pd.Timedelta(hours=48)
    assert refined.end_datetime <= max_end + pd.Timedelta(minutes=5)


def test_duration_clamped_to_next_event():
    df = _series([
        {"at": 24, "rain_hours": 3, "rain_rate": 0.3, "flow_hours": 40, "flow_peak": 6.0},
        {"at": 60, "rain_hours": 3, "rain_rate": 0.3, "flow_hours": 20, "flow_peak": 6.0},
    ], n_hours=160)
    fr = _fr(df)
    e1 = _event(df, 23, 70, 26)
    e1.total_rain = 0.9
    e2 = _event(df, 59, 100, 62)
    e2.total_rain = 0.9
    refined = refine_event_durations([e1, e2], df, fr)
    # First event must not run into the second event's start.
    assert refined[0].end_datetime < refined[1].start_datetime


def test_duration_not_clamped_by_nonqualifying_sprinkle():
    # A qualifying storm followed by a tiny <0.25" sprinkle before it has fully
    # receded. The sprinkle must NOT truncate the qualifying event's recession;
    # only the next *qualifying* event may clamp it (MSDGC wording).
    df = _series([
        {"at": 24, "rain_hours": 3, "rain_rate": 0.3, "flow_hours": 60, "flow_peak": 6.0},
        {"at": 40, "rain_hours": 1, "rain_rate": 0.01, "flow_hours": 4, "flow_peak": 0.3},  # 0.12" sprinkle
    ], n_hours=160)
    fr = _fr(df)
    # Post-split windows: the storm ends before the sprinkle's rain begins.
    big = _event(df, 23, 39, 26)
    big.total_rain = 0.9
    sprinkle = _event(df, 39, 55, 41)
    sprinkle.total_rain = 0.12
    # Baseflow threshold set unreachable -> return-to-baseflow never triggers,
    # so the 48 h cap governs the end deterministically.
    refined = refine_event_durations(
        [big, sprinkle], df, fr, baseflow_frac=1.0, baseflow_threshold=-1e9,
    )
    # The qualifying event should extend well past the sprinkle's start (hour 40),
    # not be amputated at it.
    sprinkle_start = df["datetime"].iloc[0] + pd.Timedelta(hours=40)
    assert refined[0].end_datetime > sprinkle_start + pd.Timedelta(hours=5)


# ---------------------------------------------------------------------------
# split_top_peak_events
# ---------------------------------------------------------------------------

def test_split_two_top_peaks():
    df = _series([
        {"at": 24, "rain_hours": 2, "rain_rate": 0.3, "flow_hours": 6, "flow_peak": 8.0},
        {"at": 40, "rain_hours": 2, "rain_rate": 0.3, "flow_hours": 6, "flow_peak": 8.0},
    ], n_hours=120)
    fr = _fr(df)
    # One over-broad event covering both distinct peaks.
    e = _event(df, 22, 52, 25)
    out = split_top_peak_events([e], df, fr, n_top=10, sep_frac=0.5)
    assert len(out) == 2
    assert out[0].end_datetime <= out[1].start_datetime


def test_single_peak_not_split():
    df = _series([
        {"at": 24, "rain_hours": 2, "rain_rate": 0.3, "flow_hours": 8, "flow_peak": 8.0},
    ], n_hours=120)
    fr = _fr(df)
    e = _event(df, 22, 40, 26)
    out = split_top_peak_events([e], df, fr)
    assert len(out) == 1


# ---------------------------------------------------------------------------
# flag_flow_response
# ---------------------------------------------------------------------------

def test_flow_response_flag():
    df = _series([
        {"at": 24, "rain_hours": 2, "rain_rate": 0.3, "flow_hours": 8, "flow_peak": 8.0},
    ], n_hours=120)
    fr = _fr(df)
    responsive = _event(df, 23, 40, 26)
    responsive.peak_flow = 8.0
    flat = _event(df, 50, 60, 52)
    flat.peak_flow = 0.0
    flag_flow_response([responsive, flat], df, fr)
    assert responsive.flow_response is True
    assert flat.flow_response is False


# ---------------------------------------------------------------------------
# screen_min_duration
# ---------------------------------------------------------------------------

def test_min_duration_excludes_short_event():
    df = _series([])
    short = _event(df, 0, 2, 1)      # 2 h span < 3 h threshold
    short.total_rain = 0.9
    long = _event(df, 10, 15, 12)    # 5 h span
    long.total_rain = 0.9
    screen_small_rainfall([short, long], min_rain_in=0.25)  # both qualify on rain

    screen_min_duration([short, long], min_duration_hours=3.0)

    assert short.qualifying is False
    assert short.exclusion_reason == "short duration"
    assert long.qualifying is True
    assert long.exclusion_reason is None


def test_min_duration_downgrade_only_preserves_prior_reason():
    # An event already excluded for small rainfall AND shorter than the
    # threshold must keep its original "small rainfall" reason.
    df = _series([])
    tiny_short = _event(df, 0, 1, 0)
    tiny_short.total_rain = 0.05
    screen_small_rainfall([tiny_short], min_rain_in=0.25)
    assert tiny_short.exclusion_reason == "small rainfall"

    screen_min_duration([tiny_short], min_duration_hours=3.0)

    assert tiny_short.qualifying is False
    assert tiny_short.exclusion_reason == "small rainfall"


# ---------------------------------------------------------------------------
# screen_flow_response
# ---------------------------------------------------------------------------

def test_flow_response_screen_excludes_flat_event():
    df = _series([])
    responsive = _event(df, 0, 5, 2)
    responsive.total_rain = 0.9
    responsive.flow_response = True
    flat = _event(df, 10, 16, 12)
    flat.total_rain = 0.9
    flat.flow_response = False
    screen_small_rainfall([responsive, flat], min_rain_in=0.25)

    screen_flow_response([responsive, flat])

    assert responsive.qualifying is True
    assert responsive.exclusion_reason is None
    assert flat.qualifying is False
    assert flat.exclusion_reason == "no flow response"


def test_flow_response_screen_downgrade_only_preserves_prior_reason():
    # A short, non-responsive event keeps the "short duration" reason set first.
    df = _series([])
    short_flat = _event(df, 0, 2, 1)
    short_flat.total_rain = 0.9
    short_flat.flow_response = False
    screen_small_rainfall([short_flat], min_rain_in=0.25)
    screen_min_duration([short_flat], min_duration_hours=3.0)
    assert short_flat.exclusion_reason == "short duration"

    screen_flow_response([short_flat])

    assert short_flat.qualifying is False
    assert short_flat.exclusion_reason == "short duration"


# ---------------------------------------------------------------------------
# build_global_events (orchestrator)
# ---------------------------------------------------------------------------

def test_build_global_events_end_to_end():
    df = _series([
        {"at": 24, "rain_hours": 3, "rain_rate": 0.30, "flow_hours": 20, "flow_peak": 8.0},
        {"at": 90, "rain_hours": 3, "rain_rate": 0.30, "flow_hours": 20, "flow_peak": 8.0},
        {"at": 160, "rain_hours": 1, "rain_rate": 0.02, "flow_hours": 4, "flow_peak": 0.4},  # tiny
    ], n_hours=220)

    qualifying, summary = build_global_events(
        df, detect_kwargs={"window_length": 24, "polyorder": 2, "height_perc": 20.0},
    )

    # Summary categorises every candidate; qualifying is a subset.
    assert {"qualifying", "exclusion_reason", "flow_response"}.issubset(summary.columns)
    assert len(qualifying) <= len(summary)
    assert all(e.qualifying for e in qualifying)
    # Event ids are contiguous from 1.
    assert [e.event_id for e in qualifying] == list(range(1, len(qualifying) + 1))
    # The tiny <0.25" pulse must not appear among qualifying events.
    assert all(e.total_rain >= 0.25 for e in qualifying)


def test_build_global_events_applies_both_guards():
    df = _series([
        {"at": 24, "rain_hours": 3, "rain_rate": 0.30, "flow_hours": 20, "flow_peak": 8.0},
        {"at": 90, "rain_hours": 3, "rain_rate": 0.30, "flow_hours": 20, "flow_peak": 8.0},
    ], n_hours=220)

    qualifying, _ = build_global_events(
        df, detect_kwargs={"window_length": 24, "polyorder": 2, "height_perc": 20.0},
        min_duration_hours=3.0, drop_no_flow_response=True,
    )
    # Every qualifying event clears both guards.
    assert all(e.duration_hours() >= 3.0 for e in qualifying)
    assert all(getattr(e, "flow_response", True) for e in qualifying)


def test_build_global_events_guards_can_be_relaxed():
    df = _series([
        {"at": 24, "rain_hours": 3, "rain_rate": 0.30, "flow_hours": 20, "flow_peak": 8.0},
        {"at": 90, "rain_hours": 3, "rain_rate": 0.30, "flow_hours": 20, "flow_peak": 8.0},
    ], n_hours=220)

    strict, _ = build_global_events(
        df, detect_kwargs={"window_length": 24, "polyorder": 2, "height_perc": 20.0},
        min_duration_hours=3.0, drop_no_flow_response=True,
    )
    relaxed, _ = build_global_events(
        df, detect_kwargs={"window_length": 24, "polyorder": 2, "height_perc": 20.0},
        min_duration_hours=0.0, drop_no_flow_response=False,
    )
    # Relaxing the guards can only admit more (or equal) qualifying events.
    assert len(relaxed) >= len(strict)


def test_build_global_events_empty():
    df = _series([], n_hours=48)
    qualifying, summary = build_global_events(
        df, detect_kwargs={"window_length": 24, "polyorder": 2},
    )
    assert qualifying == []
    assert summary.empty or (summary["qualifying"].sum() == 0)
