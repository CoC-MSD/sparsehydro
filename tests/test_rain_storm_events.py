"""Tests for :func:`sparsehydro.events.rain_storm_events`.

Small synthetic hourly and 15-minute series with rain placed by hand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sparsehydro.calibration.splits.masks import storm_scoring_mask
from sparsehydro.events import rain_storm_events


def _data(rain_at: dict[int, float], n: int = 48, freq: str = "h") -> pd.DataFrame:
    """Return a frame of *n* steps with rain only at the given row positions."""
    rain = np.zeros(n)
    for row, depth in rain_at.items():
        rain[row] = depth
    return pd.DataFrame({"datetime": pd.date_range("2025-01-01", periods=n, freq=freq),
                         "rainfall_in": rain})


def test_gap_longer_than_interevent_gives_two_storms():
    # Wet at rows 2 and 10: 8 hours apart, more than 6
    storms = rain_storm_events(_data({2: 0.5, 10: 0.5}), interevent_hours=6.0)
    assert storms["start_idx"].tolist() == [2, 10]
    assert storms["end_idx"].tolist() == [2, 10]
    assert storms["event_id"].tolist() == [1, 2]


def test_gap_shorter_than_interevent_gives_one_storm():
    # Wet at rows 2, 3 and 7: 4 hours apart, less than 6
    storms = rain_storm_events(_data({2: 0.5, 3: 0.25, 7: 0.25}), interevent_hours=6.0)
    assert len(storms) == 1
    storm = storms.iloc[0]
    assert (storm.start_idx, storm.end_idx) == (2, 7)
    assert storm.rain_in == pytest.approx(1.0)
    assert storm.peak_rain_in == pytest.approx(0.5)
    assert storm.rain_start == pd.Timestamp("2025-01-01 02:00")
    assert storm.rain_end == pd.Timestamp("2025-01-01 08:00")   # one step after the last wet step
    assert storm.eligible and storm.split == "train"


@pytest.mark.parametrize("freq", ["5min", "15min", "1h"])
def test_same_storms_at_any_time_step(freq):
    # Two bursts 8 hours apart give two storms whatever the time step
    step = pd.Timedelta(freq)
    n = int(pd.Timedelta(hours=24) / step)
    rows = {int(pd.Timedelta(hours=h) / step): 0.5 for h in (2, 10)}
    storms = rain_storm_events(_data(rows, n=n, freq=freq))
    assert len(storms) == 2
    assert (storms["rain_end"] - storms["rain_start"]).eq(step).all()


def test_missing_rain_counts_as_dry():
    data = _data({2: 0.5, 10: 0.5})
    data.loc[5, "rainfall_in"] = np.nan
    assert len(rain_storm_events(data)) == 2


def test_minimum_rain_filter_drops_small_storms():
    data = _data({2: 0.5, 20: 0.1, 21: 0.1})
    storms = rain_storm_events(data, min_rain_in=0.25)
    assert storms["start_idx"].tolist() == [2]
    assert storms["event_id"].tolist() == [1]
    assert len(rain_storm_events(data, min_rain_in=0.2)) == 2
    assert rain_storm_events(_data({})).empty


def test_split_column():
    data = _data({5: 0.5, 30: 0.5})
    storms = rain_storm_events(data, split_at="2025-01-02 00:00")
    assert storms["split"].tolist() == ["train", "validation"]
    assert rain_storm_events(data)["split"].eq("train").all()


def test_round_trip_through_storm_scoring_mask():
    data = _data({5: 0.5, 6: 0.25, 30: 0.5})
    split_at = "2025-01-02 00:00"
    storms = rain_storm_events(data, split_at=split_at)
    train = (data["datetime"] < split_at).to_numpy()
    mask = storm_scoring_mask(storms, train, 1.0, split="train", pre_hours=2, tail_hours=3)
    expected = np.zeros(len(data), bool)
    expected[3:10] = True   # 2 h before row 5 to 3 h after row 6
    np.testing.assert_array_equal(mask, expected)


def test_storm_set_not_eligible_is_not_scored():
    # The caller can leave a storm out, for example one with too much missing flow
    data = _data({5: 0.5, 30: 0.5})
    storms = rain_storm_events(data)
    storms.loc[1, "eligible"] = False
    mask = storm_scoring_mask(storms, np.ones(len(data), bool), 1.0, pre_hours=1, tail_hours=2)
    assert mask[4:8].all() and not mask[29:33].any()


@pytest.mark.parametrize("kwargs, message", [
    (dict(rain_col="rain"), "no column"),
    (dict(interevent_hours=0), "interevent_hours"),
    (dict(min_rain_in=-1), "min_rain_in"),
    (dict(min_rain_in=np.nan), "min_rain_in"),
])
def test_bad_settings_raise(kwargs, message):
    with pytest.raises(ValueError, match=message):
        rain_storm_events(_data({5: 0.5}), **kwargs)


def test_bad_data_raises():
    data = _data({5: 0.5})
    with pytest.raises(ValueError, match="ordered"):
        rain_storm_events(data.iloc[::-1])
    data.loc[3, "rainfall_in"] = -0.1
    with pytest.raises(ValueError, match="negative"):
        rain_storm_events(data)
