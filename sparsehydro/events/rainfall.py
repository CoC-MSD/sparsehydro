"""sparsehydro.events.rainfall: storm events found from the rainfall alone.

:func:`rain_storm_events` groups wet time steps into storms and does not look
at the flow to find them. This helps at sites such as pump stations, where the
flow is too noisy for :func:`~sparsehydro.events.detect_events`. Its table
goes straight into
:func:`sparsehydro.calibration.splits.masks.storm_scoring_mask`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["rain_storm_events"]

_COLUMNS = [
    "event_id", "start_idx", "end_idx", "rain_start", "rain_end", "rain_in",
    "peak_rain_in", "eligible", "split",
]


def rain_storm_events(
    data: pd.DataFrame,
    *,
    rain_col: str = "rainfall_in",
    min_rain_in: float = 0.25,
    interevent_hours: float = 6.0,
    split_at=None,
) -> pd.DataFrame:
    """Find storms from the rainfall alone, one row for each storm.

    Every step with rain above zero is a wet step, and a missing rain value
    counts as dry. A new storm starts when the time from one wet step to the
    next is more than *interevent_hours*. Storms with less than *min_rain_in*
    of rain in total are dropped. A storm whose rain starts before *split_at*
    goes to ``"train"`` and any other storm to ``"validation"``; with no
    *split_at*, every storm goes to ``"train"``. Every storm starts with
    ``eligible = True``; set it to ``False`` to leave a storm out of
    :func:`~sparsehydro.calibration.splits.masks.storm_scoring_mask`, for
    example when too much of its flow is missing.

    :param data: Frame with a ``datetime`` column in time order and a rain column.
    :type data: pandas.DataFrame
    :param rain_col: Name of the rain column (depth in each step, any unit).
    :type rain_col: str
    :param min_rain_in: Smallest storm rain total to keep, in the unit of *rain_col*.
    :type min_rain_in: float
    :param interevent_hours: A longer time than this between two wet steps
        starts a new storm, in hours.
    :type interevent_hours: float
    :param split_at: First time of the validation period, or ``None``.
    :type split_at: str or pandas.Timestamp or None
    :returns: One row for each storm with ``event_id`` (from 1), ``start_idx``
        and ``end_idx`` (row positions of the first and last wet step),
        ``rain_start``, ``rain_end`` (one time step after the last wet step),
        ``rain_in``, ``peak_rain_in`` (largest single step), ``eligible`` and
        ``split``.
    :rtype: pandas.DataFrame
    :raises ValueError: If a column is missing, the times are not in order and
        unique, the rain is negative, or a setting is out of range.

    Example::

        times = pd.date_range("2025-01-01", periods=12, freq="h")
        rain = [0, 0.5, 0.25, 0, 0, 0, 0, 0, 0, 0.1, 0, 0]
        data = pd.DataFrame({"datetime": times, "rainfall_in": rain})
        storms = rain_storm_events(data)
        # One storm: rows 1 to 2, 0.75 in, from 01:00 to 03:00.
        # The 0.1 in at 09:00 is below 0.25 in, so it is dropped.
        mask = storm_scoring_mask(storms, np.ones(12, bool), 1.0,
                                  pre_hours=1, tail_hours=2)
        # mask is True for rows 0 to 4
    """
    for name in ("datetime", rain_col):
        if name not in data.columns:
            raise ValueError(f"data has no column {name!r}")
    times = pd.DatetimeIndex(data["datetime"])
    if len(times) < 2 or times.hasnans or times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError("Timestamps must be ordered, unique and at least two")
    if not np.isfinite(min_rain_in) or min_rain_in < 0:
        raise ValueError("min_rain_in must be a finite number, zero or more")
    if not np.isfinite(interevent_hours) or interevent_hours <= 0:
        raise ValueError("interevent_hours must be a finite number above zero")
    rain = np.nan_to_num(data[rain_col].to_numpy(dtype=float), nan=0.0)
    if (rain < 0).any():
        raise ValueError(f"{rain_col} cannot be negative")
    step = pd.Series(times).diff().median()

    # Wet steps, and a dry gap longer than interevent_hours starts a new storm
    wet = np.flatnonzero(rain > 0)
    wet_times = times[wet]
    new_storm = np.asarray(wet_times[1:] - wet_times[:-1] > pd.Timedelta(hours=interevent_hours))
    rows = []
    for steps in np.split(wet, np.flatnonzero(new_storm) + 1) if len(wet) else []:
        rows.append(dict(
            start_idx=int(steps[0]), end_idx=int(steps[-1]),
            rain_start=times[steps[0]], rain_end=times[steps[-1]] + step,
            rain_in=float(rain[steps].sum()), peak_rain_in=float(rain[steps].max()),
        ))
    storms = pd.DataFrame(rows, columns=_COLUMNS[1:7])

    # Drop the small storms and number the rest from 1
    storms = storms[storms["rain_in"] >= min_rain_in].reset_index(drop=True)
    storms.insert(0, "event_id", np.arange(1, len(storms) + 1))

    # Every storm can be scored (set eligible to False to leave one out); train before split_at
    storms["eligible"] = True
    if split_at is None:
        storms["split"] = "train"
    else:
        storms["split"] = np.where(storms["rain_start"] < pd.Timestamp(split_at), "train", "validation")
    return storms[_COLUMNS]
