"""Scoring masks: which time steps an objective sees.

* :func:`dry_weather_mask` — steps with no rain in the preceding window.
* :func:`storm_scoring_mask` — event windows, padded before and with a recession tail.
* :func:`melt_scoring_mask` — windows following possible snowmelt pulses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def dry_weather_mask(data: pd.DataFrame, dt_hours: float, exclusion_hours: float) -> np.ndarray:
    """Exclude every positive rain pulse and an antecedent recession interval."""
    steps = int(np.ceil(exclusion_hours / dt_hours)) + 1
    wet = data["rainfall_in"].gt(0)
    return wet.rolling(steps, min_periods=steps).sum().eq(0).to_numpy()


def storm_scoring_mask(
    events: pd.DataFrame, split_mask: np.ndarray, dt_hours: float, *, split: str = "train",
    pre_hours: float = 3.0, tail_hours: float = 48.0,
) -> np.ndarray:
    """The original notebook's storm CCC window, built from the complete events.

    Each complete ``split`` event is padded by a pre-event buffer and a recession
    tail, then restricted to ``split_mask`` so padding never spills into another
    split. These are objective windows only; acceptance still uses complete events.
    """
    split_mask = np.asarray(split_mask, dtype=bool)
    if dt_hours <= 0 or pre_hours < 0 or tail_hours < 0:
        raise ValueError("Timestep must be positive and padding non-negative")
    n = len(split_mask)
    pre = int(round(pre_hours / dt_hours))
    tail = int(round(tail_hours / dt_hours))
    mask = np.zeros(n, dtype=bool)
    chosen = events[events["eligible"].astype(bool) & events["split"].eq(split)]
    for event in chosen.itertuples():
        mask[max(0, event.start_idx - pre):min(n, event.end_idx + tail + 1)] = True
    return mask & split_mask


def melt_scoring_mask(
    melt_in: np.ndarray, split_mask: np.ndarray, dt_hours: float, *,
    pre_hours: float = 3.0, tail_hours: float = 48.0,
) -> np.ndarray:
    """Pad every positive melt pulse, including rain-free thaw, then clip to split.

    Forcing-only windows supplement, but never redefine, MSD precipitation events.
    Antecedent melt may cross a split edge just as physical storage does; no
    observed flow is consulted in constructing these windows.
    """
    melt = np.asarray(melt_in, dtype=float)
    split = np.asarray(split_mask, dtype=bool)
    if (melt.ndim != 1 or split.shape != melt.shape or not np.isfinite(melt).all()
            or (melt < 0).any() or not np.isfinite([dt_hours, pre_hours, tail_hours]).all()
            or dt_hours <= 0 or pre_hours < 0 or tail_hours < 0):
        raise ValueError("Melt/mask must align; melt and padding nonnegative; timestep positive")
    pre, tail = int(round(pre_hours / dt_hours)), int(round(tail_hours / dt_hours))
    pulses = np.flatnonzero(melt > 0)
    changes = np.zeros(len(melt) + 1, dtype=np.int64)
    np.add.at(changes, np.maximum(0, pulses - pre), 1)
    np.add.at(changes, np.minimum(len(melt), pulses + tail + 1), -1)
    return (np.cumsum(changes[:-1]) > 0) & split


__all__ = ["dry_weather_mask", "storm_scoring_mask", "melt_scoring_mask"]
