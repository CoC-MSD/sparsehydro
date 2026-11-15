"""Shared pytest fixtures for the SparseHydro test suite.

Provides loaders for the ``MU-UM-019`` monitoring dataset (rainfall + stormflow
``.tsf`` files) and the co-located air-temperature CSV, plus a cached
event-hierarchy detection over a bounded window to keep tests fast.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sparsehydro.data.utilities import read_tsf

DATA_DIR = Path(__file__).parent / "data"
MUUM_DIR = DATA_DIR / "MU-UM-019"

MGD_TO_CFS = 1.5472286
# Bounded window (~3 weeks) with several storms — keeps fitting tests quick.
WINDOW = ("2022-07-01", "2022-07-22")


def _load_rain_stormflow() -> pd.DataFrame:
    """Load MU-UM-019 rain + stormflow, convert MGD→CFS, merge on the 5-min grid."""
    rain = read_tsf(str(MUUM_DIR / "rain_list.tsf")).iloc[:, 0].rename("rain")
    storm = (read_tsf(str(MUUM_DIR / "lstStormTs.tsf")).iloc[:, 0] * MGD_TO_CFS).rename("stormflow")
    df = pd.concat([rain, storm], axis=1, join="inner").reset_index()
    df.columns = ["datetime", "rain", "stormflow"]
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["stormflow"] = df["stormflow"].clip(lower=0.0)
    return df


@pytest.fixture(scope="session")
def muum019_full() -> pd.DataFrame:
    """MU-UM-019 rain + stormflow (CFS); the repo keeps 2022-07-01 to 2022-07-23 only."""
    return _load_rain_stormflow()


@pytest.fixture(scope="session")
def muum019_window(muum019_full: pd.DataFrame) -> pd.DataFrame:
    """MU-UM-019 rain + stormflow restricted to the bounded test window."""
    df = muum019_full
    mask = (df["datetime"] >= WINDOW[0]) & (df["datetime"] <= WINDOW[1])
    return df.loc[mask].reset_index(drop=True)


@pytest.fixture(scope="session")
def muum019_hierarchy(muum019_window: pd.DataFrame):
    """Detected ``(global_events, sub_events, savgol_result)`` over the window."""
    from sparsehydro.events import detect_event_hierarchy

    return detect_event_hierarchy(muum019_window)


@pytest.fixture(scope="session")
def synthetic_rain_stormflow() -> pd.DataFrame:
    """A short synthetic 5-min series with three gamma-response storms."""
    n = 6 * 24 * 12
    idx = pd.date_range("2023-05-01", periods=n, freq="5min")
    rng = np.random.default_rng(0)
    rain = np.zeros(n)
    for start, amt in [(120, 0.06), (900, 0.09), (2000, 0.05)]:
        rain[start:start + 8] = amt
    kernel = np.exp(-np.arange(60) / 8.0) * (np.arange(60) / 8.0)
    kernel /= kernel.sum()
    flow = np.convolve(rain, kernel * 200, mode="full")[:n] + rng.normal(0, 0.02, n).clip(0)
    return pd.DataFrame({"datetime": idx, "rain": rain, "stormflow": np.maximum(flow, 0.0)})
