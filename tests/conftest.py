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
AIR_TEMP_CSV = DATA_DIR / "air_temperature.csv"

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


def _load_temperature() -> pd.DataFrame:
    """Load the air-temperature CSV as ``datetime`` (naive local) + ``temperature_c``."""
    temp = pd.read_csv(AIR_TEMP_CSV, comment="#")
    temp["Date_Time"] = (
        pd.to_datetime(temp["Date_Time"], utc=True)
        .dt.tz_convert("US/Eastern")
        .dt.tz_localize(None)
    )
    temp = temp[["Date_Time", "air_temp_set_1"]].dropna().sort_values("Date_Time")
    return temp.rename(columns={"Date_Time": "datetime", "air_temp_set_1": "temperature_c"})


@pytest.fixture(scope="session")
def muum019_full() -> pd.DataFrame:
    """Full MU-UM-019 rain + stormflow (CFS) series."""
    return _load_rain_stormflow()


@pytest.fixture(scope="session")
def muum019_window(muum019_full: pd.DataFrame) -> pd.DataFrame:
    """MU-UM-019 rain + stormflow restricted to the bounded test window."""
    df = muum019_full
    mask = (df["datetime"] >= WINDOW[0]) & (df["datetime"] <= WINDOW[1])
    return df.loc[mask].reset_index(drop=True)


@pytest.fixture(scope="session")
def muum019_window_temp(muum019_window: pd.DataFrame) -> pd.DataFrame:
    """MU-UM-019 window with an aligned ``temperature_c`` column (°C)."""
    temp = _load_temperature()
    merged = pd.merge_asof(
        muum019_window.sort_values("datetime"), temp, on="datetime", direction="nearest"
    )
    merged["temperature_c"] = merged["temperature_c"].astype(float)
    return merged.reset_index(drop=True)


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
