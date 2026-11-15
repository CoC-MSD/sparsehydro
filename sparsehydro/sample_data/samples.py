"""Lookup and loading of the bundled sample datasets."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent

_SAMPLES = {
    "tc-nb-001": "tc_nb_001_rain_stormflow.csv",
}


def sample_names() -> list[str]:
    """Return the names accepted by :func:`load_sample`."""
    return sorted(_SAMPLES)


def sample_path(name: str = "tc-nb-001") -> Path:
    """Return the file path of a bundled sample.

    :raises KeyError: If *name* is not a bundled sample.
    """
    try:
        return _HERE / _SAMPLES[name]
    except KeyError:
        raise KeyError(f"Unknown sample {name!r}; available: {sample_names()}") from None


def load_sample(name: str = "tc-nb-001") -> pd.DataFrame:
    """Load a bundled sample as a DataFrame with a parsed ``datetime`` column."""
    df = pd.read_csv(sample_path(name))
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df
