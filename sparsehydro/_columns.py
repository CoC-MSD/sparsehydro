"""Picklable column readers shared by models and calibration problems."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class ColumnExtractor:
    """Picklable ``(data) -> ndarray`` reader for a single column.

    Binding the column name in a closure would be shorter, but a closure
    defined inside a function pickles by reference to a name that does not
    exist at module scope, so it cannot cross a process boundary. That blocks
    parallel calibration, where the problem is shipped to each worker once. A
    module-level class pickles as ``(class, state)`` and reconstructs fine.

    :param key: Column to read from a DataFrame or mapping of arrays.
    :param dtype: Optional dtype for the returned array; ``None`` keeps the
        column's own dtype.
    """

    __slots__ = ("key", "dtype")

    def __init__(self, key: str, dtype: Any = None) -> None:
        self.key = key
        self.dtype = dtype

    def __call__(self, data: Any) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            return data[self.key].to_numpy(dtype=self.dtype)
        return np.asarray(data[self.key], dtype=self.dtype)

    def __repr__(self) -> str:
        return f"ColumnExtractor({self.key!r})"

    def __getstate__(self) -> tuple[str, Any]:
        return self.key, self.dtype

    def __setstate__(self, state: Any) -> None:
        # Accept the bare-key state written by the earlier private classes.
        self.key, self.dtype = state if isinstance(state, tuple) else (state, None)

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, ColumnExtractor)
                and (self.key, self.dtype) == (other.key, other.dtype))

    def __hash__(self) -> int:
        return hash((self.key, self.dtype))
