"""Small argument validators shared across the package."""

from __future__ import annotations

from numbers import Integral

import numpy as np


def positive_integer(name: str, value: object, minimum: int = 1) -> None:
    """Raise ``ValueError`` unless *value* is an integer (not bool) >= *minimum*.

    NumPy integer types are accepted.
    """
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


class VerificationError(RuntimeError):
    """A run-time consistency check failed (a result does not agree with itself)."""


def verify_close(actual, expected, *, rtol: float = 1e-7, atol: float = 0.0, what: str = "values") -> None:
    """Raise :class:`VerificationError` unless *actual* and *expected* agree elementwise."""
    a = np.asarray(actual, dtype=float)
    b = np.asarray(expected, dtype=float)
    if a.shape != b.shape or not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
        diff = np.nanmax(np.abs(a - b)) if a.shape == b.shape and a.size else float("nan")
        raise VerificationError(f"{what} do not agree (max |difference| = {diff:.3g})")


def verify_equal(actual, expected, *, what: str = "values") -> None:
    """Raise :class:`VerificationError` unless *actual* and *expected* are identical."""
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or not np.array_equal(a, b, equal_nan=a.dtype.kind == "f"):
        raise VerificationError(f"{what} changed")

