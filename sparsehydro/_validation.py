"""Small argument validators shared across the package."""

from __future__ import annotations

from numbers import Integral


def positive_integer(name: str, value: object, minimum: int = 1) -> None:
    """Raise ``ValueError`` unless *value* is an integer (not bool) >= *minimum*.

    NumPy integer types are accepted.
    """
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
