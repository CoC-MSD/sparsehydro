"""Shared unit-hydrograph kernel helpers and truncation constants.

Unit hydrographs in sparsehydro are defined by the cumulative distribution
``F(t)`` of a unit-area instantaneous response, with ``t`` in **hours**. The
discrete kernel for a time step ``dt`` is the exact bin average::

    k[j] = (F((j + 1) * dt) - F(j * dt)) / dt        [1/hr]

so ``sum(k) * dt == 1`` (to rounding) and the response shape does not depend on
``dt``. Convolving rainfall depth per step with ``k`` gives the instantaneous
outflow rate at the end of each step for rain falling uniformly within each
step.

Responses with infinite support are truncated at the ``1 - TAIL_EPS`` quantile
or ``MAX_DURATION_HOURS``, whichever comes first, and renormalised.

``MAX_STEPS``, ``finalize_kernel`` and ``normalize_kernel`` belong to the
earlier step-based kernels and are kept only for code that imported them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Tail mass left out when an infinite-support response is truncated.
TAIL_EPS: float = 1e-6

#: Longest response retained [hr] (30 days).
MAX_DURATION_HOURS: float = 720.0

#: Hard cap on the number of kernel ordinates.
MAX_ORDINATES: int = 500_000

#: Deprecated: length cap of the earlier step-based kernels (72 h at 5 min).
MAX_STEPS: int = 864

#: Cumulative-mass fraction at which to truncate a kernel's tail, or ``None`` to
#: keep the full support.
#:
#: ``None`` by default, deliberately.  ``docs/unithydrograph_strategy.md``
#: specifies truncating at 99.9%, but measured on this project's kernels it does
#: not pay: because ``max_steps`` is already set near ``5 * tp``, kernels are cut
#: close to their natural support and there is no long negligible tail left to
#: trim.  Truncating at 99.9% shrinks them 0-3% while shifting predicted flow by
#: ~1.9e-3 relative, since renormalising after dropping 0.1% of the mass scales
#: every surviving ordinate by ~1/0.999.  With convolution at ~1% of calibration
#: runtime, a 3% shorter kernel is ~0.03% overall -- a real numerical change for
#: no measurable gain.
#:
#: The machinery is here so the trade can be revisited per-call or flipped
#: globally; nothing else needs to change.
DEFAULT_MASS_FRACTION: float | None = None


def infer_dt_hours(data: pd.DataFrame) -> float:
    """Infer the timestep size from a frame's ``datetime`` column.

    :param data: DataFrame containing a ``datetime`` column.
    :type data: pandas.DataFrame
    :returns: Median timestep size [hr]; ``1/12`` (5 min) for fewer than 2 rows.
    :rtype: float
    """
    if len(data) < 2:
        return 1.0 / 12.0
    dt = data["datetime"].diff().median()
    return dt.total_seconds() / 3600.0


def infer_dt_hours_from_values(values: np.ndarray) -> float:
    """Infer the timestep size from a ``datetime64`` array.

    The ndarray equivalent of :func:`infer_dt_hours`, avoiding the pandas
    ``Series.diff().median()`` round-trip on the calibration hot path.

    The final division is written as ``(ns / 1e9) / 3600.0`` rather than the
    tidier ``ns / 3.6e12`` so it reproduces ``Timedelta.total_seconds() / 3600``
    to the last bit -- the two spellings differ by an ulp for some inputs, and
    ``dt_hours`` scales every kernel ordinate.

    :param values: 1-D ``datetime64`` array.
    :type values: numpy.ndarray
    :returns: Median timestep size [hr]; ``1/12`` (5 min) for fewer than 2
        elements or a non-positive median.
    :rtype: float
    """
    if values is None or len(values) < 2:
        return 1.0 / 12.0
    diffs = np.diff(np.asarray(values, dtype="datetime64[ns]").view("i8"))
    if diffs.size == 0:
        return 1.0 / 12.0
    median_ns = float(np.median(diffs))
    if not np.isfinite(median_ns) or median_ns <= 0.0:
        return 1.0 / 12.0
    return (median_ns / 1e9) / 3600.0


def normalize_kernel(raw: np.ndarray, dt_hours: float) -> np.ndarray:
    """Normalise ordinates so ``sum(result) * dt_hours ~ 1.0``.

    :param raw: Unnormalised kernel ordinates.
    :type raw: numpy.ndarray
    :param dt_hours: Timestep size [hr].
    :type dt_hours: float
    :returns: Normalised kernel [1/hr]; all zeros if the raw sum is non-positive.
    :rtype: numpy.ndarray
    """
    total = float(np.sum(raw))
    if total <= 0.0:
        return np.zeros_like(raw)
    return raw / (total * dt_hours)


def trim_pad(arr: np.ndarray, n_steps: int | None) -> np.ndarray:
    """Trim or zero-pad *arr* to exactly *n_steps* elements.

    :param arr: Input array.
    :type arr: numpy.ndarray
    :param n_steps: Target length, or ``None`` to leave *arr* unchanged.
    :type n_steps: int | None
    :returns: Array of length *n_steps* (or *arr* unchanged when *n_steps* is
        ``None``).
    :rtype: numpy.ndarray
    """
    if n_steps is None:
        return arr
    if len(arr) >= n_steps:
        return arr[:n_steps]
    return np.pad(arr, (0, n_steps - len(arr)))


def kernel_errstate() -> np.errstate:
    """Return the numpy error state to evaluate raw kernel shapes under.

    Kernel shapes are closed forms like ``(t/tp) ** tt * exp(-t/tp)`` evaluated
    over the whole parameter box, so an optimiser probing a corner will overflow
    intermediates.  That is expected and handled -- :func:`finalize_kernel`
    sanitises the result -- so the warnings are noise, and noise emitted once per
    objective evaluation is worse than noise.

    :returns: Context manager suppressing overflow, invalid and divide warnings.
    :rtype: numpy.errstate
    """
    return np.errstate(over="ignore", invalid="ignore", divide="ignore")


def finalize_kernel(
    raw: np.ndarray,
    dt_hours: float,
    n_steps: int | None = None,
    *,
    max_steps: int | None = MAX_STEPS,
    mass_fraction: float | None = DEFAULT_MASS_FRACTION,
) -> np.ndarray:
    """Sanitise, bound, normalise and size a raw kernel.

    The single exit point every ``get_kernel`` goes through.  Step order is
    load-bearing:

    1. **Sanitise.**  Replace NaN/inf with zero and clamp negatives away.  This
       is what stops a parameter corner from poisoning a whole calibration:
       ``NashUH(n=100, k=500)`` evaluates ``t ** 99`` for ``t`` up to 250,000,
       which overflows to ``inf``, then ``inf * exp(-t/k)`` gives ``nan``.  The
       old code summed that to ``nan``, and since ``nan <= 0.0`` is ``False`` it
       divided by ``nan`` and returned an all-NaN kernel -- 250,001 elements,
       ~125 ms to build and convolve, and a result the optimiser could only ever
       read as the generic 1e12 penalty.
    2. **Cap** to *max_steps*, before normalising, so the unit-area invariant
       holds over the support actually retained.
    3. **Truncate** at *mass_fraction* of cumulative mass, when enabled.
    4. **Normalise** so ``sum * dt_hours ~ 1``.
    5. **Trim or pad** to *n_steps*.

    :param raw: Unnormalised ordinates.
    :type raw: numpy.ndarray
    :param dt_hours: Timestep size [hr].
    :type dt_hours: float
    :param n_steps: Target output length, or ``None`` for the natural support.
    :type n_steps: int or None
    :param max_steps: Hard length cap, or ``None`` to leave the length alone.
    :type max_steps: int or None
    :param mass_fraction: Cumulative-mass fraction to retain, or ``None`` to keep
        the full support.  See :data:`DEFAULT_MASS_FRACTION`.
    :type mass_fraction: float or None
    :returns: Normalised, non-negative, finite ordinates [1/hr].
    :rtype: numpy.ndarray
    """
    raw = np.asarray(raw, dtype=float)
    # Probe before rewriting: both checks are allocation-free single passes, and
    # a healthy kernel -- the overwhelmingly common case on the hot path -- then
    # skips two full-array copies.
    if not np.isfinite(raw.sum()):
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    if raw.size and raw.min() < 0.0:
        raw = np.maximum(raw, 0.0)

    if max_steps is not None and raw.shape[0] > max_steps:
        raw = raw[:max_steps]

    if mass_fraction is not None and 0.0 < mass_fraction < 1.0 and raw.size:
        cumulative = np.cumsum(raw)
        total = cumulative[-1]
        if total > 0.0:
            keep = int(np.searchsorted(cumulative, mass_fraction * total)) + 1
            raw = raw[:keep]

    return trim_pad(normalize_kernel(raw, dt_hours), n_steps)


__all__ = [
    "TAIL_EPS",
    "MAX_DURATION_HOURS",
    "MAX_ORDINATES",
    "MAX_STEPS",
    "DEFAULT_MASS_FRACTION",
    "finalize_kernel",
    "kernel_errstate",
    "infer_dt_hours",
    "infer_dt_hours_from_values",
    "normalize_kernel",
    "trim_pad",
]
