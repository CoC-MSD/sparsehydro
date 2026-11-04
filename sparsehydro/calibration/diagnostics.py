"""Parameter bounds-proximity diagnostics for sparsehydro calibration.

A calibrated parameter whose value lands at (or very near) one of its search
bounds is a warning sign: the optimizer wanted a value the bounds did not
allow, so the parameter is often absorbing a structural model deficiency
rather than describing the watershed.  A parameter at a bound can, however,
legitimately be the physical answer, so these functions **report** proximity
and never reject anything on their own.

Two entry points are provided:

* :func:`parameters_near_bounds` inspects a single model's *current* parameter
  values against their bounds.
* :func:`pareto_parameters_near_bounds` scans every solution on a
  :class:`~sparsehydro.calibration.result.CalibrationResult` Pareto front.

:func:`warn_if_near_bounds` turns either diagnostic frame into a one-line
warning summary.

Proximity is measured **linearly** as a fraction of the ``upper - lower``
span, matching :meth:`sparsehydro.parameters.ScalarParameter.normalize`.  The
package carries no log-scale metadata on parameters, so log-space proximity is
intentionally out of scope.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..models.base import IModel
    from .result import CalibrationResult


def _bound_proximity(
    value: float, lower: float, upper: float
) -> tuple[float, str | None]:
    """Fractional distance of *value* from its nearest bound.

    The fraction is ``min(value - lower, upper - value) / (upper - lower)``:
    ``0.0`` sits exactly on a bound, ``0.5`` sits at the midpoint.  A
    degenerate or inverted span (``upper <= lower``) is treated as sitting on
    the bound (fraction ``0.0``), mirroring the ``span == 0`` guard in
    :meth:`ScalarParameter.normalize`.

    :param value: Parameter value to test.
    :type value: float
    :param lower: Lower bound.
    :type lower: float
    :param upper: Upper bound.
    :type upper: float
    :returns: ``(fraction, which)`` where *fraction* is the distance from the
        nearest bound as a fraction of the span, and *which* is ``"lower"`` or
        ``"upper"`` for the nearer bound (``"lower"`` on a tie or degenerate
        span).
    :rtype: tuple[float, str | None]
    """
    span = upper - lower
    if span <= 0.0:
        return 0.0, "lower"
    dist_lower = value - lower
    dist_upper = upper - value
    if dist_lower <= dist_upper:
        return dist_lower / span, "lower"
    return dist_upper / span, "upper"


def parameters_near_bounds(
    model: "IModel",
    *,
    warn_pct: float = 5.0,
    calibrated_only: bool = True,
) -> pd.DataFrame:
    """Flag a model's current parameter values that sit near a bound.

    Scalar parameters and the elements of vector parameters are inspected.
    Vector elements are reported with flattened ``name[i]`` identifiers,
    matching the decision-vector convention used by
    :class:`~sparsehydro.calibration.problem.CalibrationProblem`.

    :param model: Any model exposing the standard parameter-enumeration API
        (:attr:`scalar_parameter_names`, :meth:`get_scalar_parameter`, etc.).
    :type model: sparsehydro.models.base.IModel
    :param warn_pct: Proximity threshold as a percentage of the ``upper -
        lower`` span.  A parameter is flagged when its distance to the nearest
        bound is at most this fraction of the span.  Defaults to ``5.0``.
    :type warn_pct: float
    :param calibrated_only: When ``True`` (default) only parameters with
        ``calibrate=True`` are examined; fixed parameters cannot be "pinned"
        by an optimizer and are skipped.
    :type calibrated_only: bool
    :returns: One row per examined parameter with columns ``parameter``,
        ``value``, ``lower_bound``, ``upper_bound``, ``units``,
        ``distance_pct`` (``100 x`` the fractional distance), ``near_bound``
        (``bool``) and ``bound`` (``"lower"``/``"upper"``, or ``""`` when not
        flagged).  Filter with ``df[df.near_bound]`` for just the flagged rows.
    :rtype: pandas.DataFrame
    """
    frac = warn_pct / 100.0
    rows: list[dict[str, Any]] = []

    for name in model.scalar_parameter_names:
        p = model.get_scalar_parameter(name)
        if calibrated_only and not p.calibrate:
            continue
        dist, which = _bound_proximity(p.value, p.lower_bound, p.upper_bound)
        near = dist <= frac
        rows.append({
            "parameter":   name,
            "value":       p.value,
            "lower_bound": p.lower_bound,
            "upper_bound": p.upper_bound,
            "units":       p.units,
            "distance_pct": 100.0 * dist,
            "near_bound":  near,
            "bound":       which if near else "",
        })

    for name in model.vector_parameter_names:
        vp = model.get_vector_parameter(name)
        if calibrated_only and not vp.calibrate:
            continue
        for i in range(vp.size):
            lo = float(vp.lower_bounds[i])
            hi = float(vp.upper_bounds[i])
            val = float(vp.values[i])
            dist, which = _bound_proximity(val, lo, hi)
            near = dist <= frac
            rows.append({
                "parameter":   f"{name}[{i}]",
                "value":       val,
                "lower_bound": lo,
                "upper_bound": hi,
                "units":       vp.units,
                "distance_pct": 100.0 * dist,
                "near_bound":  near,
                "bound":       which if near else "",
            })

    return pd.DataFrame(
        rows,
        columns=[
            "parameter", "value", "lower_bound", "upper_bound", "units",
            "distance_pct", "near_bound", "bound",
        ],
    )


def pareto_parameters_near_bounds(
    result: "CalibrationResult",
    bounds: "tuple[np.ndarray, np.ndarray] | Any",
    *,
    warn_pct: float = 5.0,
) -> pd.DataFrame:
    """Flag near-bound parameters across every Pareto-front solution.

    Each column of ``result.pareto_X`` corresponds positionally to
    ``result.param_names`` and to the supplied lower/upper bound arrays (this
    is the ordering produced by
    :class:`~sparsehydro.calibration.problem.CalibrationProblem`).  Every
    ``(solution, parameter)`` pair whose value lands within *warn_pct* of a
    bound is returned as one row.

    :param result: A calibration result holding ``pareto_X`` and
        ``param_names``.
    :type result: sparsehydro.calibration.result.CalibrationResult
    :param bounds: Either the ``(lower, upper)`` arrays returned by
        :attr:`CalibrationProblem.bounds`, or a ``CalibrationProblem`` (any
        object exposing a ``bounds`` attribute/property yielding that tuple),
        column-aligned with ``result.pareto_X``.
    :type bounds: tuple[numpy.ndarray, numpy.ndarray] | Any
    :param warn_pct: Proximity threshold as a percentage of each parameter's
        span.  Defaults to ``5.0``.
    :type warn_pct: float
    :returns: One row per flagged ``(solution, parameter)`` pair with columns
        ``solution`` (Pareto row index), ``parameter``, ``value``,
        ``lower_bound``, ``upper_bound``, ``distance_pct`` and ``bound``
        (``"lower"``/``"upper"``).  Empty (with those columns) when nothing is
        near a bound.
    :rtype: pandas.DataFrame
    :raises ValueError: If the bounds length does not match the number of
        parameter columns.
    """
    # Accept a CalibrationProblem (or anything exposing `.bounds`) directly.
    if hasattr(bounds, "bounds"):
        lower, upper = bounds.bounds
    else:
        lower, upper = bounds
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    X = result.pareto_X
    names = result.param_names
    n_params = X.shape[1] if X.ndim == 2 else 0

    if lower.shape[0] != n_params or upper.shape[0] != n_params:
        raise ValueError(
            f"bounds length ({lower.shape[0]}, {upper.shape[0]}) does not match "
            f"the number of parameter columns ({n_params})."
        )

    frac = warn_pct / 100.0
    rows: list[dict[str, Any]] = []

    for s in range(X.shape[0]):
        for i in range(n_params):
            val = float(X[s, i])
            lo = float(lower[i])
            hi = float(upper[i])
            dist, which = _bound_proximity(val, lo, hi)
            if dist <= frac:
                rows.append({
                    "solution":    s,
                    "parameter":   names[i],
                    "value":       val,
                    "lower_bound": lo,
                    "upper_bound": hi,
                    "distance_pct": 100.0 * dist,
                    "bound":       which,
                })

    return pd.DataFrame(
        rows,
        columns=[
            "solution", "parameter", "value", "lower_bound", "upper_bound",
            "distance_pct", "bound",
        ],
    )


def warn_if_near_bounds(
    df: pd.DataFrame,
    *,
    warn_pct: float = 5.0,
    logger: logging.Logger | None = None,
) -> int:
    """Emit a one-line warning summarising near-bound parameters.

    Accepts a frame from either :func:`parameters_near_bounds` (which has a
    ``near_bound`` column) or :func:`pareto_parameters_near_bounds` (whose
    every row is already a flagged pair).  The distinct near-bound parameter
    names are collected and, if any exist, reported as a single WARNING line.

    :param df: A diagnostic frame from one of the functions in this module.
    :type df: pandas.DataFrame
    :param warn_pct: Threshold echoed in the message text (informational).
        Defaults to ``5.0``.
    :type warn_pct: float
    :param logger: Logger to emit the warning through.  When ``None``
        (default) the message is printed to stdout instead.
    :type logger: logging.Logger | None
    :returns: The number of distinct parameters flagged.
    :rtype: int
    """
    if "near_bound" in df.columns:
        flagged = df.loc[df["near_bound"], "parameter"]
    else:
        flagged = df["parameter"] if "parameter" in df.columns else pd.Series(dtype=object)

    names = list(dict.fromkeys(flagged.tolist()))  # distinct, order-preserving
    if not names:
        return 0

    message = (
        f"*** WARNING - {len(names)} parameter(s) within {warn_pct:g}% of a "
        f"search bound: {', '.join(str(n) for n in names)}"
    )
    if logger is not None:
        logger.warning(message)
    else:
        print(message)
    return len(names)
