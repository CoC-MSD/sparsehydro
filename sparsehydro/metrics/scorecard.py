"""Scorecards: criteria with pass and fail bands, combined into one grade.

A :class:`Criterion` states what a metric must achieve: a *pass* range, and
optionally a wider *fail* boundary (a value outside it fails outright). A
:class:`Scorecard` evaluates criteria on metric values and grades the whole:

``fail``
    any criterion fails;
``pass``
    every criterion passes;
``conditional``
    otherwise (some criterion is between its pass and fail bands, or is
    missing).

Nothing here is specific to a study; e.g. the MSDGC calibration rules
(:mod:`sparsehydro.studies.msdgc`) are three criteria on Lin's CCC (pass at
``>= 0.9``, no fail band) and two on percent errors (pass within 10 %, fail
beyond 20 %).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

__all__ = ["Criterion", "CriterionResult", "Scorecard", "ScorecardResult", "GRADES"]

#: Grades from best to worst.
GRADES = ("pass", "conditional", "fail")


@dataclass(frozen=True)
class Criterion:
    """A requirement on one metric.

    :param metric: Key of the value in the mapping given to :meth:`Scorecard.evaluate`.
    :param label: Readable name.
    :param pass_range: ``(lower, upper)`` within which the criterion passes
        (``None`` for an open side).
    :param fail_range: ``(lower, upper)`` outside which it fails; ``None``
        means it never fails outright (outside the pass range it is
        conditional).
    :param units: Units of the value (for display).
    :param required: A missing (NaN) value makes the grade conditional when
        true; when false the criterion is reported but not graded.
    """

    metric: str
    label: str = ""
    pass_range: tuple[float | None, float | None] = (None, None)
    fail_range: tuple[float | None, float | None] | None = None
    units: str = ""
    required: bool = True

    def status(self, value: float) -> str:
        """``"pass"``, ``"conditional"``, ``"fail"`` or ``"missing"`` (a non-finite value)."""
        if value is None or not math.isfinite(value):
            return "missing"
        if _inside(value, self.pass_range):
            return "pass"
        if self.fail_range is not None and not _inside(value, self.fail_range):
            return "fail"
        return "conditional"


def _inside(value: float, bounds: tuple[float | None, float | None]) -> bool:
    lo, hi = bounds
    eps = 1e-12
    return (lo is None or value >= lo - eps) and (hi is None or value <= hi + eps)


@dataclass(frozen=True)
class CriterionResult:
    criterion: Criterion
    value: float
    status: str


@dataclass(frozen=True)
class ScorecardResult:
    """Outcome of :meth:`Scorecard.evaluate`.

    :ivar grade: ``"pass"``, ``"conditional"`` or ``"fail"``.
    :ivar results: One :class:`CriterionResult` per criterion, in order.
    """

    grade: str
    results: tuple[CriterionResult, ...]

    def as_rows(self) -> list[dict]:
        """One dict per criterion (metric, label, value, units, pass/fail bounds, status) for tables."""
        rows = []
        for r in self.results:
            c = r.criterion
            rows.append({"metric": c.metric, "label": c.label or c.metric, "value": r.value, "units": c.units,
                         "pass_lower": c.pass_range[0], "pass_upper": c.pass_range[1],
                         "fail_lower": c.fail_range[0] if c.fail_range else None,
                         "fail_upper": c.fail_range[1] if c.fail_range else None,
                         "status": r.status, "required": c.required})
        return rows


@dataclass(frozen=True)
class Scorecard:
    """Criteria graded together (see the module docstring for the rule)."""

    criteria: tuple[Criterion, ...]
    name: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)

    def evaluate(self, values: Mapping[str, float]) -> ScorecardResult:
        """Grade *values* (``{metric: value}``; absent metrics are missing)."""
        results = tuple(CriterionResult(c, float(values.get(c.metric, math.nan)),
                                        c.status(float(values.get(c.metric, math.nan))))
                        for c in self.criteria)
        graded = [r.status for r in results if r.criterion.required]
        if "fail" in graded:
            grade = "fail"
        elif graded and all(s == "pass" for s in graded):
            grade = "pass"
        else:
            grade = "conditional"
        return ScorecardResult(grade, results)
