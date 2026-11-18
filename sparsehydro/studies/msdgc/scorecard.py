"""MSDGC calibration scorecard for a hydrology monitor.

Implements the compliance check of the MSDGC Modeling Guidelines (Revision
5.8, October 2025), Volume II:

* **Storm volume** (Section 6.4.1): observed and modelled base flows are
  each adjusted, storm by storm, to match their total flow over the
  ``prior_hours`` (24 h) before the event, then subtracted; the **total
  volume** approach (6.4.2) compares total flows directly.
* **Zones** (6.4.3): the largest moving-window volume within each event over
  conveyance, overflow/storage and treatment windows whose lengths follow the
  event duration (:class:`~.assessment.MSDGCSettings`), found independently
  for observed and modelled flow.
* **Criteria** (7.1): Lin's concordance correlation coefficient of the zone
  volumes over all non-excluded events (pass at ``>= 0.9`` for each zone;
  their mean is the "Cinci score"), the critical flow volume Qcritical
  (volume above the system capacity; pass within ±10 %, fail beyond ±20 %)
  and Con15 (zero-intercept slope of modelled on observed conveyance volumes
  of the largest 15 % of events, as a percent error; same bands). Any fail
  fails the calibration, all passing passes it, anything else is a
  conditional pass.
* **Qcritical levels** (7.1.3, Table 7-2): Qc1 is a factor times the average
  observed dry weather flow (ODWF-A) — 2.5 / 1.5 (total volume, combined /
  sanitary system) or 1.5 / 0.5 (storm volume) — and Qc2..Qc4 divide the
  range from Qc1 to the largest observed event peak into three equal parts.
  The grade uses Qc1, the defined system capacity; Qc2..Qc4 are reported.
  Read literally, Qc4 is the largest observed peak, so no observed volume is
  above it and its error is undefined; the text also calls the analysis
  "quartile-based", which would place the levels at quarters of the range.
  Which the MSDGC tool does is to be confirmed against its output.
* **Reporting** (Table 7-3): CCCs for all months, the critical months
  (April-July) and the other months.

Inputs the guidelines take from elsewhere (event definitions and exclusions,
ODWF-A, the system type, the base flows) are parameters; whatever is missing
is reported in :attr:`MSDGCScorecardResult.warnings` and the result is then
marked not official.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sparsehydro.metrics.functions import paired_summary, peak_regression_slope, sum_above, top_peak_indices
from sparsehydro.metrics.scorecard import Criterion, Scorecard, ScorecardResult
from sparsehydro.units import CFS_H_TO_MG

from .assessment import MSDGCSettings, _zone_window

__all__ = [
    "GUIDELINE_ZONES",
    "TOOL_ZONES",
    "ZONE_LABELS",
    "MSDGCCriteria",
    "MSDGCScorecardResult",
    "assess_calibration",
    "event_adjusted_storm_flow",
    "qcritical_levels",
]

#: Zone windows of the MSDGC assessment tool v5.0.5 (treatment capped at 24 h).
TOOL_ZONES = MSDGCSettings()
#: Zone windows as written in the Guidelines Rev 5.8 (Table 6-1): no cap on the treatment zone.
GUIDELINE_ZONES = MSDGCSettings(maximum_minutes=(30.0, 240.0, 1.0e7), version="msdgc-guidelines-5.8-zones")
ZONE_LABELS = {"conveyance": "Conveyance", "storage": "Overflow/storage", "treatment": "Treatment"}


@dataclass(frozen=True)
class MSDGCCriteria:
    """Compliance thresholds (Guidelines Rev 5.8, Volume II Section 7.1)."""

    ccc_min: float = 0.90
    pass_percent: float = 10.0
    fail_percent: float = 20.0
    top_fraction: float = 0.15
    critical_months: tuple[int, ...] = (4, 5, 6, 7)
    prior_hours: float = 24.0
    #: Qc1 / ODWF-A by calibration volume approach and sewer system (Table 7-2).
    qcritical_factors: tuple[tuple[str, str, float], ...] = (
        ("total", "combined", 2.5), ("total", "sanitary", 1.5),
        ("storm", "combined", 1.5), ("storm", "sanitary", 0.5),
    )
    version: str = "msdgc-guidelines-5.8"

    def __post_init__(self) -> None:
        if not (0 < self.ccc_min <= 1 and 0 < self.pass_percent < self.fail_percent
                and 0 < self.top_fraction <= 1 and self.prior_hours >= 0):
            raise ValueError("Invalid MSDGC criteria")

    def qcritical_factor(self, approach: str, system: str) -> float:
        for a, s, f in self.qcritical_factors:
            if a == approach and s == system:
                return f
        raise KeyError(f"No Qcritical factor for approach {approach!r} and system {system!r}")

    def scorecard(self) -> Scorecard:
        p, f = self.pass_percent, self.fail_percent
        return Scorecard((
            Criterion("conveyance_ccc", "Lin conveyance (rcc)", (self.ccc_min, None)),
            Criterion("storage_ccc", "Lin overflow/storage (rco)", (self.ccc_min, None)),
            Criterion("treatment_ccc", "Lin treatment (rct)", (self.ccc_min, None)),
            Criterion("qcritical_pct", "Qcritical volume error", (-p, p), (-f, f), units="%"),
            Criterion("con15_pct", "Con15 peak-fit error", (-p, p), (-f, f), units="%"),
            Criterion("cinci_score", "Cinci score (mean rc)", (self.ccc_min, None), required=False),
        ), name=self.version)


def event_adjusted_storm_flow(total, base, events: pd.DataFrame, dt_hours: float,
                              prior_hours: float = 24.0) -> np.ndarray:
    """Storm flow with the base flow adjusted to each storm (Guidelines Section 6.4.1).

    For each event the base flow is shifted by the mean difference between
    total and base flow over the *prior_hours* before the event start, so the
    storm flow averages zero just before the storm; the shifted base flow is
    subtracted over the event. Outside events the unshifted base flow is
    subtracted. With ``base=None`` (flow that already has its base flow
    removed) only the pre-event shift is applied.
    """
    q = np.asarray(total, dtype=float)
    b = np.zeros_like(q) if base is None else np.asarray(base, dtype=float)
    if b.shape != q.shape:
        raise ValueError("Base flow must align with total flow")
    storm = q - b
    n_prior = int(round(prior_hours / dt_hours)) if dt_hours > 0 else 0
    for e in events.itertuples():
        start, end = int(e.start_idx), int(e.end_idx)
        pre = slice(max(0, start - n_prior), start)
        diff = (q[pre] - b[pre]) if n_prior else np.empty(0)
        diff = diff[np.isfinite(diff)]
        offset = float(diff.mean()) if diff.size else 0.0
        storm[start:end + 1] = q[start:end + 1] - (b[start:end + 1] + offset)
    return storm


def qcritical_levels(odwf_a: float, factor: float, upper: float) -> np.ndarray:
    """``[Qc1, Qc2, Qc3, Qc4]``: Qc1 = factor x ODWF-A, the rest split ``[Qc1, upper]`` in three."""
    qc1 = float(factor) * float(odwf_a)
    step = max(float(upper) - qc1, 0.0) / 3.0
    return qc1 + step * np.arange(4)


@dataclass
class MSDGCScorecardResult:
    """Everything the scorecard computed.

    :ivar events: One row per event: identifiers, dates, month, ``critical``,
        ``excluded``, ``included``, observed/modelled volume (MG) for each zone
        and the whole event, peaks (cfs), ``con15`` selection.
    :ivar summary: CCC per zone and Cinci score for ``all``, ``critical`` and
        ``non_critical`` months (Table 7-3), with event counts.
    :ivar qcritical: Qc1..Qc4: level (cfs), observed and modelled volume above
        it (MG) and percent difference.
    :ivar con15: ``slope``, ``percent``, ``n`` and the selected ``event_ids``.
    :ivar scorecard: Graded criteria (:class:`~sparsehydro.metrics.scorecard.ScorecardResult`).
    :ivar warnings: Why the result is not the official grade, or what was assumed.
    """

    events: pd.DataFrame
    summary: pd.DataFrame
    qcritical: pd.DataFrame
    con15: dict
    scorecard: ScorecardResult
    approach: str
    system: str
    criteria: MSDGCCriteria
    zones: MSDGCSettings
    warnings: list[str] = field(default_factory=list)

    @property
    def grade(self) -> str:
        return self.scorecard.grade

    @property
    def official(self) -> bool:
        """``True`` when nothing the guidelines require was missing or assumed."""
        return not self.warnings

    def criteria_table(self) -> pd.DataFrame:
        return pd.DataFrame(self.scorecard.as_rows())

    def zone_pairs(self, zone: str, group: str = "all") -> tuple[np.ndarray, np.ndarray]:
        """Observed and modelled volumes (MG) of the included events of *group* for a scatter plot."""
        ev = self.events[self.events["included"]]
        if group == "critical":
            ev = ev[ev["critical"]]
        elif group == "non_critical":
            ev = ev[~ev["critical"]]
        return ev[f"observed_{zone}_mg"].to_numpy(float), ev[f"modeled_{zone}_mg"].to_numpy(float)


def assess_calibration(
    observed, modeled, datetimes, events: pd.DataFrame, dt_hours: float, *,
    approach: str = "storm", system: str = "sanitary", odwf_a: float | None = None,
    observed_base=None, modeled_base=None,
    zones: MSDGCSettings = TOOL_ZONES, criteria: MSDGCCriteria = MSDGCCriteria(),
) -> MSDGCScorecardResult:
    """Grade a calibration the way the MSDGC guidelines do (see the module docstring).

    :param observed: Observed flow [cfs], total flow or (with ``base`` omitted) storm flow.
    :param modeled: Modelled flow on the same axis and basis.
    :param datetimes: Timestamps of the samples (for months).
    :param events: ``event_id``, ``start_idx``, ``end_idx``, ``duration_hours`` and
        optionally ``excluded`` (bool), e.g. from :func:`~.detector.find_events`.
    :param dt_hours: Time step [h].
    :param approach: ``"storm"`` (storm volume, the default) or ``"total"`` (total volume).
    :param system: ``"sanitary"`` or ``"combined"`` (selects the Qcritical factor).
    :param odwf_a: Average observed dry weather flow [cfs]; without it Qcritical is not graded.
    :param observed_base: Observed base flow (storm approach); ``None`` treats *observed*
        as flow whose base flow is already removed.
    :param modeled_base: Modelled base flow likewise (the model run with rain off).
    :param zones: Zone window rules (:data:`TOOL_ZONES` or :data:`GUIDELINE_ZONES`).
    :param criteria: Thresholds.
    """
    if approach not in ("storm", "total"):
        raise ValueError("approach must be 'storm' or 'total'")
    if system not in ("sanitary", "combined"):
        raise ValueError("system must be 'sanitary' or 'combined'")
    obs = np.asarray(observed, dtype=float)
    mod = np.asarray(modeled, dtype=float)
    if obs.shape != mod.shape or obs.ndim != 1:
        raise ValueError("Observed and modelled flow must be aligned 1-D series")
    times = pd.to_datetime(pd.Series(np.asarray(datetimes)))
    required = {"event_id", "start_idx", "end_idx", "duration_hours"}
    if not required <= set(events.columns):
        raise ValueError(f"events need columns {sorted(required)}")
    warnings: list[str] = []

    if approach == "storm":
        if observed_base is None or modeled_base is None:
            warnings.append("No base flow given: the flows are taken as storm flow (base flow already removed) and "
                            f"each event is shifted to a zero mean over the {criteria.prior_hours:g} h before it.")
        obs_f = event_adjusted_storm_flow(obs, observed_base, events, dt_hours, criteria.prior_hours)
        mod_f = event_adjusted_storm_flow(mod, modeled_base, events, dt_hours, criteria.prior_hours)
    else:
        obs_f, mod_f = obs, mod

    rows = []
    undefined = 0
    for e in events.to_dict("records"):
        start, end = int(e["start_idx"]), int(e["end_idx"])
        month = int(times.iloc[start].month)
        row = dict(event_id=e["event_id"], start_idx=start, end_idx=end, start=times.iloc[start], end=times.iloc[end],
                   duration_hours=float(e["duration_hours"]), month=month,
                   critical=month in criteria.critical_months, excluded=bool(e.get("excluded", False)))
        ok = True
        for zone in zones.zones:
            o = _zone_window(obs_f, start, end, row["duration_hours"], dt_hours, zones, zone)
            m = _zone_window(mod_f, start, end, row["duration_hours"], dt_hours, zones, zone)
            row[f"observed_{zone}_mg"], row[f"modeled_{zone}_mg"] = o["volume_mg"], m["volume_mg"]
            row[f"{zone}_minutes"] = o["effective_minutes"]
            ok = ok and o["status"] == "defined" and m["status"] == "defined"
        seg_o, seg_m = obs_f[start:end + 1], mod_f[start:end + 1]
        row["observed_event_mg"] = float(np.nansum(seg_o) * dt_hours * CFS_H_TO_MG)
        row["modeled_event_mg"] = float(np.nansum(seg_m) * dt_hours * CFS_H_TO_MG)
        row["observed_peak_cfs"] = float(np.nanmax(seg_o)) if seg_o.size else math.nan
        row["modeled_peak_cfs"] = float(np.nanmax(seg_m)) if seg_m.size else math.nan
        row["included"] = ok and not row["excluded"]
        undefined += not ok
        rows.append(row)
    table = pd.DataFrame(rows)
    if undefined:
        warnings.append(f"{undefined} event(s) have an undefined zone window (event shorter than the window, or "
                        "missing flow) and are left out.")
    if table.empty:
        table = pd.DataFrame(columns=["event_id", "included", "critical", "excluded"])
    included = table[table["included"]] if len(table) else table
    if len(included) < 2:
        warnings.append("Fewer than two events are included: the concordance correlations are undefined.")

    # Concordance correlations by month group (Table 7-3)
    summary_rows = []
    for group, part in (("all", included),
                        ("critical", included[included["critical"]] if len(included) else included),
                        ("non_critical", included[~included["critical"]] if len(included) else included)):
        srow = {"group": group, "n": int(len(part))}
        cccs = []
        for zone in zones.zones:
            value = (paired_summary(part[f"observed_{zone}_mg"], part[f"modeled_{zone}_mg"])["ccc"]
                     if len(part) else math.nan)
            srow[f"{zone}_ccc"] = value
            cccs.append(value)
        srow["cinci_score"] = float(np.mean(cccs)) if all(np.isfinite(cccs)) else math.nan
        summary_rows.append(srow)
    summary = pd.DataFrame(summary_rows)

    # Con15: zero-intercept slope over the largest conveyance volumes
    con15 = {"slope": math.nan, "percent": math.nan, "n": 0, "event_ids": []}
    table["con15"] = False
    if len(included):
        x = included["observed_conveyance_mg"].to_numpy(float)
        y = included["modeled_conveyance_mg"].to_numpy(float)
        sel = top_peak_indices(x, criteria.top_fraction)
        slope = peak_regression_slope(x[sel], y[sel])
        ids = included["event_id"].to_numpy()[sel].tolist()
        con15 = {"slope": slope, "percent": 100.0 * (slope - 1.0) if np.isfinite(slope) else math.nan,
                 "n": int(len(sel)), "event_ids": ids}
        table.loc[table["event_id"].isin(ids), "con15"] = True

    # Qcritical
    q_rows = []
    qcrit_pct = math.nan
    if odwf_a is None or not np.isfinite(odwf_a) or odwf_a <= 0:
        warnings.append("Average observed dry weather flow (ODWF-A) is not given: Qcritical cannot be computed.")
    elif len(included):
        factor = criteria.qcritical_factor(approach, system)
        upper = float(included["observed_peak_cfs"].max())
        levels = qcritical_levels(odwf_a, factor, upper)
        for i, level in enumerate(levels, 1):
            vo = vm = 0.0
            for e in included.itertuples():
                s = slice(int(e.start_idx), int(e.end_idx) + 1)
                vo += sum_above(obs_f[s], level) * dt_hours * CFS_H_TO_MG
                vm += sum_above(mod_f[s], level) * dt_hours * CFS_H_TO_MG
            pct = 100.0 * (vm - vo) / vo if vo > 0 else math.nan
            q_rows.append({"level": f"Qc{i}", "flow_cfs": float(level), "observed_mg": vo, "modeled_mg": vm,
                           "percent": pct})
        qcrit_pct = q_rows[0]["percent"]
        if not np.isfinite(qcrit_pct):
            warnings.append("No observed flow exceeds Qc1: the Qcritical volume error is undefined.")
    qcritical = pd.DataFrame(q_rows, columns=["level", "flow_cfs", "observed_mg", "modeled_mg", "percent"])

    all_row = summary.iloc[0]
    values = {f"{z}_ccc": float(all_row[f"{z}_ccc"]) for z in zones.zones}
    values.update(cinci_score=float(all_row["cinci_score"]), qcritical_pct=qcrit_pct, con15_pct=con15["percent"])
    result = criteria.scorecard().evaluate(values)
    return MSDGCScorecardResult(table, summary, qcritical, con15, result, approach, system, criteria, zones,
                                warnings)
