"""MSDGC calibration scorecard (Guidelines Rev 5.8, Volume II Sections 6.4 and 7.1)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from sparsehydro.studies.msdgc.assessment import MSDGCAssessment
from sparsehydro.studies.msdgc.scorecard import (
    GUIDELINE_ZONES,
    TOOL_ZONES,
    MSDGCCriteria,
    assess_calibration,
    event_adjusted_storm_flow,
    qcritical_levels,
)

DT = 5 / 60


def synthetic(n_events: int = 12, seed: int = 3):
    """Storm flow with *n_events* triangular storms of varied size, spread over April to October."""
    rng = np.random.default_rng(seed)
    n = n_events * 600
    t = pd.date_range("2023-04-01", periods=n, freq="5min") + pd.to_timedelta(
        np.repeat(np.arange(n_events) * 15, 600), unit="D")
    q = np.zeros(n)
    rows = []
    for k in range(n_events):
        start = k * 600 + 100
        dur = int(rng.integers(60, 300))
        peak = float(rng.uniform(1, 20))
        rise = dur // 4
        q[start:start + rise] = np.linspace(0, peak, rise)
        q[start + rise:start + dur] = np.linspace(peak, 0, dur - rise)
        rows.append({"event_id": k + 1, "start_idx": start, "end_idx": start + dur,
                     "duration_hours": dur * DT})
    return q, t, pd.DataFrame(rows)


def test_perfect_model_passes() -> None:
    q, t, ev = synthetic()
    res = assess_calibration(q, q.copy(), t, ev, DT, odwf_a=1.0)
    assert res.grade == "pass"
    assert res.summary.set_index("group").loc["all", ["conveyance_ccc", "storage_ccc", "treatment_ccc"]].tolist() == \
        pytest.approx([1.0, 1.0, 1.0])
    assert res.con15["percent"] == pytest.approx(0.0) and res.con15["n"] == 2  # ceil(15 % of 12)
    assert res.qcritical["percent"].iloc[:3].tolist() == pytest.approx([0.0] * 3)
    assert math.isnan(res.qcritical["percent"].iloc[3])  # Qc4 is the largest observed peak: nothing is above it
    assert len(res.events) == 12 and res.events["included"].all()
    # storm flow given without base flows: allowed, but said
    assert not res.official and "storm flow" in res.warnings[0]


@pytest.mark.parametrize("scale, grade", [(1.05, "pass"), (1.15, "conditional"), (1.30, "fail")])
def test_bias_moves_the_grade(scale, grade) -> None:
    q, t, ev = synthetic()
    res = assess_calibration(q, scale * q, t, ev, DT, odwf_a=1.0)
    # zone volumes are rounded to 4 decimals of MG, as the MSDGC tool reports them
    assert res.con15["percent"] == pytest.approx(100 * (scale - 1), abs=0.1)
    assert res.grade == grade
    statuses = {r["metric"]: r["status"] for r in res.scorecard.as_rows()}
    assert statuses["conveyance_ccc"] == ("pass" if scale <= 1.05 else statuses["conveyance_ccc"])


def test_missing_odwf_leaves_qcritical_ungraded() -> None:
    q, t, ev = synthetic()
    res = assess_calibration(q, q, t, ev, DT)
    assert res.qcritical.empty and res.grade == "conditional"
    assert any("ODWF-A" in w for w in res.warnings)
    assert {r["metric"]: r["status"] for r in res.scorecard.as_rows()}["qcritical_pct"] == "missing"


def test_storm_flow_is_shifted_to_zero_before_each_event() -> None:
    q, t, ev = synthetic(3)
    raised = q + 0.7  # e.g. an uncorrected base flow under the storms
    sf = event_adjusted_storm_flow(raised, None, ev, DT, prior_hours=6)
    for e in ev.itertuples():
        np.testing.assert_allclose(sf[e.start_idx:e.end_idx + 1], q[e.start_idx:e.end_idx + 1], atol=1e-12)
    base = np.full_like(q, 0.5)
    sf2 = event_adjusted_storm_flow(raised, base, ev, DT, prior_hours=6)
    for e in ev.itertuples():  # base + per-event offset removes the same 0.7 under storms
        np.testing.assert_allclose(sf2[e.start_idx:e.end_idx + 1], sf[e.start_idx:e.end_idx + 1], atol=1e-12)
    # assessed on storm flow, a constant shift of the model changes nothing
    res = assess_calibration(q, q + 3.0, t, ev, DT, odwf_a=1.0)
    assert res.con15["percent"] == pytest.approx(0.0, abs=1e-9)


def test_qcritical_levels_follow_table_7_2() -> None:
    c = MSDGCCriteria()
    assert c.qcritical_factor("total", "combined") == 2.5 and c.qcritical_factor("storm", "sanitary") == 0.5
    np.testing.assert_allclose(qcritical_levels(2.0, 1.5, 12.0), [3.0, 6.0, 9.0, 12.0])
    q, t, ev = synthetic()
    res = assess_calibration(q, 0.8 * q, t, ev, DT, odwf_a=2.0, approach="storm", system="combined")
    assert res.qcritical["flow_cfs"].iloc[0] == pytest.approx(3.0)  # 1.5 x ODWF-A
    assert res.qcritical["flow_cfs"].iloc[-1] == pytest.approx(res.events["observed_peak_cfs"].max())
    assert res.qcritical["percent"].iloc[0] < -20 and res.grade == "fail"


def test_month_groups_exclusions_and_undefined_windows() -> None:
    q, t, ev = synthetic()
    ev["excluded"] = False
    ev.loc[0, "excluded"] = True
    res = assess_calibration(q, q, t, ev, DT, odwf_a=1.0)
    groups = res.summary.set_index("group")["n"]
    assert groups["all"] == 11 and groups["critical"] + groups["non_critical"] == 11
    crit = res.events[res.events["included"] & res.events["critical"]]
    assert set(crit["month"]) <= {4, 5, 6, 7}
    assert not res.events.loc[0, "included"]
    # an event shorter than its window is left out with a warning
    short = ev.copy()
    short.loc[1, "end_idx"] = short.loc[1, "start_idx"] + 1
    res2 = assess_calibration(q, q, t, short, DT, odwf_a=1.0)
    assert not res2.events.loc[1, "included"] and any("undefined zone window" in w for w in res2.warnings)


def test_total_flow_zone_cccs_match_the_existing_assessment() -> None:
    q, t, ev = synthetic()
    model = 0.9 * q + 0.05 * np.roll(q, 6)
    res = assess_calibration(q, model, t, ev, DT, approach="total", odwf_a=1.0)
    legacy = MSDGCAssessment(q, ev, np.ones_like(q, dtype=bool), DT).metrics(model).set_index("metric")["value"]
    row = res.summary.set_index("group").loc["all"]
    for zone in ("conveyance", "storage", "treatment"):
        assert row[f"{zone}_ccc"] == pytest.approx(legacy[f"{zone}_ccc"], rel=1e-12)
    assert res.con15["slope"] == pytest.approx(legacy["con15_slope"], rel=1e-12)


def test_guideline_zones_drop_the_tools_treatment_cap() -> None:
    assert TOOL_ZONES.window(48.0, DT, "treatment")[0] == 1440.0
    assert GUIDELINE_ZONES.window(48.0, DT, "treatment")[0] == pytest.approx(0.75 * 48 * 60)
    assert GUIDELINE_ZONES.window(8.0, DT, "conveyance")[0] == pytest.approx(20.16)  # Table 6-1: 20 min
    assert math.isclose(GUIDELINE_ZONES.window(24.0, DT, "storage")[0], 240.0)  # Table 6-1: 240 min


def test_inputs_are_checked() -> None:
    q, t, ev = synthetic(2)
    with pytest.raises(ValueError):
        assess_calibration(q, q[:-1], t, ev, DT)
    with pytest.raises(ValueError):
        assess_calibration(q, q, t, ev, DT, approach="mixed")
    with pytest.raises(ValueError):
        assess_calibration(q, q, t, ev.drop(columns="duration_hours"), DT)
