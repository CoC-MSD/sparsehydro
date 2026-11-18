"""Generic assessment helpers: windowed sums, volume above a threshold, paired summaries, scorecards."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sparsehydro.metrics import (
    Criterion,
    Scorecard,
    concordance_correlation,
    max_window_sum,
    paired_summary,
    sum_above,
)


def test_max_window_sum_and_first_winner_ties() -> None:
    assert max_window_sum([1, 3, 2, 5, 0], 2) == (7.0, 2)
    assert max_window_sum([2, 2, 2, 2], 2) == (4.0, 0)  # ties: the earliest window wins
    assert max_window_sum([4.0], 1) == (4.0, 0)
    v = np.random.default_rng(1).random(200)
    best, pos = max_window_sum(v, 17)
    brute = [v[i:i + 17].sum() for i in range(200 - 16)]
    assert best == pytest.approx(max(brute)) and pos == int(np.argmax(brute))
    for bad in (0, 6):
        with pytest.raises(ValueError):
            max_window_sum([1, 2, 3, 4, 5], bad)


def test_sum_above() -> None:
    assert sum_above([0, 2, 5, 1, np.nan], 1.5) == pytest.approx(0.5 + 3.5)
    assert sum_above([1, 1], 3) == 0.0


def test_paired_summary() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    s = paired_summary(x, 2 * x)
    assert s["slope0"] == pytest.approx(2.0) and s["slope"] == pytest.approx(2.0) and s["intercept"] == pytest.approx(0)
    assert s["r2"] == pytest.approx(1.0) and s["mean_ratio"] == pytest.approx(2.0) and s["n"] == 4
    assert s["ccc"] == pytest.approx(concordance_correlation(x, 2 * x))
    one = paired_summary([1.0], [2.0])
    assert math.isnan(one["ccc"]) and one["slope0"] == pytest.approx(2.0)
    assert paired_summary([1.0, np.nan, 3.0], [1.0, 5.0, 3.0])["n"] == 2


def test_lin_ccc_reproduces_the_msdgc_guideline_example() -> None:
    """Volume II Table 7-1 / Figure 7-2 of the MSDGC Modeling Guidelines (Rev 5.8)."""
    conveyance = ([0.0130, 0.0282, 0.1072, 0.0608, 0.0541], [0.0174, 0.0297, 0.1013, 0.0565, 0.0542])
    overflow = ([0.0219, 0.0453, 0.1564, 0.0802, 0.0908], [0.0278, 0.0414, 0.1583, 0.0752, 0.0966])
    treatment = ([0.0777, 0.1498, 0.4239, 0.3659, 0.5667], [0.2744, 0.2933, 0.9606, 0.5429, 1.2334])
    got = [concordance_correlation(*pair) for pair in (conveyance, overflow, treatment)]
    # The table lists volumes to four decimals; the published coefficients come from unrounded volumes.
    assert got == pytest.approx([0.9919, 0.9947, 0.4398], abs=2e-4)
    assert np.mean(got) == pytest.approx(0.8088, abs=2e-4)  # the "Cinci score"


def test_scorecard_grades() -> None:
    card = Scorecard((
        Criterion("ccc", pass_range=(0.9, None)),
        Criterion("err", pass_range=(-10, 10), fail_range=(-20, 20), units="%"),
        Criterion("info", pass_range=(0, 1), required=False),
    ))
    assert card.evaluate({"ccc": 0.95, "err": 5}).grade == "pass"
    assert card.evaluate({"ccc": 0.85, "err": 5}).grade == "conditional"  # no fail band on CCC
    assert card.evaluate({"ccc": 0.95, "err": -15}).grade == "conditional"
    assert card.evaluate({"ccc": 0.95, "err": 25}).grade == "fail"
    missing = card.evaluate({"ccc": 0.95})
    assert missing.grade == "conditional" and missing.results[1].status == "missing"
    rows = card.evaluate({"ccc": 0.95, "err": 5, "info": 7}).as_rows()
    assert [r["status"] for r in rows] == ["pass", "pass", "conditional"] and rows[1]["fail_upper"] == 20
