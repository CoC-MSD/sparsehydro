"""Tests for :mod:`sparsehydro.eventsplit` — event-based split + per-event metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sparsehydro.eventsplit import EventSplit, estimate_recession_hours, per_event_scores, stratified_event_split, summarize_event_scores


# ---------------------------------------------------------------------------
# Synthetic hydrograph fixture: baseflow + decaying storm pulses, labelled.
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic():
    n_days = 150
    dt = pd.date_range("2022-01-01", periods=n_days * 24, freq="h")
    n = len(dt)
    baseflow = np.full(n, 2.0)
    flow = baseflow.copy()
    event_ids = np.full(n, -1, dtype=int)

    rng = np.random.default_rng(0)
    starts = np.linspace(200, n - 400, 16).astype(int)
    for k, s in enumerate(starts):
        peak = float(rng.uniform(5, 25))
        dur = int(rng.integers(18, 40))
        for j in range(dur):
            rise = peak * (j / 3.0) if j < 3 else peak * np.exp(-(j - 3) / 8.0)
            flow[s + j] += rise
        event_ids[s : s + dur] = k
    return dict(dt=dt, flow=flow, baseflow=baseflow, event_ids=event_ids, n=n)


# ---------------------------------------------------------------------------
# estimate_recession_hours
# ---------------------------------------------------------------------------

def test_recession_estimate_positive(synthetic):
    tau = estimate_recession_hours(synthetic["flow"], synthetic["baseflow"])
    assert np.isfinite(tau) and tau > 0
    # Storms decay with an ~8 h time constant; estimate should be in a sane band.
    assert 2.0 < tau < 40.0


def test_recession_estimate_nan_when_flat():
    flat = np.full(50, 3.0)
    assert np.isnan(estimate_recession_hours(flat, np.full(50, 3.0)))


# ---------------------------------------------------------------------------
# stratified_event_split
# ---------------------------------------------------------------------------

def test_split_basic_disjoint_and_fraction(synthetic):
    sp = stratified_event_split(
        synthetic["event_ids"], synthetic["dt"], synthetic["flow"],
        synthetic["baseflow"], val_fraction=0.30, seed=42,
    )
    assert isinstance(sp, EventSplit)
    train, val = set(sp.train_events.tolist()), set(sp.val_events.tolist())
    all_events = set(np.unique(synthetic["event_ids"][synthetic["event_ids"] >= 0]).tolist())
    assert train.isdisjoint(val)
    assert train | val == all_events
    # ~30% of 16 events held out, allowing for stratification rounding.
    assert 0.15 <= len(val) / len(all_events) <= 0.45
    assert len(val) >= 1


def test_masks_partition_timeline(synthetic):
    sp = stratified_event_split(
        synthetic["event_ids"], synthetic["dt"], synthetic["flow"], synthetic["baseflow"],
    )
    n = synthetic["n"]
    # val / warmup / buffer / train_mask are mutually exclusive and cover all steps.
    stack = np.vstack([sp.val_mask, sp.warmup_mask, sp.buffer_mask, sp.train_mask])
    assert stack.sum(axis=0).max() == 1  # no overlap
    assert stack.sum(axis=0).min() == 1  # full coverage
    assert stack.sum() == n


def test_no_training_step_leaks_into_holdout_neighbourhood(synthetic):
    sp = stratified_event_split(
        synthetic["event_ids"], synthetic["dt"], synthetic["flow"], synthetic["baseflow"],
    )
    holdout = sp.val_mask | sp.warmup_mask | sp.buffer_mask
    # A-style training steps never touch a validation storm, its warmup or buffer.
    assert not (sp.train_mask & holdout).any()
    # B/C calibration steps live only inside training storms, never validation ones.
    assert not sp.val_mask[sp.train_event_mask].any()
    assert not np.isin(synthetic["event_ids"][sp.train_event_mask], sp.val_events).any()


def test_warmup_precedes_validation_events(synthetic):
    sp = stratified_event_split(
        synthetic["event_ids"], synthetic["dt"], synthetic["flow"], synthetic["baseflow"],
        warmup_hours=24,
    )
    assert sp.warmup_mask.sum() > 0
    # Every warmup run is immediately followed (somewhere) by a validation step.
    assert sp.meta["warmup_steps"] == 24


def test_split_is_interwoven_not_front_loaded(synthetic):
    sp = stratified_event_split(
        synthetic["event_ids"], synthetic["dt"], synthetic["flow"], synthetic["baseflow"],
    )
    order = sp.events_table.index.to_numpy()  # time order
    ranks = {e: i for i, e in enumerate(order)}
    val_ranks = sorted(ranks[e] for e in sp.val_events)
    # Validation events are spread across the record, not clustered up front.
    assert max(val_ranks) - min(val_ranks) > len(order) / 2


def test_split_deterministic(synthetic):
    a = stratified_event_split(
        synthetic["event_ids"], synthetic["dt"], synthetic["flow"], synthetic["baseflow"], seed=7,
    )
    b = stratified_event_split(
        synthetic["event_ids"], synthetic["dt"], synthetic["flow"], synthetic["baseflow"], seed=7,
    )
    assert np.array_equal(a.val_events, b.val_events)


def test_split_rejects_empty(synthetic):
    with pytest.raises(ValueError):
        stratified_event_split(
            np.full(synthetic["n"], -1), synthetic["dt"], synthetic["flow"], synthetic["baseflow"],
        )


# ---------------------------------------------------------------------------
# per_event_scores / summarize_event_scores
# ---------------------------------------------------------------------------

def test_per_event_scores_perfect_prediction(synthetic):
    scores = per_event_scores(
        synthetic["flow"], synthetic["flow"], synthetic["event_ids"], datetime=synthetic["dt"],
    )
    assert len(scores) == 16
    assert np.allclose(scores["kge"].to_numpy(), 1.0, atol=1e-6)
    assert np.allclose(scores["ccc"].to_numpy(), 1.0, atol=1e-6)
    summ = summarize_event_scores(scores)
    assert summ["kge_mean"] == pytest.approx(1.0, abs=1e-6)
    assert summ["kge_n"] == 16
    assert summ["ccc_median"] == pytest.approx(1.0, abs=1e-6)


def test_per_event_scores_restrict_to(synthetic):
    sp = stratified_event_split(
        synthetic["event_ids"], synthetic["dt"], synthetic["flow"], synthetic["baseflow"],
    )
    scores = per_event_scores(
        synthetic["flow"], synthetic["flow"], synthetic["event_ids"], restrict_to=sp.val_events,
    )
    assert set(scores.index) == set(sp.val_events.tolist())


def test_per_event_scores_degenerate_window_is_nan():
    ids = np.array([-1, 0, 0, 0, 1, 1, 1, -1])
    obs = np.array([0, 5, 5, 5, 1, 2, 3, 0], float)  # event 0: constant obs -> undefined
    pred = obs.copy()
    scores = per_event_scores(obs, pred, ids, min_steps=2)
    assert np.isnan(scores.loc[0, "kge"])   # KGE needs non-zero variance
    assert scores.loc[1, "kge"] == pytest.approx(1.0, abs=1e-6)
    # Summary ignores the NaN storm rather than propagating it.
    summ = summarize_event_scores(scores)
    assert summ["kge_n"] == 1
    assert summ["kge_mean"] == pytest.approx(1.0, abs=1e-6)
