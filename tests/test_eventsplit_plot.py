"""Smoke tests for the event-split visualization helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("plotly")

from sparsehydro.eventsplit import per_event_scores, stratified_event_split
from sparsehydro.visualization.eventsplit_plot import plot_per_event_scores, plot_split_timeline


@pytest.fixture
def split_demo():
    rng = np.random.default_rng(0)
    n = 24 * 90  # 90 days hourly
    dt = pd.date_range("2022-01-01", periods=n, freq="h")
    baseflow = np.full(n, 2.0)
    flow = baseflow.copy()
    rainfall = np.zeros(n)
    temperature = 10 + 8 * np.sin(np.arange(n) * 2 * np.pi / n)
    event_ids = np.full(n, -1)
    # Plant a dozen triangular storms spread across the record.
    for k, start in enumerate(range(48, n - 48, (n - 96) // 12)):
        width = rng.integers(12, 30)
        peak = rng.uniform(4, 20)
        ramp = np.concatenate([np.linspace(0, peak, width), np.linspace(peak, 0, width)])
        seg = slice(start, start + ramp.size)
        flow[seg] += ramp
        rainfall[start:start + 4] += peak / 4
        event_ids[start:start + ramp.size] = k
    return dt, flow, baseflow, rainfall, temperature, event_ids


def test_split_timeline_builds_with_all_panels(split_demo):
    dt, flow, baseflow, rainfall, temperature, event_ids = split_demo
    split = stratified_event_split(event_ids, dt, flow, baseflow, val_fraction=0.3, seed=1)
    fig = plot_split_timeline(dt, flow, split, rainfall=rainfall,
                              temperature=temperature, title="demo")
    # 3 data panels → rain + flow + temp traces, plus 4 legend swatches.
    names = [t.name for t in fig.data]
    for swatch in ("Training storm", "Validation storm",
                   "Warmup (unscored)", "Recession buffer"):
        assert swatch in names
    # A shaded band per event + guard bands were added as shapes.
    assert len(fig.layout.shapes) >= split.events_table.shape[0]


def test_split_timeline_flow_only(split_demo):
    dt, flow, baseflow, _rain, _temp, event_ids = split_demo
    split = stratified_event_split(event_ids, dt, flow, baseflow, seed=1)
    fig = plot_split_timeline(dt, flow, split, show_guard_bands=False)
    names = [t.name for t in fig.data]
    assert "Warmup (unscored)" not in names  # suppressed with guard bands off


def test_per_event_scores_plot(split_demo):
    dt, flow, baseflow, _rain, _temp, event_ids = split_demo
    split = stratified_event_split(event_ids, dt, flow, baseflow, seed=1)
    sim = flow * 1.1  # a biased "model"
    sc = per_event_scores(flow, sim, split.event_ids,
                          restrict_to=split.val_events, datetime=dt)
    fig = plot_per_event_scores(sc, title="demo")
    metric_traces = [t for t in fig.data if t.name in ("KGE", "CCC")]
    assert metric_traces
    # One bar per validation storm.
    assert len(metric_traces[0].x) == len(sc)
