"""Tests for the new unit-hydrograph Plotly visualizations."""

from __future__ import annotations

import pytest

pytest.importorskip("plotly")
import plotly.graph_objects as go  # noqa: E402

from sparsehydro.events import detect_event_hierarchy  # noqa: E402
from sparsehydro.models.unithydrograph import (  # noqa: E402
    GammaUH, RectangleUH, PeakTailUH, SequentialFitter,
)
from sparsehydro.visualization import (  # noqa: E402
    plot_variable_savgol, plot_event_hierarchy, plot_uh_shapes, plot_convolution,
    plot_effective_area,
)


@pytest.fixture(scope="module")
def hierarchy(synthetic_rain_stormflow):
    return detect_event_hierarchy(synthetic_rain_stormflow)


def test_plot_variable_savgol(synthetic_rain_stormflow, hierarchy):
    _, _, savgol = hierarchy
    fig = plot_variable_savgol(savgol, synthetic_rain_stormflow)
    assert isinstance(fig, go.Figure)
    fig2 = plot_variable_savgol(savgol)  # no rain panel
    assert isinstance(fig2, go.Figure)


def test_plot_event_hierarchy(synthetic_rain_stormflow, hierarchy):
    globs, subs, savgol = hierarchy
    fig = plot_event_hierarchy(synthetic_rain_stormflow, globs, subs, savgol)
    assert isinstance(fig, go.Figure)


def test_plot_event_hierarchy_empty(synthetic_rain_stormflow):
    fig = plot_event_hierarchy(synthetic_rain_stormflow, [], [])
    assert isinstance(fig, go.Figure)


def test_plot_uh_shapes_variants():
    assert isinstance(plot_uh_shapes(GammaUH()), go.Figure)
    assert isinstance(plot_uh_shapes([("g", GammaUH()), ("r", RectangleUH())]), go.Figure)
    assert isinstance(plot_uh_shapes({"pt": PeakTailUH()}), go.Figure)


def test_plot_effective_area_subevents(hierarchy):
    _, subs, _ = hierarchy
    fig = plot_effective_area(subs)
    assert isinstance(fig, go.Figure)


def test_plot_convolution(synthetic_rain_stormflow, hierarchy):
    _, subs, _ = hierarchy
    summary = SequentialFitter(lambda: GammaUH(), synthetic_rain_stormflow, subs).fit(verbose=False)
    fig = plot_convolution(summary)
    assert isinstance(fig, go.Figure)
    if subs:
        fig_ev = plot_convolution(summary, event=subs[0])
        assert isinstance(fig_ev, go.Figure)
