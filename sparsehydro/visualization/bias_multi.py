"""Multi-method bias diagnostics — overlay several calibrations on shared axes.

The single-series :func:`~sparsehydro.visualization.timeseries.plot_cumulative_volume`
and :func:`~sparsehydro.visualization.timeseries.plot_residuals_scatter` each draw
*one* observed-vs-predicted comparison per figure.  To compare calibration
methods (e.g. A·1a, A·1b, B, C) on the **same** held-out validation window they
must share axes with one legend entry per method.

This module provides overlay variants that take an ``observed`` series plus a
list of :class:`MethodSeries` (name + predicted series + colour) and, optionally,
a boolean ``mask`` selecting the validation timesteps every method is judged on:

- :func:`plot_cumulative_volume_multi` — cumulative volume and cumulative volume
  error, one coloured line per method, over the (masked) window.
- :func:`plot_residuals_multi` — observed-vs-predicted scatter (with a 1:1 line)
  plus a per-method residual box plot, one colour per method.

Both are conditional on plotly being installed (mirroring the rest of
:mod:`sparsehydro.visualization`); import-time stubs raise a clear error if not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["MethodSeries", "DEFAULT_METHOD_COLORS", "plot_cumulative_volume_multi", "plot_residuals_multi"]


# A stable colour per method label so the same method reads the same across figures.
DEFAULT_METHOD_COLORS = {
    "A·1a": "#2166AC",  # blue
    "A·1b": "#67A9CF",  # light blue
    "B": "#1A9850",     # green
    "C": "#D6604D",     # red
}
_FALLBACK_CYCLE = ["#2166AC", "#67A9CF", "#1A9850", "#D6604D", "#762A83", "#E08214", "#4D4D4D"]


@dataclass
class MethodSeries:
    """One method's predicted series for an overlay plot.

    :ivar name: Legend label (e.g. ``"A·1a"``, ``"B"``).
    :ivar predicted: Predicted series aligned to the shared ``observed`` array.
    :ivar color: Optional explicit colour; falls back to
        :data:`DEFAULT_METHOD_COLORS` then a default cycle.
    """

    name: str
    predicted: np.ndarray
    color: str | None = None


def _resolve_colors(methods: list[MethodSeries]) -> list[str]:
    colors: list[str] = []
    for i, m in enumerate(methods):
        colors.append(m.color or DEFAULT_METHOD_COLORS.get(m.name, _FALLBACK_CYCLE[i % len(_FALLBACK_CYCLE)]))
    return colors


def _prep(observed, methods, mask):
    """Coerce inputs and apply the (optional) shared validation mask, in time order."""
    obs = np.asarray(observed, dtype=float)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != obs.shape[0]:
            raise ValueError("mask length must match observed length.")
    sel = slice(None) if mask is None else mask
    obs_sel = obs[sel]
    preds_sel = [np.asarray(m.predicted, dtype=float)[sel] for m in methods]
    return obs_sel, preds_sel


try:
    import plotly.graph_objects as go  # type: ignore[import]
    from plotly.subplots import make_subplots  # type: ignore[import]

    def plot_cumulative_volume_multi(
        datetime,
        observed,
        methods: list[MethodSeries],
        *,
        mask=None,
        title: str = "Cumulative Volume — method comparison",
        flow_label: str = "Flow",
    ) -> "go.Figure":
        """Overlay cumulative volume and cumulative volume error across methods.

        **Row 1** — cumulative observed volume (black) and each method's
        cumulative predicted volume (coloured).

        **Row 2** — cumulative volume error ``cumsum(obs) − cumsum(pred)`` per
        method, with a zero reference line; a line drifting away from zero shows
        a method accumulating volume bias.

        :param datetime: Time axis aligned to *observed* (masked with *mask* if
            given).
        :param observed: Observed series.
        :param methods: One :class:`MethodSeries` per calibration to overlay.
        :param mask: Optional boolean validation mask (same length as
            *observed*); when given, every series is restricted to these steps so
            the methods are compared on identical timesteps.
        :param title: Figure title.
        :type title: str
        :param flow_label: Units label for the volume axis.
        :type flow_label: str
        :returns: Plotly figure with 2 rows.
        :rtype: plotly.graph_objects.Figure
        """
        obs_sel, preds_sel = _prep(observed, methods, mask)
        colors = _resolve_colors(methods)
        dt = np.asarray(list(datetime))
        dt_sel = dt if mask is None else dt[np.asarray(mask, dtype=bool)]

        cum_obs = np.cumsum(obs_sel)
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
            subplot_titles=("Cumulative volume", "Cumulative volume error (obs − pred)"),
        )
        fig.add_trace(
            go.Scatter(x=dt_sel, y=cum_obs, name="Observed", legendgroup="obs",
                       line=dict(color="#000000", width=2.4)),
            row=1, col=1,
        )
        for m, pred_sel, color in zip(methods, preds_sel, colors):
            cum_pred = np.cumsum(pred_sel)
            fig.add_trace(
                go.Scatter(x=dt_sel, y=cum_pred, name=m.name, legendgroup=m.name,
                           line=dict(color=color, width=1.6)),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(x=dt_sel, y=cum_obs - cum_pred, name=m.name, legendgroup=m.name,
                           showlegend=False, line=dict(color=color, width=1.6)),
                row=2, col=1,
            )
        fig.add_hline(y=0.0, line=dict(color="#888888", width=1, dash="dash"), row=2, col=1)
        fig.update_yaxes(title_text=f"Cumulative {flow_label}", row=1, col=1)
        fig.update_yaxes(title_text=f"Error [{flow_label}]", row=2, col=1)
        fig.update_layout(title=title, hovermode="x unified", legend_title_text="Method")
        return fig

    def plot_residuals_multi(
        observed,
        methods: list[MethodSeries],
        *,
        mask=None,
        title: str = "Residual diagnostics — method comparison",
        flow_label: str = "Flow",
    ) -> "go.Figure":
        """Overlay observed-vs-predicted scatter and residual spread across methods.

        **Row 1** — observed (x) vs predicted (y) scatter, one colour per method,
        with a 1:1 reference line.  Points below the line are under-prediction.

        **Row 2** — box plot of residuals ``obs − pred`` per method; a box
        centred off zero flags systematic bias, its width shows scatter.

        :param observed: Observed series.
        :param methods: One :class:`MethodSeries` per calibration to overlay.
        :param mask: Optional boolean validation mask (same length as
            *observed*) restricting every series to identical timesteps.
        :param title: Figure title.
        :type title: str
        :param flow_label: Axis label for the observed / predicted axes.
        :type flow_label: str
        :returns: Plotly figure with 2 rows.
        :rtype: plotly.graph_objects.Figure
        """
        obs_sel, preds_sel = _prep(observed, methods, mask)
        colors = _resolve_colors(methods)

        fig = make_subplots(
            rows=2, cols=1, vertical_spacing=0.12, row_heights=[0.62, 0.38],
            subplot_titles=("Observed vs predicted", "Residual distribution (obs − pred)"),
        )
        finite_obs = obs_sel[np.isfinite(obs_sel)]
        lo = float(np.nanmin(finite_obs)) if finite_obs.size else 0.0
        hi = float(np.nanmax(finite_obs)) if finite_obs.size else 1.0
        for m, pred_sel, color in zip(methods, preds_sel, colors):
            fig.add_trace(
                go.Scatter(x=obs_sel, y=pred_sel, mode="markers", name=m.name,
                           legendgroup=m.name, marker=dict(color=color, size=4, opacity=0.55)),
                row=1, col=1,
            )
            hi = max(hi, float(np.nanmax(pred_sel)) if np.isfinite(pred_sel).any() else hi)
            fig.add_trace(
                go.Box(y=obs_sel - pred_sel, name=m.name, legendgroup=m.name,
                       showlegend=False, marker_color=color, boxmean=True),
                row=2, col=1,
            )
        fig.add_trace(
            go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="1:1",
                       line=dict(color="#888888", width=1, dash="dash"), showlegend=False),
            row=1, col=1,
        )
        fig.add_hline(y=0.0, line=dict(color="#888888", width=1, dash="dash"), row=2, col=1)
        fig.update_xaxes(title_text=f"Observed {flow_label}", row=1, col=1)
        fig.update_yaxes(title_text=f"Predicted {flow_label}", row=1, col=1)
        fig.update_yaxes(title_text=f"Residual [{flow_label}]", row=2, col=1)
        fig.update_layout(title=title, legend_title_text="Method")
        return fig

except ImportError:  # pragma: no cover - matches package-wide plotly guard

    def plot_cumulative_volume_multi(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("plotly is required for plot_cumulative_volume_multi. Install with: pip install plotly")

    def plot_residuals_multi(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("plotly is required for plot_residuals_multi. Install with: pip install plotly")
