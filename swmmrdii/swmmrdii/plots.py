"""
Plotting helpers for swmmrdii.

Provides three families of plots:

1. ``plot_timeseries()``
   A thin wrapper around ``sparsehydro.visualization.plot_calibration_timeseries``
   that adds the optional Pareto uncertainty band and keeps the call signature
   consistent with how the notebooks use it.

2. ``plot_aggregated()``
   Time-based aggregation comparison: observed vs. predicted grouped by
   'byWeek', 'byMonth', 'bySeason', or 'byYear'.  Returns a Plotly figure
   with bar + line subplots so both the seasonal pattern and the fit quality
   are visible at a glance.

3. ``plot_temporal_scatter()``
   Observed vs. predicted scatter plots aggregated at annual, seasonal, and
   monthly timescales, shown as a 1×3 panel figure.  Each panel includes a
   1:1 reference line, a best-fit regression line, ±10%/±20% tolerance bands,
   and CCC / R² / NSE annotations.

All functions return Plotly Figure objects so the caller can call
``.show()`` or ``.write_html()`` as needed.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

try:
    from sparsehydro.visualization import plot_calibration_timeseries as _sh_plot_ts
    _HAS_SH_VIZ = True
except ImportError:
    _HAS_SH_VIZ = False

try:
    from sparsehydro.visualization import plot_cumulative_volume as _sh_plot_cumvol
    _HAS_SH_CUMVOL = True
except ImportError:
    _HAS_SH_CUMVOL = False


def _require_plotly() -> None:
    if not _HAS_PLOTLY:
        raise ImportError("plotly is required for swmmrdii plots. Install with: pip install plotly")


# ---------------------------------------------------------------------------
# Timeseries plot (wrapper around sparsehydro.visualization)
# ---------------------------------------------------------------------------

def plot_timeseries(
    datetime: pd.Series,
    observed: np.ndarray,
    predicted: np.ndarray,
    exogenous: Optional[dict] = None,
    pareto_predictions: Optional[np.ndarray] = None,
    confidence_percentiles: tuple[float, float] = (5.0, 95.0),
    title: str = "RDII Calibration",
    width: int = 1600,
    height: int = 800,
) -> "go.Figure":
    """
    Plot observed vs. predicted RDII time series with optional forcing panels
    and a Pareto uncertainty band.

    Parameters
    ----------
    datetime : pd.Series
        Datetime values aligned with *observed* and *predicted*.
    observed : np.ndarray
        Observed flow (CFS).
    predicted : np.ndarray
        Best-estimate predicted flow (CFS).
    exogenous : dict or None
        Mapping of panel label → (values_array, units_string).  Each entry
        produces an additional sub-panel (e.g. rainfall, temperature).
        Example: ``{'Rainfall (in)': (rain_array, 'in')}``.
    pareto_predictions : np.ndarray or None
        2-D array of shape (N_top, N_timesteps) containing the top-N
        predictions from the Optuna study.  When provided, the 5th–95th
        percentile band is shaded as an uncertainty envelope.
    confidence_percentiles : (float, float)
        Lower and upper percentile bounds for the uncertainty band.
    title : str
        Figure title.
    width, height : int
        Figure dimensions in pixels.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    _require_plotly()

    # An empty window is a legitimate configuration — a calibration with no
    # validation period leaves the validation window with nothing in it — and
    # must not abort the notebook.  sparsehydro's plotter reduces over the
    # arrays (max/min for axis ranges) and raises on zero-size input, so the
    # empty case is handled here instead of being passed down.
    if len(np.asarray(observed)) == 0 and len(np.asarray(predicted)) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No data in this window<br>"
                 "<span style='font-size:12px'>"
                 "(empty date range — nothing to plot)</span>",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, align="center",
        )
        fig.update_layout(
            title=title, width=width, height=height,
            xaxis={"visible": False}, yaxis={"visible": False},
        )
        return fig

    if _HAS_SH_VIZ:
        # Delegate to sparsehydro's own plot function which handles the
        # multi-panel layout, axis formatting, and band shading.
        kwargs: dict = dict(
            datetime=datetime,
            observed=observed,
            predicted=predicted,
            title=title,
        )
        if exogenous is not None:
            kwargs["exogenous"] = exogenous
        if pareto_predictions is not None:
            kwargs["pareto_predictions"] = pareto_predictions
            kwargs["confidence_percentiles"] = confidence_percentiles

        fig = _sh_plot_ts(**kwargs)
        fig.update_layout(width=width, height=height)
        if exogenous is None:
            # No forcing panels were requested — drop the floating
            # "Exogenous Inputs" subplot title left over the empty top row.
            fig.layout.annotations = tuple(
                a for a in fig.layout.annotations if a.text != "Exogenous Inputs"
            )
        return fig

    # Fallback: build a simple two-panel figure if sparsehydro.visualization
    # is not importable.
    n_rows = 1 + (len(exogenous) if exogenous else 0)
    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[title] + (list(exogenous.keys()) if exogenous else []),
    )

    fig.add_trace(
        go.Scatter(x=datetime, y=observed, name="Observed", line=dict(color="royalblue")),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=datetime, y=predicted, name="Predicted", line=dict(color="firebrick")),
        row=1, col=1,
    )

    if pareto_predictions is not None:
        lo = np.percentile(pareto_predictions, confidence_percentiles[0], axis=0)
        hi = np.percentile(pareto_predictions, confidence_percentiles[1], axis=0)
        fig.add_trace(
            go.Scatter(
                x=pd.concat([datetime, datetime[::-1]]),
                y=np.concatenate([hi, lo[::-1]]),
                fill="toself",
                fillcolor="rgba(220,80,80,0.2)",
                line=dict(color="rgba(0,0,0,0)"),
                name=f"Uncertainty ({confidence_percentiles[0]:.0f}–{confidence_percentiles[1]:.0f}%)",
            ),
            row=1, col=1,
        )

    if exogenous:
        for i, (label, (vals, units)) in enumerate(exogenous.items(), start=2):
            fig.add_trace(
                go.Bar(x=datetime, y=vals, name=label, marker_color="steelblue"),
                row=i, col=1,
            )

    fig.update_layout(width=width, height=height, title_text=title)
    return fig


# ---------------------------------------------------------------------------
# Cumulative volume plot (wrapper around sparsehydro.visualization)
# ---------------------------------------------------------------------------

def plot_cumulative_volume(
    datetime: pd.Series,
    observed: np.ndarray,
    predicted: np.ndarray,
    title: str = "Cumulative Volume",
    flow_label: str = "Flow",
) -> "go.Figure":
    """
    Two-panel cumulative-volume comparison.

    - **Top** — cumulative observed vs. predicted flow, with a shaded band
      between the two curves highlighting the running volume discrepancy.
    - **Bottom** — cumulative volume error (observed − predicted cumulative
      sums) with a zero reference line.

    This delegates to ``sparsehydro.visualization.plot_cumulative_volume``
    when available (matching the original notebook output exactly), and falls
    back to an equivalent Plotly figure otherwise.

    Parameters
    ----------
    datetime : pd.Series
        Datetime values aligned with *observed* and *predicted*.
    observed : np.ndarray
        Observed flow (CFS).
    predicted : np.ndarray
        Predicted flow (CFS).
    title : str
        Figure title.
    flow_label : str
        Units label used on the cumulative-volume y-axis.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    _require_plotly()

    if _HAS_SH_CUMVOL:
        return _sh_plot_cumvol(
            datetime, observed, predicted, title=title, flow_label=flow_label
        )

    # Fallback: replicate the two-panel layout with Plotly directly.
    obs = np.asarray(observed, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    cum_obs = np.cumsum(obs)
    cum_pred = np.cumsum(pred)
    cum_error = cum_obs - cum_pred
    dt = list(datetime)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.4],
        vertical_spacing=0.06,
        subplot_titles=["Cumulative Volume", "Cumulative Volume Error"],
    )
    fig.add_trace(
        go.Scatter(x=dt, y=cum_obs, mode="lines", name="Observed cumulative",
                   line=dict(color="black", width=2)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=dt, y=cum_pred, mode="lines", name="Predicted cumulative",
                   fill="tonexty", fillcolor="rgba(220,50,50,0.15)",
                   line=dict(color="crimson", width=2, dash="dash")),
        row=1, col=1,
    )
    fig.update_yaxes(title_text=f"Cumulative {flow_label}", row=1, col=1)

    fig.add_trace(
        go.Scatter(x=dt, y=cum_error, mode="lines", name="Volume error",
                   line=dict(color="steelblue", width=2), showlegend=False),
        row=2, col=1,
    )
    fig.add_hline(y=0, line=dict(color="black", width=1, dash="dot"), row=2, col=1)
    fig.update_yaxes(title_text="Obs - Pred (cumul.)", row=2, col=1)
    fig.update_xaxes(title_text="Date / Time", row=2, col=1)

    fig.update_layout(
        title=title, height=600, hovermode="x unified",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.7)"),
    )
    return fig


# ---------------------------------------------------------------------------
# Time-aggregation plot
# ---------------------------------------------------------------------------

_PERIOD_MAP = {
    "byweek":   "W",
    "bymonth":  "ME",
    "byseason": "QS-DEC",  # seasonal quarters starting December
    "byyear":   "YE",
}

_SEASON_LABELS = {12: "DJF", 3: "MAM", 6: "JJA", 9: "SON"}


def plot_aggregated(
    datetime: pd.Series,
    observed: np.ndarray,
    predicted: np.ndarray,
    period: str = "byMonth",
    agg: str = "mean",
    title: Optional[str] = None,
    width: int = 1200,
    height: int = 500,
) -> "go.Figure":
    """
    Compare observed vs. predicted flow aggregated over a time period.

    Produces a grouped bar chart with two series (Observed, Predicted) and
    an overlaid line showing the predicted/observed ratio.

    Parameters
    ----------
    datetime : pd.Series
        Datetime values aligned with *observed* and *predicted*.
    observed : np.ndarray
        Observed flow (CFS) at the original (daily) timestep.
    predicted : np.ndarray
        Predicted flow (CFS) at the same timestep.
    period : str
        Aggregation period.  Case-insensitive.  Options:

            - ``'byWeek'``   — calendar weeks (ISO week)
            - ``'byMonth'``  — calendar months
            - ``'bySeason'`` — meteorological seasons (DJF/MAM/JJA/SON)
            - ``'byYear'``   — calendar years
    agg : str
        Aggregation function: ``'mean'`` (default) or ``'sum'``.  Use
        ``'mean'`` for flow rates (CFS) and ``'sum'`` for depths/volumes.
    title : str or None
        Figure title.  Defaults to ``'Observed vs Predicted — <period>'``.
    width, height : int
        Figure dimensions in pixels.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    _require_plotly()

    period_key = period.lower().replace(" ", "")
    if period_key not in _PERIOD_MAP:
        raise ValueError(
            f"Unknown period {period!r}. "
            f"Choose from: {list(_PERIOD_MAP.keys())}"
        )

    freq = _PERIOD_MAP[period_key]
    agg_fn = "mean" if agg == "mean" else "sum"

    # Build a working DataFrame with a datetime index.
    df = pd.DataFrame(
        {"obs": observed, "pred": predicted},
        index=pd.to_datetime(datetime.values),
    )

    # Resample to the requested period.
    grouped = df.resample(freq).agg(agg_fn)

    # Format x-axis labels depending on the period.
    if period_key == "byseason":
        # Map each period-start month to a season label.
        grouped["label"] = grouped.index.map(
            lambda d: f"{_SEASON_LABELS.get(d.month, str(d.month))} {d.year}"
        )
    elif period_key == "byyear":
        grouped["label"] = grouped.index.year.astype(str)
    elif period_key == "bymonth":
        grouped["label"] = grouped.index.strftime("%b %Y")
    else:  # byweek
        grouped["label"] = grouped.index.strftime("W%V %Y")

    x_labels = grouped["label"].tolist()

    # Compute ratio (predicted / observed) for the line overlay.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(grouped["obs"] > 0, grouped["pred"] / grouped["obs"], np.nan)

    if title is None:
        title = f"Observed vs. Predicted — {period}"

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[title, "Predicted / Observed ratio"],
        row_heights=[0.7, 0.3],
    )

    # Grouped bar chart (upper panel).
    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=grouped["obs"].tolist(),
            name="Observed",
            marker_color="royalblue",
            opacity=0.85,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=grouped["pred"].tolist(),
            name="Predicted",
            marker_color="firebrick",
            opacity=0.85,
        ),
        row=1, col=1,
    )

    # Ratio line (lower panel).
    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=ratio.tolist(),
            mode="lines+markers",
            name="Pred / Obs ratio",
            line=dict(color="darkorange"),
        ),
        row=2, col=1,
    )
    # Reference line at ratio = 1.0.
    fig.add_hline(y=1.0, line_dash="dash", line_color="grey", row=2, col=1)

    y_label = f"Flow ({agg_fn} CFS)"
    fig.update_yaxes(title_text=y_label, row=1, col=1)
    fig.update_yaxes(title_text="Ratio", row=2, col=1)
    fig.update_layout(
        barmode="group",
        width=width,
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ---------------------------------------------------------------------------
# Temporal scatter plot  (observed vs predicted at annual/seasonal/monthly)
# ---------------------------------------------------------------------------

def plot_temporal_scatter(
    datetime: pd.Series,
    observed: np.ndarray,
    predicted: np.ndarray,
    title: str = "Observed vs Predicted",
    agg: str = "mean",
    width: int = 1400,
    height: int = 550,
) -> "go.Figure":
    """
    Three-panel scatter plot of observed vs. predicted flow aggregated at
    annual, seasonal, and monthly timescales.

    Each panel shows:

    - Scatter points (observed on x-axis, predicted on y-axis).
    - 1:1 reference line (grey).
    - Best-fit OLS regression line (red), labelled with slope.
    - ±10% and ±20% tolerance band lines (dashed green / dotted gold).
    - Subtitle annotation with CCC, R², and NSE for that aggregation period.

    Parameters
    ----------
    datetime : pd.Series
        Datetime values aligned with *observed* and *predicted*.
    observed : np.ndarray
        Observed flow (CFS) at the original (daily) timestep.
    predicted : np.ndarray
        Predicted flow (CFS) at the same timestep.
    title : str
        Overall figure title prefix.  Each panel appends its period label.
    agg : str
        Aggregation function applied before computing scatter: ``'mean'``
        (default) or ``'sum'``.
    width, height : int
        Figure dimensions in pixels.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    _require_plotly()

    df = pd.DataFrame(
        {"obs": observed, "pred": predicted},
        index=pd.to_datetime(datetime.values),
    )
    agg_fn = "mean" if agg == "mean" else "sum"

    # Periods to plot: (label, resample_freq)
    periods = [
        ("Annual Mean",   "YE"),
        ("Seasonal Mean", "QS-DEC"),
        ("Monthly Mean",  "ME"),
    ]

    def _metrics(o, p):
        """Return (CCC, R², NSE) for paired arrays."""
        o, p = np.asarray(o, float), np.asarray(p, float)
        mask = np.isfinite(o) & np.isfinite(p)
        o, p = o[mask], p[mask]
        if len(o) < 2:
            return float("nan"), float("nan"), float("nan")
        o_mean = o.mean()
        p_mean = p.mean()
        o_var  = np.var(o)
        p_var  = np.var(p)
        cov    = np.mean((o - o_mean) * (p - p_mean))
        ccc    = 2.0 * cov / (o_var + p_var + (o_mean - p_mean) ** 2)
        ss_res = np.sum((o - p) ** 2)
        ss_tot = np.sum((o - o_mean) ** 2)
        nse    = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        r2     = (cov / (np.std(o) * np.std(p))) ** 2 if (np.std(o) > 0 and np.std(p) > 0) else float("nan")
        return ccc, r2, nse

    fig = make_subplots(
        rows=1, cols=3,
        horizontal_spacing=0.08,
        subplot_titles=[p[0] for p in periods],
    )

    # Shared colour / dash scheme matching the screenshot legend.
    _LINE_11    = dict(color="grey",       width=1.5)
    _LINE_FIT   = dict(color="firebrick",  width=2)
    _LINE_10    = dict(color="limegreen",  width=1, dash="dash")
    _LINE_20    = dict(color="gold",       width=1, dash="dot")
    _MARKER     = dict(size=7, color="royalblue", opacity=0.8)

    show_legend = True  # only add legend entries for the first panel

    for col, (period_label, freq) in enumerate(periods, start=1):
        g = df.resample(freq).agg(agg_fn).dropna()
        obs_agg  = g["obs"].values
        pred_agg = g["pred"].values

        ccc, r2, nse = _metrics(obs_agg, pred_agg)

        # Axis range: common scale so 1:1 line is diagonal.
        xy_max = max(obs_agg.max(), pred_agg.max()) * 1.05
        xy_min = 0.0
        ref = np.array([xy_min, xy_max])

        # OLS slope through origin (no intercept) for best-fit line.
        slope = (
            np.dot(obs_agg, pred_agg) / np.dot(obs_agg, obs_agg)
            if np.dot(obs_agg, obs_agg) > 0 else 1.0
        )

        # ── Tolerance bands ──────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=ref, y=ref * 1.20,
            mode="lines", line=_LINE_20,
            name="±20%", showlegend=show_legend, legendgroup="20pct",
        ), row=1, col=col)
        fig.add_trace(go.Scatter(
            x=ref, y=ref * 0.80,
            mode="lines", line=_LINE_20,
            name="±20%", showlegend=False, legendgroup="20pct",
        ), row=1, col=col)
        fig.add_trace(go.Scatter(
            x=ref, y=ref * 1.10,
            mode="lines", line=_LINE_10,
            name="±10%", showlegend=show_legend, legendgroup="10pct",
        ), row=1, col=col)
        fig.add_trace(go.Scatter(
            x=ref, y=ref * 0.90,
            mode="lines", line=_LINE_10,
            name="±10%", showlegend=False, legendgroup="10pct",
        ), row=1, col=col)

        # ── 1:1 line ─────────────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=ref, y=ref,
            mode="lines", line=_LINE_11,
            name="1:1", showlegend=show_legend, legendgroup="11",
        ), row=1, col=col)

        # ── Best-fit line ─────────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=ref, y=slope * ref,
            mode="lines", line=_LINE_FIT,
            name=f"Best fit (y={slope:.2f}x)",
            showlegend=show_legend, legendgroup="fit",
        ), row=1, col=col)

        # ── Scatter points ────────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=obs_agg, y=pred_agg,
            mode="markers", marker=_MARKER,
            name="Data", showlegend=show_legend, legendgroup="data",
        ), row=1, col=col)

        # ── Metric annotation (subtitle below panel title) ────────────────
        # Plotly subplot_titles go at the top; we add a second annotation
        # just below to carry the metric line.
        fig.add_annotation(
            text=f"CCC={ccc:.3f}  R²={r2:.3f}  NSE={nse:.3f}",
            xref=f"x{col} domain" if col > 1 else "x domain",
            yref=f"y{col} domain" if col > 1 else "y domain",
            x=0.5, y=1.07,
            xanchor="center", yanchor="bottom",
            showarrow=False,
            font=dict(size=11),
        )

        # Fix axis range and labels.
        axis_sfx = str(col) if col > 1 else ""
        fig.update_xaxes(
            range=[xy_min, xy_max],
            title_text="Observed (CFS)",
            row=1, col=col,
        )
        fig.update_yaxes(
            range=[xy_min, xy_max],
            title_text="Predicted (CFS)" if col == 1 else "",
            row=1, col=col,
            scaleanchor=f"x{axis_sfx}",
            scaleratio=1,
        )

        show_legend = False  # legend entries only on first panel

    fig.update_layout(
        title_text=title,
        width=width,
        height=height,
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5),
        plot_bgcolor="aliceblue",
    )
    return fig
