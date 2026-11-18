"""Time-domain interactive plots.

Public API (all conditional on plotly being installed):

- :func:`plot_timeseries`        — rainfall + observed/predicted subplots
- :func:`plot_residuals_scatter` — observed-vs-predicted, residual bars, autocorrelation
- :func:`plot_cumulative_volume` — cumulative sums + volume error
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..metrics.functions import nash_sutcliffe
import pandas as pd


if TYPE_CHECKING:

    try:
        import plotly.graph_objects as go
    except ImportError:  # pragma: no cover
        pass


try:
    import plotly.graph_objects as go  # type: ignore[import]
    from plotly.subplots import make_subplots  # type: ignore[import]

    # ------------------------------------------------------------------
    # plot_timeseries
    # ------------------------------------------------------------------

    def plot_timeseries(
        datetime,
        rainfall_mm: np.ndarray,
        observed_flow,
        predicted_flow: np.ndarray,
        title: str = "Model Time Series",
        rainfall_label: str = "Rainfall [mm]",
        flow_label: str = "Flow",
    ) -> go.Figure:
        """Dual-axis time series: rainfall (top) and flow (bottom).

        The rainfall subplot uses an **inverted Y-axis** (bars grow downward)
        and shares the X-axis with the flow subplot.

        :param datetime: Time axis values (array-like of dates or timestamps).
        :type datetime: array-like
        :param rainfall_mm: Rainfall depth per time step [mm].
        :type rainfall_mm: numpy.ndarray
        :param observed_flow: Observed flow values, or ``None`` to omit.
        :type observed_flow: array-like or None
        :param predicted_flow: Model-predicted output [same units as observed].
        :type predicted_flow: numpy.ndarray
        :param title: Figure title.
        :type title: str
        :param rainfall_label: Y-axis label for the rainfall panel.
        :type rainfall_label: str
        :param flow_label: Y-axis label for the flow panel.
        :type flow_label: str
        :returns: Plotly Figure with 2 rows, shared X-axis.
        :rtype: plotly.graph_objects.Figure
        """
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.3, 0.7],
            vertical_spacing=0.04,
            subplot_titles=["Rainfall", "Flow"],
        )

        fig.add_trace(
            go.Bar(
                x=datetime,
                y=rainfall_mm,
                name=rainfall_label,
                marker_color="steelblue",
                opacity=0.7,
            ),
            row=1,
            col=1,
        )
        fig.update_yaxes(autorange="reversed", title_text=rainfall_label, row=1, col=1)

        if observed_flow is not None:
            fig.add_trace(
                go.Scatter(
                    x=datetime,
                    y=np.asarray(observed_flow),
                    name="Observed",
                    mode="lines",
                    line=dict(color="black", width=1.5),
                ),
                row=2,
                col=1,
            )

        fig.add_trace(
            go.Scatter(
                x=datetime,
                y=np.asarray(predicted_flow),
                name="Predicted",
                mode="lines",
                line=dict(color="crimson", width=2, dash="dash"),
            ),
            row=2,
            col=1,
        )
        fig.update_yaxes(title_text=flow_label, row=2, col=1)
        fig.update_xaxes(title_text="Date / Time", row=2, col=1)
        fig.update_layout(
            title=title,
            height=600,
            legend=dict(x=0.01, y=0.45, bgcolor="rgba(255,255,255,0.7)"),
            hovermode="x unified",
        )
        return fig

    # ------------------------------------------------------------------
    # plot_residuals_scatter
    # ------------------------------------------------------------------

    def plot_residuals_scatter(
        datetime,
        observed: np.ndarray,
        predicted: np.ndarray,
        title: str = "Residual Diagnostics",
        flow_label: str = "Flow",
    ) -> go.Figure:
        """Three-panel residual diagnostic plot.

        **Row 1** — Observed vs predicted scatter with a 1:1 diagonal line.

        **Row 2** — Residual (observed − predicted) bar chart, coloured red
        (over-prediction) / blue (under-prediction).

        **Row 3** — Residual autocorrelation bars at lags 0–30, showing
        systematic bias structure.

        :param datetime: Time axis for the residual bar chart.
        :type datetime: array-like
        :param observed: Observed values.
        :type observed: numpy.ndarray
        :param predicted: Predicted values.
        :type predicted: numpy.ndarray
        :param title: Figure title.
        :type title: str
        :param flow_label: Axis label used for observed / predicted axes.
        :type flow_label: str
        :returns: Plotly Figure with 3 rows.
        :rtype: plotly.graph_objects.Figure
        """
        obs = np.asarray(observed, dtype=float)
        pred = np.asarray(predicted, dtype=float)
        residuals = obs - pred

        # Goodness-of-fit metrics for the 1v1 panel
        r = float(np.corrcoef(obs, pred)[0, 1])
        nse = nash_sutcliffe(obs, pred, undefined="nan")

        max_lags = min(30, len(residuals) - 1)
        std = np.std(residuals)
        if std == 0:
            acf = np.zeros(max_lags + 1)
        else:
            full = np.correlate(residuals - residuals.mean(), residuals - residuals.mean(), mode="full")
            full /= full[len(residuals) - 1]
            acf = full[len(residuals) - 1 : len(residuals) + max_lags]

        colors = ["crimson" if r < 0 else "steelblue" for r in residuals]

        fig = make_subplots(
            rows=3,
            cols=1,
            row_heights=[0.35, 0.35, 0.3],
            vertical_spacing=0.08,
            subplot_titles=[
                "Observed vs Predicted",
                f"Residuals ({flow_label})",
                "Residual Autocorrelation",
            ],
        )

        # Row 1 — observed vs predicted
        lo = min(obs.min(), pred.min())
        hi = max(obs.max(), pred.max())
        fig.add_trace(
            go.Scatter(
                x=[lo, hi],
                y=[lo, hi],
                mode="lines",
                line=dict(color="grey", dash="dot", width=1),
                name="1:1 line",
                showlegend=False,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=obs,
                y=pred,
                mode="markers",
                marker=dict(color="steelblue", size=5, opacity=0.6),
                name="Solutions",
                showlegend=False,
            ),
            row=1, col=1,
        )
        fig.update_xaxes(title_text=f"Observed {flow_label}", row=1, col=1)
        fig.update_yaxes(title_text=f"Predicted {flow_label}", row=1, col=1)
        fig.add_annotation(
            xref="x domain", yref="y domain",
            x=0.05, y=0.95,
            text=f"r = {r:.3f}<br>NSE = {nse:.3f}",
            showarrow=False,
            align="left",
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="grey",
            borderwidth=1,
            font=dict(size=12),
            row=1, col=1,
        )

        # Row 2 — residual bars
        fig.add_trace(
            go.Bar(
                x=datetime,
                y=residuals,
                marker_color=colors,
                name="Residual",
                showlegend=False,
            ),
            row=2, col=1,
        )
        fig.add_hline(y=0, line=dict(color="black", width=1), row=2, col=1)
        fig.update_yaxes(title_text="Obs − Pred", row=2, col=1)

        # Row 3 — autocorrelation
        lags = list(range(max_lags + 1))
        acf_colors = ["crimson" if a < 0 else "steelblue" for a in acf]
        fig.add_trace(
            go.Bar(x=lags, y=acf, marker_color=acf_colors, name="ACF", showlegend=False),
            row=3, col=1,
        )
        # 95% confidence bounds (±1.96/sqrt(n))
        ci = 1.96 / np.sqrt(len(residuals))
        fig.add_hline(y=ci, line=dict(color="grey", dash="dash", width=1), row=3, col=1)
        fig.add_hline(y=-ci, line=dict(color="grey", dash="dash", width=1), row=3, col=1)
        fig.update_xaxes(title_text="Lag", row=3, col=1)
        fig.update_yaxes(title_text="Autocorrelation", row=3, col=1)

        fig.update_layout(
            title=title,
            height=800,
            hovermode="x unified",
        )
        return fig

    # ------------------------------------------------------------------
    # plot_cumulative_volume
    # ------------------------------------------------------------------

    def plot_cumulative_volume(
        datetime,
        observed: np.ndarray,
        predicted: np.ndarray,
        title: str = "Cumulative Volume",
        flow_label: str = "Flow",
    ) -> go.Figure:
        """Two-panel cumulative volume comparison.

        **Row 1** — Cumulative sums of observed and predicted with a shaded
        fill between the two lines, highlighting volume discrepancy.

        **Row 2** — Cumulative volume error (obs_cumsum − pred_cumsum) with a
        zero-reference line.

        :param datetime: Time axis values.
        :type datetime: array-like
        :param observed: Observed values.
        :type observed: numpy.ndarray
        :param predicted: Predicted values.
        :type predicted: numpy.ndarray
        :param title: Figure title.
        :type title: str
        :param flow_label: Units label for volume axis.
        :type flow_label: str
        :returns: Plotly Figure with 2 rows.
        :rtype: plotly.graph_objects.Figure
        """
        obs = np.asarray(observed, dtype=float)
        pred = np.asarray(predicted, dtype=float)
        cum_obs = np.cumsum(obs)
        cum_pred = np.cumsum(pred)
        cum_error = cum_obs - cum_pred

        dt = list(datetime)

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.6, 0.4],
            vertical_spacing=0.06,
            subplot_titles=["Cumulative Volume", "Cumulative Volume Error"],
        )

        # Row 1 — cumulative sums + fill between
        fig.add_trace(
            go.Scatter(
                x=dt,
                y=cum_obs,
                mode="lines",
                name="Observed cumulative",
                line=dict(color="black", width=2),
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=dt,
                y=cum_pred,
                mode="lines",
                name="Predicted cumulative",
                fill="tonexty",
                fillcolor="rgba(220,50,50,0.15)",
                line=dict(color="crimson", width=2, dash="dash"),
            ),
            row=1, col=1,
        )
        fig.update_yaxes(title_text=f"Cumulative {flow_label}", row=1, col=1)

        # Row 2 — volume error
        err_color = np.where(cum_error >= 0, "steelblue", "crimson").tolist()
        fig.add_trace(
            go.Scatter(
                x=dt,
                y=cum_error,
                mode="lines",
                name="Volume error",
                line=dict(color="steelblue", width=2),
                showlegend=False,
            ),
            row=2, col=1,
        )
        fig.add_hline(y=0, line=dict(color="black", width=1, dash="dot"), row=2, col=1)
        fig.update_yaxes(title_text="Obs − Pred (cumul.)", row=2, col=1)
        fig.update_xaxes(title_text="Date / Time", row=2, col=1)

        fig.update_layout(
            title=title,
            height=600,
            hovermode="x unified",
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.7)"),
        )
        return fig


    # ------------------------------------------------------------------
    # plot_calibration_timeseries
    # ------------------------------------------------------------------

    def plot_calibration_timeseries(
        datetime,
        observed: np.ndarray,
        predicted: np.ndarray,
        exogenous: "dict[str, tuple[np.ndarray, str]] | None" = None,
        pareto_predictions: "np.ndarray | None" = None,
        confidence_percentiles: tuple = (25, 75),
        tolerance_angles: "list[float] | None" = None,
        rainfall_label: str = "Rainfall",
        observed_label: str = "Observed",
        predicted_label: str = "Predicted",
        title: str = "Calibration Time Series",
    ) -> "go.Figure":
        """Two-row calibration dashboard with linked time axes.

        **Row 1 (full width)** — Exogenous inputs.  Every entry in *exogenous*
        is normalised to [0, 1] and overlaid on a single shared y-axis.  The
        legend label for each trace includes the original value range and units
        (e.g. ``Rainfall [0-12.3 mm]``).  Traces whose label contains "rain"
        or "precip" (case-insensitive) are rendered as bars; everything else is
        a line.

        **Row 2 Left** — Predicted vs Observed time series.  If
        *pareto_predictions* is supplied, a shaded IQR band (between
        *confidence_percentiles*) is drawn around the predicted line.

        **Row 2 Right** — 1v1 scatter of observed vs predicted.  When
        *pareto_predictions* is supplied, vertical box-whiskers summarise the
        Pareto range at each time step.  A dashed 45° "perfect fit" line is
        always drawn.  Optional *tolerance_angles* draw additional lines
        radiating from the origin at ``45° +/- theta``.

        The x-axes of row 1 and row 2-left are **linked**: zooming or panning
        either panel pans the other.

        :param datetime: 1-D datetime-like array aligned with *observed* /
            *predicted*.
        :type datetime: array-like
        :param observed: 1-D array of observed values.
        :type observed: numpy.ndarray
        :param predicted: 1-D array of predicted values (best solution).
        :type predicted: numpy.ndarray
        :param exogenous: ``{label: (values_array, units_string)}`` mapping.
            All traces are normalised to [0, 1] on a shared y-axis.  Rainfall-like
            traces (label contains "rain" / "precip") are plotted as bars.
        :type exogenous: dict[str, tuple[numpy.ndarray, str]] or None
        :param pareto_predictions: Array of shape ``(n_solutions, n_timesteps)``
            containing all Pareto-front predictions.  Used for the IQR band and
            scatter box-whiskers.
        :type pareto_predictions: numpy.ndarray or None
        :param confidence_percentiles: ``(lower_pct, upper_pct)`` for the IQR band.
        :type confidence_percentiles: tuple
        :param tolerance_angles: Angles in degrees from the 45° line.  For each
            ``θ`` two lines are drawn at slopes ``tan(45° ± θ)``.
        :type tolerance_angles: list[float] or None
        :param rainfall_label: String to match when auto-detecting rainfall traces.
        :type rainfall_label: str
        :param observed_label: Legend label for the observed trace.
        :type observed_label: str
        :param predicted_label: Legend label for the predicted trace.
        :type predicted_label: str
        :param title: Figure title.
        :type title: str
        :returns: Plotly Figure.
        :rtype: plotly.graph_objects.Figure
        """
        import math

        observed  = np.asarray(observed,  dtype=float)
        predicted = np.asarray(predicted, dtype=float)
        dt_series = pd.to_datetime(datetime)

        # Goodness-of-fit metrics for the 1v1 panel
        _r = float(np.corrcoef(observed, predicted)[0, 1])
        _nse = nash_sutcliffe(observed, predicted, undefined="nan")

        exogenous = exogenous or {}

        # ── Build subplot grid ───────────────────────────────────────────
        # Col 1 (65 %): Row 1 = Exogenous, Row 2 = Time series
        # Col 2 (35 %): Rows 1-2 = 1:1 Scatter (rowspan=2)
        #
        # Plotly axis numbering (reading order, skipping None):
        #   (1,1) → xaxis,  yaxis   — Exogenous
        #   (1,2) → xaxis2, yaxis2  — Scatter (rowspan=2)
        #   (2,1) → xaxis3, yaxis3  — Time series
        fig = make_subplots(
            rows=2, cols=2,
            column_widths=[0.65, 0.35],
            row_heights=[0.35, 0.65],
            specs=[
                [{},  {"rowspan": 2}],
                [{},  None          ],
            ],
            shared_xaxes=False,
            vertical_spacing=0.08,
            horizontal_spacing=0.08,
            subplot_titles=["Exogenous Inputs", "1:1 Scatter", "Predicted vs Observed"],
        )

        # ── Row 1, Col 1: exogenous traces ──────────────────────────────
        # All inputs are normalized to [0, 1] and share a single y-axis.
        # The legend label carries the original min–max range and units.
        for label, (values, units) in exogenous.items():
            values = np.asarray(values, dtype=float)
            is_rain = any(kw in label.lower() for kw in ("rain", "precip")) or \
                      any(kw in rainfall_label.lower() for kw in ("rain", "precip"))

            v_min, v_max = float(np.nanmin(values)), float(np.nanmax(values))
            v_range = v_max - v_min
            normalized = (values - v_min) / v_range if v_range > 0 else np.zeros_like(values)

            range_str = f"{v_min:.3g}–{v_max:.3g}"
            full_label = f"{label} [{range_str} {units}]" if units else f"{label} [{range_str}]"

            if is_rain:
                fig.add_trace(
                    go.Bar(
                        x=dt_series, y=normalized,
                        name=full_label,
                        marker_opacity=0.7,
                        showlegend=True,
                    ),
                    row=1, col=1,
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=dt_series, y=normalized,
                        mode="lines", name=full_label,
                    ),
                    row=1, col=1,
                )

        fig.update_yaxes(title_text="Normalized [0–1]", range=[0, 1], row=1, col=1)

        # ── Row 2, Col 1: predicted vs observed time series ─────────────
        if pareto_predictions is not None:
            pp = np.asarray(pareto_predictions, dtype=float)
            lo = np.percentile(pp, confidence_percentiles[0], axis=0)
            hi = np.percentile(pp, confidence_percentiles[1], axis=0)
            fig.add_trace(
                go.Scatter(
                    x=pd.concat([pd.Series(dt_series), pd.Series(dt_series[::-1])]),
                    y=np.concatenate([hi, lo[::-1]]),
                    fill="toself",
                    fillcolor="rgba(31,119,180,0.15)",
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=True,
                    name=f"IQR ({confidence_percentiles[0]}–{confidence_percentiles[1]}%)",
                ),
                row=2, col=1,
            )

        fig.add_trace(
            go.Scatter(x=dt_series, y=observed,  mode="lines",
                       name=observed_label,  line=dict(color="#2ca02c")),
            row=2, col=1,
        )
        fig.add_trace(
            go.Scatter(x=dt_series, y=predicted, mode="lines",
                       name=predicted_label, line=dict(color="#1f77b4")),
            row=2, col=1,
        )

        # ── Rows 1-2, Col 2: 1v1 scatter ────────────────────────────────
        max_val = float(np.nanmax([observed, predicted]))

        # Pareto box-whiskers
        if pareto_predictions is not None:
            pp = np.asarray(pareto_predictions, dtype=float)
            for t in range(len(observed)):
                col_vals = pp[:, t]
                fig.add_trace(
                    go.Box(
                        x=[observed[t]] * len(col_vals),
                        y=col_vals,
                        marker_color="rgba(31,119,180,0.3)",
                        line_color="rgba(31,119,180,0.5)",
                        showlegend=False,
                        boxpoints=False,
                    ),
                    row=1, col=2,
                )

        # Best solution markers
        fig.add_trace(
            go.Scatter(
                x=observed, y=predicted,
                mode="markers",
                name="Best solution",
                marker=dict(color="#1f77b4", size=5, opacity=0.7),
            ),
            row=1, col=2,
        )

        # 45° perfect-fit line
        fig.add_trace(
            go.Scatter(
                x=[0, max_val], y=[0, max_val],
                mode="lines",
                name="Perfect fit (45°)",
                line=dict(color="black", dash="dash", width=1.5),
            ),
            row=1, col=2,
        )

        # Tolerance angle lines
        tolerance_angles = tolerance_angles or []
        colors = ["#d62728", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2"]
        for i, theta in enumerate(tolerance_angles):
            color = colors[i % len(colors)]
            for sign, label_suffix in [(+1, "+"), (-1, "−")]:
                slope = math.tan(math.radians(45 + sign * theta))
                fig.add_trace(
                    go.Scatter(
                        x=[0, max_val],
                        y=[0, slope * max_val],
                        mode="lines",
                        name=f"45°{label_suffix}{theta}°",
                        line=dict(color=color, dash="dot", width=1),
                    ),
                    row=1, col=2,
                )

        # ── Synchronize time axes; rangeslider on the bottom time panel ──
        # xaxis  (row 1, col 1) = Exogenous  → follows xaxis3
        # xaxis3 (row 2, col 1) = Time series → carries the rangeslider
        # xaxis2 (row 1, col 2) = Scatter     → independent (static)
        fig.update_layout(
            title_text=title,
            barmode="overlay",
            legend=dict(groupclick="toggleitem"),
            height=750,
            xaxis=dict(matches="x3"),
            xaxis3=dict(rangeslider=dict(visible=True, thickness=0.05)),
        )
        fig.update_xaxes(title_text="Observed",  row=1, col=2)
        fig.update_yaxes(title_text="Predicted", row=1, col=2)
        fig.add_annotation(
            xref="x2 domain", yref="y2 domain",
            x=0.05, y=0.95,
            text=f"r = {_r:.3f}<br>NSE = {_nse:.3f}",
            showarrow=False,
            align="left",
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="grey",
            borderwidth=1,
            font=dict(size=12),
        )

        return fig

    # ------------------------------------------------------------------
    # plot_data_explorer
    # ------------------------------------------------------------------

    def plot_data_explorer(
        df: pd.DataFrame,
        event_id_col: str = "event_id",
        datetime_col: str = "datetime",
        train_event_ids=None,
        val_event_ids=None,
        variables=None,
        rainfall_cols=None,
        title: str = "Data Explorer",
    ) -> "go.Figure":
        """Interactive multi-panel data explorer with train/validation event highlighting.

        One subplot row is created per variable in *variables*, all sharing the
        same X-axis.  Rainfall columns are rendered as **inverted bar charts**
        (bars grow downward); all other variables are rendered as lines.  When
        *train_event_ids* or *val_event_ids* are supplied, the corresponding
        event time spans are shaded in blue and orange respectively.

        :param df: Input DataFrame containing a datetime column and at least one
            numeric variable column.
        :type df: pandas.DataFrame
        :param event_id_col: Column name that identifies individual events.
        :type event_id_col: str
        :param datetime_col: Column name for the datetime axis.
        :type datetime_col: str
        :param train_event_ids: Iterable of event IDs to shade as training data
            (blue, ``rgba(0,120,215,0.10)``).
        :type train_event_ids: collections.abc.Iterable or None
        :param val_event_ids: Iterable of event IDs to shade as validation data
            (orange, ``rgba(230,115,0,0.10)``).
        :type val_event_ids: collections.abc.Iterable or None
        :param variables: ``{col_name: (axis_label, units)}`` mapping describing
            which columns to plot and how to label them.  When ``None`` all
            numeric columns (excluding *event_id_col* and *datetime_col*) are
            included with their column names as labels.
        :type variables: dict[str, tuple[str, str]] or None
        :param rainfall_cols: Column names that should be rendered as inverted
            bar charts (rainfall-style).
        :type rainfall_cols: collections.abc.Iterable[str] or None
        :param title: Figure title.
        :type title: str
        :returns: Plotly Figure with *N* rows (one per variable), shared X-axis.
        :rtype: plotly.graph_objects.Figure
        """
        if variables is None:
            skip = {event_id_col, datetime_col}
            variables = {
                col: (col, "")
                for col in df.columns
                if col not in skip and pd.api.types.is_numeric_dtype(df[col])
            }

        rainfall_set = set(rainfall_cols or [])
        col_names = list(variables.keys())
        n = len(col_names)
        if n == 0:
            raise ValueError("plot_data_explorer: no numeric variables found to plot.")

        # Row heights: rainfall rows 20%, other rows share remaining equally
        rain_rows = [c in rainfall_set for c in col_names]
        n_rain = sum(rain_rows)
        n_other = n - n_rain
        rain_h = 0.20
        other_h = (1.0 - n_rain * rain_h) / max(n_other, 1)
        heights = [rain_h if r else other_h for r in rain_rows]

        subplot_titles = [
            f"{variables[c][0]} [{variables[c][1]}]" if variables[c][1] else variables[c][0]
            for c in col_names
        ]

        fig = make_subplots(
            rows=n,
            cols=1,
            shared_xaxes=True,
            row_heights=heights,
            vertical_spacing=0.04,
            subplot_titles=subplot_titles,
        )

        dt = df[datetime_col]

        # ── Variable traces ──────────────────────────────────────────────
        for i, col in enumerate(col_names, start=1):
            label, units = variables[col]
            y_title = f"{label} [{units}]" if units else label

            if col in rainfall_set:
                fig.add_trace(
                    go.Bar(
                        x=dt,
                        y=df[col],
                        name=label,
                        marker_color="steelblue",
                        opacity=0.7,
                    ),
                    row=i, col=1,
                )
                fig.update_yaxes(autorange="reversed", title_text=y_title, row=i, col=1)
            else:
                fig.add_trace(
                    go.Scatter(x=dt, y=df[col], mode="lines", name=label),
                    row=i, col=1,
                )
                fig.update_yaxes(title_text=y_title, row=i, col=1)

        fig.update_xaxes(title_text="Date / Time", row=n, col=1)

        # ── Event highlighting ───────────────────────────────────────────
        if event_id_col in df.columns and (
            train_event_ids is not None or val_event_ids is not None
        ):
            spans = (
                df.groupby(event_id_col)[datetime_col]
                .agg(["min", "max"])
                .to_dict(orient="index")
            )

            for eid in (train_event_ids or []):
                if eid in spans:
                    fig.add_vrect(
                        x0=spans[eid]["min"],
                        x1=spans[eid]["max"],
                        fillcolor="rgba(0,120,215,0.10)",
                        line_width=0,
                        row="all",
                        col="all",
                    )
            for eid in (val_event_ids or []):
                if eid in spans:
                    fig.add_vrect(
                        x0=spans[eid]["min"],
                        x1=spans[eid]["max"],
                        fillcolor="rgba(230,115,0,0.10)",
                        line_width=0,
                        row="all",
                        col="all",
                    )

            # Legend entries for train / validation bands
            if train_event_ids:
                fig.add_trace(
                    go.Scatter(
                        x=[None], y=[None], mode="lines",
                        name="Training",
                        line=dict(color="rgba(0,120,215,0.7)", width=10),
                        showlegend=True,
                    ),
                    row=1, col=1,
                )
            if val_event_ids:
                fig.add_trace(
                    go.Scatter(
                        x=[None], y=[None], mode="lines",
                        name="Validation",
                        line=dict(color="rgba(230,115,0,0.7)", width=10),
                        showlegend=True,
                    ),
                    row=1, col=1,
                )

        fig.update_layout(
            title=title,
            height=max(300 * n, 400),
            hovermode="x unified",
            legend=dict(x=1.02, y=1.0),
            barmode="overlay",
        )
        return fig

    # ------------------------------------------------------------------
    # plot_ensemble_timeseries
    # ------------------------------------------------------------------

    def plot_ensemble_timeseries(
        datetime,
        rainfall_mm,
        observed: np.ndarray,
        pred_df: pd.DataFrame,
        aliases: list,
        output_name: str = "ensemble_output",
        observed_label: str = "Observed",
        rainfall_label: str = "Rainfall [mm]",
        title: str = "Ensemble Components",
    ) -> "go.Figure":
        """Two-panel ensemble component breakdown.

        **Row 1 (25%)** — Rainfall bars with an inverted Y-axis.  Omitted
        entirely when *rainfall_mm* is ``None``.

        **Row 2 (75%)** — Observed flow (black solid line), total predicted
        (crimson dashed line), and one **stacked area** per alias showing each
        component's individual contribution.

        :param datetime: Time axis aligned with *observed*.
        :type datetime: array-like
        :param rainfall_mm: Rainfall depth array, or ``None`` to skip the top
            panel.
        :type rainfall_mm: numpy.ndarray or None
        :param observed: 1-D array of observed flow values.
        :type observed: numpy.ndarray
        :param pred_df: Output of :meth:`~sparsehydro.models.EnsembleModel.predict`
            containing ``{alias}_output`` columns and the *output_name* column.
        :type pred_df: pandas.DataFrame
        :param aliases: List of alias strings matching the EnsembleModel's
            ``aliases`` attribute (e.g. ``["rdii", "sanitary"]``).
        :type aliases: list[str]
        :param output_name: Column name for the combined output in *pred_df*.
        :type output_name: str
        :param observed_label: Legend label for the observed trace.
        :type observed_label: str
        :param rainfall_label: Y-axis label and legend label for rainfall.
        :type rainfall_label: str
        :param title: Figure title.
        :type title: str
        :returns: Plotly Figure with 1–2 rows.
        :rtype: plotly.graph_objects.Figure
        """
        observed = np.asarray(observed, dtype=float)
        has_rain = rainfall_mm is not None

        component_colors = [
            ("rgba(31,119,180,0.45)",  "#1f77b4"),
            ("rgba(44,160,44,0.45)",   "#2ca02c"),
            ("rgba(255,127,14,0.45)",  "#ff7f0e"),
            ("rgba(148,103,189,0.45)", "#9467bd"),
            ("rgba(214,39,40,0.45)",   "#d62728"),
        ]

        if has_rain:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                row_heights=[0.25, 0.75],
                vertical_spacing=0.04,
                subplot_titles=["Rainfall", "Flow Components"],
            )
            fig.add_trace(
                go.Bar(
                    x=datetime,
                    y=np.asarray(rainfall_mm, dtype=float),
                    name=rainfall_label,
                    marker_color="steelblue",
                    opacity=0.7,
                ),
                row=1, col=1,
            )
            fig.update_yaxes(autorange="reversed", title_text=rainfall_label, row=1, col=1)
            flow_row = 2
        else:
            fig = make_subplots(rows=1, cols=1, subplot_titles=["Flow Components"])
            flow_row = 1

        pred_dt = pred_df["datetime"] if "datetime" in pred_df.columns else datetime

        # Stacked component areas (drawn first so they appear behind the lines)
        for i, alias in enumerate(aliases):
            col_name = f"{alias}_output"
            if col_name not in pred_df.columns:
                continue
            fill_color, line_color = component_colors[i % len(component_colors)]
            fig.add_trace(
                go.Scatter(
                    x=pred_dt,
                    y=pred_df[col_name].to_numpy(dtype=float),
                    mode="lines",
                    name=alias.replace("_", " ").title(),
                    stackgroup="components",
                    fillcolor=fill_color,
                    line=dict(color=line_color, width=1),
                ),
                row=flow_row, col=1,
            )

        # Total predicted (dashed line, on top of stacked areas)
        if output_name in pred_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=pred_dt,
                    y=pred_df[output_name].to_numpy(dtype=float),
                    mode="lines",
                    name="Predicted (total)",
                    line=dict(color="crimson", width=2, dash="dash"),
                ),
                row=flow_row, col=1,
            )

        # Observed (solid line, drawn last to appear on top)
        fig.add_trace(
            go.Scatter(
                x=datetime,
                y=observed,
                mode="lines",
                name=observed_label,
                line=dict(color="black", width=1.5),
            ),
            row=flow_row, col=1,
        )

        fig.update_yaxes(title_text="Flow", row=flow_row, col=1)
        fig.update_xaxes(title_text="Date / Time", row=flow_row, col=1)
        fig.update_layout(
            title=title,
            height=600,
            hovermode="x unified",
            legend=dict(x=1.02, y=1.0),
        )
        return fig

    # ------------------------------------------------------------------
    # plot_ensemble_components
    # ------------------------------------------------------------------

    def plot_ensemble_components(
        datetime,
        rainfall,
        observed: np.ndarray,
        pred_df: pd.DataFrame,
        aliases: list,
        output_name: str = "ensemble_output",
        pareto_predictions: "np.ndarray | None" = None,
        confidence_percentiles: tuple = (10, 90),
        component_labels: "dict | None" = None,
        observed_label: str = "Observed",
        rainfall_label: str = "Rainfall [mm]",
        flow_label: str = "Flow",
        title: str = "Ensemble Components",
    ) -> "go.Figure":
        """Multi-panel figure: rainfall | component signals | combined + Pareto band.

        Three rows sharing a common X-axis (two when *rainfall* is ``None``):

        * **Row 1** — Rainfall bars (inverted axis).  Omitted when *rainfall* is
          ``None``.
        * **Row 2** — Individual component signals, one filled-area trace per alias.
          Components share a common Y-axis so their magnitudes can be compared
          directly (not stacked).
        * **Row 3** — Combined predicted signal (dashed), observed (solid black), and
          an optional Pareto uncertainty band (shaded IQR between
          *confidence_percentiles*).  Component outlines (thin dotted) are also
          repeated here to show each contribution in context.

        :param datetime: 1-D time axis aligned with *observed*.
        :type datetime: array-like
        :param rainfall: Rainfall array, or ``None`` to omit the top panel.
        :type rainfall: numpy.ndarray or None
        :param observed: 1-D observed flow array.
        :type observed: numpy.ndarray
        :param pred_df: Output of
            :meth:`~sparsehydro.models.EnsembleModel.predict`, containing
            ``{alias}_output`` columns and *output_name*.
        :type pred_df: pandas.DataFrame
        :param aliases: List of alias strings (e.g. ``["rdii", "seas"]``).
        :type aliases: list[str]
        :param output_name: Column name for the combined output in *pred_df*.
        :type output_name: str
        :param pareto_predictions: Array of shape ``(n_solutions, n_timesteps)``
            of combined-output predictions for all Pareto-front solutions.
            Produced by
            :meth:`~sparsehydro.models.EnsembleModel.collect_pareto_predictions`.
            When supplied, a shaded band is drawn in Row 3.
        :type pareto_predictions: numpy.ndarray or None
        :param confidence_percentiles: ``(lower_pct, upper_pct)`` for the band.
        :type confidence_percentiles: tuple
        :param component_labels: ``{alias: display_label}`` overrides.  Missing
            aliases default to the alias string (title-cased).
        :type component_labels: dict or None
        :param observed_label: Legend label for the observed trace.
        :type observed_label: str
        :param rainfall_label: Rainfall panel label.
        :type rainfall_label: str
        :param flow_label: Y-axis label for component and combined panels.
        :type flow_label: str
        :param title: Figure title.
        :type title: str
        :returns: Plotly Figure with 2–3 rows, shared X-axis.
        :rtype: plotly.graph_objects.Figure
        """
        observed = np.asarray(observed, dtype=float)
        has_rain = rainfall is not None
        labels = component_labels or {}

        _COLORS = [
            ("#1f77b4", "rgba(31,119,180,0.18)"),
            ("#2ca02c", "rgba(44,160,44,0.18)"),
            ("#ff7f0e", "rgba(255,127,14,0.18)"),
            ("#9467bd", "rgba(148,103,189,0.18)"),
            ("#8c564b", "rgba(140,86,75,0.18)"),
        ]

        if has_rain:
            n_rows = 3
            row_heights = [0.13, 0.37, 0.50]
            subplot_titles = [rainfall_label, "Component Signals", "Combined vs Observed"]
        else:
            n_rows = 2
            row_heights = [0.40, 0.60]
            subplot_titles = ["Component Signals", "Combined vs Observed"]

        fig = make_subplots(
            rows=n_rows, cols=1,
            shared_xaxes=True,
            row_heights=row_heights,
            vertical_spacing=0.04,
            subplot_titles=subplot_titles,
        )

        comp_row = 2 if has_rain else 1
        comb_row = 3 if has_rain else 2

        pred_dt = pred_df["datetime"] if "datetime" in pred_df.columns else datetime

        # ── Row 1: Rainfall ──────────────────────────────────────────────
        if has_rain:
            fig.add_trace(
                go.Bar(
                    x=datetime,
                    y=np.asarray(rainfall, dtype=float),
                    name=rainfall_label,
                    marker_color="steelblue",
                    opacity=0.7,
                    showlegend=False,
                ),
                row=1, col=1,
            )
            fig.update_yaxes(autorange="reversed", title_text=rainfall_label, row=1, col=1)

        # ── Row 2: Individual component signals ──────────────────────────
        for i, alias in enumerate(aliases):
            col_name = f"{alias}_output"
            if col_name not in pred_df.columns:
                continue
            line_color, fill_color = _COLORS[i % len(_COLORS)]
            label = labels.get(alias, alias.replace("_", " ").title())
            fig.add_trace(
                go.Scatter(
                    x=pred_dt,
                    y=pred_df[col_name].to_numpy(dtype=float),
                    mode="lines",
                    name=label,
                    line=dict(color=line_color, width=1.5),
                    fill="tozeroy",
                    fillcolor=fill_color,
                    legendgroup=alias,
                ),
                row=comp_row, col=1,
            )
        fig.update_yaxes(title_text=flow_label, row=comp_row, col=1)

        # ── Row 3: Combined + Pareto band + observed ─────────────────────
        if pareto_predictions is not None:
            pp = np.asarray(pareto_predictions, dtype=float)
            lo = np.percentile(pp, confidence_percentiles[0], axis=0)
            hi = np.percentile(pp, confidence_percentiles[1], axis=0)
            dt_arr = np.asarray(pred_dt)
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([dt_arr, dt_arr[::-1]]),
                    y=np.concatenate([hi, lo[::-1]]),
                    fill="toself",
                    fillcolor="rgba(220,50,50,0.12)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name=(
                        f"Pareto band "
                        f"({confidence_percentiles[0]}–{confidence_percentiles[1]}%)"
                    ),
                    showlegend=True,
                ),
                row=comb_row, col=1,
            )

        # Component outlines (thin dotted) for context in the combined panel
        for i, alias in enumerate(aliases):
            col_name = f"{alias}_output"
            if col_name not in pred_df.columns:
                continue
            line_color, _ = _COLORS[i % len(_COLORS)]
            label = labels.get(alias, alias.replace("_", " ").title())
            fig.add_trace(
                go.Scatter(
                    x=pred_dt,
                    y=pred_df[col_name].to_numpy(dtype=float),
                    mode="lines",
                    name=label,
                    line=dict(color=line_color, width=1, dash="dot"),
                    showlegend=False,
                    legendgroup=alias,
                ),
                row=comb_row, col=1,
            )

        if output_name in pred_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=pred_dt,
                    y=pred_df[output_name].to_numpy(dtype=float),
                    mode="lines",
                    name="Predicted (total)",
                    line=dict(color="crimson", width=2, dash="dash"),
                ),
                row=comb_row, col=1,
            )

        fig.add_trace(
            go.Scatter(
                x=datetime,
                y=observed,
                mode="lines",
                name=observed_label,
                line=dict(color="black", width=1.5),
            ),
            row=comb_row, col=1,
        )
        fig.update_yaxes(title_text=flow_label, row=comb_row, col=1)
        fig.update_xaxes(title_text="Date / Time", row=comb_row, col=1)

        fig.update_layout(
            title=title,
            height=700,
            hovermode="x unified",
            legend=dict(x=1.02, y=1.0),
            barmode="overlay",
        )
        return fig

    # ------------------------------------------------------------------
    # plot_ensemble_members_vs_observed
    # ------------------------------------------------------------------

    def plot_ensemble_members_vs_observed(
        datetime,
        observed: np.ndarray,
        pred_df: pd.DataFrame,
        aliases: list,
        output_name: str = "ensemble_output",
        rainfall=None,
        component_labels: "dict | None" = None,
        observed_label: str = "Observed",
        rainfall_label: str = "Rainfall [mm]",
        flow_label: str = "Flow",
        title: str = "Ensemble Members vs Observed",
    ) -> "go.Figure":
        """Small-multiples grid: each ensemble member vs. the observed target, plus combined.

        One row per alias overlays that component's ``{alias}_output`` series with
        the observed target, followed by a final row for the combined
        *output_name* series vs. observed.  This differs from
        :func:`plot_ensemble_components`, which overlays members with each other
        (not against observed) and only shows the combined series vs. observed —
        here every row repeats the observed trace so each member's individual fit
        can be judged directly.

        :param datetime: 1-D time axis aligned with *observed*.
        :type datetime: array-like
        :param observed: 1-D observed flow array.
        :type observed: numpy.ndarray
        :param pred_df: Output of
            :meth:`~sparsehydro.models.EnsembleModel.predict`, containing
            ``{alias}_output`` columns and *output_name*.
        :type pred_df: pandas.DataFrame
        :param aliases: List of alias strings — one row per alias.
        :type aliases: list[str]
        :param output_name: Column name for the combined output in *pred_df*.
        :type output_name: str
        :param rainfall: Optional rainfall array for a shared top panel.
        :type rainfall: numpy.ndarray or None
        :param component_labels: ``{alias: display_label}`` overrides.  Missing
            aliases default to the alias string (title-cased).
        :type component_labels: dict or None
        :param observed_label: Legend label for the observed trace.
        :type observed_label: str
        :param rainfall_label: Rainfall panel label.
        :type rainfall_label: str
        :param flow_label: Y-axis label for all flow panels.
        :type flow_label: str
        :param title: Figure title.
        :type title: str
        :returns: Plotly Figure with one row per member, one combined row, and
            an optional leading rainfall row — all sharing the X-axis.
        :rtype: plotly.graph_objects.Figure
        """
        observed = np.asarray(observed, dtype=float)
        has_rain = rainfall is not None
        labels = component_labels or {}

        _COLORS = [
            ("#1f77b4", "rgba(31,119,180,0.18)"),
            ("#2ca02c", "rgba(44,160,44,0.18)"),
            ("#ff7f0e", "rgba(255,127,14,0.18)"),
            ("#9467bd", "rgba(148,103,189,0.18)"),
            ("#8c564b", "rgba(140,86,75,0.18)"),
        ]

        n_members = len(aliases)
        n_rows = n_members + 1 + (1 if has_rain else 0)

        if has_rain:
            row_heights = [0.12] + [0.88 / (n_members + 1)] * (n_members + 1)
        else:
            row_heights = [1.0 / (n_members + 1)] * (n_members + 1)

        subplot_titles = [rainfall_label] if has_rain else []
        for alias in aliases:
            subplot_titles.append(labels.get(alias, alias.replace("_", " ").title()))
        subplot_titles.append("Combined")

        fig = make_subplots(
            rows=n_rows, cols=1,
            shared_xaxes=True,
            row_heights=row_heights,
            vertical_spacing=0.03,
            subplot_titles=subplot_titles,
        )

        row_offset = 1
        if has_rain:
            fig.add_trace(
                go.Bar(
                    x=datetime,
                    y=np.asarray(rainfall, dtype=float),
                    name=rainfall_label,
                    marker_color="steelblue",
                    opacity=0.7,
                    showlegend=False,
                ),
                row=1, col=1,
            )
            fig.update_yaxes(autorange="reversed", title_text=rainfall_label, row=1, col=1)
            row_offset = 2

        pred_dt = pred_df["datetime"] if "datetime" in pred_df.columns else datetime

        for i, alias in enumerate(aliases):
            row = row_offset + i
            col_name = f"{alias}_output"
            line_color, _ = _COLORS[i % len(_COLORS)]
            label = labels.get(alias, alias.replace("_", " ").title())
            fig.add_trace(
                go.Scatter(
                    x=datetime, y=observed,
                    mode="lines", name=observed_label,
                    line=dict(color="black", width=1.2),
                    legendgroup="observed", showlegend=(i == 0),
                ),
                row=row, col=1,
            )
            if col_name in pred_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=pred_dt, y=pred_df[col_name].to_numpy(dtype=float),
                        mode="lines", name=label,
                        line=dict(color=line_color, width=1.5),
                    ),
                    row=row, col=1,
                )
            fig.update_yaxes(title_text=flow_label, row=row, col=1)

        comb_row = row_offset + n_members
        fig.add_trace(
            go.Scatter(
                x=datetime, y=observed,
                mode="lines", name=observed_label,
                line=dict(color="black", width=1.2),
                legendgroup="observed", showlegend=False,
            ),
            row=comb_row, col=1,
        )
        if output_name in pred_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=pred_dt, y=pred_df[output_name].to_numpy(dtype=float),
                    mode="lines", name="Combined",
                    line=dict(color="crimson", width=2, dash="dash"),
                ),
                row=comb_row, col=1,
            )
        fig.update_yaxes(title_text=flow_label, row=comb_row, col=1)
        fig.update_xaxes(title_text="Date / Time", row=comb_row, col=1)

        fig.update_layout(
            title=title,
            height=220 * n_rows,
            hovermode="x unified",
        )
        return fig

    # ------------------------------------------------------------------
    # plot_one_to_one_band
    # ------------------------------------------------------------------

    def plot_one_to_one_band(
        observed,
        predicted,
        band_pct: float = 10.0,
        title: str = "1:1 Comparison",
        observed_label: str = "Observed",
        predicted_label: str = "Predicted",
        flow_label: str = "Flow",
    ) -> "go.Figure":
        """CIWEM-style 1:1 scatter with a proportional +/- band tolerance wedge.

        Draws the observed-vs-predicted point cloud, a dashed 45 degree
        "perfect fit" line, and a shaded wedge bounded by
        ``y = (1 - band_pct/100)*x`` and ``y = (1 + band_pct/100)*x`` — a band
        that starts at the origin and widens proportionally with magnitude, per
        common CIWEM-style flow-model acceptance criteria.  This differs from
        the ``tolerance_angles`` option on :func:`plot_calibration_timeseries`,
        which draws angular offsets from the 45 degree line rather than a fixed
        percentage of the value.

        :param observed: 1-D observed values.
        :type observed: numpy.ndarray
        :param predicted: 1-D predicted values, same length as *observed*.
        :type predicted: numpy.ndarray
        :param band_pct: Tolerance band half-width as a percentage of value
            (e.g. ``10.0`` draws the wedge between +/-10% of the 1:1 line).
        :type band_pct: float
        :param title: Figure title.
        :type title: str
        :param observed_label: X-axis label.
        :type observed_label: str
        :param predicted_label: Y-axis label.
        :type predicted_label: str
        :param flow_label: Units/quantity label appended to axis titles.
        :type flow_label: str
        :returns: Plotly Figure with the 1:1 scatter and tolerance band.
        :rtype: plotly.graph_objects.Figure
        """
        obs = np.asarray(observed, dtype=float)
        pred = np.asarray(predicted, dtype=float)

        r = float(np.corrcoef(obs, pred)[0, 1])
        nse = nash_sutcliffe(obs, pred, undefined="nan")

        lo_slope = 1.0 - band_pct / 100.0
        hi_slope = 1.0 + band_pct / 100.0
        in_band = np.where(
            obs > 0,
            (pred >= lo_slope * obs) & (pred <= hi_slope * obs),
            pred == 0,
        )
        pct_in_band = 100.0 * float(np.mean(in_band))

        max_val = float(np.nanmax([obs.max(), pred.max(), 0.0]))

        fig = go.Figure()

        # Shaded wedge from the origin: (0,0) -> (max, lo*max) -> (max, hi*max)
        fig.add_trace(go.Scatter(
            x=[0, max_val, max_val, 0],
            y=[0, lo_slope * max_val, hi_slope * max_val, 0],
            fill="toself",
            fillcolor="rgba(44,160,44,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name=f"+/-{band_pct:g}% band",
            showlegend=True,
        ))

        fig.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode="lines", name="Perfect fit (45 deg)",
            line=dict(color="black", dash="dash", width=1.5),
        ))

        marker_colors = np.where(in_band, "#2ca02c", "#d62728")
        fig.add_trace(go.Scatter(
            x=obs, y=pred,
            mode="markers",
            name="Solutions",
            marker=dict(color=marker_colors, size=6, opacity=0.7),
        ))

        fig.update_xaxes(title_text=f"{observed_label} {flow_label}".strip())
        fig.update_yaxes(title_text=f"{predicted_label} {flow_label}".strip())
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            text=(
                f"r = {r:.3f}<br>NSE = {nse:.3f}<br>"
                f"{pct_in_band:.1f}% within +/-{band_pct:g}%"
            ),
            showarrow=False,
            align="left",
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="grey",
            borderwidth=1,
            font=dict(size=12),
        )
        fig.update_layout(
            title=title,
            height=600,
            width=650,
        )
        return fig

except ImportError:

    def plot_timeseries(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("plotly is required for plot_timeseries. Install with: pip install plotly")

    def plot_residuals_scatter(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("plotly is required for plot_residuals_scatter. Install with: pip install plotly")

    def plot_cumulative_volume(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("plotly is required for plot_cumulative_volume. Install with: pip install plotly")

    def plot_calibration_timeseries(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("plotly is required for plot_calibration_timeseries. Install with: pip install plotly")

    def plot_data_explorer(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("plotly is required for plot_data_explorer. Install with: pip install plotly")

    def plot_ensemble_timeseries(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("plotly is required for plot_ensemble_timeseries. Install with: pip install plotly")

    def plot_ensemble_components(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("plotly is required for plot_ensemble_components. Install with: pip install plotly")

    def plot_ensemble_members_vs_observed(*args, **kwargs):  # type: ignore[misc]
        raise ImportError(
            "plotly is required for plot_ensemble_members_vs_observed. Install with: pip install plotly"
        )

    def plot_one_to_one_band(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("plotly is required for plot_one_to_one_band. Install with: pip install plotly")

