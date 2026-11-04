"""Full-resolution diagnostics extracted from the pattern whole-flow notebook.

Builders return figures without displaying or exporting them. Clients can thin
inline copies while retaining the complete HTML artifact.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sparsehydro.calibration.objectives import ConcordanceCorrelationCoefficient, NashSutcliffe
from sparsehydro.calibration.msdgc_pareto import MSDGC_METRIC_ORDER
from sparsehydro.calibration.result import CalibrationResult
from sparsehydro.data.services import DownloadResult
from sparsehydro.models.wholeflow_pattern import CenteredWeeklyPattern, PatternWholeFlowModel
from .calibration import plot_pareto_scatter_matrix
from .display import add_val_vrects


def plot_water_balance(
    data: pd.DataFrame, prediction: pd.DataFrame, model: PatternWholeFlowModel, *, dt_hours: float,
) -> go.Figure:
    """Verify and plot the full selected-area precipitation/storage budget."""
    area = float(model.get_scalar_parameter("area_acres").value)
    precipitation = model.budget["initial_storage_in"] + np.cumsum(data["rainfall_in"].to_numpy())
    accounted = np.cumsum(
        prediction["et_in"] + (prediction["gwi_cfs"] + prediction["rdii_cfs"])
        * (dt_hours * 3600) / (area * 3630) + prediction["unrouted_excess_in"])
    accounted += (prediction["S_in"] + prediction["G_in"]
                  + prediction["routing_storage_in"] + prediction["SWE_in"])
    np.testing.assert_allclose(precipitation, accounted, atol=1e-8, rtol=0,
                               err_msg="Displayed budget must close using the selected contributing area")
    fig = go.Figure()
    for values, name in ((precipitation, "Precipitation + initial storage"),
                         (accounted, "Outputs + S + G + routing + SWE")):
        fig.add_trace(go.Scattergl(x=data["datetime"], y=values, name=name, mode="lines"))
    fig.update_layout(title="Complete precipitation budget including snow", yaxis_title="Depth [in]")
    return fig


def plot_snow_diagnostics(
    data: pd.DataFrame, prediction: pd.DataFrame, model: PatternWholeFlowModel, *,
    data_title: str, warmup_steps: int, val_mask: np.ndarray,
) -> go.Figure:
    """Show upstream snow partition, liquid forcing, storage, and selected flow."""
    times = pd.to_datetime(data["datetime"])
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=.04,
        subplot_titles=("Temperature / rain-snow threshold", "Precipitation partition and melt",
                        "Snow water equivalent (stored water)", "Observed and modeled total flow"))
    fig.add_trace(go.Scattergl(x=times, y=data["temperature_c"],
                              name="Temperature [C]", mode="lines"), row=1, col=1)
    if model.snow_config is not None:
        fig.add_hline(y=model.snow_config.threshold_c, line_dash="dash", row=1, col=1)
    for column, label in (("rain_liquid_in", "Liquid rain"), ("snowfall_in", "Snowfall (water equivalent)"),
                          ("melt_in", "Snowmelt"), ("liquid_input_in", "Rain + melt to S/G")):
        fig.add_trace(go.Scattergl(x=times, y=prediction[column], name=label, mode="lines"), row=2, col=1)
    fig.add_trace(go.Scattergl(x=times, y=prediction["SWE_in"], name="SWE",
                              mode="lines", fill="tozeroy"), row=3, col=1)
    for values, label in ((data["flow_cfs"], "Observed total"), (prediction["total_cfs"], "Modeled total"),
                          (prediction["gwi_cfs"], "G-store discharge"), (prediction["rdii_cfs"], "RTK RDII")):
        fig.add_trace(go.Scattergl(x=times, y=values, name=label, mode="lines"), row=4, col=1)
    add_val_vrects(fig, times, val_mask, row=4, col=1)
    if warmup_steps > 0:
        fig.add_vrect(x0=times.iloc[0], x1=times.iloc[warmup_steps - 1],
                      fillcolor="#9E9E9E", opacity=.12, line_width=0)
    for row, unit in enumerate(("C", "in/step", "in SWE", "cfs"), 1):
        fig.update_yaxes(title_text=unit, row=row, col=1)
    fig.update_layout(template="plotly_white", height=1000, hovermode="x unified",
                      title=f"{data_title} - Snow upstream of unchanged S/G/RTK physics")
    return fig


def plot_baseflow_components(data: pd.DataFrame, *, val_mask: np.ndarray) -> go.Figure:
    """Distinguish the detection reference from fitted sanitary and GWI flow."""
    fig = go.Figure()
    for column, name in (("baseflow_cfs", "Training-derived detection reference (estimate)"),
                         ("joint_baseflow_cfs", "Predicted sanitary + GWI"),
                         ("joint_sanitary_cfs", "Jointly fitted sanitary"), ("gwi_cfs", "Separate-store GWI")):
        fig.add_trace(go.Scattergl(x=data["datetime"], y=data[column], name=name, mode="lines"))
    add_val_vrects(fig, data["datetime"], val_mask)
    fig.update_layout(title="Baseflow components; reference is not observed truth", yaxis_title="Flow [cfs]")
    return fig


def _series(result: DownloadResult | None) -> pd.Series | None:
    """First value column of a parsed DownloadResult as a Series, or None."""
    if result is None or result.dataframe is None or result.dataframe.empty:
        return None
    return result.dataframe.iloc[:, 0]


def plot_input_timeseries(results: Mapping[str, DownloadResult], data_title: str) -> go.Figure:
    """Plot native rainfall and available flow inputs without resampling."""
    # ── Plot: rainfall hyetograph (top, inverted) over flow series (bottom) ──────





    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.28, 0.72], vertical_spacing=0.05,
        subplot_titles=("Rainfall", "Flow"),
    )

    # Rainfall on the top row as a WebGL filled area on an inverted axis, so it
    # hangs from the top like a hyetograph. (A filled Scattergl trace stays fast for
    # a full year of 5-minute data, where 100k+ Bar rects would not.)
    rain = _series(results.get("Rain"))

    if rain is not None:
        fig.add_trace(
            go.Scattergl(x=rain.index, y=rain.values, name="Rainfall", mode="lines",
                         line=dict(color="#1f77b4", width=0.5), fill="tozeroy",
                         fillcolor="rgba(31,119,180,0.45)"),
            row=1, col=1,
        )
        fig.update_yaxes(title_text="Rain [in]", autorange="reversed", row=1, col=1)

    # Flow / base-flow / storm-flow on the bottom row as WebGL line traces.
    for data_type, color in (("Flow", "#d62728"), ("BaseFlow", "#2ca02c"),
                             ("StormFlow", "#ff7f0e")):
        s = _series(results.get(data_type))
        if s is not None:
            fig.add_trace(
                go.Scattergl(x=s.index, y=s.values, mode="lines", name=data_type,
                             line=dict(color=color, width=1)),
                row=2, col=1,
            )

    fig.update_yaxes(title_text="Flow [MGD]", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_layout(
        title=f"FlowFinity — {data_title}",
        height=800, 
        width=1600,
        hovermode="x unified",
        # legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0),
    )
    #thin_for_display(fig, DISPLAY_MAX_POINTS).show()
    return fig


def plot_sanitary_shapes(pattern: CenteredWeeklyPattern) -> go.Figure:
    """Show centered training-derived hourly and weekday anomalies."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Centered hourly shape", "Centered weekday shape"))
    fig.add_trace(go.Scattergl(x=list(range(24)), y=pattern.hourly,
                              name="hour anomaly"), row=1, col=1)
    fig.add_trace(go.Scattergl(x=list(range(7)), y=pattern.weekday,
                              name="weekday anomaly"), row=1, col=2)
    fig.update_yaxes(title_text="Anomaly [cfs]")
    fig.update_layout(title="Training-derived shapes only; mean and amplitudes are fitted jointly")
    return fig


def plot_ia_store(data: pd.DataFrame, model: PatternWholeFlowModel, *, data_title: str, warmup_steps: int) -> go.Figure:
    """Show the selected IA storage and liquid input, including the unscored warmup."""
    # ── P1 · IA store state S(t) vs rainfall ─────────────────────────────────────

    _t = pd.to_datetime(data["datetime"])
    _S = data["S_in"].to_numpy()
    _rain = data["liquid_input_in"].to_numpy()
    _s_max = model.get_scalar_parameter("s_max").value
    _rmax = max(float(np.nanmax(_rain)), 1e-6)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scattergl(x=_t, y=_S, name="IA store S = filled depth [in]",
                               mode="lines", line=dict(color="#8c564b", width=1.3),
                               fill="tozeroy", fillcolor="rgba(140,86,75,0.18)"),
                  secondary_y=False)
    # fig.add_hline(y=_s_max, line=dict(color="#8c564b", width=1, dash="dot"),
    #               annotation_text=f"s_max = {_s_max:.3f} in", annotation_position="bottom right")
    # fig.add_trace(go.Bar(x=_t, y=_rain, name="Rainfall [in/step]",
    #                      marker=dict(color="#1f77b4")), secondary_y=True)
    # if not data.empty:
    if not data.empty:
        fig.add_trace(
            go.Scattergl(x=_t, y=_rain, name="Rain + snowmelt", mode="lines",
                         line=dict(color="#faa811", width=1.0), fill=None,
                         fillcolor="rgba(31,119,180,0.45)"), secondary_y=True,
            row=1, col=1,
        )


    if warmup_steps > 0:
        fig.add_vrect(x0=_t.iloc[0], x1=_t.iloc[warmup_steps - 1], fillcolor="#9E9E9E",
                      opacity=0.10, line_width=0, annotation_text="warmup",
                      annotation_position="top left")
    fig.update_yaxes(title_text="IA store S [in]", secondary_y=False)
    fig.update_yaxes(title_text="Liquid input [in/step]", range=[_rmax * 4.0, 0.0],
                     secondary_y=True, showgrid=False)
    fig.update_layout(template="plotly_white", height=500, width=1500, hovermode="x unified",
                      title=f"{data_title} - Method B: IA store S(t) vs rain + snowmelt",
                      xaxis_title="Date", bargap=0,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    return fig


def plot_groundwater_store(data: pd.DataFrame, prediction: pd.DataFrame, model: PatternWholeFlowModel, *, data_title: str, dt_hours: float) -> go.Figure:
    """Show separate S/G depth scales, slow groundwater release, and capacity spill."""
    _t = pd.to_datetime(data["datetime"])
    _S = data["S_in"].to_numpy()
    # ── P2 · Deep store G(t) alongside S(t), and the GWI it releases ─────────────
    _G = data["G_in"].to_numpy()
    _g_max = model.get_scalar_parameter("g_max").value

    _depth_to_cfs = float(model.get_scalar_parameter("area_acres").value) * 3630 / (dt_hours * 3600)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.3, 0.3, 0.4],
                        vertical_spacing=0.07,
                        subplot_titles=("S: source store", "G: own depth scale, with capacity",
                                        "G-store discharge = slow release + spill"))
    fig.add_trace(go.Scattergl(x=_t, y=_S, name="S [in]", mode="lines",
                               line=dict(color="#8c564b", width=1.0)), row=1, col=1)
    fig.add_trace(go.Scattergl(x=_t, y=_G, name="G [in] (deep store)", mode="lines",
                               line=dict(color="#9467bd", width=1.5),
                               fill="tozeroy", fillcolor="rgba(148,103,189,0.20)"), row=2, col=1)
    fig.add_hline(y=_g_max, line=dict(color="#9467bd", width=1, dash="dot"),
                  annotation_text=f"g_max = {_g_max:.6g} in", annotation_position="top right",
                  row=2, col=1)
    fig.add_trace(go.Scattergl(x=_t, y=data["gwi_cfs"], name="Total G discharge [cfs]", mode="lines",
                               line=dict(color="#1f77b4", width=1.3)), row=3, col=1)
    for _column, _label in (("gwi_release_in", "Slow G release"), ("gwi_spill_in", "G capacity spill")):
        fig.add_trace(go.Scattergl(x=_t, y=prediction[_column] * _depth_to_cfs,
            name=f"{_label} [cfs]", mode="lines"), row=3, col=1)
    fig.update_yaxes(title_text="S [in]", row=1, col=1)
    fig.update_yaxes(title_text="G [in]", row=2, col=1)
    fig.update_yaxes(title_text="Flow [cfs]", row=3, col=1)
    fig.update_layout(template="plotly_white", height=850, width=1500, hovermode="x unified",
                      title=f"{data_title} - Deep store G: storage, slow release and spill",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    return fig


def plot_stormflow(data: pd.DataFrame, events: pd.DataFrame, *, data_title: str, warmup_steps: int, val_mask: np.ndarray) -> go.Figure:
    """Compare selected RDII and the detection reference over frozen assessment events."""
    # --- P5 - RTK RDII vs observed stormflow, event windows shaded ---------------

    _t = pd.to_datetime(data["datetime"])
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.28, 0.28, 0.44],
                        vertical_spacing=0.05,
                        subplot_titles=("Live liquid excess (rain + melt, after S partition)", "Stormflow", "Store S (Antecedent Condition)"))
    fig.add_trace(go.Scattergl(x=_t, y=data["excess_in"], name="excess [in/step]",
                               mode="lines", line=dict(color="#1f77b4", width=0.6),
                               fill="tozeroy", fillcolor="rgba(31,119,180,0.40)"), row=1, col=1)
    fig.update_yaxes(title_text="excess [in]", autorange="reversed", row=1, col=1)
    fig.add_trace(go.Scattergl(x=_t, y=data["stormflow"], name="Observed stormflow",
                               mode="lines", line=dict(color="#444444", width=1)), row=2, col=1)
    fig.add_trace(go.Scattergl(x=_t, y=data["rdii_cfs"], name="RTK RDII (model)",
                               mode="lines", line=dict(color="#d62728", width=1.2)), row=2, col=1)
    for _e in events.itertuples():
        fig.add_vrect(x0=_t.iloc[_e.start_idx], x1=_t.iloc[min(_e.end_idx, len(_t) - 1)],
                      fillcolor="#00ff00", opacity=0.12, line_width=0, row=2, col=1)
    if warmup_steps > 0:
        fig.add_vrect(x0=_t.iloc[0], x1=_t.iloc[warmup_steps - 1], fillcolor="#9E9E9E",
                      opacity=0.14, line_width=0, row=2, col=1)
    add_val_vrects(fig, _t, val_mask, row=2, col=1)
    fig.update_yaxes(title_text="Flow [cfs]", row=2, col=1)
    fig.update_yaxes(title_text="Store S [in]", row=3, col=1)
    fig.add_trace(go.Scattergl(x=_t, y=data["S_in"].to_numpy(), name="IA store S = filled depth [in]",
                               mode="lines", line=dict(color="#8c564b", width=1.3),
                               fill="tozeroy", fillcolor="rgba(140,86,75,0.18)"),
                  row=3, col=1)
    fig.update_layout(template="plotly_white", height=650, width=1500, hovermode="x unified",
                      title=f"{data_title} - JOINT: RTK RDII vs observed stormflow",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    #thin_for_display(fig, DISPLAY_MAX_POINTS).show()
    return fig


def plot_total_flow(data: pd.DataFrame, prediction: pd.DataFrame, events: pd.DataFrame, *, warmup_steps: int, train_mask: np.ndarray, val_mask: np.ndarray) -> go.Figure:
    """Compare selected total flow with observations and annotate split metrics."""
    _ccc, _nse = ConcordanceCorrelationCoefficient(), NashSutcliffe()
    # --- P6 - Whole hydrograph: sanitary + GWI + RDII vs observed total flow -----

    _t = pd.to_datetime(data["datetime"])
    _floor = data["joint_sanitary_cfs"].to_numpy()   # fitted sanitary (SANITARY_MODE)
    _fg = prediction["baseflow_cfs"].to_numpy()                       # sanitary + GWI (jointly fit baseflow)
    _tot = data["total_cfs"].to_numpy()
    ccc_tot = _ccc.compute(data["flow_cfs"].to_numpy(), _tot, mask=train_mask)
    nse_tot = _nse.compute(data["flow_cfs"].to_numpy(), _tot, mask=train_mask)
    _vtxt = (f"  |  val CCC={_ccc.compute(data['flow_cfs'].to_numpy(), _tot, mask=val_mask):.3f}  "
             f"NSE={_nse.compute(data['flow_cfs'].to_numpy(), _tot, mask=val_mask):.3f}" if val_mask.any() else "")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.25, 0.75],
                        vertical_spacing=0.05, subplot_titles=("Rainfall", "Total flow"))
    fig.add_trace(go.Scattergl(x=_t, y=data["rainfall_in"], name="Rainfall",
                               mode="lines", line=dict(color="#1f77b4", width=0.5),
                               fill="tozeroy", fillcolor="rgba(31,119,180,0.45)"),
                  row=1, col=1)
    fig.add_trace(go.Scattergl(x=_t, y=data["flow_cfs"].to_numpy(), name="Observed total flow",
                               mode="lines", line=dict(color="#000000", width=1)), row=2, col=1)
    fig.add_trace(go.Scattergl(x=_t, y=_tot, name="Model total (sanitary + GWI + RDII)",
                               mode="lines", line=dict(color="#fc0000", width=1.3)), row=2, col=1)
    # fig.add_trace(go.Scattergl(x=_t, y=_fg, name="sanitary + GWI (baseflow)",
    #                            mode="lines", line=dict(color="#2ca02c", width=1, dash="dash")), row=2, col=1)
    # fig.add_trace(go.Scattergl(x=_t, y=_floor, name=f"sanitary ({SANITARY_MODE})",
    #                            mode="lines", line=dict(color="#8c564b", width=1, dash="dot")), row=2, col=1)
    if warmup_steps > 0:
        fig.add_vrect(x0=_t.iloc[0], x1=_t.iloc[warmup_steps - 1], fillcolor="#9E9E9E",
                      opacity=0.10, line_width=0, annotation_text="warmup (unscored)",
                      annotation_position="top left", row=2, col=1)
    add_val_vrects(fig, _t, val_mask, row=2, col=1)
    for _e in events.itertuples():
        fig.add_vrect(x0=_t.iloc[_e.start_idx], x1=_t.iloc[min(_e.end_idx, len(_t) - 1)],
                      fillcolor="#55ff00", opacity=0.08, line_width=0, row=2, col=1)
    fig.update_yaxes(title_text="Rain [in/step]", autorange="reversed", rangemode="tozero", row=1, col=1)
    fig.update_yaxes(title_text="Flow [cfs]", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_layout(template="plotly_white", height=750, width=1500, hovermode="x unified",
                      title=f"JOINT whole-flow model vs observed  "
                            f"[train CCC={ccc_tot:.3f}  NSE={nse_tot:.3f}{_vtxt}]",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))

    #thin_for_display(fig, DISPLAY_MAX_POINTS).show()
    return fig


def plot_msdgc_pareto(result: CalibrationResult, metrics: pd.DataFrame) -> tuple[go.Figure, go.Figure, pd.DataFrame]:
    """Plot final training objectives and summarize every candidate's whole-flow metric spread."""
    _msd_whole = metrics.loc[metrics.window.eq('whole')]
    _msd_labels = ['Conveyance CCC', 'Storage CCC', 'Treatment CCC',
                   'Three-zone mean CCC', 'Event-volume CCC', 'Con15 slope']

    if len(result.objective_names) > 1:
        msd_front_fig = plot_pareto_scatter_matrix(
            result, title='Final Pareto front - configured TRAINING objectives')
        msd_front_fig.update_traces(
            customdata=np.arange(len(result.pareto_X)),
            hovertemplate='Pareto %{customdata}<br>x=%{x:.4f}<br>y=%{y:.4f}<extra></extra>')
    else:
        print('Single-objective result: one final optimum is not a multiobjective trade-off front.')
        msd_front_fig = go.Figure(go.Scatter(
            x=result.objective_display_values()[:, 0],
            y=np.arange(len(result.pareto_X)), mode='markers',
            marker=dict(size=12), name='Final candidate',
            hovertemplate='Pareto %{y}<br>Objective=%{x:.5f}<extra></extra>'))
        msd_front_fig.update_layout(
            title='Single-objective final candidate (no multiobjective front)',
            xaxis_title=result.objective_names[0], yaxis_title='Pareto index')
    msd_front_fig.update_layout(template='plotly_white')

    msd_spread_fig = go.Figure()
    _msd_spread_rows = []
    for _metric, _label in zip(MSDGC_METRIC_ORDER, _msd_labels):
        _part = _msd_whole.loc[_msd_whole.metric.eq(_metric)]
        _defined = _part.loc[np.isfinite(_part.value) & _part.status.ne('undefined')]
        _bounds = _part[['lower', 'upper']].drop_duplicates()
        if len(_bounds) != 1:
            raise ValueError(f'Inconsistent whole-flow bounds for {_metric}')
        _lower, _upper = _bounds.iloc[0]
        _quantiles = _defined.value.quantile([0, .25, .5, .75, 1])
        _msd_spread_rows.append(dict(
            metric=_metric, lower=_lower, upper=_upper, candidates=len(_part),
            defined=len(_defined), undefined=len(_part) - len(_defined),
            passes=int(_part.status.eq('pass').sum()),
            minimum=_quantiles.loc[0], q25=_quantiles.loc[.25], median=_quantiles.loc[.5],
            q75=_quantiles.loc[.75], maximum=_quantiles.loc[1]))
        msd_spread_fig.add_trace(go.Box(
            name=_label, y=_defined.value, boxpoints='all', jitter=.25, pointpos=0,
            customdata=_defined.pareto_index, marker=dict(size=5),
            hovertemplate='Pareto %{customdata}<br>%{y:.5f}<extra>%{fullData.name}</extra>'))
        msd_spread_fig.add_shape(type='rect', xref='x', yref='y',
            x0=len(_msd_spread_rows) - 1.45, x1=len(_msd_spread_rows) - .55,
            y0=_lower, y1=_upper, fillcolor='rgba(44,160,44,0.12)', line_width=0, layer='below')
    msd_spread_fig.add_hline(y=1., line_dash='dot', line_color='gray')
    msd_spread_fig.update_layout(
        template='plotly_white', height=550, width=1400, showlegend=False,
        title='Final Pareto candidates - WHOLE total-flow MSDGC spread (green = accepted band)',
        yaxis_title='CCC / Con15 slope',
        xaxis=dict(categoryorder='array', categoryarray=_msd_labels))
    msd_whole_spread = pd.DataFrame(_msd_spread_rows)
    print('Full-event-volume CCC is diagnostic; the three-zone mean is not a substitute for each zone passing.')
    return msd_front_fig, msd_spread_fig, msd_whole_spread


def plot_msdgc_diagnostic(data: pd.DataFrame, predictions: pd.DataFrame, *, warmup_mask: np.ndarray, data_title: str, index: int, status: str) -> go.Figure:
    """Plot a diagnostic candidate without warmup; unscored observations remain gaps."""
    _msd_record = ~warmup_mask
    _msd_time = pd.to_datetime(data.datetime)
    _msd_observed = predictions["observed_cfs"].to_numpy()
    msd_diagnostic_fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[.25, .75], vertical_spacing=.07,
        subplot_titles=('Rainfall (native model timestep)', 'Observed vs modeled TOTAL flow'))
    msd_diagnostic_fig.add_trace(go.Scattergl(
        x=_msd_time[_msd_record], y=data.rainfall_in[_msd_record],
        name='Rainfall', mode='lines', fill='tozeroy', line=dict(color='#1f77b4', width=.6)), row=1, col=1)
    msd_diagnostic_fig.add_trace(go.Scattergl(
        x=_msd_time[_msd_record], y=_msd_observed[_msd_record], name='Observed total flow',
        mode='lines', connectgaps=False, line=dict(color='black', width=1)), row=2, col=1)
    msd_diagnostic_fig.add_trace(go.Scattergl(
        x=_msd_time[_msd_record], y=predictions["modeled_cfs"].to_numpy()[_msd_record], name='Modeled total flow',
        mode='lines', line=dict(color='#d62728', width=1.2)), row=2, col=1)
    #add_val_vrects(msd_diagnostic_fig, _msd_time, row=2, col=1)
    msd_diagnostic_fig.update_yaxes(title_text='Rain [in/step]', autorange='reversed', row=1, col=1)
    msd_diagnostic_fig.update_yaxes(title_text='Flow [cfs]', row=2, col=1)
    msd_diagnostic_fig.update_xaxes(title_text='Date (fixed EST)', row=2, col=1)
    msd_diagnostic_fig.update_layout(
        template='plotly_white', height=750, width=1500, hovermode='x unified',
        title=f'{data_title} - diagnostic Pareto {index}: {status}',
        legend=dict(orientation='h', yanchor='bottom', y=1.05, x=0))
    return msd_diagnostic_fig

