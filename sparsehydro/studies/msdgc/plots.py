"""MSDGC study figures: Pareto metrics, candidate diagnostics, the presentation figure."""

from __future__ import annotations

from html import escape
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sparsehydro.calibration.result import CalibrationResult
from sparsehydro.studies.msdgc.pareto import MSDGC_METRIC_ORDER
from sparsehydro.visualization.calibration import plot_pareto_scatter_matrix


def _shade(
    fig: go.Figure,
    times: pd.DatetimeIndex,
    step: pd.Timedelta,
    mask: np.ndarray,
    name: str,
    color: str,
    opacity: float,
) -> None:
    edges = np.diff(np.r_[0, mask.astype(int), 0])
    shapes = list(fig.layout.shapes)
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        # A mask covers whole timesteps, including single-point and final runs.
        shapes.append(dict(
            type="rect", xref="x2", yref="paper",
            x0=times[start].isoformat(), x1=(times[stop - 1] + step).isoformat(), y0=0, y1=1,
            fillcolor=color, opacity=opacity, line_width=0, layer="below",
            name=name,
        ))
    fig.update_layout(shapes=shapes)
    if mask.any():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name=name,
            marker=dict(symbol="square", size=14, color=color),
            hoverinfo="skip",
        ), row=2, col=1)


def _event_mask(events: pd.DataFrame, size: int) -> np.ndarray:
    mask = np.zeros(size, dtype=bool)
    for event in events.loc[events["eligible"]].itertuples():
        start, end = int(event.start_idx), int(event.end_idx)
        if start != event.start_idx or end != event.end_idx or not 0 <= start <= end < size:
            raise ValueError("Eligible event indices must be inclusive bounds inside the model record")
        mask[start:end + 1] = True
    return mask


def plot_wholeflow_presentation(
    data: pd.DataFrame,
    prediction: pd.DataFrame,
    *,
    warmup_mask: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    event_source: Literal["auto", "file", "hybrid"],
    events: pd.DataFrame,
    calibration_events: pd.DataFrame,
    event_mask: np.ndarray,
    data_title: str,
    candidate_label: str,
    selection_status: str,
    spinup_mode: str,
) -> dict[str, go.Figure]:
    """Build four full-resolution figures without fitting or mutating inputs.

    ``prediction`` must contain the selected candidate's datetime, total_cfs,
    baseflow_cfs and rdii_cfs, aligned with ``data``. Masks are the final scoring
    masks, including data-quality exclusions. ``events`` is the source inventory,
    not the later assessment-filtered inventory; hybrid mode additionally uses
    ``calibration_events``. ``event_mask`` is actual padded/melt scoring coverage.
    """
    if event_source not in ("auto", "file", "hybrid"):
        raise ValueError("event_source must be 'auto', 'file' or 'hybrid'")
    times = pd.DatetimeIndex(data["datetime"])
    size = len(times)
    if size < 2 or times.hasnans or not times.is_monotonic_increasing or times.has_duplicates:
        raise ValueError("Presentation figures require at least two unique, ordered timestamps")
    step = times[1] - times[0]
    if not ((times[1:] - times[:-1]) == step).all():
        raise ValueError("Presentation figures require a regular modeling grid")
    if not times.equals(pd.DatetimeIndex(prediction["datetime"])):
        raise ValueError("Selected prediction timestamps must match the modeling data")
    for name, mask in (
        ("warmup", warmup_mask), ("training", train_mask),
        ("validation", val_mask), ("event", event_mask),
    ):
        if not isinstance(mask, np.ndarray) or mask.dtype != bool or mask.shape != (size,):
            raise ValueError(f"{name} mask must be a boolean array of length {size}")
    if ((warmup_mask & train_mask) | (warmup_mask & val_mask) | (train_mask & val_mask)).any():
        raise ValueError("Warmup, training and validation masks must not overlap")
    scored = train_mask | val_mask
    if (event_mask & ~scored).any():
        raise ValueError("Storm scoring coverage must be inside training/validation masks")
    record = ~warmup_mask
    if not record.any():
        raise ValueError("No post-warmup record is available for presentation")
    total = prediction["total_cfs"].to_numpy(dtype=float)
    base = prediction["baseflow_cfs"].to_numpy(dtype=float)
    storm = prediction["rdii_cfs"].to_numpy(dtype=float)
    if not np.allclose(total, base + storm, equal_nan=True):
        raise ValueError("Selected total flow must equal selected baseflow + RDII")
    if not np.isfinite(np.column_stack((total, base, storm))[scored]).all():
        raise ValueError("Selected prediction has nonfinite flow on scored rows")
    observed = data["flow_cfs"].to_numpy(dtype=float)
    rain = data["rainfall_in"].to_numpy(dtype=float)
    if not np.isfinite(observed[scored]).all() or not np.isfinite(rain).all() or (rain < 0).any():
        raise ValueError("Scored observations and rainfall must be finite; rain must be nonnegative")
    observed = np.where(scored, observed, np.nan)
    source_label = {
        "auto": "Automatic detection", "file": "Imported approved events",
        "hybrid": "Hybrid: imported calibration + approved events",
    }[event_source]
    candidate = escape(candidate_label)
    site = escape(data_title)
    minutes = step.total_seconds() / 60

    def figure(title: str, subtitle: str, keep: np.ndarray, footer: str,
               separate_base: bool = False) -> go.Figure:
        rows = 3 if separate_base else 2
        fig = make_subplots(
            rows=rows, cols=1, shared_xaxes=True,
            row_heights=[0.18, 0.53, 0.29] if separate_base else [0.22, 0.78],
            vertical_spacing=0.07,
        )
        fig.add_trace(go.Scattergl(
            x=times[keep], y=rain[keep], name="Rainfall", mode="lines",
            line=dict(color="#0072B2", width=1), fill="tozeroy",
            fillcolor="rgba(0,114,178,0.35)", connectgaps=False,
        ), row=1, col=1)
        fig.update_layout(
            template="plotly_white", width=1600, height=900,
            font=dict(family="Arial", size=19, color="#253247"),
            title=dict(text=f"{title}<br><sup>{subtitle}</sup>",
                       x=0.06, y=0.97, yanchor="top", font=dict(size=28)),
            margin=dict(l=100, r=45, t=155, b=150),
            legend=dict(orientation="h", x=0, y=1.03, yanchor="bottom",
                        font=dict(size=16)),
            hovermode="x unified",
        )
        fig.update_yaxes(
            title_text=f"Rain [in/{minutes:g} min]", autorange="reversed",
            rangemode="tozero", row=1, col=1,
        )
        fig.update_yaxes(title_text="Flow [cfs]", rangemode="tozero", row=2, col=1)
        if separate_base:
            fig.update_yaxes(title_text="Baseflow [cfs]", rangemode="tozero", row=3, col=1)
        fig.update_xaxes(
            range=[times[keep][0].isoformat(), (times[keep][-1] + step).isoformat()],
            showgrid=False, ticks="outside",
        )
        fig.update_xaxes(title_text="Date (model time)", row=rows, col=1)
        fig.add_annotation(
            text=footer, x=0, y=-0.15, xref="paper", yref="paper",
            xanchor="left", yanchor="top", showarrow=False,
            font=dict(size=14, color="#596579"), align="left",
        )
        return fig

    def flow(fig: go.Figure, values: np.ndarray, keep: np.ndarray,
             name: str, color: str, dash: str = "solid", row: int = 2) -> None:
        fig.add_trace(go.Scattergl(
            x=times[keep], y=values[keep], name=name, mode="lines",
            line=dict(color=color, width=2, dash=dash), connectgaps=False,
        ), row=row, col=1)

    diagnostic_note = (
        "Post-hoc, validation-informed MSDGC comparison; not calibration selection or official compliance."
        "<br>Warmup omitted; unscored observations are gaps."
    )
    best = figure(
        "Best-available solution | observed vs modeled flow",
        f"{site} | {candidate} | {escape(selection_status.replace('_', ' '))}",
        record, diagnostic_note,
    )
    flow(best, observed, record, "Observed total flow", "#252525")
    flow(best, total, record, "Modeled total flow", "#D55E00")

    components = figure(
        "Modeled flow components",
        f"{site} | {candidate} | same candidate as the best-available hydrograph",
        record,
        "Total = baseflow + stormflow (RDII). Baseflow includes sanitary/background flow and GWI when enabled."
        "<br>Baseflow has its own vertical scale. Post-hoc diagnostic candidate; warmup omitted.",
        separate_base=True,
    )
    flow(components, total, record, "Modeled total flow", "#D55E00")
    flow(components, base, record, "Modeled baseflow", "#009E73", row=3)
    flow(components, storm, record, "Modeled stormflow (RDII)", "#0072B2")

    zones = figure(
        "Warmup, training and validation zones",
        f"{site} | actual scoring masks | warmup: {escape(spinup_mode)}",
        np.ones(size, dtype=bool),
        "Warmup is unscored; synthetic spin-up has forcing only, not observed flow."
        "<br>Other unscored rows are data-quality exclusions; training/validation retain their actual block boundaries.",
    )
    flow(zones, observed, np.ones(size, dtype=bool), "Observed total flow", "#252525")
    for mask, label, color in (
        (warmup_mask, "Warmup", "#8B95A5"),
        (train_mask, "Training", "#009E73"),
        (val_mask, "Validation", "#E69F00"),
        (~(warmup_mask | scored), "Excluded / unscored", "#CC79A7"),
    ):
        days = mask.sum() * step.total_seconds() / 86400
        _shade(zones, times, step, mask, f"{label} ({days:.1f} d)", color, 0.18)

    inventory = calibration_events if event_source == "hybrid" else events
    detected = _event_mask(inventory, size)
    storms = figure(
        "Storm windows in the observed record",
        f"{site} | {source_label} | {int(inventory.eligible.sum())} eligible of {len(inventory)} inventory windows",
        record,
        "Shading follows the selected event source, not the later MSDGC assessment exclusions."
        "<br>Amber = actual scoring coverage (padding/melt where enabled); colored windows = eligible source events.",
    )
    flow(storms, observed, record, "Observed total flow", "#252525")
    _shade(storms, times, step, event_mask, "Storm scoring coverage", "#E69F00", 0.12)
    _shade(storms, times, step, detected & record,
           "Calibration EVT windows" if event_source == "hybrid" else source_label,
           "#009E73", 0.25)
    if event_source == "hybrid":
        _shade(storms, times, step, _event_mask(events, size) & record,
               "Approved event windows", "#CC79A7", 0.25)
    return {
        "01_best_solution_hydrograph": best,
        "02_modeled_flow_components": components,
        "03_warmup_training_validation": zones,
        "04_selected_storm_windows": storms,
    }


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
