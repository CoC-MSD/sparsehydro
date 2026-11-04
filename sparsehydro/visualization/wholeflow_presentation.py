"""Presentation figures from an already selected whole-flow prediction."""

from __future__ import annotations

from html import escape
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
