"""Visualize the stratified event-based train/validation split.

The overview plot in the notebook (rainfall / flow / temperature with a single
orange "validation" band after ``TRAIN_END``) reflects the *old* chronological
cutoff.  The event-based split instead scatters validation storms **through** the
record, each guarded by an unscored warmup lead-in and a recession buffer.  These
helpers draw that partition on the real timeline so the interwoven storms — and
the fact that rainfall and temperature ride the *same* masks as flow — are easy
to see:

- :func:`plot_split_timeline` — rainfall / flow / temperature panels with every
  detected storm shaded by its split assignment (train vs validation) plus the
  warmup and buffer guard bands; validation storms are numbered.
- :func:`plot_per_event_scores` — one bar group per held-out validation storm
  showing its KGE and CCC, with the mean and median drawn in, so you can see
  exactly which storms the summary statistics pool over.

Both are conditional on plotly being installed (mirroring the rest of
:mod:`sparsehydro.visualization`); import-time stubs raise a clear error if not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["plot_split_timeline", "plot_per_event_scores"]


# Region fills — event bands by split, plus the two guard bands.
_TRAIN_FILL = "rgba(33,102,172,0.13)"    # blue   — training storm
_VAL_FILL = "rgba(214,96,77,0.22)"       # red    — validation storm (scored)
_WARMUP_FILL = "rgba(230,159,0,0.28)"    # amber  — unscored warmup lead-in
_BUFFER_FILL = "rgba(120,120,120,0.20)"  # gray   — recession buffer
_LEGEND_SWATCH = [
    ("Training storm", "#3E6DA6"),
    ("Validation storm", "#D6604D"),
    ("Warmup (unscored)", "#E69F00"),
    ("Recession buffer", "#787878"),
]

_KGE_COLOR = "#2166AC"
_CCC_COLOR = "#D6604D"


def _as_dt(datetime) -> np.ndarray:
    """Coerce a datetime-like series to a numpy datetime64 array."""
    return np.asarray(pd.to_datetime(pd.Series(datetime)).to_numpy())


def _event_spans(split):
    """List ``(event_id, split_label, first_idx, last_idx)`` in time order."""
    ids = np.asarray(split.event_ids)
    spans = []
    for eid, row in split.events_table.iterrows():
        pos = np.flatnonzero(ids == int(eid))
        if pos.size:
            spans.append((int(eid), str(row["split"]), int(pos[0]), int(pos[-1])))
    return spans


def _mask_runs(mask):
    """Return ``(start_idx, end_idx)`` inclusive spans of contiguous True runs."""
    m = np.asarray(mask, dtype=bool).astype(np.int8)
    edges = np.diff(np.concatenate([[0], m, [0]]))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


try:
    import plotly.graph_objects as go  # type: ignore[import]
    from plotly.subplots import make_subplots  # type: ignore[import]

    def plot_split_timeline(
        datetime,
        flow,
        split,
        *,
        rainfall=None,
        temperature=None,
        flow_label: str = "Flow [cfs]",
        title: str = "Event-based split",
        show_guard_bands: bool = True,
    ) -> "go.Figure":
        """Draw the stratified train/validation split on the real timeline.

        Each detected storm is shaded by its split assignment (blue = training,
        red = validation); when *show_guard_bands* is set, the unscored warmup
        lead-in (amber) and recession buffer (gray) around each validation storm
        are shaded too.  Validation storms are numbered in start-time order,
        matching :func:`plot_per_event_scores`.

        :param datetime: Timestamps aligned to the full model timeline.
        :param flow: Observed flow series (same length as *datetime*).
        :param split: An :class:`~sparsehydro.eventsplit.EventSplit`.
        :param rainfall: Optional rainfall series; drawn as an inverted
            hyetograph on a top panel, shaded with the same split bands.
        :param temperature: Optional air-temperature series; drawn on a bottom
            panel, shaded with the same split bands.
        :param flow_label: Y-axis label for the flow panel.
        :type flow_label: str
        :param title: Figure title.
        :type title: str
        :param show_guard_bands: Shade the warmup and recession-buffer regions.
        :type show_guard_bands: bool
        :returns: Plotly figure (1–3 stacked panels sharing the time axis).
        :rtype: plotly.graph_objects.Figure
        """
        dt = _as_dt(datetime)
        flow = np.asarray(flow, dtype=float)
        n = dt.shape[0]

        # Assemble the present panels (rainfall on top, temperature on the bottom).
        panels, weights = [], []
        if rainfall is not None:
            panels.append("rain"); weights.append(0.22)
        panels.append("flow"); weights.append(0.52)
        if temperature is not None:
            panels.append("temp"); weights.append(0.26)
        total = sum(weights)
        row_heights = [w / total for w in weights]
        titles = {"rain": "Rainfall", "flow": "Flow", "temp": "Air temperature"}
        row_of = {name: i + 1 for i, name in enumerate(panels)}
        nrows = len(panels)

        fig = make_subplots(
            rows=nrows, cols=1, shared_xaxes=True, vertical_spacing=0.045,
            row_heights=row_heights,
            subplot_titles=tuple(titles[p] for p in panels),
        )

        if "rain" in row_of:
            r = row_of["rain"]
            fig.add_trace(
                go.Scattergl(x=dt, y=np.asarray(rainfall, dtype=float),
                             name="Rainfall", mode="lines", fill="tozeroy",
                             line=dict(color="#1f77b4", width=0.6),
                             fillcolor="rgba(31,119,180,0.45)", showlegend=False),
                row=r, col=1,
            )
            fig.update_yaxes(title_text="Rain [in/hr]", autorange="reversed", row=r, col=1)

        rf = row_of["flow"]
        fig.add_trace(
            go.Scattergl(x=dt, y=flow, name="Observed flow", mode="lines",
                         line=dict(color="#d62728", width=1), showlegend=False),
            row=rf, col=1,
        )
        fig.update_yaxes(title_text=flow_label, row=rf, col=1)

        if "temp" in row_of:
            r = row_of["temp"]
            fig.add_trace(
                go.Scattergl(x=dt, y=np.asarray(temperature, dtype=float),
                             name="Air temperature", mode="lines",
                             line=dict(color="#9467bd", width=1), showlegend=False),
                row=r, col=1,
            )
            fig.update_yaxes(title_text="T [°C]", row=r, col=1)

        def _band(a, b, fill, rows, annotation=None, arow=None):
            # x1 nudged to the next step so single/short spans stay visible.
            x0 = dt[a]
            x1 = dt[min(b + 1, n - 1)]
            for rr in rows:
                kw = dict(x0=x0, x1=x1, row=rr, col=1, fillcolor=fill,
                          line_width=0, layer="below")
                if annotation is not None and rr == arow:
                    kw.update(annotation_text=annotation, annotation_position="top left",
                              annotation=dict(font=dict(size=10, color="#D6604D")))
                fig.add_vrect(**kw)

        all_rows = list(range(1, nrows + 1))

        # Guard bands first (below the storm bands).
        if show_guard_bands:
            for a, b in _mask_runs(split.warmup_mask):
                _band(a, b, _WARMUP_FILL, all_rows)
            for a, b in _mask_runs(split.buffer_mask):
                _band(a, b, _BUFFER_FILL, all_rows)

        # Storm bands; number the validation storms in start-time order.
        val_num = 0
        for _eid, label, a, b in _event_spans(split):
            if label == "val":
                val_num += 1
                _band(a, b, _VAL_FILL, all_rows, annotation=f"V{val_num}", arow=rf)
            else:
                _band(a, b, _TRAIN_FILL, all_rows)

        # Legend swatches (dummy traces — one per shaded category).
        for name, color in _LEGEND_SWATCH:
            if not show_guard_bands and name in ("Warmup (unscored)", "Recession buffer"):
                continue
            fig.add_trace(
                go.Scatter(x=[dt[0]], y=[None], mode="markers", name=name,
                           marker=dict(size=12, color=color, symbol="square"),
                           hoverinfo="skip"),
                row=rf, col=1,
            )

        c = split.counts()
        subtitle = (f"{c['n_train_events']} train / {c['n_val_events']} val storms · "
                    f"warmup {split.meta['warmup_steps']} steps · "
                    f"buffer {c['buffer_steps']} steps")
        fig.update_layout(
            title=f"{title}<br><sup>{subtitle}</sup>",
            template="plotly_white", hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            margin=dict(t=90, r=20, b=40, l=60),
        )
        return fig

    def plot_per_event_scores(
        scores,
        *,
        metrics=("kge", "ccc"),
        title: str = "Per-event validation scores",
    ) -> "go.Figure":
        """Bar chart of per-event KGE/CCC over a method's validation storms.

        One x-axis entry per held-out validation storm (numbered ``V1, V2, …`` in
        start-time order, matching :func:`plot_split_timeline`, and labeled with
        the storm start date).  The mean (dotted) and median (dashed) of each
        metric are drawn in, so it is obvious which storms drag the mean below the
        median.

        :param scores: A :func:`~sparsehydro.eventsplit.per_event_scores` frame
            (index ``event_id``; columns ``kge``, ``ccc``, optionally ``start`` /
            ``n_steps``).
        :param metrics: Metric columns to plot (default KGE + CCC).
        :param title: Figure title.
        :type title: str
        :returns: Plotly bar figure.
        :rtype: plotly.graph_objects.Figure
        """
        df = scores.copy()
        if "start" in df.columns:
            df = df.sort_values("start")

        n = len(df)
        labels, hover = [], []
        for i in range(n):
            start = df["start"].iloc[i] if "start" in df.columns else None
            date = pd.Timestamp(start).strftime("%Y-%m-%d") if start is not None and pd.notna(start) else ""
            labels.append(f"V{i + 1}<br>{date}")
            nsteps = int(df["n_steps"].iloc[i]) if "n_steps" in df.columns else None
            hover.append(f"event_id {df.index[i]}"
                         + (f" · {nsteps} steps" if nsteps is not None else ""))

        colors = {"kge": _KGE_COLOR, "ccc": _CCC_COLOR}
        fig = go.Figure()
        for metric in metrics:
            if metric not in df.columns:
                continue
            fig.add_trace(go.Bar(
                x=labels, y=df[metric].to_numpy(dtype=float), name=metric.upper(),
                marker_color=colors.get(metric, "#4D4D4D"), customdata=hover,
                hovertemplate="%{x}<br>%{customdata}<br>" + metric.upper() + "=%{y:.3f}<extra></extra>",
            ))

        # Mean (dotted) + median (dashed) reference lines per metric.
        for metric in metrics:
            if metric not in df.columns:
                continue
            vals = df[metric].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if not vals.size:
                continue
            color = colors.get(metric, "#4D4D4D")
            fig.add_hline(y=float(np.median(vals)), line=dict(color=color, width=1.4, dash="dash"),
                          annotation_text=f"{metric.upper()} median {np.median(vals):+.2f}",
                          annotation_position="top right", annotation_font_color=color)
            fig.add_hline(y=float(vals.mean()), line=dict(color=color, width=1.0, dash="dot"),
                          annotation_text=f"{metric.upper()} mean {vals.mean():+.2f}",
                          annotation_position="bottom right", annotation_font_color=color)

        fig.add_hline(y=0.0, line=dict(color="#000000", width=1))
        fig.update_layout(
            title=title, template="plotly_white", barmode="group",
            yaxis_title="Score", xaxis_title="Validation storm",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            margin=dict(t=70, r=120, b=60, l=60),
        )
        return fig

except ImportError:  # pragma: no cover - matches package-wide plotly guard

    def plot_split_timeline(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("plotly is required for plot_split_timeline. Install with: pip install plotly")

    def plot_per_event_scores(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("plotly is required for plot_per_event_scores. Install with: pip install plotly")
