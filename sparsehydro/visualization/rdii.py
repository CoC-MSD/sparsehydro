"""RDII / RTK-specific interactive plots.

Public API (all conditional on plotly being installed):

- :func:`plot_rtk_shape`       — unit hydrograph shape per RTKTriangle
- :func:`plot_rdii_components` — stacked per-triangle RDII contributions over time
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    try:
        import plotly.graph_objects as go
    except ImportError:  # pragma: no cover
        pass


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a ``#rrggbb`` hex colour to an ``rgba(...)`` string.

    :param hex_color: Hex colour string (e.g. ``"#1f77b4"``).
    :type hex_color: str
    :param alpha: Alpha (opacity) value in ``[0, 1]``.
    :type alpha: float
    :returns: CSS ``rgba(...)`` colour string.
    :rtype: str
    """
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


_TRACE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


try:
    import plotly.graph_objects as go  # type: ignore[import]
    from plotly.subplots import make_subplots  # type: ignore[import]

    # ------------------------------------------------------------------
    # plot_rtk_shape
    # ------------------------------------------------------------------

    def plot_rtk_shape(
        triangles,
        dt_hours: float = 1.0,
        title: str = "RTK Unit Hydrograph Shapes",
    ) -> go.Figure:
        """Plot the triangular unit hydrograph shape for each RTKTriangle.

        Useful for sanity-checking T (time-to-peak) and K (recession ratio)
        before calibration.

        :param triangles: An :class:`~sparsehydro.models.rdii.RTKTriangle` instance or
            a list/tuple of instances.
        :type triangles: RTKTriangle or list[RTKTriangle] or tuple[RTKTriangle, ...]
        :param dt_hours: Time step in hours.
        :type dt_hours: float
        :param title: Figure title.
        :type title: str
        :returns: Plotly Figure — one filled trace per triangle.
        :rtype: plotly.graph_objects.Figure
        """
        from ..models.rdii.rtk_triangle import triangular_uh

        if not isinstance(triangles, (list, tuple)):
            triangles = [triangles]

        fig = go.Figure()

        for idx, tri in enumerate(triangles):
            ordinates = triangular_uh(tri, dt_hours)
            t = np.arange(len(ordinates)) * dt_hours
            color = _TRACE_COLORS[idx % len(_TRACE_COLORS)]
            label = f"Triangle {idx + 1}  (T={tri.T:.2f}h, K={tri.K:.2f}, R={tri.R:.3f})"

            fig.add_trace(
                go.Scatter(
                    x=t.tolist(),
                    y=ordinates.tolist(),
                    mode="lines",
                    fill="tozeroy",
                    fillcolor=_hex_to_rgba(color, 0.15),
                    line=dict(color=color, width=2),
                    name=label,
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title="Time [hours]",
            yaxis_title="UH Ordinate [1/hr]",
            height=450,
            hovermode="x unified",
            legend=dict(x=0.65, y=0.99),
        )
        return fig

    # ------------------------------------------------------------------
    # plot_rdii_components
    # ------------------------------------------------------------------

    def plot_rdii_components(
        result_df,
        title: str = "RDII Components",
        rainfall_label: str = "Rainfall [mm]",
        component_label: str = "RDII [mm/step]",
    ) -> go.Figure:
        """Stacked RDII component traces alongside rainfall.

        The DataFrame must contain a ``datetime`` column (or similar time index)
        and any number of columns named ``rdii_component_N`` (N = 0, 1, 2, …).
        A ``rainfall_mm`` (or ``rainfall_in``) column is optional; if absent,
        no rainfall panel is shown.

        :param result_df: DataFrame produced by an RDIIModel run (e.g.
            :meth:`~sparsehydro.models.rdii.RDIIModel.predict_components`).
            Must contain ``rdii_component_N`` columns.
        :type result_df: pandas.DataFrame
        :param title: Figure title.
        :type title: str
        :param rainfall_label: Y-axis label for the rainfall panel.
        :type rainfall_label: str
        :param component_label: Y-axis label for the stacked component panel.
        :type component_label: str
        :returns: Plotly Figure with 2 rows (rainfall + stacked components) when
            rainfall is available, or 1 row otherwise.
        :rtype: plotly.graph_objects.Figure
        """
        import pandas as pd

        df = result_df if hasattr(result_df, "columns") else pd.DataFrame(result_df)

        # Find component columns
        comp_cols = sorted(
            [c for c in df.columns if c.startswith("rdii_component_")],
            key=lambda c: int(c.split("_")[-1]),
        )

        # Determine time column
        time_col = "datetime"
        if time_col not in df.columns:
            for c in df.columns:
                if "time" in c.lower() or "date" in c.lower():
                    time_col = c
                    break
            else:
                time_col = df.index

        rainfall_col = next(
            (c for c in ("rainfall_mm", "rainfall_in") if c in df.columns), None
        )
        has_rainfall = rainfall_col is not None
        n_rows = 2 if has_rainfall else 1
        row_heights = [0.25, 0.75] if has_rainfall else [1.0]

        fig = make_subplots(
            rows=n_rows,
            cols=1,
            shared_xaxes=True,
            row_heights=row_heights,
            vertical_spacing=0.04,
            subplot_titles=(["Rainfall", "RDII Components"] if has_rainfall else ["RDII Components"]),
        )

        dt = df[time_col] if time_col in df.columns else df.index

        if has_rainfall:
            fig.add_trace(
                go.Bar(
                    x=dt,
                    y=df[rainfall_col].tolist(),
                    name=rainfall_label,
                    marker_color="steelblue",
                    opacity=0.7,
                ),
                row=1, col=1,
            )
            fig.update_yaxes(autorange="reversed", title_text=rainfall_label, row=1, col=1)

        comp_row = 2 if has_rainfall else 1
        for idx, col in enumerate(comp_cols):
            color = _TRACE_COLORS[idx % len(_TRACE_COLORS)]
            fig.add_trace(
                go.Scatter(
                    x=dt,
                    y=df[col].tolist(),
                    mode="lines",
                    name=col.replace("rdii_component_", "Component "),
                    stackgroup="rdii",
                    line=dict(width=0.5, color=color),
                    fillcolor=_hex_to_rgba(color, 0.5),
                ),
                row=comp_row, col=1,
            )

        fig.update_yaxes(title_text=component_label, row=comp_row, col=1)
        fig.update_xaxes(title_text="Date / Time", row=comp_row, col=1)
        fig.update_layout(
            title=title,
            height=550,
            hovermode="x unified",
        )
        return fig

except ImportError:

    def plot_rtk_shape(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("plotly is required for plot_rtk_shape. Install with: pip install plotly")

    def plot_rdii_components(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("plotly is required for plot_rdii_components. Install with: pip install plotly")
