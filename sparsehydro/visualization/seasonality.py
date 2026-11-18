"""Seasonality (sanitary / dry-weather flow) model visualization.

Public API (conditional on plotly being installed):

- :func:`plot_seasonality_components` — bar-chart grid of the calibrated
  hour-of-day, day-of-week, and month-of-year peaking factors of a
  :class:`~sparsehydro.models.SeasonalityModel`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..models import SeasonalityModel

    try:
        import plotly.graph_objects as go
    except ImportError:  # pragma: no cover
        pass

_DOW_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_MONTH_LABELS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]
_DIM_COLORS = {"hour": "#1f77b4", "dow": "#2ca02c", "month": "#ff7f0e"}


try:
    import plotly.graph_objects as go  # type: ignore[import]
    from plotly.subplots import make_subplots  # type: ignore[import]

    def plot_seasonality_components(
        model: "SeasonalityModel",
        title: str = "Seasonality Model Components",
        flow_units: str = "",
    ) -> go.Figure:
        """Bar-chart breakdown of a :class:`~sparsehydro.models.SeasonalityModel`.

        One subplot per active dimension (hour-of-day, day-of-week,
        month-of-year) showing the **mean-normalised** peaking factors exactly
        as the model applies them in ``predict()``.  Each subplot title carries
        the dimension's mixing weight, and the figure title reports the
        baseline flow.

        :param model: An initialized :class:`~sparsehydro.models.SeasonalityModel`
            whose parameters hold the values to display (e.g. after
            calibration).
        :type model: SeasonalityModel
        :param title: Figure title prefix; baseline and weights are appended.
        :type title: str
        :param flow_units: Units label appended to the baseline annotation
            (e.g. ``"cfs"``).
        :type flow_units: str
        :returns: Plotly Figure with one row per active dimension.
        :rtype: plotly.graph_objects.Figure
        """
        dims: list[tuple[str, str, list]] = []
        if model.include_hour:
            dims.append(("hour", "pf_hour", [f"{h:02d}:00" for h in range(24)]))
        if model.include_dow:
            dims.append(("dow", "pf_dow", _DOW_LABELS))
        if model.include_month:
            dims.append(("month", "pf_month", _MONTH_LABELS))

        raw_weights = {
            d: model.get_scalar_parameter(f"w_{d}").value for d, _, _ in dims
        }
        w_sum = sum(raw_weights.values())
        weights = (
            {d: v / w_sum for d, v in raw_weights.items()} if w_sum > 0 else raw_weights
        )
        baseline = model.get_scalar_parameter("baseline").value

        dim_titles = {
            "hour": "Hour of day",
            "dow": "Day of week",
            "month": "Month of year (seasonality)",
        }
        subplot_titles = [
            f"{dim_titles[d]} — weight w = {weights[d]:.3f}" for d, _, _ in dims
        ]

        fig = make_subplots(
            rows=len(dims),
            cols=1,
            vertical_spacing=0.14,
            subplot_titles=subplot_titles,
        )

        for row, (d, pf_name, labels) in enumerate(dims, start=1):
            pf = np.asarray(model.get_vector_parameter(pf_name).values, dtype=float)
            pf_norm = pf / pf.mean() if pf.mean() > 0 else pf
            fig.add_trace(
                go.Bar(
                    x=labels,
                    y=pf_norm.tolist(),
                    name=dim_titles[d],
                    marker_color=_DIM_COLORS[d],
                    showlegend=False,
                ),
                row=row, col=1,
            )
            fig.add_hline(
                y=1.0,
                line=dict(color="grey", width=1, dash="dash"),
                row=row, col=1,
            )
            fig.update_yaxes(title_text="Peaking factor [-]", row=row, col=1)

        units = f" {flow_units}" if flow_units else ""
        fig.update_layout(
            title=f"{title} — baseline = {baseline:.3f}{units}",
            height=max(300, 260 * len(dims)),
            bargap=0.15,
        )
        return fig

except ImportError:

    def plot_seasonality_components(*args, **kwargs):  # type: ignore[misc]
        raise ImportError(
            "plotly is required for plot_seasonality_components. "
            "Install with: pip install plotly"
        )
