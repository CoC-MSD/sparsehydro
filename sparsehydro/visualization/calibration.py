"""Calibration result and optimizer interactive plots.

Public API (all conditional on plotly being installed):

- :func:`plot_pareto_evolution`       — animated Pareto front with generation slider
- :func:`plot_parallel_coordinates`   — parallel-axis trade-off explorer
- :func:`plot_objective_convergence`  — best-value + percentile bands per generation
- :func:`plot_parameter_distributions`— violin plots of Pareto parameter values
- :func:`plot_sensitivity_heatmap`    — parameter-objective Pearson correlation heat map
- :func:`plot_pareto_scatter_matrix`  — scatter-plot matrix (SPLOM) of all objectives
- :func:`plot_splom`                  — SPLOM of all final-generation solutions, non-dominated highlighted
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Union

import numpy as np
import pandas as pd

from ._utils import _display_col, _resolve_obj

if TYPE_CHECKING:
    from ..calibration.result import CalibrationResult

    try:
        import plotly.graph_objects as go
    except ImportError:  # pragma: no cover
        pass


try:
    import plotly.graph_objects as go  # type: ignore[import]
    from plotly.subplots import make_subplots  # type: ignore[import]

    # ------------------------------------------------------------------
    # plot_pareto_evolution
    # ------------------------------------------------------------------

    def plot_pareto_evolution(
        result: "CalibrationResult",
        x_obj: Union[int, str] = 0,
        y_obj: Union[int, str] = 1,
        title: str = "Pareto Front Evolution",
        show_knee: bool = False,
        show_bounds: bool = False,
    ) -> go.Figure:
        """Animated scatter showing all solutions and the Pareto front per generation.

        A slider at the bottom navigates between generations.  A Play/Pause
        button animates the evolution automatically.

        :param result: Calibration result with per-generation history.
        :type result: CalibrationResult
        :param x_obj: X-axis objective — zero-based index or objective name.
        :type x_obj: int or str
        :param y_obj: Y-axis objective — zero-based index or objective name.
        :type y_obj: int or str
        :param title: Figure title.
        :type title: str
        :param show_knee: Overlay a star marker on the knee (compromise)
            solution of the **final** Pareto front (see
            :meth:`~sparsehydro.calibration.result.CalibrationResult.knee_index`).
        :type show_knee: bool
        :param show_bounds: Overlay a dashed rectangle spanning the bounds of
            the **final** Pareto-optimal objective values.
        :type show_bounds: bool
        :returns: Plotly Figure with animation frames and slider.
        :rtype: plotly.graph_objects.Figure
        """
        from ..calibration.result import identify_pareto

        obj_names = result.objective_names
        min_flags = result.minimize_flags

        xi = _resolve_obj(x_obj, obj_names)
        yi = _resolve_obj(y_obj, obj_names)
        x_label = obj_names[xi]
        y_label = obj_names[yi]

        frames = []
        slider_steps = []

        for rec in result.history:
            all_x = _display_col(rec.F, xi, min_flags)
            all_y = _display_col(rec.F, yi, min_flags)
            is_pareto = identify_pareto(rec.F)
            par_x = all_x[is_pareto]
            par_y = all_y[is_pareto]

            frame_data = [
                go.Scatter(
                    x=all_x, y=all_y,
                    mode="markers",
                    marker=dict(color="lightgrey", size=5),
                    name="All solutions",
                    showlegend=False,
                ),
                go.Scatter(
                    x=par_x, y=par_y,
                    mode="markers",
                    marker=dict(
                        color=par_y,
                        colorscale="Viridis",
                        size=9,
                        showscale=True,
                        colorbar=dict(title=y_label, x=1.02),
                        line=dict(color="black", width=0.5),
                    ),
                    name="Pareto front",
                    showlegend=False,
                ),
            ]
            # Restrict frames to the two animated traces so any static
            # knee/bounds overlay traces appended below are left untouched.
            frames.append(
                go.Frame(data=frame_data, name=str(rec.generation), traces=[0, 1])
            )
            slider_steps.append(
                dict(
                    args=[
                        [str(rec.generation)],
                        {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}},
                    ],
                    label=str(rec.generation),
                    method="animate",
                )
            )

        first = result.history[0]
        is_pareto0 = identify_pareto(first.F)
        init_x = _display_col(first.F, xi, min_flags)
        init_y = _display_col(first.F, yi, min_flags)
        init_data = [
            go.Scatter(
                x=init_x, y=init_y,
                mode="markers",
                marker=dict(color="lightgrey", size=5),
                name="All solutions",
            ),
            go.Scatter(
                x=init_x[is_pareto0], y=init_y[is_pareto0],
                mode="markers",
                marker=dict(
                    color=init_y[is_pareto0],
                    colorscale="Viridis",
                    size=9,
                    showscale=True,
                    colorbar=dict(title=y_label, x=1.02),
                    line=dict(color="black", width=0.5),
                ),
                name="Pareto front",
            ),
        ]

        if (show_bounds or show_knee) and len(result.pareto_F) > 0:
            final_x = _display_col(result.pareto_F, xi, min_flags)
            final_y = _display_col(result.pareto_F, yi, min_flags)

            if show_bounds:
                x0, x1 = float(final_x.min()), float(final_x.max())
                y0, y1 = float(final_y.min()), float(final_y.max())
                init_data.append(
                    go.Scatter(
                        x=[x0, x1, x1, x0, x0],
                        y=[y0, y0, y1, y1, y0],
                        mode="lines",
                        line=dict(color="rgba(214,39,40,0.7)", width=1.5, dash="dash"),
                        name="Final Pareto bounds",
                        hoverinfo="skip",
                    )
                )

            if show_knee:
                ki = result.knee_index()
                init_data.append(
                    go.Scatter(
                        x=[float(final_x[ki])],
                        y=[float(final_y[ki])],
                        mode="markers+text",
                        marker=dict(
                            symbol="star",
                            size=18,
                            color="crimson",
                            line=dict(color="black", width=1),
                        ),
                        text=["knee"],
                        textposition="top center",
                        name="Knee (compromise)",
                    )
                )

        layout = go.Layout(
            title=title,
            xaxis=dict(title=x_label),
            yaxis=dict(title=y_label),
            height=550,
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    x=0.0, y=-0.12,
                    xanchor="left", yanchor="top",
                    buttons=[
                        dict(
                            label="▶ Play",
                            method="animate",
                            args=[None, {"frame": {"duration": 150, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}],
                        ),
                        dict(
                            label="⏸ Pause",
                            method="animate",
                            args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}],
                        ),
                    ],
                )
            ],
            sliders=[
                dict(
                    active=0,
                    steps=slider_steps,
                    x=0.0, len=1.0, y=-0.05,
                    currentvalue=dict(prefix="Generation: ", visible=True, xanchor="center"),
                    transition=dict(duration=0),
                )
            ],
        )

        return go.Figure(data=init_data, frames=frames, layout=layout)

    # ------------------------------------------------------------------
    # plot_parallel_coordinates
    # ------------------------------------------------------------------

    def plot_parallel_coordinates(
        result: "CalibrationResult",
        color_by: Union[str, None] = None,
        title: str = "Pareto Front — Parallel Coordinates",
        use_final_pareto_only: bool = True,
        include_history: bool = False,
        highlight_index: Union[int, None] = None,
    ) -> go.Figure:
        """Parallel-axis trade-off explorer for Pareto solutions.

        Each line represents one solution.  Axes are the calibrated parameters
        followed by each objective (in display form).  Drag the endpoints of
        any axis to filter solutions.

        :param result: Calibration result.
        :type result: CalibrationResult
        :param color_by: Objective name to use for line colour.  Defaults to
            first objective, or ``"generation"`` when *include_history* is set.
        :type color_by: str or None
        :param title: Figure title.
        :type title: str
        :param use_final_pareto_only: Plot only the final Pareto front when
            ``True``; plot all final-generation solutions when ``False``.
            Ignored when *include_history* is ``True``.
        :type use_final_pareto_only: bool
        :param include_history: Plot every solution from **every** generation,
            prepending a ``generation`` axis and colouring lines by generation
            (unless overridden with *color_by*) so the evolution of the search
            is visible.
        :type include_history: bool
        :param highlight_index: Row index (into the plotted solutions, e.g.
            :meth:`~sparsehydro.calibration.result.CalibrationResult.knee_index`
            when plotting the final Pareto front) of a solution to highlight.
            A narrow ``constraintrange`` is applied around that solution's
            objective values so it stays coloured while all other lines fade.
        :type highlight_index: int or None
        :returns: Plotly Figure with ``go.Parcoords`` trace.
        :rtype: plotly.graph_objects.Figure
        """
        df = result.to_pareto_dataframe()
        if include_history:
            plot_df = df.reset_index(drop=True)
        elif use_final_pareto_only:
            plot_df = df[df["is_pareto"]].reset_index(drop=True)
        else:
            final_gen = df["generation"].max()
            plot_df = df[df["generation"] == final_gen].reset_index(drop=True)

        if len(plot_df) == 0:
            final_gen = df["generation"].max()
            plot_df = df[df["generation"] == final_gen].reset_index(drop=True)

        obj_names = result.objective_names
        if color_by and (color_by in obj_names or color_by in plot_df.columns):
            color_col = color_by
        elif include_history:
            color_col = "generation"
        else:
            color_col = obj_names[0]

        def _dim(col: str) -> dict:
            vals = plot_df[col].tolist()
            lo, hi = min(vals), max(vals)
            if lo == hi:
                lo -= 1e-6
                hi += 1e-6
            return dict(label=col, values=vals, range=[lo, hi])

        dims = []
        if include_history:
            dims.append(_dim("generation"))
        dims += [_dim(p) for p in result.param_names]
        dims += [_dim(o) for o in obj_names]

        if highlight_index is not None:
            if not 0 <= highlight_index < len(plot_df):
                raise IndexError(
                    f"highlight_index {highlight_index} is out of range for "
                    f"{len(plot_df)} plotted solutions."
                )
            for dim in dims:
                if dim["label"] not in obj_names:
                    continue
                value = float(plot_df.loc[highlight_index, dim["label"]])
                lo, hi = dim["range"]
                eps = 0.02 * (hi - lo)
                dim["constraintrange"] = [value - eps, value + eps]

        color_vals = plot_df[color_col].tolist()
        fig = go.Figure(
            data=go.Parcoords(
                line=dict(
                    color=color_vals,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title=color_col),
                    cmin=min(color_vals),
                    cmax=max(color_vals),
                ),
                dimensions=dims,
            )
        )
        fig.update_layout(title=title, height=500)
        return fig

    # ------------------------------------------------------------------
    # plot_objective_convergence
    # ------------------------------------------------------------------

    def plot_objective_convergence(
        result: "CalibrationResult",
        title: str = "Objective Convergence",
    ) -> go.Figure:
        """Per-objective convergence plot with best-value line and percentile band.

        One subplot row per objective.  Each row shows:

        - A solid line of the best value achieved at each generation.
        - A shaded 10th–90th percentile band across the Pareto population.

        :param result: Calibration result with per-generation history.
        :type result: CalibrationResult
        :param title: Figure title.
        :type title: str
        :returns: Plotly Figure with one row per objective.
        :rtype: plotly.graph_objects.Figure
        """
        n_obj = len(result.objective_names)
        min_flags = result.minimize_flags

        fig = make_subplots(
            rows=n_obj,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=result.objective_names,
        )

        generations = [rec.generation for rec in result.history]

        for j, (obj_name, is_min) in enumerate(zip(result.objective_names, min_flags)):
            best_vals, p10_vals, p90_vals = [], [], []

            for rec in result.history:
                col = _display_col(rec.F, j, min_flags)
                best_vals.append(col.min() if is_min else col.max())
                p10_vals.append(float(np.percentile(col, 10)))
                p90_vals.append(float(np.percentile(col, 90)))

            row = j + 1
            color = f"hsl({int(j * 360 / max(n_obj, 1))}, 70%, 45%)"
            fill_color = f"hsla({int(j * 360 / max(n_obj, 1))}, 70%, 70%, 0.2)"

            # Percentile band (area between p10 and p90)
            x_band = generations + generations[::-1]
            y_band = p10_vals + p90_vals[::-1]
            fig.add_trace(
                go.Scatter(
                    x=x_band,
                    y=y_band,
                    fill="toself",
                    fillcolor=fill_color,
                    line=dict(color="rgba(0,0,0,0)"),
                    name=f"{obj_name} 10–90%",
                    showlegend=(row == 1),
                    legendgroup="band",
                ),
                row=row, col=1,
            )

            # Best-value line
            fig.add_trace(
                go.Scatter(
                    x=generations,
                    y=best_vals,
                    mode="lines",
                    line=dict(color=color, width=2),
                    name=f"{obj_name} best",
                    showlegend=(row == 1),
                    legendgroup="best",
                ),
                row=row, col=1,
            )

            fig.update_yaxes(title_text=obj_name, row=row, col=1)

        fig.update_xaxes(title_text="Generation", row=n_obj, col=1)
        fig.update_layout(title=title, height=max(350, 200 * n_obj), hovermode="x unified")
        return fig

    # ------------------------------------------------------------------
    # plot_parameter_distributions
    # ------------------------------------------------------------------

    def plot_parameter_distributions(
        result: "CalibrationResult",
        use_final_pareto_only: bool = True,
        title: str = "Parameter Distributions (Pareto Solutions)",
    ) -> go.Figure:
        """Grid of violin plots showing the spread of calibrated parameters.

        Each violin shows the distribution of a parameter across Pareto
        solutions, with a box-and-whisker overlay and a jittered scatter of
        individual points.

        :param result: Calibration result.
        :type result: CalibrationResult
        :param use_final_pareto_only: Use only the final Pareto front when
            ``True``; use all history solutions when ``False``.
        :type use_final_pareto_only: bool
        :param title: Figure title.
        :type title: str
        :returns: Plotly Figure with a grid of violin traces.
        :rtype: plotly.graph_objects.Figure
        """
        n_params = len(result.param_names)
        n_cols = min(3, n_params)
        n_rows = math.ceil(n_params / n_cols)

        if use_final_pareto_only:
            X = result.pareto_X
        else:
            if result.history:
                X = np.vstack([rec.X for rec in result.history])
            else:
                X = result.pareto_X

        if X.shape[0] == 0:
            fig = go.Figure()
            fig.update_layout(title=title + " (no data)")
            return fig

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=result.param_names,
            vertical_spacing=0.1,
            horizontal_spacing=0.08,
        )

        for i, pname in enumerate(result.param_names):
            row = i // n_cols + 1
            col = i % n_cols + 1
            vals = X[:, i].tolist()

            fig.add_trace(
                go.Violin(
                    y=vals,
                    name=pname,
                    box_visible=True,
                    meanline_visible=True,
                    points="all",
                    jitter=0.3,
                    pointpos=0,
                    marker=dict(size=4, opacity=0.5),
                    fillcolor="rgba(100,149,237,0.4)",
                    line_color="cornflowerblue",
                    showlegend=False,
                ),
                row=row, col=col,
            )

        fig.update_layout(
            title=title,
            height=max(350, 300 * n_rows),
        )
        return fig

    # ------------------------------------------------------------------
    # plot_sensitivity_heatmap
    # ------------------------------------------------------------------

    def plot_sensitivity_heatmap(
        result: "CalibrationResult",
        title: str = "Parameter–Objective Sensitivity (Pearson r)",
    ) -> go.Figure:
        """Heatmap of absolute Pearson correlation between parameters and objectives.

        Rows = objectives, columns = parameters.  Cell colour = Pearson *r* on a
        RdBu diverging colorscale centred at 0.  Numeric annotation on each cell.

        :param result: Calibration result.
        :type result: CalibrationResult
        :param title: Figure title.
        :type title: str
        :returns: Plotly Figure with ``go.Heatmap`` trace.
        :rtype: plotly.graph_objects.Figure
        """
        X = result.pareto_X
        min_flags = result.minimize_flags
        n_obj = len(result.objective_names)

        if X.shape[0] < 3:
            fig = go.Figure()
            fig.update_layout(title=title + " (need ≥ 3 Pareto solutions)")
            return fig

        F_display = np.column_stack([
            _display_col(result.pareto_F, j, min_flags) for j in range(n_obj)
        ])

        n_params = len(result.param_names)
        corr_matrix = np.zeros((n_obj, n_params))
        for j in range(n_obj):
            for i in range(n_params):
                if np.std(X[:, i]) > 0 and np.std(F_display[:, j]) > 0:
                    corr_matrix[j, i] = float(np.corrcoef(X[:, i], F_display[:, j])[0, 1])

        text = [[f"{corr_matrix[r, c]:.2f}" for c in range(n_params)] for r in range(n_obj)]

        fig = go.Figure(
            data=go.Heatmap(
                z=corr_matrix,
                x=result.param_names,
                y=result.objective_names,
                colorscale="RdBu",
                zmid=0,
                zmin=-1,
                zmax=1,
                text=text,
                texttemplate="%{text}",
                colorbar=dict(title="Pearson r"),
            )
        )
        fig.update_layout(
            title=title,
            height=max(350, 80 * n_obj + 150),
            xaxis=dict(title="Parameter"),
            yaxis=dict(title="Objective"),
        )
        return fig

    # ------------------------------------------------------------------
    # plot_pareto_scatter_matrix
    # ------------------------------------------------------------------

    def plot_pareto_scatter_matrix(
        result: "CalibrationResult",
        title: str = "Pareto Scatter Matrix",
    ) -> go.Figure:
        """Scatter-plot matrix (SPLOM) of all objectives.

        Each off-diagonal cell is a scatter of one objective vs another.
        Points are coloured by the first objective.  Useful for exploring
        trade-off structure when there are three or more objectives.

        :param result: Calibration result.
        :type result: CalibrationResult
        :param title: Figure title.
        :type title: str
        :returns: Plotly Figure with ``go.Splom`` trace.
        :rtype: plotly.graph_objects.Figure
        """
        min_flags = result.minimize_flags
        n_obj = len(result.objective_names)

        F_display = np.column_stack([
            _display_col(result.pareto_F, j, min_flags) for j in range(n_obj)
        ])

        dimensions = [
            dict(label=name, values=F_display[:, j].tolist())
            for j, name in enumerate(result.objective_names)
        ]

        color_vals = F_display[:, 0].tolist()

        fig = go.Figure(
            data=go.Splom(
                dimensions=dimensions,
                marker=dict(
                    color=color_vals,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title=result.objective_names[0]),
                    size=5,
                    opacity=0.7,
                ),
                diagonal_visible=True,
                showupperhalf=False,
            )
        )
        fig.update_layout(
            title=title,
            height=max(400, 200 * n_obj),
            dragmode="select",
        )
        return fig

    # ------------------------------------------------------------------
    # plot_splom
    # ------------------------------------------------------------------

    def plot_splom(
        result: "CalibrationResult",
        space: str = "objectives",
        title: str = "SPLOM — Non-Dominated Solutions Highlighted",
    ) -> go.Figure:
        """Scatter-plot matrix highlighting non-dominated vs dominated solutions.

        Draws the full final-generation population as a SPLOM.  Dominated
        solutions are shown in grey at low opacity; non-dominated (Pareto-front)
        solutions are overlaid in orange at full opacity.

        :param result: Calibration result with per-generation history.
        :type result: CalibrationResult
        :param space: Which dimensions to display — ``"objectives"`` (default)
            or ``"parameters"``.
        :type space: str
        :param title: Figure title.
        :type title: str
        :returns: Plotly Figure with two ``go.Splom`` traces.
        :rtype: plotly.graph_objects.Figure
        :raises ValueError: If *space* is not ``"objectives"`` or ``"parameters"``.
        """
        if space not in ("objectives", "parameters"):
            raise ValueError(
                f"space must be 'objectives' or 'parameters', got {space!r}"
            )

        cols = result.objective_names if space == "objectives" else result.param_names

        df = result.to_pareto_dataframe()

        if df.empty:
            # No history recorded — treat all pareto solutions as non-dominated.
            if space == "objectives":
                F_disp = result.objective_display_values()
                data = {c: F_disp[:, j].tolist() for j, c in enumerate(cols)}
            else:
                data = {c: result.pareto_X[:, j].tolist() for j, c in enumerate(cols)}
            df_final = pd.DataFrame(data)
            df_final["is_pareto"] = True
        else:
            last_gen = int(df["generation"].max())
            df_final = df[df["generation"] == last_gen].copy()

        dominated = df_final[~df_final["is_pareto"]]
        non_dominated = df_final[df_final["is_pareto"]]

        def _dims(sub: pd.DataFrame) -> list:
            return [dict(label=c, values=sub[c].tolist()) for c in cols]

        traces: list = []

        if not dominated.empty:
            traces.append(
                go.Splom(
                    dimensions=_dims(dominated),
                    name="Dominated",
                    showlegend=True,
                    showupperhalf=False,
                    diagonal_visible=True,
                    marker=dict(color="rgba(150,150,150,0.3)", size=4, line=dict(width=0)),
                )
            )

        if not non_dominated.empty:
            traces.append(
                go.Splom(
                    dimensions=_dims(non_dominated),
                    name="Non-dominated",
                    showlegend=True,
                    showupperhalf=False,
                    diagonal_visible=True,
                    marker=dict(
                        color="#ff7f0e",
                        size=7,
                        opacity=0.85,
                        line=dict(width=0.5, color="white"),
                    ),
                )
            )

        n_dim = len(cols)
        fig = go.Figure(data=traces)
        fig.update_layout(
            title=title,
            height=max(400, 180 * n_dim),
            dragmode="select",
            legend=dict(itemsizing="constant"),
        )
        return fig

except ImportError:

    def plot_pareto_evolution(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("plotly is required for plot_pareto_evolution. Install with: pip install plotly")

    def plot_parallel_coordinates(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("plotly is required for plot_parallel_coordinates. Install with: pip install plotly")

    def plot_objective_convergence(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("plotly is required for plot_objective_convergence. Install with: pip install plotly")

    def plot_parameter_distributions(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("plotly is required for plot_parameter_distributions. Install with: pip install plotly")

    def plot_sensitivity_heatmap(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("plotly is required for plot_sensitivity_heatmap. Install with: pip install plotly")

    def plot_pareto_scatter_matrix(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("plotly is required for plot_pareto_scatter_matrix. Install with: pip install plotly")

    def plot_splom(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("plotly is required for plot_splom. Install with: pip install plotly")
