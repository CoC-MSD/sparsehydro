"""``VisualizationModel``: an :class:`~sparsehydro.models.IModel` that produces figures.

Wraps the time-series and calibration plots behind the standard model
lifecycle so a pipeline can treat plotting like any other stage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..enums import ModelState
from ..models import IModel

if TYPE_CHECKING:
    from ..calibration.result import CalibrationResult

try:
    import plotly.graph_objects as go  # noqa: F401  (availability check)

    from .timeseries import plot_calibration_timeseries, plot_timeseries  # noqa: F401

    class VisualizationModel(IModel):
        """General-purpose visualization model for any :class:`~sparsehydro.models.IModel` output.

        Wraps :func:`plot_timeseries`, :func:`plot_pareto_evolution`, and
        :func:`plot_parallel_coordinates` behind the standard
        :class:`~sparsehydro.models.IModel` lifecycle.

        Column names for the input DataFrame are specified at :meth:`prepare`
        time, making this model compatible with any model's output format.

        :meth:`predict` returns the input DataFrame unchanged so the model can
        be composed in a pipeline.  Figures are available as read-only
        properties afterward::

            viz.timeseries_figure    # always generated
            viz.pareto_figure        # generated when calibration_result is supplied
            viz.parallel_figure      # generated when calibration_result is supplied

        :param title: Base title used across all figures.
        :type title: str
        :param rainfall_label: Y-axis label for the rainfall panel.
        :type rainfall_label: str
        :param flow_label: Y-axis label for the flow panel.
        :type flow_label: str
        """

        model_name = "visualization"

        def __init__(
            self,
            title: str = "Model Time Series",
            rainfall_label: str = "Rainfall [mm]",
            flow_label: str = "Flow",
        ) -> None:
            super().__init__()
            self._title = title
            self._rainfall_label = rainfall_label
            self._flow_label = flow_label
            self._data: pd.DataFrame | None = None
            self._datetime_col: str = "datetime"
            self._predicted_col: str = "predicted"
            self._observed_col: str | None = None
            self._rainfall_col: str | None = None
            self._calibration_result: "CalibrationResult | None" = None
            self._timeseries_fig: go.Figure | None = None
            self._pareto_fig: go.Figure | None = None
            self._parallel_fig: go.Figure | None = None

        def initialize(self) -> None:
            """Advance to INITIALIZED."""
            self._state = ModelState.INITIALIZED

        def validate(self) -> bool:
            """Validate and advance to VALIDATED.

            :returns: Always ``True``.
            """
            self._state = ModelState.VALIDATED
            return True

        def prepare(
            self,
            data: pd.DataFrame,
            datetime_col: str = "datetime",
            predicted_col: str = "predicted",
            observed_col: str | None = None,
            rainfall_col: str | None = None,
            calibration_result: "CalibrationResult | None" = None,
        ) -> None:
            """Load model output data and configure column mapping.

            :param data: Any model's output DataFrame.
            :type data: pandas.DataFrame
            :param datetime_col: Column name for the time axis.
            :type datetime_col: str
            :param predicted_col: Column name for the predicted output signal.
            :type predicted_col: str
            :param observed_col: Column name for observed data, or ``None`` to omit.
            :type observed_col: str or None
            :param rainfall_col: Column name for rainfall forcing, or ``None`` to omit.
            :type rainfall_col: str or None
            :param calibration_result: When provided, :meth:`predict` also
                generates Pareto-evolution and parallel-coordinates figures.
            :type calibration_result: CalibrationResult or None
            :raises ValueError: If required columns are absent.
            """
            missing = [c for c in (datetime_col, predicted_col) if c not in data.columns]
            if missing:
                raise ValueError(f"prepare() data is missing required columns: {missing}")
            self._data = data.copy()
            self._datetime_col = datetime_col
            self._predicted_col = predicted_col
            self._observed_col = observed_col
            self._rainfall_col = rainfall_col
            self._calibration_result = calibration_result
            self._timeseries_fig = None
            self._pareto_fig = None
            self._parallel_fig = None
            self._state = ModelState.PREPARED

        def predict(self) -> pd.DataFrame:
            """Generate all applicable figures and return the input DataFrame.

            :returns: The DataFrame supplied to :meth:`prepare` (unchanged).
            :raises RuntimeError: If :meth:`prepare` has not been called.
            """
            if self._data is None:
                raise RuntimeError("Call prepare(data) before predict().")

            from .calibration import plot_parallel_coordinates, plot_pareto_evolution

            df = self._data
            datetime = df[self._datetime_col]
            predicted = df[self._predicted_col].to_numpy(dtype=float)
            observed = (
                df[self._observed_col].to_numpy(dtype=float)
                if self._observed_col and self._observed_col in df.columns
                else None
            )
            rainfall = (
                df[self._rainfall_col].to_numpy(dtype=float)
                if self._rainfall_col and self._rainfall_col in df.columns
                else np.zeros(len(df))
            )

            self._timeseries_fig = plot_timeseries(
                datetime=datetime,
                rainfall_mm=rainfall,
                observed_flow=observed,
                predicted_flow=predicted,
                title=self._title,
                rainfall_label=self._rainfall_label,
                flow_label=self._flow_label,
            )

            if self._calibration_result is not None:
                self._pareto_fig = plot_pareto_evolution(
                    self._calibration_result,
                    title=f"{self._title} — Pareto Front Evolution",
                )
                self._parallel_fig = plot_parallel_coordinates(
                    self._calibration_result,
                    title=f"{self._title} — Parallel Coordinates",
                )

            self._state = ModelState.PREDICTED
            return df

        def finalize(self) -> None:
            """Release stored data and figures and advance to FINALIZED."""
            self._data = None
            self._calibration_result = None
            self._timeseries_fig = None
            self._pareto_fig = None
            self._parallel_fig = None
            self._state = ModelState.FINALIZED

        @property
        def timeseries_figure(self) -> "go.Figure | None":
            """Dual-axis rainfall/flow time series, or ``None`` before :meth:`predict`."""
            return self._timeseries_fig

        @property
        def pareto_figure(self) -> "go.Figure | None":
            """Animated Pareto-front evolution, or ``None`` when no result was supplied."""
            return self._pareto_fig

        @property
        def parallel_figure(self) -> "go.Figure | None":
            """Parallel-coordinates trade-off explorer, or ``None`` when no result was supplied."""
            return self._parallel_fig

except ImportError:  # pragma: no cover - plotly is optional

    class VisualizationModel(IModel):  # type: ignore[no-redef]
        model_name = "visualization"

        def initialize(self) -> None:
            raise ImportError("plotly is required for VisualizationModel. Install with: pip install plotly")

        def validate(self) -> bool:  # type: ignore[return]
            raise ImportError("plotly is required for VisualizationModel. Install with: pip install plotly")

        def prepare(self, data, **kwargs) -> None:
            raise ImportError("plotly is required for VisualizationModel. Install with: pip install plotly")

        def predict(self):  # type: ignore[return]
            raise ImportError("plotly is required for VisualizationModel. Install with: pip install plotly")

        def finalize(self) -> None:
            raise ImportError("plotly is required for VisualizationModel. Install with: pip install plotly")


__all__ = ["VisualizationModel"]
