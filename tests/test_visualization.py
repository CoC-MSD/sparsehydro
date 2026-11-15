"""Visualization module tests — all functions called with synthetic data.

Artifacts (HTML files) are written for manual review in a browser, to
``$SPARSEHYDRO_TEST_ARTIFACTS/visualization`` when that variable is set and to
the system temp directory otherwise -- never into the repository.

Run:
    python -m pytest tests/test_visualization.py -v
"""

from __future__ import annotations

import os
import pathlib
import tempfile
from typing import List

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Artifact output directory
# ---------------------------------------------------------------------------

ARTIFACTS = pathlib.Path(
    os.environ.get("SPARSEHYDRO_TEST_ARTIFACTS")
    or pathlib.Path(tempfile.gettempdir()) / "sparsehydro-test-artifacts"
) / "visualization"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Skip guard
# ---------------------------------------------------------------------------

plotly = pytest.importorskip("plotly", reason="plotly not installed")
import plotly.graph_objects as go  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N = 120  # time steps
RNG = np.random.default_rng(42)


@pytest.fixture(scope="module")
def datetime_index():
    return pd.date_range("2020-01-01", periods=N, freq="h")


@pytest.fixture(scope="module")
def synthetic_ts(datetime_index):
    """Synthetic observed/predicted/rainfall arrays."""
    t = np.linspace(0, 4 * np.pi, N)
    observed = 10 + 5 * np.sin(t) + RNG.normal(0, 0.3, N)
    predicted = 10 + 4.8 * np.sin(t + 0.1) + RNG.normal(0, 0.2, N)
    rainfall = np.clip(RNG.exponential(1.5, N), 0, None)
    return datetime_index, observed, predicted, rainfall


@pytest.fixture(scope="module")
def ts_dataframe(synthetic_ts):
    dt, obs, pred, rain = synthetic_ts
    return pd.DataFrame({
        "datetime": dt,
        "observed": obs,
        "predicted": pred,
        "rainfall_mm": rain,
    })


def _make_calibration_result(
    n_solutions: int = 30,
    n_params: int = 2,
    n_obj: int = 2,
    n_generations: int = 10,
    minimize_flags: List[bool] | None = None,
):
    """Build a synthetic CalibrationResult with history."""
    from sparsehydro.calibration.result import CalibrationResult, GenerationRecord, _identify_pareto

    if minimize_flags is None:
        minimize_flags = [True] * n_obj

    history = []
    for gen in range(1, n_generations + 1):
        X = RNG.uniform(0, 1, (n_solutions, n_params))
        F = RNG.uniform(0, 1, (n_solutions, n_obj))
        pareto_mask = _identify_pareto(F)
        history.append(
            GenerationRecord(generation=gen, X=X, F=F, n_pareto=int(pareto_mask.sum()))
        )

    last = history[-1]
    pareto_mask = _identify_pareto(last.F)
    pareto_X = last.X[pareto_mask]
    pareto_F = last.F[pareto_mask]

    return CalibrationResult(
        history=history,
        pareto_X=pareto_X,
        pareto_F=pareto_F,
        param_names=[f"param_{i}" for i in range(n_params)],
        objective_names=[f"obj_{i}" for i in range(n_obj)],
        minimize_flags=minimize_flags,
    )


@pytest.fixture(scope="module")
def calib_result():
    return _make_calibration_result()


@pytest.fixture(scope="module")
def calib_result_3obj():
    return _make_calibration_result(n_obj=3, n_params=4, n_generations=8)


@pytest.fixture(scope="module")
def calib_result_mixed_flags():
    """Two objectives: one minimised, one maximised (NSE-like)."""
    return _make_calibration_result(n_obj=2, minimize_flags=[True, False])


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _save(fig: go.Figure, name: str) -> pathlib.Path:
    """Write figure to HTML artifact and return the path."""
    path = ARTIFACTS / f"{name}.html"
    fig.write_html(str(path))
    return path


# ---------------------------------------------------------------------------
# timeseries.py
# ---------------------------------------------------------------------------


class TestPlotTimeseries:
    def test_returns_figure(self, synthetic_ts):
        from sparsehydro.visualization import plot_timeseries

        dt, obs, pred, rain = synthetic_ts
        fig = plot_timeseries(dt, rain, obs, pred, title="Test Time Series")
        assert isinstance(fig, go.Figure)

    def test_two_rows(self, synthetic_ts):
        from sparsehydro.visualization import plot_timeseries

        dt, obs, pred, rain = synthetic_ts
        fig = plot_timeseries(dt, rain, obs, pred)
        # 2 rows = at least 3 traces (Bar + Observed + Predicted)
        assert len(fig.data) >= 3

    def test_no_observed(self, synthetic_ts):
        from sparsehydro.visualization import plot_timeseries

        dt, obs, pred, rain = synthetic_ts
        fig = plot_timeseries(dt, rain, None, pred)
        # Bar (rainfall) + Predicted only
        assert len(fig.data) == 2

    def test_artifact(self, synthetic_ts):
        from sparsehydro.visualization import plot_timeseries

        dt, obs, pred, rain = synthetic_ts
        fig = plot_timeseries(dt, rain, obs, pred, title="Time Series Artifact")
        path = _save(fig, "plot_timeseries")
        assert path.exists()
        assert path.stat().st_size > 1000


class TestPlotResidualsScatter:
    def test_returns_figure(self, synthetic_ts):
        from sparsehydro.visualization import plot_residuals_scatter

        dt, obs, pred, _ = synthetic_ts
        fig = plot_residuals_scatter(dt, obs, pred, title="Residual Diagnostics")
        assert isinstance(fig, go.Figure)

    def test_three_rows(self, synthetic_ts):
        from sparsehydro.visualization import plot_residuals_scatter

        dt, obs, pred, _ = synthetic_ts
        fig = plot_residuals_scatter(dt, obs, pred)
        # at least 5 traces (1:1 line, scatter, residual bar, ACF bar, h-lines)
        assert len(fig.data) >= 4

    def test_artifact(self, synthetic_ts):
        from sparsehydro.visualization import plot_residuals_scatter

        dt, obs, pred, _ = synthetic_ts
        fig = plot_residuals_scatter(dt, obs, pred, title="Residual Scatter Artifact")
        path = _save(fig, "plot_residuals_scatter")
        assert path.exists()
        assert path.stat().st_size > 1000

    def test_constant_residuals(self, datetime_index):
        """All-zero residuals should not raise (std=0 guard)."""
        from sparsehydro.visualization import plot_residuals_scatter

        obs = np.ones(N) * 5.0
        pred = np.ones(N) * 5.0
        fig = plot_residuals_scatter(datetime_index, obs, pred)
        assert isinstance(fig, go.Figure)


class TestPlotCumulativeVolume:
    def test_returns_figure(self, synthetic_ts):
        from sparsehydro.visualization import plot_cumulative_volume

        dt, obs, pred, _ = synthetic_ts
        fig = plot_cumulative_volume(dt, obs, pred, title="Cumulative Volume")
        assert isinstance(fig, go.Figure)

    def test_two_rows(self, synthetic_ts):
        from sparsehydro.visualization import plot_cumulative_volume

        dt, obs, pred, _ = synthetic_ts
        fig = plot_cumulative_volume(dt, obs, pred)
        assert len(fig.data) >= 2

    def test_artifact(self, synthetic_ts):
        from sparsehydro.visualization import plot_cumulative_volume

        dt, obs, pred, _ = synthetic_ts
        fig = plot_cumulative_volume(dt, obs, pred, title="Cumulative Volume Artifact")
        path = _save(fig, "plot_cumulative_volume")
        assert path.exists()
        assert path.stat().st_size > 1000


# ---------------------------------------------------------------------------
# calibration.py
# ---------------------------------------------------------------------------


class TestPlotParetoEvolution:
    def test_returns_figure(self, calib_result):
        from sparsehydro.visualization import plot_pareto_evolution

        fig = plot_pareto_evolution(calib_result, title="Pareto Evolution Test")
        assert isinstance(fig, go.Figure)

    def test_has_frames(self, calib_result):
        from sparsehydro.visualization import plot_pareto_evolution

        fig = plot_pareto_evolution(calib_result)
        assert len(fig.frames) == len(calib_result.history)

    def test_obj_by_name(self, calib_result):
        from sparsehydro.visualization import plot_pareto_evolution

        fig = plot_pareto_evolution(calib_result, x_obj="obj_0", y_obj="obj_1")
        assert isinstance(fig, go.Figure)

    def test_invalid_obj_name(self, calib_result):
        from sparsehydro.visualization import plot_pareto_evolution

        with pytest.raises(ValueError, match="not found"):
            plot_pareto_evolution(calib_result, x_obj="no_such_obj")

    def test_artifact(self, calib_result):
        from sparsehydro.visualization import plot_pareto_evolution

        fig = plot_pareto_evolution(calib_result, title="Pareto Evolution Artifact")
        path = _save(fig, "plot_pareto_evolution")
        assert path.exists()
        assert path.stat().st_size > 1000

    def test_mixed_minimize_flags(self, calib_result_mixed_flags):
        from sparsehydro.visualization import plot_pareto_evolution

        fig = plot_pareto_evolution(calib_result_mixed_flags, title="Pareto Mixed Flags")
        assert isinstance(fig, go.Figure)
        path = _save(fig, "plot_pareto_evolution_mixed_flags")
        assert path.exists()


class TestPlotParallelCoordinates:
    def test_returns_figure(self, calib_result):
        from sparsehydro.visualization import plot_parallel_coordinates

        fig = plot_parallel_coordinates(calib_result, title="Parallel Coords Test")
        assert isinstance(fig, go.Figure)

    def test_all_solutions(self, calib_result):
        from sparsehydro.visualization import plot_parallel_coordinates

        fig = plot_parallel_coordinates(calib_result, use_final_pareto_only=False)
        assert isinstance(fig, go.Figure)

    def test_artifact(self, calib_result):
        from sparsehydro.visualization import plot_parallel_coordinates

        fig = plot_parallel_coordinates(calib_result, title="Parallel Coordinates Artifact")
        path = _save(fig, "plot_parallel_coordinates")
        assert path.exists()
        assert path.stat().st_size > 1000


class TestPlotObjectiveConvergence:
    def test_returns_figure(self, calib_result):
        from sparsehydro.visualization import plot_objective_convergence

        fig = plot_objective_convergence(calib_result, title="Convergence Test")
        assert isinstance(fig, go.Figure)

    def test_three_objectives(self, calib_result_3obj):
        from sparsehydro.visualization import plot_objective_convergence

        fig = plot_objective_convergence(calib_result_3obj)
        assert isinstance(fig, go.Figure)

    def test_artifact(self, calib_result):
        from sparsehydro.visualization import plot_objective_convergence

        fig = plot_objective_convergence(calib_result, title="Objective Convergence Artifact")
        path = _save(fig, "plot_objective_convergence")
        assert path.exists()
        assert path.stat().st_size > 1000

    def test_artifact_3obj(self, calib_result_3obj):
        from sparsehydro.visualization import plot_objective_convergence

        fig = plot_objective_convergence(calib_result_3obj, title="3-Objective Convergence")
        path = _save(fig, "plot_objective_convergence_3obj")
        assert path.exists()


class TestPlotParameterDistributions:
    def test_returns_figure(self, calib_result):
        from sparsehydro.visualization import plot_parameter_distributions

        fig = plot_parameter_distributions(calib_result, title="Param Distributions Test")
        assert isinstance(fig, go.Figure)

    def test_all_history(self, calib_result):
        from sparsehydro.visualization import plot_parameter_distributions

        fig = plot_parameter_distributions(calib_result, use_final_pareto_only=False)
        assert isinstance(fig, go.Figure)

    def test_many_params(self):
        from sparsehydro.visualization import plot_parameter_distributions

        result = _make_calibration_result(n_params=7, n_solutions=25)
        fig = plot_parameter_distributions(result, title="Many Params")
        assert isinstance(fig, go.Figure)

    def test_artifact(self, calib_result):
        from sparsehydro.visualization import plot_parameter_distributions

        fig = plot_parameter_distributions(calib_result, title="Parameter Distributions Artifact")
        path = _save(fig, "plot_parameter_distributions")
        assert path.exists()
        assert path.stat().st_size > 1000

    def test_empty_pareto_returns_figure(self):
        """Zero-row pareto_X should return a figure with a note, not crash."""
        from sparsehydro.calibration.result import CalibrationResult, GenerationRecord
        from sparsehydro.visualization import plot_parameter_distributions

        X = RNG.uniform(0, 1, (0, 2))
        F = RNG.uniform(0, 1, (0, 2))
        result = CalibrationResult(
            history=[GenerationRecord(1, RNG.uniform(0, 1, (5, 2)), RNG.uniform(0, 1, (5, 2)), 0)],
            pareto_X=X,
            pareto_F=F,
            param_names=["p0", "p1"],
            objective_names=["o0", "o1"],
            minimize_flags=[True, True],
        )
        fig = plot_parameter_distributions(result)
        assert isinstance(fig, go.Figure)


class TestPlotSensitivityHeatmap:
    def test_returns_figure(self, calib_result):
        from sparsehydro.visualization import plot_sensitivity_heatmap

        fig = plot_sensitivity_heatmap(calib_result, title="Sensitivity Heatmap Test")
        assert isinstance(fig, go.Figure)

    def test_few_solutions(self):
        """Only 2 Pareto solutions — should return a figure with a note."""
        from sparsehydro.calibration.result import CalibrationResult, GenerationRecord
        from sparsehydro.visualization import plot_sensitivity_heatmap

        X = RNG.uniform(0, 1, (2, 2))
        F = RNG.uniform(0, 1, (2, 2))
        result = CalibrationResult(
            history=[GenerationRecord(1, X, F, 2)],
            pareto_X=X,
            pareto_F=F,
            param_names=["p0", "p1"],
            objective_names=["o0", "o1"],
            minimize_flags=[True, True],
        )
        fig = plot_sensitivity_heatmap(result)
        assert isinstance(fig, go.Figure)

    def test_artifact(self, calib_result):
        from sparsehydro.visualization import plot_sensitivity_heatmap

        fig = plot_sensitivity_heatmap(calib_result, title="Sensitivity Heatmap Artifact")
        path = _save(fig, "plot_sensitivity_heatmap")
        assert path.exists()
        assert path.stat().st_size > 1000


class TestPlotParetoScatterMatrix:
    def test_two_objectives(self, calib_result):
        from sparsehydro.visualization import plot_pareto_scatter_matrix

        fig = plot_pareto_scatter_matrix(calib_result, title="SPLOM 2 Obj")
        assert isinstance(fig, go.Figure)

    def test_three_objectives(self, calib_result_3obj):
        from sparsehydro.visualization import plot_pareto_scatter_matrix

        fig = plot_pareto_scatter_matrix(calib_result_3obj, title="SPLOM 3 Obj")
        assert isinstance(fig, go.Figure)

    def test_artifact(self, calib_result):
        from sparsehydro.visualization import plot_pareto_scatter_matrix

        fig = plot_pareto_scatter_matrix(calib_result, title="Pareto Scatter Matrix Artifact")
        path = _save(fig, "plot_pareto_scatter_matrix")
        assert path.exists()
        assert path.stat().st_size > 1000

    def test_artifact_3obj(self, calib_result_3obj):
        from sparsehydro.visualization import plot_pareto_scatter_matrix

        fig = plot_pareto_scatter_matrix(calib_result_3obj, title="Pareto SPLOM 3-Obj Artifact")
        path = _save(fig, "plot_pareto_scatter_matrix_3obj")
        assert path.exists()


# ---------------------------------------------------------------------------
# rdii.py
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rtk_triangles():
    """Two RTKTriangle instances (initialized + prepared)."""
    from sparsehydro.models.rdii import RTKTriangle
    tri1 = RTKTriangle(R=0.05, T=1.0, K=2.0)
    tri1.initialize()
    tri1.validate()
    tri2 = RTKTriangle(R=0.02, T=3.0, K=3.5)
    tri2.initialize()
    tri2.validate()
    return [tri1, tri2]


@pytest.fixture(scope="module")
def rdii_component_df(datetime_index, rtk_triangles):
    """DataFrame with synthetic rdii_component_N columns."""
    n = len(datetime_index)
    rain = np.clip(RNG.exponential(1.0, n), 0, None)
    df = pd.DataFrame({
        "datetime": datetime_index,
        "rainfall_mm": rain,
        "rdii_component_0": np.clip(RNG.exponential(0.5, n), 0, None),
        "rdii_component_1": np.clip(RNG.exponential(0.3, n), 0, None),
    })
    return df


class TestPlotRtkShape:
    def test_single_triangle(self, rtk_triangles):
        from sparsehydro.visualization import plot_rtk_shape

        fig = plot_rtk_shape(rtk_triangles[0], dt_hours=0.25)
        assert isinstance(fig, go.Figure)

    def test_multiple_triangles(self, rtk_triangles):
        from sparsehydro.visualization import plot_rtk_shape

        fig = plot_rtk_shape(rtk_triangles, dt_hours=0.25, title="RTK Shape Multi")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2

    def test_artifact(self, rtk_triangles):
        from sparsehydro.visualization import plot_rtk_shape

        fig = plot_rtk_shape(rtk_triangles, dt_hours=0.25, title="RTK Shape Artifact")
        path = _save(fig, "plot_rtk_shape")
        assert path.exists()
        assert path.stat().st_size > 1000


class TestPlotRdiiComponents:
    def test_returns_figure(self, rdii_component_df):
        from sparsehydro.visualization import plot_rdii_components

        fig = plot_rdii_components(rdii_component_df, title="RDII Components Test")
        assert isinstance(fig, go.Figure)

    def test_no_rainfall_column(self, rdii_component_df):
        from sparsehydro.visualization import plot_rdii_components

        df = rdii_component_df.drop(columns=["rainfall_mm"])
        fig = plot_rdii_components(df, title="RDII No Rainfall")
        assert isinstance(fig, go.Figure)

    def test_artifact(self, rdii_component_df):
        from sparsehydro.visualization import plot_rdii_components

        fig = plot_rdii_components(rdii_component_df, title="RDII Components Artifact")
        path = _save(fig, "plot_rdii_components")
        assert path.exists()
        assert path.stat().st_size > 1000


# ---------------------------------------------------------------------------
# dashboard.py
# ---------------------------------------------------------------------------


class TestPlotCalibrationDashboard:
    def test_returns_figure(self, calib_result):
        from sparsehydro.visualization import plot_calibration_dashboard

        fig = plot_calibration_dashboard(calib_result, title="Dashboard Test")
        assert isinstance(fig, go.Figure)

    def test_writes_html_file(self, calib_result, tmp_path):
        from sparsehydro.visualization import plot_calibration_dashboard

        out = tmp_path / "dashboard.html"
        plot_calibration_dashboard(calib_result, output_path=str(out))
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "<div" in content

    def test_with_timeseries_df(self, calib_result, ts_dataframe, tmp_path):
        from sparsehydro.visualization import plot_calibration_dashboard

        out = tmp_path / "dashboard_ts.html"
        fig = plot_calibration_dashboard(
            calib_result,
            timeseries_df=ts_dataframe,
            observed_col="observed",
            rainfall_col="rainfall_mm",
            output_path=str(out),
            title="Dashboard with Time Series",
        )
        assert isinstance(fig, go.Figure)
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "Time Series" in content

    def test_artifact_cdn(self, calib_result, ts_dataframe):
        from sparsehydro.visualization import plot_calibration_dashboard

        out = ARTIFACTS / "dashboard_cdn.html"
        plot_calibration_dashboard(
            calib_result,
            timeseries_df=ts_dataframe,
            observed_col="observed",
            rainfall_col="rainfall_mm",
            output_path=str(out),
            use_cdn=True,
            title="Calibration Dashboard (CDN)",
        )
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "cdn.plot.ly" in content

    def test_artifact_inline(self, calib_result):
        from sparsehydro.visualization import plot_calibration_dashboard

        out = ARTIFACTS / "dashboard_inline.html"
        plot_calibration_dashboard(
            calib_result,
            output_path=str(out),
            use_cdn=False,
            title="Calibration Dashboard (Inline Plotly)",
        )
        assert out.exists()
        # Inline version embeds the JS — it should be much larger
        assert out.stat().st_size > 100_000

    def test_no_output_path(self, calib_result):
        """output_path=None should not raise and should return a figure."""
        from sparsehydro.visualization import plot_calibration_dashboard

        fig = plot_calibration_dashboard(calib_result, output_path=None)
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# VisualizationModel (IModel lifecycle)
# ---------------------------------------------------------------------------


class TestVisualizationModel:
    def test_full_lifecycle_no_calibration(self, ts_dataframe):
        from sparsehydro.visualization import VisualizationModel

        vm = VisualizationModel(title="VM Test")
        vm.initialize()
        vm.validate()
        vm.prepare(
            ts_dataframe,
            datetime_col="datetime",
            predicted_col="predicted",
            observed_col="observed",
            rainfall_col="rainfall_mm",
        )
        out = vm.predict()
        assert isinstance(out, pd.DataFrame)
        assert isinstance(vm.timeseries_figure, go.Figure)
        assert vm.pareto_figure is None
        assert vm.parallel_figure is None
        vm.finalize()
        assert vm.timeseries_figure is None

    def test_full_lifecycle_with_calibration(self, ts_dataframe, calib_result):
        from sparsehydro.visualization import VisualizationModel

        vm = VisualizationModel(title="VM Cal Test")
        vm.initialize()
        vm.validate()
        vm.prepare(
            ts_dataframe,
            datetime_col="datetime",
            predicted_col="predicted",
            observed_col="observed",
            rainfall_col="rainfall_mm",
            calibration_result=calib_result,
        )
        vm.predict()
        assert isinstance(vm.pareto_figure, go.Figure)
        assert isinstance(vm.parallel_figure, go.Figure)

    def test_prepare_missing_column_raises(self, ts_dataframe):
        from sparsehydro.visualization import VisualizationModel

        vm = VisualizationModel()
        vm.initialize()
        vm.validate()
        with pytest.raises(ValueError, match="missing required columns"):
            vm.prepare(ts_dataframe, datetime_col="no_such_col", predicted_col="predicted")

    def test_predict_without_prepare_raises(self):
        from sparsehydro.visualization import VisualizationModel

        vm = VisualizationModel()
        vm.initialize()
        vm.validate()
        with pytest.raises(RuntimeError, match="prepare"):
            vm.predict()

    def test_artifact(self, ts_dataframe, calib_result):
        from sparsehydro.visualization import VisualizationModel

        vm = VisualizationModel(title="VisualizationModel Artifact")
        vm.initialize()
        vm.validate()
        vm.prepare(
            ts_dataframe,
            datetime_col="datetime",
            predicted_col="predicted",
            observed_col="observed",
            rainfall_col="rainfall_mm",
            calibration_result=calib_result,
        )
        vm.predict()
        _save(vm.timeseries_figure, "viz_model_timeseries")
        _save(vm.pareto_figure, "viz_model_pareto")
        _save(vm.parallel_figure, "viz_model_parallel")
        assert (ARTIFACTS / "viz_model_timeseries.html").exists()


# ---------------------------------------------------------------------------
# Import backward-compatibility checks
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_sparsehydro_top_level_imports(self):
        import sparsehydro

        # Only the curated core is re-exported at the top level.
        for name in (
            "ModelState",
            "IModel",
            "IUnitHydroComponent",
            "ScalarParameter",
            "VectorParameter",
            "FieldRecord",
            "ModelRegistry",
            "registry",
        ):
            assert hasattr(sparsehydro, name)

        # Visualization names are no longer hoisted to the top level; they live
        # in the sparsehydro.visualization sub-package.
        assert not hasattr(sparsehydro, "plot_timeseries")
        assert not hasattr(sparsehydro, "VisualizationModel")

    def test_visualization_module_imports(self):
        from sparsehydro.visualization import (
            VisualizationModel,
            plot_calibration_dashboard,
            plot_cumulative_volume,
            plot_objective_convergence,
            plot_parallel_coordinates,
            plot_parameter_distributions,
            plot_pareto_evolution,
            plot_pareto_scatter_matrix,
            plot_residuals_scatter,
            plot_sensitivity_heatmap,
            plot_timeseries,
        )
        assert callable(plot_timeseries)
        assert callable(plot_calibration_dashboard)
