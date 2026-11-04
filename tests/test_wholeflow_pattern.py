from tests.pattern_workflow_support import (
    configure_state, build_model_state, prepare_data_state, event_coverage_state,
    objective_state, calibrate_state, report_state, assessment_state,
)

import numpy as np
import pandas as pd
import pytest

from sparsehydro.calibration.objectives import ConcordanceCorrelationCoefficient, NashSutcliffe, Con15
from sparsehydro.calibration.msdgc_assessment import CFS_H_TO_MG, MSDGCAssessment, MSDGCSettings
from sparsehydro.calibration.result import CalibrationResult
from sparsehydro.calibration.solvers.platypus_solver import PlatypusSolver
from sparsehydro.calibration.wholeflow_workflow import FeasibleCalibrationProblem, dry_weather_mask, select_peak_filtered_pareto
from sparsehydro.models.rdii.initial_abstraction import SnowConfig, snow_partition
from sparsehydro.models.wholeflow_pattern import CenteredWeeklyPattern, PatternWholeFlowModel, estimate_sanitary_shapes


def study_data(dt=1.0, days=100):
    t = pd.date_range("2024-01-01", periods=int(days * 24 / dt),
                      freq=pd.Timedelta(hours=dt))
    rain = np.zeros(len(t))
    rain[int(72 / dt)::int(144 / dt)] = 0.6
    hour = 0.12 * np.sin(t.hour.to_numpy() * np.pi / 12)
    dow = np.where(t.dayofweek >= 5, -0.03, 0.012)
    flow = 0.8 + hour + dow + np.arange(len(t)) / len(t) * 0.1
    flow += np.convolve(rain, np.exp(-np.arange(int(48 / dt)) * dt / 8), mode="full")[:len(t)]
    return pd.DataFrame(dict(datetime=t, rainfall_in=rain, temperature_c=10.0, flow_cfs=flow))


def make_model(pattern=None, dt=1.0, snow_config=None):
    if pattern is None:
        h = 0.1 * np.sin(np.arange(24) * np.pi / 12)
        pattern = CenteredWeeklyPattern(tuple(h), (0.0,) * 7)
    model = PatternWholeFlowModel(pattern, 20.26, dt, 0.6, 2.0, snow_config=snow_config)
    model.initialize()
    return model


def test_shapes_are_centered_and_training_only():
    data = study_data()
    train = np.arange(len(data)) // (21 * 24) % 4 != 3
    eligible = train & dry_weather_mask(data, 1, 48)
    a = estimate_sanitary_shapes(data, eligible)
    changed = data.copy()
    changed.loc[~train, "flow_cfs"] += 1000
    b = estimate_sanitary_shapes(changed, eligible)
    assert a.pattern == b.pattern
    np.testing.assert_array_equal(a.detection_baseline_cfs, b.detection_baseline_cfs)
    pd.testing.assert_frame_equal(a.coverage, b.coverage)
    assert abs(np.mean(a.pattern.hourly)) < 1e-12
    assert abs(np.mean(a.pattern.weekday)) < 1e-12
    np.testing.assert_allclose(a.pattern.hourly,
                               .12 * np.sin(np.arange(24) * np.pi / 12), atol=.015)
    assert np.ptp(a.pattern.weekday) > .02
    with pytest.raises(ValueError, match="coverage"):
        estimate_sanitary_shapes(data, eligible & data.datetime.dt.hour.ne(12).to_numpy())
    with pytest.raises(ValueError):
        CenteredWeeklyPattern((1.0,) * 24, (0.0,) * 7)


@pytest.mark.parametrize("dt", [1 / 12, .25, 1.0])
def test_joint_components_budget_and_observation_free_prediction(dt):
    model = make_model(dt=dt)
    data = study_data(dt, days=10)
    forcing = data.drop(columns="flow_cfs")
    model.prepare(forcing)
    prediction = model.predict()
    model.prepare(data.assign(flow_cfs=np.nan))
    pd.testing.assert_frame_equal(prediction, model.predict())
    np.testing.assert_allclose(prediction.total_cfs,
                               prediction.sanitary_cfs + prediction.gwi_cfs + prediction.rdii_cfs)
    np.testing.assert_allclose(prediction.rdii_cfs,
                               prediction.rtk1_cfs + prediction.rtk2_cfs + prediction.rtk3_cfs,
                               atol=1e-12)
    assert prediction.gwi_cfs.max() > 0
    assert prediction.budget_error_in.abs().max() < 1e-8
    assert (prediction.recharge_in <= data.rainfall_in).all()
    assert (prediction.sanitary_cfs >= 0).all()
    calibrated = [n for n in model.scalar_parameter_names
                  if model.get_scalar_parameter(n).calibrate]
    assert len(calibrated) == 20
    assert "area_acres" not in calibrated  # Notebook enables it; library default is unchanged.
    assert "ia_k_dep" in calibrated and "ia_theta" in calibrated
    assert "sanitary_level_cfs" not in model.scalar_parameter_names
    assert "floor_offset_cfs" not in model.scalar_parameter_names
    for name in ("s_initial_fraction", "g_initial_fraction"):
        assert model.get_scalar_parameter(name).value == 0
        assert not model.get_scalar_parameter(name).calibrate


def test_sanitary_parameters_and_complete_week_nonnegativity():
    model = make_model()
    data = study_data(days=10)
    model.prepare(data)
    a = model.predict()
    model.get_scalar_parameter("sanitary_mean_cfs").value += .2
    b = model.predict()
    np.testing.assert_allclose(b.sanitary_cfs - a.sanitary_cfs, .2)
    np.testing.assert_array_equal(a.gwi_cfs, b.gwi_cfs)
    model.get_scalar_parameter("sanitary_hour_scale").value = 2
    c = model.predict()
    assert np.ptp(c.sanitary_cfs) == pytest.approx(2 * np.ptp(b.sanitary_cfs))
    model.get_scalar_parameter("sanitary_mean_cfs").value = 0
    assert model.inequality_constraints()[-1] > 0
    with pytest.raises(ValueError, match="feasible"):
        model.predict()


def test_sanitary_constraint_includes_hours_absent_from_prediction():
    model = make_model()
    model.get_scalar_parameter("sanitary_mean_cfs").value = .01
    daytime = study_data(days=2).iloc[6:8].copy()
    model.prepare(daytime)
    assert model.inequality_constraints()[-1] > 0
    with pytest.raises(ValueError, match="feasible"):
        model.predict()


def test_gwi_is_live_and_forcing_is_strict():
    model = make_model()
    data = study_data(days=15)
    model.prepare(data)
    a = model.predict()
    model.get_scalar_parameter("k_div").value *= 2
    b = model.predict()
    assert not np.allclose(a.gwi_cfs, b.gwi_cfs)
    np.testing.assert_array_equal(a.sanitary_cfs, b.sanitary_cfs)
    for bad in (data.assign(rainfall_in=np.nan), data.assign(rainfall_in=-1),
                data.assign(temperature_c=np.nan), data.drop(index=5)):
        with pytest.raises(ValueError):
            model.prepare(bad)




def test_delivered_notebook_has_one_solve_and_report_only_checker():
    import ast
    import json
    from pathlib import Path

    path = (Path(__file__).parents[1] / "sparsehydro" / "notebooks"
            / "flowfinity_gwi_wholeflow_pattern_joint.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    ids = [c["id"] for c in notebook["cells"]]
    assert len(ids) == len(set(ids))
    for cell in notebook["cells"]:
        suffix = cell["id"].removeprefix("pattern-joint-")
        if suffix.isdigit():
            assert cell["metadata"]["original_cell_index"] == int(suffix)
    expected_reports = [
        "pattern-joint-tsf-description", "pattern-joint-tsf-export",
        "pattern-joint-conveyance-description", "pattern-joint-conveyance-report",
    ]
    assert [name for name in ids if name in expected_reports] == expected_reports
    solve_cells, metrics_cells, conveyance_cells = [], [], []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        compile(source, str(path), "exec")
        tree = ast.parse(source)
        assert not any(isinstance(n, ast.Name) and n.id in {
            "floor_cfs", "gwi_model", "sanitary_problem", "calib_result",
        } for n in ast.walk(tree))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "calibrate_pattern_model":
                    solve_cells.append(cell["metadata"]["original_cell_index"])
                if node.func.id == "export_pattern_report":
                    metrics_cells.append(cell["metadata"]["original_cell_index"])
    assert solve_cells == [49]
    assert metrics_cells == [61]
    # Solver setup audits observed volumes only; modeled report scoring stays report-only.
    import inspect
    from sparsehydro.calibration.pattern_workflow import build_pattern_objectives
    from sparsehydro.calibration.pattern_reporting import assess_pattern_prediction

    for function in (build_pattern_objectives, assess_pattern_prediction):
        body = ast.parse(inspect.getsource(function))
        assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr == "conveyance_report" for n in ast.walk(body))
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr == "solve" for n in ast.walk(body))


@pytest.mark.parametrize("dt", [1 / 12, .25, 1.0])
@pytest.mark.parametrize("area_factor", [.7, 1.3])
def test_live_area_scales_gwi_and_rtk_but_not_stores_or_sanitary(dt, area_factor):
    model = make_model(dt=dt, snow_config=SnowConfig())
    data = study_data(dt, days=10)
    data.loc[:int(96 / dt), "temperature_c"] = -5
    model.prepare(data)
    a = model.predict()
    budget = model.budget.copy()
    area = model.get_scalar_parameter("area_acres")
    area.update(calibrate=True, value=area.value * area_factor)
    b = model.predict()
    for column in ("gwi_cfs", "rdii_cfs", "rtk1_cfs", "rtk2_cfs", "rtk3_cfs"):
        np.testing.assert_allclose(b[column], a[column] * area_factor, atol=1e-12)
    np.testing.assert_array_equal(b.sanitary_cfs, a.sanitary_cfs)
    np.testing.assert_allclose(b.total_cfs,
                               a.sanitary_cfs + area_factor * (a.gwi_cfs + a.rdii_cfs))
    for column in a:
        if not column.endswith("_cfs"):
            np.testing.assert_array_equal(a[column], b[column])
    assert model.budget == budget


@pytest.fixture
def notebook_sources():
    import json
    from pathlib import Path

    path = (Path(__file__).parents[1] / "sparsehydro" / "notebooks"
            / "flowfinity_gwi_wholeflow_pattern_joint.ipynb")
    sources = {c["id"]: "".join(c["source"]) for c in json.loads(
        path.read_text(encoding="utf-8"))["cells"]}
    sources["pattern-joint-00"] = "\n".join(
        source for cell_id, source in sources.items()
        if cell_id == "pattern-joint-00" or (
            cell_id.startswith("pattern-settings-") and cell_id != "pattern-settings-index"))
    return sources


@pytest.mark.parametrize("calibrate_area", [False, True])
@pytest.mark.parametrize("snow_mode", ["disabled", "fixed", "calibrated"])
def test_notebook_explicitly_wires_area_into_calibration(notebook_sources, calibrate_area, snow_mode):
    import time

    ns = dict(np=np, time=time, phase_seconds={})
    exec(notebook_sources["pattern-joint-00"], ns)
    assert ns["CALIBRATE_AREA"] is True
    ns.update(CALIBRATE_AREA=calibrate_area, SEASONAL_BASELINE="off", GROUNDWATER_ENABLED=True,
              odwf_cfs=.8, _dt_hours=1.0,
              SNOW_ENABLED=snow_mode != "disabled", CALIBRATE_SNOW=snow_mode == "calibrated",
              sanitary_pattern=make_model().pattern, model_df=study_data(days=10))
    build_model_state(ns)
    model = ns["joint_model"]
    area = model.get_scalar_parameter("area_acres")
    assert area.calibrate == calibrate_area
    assert area.value == pytest.approx(20.26)
    assert area.lower_bound == pytest.approx(14.182)
    assert area.upper_bound == pytest.approx(26.338)
    problem = FeasibleCalibrationProblem(
        model, ns["model_df"], objectives=[NashSutcliffe()],
        column_map={"observed": "flow_cfs", "predicted": "total_cfs"})
    assert ("area_acres" in problem.param_names) == calibrate_area
    assert problem.n_params == 20 + int(calibrate_area) + 2 * int(snow_mode == "calibrated")
    assert ("snow_T" in problem.param_names) == (snow_mode == "calibrated")
    assert ("snow_ddf" in problem.param_names) == (snow_mode == "calibrated")
    if snow_mode == "calibrated":
        for name, lower, upper in (("snow_T", -2, 3), ("snow_ddf", .005, .5)):
            parameter = model.get_scalar_parameter(name)
            assert (parameter.lower_bound, parameter.upper_bound) == (lower, upper)


@pytest.mark.parametrize("warmup_steps", [0, 6])
@pytest.mark.parametrize("has_validation", [False, True])
def test_notebook_whole_hydrograph_has_linked_rainfall(
    notebook_sources, monkeypatch, tmp_path, warmup_steps, has_validation,
):
    import json
    from types import SimpleNamespace

    import plotly.graph_objects as go
    from sparsehydro.visualization.display import add_val_vrects
    from sparsehydro.visualization.wholeflow_diagnostics import plot_total_flow

    data = study_data(days=10)
    model = make_model()
    model.prepare(data)
    prediction = model.predict()
    frame = data.assign(joint_sanitary_cfs=prediction.sanitary_cfs,
                        total_cfs=prediction.total_cfs)
    val = (np.arange(len(data)) >= 180) & has_validation
    train = (np.arange(len(data)) >= warmup_steps) & ~val
    event = SimpleNamespace(start_idx=72, end_idx=100)
    shown = []
    monkeypatch.setattr(go.Figure, "show", lambda fig: shown.append(fig))
    ns = dict(pd=pd, go=go, model_df=frame, base_cfs=prediction.baseflow_cfs,
              joint_pred=prediction, plot_total_flow=plot_total_flow,
              acceptance_events=pd.DataFrame([vars(event)]),
              obs_flow=data.flow_cfs, train_mask=train, val_mask=val,
              _ccc=ConcordanceCorrelationCoefficient(), _nse=NashSutcliffe(),
              _warmup_steps=warmup_steps, _events=[event], ARTIFACT_DIR=tmp_path,
              add_val_vrects=lambda fig, t, **kw: add_val_vrects(fig, t, val, **kw))
    exec(notebook_sources["pattern-joint-57"], ns)
    fig = ns["fig"]
    assert shown == [fig]
    rain, observed, modeled = fig.data
    assert rain.name == "Rainfall"
    assert (rain.xaxis, rain.yaxis) == ("x", "y")
    assert rain.fill == "tozeroy"
    np.testing.assert_array_equal(rain.y, data.rainfall_in)
    np.testing.assert_array_equal(observed.y, data.flow_cfs)
    np.testing.assert_array_equal(modeled.y, prediction.total_cfs)
    for trace in fig.data:
        np.testing.assert_array_equal(trace.x, data.datetime)
    assert (observed.xaxis, observed.yaxis) == ("x2", "y2")
    assert (modeled.xaxis, modeled.yaxis) == ("x2", "y2")
    assert fig.layout.xaxis.matches == "x2"
    assert fig.layout.yaxis.autorange == "reversed"
    assert fig.layout.yaxis.title.text == "Rain [in/step]"
    assert fig.layout.yaxis2.title.text == "Flow [cfs]"
    assert fig.layout.xaxis2.title.text == "Date"
    ccc_tot = ConcordanceCorrelationCoefficient().compute(data.flow_cfs, prediction.total_cfs, mask=train)
    assert f"train CCC={ccc_tot:.3f}" in fig.layout.title.text
    assert ("val CCC=" in fig.layout.title.text) == has_validation
    shapes = fig.layout.shapes
    assert len(shapes) == 1 + bool(warmup_steps) + has_validation
    assert all(s.xref == "x2" and s.yref == "y2 domain" for s in shapes)
    assert any(s.x0 == data.datetime.iloc[event.start_idx]
               and s.x1 == data.datetime.iloc[event.end_idx] for s in shapes)
    if warmup_steps:
        assert any(s.x0 == data.datetime.iloc[0]
                   and s.x1 == data.datetime.iloc[warmup_steps - 1] for s in shapes)
    if has_validation:
        assert any(s.x0 == data.datetime.iloc[180]
                   and s.x1 == data.datetime.iloc[-1] for s in shapes)
    export = tmp_path / "flowfinity_wholeflow_joint_hydrograph.html"
    html = export.read_text(encoding="utf-8")
    assert '"name":"Rainfall"' in html
    assert len(rain.y) == len(data)
    json.loads(fig.to_json())
    note = notebook_sources["pattern-joint-56"]
    assert "May 2, June 7, and June 16" in note
    assert "does not exclude them" in note


def test_notebook_g_plot_diagnostics_and_budget_use_selected_area(
    notebook_sources, monkeypatch, tmp_path,
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from sparsehydro.calibration.wholeflow_workflow import parameter_table
    from sparsehydro.calibration.pattern_reporting import groundwater_diagnostics
    from sparsehydro.visualization.wholeflow_diagnostics import plot_groundwater_store

    model = make_model()
    model.get_scalar_parameter("area_acres").update(value=26.0, calibrate=True)
    model.get_scalar_parameter("g_max").value = .01
    model.get_scalar_parameter("k_G").value = 1e-5
    model.get_scalar_parameter("k_div").value = .05
    data = study_data(days=30)
    model.prepare(data)
    p = model.predict()
    frame = data.copy()
    for column in p:
        frame[column] = p[column]
    train = np.arange(len(p)) < 400
    val = ~train
    monkeypatch.setattr(go.Figure, "show", lambda *args, **kwargs: None)
    monkeypatch.setattr(go.Figure, "write_html", lambda *args, **kwargs: None)
    ns = dict(np=np, pd=pd, go=go, make_subplots=make_subplots, joint_model=model,
              joint_pred=p, model_df=frame, _t=data.datetime, _S=p.S_in,
              AREA_ACRES=20.26, fitted_area_acres=26.0, _dt_hours=1.0, DT_HOURS=1.0,
              plot_groundwater_store=plot_groundwater_store, groundwater_diagnostics=groundwater_diagnostics,
              DATA_TITLE="test", ARTIFACT_DIR=tmp_path, DISPLAY_MAX_POINTS=None,
              thin_for_display=lambda fig, *args: fig, display=lambda *args: None,
              add_val_vrects=lambda *args, **kwargs: None, _warmup_steps=0,
              obs_flow=data.flow_cfs, q_total=p.total_cfs, q_rdii=p.rdii_cfs,
              parameter_table=parameter_table, train_mask=train, val_mask=val,
              scored_mask=train | val)
    exec(notebook_sources["pattern-joint-37"], ns)
    traces = {t.name: t for t in ns["fig"].data}
    assert traces["G [in] (deep store)"].yaxis == "y2"
    factor = 26 * 3630 / 3600
    np.testing.assert_allclose(traces["Slow G release [cfs]"].y, p.gwi_release_in * factor)
    np.testing.assert_allclose(traces["G capacity spill [cfs]"].y, p.gwi_spill_in * factor)
    precipitation = model.budget["initial_storage_in"] + np.cumsum(data.rainfall_in)
    accounted = np.cumsum(p.et_in + (p.gwi_cfs + p.rdii_cfs) / factor + p.unrouted_excess_in)
    accounted += p.S_in + p.G_in + p.routing_storage_in + p.SWE_in
    np.testing.assert_allclose(precipitation, accounted, atol=1e-8, rtol=0)
    exec(notebook_sources["pattern-joint-43"], ns)
    diagnostics = ns["gwi_diagnostics"].set_index("window")
    assert diagnostics.loc["whole", "spill_share_pct"] > 90
    assert diagnostics.loc["whole", "G_max_in"] == pytest.approx(p.G_in.max())
    assert diagnostics.loc["whole", "at_cap_pct"] == pytest.approx(100 * p.G_at_cap.mean())
    assert diagnostics.loc["train", "n"] == train.sum()
    assert diagnostics.loc["validation", "n"] == val.sum()
    ns["joint_pred"] = p.assign(gwi_release_in=0.0, gwi_spill_in=0.0)
    exec(notebook_sources["pattern-joint-43"], ns)
    assert ns["gwi_diagnostics"].spill_share_pct.isna().all()
    assert ns["gwi_diagnostics"].status.eq("undefined_no_discharge").all()


@pytest.mark.parametrize("top_fraction", [.15, .3, 1.])
@pytest.mark.parametrize("scoring_mode,objective_names", [
    ("multiobjective", ["ccc", "nash_sutcliffe"]),
    ("multiobjective", ["con15_error", "nash_sutcliffe", "ccc"]),
    ("multiobjective", ["con15_error", "ccc"]),
    ("multiobjective", ["peak_weighted_mse", "ccc"]),
    ("multiobjective", ["nash_sutcliffe"]),
    ("multiobjective", ["con15_error"]),
    ("single", ["ccc"]),
    ("single", ["nash_sutcliffe"]),
    ("single", ["con15_error"]),
])
@pytest.mark.parametrize("use_filter", [False, True])
def test_notebook_calibrates_lean_but_builds_full_selected_output(
    notebook_sources, monkeypatch, tmp_path, capsys, top_fraction, scoring_mode,
    objective_names, use_filter,
):
    import time
    from sparsehydro.calibration.objectives import PBIAS

    if scoring_mode == "single":
        pytest.importorskip("optuna")
        pytest.importorskip("cmaes")
    use_objective = "con15_error" in objective_names
    expected_names = objective_names
    cma_jobs = 2 if (scoring_mode == "single" and expected_names == ["ccc"]
                     and top_fraction == .3 and not use_filter) else 1
    data = study_data(days=10)
    ns = dict(np=np, pd=pd, time=time, phase_seconds={},
              PatternWholeFlowModel=PatternWholeFlowModel,
              ConcordanceCorrelationCoefficient=ConcordanceCorrelationCoefficient,
              NashSutcliffe=NashSutcliffe, PBIAS=PBIAS, Con15=Con15)
    exec(notebook_sources["pattern-joint-00"], ns)
    ns.update(odwf_cfs=.8, _dt_hours=1.0, SEASONAL_BASELINE="off", GROUNDWATER_ENABLED=True,
              SOLVER="auto", sanitary_pattern=make_model().pattern,
              model_df=data.assign(stormflow=data.flow_cfs - .6),
              NSGA2_SETTINGS={"pop_size": 12, "generations": 4, "seed": 42},
              CON15_OBJ_FRACTION=top_fraction,
              SCORING_MODE=scoring_mode, MULTIOBJECTIVE_OBJECTIVES=objective_names,
              SINGLE_OBJECTIVE=objective_names[0], USE_TRAINING_PEAK_FILTER=use_filter,
              CMA_ES_SETTINGS={"cma_n_trials": 32, "cma_multi_start": 2, "cma_n_jobs": cma_jobs,
                               "cma_seed": 42, "cma_startup_trials": 4, "cma_progress_every": 0},
              CON15_FILTER_FRACTION=1., MAX_TRAINING_PEAK_ERROR=10.,
              ARTIFACT_DIR=tmp_path,
              msdgc_settings=MSDGCSettings(minimum_minutes=(60., 20., 30.),
                                           maximum_minutes=(60., 240., 1440.)))
    build_model_state(ns)
    snow = snow_partition(data.rainfall_in.to_numpy(), data.temperature_c.to_numpy(),
                          1.0, ns["SNOW_CONFIG"])
    for column in snow:
        ns["model_df"][column] = snow[column].to_numpy()
    ns["model_df"]["possible_melt"] = snow.melt_in.gt(0).to_numpy()
    train = np.arange(len(data)) < 168
    ns.update(train_mask=train, val_mask=~train,
              storm_mask=train & (np.arange(len(data)) >= 72),
              acceptance_events=pd.DataFrame({
                  "event_id": ["training-event", "validation-event"],
                  "start_idx": [72, 216], "end_idx": [120, 239],
                  "duration_hours": [48., 23.],
                  "split": ["train", "validation"],
              }))
    full_calls, lean_calls = [], []
    full_predict = PatternWholeFlowModel.predict
    lean_predict = PatternWholeFlowModel.predict_total_cfs

    def predict(model):
        full_calls.append(model)
        return full_predict(model)

    def predict_total(model):
        lean_calls.append(model)
        return lean_predict(model)

    monkeypatch.setattr(PatternWholeFlowModel, "predict", predict)
    monkeypatch.setattr(PatternWholeFlowModel, "predict_total_cfs", predict_total)
    calibrate_state(ns)
    if scoring_mode == "multiobjective":
        assert ns["CALIBRATION_WORKERS"] == ns["solver2"].n_workers == 4
    else:
        assert ns["solver2"].objective_optimized_ == expected_names[0]
        assert len(ns["joint_result"].pareto_X) == 1
        assert len(ns["joint_result"].history) == 64
        assert len(ns["solver2"].seed_solutions_) == 2
        assert ns["solver2"].selected_seed_ in (42, 43)
        seed_scores = [seed["opt_value"] for seed in ns["solver2"].seed_solutions_]
        best_score = min(seed_scores) if objective_names[0] in {
            "con15_error", "peak_weighted_mse"} else max(seed_scores)
        assert ns["joint_result"].objective_display_values()[0, 0] == pytest.approx(best_score)
    assert len(full_calls) == 1 and full_calls[0] is ns["joint_model"]
    if cma_jobs == 1:
        assert lean_calls and all(model is not ns["joint_model"] for model in lean_calls)
    prediction = ns["joint_pred"]
    assert isinstance(prediction, pd.DataFrame)
    assert {"gwi_release_in", "gwi_spill_in", "rtk3_cfs", "budget_error_in"} <= set(prediction)
    np.testing.assert_array_equal(ns["q_total"], lean_predict(ns["joint_model"]))
    assert ns["joint_model"].budget["max_abs_error_in"] < 1e-8
    result = ns["joint_result"]
    assert result.objective_names == expected_names
    assert result.minimize_flags == [
        name in {"con15_error", "peak_weighted_mse"} for name in objective_names]
    assert result.pareto_F.shape[1] == len(objective_names)
    np.testing.assert_allclose(ns["_objectives"], ns["_verified_objectives"])
    if "ccc" in expected_names:
        column = expected_names.index("ccc")
        original_ccc = ConcordanceCorrelationCoefficient(mask=ns["storm_mask"])
        expected_ccc = original_ccc.compute(data.flow_cfs.to_numpy(), ns["q_total"])
        assert ns["_objectives"][column] == pytest.approx(expected_ccc)
    audit = ns["peak_objective_events"]
    assert audit.event_id.tolist() == ["training-event"]
    assert audit.split.eq("train").all()
    assert audit.objective_selected.eq(use_objective).all()
    assert audit.filter_selected.eq(use_filter).all()
    observed_peak = data.flow_cfs.iloc[72:121].max()
    predicted_peak = ns["q_total"][72:121].max()
    observed_volume = round(observed_peak * CFS_H_TO_MG, 4)
    predicted_volume = round(predicted_peak * CFS_H_TO_MG, 4)
    expected_peak = abs(predicted_volume / observed_volume - 1)
    if use_objective:
        assert ns["peak_objective"].top_fraction == ns["CON15_OBJ_FRACTION"] == top_fraction
        np.testing.assert_array_equal(ns["peak_objective"].mask, train)
        np.testing.assert_array_equal(ns["peak_objective"].event_windows, [[72, 120]])
        assert ns["fit"].split_diagnostics.set_index("split").loc["validation", "con15_status"] == "defined"
        selected = result.objective_display_values()[ns["selected_index"]]
        column = objective_names.index("con15_error")
        assert selected[column] == pytest.approx(expected_peak)
        assert result.pareto_F[ns["selected_index"], column] == pytest.approx(expected_peak)
        exported = result.to_pareto_dataframe()
        np.testing.assert_array_equal(
            exported["con15_error"],
            np.concatenate([record.F[:, column] for record in result.history]),
        )
    else:
        assert ns["peak_objective"] is None
    front = ns["pareto_solutions"]
    assert len(front) == len(result.pareto_X)
    np.testing.assert_array_equal(front[result.param_names], result.pareto_X)
    assert front.selected.sum() == 1
    if use_filter:
        assert ns["peak_filter"].top_fraction == 1.
        np.testing.assert_array_equal(ns["peak_filter"].event_windows, [[72, 120]])
        assert front.loc[ns["selected_index"], "filter_peak_error"] == pytest.approx(expected_peak)
        assert ns["selection_status"] == "near_tie_peak_pass"
        assert front.filter_evaluated.all() and front.filter_pass.all()
        primary_names = [name for name in objective_names if name != "con15_error"] or objective_names
        assert front.attrs["primary_objective_names"] == primary_names
        assert front.primary_baseline.sum() == 1
        assert front.loc[ns["selected_index"], "near_tie_eligible"]
        assert front.loc[ns["selected_index"], "primary_score_increase"] <= .02 + 1e-12
        shortlisted = front.loc[front.near_tie_eligible]
        assert front.loc[ns["selected_index"], "filter_peak_error"] == shortlisted.filter_peak_error.min()
        if scoring_mode == "single":
            assert front.primary_score.tolist() == [0.]
    else:
        assert ns["peak_filter"] is None
        assert ns["selected_index"] == result.knee_index()
        assert ns["selection_status"] == "filter_disabled"
        assert front.filter_pass.isna().all() and front.filter_peak_error.isna().all()
    saved_front = pd.read_csv(tmp_path / "pareto_solutions.csv")
    assert len(saved_front) == len(result.pareto_X)
    np.testing.assert_array_equal(saved_front.pareto_index, np.arange(len(result.pareto_X)))
    np.testing.assert_allclose(saved_front[result.param_names], result.pareto_X, atol=1e-14)
    assert saved_front.loc[ns["selected_index"], "selection_status"] == ns["selection_status"]
    if top_fraction == .3:
        import json
        import plotly.graph_objects as go
        from pathlib import Path
        from types import SimpleNamespace
        from sparsehydro.calibration.splits import SplitConfig

        monkeypatch.setattr(go.Figure, "show", lambda *args, **kwargs: None)
        ns.update(use_webgl=lambda fig: fig, thin_for_display=lambda fig, points: fig)
        plot_source = next(source for source in notebook_sources.values()
                           if "fig = use_webgl(plot_parallel_coordinates(" in source)
        exec(plot_source, ns)
        assert ns["solver_label"] in ns["fig"].layout.title.text
        assert ns["_display_result"].objective_names == expected_names
        labels = [dimension.label for dimension in ns["fig"].data[0].dimensions]
        assert labels[-len(expected_names):] == expected_names

        zeros = np.zeros(len(data), bool)
        ns.update(
            Path=Path, display=lambda value: None, odwf_cfs=.1,
            dry_eligible=~data.rainfall_in.gt(0).to_numpy(),
            warmup_mask=zeros, scored_mask=np.ones(len(data), bool),
            events_df=ns["acceptance_events"].assign(eligible=True),
            param_tbl=ns["parameter_table"](ns["joint_model"]),
            seasonal_diagnostics=pd.DataFrame(), gwi_diagnostics=pd.DataFrame(),
            shape_fit=SimpleNamespace(coverage=pd.DataFrame()),
            val_storm_mask=~train, rain_storm_mask=ns["storm_mask"],
            val_rain_storm_mask=~train, melt_mask=zeros, val_melt_mask=zeros,
            SPLIT_CFG=SplitConfig(dt_hours=1., warmup_days=0, spinup_mode="none"),
            FLOW_FILE=tmp_path / "flow.tsf", RAIN_FILE=tmp_path / "rain.tsf",
            _air_csv=tmp_path / "air.csv",
            EVENT_SOURCE="file" if use_filter else "auto",
            _event_file=tmp_path / "approved.csv" if use_filter else None,
            _all_event_file=None, calibration_events_df=ns["acceptance_events"].assign(eligible=True),
            split_train_mask=train, split_val_mask=~train,
            observation_eligible=np.ones(len(data), bool),
            approved_storm_mask=ns["storm_mask"], val_approved_storm_mask=~train,
            assessment_storm_mask=ns["storm_mask"], val_assessment_storm_mask=~train,
            assessment_events_df=ns["acceptance_events"].assign(eligible=True, exclusion=""),
            event_mask=ns["storm_mask"] | ~train,
            PRE_EVENT_BUFF_H=0. if use_filter else 3.,
            RECESSION_TAIL_H=0. if use_filter else 48.,
        )
        report_state(ns)
        metadata = json.loads((tmp_path / "run_metadata.json").read_text())
        from sparsehydro.configuration import StudySettings
        saved_settings = json.loads((tmp_path / "study_settings.json").read_text())
        assert StudySettings.from_dict(saved_settings) == ns["study_settings"]
        assert metadata["groundwater_enabled"] is True
        assert metadata["n_rtk_triangles"] == 3
        assert metadata["event_source"] == ("file" if use_filter else "auto")
        assert metadata["event_overlap_policy"] == (
            "independent approved windows" if use_filter else "ordered disjoint windows")
        assert metadata["input_events"] == (str(tmp_path / "approved.csv") if use_filter else None)
        assert ("no padding or extra melt windows" in metadata["event_definition"]) is use_filter
        assert metadata["scoring_mode"] == scoring_mode
        assert metadata["objective_names"] == expected_names
        assert list(metadata["selected_objectives"]) == expected_names
        assert "objective_definitions" not in metadata
        assert "report_metric_labels" not in metadata
        if scoring_mode == "single":
            assert metadata["solver"] == "optuna.samplers.CmaEsSampler"
            assert metadata["requested_evaluations"] == 64
            assert metadata["trials_per_seed"] == 32
            assert metadata["multi_start"] == 2
            assert metadata["selected_seed"] == ns["solver2"].selected_seed_
            assert metadata["calibration_workers"] == cma_jobs
            assert metadata["evaluation_backend"] == ("seed processes" if cma_jobs > 1 else "serial")
            assert "population" not in metadata and "generations" not in metadata
        else:
            assert metadata["solver"] == "platypus.NSGAII"
            assert metadata["requested_evaluations"] == 48
            assert metadata["calibration_workers"] == 4
            assert metadata["evaluation_backend"] == "ordered threads"
        assert metadata["con15_objective_enabled"] is use_objective
        assert metadata["peak_filter"]["enabled"] is use_filter
        assert metadata["peak_filter"]["top_fraction"] == 1.
        assert metadata["peak_top_fraction"] == top_fraction
        assert metadata["selection_status"] == ns["selection_status"]
        assert metadata["final_pareto_count"] == len(front)
        assert metadata["final_pareto_file"] == "pareto_solutions.csv"
        assert metadata["peak_filter"]["selected_passes"] is (True if use_filter else None)
        assert metadata["near_tie_selection"]["enabled"] is use_filter
        assert metadata["near_tie_selection"]["allowance"] == .02
        if use_filter:
            assert "threshold report only" in metadata["selection"]
            near_tie = metadata["near_tie_selection"]
            assert near_tie["primary_objectives"] == primary_names
            assert near_tie["primary_baseline_index"] == int(front.index[front.primary_baseline][0])
            assert near_tie["best_primary_score"] == pytest.approx(front.primary_score.min())
            assert near_tie["selected_primary_score_increase"] <= .02 + 1e-12
            assert near_tie["shortlisted_candidate_count"] == int(front.near_tie_eligible.sum())
        if use_objective:
            assert metadata["selected_peak_slope"] == pytest.approx(predicted_volume / observed_volume)
        else:
            assert metadata["selected_peak_slope"] is None
        selected_ids = metadata["peak_filter"]["selected_training_event_ids"]
        assert selected_ids == (["training-event"] if use_filter else [])
        audit_export = pd.read_csv(tmp_path / "peak_objective_events.csv")
        assert audit_export.event_id.tolist() == ["training-event"]
        assert audit_export.objective_selected.eq(use_objective).all()
        assert audit_export.filter_selected.eq(use_filter).all()
        acceptance = pd.read_csv(tmp_path / "acceptance.csv")
        assert set(acceptance.metric) == set(MSDGCAssessment.report_metric_names)
        assert "metric_label" not in acceptance.columns
        assert metadata["acceptance"]["qcritical_cfs"] is None
        assert metadata["conveyance_report"]["aggregation"].startswith("CCC across event")
        assert metadata["msdgc_assessment"]["official_storm_scorecard"] is False
        assert metadata["con15_basis"] == "msdgc_independent_conveyance_volume_mg"
        assert (tmp_path / "msdgc_zone_volumes.csv").exists()
        assert (tmp_path / "assessment_event_inventory.csv").exists()
        assert (tmp_path / "conveyance_event_volumes.csv").exists()
        assert metadata["passing_pareto_report"]["enabled"] is use_filter
        if use_filter:
            passing = pd.read_csv(tmp_path / "pareto_filter_pass_metrics.csv")
            assert {"primary_score", "primary_score_increase", "near_tie_eligible"} <= set(passing)
            assert len(passing) == int(front.filter_pass.sum())
            assert metadata["passing_pareto_report"]["count"] == len(passing)
            assert set(passing.pareto_index) == set(front.index[front.filter_pass.fillna(False)])
            for split in ("train", "validation", "whole"):
                selected_scores = acceptance.loc[acceptance.window.eq(split)].set_index("metric")
                for metric in MSDGCAssessment.report_metric_names:
                    report_value = passing.loc[passing.pareto_index.eq(ns["selected_index"]),
                                               f"{split}_{metric}"].iloc[0]
                    expected = selected_scores.loc[metric, "value"]
                    np.testing.assert_allclose(report_value, expected, equal_nan=True)
        else:
            assert "passing_pareto_tbl" not in ns
            assert not (tmp_path / "pareto_filter_pass_metrics.csv").exists()
        printed = capsys.readouterr().out
        assert "train: total CCC=" in printed
        assert "validation: total CCC=" in printed
        assert "event_volume_ccc" in printed
        assert "conveyance_ccc" in printed
        assert "Verified selected objectives:" in printed
        history = pd.read_csv(tmp_path / "objective_history.csv")
        assert set(expected_names) <= set(history.columns)
        assert ("# of Con15-filter acceptable solutions:" in printed) is use_filter
        if use_filter:
            assert "train_event_volume_ccc" in printed
            assert "validation_conveyance_ccc" in printed


def peak_filter_case(means=(.5, .9, 1., 1.1, 1.5), with_objective=True):
    model = make_model(CenteredWeeklyPattern((0.,) * 24, (0.,) * 7))
    data = study_data(days=1).assign(rainfall_in=0.)
    model.prepare(data)
    observed = np.ones(24)
    observed[6:9] = .5
    observed[12:] = 1e6
    mask = np.arange(24) < 12
    peak_filter = Con15(np.array([[1, 3], [6, 8]]), top_fraction=.5, mask=mask)
    objectives = ["ccc", "nash_sutcliffe"] + (["con15_error"] if with_objective else [])
    values = np.array([[-.99, -.99], [-.95, -.8], [-.9, -.9], [-.8, -.95], [-.7, -.7]])
    values = values[:len(means)]
    if with_objective:
        # A different (all-events) objective fraction from the filter's top 50%.
        values = np.column_stack((values, np.abs(1.2 * np.asarray(means) - 1)))
    result = CalibrationResult(
        [], np.asarray(means)[:, None], values, ["sanitary_mean_cfs"],
        objectives, [False, False] + ([True] if with_objective else []),
    )
    return result, model, observed, peak_filter


@pytest.mark.parametrize("with_objective", [False, True])
def test_peak_filter_uses_independent_fraction_and_configured_objective_compromise(with_objective):
    result, model, observed, peak_filter = peak_filter_case(with_objective=with_objective)
    before = model.predict()
    original_x, original_f = result.pareto_X.copy(), result.pareto_F.copy()
    index, table = select_peak_filtered_pareto(
        result, model, observed, peak_filter=peak_filter, max_error=.1,
    )
    passing = np.array([1, 2, 3])
    subset = CalibrationResult(
        [], result.pareto_X[passing], result.pareto_F[passing],
        result.param_names, result.objective_names, result.minimize_flags,
    )
    assert index == passing[subset.knee_index()]
    np.testing.assert_allclose(table.filter_peak_slope, [.5, .9, 1., 1.1, 1.5])
    assert table.filter_pass.tolist() == [False, True, True, True, False]
    assert table.loc[index, "selection_status"] == "filter_pass_compromise"
    np.testing.assert_array_equal(table[result.param_names], original_x)
    np.testing.assert_array_equal(result.pareto_X, original_x)
    np.testing.assert_array_equal(result.pareto_F, original_f)
    pd.testing.assert_frame_equal(model.predict(), before)
    if with_objective:
        assert not np.allclose(table.con15_error, table.filter_peak_error)
    changed = observed.copy()
    changed[12:] = np.nan
    other_index, other_table = select_peak_filtered_pareto(
        result, model, changed, peak_filter=peak_filter, max_error=.1,
    )
    assert other_index == index
    pd.testing.assert_frame_equal(other_table, table)


@pytest.mark.parametrize("columns", [[2], [1], [2, 0], [1, 2, 0]])
def test_peak_filter_selection_supports_subsets_and_objective_order(columns):
    result, model, observed, peak_filter = peak_filter_case()
    result = CalibrationResult(
        [], result.pareto_X, result.pareto_F[:, columns], result.param_names,
        [result.objective_names[i] for i in columns],
        [result.minimize_flags[i] for i in columns],
    )
    passing = np.array([1, 2, 3])
    subset = CalibrationResult(
        [], result.pareto_X[passing], result.pareto_F[passing],
        result.param_names, result.objective_names, result.minimize_flags,
    )
    selected, table = select_peak_filtered_pareto(
        result, model, observed, peak_filter=peak_filter, max_error=.1,
    )
    assert selected == passing[subset.knee_index()]
    np.testing.assert_allclose(table[result.objective_names], result.objective_display_values())


@pytest.mark.parametrize("allowance", [None, .02])
def test_disabled_peak_filter_keeps_original_knee_without_predictions(monkeypatch, allowance):
    result, model, observed, _ = peak_filter_case()

    def no_prediction(*args, **kwargs):
        raise AssertionError("A disabled filter must not rerun candidate simulations")

    monkeypatch.setattr(PatternWholeFlowModel, "predict_total_cfs", no_prediction)
    index, table = select_peak_filtered_pareto(result, model, observed, near_tie_allowance=allowance)
    assert index == result.knee_index()
    assert table.filter_pass.isna().all() and table.filter_peak_slope.isna().all()
    assert not table.filter_evaluated.any()
    assert table.loc[index, "selection_status"] == "filter_disabled"
    assert len(table) == len(result.pareto_X)


def near_tie_case(means=(.5, .9, 1., 1.1, 1.5), with_objective=True):
    result, model, observed, peak_filter = peak_filter_case(
        means=means, with_objective=with_objective)
    scores = np.array([0., .02, .04, .98, 1.])
    result.pareto_F[:, :2] = (scores - 1)[:, None]
    return result, model, observed, peak_filter


@pytest.mark.parametrize("with_objective", [False, True])
def test_near_tie_keeps_primary_bound_even_when_a_better_peak_candidate_passes_outside(
    with_objective, tmp_path,
):
    result, model, observed, peak_filter = near_tie_case(with_objective=with_objective)
    before = model.predict()
    original_x, original_f = result.pareto_X.copy(), result.pareto_F.copy()
    with pytest.warns(RuntimeWarning, match="does not expand the shortlist"):
        index, table = select_peak_filtered_pareto(
            result, model, observed, peak_filter=peak_filter, max_error=.05,
            near_tie_allowance=.02,
        )
    assert index == 1
    np.testing.assert_allclose(table.primary_score, [0., .02, .04, .98, 1.])
    assert table.near_tie_eligible.tolist() == [True, True, False, False, False]
    assert table.primary_baseline.tolist() == [True, False, False, False, False]
    assert table.attrs["primary_objective_names"] == ["ccc", "nash_sutcliffe"]
    assert table.at[2, "filter_pass"] and not table.at[2, "near_tie_eligible"]
    assert table.at[index, "selection_status"] == "near_tie_peak_failed"
    assert not table.at[index, "filter_pass"]
    assert table.at[index, "filter_peak_error"] == pytest.approx(.1)
    np.testing.assert_array_equal(result.pareto_X, original_x)
    np.testing.assert_array_equal(result.pareto_F, original_f)
    pd.testing.assert_frame_equal(model.predict(), before)
    table.to_csv(tmp_path / "near_tie_front.csv", index=False)
    saved = pd.read_csv(tmp_path / "near_tie_front.csv")
    assert saved.at[index, "near_tie_eligible"]
    assert saved.at[index, "primary_score_increase"] == pytest.approx(.02)
    assert saved.at[index, "selection_status"] == "near_tie_peak_failed"
    changed = observed.copy()
    changed[12:] = np.nan
    with pytest.warns(RuntimeWarning, match="does not expand the shortlist"):
        other_index, other_table = select_peak_filtered_pareto(
            result, model, changed, peak_filter=peak_filter, max_error=.05,
            near_tie_allowance=.02,
        )
    assert other_index == index
    pd.testing.assert_frame_equal(other_table, table)


def test_near_tie_does_not_fall_back_to_global_peak_minimum_when_every_candidate_fails():
    result, model, observed, peak_filter = near_tie_case(means=(.5, .7, .95, 1.4, 1.5))
    with pytest.warns(RuntimeWarning, match="Selected near-tie candidate"):
        index, table = select_peak_filtered_pareto(
            result, model, observed, peak_filter=peak_filter, max_error=.01,
            near_tie_allowance=.02,
        )
    assert index == 1 and not table.filter_pass.any()
    assert table.filter_peak_error.idxmin() == 2
    assert not table.at[2, "near_tie_eligible"]


@pytest.mark.parametrize("candidate_score,expected", [(.02, 1), (.020001, 0)])
def test_near_tie_allowance_is_inclusive_and_does_not_renormalize_the_shortlist(
    candidate_score, expected,
):
    result, model, observed, peak_filter = near_tie_case()
    result.pareto_F[1, :2] = candidate_score - 1
    index, table = select_peak_filtered_pareto(
        result, model, observed, peak_filter=peak_filter, max_error=1.,
        near_tie_allowance=.02,
    )
    assert index == expected
    assert table.at[index, "selection_status"] == "near_tie_peak_pass"


def test_zero_near_tie_allowance_retains_the_best_primary_candidate():
    result, model, observed, peak_filter = near_tie_case()
    index, table = select_peak_filtered_pareto(
        result, model, observed, peak_filter=peak_filter, max_error=1.,
        near_tie_allowance=0.,
    )
    assert index == 0
    assert table.near_tie_eligible.tolist() == [True, False, False, False, False]


@pytest.mark.parametrize("scores,expected", [
    ([.01, 0., .01, 1., 1.], 1),
    ([0., 0., .01, 1., 1.], 0),
])
def test_near_tie_equal_peak_errors_prefer_primary_score_then_stable_order(scores, expected):
    result, model, observed, peak_filter = near_tie_case(means=(.9, .9, .9, 1.5, 1.5))
    result.pareto_F[:, :2] = (np.asarray(scores) - 1)[:, None]
    index, _ = select_peak_filtered_pareto(
        result, model, observed, peak_filter=peak_filter, max_error=.1,
        near_tie_allowance=.02,
    )
    assert index == expected


@pytest.mark.parametrize("columns", [[2], [1], [2, 0], [1, 2, 0]])
def test_near_tie_supports_objective_subsets_order_and_con15_only(columns):
    result, model, observed, peak_filter = near_tie_case()
    result = CalibrationResult(
        [], result.pareto_X, result.pareto_F[:, columns], result.param_names,
        [result.objective_names[i] for i in columns],
        [result.minimize_flags[i] for i in columns],
    )
    index, table = select_peak_filtered_pareto(
        result, model, observed, peak_filter=peak_filter, max_error=1.,
        near_tie_allowance=.02,
    )
    primary_names = [name for name in result.objective_names if name != "con15_error"]
    primary_names = primary_names or result.objective_names
    assert table.attrs["primary_objective_names"] == primary_names
    assert table.at[index, "primary_score_increase"] <= .02 + 1e-12
    assert table.at[index, "filter_peak_error"] == table.loc[table.near_tie_eligible, "filter_peak_error"].min()


def test_near_tie_minimizing_objectives_and_constant_dimensions_use_rms_score():
    result, model, observed, peak_filter = near_tie_case(with_objective=False)
    losses = np.array([[0., 1., 4.], [1., 0., 4.], [.7, .7, 4.], [.8, .8, 4.], [.9, .9, 4.]])
    result = CalibrationResult(
        [], result.pareto_X, losses, result.param_names,
        ["ccc", "peak_weighted_mse", "nash_sutcliffe"], [False, True, False],
    )
    index, table = select_peak_filtered_pareto(
        result, model, observed, peak_filter=peak_filter, max_error=1.,
        near_tie_allowance=.02,
    )
    expected = np.sqrt(np.mean(losses[:, :2] ** 2, axis=1) * 2 / 3)
    np.testing.assert_allclose(table.primary_score, expected)
    assert index == 2


def test_near_tie_excludes_infeasible_and_nonfinite_rows_from_normalization():
    result, model, observed, peak_filter = near_tie_case()
    result.pareto_X[3] = np.nan
    result.pareto_X[4] = 3.
    result.pareto_F[3:] = -1e9
    index, table = select_peak_filtered_pareto(
        result, model, observed, peak_filter=peak_filter, max_error=1.,
        near_tie_allowance=.02,
    )
    assert index == 0 and len(table) == 5
    np.testing.assert_allclose(table.primary_score[:3], [0., .5, 1.])
    assert not table.model_feasible[3:].any()
    assert table.primary_score[3:].isna().all()
    assert not table.near_tie_eligible[3:].any()
    result.pareto_X[:] = 3.
    with pytest.raises(RuntimeError, match="No feasible"):
        select_peak_filtered_pareto(
            result, model, observed, peak_filter=peak_filter, near_tie_allowance=.02,
        )


def test_near_tie_excludes_nonfinite_objective_rows():
    result, model, observed, peak_filter = near_tie_case()
    result.pareto_F[4] = np.inf
    index, table = select_peak_filtered_pareto(
        result, model, observed, peak_filter=peak_filter, max_error=1.,
        near_tie_allowance=.02,
    )
    assert index == 0
    assert not table.at[4, "model_feasible"]
    assert not table.at[4, "filter_evaluated"]
    assert not table.at[4, "near_tie_eligible"]
    assert pd.isna(table.at[4, "primary_score"])
    assert table.at[1, "primary_score"] == pytest.approx(.02 / .98)


def test_near_tie_propagates_simulation_errors(monkeypatch):
    result, model, observed, peak_filter = near_tie_case()

    def broken_prediction(*args, **kwargs):
        raise RuntimeError("Explicit simulation failure")

    monkeypatch.setattr(PatternWholeFlowModel, "predict_total_cfs", broken_prediction)
    with pytest.raises(RuntimeError, match="Explicit simulation failure"):
        select_peak_filtered_pareto(
            result, model, observed, peak_filter=peak_filter, near_tie_allowance=.02,
        )


@pytest.mark.parametrize("allowance", [-.01, 1.01, np.nan, np.inf])
def test_near_tie_rejects_invalid_allowance(allowance):
    result, model, observed, peak_filter = near_tie_case()
    with pytest.raises(ValueError, match="near_tie_allowance"):
        select_peak_filtered_pareto(
            result, model, observed, peak_filter=peak_filter, near_tie_allowance=allowance,
        )


def test_no_pass_filter_warns_and_exports_failing_lowest_error_fallback(tmp_path):
    result, model, observed, peak_filter = peak_filter_case(means=(.5, .8, 1.3))
    with pytest.warns(RuntimeWarning, match="FAILING fallback"):
        index, table = select_peak_filtered_pareto(
            result, model, observed, peak_filter=peak_filter, max_error=.05,
        )
    assert index == 1 and table.selected.sum() == 1
    assert not table.filter_pass.any()
    assert table.loc[index, "selection_status"] == "fallback_filter_failed"
    assert not table.loc[index, "filter_pass"]
    assert table.loc[index, "filter_peak_error"] == pytest.approx(.2)
    assert len(table) == 3
    table.to_csv(tmp_path / "pareto_solutions.csv", index=False)
    saved = pd.read_csv(tmp_path / "pareto_solutions.csv")
    assert saved.loc[index, "selection_status"] == "fallback_filter_failed"
    assert not saved.loc[index, "filter_pass"] and saved.loc[index, "selected"]


def test_filter_keeps_duplicate_final_vectors_and_zero_tolerance_passes():
    result, model, observed, peak_filter = peak_filter_case(means=(.8, 1., 1., 1.2))
    index, table = select_peak_filtered_pareto(
        result, model, observed, peak_filter=peak_filter, max_error=0,
    )
    assert index in (1, 2)
    assert table.filter_pass.tolist() == [False, True, True, False]
    assert len(table) == 4 and table.pareto_index.is_unique
    assert table.sanitary_mean_cfs.tolist().count(1.) == 2


def test_filter_rejects_infeasible_candidates_without_dropping_their_parameter_rows():
    result, model, observed, peak_filter = peak_filter_case(means=(3., .9, 1.))
    index, table = select_peak_filtered_pareto(
        result, model, observed, peak_filter=peak_filter, max_error=.1,
    )
    assert index != 0 and len(table) == 3
    assert not table.loc[0, "model_feasible"] and not table.loc[0, "filter_evaluated"]
    assert pd.isna(table.loc[0, "filter_pass"])
    result.pareto_X[:] = 3.
    with pytest.raises(RuntimeError, match="No feasible"):
        select_peak_filtered_pareto(result, model, observed, peak_filter=peak_filter)


@pytest.mark.parametrize("tolerance", [-.01, np.nan, np.inf])
def test_filter_validates_tolerance_and_propagates_simulation_errors(tolerance, monkeypatch):
    result, model, observed, peak_filter = peak_filter_case()
    with pytest.raises(ValueError, match="max_error"):
        select_peak_filtered_pareto(result, model, observed, peak_filter=peak_filter, max_error=tolerance)

    def broken_prediction(*args, **kwargs):
        raise RuntimeError("Explicit simulation failure")

    monkeypatch.setattr(PatternWholeFlowModel, "predict_total_cfs", broken_prediction)
    with pytest.raises(RuntimeError, match="Explicit simulation failure"):
        select_peak_filtered_pareto(result, model, observed, peak_filter=peak_filter)














@pytest.mark.parametrize("use_objective,use_filter", [(False, False), (True, False), (False, True)])
def test_notebook_only_requires_training_events_for_enabled_peak_controls(
    notebook_sources, use_objective, use_filter,
):
    import ast

    namespace = {}
    exec(notebook_sources["pattern-joint-00"], namespace)
    namespace.update(
        np=np, pd=pd, Con15=Con15,
        NSGA2_SETTINGS={"pop_size": 12, "generations": 4, "seed": 42},
        SCORING_MODE="multiobjective", EVENT_SOURCE="auto",
        MULTIOBJECTIVE_OBJECTIVES=["ccc", "nash_sutcliffe"] + (["con15_error"] if use_objective else []),
        USE_TRAINING_PEAK_FILTER=use_filter, CON15_NEAR_TIE_ALLOWANCE=.02,
        CON15_OBJ_FRACTION=.3, CON15_FILTER_FRACTION=.15, MAX_TRAINING_PEAK_ERROR=.1,
        acceptance_events=pd.DataFrame(columns=["event_id", "start_idx", "end_idx", "split", "duration_hours"]),
        _dt_hours=1 / 12, msdgc_settings=MSDGCSettings(),
        train_mask=np.ones(10, bool), model_df=pd.DataFrame({"flow_cfs": np.ones(10)}),
        ConcordanceCorrelationCoefficient=ConcordanceCorrelationCoefficient,
        NashSutcliffe=NashSutcliffe, storm_mask=np.ones(10, bool),
    )
    if use_objective or use_filter:
        with pytest.raises(ValueError, match="nonempty"):
            objective_state(namespace)
    else:
        objective_state(namespace)
        assert namespace["peak_objective"] is None and namespace["peak_filter"] is None
        assert len(namespace["calibration_objectives"]) == 2
        assert namespace["peak_objective_events"].empty


@pytest.mark.parametrize("overrides,message", [
    ({"SCORING_MODE": "unsupported"}, "scoring_mode"),
    ({"MULTIOBJECTIVE_OBJECTIVES": []}, "nonempty"),
    ({"MULTIOBJECTIVE_OBJECTIVES": ["ccc", "ccc"]}, "unique"),
    ({"MULTIOBJECTIVE_OBJECTIVES": ["unknown"]}, "objectives"),
    ({"MULTIOBJECTIVE_OBJECTIVES": "ccc"}, "objectives"),
    ({"SCORING_MODE": "single", "SINGLE_OBJECTIVE": "unknown"}, "objectives"),
])
def test_notebook_rejects_invalid_solver_objective_configuration(
    notebook_sources, overrides, message,
):
    namespace = {}
    exec(notebook_sources["pattern-joint-00"], namespace)
    namespace["SCORING_MODE"] = "multiobjective"
    namespace.update(overrides)
    with pytest.raises(ValueError, match=message):
        configure_state(namespace)


@pytest.mark.parametrize("allowance", [-.01, 1.01, np.nan, np.inf])
def test_notebook_rejects_invalid_near_tie_allowance_before_optimization(notebook_sources, allowance):
    namespace = dict(
        np=np, pd=pd, Con15=Con15,
        ConcordanceCorrelationCoefficient=ConcordanceCorrelationCoefficient,
        NashSutcliffe=NashSutcliffe,
    )
    exec(notebook_sources["pattern-joint-00"], namespace)
    namespace.update(
        SCORING_MODE="multiobjective", MULTIOBJECTIVE_OBJECTIVES=["ccc", "nash_sutcliffe"],
        USE_TRAINING_PEAK_FILTER=True, CON15_NEAR_TIE_ALLOWANCE=allowance,
        _dt_hours=1 / 12, msdgc_settings=MSDGCSettings(),
        train_mask=np.ones(10, bool), storm_mask=np.ones(10, bool),
        model_df=pd.DataFrame({"flow_cfs": np.arange(10) + 1.}),
        acceptance_events=pd.DataFrame({
            "event_id": ["complete"], "start_idx": [1], "end_idx": [4],
            "split": ["train"], "duration_hours": [4.],
        }),
    )
    with pytest.raises(ValueError, match="CON15_NEAR_TIE_ALLOWANCE"):
        objective_state(namespace)


def test_notebook_rejects_mislabeled_training_peak_events_before_solver(
    notebook_sources, monkeypatch,
):
    def no_solver(*args, **kwargs):
        raise AssertionError("Invalid training windows must fail before optimization")

    monkeypatch.setattr(PlatypusSolver, "solve", no_solver)
    namespace = dict(
        SCORING_MODE="multiobjective",
        USE_TRAINING_PEAK_FILTER=False,
        MULTIOBJECTIVE_OBJECTIVES=["ccc", "nash_sutcliffe"],
        acceptance_events=pd.DataFrame({
            "event_id": ["crossing"], "start_idx": [1], "end_idx": [4], "split": ["train"],
        }),
        train_mask=np.array([True, True, True, False, False]),
        model_df=pd.DataFrame({"flow_cfs": np.ones(5)}),
    )
    with pytest.raises(ValueError, match="wholly inside train_mask"):
        objective_state(namespace)


@pytest.mark.parametrize("control", ["disabled", "objective", "preference"])
def test_notebook_coarse_grid_diagnostics_do_not_enable_unused_con15(
    notebook_sources, control, capsys,
):
    import ast

    namespace = dict(np=np, pd=pd)
    exec(notebook_sources["pattern-joint-00"], namespace)
    mask = np.ones(10, bool)
    namespace.update(
        msdgc_settings=MSDGCSettings(), _dt_hours=1., EVENT_SOURCE="file",
        model_df=pd.DataFrame({"flow_cfs": np.ones(10)}),
        train_mask=mask, storm_mask=mask,
        acceptance_events=pd.DataFrame(dict(
            event_id=["1"], start_idx=[1], end_idx=[5], duration_hours=[4.], split=["train"])),
        MULTIOBJECTIVE_OBJECTIVES=["ccc"] + (["con15_error"] if control == "objective" else []),
        USE_TRAINING_PEAK_FILTER=control == "preference",
        ConcordanceCorrelationCoefficient=ConcordanceCorrelationCoefficient,
        NashSutcliffe=NashSutcliffe,
    )
    if control != "disabled":
        with pytest.raises(ValueError, match="zero native samples"):
            objective_state(namespace)
    else:
        objective_state(namespace)
        audit = namespace["peak_objective_events"]
        assert audit.observed_conveyance_volume_mg.isna().all()
        assert audit.observed_conveyance_reason.str.contains("zero native samples").all()
        assert "Undefined observed conveyance-volume diagnostics" in capsys.readouterr().out
        assert [o.name for o in namespace["calibration_objectives"]] == ["ccc"]




def test_notebook_requires_actual_compiled_numba_signature(notebook_sources, monkeypatch):
    import time
    from types import SimpleNamespace

    ns = dict(np=np, time=time, phase_seconds={})
    exec(notebook_sources["pattern-joint-00"], ns)
    ns.update(odwf_cfs=.8, _dt_hours=1.0, SEASONAL_BASELINE="off", sanitary_pattern=make_model().pattern,
              model_df=study_data(days=10))
    configure_state(ns)
    monkeypatch.setattr("sparsehydro.calibration.pattern_workflow._gwi_B_core",
                        SimpleNamespace(nopython_signatures=(), targetoptions={}))
    with pytest.raises(RuntimeError, match="did not compile"):
        build_model_state(ns)
