from tests.pattern_workflow_support import (
    configure_state, build_model_state, prepare_data_state, event_coverage_state,
    objective_state, calibrate_state, report_state, assessment_state,
)

import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sparsehydro.calibration.objectives import NashSutcliffe
from sparsehydro.calibration.wholeflow_workflow import FeasibleCalibrationProblem
from sparsehydro.configuration import CalibrationSettings, GROUNDWATER_PARAMETERS, ModelSettings, ParameterSettings, StudySettings
from sparsehydro.models.wholeflow_pattern import CenteredWeeklyPattern, PatternWholeFlowModel


PATTERN = CenteredWeeklyPattern((0.0,) * 24, (0.0,) * 7)


def forcing(dt=1.0):
    n = round(7 * 24 / dt)
    rain = np.zeros(n)
    rain[::round(48 / dt)] = .8
    temp = np.full(n, 12.0)
    temp[:round(24 / dt)] = -4
    return pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=n, freq=pd.Timedelta(hours=dt)),
        "rainfall_in": rain, "temperature_c": temp,
    })


def build(settings=StudySettings(model=ModelSettings(seasonal_baseline="off")), dt=1.0):
    return settings.build_model(PATTERN, 20.26, dt, .8, 2.0)


@pytest.mark.parametrize("groundwater", [False, True])
@pytest.mark.parametrize("n_triangles", [1, 2, 3, 4, 5, 12])
@pytest.mark.parametrize("snow", ["off", "fixed", "calibrated"])
@pytest.mark.parametrize("dt", [1 / 12, .25, 1.0])
def test_module_combinations_preserve_components_and_rainfall_budget(groundwater, n_triangles, snow, dt):
    settings = StudySettings(model=ModelSettings(
        groundwater_enabled=groundwater, n_triangles=n_triangles, seasonal_baseline="off",
        snow_enabled=snow != "off", calibrate_snow=snow == "calibrated"))
    model = build(settings, dt)
    model.prepare(forcing(dt))
    result = model.predict()
    np.testing.assert_array_equal(result.total_cfs, model.predict_total_cfs())
    np.testing.assert_allclose(result.total_cfs, result.sanitary_cfs + result.gwi_cfs + result.rdii_cfs)
    columns = [f"rtk{i}_cfs" for i in range(1, n_triangles + 1)]
    assert {c for c in result if c.startswith("rtk")} == set(columns)
    assert set(columns) <= set(model.output_field_names)
    np.testing.assert_allclose(result[columns].sum(axis=1), result.rdii_cfs, atol=1e-12)
    assert model.budget["max_abs_error_in"] < 1e-8
    assert (result.S_in > 0).any() and (result.et_in > 0).any()
    assert len(model.inequality_constraints()) == (4 if groundwater else 2)
    if groundwater:
        assert result.gwi_cfs.max() > 0
    else:
        for column in ("gwi_cfs", "G_in", "G_at_cap", "gwi_release_in", "gwi_spill_in"):
            np.testing.assert_array_equal(result[column], 0)
        assert all(not model.get_scalar_parameter(n).calibrate for n in GROUNDWATER_PARAMETERS)
        np.testing.assert_array_equal(result.baseflow_cfs, result.sanitary_cfs)


def test_existing_constructor_defaults_are_numerically_unchanged():
    legacy = PatternWholeFlowModel(PATTERN, 20.26, 1., .8, 2.)
    legacy.initialize()
    explicit = build(StudySettings(model=ModelSettings(groundwater_enabled=True)))
    assert legacy.scalar_parameter_names == explicit.scalar_parameter_names
    for model in (legacy, explicit):
        model.prepare(forcing())
    pd.testing.assert_frame_equal(legacy.predict(), explicit.predict(), check_exact=True)


def test_current_model_bypasses_retired_sanitary_construction(monkeypatch):
    from sparsehydro.models import wholeflow

    def retired(*args, **kwargs):
        raise AssertionError("Current construction used the retired sanitary fitter")

    monkeypatch.setattr(wholeflow, "_SanitaryTerm", retired)
    model = build()
    model.prepare(forcing())
    assert model.validate()
    assert np.isfinite(model.predict_total_cfs()).all()


@pytest.mark.parametrize("n_triangles", [4, 5, 12])
def test_all_pathways_participate_in_sum_r_constraint(n_triangles):
    model = build(StudySettings(model=ModelSettings(n_triangles=n_triangles, seasonal_baseline="off")))
    for i in range(1, n_triangles + 1):
        model.get_scalar_parameter(f"R_{i}").value = .3
    assert model.inequality_constraints()[0] == pytest.approx(.3 * n_triangles - 1)
    assert not model.validate()
    model.prepare(forcing())
    with pytest.raises(ValueError, match="feasible"):
        model.predict()


@pytest.mark.parametrize("n_triangles", [1, 2, 3, 4])
def test_legacy_pathway_starts_and_bounds_are_unchanged(n_triangles):
    model = build(StudySettings(model=ModelSettings(n_triangles=n_triangles, seasonal_baseline="off")))
    expected = [
        (.03, 1., 2., .1, 6.), (.02, 8., 2.5, 2., 48.),
        (.01, 48., 3., 12., 120.), (.005, 168., 3., 96., 720.),
    ]
    for i, (R, T, K, lo, hi) in enumerate(expected[:n_triangles], 1):
        parameters = [model.get_scalar_parameter(name) for name in (f"R_{i}", f"uh{i}_T", f"uh{i}_K")]
        assert [p.value for p in parameters] == [R, T, K]
        assert [(p.lower_bound, p.upper_bound) for p in parameters] == [(0., 1.), (lo, hi), (1.05, 4.)]


@pytest.mark.parametrize("n_triangles", [5, 12, 64])
def test_additional_pathway_defaults_prediction_and_round_trip(n_triangles):
    settings = StudySettings(model=ModelSettings(n_triangles=n_triangles, seasonal_baseline="off"))
    model = build(settings)
    assert model.n_triangles == n_triangles
    assert model.validate()
    extra_R = [model.get_scalar_parameter(f"R_{i}").value for i in range(5, n_triangles + 1)]
    np.testing.assert_allclose(extra_R, .005 / (n_triangles - 4))
    assert sum(extra_R) == pytest.approx(.005)
    extra_T = [model.get_scalar_parameter(f"uh{i}_T") for i in range(5, n_triangles + 1)]
    assert all(96. <= p.value <= 720. for p in extra_T)
    assert all((p.lower_bound, p.upper_bound) == (96., 720.) for p in extra_T)
    assert np.all(np.diff([p.value for p in extra_T]) >= 0)
    model.prepare(forcing())
    result = model.predict()
    columns = [f"rtk{i}_cfs" for i in range(1, n_triangles + 1)]
    assert {c for c in result if c.startswith("rtk")} == set(columns)
    np.testing.assert_allclose(result[columns].sum(axis=1), result.rdii_cfs, atol=1e-12)
    np.testing.assert_array_equal(result.total_cfs, model.predict_total_cfs())
    assert model.budget["max_abs_error_in"] < 1e-8
    restored = build(StudySettings.from_dict(settings.to_dict()))
    restored.prepare(forcing())
    pd.testing.assert_frame_equal(result, restored.predict(), check_exact=True)


def test_additional_pathway_overrides_reach_calibration_and_prediction():
    settings = StudySettings(
        model=ModelSettings(n_triangles=12, seasonal_baseline="off"),
        parameters={
            "R_12": ParameterSettings(value=.02),
            "uh12_T": ParameterSettings(value=800., upper_bound=1000.),
            "uh12_K": ParameterSettings(value=3.5, calibrate=False),
        },
    )
    model = build(settings)
    data = forcing().assign(flow_cfs=1.)
    problem = FeasibleCalibrationProblem(
        model, data, [NashSutcliffe()], column_map={"observed": "flow_cfs", "predicted": "total_cfs"})
    assert {"R_12", "uh12_T"} <= set(problem.param_names)
    assert "uh12_K" not in problem.param_names
    assert model.get_scalar_parameter("uh12_T").value == 800.
    before = model.predict()
    assert before.rtk12_cfs.max() > 0
    model.get_scalar_parameter("R_12").value = 0.
    after = model.predict()
    np.testing.assert_allclose(before.total_cfs - before.rtk12_cfs, after.total_cfs, atol=1e-12)
    np.testing.assert_array_equal(after.rtk12_cfs, 0.)


@pytest.mark.parametrize("n_triangles", [0, -1, True, 2.5, "5", None])
def test_direct_pattern_constructor_rejects_invalid_triangle_counts(n_triangles):
    with pytest.raises(ValueError, match="positive integer"):
        PatternWholeFlowModel(PATTERN, 20.26, 1., .8, 2., n_triangles=n_triangles)


def test_wholeflow_constructor_rejects_empty_triangle_definitions():
    from sparsehydro.models.wholeflow import WholeFlowModel

    with pytest.raises(ValueError, match="at least one RTK triangle"):
        WholeFlowModel(np.ones(2), 20.26, 1., [], sanitary_mode="offset")


def test_parameter_selection_and_fixed_values_reach_calibration_problem():
    settings = StudySettings(model=ModelSettings(seasonal_baseline="off"), parameters={
        "ia_T_ref": ParameterSettings(value=12., calibrate=True),
        "sanitary_hour_scale": ParameterSettings(value=.5, calibrate=False),
        "s_initial_fraction": ParameterSettings(value=.25, calibrate=True),
        "area_acres": ParameterSettings(value=25., lower_bound=20., upper_bound=30., calibrate=True),
    })
    model = build(settings)
    data = forcing().assign(flow_cfs=np.linspace(.8, 1.2, len(forcing())))
    problem = FeasibleCalibrationProblem(model, data, [NashSutcliffe()],
        column_map={"observed": "flow_cfs", "predicted": "total_cfs"})
    assert {"ia_T_ref", "s_initial_fraction", "area_acres"} <= set(problem.param_names)
    assert "sanitary_hour_scale" not in problem.param_names
    assert not set(GROUNDWATER_PARAMETERS) & set(problem.param_names)
    assert model.get_scalar_parameter("area_acres").value == 25
    assert model.get_scalar_parameter("sanitary_hour_scale").value == .5


@pytest.mark.parametrize("name", ["k_div", "k_G", "g_max", "g_initial_fraction", "R_4", "snow_T", "typo"])
def test_disabled_and_unknown_parameter_overrides_are_rejected(name):
    with pytest.raises(ValueError, match="disabled|not registered"):
        build(StudySettings(model=ModelSettings(seasonal_baseline="off"),
                            parameters={name: ParameterSettings(calibrate=True)}))


@pytest.mark.parametrize("kwargs", [
    {"n_triangles": 0}, {"n_triangles": -1}, {"n_triangles": True}, {"n_triangles": 2.5},
    {"n_triangles": "5"}, {"n_triangles": None},
    {"groundwater_enabled": "False"}, {"snow_enabled": 1}, {"calibrate_snow": True},
    {"snow_melt_factor": -1}, {"snow_threshold_c": float("nan")},
])
def test_invalid_module_settings_fail_explicitly(kwargs):
    with pytest.raises(ValueError):
        ModelSettings(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"solver": "typo"}, {"scoring_mode": "typo"}, {"objectives": ()},
    {"objectives": ("ccc", "ccc")}, {"objectives": ("typo",)},
    {"solver": "cmaes"}, {"solver": "differential_evolution"},
    {"scoring_mode": "single"}, {"population": True}, {"workers": 0},
    {"scoring_mode": "single", "objectives": ("ccc",), "cma_trials": 2, "cma_startup_trials": 2},
])
def test_invalid_solver_and_objective_settings_fail_explicitly(kwargs):
    with pytest.raises(ValueError):
        CalibrationSettings(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"value": np.nan}, {"lower_bound": np.inf}, {"value": "1"}, {"calibrate": 1},
    {"value": -1}, {"lower_bound": 2, "upper_bound": 1},
    {"lower_bound": .8, "upper_bound": .8, "calibrate": True},
])
def test_invalid_parameter_overrides_fail_explicitly(kwargs):
    with pytest.raises(ValueError):
        build(StudySettings(model=ModelSettings(seasonal_baseline="off"),
                            parameters={"sanitary_mean_cfs": ParameterSettings(**kwargs)}))


def test_serialization_round_trip_and_unknown_settings():
    settings = StudySettings(
        model=ModelSettings(groundwater_enabled=True, n_triangles=4),
        calibration=CalibrationSettings(solver="spea2"),
        parameters={"ia_T_ref": ParameterSettings(calibrate=True)},
        workflow={"events": {"EVENT_SOURCE": "file", "EVENT_FILE": r"C:\data\events.evt",
                             "ASSESSMENT_EXCLUDED_EVENT_IDS": ("87",)},
                  "holdout": {"VAL_EVERY": 4}})
    payload = json.loads(json.dumps(settings.to_dict()))
    assert payload["calibration"]["objectives"] == ["ccc", "nash_sutcliffe", "peak_weighted_mse"]
    restored = StudySettings.from_dict(payload)
    assert restored == settings
    assert restored.to_dict() == payload
    for model in (build(settings), build(restored)):
        model.prepare(forcing())
        assert model.predict().budget_error_in.abs().max() < 1e-8
    with pytest.raises(TypeError):
        StudySettings.from_dict({**payload, "typo": 1})
    with pytest.raises(TypeError):
        StudySettings.from_dict({**payload, "model": {"typo": True}})
    with pytest.raises(ValueError, match="schema"):
        StudySettings.from_dict({**payload, "schema_version": 4})
    with pytest.raises(ValueError):
        StudySettings(workflow={"data": {"invalid": float("nan")}})


def test_objective_factory_preserves_masks_order_and_rejects_leakage():
    train = np.array([True, True, True, False])
    storm = np.array([True, False, True, False])
    config = CalibrationSettings(objectives=("pbias", "ccc", "peak_weighted_mse", "nash_sutcliffe"))
    objectives = config.build_objectives(train, storm)
    assert [o.name for o in objectives] == list(config.objectives)
    assert [o.minimize for o in objectives] == [True, False, True, False]
    np.testing.assert_array_equal(objectives[0].mask, train)
    np.testing.assert_array_equal(objectives[1].mask, storm)
    with pytest.raises(ValueError, match="training-only"):
        config.build_objectives(train, np.ones(4, bool))
    with pytest.raises(ValueError, match="aligned"):
        config.build_objectives(train, storm.astype(int))
    with pytest.raises(ValueError, match="explicit"):
        CalibrationSettings(objectives=("con15_error",)).build_objectives(train, storm)


@pytest.mark.parametrize("method", ["nsga2", "spea2", "cmaes", "differential_evolution"])
@pytest.mark.parametrize("n_triangles", [4, 6])
def test_each_solver_executes_a_configured_core_model(method, n_triangles):
    config = CalibrationSettings(
        solver=method, scoring_mode="multiobjective" if method in ("nsga2", "spea2") else "single",
        objectives=("ccc", "nash_sutcliffe") if method in ("nsga2", "spea2") else ("nash_sutcliffe",),
        population=8, generations=2, workers=2, cma_trials=8, cma_startup_trials=2,
        cma_starts=1, cma_jobs=1, cma_progress_every=0, de_maxiter=1, de_popsize=3)
    model = build(StudySettings(model=ModelSettings(n_triangles=n_triangles, seasonal_baseline="off"), calibration=config))
    for name in model.scalar_parameter_names:
        model.get_scalar_parameter(name).calibrate = name in ("sanitary_mean_cfs", f"R_{n_triangles}")
    model.prepare(forcing())
    data = forcing().assign(flow_cfs=model.predict_total_cfs() * 1.03)
    train = np.ones(len(data), bool)
    problem = FeasibleCalibrationProblem(
        model, data, config.build_objectives(train, train),
        column_map={"observed": "flow_cfs", "predicted": lambda values: values},
        predict_fn=PatternWholeFlowModel.predict_total_cfs)
    result = config.build_solver().solve(problem)
    assert problem.n_params == 2
    assert result.pareto_X.shape[1] == 2
    assert result.pareto_F.shape[1] == len(config.objectives)
    assert np.isfinite(result.pareto_F).all()
    for x, f in zip(result.pareto_X, result.pareto_F):
        np.testing.assert_allclose(problem.evaluate(x), f)
    metadata = config.solver_metadata()
    assert metadata["requested_evaluations"] == (
        16 if method in ("nsga2", "spea2") else 8 if method == "cmaes" else None)


@pytest.fixture(scope="module")
def notebook():
    path = Path(__file__).parents[1] / "sparsehydro" / "notebooks" / "flowfinity_gwi_wholeflow_pattern_joint.ipynb"
    return json.loads(path.read_text(encoding="utf-8"))


def notebook_namespace(notebook):
    ns = dict(np=np, pd=pd, time=time, phase_seconds={})
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code" and (
                cell["id"] == "pattern-joint-00" or cell["id"].startswith("pattern-settings-")):
            exec("".join(cell["source"]), ns)
    return ns


@pytest.mark.parametrize("groundwater", [False, True])
@pytest.mark.parametrize("n_triangles", [1, 2, 3, 4, 5, 12])
def test_notebook_pages_wire_modules_bounds_values_and_parameter_switches(notebook, groundwater, n_triangles):
    sources = {c["id"]: "".join(c["source"]) for c in notebook["cells"]}
    ns = notebook_namespace(notebook)
    ns.update(
        GROUNDWATER_ENABLED=groundwater, SEASONAL_BASELINE="off", N_RTK_TRIANGLES=n_triangles, odwf_cfs=.8,
        _dt_hours=1., model_df=forcing(), sanitary_pattern=PATTERN,
        PARAMETER_OVERRIDES={"ia_T_ref": {"value": 12., "calibrate": True},
                             "ia_kT": {"value": .02, "calibrate": False}},
    )
    if n_triangles > 4:
        ns["PARAMETER_OVERRIDES"].update({
            f"R_{n_triangles}": {"value": .01},
            f"uh{n_triangles}_T": {"value": 500., "lower_bound": 200., "upper_bound": 900.},
            f"uh{n_triangles}_K": {"value": 3., "calibrate": False},
        })
    build_model_state(ns)
    model = ns["joint_model"]
    assert model.n_triangles == n_triangles
    assert model.groundwater_enabled == groundwater
    assert model.get_scalar_parameter("ia_T_ref").calibrate
    assert not model.get_scalar_parameter("ia_kT").calibrate
    assert model.get_scalar_parameter("R_1").value == ns["R_1_value"] == .08
    if n_triangles >= 4:
        assert model.get_scalar_parameter("uh4_T").value == 168
    if n_triangles > 4:
        assert model.get_scalar_parameter(f"R_{n_triangles}").value == .01
        assert model.get_scalar_parameter(f"uh{n_triangles}_T").value == 500.
        assert not model.get_scalar_parameter(f"uh{n_triangles}_K").calibrate
        assert f"rtk{n_triangles}_cfs" in model.predict()
    payload = ns["study_settings"].to_dict()
    assert payload["workflow"]["events"]["EVENT_SOURCE"] == "hybrid"
    assert payload["workflow"]["events"]["EVENT_FILE"] == ns["EVENT_FILE"]
    restored = StudySettings.from_dict(payload).build_model(
        PATTERN, ns["AREA_ACRES"], ns["_dt_hours"], ns["sanitary_start"], ns["sanitary_upper"])
    restored.prepare(forcing())
    pd.testing.assert_frame_equal(model.predict(), restored.predict())


def test_notebook_snow_bound_overrides_are_applied_before_frozen_masks(notebook):
    sources = {c["id"]: "".join(c["source"]) for c in notebook["cells"]}
    ns = notebook_namespace(notebook)
    ns.update(SNOW_ENABLED=True, CALIBRATE_SNOW=True)
    ns["PARAMETER_OVERRIDES"] = {"snow_T": {"value": 1., "lower_bound": -3., "upper_bound": 4.}}
    configure_state(ns)
    snow = ns["study_settings"].parameters["snow_T"]
    assert (snow.lower_bound, snow.upper_bound) == (-3., 4.)
    assert ns["SNOW_CONFIG"].threshold_c == 1.


def test_notebook_observed_background_is_training_only_and_rejects_gwi(notebook):
    sources = {c["id"]: "".join(c["source"]) for c in notebook["cells"]}
    frame = pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=56 * 24, freq="h"),
        "rainfall_in": 0., "temperature_c": 10., "dry_eligible": True,
    })
    frame["flow_cfs"] = .8 + .001 * np.arange(len(frame)) / 24
    train = np.arange(len(frame)) // (3 * 24) % 4 != 2
    train[:12] = False
    train[-48:] = False
    learned = []
    for perturb in (False, True):
        ns = notebook_namespace(notebook)
        changed = frame.copy()
        if perturb:
            changed.loc[~train, "flow_cfs"] += 10000.
        ns.update(
            GROUNDWATER_ENABLED=False, SEASONAL_BASELINE="observed",
            SNOW_ENABLED=False, CALIBRATE_SNOW=False, model_df=changed, _dt_hours=1.,
            train_mask=train, val_mask=~train,
            scored_mask=np.ones(len(frame), bool), display=lambda _: None,
            plot_sanitary_shapes=lambda pattern: SimpleNamespace(show=lambda: None))
        original_train, original_val, original_scored = train.copy(), (~train).copy(), ns["scored_mask"].copy()
        exec(sources["pattern-joint-27"], ns)
        build_model_state(ns)
        model = ns["joint_model"]
        assert "sanitary_mean_cfs" not in model.scalar_parameter_names
        assert model.predict().background_supported.all()
        held = model.predict().background_endpoint_held.to_numpy()
        assert held[:12].all() and held[-48:].all()
        assert model.observed_background.endpoint_hold_days == ns["BACKGROUND_ENDPOINT_HOLD_DAYS"] == 3
        np.testing.assert_array_equal(ns["train_mask"], original_train)
        np.testing.assert_array_equal(ns["val_mask"], original_val)
        np.testing.assert_array_equal(ns["scored_mask"], original_scored)
        strict = StudySettings.from_dict(ns["study_settings"].to_dict())
        assert strict.model.background_endpoint_hold_days == 3
        learned.append((model.observed_background, model.predict()))
    assert learned[0][0] == learned[1][0]
    pd.testing.assert_frame_equal(learned[0][1], learned[1][1])
    ns["GROUNDWATER_ENABLED"] = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_model_state(ns)


@pytest.mark.parametrize("method", ["spea2", "differential_evolution"])
def test_notebook_alternative_solver_runs_four_pathway_core(notebook, tmp_path, method):
    from sparsehydro.calibration.msdgc_assessment import MSDGCSettings
    from sparsehydro.calibration.objectives import ConcordanceCorrelationCoefficient, PBIAS
    from sparsehydro.models.rdii.initial_abstraction import snow_partition

    sources = {c["id"]: "".join(c["source"]) for c in notebook["cells"]}
    ns = notebook_namespace(notebook)
    template = build(StudySettings(model=ModelSettings(n_triangles=4, seasonal_baseline="off")))
    overrides = {
        name: {"calibrate": name in ("sanitary_mean_cfs", "R_4")}
        for name in template.scalar_parameter_names if name not in GROUNDWATER_PARAMETERS
    }
    data = forcing()
    ns.update(
        GROUNDWATER_ENABLED=False, SEASONAL_BASELINE="off", N_RTK_TRIANGLES=4, SNOW_ENABLED=False, CALIBRATE_SNOW=False,
        PARAMETER_OVERRIDES=overrides, SOLVER=method,
        SCORING_MODE="multiobjective" if method == "spea2" else "single",
        MULTIOBJECTIVE_OBJECTIVES=["ccc", "nash_sutcliffe"], SINGLE_OBJECTIVE="nash_sutcliffe",
        NSGA2_SETTINGS={"pop_size": 8, "generations": 2, "seed": 42},
        DE_SETTINGS={"de_maxiter": 1, "de_popsize": 3, "de_polish": False}, EVENT_SOURCE="auto",
        odwf_cfs=.8, _dt_hours=1., model_df=data, sanitary_pattern=PATTERN,
        ARTIFACT_DIR=tmp_path, ConcordanceCorrelationCoefficient=ConcordanceCorrelationCoefficient,
        NashSutcliffe=NashSutcliffe, PBIAS=PBIAS,
        msdgc_settings=MSDGCSettings(minimum_minutes=(60., 20., 30.),
                                   maximum_minutes=(60., 240., 1440.)),
    )
    build_model_state(ns)
    model = ns["joint_model"]
    data["flow_cfs"] = model.predict_total_cfs() * 1.03
    data["stormflow"] = data.flow_cfs - .6
    for column, values in snow_partition(
            data.rainfall_in.to_numpy(), data.temperature_c.to_numpy(), 1., None).items():
        data[column] = values
    train = np.arange(len(data)) < 120
    ns.update(
        train_mask=train, val_mask=~train, storm_mask=train.copy(),
        acceptance_events=pd.DataFrame({
            "event_id": ["train", "validation"], "start_idx": [48, 144],
            "end_idx": [72, 167], "duration_hours": [24., 23.], "split": ["train", "validation"],
        }),
    )
    calibrate_state(ns)
    assert ns["calibration_config"].resolved_solver == method
    assert ns["joint_result"].pareto_X.shape[1] == 2
    assert ns["N_EVALUATIONS"] == (16 if method == "spea2" else None)
    assert "selected_seed" not in ns["solver_metadata"]
    assert (tmp_path / "pareto_solutions.csv").is_file()
    result = ns["joint_pred"]
    np.testing.assert_array_equal(result.gwi_cfs, 0)
    np.testing.assert_allclose(result.rdii_cfs, result.filter(regex=r"^rtk[1-4]_cfs$").sum(axis=1))
    assert model.budget["max_abs_error_in"] < 1e-8


def test_flowfinity_record_selection_uses_settings_page(notebook):
    from sparsehydro.data.wholeflow import select_flowfinity_record

    ns = notebook_namespace(notebook)
    ns.update(DATA_SOURCE="flowfinity", FLOWFINITY_RECORD_INDEX=1,
              records_df=pd.DataFrame({"DataTitle": ["first", "second"]}))
    def selected():
        return select_flowfinity_record(ns["records_df"], title=ns["FLOWFINITY_RECORD_TITLE"],
                                        index=ns["FLOWFINITY_RECORD_INDEX"])

    assert selected() == "second"
    ns["FLOWFINITY_RECORD_TITLE"] = "first"
    assert selected() == "first"
    ns["FLOWFINITY_RECORD_TITLE"] = "missing"
    with pytest.raises(ValueError, match="discovered"):
        selected()
