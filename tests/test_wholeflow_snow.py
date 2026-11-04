from tests.pattern_workflow_support import (
    configure_state, build_model_state, prepare_data_state, event_coverage_state,
    objective_state, calibrate_state, report_state, assessment_state,
)

import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pandas as pd
import pytest

from sparsehydro.calibration.wholeflow_workflow import dry_weather_mask, melt_scoring_mask, snow_melt_envelope
from sparsehydro.models.rdii.initial_abstraction import SnowConfig, snow_partition
from sparsehydro.models.wholeflow import _gwi_B_core
from sparsehydro.models.wholeflow_pattern import CenteredWeeklyPattern, PatternWholeFlowModel


def make_model(dt=1.0, config=None):
    pattern = CenteredWeeklyPattern((0.0,) * 24, (0.0,) * 7)
    model = PatternWholeFlowModel(pattern, 20.26, dt, 0.5, 2.0, snow_config=config)
    model.initialize()
    return model


def forcing(rain, temp, dt=1.0):
    return pd.DataFrame({
        "datetime": pd.date_range("2025-01-01", periods=len(rain),
                                 freq=pd.Timedelta(hours=dt)),
        "rainfall_in": rain, "temperature_c": temp,
    })


def test_snow_ledger_threshold_melt_cap_and_rain_on_snow():
    p = snow_partition(np.array([0.2, 0.3, 0.1, 0.0]), np.array([-5, 1, 3, 11]),
                       24.0, SnowConfig())
    np.testing.assert_allclose(p.snowfall_in, [0.2, 0.3, 0, 0])
    np.testing.assert_allclose(p.SWE_in, [0.2, 0.5, 0.26, 0])
    np.testing.assert_allclose(p.melt_in, [0, 0, 0.24, 0.26])
    np.testing.assert_allclose(p.liquid_input_in, [0, 0, 0.34, 0.26])
    np.testing.assert_allclose(p.rain_liquid_in + p.melt_in, p.liquid_input_in)
    np.testing.assert_allclose(np.cumsum([0.2, 0.3, 0.1, 0]) - p.liquid_input_in.cumsum(),
                               p.SWE_in, atol=1e-15)


@pytest.mark.parametrize("kwargs", [
    {"threshold_c": np.nan}, {"melt_factor_in_per_c_day": -1},
    {"melt_factor_in_per_c_day": np.inf}, {"initial_swe_in": -1},
])
def test_invalid_snow_settings(kwargs):
    with pytest.raises(ValueError):
        SnowConfig(**kwargs)


@pytest.mark.parametrize("rain,temp,dt", [
    ([-1], [0], 1), ([np.nan], [0], 1), ([0], [np.nan], 1),
    ([0, 1], [0], 1), ([], [], 1), ([0], [0], 0), ([0], [0], np.inf),
])
def test_invalid_snow_forcing(rain, temp, dt):
    with pytest.raises(ValueError):
        snow_partition(np.array(rain), np.array(temp), dt, SnowConfig())


@pytest.mark.parametrize("dt", [1 / 12, .25, 1.0])
def test_snow_groundwater_plumbing_and_complete_budget(dt):
    n = int(120 / dt)
    rain = np.zeros(n)
    rain[1] = .8
    rain[int(54 / dt)] = .2
    temp = np.where(np.arange(n) * dt < 48, -5.0, 5.0)
    data = forcing(rain, temp, dt)
    model = make_model(dt, SnowConfig(initial_swe_in=.1))
    model.prepare(data)
    p = model.predict()
    cold = np.arange(n) * dt < 48
    assert p.loc[cold, "liquid_input_in"].eq(0).all()
    np.testing.assert_allclose(p.loc[cold, ["S_in", "G_in", "gwi_cfs", "rdii_cfs"]], 0,
                               atol=1e-12)
    assert p.loc[~cold, "melt_in"].sum() > 0
    assert p.loc[~cold, "gwi_cfs"].max() > 0
    assert p.loc[~cold, "rdii_cfs"].max() > 0
    assert (p.recharge_in >= 0).all()
    assert (p.recharge_in <= p.liquid_input_in).all()
    assert p.budget_error_in.abs().max() < 1e-8
    b = model.budget
    assert b["initial_storage_in"] + b["rain_in"] == pytest.approx(
        b["et_in"] + b["gwi_in"] + b["rdii_in"] + b["unrouted_in"]
        + b["final_soil_in"] + b["final_gwi_in"] + b["final_routing_in"]
        + b["final_snow_in"], abs=1e-8)
    # Feeding liquid into the legacy no-snow model must give identical S/G/RTK.
    reference = make_model(dt)
    reference.prepare(data.assign(rainfall_in=p.liquid_input_in))
    q = reference.predict()
    for column in ("S_in", "G_in", "excess_in", "et_in", "gwi_cfs", "rdii_cfs", "total_cfs"):
        np.testing.assert_array_equal(p[column], q[column])
    pd.testing.assert_frame_equal(p, model.predict())
    model.prepare(data.assign(flow_cfs=np.nan))
    pd.testing.assert_frame_equal(p, model.predict())
    assert sum(model.get_scalar_parameter(n).calibrate for n in model.scalar_parameter_names) == 20


def test_disabled_snow_is_exact_legacy_core_even_below_freezing():
    data = forcing([.3, .1, 0, 0, .2, 0], [-5, -5, 5, 5, 5, 5])
    model = make_model()
    model.prepare(data)
    p = model.predict()
    get = lambda name: float(model.get_scalar_parameter(name).value)
    s, g, excess, et, gwi = _gwi_B_core(
        data.rainfall_in.to_numpy(), data.temperature_c.to_numpy(), 1.0,
        *[get(n) for n in ("s_max", "ia_k_dep", "ia_k0", "ia_kT", "ia_theta",
                          "ia_T_ref", "ia_T_freeze", "k_div", "k_G", "g_max")])
    for name, values in (("S_in", s), ("G_in", g), ("excess_in", excess), ("et_in", et)):
        np.testing.assert_array_equal(p[name], values)
    np.testing.assert_array_equal(p.gwi_cfs, gwi * (get("area_acres") * 3630 / 3600))
    np.testing.assert_array_equal(p.liquid_input_in, data.rainfall_in)
    assert p[["SWE_in", "melt_in", "snowfall_in"]].eq(0).all().all()


def test_existing_groundwater_releases_while_new_snow_is_stored():
    data = forcing([.1] * 24, [-5] * 24)
    model = make_model(config=SnowConfig())
    model.get_scalar_parameter("g_initial_fraction").value = 1
    model.prepare(data)
    p = model.predict()
    assert (p.gwi_cfs > 0).all()
    assert (p.G_in.diff().dropna() < 0).all()
    assert p.excess_in.eq(0).all()
    assert p.SWE_in.iloc[-1] == pytest.approx(2.4)
    assert p.budget_error_in.abs().max() < 1e-8




def test_melt_only_windows_are_padded_clipped_and_not_dry():
    melt = np.zeros(100)
    melt[30:33] = .001
    split = np.arange(100) >= 31
    actual = melt_scoring_mask(melt, split, 1, pre_hours=3, tail_hours=48)
    expected = np.zeros(100, bool)
    expected[31:81] = True
    np.testing.assert_array_equal(actual, expected)
    dry = dry_weather_mask(pd.DataFrame({"rainfall_in": melt}), 1, 48)
    assert not dry[30:81].any()
    assert dry[81:].all()
    assert not melt_scoring_mask(melt * 0, split, 1).any()
    with pytest.raises(ValueError):
        melt_scoring_mask(melt, split[:-1], 1)
    with pytest.raises(ValueError):
        melt_scoring_mask(-melt, split, 1)


NOTEBOOK = Path(__file__).resolve().parents[1] / "sparsehydro" / "notebooks" / "flowfinity_gwi_wholeflow_pattern_joint.ipynb"


def notebook_cell(cell_id):
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    return "".join(next(c["source"] for c in cells if c["id"] == cell_id))


@pytest.mark.parametrize("calibrate_snow", [False, True])
def test_notebook_spinup_snow_is_continuous_and_masks_are_melt_aware(calibrate_snow):
    from sparsehydro.calibration.splits import SplitConfig, build_masks, prepend_spinup

    rain = np.zeros(240)
    rain[12] = .5
    temp = np.where(np.arange(240) < 30, -5.0, 5.0)
    frame = forcing(rain, temp).assign(flow_cfs=.5)
    cfg = SplitConfig(dt_hours=1, warmup_days=1, spinup_mode="repeat_head",
                      split_mode="continuous", snap_edges_to_dry=False)
    ns = {}
    for cell in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]:
        if cell["cell_type"] == "code" and (
                cell["id"] == "pattern-joint-00" or cell["id"].startswith("pattern-settings-")):
            exec("".join(cell["source"]), ns)
    ns.update(np=np, pd=pd, time=time, model_df=frame, SPLIT_CFG=cfg, DT_HOURS=1,
              TIMESTEP="1h", WARMUP_DAYS=1, SPINUP_MODE="repeat_head",
              SPLIT_MODE="continuous", DRY_EXCLUSION_HOURS=48, SNOW_ENABLED=True,
              snow_T_value=1, snow_ddf_value=.12, phase_seconds={},
              CALIBRATE_SNOW=calibrate_snow, snow_T_lower=-2, snow_T_upper=3,
              snow_ddf_lower=0, snow_ddf_upper=.5, PARAMETER_OVERRIDES={}, EVENT_SOURCE="auto",
              prepend_spinup=prepend_spinup, build_masks=build_masks)
    configure_state(ns)
    prepare_data_state(ns)
    data = ns["model_df"]
    model = make_model(config=ns["SNOW_CONFIG"])
    model.prepare(data)
    p = model.predict()
    for name in ("SWE_in", "melt_in", "liquid_input_in"):
        np.testing.assert_array_equal(data[name], p[name])
    assert p.SWE_in.iloc[24] == pytest.approx(.5)
    assert p.melt_in.iloc[24:54].eq(0).all()
    assert p.SWE_in.iloc[53] == pytest.approx(1)
    assert not data.loc[data.melt_in.gt(0), "dry_eligible"].any()
    ns.update(shape_fit=SimpleNamespace(detection_baseline_cfs=np.full(len(data), .5)),
              dry_eligible=data.dry_eligible.to_numpy(), display=lambda *args: None)
    event_coverage_state(ns)
    assert not (ns["storm_mask"] & ~ns["train_mask"]).any()
    assert not (ns["val_storm_mask"] & ~ns["val_mask"]).any()
    assert ns["storm_mask"][data.melt_in.gt(0) & ns["train_mask"]].all()
    assert not (ns["dry_mask"] & data.melt_in.gt(0)).any()
    if calibrate_snow:
        for threshold, factor in ((-2, .5), (3, .01), (0, .03)):
            candidate = snow_partition(data.rainfall_in.to_numpy(), data.temperature_c.to_numpy(),
                                       1, SnowConfig(threshold, factor))
            actual_melt = candidate.melt_in.gt(0).to_numpy()
            assert not (actual_melt & ~data.possible_melt.to_numpy()).any()
            assert not (actual_melt & data.dry_eligible.to_numpy()).any()
            assert ns["storm_mask"][actual_melt & ns["train_mask"]].all()


@pytest.mark.parametrize("kind", ["compact", "pattern"])
@pytest.mark.parametrize("dt", [1 / 12, .25, 1.0])
def test_bounded_snow_candidates_refresh_without_reprepare(kind, dt, monkeypatch):
    from sparsehydro.models import wholeflow_compact
    from sparsehydro.models.wholeflow_compact import CompactWholeFlowModel, SanitaryTemplate

    def build(config, calibrate):
        if kind == "compact":
            model = CompactWholeFlowModel(
                SanitaryTemplate((1.0,) * 24, (1.0,) * 7), 20.26, dt, .5, 2,
                snow_config=config, calibrate_snow=calibrate)
        else:
            model = PatternWholeFlowModel(
                CenteredWeeklyPattern((0.0,) * 24, (0.0,) * 7), 20.26, dt, .5, 2,
                snow_config=config, calibrate_snow=calibrate)
        model.initialize()
        return model

    config = SnowConfig(threshold_c=0, melt_factor_in_per_c_day=.03, initial_swe_in=.1)
    model = build(config, True)
    assert model.get_scalar_parameter("snow_T").lower_bound == -2
    assert model.get_scalar_parameter("snow_T").upper_bound == 3
    assert model.get_scalar_parameter("snow_ddf").lower_bound == .005
    assert model.get_scalar_parameter("snow_ddf").upper_bound == .5
    calls = []
    partition = wholeflow_compact.snow_partition

    def counted(*args):
        calls.append(args[-1])
        return partition(*args)

    monkeypatch.setattr(wholeflow_compact, "snow_partition", counted)
    n = int(120 / dt)
    rain = np.zeros(n)
    rain[[1, int(25 / dt), int(60 / dt)]] = [.5, .2, .3]
    temp = np.where(np.arange(n) * dt < 24, -4,
                    np.where(np.arange(n) * dt < 48, 1, 6))
    data = forcing(rain, temp, dt)
    model.prepare(data)
    baseline = model.predict()
    assert len(calls) == 1
    np.testing.assert_array_equal(model.predict_total_cfs(), baseline.total_cfs)
    assert len(calls) == 1
    for threshold, factor in ((3, .5), (-2, .005), (0, .03)):
        model.apply_flat_parameter("snow_T", threshold)
        model.apply_flat_parameter("snow_ddf", factor)
        before = len(calls)
        lean = model.predict_total_cfs()
        actual = model.predict()
        assert len(calls) == before + 1
        np.testing.assert_array_equal(lean, actual.total_cfs)
        assert model.snow_config == SnowConfig(threshold, factor, .1)
        assert model.budget["max_abs_error_in"] < 1e-8
        reference = build(model.snow_config, False)
        reference.prepare(data)
        pd.testing.assert_frame_equal(actual, reference.predict(), check_exact=True)
    pd.testing.assert_frame_equal(baseline, model.predict(), check_exact=True)
    model.get_scalar_parameter("snow_T").value = 4
    with pytest.raises(ValueError, match="feasible"):
        model.predict_total_cfs()


def test_snow_calibration_requires_enabled_valid_start():
    pattern = CenteredWeeklyPattern((0.0,) * 24, (0.0,) * 7)
    with pytest.raises(ValueError, match="SnowConfig"):
        PatternWholeFlowModel(pattern, 20, 1, .5, 2, calibrate_snow=True)
    for config in (SnowConfig(threshold_c=4), SnowConfig(melt_factor_in_per_c_day=.6)):
        model = PatternWholeFlowModel(pattern, 20, 1, .5, 2,
                                     snow_config=config, calibrate_snow=True)
        with pytest.raises(ValueError, match="initial value"):
            model.initialize()
        fixed = make_model(config=config)
        fixed.prepare(forcing([.1, 0, 0], [-4, 5, 8]))
        assert fixed.predict().budget_error_in.abs().max() < 1e-8


@pytest.mark.parametrize("dt", [1 / 12, .25, 1.0])
@pytest.mark.parametrize("minimum", [0.0, .02])
def test_snow_envelope_contains_every_sampled_bounded_candidate(dt, minimum):
    rng = np.random.default_rng(42)
    rain = rng.choice([0, 0, 0, .01, .1], 240)
    temp = rng.uniform(-5, 8, 240)
    envelope = snow_melt_envelope(rain, temp, dt, threshold_bounds=(-2, 3),
                                  melt_factor_bounds=(minimum, .5), initial_swe_in=.1)
    for threshold in np.linspace(-2, 3, 11):
        for factor in np.linspace(minimum, .5, 11):
            snow = snow_partition(rain, temp, dt, SnowConfig(threshold, factor, .1))
            assert not (snow.melt_in.gt(0).to_numpy() & ~envelope).any()
    dry = dry_weather_mask(pd.DataFrame({"rainfall_in": rain + envelope}), dt, 3)
    assert not (dry & envelope).any()


def test_envelope_threshold_edges_initial_swe_zero_melt_and_invalid_bounds():
    envelope = snow_melt_envelope(np.array([.5, 0, 0, 0]), np.array([3, -2, 0, 5]),
                                  1, threshold_bounds=(-2, 3), melt_factor_bounds=(0, .5))
    np.testing.assert_array_equal(envelope, [False, False, True, True])
    assert snow_melt_envelope(np.zeros(2), np.array([5, 5]), 1,
                             threshold_bounds=(-2, 3), melt_factor_bounds=(0, .5),
                             initial_swe_in=.1).all()
    assert not snow_melt_envelope(np.array([.5, 0]), np.array([-5, 5]), 1,
                                 threshold_bounds=(-2, 3), melt_factor_bounds=(0, 0)).any()
    for threshold, melt in (((3, -2), (0, .5)), ((-2, np.inf), (0, .5)),
                            ((-2, 3), (-1, .5)), ((-2, 3), (.5, .1))):
        with pytest.raises(ValueError, match="bounds"):
            snow_melt_envelope(np.zeros(2), np.zeros(2), 1,
                               threshold_bounds=threshold, melt_factor_bounds=melt)


def test_bounded_snow_solver_is_thread_and_holdout_invariant():
    import platypus
    from sparsehydro.calibration.objectives import ConcordanceCorrelationCoefficient, NashSutcliffe
    from sparsehydro.calibration.solvers.platypus_solver import PlatypusSolver
    from sparsehydro.calibration.wholeflow_workflow import FeasibleCalibrationProblem

    rain = np.zeros(240)
    rain[[1, 25, 50, 90, 150, 200]] = [.5, .2, .4, .3, .2, .5]
    temp = np.tile(np.repeat([-4, 1, 5, 8], 15), 4)
    data = forcing(rain, temp)
    truth = make_model(config=SnowConfig(.8, .08))
    truth.prepare(data)
    data["flow_cfs"] = truth.predict().total_cfs + .005 * np.sin(np.arange(len(data)))
    train = np.arange(len(data)) < 180
    envelope = snow_melt_envelope(rain, temp, 1, threshold_bounds=(-2, 3),
                                  melt_factor_bounds=(0, .5))
    storm = melt_scoring_mask(envelope, train, 1)
    results = []
    for workers, perturb in ((1, False), (2, False), (2, True)):
        model = PatternWholeFlowModel(
            truth.pattern, 20.26, 1, .5, 2, snow_config=SnowConfig(0, .03),
            calibrate_snow=True)
        model.initialize()
        for name in model.scalar_parameter_names:
            model.get_scalar_parameter(name).calibrate = name in ("snow_T", "snow_ddf")
        frame = data.copy()
        if perturb:
            frame.loc[~train, "flow_cfs"] += 1000
        problem = FeasibleCalibrationProblem(
            model, frame.assign(flow_cfs=frame.flow_cfs.where(train)),
            objectives=[ConcordanceCorrelationCoefficient(mask=storm), NashSutcliffe(mask=train)],
            column_map={"observed": "flow_cfs", "predicted": lambda values: values},
            predict_fn=PatternWholeFlowModel.predict_total_cfs)
        assert problem.param_names == ["snow_T", "snow_ddf"]
        result = PlatypusSolver(platypus.NSGAII, population_size=12, n_evaluations=48,
                                seed=42, n_workers=workers).solve(problem)
        for record in result.history:
            assert ((record.X[:, 0] >= -2) & (record.X[:, 0] <= 3)).all()
            assert ((record.X[:, 1] >= .005) & (record.X[:, 1] <= .5)).all()
        assert np.ptp(np.concatenate([r.F[:, 1] for r in result.history])) > 1e-8
        for name, value in zip(result.param_names, result.pareto_X[result.knee_index()]):
            model.apply_flat_parameter(name, float(value))
        prediction = model.predict()
        assert model.budget["max_abs_error_in"] < 1e-8
        results.append((result, prediction))
    for result, prediction in results[1:]:
        np.testing.assert_array_equal(result.pareto_X, results[0][0].pareto_X)
        np.testing.assert_array_equal(result.pareto_F, results[0][0].pareto_F)
        pd.testing.assert_frame_equal(prediction, results[0][1], check_exact=True)
