import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sparsehydro.calibration.objectives import NashSutcliffe
from sparsehydro.calibration.solvers.scipy_solver import ScipySolver
from sparsehydro.calibration.wholeflow_workflow import FeasibleCalibrationProblem
from sparsehydro.configuration import ModelSettings, ParameterSettings, StudySettings
from sparsehydro.models.wholeflow_pattern import CenteredWeeklyPattern, ObservedBackground, PatternWholeFlowModel, estimate_observed_background, estimate_sanitary_shapes


PATTERN = CenteredWeeklyPattern(
    tuple(.1 * np.sin(np.arange(24) * np.pi / 12)),
    (.01, .02, .03, .01, -.01, -.02, -.04),
)


def data(days=180):
    t = pd.date_range("2024-01-01", periods=days * 24, freq="h")
    h, d = PATTERN.components(t)
    return pd.DataFrame(dict(
        datetime=t, rainfall_in=0., temperature_c=10.,
        flow_cfs=1.2 + np.arange(len(t)) / 24 * .002 + h + d,
    ))


def background(frame):
    return estimate_observed_background(frame, np.ones(len(frame), bool), PATTERN)


def build(frame, *, groundwater=False, mode="observed", snow=False, triangles=3):
    learned = background(frame) if mode == "observed" else None
    settings = StudySettings(model=ModelSettings(
        groundwater_enabled=groundwater, seasonal_baseline=mode,
        snow_enabled=snow, calibrate_snow=snow, n_triangles=triangles))
    return settings.build_model(PATTERN, 20.26, 1., 1.2, 3., observed_background=learned)


@pytest.mark.parametrize("groundwater", [False, True])
@pytest.mark.parametrize("mode", ["auto", "off", "observed"])
def test_background_modes_and_groundwater_are_exclusive(groundwater, mode):
    if groundwater and mode == "observed":
        with pytest.raises(ValueError, match="mutually exclusive"):
            ModelSettings(groundwater_enabled=True, seasonal_baseline=mode)
        return
    settings = StudySettings(model=ModelSettings(groundwater_enabled=groundwater, seasonal_baseline=mode))
    enabled = mode == "observed" or (mode == "auto" and not groundwater)
    learned = background(data()) if enabled else None
    model = settings.build_model(PATTERN, 20.26, 1., 1.2, 3., observed_background=learned)
    assert (model.observed_background is not None) == enabled
    assert ("sanitary_mean_cfs" in model.scalar_parameter_names) != enabled
    assert not any("annual" in name for name in model.scalar_parameter_names)
    assert model.groundwater_enabled == groundwater


def test_no_implicit_fit_or_calendar_mode_and_no_inactive_background():
    with pytest.raises(ValueError, match="training-derived"):
        StudySettings().build_model(PATTERN, 20.26, 1., 1.2, 3.)
    with pytest.raises(ValueError, match="exactly when"):
        StudySettings(model=ModelSettings(seasonal_baseline="off")).build_model(
            PATTERN, 20.26, 1., 1.2, 3., observed_background=background(data()))
    for mode in ("annual", "monthly", True, None, 1):
        with pytest.raises(ValueError, match="seasonal_baseline"):
            ModelSettings(seasonal_baseline=mode)
    with pytest.raises(ValueError, match="removed"):
        PatternWholeFlowModel(PATTERN, 20.26, 1., 1.2, 3., annual_seasonality=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        PatternWholeFlowModel(PATTERN, 20.26, 1., 1.2, 3., observed_background=background(data()))
    with pytest.raises(ValueError, match="not registered"):
        StudySettings(parameters={"sanitary_mean_cfs": ParameterSettings(value=1.)}).build_model(
            PATTERN, 20.26, 1., 1.2, 3., observed_background=background(data()))


@pytest.mark.parametrize("kwargs", [
    {"background_window_days": 0}, {"background_window_days": True},
    {"background_minimum_days": 22}, {"background_max_gap_days": -1},
    {"background_minimum_days": 2.5},
    {"background_endpoint_hold_days": -1}, {"background_endpoint_hold_days": True},
    {"background_endpoint_hold_days": 1.5},
])
def test_smoothing_settings_are_explicit_and_validated(kwargs):
    with pytest.raises(ValueError, match="Background"):
        ModelSettings(**kwargs)


def test_settings_and_fitted_background_persist_without_refitting():
    frame = data()
    learned = background(frame)
    restored = ObservedBackground.from_dict(json.loads(json.dumps(learned.to_dict())))
    assert restored == learned
    settings = StudySettings()
    saved_settings = StudySettings.from_dict(json.loads(json.dumps(settings.to_dict())))
    a = settings.build_model(PATTERN, 20.26, 1., 1.2, 3., observed_background=learned)
    b = saved_settings.build_model(PATTERN, 20.26, 1., 1.2, 3., observed_background=restored)
    for model in (a, b):
        model.prepare(frame.drop(columns="flow_cfs"))
    pd.testing.assert_frame_equal(a.predict(), b.predict(), check_exact=True)
    with pytest.raises(ValueError, match="options do not match"):
        StudySettings(model=ModelSettings(background_window_days=7)).build_model(
            PATTERN, 20.26, 1., 1.2, 3., observed_background=learned)


def test_legacy_stationary_studies_are_preserved_but_annual_requires_explicit_migration():
    for payload in (
        {}, {"schema_version": 1, "model": {"groundwater_enabled": False}},
        {"schema_version": 2, "model": {"seasonal_baseline": "off"}},
        {"schema_version": 2, "model": {"groundwater_enabled": True, "seasonal_baseline": "auto"}},
    ):
        settings = StudySettings.from_dict(payload)
        assert settings.schema_version == 3
        assert not settings.model.uses_observed_background
    for mode in ("annual", "auto"):
        with pytest.raises(ValueError, match="no longer supported"):
            StudySettings.from_dict(dict(
                schema_version=2, model=dict(groundwater_enabled=False, seasonal_baseline=mode)))
    legacy = PatternWholeFlowModel(PATTERN, 20.26, 1., 1.2, 3.)
    legacy.initialize()
    explicit = build(data(), groundwater=True, mode="off")
    for model in (legacy, explicit):
        model.prepare(data())
    pd.testing.assert_frame_equal(legacy.predict(), explicit.predict(), check_exact=True)


def test_background_round_trip_preserves_named_timezone_across_daylight_saving():
    frame = data(100)
    frame["datetime"] = pd.date_range(
        "2024-01-01", periods=len(frame), freq="h", tz="America/New_York")
    learned = background(frame)
    restored = ObservedBackground.from_dict(json.loads(json.dumps(learned.to_dict())))
    assert restored == learned
    t = pd.DatetimeIndex(frame.datetime)
    np.testing.assert_array_equal(restored.evaluate(t), learned.evaluate(t))
    assert np.isfinite(restored.evaluate(t)).all()


@pytest.mark.parametrize("mask_kind", ["integer", "empty", "invalid_source", "insufficient"])
def test_invalid_or_insufficient_training_sources_fail_explicitly(mask_kind):
    frame = data()
    mask = np.ones(len(frame), bool)
    if mask_kind == "integer":
        mask = mask.astype(int)
    elif mask_kind == "empty":
        mask[:] = False
    elif mask_kind == "invalid_source":
        frame.loc[10, "flow_cfs"] = np.nan
    else:
        mask[48:] = False
    with pytest.raises(ValueError, match="Background|background|Insufficient"):
        estimate_observed_background(frame, mask, PATTERN)


def test_background_uses_observations_not_calendar_and_does_not_repeat_next_year():
    frame = data(420)
    t = pd.DatetimeIndex(frame.datetime)
    mask = np.ones(len(frame), bool)
    learned = estimate_observed_background(frame, mask, PATTERN)
    values = learned.evaluate(t)
    # A monotonic source stays monotonic; the same date next year is not forced to repeat.
    assert np.all(np.diff(values) >= -1e-12)
    assert values[366 * 24] - values[0] > .7
    shifted = frame.assign(datetime=frame.datetime + pd.Timedelta(days=70))
    other = estimate_observed_background(shifted, mask, PATTERN)
    np.testing.assert_allclose(other.evaluate(pd.DatetimeIndex(shifted.datetime)), values, atol=1e-12)
    h, d = PATTERN.components(t)
    flat = estimate_observed_background(
        frame.assign(flow_cfs=1.5 + h + d), mask, PATTERN).evaluate(t)
    np.testing.assert_allclose(flat, 1.5, atol=1e-12)


def test_training_only_fit_ignores_holdout_wet_and_bad_observations():
    frame = data()
    mask = np.arange(len(frame)) // (7 * 24) % 4 != 2
    mask[24 * 50:24 * 55] = False  # excluded source interval, e.g. wet weather or data issues
    learned = estimate_observed_background(frame, mask, PATTERN)
    shapes = estimate_sanitary_shapes(frame, mask)
    changed = frame.copy()
    changed.loc[~mask, "flow_cfs"] = np.nan
    assert estimate_observed_background(changed, mask, PATTERN) == learned
    changed_shapes = estimate_sanitary_shapes(changed, mask)
    assert shapes.pattern == changed_shapes.pattern
    np.testing.assert_array_equal(shapes.detection_baseline_cfs, changed_shapes.detection_baseline_cfs)
    prediction = learned.evaluate(pd.DatetimeIndex(frame.datetime))
    h, d = PATTERN.components(pd.DatetimeIndex(frame.datetime))
    interior = (~mask) & (np.arange(len(frame)) > 21 * 24) & (np.arange(len(frame)) < 150 * 24)
    assert np.max(np.abs(prediction[interior] + h[interior] + d[interior]
                         - frame.flow_cfs.to_numpy()[interior])) < .025


def test_unobserved_days_are_not_anchors_and_large_gaps_are_not_smoothed_across():
    frame = data(180)
    mask = (frame.datetime < "2024-02-01") | (frame.datetime >= "2024-05-01")
    frame.loc[frame.datetime >= "2024-05-01", "flow_cfs"] += 5.
    learned = estimate_observed_background(frame, mask.to_numpy(), PATTERN)
    anchors = pd.DatetimeIndex(learned.datetimes)
    assert not ((anchors >= "2024-02-01") & (anchors < "2024-05-01")).any()
    values = learned.evaluate(pd.DatetimeIndex(frame.datetime))
    assert np.isnan(values[(frame.datetime >= "2024-02-01") & (frame.datetime < "2024-05-01")]).all()
    assert values[30 * 24] < 2.  # May source values did not contaminate January smoothing.
    assert values[121 * 24] > 6.


def test_unsupported_predictions_are_errors_unless_explicitly_unrequired():
    frame = data(60)
    model = build(frame)
    future = frame.assign(datetime=frame.datetime + pd.DateOffset(years=1))
    with pytest.raises(ValueError, match="no extrapolation"):
        model.prepare(future)
    extended = pd.concat([
        frame.iloc[:24].assign(datetime=frame.datetime.iloc[:24] - pd.Timedelta(days=1)),
        frame,
    ], ignore_index=True)
    extended["background_required"] = np.arange(len(extended)) >= 24
    model.prepare(extended)
    prediction = model.predict()
    assert prediction.total_cfs.iloc[:24].isna().all()
    assert not prediction.background_supported.iloc[:24].any()
    assert prediction.total_cfs.iloc[24:].notna().all()
    np.testing.assert_array_equal(model.predict_total_cfs(), prediction.total_cfs)
    with pytest.raises(ValueError, match="boolean"):
        model.prepare(frame.assign(background_required=1))


def test_endpoint_hold_is_bounded_constant_and_continuous_at_partial_source_days():
    learned = ObservedBackground(
        (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")),
        (1., 2., 4.), pd.Timestamp("2024-01-01 15:35"), pd.Timestamp("2024-01-03 02:40"),
        endpoint_hold_days=3)
    minute = pd.Timedelta(minutes=1)
    limit = pd.Timedelta(days=3)
    t = pd.DatetimeIndex([
        learned.start - limit - minute, learned.start - limit, learned.start - minute,
        learned.start, pd.Timestamp("2024-01-02"), learned.end,
        learned.end + minute, learned.end + limit, learned.end + limit + minute,
    ])
    values = learned.evaluate(t)
    assert np.isnan(values[[0, -1]]).all()
    np.testing.assert_array_equal(learned.endpoint_hold_mask(t),
                                  [False, True, True, False, False, False, True, True, False])
    start_value = 1. + (15 * 60 + 35) / (24 * 60)
    np.testing.assert_allclose(values[1:4], start_value)
    np.testing.assert_allclose(values[5:8], 4.)
    assert values[4] == 2.
    strict = replace(learned, endpoint_hold_days=0)
    assert np.isnan(strict.evaluate(t)[[0, 1, 2, 6, 7, 8]]).all()
    assert not strict.endpoint_hold_mask(t).any()


def test_endpoint_hold_does_not_bridge_long_interior_gaps():
    frame = data(180)
    mask = ((frame.datetime < "2024-02-01") | (frame.datetime >= "2024-05-01")).to_numpy()
    learned = estimate_observed_background(frame, mask, PATTERN, endpoint_hold_days=3)
    t = pd.DatetimeIndex(frame.datetime)
    gap = (t >= "2024-02-01") & (t < "2024-05-01")
    assert np.isnan(learned.evaluate(t)[gap]).all()
    assert not learned.endpoint_hold_mask(t)[gap].any()
    model = StudySettings(model=ModelSettings(background_endpoint_hold_days=3)).build_model(
        PATTERN, 20.26, 1., 1.2, 3., observed_background=learned)
    with pytest.raises(ValueError, match="interior gaps:"):
        model.prepare(frame)


def test_endpoint_hold_persistence_preserves_strict_legacy_backgrounds_and_timezone():
    frame = data(100)
    frame["datetime"] = pd.date_range(
        "2024-01-01", periods=len(frame), freq="h", tz="America/New_York")
    learned = replace(background(frame), endpoint_hold_days=3)
    restored = ObservedBackground.from_dict(json.loads(json.dumps(learned.to_dict())))
    assert restored == learned
    t = pd.DatetimeIndex([learned.start - pd.Timedelta(days=3), learned.end + pd.Timedelta(days=3)])
    np.testing.assert_array_equal(restored.evaluate(t), learned.evaluate(t))
    assert restored.endpoint_hold_mask(t).all()
    payload = learned.to_dict()
    del payload["endpoint_hold_days"]
    legacy = ObservedBackground.from_dict(payload)
    assert legacy.endpoint_hold_days == 0
    assert np.isnan(legacy.evaluate(t)).all()
    settings = StudySettings(model=ModelSettings(background_endpoint_hold_days=3))
    assert StudySettings.from_dict(settings.to_dict()) == settings
    with pytest.raises(ValueError, match="options do not match"):
        StudySettings().build_model(PATTERN, 20.26, 1., 1.2, 3., observed_background=learned)


def test_model_flags_held_endpoints_without_filling_full_warmup_or_changing_physics():
    frame = data(60)
    learned = replace(background(frame), endpoint_hold_days=3)
    extended = pd.concat([
        frame.iloc[:4 * 24].assign(datetime=frame.datetime.iloc[:4 * 24] - pd.Timedelta(days=4)),
        frame,
        frame.iloc[-2 * 24:].assign(datetime=frame.datetime.iloc[-2 * 24:] + pd.Timedelta(days=2)),
    ], ignore_index=True)
    extended["background_required"] = np.arange(len(extended)) >= 24
    model = StudySettings(model=ModelSettings(background_endpoint_hold_days=3)).build_model(
        PATTERN, 20.26, 1., 1.2, 3., observed_background=learned)
    model.prepare(extended)
    prediction = model.predict()
    assert prediction.total_cfs.iloc[:24].isna().all()
    assert prediction.total_cfs.iloc[24:].notna().all()
    np.testing.assert_array_equal(prediction.background_supported, prediction.total_cfs.notna())
    np.testing.assert_array_equal(prediction.background_endpoint_held,
                                  learned.endpoint_hold_mask(pd.DatetimeIndex(extended.datetime)))
    assert prediction.background_endpoint_held.sum() == 5 * 24
    assert "background_endpoint_held" in model.output_field_names
    np.testing.assert_array_equal(prediction.total_cfs, model.predict_total_cfs())
    budget = model.budget.copy()
    model.prepare(extended.assign(flow_cfs=np.nan))
    pd.testing.assert_frame_equal(prediction, model.predict())
    assert model.budget == budget


@pytest.mark.parametrize("snow", [False, True])
@pytest.mark.parametrize("triangles", [1, 4])
def test_background_preserves_rainfall_physics_and_ignores_prediction_observations(snow, triangles):
    frame = data(60)
    model = build(frame, snow=snow, triangles=triangles)
    frame.loc[::48, "rainfall_in"] = .5
    frame.loc[:60, "temperature_c"] = -5.
    model.prepare(frame)
    prediction = model.predict()
    budget = model.budget.copy()
    model.prepare(frame.assign(flow_cfs=-10000.))
    pd.testing.assert_frame_equal(prediction, model.predict())
    np.testing.assert_array_equal(prediction.total_cfs, model.predict_total_cfs())
    np.testing.assert_array_equal(prediction.gwi_cfs, 0)
    np.testing.assert_allclose(prediction.total_cfs,
                               prediction.sanitary_cfs + prediction.gwi_cfs + prediction.rdii_cfs)
    h, d = PATTERN.components(pd.DatetimeIndex(frame.datetime))
    np.testing.assert_allclose(prediction.sanitary_cfs, prediction.slow_background_cfs + h + d)
    stationary = build(frame, mode="off", snow=snow, triangles=triangles)
    stationary.prepare(frame)
    other = stationary.predict()
    for column in ("S_in", "G_in", "gwi_cfs", "rdii_cfs", "SWE_in", "budget_error_in"):
        np.testing.assert_array_equal(prediction[column], other[column])
    assert stationary.budget == budget == model.budget
    assert budget["max_abs_error_in"] < 1e-8


def test_joint_calibration_cannot_move_frozen_background():
    frame = data(60)
    train = np.arange(len(frame)) // (7 * 24) % 4 != 2
    learned = estimate_observed_background(frame, train, PATTERN)
    model = StudySettings().build_model(
        PATTERN, 20.26, 1., 1.2, 3., observed_background=learned)
    for name in model.scalar_parameter_names:
        model.get_scalar_parameter(name).calibrate = name == "sanitary_hour_scale"
    original = model.observed_background
    problem = FeasibleCalibrationProblem(
        model, frame.assign(flow_cfs=frame.flow_cfs.where(train)), [NashSutcliffe(mask=train)],
        column_map={"observed": "flow_cfs", "predicted": lambda values: values},
        predict_fn=PatternWholeFlowModel.predict_total_cfs)
    result = ScipySolver(method="L-BFGS-B", maxiter=20).solve(problem)
    assert result.param_names == ["sanitary_hour_scale"]
    assert model.observed_background == original
    assert np.isfinite(result.pareto_F).all()


def test_background_nonnegativity_is_not_output_clipping():
    frame = data()
    learned = background(frame)
    low = ObservedBackground(
        learned.datetimes, (.01,) * len(learned.flow_cfs), learned.start, learned.end)
    model = PatternWholeFlowModel(
        PATTERN, 20.26, 1., 1.2, 3., groundwater_enabled=False, observed_background=low)
    model.initialize()
    assert not model.validate()
    model.prepare(frame)
    with pytest.raises(ValueError, match="feasible"):
        model.predict()


@pytest.mark.parametrize("groundwater", [False, True])
def test_notebook_monthly_report_and_sensitivity_preserve_model(groundwater):
    from sparsehydro.calibration.pattern_reporting import seasonal_dry_diagnostics
    path = Path(__file__).parents[1] / "sparsehydro" / "notebooks" / "flowfinity_gwi_wholeflow_pattern_joint.ipynb"
    sources = {c["id"]: "".join(c["source"]) for c in json.loads(path.read_text(encoding="utf-8"))["cells"]}
    frame = data()
    model = build(frame, groundwater=groundwater, mode="off" if groundwater else "observed")
    model.get_scalar_parameter("s_initial_fraction").value = .3
    if groundwater:
        model.get_scalar_parameter("g_initial_fraction").value = .25
    model.prepare(frame)
    prediction = model.predict()
    saved = {n: model.get_scalar_parameter(n).value for n in model.scalar_parameter_names}
    budget = model.budget.copy()
    frame["joint_sanitary_cfs"] = prediction.sanitary_cfs
    frame["gwi_cfs"], frame["rdii_cfs"] = prediction.gwi_cfs, prediction.rdii_cfs
    train = frame.datetime.dt.month.le(3).to_numpy()
    frame["flow_cfs"] = prediction.total_cfs
    ns = dict(np=np, pd=pd, joint_model=model, joint_pred=prediction, model_df=frame,
              seasonal_dry_diagnostics=seasonal_dry_diagnostics,
              dry_mask=train, val_dry_mask=~train, scored_mask=np.ones(len(frame), bool),
              q_total=prediction.total_cfs.to_numpy(), obs_flow=prediction.total_cfs.to_numpy(),
              display=lambda _: None)
    exec(sources["pattern-joint-51"], ns)
    report = ns["seasonal_diagnostics"]
    assert len(report) == 24
    defined = report.status.eq("defined")
    assert report.loc[~defined, "total_bias_cfs"].isna().all()
    assert report.loc[defined, "total_bias_cfs"].abs().max() == 0
    assert report.loc[defined, "slow_background_mean_cfs"].notna().all()
    assert saved == {n: model.get_scalar_parameter(n).value for n in model.scalar_parameter_names}
    assert budget == model.budget
    pd.testing.assert_frame_equal(prediction, model.predict())
