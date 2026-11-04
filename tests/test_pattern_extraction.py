import ast
from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from sparsehydro.calibration.pattern_workflow import prepare_pattern_data, study_settings_from_controls
from sparsehydro.calibration.splits import SplitConfig, build_masks
from sparsehydro.data.services import DownloadResult
from sparsehydro.data.wholeflow import (
    MGD_TO_CFS, align_air_temperature, assemble_modeling_data, download_flowfinity_inputs,
    modeling_timestep, read_local_results, select_flowfinity_record,
)
from tests.pattern_workflow_support import NOTEBOOK, controls


def input_results(periods=72):
    index = pd.date_range("2025-01-01", periods=periods, freq="5min")
    return {
        name: DownloadResult(name, Path(f"{name}.csv"), pd.DataFrame(
            {"value": np.arange(periods, dtype=float) + offset}, index=index))
        for name, offset in (("Flow", 1.), ("Rain", 0.))
    }


@pytest.mark.parametrize("timestep,factor", [("5min", 1), ("15min", 3), ("60min", 12), ("1h", 12)])
def test_assembly_preserves_native_mean_flow_and_rainfall_volume(timestep, factor):
    results = input_results()
    data = assemble_modeling_data(results, timestep)
    flow = results["Flow"].dataframe.iloc[:, 0].to_numpy().reshape(-1, factor).mean(axis=1)
    rain = results["Rain"].dataframe.iloc[:, 0].to_numpy().reshape(-1, factor).sum(axis=1)
    np.testing.assert_array_equal(data.flow_mgd, flow)
    np.testing.assert_array_equal(data.rainfall_in, rain)
    np.testing.assert_allclose(data.flow_cfs, flow * MGD_TO_CFS)
    assert pd.DatetimeIndex(data.datetime).equals(results["Flow"].dataframe.index[::factor])


@pytest.mark.parametrize("bad,match", [
    ("duplicate", "unique"), ("missing_step", "irregular"), ("nan", "missing or negative"),
    ("negative", "missing or negative"), ("empty", "nonempty"),
])
def test_assembly_rejects_bad_inputs_without_filling(bad, match):
    results = input_results()
    data = results["Rain"].dataframe
    if bad == "duplicate":
        data.index = pd.DatetimeIndex([data.index[0], *data.index[:-1]])
    elif bad == "missing_step":
        data.drop(index=data.index[5], inplace=True)
    elif bad == "empty":
        data.drop(index=data.index, inplace=True)
    else:
        data.iloc[5, 0] = np.nan if bad == "nan" else -1
    with pytest.raises(ValueError, match=match):
        assemble_modeling_data(results, "15min")


def test_assembly_trims_partial_edge_bins_but_never_fills_an_interior_gap():
    results = input_results(periods=74)
    data = assemble_modeling_data(results, "15min")
    assert len(data) == 24
    assert data.datetime.iloc[-1] == results["Flow"].dataframe.index[69]
    with pytest.raises(ValueError, match="No complete"):
        assemble_modeling_data(input_results(periods=2), "60min")
    with pytest.raises(ValueError, match="multiple of the native"):
        slower = {name: DownloadResult(name, value.path, value.dataframe.iloc[::3])
                  for name, value in results.items()}
        assemble_modeling_data(slower, "5min")
    with pytest.raises(ValueError, match="TIMESTEP"):
        modeling_timestep("10min")


def test_local_input_reader_selects_columns_and_optional_files(tmp_path):
    times = pd.date_range("2025-01-01", periods=8, freq="h")
    frame = pd.DataFrame({"first": np.arange(8), "second": np.arange(8) + 10}, index=times)
    frame.to_csv(tmp_path / "rain.csv")
    frame.to_csv(tmp_path / "flow.csv")
    results = read_local_results(
        tmp_path, {"Rain": ("rain.csv", "second"), "Flow": ("flow.csv", 0), "BaseFlow": (None, 0)},
        start="2025-01-01 02:00", end="2025-01-01 04:00")
    assert set(results) == {"Rain", "Flow"}
    assert results["Rain"].dataframe.columns.tolist() == ["second"]
    assert results["Flow"].dataframe.iloc[:, 0].tolist() == [2, 3, 4]
    with pytest.raises(ValueError, match="Flow file"):
        read_local_results(tmp_path, {"Rain": ("rain.csv", 0)})
    with pytest.raises(FileNotFoundError, match="Rain file"):
        read_local_results(tmp_path, {"Rain": ("missing.csv", 0), "Flow": ("flow.csv", 0)})


@pytest.mark.parametrize("index", [True, -1, 2, "0"])
def test_flowfinity_record_index_is_validated(index):
    with pytest.raises(ValueError, match="FLOWFINITY_RECORD_INDEX"):
        select_flowfinity_record(pd.DataFrame({"DataTitle": ["first", "second"]}), title=None, index=index)


@pytest.mark.parametrize("local_auth", [False, True])
def test_flowfinity_acquisition_selects_sorted_records_without_network(tmp_path, monkeypatch, local_auth):
    from types import SimpleNamespace
    import sparsehydro.data.wholeflow as inputs

    calls = {}
    auth = SimpleNamespace(FLOWFINITY_USERNAME="test-user", FLOWFINITY_PASSWORD="test-value")

    def import_auth(name):
        assert name == "flowfinity_auth"
        if not local_auth:
            raise ModuleNotFoundError(name=name)
        return auth

    class Client:
        base_url = "test-service"

        def __init__(self, **credentials):
            calls["credentials"] = credentials

        def list_resources(self, name_filter):
            calls["name_filter"] = name_filter
            return [{"DataTitle": "second site"}, {"DataTitle": "first site"}]

        def available_types(self, title):
            assert title == "second site"
            return ["Flow", "Rain"]

        def download(self, title, **kwargs):
            calls["title"], calls["download"] = title, kwargs
            return input_results()

    monkeypatch.setattr(inputs.importlib, "import_module", import_auth)
    monkeypatch.setattr(inputs.importlib, "reload", lambda module: module)
    monkeypatch.setattr(inputs, "FlowFinityClient", Client)
    downloaded = download_flowfinity_inputs(
        tmp_path, auth_dir=tmp_path, record_title=None, record_index=1,
        name_filter="site", types=["Flow", "Rain"], start="2025-01-01", end=None)
    assert downloaded.title == calls["title"] == "second site"
    assert downloaded.directory == tmp_path / "second_site"
    assert downloaded.records.DataTitle.tolist() == ["first site", "second site"]
    assert calls["name_filter"] == "site"
    assert calls["download"] == dict(
        types=["Flow", "Rain"], output_dir=tmp_path / "second_site",
        parse=True, start="2025-01-01", end=None)
    assert calls["credentials"] == (
        {"username": "test-user", "password": "test-value"} if local_auth else
        {"username": None, "password": None})
    assert set(vars(downloaded)) == {"title", "directory", "records", "results"}


def test_flowfinity_auth_dependency_errors_do_not_become_environment_fallbacks(tmp_path, monkeypatch):
    import sparsehydro.data.wholeflow as inputs

    def broken_auth(name):
        raise ModuleNotFoundError(name="missing_auth_dependency")

    monkeypatch.setattr(inputs.importlib, "import_module", broken_auth)
    with pytest.raises(ModuleNotFoundError) as error:
        download_flowfinity_inputs(
            tmp_path, auth_dir=tmp_path, record_title=None, record_index=0,
            name_filter=None, types=["Flow", "Rain"])
    assert error.value.name == "missing_auth_dependency"


def test_temperature_alignment_uses_fixed_est_and_no_endpoint_filling(tmp_path):
    path = tmp_path / "air_temperature.csv"
    pd.DataFrame({
        "datetime": pd.date_range("2025-01-01 05:00", periods=4, freq="h", tz="UTC"),
        "temperature_c": [0., 2., 4., 6.],
    }).to_csv(path, index=False)
    data = pd.DataFrame({"datetime": pd.date_range("2024-12-31 23:30", periods=9, freq="30min")})
    aligned, source = align_air_temperature(data, [tmp_path / "missing.csv", path], 2.)
    assert source == path
    np.testing.assert_array_equal(aligned.temperature_c, np.arange(7))
    assert aligned.datetime.iloc[0] == pd.Timestamp("2025-01-01")
    assert "temperature_c" not in data
    with pytest.raises(ValueError, match="Temperature gap"):
        align_air_temperature(data, [path], .5)


@pytest.mark.parametrize("unit", ["ns", "us", "ms"])
@pytest.mark.parametrize("dt", [1 / 12, .25, 1.])
def test_preparation_is_independent_of_datetime_storage_unit(unit, dt, tmp_path):
    values = controls()
    values.update(EVENT_SOURCE="auto", SNOW_ENABLED=False, CALIBRATE_SNOW=False)
    study = study_settings_from_controls(values)
    size = 120
    times = pd.date_range("2025-01-01", periods=size, freq=pd.Timedelta(hours=dt)).as_unit(unit)
    data = pd.DataFrame({"datetime": times, "rainfall_in": 0., "temperature_c": 10., "flow_cfs": 1.})
    original = data.copy(deep=True)
    split = SplitConfig(dt_hours=dt, warmup_days=0, spinup_mode="none",
                        split_mode="continuous", train_frac=.7, snap_edges_to_dry=False)
    prepared = prepare_pattern_data(data, study, split, tmp_path)
    expected = build_masks(data.datetime, split, rain=data.rainfall_in.to_numpy())
    for actual, mask in zip((prepared.warmup_mask, prepared.train_mask, prepared.val_mask), expected[:3]):
        np.testing.assert_array_equal(actual, mask)
    np.testing.assert_array_equal(prepared.data.liquid_input_in, data.rainfall_in)
    assert not prepared.data.possible_melt.any()
    pd.testing.assert_frame_equal(data, original)
    with pytest.raises(RuntimeError, match="already carries"):
        prepare_pattern_data(prepared.data, study, split, tmp_path)


def test_all_notebook_code_compiles_and_extracted_cells_stay_lightweight():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    targets = {"pattern-joint-25", "pattern-joint-45", "pattern-joint-47", "pattern-joint-49",
               "pattern-joint-61", "pattern-joint-pareto-msd-spread", "pattern-joint-pareto-msd-scan"}
    assert targets <= {cell["id"] for cell in notebook["cells"]}
    lines = 0
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        ast.parse(source, feature_version=(3, 10))
        lines += len(source.splitlines())
        if cell["id"] in targets:
            assert len(source.splitlines()) <= 20, f"Cell {cell['id']} grew beyond orchestration"
    assert lines < 800


def test_optional_diagnostics_are_self_contained_and_preserve_budget():
    from sparsehydro.models.rdii.initial_abstraction import SnowConfig
    from sparsehydro.models.wholeflow_pattern import CenteredWeeklyPattern, PatternWholeFlowModel
    from sparsehydro.visualization.wholeflow_diagnostics import (
        plot_baseflow_components, plot_snow_diagnostics, plot_water_balance,
    )

    pattern = CenteredWeeklyPattern((0.,) * 24, (0.,) * 7)
    model = PatternWholeFlowModel(pattern, 26., 1., .8, 2., snow_config=SnowConfig())
    model.initialize()
    size = 96
    data = pd.DataFrame({
        "datetime": pd.date_range("2025-01-01", periods=size, freq="h"),
        "rainfall_in": np.where(np.arange(size) % 24 == 0, .8, 0.),
        "temperature_c": np.where(np.arange(size) < 24, -5., 10.), "flow_cfs": 1.,
    })
    model.prepare(data)
    prediction = model.predict()
    saved = prediction.copy(deep=True)
    val = np.arange(size) >= 72
    balance = plot_water_balance(data, prediction, model, dt_hours=1.)
    np.testing.assert_allclose(balance.data[0].y, balance.data[1].y, atol=1e-8, rtol=0)
    snow = plot_snow_diagnostics(data, prediction, model, data_title="Test", warmup_steps=12, val_mask=val)
    assert len(snow.data) == 10
    assert snow.layout.xaxis.matches == "x4"
    components = data.assign(
        baseflow_cfs=.7, joint_baseflow_cfs=prediction.baseflow_cfs,
        joint_sanitary_cfs=prediction.sanitary_cfs, gwi_cfs=prediction.gwi_cfs)
    baseflow = plot_baseflow_components(components, val_mask=val)
    assert len(baseflow.data) == 4
    assert "not observed truth" in baseflow.layout.title.text
    pd.testing.assert_frame_equal(prediction, saved)
    pd.testing.assert_frame_equal(model.predict(), saved)


def test_notebook_runs_end_to_end_on_synthetic_local_inputs(tmp_path, monkeypatch):
    size = 56 * 24 * 12
    times = pd.date_range("2025-01-01", periods=size, freq="5min")
    rain = np.zeros(size)
    rows = []
    for number, day in enumerate(range(3, 54, 4), 1):
        start = day * 24 * 12 + 12 * 12
        rain[start:start + 12] = .15 + number * .02
        rows.append((str(number), times[start], times[start + 72]))
    response = np.convolve(rain, np.exp(-np.arange(288) / 36), mode="full")[:size]
    flow_cfs = .8 + .1 * np.sin(np.arange(size) * np.pi / 144) + response
    pd.DataFrame({"rain": rain}, index=times).to_csv(tmp_path / "rain.csv")
    pd.DataFrame({"flow": flow_cfs / MGD_TO_CFS}, index=times).to_csv(tmp_path / "flow.csv")
    pd.DataFrame({
        "datetime": times.tz_localize("Etc/GMT+5").tz_convert("UTC"), "temperature_c": 10.,
    }).to_csv(tmp_path / "site_temperature.csv", index=False)
    pd.DataFrame(rows, columns=["event_id", "start_date", "end_date"]).to_csv(tmp_path / "events.csv", index=False)
    monkeypatch.setattr(go.Figure, "show", lambda *args, **kwargs: None)
    monkeypatch.setattr(go.Figure, "write_html", lambda *args, **kwargs: None)
    monkeypatch.setattr(go.Figure, "write_image", lambda *args, **kwargs: None)
    namespace = {"display": lambda *args, **kwargs: None}
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    ids = {cell["id"] for cell in notebook["cells"]}
    assert {"pattern-joint-02", "pattern-joint-04"} <= ids
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        if cell["id"] == "pattern-joint-02":
            namespace.update(
                DATA_SOURCE="files", DATA_TITLE="Synthetic", SOURCE_DIR=str(tmp_path),
                RAIN_FILE="rain.csv", FLOW_FILE="flow.csv", TEMP_FILE="site_temperature.csv",
                FILE_START=None, FILE_END=None,
                EVENT_SOURCE="file", EVENT_FILE="events.csv", ASSESSMENT_EXCLUDED_EVENT_IDS=(),
                WARMUP_DAYS=7, SPINUP_MODE="wrap", SPLIT_MODE="continuous",
                SNOW_ENABLED=False, CALIBRATE_SNOW=False, REQUIRE_NUMBA=False,
                NSGA2_SETTINGS={"pop_size": 12, "generations": 2, "seed": 42}, CALIBRATION_WORKERS=1,
                MULTIOBJECTIVE_OBJECTIVES=["ccc", "nash_sutcliffe"])
        exec(compile("".join(cell["source"]), f"pattern-notebook-{cell['id']}", "exec"), namespace)
        if cell["id"] in ("pattern-joint-02", "pattern-joint-04"):
            namespace["NOTEBOOK_DIR"] = tmp_path
    artifact_dir = namespace["ARTIFACT_DIR"]
    assert namespace["_air_csv"] == tmp_path / "site_temperature.csv"
    assert namespace["model_df"]["temperature_c"].eq(10.).all()
    assert namespace["selected_index"] == namespace["fit"].selected_index
    np.testing.assert_array_equal(
        namespace["joint_model"].predict_total_cfs(), namespace["fit"].prediction.total_cfs)
    assert namespace["prepared"].spinup_steps == 7 * 24 * 12
    assert len(namespace["presentation_figures"]) == 4
    for name in ("pareto_solutions.csv", "predictions.csv.gz", "scoring_masks.csv.gz",
                 "study_settings.json", "run_metadata.json", "pareto_msdgc_comparison.csv",
                 "modeled_total_flow.tsf", "modeled_stormflow.tsf",
                 "modeled_total_flow_best.tsf", "modeled_stormflow_best.tsf"):
        assert (artifact_dir / name).is_file()
    metadata = json.loads((artifact_dir / "run_metadata.json").read_text())
    saved_events = json.loads((artifact_dir / "study_settings.json").read_text())["workflow"]["events"]
    assert saved_events["STORM_DETECTION_SETTINGS"]["min_rain_in"] == .2
    assert metadata["requested_evaluations"] == 24
    assert metadata["checker_role"] == "report only"
    assert not metadata["msdgc_assessment"]["official_storm_scorecard"]
