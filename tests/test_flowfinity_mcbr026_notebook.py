"""Exercise edited notebook cells without downloading data or running a long fit."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sparsehydro.calibration import ConcordanceCorrelationCoefficient, NashSutcliffe
from sparsehydro.data.utilities import read_tsf, write_tsf


NOTEBOOK = (Path(__file__).parents[1] / "sparsehydro" / "notebooks"
            / "flowfinity_ValModel_SBU_MC-BR-026.ipynb")


@pytest.fixture(scope="module")
def cells():
    return {cell["id"]: "".join(cell["source"])
            for cell in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]}


def test_all_notebook_code_cells_compile():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{NOTEBOOK.name}:{cell['id']}", "exec")
    assert len({cell["id"] for cell in notebook["cells"]}) == len(notebook["cells"])


@pytest.fixture
def notebook_state(tmp_path):
    times = pd.date_range("2025-01-01", periods=245, freq="5min")
    observed = np.repeat([1., 2., 4., 8., 10.], 49)
    frame = pd.DataFrame(dict(datetime=times, flow_cfs=observed,
                              rainfall_in=0., stormflow=observed))
    warmup = np.arange(len(times)) < 49
    train = (np.arange(len(times)) >= 49) & (np.arange(len(times)) < 160)
    validation = ~(warmup | train)
    prediction = pd.DataFrame(dict(datetime=times, q_total_cfs=observed,
                                    rdii_output=observed * .25))
    calls = []

    def apply_and_predict(x):
        calls.append(x)
        return prediction

    return dict(np=np, pd=pd, model_df=frame, warmup_mask=warmup, train_mask=train,
                val_mask=validation, ARTIFACT_DIR=tmp_path, DT_HOURS=1 / 12,
                MGD_TO_CFS=1e6 / (7.480519 * 86400), DATA_TITLE="MC-BR-026",
                knee_x=np.array([1.]), apply_and_predict=apply_and_predict,
                prediction=prediction, calls=calls, USE_CON15_OBJECTIVE=False,
                TOP_EVENT_FRACTION=.5, NashSutcliffe=NashSutcliffe,
                ConcordanceCorrelationCoefficient=ConcordanceCorrelationCoefficient,
                ensemble=object(),
                CalibrationProblem=lambda **kw: SimpleNamespace(n_params=0, param_names=[], **kw))


def prepare_events(cells, state, empty=False):
    times = state["model_df"].datetime
    events = [] if empty else [
        SimpleNamespace(event_id=i, start_idx=start, end_idx=start + 48,
                        start_datetime=times.iloc[start], end_datetime=times.iloc[start + 48])
        for i, start in enumerate(range(0, 245, 49), 1)
    ]
    tree = ast.parse(cells["48c58d5b"])
    cutoff = next(i for i, node in enumerate(tree.body)
                  if isinstance(node, ast.Assign) and any(
                      isinstance(target, ast.Name) and target.id == "POP_SIZE"
                      for target in node.targets))
    with patch("sparsehydro.events.msdgc.find_events", return_value=events):
        exec(compile(ast.Module(tree.body[:cutoff], type_ignores=[]), "<notebook setup>", "exec"), state)


@pytest.mark.parametrize("enabled", [False, True])
def test_default_and_optional_objectives_and_reporting(cells, notebook_state, enabled):
    state = notebook_state
    state["USE_CON15_OBJECTIVE"] = enabled
    prepare_events(cells, state)
    assert [o.name for o in state["calibration_objectives"]] == (
        ["nash_sutcliffe", "ccc"] + (["con15_error"] if enabled else []))
    exec(cells["ad9a2d23"], state)
    zones = state["zone_volumes_msdgc"]
    assert zones.query("window == 'whole'").event_id.nunique() == 4
    assert zones.query("window == 'train'").event_id.nunique() == 2
    assert zones.query("window == 'validation'").event_id.nunique() == 1
    assert set(state["scorecard_msdgc"].metric) == {
        "conveyance_ccc", "storage_ccc", "treatment_ccc", "three_zone_mean_ccc", "con15_slope"}
    assert state["scorecard_msdgc"].query("window == 'whole'").status.eq("pass").all()
    assert state["scorecard_msdgc"].query(
        "window == 'validation' and metric == 'conveyance_ccc'").status.eq("undefined").all()
    assert len(state["calls"]) == 1
    assert (state["ARTIFACT_DIR"] / "msdgc_zone_volumes.csv").exists()


def test_no_events_produces_undefined_report_and_optional_objective_raises(cells, notebook_state):
    prepare_events(cells, notebook_state, empty=True)
    exec(cells["ad9a2d23"], notebook_state)
    assert notebook_state["scorecard_msdgc"].status.eq("undefined").all()
    notebook_state["USE_CON15_OBJECTIVE"] = True
    with pytest.raises(ValueError, match="nonempty"):
        prepare_events(cells, notebook_state, empty=True)


def test_actual_export_cell_restores_knee_excludes_warmup_and_converts_both_series(cells, notebook_state):
    state = notebook_state
    exec(cells["modeled_tsf_export"], state)
    assert set(state["tsf_paths"]) == {"Flow", "StormFlow"}
    expected_index = pd.DatetimeIndex(state["model_df"].datetime[~state["warmup_mask"]], name="Date/Time")
    for variable, column in (("Flow", "q_total_cfs"), ("StormFlow", "rdii_output")):
        path = state["tsf_paths"][variable]
        exported = read_tsf(path)
        pd.testing.assert_index_equal(exported.index, expected_index)
        expected = state["prediction"].loc[~state["warmup_mask"], column] / state["MGD_TO_CFS"]
        np.testing.assert_allclose(exported[variable], expected, rtol=1e-10)
        assert path.read_text().splitlines()[:3] == [
            f"IDs:\tMC-BR-026 Modeled {variable}", f"Date/Time\t{variable}", "M/d/yyyy\tMGD"]
    assert len(state["calls"]) == 1
    np.testing.assert_array_equal(state["calls"][0], state["knee_x"])


@pytest.mark.parametrize("bad", ["mask", "timestamps", "nan", "empty"])
def test_export_cell_rejects_bad_inputs_before_writing(cells, notebook_state, bad):
    state = notebook_state
    if bad == "mask":
        state["warmup_mask"] = state["warmup_mask"].astype(int)
    elif bad == "timestamps":
        state["prediction"]["datetime"] += pd.Timedelta(minutes=1)
    elif bad == "nan":
        state["prediction"].loc[100, "rdii_output"] = np.nan
    else:
        state["warmup_mask"][:] = True
    with pytest.raises(ValueError):
        exec(cells["modeled_tsf_export"], state)
    assert not list(state["ARTIFACT_DIR"].glob("*.tsf"))


@pytest.mark.parametrize("bad", ["empty", "nan", "inf", "duplicate", "unsorted", "timezone", "subsecond", "nat"])
def test_tsf_writer_rejects_invalid_series(tmp_path, bad):
    index = pd.date_range("2025-01-01", periods=3, freq="5min")
    values = np.array([1., 2., 3.])
    if bad == "empty":
        index, values = index[:0], values[:0]
    elif bad in ("nan", "inf"):
        values[1] = float(bad)
    elif bad == "duplicate":
        index = pd.DatetimeIndex([index[0], index[0], index[2]])
    elif bad == "unsorted":
        index = index[::-1]
    elif bad == "timezone":
        index = index.tz_localize("UTC")
    elif bad == "subsecond":
        index += pd.Timedelta(milliseconds=1)
    else:
        index = pd.DatetimeIndex([index[0], pd.NaT, index[2]])
    path = tmp_path / "invalid.tsf"
    with pytest.raises(ValueError):
        write_tsf(pd.Series(values, index=index), path, title="Flow", variable="Flow", unit="MGD")
    assert not path.exists()


@pytest.mark.parametrize("field,value", [
    ("title", ""), ("title", "bad\nheader"), ("variable", "bad\theader"),
    ("variable", "Date/Time"), ("unit", "bad\rheader"),
])
def test_tsf_writer_rejects_invalid_headers(tmp_path, field, value):
    header = dict(title="Flow", variable="Flow", unit="MGD")
    header[field] = value
    path = tmp_path / "invalid.tsf"
    with pytest.raises(ValueError):
        write_tsf(pd.Series([1.], index=pd.date_range("2025-01-01", periods=1)), path, **header)
    assert not path.exists()
