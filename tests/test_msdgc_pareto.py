import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sparsehydro.calibration.msdgc_assessment import MSDGCAssessment
from sparsehydro.calibration.msdgc_pareto import MSDGC_METRIC_ORDER, evaluate_msdgc_pareto, rank_msdgc_pareto
from sparsehydro.calibration.result import CalibrationResult
from sparsehydro.calibration.wholeflow_acceptance import AcceptanceContract
from sparsehydro.models.wholeflow_pattern import CenteredWeeklyPattern, PatternWholeFlowModel


def metric_report(count=2):
    rows = []
    for index in range(count):
        for window in ("train", "validation", "whole"):
            for metric in MSDGC_METRIC_ORDER:
                slope = metric == "con15_slope"
                rows.append(dict(
                    pareto_index=index, window=window, metric=metric,
                    value=1., lower=.9, upper=1.1 if slope else 1.,
                    status="pass", reason="", n=4, model_feasible=True,
                ))
    return pd.DataFrame(rows)


def set_score(report, index, window, metric, value, status="fail"):
    report.loc[
        report.pareto_index.eq(index) & report.window.eq(window) & report.metric.eq(metric),
        ["value", "status"],
    ] = [value, status]


def test_passes_require_each_zone_and_con15_in_validation_and_whole():
    report = metric_report()
    set_score(report, 0, "validation", "treatment_ccc", .899)
    selected, summary = rank_msdgc_pareto(report)
    assert selected == 1
    assert summary.passes_all_msd.tolist() == [False, True]
    assert summary.checks_passed.tolist() == [7, 8]
    assert summary.aggregate_error.tolist() == pytest.approx([.01, 0])


def test_fallback_is_sum_of_normalized_violations_not_distance_to_ideal():
    report = metric_report()
    set_score(report, 0, "validation", "storage_ccc", .85)
    set_score(report, 0, "whole", "conveyance_ccc", .88)
    set_score(report, 0, "whole", "con15_slope", 1.12)
    set_score(report, 1, "validation", "con15_slope", .80)
    selected, summary = rank_msdgc_pareto(report)
    assert selected == 0
    assert summary.aggregate_error.tolist() == pytest.approx([.9, 1.])
    assert summary.checks_failed.tolist() == [3, 1]
    assert summary.loc[0, "whole_con15_slope_normalized_violation"] == pytest.approx(.2)
    assert summary.loc[0, "diagnostic_selection_status"] == "closest_threshold_shortfall"


def test_training_mean_and_event_ccc_never_influence_pass_or_ranking():
    report = metric_report()
    set_score(report, 1, "whole", "con15_slope", 1.11)
    baseline = rank_msdgc_pareto(report)
    excluded = report.window.eq("train") | report.metric.isin(
        ["three_zone_mean_ccc", "event_volume_ccc"])
    report.loc[excluded, ["value", "status"]] = [np.nan, "undefined"]
    selected, summary = rank_msdgc_pareto(report)
    assert selected == baseline[0] == 0
    np.testing.assert_array_equal(summary.aggregate_error, baseline[1].aggregate_error)
    np.testing.assert_array_equal(summary.passes_all_msd, baseline[1].passes_all_msd)


def test_boundaries_are_inclusive_and_passing_quality_does_not_break_ties():
    report = metric_report()
    for window in ("validation", "whole"):
        for metric in MSDGC_METRIC_ORDER[:3]:
            set_score(report, 0, window, metric, .9, "pass")
    set_score(report, 0, "validation", "con15_slope", .9, "pass")
    set_score(report, 0, "whole", "con15_slope", 1.1 + 5e-13, "pass")
    selected, summary = rank_msdgc_pareto(report.sample(frac=1, random_state=4))
    assert selected == 0
    assert summary.passes_all_msd.all()
    assert summary.aggregate_error.eq(0).all()


@pytest.mark.parametrize("value,status", [(np.nan, "undefined"), (np.inf, "pass"), (.95, "undefined")])
def test_undefined_required_check_cannot_win_or_be_skipped(value, status):
    report = metric_report()
    set_score(report, 0, "whole", "storage_ccc", value, status)
    set_score(report, 1, "whole", "storage_ccc", .8)
    selected, summary = rank_msdgc_pareto(report)
    assert selected == 1
    assert summary.loc[0, "checks_undefined"] == 1
    assert np.isinf(summary.loc[0, "aggregate_error"])
    assert not summary.passes_all_msd.any()


def test_no_defined_candidate_fails_explicitly():
    report = metric_report(1)
    set_score(report, 0, "validation", "con15_slope", np.nan, "undefined")
    with pytest.raises(RuntimeError, match="eight defined"):
        rank_msdgc_pareto(report)


def test_missing_or_duplicate_checks_fail_explicitly():
    report = metric_report(1)
    with pytest.raises(ValueError, match="unique"):
        rank_msdgc_pareto(pd.concat([report, report.iloc[:1]]))
    with pytest.raises(ValueError, match="missing"):
        rank_msdgc_pareto(report.iloc[:-1])


def prepared_case():
    observed = np.repeat([1., 2., 4., 8.], 49)
    data = pd.DataFrame(dict(
        datetime=pd.date_range("2025-01-01", periods=len(observed), freq="5min"),
        rainfall_in=np.zeros(len(observed)), temperature_c=10.,
    ))
    model = PatternWholeFlowModel(
        CenteredWeeklyPattern((0.,) * 24, (0.,) * 7), 20.26, 1 / 12, .6, 2.)
    model.initialize()
    model.prepare(data)
    events = pd.DataFrame(dict(
        event_id=["1", "2", "3", "4"], start_idx=[0, 49, 98, 147],
        end_idx=[48, 97, 146, 195], duration_hours=4.,
    ))
    train = np.arange(len(observed)) < 98
    evaluators = {}
    for window, mask in (("train", train), ("validation", ~train),
                         ("whole", np.ones(len(train), bool))):
        subset = events if window == "whole" else events.loc[
            events.start_idx.lt(98) if window == "train" else events.start_idx.ge(98)]
        evaluators[window] = MSDGCAssessment(
            observed, subset, mask, mask, 1 / 12, AcceptanceContract())
    result = CalibrationResult(
        [], np.array([[.6], [.8], [-1.]]), np.array([[-.8], [-.7], [-.9]]),
        ["sanitary_mean_cfs"], ["ccc"], [False])
    return result, model, evaluators


def test_scan_evaluates_every_vector_on_private_prepared_model():
    result, model, evaluators = prepared_case()
    baseline = model.predict_total_cfs().copy()
    before_x, before_f = result.pareto_X.copy(), result.pareto_F.copy()
    report = evaluate_msdgc_pareto(result, model, evaluators)
    assert len(report) == 3 * 3 * 6
    assert report.pareto_index.drop_duplicates().tolist() == [0, 1, 2]
    assert report.loc[report.pareto_index.eq(2), "status"].eq("undefined").all()
    assert report.loc[report.pareto_index.eq(2), "reason"].str.contains("constraints").all()
    assert not report.loc[report.pareto_index.eq(2), "model_feasible"].any()
    worker = copy.deepcopy(model)
    worker.apply_flat_parameter("sanitary_mean_cfs", .8)
    expected = evaluators["whole"].metrics(worker.predict_total_cfs()).set_index("metric")
    actual = report.query("pareto_index == 1 and window == 'whole'").set_index("metric")
    np.testing.assert_allclose(actual.loc[expected.index, "value"], expected.value, equal_nan=True)
    np.testing.assert_array_equal(model.predict_total_cfs(), baseline)
    assert model.get_scalar_parameter("sanitary_mean_cfs").value == .6
    np.testing.assert_array_equal(result.pareto_X, before_x)
    np.testing.assert_array_equal(result.pareto_F, before_f)


def test_notebook_report_cells_compile_and_keep_metric_order():
    path = Path(__file__).parents[1] / "sparsehydro" / "notebooks" / "flowfinity_gwi_wholeflow_pattern_joint.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = {cell["id"]: cell for cell in notebook["cells"]}
    for cell_id in ("pattern-joint-pareto-msd-spread", "pattern-joint-pareto-msd-scan"):
        compile("".join(cells[cell_id]["source"]), cell_id, "exec")
    assert MSDGC_METRIC_ORDER == (
        "conveyance_ccc", "storage_ccc", "treatment_ccc",
        "three_zone_mean_ccc", "event_volume_ccc", "con15_slope",
    )


def test_multicandidate_notebook_plots_scan_and_exports(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tests.pattern_workflow_support import prepared_fixture
    import plotly.graph_objects as go
    from sparsehydro.calibration.pattern_reporting import build_msdgc_diagnostic, export_pattern_tsf
    from sparsehydro.visualization.wholeflow_diagnostics import plot_msdgc_pareto, plot_msdgc_diagnostic
    from sparsehydro.visualization.display import add_val_vrects, thin_for_display

    result, model, evaluators = prepared_case()
    result = CalibrationResult(
        [], result.pareto_X[:2], np.array([[-.8, .2], [-.7, .1]]),
        result.param_names, ["ccc", "con15_error"], [False, True])
    report = metric_report()
    set_score(report, 0, "validation", "storage_ccc", .85)
    monkeypatch.setattr("sparsehydro.calibration.msdgc_pareto.evaluate_msdgc_pareto",
                        lambda *args: report.copy())
    monkeypatch.setattr(go.Figure, "show", lambda *args, **kwargs: None)
    mask = evaluators["validation"].mask
    times = pd.date_range("2025-01-01", periods=len(mask), freq="5min")
    data = pd.DataFrame(dict(datetime=times, rainfall_in=0., flow_cfs=evaluators["whole"].observed))
    before = model.predict_total_cfs().copy()
    joint_pred = model.predict()
    namespace = dict(
        np=np, pd=pd, go=go, joint_result=result, joint_model=model, report_evaluators=evaluators,
        joint_pred=joint_pred, MGD_TO_CFS=1e6 / (7.480519 * 86400.0),
        ARTIFACT_DIR=tmp_path, selected_index=0, obs_flow=evaluators["whole"].observed,
        warmup_mask=np.zeros(len(mask), bool), scored_mask=np.ones(len(mask), bool),
        model_df=data, val_mask=mask, DATA_TITLE="Synthetic test", DISPLAY_MAX_POINTS=100,
        display=lambda *args: None, thin_for_display=thin_for_display,
        add_val_vrects=lambda fig, t, **kwargs: add_val_vrects(fig, t, mask, **kwargs),
        fit=SimpleNamespace(result=result, selected_index=0),
        plot_msdgc_pareto=plot_msdgc_pareto, plot_msdgc_diagnostic=plot_msdgc_diagnostic,
        build_msdgc_diagnostic=build_msdgc_diagnostic, export_pattern_tsf=export_pattern_tsf,
    )
    prepared_fixture(namespace)
    path = Path(__file__).parents[1] / "sparsehydro" / "notebooks" / "flowfinity_gwi_wholeflow_pattern_joint.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell.get("id") in ("pattern-joint-pareto-msd-spread", "pattern-joint-pareto-msd-scan",
                              "pattern-joint-tsf-export"):
            exec(compile("".join(cell["source"]), cell["id"], "exec"), namespace)
    assert namespace["msd_diagnostic_index"] == 1
    assert namespace["selected_index"] == 0
    np.testing.assert_array_equal(model.predict_total_cfs(), before)
    assert namespace["msd_whole_spread"].metric.tolist() == list(MSDGC_METRIC_ORDER)
    assert len(namespace["msd_spread_fig"].data) == 6
    assert len(namespace["msd_front_fig"].data[0].customdata) == 2
    assert len(namespace["msd_diagnostic_fig"].data) == 3
    assert namespace["msd_diagnostic_fig"].layout.yaxis.autorange == "reversed"
    assert namespace["msd_diagnostic_fig"].layout.xaxis.matches == "x2"
    selection = json.loads((tmp_path / "msd_diagnostic_selection.json").read_text())
    assert selection["passing_count"] == 1
    assert selection["required_windows"] == ["validation", "whole"]
    assert len(pd.read_csv(tmp_path / "pareto_msdgc_metrics.csv")) == 36
    assert len(pd.read_csv(tmp_path / "pareto_msdgc_comparison.csv")) == 2
    from sparsehydro.data.utilities import read_tsf

    best_pred = namespace["_msd_model"].predict()
    assert not np.array_equal(best_pred.total_cfs, joint_pred.total_cfs)
    for suffix, predictions in (("", joint_pred), ("_best", best_pred)):
        for name, variable, column in (
                ("modeled_total_flow", "Flow", "total_cfs"),
                ("modeled_stormflow", "StormFlow", "rdii_cfs")):
            parsed = read_tsf(tmp_path / f"{name}{suffix}.tsf")
            np.testing.assert_array_equal(parsed.index, predictions.datetime)
            np.testing.assert_allclose(
                parsed[variable], predictions[column] / namespace["MGD_TO_CFS"],
                rtol=1e-11, atol=0)
