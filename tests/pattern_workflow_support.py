"""Shared fixtures for package-level tests migrated from inline notebook code."""

import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from sparsehydro.calibration.pattern_workflow import (
    PreparedPatternData, PatternEventCoverage, build_pattern_event_coverage,
    build_pattern_model, build_pattern_objectives, calibrate_pattern_model,
    prepare_pattern_data, study_settings_from_controls,
)
from sparsehydro.calibration.pattern_reporting import assess_pattern_prediction, export_pattern_report
from sparsehydro.calibration.msdgc_assessment import MSDGCSettings
from sparsehydro.calibration.splits import SplitConfig
from sparsehydro.calibration.wholeflow_workflow import parameter_table
from sparsehydro.models.wholeflow_pattern import CenteredWeeklyPattern, PatternEstimate


NOTEBOOK = Path(__file__).parents[1] / "sparsehydro" / "notebooks" / "flowfinity_gwi_wholeflow_pattern_joint.ipynb"


@lru_cache(maxsize=1)
def _default_controls():
    namespace = {}
    for cell in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]:
        if cell["cell_type"] == "code" and cell["id"].startswith("pattern-settings-"):
            exec("".join(cell["source"]), namespace)
    return {name: value for name, value in namespace.items() if not name.startswith("__")}


def controls():
    return deepcopy(_default_controls())


def configure_state(namespace):
    values = {**controls(), **namespace}
    values = {name: str(value) if isinstance(value, Path) else value for name, value in values.items()}
    study = study_settings_from_controls(values)
    namespace.update(study_settings=study, SNOW_CONFIG=study.model.snow_config,
                     SNOW_CALIBRATION_ACTIVE=study.model.calibrate_snow)
    return study


def prepared_fixture(namespace):
    data = namespace["model_df"]
    size = len(data)
    if "dry_eligible" not in data:
        data["dry_eligible"] = True
    train = namespace.get("train_mask", np.ones(size, bool))
    val = namespace.get("val_mask", ~train)
    split = namespace.get("SPLIT_CFG", SplitConfig(
        dt_hours=namespace.get("_dt_hours", namespace.get("DT_HOURS", 1.)),
        warmup_days=0, spinup_mode="none"))
    previous = namespace.get("prepared")
    prepared = PreparedPatternData(
        data, split, namespace.get("DOWNLOAD_DIR", Path.cwd()),
        namespace.get("warmup_mask", np.zeros(size, bool)), train, val,
        namespace.get("split_train_mask", train.copy()),
        namespace.get("split_val_mask", val.copy()),
        namespace.get("observation_eligible", np.ones(size, bool)),
        namespace.get("_spinup_steps", 0), namespace.get("_warmup_steps", 0),
        event_file=namespace.get("_event_file"), all_event_file=namespace.get("_all_event_file"),
        events=namespace.get("events_df"), calibration_events=namespace.get("calibration_events_df"),
        phase_seconds=namespace.get("phase_seconds", {}))
    if previous is not None:
        if "events_df" not in namespace:
            prepared.events = previous.events
        if "calibration_events_df" not in namespace:
            prepared.calibration_events = previous.calibration_events
        if "_event_file" not in namespace:
            prepared.event_file = previous.event_file
        if "_all_event_file" not in namespace:
            prepared.all_event_file = previous.all_event_file
    namespace["prepared"] = prepared
    return prepared


def coverage_fixture(namespace):
    prepared = namespace["prepared"]
    size = len(prepared.data)
    empty = pd.DataFrame(columns=["event_id", "start_idx", "end_idx", "duration_hours", "split"])
    accepted = namespace.get("acceptance_events", empty)
    events = namespace.get("events_df", accepted)
    storm = namespace.get("storm_mask", prepared.train_mask)
    val_storm = namespace.get("val_storm_mask", prepared.val_mask)
    zeros = np.zeros(size, bool)
    coverage = PatternEventCoverage(
        events, namespace.get("calibration_events_df", events),
        namespace.get("assessment_events_df", accepted), accepted, storm, val_storm,
        namespace.get("rain_storm_mask", storm), namespace.get("val_rain_storm_mask", val_storm),
        namespace.get("melt_mask", zeros), namespace.get("val_melt_mask", zeros),
        namespace.get("approved_storm_mask", storm), namespace.get("val_approved_storm_mask", val_storm),
        namespace.get("assessment_storm_mask", storm), namespace.get("val_assessment_storm_mask", val_storm),
        namespace.get("PRE_EVENT_BUFF_H", 3.), namespace.get("RECESSION_TAIL_H", 48.),
        namespace.get("msdgc_settings", MSDGCSettings()))
    namespace["coverage"] = coverage
    return coverage


def shape_fixture(namespace):
    fit = namespace.get("shape_fit")
    pattern = namespace.get("sanitary_pattern", CenteredWeeklyPattern((0.,) * 24, (0.,) * 7))
    return PatternEstimate(
        pattern, getattr(fit, "detection_baseline_cfs", np.zeros(len(namespace["model_df"]))),
        getattr(fit, "coverage", pd.DataFrame()))


def build_model_state(namespace):
    study = configure_state(namespace)
    prepared = prepared_fixture(namespace)
    shape = shape_fixture(namespace)
    model = build_pattern_model(prepared, shape, namespace["odwf_cfs"], study)
    namespace.update(
        joint_model=model, FORCING_COLUMNS=prepared.forcing_columns, shape_fit=shape,
        sanitary_start=max(.75 * namespace["odwf_cfs"],
                           -min(shape.pattern.hourly) - min(shape.pattern.weekday) + 1e-6),
        sanitary_upper=study.workflow["simulation"]["SANITARY_UPPER_MULTIPLIER"] * namespace["odwf_cfs"])
    return model


def prepare_data_state(namespace):
    study = configure_state(namespace)
    prepared = prepare_pattern_data(
        namespace["model_df"], study, namespace["SPLIT_CFG"], namespace.get("DOWNLOAD_DIR", Path.cwd()))
    namespace.update(
        prepared=prepared, model_df=prepared.data, warmup_mask=prepared.warmup_mask,
        train_mask=prepared.train_mask, val_mask=prepared.val_mask, scored_mask=prepared.scored_mask,
        _warmup_steps=prepared.warmup_steps, _spinup_steps=prepared.spinup_steps,
        _dt_hours=prepared.split.dt_hours, phase_seconds=prepared.phase_seconds,
        observation_eligible=prepared.observation_eligible, _event_file=prepared.event_file,
        _all_event_file=prepared.all_event_file, dry_eligible=prepared.dry_eligible)
    if prepared.events is not None:
        namespace["events_df"] = prepared.events
        namespace["calibration_events_df"] = prepared.calibration_events
    return prepared


def event_coverage_state(namespace):
    study = configure_state(namespace)
    prepared = prepared_fixture(namespace)
    coverage = build_pattern_event_coverage(prepared, shape_fixture(namespace).detection_baseline_cfs, study)
    namespace.update(
        coverage=coverage, events_df=coverage.events, calibration_events_df=coverage.calibration_events,
        assessment_events_df=coverage.assessment_events, acceptance_events=coverage.acceptance_events,
        _event_file=prepared.event_file, _all_event_file=prepared.all_event_file,
        dry_mask=prepared.train_mask & prepared.dry_eligible,
        val_dry_mask=prepared.val_mask & prepared.dry_eligible,
        _events=list(coverage.acceptance_events.itertuples()))
    for name in ("storm_mask", "val_storm_mask", "event_mask", "rain_storm_mask", "val_rain_storm_mask",
                 "melt_mask", "val_melt_mask", "approved_storm_mask", "val_approved_storm_mask",
                 "assessment_storm_mask", "val_assessment_storm_mask"):
        namespace[name] = getattr(coverage, name)
    return coverage


def objective_state(namespace):
    study = configure_state(namespace)
    prepared = prepared_fixture(namespace)
    coverage = coverage_fixture(namespace)
    definitions = build_pattern_objectives(prepared, coverage, study)
    namespace.update(
        peak_objective=definitions.peak_objective, peak_filter=definitions.peak_filter,
        peak_objective_events=definitions.events, training_peak_events=definitions.events,
        calibration_objectives=definitions.objectives)
    return definitions


def calibrate_state(namespace):
    study = configure_state(namespace)
    prepared = prepared_fixture(namespace)
    coverage = coverage_fixture(namespace)
    fit = calibrate_pattern_model(namespace["joint_model"], prepared, coverage, study, namespace["ARTIFACT_DIR"])
    verified = [objective.compute(prepared.data.flow_cfs.to_numpy(), fit.prediction.total_cfs.to_numpy())
                for objective in study.calibration.build_objectives(
                    prepared.train_mask, coverage.storm_mask, con15=fit.peak_objective)]
    namespace.update(
        fit=fit, joint_result=fit.result, joint_pred=fit.prediction, solver2=fit.solver,
        joint_problem=fit.problem, calibration_config=study.calibration,
        solver_metadata={key: value for key, value in fit.solver_metadata.items() if key != "requested_evaluations"},
        N_EVALUATIONS=fit.solver_metadata["requested_evaluations"],
        solver_label=fit.solver_metadata["solver"], peak_objective=fit.peak_objective,
        peak_filter=fit.peak_filter, peak_objective_events=fit.peak_objective_events,
        pareto_solutions=fit.pareto_solutions, selected_index=fit.selected_index,
        selection_status=fit.selection_status, selection_rule=fit.selection_rule,
        _objectives=fit.result.objective_display_values()[fit.selected_index], _verified_objectives=verified,
        q_total=fit.prediction.total_cfs.to_numpy(), obs_flow=prepared.data.flow_cfs.to_numpy(),
        parameter_table=parameter_table)
    return fit


def assessment_state(namespace):
    prepared = prepared_fixture(namespace)
    coverage = coverage_fixture(namespace)
    report = assess_pattern_prediction(
        prepared, coverage, pd.DataFrame({"total_cfs": namespace["q_total"]}),
        allow_overlapping_events=namespace["EVENT_SOURCE"] in ("file", "hybrid"))
    capture_report(namespace, report)
    return report


def report_state(namespace):
    study = configure_state(namespace)
    prepared = prepared_fixture(namespace)
    coverage = coverage_fixture(namespace)
    report = export_pattern_report(
        prepared, coverage, namespace["joint_model"], namespace["fit"], study, namespace["ARTIFACT_DIR"],
        shape_fit=shape_fixture(namespace), odwf_cfs=namespace["odwf_cfs"],
        seasonal_diagnostics=namespace["seasonal_diagnostics"], gwi_diagnostics=namespace["gwi_diagnostics"],
        temperature_source=namespace["_air_csv"])
    capture_report(namespace, report)
    return report


def capture_report(namespace, report):
    namespace.update(report=report, acceptance_tbl=report.acceptance, events_eval=report.event_metrics,
                     conveyance_tbl=report.conveyance, zone_volumes_tbl=report.zone_volumes,
                     report_evaluators=report.evaluators, hydrograph_summary=report.hydrograph_summary)
