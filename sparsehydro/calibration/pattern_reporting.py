"""Post-fit diagnostics and reproducible exports for pattern whole-flow studies.

MSDGC comparison is report-only. The diagnostic model is always a private copy;
it never replaces the candidate selected from training objectives.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

import sparsehydro.models.wholeflow as _physics_module
import sparsehydro.models.wholeflow_pattern as _pattern_module
from sparsehydro.configuration import StudySettings
from sparsehydro.data.wholeflow import MGD_TO_CFS
from sparsehydro.data.utilities import write_tsf
from sparsehydro.models.wholeflow import _HAS_NUMBA, _gwi_B_core
from sparsehydro.models.wholeflow_pattern import PatternEstimate, PatternWholeFlowModel
from .msdgc_assessment import MSDGCAssessment
from .msdgc_pareto import MSDGC_METRIC_ORDER, rank_msdgc_pareto
from .pattern_workflow import PreparedPatternData, PatternCalibration, PatternEventCoverage
from .wholeflow_acceptance import AcceptanceContract
from .wholeflow_workflow import parameter_table, passing_pareto_metrics, hydrograph_diagnostics


@dataclass
class PatternReport:
    """Selected-candidate assessment tables and the evaluators for Pareto reports."""

    acceptance: pd.DataFrame
    event_metrics: pd.DataFrame
    scorecard: pd.DataFrame
    statuses: pd.DataFrame
    conveyance: pd.DataFrame
    zone_volumes: pd.DataFrame
    hydrograph_summary: pd.DataFrame
    evaluators: dict[str, MSDGCAssessment]
    metadata: dict = field(default_factory=dict)


@dataclass
class MSDGCDiagnostic:
    """Validation-informed diagnostic; not the calibration-selected model."""

    index: int
    status: str
    model: PatternWholeFlowModel
    flow: np.ndarray
    comparison: pd.DataFrame
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    metadata: dict


def assess_pattern_prediction(
    prepared: PreparedPatternData, coverage: PatternEventCoverage,
    prediction: pd.DataFrame, *, allow_overlapping_events: bool,
) -> PatternReport:
    """Assess one prediction without selecting candidates or writing artifacts."""
    started = time.perf_counter()
    observed, modeled = prepared.data["flow_cfs"].to_numpy(), prediction["total_cfs"].to_numpy()
    scores, events, bands, zones, evaluators = [], [], [], [], {}
    contract = AcceptanceContract(version=coverage.settings.version)
    for label, mask in (("train", prepared.train_mask), ("validation", prepared.val_mask),
                        ("whole", prepared.scored_mask)):
        inventory = (coverage.acceptance_events if label == "whole" else
                     coverage.acceptance_events.loc[coverage.acceptance_events["split"].eq(label)])
        evaluator = MSDGCAssessment(
            observed, inventory, mask, prepared.dry_eligible, prepared.split.dt_hours,
            contract, settings=coverage.settings, allow_overlapping_events=allow_overlapping_events)
        evaluators[label] = evaluator
        scores.append(evaluator.metrics(modeled).assign(window=label))
        events.append(evaluator.event_report(modeled).assign(window=label))
        if label != "whole":
            bands.append(evaluator.conveyance_report(modeled).assign(window=label))
            zones.append(evaluator.zone_report(modeled).assign(window=label))
    acceptance = pd.concat(scores, ignore_index=True)
    prepared.phase_seconds["criteria_checker"] = time.perf_counter() - started
    hydrograph_rows = []
    for label, all_rows, broad, approved in (
        ("train", prepared.train_mask, coverage.storm_mask, coverage.assessment_storm_mask),
        ("validation", prepared.val_mask, coverage.val_storm_mask, coverage.val_assessment_storm_mask),
        ("whole", prepared.scored_mask, coverage.event_mask,
         coverage.assessment_storm_mask | coverage.val_assessment_storm_mask),
    ):
        for domain, mask in (("all_quality_steps", all_rows), ("calibration_windows", broad),
                             ("report_event_windows", approved)):
            hydrograph_rows.append(dict(window=label, domain=domain,
                                        **hydrograph_diagnostics(observed[mask], modeled[mask])))
    return PatternReport(
        acceptance=acceptance, event_metrics=pd.concat(events, ignore_index=True),
        scorecard=acceptance.pivot(index="window", columns="metric", values="value"),
        statuses=acceptance.pivot(index="window", columns="metric", values="status"),
        conveyance=pd.concat(bands, ignore_index=True), zone_volumes=pd.concat(zones, ignore_index=True),
        hydrograph_summary=pd.DataFrame(hydrograph_rows), evaluators=evaluators)


def groundwater_diagnostics(model: PatternWholeFlowModel, prediction: pd.DataFrame, train_mask: np.ndarray, val_mask: np.ndarray, scored_mask: np.ndarray) -> pd.DataFrame:
    """Report G storage, release/spill shares and timescales without modifying parameters."""
    bounds_tbl = parameter_table(model)
    _g = model.get_scalar_parameter
    _k_g = float(_g("k_G").value)
    if model.groundwater_enabled:
        print(f"G slow-release e-fold time: {1 / (_k_g * 24):.2f} days; "
              f"half-life: {np.log(2) / (_k_g * 24):.2f} days (both assume no recharge)")
    else:
        print("Groundwater module disabled: no diversion, deep storage, or GWI discharge.")
    print("Near bounds:", ", ".join(bounds_tbl.loc[bounds_tbl.near_bound, "name"]))
    _gwi_rows = []
    for _split, _mask in (("train", train_mask), ("validation", val_mask), ("whole", scored_mask)):
        _p = prediction.loc[_mask]
        _release, _spill = float(_p.gwi_release_in.sum()), float(_p.gwi_spill_in.sum())
        _discharge = _release + _spill
        _gwi_rows.append(dict(window=_split, n=len(_p), G_min_in=float(_p.G_in.min()),
            G_max_in=float(_p.G_in.max()), at_cap_pct=100 * float(_p.G_at_cap.mean()),
            slow_release_in=_release, spill_in=_spill,
            spill_share_pct=100 * _spill / _discharge if _discharge > 0 else np.nan,
            status=("disabled" if not model.groundwater_enabled else
                    "defined" if _discharge > 0 else "undefined_no_discharge")))
    gwi_diagnostics = pd.DataFrame(_gwi_rows)
    print("A high spill share means G discharge mainly follows S-to-G diversion, not slow G recession.")
    print("Changing area scales GWI and RTK flow together; it does not itself change G depth or recession.")
    return gwi_diagnostics


def seasonal_dry_diagnostics(model: PatternWholeFlowModel, prediction: pd.DataFrame, data: pd.DataFrame, dry_mask: np.ndarray, val_dry_mask: np.ndarray, scored_mask: np.ndarray) -> pd.DataFrame:
    """Report monthly dry residuals and empty/full-store sensitivity without refitting."""
    _diagnostic_rows = []
    _slow_background = (prediction["slow_background_cfs"].to_numpy() if model.observed_background is not None
                        else np.full(len(data), float(model.get_scalar_parameter("sanitary_mean_cfs").value)))
    for _split, _mask in (("train", dry_mask), ("validation", val_dry_mask)):
        for _month in range(1, 13):
            _m = _mask & data["datetime"].dt.month.eq(_month).to_numpy()
            if not _m.any():
                _diagnostic_rows.append(dict(split=_split, month=_month, n=0,
                    status="undefined_no_dry_rows", observed_mean_cfs=np.nan, total_bias_cfs=np.nan,
                    sanitary_mean_cfs=np.nan, slow_background_mean_cfs=np.nan,
                    gwi_mean_cfs=np.nan, rdii_mean_cfs=np.nan))
                continue
            _diagnostic_rows.append(dict(split=_split, month=_month, n=int(_m.sum()), status="defined",
                observed_mean_cfs=float(data["flow_cfs"].to_numpy()[_m].mean()),
                total_bias_cfs=float((prediction["total_cfs"].to_numpy()[_m] - data["flow_cfs"].to_numpy()[_m]).mean()),
                sanitary_mean_cfs=float(data.loc[_m, "joint_sanitary_cfs"].mean()),
                slow_background_mean_cfs=float(_slow_background[_m].mean()),
                gwi_mean_cfs=float(data.loc[_m, "gwi_cfs"].mean()),
                rdii_mean_cfs=float(data.loc[_m, "rdii_cfs"].mean())))
    seasonal_diagnostics = pd.DataFrame(_diagnostic_rows)
    print("Months are reporting groups only; no monthly corrections are fitted.")
    if model.observed_background is not None:
        _missing_months = seasonal_diagnostics.loc[
            seasonal_diagnostics["split"].eq("train") & seasonal_diagnostics["n"].eq(0), "month"].tolist()
        if _missing_months:
            print("CAUTION: no training dry-weather observations in months:", _missing_months)
    _sensitivity_model = copy.deepcopy(model)
    _initial_names = ["s_initial_fraction"] + (["g_initial_fraction"] if model.groundwater_enabled else [])
    _initial_predictions = []
    for _fraction in (0.0, 1.0):
        for _name in _initial_names:
            _sensitivity_model.get_scalar_parameter(_name).value = _fraction
        _initial_predictions.append(_sensitivity_model.predict()["total_cfs"].to_numpy())
    print("Empty vs full active initial stores (initial SWE unchanged), max scored flow difference [cfs]:",
          float(np.max(np.abs(_initial_predictions[0][scored_mask] -
                              _initial_predictions[1][scored_mask]))))
    print("This is a fixed-parameter sensitivity check, not proof of warmup equilibration.")
    return seasonal_diagnostics


def export_pattern_report(prepared: PreparedPatternData, coverage: PatternEventCoverage, model: PatternWholeFlowModel, fit: PatternCalibration, study: StudySettings, artifact_dir: Path, *, shape_fit: PatternEstimate, odwf_cfs: float, seasonal_diagnostics: pd.DataFrame, gwi_diagnostics: pd.DataFrame, temperature_source: Path) -> PatternReport:
    """Compute report-only criteria and export tables, masks, settings and provenance."""
    artifact_dir = Path(artifact_dir)
    solver_metadata = dict(fit.solver_metadata)
    N_EVALUATIONS = solver_metadata.pop("requested_evaluations")
    _numba_signatures = getattr(_gwi_B_core, "nopython_signatures", ())
    _numba_options = getattr(_gwi_B_core, "targetoptions", {})
    passing_pareto_tbl = pd.DataFrame()
    acceptance_contract = AcceptanceContract(version=coverage.settings.version)
    report = assess_pattern_prediction(
        prepared, coverage, fit.prediction,
        allow_overlapping_events=study.workflow["events"]["EVENT_SOURCE"] in ("file", "hybrid"))
    acceptance_tbl, events_eval = report.acceptance, report.event_metrics
    conveyance_tbl, zone_volumes_tbl = report.conveyance, report.zone_volumes
    scorecard_msdgc, status_msdgc = report.scorecard, report.statuses
    report_evaluators, hydrograph_summary = report.evaluators, report.hydrograph_summary
    print("REPORT ONLY: no solver calls or solution selection in this checker.")
    print("MSDGC total-flow comparison: independent maximum-volume windows; configured assessment exclusions applied.")
    print("Not the official effective-storm scorecard; no Qcritical or overall grade is inferred.")
    print("Full-event-volume CCC is an additional diagnostic, not one of the three MSDGC zones.")
    print(scorecard_msdgc.round(4).to_string(na_rep="undefined"))
    print(status_msdgc.to_string())
    print("Individual passes do not override failures/undefined criteria; no full compliance claim.")
    print("Timestep metrics by domain; report_event_windows use the assessment inventory after exclusions:")
    if study.workflow["selection"]["USE_TRAINING_PEAK_FILTER"]:
        _started = time.perf_counter()
        passing_pareto_tbl = passing_pareto_metrics(
            fit.result, model, fit.pareto_solutions, report_evaluators)
        passing_pareto_tbl = passing_pareto_tbl.merge(fit.pareto_solutions[
            ["pareto_index", "primary_score", "primary_score_increase", "near_tie_eligible"]],
            on="pareto_index", validate="one_to_one")
        prepared.phase_seconds["passing_pareto_reporting"] = time.perf_counter() - _started
        print(f"# of Con15-filter acceptable solutions: {len(passing_pareto_tbl)} "
              f"of {len(fit.pareto_solutions)} final Pareto solutions")
        print("Acceptable here means the TRAINING conveyance-volume filter only; other metrics may fail or be undefined.")
        print("Threshold passes are report only; candidates outside the near-tie shortlist cannot be selected.")
        print(f"Filter top fraction={study.workflow['selection']['CON15_FILTER_FRACTION']:g}; standard reported Con15=0.15")
        if passing_pareto_tbl.empty:
            print("No passing solutions. The selected threshold-failing candidate is not included in this count/table.")
        else:
            _passing_columns = ["pareto_index", "selected", "primary_score", "primary_score_increase",
                "near_tie_eligible", "filter_peak_slope", "filter_peak_error"] + [
                f"{_split}_{_metric}" for _split in ("train", "validation", "whole")
                for _metric in MSDGCAssessment.report_metric_names]
            print(passing_pareto_tbl[_passing_columns].to_string(index=False, na_rep="undefined", float_format="{:.4f}".format))
        passing_pareto_tbl.to_csv(artifact_dir / "pareto_filter_pass_metrics.csv", index=False)

    _started = time.perf_counter()
    fit.prediction.to_csv(artifact_dir / "predictions.csv.gz", index=False)
    parameter_table(model).to_csv(artifact_dir / "parameters.csv", index=False)
    coverage.events.to_csv(artifact_dir / "event_inventory.csv", index=False)
    coverage.assessment_events.to_csv(artifact_dir / "assessment_event_inventory.csv", index=False)
    coverage.calibration_events.to_csv(artifact_dir / "calibration_event_inventory.csv", index=False)
    events_eval.to_csv(artifact_dir / "event_metrics.csv", index=False)
    hydrograph_summary.to_csv(artifact_dir / "hydrograph_scoring_summary.csv", index=False)
    fit.peak_objective_events.to_csv(artifact_dir / "peak_objective_events.csv", index=False)
    acceptance_tbl.to_csv(artifact_dir / "acceptance.csv", index=False)
    conveyance_tbl.to_csv(artifact_dir / "conveyance_event_volumes.csv", index=False)
    zone_volumes_tbl.to_csv(artifact_dir / "msdgc_zone_volumes.csv", index=False)
    seasonal_diagnostics.to_csv(artifact_dir / "seasonal_dry_weather.csv", index=False)
    gwi_diagnostics.to_csv(artifact_dir / "gwi_release_spill_diagnostics.csv", index=False)
    shape_fit.coverage.to_csv(artifact_dir / "shape_coverage.csv", index=False)
    pd.DataFrame(dict(datetime=prepared.data.datetime, warmup=prepared.warmup_mask, train=prepared.train_mask,
        split_train=prepared.split_train_mask, split_validation=prepared.split_val_mask,
        observation_eligible=prepared.observation_eligible,
        assessment_storm=coverage.assessment_storm_mask, validation_assessment_storm=coverage.val_assessment_storm_mask,
        approved_storm=coverage.approved_storm_mask, validation_approved_storm=coverage.val_approved_storm_mask,
        validation=prepared.val_mask, dry=prepared.dry_eligible, storm=coverage.storm_mask,
        validation_storm=coverage.val_storm_mask, precipitation_storm=coverage.rain_storm_mask,
        validation_precipitation_storm=coverage.val_rain_storm_mask,
        melt=coverage.melt_mask, validation_melt=coverage.val_melt_mask,
        possible_melt=prepared.data["possible_melt"])).to_csv(
            artifact_dir / "scoring_masks.csv.gz", index=False)
    fit.result.to_pareto_dataframe().to_csv(artifact_dir / "objective_history.csv", index=False)
    prepared.phase_seconds["exports"] = time.perf_counter() - _started
    run_metadata = dict(
        model="centered-pattern + IA/RTK with optional GWI/snow", python=platform.python_version(),
        groundwater_enabled=model.groundwater_enabled, n_rtk_triangles=model.n_triangles,
        seasonal_baseline_mode=study.model.seasonal_baseline,
        observed_background_enabled=model.observed_background is not None,
        background_use="retrospective interpolation with bounded endpoint holds, not forecasting"
            if model.observed_background is not None else "forcing-driven",
        background_endpoint_hold_days=study.model.background_endpoint_hold_days,
        background_endpoint_held_scored_rows=int((fit.prediction["background_endpoint_held"].to_numpy() & prepared.scored_mask).sum())
            if model.observed_background is not None else 0,
        background_source="eligible training dry-weather flow only; no calendar cycle"
            if model.observed_background is not None else "jointly fitted weekly mean/amplitudes",
        baseline_interpretation="empirical background including unidentified slow inflow"
            if model.observed_background is not None else "centered weekly pattern plus optional GWI",
        study_settings_schema_version=study.schema_version,
        **solver_metadata, scoring_mode=study.calibration.scoring_mode,
        requested_evaluations=N_EVALUATIONS, selected_index=fit.selected_index,
        selection=fit.selection_rule, selection_status=fit.selection_status,
        objective_names=fit.result.objective_names, objective_minimize_flags=fit.result.minimize_flags,
        selected_objectives=dict(zip(fit.result.objective_names, fit.verified_objectives.tolist())),
        con15_objective_enabled=fit.peak_objective is not None,
        peak_top_fraction=study.workflow["selection"]["CON15_OBJ_FRACTION"], peak_objective_mask="complete training assessment storms",
        con15_basis="msdgc_independent_conveyance_volume_mg",
        peak_training_event_count=len(fit.peak_objective_events),
        peak_selected_training_event_count=int(fit.peak_objective_events["objective_selected"].sum()),
        peak_selected_training_event_ids=fit.peak_objective_events.loc[
            fit.peak_objective_events["objective_selected"], "event_id"].astype(str).tolist(),
        peak_objective_loss="abs(zero-intercept modeled/observed conveyance-volume slope - 1)",
        selected_peak_slope=fit.peak_objective.slope(prepared.data["flow_cfs"].to_numpy(), fit.prediction["total_cfs"].to_numpy()) if fit.peak_objective is not None else None,
        peak_filter=dict(enabled=bool(study.workflow["selection"]["USE_TRAINING_PEAK_FILTER"]), top_fraction=study.workflow["selection"]["CON15_FILTER_FRACTION"],
            role="soft near-tie conveyance-volume preference; pass threshold report only",
            basis="msdgc_independent_conveyance_volume_mg",
            max_error=study.workflow["selection"]["MAX_TRAINING_PEAK_ERROR"], domain="complete training assessment storms only",
            selected_training_event_count=int(fit.peak_objective_events["filter_selected"].sum()),
            selected_training_event_ids=fit.peak_objective_events.loc[
                fit.peak_objective_events["filter_selected"], "event_id"].astype(str).tolist(),
            passing_candidate_count=int(fit.pareto_solutions["filter_pass"].fillna(False).sum()) if fit.peak_filter is not None else None,
            selected_passes=bool(fit.pareto_solutions.loc[fit.selected_index, "filter_pass"]) if fit.peak_filter is not None else None,
            selected_slope=float(fit.pareto_solutions.loc[fit.selected_index, "filter_peak_slope"]) if fit.peak_filter is not None else None,
            selected_error=float(fit.pareto_solutions.loc[fit.selected_index, "filter_peak_error"]) if fit.peak_filter is not None else None,
            no_pass_policy="warn if selected candidate fails; retain primary-score bound, never expand shortlist"),
        near_tie_selection=dict(enabled=bool(study.workflow["selection"]["USE_TRAINING_PEAK_FILTER"]), allowance=study.workflow["selection"]["CON15_NEAR_TIE_ALLOWANCE"],
            primary_objectives=fit.pareto_solutions.attrs["primary_objective_names"] if fit.peak_filter is not None else None,
            score="RMS distance to ideal after min-max normalization on feasible final candidates",
            primary_baseline_index=int(fit.pareto_solutions.index[fit.pareto_solutions["primary_baseline"]][0]) if fit.peak_filter is not None else None,
            best_primary_score=float(fit.pareto_solutions["primary_score"].min()) if fit.peak_filter is not None else None,
            selected_primary_score=float(fit.pareto_solutions.loc[fit.selected_index, "primary_score"]) if fit.peak_filter is not None else None,
            selected_primary_score_increase=float(fit.pareto_solutions.loc[fit.selected_index, "primary_score_increase"]) if fit.peak_filter is not None else None,
            shortlisted_candidate_count=int(fit.pareto_solutions["near_tie_eligible"].sum()) if fit.peak_filter is not None else None),
        final_pareto_count=len(fit.pareto_solutions), final_pareto_file="pareto_solutions.csv",
        conveyance_report=dict(role="report only", placement="independent maximum-volume windows",
            duration_rule="clamp(0.042 * rounded duration, 15min, 30min); nearest native samples",
            flow="total_flow", aggregation="CCC across event volume pairs",
            integration="native full-bin volumes; moving sum rounded to 0.0001 MG",
            incomplete_bands="undefined, not clipped or silently dropped"),
        msdgc_assessment=dict(**asdict(coverage.settings), flow_basis="total_flow",
            official_storm_scorecard=False, qcritical="omitted: no capacity or official storm inputs",
            excluded_event_ids=list(study.workflow["events"]["ASSESSMENT_EXCLUDED_EVENT_IDS"]),
            applied_excluded_event_ids=coverage.assessment_events.loc[
                coverage.assessment_events.exclusion.str.contains("assessment_criteria_exclusion"),
                "event_id"].astype(str).tolist(),
            assessment_event_count=len(coverage.acceptance_events),
            assessment_window_source="full_evt" if study.workflow["events"]["EVENT_SOURCE"] == "hybrid" else "approved",
            inventory_file="assessment_event_inventory.csv", volumes_file="msdgc_zone_volumes.csv",
            calibration_coverage_changed=False),
        passing_pareto_report=dict(enabled=bool(study.workflow["selection"]["USE_TRAINING_PEAK_FILTER"]),
            count=len(passing_pareto_tbl) if study.workflow["selection"]["USE_TRAINING_PEAK_FILTER"] else None,
            file="pareto_filter_pass_metrics.csv" if study.workflow["selection"]["USE_TRAINING_PEAK_FILTER"] else None,
            scope="all filter-passing final candidates; train/validation/whole report only"),
        checker_role="report only", optimized_parameters=fit.result.param_names,
        calibration_prediction="total_cfs array; full physics and budget checks",
        empirical_shape_degrees_of_freedom=29, pattern=asdict(shape_fit.pattern),
        snow_enabled=model.snow_config is not None,
        snow_initial_settings=asdict(study.model.snow_config) if study.model.snow_config is not None else None,
        snow_selected_settings=asdict(model.snow_config) if model.snow_config is not None else None,
        snow_calibrated=bool(study.model.calibrate_snow),
        snow_bounds={n: [model.get_scalar_parameter(n).lower_bound,
                         model.get_scalar_parameter(n).upper_bound]
                     for n in ("snow_T", "snow_ddf")} if study.model.calibrate_snow else None,
        snow_parameter_treatment="joint bounded threshold/melt factor; initial SWE fixed" if study.model.calibrate_snow else "fixed or disabled",
        snow_mask_treatment="frozen conservative bounded envelope" if study.model.calibrate_snow else "fixed snow ledger",
        event_source=study.workflow["events"]["EVENT_SOURCE"], input_events=str(prepared.event_file) if prepared.event_file is not None else None,
        input_calibration_events=str(prepared.all_event_file) if prepared.all_event_file is not None else None,
        calibration_event_count=int(coverage.calibration_events.eligible.sum()),
        calibration_padding_hours=dict(pre=coverage.pre_hours, tail=coverage.tail_hours),
        data_issue_excluded_rows=int((~prepared.observation_eligible & ~prepared.warmup_mask).sum()),
        approved_event_role="peak objectives/preferences and guideline reports" if study.workflow["events"]["EVENT_SOURCE"] == "hybrid" else "event scoring and reports",
        event_overlap_policy=("independent approved windows" if study.workflow["events"]["EVENT_SOURCE"] in ("file", "hybrid")
                              else "ordered disjoint windows"),
        event_definition=("reliable EVT intervals + padding + possible melt for CCC; approved exact intervals for peaks/reports; Data Issues unscored"
            if study.workflow["events"]["EVENT_SOURCE"] == "hybrid" else
            "approved file intervals; inclusive samples; no padding or extra melt windows"
            if study.workflow["events"]["EVENT_SOURCE"] == "file" else
            "measured precipitation MSD events unchanged; melt windows supplement objectives only"),
        dry_definition="no precipitation or possible melt in current/preceding exclusion window; hybrid additionally excludes EVT Data Issues; frozen before optimization",
        acceptance=asdict(acceptance_contract), odwf_training_cfs=odwf_cfs,
        budget=model.budget, phase_seconds=prepared.phase_seconds,
        physics_module=str(Path(_physics_module.__file__).resolve()),
        pattern_module=str(Path(_pattern_module.__file__).resolve()),
        numba=importlib.metadata.version("numba") if _HAS_NUMBA else "unavailable",
        numba_compiled=bool(_numba_signatures), numba_nogil=bool(_numba_options.get("nogil", False)),
        numba_fastmath=bool(_numba_options.get("fastmath", False)),
        timestep=study.workflow["data"]["TIMESTEP"], area_acres=float(model.get_scalar_parameter("area_acres").value), area_acres_initial=study.workflow["simulation"]["AREA_ACRES"],
        area_calibrated=bool(model.get_scalar_parameter("area_acres").calibrate),
        split=prepared.split.describe(),
        warmup_days=study.workflow["simulation"]["WARMUP_DAYS"], spinup_mode=study.workflow["simulation"]["SPINUP_MODE"],
        input_flow=str(study.workflow["data"]["FLOW_FILE"]), input_rain=str(study.workflow["data"]["RAIN_FILE"]), temperature_source=str(temperature_source),
        timezone="fixed EST; source TSF timezone not independently verified",
        model_input_fingerprint=str(pd.util.hash_pandas_object(
            prepared.data[prepared.forcing_columns + ["flow_cfs"]], index=False).sum()),
        limitations=["observed background is retrospective interpolation, not future-year prediction; components not uniquely identified",
                     "effective contributing area trades off with RTK fractions and S/G partition",
                     "G-store discharge includes capacity spill; inspect slow-release/spill split",
                     "MSDGC total-flow zone formulas; not the official effective-storm scorecard",
                     "snow assumes precipitation is total liquid-equivalent depth; no gauge-undercatch correction",
                     "degree-day snow; no sublimation, refreezing or frozen-soil permeability model",
                     "zero lower melt bound can retain potential SWE indefinitely and severely reduce dry coverage",
                      ("MSD precipitation events may miss delayed rain-free melt; inspect held-out all-flow CCC and snow plot"
                       if study.workflow["events"]["EVENT_SOURCE"] == "auto" else
                       "hybrid CCC includes possible melt outside approved windows" if study.workflow["events"]["EVENT_SOURCE"] == "hybrid" else
                       "file event approval is external; rain-free melt is scored only inside supplied windows"),
                      "hybrid excludes Data Issues observations only; any forcing errors or downstream effects require separate quality review",
                      "event exclusions are in event_inventory.csv"])
    (artifact_dir / "study_settings.json").write_text(
        json.dumps(study.to_dict(), indent=2, allow_nan=False), encoding="utf-8")
    if model.observed_background is not None:
        (artifact_dir / "observed_background.json").write_text(
            json.dumps(model.observed_background.to_dict(), indent=2, allow_nan=False), encoding="utf-8")
    (artifact_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    print("Phase timings [s]:", prepared.phase_seconds)
    print("Saved separate run artifacts:", artifact_dir)
    report.metadata = run_metadata
    return report


def build_msdgc_diagnostic(fit: PatternCalibration, model: PatternWholeFlowModel, metrics: pd.DataFrame, prepared: PreparedPatternData, artifact_dir: Path) -> MSDGCDiagnostic:
    """Export the all-candidate pass scan and simulate its diagnostic choice on a private model."""
    msd_diagnostic_index, pareto_msdgc_comparison = rank_msdgc_pareto(metrics)
    pareto_msdgc_comparison['calibration_selected'] = pareto_msdgc_comparison.pareto_index.eq(fit.selected_index)
    pareto_msdgc_comparison.to_csv(artifact_dir / 'pareto_msdgc_comparison.csv', index=False)
    _msd_passes = int(pareto_msdgc_comparison.passes_all_msd.sum())
    _msd_selected = pareto_msdgc_comparison.loc[
        pareto_msdgc_comparison.diagnostic_selected].iloc[0]
    _msd_choice_status = str(_msd_selected.diagnostic_selection_status)
    _msd_summary_columns = ['pareto_index', 'calibration_selected', 'diagnostic_selected',
        'passes_all_msd', 'checks_passed', 'checks_failed', 'checks_undefined', 'aggregate_error']

    msd_diagnostic_metrics = metrics.loc[
        metrics.pareto_index.eq(msd_diagnostic_index)].copy()
    msd_diagnostic_metrics['metric'] = pd.Categorical(
        msd_diagnostic_metrics.metric, categories=MSDGC_METRIC_ORDER, ordered=True)
    msd_diagnostic_metrics = msd_diagnostic_metrics.sort_values(['window', 'metric'])
    msd_diagnostic_metrics.to_csv(artifact_dir / 'msd_diagnostic_selected_metrics.csv', index=False)

    _msd_model = copy.deepcopy(model)
    for _name, _value in zip(fit.result.param_names, fit.result.pareto_X[msd_diagnostic_index]):
        _msd_model.apply_flat_parameter(_name, float(_value))
    msd_diagnostic_flow = _msd_model.predict_total_cfs()
    parameter_table(_msd_model).to_csv(artifact_dir / 'msd_diagnostic_selected_parameters.csv', index=False)
    _msd_record = ~prepared.warmup_mask
    _msd_time = pd.to_datetime(prepared.data.datetime)
    _msd_observed = np.where(prepared.scored_mask, prepared.data["flow_cfs"].to_numpy(), np.nan)
    msd_diagnostic_predictions = pd.DataFrame(dict(
        datetime=_msd_time, observed_cfs=_msd_observed, modeled_cfs=msd_diagnostic_flow,
        rainfall_in=prepared.data.rainfall_in, validation=prepared.val_mask, observation_scored=prepared.scored_mask))
    msd_diagnostic_predictions.loc[_msd_record].to_csv(
        artifact_dir / 'msd_diagnostic_selected_predictions.csv.gz', index=False)
    _msd_scan_metadata = dict(
        role='post-hoc validation-informed diagnostic; not calibration selection or official storm compliance',
        diagnostic_index=msd_diagnostic_index, calibration_selected_index=int(fit.selected_index),
        selection_status=_msd_choice_status, passing_count=_msd_passes,
        final_pareto_count=len(pareto_msdgc_comparison),
        required_windows=['validation', 'whole'],
        required_metrics=['conveyance_ccc', 'storage_ccc', 'treatment_ccc', 'con15_slope'],
        aggregate_error=float(_msd_selected.aggregate_error),
        aggregate_formula='sum(max(lower-value, value-upper, 0) / 0.10); eight checks; undefined=infinity',
        tie_break='lowest original Pareto index', official_storm_scorecard=False)
    (artifact_dir / 'msd_diagnostic_selection.json').write_text(
        json.dumps(_msd_scan_metadata, indent=2, allow_nan=False), encoding='utf-8')
    return MSDGCDiagnostic(msd_diagnostic_index, _msd_choice_status, _msd_model,
                           msd_diagnostic_flow, pareto_msdgc_comparison,
                           msd_diagnostic_metrics, msd_diagnostic_predictions, _msd_scan_metadata)


def export_pattern_tsf(prepared: PreparedPatternData, prediction: pd.DataFrame, diagnostic_prediction: pd.DataFrame, artifact_dir: Path, data_title: str) -> tuple[dict[str, Path], dict[str, Path]]:
    """Export selected and separate diagnostic total/RDII flow in MGD, excluding warmup."""
    _diagnostic_prediction = diagnostic_prediction
    _export_warmup = np.asarray(prepared.warmup_mask)
    if _export_warmup.dtype != bool or _export_warmup.shape != (len(prepared.data),):
        raise ValueError("warmup_mask must be a boolean array aligned with model_df")
    tsf_paths = {}
    best_tsf_paths = {}
    for _suffix, _prediction, _paths in (
            ("", prediction, tsf_paths),
            ("_best", _diagnostic_prediction, best_tsf_paths)):
        if not _prediction["datetime"].reset_index(drop=True).equals(
                prepared.data["datetime"].reset_index(drop=True)):
            raise ValueError(f"TSF{_suffix} prediction timestamps must align with model_df")
        _tsf_predictions = _prediction.loc[~_export_warmup,
            ["datetime", "total_cfs", "rdii_cfs"]].set_index("datetime") / MGD_TO_CFS
        if _tsf_predictions.empty or not np.isfinite(_tsf_predictions.to_numpy()).all():
            raise ValueError(f"TSF{_suffix} export requires nonempty finite predictions after warmup")
        for _column, _variable, _filename in (
                ("total_cfs", "Flow", f"modeled_total_flow{_suffix}.tsf"),
                ("rdii_cfs", "StormFlow", f"modeled_stormflow{_suffix}.tsf")):
            _path = artifact_dir / _filename
            _title_suffix = " (best available MSD diagnostic)" if _suffix else ""
            write_tsf(_tsf_predictions[_column], _path,
                      title=f"{data_title} Modeled {_variable}{_title_suffix}",
                      variable=_variable, unit="MGD")
            _paths[_variable] = _path
            print(f"{_variable}: {len(_tsf_predictions):,} rows in MGD; saved {_path}")
        print(f"TSF{_suffix} period (warmup excluded): {_tsf_predictions.index[0]} -> "
              f"{_tsf_predictions.index[-1]}")
    return tsf_paths, best_tsf_paths
