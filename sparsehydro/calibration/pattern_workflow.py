"""Preparation and joint fitting for the centered-pattern whole-flow workflow.

The notebook owns editable controls and display. These functions own reusable
calculations; all observation-derived fits use the quality-eligible training mask.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from sparsehydro.configuration import CalibrationSettings, ModelSettings, ParameterSettings, StudySettings
from sparsehydro.events.imported import imported_event_inventory, prepare_hybrid_events
from sparsehydro.models.rdii.initial_abstraction import snow_partition
from sparsehydro.models.wholeflow import _HAS_NUMBA, _gwi_B_core
from sparsehydro.models.wholeflow_pattern import (
    PatternEstimate, PatternWholeFlowModel, estimate_observed_background,
)

from .msdgc_assessment import MSDGCAssessment, MSDGCCon15, MSDGCSettings, assessment_event_inventory
from .objectives import ConcordanceCorrelationCoefficient, IObjective, NashSutcliffe
from .result import CalibrationResult
from .solvers.base import ISolver
from .splits import SplitConfig, build_masks, prepend_spinup
from .wholeflow_acceptance import AcceptanceContract
from .wholeflow_workflow import (
    FeasibleCalibrationProblem, dry_weather_mask, frozen_events, melt_scoring_mask,
    select_peak_filtered_pareto, snow_melt_envelope, storm_scoring_mask,
)


WORKFLOW_FIELDS = {
    "events": ("EVENT_SOURCE", "EVENT_FILE", "APPROVED_EVENTS_FILE", "CALIBRATION_PRE_HOURS",
               "CALIBRATION_TAIL_HOURS", "ASSESSMENT_EXCLUDED_EVENT_IDS",
               "STORM_DETECTION_SETTINGS"),
    "data": ("DATA_SOURCE", "DATA_TITLE", "SOURCE_DIR", "RAIN_FILE", "FLOW_FILE",
             "BASEFLOW_FILE", "STORMFLOW_FILE", "RAIN_COLUMN", "FLOW_COLUMN",
             "BASEFLOW_COLUMN", "STORMFLOW_COLUMN", "FILE_START", "FILE_END", "TIMESTEP",
             "TEMPERATURE_MAX_GAP_HOURS", "NAME_FILTER", "FLOWFINITY_RECORD_TITLE",
             "FLOWFINITY_RECORD_INDEX", "TYPES", "START", "END"),
    "simulation": ("WARMUP_DAYS", "SPINUP_MODE", "AREA_ACRES", "DRY_EXCLUSION_HOURS",
                   "SANITARY_UPPER_MULTIPLIER", "REQUIRE_NUMBA"),
    "holdout": ("SPLIT_MODE", "TRAIN_FRAC", "STRAT_BLOCK_DAYS", "VAL_EVERY", "SNAP_EDGES_TO_DRY",
                "EDGE_DRY_HOURS", "EDGE_SNAP_MAX_DAYS", "RAIN_EPS_IN"),
    "selection": ("CON15_OBJ_FRACTION", "USE_TRAINING_PEAK_FILTER", "CON15_NEAR_TIE_ALLOWANCE",
                  "CON15_FILTER_FRACTION", "MAX_TRAINING_PEAK_ERROR"),
    "reporting": ("DISPLAY_MAX_POINTS",),
}


def _validate_peak_preference(controls: Mapping[str, Any]) -> None:
    if not controls["USE_TRAINING_PEAK_FILTER"]:
        return
    max_error, allowance = controls["MAX_TRAINING_PEAK_ERROR"], controls["CON15_NEAR_TIE_ALLOWANCE"]
    if not np.isfinite(max_error) or max_error < 0:
        raise ValueError("MAX_TRAINING_PEAK_ERROR must be finite and nonnegative")
    if not np.isfinite(allowance) or not 0 <= allowance <= 1:
        raise ValueError("CON15_NEAR_TIE_ALLOWANCE must be finite and in [0, 1]")


def study_settings_from_controls(controls: Mapping[str, Any]) -> StudySettings:
    """Translate the notebook's named controls without capturing kernel state.

    Only whitelisted workflow controls are persisted; credentials and other
    notebook variables are never included. Snow overrides are resolved here,
    before the conservative melt envelope is frozen.
    """
    _validate_peak_preference(controls)
    snow_values = {}
    overrides = controls["PARAMETER_OVERRIDES"]
    snow_active = controls["SNOW_ENABLED"] and controls["CALIBRATE_SNOW"]
    for name in ("snow_T", "snow_ddf"):
        options = overrides.get(name, {})
        if options and not snow_active:
            raise ValueError("Snow parameter overrides require enabled snow calibration")
        ParameterSettings(**options)
        for option, suffix in (("value", "value"), ("lower_bound", "lower"), ("upper_bound", "upper")):
            snow_values[f"{name}_{suffix}"] = (
                options[option] if options.get(option) is not None else controls[f"{name}_{suffix}"])
    if snow_active:
        lo, hi = snow_values["snow_T_lower"], snow_values["snow_T_upper"]
        dlo, dhi = snow_values["snow_ddf_lower"], snow_values["snow_ddf_upper"]
        if not (np.isfinite([lo, hi, dlo, dhi]).all() and lo < hi and 0 <= dlo < dhi
                and lo <= snow_values["snow_T_value"] <= hi
                and dlo <= snow_values["snow_ddf_value"] <= dhi):
            raise ValueError("Snow starts must lie within finite ordered calibration bounds")
    modules = ModelSettings(
        groundwater_enabled=controls["GROUNDWATER_ENABLED"], n_triangles=controls["N_RTK_TRIANGLES"],
        seasonal_baseline=controls["SEASONAL_BASELINE"],
        background_window_days=controls["BACKGROUND_WINDOW_DAYS"],
        background_minimum_days=controls["BACKGROUND_MINIMUM_DAYS"],
        background_max_gap_days=controls["BACKGROUND_MAX_GAP_DAYS"],
        background_endpoint_hold_days=controls["BACKGROUND_ENDPOINT_HOLD_DAYS"],
        snow_enabled=controls["SNOW_ENABLED"], calibrate_snow=snow_active,
        snow_threshold_c=snow_values["snow_T_value"], snow_melt_factor=snow_values["snow_ddf_value"])
    nsga, cma, de = controls["NSGA2_SETTINGS"], controls["CMA_ES_SETTINGS"], controls["DE_SETTINGS"]
    calibration = CalibrationSettings(
        scoring_mode=controls["SCORING_MODE"], solver=controls["SOLVER"],
        objectives=tuple([controls["SINGLE_OBJECTIVE"]] if controls["SCORING_MODE"] == "single"
                         else controls["MULTIOBJECTIVE_OBJECTIVES"]),
        population=nsga["pop_size"], generations=nsga["generations"], seed=nsga["seed"],
        workers=controls["CALIBRATION_WORKERS"], cma_trials=cma["cma_n_trials"],
        cma_starts=cma["cma_multi_start"], cma_jobs=cma["cma_n_jobs"], cma_seed=cma["cma_seed"],
        cma_offspring_size=cma["cma_offspring_size"], cma_progress_every=cma["cma_progress_every"],
        de_maxiter=de["de_maxiter"], de_popsize=de["de_popsize"], de_polish=de["de_polish"])
    names = ["s_max", "ia_k_dep", "ia_k0", "ia_kT", "ia_theta", "ia_T_ref", "ia_T_freeze", "area_acres"]
    if modules.groundwater_enabled:
        names += ["k_div", "k_G", "g_max"]
    # The settings page exposes four pathways; later ones use model defaults or explicit overrides.
    for i in range(1, min(modules.n_triangles, 4) + 1):
        names += [f"R_{i}", f"uh{i}_T", f"uh{i}_K"]
    if snow_active:
        names += ["snow_T", "snow_ddf"]
    values = {**controls, **snow_values}
    parameters = {}
    for name in names:
        options = dict(lower_bound=values[f"{name}_lower"], upper_bound=values[f"{name}_upper"])
        if f"{name}_value" in values:
            options["value"] = values[f"{name}_value"]
        parameters[name] = options
    parameters["area_acres"]["calibrate"] = controls["CALIBRATE_AREA"]
    for name, options in overrides.items():
        parameters.setdefault(name, {}).update(options)
    return StudySettings(
        model=modules, calibration=calibration,
        parameters={name: ParameterSettings(**options) for name, options in parameters.items()},
        workflow={page: {name: controls[name] for name in fields}
                  for page, fields in WORKFLOW_FIELDS.items()})


def split_config_from_settings(study: StudySettings, dt_hours: float) -> SplitConfig:
    """Build the existing forcing-only split, retaining even small melt pulses."""
    holdout, simulation = study.workflow["holdout"], study.workflow["simulation"]
    return SplitConfig(
        dt_hours=dt_hours, warmup_days=simulation["WARMUP_DAYS"], spinup_mode=simulation["SPINUP_MODE"],
        split_mode=holdout["SPLIT_MODE"], train_frac=holdout["TRAIN_FRAC"],
        block_days=holdout["STRAT_BLOCK_DAYS"], val_every=holdout["VAL_EVERY"],
        snap_edges_to_dry=holdout["SNAP_EDGES_TO_DRY"], edge_dry_hours=holdout["EDGE_DRY_HOURS"],
        edge_snap_max_days=holdout["EDGE_SNAP_MAX_DAYS"],
        rain_eps=0.0 if study.model.snow_enabled else holdout["RAIN_EPS_IN"])


def _event_path(value: str | None, directory: Path, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must name an event file")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else directory / path).resolve()


@dataclass
class PreparedPatternData:
    """Continuous forcing plus frozen scoring masks; data is enriched in place."""

    data: pd.DataFrame
    split: SplitConfig
    download_dir: Path
    warmup_mask: np.ndarray
    train_mask: np.ndarray
    val_mask: np.ndarray
    split_train_mask: np.ndarray
    split_val_mask: np.ndarray
    observation_eligible: np.ndarray
    spinup_steps: int
    warmup_steps: int
    event_file: Path | None = None
    all_event_file: Path | None = None
    events: pd.DataFrame | None = None
    calibration_events: pd.DataFrame | None = None
    phase_seconds: dict[str, float] = field(default_factory=dict)

    @property
    def scored_mask(self) -> np.ndarray:
        return self.train_mask | self.val_mask

    @property
    def dry_eligible(self) -> np.ndarray:
        return self.data["dry_eligible"].to_numpy(dtype=bool)

    @property
    def forcing_columns(self) -> list[str]:
        columns = ["datetime", "rainfall_in", "temperature_c"]
        if "background_required" in self.data:
            columns.append("background_required")
        return columns


def prepare_pattern_data(
    data: pd.DataFrame, study: StudySettings, split: SplitConfig, download_dir: Path,
) -> PreparedPatternData:
    """Prepend spin-up once, partition snow, and exclude bad observations only."""
    if "is_spinup" in data:
        raise RuntimeError("model_df already carries a spin-up block; re-run the assemble cell.")
    source = study.workflow["events"]["EVENT_SOURCE"]
    if source not in ("auto", "file", "hybrid"):
        raise ValueError("EVENT_SOURCE must be 'auto', 'file' or 'hybrid'")
    data, spinup_steps = prepend_spinup(data, split)
    data = data.copy()
    data["is_spinup"] = np.arange(len(data)) < spinup_steps
    times = pd.DatetimeIndex(data["datetime"])
    spacing = np.diff(times.as_unit("ns").asi8) / pd.Timedelta(hours=1).value
    if not len(spacing) or not np.allclose(spacing, split.dt_hours):
        raise ValueError(f"model_df spacing must equal DT_HOURS={split.dt_hours:g}; re-run assembly.")
    started = time.perf_counter()
    rain, temperature = data["rainfall_in"].to_numpy(), data["temperature_c"].to_numpy()
    forcing = snow_partition(rain, temperature, split.dt_hours, study.model.snow_config)
    for column in forcing:
        data[column] = forcing[column].to_numpy()
    if study.model.calibrate_snow:
        threshold, melt = study.parameters["snow_T"], study.parameters["snow_ddf"]
        possible_melt = snow_melt_envelope(
            rain, temperature, split.dt_hours,
            threshold_bounds=(threshold.lower_bound, threshold.upper_bound),
            melt_factor_bounds=(melt.lower_bound, melt.upper_bound),
            initial_swe_in=study.model.initial_swe_in)
    else:
        possible_melt = data["melt_in"].to_numpy() > 0
    data["possible_melt"] = possible_melt
    # This indicator freezes masks; it is never fed to the physical model.
    wet = rain + possible_melt.astype(float)
    exclusion = study.workflow["simulation"]["DRY_EXCLUSION_HOURS"]
    data["dry_eligible"] = dry_weather_mask(data.assign(rainfall_in=wet), split.dt_hours, exclusion)
    elapsed = time.perf_counter() - started
    warmup, train, val, warmup_steps = build_masks(
        data["datetime"], split, rain=wet, warmup_steps=spinup_steps or None)
    prepared = PreparedPatternData(
        data, split, Path(download_dir), warmup, train.copy(), val.copy(), train.copy(), val.copy(),
        np.ones(len(data), bool), spinup_steps, warmup_steps,
        phase_seconds={"snow_and_dry_eligibility": elapsed})
    if source == "hybrid":
        events = study.workflow["events"]
        prepared.all_event_file = _event_path(
            events["APPROVED_EVENTS_FILE"], prepared.download_dir, "APPROVED_EVENTS_FILE")
        prepared.event_file = _event_path(events["EVENT_FILE"], prepared.download_dir, "EVENT_FILE")
        prepared.calibration_events, prepared.events, prepared.observation_eligible = prepare_hybrid_events(
            data, train, val, prepared.all_event_file, prepared.event_file)
        prepared.train_mask &= prepared.observation_eligible
        prepared.val_mask &= prepared.observation_eligible
        data["dry_eligible"] &= prepared.observation_eligible
        print(f"Hybrid data-quality exclusion: {int((~prepared.observation_eligible).sum()):,} rows; "
              "forcing and continuous simulation unchanged.")
    if not prepared.train_mask.any():
        raise ValueError("No quality-eligible training observations remain")
    print(f"Dry-weather eligibility: {data.dry_eligible.sum():,} rows after {exclusion:g} dry hours.")
    print(f"Conservative possible-melt rows: {int(possible_melt.sum()):,}; masks stay fixed.")
    print(f"Modeling dataset: {len(data):,} steps @ {split.dt_hours * 60:g} min; "
          f"{times[0]} -> {times[-1]}")
    print(f"Warmup: {warmup_steps:,} steps; "
          f"{'synthetic ' + split.spinup_mode if spinup_steps else 'real record'}")
    print(f"Training: {prepared.train_mask.sum():,}; validation: {prepared.val_mask.sum():,}; "
          f"last train step: {data.loc[prepared.train_mask, 'datetime'].iloc[-1]}")
    return prepared


@dataclass
class PatternEventCoverage:
    """Keep broad calibration coverage separate from approved assessment windows."""

    events: pd.DataFrame
    calibration_events: pd.DataFrame
    assessment_events: pd.DataFrame
    acceptance_events: pd.DataFrame
    storm_mask: np.ndarray
    val_storm_mask: np.ndarray
    rain_storm_mask: np.ndarray
    val_rain_storm_mask: np.ndarray
    melt_mask: np.ndarray
    val_melt_mask: np.ndarray
    approved_storm_mask: np.ndarray
    val_approved_storm_mask: np.ndarray
    assessment_storm_mask: np.ndarray
    val_assessment_storm_mask: np.ndarray
    pre_hours: float
    tail_hours: float
    settings: MSDGCSettings = field(default_factory=MSDGCSettings)

    @property
    def event_mask(self) -> np.ndarray:
        return self.storm_mask | self.val_storm_mask


def build_pattern_event_coverage(
    prepared: PreparedPatternData, reference: np.ndarray, study: StudySettings,
) -> PatternEventCoverage:
    """Freeze event inventories and masks without re-fitting the dry reference."""
    started = time.perf_counter()
    data, train, val = prepared.data, prepared.train_mask, prepared.val_mask
    controls, dt = study.workflow["events"], prepared.split.dt_hours
    source = controls["EVENT_SOURCE"]
    if source == "file":
        prepared.event_file = _event_path(controls["EVENT_FILE"], prepared.download_dir, "EVENT_FILE")
        events = imported_event_inventory(data, train, val, prepared.event_file)
        pre, tail = 0.0, 0.0
        calibration = events.copy()
        print(f"Overlapping approved windows: {events.overlap_previous_hours.gt(0).sum()}; not merged.")
    elif source == "auto":
        # Older saved studies lack this setting; None keeps the detector defaults.
        events = frozen_events(data, train, val, reference,
                               detector_settings=controls.get("STORM_DETECTION_SETTINGS"))
        calibration, pre, tail = events.copy(), 3.0, 48.0
    elif source == "hybrid":
        events, calibration = prepared.events, prepared.calibration_events
        if events is None or calibration is None:
            raise ValueError("Hybrid inventories must be prepared before fitting sanitary shapes")
        pre, tail = controls["CALIBRATION_PRE_HOURS"], controls["CALIBRATION_TAIL_HOURS"]
        if not np.isfinite([pre, tail]).all() or min(pre, tail) < 0:
            raise ValueError("Hybrid calibration padding must be finite and nonnegative")
    else:
        raise ValueError("EVENT_SOURCE must be 'auto', 'file' or 'hybrid'")
    rain_masks, melt_masks, approved_masks, assessment_masks = [], [], [], []
    assessment = assessment_event_inventory(
        pd.DatetimeIndex(data["datetime"]), events, train, val,
        excluded_event_ids=controls["ASSESSMENT_EXCLUDED_EVENT_IDS"],
        full_inventory=calibration if source == "hybrid" else None)
    acceptance = assessment.loc[assessment.eligible.astype(bool)].copy()
    for label, mask in (("train", train), ("validation", val)):
        rain_masks.append(storm_scoring_mask(
            calibration, mask, dt, split=label, pre_hours=pre, tail_hours=tail))
        melt_masks.append(
            melt_scoring_mask(data["possible_melt"].to_numpy(), mask, dt, pre_hours=pre, tail_hours=tail)
            if source in ("auto", "hybrid") else np.zeros(len(data), bool))
        approved_masks.append(storm_scoring_mask(events, mask, dt, split=label, pre_hours=0, tail_hours=0))
        assessment_masks.append(storm_scoring_mask(
            assessment, mask, dt, split=label, pre_hours=0, tail_hours=0))
    coverage = PatternEventCoverage(
        events, calibration, assessment, acceptance,
        rain_masks[0] | melt_masks[0], rain_masks[1] | melt_masks[1],
        *rain_masks, *melt_masks, *approved_masks, *assessment_masks, pre, tail)
    print("MSDGC TOTAL-FLOW assessment; not the official effective-storm scorecard.")
    print(f"Event source: {source}; file: {prepared.event_file}; broad file: {prepared.all_event_file}")
    print(f"Assessment exclusions requested: {controls['ASSESSMENT_EXCLUDED_EVENT_IDS']}; "
          f"eligible assessment events: {len(acceptance)}; broad fitting unchanged.")
    print(f"Eligible calibration events: {int(calibration.eligible.sum())}; "
          f"training storm steps: {coverage.storm_mask.sum():,}")
    print("Added melt-window rows: train="
          f"{int((melt_masks[0] & ~rain_masks[0]).sum()):,}, validation="
          f"{int((melt_masks[1] & ~rain_masks[1]).sum()):,}")
    for name, inventory in (("Assessment", assessment), ("Approved", events), ("Calibration", calibration)):
        excluded = inventory.loc[~inventory.eligible.astype(bool), ["event_id", "split", "exclusion"]]
        if len(excluded):
            print(f"{name} exclusions:\n{excluded.to_string(index=False)}")
    if source == "hybrid" and not acceptance["split"].eq("train").any():
        raise ValueError("Hybrid mode needs complete approved TRAINING events for preference/reporting")
    if not coverage.storm_mask.any():
        raise ValueError(f"No usable training event windows for EVENT_SOURCE={source!r}; "
                         f"file={prepared.event_file}. Inspect event exclusions.")
    prepared.phase_seconds["event_inventory"] = time.perf_counter() - started
    return coverage


def build_pattern_model(
    prepared: PreparedPatternData, shape_fit: PatternEstimate, odwf_cfs: float, study: StudySettings,
) -> PatternWholeFlowModel:
    """Build, validate, and warm the model using only continuous forcing columns."""
    simulation, modules = study.workflow["simulation"], study.model
    if simulation["REQUIRE_NUMBA"] and not _HAS_NUMBA:
        raise RuntimeError("Numba acceleration is unavailable; use the sparsehydro environment.")
    pattern = shape_fit.pattern
    upper = simulation["SANITARY_UPPER_MULTIPLIER"] * odwf_cfs
    start = max(.75 * odwf_cfs, -min(pattern.hourly) - min(pattern.weekday) + 1e-6)
    background = None
    if modules.uses_observed_background:
        background = estimate_observed_background(
            prepared.data, prepared.train_mask & prepared.dry_eligible, pattern,
            window_days=modules.background_window_days, minimum_days=modules.background_minimum_days,
            max_gap_days=modules.background_max_gap_days, endpoint_hold_days=modules.background_endpoint_hold_days)
        prepared.data["background_required"] = prepared.scored_mask
        print("RETROSPECTIVE ONLY: training-dry background includes unidentified slow inflow; no annual repetition.")
        print(f"Background support: {background.start} to {background.end}; "
              f"{len(background.datetimes)} source-day anchors; "
              f"endpoint hold limit {modules.background_endpoint_hold_days} days.")
        times = pd.DatetimeIndex(prepared.data["datetime"])
        held = background.endpoint_hold_mask(times) & prepared.scored_mask
        for label, edge in (("before", times < background.start), ("after", times > background.end)):
            mask = held & edge
            if mask.any():
                print(f"Endpoint hold {label}: {times[mask][0]} to {times[mask][-1]}; {mask.sum()} scored rows "
                      f"(training {(mask & prepared.train_mask).sum()}, "
                      f"validation {(mask & prepared.val_mask).sum()}).")
    elif start > upper:
        raise ValueError("Shapes are incompatible with sanitary mean bounds; inspect dry-weather data")
    model = study.build_model(pattern, simulation["AREA_ACRES"], prepared.split.dt_hours,
                              start, upper, observed_background=background)
    model.prepare(prepared.data[prepared.forcing_columns])
    started = time.perf_counter()
    model.predict()
    prepared.phase_seconds["numba_warmup_and_initial_prediction"] = time.perf_counter() - started
    signatures, options = getattr(_gwi_B_core, "nopython_signatures", ()), getattr(_gwi_B_core, "targetoptions", {})
    if simulation["REQUIRE_NUMBA"] and not signatures:
        raise RuntimeError("S/G recurrence did not compile with Numba; check NUMBA_DISABLE_JIT and restart the kernel.")
    print(f"Compiled Numba recurrence: {bool(signatures)}; "
          f"nogil={options.get('nogil', False)}; fastmath={options.get('fastmath', False)}")
    count = sum(model.get_scalar_parameter(n).calibrate for n in model.scalar_parameter_names)
    print(f"{count} optimized scalars; GWI={model.groundwater_enabled}; "
          f"RTK pathways={model.n_triangles}; 29 shapes estimated separately.")
    area = model.get_scalar_parameter("area_acres")
    print(f"Area: initial={area.value:g} acres, bounds=({area.lower_bound:g}, {area.upper_bound:g}), "
          f"calibrated={area.calibrate}")
    print("Initial stores default to fixed zero; PARAMETER_OVERRIDES can explicitly change active stores.")
    return model


@dataclass
class PatternCalibration:
    """Training-selected result; post-hoc diagnostic models are kept separate."""

    result: CalibrationResult
    prediction: pd.DataFrame
    pareto_solutions: pd.DataFrame
    selected_index: int
    selection_rule: str
    selection_status: str
    peak_objective: MSDGCCon15 | None
    peak_filter: MSDGCCon15 | None
    peak_objective_events: pd.DataFrame
    solver_metadata: dict[str, Any]
    verified_objectives: np.ndarray
    problem: FeasibleCalibrationProblem
    solver: ISolver
    split_diagnostics: pd.DataFrame


@dataclass
class PatternObjectives:
    """Frozen training objectives, independent Con15 controls, and their audit."""

    objectives: list[IObjective]
    peak_objective: MSDGCCon15 | None
    peak_filter: MSDGCCon15 | None
    events: pd.DataFrame


def build_pattern_objectives(
    prepared: PreparedPatternData, coverage: PatternEventCoverage, study: StudySettings,
) -> PatternObjectives:
    """Audit complete training events before a solver can access observations."""
    selection, config = study.workflow["selection"], study.calibration
    _validate_peak_preference(selection)
    enabled = selection["USE_TRAINING_PEAK_FILTER"]
    data, train, dt = prepared.data, prepared.train_mask, prepared.split.dt_hours
    events = coverage.acceptance_events.loc[coverage.acceptance_events["split"].eq("train")].copy()
    windows = events[["start_idx", "end_idx"]].to_numpy(dtype=int)
    if not all(train[start:end + 1].all() for start, end in windows):
        raise ValueError("Training peak events must lie wholly inside train_mask")
    overlapping = study.workflow["events"]["EVENT_SOURCE"] in ("file", "hybrid")

    def peak_definition(fraction: float) -> MSDGCCon15:
        return MSDGCCon15(windows, events["duration_hours"].to_numpy(), dt,
                          settings=coverage.settings, top_fraction=fraction,
                          mask=train, allow_overlapping_events=overlapping)

    peak_objective = peak_definition(selection["CON15_OBJ_FRACTION"]) if "con15_error" in config.objectives else None
    peak_filter = peak_definition(selection["CON15_FILTER_FRACTION"]) if enabled else None
    observed = data["flow_cfs"].to_numpy()
    audit = MSDGCAssessment(
        observed, events, train, np.zeros_like(train), dt,
        AcceptanceContract(version=coverage.settings.version), settings=coverage.settings,
        allow_overlapping_events=overlapping).conveyance_report(observed)
    events["observed_peak_cfs"] = [float(data["flow_cfs"].iloc[s:e + 1].max()) for s, e in windows]
    events["observed_conveyance_volume_mg"] = audit["observed_volume_mg"].to_numpy(dtype=float)
    events["observed_conveyance_status"] = audit["status"].to_numpy()
    events["observed_conveyance_reason"] = audit["reason"].to_numpy()
    events["con15_basis"] = "msdgc_independent_conveyance_volume_mg"
    if audit["status"].ne("defined").any():
        print("Undefined observed conveyance-volume diagnostics:\n" + audit.loc[
            audit["status"].ne("defined"), ["event_id", "status", "reason"]].to_string(index=False))
    for role, definition in (("objective", peak_objective), ("filter", peak_filter)):
        flag = f"{role}_selected"
        events[flag] = False
        if definition is None:
            print(f"Training Con15 {role}: disabled")
        else:
            definition.compute(observed, observed)
            indices = definition.selected_event_indices(observed)
            events.iloc[indices, events.columns.get_loc(flag)] = True
            print(f"Con15 {role}: top {definition.top_fraction:.1%} of {len(events)} complete TRAINING storms "
                  f"= {len(indices)} selected; validation excluded.")
            print(events.iloc[indices][["event_id", "observed_conveyance_volume_mg"]].to_string(index=False))
    objectives = config.build_objectives(train, coverage.storm_mask, con15=peak_objective)
    return PatternObjectives(objectives, peak_objective, peak_filter, events)


def calibrate_pattern_model(
    model: PatternWholeFlowModel, prepared: PreparedPatternData,
    coverage: PatternEventCoverage, study: StudySettings, artifact_dir: Path,
) -> PatternCalibration:
    """Solve once and verify the selected prediction against frozen objectives."""
    definitions = build_pattern_objectives(prepared, coverage, study)
    objectives, events = definitions.objectives, definitions.events
    peak_objective, peak_filter = definitions.peak_objective, definitions.peak_filter
    config, selection = study.calibration, study.workflow["selection"]
    max_error, allowance = selection["MAX_TRAINING_PEAK_ERROR"], selection["CON15_NEAR_TIE_ALLOWANCE"]
    data, train, dt = prepared.data, prepared.train_mask, prepared.split.dt_hours
    observed = data["flow_cfs"].to_numpy()
    problem = FeasibleCalibrationProblem(
        model=model, data=data.assign(flow_cfs=data["flow_cfs"].where(train)), objectives=objectives,
        column_map={"observed": "flow_cfs", "predicted": lambda values: values},
        predict_fn=PatternWholeFlowModel.predict_total_cfs)
    solver = config.build_solver()
    metadata = config.solver_metadata()
    print("Solver configuration:", metadata)
    if model.observed_background is not None:
        print("Background is frozen before optimization; validation describes historical interpolation, not forecasting.")
    started = time.perf_counter()
    result = solver.solve(problem)
    prepared.phase_seconds["joint_optimization"] = time.perf_counter() - started
    if config.resolved_solver == "cmaes":
        metadata["selected_seed"] = solver.selected_seed_
    started = time.perf_counter()
    selected, table = select_peak_filtered_pareto(
        result, model, observed, peak_filter=peak_filter, max_error=max_error,
        near_tie_allowance=allowance if peak_filter is not None else None)
    prepared.phase_seconds["pareto_filter_selection"] = time.perf_counter() - started
    status = str(table.loc[selected, "selection_status"])
    rule = ("normalized closest-to-ideal of calibration objectives; peak filter disabled"
            if peak_filter is None else
            f"lowest training conveyance-volume slope error within {allowance:g} of "
            "best normalized primary RMS score; threshold report only")
    if peak_filter is not None:
        print(f"Near-tie primary objectives: {table.attrs['primary_objective_names']}; "
              f"shortlist={int(table.near_tie_eligible.sum())} of {len(table)}; "
              f"selected primary score={table.loc[selected, 'primary_score']:.6g}; "
              f"increase={table.loc[selected, 'primary_score_increase']:.6g} <= {allowance:g}")
        print(f"Training volume-filter passes: {int(table.filter_pass.fillna(False).sum())} of {len(table)}; "
              f"selected error={table.loc[selected, 'filter_peak_error']:.6g}")
    for name, value in zip(result.param_names, result.pareto_X[selected]):
        model.apply_flat_parameter(name, float(value))
    if not model.validate():
        raise RuntimeError("No feasible selected Pareto solution; increase the search budget")
    started = time.perf_counter()
    model.prepare(data[prepared.forcing_columns])
    prediction = model.predict()
    if study.model.calibrate_snow:
        if (prediction["melt_in"].gt(0).to_numpy() & ~data["possible_melt"].to_numpy()).any():
            raise RuntimeError("Selected melt escaped the frozen envelope; rerun after changing bounds.")
    else:
        for column in ("SWE_in", "melt_in", "liquid_input_in"):
            np.testing.assert_array_equal(data[column], prediction[column], err_msg="Fixed snow forcing changed.")
    for column in prediction:
        if column not in ("datetime", "baseflow_cfs", "sanitary_cfs"):
            data[column] = prediction[column].to_numpy()
    data["joint_baseflow_cfs"] = prediction["baseflow_cfs"].to_numpy()
    data["joint_sanitary_cfs"] = prediction["sanitary_cfs"].to_numpy()
    data["joint_G_at_cap"] = prediction["G_at_cap"].to_numpy()
    np.testing.assert_allclose(prediction.total_cfs,
                               prediction.sanitary_cfs + prediction.gwi_cfs + prediction.rdii_cfs)
    verified = result.objective_display_values()[selected]
    np.testing.assert_allclose(verified, [obj.compute(observed, prediction.total_cfs.to_numpy()) for obj in objectives])
    if peak_filter is not None:
        np.testing.assert_allclose(peak_filter.compute(observed, prediction.total_cfs.to_numpy()),
                                   table.loc[selected, "filter_peak_error"])
    diagnostics = []
    for label, mask in (("train", train), ("validation", prepared.val_mask)):
        split_events = coverage.acceptance_events.loc[coverage.acceptance_events["split"].eq(label)]
        peak = None
        if peak_objective is not None and len(split_events):
            peak = MSDGCCon15(
                split_events[["start_idx", "end_idx"]].to_numpy(dtype=int),
                split_events["duration_hours"].to_numpy(), dt, settings=coverage.settings,
                top_fraction=selection["CON15_OBJ_FRACTION"], mask=mask,
                allow_overlapping_events=study.workflow["events"]["EVENT_SOURCE"] in ("file", "hybrid"))
        row = dict(
            split=label,
            ccc=ConcordanceCorrelationCoefficient().compute(observed, prediction.total_cfs.to_numpy(), mask=mask),
            nse=NashSutcliffe().compute(observed, prediction.total_cfs.to_numpy(), mask=mask),
            con15_slope=peak.slope(observed, prediction.total_cfs.to_numpy()) if peak is not None else np.nan,
            con15_error=peak.compute(observed, prediction.total_cfs.to_numpy()) if peak is not None else np.nan,
            con15_status="defined" if peak is not None else (
                "disabled" if peak_objective is None else "undefined_no_complete_assessment_storms"))
        diagnostics.append(row)
        peak_text = (f"conveyance-volume slope={row['con15_slope']:.4f}; Con15 error={row['con15_error']:.6g}"
                     if peak is not None else "objective volume diagnostic disabled"
                     if peak_objective is None else "volume slope/error=undefined (no complete assessment storms)")
        print(f"{label}: total CCC={row['ccc']:.4f}; NSE={row['nse']:.4f}; "
              f"top fraction={selection['CON15_OBJ_FRACTION']:g}; {peak_text}")
    split_diagnostics = pd.DataFrame(diagnostics)
    table.to_csv(Path(artifact_dir) / "pareto_solutions.csv", index=False)
    prepared.phase_seconds["selected_prediction"] = time.perf_counter() - started
    print("Verified selected objectives:", dict(zip(result.objective_names, verified)))
    print(f"Selected snow settings: {model.snow_config}; "
          f"contributing area: {model.get_scalar_parameter('area_acres').value:.6g} acres")
    print(f"Joint fit: {prepared.phase_seconds['joint_optimization']:.2f} s; "
          f"selected Pareto index {selected}; {status}; {rule}")
    return PatternCalibration(result, prediction, table, selected, rule, status,
                              peak_objective, peak_filter, events, metadata, verified,
                              problem, solver, split_diagnostics)
