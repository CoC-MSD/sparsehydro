"""Default controls for the MSDGC whole-flow study.

These are the values from the settings pages of the study notebook
(``flowfinity_gwi_wholeflow_pattern_joint.ipynb``), held in the package so that
tests and scripts do not need the notebook. Pass the result, with any edits, to
:func:`sparsehydro.calibration.pattern_workflow.study_settings_from_controls`.

Keys keep the notebook's upper-case names because the workflow functions read
them by those names.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

# fmt: off
_DATA = {
    # Page 1 - data and events
    "DATA_SOURCE": "files",          # "files" (local) | "flowfinity" (download)
    "DATA_TITLE": "SBU_FM",          # plot/export label; default local folder name
    "SOURCE_DIR": None,              # None -> sparsehydro/downloads/<DATA_TITLE>
    "RAIN_FILE": "WS236_Rainfall_10252024_09302025.tsf",     # rainfall depth per step [in]
    "FLOW_FILE": "MC-BR-026_Monitor_10302024_10312025.tsf",  # total flow [MGD]
    "TEMP_FILE": None,               # air temperature CSV [deg C, UTC]
    "BASEFLOW_FILE": None,
    "STORMFLOW_FILE": None,
    "START": None,                   # FlowFinity download window
    "END": None,
    "FLOWFINITY_RECORD_TITLE": None,
    "FLOWFINITY_RECORD_INDEX": 28,
    "TYPES": ["Flow", "BaseFlow", "StormFlow", "Rain"],
    "NAME_FILTER": None,
    "EVENT_SOURCE": "hybrid",        # "auto" | "file" | "hybrid"
    "EVENT_FILE": r"..\Events\QualifyingEvents_MCBR26.xlsx",
    "APPROVED_EVENTS_FILE": r"..\Events\MC-BR-026_AllEvents_3.evt",
    "CALIBRATION_PRE_HOURS": 3.0,
    "CALIBRATION_TAIL_HOURS": 48.0,
    "ASSESSMENT_EXCLUDED_EVENT_IDS": ("87",),
}

_FILE_PROCESSING = {
    # Page 2 - file processing and training/validation split
    "TIMESTEP": "5min",              # "5min" | "15min" | "60min"
    "RAIN_COLUMN": 0,
    "FLOW_COLUMN": 0,
    "BASEFLOW_COLUMN": 0,
    "STORMFLOW_COLUMN": 0,
    "FILE_START": "2024-11-01",
    "FILE_END": None,
    "SPLIT_MODE": "stratified",      # "continuous" | "stratified"
    "TRAIN_FRAC": 0.70,
    "STRAT_BLOCK_DAYS": 21,
    "VAL_EVERY": 4,
    "SNAP_EDGES_TO_DRY": True,
    "EDGE_DRY_HOURS": 48.0,
    "EDGE_SNAP_MAX_DAYS": 5.0,
    "RAIN_EPS_IN": 1e-3,
}

_SIMULATION = {
    # Page 3 - spin-up
    "WARMUP_DAYS": 91,               # 13 whole weeks keeps hour and weekday aligned
    "SPINUP_MODE": "wrap",           # "wrap" | "repeat_head" | "none"
    # Page 4 - automatic storm detection (EVENT_SOURCE = "auto")
    "STORM_DETECTION_SETTINGS": {
        "threshold_pct": 5.0,
        "interevent_hours": 6.0,
        "sliding_hours": 12.0,
        "min_rain_in": 0.2,
        "min_duration_hours": 3.0,
    },
}

_CALIBRATION = {
    # Page 5 - objectives, solver, budgets and selection
    "SCORING_MODE": "multiobjective",  # "single" | "multiobjective"
    "SOLVER": "nsga2",                 # "auto" | "nsga2" | "spea2" | "cmaes" | "differential_evolution"
    "MULTIOBJECTIVE_OBJECTIVES": ["ccc", "nash_sutcliffe", "peak_weighted_mse"],
    "SINGLE_OBJECTIVE": "ccc",
    "DRY_EXCLUSION_HOURS": 48.0,
    "NSGA2_SETTINGS": {"pop_size": 100, "generations": 100, "seed": 42},
    "CMA_ES_SETTINGS": {
        "cma_n_trials": 2000,
        "cma_multi_start": 4,
        "cma_n_jobs": 4,
        "cma_seed": 42,
        "cma_offspring_size": 10,
        "cma_progress_every": 200,
    },
    "DE_SETTINGS": {"de_maxiter": 1000, "de_popsize": 15, "de_polish": False},
    "CON15_OBJ_FRACTION": 0.3,
    "USE_TRAINING_PEAK_FILTER": False,
    "CON15_NEAR_TIE_ALLOWANCE": 0.02,
    "CON15_FILTER_FRACTION": 0.3,
    "MAX_TRAINING_PEAK_ERROR": 0.1,
    "CALIBRATION_WORKERS": 4,
    "SANITARY_UPPER_MULTIPLIER": 2.0,
    "TEMPERATURE_MAX_GAP_HOURS": 12.0,
    "REQUIRE_NUMBA": True,
}

_AREA_ACRES = 20.26

_PARAMETERS = {
    # Page 6 - start values, bounds and calibration participation
    "PARAMETER_OVERRIDES": {},
    "N_RTK_TRIANGLES": 3,
    "AREA_ACRES": _AREA_ACRES,
    "CALIBRATE_AREA": True,
    # IA store S
    "s_max_lower": 0.05, "s_max_upper": 2.0,
    "ia_k_dep_lower": 1.0, "ia_k_dep_upper": 30.0,
    "ia_k0_lower": 1e-4, "ia_k0_upper": 0.35,
    "ia_kT_lower": 0.0, "ia_kT_upper": 0.35,
    "ia_theta_lower": 0.0, "ia_theta_upper": 0.5,
    "ia_T_ref_lower": 8.0, "ia_T_ref_upper": 20.0,
    "ia_T_freeze_lower": -5.0, "ia_T_freeze_upper": 5.0,
    # Deep store G
    "k_div_lower": 1e-4, "k_div_upper": 5e-2,
    "k_G_lower": 1e-5, "k_G_upper": 2e-2,
    "g_max_lower": 1e-2, "g_max_upper": 0.5,
    "area_acres_lower": _AREA_ACRES * 0.7, "area_acres_upper": _AREA_ACRES * 1.3,
    # RTK pathways
    "R_1_value": 0.08, "R_1_lower": 0.0, "R_1_upper": 0.40,
    "R_2_value": 0.05, "R_2_lower": 0.0, "R_2_upper": 0.35,
    "R_3_value": 0.02, "R_3_lower": 0.0, "R_3_upper": 0.25,
    "R_4_value": 0.005, "R_4_lower": 0.0, "R_4_upper": 0.15,
    "uh1_T_value": 1.0, "uh1_T_lower": 0.25, "uh1_T_upper": 4.0,
    "uh1_K_value": 2.0, "uh1_K_lower": 1.1, "uh1_K_upper": 6.0,
    "uh2_T_value": 8.0, "uh2_T_lower": 4.0, "uh2_T_upper": 24.0,
    "uh2_K_value": 2.5, "uh2_K_lower": 1.5, "uh2_K_upper": 8.0,
    "uh3_T_value": 48.0, "uh3_T_lower": 24.0, "uh3_T_upper": 96.0,
    "uh3_K_value": 3.0, "uh3_K_lower": 2.0, "uh3_K_upper": 9.0,
    "uh4_T_value": 168.0, "uh4_T_lower": 96.0, "uh4_T_upper": 720.0,
    "uh4_K_value": 3.0, "uh4_K_lower": 2.0, "uh4_K_upper": 9.0,
    # Snow
    "SNOW_ENABLED": True,
    "CALIBRATE_SNOW": True,
    "snow_T_value": 0, "snow_T_lower": -2.0, "snow_T_upper": 3.0,
    "snow_ddf_value": 0.03, "snow_ddf_lower": 0.005, "snow_ddf_upper": 0.5,
}

_MODULES = {
    # Page 7 - modules and reporting
    "GROUNDWATER_ENABLED": True,
    "SEASONAL_BASELINE": "auto",     # "auto" | "observed" | "off"
    "BACKGROUND_WINDOW_DAYS": 21,
    "BACKGROUND_MINIMUM_DAYS": 3,
    "BACKGROUND_MAX_GAP_DAYS": 45,
    "BACKGROUND_ENDPOINT_HOLD_DAYS": 3,
    "DISPLAY_MAX_POINTS": 5000,
}
# fmt: on

_DEFAULTS: dict[str, Any] = {
    **_DATA, **_FILE_PROCESSING, **_SIMULATION, **_CALIBRATION, **_PARAMETERS, **_MODULES,
}


def default_controls() -> dict[str, Any]:
    """Return a fresh, independently mutable copy of the study's default controls."""
    return deepcopy(_DEFAULTS)
