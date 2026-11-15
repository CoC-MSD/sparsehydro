# Parsimonious Functions → SparseHydro Integration Plan

> **Historical design note.** The unit hydrographs live in `sparsehydro.models.unithydrograph`; sequential fitting in `sparsehydro.calibration.sequential`. Module paths below may be out of date; see the API reference for the current layout.

## Goal

Integrate the Parsimonious Functions unit-hydrograph workflow into `sparsehydro`
using the package's existing structure and interfaces. Deliver an end-to-end
pipeline: **flow + rainfall → global + sub-event detection (Savitzky-Golay first)
→ sequential convolution fitting with native UH + abstraction + seasonality
models → Plotly visualization**, plus a worked notebook and unit tests on the
`MU-UM-019` dataset with an air-temperature covariate.

## Critical context

SparseHydro **already absorbed** the modular UnitHydrograph refactor. This is an
**enhancement**, not a greenfield port. Already present:

- `events/` — `detect_events()` (single-level, Savgol-derivative segmentation with
  back-to-back trough splitting). No global/sub hierarchy, no variable-window
  Savgol, no rain-aware bimodal split, no 1-day global grouping.
- `filters/` — `apply_savgol_filter`, `compute_thresholds`, `dryweather.disaggregate`.
- `models/unithydrograph/models.py` — `GammaUH`, `NashUH`, `TriangleUH`.
- `models/unithydrograph/adapter.py` — `UnitHydrographAdapter` wrapping a legacy
  `uh_models.py` that is **not vendored** → dead path.
- `models/unithydrograph/sequential.py` — `SequentialFitter` (residual-based,
  warm-start, single-objective, fixed Savgol smooth). No zonal weighting, no
  global grouping, no abstraction/seasonality composition.
- `models/seasonality.py` — `SeasonalityModel`. `models/rdii/` — `IAModel`
  (temperature-driven abstraction), `RTKTriangle`, `RDIIModel`.
  `models/ensemble.py` — `EnsembleModel`.
- `visualization/unithydrograph.py` — event/fit/area/parameter plots (Plotly).
- `data/utilities.py` — `read_tsf`, `add_time_features`.

## Decisions (confirmed)

| # | Decision |
|---|----------|
| A | **Full port** of event detection: variable-window Savgol + rain-aware bimodal sub-events + curvature-anchored zones + 1-day global grouping. |
| B | Abstraction/seasonality: **all three** — native `IAModel` (temperature) + `SeasonalityModel` + ported tank models (V_C/V_lin/V_sroot), composed via new `AbstractionUHModel`. |
| C | UH gaps: **native subclasses** (Rectangle/Decay/GammaDelay/PeakTail) + **remove** the dead adapter. |
| D | **Include** Parsimonious-style zonal (peak/tail) weighting now. |
| E | Stormflow: tests use `lstStormTs` (MGD→CFS); notebook **demonstrates both** (also `dryweather.disaggregate`). |
| F | Notebook: `sparsehydro/notebooks/sequential_fitting_workflow.ipynb`. |

Resolved sub-decisions: new `AbstractionUHModel` (leave `RDIIModel` untouched);
treat `air_temperature.csv` as **Celsius** with a conversion flag; **multiplicative**
seasonality peaking-factor.

## Phases

Phases A–C are independent (parallelizable). D depends on A/B/C; E on C/D;
F on A–E; G alongside each.

### Phase A — Fill UH gaps + remove dead adapter
- `models/unithydrograph/models.py`: add `RectangleUH` (A, tr), `DecayUH`
  (A, alpha), `GammaDelayUH` (A, tt, tp, td) — mirror `GammaUH`
  (`get_kernel`, normalization `Σk·dt ≈ 1`, amplitude scaling).
- New `models/unithydrograph/peak_tail.py`: `PeakTailUH` (A, blend `w`, delay
  `td`, peak/tail shape params) — default Triangle peak + Gamma tail.
- Delete `models/unithydrograph/adapter.py`; update
  `models/unithydrograph/__init__.py` and `models/__init__.py` docstring.

### Phase B — Tank abstraction + composite model
- New `models/abstraction/tank.py`: `ConstantDrainTank` (V_C),
  `LinearDrainTank` (V_lin), `SqrtDrainTank` (V_sroot) — `IModel` producing
  effective rainfall (5 sub-steps; defaults V_tank=1000, Ae_min=5, Ae_max=10,
  Qd=0.1, k=0.1).
- New `models/composite.py`: `AbstractionUHModel` chaining abstraction
  (`IAModel` or tank) → UH component → optional `SeasonalityModel` multiplier,
  with prefixed params (`ia_`/`tank_`/`uh_`/`seas_`), modeled after `RDIIModel`.

### Phase C — Global + sub-event detection (full port)
Refactor `events/` into a package (re-export for back-compat):
- `records.py` — existing `EventRecord` + `SubEventRecord`, `GlobalEvent`.
- `smoothing.py` — `variable_savgol_smooth` + curvature/slope helpers +
  `VariableSavgolResult`.
- `peaks.py` — `define_peaks` (rain-driven).
- `bimodality.py` — `calculate_peak_zone_bimodality`, `calculate_rain_bimodality`.
- `zones.py` — `define_complete_event_zones` (global + sub, bimodal split,
  curvature-anchored boundaries, rain gating).
- `detection.py` — move existing `detect_events`.
- `hierarchy.py` — `detect_event_hierarchy()` → `(global_events, sub_events,
  savgol_result)`; 1-day grouping.

### Phase D — Zonal sequential fitting
- `calibration/objectives.py`: add `WeightedRMSE` (per-timestep weights).
- `models/unithydrograph/sequential.py`: extend `SequentialFitter` — Savgol
  options, temperature pass-through, peak/tail zonal weight vectors, and a
  global-event grouping mode (warm-start across globals). Keep current
  signature working.

### Phase E — Plotly visualization
`visualization/unithydrograph.py`: add `plot_event_hierarchy`,
`plot_variable_savgol`, `plot_uh_shapes`, `plot_convolution`; enhance
`plot_sequential_fit` (zonal shading) and `plot_effective_area`. Export via
`visualization/__init__.py` and `sparsehydro/__init__.py`.

### Phase F — Notebook
`sparsehydro/notebooks/sequential_fitting_workflow.ipynb`: end-to-end
`MU-UM-019` + temperature; both stormflow paths; Parsimonious defaults.

### Phase G — Tests + fixtures
- `tests/conftest.py`: fixtures — `MU-UM-019` rain+stormflow (`read_tsf`,
  MGD→CFS, merged `datetime/rain/stormflow`), temperature (resampled 5-min),
  events.
- New: `test_uh_models`, `test_tank_abstraction`, `test_composite`,
  `test_event_hierarchy`, `test_sequential_global`; extend `test_visualization`;
  remove `test_uh_adapter`.

## Default parameters (carried from Parsimonious)

- Peaks: rain_window_hours=12, total_rain_min=0.05, dry_gap_hours=6, sg_window=31,
  polyorder=2, prominence_frac=0.10, min_peak_dist=12.
- Variable Savgol: seed_window=24, win_min=5, win_max=24, polyorder=3, peak_pad=2,
  lift_pad=5, curv_env_window=48, anchor_pct=90.
- Zones: alpha=0.02, rain_lead_hours=3, min_rain_lead=0.05, decline_rise_ratio=2/3,
  end_slope_frac=0.05, end_slope_hold=6.
- Bimodality: threshold=0.12, min_sec_curv_frac=0.20, rain_lead=8h, rain_lag=0.75h,
  split_min_separation=2h, split_max_valley_frac=0.65.
- Global/sequential: gap_days=1, base_pad_days=1, peak=Triangle, tail=Gamma,
  method="Nelder-Mead", min_stormflow=0.03.
- UH: Gamma (A=1, tt=2, tp=5), Triangle (A=100, tt=50, tp=20), Nash (A=100, n=2,
  k=5). Tank V_C (V_tank=1000, Ae_min=5, Ae_max=10, Qd=0.1).

## Verification

1. `pytest` from the SparseHydro `.venv` (new tests + full suite).
2. UH kernels `Σk·dt ≈ 1` and amplitude scaling; tank mass balance
   (0 ≤ effective ≤ rain); composite `predict()` length; `detect_event_hierarchy`
   non-empty global+sub on `MU-UM-019`; global sequential fit yields reasonable
   NSE; every plot returns a `go.Figure`.
3. Execute the notebook end-to-end.

## Scope exclusions

FlowFinity/Ayyeka network calls; CustomTkinter desktop viewer; multiprocessing /
incremental signature-reuse refit; matplotlib (Plotly only); no changes to
`RDIIModel`/RTK.
