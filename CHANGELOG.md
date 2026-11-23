# Changelog

All notable changes to this project are documented in this file. This project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html) and the
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.

## [Unreleased]

### Added

- `models.baseflow.estimate_sanitary_template`: hourly and daily multipliers
  that scale a level, estimated from whole dry days, as a `SanitaryTemplate`.
- `models.baseflow.estimate_sanitary_level`: a daily dry-weather level from the
  rise between night-time and day-time flow on whole dry days, smoothed with a
  rolling median and returned as an `ObservedBackground`.

### Changed

- `SanitaryTemplateModel(level_from_input=True)` takes its level from the bound
  input `level_cfs` and calibrates `sanitary_level_scale` (start 1, bounds 0.5
  to 1.5) instead of `sanitary_level_cfs`. The default is unchanged.

## [1.0.0b1] - 2026-11-18

First beta of 1.0.0: the public API (models and the registry, composition and
configuration, metrics and objectives, calibration and solvers) is complete
and expected to change only for problems found during the beta. This release
also contains the changes noted as 1.0.0a2, which was never tagged.

### Added

- Unit hydrograph library on one convention (time in hours, kernels that are
  exact bin averages, so results do not depend on the model time step). New
  formulations, all registered by name: Nash cascade (`nash-uh`), gamma
  (`gamma-uh`), SCS/NRCS dimensionless (`scs-uh`), Snyder (`snyder-uh`), Clark
  (`clark-uh`), ModClark with lumped rainfall (`modclark-uh`), Weibull
  (`weibull-uh`), lognormal (`lognormal-uh`), generalized gamma
  (`gengamma-uh`) and mixtures of any of these (`mixture-uh`). Any UH takes an
  optional delay `td`. See `docs/unit_hydrographs.md`.
- `RDIIModel` accepts any unit hydrograph as a routing component and forwards
  component constraints.
- `CompositeModel` (`composite`): named components wired by bindings
  (`"target.input": "source.output"`), outputs as aliases, sums or products,
  shared parameters, parameter overrides and a fixed parameter order.
  `AbstractionUHModel` is now a preset built on it.
- Configuration API: `build_model`, `build_objectives`, `build_solver` and
  `load_config` (JSON, or YAML with the new `yaml` extra) build any registered
  formulation from plain data. `objective_registry` and `solver_registry` name
  objectives and solvers.
- Whole-flow components, all registered: `SoilGroundwaterStore`,
  `DegreeDaySnow`, `SanitaryTemplateModel`, `CenteredWeeklyPatternModel`,
  `ObservedBackgroundModel`, `ExternalExcess`; pre-fit estimators in
  `sparsehydro.models.baseflow` (including `profile_slot_levels`).
- Whole-flow (sanitary + groundwater infiltration + RDII) models, snow
  partitioning, a soil/groundwater store, the scalar CMA-ES solver
  (`PlatypusCmaEsSolver`), train/validation splits, diagnostics, and MSDGC
  study tooling (from the whole-flow pull request).
- `sparsehydro.studies.msdgc.default_controls()`: the study's default settings,
  previously only in a notebook.
- `sparsehydro.sample_data.load_sample("tc-nb-001")`: one month of 5-minute
  rain and stormflow shipped with the package.
- `CalibrationProblem(skip_infeasible=True)` scores constraint-violating
  parameter vectors with a penalty without running the model.
- `IModel.predict_arrays()` returning model output as plain ndarrays, and
  `CalibrationProblem.evaluate_single(x, index)`.
- `sparsehydro.models.convolution.convolve_causal()`, shared by all routing.
- Contributor guide, code of conduct, pull request and issue templates.
- `sparsehydro.metrics`: one implementation of each goodness-of-fit formula as
  a plain function (`mse`, `rmse`, `mae`, `nash_sutcliffe`, `log_nse`, `kge`,
  `pbias`, `volume_relative_error`, `index_of_agreement`,
  `concordance_correlation`, `peak_weighted_mse`, `top_peak_indices`,
  `peak_regression_slope`, `fit_summary`). Objectives, plots and the study
  code all call these instead of their own copies.
- `sparsehydro.calibration.selection` (`feasible_history`,
  `select_peak_filtered_pareto`), `calibration.splits.masks`
  (`dry_weather_mask`, `storm_scoring_mask`, `melt_scoring_mask`),
  `calibration.diagnostics.parameter_table` and a public `identify_pareto`.
- `sparsehydro.models.presets` for ready-made compositions.
- `VerificationError`, raised when an internal consistency check fails
  (previously bare `numpy.testing` assertions).
- `sparsehydro.studies.msdgc.enable_notebook_logging()`: study progress is now
  logged rather than printed; call this to see it in a notebook.
- `examples/`: four runnable scripts on the bundled sample (fitting, comparing
  formulations, composing from configuration, per-event scoring), run by
  `tests/test_examples.py`.
- Documentation page "Composing models" (`docs/composition.md`).
- `CalibrationProblem` warns when a unit hydrograph's bounds allow responses
  so slow that much of the search range predicts almost no flow within the
  data record (`calibration.diagnostics.slow_response_warnings`,
  `UnitHydrographBase.slowest_median_hours`). Default bounds are unchanged.
- `sparsehydro.load_extensions()` (in `sparsehydro.registry`): imports the
  modules installed packages declare in the `sparsehydro.extensions`
  entry-point group, plus any modules or `.py` files passed to it, so models,
  objectives and solvers defined outside sparsehydro register and can be built
  by name. Failures are logged and reported instead of raised (or raised with
  `strict=True`). See "Packaging an extension" in `docs/composition.md`.
- `tests/test_layering.py` enforces the package layers (see `CONTRIBUTING.md`):
  no upward imports, nothing imports `sparsehydro.studies`, and package
  `__init__` files only re-export.
- Parsimonious event detection with hierarchical global and sub-event records,
  variable Savitzky-Golay smoothing, peak detection, and event-zone analysis.
- Sequential and global sequential unit-hydrograph fitting with zonal
  weighting.
- Unit-hydrograph, event hierarchy, convolution, Pareto knee,
  parameter-bounds, and model-component visualizations.
- Abstraction and composite model support for integrated rainfall-runoff
  workflows.
- `calibration.PooledEventFitter`: one parameter set fitted to all global
  events together (the model runs over the whole record; a weighted RMSE over
  the events' peak and tail zones is minimised), scored per event
  (`PooledFitResult.event_scores`) — the comparison for the per-event fits.
- `GlobalSequentialFitter.fit` takes `progress(done, total, text)`,
  `should_stop()` (raises `InterruptedError`) and `warm_start` (start each
  event from the previous fit, the default, or from the model's values).
- `sparsehydro.metrics`: `max_window_sum` (largest moving-window sum, earliest
  window on ties), `sum_above` (amount above a threshold), `paired_summary`
  (CCC, r², slopes and mean ratio of paired values, e.g. event volumes) and
  `metrics.scorecard` (`Criterion`, `Scorecard`): criteria with pass and fail
  bands graded pass / conditional / fail. The MSDGC zone volumes use
  `max_window_sum` (same results). Lin's CCC is tested against the worked
  example of the MSDGC Modeling Guidelines (Rev 5.8, Table 7-1).
- `sparsehydro.studies.msdgc.scorecard.assess_calibration`: the MSDGC
  calibration scorecard of the Modeling Guidelines Rev 5.8 (Volume II 6.4 and
  7.1): storm volume with the base flow adjusted to each storm over the 24 h
  before it (or total volume), zone volumes, Lin's CCC per zone and the Cinci
  score for all, critical (April-July) and other months, Con15 as a percent
  error, Qcritical at four levels from ODWF-A (Table 7-2; Qc1 graded), and the
  pass / conditional / fail grade, with warnings for any missing input (the
  result is then not official). `TOOL_ZONES` (the v5.0.5 tool, treatment
  capped at 24 h) and `GUIDELINE_ZONES` (no cap). The existing total-flow
  `MSDGCAssessment` is unchanged.
- Model traces, a secondary output for plotting and checking:
  `IModel.predict_trace()` returns the outputs plus every registered trace
  field (`register_trace_field`, `trace_fields`, `trace_field_metadata`).
  `FieldRecord` gains `role` (`output`, `component`, `state`, `flux`) and
  `component_of`. `predict_arrays()` and calibration are unchanged.
  `RDIIModel` reports each unit hydrograph's flow (`rdii_component_i`) and its
  abstraction model's trace (`ia.*`); `MixtureUH` and `PeakTailUH` report their
  parts; `EnsembleModel` its weighted contributions (sum mode) and each
  child's outputs and trace as `alias.field`;
  `CompositeModel` every child output and trace as `child.field`, with the
  terms of a summed output as its components. Components always add up to
  the series they belong to. See "Components and internal states" in
  `docs/composition.md`.
- Internal states in the trace: `IAModel` records the remaining abstraction
  capacity (`ia_available`) and, with snow, `swe`, `snowfall` and `melt`; the
  tank models record `tank_storage` and the effective-area multiplier
  `tank_gain`. `compute_excess_series` and the compiled loops take an optional
  `record` and are unchanged without it (same results, same speed). The
  existing outputs of `SoilGroundwaterStore` and `DegreeDaySnow` are labelled
  with their roles (`S_in`, `G_in`, `SWE_in` states; recharge, infiltration,
  melt fluxes).

### Changed

- The version is defined once, in `sparsehydro/__init__.py` (`pyproject.toml`
  reads it), and the package is classified as beta.
- Beta deprecation policy (`CONTRIBUTING.md`): public names are not removed
  or renamed without a `DeprecationWarning` period.
- Releases publish to PyPI through trusted publishing after the full test
  matrix, a tag/version/changelog check, `twine check` and a wheel install;
  optional TestPyPI rehearsal. Setup and steps are in `RELEASING.md`.
- Expanded calibration objectives and solver support for weighted and
  multi-objective workflows.
- Improved time-series parsing and plotting performance for large monitoring
  datasets.
- **BREAKING:** modules regrouped so each package has one job and imports
  only from the layers below it. Names exported from a package's
  `__init__` (for example `from sparsehydro.calibration import NashSutcliffe`,
  `from sparsehydro.events import detect_events`) still work; imports from the
  old module paths do not. Old to new:

  | Old | New |
  |---|---|
  | `calibration.objectives` | `metrics.objectives` |
  | `calibration.splits.metrics` | `metrics.events` |
  | `models.unithydrograph.sequential` (and its names in `models.unithydrograph`) | `calibration.sequential` |
  | `models.wholeflow_compact`, `models.wholeflow_pattern` | `models.presets.wholeflow` |
  | baseflow names re-exported from `models.wholeflow_pattern` | `models.baseflow` |
  | snow names re-exported from `models.rdii.initial_abstraction` | `models.forcing` |
  | `events.imported` | `events.io` |
  | logic in `events/__init__`, `filters/__init__` | `events.records`, `events.detection`, `events.io`, `filters.savgol` |
  | `data.wholeflow` (assembly) | `data.assembly` |
  | `data.wholeflow` (FlowFinity, air temperature) | `studies.msdgc.data` |
  | `events.msdgc` | `studies.msdgc.detector` (`events_to_dataframe` renamed `msdgc_events_to_dataframe`) |
  | `calibration.msdgc_assessment`, `calibration.wholeflow_acceptance` | `studies.msdgc.assessment` |
  | `calibration.msdgc_pareto` | `studies.msdgc.pareto` |
  | `calibration.pattern_workflow` | `studies.msdgc.workflow` |
  | `calibration.pattern_reporting` | `studies.msdgc.reporting` |
  | `calibration.wholeflow_workflow` | `calibration.selection`, `calibration.splits.masks`, `calibration.diagnostics`, `studies.msdgc.events`, `studies.msdgc.reporting` |
  | study settings in `configuration` (`ModelSettings`, `CalibrationSettings`, `StudySettings`, ...) | `studies.msdgc.settings` |
  | `visualization.wholeflow_diagnostics` | `visualization.wholeflow` (generic plots); `studies.msdgc.plots` (MSDGC figures) |
  | `visualization.wholeflow_presentation` | `studies.msdgc.plots` |
  | `VisualizationModel` in `visualization.timeseries` | `visualization.model` |

  Pickled calibration problems or results that reference
  `sparsehydro.calibration.objectives` classes will not load.
- **BREAKING:** `download_flowfinity_inputs` takes `username`/`password` (or a
  `client`) and otherwise reads `FLOWFINITY_USERNAME`/`FLOWFINITY_PASSWORD`;
  the `auth_dir` argument, which imported a credentials module from
  `sys.path`, is gone.
- `select_peak_filtered_pareto` predicts through `predict_arrays()` and takes
  the output name (`predicted="total_cfs"`) instead of calling
  `predict_total_cfs`.
- Tank abstraction models (`tank-constant`, `tank-linear`, `tank-sqrt`) run in
  a compiled kernel: 15-40x faster (a month of 5-minute data went from 22-64 ms
  to 0.5-3 ms per run), with results identical to the previous loop. A
  subclass with its own `_drainage_full` still runs in Python.
- Tank models register `ae_min_le_ae_max` as an inequality constraint, so
  optimisers respect `ae_min <= ae_max` (previously only `validate()` checked
  it, which calibration does not call).
- `RDIIModel` passes bounds and `calibrate` flags (not only values) down to
  its abstraction model and unit hydrographs, and `CompositeModel` now also
  passes `calibrate` flags. Predictions are unchanged.
- Importing `sparsehydro` or `sparsehydro.models` no longer loads calibration,
  events, filters, metrics or plotting code.
- **BREAKING:** unit hydrograph time parameters are in hours and renamed
  (`GammaUH(tt, tp)` -> `alpha`, `theta`; `NashUH(k)` -> `K`;
  `TriangleUH(tp, tt)` -> `Tp`, `Tb`; `RectangleUH(tr)` -> `Tr`;
  `DecayUH(alpha)` -> `K`; `PeakTailUH` likewise). Old names raise `TypeError`.
- **BREAKING:** unit hydrograph prediction is `Q = A * conv(rain, kernel)`
  without a `dt` factor, so `A_new = A_old * dt_hours`; default `A` is 1.0.
  `SequentialFitSummary.calibrated_areas()` divides by `dt_hours` to stay on
  the per-step effective-area scale.
- **BREAKING:** `sparsehydro.eventsplit` moved to
  `sparsehydro.calibration.splits`.
- **BREAKING:** `FeasibleCalibrationProblem` removed; use
  `CalibrationProblem(..., skip_infeasible=True)`.
- **BREAKING:** `CalibrationProblem` no longer accepts `predict_fn`; calibrate
  on `predict_arrays()` outputs through `column_map`.
- `CompactWholeFlowModel` and `PatternWholeFlowModel` are presets over the
  whole-flow components. Parameter names, order, bounds, constraints, outputs
  and budgets match the previous classes (pinned in `tests/golden/wholeflow.npz`).
- Sequential fitters start from the model's current parameter values instead
  of the middle of the bounds.
- `RTKTriangle` lives in `sparsehydro.models.unithydrograph.rtk` (still
  importable from `sparsehydro.models.rdii`); its results are unchanged.
- Calibration evaluation no longer round-trips through pandas (a `PeakTailUH`
  evaluation over 600 steps with four objectives went from 4390 us to 181 us).
- Convolution selects direct summation or FFT overlap-add by the cost `n*m`.
  RDII results change at FFT round-off level (~1e-12 relative).
- Tests write plot artifacts to a temporary directory, never into the
  repository.
- pymoo is no longer installed with sparsehydro: it moves to the optional
  `pymoo` extra (`pip install sparsehydro[pymoo]`; included in `dev`). The
  `"nsga2"` solver already uses Platypus's NSGA-II.

### Deprecated

- `NSGAIISolver` and the `"pymoo_nsga2"` solver (pymoo's NSGA-II) emit a
  `DeprecationWarning` and will be removed in a later release; use the
  `"nsga2"` solver (`n_evaluations = pop_size * n_gen` gives a comparable
  budget).

### Removed

- **BREAKING:** `sparsehydro.viewer`, the `viewer` extra and the
  `sparsehydro-viewer` command. The viewer is now a separate application
  (`sparsehydro.viewer` repository, package `sparsehydro_viewer`).
- **BREAKING:** `lin_ccc`; use `sparsehydro.metrics.concordance_correlation`
  (`undefined="nan"` reproduces its NaN behaviour).
- Notebooks, Git LFS, generated plot artifacts and most bundled monitoring
  data from the repository (tracked data went from about 63 MB to under 1 MB).
- The bundled `swmmrdii` package, which could not be imported.
- `AcceptanceContract.qcritical_cfs`.
- **BREAKING:** `sparsehydro.models.wholeflow` (`WholeFlowModel`,
  `GwiStoreModelB`, `RTKOnFrozenExcess`). Build these as composites of the
  whole-flow components. The "profiled" sanitary mode is now the pre-fit
  function `profile_slot_levels`.

### Fixed

- Initializing a model twice no longer registers its constraints twice
  (`RDIIModel`, `EnsembleModel`, `SeasonalityModel`, `AMMModel` and the
  whole-flow models were affected).
- Long unit hydrograph responses are truncated by duration and renormalised
  instead of being cut at a fixed number of steps, which dropped part of the
  tail; extreme parameters no longer produce NaN kernels.
- Ten test modules deleted in the whole-flow pull request are restored, and no
  test depends on a notebook.

## [0.1.0a1]

First tagged pre-release.

[Unreleased]: https://github.com/CoC-MSD/sparsehydro/compare/v1.0.0b1...HEAD
[1.0.0b1]: https://github.com/CoC-MSD/sparsehydro/compare/v0.1.0a1...v1.0.0b1
[0.1.0a1]: https://github.com/CoC-MSD/sparsehydro/releases/tag/v0.1.0a1
