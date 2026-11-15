# Changelog

All notable changes to this project are documented in this file. This project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html) and the
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.

## [Unreleased]

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

### Changed

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

### Removed

- Notebooks, Git LFS, generated plot artifacts and most bundled monitoring
  data from the repository (tracked data went from about 63 MB to under 1 MB).
- The bundled `swmmrdii` package, which could not be imported.
- `AcceptanceContract.qcritical_cfs`.

### Fixed

- Initializing a model twice no longer registers its constraints twice
  (`RDIIModel`, `EnsembleModel`, `SeasonalityModel`, `AMMModel` and the
  whole-flow models were affected).
- Long unit hydrograph responses are truncated by duration and renormalised
  instead of being cut at a fixed number of steps, which dropped part of the
  tail; extreme parameters no longer produce NaN kernels.
- Ten test modules deleted in the whole-flow pull request are restored, and no
  test depends on a notebook.

## [1.0.0a2] - 2026-09-02

### Added

- Parsimonious event detection with hierarchical global and sub-event records, variable Savitzky-Golay smoothing, peak detection, and event-zone analysis.
- Sequential and global sequential unit-hydrograph fitting with zonal weighting.
- Unit-hydrograph, event hierarchy, convolution, Pareto knee, parameter-bounds, and model-component visualizations.
- Abstraction and composite model support for integrated rainfall-runoff workflows.

### Changed

- Expanded calibration objectives and solver support for weighted and multi-objective workflows.
- Improved time-series parsing and plotting performance for large monitoring datasets.

[Unreleased]: https://github.com/CoC-MSD/sparsehydro/compare/v1.0.0a2...HEAD
[1.0.0a2]: https://github.com/CoC-MSD/sparsehydro/releases/tag/v1.0.0a2
