# Changelog

All notable changes to this project are documented in this file. This project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html) and the
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.

## [Unreleased]

### Added

- `IModel.predict_arrays()` returning model output as plain ndarrays. It defaults
  to unwrapping `predict()`, so existing models are unaffected; the unit
  hydrograph, composite, abstraction, seasonality and tank models override it.
- `CalibrationProblem.evaluate_single(x, index)` for solvers that optimise one
  objective, instead of computing all of them and discarding the rest.
- `sparsehydro.models.convolution` with `convolve_causal()`, shared by the unit
  hydrograph and RDII models.
- `UnitHydrographBase`, holding the prepare/predict/finalize lifecycle that was
  duplicated across the seven UH classes, plus `prepare_arrays()` and
  `set_forcing()` for driving a UH without building DataFrames.
- Golden-output regression fixtures (`tests/golden/`) pinning kernels,
  predictions, objective values and end-to-end `evaluate()` vectors.

### Changed

- Calibration evaluation no longer round-trips through pandas. A `PeakTailUH`
  evaluation over a 600-step event window with four objectives went from 4390 us
  to 181 us (24x); `evaluate_single` is 69 us (63x). A full `SequentialFitter`
  run over a three-week window is 7.8x faster, with bit-identical fitted
  parameters and predictions.
- Convolution now selects direct summation or FFT overlap-add by the cost `n*m`
  rather than `max(n, m)`. **This changes RDII results at FFT round-off level
  (~1e-12 relative).** Short windows that previously took the FFT branch now use
  direct summation and are both faster and more accurate; long series are up to
  5.6x faster. The FFT branch clips round-off undershoot below zero when both
  operands are non-negative.
- `AbstractionUHModel` prepares its child models once in `prepare()` rather than
  on every `predict()`.

### Fixed

- Unit hydrograph kernels are capped at `MAX_STEPS` (864) in all models.
  `GammaUH`, `NashUH`, `TriangleUH` and `RectangleUH` were previously uncapped.
  **This changes results where the cap is active** — `GammaUH` with `tp > 172`,
  `NashUH` with `5*n*k > 863`, and `TriangleUH`/`RectangleUH` above 863 steps.
- `NashUH` at large `n` and `k` produced an all-NaN kernel of up to 250,001
  ordinates, costing ~125 ms per evaluation to return a value the optimiser
  could only read as a generic penalty. Kernels are now sanitised for NaN/inf
  and negatives before normalisation, and `NashUH`'s overflowing normalising
  constant — which the unit-area normalisation divides out anyway — is dropped
  when unusable. `PeakTailUH` had the same latent NaN path through its gamma
  tail.
- Raw kernel evaluation runs under a suppressed numpy error state; sweeping the
  parameter box legitimately overflows intermediates, and the warning was
  emitted once per objective evaluation.

## [1.0.0a2] - 2026-09-02

### Added

- Parsimonious event detection with hierarchical global and sub-event records, variable Savitzky-Golay smoothing, peak detection, and event-zone analysis.
- Sequential and global sequential unit-hydrograph fitting with zonal weighting.
- Unit-hydrograph, event hierarchy, convolution, Pareto knee, parameter-bounds, and model-component visualizations.
- Abstraction and composite model support for integrated rainfall-runoff workflows.

### Changed

- Expanded calibration objectives and solver support for weighted and multi-objective workflows.
- Improved time-series parsing and plotting performance for large monitoring datasets.

[1.0.0a2]: https://github.com/MSDGC-SWM/sparsehydro/releases/tag/v1.0.0a2
