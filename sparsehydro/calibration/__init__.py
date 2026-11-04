"""sparsehydro.calibration — general-purpose model calibration abstractions.

Public API
----------

**Objectives**

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Name
     - Description
   * - :class:`IObjective`
     - Abstract objective function interface
   * - :class:`MSE`
     - Mean squared error
   * - :class:`RMSE`
     - Root mean squared error
   * - :class:`MAE`
     - Mean absolute error
   * - :class:`PeakWeightedMSE`
     - Flow-weighted MSE
   * - :class:`NashSutcliffe`
     - Nash-Sutcliffe efficiency
   * - :class:`KGE`
     - Kling-Gupta efficiency
   * - :class:`PBIAS`
     - Absolute percent bias
   * - :class:`VolumeRelativeError`
     - Relative total volume error
   * - :class:`LogNSE`
     - NSE on log-transformed flows
   * - :class:`IndexOfAgreement`
     - Willmott's index of agreement
   * - :class:`ConcordanceCorrelationCoefficient`
     - Lin's concordance correlation coefficient

**Problem & results**

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Name
     - Description
   * - :class:`CalibrationProblem`
     - Wraps any IModel + observed + objectives
   * - :class:`CalibrationResult`
     - Pareto front + generation history
   * - :class:`GenerationRecord`
     - Per-generation population snapshot

**Solvers**

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Name
     - Description
   * - :class:`ISolver`
     - Abstract solver interface
   * - :class:`NSGAIISolver`
     - NSGA-II (requires pymoo)
   * - :class:`ParticleSwarmSolver`
     - SMPSO / OMOPSO (requires platypus-opt)
   * - :class:`PlatypusSolver`
     - Any Platypus algorithm (requires platypus-opt)
   * - :class:`ScipySolver`
     - SciPy single-objective (requires scipy)
   * - :class:`OptunaCmaEsSolver`
     - Single-objective CMA-ES, multi-seed / parallel (requires optuna)

**Train / validation splits** (:mod:`~sparsehydro.calibration.splits`)

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Name
     - Description
   * - :class:`SplitConfig`
     - Spin-up warmup + continuous / stratified split knobs
   * - :func:`prepend_spinup`
     - Synthetic pre-history so no real record is spent on warmup
   * - :func:`build_masks`
     - Warmup / train / validation step masks (:class:`SplitMasks`)
   * - :func:`label_events_by_split`
     - Assign storms to warmup / train / val; count edge-straddlers
"""

from .callbacks import ProgressCallback
from .objectives import (
    IObjective,
    KGE,
    MAE,
    MSE,
    RMSE,
    WeightedRMSE,
    NashSutcliffe,
    PeakWeightedMSE,
    PBIAS,
    VolumeRelativeError,
    LogNSE,
    IndexOfAgreement,
    ConcordanceCorrelationCoefficient,
    Con15,
    DisjointWindowCCC,
)
from .problem import CalibrationProblem
from .result import CalibrationResult, GenerationRecord, _identify_pareto
from .solvers import ISolver, NSGAIISolver, OptunaCmaEsSolver, ParticleSwarmSolver, PlatypusSolver, ScipySolver
from .splits import (
    SplitConfig,
    SplitMasks,
    build_masks,
    dry_edge_flags,
    label_events_by_split,
    prepend_spinup,
    wrap_seasonal_slip_days,
)

__all__ = [
    # Objectives
    "IObjective",
    "MSE",
    "RMSE",
    "WeightedRMSE",
    "MAE",
    "PeakWeightedMSE",
    "NashSutcliffe",
    "KGE",
    "PBIAS",
    "VolumeRelativeError",
    "LogNSE",
    "IndexOfAgreement",
    "ConcordanceCorrelationCoefficient",
    "Con15",
    "DisjointWindowCCC",
    # Problem
    "CalibrationProblem",
    # Result
    "CalibrationResult",
    "GenerationRecord",
    # Solvers
    "ISolver",
    "NSGAIISolver",
    "OptunaCmaEsSolver",
    "ParticleSwarmSolver",
    "PlatypusSolver",
    "ScipySolver",
    # Splits
    "SplitConfig",
    "SplitMasks",
    "prepend_spinup",
    "dry_edge_flags",
    "build_masks",
    "label_events_by_split",
    "wrap_seasonal_slip_days",
    # Callbacks
    "ProgressCallback",
]
