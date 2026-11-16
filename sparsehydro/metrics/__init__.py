"""sparsehydro.metrics — goodness-of-fit metrics and calibration objectives.

* :mod:`.functions` — plain metric functions (``nash_sutcliffe``, ``kge``,
  ``pbias``, ``concordance_correlation``, ...), the single implementation of
  each formula.
* :mod:`.objectives` — :class:`IObjective` and the calibration objectives built
  on those functions, plus :data:`objective_registry`.
* :mod:`.events` — scoring a prediction event by event.
* :mod:`.scorecard` — criteria with pass and fail bands graded together.
"""

from .events import per_event_scores, summarize_event_scores
from .functions import (
    concordance_correlation,
    fit_summary,
    index_of_agreement,
    kge,
    log_nse,
    mae,
    max_window_sum,
    mse,
    nash_sutcliffe,
    paired_summary,
    pbias,
    peak_regression_slope,
    peak_weighted_mse,
    rmse,
    sum_above,
    top_peak_indices,
    volume_relative_error,
)
from .scorecard import Criterion, Scorecard, ScorecardResult
from .objectives import (
    KGE,
    MAE,
    MSE,
    PBIAS,
    RMSE,
    Con15,
    ConcordanceCorrelationCoefficient,
    DisjointWindowCCC,
    IndexOfAgreement,
    IObjective,
    LogNSE,
    NashSutcliffe,
    PeakWeightedMSE,
    VolumeRelativeError,
    WeightedRMSE,
    objective_registry,
)

__all__ = [
    # functions
    "mse", "rmse", "mae", "nash_sutcliffe", "log_nse", "kge", "pbias",
    "volume_relative_error", "index_of_agreement", "concordance_correlation",
    "peak_weighted_mse", "top_peak_indices", "peak_regression_slope", "fit_summary",
    "max_window_sum", "sum_above", "paired_summary",
    # scorecards
    "Criterion", "Scorecard", "ScorecardResult",
    # objectives
    "IObjective", "MSE", "RMSE", "WeightedRMSE", "MAE", "PeakWeightedMSE", "NashSutcliffe",
    "KGE", "PBIAS", "VolumeRelativeError", "LogNSE", "IndexOfAgreement",
    "ConcordanceCorrelationCoefficient", "DisjointWindowCCC", "Con15", "objective_registry",
    # per-event scoring
    "per_event_scores", "summarize_event_scores",
]
