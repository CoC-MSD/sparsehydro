"""sparsehydro.filters — Savitzky-Golay filtering for stormflow preprocessing.

Provides a self-contained pipeline for smoothing raw stormflow and computing
derivative signals used by the event-detection algorithm in
:mod:`sparsehydro.events`.

Public API
----------
- :class:`FilterResult` — dataclass holding the four signal arrays + thresholds
- :func:`apply_savgol_filter` — computes sg_0, sg_1, sg_2 from a rain/stormflow DataFrame
- :func:`compute_thresholds` — attaches percentile-based detection thresholds

Quick start::

    from sparsehydro.filters import apply_savgol_filter, compute_thresholds

    result = apply_savgol_filter(rain_stormflow_df, window_length=48)
    result = compute_thresholds(result, sg_0_per=0.15, sg_1_per=0.25, sg_2_per=0.25)
    print(result.thresholds)
"""

from .dryweather import DryWeatherResult, disaggregate_dry_weather, generate_weekly_patterns
from .savgol import FilterResult, apply_savgol_filter, compute_thresholds

__all__ = [
    "FilterResult",
    "apply_savgol_filter",
    "compute_thresholds",
    "DryWeatherResult",
    "disaggregate_dry_weather",
    "generate_weekly_patterns",
]
