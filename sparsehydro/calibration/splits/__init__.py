"""Train/validation splitting for calibration.

Two complementary strategies:

* **Time blocks** (:mod:`.blocks`): forcing-only spin-up, then a continuous or
  stratified block split of the record into warmup, training and validation
  masks (:class:`SplitConfig`, :func:`build_masks`, :func:`prepend_spinup`).
* **Whole events** (:mod:`.events`): each storm event goes entirely to training
  or validation, stratified by magnitude and season, with warmup and recession
  buffers around validation storms (:func:`stratified_event_split`).

:mod:`.masks` builds scoring masks (dry weather, storms, snow melt).
Per-event scoring lives in :mod:`sparsehydro.metrics.events` and is re-exported here.
"""

from __future__ import annotations

from .blocks import (
    SPINUP_MODES,
    SPLIT_LABELS,
    SPLIT_MODES,
    SplitConfig,
    SplitMasks,
    build_masks,
    dry_edge_flags,
    label_events_by_split,
    prepend_spinup,
    wrap_seasonal_slip_days,
)
from .masks import dry_weather_mask, melt_scoring_mask, storm_scoring_mask
from .events import EventSplit, estimate_recession_hours, stratified_event_split
from ...metrics.events import per_event_scores, summarize_event_scores

__all__ = [
    "SPINUP_MODES",
    "SPLIT_MODES",
    "SPLIT_LABELS",
    "SplitConfig",
    "SplitMasks",
    "wrap_seasonal_slip_days",
    "prepend_spinup",
    "dry_edge_flags",
    "build_masks",
    "label_events_by_split",
    "dry_weather_mask",
    "storm_scoring_mask",
    "melt_scoring_mask",
    "EventSplit",
    "stratified_event_split",
    "estimate_recession_hours",
    "per_event_scores",
    "summarize_event_scores",
]
