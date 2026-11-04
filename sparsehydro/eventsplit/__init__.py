"""sparsehydro.eventsplit — event-based train/validation partitioning.

This package moves the calibration train/validation split from a single
chronological cutoff to an **event-based** partition:

- Each storm *event* (labelled by an ``event_ids`` array, ``-1`` between storms)
  is assigned *whole* to either the training or the validation set.
- Assignment is **stratified** by storm magnitude and season and **interwoven**
  in time, so neither set is dominated by one season or by the largest storms
  (which a first-N-months chronological cutoff cannot guarantee).
- Validation storms are surrounded by an unscored **warmup** lead-in and a
  dynamic **recession buffer** that are excluded from the training set, so a
  training storm never seeds the model state that a validation storm is graded
  on (no antecedent-condition leakage).

Public API
----------
- :class:`EventSplit` — the partition result (event assignments + boolean masks)
- :func:`stratified_event_split` — build a partition from an ``event_ids`` array
- :func:`estimate_recession_hours` — recession time constant used for the warmup
- :func:`per_event_scores` — per-event KGE/CCC (one row per event)
- :func:`summarize_event_scores` — mean / median / IQR across events
"""

from __future__ import annotations

from .split import EventSplit, estimate_recession_hours, stratified_event_split
from .metrics import per_event_scores, summarize_event_scores

__all__ = [
    "EventSplit",
    "stratified_event_split",
    "estimate_recession_hours",
    "per_event_scores",
    "summarize_event_scores",
]
