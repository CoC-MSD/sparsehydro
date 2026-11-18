"""Baseline (sanitary / dry-weather) flow: shapes, pre-fit estimators, components."""

from ._shapes import CenteredWeeklyPattern, ObservedBackground, SanitaryTemplate, validate_background_options
from .components import CenteredWeeklyPatternModel, ObservedBackgroundModel, SanitaryTemplateModel
from .fitting import (
    PatternEstimate,
    estimate_observed_background,
    estimate_sanitary_shapes,
    profile_slot_levels,
    slot_of_week,
)

__all__ = [
    "SanitaryTemplate",
    "CenteredWeeklyPattern",
    "ObservedBackground",
    "validate_background_options",
    "SanitaryTemplateModel",
    "CenteredWeeklyPatternModel",
    "ObservedBackgroundModel",
    "PatternEstimate",
    "estimate_observed_background",
    "estimate_sanitary_shapes",
    "profile_slot_levels",
    "slot_of_week",
]
