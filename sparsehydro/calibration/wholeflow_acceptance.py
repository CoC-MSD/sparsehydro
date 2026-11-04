"""Shared total-flow reporting infrastructure, without retired scorecard fits."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


def lin_ccc(observed: np.ndarray, modeled: np.ndarray) -> float:
    """Population-moment CCC; incomplete or degenerate pairs are undefined."""
    x, y = np.asarray(observed, float), np.asarray(modeled, float)
    if x.shape != y.shape:
        raise ValueError("CCC arrays must align")
    if x.size < 2 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return math.nan
    denominator = x.var() + y.var() + (x.mean() - y.mean()) ** 2
    return float(2 * np.mean((x - x.mean()) * (y - y.mean())) / denominator) if denominator > 0 else math.nan


@dataclass(frozen=True)
class AcceptanceContract:
    ccc_min: float = 0.90
    con15_tolerance: float = 0.10
    top_fraction: float = 0.15
    version: str = "total-events-conveyance-volume-v2"
    qcritical_cfs: None = None

    def __post_init__(self) -> None:
        if self.qcritical_cfs is not None:
            raise ValueError("Qcritical thresholds are retired from this total-flow workflow")
        if not (0 < self.ccc_min <= 1 and 0 < self.top_fraction <= 1 and self.con15_tolerance > 0):
            raise ValueError("Invalid acceptance tolerances")


class AcceptanceEvaluator(ABC):
    """Shared event inventory; subclasses supply current assessment formulas."""

    def __init__(
        self, observed: np.ndarray, events: pd.DataFrame, score_mask: np.ndarray,
        dry_mask: np.ndarray, dt_hours: float, contract: AcceptanceContract, *,
        allow_overlapping_events: bool = False,
    ):
        self.observed = np.asarray(observed, float).copy()
        self.events = events.copy()
        self.mask = np.asarray(score_mask, bool).copy()
        self.dry = np.asarray(dry_mask, bool) & self.mask
        self.dt, self.contract = dt_hours, contract
        if (
            self.observed.ndim != 1 or self.mask.shape != self.observed.shape
            or self.dry.shape != self.mask.shape or not np.isfinite(dt_hours) or dt_hours <= 0
        ):
            raise ValueError("Invalid score masks or timestep")
        previous_start, previous_end = -1, -1
        for event in self.events.itertuples():
            if not (
                isinstance(event.start_idx, (int, np.integer))
                and isinstance(event.end_idx, (int, np.integer))
                and 0 <= event.start_idx <= event.end_idx < len(self.observed)
            ):
                raise ValueError("Event outside observation domain or non-integer event indices")
            if event.start_idx < previous_start:
                raise ValueError("Assessment events must be ordered by start")
            if not allow_overlapping_events and event.start_idx <= previous_end:
                raise ValueError("Assessment events must be ordered and disjoint")
            if not self.mask[event.start_idx:event.end_idx + 1].all():
                raise ValueError("A scored event must contain only rows from its own split")
            previous_start, previous_end = event.start_idx, event.end_idx
        self.obs_volumes, self.obs_peaks = self._event_values(self.observed)

    def _event_values(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        volumes, peaks = [], []
        for event in self.events.itertuples():
            q = values[event.start_idx:event.end_idx + 1]
            volumes.append(float(q.sum() * self.dt))
            peaks.append(float(q.max()))
        return np.asarray(volumes), np.asarray(peaks)

    @staticmethod
    def _row(name: str, value: float, lower: float, upper: float, count: int, reason: str) -> dict:
        finite = np.isfinite(value)
        passed = finite and lower - 1e-12 <= value <= upper + 1e-12
        return dict(
            metric=name, value=value, lower=lower, upper=upper, n=count,
            status="pass" if passed else "fail" if finite else "undefined",
            reason="" if finite else reason,
        )

    def event_report(self, prediction: np.ndarray) -> pd.DataFrame:
        volumes, peaks = self._event_values(np.asarray(prediction))
        out = self.events.copy()
        out["observed_volume_cfs_h"], out["modeled_volume_cfs_h"] = self.obs_volumes, volumes
        out["observed_peak_cfs"], out["modeled_peak_cfs"] = self.obs_peaks, peaks
        out["con15_selected"] = False
        return out

    @abstractmethod
    def metrics(self, prediction: np.ndarray) -> pd.DataFrame:
        """Return the current assessment's aggregate metrics."""

    @abstractmethod
    def conveyance_report(self, prediction: np.ndarray) -> pd.DataFrame:
        """Return the current assessment's event conveyance volumes."""
