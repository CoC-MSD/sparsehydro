"""MSDGC v5.0.5 total-flow comparison rules, not its effective-storm scorecard."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import ClassVar

import numpy as np
import pandas as pd

from .objectives import Con15, _peak_regression_slope, _top_peak_indices
from .wholeflow_acceptance import AcceptanceContract, AcceptanceEvaluator, lin_ccc


CFS_H_TO_MG = 7.480519 * 3600 / 1e6


@dataclass(frozen=True)
class MSDGCSettings:
    """Saved zone rules; lengths are rounded to native sample counts."""

    fractions: tuple[float, float, float] = (0.042, 0.25, 0.75)
    minimum_minutes: tuple[float, float, float] = (15.0, 20.0, 30.0)
    maximum_minutes: tuple[float, float, float] = (30.0, 240.0, 1440.0)
    version: str = "msdgc-5.0.5-total-flow-moving-volumes-v1"
    zones: ClassVar[tuple[str, ...]] = ("conveyance", "storage", "treatment")

    def __post_init__(self) -> None:
        values = np.asarray([self.fractions, self.minimum_minutes, self.maximum_minutes])
        if (values.shape != (3, 3) or not np.isfinite(values).all()
                or (values <= 0).any() or (values[1] > values[2]).any()):
            raise ValueError("Three finite positive zone fractions and ordered minute limits are required")

    def window(self, duration_hours: float, dt_hours: float, zone: str) -> tuple[float, int]:
        if (not np.isfinite([duration_hours, dt_hours]).all()
                or duration_hours <= 0 or dt_hours <= 0):
            raise ValueError("Event duration and native timestep must be finite and positive")
        i = self.zones.index(zone)
        minutes = float(np.clip(
            round(float(duration_hours), 2) * 60 * self.fractions[i],
            self.minimum_minutes[i], self.maximum_minutes[i],
        ))
        return minutes, int(round(minutes / (dt_hours * 60)))


def assessment_event_inventory(
    times: pd.DatetimeIndex, approved: pd.DataFrame,
    train: np.ndarray, validation: np.ndarray, *,
    excluded_event_ids: tuple[str, ...] = ("87",),
    full_inventory: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Preserve source approval and audit exclusions without changing fitting masks.

    Hybrid assessment uses full EVT timestamps for approved IDs. Like MSDGC's
    lookup, the scan includes the first native sample on or after the end.
    Whole-event split/quality checks include that sample; no clipping or leakage.
    """
    times = pd.DatetimeIndex(times)
    train, validation = np.asarray(train, bool), np.asarray(validation, bool)
    if (len(times) < 2 or times.hasnans or times.has_duplicates
            or not times.is_monotonic_increasing or times.tz is not None
            or train.shape != (len(times),) or validation.shape != train.shape
            or (train & validation).any()):
        raise ValueError("Assessment requires ordered native timestamps and aligned disjoint split masks")
    source = approved.copy()
    if source.event_id.astype(str).duplicated().any():
        raise ValueError("Assessment event IDs must be unique")
    source["source_start"], source["source_end"] = source["start"], source["end"]
    source["assessment_window_source"] = "approved"
    if full_inventory is not None:
        full = full_inventory.copy()
        full.index = full.event_id.astype(str)
        if full.index.has_duplicates or not source.event_id.astype(str).isin(full.index).all():
            raise ValueError("Full EVT inventory must contain each approved event ID exactly once")
        for column in ("start", "end"):
            source[column] = source.event_id.astype(str).map(full[column])
        source["assessment_window_source"] = "full_evt"
    excluded = set(map(str, excluded_event_ids))
    for index, event in source.iterrows():
        start = int(times.searchsorted(event["start"], side="left"))
        end = int(times.searchsorted(event["end"], side="left"))
        reason = str(event["exclusion"]) if not event["eligible"] else ""
        if str(event["event_id"]) in excluded:
            reason = ";".join(filter(None, [reason, "assessment_criteria_exclusion"]))
        split = "excluded"
        if event["start"] < times[0] or event["end"] > times[-1]:
            reason = reason or "outside_model_record"
        elif end - start + 1 < 2:
            reason = reason or "insufficient_model_samples"
        elif train[start:end + 1].all():
            split = "train"
        elif validation[start:end + 1].all():
            split = "validation"
        else:
            reason = reason or "assessment_endpoint_crosses_split_or_unscored"
        source.loc[index, ["start_idx", "end_idx", "split", "eligible", "exclusion"]] = [
            start, end, split, not reason, reason,
        ]
    source["duration_hours"] = (
        (pd.to_datetime(source["end"]) - pd.to_datetime(source["start"])).dt.total_seconds() / 3600
    ).round(2)
    return source


def _zone_window(
    values: np.ndarray, start: int, end: int, duration: float,
    dt_hours: float, settings: MSDGCSettings, zone: str,
) -> dict:
    minutes, count = settings.window(duration, dt_hours, zone)
    result = dict(requested_minutes=minutes, samples=count, effective_minutes=count * dt_hours * 60,
                  volume_mg=math.nan, window_start_idx=math.nan, window_end_idx=math.nan,
                  status="undefined", reason="")
    q = values[start:end + 1]
    if count < 1:
        result["reason"] = "Requested window rounds to zero native samples"
    elif count > len(q):
        result["reason"] = "Requested window exceeds event; no outside-event samples are used"
    elif not np.isfinite(q).all() or np.any(q == -99):
        result["reason"] = "Non-finite flow or MSDGC invalid marker in event scan"
    else:
        bins = q * dt_hours * CFS_H_TO_MG
        # Match the native moving sum, including first-winner ties and final endpoint.
        current = float(bins[:count].sum())
        best, position = current, 0
        for i in range(count, len(bins)):
            current -= float(bins[i - count])
            current += float(bins[i])
            if current > best:
                best, position = current, i - count + 1
        result.update(volume_mg=round(best, 4), window_start_idx=start + position,
                      window_end_idx=start + position + count - 1, status="defined")
    return result


class MSDGCCon15(Con15):
    """``con15_error`` on independent conveyance-window volumes in MG.

    Retains Con15's original-axis mask validation and stable observed-only
    ranking. Native timestep, event durations and zone rules are frozen; no
    validation observations or candidate-dependent window cache is used.
    Legacy ``Con15`` remains instantaneous-peak-based for other notebooks.
    """

    basis = "msdgc_independent_conveyance_volume_mg"

    def __init__(
        self, event_windows: np.ndarray, duration_hours: np.ndarray, dt_hours: float,
        top_fraction: float = 0.15, mask: np.ndarray | None = None, *,
        settings: MSDGCSettings = MSDGCSettings(), allow_overlapping_events: bool = False,
    ) -> None:
        super().__init__(event_windows, top_fraction, mask,
                         allow_overlapping_events=allow_overlapping_events)
        durations = np.asarray(duration_hours, float)
        if (durations.shape != (len(self.event_windows),)
                or not np.isfinite(durations).all() or (durations <= 0).any()):
            raise ValueError("One finite positive duration is required per event")
        settings.window(float(durations[0]), dt_hours, "conveyance")
        self.duration_hours = durations.copy()
        self.duration_hours.setflags(write=False)
        self.dt_hours, self.settings = dt_hours, settings

    def _event_values(self, values: np.ndarray, indices: np.ndarray) -> np.ndarray:
        volumes = []
        for i in indices:
            start, end = self.event_windows[i]
            row = _zone_window(values, start, end, self.duration_hours[i],
                               self.dt_hours, self.settings, "conveyance")
            if row["status"] != "defined":
                raise ValueError(f"MSDGC Con15 event {i}: {row['reason']}")
            volumes.append(row["volume_mg"])
        return np.asarray(volumes)

    def event_values(self, observed: np.ndarray) -> np.ndarray:
        """Return all frozen event conveyance volumes for selection audit."""
        self.selected_event_indices(observed)
        return self._event_values(np.asarray(observed, float), np.arange(len(self.event_windows)))


class MSDGCAssessment(AcceptanceEvaluator):
    """Total-flow reports using ordinary overlap-allowed MSDGC zone rules.

    Qcritical and the official effective-storm grade are intentionally omitted:
    no capacity, segregated baseflow or antecedent-correction inputs are inferred.
    Invalid bands invalidate their aggregate instead of silently dropping events.
    """

    report_metric_names = (
        "con15_slope", "event_volume_ccc", "conveyance_ccc",
        "storage_ccc", "treatment_ccc", "three_zone_mean_ccc",
    )

    def __init__(
        self, observed: np.ndarray, events: pd.DataFrame, score_mask: np.ndarray,
        dry_mask: np.ndarray | float | None = None, dt_hours: float | None = None,
        contract: AcceptanceContract | None = None, *,
        settings: MSDGCSettings = MSDGCSettings(), allow_overlapping_events: bool = False,
        top_fraction: float | None = None,
    ) -> None:
        if np.isscalar(dry_mask) and dt_hours is None:
            dt_hours, dry_mask = float(dry_mask), None
        if dt_hours is None:
            raise ValueError("MSDGC assessment requires a positive timestep")
        score = np.asarray(score_mask)
        if score.dtype != bool or score.shape != np.asarray(observed).shape:
            raise ValueError("Score mask must be boolean and aligned with one-dimensional observations")
        if contract is None:
            contract = AcceptanceContract(top_fraction=0.15 if top_fraction is None else top_fraction)
        elif top_fraction is not None and top_fraction != contract.top_fraction:
            raise ValueError("top_fraction must agree with the supplied acceptance contract")
        if dry_mask is None:
            dry_mask = np.zeros_like(score)
        elif np.isscalar(dry_mask):
            raise ValueError("Dry mask must align with observations")
        required = {"event_id", "start_idx", "end_idx", "duration_hours"}
        if not required.issubset(events.columns):
            raise ValueError(f"Assessment events require columns {sorted(required)}")
        if events.event_id.duplicated().any():
            raise ValueError("Assessment event IDs must be unique")
        settings.window(1.0, dt_hours, "conveyance")
        for duration in events.duration_hours:
            settings.window(duration, dt_hours, "conveyance")
        super().__init__(observed, events, score_mask, dry_mask, dt_hours, contract,
                         allow_overlapping_events=allow_overlapping_events)
        if len(events) and "duration_hours" not in events:
            raise ValueError("MSDGC assessment requires event duration_hours")
        self.settings = settings
        self.top_fraction = contract.top_fraction

    def zone_report(self, prediction: np.ndarray) -> pd.DataFrame:
        sim = np.asarray(prediction, float)
        if sim.shape != self.observed.shape:
            raise ValueError("Predictions and observations must align")
        columns = list(self.events.columns) + [
            "zone", "requested_minutes", "effective_minutes", "n",
            "observed_window_start_idx", "observed_window_end_idx",
            "modeled_window_start_idx", "modeled_window_end_idx",
            "observed_volume_mg", "modeled_volume_mg", "status", "reason",
        ]
        rows = []
        for event in self.events.to_dict("records"):
            for zone in self.settings.zones:
                obs = _zone_window(self.observed, int(event["start_idx"]), int(event["end_idx"]),
                                   event["duration_hours"], self.dt, self.settings, zone)
                mod = _zone_window(sim, int(event["start_idx"]), int(event["end_idx"]),
                                   event["duration_hours"], self.dt, self.settings, zone)
                reason = "; ".join(dict.fromkeys(r for r in [obs["reason"], mod["reason"]] if r))
                rows.append(dict(
                    **event, zone=zone, requested_minutes=obs["requested_minutes"],
                    effective_minutes=obs["effective_minutes"], n=obs["samples"],
                    observed_window_start_idx=obs["window_start_idx"],
                    observed_window_end_idx=obs["window_end_idx"],
                    modeled_window_start_idx=mod["window_start_idx"],
                    modeled_window_end_idx=mod["window_end_idx"],
                    observed_volume_mg=obs["volume_mg"], modeled_volume_mg=mod["volume_mg"],
                    status="undefined" if reason else "defined", reason=reason,
                ))
        return pd.DataFrame(rows, columns=columns)

    def conveyance_report(self, prediction: np.ndarray) -> pd.DataFrame:
        out = self.zone_report(prediction).query("zone == 'conveyance'").copy()
        out["conveyance_minutes"] = out["effective_minutes"]
        for side in ("observed", "modeled"):
            out[f"{side}_conveyance_volume_cfs_h"] = out[f"{side}_volume_mg"] / CFS_H_TO_MG
        return out

    def event_report(self, prediction: np.ndarray) -> pd.DataFrame:
        out = super().event_report(prediction)
        bands = self.conveyance_report(prediction)
        x = bands["observed_volume_mg"].to_numpy(dtype=float)
        selected = (_top_peak_indices(x, self.contract.top_fraction)
                    if np.isfinite(x).all() else np.empty(0, dtype=int))
        out["con15_selected"] = False
        out.iloc[selected, out.columns.get_loc("con15_selected")] = True
        out["observed_conveyance_volume_mg"] = x
        out["modeled_conveyance_volume_mg"] = bands["modeled_volume_mg"].to_numpy()
        out["con15_basis"] = MSDGCCon15.basis
        return out

    def metrics(self, prediction: np.ndarray) -> pd.DataFrame:
        zones = self.zone_report(prediction)
        rows = []
        for zone in self.settings.zones:
            part = zones.loc[zones.zone.eq(zone)]
            x = part.observed_volume_mg.to_numpy(dtype=float)
            y = part.modeled_volume_mg.to_numpy(dtype=float)
            # The tool's Pearson-times-bias expression is undefined for either constant series.
            value = (lin_ccc(x, y) if len(x) >= 2 and np.var(x) > 0 and np.var(y) > 0
                     else math.nan)
            row = self._row(f"{zone}_ccc", value, self.contract.ccc_min, 1, len(part),
                            "Incomplete bands, fewer than two events, or constant/non-finite volume pairs")
            row["n_defined"] = int(part.status.eq("defined").sum())
            rows.append(row)
        mean = float(np.mean([row["value"] for row in rows]))
        rows.append(self._row("three_zone_mean_ccc", mean, self.contract.ccc_min, 1,
                              len(self.events), "One or more zone CCCs are undefined"))
        conveyance = zones.loc[zones.zone.eq("conveyance")]
        x = conveyance.observed_volume_mg.to_numpy(dtype=float)
        y = conveyance.modeled_volume_mg.to_numpy(dtype=float)
        slope = math.nan
        selected = np.empty(0, dtype=int)
        if np.isfinite(x).all():
            selected = _top_peak_indices(x, self.contract.top_fraction)
            slope = _peak_regression_slope(x[selected], y[selected])
        rows.append(self._row("con15_slope", slope, 1 - self.contract.con15_tolerance,
                              1 + self.contract.con15_tolerance, len(selected),
                              "Incomplete selected bands or zero observed conveyance-volume energy"))
        volumes, _ = self._event_values(np.asarray(prediction, float))
        rows.append(self._row("event_volume_ccc", lin_ccc(self.obs_volumes, volumes),
                              self.contract.ccc_min, 1, len(volumes),
                              "Fewer than two complete finite events or degenerate full-event volumes"))
        for row in rows:
            row.update(flow_basis="total_flow", assessment_version=self.settings.version,
                       official_storm_scorecard=False)
        return pd.DataFrame(rows)

    def conveyance_metric(self, prediction: np.ndarray) -> dict:
        return self.metrics(prediction).set_index("metric").loc["conveyance_ccc"].to_dict() | {
            "metric": "conveyance_ccc",
        }
