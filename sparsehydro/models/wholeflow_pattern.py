"""Centered sanitary shapes jointly calibrated with the existing GWI/RTK physics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sparsehydro.models.wholeflow import WholeFlowModel
from sparsehydro.models.wholeflow import J_INIT_RTK, SUPER_SLOW_RTK
from sparsehydro.models.rdii.initial_abstraction import SnowConfig
from sparsehydro.models.wholeflow_compact import (
    CompactWholeFlowModel, SanitaryTemplate, _TemplateTerm,
)
from sparsehydro.parameters import ConstraintRecord, FieldRecord, ScalarParameter


def validate_background_options(
    window_days: int, minimum_days: int, max_gap_days: int, endpoint_hold_days: int = 0,
) -> None:
    for name, value in (("window_days", window_days), ("minimum_days", minimum_days),
                        ("max_gap_days", max_gap_days)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Background {name} must be a positive integer")
    if minimum_days > window_days:
        raise ValueError("Background minimum_days must not exceed window_days")
    if (isinstance(endpoint_hold_days, bool) or not isinstance(endpoint_hold_days, int)
            or endpoint_hold_days < 0):
        raise ValueError("Background endpoint_hold_days must be a nonnegative integer")


@dataclass(frozen=True)
class ObservedBackground:
    """Frozen training-dry background for retrospective interpolation, not forecasting."""

    datetimes: tuple[pd.Timestamp, ...]
    flow_cfs: tuple[float, ...]
    start: pd.Timestamp
    end: pd.Timestamp
    window_days: int = 21
    minimum_days: int = 3
    max_gap_days: int = 45
    endpoint_hold_days: int = 0

    def __post_init__(self) -> None:
        validate_background_options(
            self.window_days, self.minimum_days, self.max_gap_days, self.endpoint_hold_days)
        t = pd.DatetimeIndex(self.datetimes)
        values = np.asarray(self.flow_cfs, dtype=float)
        if (len(t) < 2 or t.hasnans or t.has_duplicates or not t.is_monotonic_increasing
                or values.shape != (len(t),) or not np.isfinite(values).all()
                or np.any(values < 0)):
            raise ValueError("Observed background needs ordered unique anchors and finite nonnegative flows")
        start, end = pd.Timestamp(self.start), pd.Timestamp(self.end)
        if (pd.isna(start) or pd.isna(end) or start > end
                or start.normalize() != t[0].normalize() or end.normalize() != t[-1].normalize()):
            raise ValueError("Background support must begin/end on its first/last anchor days")
        object.__setattr__(self, "datetimes", tuple(t))
        object.__setattr__(self, "flow_cfs", tuple(float(v) for v in values))
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    def endpoint_hold_mask(self, datetimes: pd.DatetimeIndex) -> np.ndarray:
        """Flag only bounded extensions outside the first/last source observation."""
        t = pd.DatetimeIndex(datetimes)
        if t.hasnans or str(t.tz) != str(self.start.tz):
            raise ValueError("Background timestamps must be valid and use the fitted timezone")
        limit = pd.Timedelta(days=self.endpoint_hold_days)
        return np.asarray(
            ((t < self.start) & (t >= self.start - limit))
            | ((t > self.end) & (t <= self.end + limit)), dtype=bool,
        )

    def evaluate(self, datetimes: pd.DatetimeIndex) -> np.ndarray:
        """Interpolate or hold bounded endpoints; leave long gaps/future dates undefined."""
        t = pd.DatetimeIndex(datetimes)
        held = self.endpoint_hold_mask(t)
        anchors = pd.DatetimeIndex(self.datetimes).as_unit("ns").asi8
        query = t.as_unit("ns").asi8
        # Hold the value at the actual support boundary, not the daily midnight anchor.
        bounded_query = np.clip(query, self.start.value, self.end.value)
        values = np.interp(bounded_query - anchors[0], anchors - anchors[0], self.flow_cfs)
        supported = (t >= self.start) & (t <= self.end)
        right = np.searchsorted(anchors, query, side="left")
        inside = (right > 0) & (right < len(anchors))
        positions = np.flatnonzero(inside)
        gaps = anchors[right[inside]] - anchors[right[inside] - 1]
        # An anchor remains supported even when the preceding gap is too long.
        unsupported = ((gaps > pd.Timedelta(days=self.max_gap_days).value)
                       & (query[inside] != anchors[right[inside]]))
        supported[positions[unsupported]] = False
        return np.where(supported | held, values, np.nan)

    def to_dict(self) -> dict:
        return dict(
            datetimes=[t.isoformat() for t in self.datetimes], flow_cfs=list(self.flow_cfs),
            start=self.start.isoformat(), end=self.end.isoformat(),
            timezone=str(self.start.tz) if self.start.tz is not None else None,
            window_days=self.window_days, minimum_days=self.minimum_days,
            max_gap_days=self.max_gap_days,
            endpoint_hold_days=self.endpoint_hold_days,
        )

    @classmethod
    def from_dict(cls, values: dict) -> ObservedBackground:
        data = dict(values)
        timezone = data.pop("timezone")
        if timezone is not None:
            data["datetimes"] = tuple(pd.to_datetime(data["datetimes"], utc=True).tz_convert(timezone))
            for name in ("start", "end"):
                data[name] = pd.Timestamp(data[name]).tz_convert(timezone)
        return cls(**data)


@dataclass(frozen=True)
class CenteredWeeklyPattern:
    """Empirical cfs anomalies; no absolute sanitary level or annual adjustment."""

    hourly: tuple[float, ...]
    weekday: tuple[float, ...]

    def __post_init__(self) -> None:
        for values, size in ((self.hourly, 24), (self.weekday, 7)):
            array = np.asarray(values, dtype=float)
            if array.shape != (size,) or not np.isfinite(array).all():
                raise ValueError("Sanitary shapes require 24 hourly and seven weekday values")
            if not np.isclose(array.mean(), 0.0, atol=1e-12, rtol=0):
                raise ValueError("Sanitary anomaly shapes must have zero reference-week mean")

    def components(self, datetimes: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
        return (np.asarray(self.hourly)[datetimes.hour],
                np.asarray(self.weekday)[datetimes.dayofweek])


@dataclass
class PatternEstimate:
    pattern: CenteredWeeklyPattern
    detection_baseline_cfs: np.ndarray
    coverage: pd.DataFrame


def estimate_observed_background(
    data: pd.DataFrame, eligible: np.ndarray, pattern: CenteredWeeklyPattern, *,
    window_days: int = 21, minimum_days: int = 3, max_gap_days: int = 45,
    endpoint_hold_days: int = 0,
) -> ObservedBackground:
    """Smooth daily training-dry means after removing the frozen weekly anomalies.

    Anchors exist only on observed eligible days, not in empty rolling windows.
    Smoothing never bridges a source gap exceeding max_gap_days. The window and
    gap limit are explicit estimation assumptions, not a prescribed seasonal shape.
    """
    validate_background_options(window_days, minimum_days, max_gap_days, endpoint_hold_days)
    t = pd.DatetimeIndex(data["datetime"])
    mask = np.asarray(eligible)
    q = data["flow_cfs"].to_numpy(dtype=float)
    if (mask.dtype != bool or mask.shape != q.shape or not mask.any()
            or t.hasnans or t.has_duplicates or not t.is_monotonic_increasing):
        raise ValueError("Background fitting requires aligned boolean eligibility and ordered timestamps")
    if not np.isfinite(q[mask]).all() or np.any(q[mask] < 0):
        raise ValueError("Background fitting requires finite nonnegative training dry-weather flow")
    h, d = pattern.components(t)
    daily = pd.Series(np.where(mask, q - h - d, np.nan), index=t).resample("D").mean().dropna()
    segments = daily.index.to_series().diff().gt(pd.Timedelta(days=max_gap_days)).cumsum()
    slow = pd.concat([
        group.rolling(f"{window_days}D", center=True, min_periods=minimum_days).mean()
        for _, group in daily.groupby(segments)
    ]).dropna()
    if len(slow) < 2:
        raise ValueError("Insufficient supported training dry-weather days for observed background")
    source = t[mask]
    start = max(source[0], slow.index[0])
    end = min(source[-1], slow.index[-1] + pd.DateOffset(days=1) - pd.Timedelta(nanoseconds=1))
    return ObservedBackground(
        tuple(slow.index), tuple(slow.to_numpy()), start, end,
        window_days, minimum_days, max_gap_days, endpoint_hold_days,
    )


def estimate_sanitary_shapes(data: pd.DataFrame, eligible: np.ndarray) -> PatternEstimate:
    """Estimate shapes, not sanitary level, from training dry-weather rows only.

    A centered 21-day trend of eligible daily means removes slow variation before
    a weighted two-way hour/weekday fit. Its intercept is discarded. Neither the
    trend nor filtered observations are used in model prediction. This assumes
    slow GWI relative to the daily pattern; it does not identify sanitary flow.
    """
    t = pd.DatetimeIndex(data["datetime"])
    mask = np.asarray(eligible, dtype=bool)
    q = data["flow_cfs"].to_numpy(dtype=float)
    if (mask.shape != q.shape or len(t) < 2 or t.hasnans or t.has_duplicates
            or not t.is_monotonic_increasing):
        raise ValueError("Shape fitting requires aligned masks and ordered unique timestamps")
    if not mask.any() or not np.isfinite(q[mask]).all() or np.any(q[mask] < 0):
        raise ValueError("Shape fitting requires finite nonnegative training dry-weather flow")
    daily = pd.Series(np.where(mask, q, np.nan), index=t).resample("D").mean()
    slow = daily.rolling(21, center=True, min_periods=3).mean()
    trend = slow.reindex(t.normalize()).to_numpy()
    fit = mask & np.isfinite(trend) & (trend > 0)
    grouped = pd.DataFrame({
        "hour": t.hour[fit], "weekday": t.dayofweek[fit],
        "residual": q[fit] - trend[fit],
    }).groupby(["hour", "weekday"])["residual"].agg(["mean", "count"]).reset_index()
    if grouped["hour"].nunique() != 24 or grouped["weekday"].nunique() != 7:
        raise ValueError("Insufficient training dry-weather coverage for all hours/weekdays")
    design = np.column_stack((
        np.ones(len(grouped)),
        np.eye(24)[grouped["hour"].to_numpy(), 1:],
        np.eye(7)[grouped["weekday"].to_numpy(), 1:],
    ))
    weight = np.sqrt(grouped["count"].to_numpy())
    coefficients, _, rank, _ = np.linalg.lstsq(
        design * weight[:, None], grouped["mean"].to_numpy() * weight, rcond=None,
    )
    if rank != design.shape[1]:
        raise ValueError("Training hour/weekday coverage cannot identify both pattern shapes")
    hourly, weekday = np.r_[0.0, coefficients[1:24]], np.r_[0.0, coefficients[24:]]
    hourly -= hourly.mean()
    weekday -= weekday.mean()
    pattern = CenteredWeeklyPattern(tuple(hourly), tuple(weekday))
    anchors = slow.dropna()
    if len(anchors) < 2:
        raise ValueError("Insufficient training days for the event-detection reference")
    background = np.interp(t.as_unit("ns").asi8, anchors.index.as_unit("ns").asi8,
                           anchors.to_numpy())
    h, d = pattern.components(t)
    reference = background + h + d
    if not np.isfinite(reference).all() or np.any(reference <= 0):
        raise ValueError("Nonpositive event-detection reference; inspect training flow and shapes")
    coverage = pd.DataFrame({
        "month": t.month, "weekday": t.dayofweek, "hour": t.hour,
        "eligible": mask, "shape_fit": fit,
    }).groupby(["month", "weekday", "hour"]).sum().reset_index()
    return PatternEstimate(pattern, reference, coverage)


class _CenteredPatternTerm(_TemplateTerm):
    def __init__(
        self, pattern: CenteredWeeklyPattern, level: float, upper: float,
        observed_background: ObservedBackground | None = None,
    ):
        self.pattern = pattern
        self.level, self.upper = level, upper
        self.hour_values: np.ndarray | None = None
        self.weekday_values: np.ndarray | None = None
        self.observed_background = observed_background
        self.background_values: np.ndarray | None = None

    def register(self, model: WholeFlowModel) -> None:
        if self.observed_background is None:
            model.register_scalar_parameter(ScalarParameter(
                "sanitary_mean_cfs", self.level, 0.0, self.upper, units="cfs",
                description="Jointly fitted sanitary mean over an equally weighted reference week",
            ))
        for name, value, lower, upper, units, description in (
            ("sanitary_hour_scale", 1.0, 0.0, 3.0, "-",
             "Jointly fitted amplitude of the training hourly anomaly"),
            ("sanitary_weekday_scale", 1.0, 0.0, 3.0, "-",
             "Jointly fitted amplitude of the training weekday anomaly"),
        ):
            model.register_scalar_parameter(ScalarParameter(
                name, value, lower, upper, units=units, description=description,
            ))

    def prepare(self, datetimes: pd.DatetimeIndex) -> None:
        self.hour_values, self.weekday_values = self.pattern.components(datetimes)
        self.background_values = (self.observed_background.evaluate(datetimes)
                                  if self.observed_background is not None else None)

    def evaluate(self, model: WholeFlowModel) -> np.ndarray:
        if self.hour_values is None or self.weekday_values is None:
            raise RuntimeError("Prepare the sanitary pattern before prediction")
        g = lambda name: float(model.get_scalar_parameter(name).value)
        if self.observed_background is None:
            level = g("sanitary_mean_cfs")
        elif self.background_values is None:
            raise RuntimeError("Prepare the observed background before prediction")
        else:
            level = self.background_values
        baseline = (level + g("sanitary_hour_scale") * self.hour_values
                    + g("sanitary_weekday_scale") * self.weekday_values)
        return baseline


class PatternWholeFlowModel(CompactWholeFlowModel):
    """Centered sanitary patterns + IA/RTK, with optional groundwater and snow.

    Reuses the existing Numba recurrence, strict forcing preparation, convolution,
    and complete rainfall-budget implementation. Unlike the compact experiment,
    no IA coefficients are fixed and initial storage defaults to fixed zero for
    the original notebook's forcing-only spin-up. No observed flow is required.
    Optional bounded snow calibration is inherited from the compact model.
    A frozen ``observed_background`` replaces the fitted mean for retrospective
    analysis without GWI. ``sanitary_cfs`` then includes unidentified slow inflow;
    it is independent of the rainfall budget and is not a forecast.
    """

    model_name = "wholeflow-pattern-joint"

    def __init__(
        self, pattern: CenteredWeeklyPattern, area_acres: float, dt_hours: float,
        sanitary_mean_cfs: float, sanitary_upper_cfs: float, *, snow_config: SnowConfig | None = None,
        calibrate_snow: bool = False,
        groundwater_enabled: bool = True, n_triangles: int = 3,
        observed_background: ObservedBackground | None = None,
        annual_seasonality: bool = False,
    ):
        if isinstance(n_triangles, bool) or not isinstance(n_triangles, int) or not 1 <= n_triangles <= 4:
            raise ValueError("n_triangles must be an integer from 1 to 4")
        if annual_seasonality is not False:
            raise ValueError("Calendar annual_seasonality has been removed; use a training-derived observed_background")
        if observed_background is not None:
            if not isinstance(observed_background, ObservedBackground):
                raise ValueError("observed_background must be an ObservedBackground")
            if groundwater_enabled:
                raise ValueError("Observed background and physical GWI are mutually exclusive")
        init_rtk = tuple((*J_INIT_RTK, SUPER_SLOW_RTK)[:n_triangles])
        super().__init__(SanitaryTemplate((1.0,) * 24, (1.0,) * 7),
                         area_acres, dt_hours, sanitary_mean_cfs, sanitary_upper_cfs,
                         snow_config=snow_config, calibrate_snow=calibrate_snow,
                         groundwater_enabled=groundwater_enabled, init_rtk=init_rtk)
        self.sanitary = _CenteredPatternTerm(
            pattern, sanitary_mean_cfs, sanitary_upper_cfs, observed_background)
        self.pattern = pattern
        self.observed_background = observed_background

    def initialize(self) -> None:
        super().initialize()
        for name in ("ia_k_dep", "ia_theta"):
            self.get_scalar_parameter(name).calibrate = True
        for name in ("s_initial_fraction", "g_initial_fraction"):
            self.get_scalar_parameter(name).update(value=0.0, calibrate=False)
        self.register_inequality_constraint(ConstraintRecord(
            name="sanitary_nonnegative",
            description="Minimum background plus weekly anomalies >= 0, without clipping",
        ))
        if self.observed_background is not None:
            self.register_output_field(FieldRecord(
                name="slow_background_cfs", units="cfs",
                description="Frozen training-dry background, already included in sanitary_cfs; not a rainfall flux",
            ))
            self.register_output_field(FieldRecord(
                name="background_supported", units="-",
                description="Background has interpolation support or an explicitly bounded endpoint hold",
            ))
            self.register_output_field(FieldRecord(
                name="background_endpoint_held", units="-",
                description="Background uses a constant source-boundary value, not interpolation",
            ))

    def prepare(self, data: pd.DataFrame) -> None:
        if self.observed_background is not None:
            values = self.observed_background.evaluate(pd.DatetimeIndex(data["datetime"]))
            required = (data["background_required"].to_numpy()
                        if "background_required" in data else np.ones(len(data), dtype=bool))
            if required.dtype != bool or required.shape != values.shape:
                raise ValueError("background_required must be an aligned boolean column")
            missing = required & ~np.isfinite(values)
            if missing.any():
                first = data.loc[missing, "datetime"].iloc[0]
                t = pd.DatetimeIndex(data["datetime"])
                before = int((missing & (t < self.observed_background.start)).sum())
                after = int((missing & (t > self.observed_background.end)).sum())
                interior = int(missing.sum()) - before - after
                raise ValueError(
                    f"Observed background lacks support for {int(missing.sum())} required rows, first {first}. "
                    f"Before source support: {before}; interior gaps: {interior}; after source support: {after}. "
                    f"Endpoint hold limit: {self.observed_background.endpoint_hold_days} days. "
                    "Restrict the scoring period to supported interpolation, supply more training dry data, "
                    "or use physical GWI with seasonal_baseline='off'; no extrapolation beyond the endpoint "
                    "hold limit or filling of long interior gaps is performed."
                )
        super().prepare(data)

    def predict(self) -> pd.DataFrame:
        prediction = super().predict()
        if self.observed_background is not None:
            prediction["slow_background_cfs"] = self.sanitary.background_values
            prediction["background_supported"] = np.isfinite(self.sanitary.background_values)
            prediction["background_endpoint_held"] = self.observed_background.endpoint_hold_mask(
                pd.DatetimeIndex(prediction["datetime"]))
        return prediction

    def inequality_constraints(self) -> list[float]:
        g = lambda name: float(self.get_scalar_parameter(name).value)
        level = (g("sanitary_mean_cfs") if self.observed_background is None
                 else min(self.observed_background.flow_cfs))
        minimum = (level
                   + g("sanitary_hour_scale") * min(self.pattern.hourly)
                   + g("sanitary_weekday_scale") * min(self.pattern.weekday))
        return [*super().inequality_constraints(), -minimum]
