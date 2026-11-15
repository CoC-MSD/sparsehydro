"""Fitted sanitary shapes and the observed background (plain data classes)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SanitaryTemplate:
    """24 hourly and 7 weekday factors; 29 independent normalized shape values."""

    hourly: tuple[float, ...]
    weekday: tuple[float, ...]

    def __post_init__(self) -> None:
        for values, size in ((self.hourly, 24), (self.weekday, 7)):
            a = np.asarray(values)
            if a.shape != (size,) or not np.all(np.isfinite(a)) or np.any(a < 0):
                raise ValueError("Template factors must be finite, nonnegative and complete")
            if not np.isclose(a.mean(), 1.0):
                raise ValueError("Each template dimension must have mean one")

    def predict(self, datetimes: pd.Series | pd.DatetimeIndex) -> np.ndarray:
        t = pd.DatetimeIndex(datetimes)
        return np.asarray(self.hourly)[t.hour] * np.asarray(self.weekday)[t.dayofweek]


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
