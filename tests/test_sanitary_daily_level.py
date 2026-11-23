"""Daily dry-weather level: hourly and daily multipliers and a level from the night-to-day rise.

The test data is 15-minute flow made from known hourly and daily multipliers
times a sewage level that steps up half way, plus steady groundwater, plus
extra flow on a few rainy days.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sparsehydro.calibration.splits import dry_weather_mask
from sparsehydro.models.abstraction import ExternalExcess
from sparsehydro.models.baseflow import (
    ObservedBackground,
    ObservedBackgroundModel,
    SanitaryTemplate,
    SanitaryTemplateModel,
    estimate_sanitary_level,
    estimate_sanitary_template,
)
from sparsehydro.models.composite import CompositeModel
from sparsehydro.models.rdii import RDIIModel

HOURLY = 1 + 0.6 * np.sin(2 * np.pi * (np.arange(24) - 9) / 24)    # lowest at 03:00, mean 1
WEEKDAY = np.array([0.95, 0.95, 0.95, 0.95, 1.0, 1.1, 1.1])         # Monday first, mean 1
TRUE = SanitaryTemplate(tuple(HOURLY), tuple(WEEKDAY))
N_DAYS, STEP_DAY = 120, 60
LOW, HIGH = 0.10, 0.20       # sewage level before and after the step [cfs]
RAIN_DAYS = (20, 45, 80, 100)


def make_data(groundwater_cfs: float = 0.05, high: float = HIGH) -> pd.DataFrame:
    times = pd.date_range("2024-01-01", periods=N_DAYS * 96, freq="15min")     # starts on a Monday
    day_number = (times - times[0]).days.to_numpy()
    level = np.where(day_number < STEP_DAY, LOW, high)
    flow = level * TRUE.predict(times) + groundwater_cfs
    rain = np.zeros(len(times))
    for rain_day in RAIN_DAYS:
        rain[day_number == rain_day] = 0.01                          # 0.96 in over the day
        # Storm water on the rainy day and the two days after
        for days_after, extra_cfs in ((0, 0.10), (1, 0.05), (2, 0.02)):
            flow[day_number == rain_day + days_after] += extra_cfs
    return pd.DataFrame({"datetime": times, "rainfall_in": rain, "flow_cfs": flow})


def dry_rows(data: pd.DataFrame) -> np.ndarray:
    """Rows with no rain in the 72 hours before and a known flow."""
    return dry_weather_mask(data, 0.25, 72) & data["flow_cfs"].notna().to_numpy()


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

def test_template_recovers_the_true_factors():
    data = make_data(groundwater_cfs=0.0)
    template = estimate_sanitary_template(data, dry_rows(data))
    np.testing.assert_allclose(template.hourly, HOURLY, rtol=0.02)
    np.testing.assert_allclose(template.weekday, WEEKDAY, rtol=0.03)
    assert min(template.hourly) >= 0 and min(template.weekday) >= 0


def test_level_is_sewage_without_the_groundwater():
    data = make_data(groundwater_cfs=0.05)
    level = estimate_sanitary_level(data, dry_rows(data), TRUE)
    assert isinstance(level, ObservedBackground)
    assert (level.window_days, level.minimum_days, level.max_gap_days) == (31, 5, 45)
    values = pd.Series(level.flow_cfs, index=pd.DatetimeIndex(level.datetimes))
    start = data["datetime"][0]
    before = values[start + pd.Timedelta(days=20):start + pd.Timedelta(days=STEP_DAY - 16)]
    after = values[start + pd.Timedelta(days=STEP_DAY + 16):start + pd.Timedelta(days=110)]
    np.testing.assert_allclose(before, LOW, rtol=0.01)       # not LOW + 0.05
    np.testing.assert_allclose(after, HIGH, rtol=0.01)       # not HIGH + 0.05


def test_only_whole_dry_days_are_used():
    data = make_data()
    eligible = dry_rows(data)
    day_number = (data["datetime"] - data["datetime"][0]).dt.days.to_numpy()
    # Day 30 is dry for only 18 of its 24 hours
    eligible[(day_number == 30) & (data["datetime"].dt.hour < 6).to_numpy()] = False
    level = estimate_sanitary_level(data, eligible, TRUE)
    days_used = set(pd.DatetimeIndex(level.datetimes).normalize())
    start = data["datetime"][0]
    assert start + pd.Timedelta(days=30) not in days_used
    assert start + pd.Timedelta(days=29) not in days_used      # its night after is on day 30
    assert start + pd.Timedelta(days=33) in days_used


def test_estimated_template_and_level_describe_steady_dry_flow():
    """A template from flow with groundwater is flatter, so the level takes in the groundwater.

    While the groundwater share stays steady, the pair still gives the whole dry flow.
    """
    data = make_data(groundwater_cfs=0.05, high=LOW)
    eligible = dry_rows(data)
    template = estimate_sanitary_template(data, eligible)
    assert np.ptp(template.hourly) < 0.8 * np.ptp(HOURLY)
    level = estimate_sanitary_level(data, eligible, template)
    times = pd.DatetimeIndex(data["datetime"])
    day_number = (times - times[0]).days.to_numpy()
    inside = eligible & (day_number > 20) & (day_number < 110)
    np.testing.assert_allclose(level.evaluate(times)[inside], LOW + 0.05, rtol=0.03)
    fitted = level.evaluate(times) * template.predict(times)
    np.testing.assert_allclose(fitted[inside], data["flow_cfs"].to_numpy()[inside], rtol=0.05)


def test_estimators_reject_bad_input():
    data = make_data()
    eligible = dry_rows(data)
    flat = SanitaryTemplate((1.0,) * 24, (1.0,) * 7)
    with pytest.raises(ValueError, match="day-night difference"):
        estimate_sanitary_level(data, eligible, flat)
    few = eligible & (data["datetime"] < "2024-01-07").to_numpy()
    with pytest.raises(ValueError, match="Too few eligible days"):
        estimate_sanitary_level(data, few, TRUE)
    with pytest.raises(ValueError, match="every hour and day of the week"):
        estimate_sanitary_template(data, few)


# ---------------------------------------------------------------------------
# SanitaryTemplateModel
# ---------------------------------------------------------------------------

def test_default_template_model_is_unchanged():
    data = make_data().iloc[: 96 * 7]
    model = SanitaryTemplateModel(TRUE, 0.8, 2.0)
    model.initialize()
    assert model.scalar_parameter_names == ["sanitary_level_cfs"]
    assert model.input_field_names == []
    model.prepare(data)
    out = model.predict_arrays()["sanitary_cfs"]
    np.testing.assert_allclose(out, 0.8 * TRUE.predict(data["datetime"]))
    with pytest.raises(KeyError):
        model.set_input("level_cfs", np.ones(len(data)))


def test_level_from_input_inside_a_composite():
    data = make_data()
    eligible = dry_rows(data)
    level = estimate_sanitary_level(data, eligible, TRUE)
    in_window = (data["datetime"] >= "2024-02-01") & (data["datetime"] < "2024-04-01")
    window = data[in_window].reset_index(drop=True)
    model = CompositeModel(
        {"background": ObservedBackgroundModel(level),
         "sewage": SanitaryTemplateModel(TRUE, level_from_input=True),
         "rdii": RDIIModel(ia_model=ExternalExcess())},
        bindings={"sewage.level_cfs": "background.background_cfs",
                  "rdii.p_excess_in": "data.rainfall_in"},
        outputs={"sanitary_cfs": "sewage.sanitary_cfs",
                 "total_cfs": ["sewage.sanitary_cfs", "rdii.rdii_cfs"]},
    )
    model.initialize()
    scale = model.get_scalar_parameter("sewage.sanitary_level_scale")
    assert (scale.value, scale.lower_bound, scale.upper_bound) == (1.0, 0.5, 1.5)
    assert "sewage.sanitary_level_cfs" not in model.scalar_parameter_names
    model.prepare(window)
    model.apply_flat_parameter("sewage.sanitary_level_scale", 1.2)
    arrays = model.predict_arrays()
    t = pd.DatetimeIndex(window["datetime"])
    np.testing.assert_allclose(arrays["sanitary_cfs"], 1.2 * level.evaluate(t) * TRUE.predict(t))
    assert np.all(arrays["total_cfs"] >= arrays["sanitary_cfs"])
    frame = model.predict()
    for name in ("sanitary_cfs", "total_cfs"):
        np.testing.assert_array_equal(frame[name].to_numpy(), arrays[name])
