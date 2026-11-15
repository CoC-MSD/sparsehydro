"""Whole-flow building blocks as standalone registered components."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sparsehydro.models.forcing import DegreeDaySnow, SnowConfig, snow_partition
from sparsehydro.registry import registry


def forcing(n: int = 240) -> pd.DataFrame:
    t = pd.date_range("2024-01-01", periods=n, freq="h")
    rain = np.zeros(n)
    rain[10:20] = 0.1
    rain[120:126] = 0.2
    temp = np.where(np.arange(n) < 100, -4.0, 6.0)
    return pd.DataFrame({"datetime": t, "rainfall_in": rain, "temperature_c": temp})


class TestDegreeDaySnow:
    def test_matches_snow_partition_and_closes_water_balance(self):
        data = forcing()
        config = SnowConfig(threshold_c=0.5, melt_factor_in_per_c_day=0.2, initial_swe_in=0.1)
        snow = DegreeDaySnow(config)
        snow.initialize()
        snow.prepare(data)
        out = snow.predict_arrays()
        expected = snow_partition(data.rainfall_in.to_numpy(), data.temperature_c.to_numpy(), 1.0, config)
        for col in expected:
            np.testing.assert_array_equal(out[col], expected[col].to_numpy())
        storage = config.initial_swe_in + np.cumsum(data.rainfall_in - out["liquid_input_in"])
        np.testing.assert_allclose(storage, out["SWE_in"], atol=1e-12)
        assert snow.scalar_parameter_names == []

    def test_calibrated_coefficients_drive_the_partition(self):
        snow = registry.create("degree-day-snow", SnowConfig(), calibrate=True)
        snow.initialize()
        assert snow.scalar_parameter_names == ["snow_T", "snow_ddf"]
        snow.prepare(forcing())
        before = snow.predict_arrays()["liquid_input_in"].copy()
        snow.apply_flat_parameter("snow_ddf", 0.01)
        assert snow.effective_config.melt_factor_in_per_c_day == 0.01
        assert not np.array_equal(snow.predict_arrays()["liquid_input_in"], before)

    def test_disabled_snow_passes_precipitation_through(self):
        data = forcing()
        snow = DegreeDaySnow(None)
        snow.initialize()
        snow.prepare(data)
        np.testing.assert_array_equal(snow.predict_arrays()["liquid_input_in"], data.rainfall_in)
        with pytest.raises(ValueError, match="requires an enabled"):
            DegreeDaySnow(None, calibrate=True)

    def test_initial_values_must_lie_within_bounds(self):
        with pytest.raises(ValueError, match="within"):
            DegreeDaySnow(SnowConfig(threshold_c=9.0), calibrate=True).initialize()
