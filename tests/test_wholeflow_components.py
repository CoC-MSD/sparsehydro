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


# ---------------------------------------------------------------------------
# Soil/groundwater store
# ---------------------------------------------------------------------------

from sparsehydro.configuration import build_model  # noqa: E402
from sparsehydro.models.abstraction import SoilGroundwaterStore  # noqa: E402
from sparsehydro.models.baseflow import (  # noqa: E402
    CenteredWeeklyPattern, CenteredWeeklyPatternModel, ObservedBackgroundModel,
    SanitaryTemplate, SanitaryTemplateModel, estimate_observed_background,
)
from sparsehydro.models.wholeflow_pattern import PatternWholeFlowModel  # noqa: E402

PATTERN = CenteredWeeklyPattern(tuple(0.05 * np.sin(np.arange(24) * np.pi / 12)), (0.0,) * 7)


def warm_forcing(n: int = 480) -> pd.DataFrame:
    data = forcing(n)
    return data.assign(temperature_c=8.0 + 4.0 * np.sin(np.arange(n) / 24.0))


class TestSoilGroundwaterStore:
    def test_mass_balance_and_outputs(self):
        store = SoilGroundwaterStore(50.0)
        store.initialize()
        store.prepare(warm_forcing())
        out = store.predict_arrays()
        assert np.abs(out["budget_error_in"]).max() < 1e-12
        np.testing.assert_allclose(out["gwi_release_in"] + out["gwi_spill_in"], out["gwi_in"])
        np.testing.assert_allclose(out["gwi_cfs"], out["gwi_in"] * 50.0 * 3630.0 / 3600.0)
        assert store.inequality_constraint_names == ["g_max_lt_s_max", "k_G_lt_k0"]

    def test_without_groundwater_there_is_no_gwi(self):
        store = SoilGroundwaterStore(50.0, groundwater=False)
        store.initialize()
        assert store.get_scalar_parameter("k_div").value == 0.0
        assert not store.get_scalar_parameter("g_max").calibrate
        store.prepare(warm_forcing())
        assert store.predict_arrays()["gwi_in"].sum() == 0.0
        assert store.inequality_constraints() == []

    def test_validate_rejects_inconsistent_stores(self):
        store = SoilGroundwaterStore()
        store.initialize()
        assert store.validate()
        store.apply_flat_parameter("g_max", 0.3)
        store.apply_flat_parameter("s_max", 0.2)
        assert not store.validate()
        assert store.inequality_constraints()[0] == pytest.approx(0.1)


class TestBaselines:
    def test_template_scales_a_fixed_shape(self):
        hourly = np.r_[np.full(12, 0.5), np.full(12, 1.5)]
        model = SanitaryTemplateModel(SanitaryTemplate(tuple(hourly), (1.0,) * 7), 0.8, 2.0)
        model.initialize()
        model.prepare(forcing(48))
        out = model.predict_arrays()["sanitary_cfs"]
        np.testing.assert_allclose(out[:12], 0.4)
        np.testing.assert_allclose(out[12:24], 1.2)

    def test_pattern_needs_a_fit_and_enforces_nonnegativity(self):
        model = CenteredWeeklyPatternModel(mean_cfs=0.02, mean_upper_cfs=1.0)
        model.initialize()
        with pytest.raises(ValueError, match="fitted pattern"):
            model.prepare(forcing(48))
        model.prepare(forcing(48), pattern=PATTERN)
        assert model.inequality_constraints()[0] > 0      # 0.02 - 0.05 < 0
        model.apply_flat_parameter("sanitary_mean_cfs", 0.5)
        assert model.inequality_constraints()[0] < 0

    def test_background_reports_missing_support(self):
        data = warm_forcing(24 * 20).assign(flow_cfs=0.7)
        bg = estimate_observed_background(data, np.arange(len(data)) < 24 * 10, PATTERN,
                                          window_days=3, minimum_days=1, max_gap_days=5)
        model = ObservedBackgroundModel(bg)
        model.initialize()
        with pytest.raises(ValueError, match="lacks support"):
            model.prepare(data)
        model.prepare(data.assign(background_required=np.arange(len(data)) < 24 * 9))
        assert model.predict_arrays()["background_supported"][: 24 * 9].all()


def test_whole_flow_model_as_configuration_matches_the_preset():
    data = warm_forcing()
    config = {
        "model": {
            "type": "composite",
            "components": {
                "snow": {"type": "degree-day-snow", "config": None},
                "store": {"type": "soil-groundwater-store", "area_acres": 20.0,
                          "inputs": {"liquid_input_in": "snow.liquid_input_in"}},
                "rdii": {"type": "rdii", "excess": "store.p_excess_in",
                         "routing": [{"type": "rtk-triangle", "R": 0.03, "T": 1.0, "K": 2.0},
                                     {"type": "rtk-triangle", "R": 0.02, "T": 8.0, "K": 2.5},
                                     {"type": "rtk-triangle", "R": 0.01, "T": 48.0, "K": 3.0}]},
                "baseline": {"type": "centered-weekly-pattern", "mean_cfs": 0.6, "mean_upper_cfs": 2.0},
            },
            "prefixes": {"snow": "", "store": "", "rdii": "", "baseline": ""},
            "shared": {"area_acres": ["store.area_acres", "rdii.area_acres"]},
            "outputs": {"total_cfs": ["baseline.sanitary_cfs", "store.gwi_cfs", "rdii.rdii_cfs"]},
        },
    }
    from_config = build_model(config)
    from_config.prepare(data, baseline={"pattern": PATTERN})
    preset = PatternWholeFlowModel(PATTERN, 20.0, 1.0, 0.6, 2.0)
    preset.initialize()
    preset.prepare(data)
    np.testing.assert_allclose(from_config.predict_arrays()["total_cfs"],
                               preset.predict_arrays()["total_cfs"], rtol=1e-12, atol=1e-14)
