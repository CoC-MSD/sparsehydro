"""Tests for the storage-based rainfall-abstraction (tank) models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sparsehydro.enums import ModelState
from sparsehydro.models.abstraction import (
    ConstantDrainTank, LinearDrainTank, SqrtDrainTank, TankAbstractionModel,
)
from sparsehydro.registry import registry

ALL_TANKS = [ConstantDrainTank, LinearDrainTank, SqrtDrainTank]


def _rain_df(n: int = 300) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=n, freq="5min")
    rain = np.zeros(n)
    rain[20:40] = 0.05
    rain[150:170] = 0.05
    return pd.DataFrame({"datetime": idx, "rainfall_in": rain})


@pytest.mark.parametrize("cls", ALL_TANKS)
def test_full_lifecycle(cls):
    m = cls()
    assert m.state is ModelState.CREATED
    m.initialize()
    assert m.validate()
    m.prepare(_rain_df())
    out = m.predict()
    assert "p_excess_in" in out.columns
    assert len(out) == len(_rain_df())
    m.finalize()
    assert m.is_finalized()


@pytest.mark.parametrize("cls", ALL_TANKS)
def test_effective_nonnegative(cls):
    m = cls()
    m.initialize()
    m.validate()
    m.prepare(_rain_df())
    out = m.predict()
    assert np.all(out["p_excess_in"].to_numpy() >= 0.0)


@pytest.mark.parametrize("cls", ALL_TANKS)
def test_gain_within_bounds(cls):
    m = cls(ae_min=5.0, ae_max=10.0)
    m.initialize()
    m.validate()
    df = _rain_df()
    m.prepare(df)
    out = m.predict()
    r = df["rainfall_in"].to_numpy()
    ie = out["p_excess_in"].to_numpy()
    ratio = ie[r > 0] / r[r > 0]
    assert ratio.min() >= 5.0 - 1e-6
    assert ratio.max() <= 10.0 + 1e-6


def test_validate_rejects_bad_ae_order():
    m = ConstantDrainTank(ae_min=10.0, ae_max=5.0)
    m.initialize()
    assert not m.validate()


@pytest.mark.parametrize("name", ["tank-constant", "tank-linear", "tank-sqrt"])
def test_registered(name):
    assert name in registry.names()
    m = registry.create(name)
    assert isinstance(m, TankAbstractionModel)


def test_state_dependent_gain_drops_as_tank_fills():
    # Small tank so sustained rain fills it: gain should decline over a long burst.
    m = ConstantDrainTank(V_tank=0.5, ae_min=1.0, ae_max=10.0, drain=0.001)
    m.initialize()
    m.validate()
    n = 200
    idx = pd.date_range("2023-01-01", periods=n, freq="5min")
    rain = np.full(n, 0.05)
    m.prepare(pd.DataFrame({"datetime": idx, "rainfall_in": rain}))
    ie = m.predict()["p_excess_in"].to_numpy()
    ratio = ie / rain
    assert ratio[0] > ratio[-1]  # drier start → higher gain than saturated end


@pytest.mark.parametrize("cls", ALL_TANKS)
def test_compiled_kernel_matches_python_reference(cls):
    rng = np.random.default_rng(3)
    df = _rain_df(2000)
    df["rainfall_in"] = np.where(rng.random(2000) < 0.1, rng.gamma(0.5, 0.05, 2000), 0.0)
    m = cls(V_tank=0.8, ae_min=2.0, ae_max=9.0, drain=0.05)
    m.initialize()
    assert m.validate()
    m.prepare(df)
    fast = m.predict_arrays()["p_excess_in"]
    p = {k: m.get_scalar_parameter(k).value for k in m.scalar_parameter_names}
    ref = m._advance_python(df["rainfall_in"].to_numpy(dtype=float), p["tank_V"], p["tank_ae_min"],
                            p["tank_ae_max"], p[m._drain_param_name], m._n_substeps)
    np.testing.assert_array_equal(fast, ref)


def test_custom_drainage_law_runs_in_python():
    class HalfDrainTank(TankAbstractionModel):
        model_name = "test-half-drain-tank"
        _drain_param_name = "tank_k"

        def _drainage_full(self, V, drain):
            return 0.5 * V

    m = HalfDrainTank()
    m.initialize()
    assert m.validate()
    m.prepare(_rain_df())
    assert m.predict_arrays()["p_excess_in"].max() > 0


@pytest.mark.parametrize("cls", ALL_TANKS)
def test_gain_ordering_is_a_constraint_optimisers_see(cls):
    m = cls(ae_min=5.0, ae_max=10.0)
    m.initialize()
    assert m.inequality_constraint_names == ["ae_min_le_ae_max"]
    assert m.inequality_constraints() == [-5.0]
    m.get_scalar_parameter("tank_ae_min").value = 12.0
    assert m.inequality_constraints() == [2.0]
