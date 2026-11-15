"""Tests for AbstractionUHModel (abstraction → UH → optional seasonality)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sparsehydro.enums import ModelState
from sparsehydro.models import AbstractionUHModel, ConstantDrainTank, SeasonalityModel
from sparsehydro.models.rdii import IAModel
from sparsehydro.models.unithydrograph import GammaUH, PeakTailUH
from sparsehydro.registry import registry


def _df(n: int = 400, temp: float | None = None) -> pd.DataFrame:
    idx = pd.date_range("2023-03-01", periods=n, freq="5min")
    rain = np.zeros(n)
    rain[30:45] = 0.06
    rain[200:210] = 0.05
    d = {"datetime": idx, "rain": rain}
    if temp is not None:
        d["temperature_c"] = np.full(n, temp)
    return pd.DataFrame(d)


def test_full_lifecycle_default():
    m = AbstractionUHModel(abstraction=IAModel(), uh=PeakTailUH())
    assert m.state is ModelState.CREATED
    m.initialize()
    assert m.validate()
    m.prepare(_df(temp=15.0))
    out = m.predict()
    assert list(out.columns) == ["datetime", "Q_pred"]
    assert len(out) == 400
    m.finalize()
    assert m.is_finalized()


def test_param_prefixing():
    m = AbstractionUHModel(abstraction=IAModel(), uh=GammaUH(A=1.0))
    m.initialize()
    names = m.scalar_parameter_names
    assert any(n.startswith("ia_") for n in names)
    assert "uh_A" in names and "uh_tt" in names and "uh_tp" in names


def test_registered():
    assert "abstraction-uh" in registry.names()


def test_temperature_threaded():
    # Below-freeze vs warm should change IA recovery → different predicted flow.
    cold = AbstractionUHModel(abstraction=IAModel(), uh=GammaUH(A=1.0))
    cold.initialize()
    cold.validate()
    cold.prepare(_df(temp=-10.0))
    q_cold = float(cold.predict()["Q_pred"].sum())

    warm = AbstractionUHModel(abstraction=IAModel(), uh=GammaUH(A=1.0))
    warm.initialize()
    warm.validate()
    warm.prepare(_df(temp=25.0))
    q_warm = float(warm.predict()["Q_pred"].sum())

    assert np.isfinite(q_cold) and np.isfinite(q_warm)
    assert q_cold != q_warm


def test_tank_abstraction_composition():
    m = AbstractionUHModel(abstraction=ConstantDrainTank(), uh=GammaUH(A=1.0))
    m.initialize()
    assert m.validate()
    m.prepare(_df())
    out = m.predict()
    assert len(out) == 400
    assert any(n.startswith("tank_") for n in m.scalar_parameter_names)


def test_seasonality_composition():
    seas = SeasonalityModel(include_hour=False, include_dow=False, include_month=True, output_name="pf")
    m = AbstractionUHModel(abstraction=IAModel(), uh=GammaUH(A=1.0), seasonality=seas)
    m.initialize()
    assert m.validate()
    # seasonality parameters forwarded with seas_ prefix (+ vector + constraints)
    assert any(n.startswith("seas_") for n in m.scalar_parameter_names)
    assert "seas_pf_month" in m.vector_parameter_names
    assert len(m.inequality_constraints()) >= 2
    # seas_baseline is fixed (not calibrated)
    assert not m.get_scalar_parameter("seas_baseline").calibrate
    m.prepare(_df(temp=12.0))
    out = m.predict()
    assert len(out) == 400
