"""Tests for the native unit hydrograph models (Rectangle, Decay, GammaDelay, PeakTail)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sparsehydro.enums import ModelState
from sparsehydro.models.unithydrograph import (
    GammaUH, NashUH, TriangleUH, RectangleUH, DecayUH, GammaDelayUH, PeakTailUH,
)

DT_HOURS = 5.0 / 60.0
ALL_UH = [GammaUH, NashUH, TriangleUH, RectangleUH, DecayUH, GammaDelayUH, PeakTailUH]


def _rain_df(n: int = 240) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=n, freq="5min")
    rain = np.zeros(n)
    rain[10:16] = 0.1
    return pd.DataFrame({"datetime": idx, "rain": rain})


@pytest.mark.parametrize("cls", ALL_UH)
def test_full_lifecycle(cls):
    m = cls()
    assert m.state is ModelState.CREATED
    m.initialize()
    assert m.state is ModelState.INITIALIZED
    assert m.validate()
    m.prepare(_rain_df())
    result = m.predict()
    assert "Q_pred" in result.columns
    assert len(result) == len(_rain_df())
    m.finalize()
    assert m.is_finalized()


@pytest.mark.parametrize("cls", ALL_UH)
def test_kernel_unit_area(cls):
    m = cls()
    m.initialize()
    m.validate()
    kernel = m.get_kernel(DT_HOURS)
    assert np.all(kernel >= 0.0)
    assert abs(float(np.sum(kernel)) * DT_HOURS - 1.0) < 1e-3


@pytest.mark.parametrize("cls", [RectangleUH, DecayUH, GammaDelayUH, PeakTailUH])
def test_amplitude_scales_output(cls):
    df = _rain_df()
    m1 = cls()
    m1.initialize()
    a = m1.get_scalar_parameter("A")
    a.update(value=50.0)
    m1.validate()
    m1.prepare(df)
    q1 = m1.predict()["Q_pred"].sum()

    m2 = cls()
    m2.initialize()
    m2.get_scalar_parameter("A").update(value=100.0)
    m2.validate()
    m2.prepare(df)
    q2 = m2.predict()["Q_pred"].sum()

    assert q1 > 0
    assert abs(q2 / q1 - 2.0) < 0.05


def test_rectangle_is_flat():
    m = RectangleUH(A=1.0, tr=10.0)
    m.initialize()
    m.validate()
    k = m.get_kernel(DT_HOURS)
    nz = k[k > 0]
    assert np.allclose(nz, nz[0])  # constant pulse


def test_decay_is_monotonic():
    m = DecayUH(A=1.0, alpha=0.7)
    m.initialize()
    m.validate()
    k = m.get_kernel(DT_HOURS)
    assert np.all(np.diff(k) <= 1e-12)  # non-increasing


def test_gamma_delay_shifts_peak():
    base = GammaDelayUH(A=1.0, tt=2.0, tp=5.0, td=0.0)
    base.initialize()
    base.validate()
    delayed = GammaDelayUH(A=1.0, tt=2.0, tp=5.0, td=10.0)
    delayed.initialize()
    delayed.validate()
    k0 = base.get_kernel(DT_HOURS)
    kd = delayed.get_kernel(DT_HOURS)
    assert int(np.argmax(kd)) > int(np.argmax(k0))


def test_peak_tail_validate_ordering():
    m = PeakTailUH()
    m.initialize()
    m.get_scalar_parameter("peak_tp").update(value=60.0)
    m.get_scalar_parameter("peak_tt").update(value=50.0)  # tp >= tt invalid
    assert not m.validate()


def test_peak_tail_blend_weight():
    # The blended kernel must equal (1-w)*peak_kernel + w*tail_kernel elementwise,
    # since both component kernels are normalised before blending and share support.
    peak = PeakTailUH(w=0.0, td=0.0)
    peak.initialize()
    peak.validate()
    tail = PeakTailUH(w=1.0, td=0.0)
    tail.initialize()
    tail.validate()
    blend = PeakTailUH(w=0.5, td=0.0)
    blend.initialize()
    blend.validate()

    kp = peak.get_kernel(DT_HOURS)
    kt = tail.get_kernel(DT_HOURS)
    kb = blend.get_kernel(DT_HOURS)

    # All three share the same support length (independent of w).
    assert len(kp) == len(kt) == len(kb)
    # Each component kernel integrates to ~1 and the blend does too.
    assert np.sum(kp) * DT_HOURS == pytest.approx(1.0, abs=1e-4)
    assert np.sum(kt) * DT_HOURS == pytest.approx(1.0, abs=1e-4)
    assert np.sum(kb) * DT_HOURS == pytest.approx(1.0, abs=1e-4)
    # Blend identity.
    np.testing.assert_allclose(kb, 0.5 * kp + 0.5 * kt, rtol=1e-6, atol=1e-9)
    # The pure peak (triangle) and pure tail (gamma) shapes differ.
    assert not np.allclose(kp, kt)

