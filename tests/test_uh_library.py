"""Conformance tests for every registered unit hydrograph.

The key acceptance test is timestep invariance: the same physical parameters
and rainfall give the same response at 5 minutes, 15 minutes and 1 hour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import sparsehydro.models  # noqa: F401  (registers built-ins)
from sparsehydro.models.rdii import RDIIModel
from sparsehydro.models.rdii.rtk_triangle import triangular_uh
from sparsehydro.models.unithydrograph import (
    SCSUH, ClarkUH, DecayUH, GammaUH, GeneralizedGammaUH, LognormalUH, MixtureUH,
    ModClarkUH, NashUH, PeakTailUH, RTKTriangle, SnyderUH, TriangleUH,
    UnitHydrographBase, WeibullUH,
)
from sparsehydro.models.unithydrograph.synthetic import PRF_DIMENSIONLESS_DIVISOR
from sparsehydro.registry import registry

DTS = (5 / 60, 0.25, 1.0)

UH_NAMES = sorted(
    name for name in registry.names()
    if issubclass(registry.get(name), UnitHydrographBase)
)


def make(name: str) -> UnitHydrographBase:
    m = registry.create(name)
    m.initialize()
    assert m.validate()
    return m


def first_moments(m: UnitHydrographBase, dt: float = 1 / 60) -> tuple[float, float]:
    k = m.get_kernel(dt)
    t = (np.arange(len(k)) + 0.5) * dt
    w = k * dt
    mean = float(np.sum(w * t))
    return mean, float(np.sum(w * (t - mean) ** 2))


def test_library_is_complete():
    expected = {
        "nash-uh", "gamma-uh", "gamma-delay-uh", "scs-uh", "snyder-uh", "clark-uh",
        "modclark-uh", "weibull-uh", "lognormal-uh", "gengamma-uh", "mixture-uh",
        "peak-tail-uh", "triangle-uh", "rectangle-uh", "decay-uh", "rtk-triangle",
    }
    assert expected <= set(UH_NAMES)


@pytest.mark.parametrize("name", UH_NAMES)
@pytest.mark.parametrize("dt", DTS)
def test_unit_area_and_non_negative(name, dt):
    k = make(name).get_kernel(dt)
    assert np.all(np.isfinite(k)) and np.all(k >= 0.0)
    assert abs(float(k.sum()) * dt - 1.0) < 1e-9


@pytest.mark.parametrize("name", [n for n in UH_NAMES if n != "rtk-triangle"])
def test_timestep_invariance(name):
    """Hourly rain disaggregated to 15 and 5 min gives the same response.

    Each ordinate is the instantaneous flow at the end of its step, so the fine
    run sampled at hour boundaries equals the hourly run, and volumes match.
    """
    rng = np.random.default_rng(7)
    hours = 160
    rain_h = np.where(rng.random(hours) < 0.25, rng.gamma(0.8, 0.2, hours), 0.0)
    rain_h[60:] = 0.0  # let the response drain inside the record
    results = {}
    for dt in DTS:
        m_per_h = int(round(1 / dt))
        rain = np.repeat(rain_h / m_per_h, m_per_h)
        m = make(name)
        m.get_scalar_parameter("A").update(value=10.0)
        idx = pd.date_range("2024-01-01", periods=len(rain), freq=pd.Timedelta(hours=dt))
        m.prepare(pd.DataFrame({"datetime": idx, "rain": rain}))
        q = m.predict()["Q_pred"].to_numpy()
        results[dt] = (q, m_per_h)
    q_h, _ = results[1.0]
    for dt in (5 / 60, 0.25):
        q, m_per_h = results[dt]
        np.testing.assert_allclose(q[m_per_h - 1::m_per_h], q_h, rtol=1e-9, atol=1e-12)
        assert float(q.sum() * dt) == pytest.approx(float(q_h.sum()), rel=1e-6)


def test_volume_equals_amplitude_times_rain():
    m = NashUH(A=12.5, n=2.0, K=0.5)
    m.initialize()
    rain = np.zeros(400)
    rain[5:12] = 0.1
    m.prepare(pd.DataFrame({"datetime": pd.date_range("2024", periods=400, freq="5min"), "rain": rain}))
    q = m.predict()["Q_pred"].to_numpy()
    assert float(q.sum() * (5 / 60)) == pytest.approx(12.5 * rain.sum(), rel=1e-8)


# ---------------------------------------------------------------------------
# Analytical properties and equivalences
# ---------------------------------------------------------------------------

def test_nash_moments():
    m = NashUH(n=3.0, K=0.8)
    m.initialize()
    mean, var = first_moments(m)
    assert mean == pytest.approx(3.0 * 0.8, rel=1e-3)
    assert var == pytest.approx(3.0 * 0.8 ** 2, rel=2e-3)


def test_gamma_equals_nash():
    g = GammaUH(alpha=2.5, theta=0.7)
    n = NashUH(n=2.5, K=0.7)
    g.initialize()
    n.initialize()
    np.testing.assert_allclose(g.get_kernel(0.25), n.get_kernel(0.25), rtol=1e-14)


def test_decay_equals_single_reservoir():
    d = DecayUH(K=0.6)
    n = NashUH(n=1.0, K=0.6)
    d.initialize()
    n.initialize()
    np.testing.assert_allclose(d.get_kernel(0.25), n.get_kernel(0.25), rtol=1e-10, atol=1e-14)


def test_generalized_gamma_special_cases():
    gg = GeneralizedGammaUH(a=2.5, c=1.0, scale=0.7)
    ga = GammaUH(alpha=2.5, theta=0.7)
    gw = GeneralizedGammaUH(a=1.0, c=1.7, scale=1.2)
    wb = WeibullUH(k=1.7, lam=1.2)
    for m in (gg, ga, gw, wb):
        m.initialize()
    np.testing.assert_allclose(gg.get_kernel(0.25), ga.get_kernel(0.25), rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(gw.get_kernel(0.25), wb.get_kernel(0.25), rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("model,dist", [
    (WeibullUH(k=1.7, lam=1.2), stats.weibull_min(1.7, scale=1.2)),
    (LognormalUH(mu=0.3, sigma=0.6), stats.lognorm(0.6, scale=np.exp(0.3))),
    (GeneralizedGammaUH(a=2.0, c=1.5, scale=0.8), stats.gengamma(2.0, 1.5, scale=0.8)),
])
def test_distributions_match_scipy(model, dist):
    model.initialize()
    dt = 0.25
    k = model.get_kernel(dt)
    edges = np.minimum(np.arange(len(k) + 1) * dt, model.support_hours())
    F = dist.cdf(edges)
    expected = np.diff(F) / F[-1] / dt
    np.testing.assert_allclose(k, expected, rtol=1e-8, atol=1e-12)


def test_scs_peak_time_and_rate():
    m = SCSUH(Tp=2.0)
    m.initialize()
    dt = 1 / 120
    k = m.get_kernel(dt)
    t_peak = (np.argmax(k) + 0.5) * dt
    assert t_peak == pytest.approx(2.0, abs=dt)
    assert k.max() * 2.0 == pytest.approx(484.0 / PRF_DIMENSIONLESS_DIVISOR, rel=2e-3)
    assert float(m.cdf(np.array([2.0]))[0]) == pytest.approx(0.375, abs=0.02)


def test_snyder_is_scs_with_prf_640_cp():
    s = SnyderUH(tL=1.5, Cp=0.7)
    c = SCSUH(Tp=1.5, PRF=640 * 0.7)
    s.initialize()
    c.initialize()
    np.testing.assert_allclose(s.get_kernel(0.25), c.get_kernel(0.25), rtol=1e-12)


def test_clark_tends_to_time_area_as_storage_vanishes():
    m = ClarkUH(Tc=2.0, Rs=1e-3)
    m.initialize()
    tau, cum = m.time_area
    t = np.linspace(0, 2.0, 41)
    np.testing.assert_allclose(m.cdf(t), np.interp(t / 2.0, tau, cum), atol=2e-3)


def test_clark_mean_is_time_area_mean_plus_storage():
    m = ClarkUH(Tc=3.0, Rs=1.5)
    m.initialize()
    mean, _ = first_moments(m)
    assert mean == pytest.approx(0.5 * 3.0 + 1.5, rel=2e-3)


def test_clark_custom_time_area_and_validation():
    m = ClarkUH(Tc=2.0, Rs=0.5, time_area=([0.0, 0.25, 1.0], [0.0, 0.5, 1.0]))
    m.initialize()
    assert abs(m.get_kernel(0.1).sum() * 0.1 - 1.0) < 1e-9
    with pytest.raises(ValueError):
        ClarkUH(time_area=([0.0, 1.0], [0.0, 0.9]))


def test_modclark_histogram_reduces_to_clark():
    travel = [0.0, 0.5, 0.5, 1.0]
    mc = ModClarkUH(Tc=2.0, Rs=0.7, travel_index=travel, n_bins=2)
    curve = ModClarkUH.time_area_from_cells(travel, n_bins=2)
    cl = ClarkUH(Tc=2.0, Rs=0.7, time_area=curve)
    mc.initialize()
    cl.initialize()
    np.testing.assert_allclose(mc.get_kernel(0.1), cl.get_kernel(0.1))
    np.testing.assert_allclose(curve[1], [0.0, 0.25, 1.0])
    with pytest.raises(ValueError):
        ModClarkUH.time_area_from_cells([-1.0, 1.0])


def test_mixture_with_one_dominant_component_equals_it():
    fast = GammaUH(alpha=3.0, theta=0.3)
    mix = MixtureUH([GammaUH(alpha=3.0, theta=0.3), GammaUH(alpha=2.0, theta=4.0)], weights=[1.0, 0.0])
    fast.initialize()
    mix.initialize()
    k_fast = fast.get_kernel(0.25)
    k_mix = mix.get_kernel(0.25)
    np.testing.assert_allclose(k_mix[:len(k_fast)], k_fast, rtol=1e-6, atol=1e-9)


def test_mixture_parameters_weights_and_constraints():
    mix = MixtureUH([NashUH(), GammaUH(), TriangleUH()])
    mix.initialize()
    names = mix.scalar_parameter_names
    assert {"A", "w_2", "w_3", "c1_n", "c1_K", "c2_alpha", "c3_Tp", "c3_Tb"} <= set(names)
    assert "c1_A" not in names
    constraints = [c.name for c in mix._constraint_registry]
    assert constraints == ["weights_sum_leq_1", "c3_Tp_lt_Tb"]
    mix.apply_flat_parameter("w_2", 0.8)
    mix.apply_flat_parameter("w_3", 0.5)
    assert mix.inequality_constraints()[0] == pytest.approx(0.3)
    assert not mix.validate()


def test_peak_tail_blend_and_components():
    m = PeakTailUH(w=0.3, td=0.5)
    m.initialize()
    parts = m.component_kernels(0.25)
    np.testing.assert_allclose(parts["peak"] + parts["tail"], parts["combined"], rtol=1e-12)
    np.testing.assert_allclose(parts["combined"], m.get_kernel(0.25), rtol=1e-12, atol=1e-15)
    assert parts["tail"].sum() * 0.25 == pytest.approx(0.3, rel=1e-6)
    assert np.all(m.get_kernel(0.25)[:2] == 0.0)  # delayed by 0.5 h
    m.apply_flat_parameter("peak_Tp", 5.0)
    m.apply_flat_parameter("peak_Tb", 4.0)
    assert m.inequality_constraints() == [pytest.approx(1.0)]
    assert not m.validate()


def test_rtk_generic_cdf_matches_triangular_uh():
    tri = RTKTriangle(R=0.1, T=1.5, K=2.0)
    tri.initialize()
    for dt in DTS:
        edges = np.arange(len(triangular_uh(tri, dt)) + 1) * dt
        generic = np.diff(tri.cdf(edges)) / dt
        np.testing.assert_allclose(generic, triangular_uh(tri, dt), rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize("cls,old", [
    (GammaUH, {"tt": 2.0}), (NashUH, {"k": 5.0}), (TriangleUH, {"tp": 3.0}),
    (PeakTailUH, {"peak_tp": 2.0}),
])
def test_step_based_keywords_fail_loudly(cls, old):
    with pytest.raises(TypeError, match="hours"):
        cls(**old)


def test_delay_shifts_response_exactly():
    base = NashUH(n=2.0, K=0.5)
    late = NashUH(n=2.0, K=0.5, delay=1.0)
    base.initialize()
    late.initialize()
    k0, k1 = base.get_kernel(0.25), late.get_kernel(0.25)
    np.testing.assert_allclose(k1[4:4 + len(k0)], k0, rtol=1e-10, atol=1e-14)
    assert np.all(k1[:4] == 0.0)


def test_truncation_is_bounded_for_heavy_tails():
    m = LognormalUH(mu=5.5, sigma=2.0)
    m.initialize()
    k = m.get_kernel(1.0)
    assert len(k) <= int(m.max_duration_hours) + 2
    assert abs(k.sum() - 1.0) < 1e-9


def test_rdii_accepts_any_unit_hydrograph():
    uhs = [RTKTriangle(R=0.02, T=1.0, K=2.0), NashUH(A=50.0), ClarkUH(),
           MixtureUH([GammaUH(), TriangleUH()])]
    m = RDIIModel(uh_components=uhs)
    m.initialize()
    assert m.validate()
    names = m.scalar_parameter_names
    assert {"uh2_n", "uh3_Tc", "uh4_c2_Tp"} <= set(names)
    assert "uh2_A" not in names
    assert "uh4_c2_Tp_lt_Tb" in [c.name for c in m._constraint_registry]
    for i in range(1, 5):
        assert 0.0 <= m.get_scalar_parameter(f"R_{i}").value <= 1.0
    n = 600
    rain = np.zeros(n)
    rain[10:30] = 0.1
    df = pd.DataFrame({
        "datetime": pd.date_range("2024-06-01", periods=n, freq="5min"),
        "rainfall_in": rain, "temperature_c": 20.0,
    })
    m.prepare(df)
    out = m.predict()
    assert np.all(np.isfinite(out["rdii_cfs"])) and out["rdii_cfs"].max() > 0
