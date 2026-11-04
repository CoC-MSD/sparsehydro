import numpy as np
import pandas as pd
import pytest

from sparsehydro.calibration.wholeflow_workflow import frozen_events
from sparsehydro.models.wholeflow_compact import CompactWholeFlowModel, SanitaryTemplate


def synthetic(days=120, start="2024-01-01"):
    t = pd.date_range(start, periods=days * 24, freq="h")
    rain = np.zeros(len(t))
    rain[72::144] = 1.0
    flow = .5 * (1 + .25 * np.sin(t.hour.to_numpy() * np.pi / 12))
    flow *= np.where(t.dayofweek >= 5, .9, 1)
    flow += np.convolve(rain, 5 * np.exp(-np.arange(48) / 8), mode="full")[:len(t)]
    return pd.DataFrame(dict(datetime=t, rainfall_in=rain, temperature_c=10.0, flow_cfs=flow))


def model(dt=1.0):
    m = CompactWholeFlowModel(SanitaryTemplate((1.0,) * 24, (1.0,) * 7), 20.26, dt, .5, 2)
    m.initialize()
    return m


@pytest.mark.parametrize("dt", [1 / 12, .25, 1.0])
def test_components_and_complete_budget(dt):
    m = model(dt)
    n = 1000
    rain = np.zeros(n)
    rain[[1, 100, 950]] = [.01, 20, 2]
    df = pd.DataFrame(dict(datetime=pd.date_range("2024-01-01", periods=n,
                                                freq=pd.Timedelta(hours=dt)),
                           rainfall_in=rain, temperature_c=10.0))
    m.prepare(df)
    p = m.predict()
    np.testing.assert_allclose(p.total_cfs, p.sanitary_cfs + p.gwi_cfs + p.rdii_cfs)
    np.testing.assert_allclose(p.rdii_cfs, p.rtk1_cfs + p.rtk2_cfs + p.rtk3_cfs, atol=1e-12)
    assert (p.recharge_in >= 0).all() and (p.recharge_in <= rain).all()
    assert np.isfinite(p.drop(columns="datetime")).all().all()
    assert p.budget_error_in.abs().max() < 1e-8
    assert m.budget["final_routing_in"] > 0
    assert m.budget["initial_storage_in"] > 0
    assert "floor_offset_cfs" not in m.scalar_parameter_names
    assert not any("month" in n or "harmonic" in n for n in m.scalar_parameter_names)


def test_sanitary_only_and_recession():
    m = model()
    df = synthetic(30).drop(columns="flow_cfs").assign(rainfall_in=0)
    for name in ("s_initial_fraction", "g_initial_fraction"):
        m.get_scalar_parameter(name).value = 0
    m.prepare(df)
    p = m.predict()
    np.testing.assert_allclose(p.total_cfs, .5)
    m.get_scalar_parameter("g_initial_fraction").value = 1
    p = m.predict()
    assert (np.diff(p.gwi_cfs) < 0).all()
    assert (p.rdii_cfs == 0).all()


@pytest.mark.parametrize("column,value", [("rainfall_in", np.nan), ("rainfall_in", -1),
                                         ("temperature_c", np.nan)])
def test_missing_and_negative_forcing_rejected(column, value):
    df = synthetic(10)
    df.loc[3, column] = value
    with pytest.raises(ValueError):
        model().prepare(df)


def test_parameter_feasibility_and_irregular_grid():
    m = model()
    for name in ("R_1", "R_2", "R_3"):
        m.get_scalar_parameter(name).value = .5
    m.prepare(synthetic(10))
    with pytest.raises(ValueError):
        m.predict()
    with pytest.raises(ValueError):
        m.prepare(synthetic(10).drop(index=10))


















def test_rain_span_crossing_split_is_not_a_complete_event():
    t = pd.date_range("2024-01-01", periods=72, freq="h")
    rain = np.zeros(72)
    rain[[9, 12, 30, 50]] = .5
    flow = np.ones(72)
    flow[9:17] = 3
    flow[30:35] = 3
    flow[50:55] = 3
    data = pd.DataFrame(dict(datetime=t, rainfall_in=rain, flow_cfs=flow))
    train = np.arange(72) < 10
    events = frozen_events(data, train, ~train, np.ones(72))
    clipped = events.loc[events.start_idx == 12]
    assert len(clipped) == 1
    assert not clipped.iloc[0].eligible
    assert clipped.iloc[0].exclusion == "rain_span_crosses_split_boundary"
    assert events.loc[events.start_idx == 30, "eligible"].all()
    separated = frozen_events(data, train, ~train, np.ones(72),
                              detector_settings={"interevent_hours": 2.0})
    assert separated.loc[separated.start_idx == 12, "eligible"].tolist() == [True]


def test_frozen_event_settings_defaults_and_rainfall_threshold():
    rain = np.zeros(72)
    rain[[5, 25, 50]] = .5
    flow = np.ones(72)
    for start in (5, 25, 50):
        flow[start:start + 5] = 3
    data = pd.DataFrame(dict(datetime=pd.date_range("2024-01-01", periods=72, freq="h"),
                             rainfall_in=rain, flow_cfs=flow))
    train = np.ones(72, bool)
    defaults = dict(threshold_pct=5.0, interevent_hours=6.0, sliding_hours=12.0,
                    min_rain_in=.2, min_duration_hours=3.0)
    events = frozen_events(data, train, ~train, np.ones(72))
    assert not events.empty
    pd.testing.assert_frame_equal(
        events, frozen_events(data, train, ~train, np.ones(72), detector_settings=defaults))
    assert frozen_events(data, train, ~train, np.ones(72),
                         detector_settings={"min_rain_in": 1.0}).empty


@pytest.mark.parametrize("settings", [
    {"threshold_pct": -1.0}, {"threshold_pct": 101.0},
    {"interevent_hours": 0.0}, {"sliding_hours": 0.0},
    {"min_rain_in": -1.0}, {"min_duration_hours": -1.0},
    {"sliding_hours": np.nan}, {"min_rain_in": np.inf}, {"unknown": 1.0},
])
def test_frozen_event_settings_reject_invalid_values(settings):
    empty = np.array([], bool)
    with pytest.raises(ValueError):
        frozen_events(pd.DataFrame(), empty, empty, np.array([]), detector_settings=settings)




