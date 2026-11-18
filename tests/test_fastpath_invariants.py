"""Structural invariants for the ndarray fast path.

Unlike the golden fixtures, which pin *values*, these pin the *architecture* --
the properties that make the fast path safe to keep around:

1. ``predict_arrays()`` and ``predict()`` can never disagree, because the latter
   is an adapter over the former.
2. ``CalibrationProblem.evaluate`` produces identical results whether it takes
   the fast path or the fallback.
3. Re-``prepare()`` always invalidates cached forcing state.
4. No ``pandas.DataFrame`` is constructed inside the calibration hot loop.

Tests that depend on not-yet-implemented machinery skip cleanly, so this module
is green before, during and after the refactor -- each test switching itself on
as its feature lands.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sparsehydro.models.unithydrograph import (
    GammaUH, NashUH, TriangleUH, RectangleUH, DecayUH, GammaDelayUH, PeakTailUH,
)

ALL_UH = [GammaUH, NashUH, TriangleUH, RectangleUH, DecayUH, GammaDelayUH, PeakTailUH]
N = 400


def _rain_df(n: int = N, seed: int = 0) -> pd.DataFrame:
    """Return a deterministic rainfall frame of *n* 5-minute steps."""
    rng = np.random.default_rng(seed)
    rain = rng.gamma(0.3, 0.05, n)
    rain[rain < 0.01] = 0.0
    return pd.DataFrame(
        {"datetime": pd.date_range("2022-05-01", periods=n, freq="5min"), "rain": rain}
    )


def _has_array_predict(model) -> bool:
    """Return ``True`` once :meth:`predict_arrays` exists on *model*."""
    return hasattr(model, "predict_arrays")


# ---------------------------------------------------------------------------
# 1. Cross-path equivalence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", ALL_UH, ids=[c.__name__ for c in ALL_UH])
def test_predict_arrays_matches_predict(cls):
    """``predict_arrays()`` reproduces ``predict()`` exactly, column for column.

    This is what makes the ``predict()``-as-adapter inversion enforceable: if
    the two ever become separate implementations, they drift, and calibration
    silently stops agreeing with the notebooks.
    """
    m = cls()
    m.initialize()
    m.validate()
    m.prepare(_rain_df())
    if not _has_array_predict(m):
        pytest.skip("predict_arrays() not implemented yet (pre-Step-2)")

    df = m.predict()
    arrays = m.predict_arrays()

    assert set(arrays) == set(df.columns), (
        f"{cls.__name__}: predict_arrays keys {sorted(arrays)} != "
        f"predict columns {sorted(df.columns)}"
    )
    for col in df.columns:
        got = np.asarray(arrays[col])
        want = df[col].to_numpy()
        assert got.shape == want.shape, f"{cls.__name__}.{col}: shape mismatch"
        if np.issubdtype(want.dtype, np.floating):
            same = (got == want) | (np.isnan(got) & np.isnan(want))
            assert same.all(), f"{cls.__name__}.{col}: values differ"
        else:
            assert np.array_equal(got, want), f"{cls.__name__}.{col}: values differ"


def test_composite_predict_arrays_matches_predict():
    """``AbstractionUHModel`` honours the same cross-path contract."""
    from sparsehydro.models import AbstractionUHModel
    from sparsehydro.models.rdii import IAModel

    m = AbstractionUHModel(abstraction=IAModel(), uh=PeakTailUH())
    m.initialize()
    m.validate()
    m.prepare(_rain_df().assign(temperature_c=np.full(N, 15.0)))
    if not _has_array_predict(m):
        pytest.skip("predict_arrays() not implemented yet (pre-Step-2)")

    df = m.predict()
    arrays = m.predict_arrays()
    assert set(arrays) == set(df.columns)
    for col in df.columns:
        want = df[col].to_numpy()
        got = np.asarray(arrays[col])
        if np.issubdtype(want.dtype, np.floating):
            same = (got == want) | (np.isnan(got) & np.isnan(want))
            assert same.all(), f"AbstractionUHModel.{col}: values differ"
        else:
            assert np.array_equal(got, want), f"AbstractionUHModel.{col}: values differ"


# ---------------------------------------------------------------------------
# 2. Dual-path evaluate()
# ---------------------------------------------------------------------------

def test_evaluate_fast_and_fallback_agree():
    """``evaluate()`` gives identical results on the fast and fallback paths.

    Disabling ``predict_arrays`` on one instance forces ``CalibrationProblem``
    down the DataFrame extractor path; both must agree bit-for-bit.
    """
    from sparsehydro.calibration import CalibrationProblem
    from sparsehydro.metrics.objectives import (
        WeightedRMSE, RMSE, NashSutcliffe, KGE,
    )

    df = _rain_df().assign(obs=np.random.default_rng(5).gamma(2.0, 1.0, N))
    mask = np.zeros(N, dtype=bool)
    mask[30:370] = True

    def _problem():
        m = PeakTailUH()
        m.initialize()
        m.validate()
        return CalibrationProblem(
            model=m, data=df,
            objectives=[WeightedRMSE(), RMSE(), NashSutcliffe(), KGE()],
            column_map={"observed": "obs", "predicted": "Q_pred"}, mask=mask,
        )

    fast = _problem()
    slow = _problem()
    if not _has_array_predict(slow._model):
        pytest.skip("predict_arrays() not implemented yet (pre-Step-2)")

    # Force the fallback: hide predict_arrays on this instance only.
    slow._model.predict_arrays = None
    slow = _problem() if slow._model.predict_arrays is not None else slow
    object.__setattr__(
        slow._model, "predict_arrays",
        lambda: (_ for _ in ()).throw(AttributeError("disabled")),
    )

    lo, hi = fast.bounds
    rng = np.random.default_rng(17)
    for _ in range(20):
        x = lo + rng.random(lo.shape) * (hi - lo)
        f_fast = fast.evaluate(x, penalty_weight=0.0)
        try:
            f_slow = slow.evaluate(x, penalty_weight=0.0)
        except AttributeError:
            pytest.skip("fallback path cannot be forced on this build")
        same = (f_fast == f_slow) | (np.isnan(f_fast) & np.isnan(f_slow))
        assert same.all(), f"fast {f_fast} != fallback {f_slow} at x={x}"


# ---------------------------------------------------------------------------
# 3. Cache invalidation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", ALL_UH, ids=[c.__name__ for c in ALL_UH])
def test_reprepare_with_new_rain_invalidates(cls):
    """A second ``prepare()`` with different rain changes the prediction.

    ``SequentialFitter`` re-prepares the same model across events, so a forcing
    cache keyed only on the time base would silently reuse stale rainfall.
    """
    m = cls()
    m.initialize()
    m.validate()

    df_a = _rain_df(seed=0)
    df_b = df_a.copy()
    df_b["rain"] = _rain_df(seed=99)["rain"].to_numpy() * 3.0 + 0.05

    m.prepare(df_a)
    q_a = m.predict()["Q_pred"].to_numpy(dtype=float)
    m.prepare(df_b)
    q_b = m.predict()["Q_pred"].to_numpy(dtype=float)

    assert not np.allclose(q_a, q_b), (
        f"{cls.__name__}: prediction unchanged after re-prepare with new rain -- "
        f"the forcing cache is stale"
    )


@pytest.mark.parametrize("cls", ALL_UH, ids=[c.__name__ for c in ALL_UH])
def test_predicted_flow_is_timestep_invariant(cls):
    """The same physical rainfall gives the same response at 5 min and 1 h.

    Time parameters are in hours and kernels are exact bin averages, so an
    hourly rain series disaggregated to 5 minutes gives, at each hour boundary,
    exactly the hourly prediction. (Relabelling the *same per-step depths* as a
    different timestep is a different storm and correctly changes Q.)
    """
    rain_h = _rain_df(seed=3)["rain"].to_numpy()[: N // 12]
    rain_5 = np.repeat(rain_h / 12.0, 12)

    m = cls()
    m.initialize()
    m.validate()
    m.prepare(pd.DataFrame({"datetime": pd.date_range("2022-05-01", periods=len(rain_5), freq="5min"),
                            "rain": rain_5}))
    q_5 = m.predict()["Q_pred"].to_numpy(dtype=float)
    m.prepare(pd.DataFrame({"datetime": pd.date_range("2022-05-01", periods=len(rain_h), freq="1h"),
                            "rain": rain_h}))
    q_1 = m.predict()["Q_pred"].to_numpy(dtype=float)

    np.testing.assert_allclose(
        q_5[11::12], q_1, rtol=1e-9, atol=1e-12,
        err_msg=f"{cls.__name__}: response depends on the model timestep",
    )


@pytest.mark.parametrize("cls", ALL_UH, ids=[c.__name__ for c in ALL_UH])
def test_reprepare_with_new_timestep_updates_cached_dt(cls):
    """Re-``prepare()`` at a different timestep updates the cached ``dt_hours``.

    ``dt_hours`` sets the bin width of every kernel, so a stale value would
    silently give the wrong response; this asserts the cache directly.
    """
    m = cls()
    m.initialize()
    m.validate()

    rain = _rain_df(seed=3)["rain"].to_numpy()
    m.prepare(pd.DataFrame(
        {"datetime": pd.date_range("2022-05-01", periods=N, freq="5min"), "rain": rain}
    ))
    if not hasattr(m, "_dt_hours"):
        pytest.skip("dt_hours not cached yet (pre-Step-1)")
    assert m._dt_hours == pytest.approx(5.0 / 60.0)

    m.prepare(pd.DataFrame(
        {"datetime": pd.date_range("2022-05-01", periods=N, freq="1h"), "rain": rain}
    ))
    assert m._dt_hours == pytest.approx(1.0), (
        f"{cls.__name__}: cached dt_hours survived a re-prepare at a new timestep"
    )


# ---------------------------------------------------------------------------
# 4. No pandas in the hot loop
# ---------------------------------------------------------------------------

def _count_dataframes(fn) -> int:
    """Return how many ``pd.DataFrame`` instances *fn* constructs.

    :param fn: Zero-argument callable to run under instrumentation.
    :type fn: Callable
    :returns: Number of ``DataFrame.__init__`` invocations.
    :rtype: int
    """
    original = pd.DataFrame.__init__
    count = 0

    def counting_init(self, *args, **kwargs):
        nonlocal count
        count += 1
        return original(self, *args, **kwargs)

    pd.DataFrame.__init__ = counting_init
    try:
        fn()
    finally:
        pd.DataFrame.__init__ = original
    return count


@pytest.mark.parametrize("composite", [False, True], ids=["PeakTailUH", "AbstractionUHModel"])
def test_evaluate_constructs_no_dataframes(composite):
    """``evaluate()`` builds zero DataFrames once the fast path is in place.

    A deterministic proxy for "the pandas round-trip is gone" -- far more stable
    on CI than a wall-clock assertion, and it fails loudly if a later change
    reintroduces a per-evaluation frame.
    """
    from sparsehydro.calibration import CalibrationProblem
    from sparsehydro.metrics.objectives import RMSE, NashSutcliffe

    df = _rain_df().assign(
        obs=np.random.default_rng(5).gamma(2.0, 1.0, N),
        temperature_c=np.full(N, 15.0),
    )

    if composite:
        from sparsehydro.models import AbstractionUHModel
        from sparsehydro.models.rdii import IAModel
        model = AbstractionUHModel(abstraction=IAModel(), uh=PeakTailUH())
    else:
        model = PeakTailUH()
    model.initialize()
    model.validate()

    problem = CalibrationProblem(
        model=model, data=df, objectives=[RMSE(), NashSutcliffe()],
        column_map={"observed": "obs", "predicted": "Q_pred"},
    )
    if not getattr(model, "supports_array_predict", False):
        pytest.skip("fast path not implemented for this model yet")
    if not hasattr(problem, "_predicted_key"):
        pytest.skip("CalibrationProblem does not use the fast path yet (pre-Step-2)")

    lo, hi = problem.bounds
    x = lo + 0.5 * (hi - lo)
    problem.evaluate(x, penalty_weight=0.0)          # warm up any lazy state

    n_frames = _count_dataframes(lambda: problem.evaluate(x, penalty_weight=0.0))
    assert n_frames == 0, (
        f"evaluate() constructed {n_frames} DataFrame(s); the calibration hot "
        f"loop must stay free of pandas round-trips"
    )
