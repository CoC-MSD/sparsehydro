"""Golden-output regression tests for the convolution / prediction path.

These pin the numerical results produced by ``tests/golden/generate.py``.  The
refactor they guard is described in the plan: an ndarray fast path through
``CalibrationProblem.evaluate``, adaptive convolution, and kernel-length safety.

**Tolerance tiers.**  Most fixtures must match *bit-for-bit* -- ``np.array_equal``
semantics with NaN treated as equal -- not ``allclose``.  A pure refactor that
changes the last ulp is a refactor that reassociated a floating-point expression,
and that is exactly the class of accident these tests exist to catch.

Families with a different tier:

``UH_RTOL``
    Unit hydrograph kernels and everything built on them (UH predictions,
    composite predictions, ``evaluate``) are computed with ``scipy.special``
    functions whose last bits can differ between SciPy builds, so they are
    pinned to ``rtol=1e-12`` rather than bit-for-bit.
``RDII_RTOL``
    RDII predictions may shift at FFT round-off level.

The UH fixtures were re-baselined when unit hydrographs moved from time steps
to hours (kernels are now exact bin averages of the continuous response).

If a fixture file is missing, the tests skip rather than fail -- a fresh
checkout should not be blocked on running the generator first.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.golden import cases as C

FIXTURE_DIR = Path(__file__).resolve().parent / "golden"

# Step 5: RDII's convolution-branch rule changes; FFT round-off only.
RDII_RTOL = 1e-9
RDII_ATOL = 1e-12

# Unit hydrograph pins: scipy.special results can differ in the last bits
# across SciPy builds.
UH_RTOL = 1e-12
UH_ATOL = 1e-15


def _load(name: str) -> dict[str, np.ndarray]:
    """Load a fixture archive, or skip the test if it has not been generated.

    :param name: Fixture base name (e.g. ``"kernels"``).
    :type name: str
    :returns: Mapping of fixture key to array.
    :rtype: dict[str, numpy.ndarray]
    """
    path = FIXTURE_DIR / f"{name}.npz"
    if not path.exists():
        pytest.skip(
            f"{path.name} not generated; run: python tests/golden/generate.py --regenerate"
        )
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def assert_bit_identical(got: np.ndarray, want: np.ndarray, key: str) -> None:
    """Assert two arrays are equal bit-for-bit, treating NaN as equal.

    :param got: Freshly computed array.
    :type got: numpy.ndarray
    :param want: Pinned reference array.
    :type want: numpy.ndarray
    :param key: Fixture key, for the failure message.
    :type key: str
    :raises AssertionError: If shapes differ or any element differs.
    """
    got = np.asarray(got, dtype=float)
    want = np.asarray(want, dtype=float)
    assert got.shape == want.shape, (
        f"{key}: shape changed {want.shape} -> {got.shape}"
    )
    same = (got == want) | (np.isnan(got) & np.isnan(want))
    if same.all():
        return

    bad = np.flatnonzero(~same)
    with np.errstate(invalid="ignore", divide="ignore"):
        delta = np.abs(got[bad] - want[bad])
    raise AssertionError(
        f"{key}: {bad.size} of {got.size} elements differ "
        f"(max |delta| = {np.nanmax(delta):.3e}, first at index {bad[0]}: "
        f"{want[bad[0]]!r} -> {got[bad[0]]!r}).\n"
        f"Expected bit-identical results.  If this change is intended, it belongs "
        f"to a plan step that sanctions it -- re-baseline with "
        f"`python tests/golden/generate.py --regenerate` and say so in the commit."
    )


def assert_uh_close(got: np.ndarray, want: np.ndarray, key: str) -> None:
    """Assert a unit-hydrograph-derived result matches its pin to UH_RTOL."""
    got = np.asarray(got, dtype=float)
    want = np.asarray(want, dtype=float)
    assert got.shape == want.shape, f"{key}: shape changed {want.shape} -> {got.shape}"
    np.testing.assert_allclose(got, want, rtol=UH_RTOL, atol=UH_ATOL, equal_nan=True,
                               err_msg=f"{key}: drifted from its pin")


def _is_rdii(key: str) -> bool:
    """Return ``True`` when *key* names an RDII fixture (relaxed tolerance)."""
    return key.startswith("RDIIModel/")


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

def test_kernels_pinned():
    """Every UH kernel matches its pin."""
    want = _load("kernels")
    checked = 0
    for case_id, model_name, params in C.iter_uh_cases():
        for dt_label, dt in C.DT_CASES.items():
            key = f"{case_id}|{dt_label}"
            if key not in want:
                pytest.fail(f"fixture missing key {key!r}; regenerate the archive")
            m = C.build(model_name, params)
            assert_uh_close(m.get_kernel(dt), want[key], key)
            checked += 1
    assert checked > 0


@pytest.mark.parametrize("case_id", sorted(C.TRUNCATION_ACTIVE))
@pytest.mark.parametrize("dt_label", sorted(C.DT_CASES))
def test_truncated_kernels_are_finite_bounded_and_unit_area(case_id, dt_label):
    """Responses longer than MAX_DURATION_HOURS stay finite, bounded and unit-area.

    These are the corners that used to break: a step-based ``NashUH(n=100,
    k=500)`` built a 250,001-element all-NaN kernel.
    """
    from sparsehydro.models.unithydrograph.kernels import MAX_DURATION_HOURS

    model_name, label = case_id.split("/", 1)
    params = dict(dict(C.UH_CASES[model_name])[label])
    m = C.build(model_name, params)
    dt = C.DT_CASES[dt_label]
    k = np.asarray(m.get_kernel(dt), dtype=float)

    assert np.isfinite(k).all(), f"{case_id}: kernel contains NaN/inf"
    assert (k >= 0.0).all(), f"{case_id}: kernel has negative ordinates"
    assert k.size <= int(np.ceil(MAX_DURATION_HOURS / dt)) + 2, f"{case_id}: kernel too long"
    assert abs(k.sum() * dt - 1.0) < 1e-9, f"{case_id}: truncated kernel lost unit area"


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

def _uses_fft(n: int, kernel_len: int) -> bool:
    """Return whether this (n, kernel) pair takes the FFT convolution branch.

    Asked of the production heuristic rather than hard-coded, so the tolerance
    tier follows the code instead of drifting away from it.
    """
    try:
        from sparsehydro.models.convolution import convolution_method
    except ImportError:
        return False
    return convolution_method(n, kernel_len) != "direct"


def test_uh_predictions_pinned():
    """Every UH ``predict()["Q_pred"]`` matches its pin.

    Bit-for-bit on the direct branch, which covers every event-scale window.
    Cases large enough to take the FFT branch are allowed FFT round-off, and are
    additionally checked for the non-negativity the clipping in
    ``convolve_causal`` restores.
    """
    want = _load("predictions")
    checked = fft_cases = 0
    for n in C.PREDICT_LENGTHS:
        df = C.rain_series(n)
        for case_id, model_name, params in C.predict_cases(n):
            key = f"{case_id}|n{n}"
            if key not in want:
                pytest.fail(f"fixture missing key {key!r}; regenerate the archive")
            m = C.build(model_name, params)
            m.prepare(df)
            got = m.predict()["Q_pred"].to_numpy(dtype=float)

            if _uses_fft(n, len(m.get_kernel(C.DT_CASES["5min"]))):
                np.testing.assert_allclose(
                    got, want[key], rtol=RDII_RTOL, atol=RDII_ATOL,
                    err_msg=f"{key}: drifted beyond FFT round-off",
                )
                assert (got >= 0.0).all(), (
                    f"{key}: FFT branch produced negative flow from non-negative "
                    f"rainfall and kernel; the round-off clip is not being applied"
                )
                fft_cases += 1
            else:
                assert_uh_close(got, want[key], key)
                checked += 1
    assert checked > 0
    # The n=5000 long-kernel cases must actually exercise the FFT branch,
    # otherwise this test silently stops covering it.
    assert fft_cases > 0


def test_composite_predictions_pinned():
    """``AbstractionUHModel`` predictions match their pins.

    This is the pin that guards Step 3, which hoists the child ``prepare()``
    calls out of ``predict()``.
    """
    want = _load("predictions")
    from sparsehydro.models import AbstractionUHModel
    from sparsehydro.models.rdii import IAModel
    from sparsehydro.models.unithydrograph import GammaUH, PeakTailUH

    n = 600
    df = C.rain_series(n).assign(temperature_c=np.full(n, 15.0))
    for label, uh in (("peaktail", PeakTailUH(A=20.0)), ("gamma", GammaUH(A=20.0))):
        key = f"AbstractionUHModel/{label}|n{n}"
        if key not in want:
            pytest.fail(f"fixture missing key {key!r}; regenerate the archive")
        m = AbstractionUHModel(abstraction=IAModel(), uh=uh)
        m.initialize()
        m.validate()
        m.prepare(df)
        assert_uh_close(m.predict()["Q_pred"].to_numpy(dtype=float), want[key], key)


def test_rdii_predictions_within_fft_tolerance():
    """RDII predictions match their pins to FFT round-off.

    Step 5 changes which convolution branch RDII takes, so this family gets
    ``rtol=1e-9`` rather than bit-identity.
    """
    want = _load("predictions")
    keys = [k for k in want if _is_rdii(k)]
    if not keys:
        pytest.skip("no RDII fixtures recorded")

    from sparsehydro.models.rdii import RDIIModel

    n = 600
    df = C.rain_series(n).assign(temperature_c=np.full(n, 15.0))
    m = RDIIModel()
    m.initialize()
    m.validate()
    m.prepare(df.rename(columns={"rain": "rainfall_in"}))
    pred = m.predict()

    for key in keys:
        col = key.split("|", 1)[0].split("/", 1)[1]
        got = pred[col].to_numpy(dtype=float)
        np.testing.assert_allclose(
            got, want[key], rtol=RDII_RTOL, atol=RDII_ATOL,
            err_msg=f"{key}: RDII prediction drifted beyond FFT round-off",
        )


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------

def test_objectives_pinned():
    """Every objective value matches its pin bit-for-bit across all scenarios.

    Guards Step 2, which re-expresses ``compute()`` on top of a new
    ``compute_selected()`` hook.  Reassociating a reduction would show up here.
    """
    from tests.golden.generate import build_objectives

    want = _load("objectives")
    got = build_objectives()

    missing = sorted(set(want) - set(got))
    assert not missing, f"fixture keys no longer produced: {missing[:5]}"

    for key, value in got.items():
        if key not in want:
            pytest.fail(f"fixture missing key {key!r}; regenerate the archive")
        assert_bit_identical(value, want[key], key)


# ---------------------------------------------------------------------------
# End-to-end evaluate()
# ---------------------------------------------------------------------------

def test_evaluate_pinned():
    """``CalibrationProblem.evaluate`` matches its pins (UH tolerance tier).

    The single most important regression test in this module: it exercises
    parameter application, ``predict()``, result extraction, masking, every
    objective, and the constraint penalty in one call.
    """
    from tests.golden.generate import build_evaluate

    want = _load("evaluate")
    got = build_evaluate()

    missing = sorted(set(want) - set(got))
    assert not missing, f"fixture keys no longer produced: {missing[:5]}"

    for key, value in got.items():
        if key not in want:
            pytest.fail(f"fixture missing key {key!r}; regenerate the archive")
        assert_uh_close(value, want[key], key)
