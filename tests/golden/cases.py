"""Case definitions shared by the golden-fixture generator and its tests.

Kept separate from :mod:`generate` so the test module can enumerate the same
cases without importing generation/CLI machinery.

Every case is a ``(case_id, factory)`` pair where ``factory()`` returns a freshly
built, INITIALIZED model with the case's parameters applied.  Case ids double as
``.npz`` array keys, so they must be stable: renaming one silently drops its pin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sparsehydro.models.unithydrograph import (
    SCSUH, ClarkUH, DecayUH, GammaDelayUH, GammaUH, GeneralizedGammaUH, LognormalUH,
    MixtureUH, ModClarkUH, NashUH, PeakTailUH, RectangleUH, SnyderUH, TriangleUH, WeibullUH,
)

# Time-step sizes exercised for every kernel case.
DT_CASES: dict[str, float] = {"5min": 5.0 / 60.0, "15min": 0.25, "1hr": 1.0}

# Parameter grids per model, all times in hours.  Each entry is (label, {param: value}).
# Coverage intent: defaults, low/mid/high per parameter, degenerate edges, and
# responses long enough to hit the duration cap (TRUNCATION_ACTIVE).
UH_CASES: dict[str, list[tuple[str, dict[str, float]]]] = {
    "GammaUH": [
        ("default", {}),
        ("low", {"alpha": 0.1, "theta": 0.01}),
        ("mid", {"alpha": 3.0, "theta": 4.0}),
        ("high", {"alpha": 5.0, "theta": 20.0}),
        ("extreme", {"alpha": 50.0, "theta": 240.0}),
    ],
    "NashUH": [
        ("default", {}),
        ("low", {"n": 0.1, "K": 0.01}),
        ("mid", {"n": 3.0, "K": 1.0}),
        ("high", {"n": 10.0, "K": 2.0}),
        ("extreme", {"n": 50.0, "K": 240.0}),
    ],
    "TriangleUH": [
        ("default", {}),
        ("infeasible_tp_gt_tb", {"Tp": 5.0, "Tb": 2.0}),
        ("mid", {"Tp": 5.0, "Tb": 16.0}),
        ("high", {"Tp": 40.0, "Tb": 80.0}),
    ],
    "RectangleUH": [
        ("default", {}),
        ("low", {"Tr": 0.01}),
        ("mid", {"Tr": 16.0}),
        ("high", {"Tr": 80.0}),
    ],
    "DecayUH": [
        ("default", {}),
        ("mid", {"K": 2.0}),
        ("high", {"K": 80.0}),
    ],
    "GammaDelayUH": [
        ("default", {}),
        ("no_delay", {"td": 0.0}),
        ("mid", {"alpha": 3.0, "theta": 2.0, "td": 2.0}),
    ],
    "PeakTailUH": [
        ("default", {}),
        ("peak_only", {"w": 0.0}),
        ("tail_only", {"w": 1.0}),
        ("mid", {"w": 0.4, "td": 1.0, "peak_Tp": 5.0, "peak_Tb": 16.0,
                 "tail_alpha": 3.0, "tail_theta": 5.0}),
    ],
    "SCSUH": [("default", {}), ("mid", {"Tp": 4.0})],
    "SnyderUH": [("default", {}), ("mid", {"tL": 3.0, "Cp": 0.4})],
    "ClarkUH": [("default", {}), ("mid", {"Tc": 4.0, "Rs": 3.0}), ("small_storage", {"Rs": 0.001})],
    "ModClarkUH": [("default", {})],
    "WeibullUH": [("default", {}), ("mid", {"k": 3.0, "lam": 5.0})],
    "LognormalUH": [("default", {}), ("heavy", {"mu": 5.5, "sigma": 2.0})],
    "GeneralizedGammaUH": [("default", {}), ("mid", {"a": 3.0, "c": 0.7, "scale": 2.0})],
    "MixtureUH": [("default", {})],
}

_CLASSES = {
    "GammaUH": GammaUH, "NashUH": NashUH, "TriangleUH": TriangleUH,
    "RectangleUH": RectangleUH, "DecayUH": DecayUH, "GammaDelayUH": GammaDelayUH,
    "PeakTailUH": PeakTailUH, "SCSUH": SCSUH, "SnyderUH": SnyderUH, "ClarkUH": ClarkUH,
    "ModClarkUH": ModClarkUH, "WeibullUH": WeibullUH, "LognormalUH": LognormalUH,
    "GeneralizedGammaUH": GeneralizedGammaUH, "MixtureUH": MixtureUH,
}

# Responses that reach MAX_DURATION_HOURS and are truncated and renormalised.
TRUNCATION_ACTIVE: frozenset[str] = frozenset({
    "GammaUH/extreme", "NashUH/extreme", "LognormalUH/heavy",
})

# Long-kernel cases also pinned on a 5000-step series, which takes the
# overlap-add (FFT) convolution branch.
LONG_CASES: frozenset[str] = frozenset({
    "GammaUH/mid", "NashUH/high", "DecayUH/high", "PeakTailUH/mid",
})


def build(model_name: str, params: dict[str, float]):
    """Return a freshly built, INITIALIZED model with *params* applied.

    :param model_name: Key into the UH class table (e.g. ``"PeakTailUH"``).
    :type model_name: str
    :param params: Flat parameter name -> value overrides applied after
        ``initialize()``.
    :type params: dict[str, float]
    :returns: An initialized model instance.
    :rtype: IUnitHydroComponent
    """
    m = _CLASSES[model_name]()
    m.initialize()
    for k, v in params.items():
        m.apply_flat_parameter(k, float(v))
    return m


def iter_uh_cases():
    """Yield ``(case_id, model_name, params)`` for every UH parameter case.

    :returns: Iterator of ``(str, str, dict)`` triples; ``case_id`` is
        ``"{model_name}/{label}"``.
    :rtype: Iterator[tuple[str, str, dict[str, float]]]
    """
    for model_name, cases in UH_CASES.items():
        for label, params in cases:
            yield f"{model_name}/{label}", model_name, params


def rain_series(n: int, seed: int = 0) -> pd.DataFrame:
    """Return a deterministic synthetic rainfall frame of *n* 5-minute steps.

    Uses a fixed-seed gamma draw with a hard zero floor so the series has
    realistic intermittency (long dry stretches punctuated by bursts) rather
    than uniform drizzle.

    :param n: Number of timesteps.
    :type n: int
    :param seed: PRNG seed.
    :type seed: int
    :returns: DataFrame with ``datetime`` and ``rain`` columns.
    :rtype: pandas.DataFrame
    """
    rng = np.random.default_rng(seed)
    rain = rng.gamma(0.3, 0.05, n)
    rain[rain < 0.01] = 0.0
    return pd.DataFrame(
        {"datetime": pd.date_range("2022-01-01", periods=n, freq="5min"), "rain": rain}
    )


def observed_series(n: int, seed: int = 1) -> np.ndarray:
    """Return a deterministic synthetic observed-flow array with embedded NaNs.

    :param n: Number of timesteps.
    :type n: int
    :param seed: PRNG seed.
    :type seed: int
    :returns: Float array of length *n* with a NaN gap at ``[n//3, n//3 + 5)``.
    :rtype: numpy.ndarray
    """
    rng = np.random.default_rng(seed)
    obs = rng.gamma(2.0, 1.0, n)
    obs[n // 3: n // 3 + 5] = np.nan
    return obs


# Prediction series lengths: every case at 600 steps (direct convolution
# branch); LONG_CASES also at 5000 steps (FFT branch).
PREDICT_LENGTHS: tuple[int, ...] = (600, 5000)


def predict_cases(n: int):
    """Yield the UH cases pinned at series length *n*."""
    for case_id, model_name, params in iter_uh_cases():
        if n == PREDICT_LENGTHS[0] or case_id in LONG_CASES:
            yield case_id, model_name, params
