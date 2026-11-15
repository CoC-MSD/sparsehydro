"""Generate the golden-output fixtures that pin SparseHydro's numerical results.

Run this **before** any refactor of the convolution / prediction path, commit the
resulting ``.npz`` files, and then assert against them with
``tests/test_golden_convolution.py`` after every change.

Usage::

    python tests/golden/generate.py --regenerate

The ``--regenerate`` flag is required: without it the script refuses to
overwrite existing fixtures, so an accidental run cannot silently re-baseline a
regression.

Four fixture files are written next to this module:

``kernels.npz``
    ``get_kernel(dt)`` for every UH class x parameter case x timestep size.
``predictions.npz``
    ``predict()["Q_pred"]`` for every UH case over two synthetic rain series,
    plus the composite / ensemble / RDII models.
``objectives.npz``
    Every objective in :mod:`sparsehydro.calibration.objectives` over a fixed
    observed/predicted pair, including NaN, ``+inf`` and all-masked edge cases.
``evaluate.npz``
    Full ``CalibrationProblem.evaluate(x)`` objective vectors -- the pin that
    actually matters, since it exercises the whole chain end to end.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

# Allow `python tests/golden/generate.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.golden import cases as C  # noqa: E402

FIXTURE_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

def build_kernels() -> dict[str, np.ndarray]:
    """Return ``{case_id + "|" + dt_label: kernel}`` for every UH case.

    :returns: Mapping of fixture key to kernel ordinate array.
    :rtype: dict[str, numpy.ndarray]
    """
    out: dict[str, np.ndarray] = {}
    for case_id, model_name, params in C.iter_uh_cases():
        for dt_label, dt in C.DT_CASES.items():
            m = C.build(model_name, params)
            # NashUH's overflow corner legitimately warns today; the fixture
            # records the (NaN) result rather than suppressing the case.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out[f"{case_id}|{dt_label}"] = np.asarray(m.get_kernel(dt), dtype=float)
    return out


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

def build_predictions() -> dict[str, np.ndarray]:
    """Return ``{case_id + "|n" + length: Q_pred}`` plus composite-model cases.

    :returns: Mapping of fixture key to predicted-flow array.
    :rtype: dict[str, numpy.ndarray]
    """
    out: dict[str, np.ndarray] = {}
    for n in C.PREDICT_LENGTHS:
        df = C.rain_series(n)
        for case_id, model_name, params in C.predict_cases(n):
            m = C.build(model_name, params)
            m.prepare(df)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                q = m.predict()["Q_pred"].to_numpy(dtype=float)
            out[f"{case_id}|n{n}"] = q

    out.update(_composite_predictions())
    return out


def _composite_predictions() -> dict[str, np.ndarray]:
    """Return predicted-flow pins for the composite / ensemble / RDII models.

    These exercise the child-model plumbing that Step 3 rewires, which the bare
    UH cases do not touch.

    :returns: Mapping of fixture key to predicted-flow array.
    :rtype: dict[str, numpy.ndarray]
    """
    from sparsehydro.models import AbstractionUHModel
    from sparsehydro.models.rdii import IAModel
    from sparsehydro.models.unithydrograph import GammaUH, PeakTailUH

    out: dict[str, np.ndarray] = {}
    n = 600
    df = C.rain_series(n)
    df = df.assign(temperature_c=np.full(n, 15.0))

    for label, uh in (("peaktail", PeakTailUH(A=20.0)), ("gamma", GammaUH(A=20.0))):
        m = AbstractionUHModel(abstraction=IAModel(), uh=uh)
        m.initialize()
        m.validate()
        m.prepare(df)
        out[f"AbstractionUHModel/{label}|n{n}"] = (
            m.predict()["Q_pred"].to_numpy(dtype=float)
        )

    out.update(_rdii_predictions(df, n))
    return out


def _rdii_predictions(df, n: int) -> dict[str, np.ndarray]:
    """Return RDII model pins, or ``{}`` if the model cannot be built here.

    RDII construction varies with optional configuration; a failure to build is
    recorded as a skip rather than aborting the whole fixture run.

    :param df: Rainfall frame with a ``temperature_c`` column.
    :type df: pandas.DataFrame
    :param n: Series length.
    :type n: int
    :returns: Mapping of fixture key to predicted-flow array (possibly empty).
    :rtype: dict[str, numpy.ndarray]
    """
    try:
        from sparsehydro.models.rdii import RDIIModel
    except Exception as exc:  # pragma: no cover - optional path
        print(f"  ! RDIIModel unavailable, skipping: {exc}")
        return {}

    try:
        m = RDIIModel()
        m.initialize()
        m.validate()
        # RDIIModel expects the rainfall column under its own name.
        m.prepare(df.rename(columns={"rain": "rainfall_in"}))
        pred = m.predict()
        return {
            f"RDIIModel/{col}|n{n}": pred[col].to_numpy(dtype=float)
            for col in pred.columns
            if col != "datetime"
        }
    except Exception as exc:  # pragma: no cover - optional path
        print(f"  ! RDIIModel default construction failed, skipping: {exc}")
        return {}


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------

def _objective_instances() -> list[tuple[str, object]]:
    """Return ``(label, objective)`` pairs covering every metric class.

    :returns: List of labelled objective instances.
    :rtype: list[tuple[str, object]]
    """
    from sparsehydro.calibration.objectives import (
        MSE, RMSE, WeightedRMSE, MAE, PeakWeightedMSE, NashSutcliffe, KGE,
        PBIAS, VolumeRelativeError, LogNSE, IndexOfAgreement,
        ConcordanceCorrelationCoefficient,
    )

    n = 200
    rng = np.random.default_rng(7)
    weights = rng.random(n) * 2.0

    return [
        ("MSE", MSE()),
        ("RMSE", RMSE()),
        ("WeightedRMSE_noweights", WeightedRMSE()),
        ("WeightedRMSE_weighted", WeightedRMSE(weights=weights)),
        ("MAE", MAE()),
        ("PeakWeightedMSE", PeakWeightedMSE()),
        ("PeakWeightedMSE_p2", PeakWeightedMSE(power=2.0)),
        ("NashSutcliffe", NashSutcliffe()),
        ("KGE", KGE()),
        ("PBIAS", PBIAS()),
        ("VolumeRelativeError", VolumeRelativeError()),
        ("LogNSE", LogNSE()),
        ("IndexOfAgreement", IndexOfAgreement()),
        ("ConcordanceCorrelationCoefficient", ConcordanceCorrelationCoefficient()),
    ]


def build_objectives() -> dict[str, np.ndarray]:
    """Return ``{label + "|" + scenario: value}`` for every objective.

    Scenarios cover the plain case, NaN on both sides, a ``+inf`` in predicted
    (which today is *not* excluded -- only NaN is), a partial mask, and a mask
    that selects nothing (recorded as NaN, since it raises).

    :returns: Mapping of fixture key to 0-d float array.
    :rtype: dict[str, numpy.ndarray]
    """
    n = 200
    rng = np.random.default_rng(11)
    obs = rng.gamma(2.0, 1.0, n)
    pred = obs * 0.85 + rng.normal(0.0, 0.3, n)
    pred = np.abs(pred)

    obs_nan = obs.copy()
    obs_nan[10:15] = np.nan
    pred_nan = pred.copy()
    pred_nan[100:103] = np.nan

    pred_inf = pred.copy()
    pred_inf[50] = np.inf

    mask_partial = np.zeros(n, dtype=bool)
    mask_partial[20:150] = True
    mask_empty = np.zeros(n, dtype=bool)

    scenarios = {
        "plain": (obs, pred, None),
        "nan_both": (obs_nan, pred_nan, None),
        "inf_pred": (obs, pred_inf, None),
        "masked": (obs_nan, pred_nan, mask_partial),
        "mask_empty": (obs, pred, mask_empty),
    }

    out: dict[str, np.ndarray] = {}
    for label, obj in _objective_instances():
        for scen, (o, p, m) in scenarios.items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    val = float(obj.compute(o, p, mask=m))
            except Exception:
                # Recorded as NaN: the pin is that this scenario still raises.
                val = np.nan
            out[f"{label}|{scen}"] = np.array(val, dtype=float)
    return out


# ---------------------------------------------------------------------------
# End-to-end CalibrationProblem.evaluate
# ---------------------------------------------------------------------------

def build_evaluate() -> dict[str, np.ndarray]:
    """Return ``{problem_label + "|x" + i: F}`` for fixed parameter vectors.

    This is the pin that matters most: it exercises parameter application,
    ``predict()``, result extraction, masking and every objective in one shot.

    :returns: Mapping of fixture key to objective vector.
    :rtype: dict[str, numpy.ndarray]
    """
    from sparsehydro.calibration import CalibrationProblem
    from sparsehydro.calibration.objectives import (
        WeightedRMSE, RMSE, NashSutcliffe, KGE,
    )
    from sparsehydro.models import AbstractionUHModel
    from sparsehydro.models.rdii import IAModel
    from sparsehydro.models.unithydrograph import PeakTailUH

    n = 600
    df = C.rain_series(n).assign(
        obs=C.observed_series(n), temperature_c=np.full(n, 15.0)
    )
    mask = np.zeros(n, dtype=bool)
    mask[50:550] = True
    weights = np.ones(n)
    weights[200:300] = 5.0

    out: dict[str, np.ndarray] = {}

    def _objectives():
        return [WeightedRMSE(weights=weights), RMSE(), NashSutcliffe(), KGE()]

    # --- bare PeakTailUH ---------------------------------------------------
    m = PeakTailUH()
    m.initialize()
    m.validate()
    prob = CalibrationProblem(
        model=m, data=df, objectives=_objectives(),
        column_map={"observed": "obs", "predicted": "Q_pred"}, mask=mask,
    )
    for i, values in enumerate(_peaktail_vectors()):
        x = np.array([values[name] for name in prob.param_names], dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out[f"PeakTailUH|x{i}"] = prob.evaluate(x, penalty_weight=0.0)
        out[f"PeakTailUH_pen|x{i}"] = prob.evaluate(x, penalty_weight=1e6)

    # --- AbstractionUHModel ------------------------------------------------
    cm = AbstractionUHModel(abstraction=IAModel(), uh=PeakTailUH())
    cm.initialize()
    cm.validate()
    prob_c = CalibrationProblem(
        model=cm, data=df, objectives=_objectives(),
        column_map={"observed": "obs", "predicted": "Q_pred"}, mask=mask,
    )
    lo, hi = prob_c.bounds
    mid = 0.5 * (lo + hi)
    rng = np.random.default_rng(23)
    for i in range(5):
        frac = np.clip(0.5 + 0.15 * rng.normal(size=mid.shape), 0.05, 0.95)
        x = lo + frac * (hi - lo)
        out[f"AbstractionUHModel|x{i}"] = np.asarray(x, dtype=float)  # pin the input too
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out[f"AbstractionUHModel_F|x{i}"] = prob_c.evaluate(x, penalty_weight=0.0)

    return out


def _peaktail_vectors() -> list[dict[str, float]]:
    """Return the five fixed PeakTailUH parameter sets (times in hours).

    Given by name so they do not depend on parameter registration order.
    Vector 4 violates ``peak_Tp_lt_Tb`` so the constraint penalty is pinned.
    """
    names = ("A", "w", "td", "peak_Tp", "peak_Tb", "tail_alpha", "tail_theta")
    rows = [
        (1.0, 0.5, 0.25, 1.0, 3.0, 2.0, 2.0),
        (0.1, 0.0, 0.0, 0.05, 0.1, 0.1, 0.01),        # lower corner
        (500.0, 1.0, 48.0, 240.0, 960.0, 50.0, 240.0),  # upper corner
        (20.0, 0.3, 1.0, 3.0, 15.0, 2.5, 4.0),
        (8.0, 0.85, 2.5, 9.0, 6.0, 6.0, 12.0),         # infeasible: Tp > Tb
    ]
    return [dict(zip(names, row)) for row in rows]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

BUILDERS = {
    "kernels": build_kernels,
    "predictions": build_predictions,
    "objectives": build_objectives,
    "evaluate": build_evaluate,
}


def main(argv: list[str] | None = None) -> int:
    """Generate and write all fixture files.

    :param argv: Argument list (defaults to ``sys.argv[1:]``).
    :type argv: list[str] or None
    :returns: Process exit code.
    :rtype: int
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--regenerate", action="store_true",
        help="Required to overwrite existing fixtures.",
    )
    ap.add_argument(
        "--only", choices=sorted(BUILDERS), default=None,
        help="Regenerate a single fixture file instead of all four.",
    )
    args = ap.parse_args(argv)

    targets = [args.only] if args.only else list(BUILDERS)
    existing = [t for t in targets if (FIXTURE_DIR / f"{t}.npz").exists()]
    if existing and not args.regenerate:
        print(
            f"Refusing to overwrite existing fixtures: {', '.join(existing)}.\n"
            f"Pass --regenerate if you intend to re-baseline them.",
            file=sys.stderr,
        )
        return 1

    for name in targets:
        print(f"building {name} ...")
        data = BUILDERS[name]()
        path = FIXTURE_DIR / f"{name}.npz"
        np.savez_compressed(path, **data)
        print(f"  wrote {path.name}  ({len(data)} arrays, {path.stat().st_size / 1024:.1f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
