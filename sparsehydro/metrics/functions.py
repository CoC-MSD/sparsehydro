"""Goodness-of-fit metrics as plain functions: arrays in, a number out.

This is the single implementation of every metric in sparsehydro. The
calibration objectives in :mod:`sparsehydro.metrics.objectives`, the plots,
the viewer and the reports all call these functions rather than writing the
formulas again.

All functions take ``observed`` first and ``predicted`` second, convert both to
float arrays, and raise ``ValueError`` when the shapes differ or the metric is
undefined for the input (for example NSE of a constant series), unless a
function documents another behaviour.

Sign conventions
----------------
``pbias`` is signed: positive means the model over-predicts volume. The
``PBIAS`` calibration objective minimises its absolute value.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "mse",
    "rmse",
    "mae",
    "nash_sutcliffe",
    "log_nse",
    "kge",
    "pbias",
    "volume_relative_error",
    "index_of_agreement",
    "concordance_correlation",
    "peak_weighted_mse",
    "top_peak_indices",
    "peak_regression_slope",
    "fit_summary",
]


def _pair(observed, predicted) -> tuple[np.ndarray, np.ndarray]:
    obs = np.asarray(observed, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    if obs.shape != pred.shape:
        raise ValueError(f"Shape mismatch: observed {obs.shape} vs predicted {pred.shape}")
    return obs, pred


def mse(observed, predicted) -> float:
    """Mean squared error."""
    obs, pred = _pair(observed, predicted)
    return float(np.mean((obs - pred) ** 2))


def rmse(observed, predicted) -> float:
    """Root mean squared error."""
    obs, pred = _pair(observed, predicted)
    return float(np.sqrt(np.mean((obs - pred) ** 2)))


def mae(observed, predicted) -> float:
    """Mean absolute error."""
    obs, pred = _pair(observed, predicted)
    return float(np.mean(np.abs(obs - pred)))


def _check_undefined(undefined: str) -> None:
    if undefined not in ("raise", "nan"):
        raise ValueError("undefined must be 'raise' or 'nan'")


def nash_sutcliffe(observed, predicted, *, undefined: str = "raise") -> float:
    """Nash-Sutcliffe efficiency (1 is perfect).

    :param undefined: ``"raise"`` (default) raises ``ValueError`` for a
        constant observed series; ``"nan"`` returns NaN instead, and also for
        an empty series.
    """
    _check_undefined(undefined)
    obs, pred = _pair(observed, predicted)
    if undefined == "nan" and obs.size == 0:
        return math.nan
    ss_res = float(np.sum((obs - pred) ** 2))
    ss_tot = float(np.sum((obs - np.mean(obs)) ** 2))
    if ss_tot == 0.0:
        if undefined == "nan":
            return math.nan
        raise ValueError("SS_tot is zero; observed series is constant — NSE is undefined.")
    return 1.0 - ss_res / ss_tot


def log_nse(observed, predicted, epsilon: float = 0.01) -> float:
    """Nash-Sutcliffe efficiency of ``log(flow + epsilon)`` (predictions clipped at 0).

    :raises ValueError: If ``epsilon <= 0`` or the log-observed series is constant.
    """
    obs, pred = _pair(observed, predicted)
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be > 0; got {epsilon!r}")
    log_obs = np.log(obs + epsilon)
    log_pred = np.log(np.clip(pred, 0.0, None) + epsilon)
    ss_res = float(np.sum((log_obs - log_pred) ** 2))
    ss_tot = float(np.sum((log_obs - np.mean(log_obs)) ** 2))
    if ss_tot == 0.0:
        raise ValueError("SS_tot is zero; log-transformed observed series is constant.")
    return float(1.0 - ss_res / ss_tot)


def kge(observed, predicted) -> float:
    """Kling-Gupta efficiency ``1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)``.

    ``r`` is the Pearson correlation, ``alpha`` the ratio of standard
    deviations and ``beta`` the ratio of means.

    :raises ValueError: If ``std(observed)`` or ``mean(observed)`` is zero.
    """
    obs, pred = _pair(observed, predicted)
    std_obs = float(np.std(obs))
    mean_obs = float(np.mean(obs))
    if std_obs == 0.0:
        raise ValueError("std(observed) == 0; KGE is undefined.")
    if mean_obs == 0.0:
        raise ValueError("mean(observed) == 0; KGE bias ratio is undefined.")
    r = float(np.corrcoef(obs, pred)[0, 1])
    alpha = float(np.std(pred)) / std_obs
    beta = float(np.mean(pred)) / mean_obs
    return float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def pbias(observed, predicted, *, undefined: str = "raise") -> float:
    """Signed percent bias ``100 * sum(predicted - observed) / sum(observed)``.

    Positive values mean the model over-predicts volume.

    :param undefined: ``"raise"`` (default) raises ``ValueError`` when
        ``sum(observed)`` is zero; ``"nan"`` returns NaN instead.
    """
    _check_undefined(undefined)
    obs, pred = _pair(observed, predicted)
    total_obs = float(np.sum(obs))
    if total_obs == 0.0:
        if undefined == "nan":
            return math.nan
        raise ValueError("sum(observed) is zero; PBIAS is undefined.")
    return float(100.0 * np.sum(pred - obs) / total_obs)


def volume_relative_error(observed, predicted) -> float:
    """Absolute relative error of total volume, ``|sum(pred) - sum(obs)| / sum(obs)``.

    :raises ValueError: If ``sum(observed)`` is zero.
    """
    obs, pred = _pair(observed, predicted)
    total_obs = float(np.sum(obs))
    if total_obs == 0.0:
        raise ValueError("sum(observed) is zero; volume relative error is undefined.")
    return float(abs(float(np.sum(pred)) - total_obs) / total_obs)


def index_of_agreement(observed, predicted) -> float:
    """Willmott's index of agreement in [0, 1] (1 is perfect).

    :raises ValueError: If the potential error is zero.
    """
    obs, pred = _pair(observed, predicted)
    mean_obs = float(np.mean(obs))
    ss_res = float(np.sum((obs - pred) ** 2))
    ss_pot = float(np.sum((np.abs(pred - mean_obs) + np.abs(obs - mean_obs)) ** 2))
    if ss_pot == 0.0:
        raise ValueError("Potential error (SS_pot) is zero; Index of Agreement is undefined.")
    return float(1.0 - ss_res / ss_pot)


def concordance_correlation(observed, predicted, *, undefined: str = "raise") -> float:
    """Lin's concordance correlation coefficient in [-1, 1] (1 is perfect).

    :param undefined: ``"raise"`` (default) raises ``ValueError`` when the
        coefficient is undefined (zero denominator); ``"nan"`` returns NaN
        instead, and also for fewer than two points or non-finite values,
        which suits reports.
    """
    _check_undefined(undefined)
    obs, pred = _pair(observed, predicted)
    if undefined == "nan" and (obs.size < 2 or not np.all(np.isfinite(obs))
                               or not np.all(np.isfinite(pred))):
        return math.nan
    mean_obs = float(np.mean(obs))
    mean_pred = float(np.mean(pred))
    var_obs = float(np.var(obs))
    var_pred = float(np.var(pred))
    covariance = float(np.mean((obs - mean_obs) * (pred - mean_pred)))
    denominator = var_obs + var_pred + (mean_obs - mean_pred) ** 2
    if denominator == 0.0:
        if undefined == "nan":
            return math.nan
        raise ValueError("Denominator is zero; CCC is undefined.")
    return float(2.0 * covariance / denominator)


def peak_weighted_mse(observed, predicted, power: float = 1.0) -> float:
    """MSE weighted by ``(observed / mean(observed)) ** power`` (peaks count more).

    :raises ValueError: If ``power < 0`` or the observed series is all zeros.
    """
    obs, pred = _pair(observed, predicted)
    if power < 0.0:
        raise ValueError(f"power must be >= 0; got {power!r}")
    mean_obs = float(np.mean(obs))
    if mean_obs == 0.0:
        raise ValueError("observed is all zeros; peak-weighted MSE is undefined.")
    weights = (obs / mean_obs) ** power
    w_sum = float(np.sum(weights))
    if w_sum == 0.0:
        return 0.0
    return float(np.sum(weights * (obs - pred) ** 2) / w_sum)


def top_peak_indices(observed_peaks, top_fraction: float) -> np.ndarray:
    """Indices of the largest ``top_fraction`` of observed peaks (stable order, at least one).

    :raises ValueError: Unless ``0 < top_fraction <= 1``.
    """
    if not np.isfinite(top_fraction) or not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be finite and in (0, 1]")
    observed_peaks = np.asarray(observed_peaks)
    count = max(1, int(np.ceil(len(observed_peaks) * top_fraction)))
    return np.argsort(-observed_peaks, kind="stable")[:count]


def peak_regression_slope(observed_peaks, modeled_peaks) -> float:
    """Zero-intercept regression slope of modelled on observed peaks.

    Returns NaN when the slope is undefined (no points, non-finite values, or
    all-zero observed peaks).
    """
    x, y = np.asarray(observed_peaks, float), np.asarray(modeled_peaks, float)
    if x.ndim != 1 or x.shape != y.shape:
        raise ValueError("Peak arrays must be aligned and one-dimensional")
    if not len(x) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return float("nan")
    denominator = float(x @ x)
    return float(x @ y / denominator) if denominator > 0 else float("nan")


def fit_summary(observed, predicted) -> dict[str, float]:
    """NSE, signed PBIAS, R-squared and peak error for display.

    Pairs with a non-finite value are dropped first; undefined entries are NaN
    rather than errors.

    :returns: ``{"nse", "pbias", "r2", "peak_error_pct", "n"}``.
    """
    obs, pred = _pair(observed, predicted)
    keep = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[keep], pred[keep]
    out = {"nse": math.nan, "pbias": math.nan, "r2": math.nan,
           "peak_error_pct": math.nan, "n": float(obs.size)}
    if obs.size < 2:
        return out
    out["nse"] = nash_sutcliffe(obs, pred, undefined="nan")
    out["pbias"] = pbias(obs, pred, undefined="nan")
    if np.std(obs) > 0 and np.std(pred) > 0:
        out["r2"] = float(np.corrcoef(obs, pred)[0, 1] ** 2)
    peak = float(np.max(obs))
    if peak != 0:
        out["peak_error_pct"] = float(100.0 * (np.max(pred) - peak) / peak)
    return out
