"""
CalOptimizer — calibration engine for swmmrdii.

Uses sparsehydro's ``CalibrationProblem`` + solver API with Platypus, pymoo
and SciPy. Every evaluation proposes parameters, simulates all subcatchments,
and computes the configured objectives.

    Samplers:  'CMA-ES' (Platypus, default; single objective),
               'NSGAII' (alias for sh_nsga2),
               'scipy' (differential_evolution, default scipy method),
               'sh_nsga2'   (pymoo NSGA-II via sparsehydro),
               'sh_pso'     (SMPSO particle-swarm via platypus),
               'sh_platypus' (generic platypus wrapper — pass algorithm_class
                              kwarg to optimize())
    Budget:    controlled by ``n_trials`` (mapped to maxiter / n_gen / n_evals
               depending on the solver)

All solvers share the objective registry, warm-up handling, BFI R₃ scaling,
network lags, physical feasibility gates and calibration-window slicing.
TPE is not supported.
"""

from __future__ import annotations

import copy
import json
import math
import os
import warnings
from datetime import datetime as _dt
from typing import Optional

import numpy as np
from sparsehydro.calibration.problem import CalibrationProblem
import pandas as pd

from .model import RdiiModel

# Seed-selection rules that run the calibration-window screening.  'objective'
# is the only rule that skips it.  Both the constructor's validation and
# optimize()'s dispatch read these, so adding a rule in one place cannot leave
# the other silently falling back to the legacy path.
_SCREENED_SELECTIONS = ("robust", "balanced")
_ALL_SELECTIONS = _SCREENED_SELECTIONS + ("objective",)


# ---------------------------------------------------------------------------
# Inline metric functions
# (Fast NumPy implementations — avoids sparsehydro class overhead inside
# the calibration hot loop where these are called millions of times.)
# ---------------------------------------------------------------------------

def _fast_ccc(
    obs: np.ndarray,
    pred: np.ndarray,
    obs_mean: float,
    obs_var: float,
) -> float:
    """
    Lin's Concordance Correlation Coefficient.

    CCC = 2 * cov(obs, pred) / (var(obs) + var(pred) + (mean_obs - mean_pred)^2)

    Ranges from -1 to 1; 1 is perfect agreement.  Penalises both correlation
    and location/scale shifts, making it sensitive to volume bias.
    """
    pred_mean = pred.mean()
    pred_var  = np.var(pred)
    cov = np.mean((obs - obs_mean) * (pred - pred_mean))
    return 2.0 * cov / (obs_var + pred_var + (obs_mean - pred_mean) ** 2)


def _pbias(obs: np.ndarray, pred: np.ndarray) -> float:
    """
    Percent bias (dimensionless ratio, NOT multiplied by 100 here).

    Positive → over-prediction, negative → under-prediction.
    """
    return (pred.sum() - obs.sum()) / obs.sum()


def _kge(
    obs: np.ndarray,
    pred: np.ndarray,
    obs_mean: float,
    obs_std: float,
) -> float:
    """
    Kling-Gupta Efficiency (Gupta et al. 2009).

    KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)

    where:
        r     = Pearson correlation coefficient
        alpha = pred_std / obs_std    (variability ratio)
        beta  = pred_mean / obs_mean  (bias ratio)

    Ranges from -inf to 1; 1 is perfect.  Values > 0.5 are generally
    considered satisfactory for daily streamflow.
    """
    pred_mean = pred.mean()
    pred_std  = pred.std()
    cov = np.mean((obs - obs_mean) * (pred - pred_mean))
    r = cov / (obs_std * pred_std) if (obs_std > 0 and pred_std > 0) else 0.0
    alpha = pred_std / obs_std if obs_std > 0 else 0.0
    beta  = pred_mean / obs_mean if obs_mean != 0 else 0.0
    return 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)


def _nse(obs: np.ndarray, pred: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency.  1 = perfect, 0 = as good as mean flow."""
    denom = np.sum((obs - obs.mean()) ** 2)
    return float(1.0 - np.sum((obs - pred) ** 2) / denom) if denom > 0 else float("nan")


def _log_nse(obs: np.ndarray, pred: np.ndarray, epsilon: float = 0.01) -> float:
    """NSE on log-transformed flows.  Epsilon avoids log(0)."""
    log_obs  = np.log(obs  + epsilon)
    log_pred = np.log(pred + epsilon)
    denom = np.sum((log_obs - log_obs.mean()) ** 2)
    return float(1.0 - np.sum((log_obs - log_pred) ** 2) / denom) if denom > 0 else float("nan")


def _rmse(obs: np.ndarray, pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((obs - pred) ** 2)))


def _mae(obs: np.ndarray, pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(obs - pred)))


def _mse(obs: np.ndarray, pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean((obs - pred) ** 2))


def _peak_weighted_mse(obs: np.ndarray, pred: np.ndarray, power: float = 1.0) -> float:
    """Flow-weighted MSE (high flows penalised more).  power=1 = linear weighting."""
    weights = np.power(obs, power)
    return float(np.sum(weights * (obs - pred) ** 2) / np.sum(weights)) if weights.sum() > 0 else float("nan")


def _volume_relative_error(obs: np.ndarray, pred: np.ndarray) -> float:
    """Absolute relative total-volume error (0 = perfect, 1 = 100% error)."""
    return float(abs(pred.sum() - obs.sum()) / obs.sum()) if obs.sum() != 0 else float("nan")


def _index_of_agreement(obs: np.ndarray, pred: np.ndarray) -> float:
    """
    Willmott's Index of Agreement.

    d = 1 - sum((obs-pred)^2) / sum((|pred-obs_mean| + |obs-obs_mean|)^2)

    Ranges 0–1; 1 = perfect.
    """
    obs_mean = obs.mean()
    num   = np.sum((obs - pred) ** 2)
    denom = np.sum((np.abs(pred - obs_mean) + np.abs(obs - obs_mean)) ** 2)
    return float(1.0 - num / denom) if denom > 0 else float("nan")


# ---------------------------------------------------------------------------
# Objective registry
# ---------------------------------------------------------------------------
# Each entry: key → (compute_fn, minimize, short_description)
#   compute_fn(obs, pred, obs_mean, obs_std, obs_var) → float
#   minimize=True  means calibration direction='minimize'
#   minimize=False means calibration direction='maximize'
#
# Keys match the sparsehydro CalibrationProblem objective name strings
# (lowercase).  Aliases (e.g. 'nse') are also registered for convenience.
#
# All compute_fn signatures accept (obs, pred, obs_mean, obs_std, obs_var);
# functions that don't use all five arguments simply ignore extras.

def _obj_ccc(obs, pred, obs_mean, obs_std, obs_var):
    return _fast_ccc(obs, pred, obs_mean, obs_var)

def _obj_kge(obs, pred, obs_mean, obs_std, obs_var):
    return _kge(obs, pred, obs_mean, obs_std)

def _obj_nse(obs, pred, obs_mean, obs_std, obs_var):
    return _nse(obs, pred)

def _obj_log_nse(obs, pred, obs_mean, obs_std, obs_var):
    return _log_nse(obs, pred)

def _obj_pbias(obs, pred, obs_mean, obs_std, obs_var):
    return abs(_pbias(obs, pred))  # stored as fraction; minimise absolute value

def _obj_volume_relative_error(obs, pred, obs_mean, obs_std, obs_var):
    return _volume_relative_error(obs, pred)

def _obj_rmse(obs, pred, obs_mean, obs_std, obs_var):
    return _rmse(obs, pred)

def _obj_mae(obs, pred, obs_mean, obs_std, obs_var):
    return _mae(obs, pred)

def _obj_mse(obs, pred, obs_mean, obs_std, obs_var):
    return _mse(obs, pred)

def _obj_peak_weighted_mse(obs, pred, obs_mean, obs_std, obs_var):
    return _peak_weighted_mse(obs, pred)

def _obj_index_of_agreement(obs, pred, obs_mean, obs_std, obs_var):
    return _index_of_agreement(obs, pred)


# Registry: lowercase key → (fn, minimize, display_name, description)
OBJECTIVE_REGISTRY: dict[str, tuple] = {
    "ccc": (
        _obj_ccc, False, "CCC",
        "Lin's Concordance Correlation Coefficient. Penalises correlation AND "
        "location/scale shift. Can leave a small volume bias. maximize.",
    ),
    "kge": (
        _obj_kge, False, "KGE",
        "Kling-Gupta Efficiency (Gupta et al. 2009). Balances r, variability "
        "ratio, and bias ratio. Recommended default. maximize.",
    ),
    "nash_sutcliffe": (
        _obj_nse, False, "NSE",
        "Nash-Sutcliffe Efficiency. 1=perfect, 0=mean-flow baseline. "
        "Sensitive to high-flow peaks. maximize.",
    ),
    "log_nse": (
        _obj_log_nse, False, "LogNSE",
        "NSE on log-transformed flows. Emphasises low-flow fit. maximize.",
    ),
    "index_of_agreement": (
        _obj_index_of_agreement, False, "IoA",
        "Willmott's Index of Agreement. 1=perfect, 0=no skill. maximize.",
    ),
    "pbias": (
        _obj_pbias, True, "|PBIAS|",
        "Absolute percent bias (volume balance error as fraction). "
        "0=perfect, higher=worse. minimize.",
    ),
    "volume_relative_error": (
        _obj_volume_relative_error, True, "VRE",
        "Absolute relative total-volume error. 0=perfect. minimize.",
    ),
    "rmse": (
        _obj_rmse, True, "RMSE",
        "Root Mean Squared Error. Same units as flow (CFS). minimize.",
    ),
    "mae": (
        _obj_mae, True, "MAE",
        "Mean Absolute Error. Same units as flow (CFS). minimize.",
    ),
    "mse": (
        _obj_mse, True, "MSE",
        "Mean Squared Error. Penalises large errors more than MAE. minimize.",
    ),
    "peak_weighted_mse": (
        _obj_peak_weighted_mse, True, "PeakMSE",
        "Flow-weighted MSE — high flows penalised more. minimize.",
    ),
}

# Convenience aliases so users don't need to remember the exact sparsehydro key.
OBJECTIVE_REGISTRY["nse"]                          = OBJECTIVE_REGISTRY["nash_sutcliffe"]
OBJECTIVE_REGISTRY["concordance_correlation_coefficient"] = OBJECTIVE_REGISTRY["ccc"]
OBJECTIVE_REGISTRY["kling_gupta_efficiency"]       = OBJECTIVE_REGISTRY["kge"]


def _resolve_objectives(raw) -> list[str]:
    """
    Normalise the ``objective`` constructor argument to a list of registry keys.

    Accepts:
        - A single string: ``'KGE'``, ``'ccc'``, etc.
        - A list of strings: ``['KGE', 'pbias']``, ``['CCC', 'NSE', 'pbias']``
        - The legacy combined string: ``'CCC+PBIAS'``  (split on '+')

    All keys are lowercased and validated against OBJECTIVE_REGISTRY.
    """
    if isinstance(raw, str):
        parts = [p.strip().lower() for p in raw.split("+")]
    else:
        parts = [p.strip().lower() for p in raw]

    unknown = [p for p in parts if p not in OBJECTIVE_REGISTRY]
    if unknown:
        valid = ", ".join(f"'{k}'" for k in sorted(OBJECTIVE_REGISTRY)
                         if not k.startswith(("concordance", "kling")))
        raise ValueError(
            f"Unknown objective(s): {unknown}. "
            f"Valid keys: {valid}"
        )
    return parts


# ---------------------------------------------------------------------------
# Sampler registry
# ---------------------------------------------------------------------------
# Each entry: key → (backend, display_name, single_ok, multi_ok, description)
#   backend    : 'calibration' or 'sparsehydro'
#   single_ok  : works for single-objective runs
#   multi_ok   : works for multi-objective Pareto runs

SAMPLER_REGISTRY: dict[str, tuple] = {
    "cma-es": (
        "sparsehydro", "CMA-ES", True, False,
        "Single-objective CMA-ES via Platypus. n_trials is evaluations per seed.",
    ),
    # ── sparsehydro backend ──────────────────────────────────────────────────
    "scipy": (
        "sparsehydro", "SciPy-DE", True, False,
        "Differential Evolution via scipy (sparsehydro backend). "
        "Global stochastic single-objective search. n_trials -> maxiter.",
    ),
    "sh_nsga2": (
        "sparsehydro", "SH-NSGA2", True, True,
        "NSGA-II via pymoo (sparsehydro backend). "
        "Multi-objective Pareto. n_trials -> n_gen (pop_size fixed at n_startup).",
    ),
    "sh_pso": (
        "sparsehydro", "SH-PSO", True, True,
        "SMPSO Particle Swarm via platypus (sparsehydro backend). "
        "Multi-objective capable. n_trials -> n_evaluations.",
    ),
    "sh_platypus": (
        "sparsehydro", "SH-Platypus", True, True,
        "Generic platypus algorithm wrapper (sparsehydro backend). "
        "Pass algorithm_class=<platypus.Algorithm subclass> to optimize(). "
        "n_trials -> n_evaluations.",
    ),
}

# Normalise alias keys (remove hyphens, lowercase).
def _resolve_sampler(key: str) -> str:
    """Return the canonical SAMPLER_REGISTRY key for *key*, or raise ValueError."""
    k = key.strip().lower()
    # Explicit alias table — maps common variants to canonical registry keys.
    aliases = {
        "nsgaii":                 "sh_nsga2",
        "nsga-ii":                "sh_nsga2",
        "nsga2":                  "sh_nsga2",
        "nsga_ii":                "sh_nsga2",
        "cmaes":                  "cma-es",
        "cma_es":                 "cma-es",
        "differential_evolution": "scipy",
        "de":                     "scipy",
        "pso":                    "sh_pso",
        "smpso":                  "sh_pso",
        "sh-nsga2":               "sh_nsga2",
        "sh-pso":                 "sh_pso",
        "sh-platypus":            "sh_platypus",
    }
    k = aliases.get(k, k)
    if k not in SAMPLER_REGISTRY:
        valid = ", ".join(f"'{s}'" for s in sorted(SAMPLER_REGISTRY))
        raise ValueError(
            f"Unknown sampler {key!r}. Valid keys: {valid}"
        )
    return k


# ---------------------------------------------------------------------------
# CalOptimizer
# ---------------------------------------------------------------------------


class _RdiiCalibrationProblem(CalibrationProblem):
    """Isolated multi-subcatchment evaluation, including logarithmic bounds."""

    def __init__(self, owner, forcing):
        self.owner = owner
        self.forcing = forcing
        self._param_names = [name for name, *_ in owner.model._param_meta]
        self._log = np.array([log for _, _, _, log in owner.model._param_meta], dtype=bool)
        bounds = np.array([(lo, hi) for _, lo, hi, _ in owner.model._param_meta], dtype=float)
        if (not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1])
                or np.any(bounds[self._log] <= 0)):
            raise ValueError("Calibration requires finite ordered bounds, positive for log parameters")
        bounds[self._log] = np.log(bounds[self._log])
        self._xl, self._xu = bounds.T
        self._n_ieq_constr = 0
        self._constraint_names = []
        self._constraint_descriptions = []

    @property
    def n_objectives(self):
        return len(self.owner._obj_keys)

    @property
    def objective_names(self):
        return list(self.owner._obj_keys)

    @property
    def minimize_flags(self):
        return [OBJECTIVE_REGISTRY[key][1] for key in self.owner._obj_keys]

    def to_physical(self, x):
        values = np.array(x, dtype=float, copy=True)
        values[..., self._log] = np.exp(values[..., self._log])
        return values

    def to_search(self, params):
        values = np.array([params[name] for name in self.param_names], dtype=float)
        if np.any(values[self._log] <= 0):
            raise ValueError("Warm-start log parameters must be positive")
        values[self._log] = np.log(values[self._log])
        return values

    def evaluate(self, x, penalty_weight=1e6):
        owner = self.owner
        params = dict(zip(self.param_names, self.to_physical(x)))
        if sum(params[name] for name in owner._r_sorted) > 1:
            return np.full(self.n_objectives, 1e12)
        prediction = owner._simulate_cal(params, self.forcing)
        if prediction is None:
            return np.full(self.n_objectives, 1e12)
        scores = []
        for key in owner._obj_keys:
            fn, minimize, *_ = OBJECTIVE_REGISTRY[key]
            value = fn(owner._obs_cal, prediction, owner._obs_cal_mean,
                       owner._obs_cal_std, owner._obs_cal_var)
            if not np.isfinite(value):
                raise ValueError(f"Non-finite calibration objective: {key}")
            scores.append(value if minimize else -value)
        return np.asarray(scores, dtype=float)

    def evaluate_single(self, x, index, penalty_weight=1e6):
        return float(self.evaluate(x, penalty_weight)[index])

    def make_copy(self):
        return copy.deepcopy(self)


class CalOptimizer:
    """
    Solver-agnostic optimizer for an ``RdiiModel``.

    Parameters
    ----------
    model : RdiiModel
        A model with forcing and observed data already loaded.
    objective : str or list of str
        One or more objective keys.  A single string runs single-objective
        optimization; a list (or a ``'+'``-separated string) runs
        multi-objective Pareto optimization and forces the NSGA-II sampler.

        Single-objective examples::

            'KGE'              # recommended default
            'CCC'
            'NSE'
            'pbias'            # minimize absolute percent bias

        Multi-objective examples::

            ['KGE', 'pbias']           # maximize KGE + minimize |PBIAS|
            'CCC+PBIAS'                # legacy syntax, still accepted
            ['CCC', 'NSE', 'pbias']    # three-objective Pareto

        See ``CalOptimizer.list_objectives()`` for all available keys and
        their minimize/maximize directions.
    sampler : str
        Solver name: ``'CMA-ES'`` (Platypus, default), ``'NSGAII'``
        (existing pymoo implementation), or another registry entry.
        CMA-ES requires a single objective; TPE is no longer supported.
    n_trials : int
        Total number of calibration trials.
    n_startup : int or None
        Population size: CMA-ES offspring size (default 10); otherwise the
        existing population/swarm setting. CMA-ES has no startup phase.
    pbias_target : float
        In multi-objective mode, the maximum acceptable |PBIAS| (as a
        fraction, e.g. ``0.05`` = 5%) when selecting the Pareto-front knee
        trial.  The knee is the Pareto-optimal trial with the best primary
        maximize metric that also satisfies this constraint.
    seed_initial : bool
        If True, enqueue the RTKInitializer default parameter vector as the
        first candidate in every CMA-ES seed; it counts against the evaluation budget.
    multi_start : int
        Number of independent CMA-ES runs (different seeds), keeping the
        global best.  Default 1 (single run, backward-compatible).  CMA-ES is
        basin-sensitive: with a single seed a hairline change in the
        area-scaled start point can trap the search in a worse local optimum
        even though every run is internally reproducible.  Setting e.g.
        ``multi_start=8`` runs seeds 42..49 and keeps the best, making the
        result robust to that fragility without relying on a lucky start.
    multi_start_seeds : list[int] or None
        Explicit seed list overriding the default ``42, 43, ...`` sequence.
        When given, ``multi_start`` is ignored and ``len(multi_start_seeds)``
        runs are performed.
    warmup_buffer : int
        Days of warm-up prepended before the calibration/validation scoring
        window.  The model is run over [window_start − warmup_buffer, window_end]
        but only scored over [window_start, window_end].  Allows the IA
        storage to spin up from zero.  Call ``set_warmup_buffer()`` to change
        this after construction.
    progress_every : int
        Print a progress line every this many trials (default 100).
        Sequential CMA-ES prints evaluation progress; parallel runs report seed completion.
    ia_max_ub : float or None
        Forwarded to ``RdiiModel._build_templates()``.  Superseded by
        ``param_bound_overrides={'ia_max': (None, x)}``; kept for
        backward compatibility with existing notebooks.
    uh1_t_ub : float or None
        Forwarded to ``RdiiModel._build_templates()``.  Superseded by
        ``param_bound_overrides={'uh1_T': (None, x)}``; kept for
        backward compatibility with existing notebooks.
    param_bound_overrides : dict or None
        Per-parameter search-bound overrides applied after the templates are
        built, e.g. ``{'R_3': (0.10, None)}`` to raise R_3's lower bound.
        Keys match a calibratable parameter by exact name or by suffix
        (``'R_3'`` matches ``m01_R_3``); values are ``(lower, upper)`` tuples
        where ``None`` keeps the existing bound.  Overrides update both the
        calibration search space and the template parameters (initial values are
        clamped into the new bounds so the ``seed_initial`` trial stays
        valid).  A key that matches no parameter raises ``ValueError`` at
        ``optimize()`` start.  Note R_3 is the watershed-level value — with
        BFI scaling active, per-subcatchment R_3 is this value scaled by
        BFI/mean-BFI, so individual subcatchments may still fall outside the
        override range.
    ia_kt : float
        Fixed (or seed) value for the IA temperature coefficient.
    ia_t_ref : float
        Fixed (or seed) value for the IA reference temperature (°C).
    ia_t_freeze : float
        Fixed (or seed) value for the IA freeze threshold (°C).
    calibrate_temperature_params : bool
        Whether to include IA temperature parameters in the search space.
    seed_selection : str
        How the winning seed is chosen when multi-start runs several
        (ignored for a single seed):

        * ``'balanced'`` (default) — screen every seed on the CALIBRATION
          window, then rank survivors by
          ``mean(KGE, CCC) - seasonal_weight * |worst-season PBIAS| / 100``.
          A seed must first pass two gates: overall
          ``|PBIAS| <= pbias_max`` and worst-season
          ``|PBIAS| <= seasonal_pbias_max``.  Trading fit against seasonal
          balance continuously (rather than treating the seasonal gate as
          pass/fail and then maximising fit) means a seed that fits
          marginally worse but spreads its error evenly across seasons can
          win — and even seasonal error is what keeps volume from drifting
          outside the calibration window.
        * ``'robust'`` — same two gates, but survivors rank on
          ``mean(KGE, CCC)`` alone, so seasonal balance only ever acts as a
          threshold.  Use it when you want the best-fitting seed among those
          that are merely acceptable on seasonal balance.
        * ``'objective'`` — legacy behaviour: take whichever seed scored
          best on the search objective, with no screening at all.  Prone to
          selecting an over-fitted seed, since nothing rewards
          generalisation.  Use it to reproduce a pre-screening calibration.

        The seasonal gate is what makes any of this work: an aggregate PBIAS
        near zero is often a winter deficit cancelling a spring surplus,
        which looks like a volume-balanced model during calibration and then
        drifts once the seasons stop cancelling.

        Both scores are printed for every seed whichever rule is active, and
        the note says when the other ranking would have chosen differently.

        Every rule uses the calibration window ONLY.  The validation window
        is never consulted, so it remains a blind holdout and validation
        metrics stay honest.
    seed_override : int or None
        Force a specific seed instead of whichever the rule would choose
        (default None = let the rule decide).  Must be one of the seeds
        actually being run, or ``ValueError`` is raised at construction.

        Use this after reading the screening table, when the automatic
        ranking picks a seed you disagree with — for example when the
        top-ranked seed wins on ``mean(KGE, CCC)`` by a hair but a
        runner-up has markedly better seasonal balance, which is the
        property that governs whether volume holds up outside the
        calibration window.

        The screening table still prints in full, the note records that an
        override was applied and which seed the rule would otherwise have
        chosen, and a seed that fails the gates is honoured but flagged.
        So the override is documented, never silent.
    seasonal_weight : float
        Penalty weight on worst-season ``|PBIAS|`` in the ``'balanced'``
        score (default 1.0).  Both terms are fractions of 1.0, so at 1.0 a
        10-point worst-season PBIAS costs 0.10 of ``mean(KGE, CCC)``.
        Raise it to prioritise seasonal balance more strongly, lower it
        toward 0 to approach plain ``'robust'`` ranking.  Only affects
        ranking, never the gates.
    pbias_max : float
        Overall ``|PBIAS|`` gate for ``seed_selection='robust'``, in percent
        (default 10.0).
    seasonal_pbias_max : float
        Worst-season ``|PBIAS|`` gate for ``seed_selection='robust'``, in
        percent (default 15.0).  Seasons are meteorological DJF/MAM/JJA/SON.
    bound_warn_pct : float
        Report a parameter as pinned when it lands within this percentage of
        its search range from either bound (default 5.0).  Pinned parameters
        are printed as a WARNING and recorded, but never gate selection — a
        parameter at a bound can be physically genuine, so this informs you
        rather than vetoing the seed.
    """

    def __init__(
        self,
        model: RdiiModel,
        objective = "KGE",
        sampler: str = "CMA-ES",
        n_trials: int = 10_000,
        n_startup: Optional[int] = None,
        pbias_target: float = 0.05,
        seed_initial: bool = True,
        multi_start: int = 1,
        multi_start_seeds: Optional[list] = None,
        warmup_buffer: int = 365,
        progress_every: int = 100,
        ia_max_ub: Optional[float] = None,
        uh1_t_ub: Optional[float] = None,
        param_bound_overrides: Optional[dict] = None,
        ia_kt: float = 0.05,
        ia_t_ref: float = 20.0,
        ia_t_freeze: float = 0.0,
        calibrate_temperature_params: bool = False,
        imperv_field: Optional[str] = None,
        bfi_field: Optional[str] = None,
        obs_scale: float = 1.0,
        seed_selection: str = "balanced",
        seed_override: Optional[int] = None,
        seasonal_weight: float = 1.0,
        pbias_max: float = 10.0,
        seasonal_pbias_max: float = 15.0,
        bound_warn_pct: float = 5.0,
    ) -> None:
        self.model = model

        # Resolve objective argument → list of registry keys.
        self._obj_keys: list[str] = _resolve_objectives(objective)
        self._is_multi: bool = len(self._obj_keys) > 1

        # Human-readable label for summary/file naming (e.g. 'KGE' or 'CCC+PBIAS').
        self.objective_mode: str = "+".join(
            OBJECTIVE_REGISTRY[k][2] for k in self._obj_keys
        )

        # Resolve sampler key → canonical name and backend.
        self._sampler_key: str = _resolve_sampler(sampler)
        entry = SAMPLER_REGISTRY[self._sampler_key]
        self.sampler_type: str = entry[1]          # display name, e.g. 'CMA-ES'
        self._backend: str     = entry[0]

        self.n_trials = n_trials
        self.n_startup = int(n_startup if n_startup is not None else
                             (10 if self._sampler_key == "cma-es" else max(50, n_trials * 0.1)))
        self.pbias_target = pbias_target
        self.seed_initial = seed_initial
        # Multi-start ensemble: run several CMA-ES studies from different seeds
        # and keep the global best.  Robust-by-construction against the
        # local-basin fragility.  multi_start is
        # the number of seeds; multi_start_seeds overrides the default
        # 42, 43, ... sequence with an explicit list.
        if multi_start_seeds is not None:
            self._seeds = [int(s) for s in multi_start_seeds]
        else:
            self._seeds = [42 + i for i in range(max(1, int(multi_start)))]
        self.warmup_buffer = warmup_buffer
        self.progress_every = progress_every
        self._ia_max_ub = ia_max_ub
        self._uh1_t_ub = uh1_t_ub
        # Per-parameter search-bound overrides, validated for shape here and
        # applied to the built templates at optimize() start (name resolution
        # needs _param_meta, which does not exist until templates are built).
        self._param_bound_overrides: dict[str, tuple] = {}
        for key, bounds in (param_bound_overrides or {}).items():
            try:
                new_lb, new_ub = bounds
            except (TypeError, ValueError):
                raise ValueError(
                    f"param_bound_overrides['{key}'] must be a (lower, upper) "
                    f"tuple (use None to keep a bound), got {bounds!r}"
                ) from None
            if new_lb is not None and new_ub is not None and float(new_lb) > float(new_ub):
                raise ValueError(
                    f"param_bound_overrides['{key}']: lower ({new_lb}) > upper ({new_ub})"
                )
            self._param_bound_overrides[str(key)] = (new_lb, new_ub)
        self._ia_kt = ia_kt
        self._ia_t_ref = ia_t_ref
        self._ia_t_freeze = ia_t_freeze
        self._calibrate_temp = calibrate_temperature_params

        # Per-subcatchment imperviousness scaling of ia_max (linear in pervious fraction).
        # imperv_field : column in model._sc_df holding %Imperv (None = off).
        self._imperv_field = imperv_field
        # Resolved sc_df column for imperv_field (PCSWMM DB user-fields are
        # merged under sanitised lower-case names, e.g. 'Imperv' -> 'imperv').
        # Set in _prepare_objective_data(); raises there if the declared
        # field cannot be found.
        self._imperv_col: Optional[str] = None
        # ---- Multi-start seed selection -----------------------------------
        # 'robust' (default) screens every seed on the CALIBRATION window only
        # — overall PBIAS gate + worst-season PBIAS gate, survivors ranked by
        # mean(KGE, CCC).  The validation window is never consulted, so it
        # stays a blind holdout.  'objective' restores the legacy behaviour of
        # taking whichever seed scored best on the search objective, which
        # tends to pick an over-fitted seed.
        _sel = str(seed_selection).lower()
        if _sel not in _ALL_SELECTIONS:
            raise ValueError(
                f"seed_selection must be one of "
                f"{', '.join(repr(s) for s in _ALL_SELECTIONS)}, got "
                f"{seed_selection!r}."
            )
        self.seed_selection = _sel
        # Manual override: force a specific seed regardless of the rule.  The
        # screening table still prints, so the choice stays auditable and the
        # override is recorded rather than looking like an automatic pick.
        if seed_override is not None:
            seed_override = int(seed_override)
            if seed_override not in self._seeds:
                raise ValueError(
                    f"seed_override={seed_override} is not one of the seeds "
                    f"being run: {self._seeds}. Set multi_start_seeds to "
                    f"include it, or pick from that list."
                )
        self.seed_override = seed_override
        self.seasonal_weight = float(seasonal_weight)
        self.pbias_max = float(pbias_max)
        self.seasonal_pbias_max = float(seasonal_pbias_max)
        self.bound_warn_pct = float(bound_warn_pct)
        # Per-seed screening table, filled by optimize() when >1 seed is run.
        self.seed_screen: list[dict] = []
        self.seed_selection_note: str = ""
        self.best_seed: Optional[int] = None

        # BFI field name used at calibration time (None = flat bfi=1.0 default).
        self._bfi_field = bfi_field
        # Observed series scale factor: calibration target = observed * obs_scale.
        # Values > 1.0 inflate the target to compensate for SWMM routing attenuation.
        self._obs_scale: float = float(obs_scale)

        # Calibration / validation window strings — set by set_windows().
        self._cal_start: Optional[str] = None
        self._cal_end:   Optional[str] = None
        self._val_start: Optional[str] = None
        self._val_end:   Optional[str] = None

        # Set by optimize() — available afterwards for plotting and export.
        # self.study holds a solver-agnostic CalibrationResult.
        # Both support self.best_params and calibration metrics.
        self.study = None
        self.best_params: Optional[dict[str, float]] = None
        self.best_trial = None   # retained for summary compatibility

        # Calibration metrics populated by optimize().
        self.cal_ccc:   float = float("nan")
        self.cal_kge:   float = float("nan")
        self.cal_nse:   float = float("nan")
        self.cal_pbias: float = float("nan")

        # Persistent prepared models + cal-window masks (built in optimize()
        # when n_jobs == 1) to avoid per-trial deepcopy + prepare.
        self._prepared_models = None
        self._cal_masks = None

        # Pre-computed arrays for use inside the objective function.
        # Populated in _prepare_objective_data().
        self._obs_cal: Optional[np.ndarray] = None
        self._obs_cal_mean: float = 0.0
        self._obs_cal_var:  float = 0.0
        self._obs_cal_std:  float = 0.0

        # BFI R_3 multiplier per subcatchment (computed once, reused per trial).
        self._r3_multiplier: dict[str, float] = {}
        # Area-weighted mean BFI at calibration time (for future-condition rescaling).
        self._mean_bfi: Optional[float] = None

        # Per-subcatchment imperviousness multiplier on ia_max (computed once).
        self._imp_multiplier: dict[str, float] = {}
        # Area-weighted mean pervious fraction at calibration time (for future-condition rescaling).
        self._mean_perv: Optional[float] = None

        # Per-SC network lag (junction → gage travel time, in days).  Set via
        # set_network_lags(); empty dict = no lag correction.
        self._network_lags: dict[str, float] = {}

        # Upstream boundary-condition inflows for downstream multi-gage
        # calibration.  A list of components, each {"series", "label",
        # "lag_days"}: set ONE via load_bc_inflow()/set_bc_inflow() (replace
        # semantics) or several via add_bc_inflow(); empty list = no BC.
        # _bc_cal is the calibration-window-aligned SUM of all components,
        # built once in _prepare_objective_data() and added to every trial's
        # summed prediction.  Never scaled by obs_scale.  Each component is
        # forward-shifted by its OWN lag_days (entry node -> gage travel
        # time) — see DESIGN_downstream_bc_calibration.md.  Components must
        # be genuinely pass-through inflows (upstream gage flow, point
        # sources, transfers) — NOT storage-mediated series like reservoir
        # net precip/evap, which do not translate to the gage as a lag.
        self._bc_components: list[dict] = []
        self._bc_cal: Optional[np.ndarray] = None

        # Sorted names of R parameters from the reference model's param_meta.
        self._r_sorted: list[str] = []

    # -----------------------------------------------------------------------
    # Window configuration
    # -----------------------------------------------------------------------

    def set_windows(
        self,
        cal_start: str,
        cal_end: str,
        val_start: str,
        val_end: str,
    ) -> None:
        """
        Set calibration and validation date windows.

        Parameters
        ----------
        cal_start, cal_end : str
            Inclusive ISO-format dates (``'YYYY-MM-DD'``) for the calibration
            scoring window.
        val_start, val_end : str
            Inclusive ISO-format dates for the validation window.
        """
        self._cal_start = cal_start
        self._cal_end   = cal_end
        self._val_start = val_start
        self._val_end   = val_end

    def set_warmup_buffer(self, days: int) -> None:
        """
        Set the warm-up buffer length in days.

        The warm-up buffer is prepended *before* the calibration and validation
        scoring windows so that the IA (initial abstraction) soil storage can
        spin up from its zero initial condition.  Without a buffer the first
        few months of each window will have an artificially high runoff ratio
        because the IA reservoir starts empty.

        The model is **run** over ``[window_start - days, window_end]`` but
        only **scored** over ``[window_start, window_end]``.

        Parameters
        ----------
        days : int
            Warm-up length in days.  365 (one year) is the recommended default
            for watersheds with significant inter-annual IA variability.
            Use a smaller value only if forcing data is too short to support a
            full year of spin-up.

        Example
        -------
        >>> cal = CalOptimizer(model=m, objective='KGE', n_trials=5_000)
        >>> cal.set_windows('2008-01-01', '2015-12-31', '2016-01-01', '2022-12-31')
        >>> cal.set_warmup_buffer(365)
        >>> cal.optimize()
        """
        self.warmup_buffer = int(days)

    def set_network_lags(self, lags: dict[str, float]) -> None:
        """
        Set per-SC network travel-time lags for use during calibration scoring.

        Each SC's predicted RDII is shifted forward in time by ``round(lag)``
        days before being summed with other SCs and compared to the observed
        series at the calibration gage.  This accounts for the travel time
        from the SC's SWMM junction to the gage node through the channel
        network.

        Lags are typically pre-computed by ``compute_network_lags()`` in
        ``swmm_runner.py``, which runs a parallel SWMM pulse test for each
        unique SC outlet junction.

        Parameters
        ----------
        lags : dict
            Mapping of subcatchment name → lag in days (float).  Subcatchments
            not listed default to 0.0 (no shift).

        Notes
        -----
        - Lags are rounded to the nearest integer number of days before
          application (sub-day precision is meaningless at daily timestep).
        - Setting lags after ``optimize()`` has already run does NOT
          retroactively change the stored results; call ``optimize()`` again
          to re-calibrate with the new lags.
        - To disable lag correction, call ``set_network_lags({})``.
        """
        self._network_lags = {k: float(v) for k, v in lags.items()}

    # -----------------------------------------------------------------------
    # Upstream boundary-condition inflow (downstream multi-gage calibration)
    # -----------------------------------------------------------------------

    def load_bc_inflow(
        self,
        tsf_path: str,
        series_id: str,
        bc_lag_days: float = 0.0,
    ) -> None:
        """
        Load an upstream boundary-condition inflow from a PCSWMM/SWMM TSF file.

        For downstream multi-gage calibration: when part of the basin upstream
        of the calibration gage is already calibrated, its contribution enters
        this calibration as a fixed inflow series added to the summed
        sparseHydro prediction each trial.  The observed target stays the
        *full* downstream gage signal, so only the local (not-yet-calibrated)
        subcatchments — selected via ``RdiiModel.from_swmm(where=...)`` — are
        fitted to the remainder.

        The BC series is never scaled by ``obs_scale``.  It IS forward-shifted
        by ``bc_lag_days`` (rounded to the nearest day, same convention as
        ``set_network_lags()``) when non-zero — this is the travel time from
        the node where the BC enters the model (``BC_NODE_ID``) to the
        calibration gage.  A BC injected far upstream needs this; a BC that
        happens to enter at the gage node itself naturally has lag 0 (there is
        no path to travel). Compute ``bc_lag_days`` the same way as per-SC
        network lags, e.g. via ``compute_network_lags()`` treating
        ``BC_NODE_ID`` as an outlet junction, or by tracing the path yourself.
        Include any warm-up period you need when generating the tsf — if the
        series does not cover the calibration (and validation) windows,
        ``optimize()`` raises a ``ValueError`` up front.

        The final production run needs no BC: place the upstream and local
        PKL bundles in one directory and ``run_sparsehydro()`` injects them
        all.  See ``DESIGN_downstream_bc_calibration.md``.

        Parameters
        ----------
        tsf_path : str
            Path to the .tsf file containing the BC inflow series.
        series_id : str
            Identifier string that appears in the TSF block's ``IDs:`` header
            line (e.g. an upstream gage number or a node name).
        bc_lag_days : float
            Travel time in days from the BC injection node to the calibration
            gage. Rounded to the nearest integer day before being applied.
            Default 0.0 (no shift).

        Notes
        -----
        REPLACE semantics: any previously set BC component(s) are cleared
        first.  To combine several BC inflows (each with its own entry node
        and lag), call ``add_bc_inflow()`` once per component instead.
        """
        self._bc_components = []
        self.add_bc_inflow(tsf_path, series_id, bc_lag_days=bc_lag_days)

    def set_bc_inflow(
        self,
        series: Optional[pd.Series],
        label: str = "",
        bc_lag_days: float = 0.0,
    ) -> None:
        """
        Set the upstream BC inflow directly from a pandas Series.

        Twin of ``load_bc_inflow()`` for when the series comes from a source
        other than a TSF file.  Pass ``None`` to clear ALL previously set BC
        components.  REPLACE semantics — use ``add_bc_inflow()`` to combine
        several BC inflows.

        Parameters
        ----------
        series : pd.Series or None
            Flow in the same unit as the observed series (cfs), indexed by
            datetime.  Must cover the calibration and validation windows.
        label : str
            Human-readable identifier used in ``summary()`` and
            ``save_params()`` metadata.
        bc_lag_days : float
            Travel time in days from the BC injection node (``BC_NODE_ID``) to
            the calibration gage.  Rounded to the nearest integer day before
            being applied.  Default 0.0 (no shift).
        """
        self._bc_components = []
        if series is None:
            return
        self.add_bc_inflow(series, label=label, bc_lag_days=bc_lag_days)

    def add_bc_inflow(
        self,
        source,
        series_id: Optional[str] = None,
        label: str = "",
        bc_lag_days: float = 0.0,
    ) -> None:
        """
        APPEND one BC inflow component (accumulating semantics).

        Call once per upstream/external inflow entering the calibrated area;
        every component is independently lag-shifted by its own
        ``bc_lag_days`` and the shifted components are summed into the fixed
        baseline added to each trial's prediction.  Components must be
        genuinely pass-through inflows (upstream calibrated-gage flow, point
        sources, interbasin transfers) — a storage-mediated series such as a
        reservoir's net precip/evap does NOT reach the gage as a pure time
        shift and must not be added here.

        Parameters
        ----------
        source : str or pd.Series
            Path to a .tsf file (requires *series_id*) or a flow Series in
            the observed unit (cfs), indexed by datetime.
        series_id : str or None
            TSF block identifier — required when *source* is a path,
            ignored for a Series.
        label : str
            Identifier for ``summary()``/``save_params()``; defaults to
            ``"<basename> [<series_id>]"`` for a TSF source.
        bc_lag_days : float
            Travel time in days from this component's entry node to the
            calibration gage (e.g. from ``bc_node_lag()``).
        """
        if isinstance(source, str):
            from .forcing import load_tsf

            if series_id is None:
                raise ValueError(
                    "series_id is required when source is a TSF path."
                )
            df = load_tsf(source, series_id=series_id, value_col="bc_cfs")
            series = df["bc_cfs"] if "bc_cfs" in df.columns else df.iloc[:, 0]
            label = label or f"{os.path.basename(source)} [{series_id}]"
        elif isinstance(source, pd.Series):
            series = source
        else:
            raise TypeError(
                "source must be a .tsf path or a pandas Series indexed by "
                f"datetime (got {type(source).__name__})."
            )
        s = series.copy()
        s.index = pd.to_datetime(s.index)
        s = s[~s.index.duplicated(keep="first")].sort_index().astype(float)
        self._bc_components.append({
            "series": s,
            "label": label or f"BC component {len(self._bc_components) + 1}",
            "lag_days": float(bc_lag_days),
        })

    @property
    def bc_inflow(self) -> Optional[pd.Series]:
        """
        The raw BC series when exactly ONE component is set, else ``None``.

        The series is AS LOADED: NOT resampled and NOT shifted by its lag.
        Anything that sums the BC into a prediction or a mass balance must
        use ``bc_aligned(index)`` instead — mixing this unshifted series
        with a calibration that scored the shifted one mis-times the
        upstream contribution by exactly the lag.  With multiple components
        this returns ``None``; use ``bc_inflows`` for the full list.
        """
        if len(self._bc_components) == 1:
            return self._bc_components[0]["series"]
        return None

    @property
    def bc_inflows(self) -> list[dict]:
        """
        Copies of all BC components: ``{'label', 'lag_days', 'series'}``.

        The series are raw (unresampled, unshifted) — use
        ``bc_aligned(index)`` for the analysis-ready combined baseline.
        """
        return [dict(c) for c in self._bc_components]

    def bc_aligned(self, index) -> pd.Series:
        """
        Return the BC inflow aligned to *index* (all zeros when no BC is set).

        Notebook cells that assemble the summed prediction outside the
        optimizer (validation scoring, timeseries/scatter plots, mass-balance
        style comparisons built from ``build_calibrated_models()``) must add
        this so reported metrics include the upstream contribution::

            pred_total['rdii_cfs'] += cal.bc_aligned(pred_total.index).to_numpy()

        Raises ``ValueError`` if the BC does not cover every timestamp in
        *index* — same contract as calibration itself.
        """
        index = pd.DatetimeIndex(index)
        if not self._bc_components:
            return pd.Series(0.0, index=index)
        return self._align_bc(index, "requested")

    def _align_bc(
        self, index: pd.DatetimeIndex, context: str, verbose: bool = True
    ) -> pd.Series:
        """
        Align every stored BC component to *index* and return their SUM,
        erroring on any coverage gap in any component.

        Per component: if the series is at a finer timestep than *index*
        (e.g. a sub-daily extraction from a SWMM .out), it is mean-resampled
        to the target timestep first so values are period means rather than
        instantaneous samples.  A non-zero ``lag_days`` then forward-shifts
        the resampled series by ``round(lag_days / grid_step_in_days)``
        steps — the lag is a physical travel time in DAYS and is converted
        to index steps of the target grid, so it shifts by the same
        wall-clock amount on a daily or sub-daily grid (rounding to whole
        steps matches the per-SC network-lag convention on the daily scoring
        grid).  The shift consumes ``lag_steps`` of lead-in: each component
        must start at least its own ``lag_steps`` before *index* or the
        coverage error fires, naming the component.
        """
        target_step = pd.Timedelta(days=1)
        if len(index) > 1:
            target_step = pd.Timedelta(np.median(np.diff(index.values)))
        total = pd.Series(0.0, index=index)
        for comp in self._bc_components:
            bc = comp["series"]
            name = comp["label"] or "(unnamed)"
            if len(index) > 1 and len(bc.index) > 1:
                bc_step = np.median(np.diff(bc.index.values))
                if bc_step < target_step.to_timedelta64():
                    bc = bc.resample(target_step).mean()
                    if verbose:
                        print(
                            f"BC inflow '{name}': resampled "
                            f"from {pd.Timedelta(bc_step)} to {target_step} "
                            f"timestep (period means) to match the observed grid."
                        )
            # Convert the day-based lag to steps of the TARGET grid.  On the
            # daily scoring grid this equals round(lag_days); on a sub-daily
            # grid it shifts the same wall-clock travel time (a 2.0-day lag
            # on an hourly grid is 48 steps, not 2).
            lag_steps = round(pd.Timedelta(days=float(comp["lag_days"])) / target_step)
            if lag_steps > 0:
                # shift() leaves NaN at the component's first lag_steps
                # timestamps — values from before the series start.  Do NOT
                # zero-fill them: when they fall inside *index* they are a
                # genuine coverage gap (the series must start lag_steps
                # before the window) and the n_missing check below must
                # fire; when there is enough lead-in they fall before
                # *index* and reindex drops them.
                bc = bc.shift(lag_steps)
                if verbose:
                    print(
                        f"BC inflow '{name}': shifted forward "
                        f"{lag_steps} step(s) (bc_lag_days={comp['lag_days']:g})."
                    )
            aligned = bc.reindex(index)
            n_missing = int(aligned.isna().sum())
            if n_missing:
                raise ValueError(
                    f"BC inflow '{name}' does not cover the "
                    f"{context} window: required {index.min():%Y-%m-%d} → "
                    f"{index.max():%Y-%m-%d}, BC covers "
                    f"{bc.index.min():%Y-%m-%d} → {bc.index.max():%Y-%m-%d} "
                    f"({n_missing} of {len(index)} timestamps missing or NaN). "
                    f"Regenerate the BC tsf to span the full window, including "
                    f"any warm-up period you intend it to carry."
                    + (
                        f" Note: bc_lag_days={comp['lag_days']:g} shifts this "
                        f"component forward {lag_steps} step(s), so its series "
                        f"must start at least {lag_steps} step(s) BEFORE the "
                        f"window."
                        if lag_steps > 0 else ""
                    )
                )
            total = total.add(aligned)
        return total

    # -----------------------------------------------------------------------
    # Internal: apply user search-bound overrides to the built templates
    # -----------------------------------------------------------------------

    def _apply_param_bound_overrides(self) -> None:
        """
        Apply ``param_bound_overrides`` to the built parameter metadata.

        Updates both ``model._param_meta`` (the calibration search space) and the
        per-subcatchment template parameters, clamping each template's current
        value into the new bounds so the ``seed_initial`` enqueued trial stays
        inside the sampled range.  The log-sampling flag is dropped if a new
        lower bound is no longer positive.  Idempotent, so a re-run of
        ``optimize()`` on already-built templates is safe.
        """
        if not self._param_bound_overrides:
            return

        meta = self.model._param_meta
        names = [name for name, *_ in meta]
        for key, (new_lb, new_ub) in self._param_bound_overrides.items():
            matches = [n for n in names if n == key or n.endswith(f"_{key}")]
            if not matches:
                raise ValueError(
                    f"param_bound_overrides: '{key}' matches no calibratable "
                    f"parameter. Available: {names}"
                )
            for n in matches:
                i = names.index(n)
                _, lb, ub, log = meta[i]
                lb = float(new_lb) if new_lb is not None else lb
                ub = float(new_ub) if new_ub is not None else ub
                if lb > ub:
                    raise ValueError(
                        f"param_bound_overrides: '{n}' resolves to lower "
                        f"({lb}) > upper ({ub})"
                    )
                meta[i] = (n, lb, ub, log and lb > 0)
                for template in self.model._templates:
                    cur = float(template.get_scalar_parameter(n).value)
                    template.set_parameter(
                        n,
                        value=float(np.clip(cur, lb, ub)),
                        lower_bound=lb,
                        upper_bound=ub,
                    )
                print(f"Bound override: {n} -> [{lb:g}, {ub:g}]")

    # -----------------------------------------------------------------------
    # Internal: prepare observed arrays and BFI weights
    # -----------------------------------------------------------------------

    def _prepare_objective_data(self) -> None:
        """
        Pre-compute observed calibration arrays and BFI multipliers.

        Called once at the start of ``optimize()``.  Separating this from the
        objective function avoids repeated slicing/computation inside the hot
        loop.
        """
        if self.model._observed is None:
            raise RuntimeError(
                "No observed flow data. Call model.load_observed() before optimize()."
            )
        if self._cal_start is None:
            raise RuntimeError(
                "Calibration window not set. Call set_windows() before optimize()."
            )

        obs = self.model._observed
        cal_mask = (obs.index >= self._cal_start) & (obs.index <= self._cal_end)
        self._obs_cal = np.ascontiguousarray(obs.loc[cal_mask].to_numpy() * self._obs_scale)
        # Datetime index of the calibration scoring window, kept so seed
        # screening can decompose performance by season (see _screen_seeds).
        self._obs_cal_index = obs.index[cal_mask]
        self._obs_cal_mean = float(self._obs_cal.mean())
        self._obs_cal_var  = float(np.var(self._obs_cal))
        self._obs_cal_std  = float(np.std(self._obs_cal))

        # Upstream BC inflow: align to the observed calibration index once.
        # The aligned array is added to every trial's summed prediction as a
        # fixed baseline (never scaled by obs_scale; forward-shifted by
        # bc_lag_days when set — see _align_bc).  The validation window is
        # checked here too so a short BC tsf fails fast instead of after
        # thousands of trials.
        self._bc_cal = None
        if self._bc_components:
            obs_idx = obs.index[cal_mask]
            bc_cal = self._align_bc(obs_idx, "calibration")
            if self._val_start is not None:
                val_idx = obs.index[
                    (obs.index >= self._val_start) & (obs.index <= self._val_end)
                ]
                if len(val_idx):
                    self._align_bc(val_idx, "validation", verbose=False)
            n_neg = int((bc_cal < 0).sum())
            if n_neg:
                bc_names = " + ".join(
                    c["label"] or "(unnamed)" for c in self._bc_components
                )
                warnings.warn(
                    f"Combined BC inflow '{bc_names}' has {n_neg} "
                    f"negative values in the calibration window (extracted "
                    f"routed series can dip slightly negative). Values are "
                    f"used as-is.",
                    stacklevel=2,
                )
            self._bc_cal = np.ascontiguousarray(bc_cal.to_numpy(), dtype=np.float64)

        # BFI-preserving R_3 scaling:
        # The calibration searches for a single watershed-level R_3 value.
        # Each subcatchment's R_3 is then scaled proportionally to its BFI
        # relative to the area-weighted mean BFI.  This preserves the total
        # watershed volume while distributing the slow (groundwater) component
        # in a physically meaningful way.
        sc_df = self.model._sc_df
        mean_bfi = float((sc_df["bfi"] * sc_df["area_weight"]).sum())
        if mean_bfi == 0:
            mean_bfi = 1.0  # Guard against all-zero BFI
        self._mean_bfi = mean_bfi

        self._r3_multiplier = {
            row["name"]: row["bfi"] / mean_bfi
            for _, row in sc_df.iterrows()
        }

        # Imperviousness-preserving ia_max scaling (linear in pervious fraction):
        #   ia_max_i = ia_max * (1 - imp_i/100) / (1 - mean_imp/100)
        # A 100%-impervious SC -> 0 IA (full runoff); a 0% SC -> above-mean IA.
        self._imp_multiplier: dict[str, float] = {}
        self._mean_perv: Optional[float] = None
        self._imperv_col = None
        if self._imperv_field is not None:
            # Resolve the declared field to an actual sc_df column.  PCSWMM
            # DB user-fields are merged under sanitised lower-case names
            # ('Imperv' -> 'imperv', '%Imperv' -> 'pct_imperv'), so accept
            # either form.  A declared-but-missing field is an ERROR — the
            # old behavior silently calibrated without imperviousness
            # scaling while the bundles recorded imperv_field_used as set.
            from .swmm_parse import _sanitize_field_name
            _sanitized = _sanitize_field_name(self._imperv_field)
            for _cand in (self._imperv_field, _sanitized):
                if _cand in sc_df.columns:
                    self._imperv_col = _cand
                    break
            if self._imperv_col is None:
                raise ValueError(
                    f"imperv_field={self._imperv_field!r} was declared but no "
                    f"matching column exists in the subcatchment table "
                    f"(checked {self._imperv_field!r} and {_sanitized!r}).\n"
                    f"Available columns: {sorted(sc_df.columns)}\n"
                    f"Pass the correct field name, or imperv_field=None to "
                    f"calibrate without imperviousness scaling."
                )
            imp = pd.to_numeric(sc_df[self._imperv_col], errors="coerce").fillna(0.0)
            perv = (1.0 - imp / 100.0).clip(lower=0.0)
            mean_perv = float((perv * sc_df["area_weight"]).sum())
            if mean_perv <= 0:
                mean_perv = 1.0
            self._mean_perv = mean_perv
            self._imp_multiplier = {
                row["name"]: float(perv.iloc[i] / mean_perv)
                for i, (_, row) in enumerate(sc_df.iterrows())
            }

        # R parameter names sorted so R_1 < R_2 < R_3 regardless of how
        # calibration discovers them.
        self._r_sorted = sorted(
            name for name, _, _, _ in self.model._param_meta if "_R_" in name
        )

    # -----------------------------------------------------------------------
    # Internal: run all subcatchments for one trial
    # -----------------------------------------------------------------------

    def _apply_imp_scaling(self, m, shape_params: dict, sc_name: str) -> None:
        """Scale this subcatchment's ia_max by its pervious-fraction ratio."""
        if not self._imp_multiplier:
            return
        factor = self._imp_multiplier.get(sc_name, 1.0)
        if factor == 1.0:
            return
        overrides = {
            k: v * factor for k, v in shape_params.items() if k.endswith("_ia_max")
        }
        if overrides:
            m.apply_parameter_values(overrides)


    def _cal_window_mask(self, datetimes) -> np.ndarray:
        """
        Boolean mask selecting exactly the observed calibration days.

        The calibration forcing spans every day that has precipitation and
        temperature — including warm-up days before the gage record begins,
        and interior gaps in it — so the model integrates continuously
        instead of jumping over missing days.  Scoring still has to line up
        element-for-element with ``_obs_cal``, so the mask is built by
        matching the observed index rather than by date range (a date-range
        mask would pick up warm-up and gap days that have no observation).

        Raises
        ------
        ValueError
            If the selected days do not exactly reproduce the observed
            calibration index — which would silently misalign predictions
            against observations and corrupt every metric.
        """
        dt = pd.DatetimeIndex(datetimes)
        mask = np.asarray(dt.isin(self._obs_cal_index))
        if not np.array_equal(dt[mask].to_numpy(),
                              self._obs_cal_index.to_numpy()):
            missing = self._obs_cal_index.difference(dt)
            raise ValueError(
                f"Calibration forcing does not cover every observed day in "
                f"[{self._cal_start} .. {self._cal_end}]: matched "
                f"{int(mask.sum())} of {len(self._obs_cal_index)} observed "
                f"days."
                + (f"  First missing: "
                   f"{', '.join(str(d.date()) for d in missing[:5])}"
                   f"{' ...' if len(missing) > 5 else ''}"
                   if len(missing) else "")
                + "\nThe precipitation/temperature record must span the whole "
                  "calibration window. Check the rain and temperature series "
                  "referenced by the INP."
            )
        return mask

    def _build_prepared_models(self, sc_forcing_cal: list) -> tuple:
        """Deep-copy + prepare one model per subcatchment and compute its
        calibration-window mask.

        prepare() depends only on the forcing (constant across trials); the
        returned models are then re-run each trial via apply_parameter_values
        + predict(), avoiding a per-trial deepcopy/prepare.  Returns
        ``(prepared, masks)``.  Each call yields an INDEPENDENT model set, so
        parallel multi-start seeds can each hold their own without colliding on
        the models' mutable internal state.
        """
        prepared, masks = [], []
        for template, df_wu_cal in zip(self.model._templates, sc_forcing_cal):
            m = copy.deepcopy(template)
            m.prepare(df_wu_cal)
            pred = m.predict()
            mask = self._cal_window_mask(pred["datetime"])
            prepared.append(m)
            masks.append(mask)
        return prepared, masks

    def _simulate_cal(
        self,
        params: dict[str, float],
        sc_forcing: list[pd.DataFrame],
        prepared: "Optional[list]" = None,
        masks: "Optional[list]" = None,
    ) -> Optional[np.ndarray]:
        """
        Apply *params* to all subcatchment templates and return the summed
        predicted CFS array for the calibration scoring window.

        Parameters
        ----------
        params : dict
            Full parameter dict from the calibration trial.
        sc_forcing : list of DataFrames
            One warm-up-padded calibration forcing DataFrame per subcatchment,
            pre-sliced to [cal_start − warmup, cal_end].
        prepared, masks : list or None
            Per-call override for the persistent prepared models and their
            calibration-window masks.  When given (used by parallel multi-start
            seeds, which each own an isolated model set), these are used instead
            of ``self._prepared_models`` / ``self._cal_masks``.  This avoids
            mutable-state collisions between seeds running on different threads.
            When None, the shared ``self._*`` attributes are used (sequential
            path / single run).

        Returns
        -------
        np.ndarray or None
            Summed predicted CFS over the calibration scoring window
            (including the upstream BC inflow baseline when one is set), or
            None if any subcatchment violates the physical R_sum ≤ 1
            constraint.
        """
        # Split watershed-level R values from shape parameters.
        shape_params = {k: v for k, v in params.items() if "_R_" not in k}

        r_vals = [params[name] for name in self._r_sorted]
        # For single-SC models the R values are applied directly.
        # For multi-SC models, the last R (slow/R_3) is scaled per subcatchment.
        has_bfi_scaling = len(self._r3_multiplier) > 1

        # Upstream BC inflow enters as a fixed baseline (constant across
        # trials); local SC predictions are summed on top of it.
        if self._bc_cal is not None:
            total_pred = self._bc_cal.copy()
        else:
            total_pred = np.zeros(len(self._obs_cal))

        # Fast path: reuse persistent prepared models (built once in optimize()).
        # Only parameters change between trials; predict() re-reads them via the
        # model's _sync_from_params(), so no deepcopy / prepare is needed here.
        # A per-call `prepared`/`masks` (parallel seeds) overrides the shared set.
        if prepared is None:
            prepared = self._prepared_models
            masks = self._cal_masks
        use_persistent = prepared is not None

        sc_iter = zip(
            self.model._templates,
            self.model._sc_df.itertuples(),
            sc_forcing,
            range(len(self.model._templates)),
        )

        for template, sc_row, df_wu_cal, idx in sc_iter:
            sc_name = sc_row.name

            # Compute per-subcatchment R values (BFI scaling for R_3 in multi-SC).
            if has_bfi_scaling and len(self._r_sorted) >= 3:
                r3_i = r_vals[-1] * self._r3_multiplier.get(sc_name, 1.0)
                sc_r_vals = r_vals[:-1] + [r3_i]
            else:
                sc_r_vals = r_vals

            # Physical gate: sum of R fractions must not exceed 1.0.
            # Pruning here avoids running the full convolution for infeasible params.
            if sum(sc_r_vals) > 1.0:
                return None

            if use_persistent:
                m = prepared[idx]
            else:
                # Parallel-safe fallback: fresh copy + prepare each trial.
                m = copy.deepcopy(template)

            m.apply_parameter_values(shape_params)
            m.apply_parameter_values(
                {name: val for name, val in zip(self._r_sorted, sc_r_vals)}
            )
            self._apply_imp_scaling(m, shape_params, sc_name)

            # Lag shift: shift the SC's full prediction forward by network lag
            # (junction → gage travel time) before slicing to the cal window.
            lag_days = round(self._network_lags.get(sc_name, 0.0))

            if use_persistent:
                # Forcing already prepared; just re-run with new parameters.
                pred_full = m.predict()
                sc_rdii = pred_full["rdii_cfs"].to_numpy()
                if lag_days > 0:
                    sc_rdii = np.roll(sc_rdii, lag_days)
                    sc_rdii[:lag_days] = 0.0
                cal_mask = masks[idx]
                total_pred += sc_rdii[cal_mask]
            else:
                m.prepare(df_wu_cal)
                pred_full = m.predict()
                sc_rdii = pred_full["rdii_cfs"].to_numpy()
                if lag_days > 0:
                    sc_rdii = np.roll(sc_rdii, lag_days)
                    sc_rdii[:lag_days] = 0.0
                cal_mask = self._cal_window_mask(pred_full["datetime"])
                total_pred += sc_rdii[cal_mask]

        return total_pred

    # -----------------------------------------------------------------------
    # Solver-agnostic calibration
    # -----------------------------------------------------------------------

    def _optimize_sparsehydro(
        self, sc_forcing_cal, obs_cal, obs_mean, obs_std, obs_var, **solver_kwargs,
    ) -> None:
        """Calibrate through the common solver API with isolated model state."""
        from types import SimpleNamespace
        from sparsehydro.calibration.solvers import (
            NSGAIISolver, ScipySolver, ParticleSwarmSolver, PlatypusSolver,
            PlatypusCmaEsSolver,
        )

        problem = _RdiiCalibrationProblem(self, sc_forcing_cal)
        if self._sampler_key == "cma-es":
            defaults = {
                name: float(self.model._templates[0].get_scalar_parameter(name).value)
                if name in self.model._templates[0].scalar_parameter_names else (lo + hi) / 2
                for name, lo, hi, _ in self.model._param_meta
            }
            initial = [problem.to_search(defaults)] if self.seed_initial else []
            warmstart = getattr(self, "_warmstart_params", None)
            if warmstart:
                unknown = set(warmstart) - set(problem.param_names)
                if unknown:
                    raise ValueError(f"Unknown warm-start parameters: {sorted(unknown)}")
                initial.append(problem.to_search({**defaults, **warmstart}))
            solver = PlatypusCmaEsSolver(
                objective_index=0, n_trials=self.n_trials,
                offspring_size=solver_kwargs.pop("offspring_size", self.n_startup),
                seeds=self._seeds, progress_every=self.progress_every,
                initial_points=initial or None, **solver_kwargs,
            )
        elif self._sampler_key == "scipy":
            solver = ScipySolver(
                method=solver_kwargs.pop("method", "differential_evolution"),
                maxiter=self.n_trials, seed=42, **solver_kwargs,
            )
        elif self._sampler_key == "sh_nsga2":
            solver = NSGAIISolver(
                pop_size=solver_kwargs.pop("pop_size", self.n_startup),
                n_gen=self.n_trials, seed=42, **solver_kwargs,
            )
        elif self._sampler_key == "sh_pso":
            solver = ParticleSwarmSolver(
                swarm_size=solver_kwargs.pop("swarm_size", self.n_startup),
                n_evaluations=self.n_trials, seed=42, **solver_kwargs,
            )
        elif self._sampler_key == "sh_platypus":
            algorithm = solver_kwargs.pop("algorithm_class", None)
            if algorithm is None:
                raise ValueError("sh_platypus requires algorithm_class=<platypus.Algorithm>")
            solver = PlatypusSolver(
                algorithm_class=algorithm, n_evaluations=self.n_trials,
                seed=42, **solver_kwargs,
            )
        else:
            raise ValueError(f"Unrecognised sampler: {self._sampler_key!r}")

        print(f"[sparsehydro/{self.sampler_type}] objectives={self.objective_mode} "
              f"n_trials={self.n_trials} ({problem.n_params} params)")
        result = solver.solve(problem)
        self.seed_screen = []
        self.seed_selection_note = ""
        if self._sampler_key == "cma-es":
            candidates = [
                (entry["seed"],
                 entry["opt_value"] * (-1 if problem.minimize_flags[0] else 1),
                 SimpleNamespace(params=dict(zip(problem.param_names, problem.to_physical(entry["x"])))),
                 result)
                for entry in solver.seed_solutions_
            ]
            if len(candidates) > 1 and (self.seed_selection in _SCREENED_SELECTIONS
                                         or self.seed_override is not None):
                selected, self.seed_screen, self.seed_selection_note = self._screen_seeds(
                    candidates, sc_forcing_cal)
            else:
                selected = max(range(len(candidates)), key=lambda i: candidates[i][1])
            self.best_seed, _, best, _ = candidates[selected]
            best_x = problem.to_search(best.params)
            result.pareto_X = best_x.reshape(1, -1)
            result.pareto_F = problem.evaluate(best_x).reshape(1, -1)
        else:
            display = result.objective_display_values()
            candidates = [
                SimpleNamespace(params=dict(zip(problem.param_names, problem.to_physical(x))),
                                user_attrs=dict(zip(problem.objective_names, f)))
                for x, f in zip(result.pareto_X, display)
            ]
            best = self._select_best_multi(candidates)
            if best is None:
                raise RuntimeError("Calibration produced no selectable solution")
        self.best_params = dict(best.params)
        if sum(self.best_params[name] for name in self._r_sorted) > 1:
            raise RuntimeError("Calibration selected physically infeasible watershed R fractions")
        # Public diagnostics and saved parameters always use physical units.
        result.pareto_X = problem.to_physical(result.pareto_X)
        result.history = [
            record._replace(X=problem.to_physical(record.X)) for record in result.history
        ]
        self.study = result
        self.studies = [result]
        self.best_trial = None
        total_best = self._simulate_cal(self.best_params, sc_forcing_cal)
        if total_best is None:
            raise RuntimeError("Calibration selected physically infeasible R fractions")
        self.cal_ccc = float(_fast_ccc(obs_cal, total_best, obs_mean, obs_var))
        self.cal_kge = float(_kge(obs_cal, total_best, obs_mean, obs_std))
        self.cal_nse = float(_nse(obs_cal, total_best))
        self.cal_pbias = float(_pbias(obs_cal, total_best)) * 100
        self._store_selected_diagnostics(sc_forcing_cal)

    # -----------------------------------------------------------------------
    # Multi-start seed screening (calibration window only)
    # -----------------------------------------------------------------------

    def _pinned_params(self, params: dict[str, float]) -> list[str]:
        """
        Names of parameters sitting within ``bound_warn_pct`` of a search bound.

        A pinned parameter usually means the optimizer wanted a value the
        bounds do not allow, and is often a sign that the parameter is
        absorbing a structural model deficiency rather than describing the
        watershed.  Reported only — never used to reject a seed, because a
        parameter at a bound can legitimately be the physical answer.
        """
        pinned = []
        frac = self.bound_warn_pct / 100.0
        for name, lb, ub, log in self.model._param_meta:
            if name not in params:
                continue
            val = params[name]
            if log and lb > 0 and ub > 0 and val > 0:
                # Judge proximity in the space the search actually explored.
                lo, hi, v = math.log(lb), math.log(ub), math.log(val)
            else:
                lo, hi, v = lb, ub, val
            span = hi - lo
            if span <= 0:
                continue
            if (v - lo) <= frac * span or (hi - v) <= frac * span:
                pinned.append(name)
        return pinned

    def _screen_seed(self, params: dict[str, float], sc_forcing_cal: list) -> dict:
        """
        Score one seed's parameters on the CALIBRATION window.

        Returns a dict of overall metrics, per-season PBIAS, the worst-season
        |PBIAS|, and any bound-pinned parameter names.  Every number comes
        from the calibration window — the validation window is not touched.
        """
        obs   = self._obs_cal
        index = self._obs_cal_index
        pred  = self._simulate_cal(params, sc_forcing_cal)

        row: dict = {
            "kge": float("nan"), "ccc": float("nan"), "nse": float("nan"),
            "pbias": float("nan"), "season_pbias": {},
            "worst_season": None, "worst_season_pbias": float("nan"),
            "pinned": self._pinned_params(params),
            "valid": pred is not None,
        }
        if pred is None:
            # R_sum > 1 — physically impossible, so the seed cannot be used.
            return row

        row["kge"]   = float(_kge(obs, pred, self._obs_cal_mean, self._obs_cal_std))
        row["ccc"]   = float(_fast_ccc(obs, pred, self._obs_cal_mean, self._obs_cal_var))
        row["nse"]   = float(_nse(obs, pred))
        row["pbias"] = float(_pbias(obs, pred)) * 100.0

        # Meteorological seasons: DJF / MAM / JJA / SON.
        month = pd.DatetimeIndex(index).month.to_numpy()
        seasons = {
            "DJF": np.isin(month, (12, 1, 2)),
            "MAM": np.isin(month, (3, 4, 5)),
            "JJA": np.isin(month, (6, 7, 8)),
            "SON": np.isin(month, (9, 10, 11)),
        }
        worst_name, worst_val = None, 0.0
        for label, mask in seasons.items():
            if not mask.any():
                continue
            o_sum = float(obs[mask].sum())
            if o_sum <= 0:
                continue
            sp = (float(pred[mask].sum()) - o_sum) / o_sum * 100.0
            row["season_pbias"][label] = sp
            if abs(sp) > abs(worst_val):
                worst_name, worst_val = label, sp
        row["worst_season"] = worst_name
        row["worst_season_pbias"] = worst_val if worst_name else float("nan")
        return row

    def _store_selected_diagnostics(self, sc_forcing_cal: list) -> None:
        """
        Record seasonal PBIAS and pinned parameters for the selected model.

        Runs for every calibration, single-seed included — the seasonal
        breakdown is the diagnostic that reveals an aggregate PBIAS near zero
        which is really a winter deficit cancelling a summer surplus.
        """
        self.cal_season_pbias: dict = {}
        self.cal_worst_season: Optional[str] = None
        self.cal_worst_season_pbias: float = float("nan")
        self.pinned_params: list[str] = []
        if self.best_params is None:
            return
        try:
            row = self._screen_seed(dict(self.best_params), sc_forcing_cal)
        except Exception as exc:      # diagnostics must never break a run
            warnings.warn(f"Seasonal diagnostics unavailable: {exc}", stacklevel=2)
            return
        self.cal_season_pbias       = row["season_pbias"]
        self.cal_worst_season       = row["worst_season"]
        self.cal_worst_season_pbias = row["worst_season_pbias"]
        self.pinned_params          = row["pinned"]

    def _screen_seeds(self, seed_results: list, sc_forcing_cal: list) -> tuple:
        """
        Screen every multi-start seed and pick a winner.

        Applies the ``seed_selection='robust'`` rule described in the class
        docstring, prints the full table, and returns
        ``(chosen_index, table, note)``.  Selection uses the calibration
        window only, so the validation window remains a blind holdout.
        """
        table = []
        for idx, (seed, score, best, _study) in enumerate(seed_results):
            row = self._screen_seed(dict(best.params), sc_forcing_cal)
            row.update(seed=seed, obj_score=score, index=idx)
            row["fit"] = (row["kge"] + row["ccc"]) / 2.0
            # Balanced score: fit minus a penalty for seasonal imbalance.
            # Both terms are fractions of 1.0, so seasonal_weight=1.0 means a
            # 10-point worst-season PBIAS costs 0.10 of mean(KGE, CCC).
            _ws = row["worst_season_pbias"]
            row["bal"] = (row["fit"]
                          - self.seasonal_weight * abs(_ws) / 100.0
                          if math.isfinite(_ws) else row["fit"])
            row["pass_pbias"]  = (row["valid"]
                                  and abs(row["pbias"]) <= self.pbias_max)
            row["pass_season"] = (row["valid"]
                                  and math.isfinite(row["worst_season_pbias"])
                                  and abs(row["worst_season_pbias"])
                                     <= self.seasonal_pbias_max)
            row["passed"] = row["pass_pbias"] and row["pass_season"]
            table.append(row)

        # A manual override wins outright — but the table above still prints,
        # so the reader sees what the automatic rule would have chosen and how
        # the forced seed compares.
        if self.seed_override is not None:
            forced = next(r for r in table if r["seed"] == self.seed_override)
            if not forced["valid"]:
                raise ValueError(
                    f"seed_override={self.seed_override} produced a physically "
                    f"invalid result (R_1+R_2+R_3 > 1) and cannot be used."
                )
            auto = max((r for r in table if r["passed"]),
                       key=lambda r: r["fit"], default=None)
            note = (f"MANUAL OVERRIDE — seed {self.seed_override} forced by "
                    f"seed_override")
            if auto is not None and auto["index"] != forced["index"]:
                note += f" (automatic rule would have picked seed {auto['seed']})"
            if not forced["passed"]:
                note += "; NOTE this seed does NOT pass the PBIAS gates"
            self._print_screen_table(table, forced, note)
            return forced["index"], table, note

        rank_key = "bal" if self.seed_selection == "balanced" else "fit"
        rank_label = (
            f"mean(KGE, CCC) - {self.seasonal_weight:g}x|worst-season PBIAS|"
            if rank_key == "bal" else "mean(KGE, CCC)"
        )

        survivors = [r for r in table if r["passed"]]
        if survivors:
            winner = max(survivors, key=lambda r: r[rank_key])
            note = (f"{len(survivors)}/{len(table)} seeds passed both PBIAS "
                    f"gates; ranked by {rank_label}.")
            # When the two rankings disagree, say so — the reader should know
            # a different seed would have won under the other rule.
            other_key = "fit" if rank_key == "bal" else "bal"
            other = max(survivors, key=lambda r: r[other_key])
            if other["index"] != winner["index"]:
                other_label = ("mean(KGE, CCC)" if other_key == "fit"
                               else "the balanced score")
                note += (f"  (ranking by {other_label} would have picked "
                         f"seed {other['seed']}.)")
        else:
            # Nothing cleared the gates.  Rank everything by fit minus a
            # penalty for how far each seed overshot, and say so loudly —
            # silently falling back would hide that no seed is trustworthy.
            usable = [r for r in table if r["valid"]]
            if not usable:
                raise RuntimeError(
                    "Every multi-start seed produced a physically invalid "
                    "result (R_1+R_2+R_3 > 1). Nothing can be selected — "
                    "check the R parameter bounds and the forcing data."
                )
            def _penalized(r):
                over_tot = max(0.0, abs(r["pbias"]) - self.pbias_max)
                over_sea = max(0.0, abs(r["worst_season_pbias"])
                                    - self.seasonal_pbias_max)
                return r[rank_key] - (over_tot + over_sea) / 100.0
            winner = max(usable, key=_penalized)
            note = (f"*** NO seed passed the PBIAS gates "
                    f"(overall <= {self.pbias_max:g}%, worst-season <= "
                    f"{self.seasonal_pbias_max:g}%). Fell back to "
                    f"penalized mean(KGE, CCC). Treat this calibration as "
                    f"volume-unreliable.")

        self._print_screen_table(table, winner, note)
        return winner["index"], table, note

    def _print_screen_table(self, table: list, winner: dict, note: str) -> None:
        """Print the per-seed screening table, the deciding rule, and warnings."""
        print()
        print("  Multi-start seed screening (CALIBRATION window only — "
              "validation not consulted):")
        rank_key = "bal" if self.seed_selection == "balanced" else "fit"
        # Star the column actually doing the ranking.
        h_fit = "mean(K,C)" + ("" if rank_key == "fit" else " ")
        h_bal = "balanced" + ("" if rank_key == "bal" else " ")
        print(f"  {'Seed':>5}  {'KGE':>7}  {'CCC':>7}  {'NSE':>7}  "
              f"{'PBIAS':>8}  {'WorstSeas':>12}  "
              f"{('*' + h_fit) if rank_key == 'fit' else ' ' + h_fit:>10}  "
              f"{('*' + h_bal) if rank_key == 'bal' else ' ' + h_bal:>10}  Gates")
        print("  " + "-" * 98)
        for r in sorted(table, key=lambda r: -r[rank_key] if r["valid"] else 1):
            if not r["valid"]:
                print(f"  {r['seed']:>5}  {'— invalid (sum(R) > 1) —':>50}")
                continue
            ws = (f"{r['worst_season']} {r['worst_season_pbias']:+.1f}%"
                  if r["worst_season"] else "n/a")
            gates = "PASS" if r["passed"] else "fail:" + ",".join(
                g for g, ok in (("PBIAS", r["pass_pbias"]),
                                ("season", r["pass_season"])) if not ok
            )
            mark = " <-- selected" if r["index"] == winner["index"] else ""
            print(f"  {r['seed']:>5}  {r['kge']:>7.4f}  {r['ccc']:>7.4f}  "
                  f"{r['nse']:>7.4f}  {r['pbias']:>+7.2f}%  {ws:>12}  "
                  f"{r['fit']:>10.4f}  {r['bal']:>10.4f}  {gates}{mark}")
        print(f"  balanced = mean(KGE, CCC) - {self.seasonal_weight:g} x "
              f"|worst-season PBIAS|/100   (* = column used for ranking)")
        print(f"  Rule: {note}")

        if winner["pinned"]:
            print(f"  *** WARNING — selected seed {winner['seed']} has "
                  f"{len(winner['pinned'])} parameter(s) within "
                  f"{self.bound_warn_pct:g}% of a search bound:")
            print(f"      {', '.join(winner['pinned'])}")
            print("      A pinned parameter may be physically real, but it can "
                  "also mean the optimizer is")
            print("      absorbing a structural deficiency and the model will "
                  "extrapolate poorly. Review before use.")
        print()

    # -----------------------------------------------------------------------
    # Pareto selection helper
    # -----------------------------------------------------------------------

    def _select_best_multi(
        self,
        trials: list,
    ) -> Optional[object]:
        """
        Select the Pareto-front knee trial for multi-objective optimization.

        Strategy
        --------
        1. Among Pareto-front trials, find those with all *minimize* objectives
           ≤ their respective targets (``pbias_target`` applies to any objective
           whose registry key contains 'pbias' or 'volume').
        2. From qualifying trials, return the one with the best value on the
           first *maximize* objective (primary score).
        3. If no trials satisfy the targets, return the one with the smallest
           sum of normalized minimize-objective values as the least-bad option.

        Parameters
        ----------
        trials : list
            Pareto candidates with params and user_attrs.
        """
        valid = [t for t in trials if t.user_attrs]
        if not valid:
            return None

        # Identify the first maximize-objective key for primary scoring.
        maximize_keys = [k for k in self._obj_keys if not OBJECTIVE_REGISTRY[k][1]]
        minimize_keys = [k for k in self._obj_keys if OBJECTIVE_REGISTRY[k][1]]
        primary_key   = maximize_keys[0] if maximize_keys else None

        def _get(t, key):
            return t.user_attrs.get(key, float("nan"))

        # Apply per-key targets: pbias_target for any error/bias metric.
        def _within_target(t):
            for k in minimize_keys:
                val = _get(t, k)
                if np.isnan(val):
                    return False
                if "pbias" in k or "volume" in k:
                    if val > self.pbias_target:
                        return False
            return True

        qualified = [t for t in valid if _within_target(t)]

        if qualified and primary_key:
            return max(qualified, key=lambda t: _get(t, primary_key))
        if qualified:
            # All objectives are minimize — pick smallest first.
            return min(qualified, key=lambda t: _get(t, minimize_keys[0]))

        # No trial satisfies all targets — pick the least-bad minimize sum.
        if minimize_keys:
            return min(valid, key=lambda t: sum(_get(t, k) for k in minimize_keys))
        if primary_key:
            return max(valid, key=lambda t: _get(t, primary_key))
        return valid[0]

    # -----------------------------------------------------------------------
    # Main optimize() entry point
    # -----------------------------------------------------------------------

    def optimize(self, n_jobs: int = 1, allow_parallel_in_kernel: bool = True,
                 **solver_kwargs) -> None:
        """
        Run calibration with the selected sparsehydro solver.

        This method:

        1. Builds sparsehydro model templates (via ``RdiiModel._build_templates``)
           if not already built.
        2. Pre-slices all forcing DataFrames for the calibration window + warm-up.
        3. Evaluates isolated multi-subcatchment model copies through the
           selected solver, preserving physical gates and BFI/lag scaling.

        4. Selects the best solution and populates ``best_params`` and
           ``cal_ccc / cal_kge / cal_nse / cal_pbias``.

        Parameters
        ----------
        n_jobs : int
            Number of multi-start SEEDS to run concurrently (default 1).  With
            multi_start > 1 and n_jobs > 1, that many CMA-ES seeds run in
            parallel as separate PROCESSES (joblib/loky), each with its own
            isolated prepared models; each seed itself stays single-threaded.
            Processes (not threads) are used because the hot path is a
            pure-Python loop that holds the GIL, so threads give no speedup;
            processes deliver ~Nx (e.g. ~3x on 4 seeds, minus spawn/pickle
            overhead).  This replaces the old per-trial parallelism, which was
            a net loss for the fast objective.  n_jobs is capped at the number
            of seeds. Other solvers use their own worker options.
        allow_parallel_in_kernel : bool
            Default True: parallelise seeds (n_jobs > 1) even inside a
            Jupyter/IPython kernel.  Set False as an escape hatch to force
            sequential seeds in a notebook — loky's child-process spawn can
            destabilise an embedded kernel on some setups, so flip this off if a
            kernel struggles.  Has no effect outside a kernel (scripts always
            parallelise when n_jobs > 1).
        **solver_kwargs
            Extra keyword arguments forwarded to the sparsehydro solver
            constructor.  Examples:

            - ``pop_size=50``        for sh_nsga2
            - ``swarm_size=80``      for sh_pso
            - ``algorithm_class=platypus.SPEA2``  for sh_platypus
            - ``method='Nelder-Mead'``             for scipy (local search)
        """
        if self._cal_start is None:
            raise RuntimeError("Call set_windows() before optimize().")

        # Build templates if not yet done.
        if not self.model._templates_built:
            self.model._build_templates(
                ia_max_ub=self._ia_max_ub,
                uh1_t_ub=self._uh1_t_ub,
                ia_kt=self._ia_kt,
                ia_t_ref=self._ia_t_ref,
                ia_t_freeze=self._ia_t_freeze,
                calibrate_temperature_params=self._calibrate_temp,
            )

        # Apply user search-bound overrides (no-op when none were given).
        self._apply_param_bound_overrides()

        # Pre-compute observed arrays and BFI weights.
        self._prepare_objective_data()

        # Pre-slice calibration forcing DataFrames (warm-up + scoring window).
        #
        # observed=None is deliberate: it keeps EVERY day that has
        # precipitation and temperature, rather than only days that also have
        # an observed flow value.  Joining against the observed series here
        # would silently delete the warm-up whenever the gage record starts at
        # (or after) the calibration start — the model would then begin cold,
        # with an empty IA store and an unspun RTK convolution, which matters
        # most for the slow component whose time base can approach a year.
        # It would also skip over interior gaps in the gage record instead of
        # integrating through them.  Scoring is aligned separately, by
        # _cal_window_mask(), so predictions still match observations
        # element-for-element.
        warmup_cal_start = pd.Timestamp(self._cal_start) - pd.Timedelta(days=self.warmup_buffer)
        sc_forcing_cal: list[pd.DataFrame] = []
        for sc_row in self.model._sc_df.itertuples():
            df = self.model._get_sc_df(
                sc_row=self.model._sc_df.loc[self.model._sc_df["name"] == sc_row.name].iloc[0],
                observed=None,
                start=warmup_cal_start,
                end=pd.Timestamp(self._cal_end),
            )
            sc_forcing_cal.append(df)

        # Warn when the forcing cannot actually deliver the warm-up asked for.
        # The buffer is a request, not a guarantee: it is limited by how far
        # the rain/temperature record extends before the calibration start.
        if sc_forcing_cal and self.warmup_buffer > 0:
            first = min(df["datetime"].min() for df in sc_forcing_cal
                        if len(df))
            got = (pd.Timestamp(self._cal_start) - pd.Timestamp(first)).days
            if got < self.warmup_buffer:
                print(
                    f"  *** WARNING — warm-up is {got} days, not the "
                    f"{self.warmup_buffer} requested: the forcing record "
                    f"starts {pd.Timestamp(first):%Y-%m-%d}.\n"
                    f"      The model begins with an unspun slow-response "
                    f"store, which biases the early calibration years.\n"
                    f"      Extend the rain/temperature series earlier, or "
                    f"move CAL_START later.",
                    flush=True,
                )

        self._prepared_models, self._cal_masks = self._build_prepared_models(sc_forcing_cal)
        if self._sampler_key == "cma-es":
            solver_kwargs["n_jobs"] = n_jobs
            solver_kwargs["allow_parallel_in_kernel"] = allow_parallel_in_kernel
        self._optimize_sparsehydro(
            sc_forcing_cal=sc_forcing_cal, obs_cal=self._obs_cal,
            obs_mean=self._obs_cal_mean, obs_std=self._obs_cal_std,
            obs_var=self._obs_cal_var, **solver_kwargs,
        )

    # -----------------------------------------------------------------------
    # Warm-start from JSON
    # -----------------------------------------------------------------------

    def load_warmstart(self, json_path: str) -> None:
        """
        Add initial CMA-ES evaluations from a previously saved
        ``best_params.json`` file.

        Must be called after ``set_windows()`` but before ``optimize()``.
        Parameters are evaluated before the evolutionary search in every
        CMA-ES seed and count against its budget. Other solvers do not use
        these points; unknown or out-of-bounds parameters raise an error.

        Parameters
        ----------
        json_path : str
            Path to a JSON file in the format written by ``save_params()``.
        """
        if self._sampler_key != "cma-es":
            raise ValueError("load_warmstart is supported only by CMA-ES")
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self._warmstart_params = payload.get("parameters", {})
        print(f"Warm-start: loaded {len(self._warmstart_params)} parameters from {json_path!r}")

    # -----------------------------------------------------------------------
    # Apply best parameters to a copy of each subcatchment template
    # -----------------------------------------------------------------------

    def build_calibrated_models(self) -> list:
        """
        Return a list of sparsehydro models with the best calibrated parameters
        already applied, one per subcatchment.

        These are deep copies of the templates — modifying them does not affect
        the stored templates.

        Returns
        -------
        list of EnsembleModel
        """
        if self.best_params is None:
            raise RuntimeError("Run optimize() first.")

        sc_df = self.model._sc_df
        shape_params = {k: v for k, v in self.best_params.items() if "_R_" not in k}
        r_vals = [self.best_params[name] for name in self._r_sorted]
        has_bfi = len(self._r3_multiplier) > 1

        models = []
        for template, sc_row in zip(self.model._templates, sc_df.itertuples()):
            m = copy.deepcopy(template)
            m.apply_parameter_values(shape_params)

            if has_bfi and len(self._r_sorted) >= 3:
                r3_i = r_vals[-1] * self._r3_multiplier.get(sc_row.name, 1.0)
                sc_r_vals = r_vals[:-1] + [r3_i]
            else:
                sc_r_vals = r_vals

            m.apply_parameter_values(
                {name: val for name, val in zip(self._r_sorted, sc_r_vals)}
            )
            self._apply_imp_scaling(m, shape_params, sc_row.name)
            models.append(m)

        return models

    def save_sc_models(self, out_dir: str) -> dict[str, str]:
        """
        Serialize each calibrated per-SC sparsehydro model to a cloudpickle file.

        Each file is a bundle dict containing the calibrated EnsembleModel plus
        traceability metadata (SC name, area, BFI, imperviousness) so that
        future-condition scaling can be applied without re-calibrating.

        Returns
        -------
        dict
            Mapping of subcatchment name → absolute path to the ``.pkl`` file.
        """
        import cloudpickle

        from .io import SH_SERIES_PREFIX

        if self.best_params is None:
            raise RuntimeError("Run optimize() first.")

        models = self.build_calibrated_models()
        sc_df = self.model._sc_df
        paths: dict[str, str] = {}
        os.makedirs(out_dir, exist_ok=True)

        for m, sc_row in zip(models, sc_df.itertuples()):
            imperv_pct = None
            imp_multiplier = None
            if self._imperv_col is not None and self._imp_multiplier:
                raw = getattr(sc_row, self._imperv_col, None)
                if raw is not None:
                    try:
                        imperv_pct = float(raw)
                    except (TypeError, ValueError):
                        pass
                imp_multiplier = self._imp_multiplier.get(sc_row.name)

            # Network lag: travel time from this SC's outlet junction to the
            # calibration gage node (decimal days).  Stored so run_sparsehydro()
            # can apply the correct lag without re-running compute_network_lags().
            lag_d = 0.0
            if self._network_lags:
                lag_d = float(self._network_lags.get(sc_row.name, 0.0))

            # Compute per-SC R values for the params_v2 reconstruction recipe.
            shape_params = {k: v for k, v in self.best_params.items() if "_R_" not in k}
            r_vals = [self.best_params[name] for name in self._r_sorted]
            has_bfi = len(self._r3_multiplier) > 1
            if has_bfi and len(self._r_sorted) >= 3:
                r3_i = r_vals[-1] * self._r3_multiplier.get(sc_row.name, 1.0)
                sc_r_vals = r_vals[:-1] + [r3_i]
            else:
                sc_r_vals = r_vals
            sc_params = dict(shape_params)
            sc_params.update({name: val for name, val in zip(self._r_sorted, sc_r_vals)})
            if self._imp_multiplier:
                factor = self._imp_multiplier.get(sc_row.name, 1.0)
                if factor != 1.0:
                    for k, v in shape_params.items():
                        if k.endswith("_ia_max"):
                            sc_params[k] = v * factor

            bundle = {
                "sc_name": sc_row.name,
                "area_acres": float(sc_row.area_acres),
                "bfi": float(getattr(sc_row, "bfi", float("nan"))),
                "imperv_pct": imperv_pct,
                "imp_multiplier": imp_multiplier,
                "mean_perv": self._mean_perv,
                # Metershed area-weighted mean BFI and this SC's R_3 multiplier
                # (bfi / mean_bfi), stored so rescale_bfi() can reverse the
                # calibration-time R_3 scaling for a new BFI value.
                "mean_bfi": self._mean_bfi,
                "r3_multiplier": self._r3_multiplier.get(sc_row.name, 1.0),
                "network_lag_days": lag_d,
                "bfi_field_used": self._bfi_field,
                "imperv_field_used": self._imperv_field,
                # params_v2: Python-version-independent reconstruction recipe.
                # Stores plain dicts only — no live Python objects, no bytecode.
                # Portable across Python versions. The "model" key is omitted
                # intentionally so cloudpickle never tries to serialize bytecode.
                "params_v2": {
                    "params": sc_params,
                    "n_models": self.model.n_models,
                    "n_triangles": self.model.n_triangles,
                    "units": self.model.units,
                    "ia_defaults": dict(self.model.ia_defaults),
                },
            }

            path = os.path.join(out_dir, f"{SH_SERIES_PREFIX}{sc_row.name}.pkl")
            with open(path, "wb") as f:
                cloudpickle.dump(bundle, f)
            paths[sc_row.name] = os.path.abspath(path)
            print(f"  saved: {path}")

        return paths

    @staticmethod
    def load_sc_model(pkl_path: str, expected_sc_name: Optional[str] = None) -> dict:
        """
        Load a cloudpickle bundle written by ``save_sc_models()``.

        Parameters
        ----------
        pkl_path : str
            Path to the ``.pkl`` file.
        expected_sc_name : str or None
            If provided, raises ``ValueError`` if the loaded bundle's
            ``sc_name`` does not match (guards against loading the wrong file).

        Returns
        -------
        dict
            Bundle with keys: ``sc_name``, ``area_acres``, ``bfi``,
            ``imperv_pct``, ``imp_multiplier``, ``mean_perv``, ``mean_bfi``,
            ``r3_multiplier``, ``network_lag_days``, ``bfi_field_used``,
            ``imperv_field_used``, and ``params_v2`` (the version-independent
            reconstruction recipe: plain dicts of calibrated parameter values).
            Legacy bundles (pre-``params_v2``) instead carry a pickled
            ``model`` object and may lack the newer metadata keys.
        """
        import cloudpickle

        with open(pkl_path, "rb") as f:
            bundle = cloudpickle.load(f)
        if expected_sc_name is not None and bundle.get("sc_name") != expected_sc_name:
            raise ValueError(
                f"Loaded SC name {bundle.get('sc_name')!r} != expected {expected_sc_name!r}"
            )
        return bundle

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    def summary(self) -> None:
        """
        Print the best calibrated parameters and calibration period metrics.
        """
        if self.best_params is None:
            print("CalOptimizer: no results yet — call optimize() first.")
            return

        print(f"CalOptimizer summary — {self.model.name or '(unnamed)'}")
        print(
            f"  Objective: {self.objective_mode}  |  Sampler: {self.sampler_type}  "
            f"|  Best trial: #{getattr(self.best_trial, 'number', 'N/A')}"
        )
        print(
            f"  Cal window: {self._cal_start} → {self._cal_end}  "
            f"|  Warmup: {self.warmup_buffer} days"
        )
        if len(self._seeds) > 1:
            print(
                f"  Seed select: {self.seed_selection}  "
                f"|  chose seed {self.best_seed} of {len(self._seeds)}"
            )
            if self.seed_selection_note:
                print(f"               {self.seed_selection_note}")
        for comp in self._bc_components:
            bc_note = comp["label"] or "(unnamed series)"
            if comp["lag_days"]:
                applied = round(comp["lag_days"])
                bc_note += f"  |  lag: {comp['lag_days']:g} day(s)"
                if applied == 0:
                    bc_note += " (rounds to 0 at the daily step - NO shift applied)"
            print(f"  BC inflow: {bc_note}")
        if self._bc_components and self._bc_cal is not None:
            n = len(self._bc_components)
            print(
                f"  BC {'combined ' if n > 1 else ''}cal-window mean: "
                f"{self._bc_cal.mean():.2f} cfs"
            )
        print()
        print(f"  {'Parameter':<40}  {'Value':>12}  {'[Lower, Upper]':<20}  Scale")
        print("  " + "-" * 80)
        for name, lb, ub, log in self.model._param_meta:
            val   = self.best_params[name]
            scale = "LOG" if log else "LIN"
            print(f"  {name:<40}  {val:>12.6f}  [{lb:.4f}, {ub:.4f}]{'':>4}  {scale}")
        print()

        if len(self._r3_multiplier) <= 1:
            bfi_status, bfi_detail = "n/a", "single subcatchment - no scaling applied"
        elif self._bfi_field is None:
            bfi_status, bfi_detail = "flat", "no bfi_field given - all BFI=1.0, R_3 uniform"
        else:
            bfi_status = "enabled"
            bfi_detail = f"field={self._bfi_field!r}, area-wtd mean BFI={self._mean_bfi:.3f}"
        print(f"  {'BFI-preserving R_3 scaling':<40}  {bfi_status:>12}  {bfi_detail}")

        if len(self._r3_multiplier) <= 1:
            imp_status, imp_detail = "n/a", "single subcatchment - no scaling applied"
        elif self._imperv_field is not None and self._imp_multiplier:
            imp_status = "enabled"
            imp_detail = f"field={self._imperv_field!r}, linear in pervious fraction"
        else:
            imp_status, imp_detail = "disabled", "no imperv_field given - ia_max uniform"
        print(
            f"  {'Imperviousness-preserving ia_max scaling':<40}  "
            f"{imp_status:>12}  {imp_detail}"
        )
        print()
        print("  Calibration metrics:")
        print(f"    CCC   : {self.cal_ccc:.6f}")
        print(f"    KGE   : {self.cal_kge:.6f}")
        print(f"    NSE   : {self.cal_nse:.6f}")
        print(f"    PBIAS : {self.cal_pbias:+.2f}%")

        season = getattr(self, "cal_season_pbias", {}) or {}
        if season:
            cells = "   ".join(
                f"{k} {season[k]:+.1f}%" for k in ("DJF", "MAM", "JJA", "SON")
                if k in season
            )
            print(f"    PBIAS by season : {cells}")
            worst = getattr(self, "cal_worst_season_pbias", float("nan"))
            if math.isfinite(worst) and abs(worst) > self.seasonal_pbias_max:
                print(f"    *** WARNING — worst season "
                      f"({self.cal_worst_season}) is {worst:+.1f}%, beyond "
                      f"±{self.seasonal_pbias_max:g}%.")
                print(f"        The overall {self.cal_pbias:+.2f}% is seasonal "
                      f"cancellation, not volume balance. Expect volume drift")
                print( "        in any period where the seasons stop "
                       "offsetting each other.")

        pinned = getattr(self, "pinned_params", []) or []
        if pinned:
            print(f"    *** WARNING — {len(pinned)} parameter(s) within "
                  f"{self.bound_warn_pct:g}% of a search bound: "
                  f"{', '.join(pinned)}")
            print( "        May be physically real, but can indicate the "
                   "optimizer is absorbing a")
            print( "        structural deficiency. Review before relying on "
                   "extrapolation.")

    # -----------------------------------------------------------------------
    # JSON export
    # -----------------------------------------------------------------------

    def save_params(self, path: str) -> None:
        """
        Save the best calibrated parameters and metrics to a JSON file.

        The output format is compatible with ``load_warmstart()`` so a
        previous calibration can seed a new run.

        Parameters
        ----------
        path : str
            Output file path (e.g. ``'artifacts/best_params.json'``).
        """
        if self.best_params is None:
            raise RuntimeError("Run optimize() first.")

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        payload = {
            "parameters": {k: round(v, 6) for k, v in self.best_params.items()},
            "objectives": {
                "cal_ccc":   round(self.cal_ccc,   6),
                "cal_kge":   round(self.cal_kge,   6),
                "cal_nse":   round(self.cal_nse,   6),
                "cal_pbias": round(self.cal_pbias,  4),
                "cal_pbias_by_season": {
                    k: round(v, 4)
                    for k, v in (getattr(self, "cal_season_pbias", {}) or {}).items()
                } or None,
                "cal_worst_season":       getattr(self, "cal_worst_season", None),
                "cal_worst_season_pbias": (
                    round(self.cal_worst_season_pbias, 4)
                    if math.isfinite(getattr(self, "cal_worst_season_pbias",
                                             float("nan")))
                    else None
                ),
            },
            "seed_selection": {
                "rule":            self.seed_selection,
                "seed_override":   self.seed_override,
                "seasonal_weight": self.seasonal_weight,
                "note":            self.seed_selection_note or None,
                "selected_seed":   self.best_seed,
                "n_seeds":         len(self._seeds),
                "pbias_max":       self.pbias_max,
                "seasonal_pbias_max": self.seasonal_pbias_max,
                # Screening is calibration-only; the validation window is
                # never used to choose a seed, so it stays a blind holdout.
                "windows_used":    "calibration only",
                "pinned_params":   getattr(self, "pinned_params", []) or None,
                "table": [
                    {
                        "seed":  r["seed"],
                        "kge":   round(r["kge"],   6),
                        "ccc":   round(r["ccc"],   6),
                        "nse":   round(r["nse"],   6),
                        "pbias": round(r["pbias"], 4),
                        "fit":   round(r["fit"],   6),
                        "balanced": round(r["bal"], 6),
                        "worst_season":       r["worst_season"],
                        "worst_season_pbias": (
                            round(r["worst_season_pbias"], 4)
                            if math.isfinite(r["worst_season_pbias"]) else None
                        ),
                        "passed": r["passed"],
                        "pinned": r["pinned"] or None,
                    }
                    for r in self.seed_screen
                ] or None,
            },
            "metadata": {
                "model_name":     self.model.name,
                "area_acres":     self.model.area_acres,
                "n_models":       self.model.n_models,
                "n_triangles":    self.model.n_triangles,
                "objective_mode": self.objective_mode,
                "sampler":        self.sampler_type,
                "n_trials":       self.n_trials,
                "cal_start":      self._cal_start,
                "cal_end":        self._cal_end,
                "val_start":      self._val_start,
                "val_end":        self._val_end,
                "bc_inflow":      " + ".join(
                                      c["label"] or "(unnamed series)"
                                      for c in self._bc_components
                                  ) if self._bc_components else None,
                "bc_lag_days":    float(self._bc_components[0]["lag_days"])
                                  if len(self._bc_components) == 1 else None,
                "bc_inflows":     [
                                      {"label": c["label"], "lag_days": c["lag_days"]}
                                      for c in self._bc_components
                                  ] or None,
                "param_bound_overrides": self._param_bound_overrides or None,
                "best_trial":     getattr(self.best_trial, "number", None),
                "saved_at":       _dt.utcnow().isoformat(timespec="seconds") + "Z",
            },
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(f"Parameters saved → {path}")

    # -----------------------------------------------------------------------
    # Diagnostic plots
    # -----------------------------------------------------------------------

    def plot_diagnostics(self, prefix: str = "", param_importances: bool = False) -> list:
        """
        Produce optimizer diagnostic plots appropriate for the active backend.

        Works with every supported sampler.
        Returns a list of Plotly figures so the caller can display or further
        customise them.

        Produces up to two figures:

        - *Convergence* — number of Pareto solutions per generation (or, for
          single-objective, best objective value per generation).
        - *Pareto front* (multi-objective) — scatter of the final Pareto
          front; if only one objective was used, a horizontal bar chart of
          the single solution's metrics is shown instead.

        Parameters
        ----------
        prefix : str
            String prepended to output filenames, e.g. the model name.
            Default ``''``.
        param_importances : bool
            Retained for call compatibility. True emits a warning because
            parameter-importance plots are unavailable. Default False.

        Returns
        -------
        list of (fig, filename) tuples
            Each element is a ``(plotly.graph_objects.Figure, str)`` pair.
            Call ``save(fig, fname).show()`` on each in the notebook cell so
            VS Code renders the figure inline (matching the rest of the notebook).
        """
        import plotly.graph_objects as go
        import plotly.express as px

        if param_importances:
            warnings.warn("Parameter importance plots are not supported by this solver backend",
                          stacklevel=2)

        if self.study is None:
            raise RuntimeError("Run optimize() before calling plot_diagnostics().")

        figs = []

        def _save(fig, name):
            fname = f"{prefix}{name}" if prefix else name
            figs.append((fig, fname))

        # Remove unused save_fn parameter — saving is handled by the caller.

        obj_labels = [OBJECTIVE_REGISTRY[k][2] for k in self._obj_keys]
        title_base = f"{self.model.name or 'Model'} — {self.objective_mode} ({self.sampler_type})"

        result = self.study  # CalibrationResult object

        # ── Convergence plot ─────────────────────────────────────────────
        if result.history:
            df_hist = result.to_pareto_dataframe()
            # n_pareto per generation across all evaluated solutions.
            gen_pareto = (
                df_hist.groupby("generation")["is_pareto"]
                .sum()
                .reset_index()
                .rename(columns={"is_pareto": "n_pareto"})
            )
            if self._is_multi:
                # Multi-objective: track Pareto front size over generations.
                fig_c = go.Figure(
                    go.Scatter(
                        x=gen_pareto["generation"],
                        y=gen_pareto["n_pareto"],
                        mode="lines+markers",
                        name="Pareto solutions",
                    )
                )
                fig_c.update_layout(
                    title=f"{title_base} — Pareto Front Size per Generation",
                    xaxis_title="Generation",
                    yaxis_title="# Pareto solutions",
                    width=1200, height=500,
                )
            else:
                # Single-objective: track best objective value per generation.
                obj_name = result.objective_names[0]
                minimize = result.minimize_flags[0]
                best_per_gen = (
                    df_hist.groupby("generation")[obj_name]
                    .agg("min" if minimize else "max")
                    .reset_index()
                )
                values = best_per_gen[obj_name]
                best_per_gen[obj_name] = values.cummin() if minimize else values.cummax()
                fig_c = go.Figure(
                    go.Scatter(
                        x=best_per_gen["generation"],
                        y=best_per_gen[obj_name],
                        mode="lines+markers",
                        name=obj_name,
                    )
                )
                fig_c.update_layout(
                    title=f"{title_base} — Best {obj_labels[0]} per Generation",
                    xaxis_title="Generation",
                    yaxis_title=obj_labels[0],
                    width=1200, height=500,
                )
            _save(fig_c, "sh_convergence.html")

        # ── Pareto / solution plot ───────────────────────────────────────
        disp = result.objective_display_values()   # (n_sol, n_obj), un-negated
        n_obj = len(result.objective_names)

        if n_obj == 1:
            # Single-objective: bar chart of the one solution's metrics.
            met_names = ["CCC", "KGE", "NSE", "PBIAS (%)"]
            met_vals  = [self.cal_ccc, self.cal_kge, self.cal_nse, self.cal_pbias]
            fig_s = go.Figure(
                go.Bar(x=met_names, y=met_vals, marker_color="#4C72B0")
            )
            fig_s.update_layout(
                title=f"{title_base} — Calibration Metrics",
                yaxis_title="Value",
                width=700, height=450,
            )
            _save(fig_s, "sh_metrics.html")

        elif n_obj == 2:
            # Two-objective: scatter of Pareto front.
            x_name, y_name = result.objective_names
            x_lbl,  y_lbl  = obj_labels[0], obj_labels[1]
            fig_pf = go.Figure(
                go.Scatter(
                    x=disp[:, 0], y=disp[:, 1],
                    mode="markers",
                    marker=dict(size=8, color="#4C72B0"),
                    text=[f"sol {i}" for i in range(len(disp))],
                )
            )
            # Mark the selected best solution.
            best_x = np.array([
                self.best_params[n] for n in result.param_names
            ])
            # Find the matching row in pareto_X.
            dists = np.linalg.norm(result.pareto_X - best_x, axis=1)
            bi = int(np.argmin(dists))
            fig_pf.add_trace(go.Scatter(
                x=[disp[bi, 0]], y=[disp[bi, 1]],
                mode="markers",
                marker=dict(size=14, color="red", symbol="star"),
                name="Selected",
            ))
            fig_pf.update_layout(
                title=f"{title_base} — Pareto Front",
                xaxis_title=x_lbl,
                yaxis_title=y_lbl,
                width=900, height=650,
            )
            _save(fig_pf, "sh_pareto.html")

        else:
            # 3+ objectives: scatter matrix of all objective pairs.
            import pandas as _pd
            df_pf = _pd.DataFrame(disp, columns=obj_labels)
            fig_sm = px.scatter_matrix(
                df_pf,
                dimensions=obj_labels,
                title=f"{title_base} — Pareto Front (scatter matrix)",
            )
            fig_sm.update_layout(width=1000, height=1000)
            _save(fig_sm, "sh_pareto_matrix.html")

        return figs

    # -----------------------------------------------------------------------
    # Class-level reference helpers
    # -----------------------------------------------------------------------

    @classmethod
    def list_objectives(cls) -> None:
        """
        Print a reference table of all available objective keys.

        Shows the key to pass to the ``objective`` parameter, the calibration
        direction (maximize/minimize), the display name, and a short
        description.  Any single key can be used alone (single-objective)
        or combined in a list or ``'+'``-separated string (multi-objective
        Pareto).

        Examples
        --------
        >>> CalOptimizer.list_objectives()

        Single-objective::

            CalOptimizer(model=m, objective='KGE', ...)

        Multi-objective::

            CalOptimizer(model=m, objective=['KGE', 'pbias'], ...)
            CalOptimizer(model=m, objective='CCC+pbias+volume_relative_error', ...)
        """
        # Deduplicate (aliases point to the same tuple) and sort by direction
        # then name for a clean two-section table.
        seen = set()
        entries = []
        for key, val in OBJECTIVE_REGISTRY.items():
            fn, minimize, display, desc = val
            if fn in seen:
                continue  # skip aliases
            seen.add(fn)
            entries.append((key, minimize, display, desc))

        entries.sort(key=lambda e: (e[1], e[2]))  # minimize=False first, then True

        w_key  = max(len(e[0]) for e in entries) + 2
        w_dir  = 10
        w_name = max(len(e[2]) for e in entries) + 2

        header = f"  {'Key':<{w_key}}  {'Direction':<{w_dir}}  {'Name':<{w_name}}  Description"
        print("Available calibration objectives (from sparsehydro + swmmrdii):")
        print()
        print("  — maximize objectives (higher = better) —")
        print(header)
        print("  " + "-" * (len(header) - 2))
        for key, minimize, display, desc in entries:
            if not minimize:
                print(f"  {key:<{w_key}}  {'maximize':<{w_dir}}  {display:<{w_name}}  {desc}")

        print()
        print("  — minimize objectives (lower = better) —")
        print(header)
        print("  " + "-" * (len(header) - 2))
        for key, minimize, display, desc in entries:
            if minimize:
                print(f"  {key:<{w_key}}  {'minimize':<{w_dir}}  {display:<{w_name}}  {desc}")

        print()
        print("  Pass a single key for single-objective, or a list / '+'-joined string")
        print("  for multi-objective Pareto (NSGA-II forced automatically).")
        print("  Examples:  objective='KGE'")
        print("             objective=['KGE', 'pbias']")
        print("             objective='CCC+pbias'")
        print("             objective=['KGE', 'pbias', 'peak_weighted_mse']")

    @classmethod
    def list_samplers(cls) -> None:
        """
        Print a reference table of all available sampler keys and their backends.

        Solvers use pymoo, scipy or Platypus. CMA-ES counts model
        evaluations per seed; sh_nsga2 counts generations. TPE is removed.

        Examples
        --------
        >>> CalOptimizer.list_samplers()

        Usage::

            CalOptimizer(model=m, sampler='CMA-ES', ...)        # calibration
            CalOptimizer(model=m, sampler='scipy',  ...)        # sparsehydro
            CalOptimizer(model=m, sampler='sh_nsga2', ...)      # sparsehydro
        """
        sh_entries = [
            (k, v) for k, v in SAMPLER_REGISTRY.items() if v[0] == "sparsehydro"
        ]

        w_key  = max(len(k) for k in SAMPLER_REGISTRY) + 2
        w_name = max(v[1].__len__() for v in SAMPLER_REGISTRY.values()) + 2

        def _row(k, v):
            _, name, single, multi, desc = v
            modes = ("single" if single else "") + ("" if not multi else "+multi")
            print(f"  {k:<{w_key}}  {name:<{w_name}}  [{modes:<12}]  {desc}")

        print("Available samplers:")
        print()
        print("  -- sparsehydro backend (pymoo / scipy / platypus) -----------------------")
        for k, v in sh_entries:
            _row(k, v)
        print()
        print("  NSGAII aliases use sh_nsga2. CMA-ES and scipy require one objective.")
