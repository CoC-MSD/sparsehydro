"""Objective function abstractions for model calibration.

:class:`IObjective` defines the interface that every calibration objective
must implement.  Concrete subclasses encapsulate both the direction of
optimisation (``minimize``) and the computation of a scalar score from
observed vs predicted arrays.

Built-in objectives
-------------------

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Class
     - minimize
     - Description
   * - :class:`MSE`
     - True
     - Mean squared error
   * - :class:`RMSE`
     - True
     - Root mean squared error
   * - :class:`MAE`
     - True
     - Mean absolute error
   * - :class:`PeakWeightedMSE`
     - True
     - Flow-weighted MSE (peaks penalised)
   * - :class:`PBIAS`
     - True
     - Absolute percent bias (volume balance)
   * - :class:`VolumeRelativeError`
     - True
     - Relative total runoff volume error
   * - :class:`NashSutcliffe`
     - False
     - Nash–Sutcliffe efficiency
   * - :class:`LogNSE`
     - False
     - NSE on log-transformed flows
   * - :class:`KGE`
     - False
     - Kling–Gupta efficiency
   * - :class:`IndexOfAgreement`
     - False
     - Willmott's Index of Agreement
   * - :class:`ConcordanceCorrelationCoefficient`
     - False
     - Lin's Concordance Correlation Coefficient

Custom objectives
-----------------
Subclass :class:`IObjective`, set ``name`` and ``minimize``, and implement
:meth:`evaluate`::

    class MyObjective(IObjective):
        name = "my_obj"
        minimize = True

        def evaluate(self, observed, predicted):
            return float(np.sum(np.abs(observed - predicted)))
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np


class IObjective(ABC):
    """Abstract calibration objective.

    Subclasses must set the class variables :attr:`name` and :attr:`minimize`
    and implement :meth:`evaluate`.

    :cvar name: Short snake_case identifier (used as column name in result
        DataFrames).
    :cvar minimize: ``True`` if lower scores are better; ``False`` for
        objectives like NSE where higher is better.
    """

    name: ClassVar[str]
    minimize: ClassVar[bool]

    def __init__(self, mask: np.ndarray | None = None) -> None:
        """Store an optional per-objective boolean mask.

        :param mask: Boolean array selecting the timesteps this objective is
            computed over.  Applied by :meth:`compute` unless overridden by an
            explicit ``mask`` argument.  ``None`` (default) uses all values.
            Column-name / callable mask specifications are resolved by
            :class:`~sparsehydro.calibration.problem.CalibrationProblem`, not
            here — pass a boolean array when using an objective standalone.
        :type mask: numpy.ndarray or None
        """
        self.mask = mask

    @abstractmethod
    def evaluate(self, observed: np.ndarray, predicted: np.ndarray) -> float:
        """Compute a scalar score from observed and predicted arrays.

        :param observed: 1-D array of observed values.
        :type observed: numpy.ndarray
        :param predicted: 1-D array of predicted values (same shape).
        :type predicted: numpy.ndarray
        :returns: Scalar objective value.
        :rtype: float
        :raises ValueError: If arrays are incompatible or the score is
            undefined (e.g. constant observed series for NSE).
        """

    def compute(
        self,
        observed: np.ndarray,
        predicted: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> float:
        """Evaluate the objective over a selected subset of timesteps.

        Positions where ``observed`` or ``predicted`` is ``NaN`` are **always**
        excluded, in addition to any boolean mask.  The *effective* mask is the
        explicit ``mask`` argument when supplied, otherwise the objective's own
        :attr:`mask` (an explicit argument wins — this lets
        :class:`~sparsehydro.calibration.problem.CalibrationProblem` apply a
        problem-level default while honouring per-objective overrides).

        :param observed: 1-D array of observed values.
        :type observed: numpy.ndarray
        :param predicted: 1-D array of predicted values (same shape).
        :type predicted: numpy.ndarray
        :param mask: Boolean array of the same length as *observed*.  Only
            positions where ``mask`` is ``True`` are included.  ``None``
            (default) falls back to :attr:`mask`.
        :type mask: numpy.ndarray or None
        :returns: Scalar objective value computed over the selected subset.
        :rtype: float
        :raises TypeError: If the effective mask is a column name or callable
            (those must be resolved via ``CalibrationProblem``).
        :raises ValueError: If the mask length does not match *observed*, or if
            no elements remain after masking and NaN removal.
        """
        obs = np.asarray(observed, dtype=float)
        pred = np.asarray(predicted, dtype=float)
        if obs.shape != pred.shape:
            raise ValueError(
                f"Shape mismatch: observed {obs.shape} vs predicted {pred.shape}"
            )

        effective = mask if mask is not None else self.mask

        # Always drop NaN positions in either series.
        selector = ~(np.isnan(obs) | np.isnan(pred))

        if effective is not None:
            if isinstance(effective, str) or callable(effective):
                raise TypeError(
                    "A column-name or callable mask cannot be applied directly; "
                    "resolve it through CalibrationProblem, or pass a boolean array."
                )
            m = np.asarray(effective, dtype=bool)
            if m.shape != obs.shape:
                raise ValueError(
                    f"mask shape {m.shape} does not match observed shape {obs.shape}"
                )
            selector &= m

        if not selector.any():
            raise ValueError(
                "mask (and NaN removal) selects no elements; cannot compute objective."
            )
        return self.compute_selected(
            obs[selector], pred[selector], self.prepare_observed(obs[selector], selector)
        )

    # ------------------------------------------------------------------
    # Pre-selected fast path
    # ------------------------------------------------------------------

    def prepare_observed(self, observed: np.ndarray, selector: np.ndarray) -> Any:
        """Precompute whatever this objective can derive from *observed* alone.

        :class:`~sparsehydro.calibration.problem.CalibrationProblem` calls this
        once at construction and hands the result back to every
        :meth:`compute_selected` call, so observed-only work (weights, variance,
        means) is not redone on every parameter evaluation.

        The returned context must be treated as immutable: one context is shared
        across every evaluation and across problem copies.

        :param observed: Observed values, already NaN- and mask-filtered.
        :type observed: numpy.ndarray
        :param selector: Boolean selector that produced *observed*, for
            objectives carrying full-length auxiliary arrays (e.g. weights).
        :type selector: numpy.ndarray
        :returns: Opaque context passed to :meth:`compute_selected`; ``None``
            when the objective needs none.
        :rtype: typing.Any
        """
        return None

    def compute_selected(
        self,
        observed: np.ndarray,
        predicted: np.ndarray,
        context: Any = None,
    ) -> float:
        """Score already-selected arrays, skipping NaN and mask filtering.

        Both inputs must already be NaN-free and mask-filtered.  This is the
        method the calibration hot path calls; :meth:`compute` does the
        filtering and then delegates here, so the two can never disagree.

        :param observed: Selected observed values.
        :type observed: numpy.ndarray
        :param predicted: Selected predicted values (same shape).
        :type predicted: numpy.ndarray
        :param context: Value returned by :meth:`prepare_observed`.
        :type context: typing.Any
        :returns: Scalar objective value.
        :rtype: float
        """
        return self.evaluate(observed, predicted)


# ---------------------------------------------------------------------------
# Error metrics (minimize = True)
# ---------------------------------------------------------------------------

class MSE(IObjective):
    """Mean squared error.

    :cvar name: ``"mse"``
    :cvar minimize: ``True``
    """

    name = "mse"
    minimize = True

    def evaluate(self, observed: np.ndarray, predicted: np.ndarray) -> float:
        obs = np.asarray(observed, dtype=float)
        pred = np.asarray(predicted, dtype=float)
        if obs.shape != pred.shape:
            raise ValueError(
                f"Shape mismatch: observed {obs.shape} vs predicted {pred.shape}"
            )
        return float(np.mean((obs - pred) ** 2))


class RMSE(IObjective):
    """Root mean squared error.

    :cvar name: ``"rmse"``
    :cvar minimize: ``True``
    """

    name = "rmse"
    minimize = True

    def evaluate(self, observed: np.ndarray, predicted: np.ndarray) -> float:
        obs = np.asarray(observed, dtype=float)
        pred = np.asarray(predicted, dtype=float)
        if obs.shape != pred.shape:
            raise ValueError(
                f"Shape mismatch: observed {obs.shape} vs predicted {pred.shape}"
            )
        return float(np.sqrt(np.mean((obs - pred) ** 2)))


class WeightedRMSE(IObjective):
    """Weighted root mean squared error with per-timestep weights.

    ``wrmse = sqrt( Σ w_i (obs_i - pred_i)² / Σ w_i )``

    Used for zonal (peak vs tail) weighting in sequential unit-hydrograph
    fitting: assign a larger weight to the peak zone so the fitted UH reproduces
    the crest shape.  When no *weights* are supplied it reduces to unweighted
    :class:`RMSE`.

    :param weights: Full-length per-timestep weight vector aligned with the
        observed/predicted arrays passed to :meth:`compute`.  ``None`` (default)
        falls back to uniform weighting.
    :type weights: numpy.ndarray or None
    :param mask: Optional per-objective boolean mask (see :class:`IObjective`).
    :type mask: numpy.ndarray or None
    """

    name = "wrmse"
    minimize = True

    def __init__(self, weights: np.ndarray | None = None, mask: np.ndarray | None = None) -> None:
        super().__init__(mask=mask)
        self.weights = None if weights is None else np.asarray(weights, dtype=float)

    def evaluate(self, observed: np.ndarray, predicted: np.ndarray) -> float:
        """Unweighted RMSE fallback (used when no weights are set).

        :param observed: 1-D array of observed values.
        :type observed: numpy.ndarray
        :param predicted: 1-D array of predicted values (same shape).
        :type predicted: numpy.ndarray
        :returns: Unweighted root mean squared error.
        :rtype: float
        """
        obs = np.asarray(observed, dtype=float)
        pred = np.asarray(predicted, dtype=float)
        if obs.shape != pred.shape:
            raise ValueError(
                f"Shape mismatch: observed {obs.shape} vs predicted {pred.shape}"
            )
        return float(np.sqrt(np.mean((obs - pred) ** 2)))

    def prepare_observed(self, observed: np.ndarray, selector: np.ndarray) -> Any:
        """Select and validate the weight vector for this selector.

        Weights are full-length and aligned with the unfiltered series, so they
        must be subset by the same selector -- doing it once here keeps it off
        the hot path.

        :param observed: Selected observed values.
        :type observed: numpy.ndarray
        :param selector: Boolean selector that produced *observed*.
        :type selector: numpy.ndarray
        :returns: Clipped, selected weights, or ``None`` to fall back to
            unweighted RMSE (no weights set, wrong length, or non-positive sum).
        :rtype: numpy.ndarray or None
        :raises ValueError: If the weight vector length does not match *selector*.
        """
        if self.weights is None:
            return None
        w = self.weights
        if w.shape != selector.shape:
            raise ValueError(
                f"weights shape {w.shape} does not match observed shape {selector.shape}"
            )
        w_sel = np.clip(w[selector], 0.0, None)
        if float(np.sum(w_sel)) <= 0.0:
            return None
        return w_sel

    def compute_selected(
        self,
        observed: np.ndarray,
        predicted: np.ndarray,
        context: Any = None,
    ) -> float:
        """Compute weighted RMSE over already-selected arrays.

        :param observed: Selected observed values.
        :type observed: numpy.ndarray
        :param predicted: Selected predicted values (same shape).
        :type predicted: numpy.ndarray
        :param context: Selected weights from :meth:`prepare_observed`, or
            ``None`` for the unweighted fallback.
        :type context: numpy.ndarray or None
        :returns: Weighted root mean squared error.
        :rtype: float
        """
        e = observed - predicted
        if context is None:
            return float(np.sqrt(np.mean(e ** 2)))
        wsum = float(np.sum(context))
        return float(np.sqrt(float(np.sum(context * e ** 2)) / wsum))


class MAE(IObjective):
    """Mean absolute error.

    :cvar name: ``"mae"``
    :cvar minimize: ``True``
    """

    name = "mae"
    minimize = True

    def evaluate(self, observed: np.ndarray, predicted: np.ndarray) -> float:
        obs = np.asarray(observed, dtype=float)
        pred = np.asarray(predicted, dtype=float)
        if obs.shape != pred.shape:
            raise ValueError(
                f"Shape mismatch: observed {obs.shape} vs predicted {pred.shape}"
            )
        return float(np.mean(np.abs(obs - pred)))


class PeakWeightedMSE(IObjective):
    """Peak-weighted mean squared error.

    Weights each squared residual by the observed flow at that step relative
    to the mean observed flow, raised to ``power``, penalising errors during
    high-flow events.

    .. math::

        w_t = (Q_{obs,t} / \\bar{Q}_{obs})^{p}

        PWMSE = \\frac{\\sum_t w_t (Q_{obs,t} - Q_{pred,t})^2}{\\sum_t w_t}

    ``power=1.0`` (default) gives linear weighting; larger values sharpen the
    focus on peaks.  ``power=0.0`` reduces to plain MSE.

    :param power: Exponent applied to the normalised-flow weights (>= 0).
    :param mask: Optional boolean array selecting the timesteps to score over.
    :cvar name: ``"peak_weighted_mse"``
    :cvar minimize: ``True``
    """

    name = "peak_weighted_mse"
    minimize = True

    def __init__(self, power: float = 1.0, mask: np.ndarray | None = None) -> None:
        super().__init__(mask=mask)
        if power < 0.0:
            raise ValueError(f"power must be >= 0; got {power!r}")
        self.power = float(power)

    def evaluate(self, observed: np.ndarray, predicted: np.ndarray) -> float:
        return _peak_weighted_mse(observed, predicted, power=self.power)


# ---------------------------------------------------------------------------
# Efficiency metrics (minimize = False)
# ---------------------------------------------------------------------------

class NashSutcliffe(IObjective):
    """Nash–Sutcliffe Efficiency (NSE).

    .. math::

        NSE = 1 - \\frac{\\sum_t (Q_{obs,t} - Q_{pred,t})^2}
                        {\\sum_t (Q_{obs,t} - \\bar{Q}_{obs})^2}

    NSE = 1 for a perfect predictor; 0 for the mean-flow predictor;
    negative for worse than the mean.

    :cvar name: ``"nash_sutcliffe"``
    :cvar minimize: ``False``
    """

    name = "nash_sutcliffe"
    minimize = False

    def evaluate(self, observed: np.ndarray, predicted: np.ndarray) -> float:
        return _nash_sutcliffe(observed, predicted)


class KGE(IObjective):
    """Kling–Gupta Efficiency (KGE).

    Decomposes model error into correlation, bias, and variability components:

    .. math::

        KGE = 1 - \\sqrt{(r-1)^2 + (\\alpha-1)^2 + (\\beta-1)^2}

    where :math:`r` is the Pearson correlation, :math:`\\alpha = \\sigma_p /
    \\sigma_o` is the variability ratio, and :math:`\\beta = \\mu_p / \\mu_o`
    is the bias ratio.  KGE = 1 for a perfect predictor.

    :cvar name: ``"kge"``
    :cvar minimize: ``False``

    :raises ValueError: If ``std(observed) == 0`` or ``mean(observed) == 0``.
    """

    name = "kge"
    minimize = False

    def evaluate(self, observed: np.ndarray, predicted: np.ndarray) -> float:
        obs = np.asarray(observed, dtype=float)
        pred = np.asarray(predicted, dtype=float)
        if obs.shape != pred.shape:
            raise ValueError(
                f"Shape mismatch: observed {obs.shape} vs predicted {pred.shape}"
            )
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


class PBIAS(IObjective):
    """Absolute percent bias.

    Measures the tendency of the model to over- or under-predict total volume,
    expressed as a percentage of observed volume.  The absolute value is
    returned so lower is always better.

    .. math::

        PBIAS = 100 \\cdot \\left|
            \\frac{\\sum_t (Q_{obs,t} - Q_{pred,t})}{\\sum_t Q_{obs,t}}
        \\right|

    :cvar name: ``"pbias"``
    :cvar minimize: ``True``
    """

    name = "pbias"
    minimize = True

    def evaluate(self, observed: np.ndarray, predicted: np.ndarray) -> float:
        return _pbias(observed, predicted)


class VolumeRelativeError(IObjective):
    """Relative total volume error.

    Fractional difference between predicted and observed cumulative runoff
    volume.  The absolute value is returned so lower is always better.

    .. math::

        VRE = \\left|
            \\frac{\\sum_t Q_{pred,t} - \\sum_t Q_{obs,t}}{\\sum_t Q_{obs,t}}
        \\right|

    :cvar name: ``"volume_relative_error"``
    :cvar minimize: ``True``
    """

    name = "volume_relative_error"
    minimize = True

    def evaluate(self, observed: np.ndarray, predicted: np.ndarray) -> float:
        return _volume_relative_error(observed, predicted)


class LogNSE(IObjective):
    """Nash–Sutcliffe Efficiency on log-transformed flows (Log-NSE).

    The log transform de-emphasises large peak errors and gives more weight to
    low-flow reproduction — complementary to the standard :class:`NashSutcliffe`
    when both peak and low-flow performance matter.

    .. math::

        \\text{LogNSE} = 1 - \\frac{
            \\sum_t (\\ln(Q_{obs,t}+\\varepsilon) - \\ln(Q_{pred,t}+\\varepsilon))^2
        }{
            \\sum_t (\\ln(Q_{obs,t}+\\varepsilon) -
                     \\overline{\\ln(Q_{obs}+\\varepsilon)})^2
        }

    :param epsilon: Offset added before the log transform to avoid
        ``log(0)``.  Default ``0.01``.
    :type epsilon: float
    :param mask: Optional boolean array selecting the timesteps to score over.
    :cvar name: ``"log_nse"``
    :cvar minimize: ``False``
    """

    name = "log_nse"
    minimize = False

    def __init__(self, epsilon: float = 0.01, mask: np.ndarray | None = None) -> None:
        super().__init__(mask=mask)
        self.epsilon = epsilon

    def evaluate(self, observed: np.ndarray, predicted: np.ndarray) -> float:
        return _log_nse(observed, predicted, epsilon=self.epsilon)


class IndexOfAgreement(IObjective):
    """Willmott's Index of Agreement (d).

    A dimensionless goodness-of-fit measure in [0, 1] that is less sensitive
    to large errors than NSE while still rewarding accurate reproduction of
    timing, phase, and magnitude.

    .. math::

        d = 1 - \\frac{\\sum_t (Q_{obs,t} - Q_{pred,t})^2}
                      {\\sum_t (|Q_{pred,t} - \\bar{Q}_{obs}| +
                                |Q_{obs,t}  - \\bar{Q}_{obs}|)^2}

    :cvar name: ``"index_of_agreement"``
    :cvar minimize: ``False``
    """

    name = "index_of_agreement"
    minimize = False

    def evaluate(self, observed: np.ndarray, predicted: np.ndarray) -> float:
        return _index_of_agreement(observed, predicted)


class ConcordanceCorrelationCoefficient(IObjective):
    """Lin's Concordance Correlation Coefficient (CCC).

    Combines precision (Pearson correlation) and accuracy (closeness to the
    perfect-concordance line) into a single coefficient bounded to [−1, 1]:

    .. math::

        \\rho_c = \\frac{2\\,\\sigma_{op}}
                        {\\sigma_o^2 + \\sigma_p^2 + (\\mu_o - \\mu_p)^2}

    CCC = 1 for perfect agreement between observed and predicted.  Unlike
    Pearson correlation alone, CCC penalises systematic bias and scale
    differences, making it sensitive to both random and systematic error.

    :cvar name: ``"ccc"``
    :cvar minimize: ``False``

    :raises ValueError: If both series are constant and equal (denominator zero).
    """

    name = "ccc"
    minimize = False

    def evaluate(self, observed: np.ndarray, predicted: np.ndarray) -> float:
        return _concordance_correlation_coefficient(observed, predicted)


class DisjointWindowCCC(IObjective):
    """Composite CCC over two disjoint windows on ONE observed/predicted pair.

    Grades the same ``(observed, predicted)`` arrays on ``dry_mask`` and
    ``storm_mask`` separately and returns ``w_dry*CCC_dry + w_storm*CCC_storm``.
    Overrides :meth:`compute` to apply its own masks, ignoring any problem-level
    mask (each window scores the whole-flow signal on a different portion of the
    record — dry windows grade the baseflow, storm windows grade the peaks).

    Defined at package scope (not in a notebook cell) so ``cloudpickle`` can
    serialise a :class:`~sparsehydro.calibration.problem.CalibrationProblem`
    carrying it into ``loky`` worker processes — the prerequisite for the
    multi-seed *parallel* CMA-ES path (``OptunaCmaEsSolver(n_jobs>1)``).

    :param dry_mask: Boolean mask selecting the dry (baseflow) scoring window.
    :param storm_mask: Boolean mask selecting the storm (peak) scoring window.
    :param w_dry: Weight on the dry-window CCC.
    :param w_storm: Weight on the storm-window CCC.
    """

    name = "disjoint_ccc"
    minimize = False

    def __init__(self, dry_mask, storm_mask, w_dry=0.5, w_storm=0.5, mask=None):
        super().__init__(mask=mask)
        self._dry = np.asarray(dry_mask, dtype=bool)
        self._storm = np.asarray(storm_mask, dtype=bool)
        self._w_dry = float(w_dry)
        self._w_storm = float(w_storm)
        self._ccc = ConcordanceCorrelationCoefficient()

    def evaluate(self, observed, predicted):        # only hit if called mask-less
        return self._ccc.evaluate(observed, predicted)

    def compute(self, observed, predicted, mask=None):
        cd = self._ccc.compute(observed, predicted, mask=self._dry)
        cs = self._ccc.compute(observed, predicted, mask=self._storm)
        return self._w_dry * cd + self._w_storm * cs


# ---------------------------------------------------------------------------
# Pure function implementations (no external dependencies)
# ---------------------------------------------------------------------------

def _peak_weighted_mse(
    observed: np.ndarray,
    predicted: np.ndarray,
    power: float = 1.0,
) -> float:
    """Compute the peak-weighted mean squared error.

    :param observed: 1-D array of observed values.
    :type observed: numpy.ndarray
    :param predicted: 1-D array of predicted values (same shape).
    :type predicted: numpy.ndarray
    :param power: Exponent applied to the normalised-flow weights (>= 0).
    :type power: float
    :returns: Peak-weighted MSE.
    :rtype: float
    :raises ValueError: If shapes mismatch, *power* < 0, or observed is all zeros.
    """
    obs = np.asarray(observed, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    if obs.shape != pred.shape:
        raise ValueError(f"Shape mismatch: observed {obs.shape} vs predicted {pred.shape}")
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


def _nash_sutcliffe(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Compute the Nash-Sutcliffe efficiency.

    :param observed: 1-D array of observed values.
    :type observed: numpy.ndarray
    :param predicted: 1-D array of predicted values (same shape).
    :type predicted: numpy.ndarray
    :returns: Nash-Sutcliffe efficiency (1 is perfect).
    :rtype: float
    :raises ValueError: If shapes mismatch or the observed series is constant.
    """
    obs = np.asarray(observed, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    if obs.shape != pred.shape:
        raise ValueError(f"Shape mismatch: observed {obs.shape} vs predicted {pred.shape}")
    ss_res = float(np.sum((obs - pred) ** 2))
    ss_tot = float(np.sum((obs - np.mean(obs)) ** 2))
    if ss_tot == 0.0:
        raise ValueError("SS_tot is zero; observed series is constant — NSE is undefined.")
    return 1.0 - ss_res / ss_tot


def _pbias(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Compute the absolute percent bias.

    :param observed: 1-D array of observed values.
    :type observed: numpy.ndarray
    :param predicted: 1-D array of predicted values (same shape).
    :type predicted: numpy.ndarray
    :returns: Absolute percent bias (0 is perfect).
    :rtype: float
    :raises ValueError: If shapes mismatch or ``sum(observed)`` is zero.
    """
    obs = np.asarray(observed, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    if obs.shape != pred.shape:
        raise ValueError(f"Shape mismatch: observed {obs.shape} vs predicted {pred.shape}")
    total_obs = float(np.sum(obs))
    if total_obs == 0.0:
        raise ValueError("sum(observed) is zero; PBIAS is undefined.")
    return float(abs(100.0 * np.sum(obs - pred) / total_obs))


def _log_nse(
    observed: np.ndarray,
    predicted: np.ndarray,
    epsilon: float = 0.01,
) -> float:
    """Compute the Nash-Sutcliffe efficiency on log-transformed flows.

    :param observed: 1-D array of observed values.
    :type observed: numpy.ndarray
    :param predicted: 1-D array of predicted values (same shape).
    :type predicted: numpy.ndarray
    :param epsilon: Offset added before the log transform to avoid ``log(0)``.
    :type epsilon: float
    :returns: Log-space Nash-Sutcliffe efficiency (1 is perfect).
    :rtype: float
    :raises ValueError: If shapes mismatch, *epsilon* <= 0, or the
        log-transformed observed series is constant.
    """
    obs = np.asarray(observed, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    if obs.shape != pred.shape:
        raise ValueError(f"Shape mismatch: observed {obs.shape} vs predicted {pred.shape}")
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be > 0; got {epsilon!r}")
    log_obs = np.log(obs + epsilon)
    log_pred = np.log(np.clip(pred, 0.0, None) + epsilon)
    ss_res = float(np.sum((log_obs - log_pred) ** 2))
    ss_tot = float(np.sum((log_obs - np.mean(log_obs)) ** 2))
    if ss_tot == 0.0:
        raise ValueError("SS_tot is zero; log-transformed observed series is constant.")
    return float(1.0 - ss_res / ss_tot)


def _volume_relative_error(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Compute the relative total volume error.

    :param observed: 1-D array of observed values.
    :type observed: numpy.ndarray
    :param predicted: 1-D array of predicted values (same shape).
    :type predicted: numpy.ndarray
    :returns: Absolute relative volume error (0 is perfect).
    :rtype: float
    :raises ValueError: If shapes mismatch or ``sum(observed)`` is zero.
    """
    obs = np.asarray(observed, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    if obs.shape != pred.shape:
        raise ValueError(f"Shape mismatch: observed {obs.shape} vs predicted {pred.shape}")
    total_obs = float(np.sum(obs))
    if total_obs == 0.0:
        raise ValueError("sum(observed) is zero; volume relative error is undefined.")
    return float(abs(float(np.sum(pred)) - total_obs) / total_obs)


def _concordance_correlation_coefficient(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Compute Lin's concordance correlation coefficient.

    :param observed: 1-D array of observed values.
    :type observed: numpy.ndarray
    :param predicted: 1-D array of predicted values (same shape).
    :type predicted: numpy.ndarray
    :returns: Concordance correlation coefficient in [-1, 1] (1 is perfect).
    :rtype: float
    :raises ValueError: If shapes mismatch or the denominator is zero.
    """
    obs = np.asarray(observed, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    if obs.shape != pred.shape:
        raise ValueError(f"Shape mismatch: observed {obs.shape} vs predicted {pred.shape}")
    mean_obs = float(np.mean(obs))
    mean_pred = float(np.mean(pred))
    var_obs = float(np.var(obs))
    var_pred = float(np.var(pred))
    covariance = float(np.mean((obs - mean_obs) * (pred - mean_pred)))
    denominator = var_obs + var_pred + (mean_obs - mean_pred) ** 2
    if denominator == 0.0:
        raise ValueError("Denominator is zero; CCC is undefined.")
    return float(2.0 * covariance / denominator)


def _index_of_agreement(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Compute Willmott's index of agreement.

    :param observed: 1-D array of observed values.
    :type observed: numpy.ndarray
    :param predicted: 1-D array of predicted values (same shape).
    :type predicted: numpy.ndarray
    :returns: Index of agreement in [0, 1] (1 is perfect).
    :rtype: float
    :raises ValueError: If shapes mismatch or the potential error is zero.
    """
    obs = np.asarray(observed, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    if obs.shape != pred.shape:
        raise ValueError(f"Shape mismatch: observed {obs.shape} vs predicted {pred.shape}")
    mean_obs = float(np.mean(obs))
    ss_res = float(np.sum((obs - pred) ** 2))
    ss_pot = float(np.sum((np.abs(pred - mean_obs) + np.abs(obs - mean_obs)) ** 2))
    if ss_pot == 0.0:
        raise ValueError("Potential error (SS_pot) is zero; Index of Agreement is undefined.")
    return float(1.0 - ss_res / ss_pot)


class Con15(IObjective):
    """Absolute error of a zero-intercept event-peak regression slope.

    Rank complete events by observed peak, select
    ``ceil(top_fraction * event_count)``, and minimize ``abs(slope - 1)``.
    ``top_fraction=0.15`` reproduces the Con15 selection; ``1.0`` uses all
    eligible events. The fraction must be finite and in ``(0, 1]``.

    ``event_windows`` is an integer array of inclusive ``[start_idx, end_idx]``
    pairs on the original time axis, ordered and disjoint by default.
    ``allow_overlapping_events=True`` preserves approved overlapping windows
    as independent peak observations. Only events wholly
    inside the effective mask count. Unlike timestep objectives, this class
    never compresses arrays or drops missing samples: incomplete observed
    events and missing selected modeled peaks raise ``ValueError``.

    Peaks are maxima within each fixed event, not necessarily at the same
    timestamp. This objective measures aggregate peak scaling, not peak timing,
    crest shape, or individual-event accuracy. Opposing peak errors can cancel.
    Instances do not cache candidate-dependent state and can be shared by workers.

    :cvar name: ``"con15_error"``
    :cvar minimize: ``True``
    """

    name = "con15_error"
    minimize = True

    def __init__(
        self,
        event_windows: np.ndarray,
        top_fraction: float = 0.15,
        mask: np.ndarray | None = None,
        *, allow_overlapping_events: bool = False,
    ) -> None:
        super().__init__(mask=mask)
        _top_peak_indices(np.empty(0), top_fraction)
        windows = np.asarray(event_windows)
        if (windows.ndim != 2 or windows.shape[1] != 2 or not len(windows)
                or windows.dtype.kind not in "iu"):
            raise ValueError("event_windows must be a nonempty integer array of shape (n, 2)")
        if (np.any(windows[:, 0] < 0) or np.any(windows[:, 1] < windows[:, 0])
                or np.any(windows[1:, 0] < windows[:-1, 0])):
            raise ValueError("Event windows must be nonnegative, ordered by start, and inclusive")
        if not allow_overlapping_events and np.any(windows[1:, 0] <= windows[:-1, 1]):
            raise ValueError("Event windows must be nonnegative, ordered, disjoint, and inclusive")
        self.event_windows = windows.copy()
        self.event_windows.setflags(write=False)
        self.top_fraction = float(top_fraction)

    def _event_values(self, values: np.ndarray, indices: np.ndarray) -> np.ndarray:
        return np.asarray([values[start:end + 1].max()
                           for start, end in self.event_windows[indices]])

    def selected_event_indices(
        self,
        observed: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return selected rows of ``event_windows``, ranking within the mask only.

        Filter complete events first, then compute the fraction's event count
        and observed-peak ranking. Neither the number nor size of events outside
        the effective mask can affect selection. Returned rows are in peak rank
        order, with stable ties.
        """
        obs = np.asarray(observed, dtype=float)
        if obs.ndim != 1:
            raise ValueError("Observed must be a one-dimensional array")
        if np.any(self.event_windows[:, 1] >= len(obs)):
            raise ValueError("Event window outside the observation domain")
        effective = mask if mask is not None else self.mask
        if isinstance(effective, str) or callable(effective):
            raise TypeError("Resolve column-name or callable masks through CalibrationProblem")
        selector = (np.ones(obs.shape, dtype=bool) if effective is None
                    else np.asarray(effective, dtype=bool))
        if selector.shape != obs.shape:
            raise ValueError("mask shape must match observed shape")
        eligible = np.flatnonzero([
            selector[start:end + 1].all() for start, end in self.event_windows
        ])
        if not len(eligible):
            raise ValueError("Con15 requires at least one complete event inside the mask")
        windows = self.event_windows[eligible]
        obs_events = [obs[start:end + 1] for start, end in windows]
        if not all(np.isfinite(event).all() for event in obs_events):
            raise ValueError("Non-finite samples in Con15 observed events; all samples must be finite")
        obs_peaks = self._event_values(obs, eligible)
        return eligible[_top_peak_indices(obs_peaks, self.top_fraction)]

    def slope(
        self,
        observed: np.ndarray,
        predicted: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> float:
        """Return the raw peak slope using complete events in the effective mask."""
        obs = np.asarray(observed, dtype=float)
        pred = np.asarray(predicted, dtype=float)
        if obs.ndim != 1 or obs.shape != pred.shape:
            raise ValueError("Observed and predicted must be aligned one-dimensional arrays")
        selected = self.selected_event_indices(obs, mask=mask)
        windows = self.event_windows[selected]
        obs_peaks = self._event_values(obs, selected)
        pred_events = [pred[start:end + 1] for start, end in windows]
        if not all(np.isfinite(event).all() for event in pred_events):
            raise ValueError("Con15 selected modeled events must contain only finite samples")
        pred_peaks = self._event_values(pred, selected)
        value = _peak_regression_slope(obs_peaks, pred_peaks)
        if not np.isfinite(value):
            raise ValueError("Con15 requires finite selected peaks and positive observed peak energy")
        return value

    def compute(
        self,
        observed: np.ndarray,
        predicted: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> float:
        return abs(self.slope(observed, predicted, mask=mask) - 1.0)

    def evaluate(self, observed: np.ndarray, predicted: np.ndarray) -> float:
        """Evaluate on the original time axis, honoring this objective's mask."""
        return self.compute(observed, predicted)


    def prepare_observed(
        self, observed: np.ndarray, selector: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Preserve event indices despite CalibrationProblem compressing timestep arrays.
        if self.mask is None and not selector.all():
            raise ValueError("Use original-axis compute for a problem-level partial mask")
        full = np.full(selector.shape, np.nan)
        full[selector] = observed
        self.selected_event_indices(full)
        full.setflags(write=False)
        selected = selector.copy()
        selected.setflags(write=False)
        return full, selected


    def compute_selected(
        self, observed: np.ndarray, predicted: np.ndarray,
        context: tuple[np.ndarray, np.ndarray],
    ) -> float:
        full_observed, selector = context
        full_predicted = np.full(selector.shape, np.nan)
        full_predicted[selector] = predicted
        return self.compute(full_observed, full_predicted)



def _top_peak_indices(observed_peaks: np.ndarray, top_fraction: float) -> np.ndarray:
    """Stable observed-only ranking shared by calibration and acceptance."""
    if not np.isfinite(top_fraction) or not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be finite and in (0, 1]")
    count = max(1, int(np.ceil(len(observed_peaks) * top_fraction)))
    return np.argsort(-observed_peaks, kind="stable")[:count]


def _peak_regression_slope(observed_peaks: np.ndarray, modeled_peaks: np.ndarray) -> float:
    """Return a zero-intercept slope, or NaN for an undefined report metric."""
    x, y = np.asarray(observed_peaks, float), np.asarray(modeled_peaks, float)
    if x.ndim != 1 or x.shape != y.shape:
        raise ValueError("Peak arrays must be aligned and one-dimensional")
    if not len(x) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return float("nan")
    denominator = float(x @ x)
    return float(x @ y / denominator) if denominator > 0 else float("nan")
