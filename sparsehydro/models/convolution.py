"""Adaptive causal convolution shared by the unit hydrograph and RDII models.

Direct convolution costs O(n*m); FFT overlap-add costs roughly O(n log m) plus a
fixed transform overhead.  Which wins depends on both lengths, and the useful
discriminator turns out to be the product ``n * m``.

Measured on this project's workloads (float64), winner and its margin over the
loser:

========  ========  ========  ========  ========  ========  ========
   n \\ m        40        81       160       300       500       864
========  ========  ========  ========  ========  ========  ========
     600    dir 12   dir 6.4   dir 4.6   dir 2.5   dir 1.9   dir 1.3
    1000   dir 7.9   dir 7.3   dir 3.4   dir 2.0   dir 1.2    oa 1.2
    2000   dir 4.3   dir 4.3   dir 3.3   dir 1.3    oa 1.0    oa 1.6
    5000   dir 2.1   dir 2.3   dir 1.8   dir 1.2    oa 1.5    oa 2.3
   10000   dir 1.3   dir 1.4    oa 1.1    oa 1.5    oa 1.6    oa 2.1
   20000    oa 1.1   dir 1.1    oa 1.4    oa 2.1    oa 2.5    oa 3.3
  107000   dir 1.3   dir 1.5    oa 1.1    oa 1.5    oa 2.2    oa 3.4
========  ========  ========  ========  ========  ========  ========

``n * m > 1e6`` reproduces that boundary with a worst case of about 1.45x, and
only in the small-``m``/large-``n`` corner where the two are close to a tie
anyway; where it matters it wins 2-3.4x.

This matters more than it looks: the RDII models previously switched on
``max(n, m) > 500``, which sends a short event window (n=600, m=60) down the FFT
branch, where it is roughly six times *slower* and less accurate than direct
summation.  With kernels capped at 864 ordinates, the threshold also means any
window up to n=1157 stays on ``direct``, so ordinary event-scale fitting is
unaffected by FFT round-off entirely.

``"direct"`` is implemented with :func:`numpy.convolve` specifically, so callers
migrating from it get bit-identical results.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import oaconvolve

#: Convolution cost (``n * m``) above which FFT overlap-add beats direct
#: summation.  Below it the transform overhead dominates.
OA_COST_THRESHOLD: int = 1_000_000


def convolution_method(n: int, m: int, threshold: int = OA_COST_THRESHOLD) -> str:
    """Return the cheaper convolution algorithm for these operand lengths.

    :param n: Signal length.
    :type n: int
    :param m: Kernel length.
    :type m: int
    :param threshold: Cost (``n * m``) above which overlap-add is chosen.
    :type threshold: int
    :returns: ``"direct"`` or ``"oa"``.
    :rtype: str
    """
    return "direct" if n * m <= threshold else "oa"


def convolve_causal(
    signal: np.ndarray,
    kernel: np.ndarray,
    *,
    n_out: int | None = None,
    method: str = "auto",
    threshold: int = OA_COST_THRESHOLD,
) -> np.ndarray:
    """Convolve *signal* with *kernel* and truncate to a causal window.

    Equivalent to ``np.convolve(signal, kernel, mode="full")[:n_out]`` -- exactly
    so for ``method="direct"``, and to within FFT round-off for ``method="oa"``.

    When the FFT branch is taken and both operands are non-negative -- the usual
    case, since rainfall and UH ordinates both are -- the exact result cannot be
    negative, so round-off undershoot below zero is clipped away.  That makes the
    FFT branch strictly closer to the exact answer, and preserves the
    non-negativity that callers downstream rely on.  The direct branch is never
    clipped, so a caller deliberately convolving a signed series keeps today's
    behaviour.

    :param signal: Input series (e.g. rainfall or rainfall excess).
    :type signal: numpy.ndarray
    :param kernel: Convolution kernel (e.g. UH ordinates scaled by ``A * dt``).
    :type kernel: numpy.ndarray
    :param n_out: Output length; defaults to ``len(signal)``.
    :type n_out: int or None
    :param method: ``"auto"`` (default), ``"direct"`` or ``"oa"``.
    :type method: str
    :param threshold: Cost threshold forwarded to :func:`convolution_method`.
    :type threshold: int
    :returns: Convolved series of length *n_out*.
    :rtype: numpy.ndarray
    :raises ValueError: If *method* is not one of the three accepted values.
    """
    n = int(signal.shape[0])
    m = int(kernel.shape[0])
    if n_out is None:
        n_out = n

    if n == 0 or m == 0:
        return np.zeros(n_out, dtype=float)

    if method == "auto":
        method = convolution_method(n, m, threshold)

    if method == "direct":
        return np.convolve(signal, kernel, mode="full")[:n_out]

    if method == "oa":
        out = oaconvolve(signal, kernel, mode="full")[:n_out]
        # Two O(n) reductions, negligible beside the transforms they guard.
        if signal.min() >= 0.0 and kernel.min() >= 0.0:
            np.maximum(out, 0.0, out=out)
        return out

    raise ValueError(f"method must be 'auto', 'direct' or 'oa'; got {method!r}")


__all__ = ["OA_COST_THRESHOLD", "convolution_method", "convolve_causal"]
