"""Weighted mixtures of unit hydrographs (fast and slow pathways in one UH).

``F(t) = sum_i w_i F_i(t)`` with weights on the simplex. Any
:class:`~.base.UnitHydrographBase` can be a component, so a mixture of gammas,
of Nash cascades, or of a triangle and a Clark response are all the same class.

Parameters
    The mixture owns the amplitude ``A`` and an optional shared delay ``td``.
    Component ``i`` (1-based) contributes its shape parameters as
    ``{prefix_i}{name}`` (default prefix ``c{i}_``) and its own delay, if it has
    one; component amplitudes are ignored. Weights ``w_2 ... w_K`` are free in
    ``[0, 1]`` and ``w_1 = 1 - (w_2 + ... + w_K)``; with three or more
    components the constraint ``weights_sum_leq_1`` keeps ``w_1 >= 0``.
    Component constraints are forwarded with the component prefix.
"""

from __future__ import annotations

from typing import ClassVar, Sequence

import numpy as np

from ...parameters import ConstraintRecord, ScalarParameter
from ...registry import registry
from .base import UnitHydrographBase
from .models import GammaUH

__all__ = ["MixtureUH"]


@registry.register
class MixtureUH(UnitHydrographBase):
    """Weighted mixture of unit hydrograph components.

    :param components: Two or more unit hydrographs. Default: a fast and a
        slow :class:`~.models.GammaUH`.
    :param weights: Initial weights (normalised to sum to 1). Default equal.
    :param A: Amplitude.
    :param prefixes: Parameter prefix per component (default ``c1_``, ``c2_``...).
    :param weight_names: Names for the free weights of components 2..K
        (default ``w_2``... ).
    :param delay: Optional delay ``td`` [hr] applied to the whole mixture.
    """

    model_name: ClassVar[str] = "mixture-uh"

    def __init__(
        self,
        components: Sequence[UnitHydrographBase] | None = None,
        weights: Sequence[float] | None = None,
        A: float = 1.0,
        *,
        prefixes: Sequence[str] | None = None,
        weight_names: Sequence[str] | None = None,
        delay: float | None = None,
    ) -> None:
        super().__init__(A, delay=delay)
        if components is None:
            components = [GammaUH(alpha=3.0, theta=0.3), GammaUH(alpha=2.0, theta=4.0)]
        components = list(components)
        if len(components) < 2:
            raise ValueError("MixtureUH needs at least two components")
        for c in components:
            if not isinstance(c, UnitHydrographBase):
                raise TypeError(f"Mixture components must be unit hydrographs; got {type(c).__name__}")
        k = len(components)
        w = np.full(k, 1.0 / k) if weights is None else np.asarray(weights, dtype=float)
        if w.shape != (k,) or np.any(w < 0) or w.sum() <= 0:
            raise ValueError("weights must be non-negative, one per component, with a positive sum")
        self._weights_init = w / w.sum()
        self._prefixes = list(prefixes) if prefixes is not None else [f"c{i}_" for i in range(1, k + 1)]
        self._weight_names = (list(weight_names) if weight_names is not None
                              else [f"w_{i}" for i in range(2, k + 1)])
        if len(self._prefixes) != k or len(set(self._prefixes)) != k:
            raise ValueError("prefixes must be unique, one per component")
        if len(self._weight_names) != k - 1:
            raise ValueError("weight_names needs one name per component after the first")
        self._components = components
        self._param_map: dict[str, tuple[UnitHydrographBase, str]] = {}

    @property
    def components(self) -> list[UnitHydrographBase]:
        """The component unit hydrographs."""
        return list(self._components)

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def _register_shape_parameters(self) -> None:
        for name, value in zip(self._weight_names, self._weights_init[1:]):
            self.register_scalar_parameter(ScalarParameter(
                name, value=float(value), lower_bound=0.0, upper_bound=1.0,
                description="Mixture weight"))
        if len(self._components) >= 3:
            self.register_inequality_constraint(ConstraintRecord(
                "weights_sum_leq_1", "Sum of free mixture weights <= 1"))
        self._param_map = {}
        for comp, prefix in zip(self._components, self._prefixes):
            comp.initialize()
            for pname in comp.scalar_parameter_names:
                if pname == comp._amplitude_param_name:
                    continue
                child = comp.get_scalar_parameter(pname)
                name = f"{prefix}{pname}"
                self.register_scalar_parameter(ScalarParameter(
                    name, value=child.value, lower_bound=child.lower_bound,
                    upper_bound=child.upper_bound, units=child.units,
                    description=child.description, calibrate=child.calibrate))
                self._param_map[name] = (comp, pname)
            for record in comp._constraint_registry:
                self.register_inequality_constraint(
                    ConstraintRecord(f"{prefix}{record.name}", record.description))

    def _sync(self) -> None:
        for name, (comp, pname) in self._param_map.items():
            comp._scalar_parameters[pname].value = self._scalar_parameters[name].value

    def weights(self) -> np.ndarray:
        """Current component weights ``[w_1, ..., w_K]`` (``w_1`` may be < 0 if infeasible)."""
        free = np.array([self._p(n) for n in self._weight_names], dtype=float)
        return np.concatenate([[1.0 - free.sum()], free])

    def _constraint_residuals(self) -> list[float]:
        self._sync()
        out = []
        if len(self._components) >= 3:
            out.append(float(sum(self._p(n) for n in self._weight_names) - 1.0))
        for comp in self._components:
            out.extend(comp._constraint_residuals())
        return out

    def _extra_valid(self) -> bool:
        self._sync()
        return all(c.parameters_valid() and c._extra_valid() for c in self._components)

    # ------------------------------------------------------------------
    # Shape
    # ------------------------------------------------------------------

    def _support_hours(self) -> float:
        self._sync()
        return min(max(c.support_hours() for c in self._components), self.max_duration_hours)

    def _cdf(self, t):
        self._sync()
        w = np.clip(self.weights(), 0.0, None)
        t = np.asarray(t, dtype=float)
        return sum(wi * c.cdf(t) for wi, c in zip(w, self._components))

    def component_kernels(self, dt_hours: float, n_steps: int | None = None) -> dict[str, np.ndarray]:
        """Weighted kernel of each component and their sum, on one time base.

        Keys are the component prefixes without a trailing underscore, plus
        ``"combined"``; ``sum(components) == combined``.
        """
        from .kernels import trim_pad

        n = self.natural_length(dt_hours) if n_steps is None else int(n_steps)
        edges = np.arange(n + 1, dtype=float) * dt_hours
        x = np.clip(edges - self.delay_hours, 0.0, None)
        self._sync()
        w = np.clip(self.weights(), 0.0, None)
        out = {}
        total = np.zeros(n)
        for wi, comp, prefix in zip(w, self._components, self._prefixes):
            F = comp.cdf(x)
            k = wi * (F[1:] - F[:-1]) / dt_hours
            out[prefix.rstrip("_")] = trim_pad(k, n_steps)
            total = total + k
        out["combined"] = trim_pad(total, n_steps)
        return out
