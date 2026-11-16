"""Typed parameter classes with boundary enforcement for sparsehydro models.

Parameters are the primary calibration handles exposed by a model.  Each
parameter carries its name, current value(s), and the permissible range so
that calibration algorithms can operate in a well-defined search space.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ScalarParameter:
    """A named scalar parameter with lower and upper bounds.

    :param name: Unique identifier for the parameter within a model.
    :type name: str
    :param value: Current parameter value.
    :type value: float
    :param lower_bound: Minimum permissible value (inclusive).
    :type lower_bound: float
    :param upper_bound: Maximum permissible value (inclusive).
    :type upper_bound: float
    :param units: Physical units (informational only).
    :type units: str
    :param description: Human-readable description (informational only).
    :type description: str

    :raises ValueError: If ``lower_bound > upper_bound``.

    Example::

        p = ScalarParameter("k", value=0.5, lower_bound=0.0, upper_bound=1.0)
        assert p.is_valid()
        assert p.normalize() == 0.5
    """

    name: str
    value: float
    lower_bound: float
    upper_bound: float
    units: str = ""
    description: str = ""
    calibrate: bool = True
    """When ``False`` the parameter is held fixed and excluded from the
    calibration search space.  Its value is still applied during ``predict()``
    but the optimizer will never modify it."""

    def __post_init__(self) -> None:
        if self.lower_bound > self.upper_bound:
            raise ValueError(
                f"Parameter '{self.name}': lower_bound ({self.lower_bound}) "
                f"must be <= upper_bound ({self.upper_bound})."
            )

    def is_valid(self) -> bool:
        """Return ``True`` if ``value`` lies within ``[lower_bound, upper_bound]``.

        :returns: ``True`` when the value satisfies the bounds constraint.
        :rtype: bool
        """
        return self.lower_bound <= self.value <= self.upper_bound

    def normalize(self) -> float:
        """Return ``value`` mapped linearly to ``[0, 1]`` within the bounds.

        Returns ``0.0`` when ``lower_bound == upper_bound``.

        :returns: Normalized value in ``[0, 1]``.
        :rtype: float
        """
        span = self.upper_bound - self.lower_bound
        if span == 0.0:
            return 0.0
        return (self.value - self.lower_bound) / span

    def clamp(self) -> ScalarParameter:
        """Return a copy of this parameter with ``value`` clamped to the bounds.

        :returns: New :class:`ScalarParameter` with value in
            ``[lower_bound, upper_bound]``.
        :rtype: ScalarParameter
        """
        return ScalarParameter(
            name=self.name,
            value=float(np.clip(self.value, self.lower_bound, self.upper_bound)),
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            units=self.units,
            description=self.description,
            calibrate=self.calibrate,
        )

    def update(
        self,
        *,
        value: float | None = None,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
        units: str | None = None,
        description: str | None = None,
        calibrate: bool | None = None,
    ) -> None:
        """Update one or more mutable attributes in-place.

        Only the keyword arguments you supply are changed; others are left
        untouched.  Raises :class:`ValueError` if the resulting bounds are
        invalid (``lower_bound > upper_bound``).

        :param value: New parameter value.
        :type value: float | None
        :param lower_bound: New lower bound.
        :type lower_bound: float | None
        :param upper_bound: New upper bound.
        :type upper_bound: float | None
        :param units: New units string (e.g. ``"in"``, ``"m/s"``).
        :type units: str | None
        :param description: New human-readable description.
        :type description: str | None
        :param calibrate: New calibration flag.
        :type calibrate: bool | None
        :raises ValueError: If ``lower_bound > upper_bound`` after the update.
        """
        if value is not None:
            self.value = float(value)
        if lower_bound is not None:
            self.lower_bound = float(lower_bound)
        if upper_bound is not None:
            self.upper_bound = float(upper_bound)
        if units is not None:
            self.units = units
        if description is not None:
            self.description = description
        if calibrate is not None:
            self.calibrate = bool(calibrate)
        if self.lower_bound > self.upper_bound:
            raise ValueError(
                f"Parameter '{self.name}': lower_bound ({self.lower_bound}) "
                f"must be <= upper_bound ({self.upper_bound})."
            )


@dataclass
class FieldRecord:
    """Metadata for a single column in a model's ``predict()`` output DataFrame.

    Register one :class:`FieldRecord` per output column during
    :meth:`~sparsehydro.models.IModel.initialize` so that calibration
    tooling, notebooks, and downstream consumers can discover what each column
    contains without inspecting the DataFrame itself.

    :param name: Column name exactly as it appears in the ``predict()`` output
        (e.g. ``"rdii_cfs"``, ``"total_cfs"``).
    :type name: str
    :param units: Physical units string (e.g. ``"CFS"``, ``"mm"``, ``"in"``).
        Use ``""`` for dimensionless or non-physical columns such as
        ``"datetime"``.
    :type units: str
    :param description: Human-readable explanation of what the field represents.
    :type description: str
    :param calibratable: ``True`` when this column can serve as the
        ``predicted`` column in a
        :class:`~sparsehydro.calibration.CalibrationProblem` — i.e. it is a
        flow-rate or quantity directly comparable to an observed signal.
        Diagnostic columns (depth, excess, datetime) should be ``False``.
    :type calibratable: bool
    :param role: What the series is, for plotting and checks (see
        :data:`FIELD_ROLES`): ``"output"`` (a model result, the default),
        ``"component"`` (an additive part of another series, named by
        *component_of*), ``"state"`` (a storage or level at the end of each
        step, e.g. soil moisture or remaining abstraction) or ``"flux"`` (a
        rate or per-step amount moving between stores, e.g. recharge or melt).
        Trace fields (:meth:`~sparsehydro.models.IModel.register_trace_field`)
        use the last three.
    :type role: str
    :param component_of: For ``role="component"``: the series this one adds up
        to, in the same model (components of one series sum to it).
    :type component_of: str

    Example::

        from sparsehydro.parameters import FieldRecord

        class MyModel(IModel):
            def initialize(self) -> None:
                self.register_output_field(FieldRecord(
                    name="datetime",
                    description="Simulation time step",
                ))
                self.register_output_field(FieldRecord(
                    name="q_cfs",
                    units="CFS",
                    description="Predicted flow rate",
                    calibratable=True,
                ))
                self._state = ModelState.INITIALIZED
    """

    name: str
    units: str = ""
    description: str = ""
    calibratable: bool = False
    role: str = "output"
    component_of: str = ""

    def __post_init__(self) -> None:
        if self.role not in FIELD_ROLES:
            raise ValueError(f"FieldRecord {self.name!r}: role must be one of {FIELD_ROLES}; got {self.role!r}")
        if self.component_of and self.role != "component":
            raise ValueError(f"FieldRecord {self.name!r}: component_of is only meaningful with role='component'")
        if self.role == "component" and not self.component_of:
            raise ValueError(f"FieldRecord {self.name!r}: a component must name the series it adds up to")


#: Roles a :class:`FieldRecord` can have (see its ``role`` parameter).
FIELD_ROLES: tuple[str, ...] = ("output", "component", "state", "flux")


@dataclass
class ConstraintRecord:
    """Metadata for a single inequality constraint registered by a model.

    Complements :class:`ScalarParameter` for the constraint side of the
    optimisation problem.  The residual values themselves are returned by
    :meth:`~sparsehydro.models.IModel.inequality_constraints`; this class
    carries only the *identity* of each constraint so that solvers and
    notebooks can display meaningful labels.

    :param name: Short machine-readable identifier (e.g. ``"sum_R_leq_1"``).
    :type name: str
    :param description: Human-readable description of what the constraint
        enforces (e.g. ``"Sum of R_i fractions ≤ 1.0"``).
    :type description: str
    """

    name: str
    description: str = ""


@dataclass
class VectorParameter:
    """A named vector parameter with element-wise lower and upper bounds.

    A scalar supplied for ``lower_bounds`` or ``upper_bounds`` is broadcast to
    match the length of ``values``.

    :param name: Unique identifier for the parameter within a model.
    :type name: str
    :param values: 1-D array of current parameter values.
    :type values: numpy.ndarray
    :param lower_bounds: 1-D array of lower bounds, one per element.
        A scalar is broadcast to match the length of ``values``.
    :type lower_bounds: numpy.ndarray
    :param upper_bounds: 1-D array of upper bounds, one per element.
        A scalar is broadcast to match the length of ``values``.
    :type upper_bounds: numpy.ndarray
    :param units: Physical units (informational only).
    :type units: str
    :param description: Human-readable description (informational only).
    :type description: str
    :param calibrate: When ``False`` the parameter is held fixed and excluded
        from the calibration search space.
    :type calibrate: bool

    :raises ValueError: If ``values`` is not 1-D, if the bounds arrays do not
        match the length of ``values`` after broadcasting, or if any
        ``lower_bound > upper_bound``.

    Example::

        import numpy as np
        p = VectorParameter("beta", values=[0.2, 0.8],
                            lower_bounds=0.0, upper_bounds=1.0)
        assert p.is_valid()
        np.testing.assert_allclose(p.normalize(), [0.2, 0.8])
    """

    name: str
    values: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    units: str = ""
    description: str = ""
    calibrate: bool = True
    """When ``False`` the parameter is held fixed and excluded from the
    calibration search space.  Its values are still applied during
    ``predict()`` but the optimizer will never modify them."""

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=float)
        self.lower_bounds = np.asarray(self.lower_bounds, dtype=float)
        self.upper_bounds = np.asarray(self.upper_bounds, dtype=float)

        if self.values.ndim != 1:
            raise ValueError(
                f"Parameter '{self.name}': values must be a 1-D array; "
                f"got shape {self.values.shape}."
            )

        n = len(self.values)

        if self.lower_bounds.ndim == 0:
            self.lower_bounds = np.full(n, float(self.lower_bounds))
        if self.upper_bounds.ndim == 0:
            self.upper_bounds = np.full(n, float(self.upper_bounds))

        if len(self.lower_bounds) != n:
            raise ValueError(
                f"Parameter '{self.name}': lower_bounds length ({len(self.lower_bounds)}) "
                f"does not match values length ({n})."
            )
        if len(self.upper_bounds) != n:
            raise ValueError(
                f"Parameter '{self.name}': upper_bounds length ({len(self.upper_bounds)}) "
                f"does not match values length ({n})."
            )
        if np.any(self.lower_bounds > self.upper_bounds):
            bad = np.where(self.lower_bounds > self.upper_bounds)[0].tolist()
            raise ValueError(
                f"Parameter '{self.name}': lower_bound > upper_bound at indices {bad}."
            )

    @property
    def size(self) -> int:
        """Number of elements in the parameter vector.

        :rtype: int
        """
        return len(self.values)

    def is_valid(self) -> bool:
        """Return ``True`` if all values lie within their respective bounds.

        :returns: ``True`` when every element satisfies its bounds constraint.
        :rtype: bool
        """
        return bool(
            np.all(self.values >= self.lower_bounds)
            and np.all(self.values <= self.upper_bounds)
        )

    def normalize(self) -> np.ndarray:
        """Return values mapped element-wise to ``[0, 1]`` within the bounds.

        Elements with ``lower_bound == upper_bound`` are mapped to ``0.0``.

        :returns: Array of normalized values, each in ``[0, 1]``.
        :rtype: numpy.ndarray
        """
        span = self.upper_bounds - self.lower_bounds
        span = np.where(span == 0.0, 1.0, span)
        return (self.values - self.lower_bounds) / span

    def clamp(self) -> VectorParameter:
        """Return a copy with all values clamped element-wise to their bounds.

        :returns: New :class:`VectorParameter` with every value within its bounds.
        :rtype: VectorParameter
        """
        return VectorParameter(
            name=self.name,
            values=np.clip(self.values, self.lower_bounds, self.upper_bounds).copy(),
            lower_bounds=self.lower_bounds.copy(),
            upper_bounds=self.upper_bounds.copy(),
            units=self.units,
            description=self.description,
            calibrate=self.calibrate,
        )

    def update(
        self,
        *,
        values: np.ndarray | None = None,
        lower_bounds: np.ndarray | None = None,
        upper_bounds: np.ndarray | None = None,
        units: str | None = None,
        description: str | None = None,
        calibrate: bool | None = None,
    ) -> None:
        """Update one or more mutable attributes in-place.

        Only the keyword arguments you supply are changed; others are left
        untouched.  Scalar bounds are broadcast to the vector length.
        Raises :class:`ValueError` if the resulting bounds or shape are invalid.

        :param values: New values array (must match current length).
        :type values: numpy.ndarray | None
        :param lower_bounds: New lower bounds (scalar broadcast supported).
        :type lower_bounds: numpy.ndarray | None
        :param upper_bounds: New upper bounds (scalar broadcast supported).
        :type upper_bounds: numpy.ndarray | None
        :param units: New units string.
        :type units: str | None
        :param description: New human-readable description.
        :type description: str | None
        :param calibrate: New calibration flag.
        :type calibrate: bool | None
        :raises ValueError: If the resulting configuration is invalid.
        """
        if values is not None:
            self.values = np.asarray(values, dtype=float)
        if lower_bounds is not None:
            self.lower_bounds = np.asarray(lower_bounds, dtype=float)
        if upper_bounds is not None:
            self.upper_bounds = np.asarray(upper_bounds, dtype=float)
        if units is not None:
            self.units = units
        if description is not None:
            self.description = description
        if calibrate is not None:
            self.calibrate = bool(calibrate)
        self.__post_init__()
