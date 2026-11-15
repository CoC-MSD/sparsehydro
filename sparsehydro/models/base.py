"""Abstract base interfaces for parsimonious hydrological models.

All concrete models must inherit from :class:`IModel` and implement the five
lifecycle methods.  The class also provides a built-in parameter registry
so that calibration algorithms can discover and manipulate named parameters
without knowing the internal model structure.
"""

from __future__ import annotations

import inspect
import re
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from ..enums import ModelState
from ..parameters import ConstraintRecord, FieldRecord, ScalarParameter, VectorParameter

# Matches a flattened vector-parameter element name, e.g. "pf_dow[3]" -> ("pf_dow", 3).
_FLAT_VECTOR_INDEX_RE = re.compile(r"^(.+)\[(\d+)\]$")


class IModel(ABC):
    """Abstract interface for a parsimonious hydrological model.

    **Model name**

    Every concrete subclass must define a ``model_name`` class variable — a
    non-empty string that uniquely identifies the model type.  This is
    enforced at class-definition time::

        class MyModel(IModel):
            model_name = "my-model"
            ...

    Attempting to define a concrete subclass without ``model_name`` raises
    :class:`TypeError` immediately.

    **Lifecycle**

    Concrete subclasses must advance ``self._state`` as each method completes::

        CREATED → INITIALIZED → VALIDATED → PREPARED → PREDICTED → FINALIZED

    **Parameter registry**

    Parameters are registered during :meth:`initialize` via
    :meth:`register_scalar_parameter` and :meth:`register_vector_parameter`.
    Calibration tools retrieve them by name through
    :meth:`get_scalar_parameter` and :meth:`get_vector_parameter`.

    Minimal concrete implementation::

        class MyModel(IModel):
            model_name = "my-model"

            def initialize(self) -> None:
                self.register_scalar_parameter(
                    ScalarParameter("alpha", value=0.3,
                                    lower_bound=0.0, upper_bound=1.0)
                )
                self._state = ModelState.INITIALIZED

            def validate(self) -> bool:
                ok = self.parameters_valid()
                if ok:
                    self._state = ModelState.VALIDATED
                return ok

            def prepare(self, forcing) -> None:
                self._forcing = forcing
                self._state = ModelState.PREPARED

            def predict(self) -> pd.Series:
                alpha = self.get_scalar_parameter("alpha").value
                self._state = ModelState.PREDICTED
                return pd.Series(self._forcing * alpha)

            def finalize(self) -> None:
                self._state = ModelState.FINALIZED
    """

    #: Unique string identifier for this model type.
    #: Must be defined by every concrete subclass.
    model_name: ClassVar[str]

    #: Whether this model implements :meth:`predict_arrays` natively rather than
    #: inheriting the DataFrame-unwrapping default.  Informational only --
    #: :class:`~sparsehydro.calibration.problem.CalibrationProblem` calls
    #: :meth:`predict_arrays` unconditionally, so nothing branches on this.
    supports_array_predict: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Abstract subclasses (those that still have unimplemented abstract
        # methods) are permitted to defer the model_name definition.
        if inspect.isabstract(cls):
            return
        if "model_name" not in vars(cls) and not hasattr(cls, "model_name"):
            raise TypeError(
                f"Concrete model '{cls.__qualname__}' must define a "
                "'model_name' class variable (a non-empty string that "
                "uniquely identifies the model type)."
            )
        name = getattr(cls, "model_name", None)
        if not isinstance(name, str) or not name.strip():
            raise TypeError(
                f"'{cls.__qualname__}.model_name' must be a non-empty string; "
                f"got {name!r}."
            )

    def __init__(self) -> None:
        self._state: ModelState = ModelState.CREATED
        self._scalar_parameters: dict[str, ScalarParameter] = {}
        self._vector_parameters: dict[str, VectorParameter] = {}
        self._constraint_registry: list[ConstraintRecord] = []
        self._output_fields: dict[str, FieldRecord] = {}

    def _reset_registries(self) -> None:
        """Clear registered parameters, outputs and constraints.

        Call at the start of :meth:`initialize` so that initializing twice
        re-registers everything once instead of appending duplicates.
        """
        self._scalar_parameters = {}
        self._vector_parameters = {}
        self._constraint_registry = []
        self._output_fields = {}

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    @property
    def state(self) -> ModelState:
        """Current lifecycle state of the model.

        :returns: The model's current lifecycle state.
        :rtype: ModelState
        """
        return self._state

    def is_created(self) -> bool:
        """Return whether the model is in the ``CREATED`` state.

        :returns: ``True`` if the model has been constructed but not yet
            initialized.
        :rtype: bool
        """
        return self._state is ModelState.CREATED

    def is_initialized(self) -> bool:
        """Return whether the model is in the ``INITIALIZED`` state.

        :returns: ``True`` if :meth:`initialize` has completed.
        :rtype: bool
        """
        return self._state is ModelState.INITIALIZED

    def is_validated(self) -> bool:
        """Return whether the model is in the ``VALIDATED`` state.

        :returns: ``True`` if :meth:`validate` has completed successfully.
        :rtype: bool
        """
        return self._state is ModelState.VALIDATED

    def is_prepared(self) -> bool:
        """Return whether the model is in the ``PREPARED`` state.

        :returns: ``True`` if :meth:`prepare` has completed.
        :rtype: bool
        """
        return self._state is ModelState.PREPARED

    def is_predicted(self) -> bool:
        """Return whether the model is in the ``PREDICTED`` state.

        :returns: ``True`` if :meth:`predict` has completed.
        :rtype: bool
        """
        return self._state is ModelState.PREDICTED

    def is_finalized(self) -> bool:
        """Return whether the model is in the ``FINALIZED`` state.

        :returns: ``True`` if :meth:`finalize` has completed.
        :rtype: bool
        """
        return self._state is ModelState.FINALIZED

    # ------------------------------------------------------------------
    # Lifecycle (abstract)
    # ------------------------------------------------------------------

    @abstractmethod
    def initialize(self, *args: Any, **kwargs: Any) -> None:
        """Set up model structure and register parameters.

        Called once before :meth:`validate`.  Implementations register every
        :class:`~sparsehydro.parameters.ScalarParameter` and
        :class:`~sparsehydro.parameters.VectorParameter` here and advance the
        state to :attr:`~sparsehydro.enums.ModelState.INITIALIZED`.

        :param args: Optional positional configuration defined by the subclass.
        :type args: typing.Any
        :param kwargs: Optional keyword configuration defined by the subclass.
        :type kwargs: typing.Any
        :returns: Nothing.
        :rtype: None
        """

    @abstractmethod
    def validate(self) -> bool:
        """Validate the model configuration and parameter bounds.

        Implementations should advance the state to
        :attr:`~sparsehydro.enums.ModelState.VALIDATED` when returning ``True``.

        :returns: ``True`` if the model is valid and ready for :meth:`prepare`;
            ``False`` otherwise.
        :rtype: bool
        """

    @abstractmethod
    def prepare(self, *args: Any, **kwargs: Any) -> None:
        """Load and pre-process forcing/input data.

        :param args: Positional forcing/input data (e.g. a DataFrame of rainfall).
        :type args: typing.Any
        :param kwargs: Keyword forcing/input options.
        :type kwargs: typing.Any
        :returns: Nothing.
        :rtype: None
        """

    @abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> pd.DataFrame | pd.Series:
        """Execute the model and return its outputs.

        :param args: Positional run-time arguments.
        :type args: typing.Any
        :param kwargs: Keyword run-time arguments.
        :type kwargs: typing.Any
        :returns: Model outputs, typically with a predicted-flow column.
        :rtype: pandas.DataFrame | pandas.Series
        """

    def predict_arrays(self) -> dict[str, np.ndarray]:
        """Return :meth:`predict`'s output as plain arrays, without a DataFrame.

        The calibration loop calls this instead of :meth:`predict` so the hot
        path never builds a DataFrame that the caller immediately unwraps again.

        This default implementation simply unwraps :meth:`predict`, so every
        existing model works unchanged and is no slower than before.  Models that
        can produce the arrays directly should override it -- and must return
        exactly the keys of ``predict().columns``, with values equal to
        ``predict()[col].to_numpy()``.  ``tests/test_fastpath_invariants.py``
        enforces that equivalence.

        Prefer building :meth:`predict` on top of this method rather than the
        other way around: a model with two independent implementations will
        eventually have two different answers.

        Returned arrays may be views into model state; callers must not mutate
        them.

        :returns: Mapping of output column name to 1-D array.
        :rtype: dict[str, numpy.ndarray]
        """
        out = self.predict()
        if isinstance(out, pd.Series):
            return {str(out.name): out.to_numpy()}
        # Deliberately untyped: forcing dtype=float here would corrupt a
        # datetime64 column.
        return {str(c): out[c].to_numpy() for c in out.columns}

    @abstractmethod
    def finalize(self) -> None:
        """Release resources and wrap up computation.

        :returns: Nothing.
        :rtype: None
        """

    # ------------------------------------------------------------------
    # Parameter registry
    # ------------------------------------------------------------------

    def register_scalar_parameter(self, param: ScalarParameter) -> None:
        """Register a scalar parameter with the model.

        :param param: Parameter to register.  Replaces any existing parameter
            sharing the same :attr:`~sparsehydro.parameters.ScalarParameter.name`.
        :type param: ScalarParameter
        :returns: Nothing.
        :rtype: None
        """
        self._scalar_parameters[param.name] = param

    def register_vector_parameter(self, param: VectorParameter) -> None:
        """Register a vector parameter with the model.

        :param param: Parameter to register.  Replaces any existing parameter
            sharing the same :attr:`~sparsehydro.parameters.VectorParameter.name`.
        :type param: VectorParameter
        :returns: Nothing.
        :rtype: None
        """
        self._vector_parameters[param.name] = param

    def get_scalar_parameter(self, name: str) -> ScalarParameter:
        """Retrieve a registered scalar parameter by name.

        :param name: Name of the scalar parameter to retrieve.
        :type name: str
        :returns: The registered scalar parameter.
        :rtype: ScalarParameter
        :raises KeyError: If no scalar parameter named *name* is registered.
        """
        if name not in self._scalar_parameters:
            available = list(self._scalar_parameters)
            raise KeyError(
                f"Scalar parameter '{name}' not found. "
                f"Available parameters: {available}"
            )
        return self._scalar_parameters[name]

    def get_vector_parameter(self, name: str) -> VectorParameter:
        """Retrieve a registered vector parameter by name.

        :param name: Name of the vector parameter to retrieve.
        :type name: str
        :returns: The registered vector parameter.
        :rtype: VectorParameter
        :raises KeyError: If no vector parameter named *name* is registered.
        """
        if name not in self._vector_parameters:
            available = list(self._vector_parameters)
            raise KeyError(
                f"Vector parameter '{name}' not found. "
                f"Available parameters: {available}"
            )
        return self._vector_parameters[name]

    def rename_scalar_parameter(self, old_name: str, new_name: str) -> None:
        """Rename a registered scalar parameter.

        :param old_name: Current name of the parameter.
        :type old_name: str
        :param new_name: New name to assign to the parameter.
        :type new_name: str
        :returns: Nothing.
        :rtype: None
        :raises ValueError: If a scalar parameter named *new_name* already exists.
        :raises KeyError: If no scalar parameter named *old_name* is registered.
        """
        if new_name in self._scalar_parameters:
            raise ValueError(f"Scalar parameter '{new_name}' already exists.")
        param = self.get_scalar_parameter(old_name)
        param.name = new_name
        self._scalar_parameters[new_name] = param
        del self._scalar_parameters[old_name]

    def rename_vector_parameter(self, old_name: str, new_name: str) -> None:
        """Rename a registered vector parameter.

        :param old_name: Current name of the parameter.
        :type old_name: str
        :param new_name: New name to assign to the parameter.
        :type new_name: str
        :returns: Nothing.
        :rtype: None
        :raises ValueError: If a vector parameter named *new_name* already exists.
        :raises KeyError: If no vector parameter named *old_name* is registered.
        """
        if new_name in self._vector_parameters:
            raise ValueError(f"Vector parameter '{new_name}' already exists.")
        param = self.get_vector_parameter(old_name)
        param.name = new_name
        self._vector_parameters[new_name] = param
        del self._vector_parameters[old_name]

    @property
    def scalar_parameter_names(self) -> list[str]:
        """Ordered list of registered scalar parameter names.

        :returns: Scalar parameter names in registration order.
        :rtype: list[str]
        """
        return list(self._scalar_parameters)

    @property
    def vector_parameter_names(self) -> list[str]:
        """Ordered list of registered vector parameter names.

        :returns: Vector parameter names in registration order.
        :rtype: list[str]
        """
        return list(self._vector_parameters)

    def parameters_valid(self) -> bool:
        """Return whether every registered parameter is within its bounds.

        :returns: ``True`` when all scalar and vector parameters satisfy their
            bounds; ``False`` otherwise.
        :rtype: bool
        """
        return all(p.is_valid() for p in self._scalar_parameters.values()) and all(
            p.is_valid() for p in self._vector_parameters.values()
        )

    # ------------------------------------------------------------------
    # Output field registry
    # ------------------------------------------------------------------

    def apply_parameter_values(
        self,
        values: "dict[str, float]",
        *,
        skip_missing: bool = True,
    ) -> "list[str]":
        """Apply a mapping of parameter names to values in-place.

        :param values: Mapping of scalar parameter name to its new value.
        :type values: dict[str, float]
        :param skip_missing: When ``True`` (default) names that are not
            registered are ignored; when ``False`` a missing name raises.
        :type skip_missing: bool
        :returns: Names of the parameters that were actually updated.
        :rtype: list[str]
        :raises KeyError: If *skip_missing* is ``False`` and a name in *values*
            is not registered.
        """
        applied: list[str] = []
        for name, value in values.items():
            if name not in self._scalar_parameters:
                if not skip_missing:
                    available = list(self._scalar_parameters)
                    raise KeyError(
                        f"Parameter '{name}' is not registered. "
                        f"Available parameters: {available}"
                    )
                continue
            self._scalar_parameters[name].value = float(value)
            applied.append(name)
        return applied

    def apply_flat_parameter(self, name: str, value: float) -> None:
        """Apply one value from a flattened calibration search vector.

        Dispatches on the parameter name's shape: a plain name (e.g.
        ``"HHL"``) is treated as a scalar parameter; a name with a trailing
        index (e.g. ``"pf_dow[3]"``, as produced by
        :class:`~sparsehydro.calibration.CalibrationProblem` for
        :class:`~sparsehydro.parameters.VectorParameter` elements) sets that
        single element of the named vector parameter in-place.

        :param name: Scalar parameter name, or ``"{vector_name}[{index}]"``.
        :type name: str
        :param value: New value to assign.
        :type value: float
        :returns: Nothing.
        :rtype: None
        :raises KeyError: If the named scalar or vector parameter is not
            registered, or *index* is out of range for the vector parameter.
        """
        match = _FLAT_VECTOR_INDEX_RE.match(name)
        if match is None:
            self.get_scalar_parameter(name).value = float(value)
            return
        vec_name, index_str = match.group(1), match.group(2)
        vp = self.get_vector_parameter(vec_name)
        index = int(index_str)
        if not (0 <= index < vp.size):
            raise KeyError(
                f"Index {index} out of range for vector parameter "
                f"'{vec_name}' (size {vp.size})."
            )
        vp.values[index] = float(value)

    def read_flat_parameter(self, name: str) -> float:
        """Read one value addressed the same way as :meth:`apply_flat_parameter`.

        :param name: Scalar parameter name, or ``"{vector_name}[{index}]"``.
        :type name: str
        :returns: The current value at that address.
        :rtype: float
        :raises KeyError: If the named scalar or vector parameter is not
            registered, or *index* is out of range for the vector parameter.
        """
        match = _FLAT_VECTOR_INDEX_RE.match(name)
        if match is None:
            return self.get_scalar_parameter(name).value
        vec_name, index_str = match.group(1), match.group(2)
        vp = self.get_vector_parameter(vec_name)
        index = int(index_str)
        if not (0 <= index < vp.size):
            raise KeyError(
                f"Index {index} out of range for vector parameter "
                f"'{vec_name}' (size {vp.size})."
            )
        return float(vp.values[index])

    def register_output_field(self, field: FieldRecord) -> None:
        """Register metadata for one column of the ``predict()`` output.

        :param field: Output field metadata.  Replaces any existing field with
            the same name.
        :type field: FieldRecord
        :returns: Nothing.
        :rtype: None
        """
        self._output_fields[field.name] = field

    def get_output_field(self, name: str) -> FieldRecord:
        """Return the :class:`~sparsehydro.parameters.FieldRecord` for *name*.

        :param name: Name of the output field to retrieve.
        :type name: str
        :returns: The registered output-field metadata.
        :rtype: FieldRecord
        :raises KeyError: If no output field named *name* is registered.
        """
        if name not in self._output_fields:
            available = list(self._output_fields)
            raise KeyError(
                f"Output field '{name}' not found. "
                f"Available fields: {available}"
            )
        return self._output_fields[name]

    @property
    def output_field_names(self) -> list[str]:
        """Ordered list of registered output column names.

        :returns: Output column names in registration order.
        :rtype: list[str]
        """
        return list(self._output_fields)

    @property
    def output_fields(self) -> list[FieldRecord]:
        """Ordered list of all registered output-field records.

        :returns: :class:`~sparsehydro.parameters.FieldRecord` objects in
            registration order.
        :rtype: list[FieldRecord]
        """
        return list(self._output_fields.values())

    @property
    def calibratable_output_names(self) -> list[str]:
        """Names of output columns flagged as calibratable.

        :returns: Names of output fields whose ``calibratable`` flag is ``True``.
        :rtype: list[str]
        """
        return [f.name for f in self._output_fields.values() if f.calibratable]

    def output_field_metadata(self) -> "pd.DataFrame":
        """Return a DataFrame describing every registered output field.

        :returns: One row per output field with ``field``, ``units``,
            ``calibratable`` and ``description`` columns.
        :rtype: pandas.DataFrame
        """
        rows = [
            {
                "field":        f.name,
                "units":        f.units,
                "calibratable": f.calibratable,
                "description":  f.description,
            }
            for f in self._output_fields.values()
        ]
        return pd.DataFrame(rows)

    def register_inequality_constraint(self, record: ConstraintRecord) -> None:
        """Register a named inequality constraint.

        :param record: Constraint metadata to append to the registry.
        :type record: ConstraintRecord
        :returns: Nothing.
        :rtype: None
        """
        self._constraint_registry.append(record)

    @property
    def inequality_constraint_names(self) -> list[str]:
        """Ordered list of registered inequality constraint names.

        :returns: Constraint names in registration order.
        :rtype: list[str]
        """
        return [r.name for r in self._constraint_registry]

    @property
    def inequality_constraint_descriptions(self) -> list[str]:
        """Ordered list of registered inequality constraint descriptions.

        :returns: Constraint descriptions in registration order.
        :rtype: list[str]
        """
        return [r.description for r in self._constraint_registry]

    def inequality_constraints(self) -> list[float]:
        """Inequality constraint residuals where ``g_j <= 0`` means feasible.

        Override in subclasses to expose model-specific constraints to the
        calibration framework.  The default returns an empty list (unconstrained).

        :returns: Constraint residuals; an empty list indicates no constraints.
        :rtype: list[float]
        """
        return []


class IUnitHydroComponent(IModel, ABC):
    """Abstract base for unit hydrograph components used within :class:`RDIIModel`.

    Extends :class:`IModel` with a single additional method, :meth:`get_kernel`,
    that returns a normalized ordinate array suitable for convolution with a
    rainfall-excess series.

    **Amplitude parameter**

    Many UH models carry a parameter that scales the total response volume
    (``R`` for RTK triangles, ``A`` for Nash/Gamma UH shapes).  When a
    component is embedded in :class:`RDIIModel`, that composite manages the
    fraction ``R_i`` separately and re-registers only the *shape* parameters.
    Declare the name of the amplitude/scaling parameter as a class variable
    so the composite can exclude it::

        class MyUH(IUnitHydroComponent):
            _amplitude_param_name = "A"   # or "R", or None if none
    """

    _amplitude_param_name: ClassVar[str | None] = None

    @abstractmethod
    def get_kernel(
        self,
        dt_hours: float,
        n_steps: int | None = None,
    ) -> np.ndarray:
        """Return normalized unit hydrograph ordinates.

        The returned array satisfies::

            np.sum(ordinates) * dt_hours ≈ 1.0

        :param dt_hours: Time-step size [hr] of the forcing data.
        :param n_steps: If provided, trim or zero-pad the result to this length.
        :returns: 1-D ordinate array [1/hr].
        :rtype: numpy.ndarray
        """
