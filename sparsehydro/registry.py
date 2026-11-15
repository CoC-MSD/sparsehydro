"""Model registration service for sparsehydro.

:class:`ModelRegistry` tracks compatible :class:`~sparsehydro.models.IModel`
implementations and lets calibration frameworks discover and instantiate models
by name without importing them directly.

A module-level default registry instance is provided as :data:`registry`.
Models can self-register at definition time using it as a decorator::

    from sparsehydro.registry import registry

    @registry.register
    class LinearReservoir(IModel):
        model_name = "linear-reservoir"
        ...
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from .models.base import IModel


class NamedRegistry:
    """A simple name -> factory registry (used for objectives and solvers).

    A factory is any callable returning the object, usually a class::

        objective_registry.add("my_metric", MyMetric)
        objective_registry.create("my_metric", mask=train)

    :param kind: Human-readable kind for error messages (e.g. ``"objective"``).
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._factories: dict[str, Any] = {}

    def add(self, name: str, factory: Any, *, replace: bool = False) -> Any:
        """Register *factory* under *name* and return it.

        :raises ValueError: If *name* is taken and *replace* is false.
        """
        if not isinstance(name, str) or not name:
            raise ValueError(f"{self.kind} name must be a non-empty string")
        if not callable(factory):
            raise TypeError(f"{self.kind} factory for {name!r} is not callable")
        if name in self._factories and not replace:
            raise ValueError(f"A {self.kind} named {name!r} is already registered")
        self._factories[name] = factory
        return factory

    def register(self, name: str):
        """Decorator form of :meth:`add`."""
        def decorator(factory):
            return self.add(name, factory)
        return decorator

    def unregister(self, name: str) -> None:
        """Remove *name*."""
        self.get(name)
        del self._factories[name]

    def get(self, name: str) -> Any:
        """Return the factory for *name*.

        :raises KeyError: Listing the registered names when *name* is unknown.
        """
        try:
            return self._factories[name]
        except KeyError:
            raise KeyError(f"No {self.kind} named {name!r}. Registered: {self.names()}") from None

    def create(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Call the factory for *name* with the given arguments."""
        return self.get(name)(*args, **kwargs)

    def names(self) -> list[str]:
        """Sorted registered names."""
        return sorted(self._factories)

    def __contains__(self, name: object) -> bool:
        return name in self._factories

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())


class ModelRegistry:
    """Central registry for tracking compatible parsimonious model classes.

    A model class is considered *compatible* if it:

    1. Is a **concrete** (non-abstract) subclass of
       :class:`~sparsehydro.models.IModel`.
    2. Defines a non-empty ``model_name`` class variable.

    The registry can be used directly as a class decorator::

        registry = ModelRegistry()

        @registry.register
        class LinearReservoir(IModel):
            model_name = "linear-reservoir"
            ...

        model = registry.create("linear-reservoir")
    """

    def __init__(self) -> None:
        self._models: dict[str, type] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, model_cls: type) -> type:
        """Register a compatible model class.

        May be used as a class decorator.

        :param model_cls: The model class to register.  Must be a concrete
            subclass of :class:`~sparsehydro.models.IModel` that defines
            a non-empty ``model_name`` class variable.
        :type model_cls: type[IModel]
        :returns: The model class unchanged (enables decorator use).
        :rtype: type[IModel]
        :raises TypeError: If ``model_cls`` is not a concrete
            :class:`~sparsehydro.models.IModel` subclass or does not
            define a valid ``model_name``.
        :raises ValueError: If a model with the same name is already
            registered.
        """
        self._assert_compatible(model_cls)
        name: str = model_cls.model_name
        if name in self._models:
            existing = self._models[name].__qualname__
            raise ValueError(
                f"A model named '{name}' is already registered "
                f"({existing}). Unregister it first or choose a "
                "different model_name."
            )
        self._models[name] = model_cls
        return model_cls

    def unregister(self, name: str) -> None:
        """Remove the model registered under ``name``.

        :param name: The ``model_name`` of the model to remove.
        :type name: str
        :raises KeyError: If no model with ``name`` is registered.
        """
        if name not in self._models:
            raise KeyError(
                f"No model named '{name}' is registered. "
                f"Registered names: {self.names()}"
            )
        del self._models[name]

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get(self, name: str) -> type[IModel]:
        """Return the model class registered under ``name``.

        :param name: The ``model_name`` of the desired model.
        :type name: str
        :returns: The registered model class.
        :rtype: type[IModel]
        :raises KeyError: If no model with ``name`` is registered.
        """
        if name not in self._models:
            raise KeyError(
                f"No model named '{name}' is registered. "
                f"Registered names: {self.names()}"
            )
        return self._models[name]

    def create(self, name: str, *args: Any, **kwargs: Any) -> IModel:
        """Instantiate the model class registered under ``name``.

        All positional and keyword arguments are forwarded to the model
        constructor.

        :param name: The ``model_name`` of the model to instantiate.
        :type name: str
        :returns: A new instance of the registered model.
        :rtype: IModel
        :raises KeyError: If no model with ``name`` is registered.
        """
        return self.get(name)(*args, **kwargs)

    def names(self) -> list[str]:
        """Return a sorted list of all registered model names.

        :returns: Sorted list of ``model_name`` strings.
        :rtype: list[str]
        """
        return sorted(self._models)

    def is_registered(self, name: str) -> bool:
        """Return ``True`` if a model with ``name`` is registered.

        :param name: Model name to check.
        :type name: str
        :rtype: bool
        """
        return name in self._models

    # ------------------------------------------------------------------
    # Compatibility check
    # ------------------------------------------------------------------

    @staticmethod
    def _assert_compatible(model_cls: type) -> None:
        """Raise :class:`TypeError` if ``model_cls`` is not a compatible model.

        :param model_cls: The class to check.
        :raises TypeError: If incompatible.
        """
        from .models.base import IModel as _IModel  # lazy to avoid circular import
        if not (isinstance(model_cls, type) and issubclass(model_cls, _IModel)):
            raise TypeError(
                f"{model_cls!r} is not a subclass of IModel."
            )
        if inspect.isabstract(model_cls):
            abstract = sorted(model_cls.__abstractmethods__)
            raise TypeError(
                f"'{model_cls.__qualname__}' is abstract and cannot be "
                f"registered. Unimplemented methods: {abstract}"
            )
        name = getattr(model_cls, "model_name", None)
        if not isinstance(name, str) or not name.strip():
            raise TypeError(
                f"'{model_cls.__qualname__}' must define a non-empty "
                f"'model_name' class variable; got {name!r}."
            )

    # ------------------------------------------------------------------
    # Collections protocol
    # ------------------------------------------------------------------

    def __contains__(self, name: object) -> bool:
        return name in self._models

    def __len__(self) -> int:
        return len(self._models)

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._models))

    def __repr__(self) -> str:  # pragma: no cover
        return f"ModelRegistry({self.names()})"


#: Default module-level registry.  Import and use this instance directly
#: rather than creating your own unless you need isolation.
registry: ModelRegistry = ModelRegistry()
