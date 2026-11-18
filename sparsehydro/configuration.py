"""Configuration: build models, objectives and solvers from plain data.

Generic API
-----------
:func:`build_model`, :func:`build_objectives`, :func:`build_solver` and
:func:`load_config` turn JSON/YAML-style dictionaries into objects using the
registries, so any registered formulation can be configured without code::

    model:
      type: composite
      components:
        ia:   {type: ia}
        rdii: {type: rdii, excess: ia.p_excess_in,
               routing: [{type: rtk-triangle}, {type: nash-uh, n: 3, K: 2.0}]}
      outputs: {rdii_cfs: rdii.rdii_cfs}
    parameters:
      rdii.R_1: {value: 0.05, upper_bound: 0.4}

A model spec is ``{"type": <registry name>, <constructor arguments>...}``.
Nested specs with a ``type`` are built recursively; lists are built element by
element. A class can customise this with a ``from_config(spec, build)``
classmethod (``CompositeModel`` and ``RDIIModel`` do).

Study settings
--------------
The dataclasses further down configure the MSDGC whole-flow study and will
move to :mod:`sparsehydro.studies.msdgc`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any


from sparsehydro.calibration.solvers import solver_registry
from sparsehydro.calibration.solvers.base import ISolver
from sparsehydro.metrics.objectives import IObjective, objective_registry
from sparsehydro.parameters import ScalarParameter
from sparsehydro.registry import registry


@dataclass(frozen=True)
class ParameterSettings:
    """Overrides for any registered scalar, including fixed/calibrated status."""

    value: float | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    calibrate: bool | None = None

    def __post_init__(self) -> None:
        for name in ("value", "lower_bound", "upper_bound"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(value)):
                raise ValueError(f"Parameter {name} must be finite")
        if self.calibrate is not None and not isinstance(self.calibrate, bool):
            raise ValueError("Parameter calibrate must be a boolean")

    def apply(self, parameter: ScalarParameter) -> None:
        values = asdict(parameter)
        values.update({k: v for k, v in asdict(self).items() if v is not None})
        candidate = ScalarParameter(**values)
        if not candidate.is_valid():
            raise ValueError(f"Parameter {candidate.name}: value must lie within its bounds")
        if candidate.calibrate and candidate.lower_bound == candidate.upper_bound:
            raise ValueError(f"Parameter {candidate.name}: calibrated bounds must have positive width")
        parameter.update(**{k: v for k, v in asdict(self).items() if v is not None})


# ---------------------------------------------------------------------------
# Generic configuration API
# ---------------------------------------------------------------------------

def build_component(spec: Any) -> Any:
    """Build one value of a configuration tree.

    ``{"type": name, ...}`` becomes ``registry.create(name, **built_kwargs)``
    (or ``cls.from_config(spec, build_component)`` when the class defines it);
    lists and other dicts are built element-wise; anything else is returned
    unchanged.

    :raises KeyError: For an unknown ``type`` (the message lists known names).
    """
    if isinstance(spec, Mapping):
        if "type" in spec:
            cls = registry.get(spec["type"])
            body = {k: v for k, v in spec.items() if k != "type"}
            from_config = getattr(cls, "from_config", None)
            if from_config is not None:
                return from_config(body, build_component)
            return cls(**{k: build_component(v) for k, v in body.items()})
        return {k: build_component(v) for k, v in spec.items()}
    if isinstance(spec, (list, tuple)):
        return [build_component(v) for v in spec]
    return spec


def apply_parameter_settings(model: Any, parameters: Mapping[str, Any]) -> None:
    """Apply ``{name: {value, lower_bound, upper_bound, calibrate}}`` to *model*.

    :raises KeyError: For a name the model does not register.
    :raises ValueError: For invalid settings (e.g. a value outside its bounds).
    """
    for name, options in parameters.items():
        if name not in model.scalar_parameter_names:
            raise KeyError(f"Unknown parameter {name!r}. Known: {model.scalar_parameter_names}")
        settings = options if isinstance(options, ParameterSettings) else ParameterSettings(**options)
        settings.apply(model.get_scalar_parameter(name))


def build_model(config: Mapping[str, Any]) -> Any:
    """Build and initialize a model from a configuration mapping.

    *config* is either a model spec (``{"type": ...}``) or a mapping with
    ``model`` (the spec) and optional ``parameters`` (flat parameter settings
    applied after initialization).

    :returns: An initialized model.
    """
    if "model" in config:
        spec, parameters = config["model"], config.get("parameters") or {}
    else:
        spec, parameters = config, {}
    if not isinstance(spec, Mapping) or "type" not in spec:
        raise ValueError("A model configuration needs a 'type'")
    model = build_component(spec)
    model.initialize()
    apply_parameter_settings(model, parameters)
    return model


def build_objectives(specs: Sequence[Any], masks: Mapping[str, Any] | None = None) -> list[IObjective]:
    """Build objectives from names or ``{"type": name, "mask": ..., ...}`` specs.

    A string ``mask`` is looked up in *masks*; other keyword arguments go to
    the objective's constructor.

    :raises KeyError: For an unknown objective or mask name.
    """
    masks = dict(masks or {})
    objectives = []
    for spec in specs:
        if isinstance(spec, str):
            spec = {"type": spec}
        body = dict(spec)
        name = body.pop("type")
        mask = body.get("mask")
        if isinstance(mask, str):
            if mask not in masks:
                raise KeyError(f"Unknown mask {mask!r} for objective {name!r}. Known: {sorted(masks)}")
            body["mask"] = masks[mask]
        objectives.append(objective_registry.create(name, **body))
    return objectives


def build_solver(spec: Any) -> ISolver:
    """Build a solver from a name or ``{"type": name, **constructor_kwargs}``."""
    if isinstance(spec, str):
        spec = {"type": spec}
    body = dict(spec)
    return solver_registry.create(body.pop("type"), **body)


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a configuration file (``.json``, ``.yaml`` or ``.yml``).

    YAML needs PyYAML (``pip install "sparsehydro[yaml]"``).
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in (".json", ".yaml", ".yml"):
        raise ValueError(f"Unsupported configuration format: {path.suffix!r} (use .json, .yaml or .yml)")
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        return json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError('Reading YAML needs PyYAML: pip install "sparsehydro[yaml]"') from exc
        return yaml.safe_load(text)


