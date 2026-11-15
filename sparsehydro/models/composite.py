"""Composite models: registered components wired together by name.

:class:`CompositeModel` is how formulations are combined in sparsehydro. It
holds named child models and three declarative pieces of wiring:

``bindings``
    ``{"target.input": "source.output"}``. Before each prediction the source
    array is handed to the target with :meth:`IModel.set_input`. The source may
    be another child (``"store.p_excess_in"``) or a prepared data column
    (``"data.rainfall_in"``). Children run in an order that respects bindings.

``outputs``
    What the composite returns besides ``datetime``. Each entry is an alias
    (``"rdii_cfs": "rdii.rdii_cfs"``), a sum (a list of addresses, or
    ``{"sum": [...]}``) or a product (``{"product": [...]}``). Anything more
    elaborate belongs in its own component. With ``outputs=None`` every child
    output is exposed as ``"child.field"``.

``shared``
    ``{"area_acres": ["store.area_acres", "rdii.area_acres"]}`` registers one
    composite parameter that drives several child parameters.

Parameters
    Child parameters are registered as ``{prefix}{name}`` with the default
    prefix ``"{child}."`` (e.g. ``store.k_div``); ``prefixes`` overrides it per
    child (``""`` keeps the child's own names). ``parameter_overrides`` changes
    values, bounds or ``calibrate`` flags after forwarding, and
    ``parameter_order`` fixes the order of the calibration vector. Child
    constraints are forwarded with the same prefix.

:class:`AbstractionUHModel` is a preset built on :class:`CompositeModel`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from ..enums import ModelState
from ..parameters import ConstraintRecord, FieldRecord, ScalarParameter, VectorParameter
from ..registry import registry
from .base import IModel, IUnitHydroComponent
from .rdii import IAModel
from .seasonality import SeasonalityModel
from .unithydrograph import PeakTailUH

_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_DATA = "data"


def _component_class(spec: Any) -> type | None:
    if isinstance(spec, Mapping) and "type" in spec:
        return registry.get(spec["type"])
    return None


def _split(address: str) -> tuple[str, str]:
    if not isinstance(address, str) or "." not in address:
        raise ValueError(f"expected 'component.field', got {address!r}")
    child, field = address.split(".", 1)
    return child, field


@registry.register
class CompositeModel(IModel):
    """Named components wired by bindings, with combined outputs.

    :param components: ``{name: model}``; names are identifiers and ``"data"``
        is reserved.
    :param bindings: ``{"target.input": "source.output"}``.
    :param outputs: Output specification (see module docstring).
    :param prefixes: Parameter-name prefix per component (default ``"name."``).
    :param shared: ``{composite_name: ["child.param", ...]}``.
    :param parameter_overrides: ``{name: {"value": ..., "lower_bound": ...,
        "upper_bound": ..., "calibrate": ...}}`` applied after forwarding.
    :param parameter_order: Names placed first, in this order, in the
        parameter registry (and hence in the calibration vector).
    :param require_feasible: Raise ``ValueError`` from :meth:`predict_arrays`
        when any constraint is violated.
    """

    model_name: ClassVar[str] = "composite"
    supports_array_predict: ClassVar[bool] = True

    def __init__(
        self,
        components: Mapping[str, IModel],
        *,
        bindings: Mapping[str, str] | None = None,
        outputs: Mapping[str, Any] | None = None,
        prefixes: Mapping[str, str] | None = None,
        shared: Mapping[str, Sequence[str]] | None = None,
        parameter_overrides: Mapping[str, Mapping[str, Any]] | None = None,
        parameter_order: Sequence[str] | None = None,
        require_feasible: bool = False,
    ) -> None:
        super().__init__()
        if not components:
            raise ValueError("CompositeModel needs at least one component")
        for name, model in components.items():
            if not isinstance(name, str) or not _NAME_RE.match(name) or name == _DATA:
                raise ValueError(f"invalid component name {name!r}")
            if not isinstance(model, IModel):
                raise TypeError(f"component {name!r} is not an IModel")
        self._components: dict[str, IModel] = dict(components)
        self._bindings: dict[str, str] = dict(bindings or {})
        self._outputs_spec = None if outputs is None else dict(outputs)
        self._prefixes = {name: f"{name}." for name in self._components}
        for name, prefix in (prefixes or {}).items():
            if name not in self._components:
                raise ValueError(f"prefix given for unknown component {name!r}")
            self._prefixes[name] = prefix
        self._shared = {k: list(v) for k, v in (shared or {}).items()}
        self._overrides = {k: dict(v) for k, v in (parameter_overrides or {}).items()}
        self._parameter_order = list(parameter_order or [])
        self.require_feasible = bool(require_feasible)

        self._run_order: list[str] = []
        self._inputs: dict[str, list[tuple[str, str, str]]] = {}
        self._param_map: dict[str, list[tuple[IModel, str, str]]] = {}
        self._output_plan: list[tuple[str, str, list[tuple[str, str]]]] = []
        self._datetime: np.ndarray | None = None
        self._data_columns: dict[str, np.ndarray] = {}

    @classmethod
    def from_config(cls, spec: Mapping[str, Any], build) -> "CompositeModel":
        """Build from configuration.

        Keys: ``components`` (``{name: spec}``; a component spec may carry
        ``inputs: {field: "source.output"}``), ``bindings``, ``outputs`` (or
        ``output``), ``prefixes``, ``shared``, ``parameter_overrides``,
        ``parameter_order``, ``require_feasible``.
        """
        spec = dict(spec)
        components: dict[str, IModel] = {}
        bindings = dict(spec.pop("bindings", None) or {})
        for name, child_spec in (spec.pop("components", None) or {}).items():
            child_spec = dict(child_spec)
            for field, source in (child_spec.pop("inputs", None) or {}).items():
                bindings[f"{name}.{field}"] = source
            implied = getattr(_component_class(child_spec), "config_inputs", None)
            if implied is not None:
                for field, source in implied(child_spec).items():
                    bindings[f"{name}.{field}"] = source
            components[name] = build(child_spec)
        if "output" in spec:
            spec["outputs"] = spec.pop("output")
        unknown = set(spec) - {"outputs", "prefixes", "shared", "parameter_overrides",
                               "parameter_order", "require_feasible"}
        if unknown:
            raise ValueError(f"Unknown composite configuration keys: {sorted(unknown)}")
        return cls(components, bindings=bindings, **spec)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def components(self) -> dict[str, IModel]:
        """The named child models."""
        return dict(self._components)

    def component(self, name: str) -> IModel:
        """Return child *name*."""
        return self._components[name]

    @property
    def run_order(self) -> list[str]:
        """Component names in execution order (set by :meth:`initialize`)."""
        return list(self._run_order)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Initialize children, check wiring, and forward parameters and constraints."""
        self._reset_registries()
        for child in self._components.values():
            child.initialize()
        self._resolve_bindings()
        self._forward_parameters()
        for name, child in self._components.items():
            prefix = self._prefixes[name]
            for cname, cdesc in zip(child.inequality_constraint_names,
                                    child.inequality_constraint_descriptions):
                self.register_inequality_constraint(ConstraintRecord(f"{prefix}{cname}", cdesc))
        self._apply_overrides()
        self._apply_order()
        self._plan_outputs()
        self._state = ModelState.INITIALIZED

    def _resolve_bindings(self) -> None:
        self._inputs = {name: [] for name in self._components}
        deps: dict[str, set[str]] = {name: set() for name in self._components}
        for target, source in self._bindings.items():
            t_child, t_field = _split(target)
            s_child, s_field = _split(source)
            if t_child not in self._components:
                raise ValueError(f"binding target {target!r}: unknown component {t_child!r}")
            available = self._components[t_child].input_field_names
            if t_field not in available:
                raise ValueError(
                    f"binding target {target!r}: {t_child!r} accepts inputs {available}")
            if s_child != _DATA:
                if s_child not in self._components:
                    raise ValueError(f"binding source {source!r}: unknown component {s_child!r}")
                if s_child == t_child:
                    raise ValueError(f"binding {target!r} <- {source!r} feeds a component itself")
                deps[t_child].add(s_child)
            self._inputs[t_child].append((t_field, s_child, s_field))

        order: list[str] = []
        remaining = list(self._components)
        while remaining:
            ready = [n for n in remaining if deps[n] <= set(order)]
            if not ready:
                raise ValueError(f"bindings form a cycle among {remaining}")
            order.append(ready[0])
            remaining.remove(ready[0])
        self._run_order = order

    def _register_forwarded(self, name: str, source: ScalarParameter | VectorParameter) -> None:
        if name in self._scalar_parameters or name in self._vector_parameters:
            raise ValueError(f"parameter name collision: {name!r}")
        if isinstance(source, VectorParameter):
            self.register_vector_parameter(VectorParameter(
                name, values=np.array(source.values, dtype=float).copy(),
                lower_bounds=np.array(source.lower_bounds, dtype=float).copy(),
                upper_bounds=np.array(source.upper_bounds, dtype=float).copy(),
                units=source.units, description=source.description, calibrate=source.calibrate))
        else:
            self.register_scalar_parameter(ScalarParameter(
                name, value=source.value, lower_bound=source.lower_bound,
                upper_bound=source.upper_bound, units=source.units,
                description=source.description, calibrate=source.calibrate))

    def _forward_parameters(self) -> None:
        self._param_map = {}
        shared_of: dict[tuple[str, str], str] = {}
        for shared_name, members in self._shared.items():
            if not members:
                raise ValueError(f"shared parameter {shared_name!r} has no members")
            for address in members:
                child, pname = _split(address)
                if child not in self._components:
                    raise ValueError(f"shared {shared_name!r}: unknown component {child!r}")
                if pname not in self._components[child].scalar_parameter_names:
                    raise ValueError(f"shared {shared_name!r}: {child!r} has no parameter {pname!r}")
                shared_of[(child, pname)] = shared_name

        for name, child in self._components.items():
            prefix = self._prefixes[name]
            for pname in child.scalar_parameter_names:
                shared_name = shared_of.get((name, pname))
                if shared_name is not None:
                    if shared_name not in self._param_map:
                        self._register_forwarded(shared_name, child.get_scalar_parameter(pname))
                        self._param_map[shared_name] = []
                    self._param_map[shared_name].append((child, pname, "scalar"))
                    continue
                cname = f"{prefix}{pname}"
                self._register_forwarded(cname, child.get_scalar_parameter(pname))
                self._param_map[cname] = [(child, pname, "scalar")]
            for vname in child.vector_parameter_names:
                cname = f"{prefix}{vname}"
                self._register_forwarded(cname, child.get_vector_parameter(vname))
                self._param_map[cname] = [(child, vname, "vector")]

    def _apply_overrides(self) -> None:
        allowed = {"value", "lower_bound", "upper_bound", "calibrate", "units", "description"}
        for name, options in self._overrides.items():
            unknown = set(options) - allowed
            if unknown:
                raise ValueError(f"override {name!r}: unknown fields {sorted(unknown)}")
            if name not in self._scalar_parameters:
                raise ValueError(f"override for unknown parameter {name!r}")
            self._scalar_parameters[name].update(**options)
            bounds = {k: v for k, v in options.items() if k in ("lower_bound", "upper_bound")}
            for child, pname, kind in self._param_map.get(name, []):
                if kind == "scalar" and bounds:
                    child.get_scalar_parameter(pname).update(**bounds)

    def _apply_order(self) -> None:
        if not self._parameter_order:
            return
        unknown = [n for n in self._parameter_order if n not in self._scalar_parameters]
        if unknown:
            raise ValueError(f"parameter_order names unknown parameters: {unknown}")
        ordered = {n: self._scalar_parameters[n] for n in self._parameter_order}
        ordered.update({n: p for n, p in self._scalar_parameters.items() if n not in ordered})
        self._scalar_parameters = ordered

    def _plan_outputs(self) -> None:
        self._output_plan = []
        self.register_output_field(FieldRecord("datetime", description="Simulation time step"))
        spec = self._outputs_spec
        if spec is None:
            spec = {f"{name}.{field}": f"{name}.{field}"
                    for name, child in self._components.items()
                    for field in child.output_field_names if field != "datetime"}
        for out_name, entry in spec.items():
            if isinstance(entry, str):
                kind, terms = "alias", [entry]
            elif isinstance(entry, Mapping) and len(entry) == 1 and next(iter(entry)) in ("sum", "product"):
                kind = next(iter(entry))
                terms = list(entry[kind])
            elif isinstance(entry, Sequence):
                kind, terms = "sum", list(entry)
            else:
                raise ValueError(f"output {out_name!r}: expected an address, a list, or "
                                 f"{{'sum'|'product': [...]}}; got {entry!r}")
            if not terms:
                raise ValueError(f"output {out_name!r} has no terms")
            parsed = []
            for term in terms:
                child, field = _split(term)
                if child != _DATA and child not in self._components:
                    raise ValueError(f"output {out_name!r}: unknown component {child!r}")
                parsed.append((child, field))
            self._output_plan.append((out_name, kind, parsed))
            first_child, first_field = parsed[0]
            record = (self._components[first_child]._output_fields.get(first_field)
                      if first_child != _DATA else None)
            self.register_output_field(FieldRecord(
                out_name, units=record.units if record else "",
                description=record.description if record else "",
                calibratable=True))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _sync_children(self) -> None:
        # The composite's value, bounds and calibrate flag are authoritative for
        # forwarded parameters, so settings applied to the composite reach the
        # children.
        for cname, members in self._param_map.items():
            if members[0][2] == "scalar":
                param = self._scalar_parameters[cname]
                for child, pname, _ in members:
                    target = child._scalar_parameters[pname]
                    target.lower_bound = param.lower_bound
                    target.upper_bound = param.upper_bound
                    target.value = param.value
                    target.calibrate = param.calibrate
            else:
                values = self._vector_parameters[cname].values
                for child, pname, _ in members:
                    child._vector_parameters[pname].values[:] = values

    def validate(self) -> bool:
        """Validate composite bounds and every child."""
        if not self.parameters_valid():
            return False
        self._sync_children()
        if not all(child.validate() for child in self._components.values()):
            return False
        self._state = ModelState.VALIDATED
        return True

    def prepare(self, data: pd.DataFrame, **child_kwargs: Mapping[str, Any]) -> None:
        """Prepare every child once on *data*.

        Bound inputs missing from *data* are added as zero columns so children
        that read them in ``prepare`` see the right length; the real values
        arrive with :meth:`IModel.set_input` before each prediction.

        :param data: Forcing frame with a ``datetime`` column.
        :param child_kwargs: ``{component_name: {keyword: value}}`` passed to
            that child's ``prepare`` (e.g. pre-fitted patterns).
        """
        unknown = set(child_kwargs) - set(self._components)
        if unknown:
            raise ValueError(f"prepare() keywords for unknown components: {sorted(unknown)}")
        self._datetime = pd.to_datetime(data["datetime"]).to_numpy()
        self._data_columns = {}
        for inputs in self._inputs.values():
            for _, s_child, s_field in inputs:
                if s_child == _DATA:
                    self._data_columns[s_field] = data[s_field].to_numpy(dtype=float)
        for _, _, terms in self._output_plan:
            for child, field in terms:
                if child == _DATA:
                    self._data_columns[field] = data[field].to_numpy(dtype=float)
        self._sync_children()
        for name in self._run_order:
            child = self._components[name]
            missing = {field: 0.0 for field, _, _ in self._inputs[name] if field not in data.columns}
            frame = data.assign(**missing) if missing else data
            child.prepare(frame, **dict(child_kwargs.get(name, {})))
        self._state = ModelState.PREPARED

    def _check_feasible(self) -> None:
        residuals = self.inequality_constraints()
        bad = [n for n, g in zip(self.inequality_constraint_names, residuals) if g > 0.0]
        if bad:
            raise ValueError(f"parameters violate constraints: {bad}")

    def predict_arrays(self) -> dict[str, np.ndarray]:
        """Run the children in order and assemble the outputs."""
        if self._datetime is None:
            raise RuntimeError("Call prepare(data) before predict().")
        self._sync_children()
        if self.require_feasible:
            self._check_feasible()
        results: dict[str, Mapping[str, np.ndarray]] = {}
        for name in self._run_order:
            child = self._components[name]
            for field, s_child, s_field in self._inputs[name]:
                source = (self._data_columns[s_field] if s_child == _DATA
                          else results[s_child][s_field])
                child.set_input(field, source)
            results[name] = child.predict_arrays()

        def lookup(child: str, field: str) -> np.ndarray:
            if child == _DATA:
                return self._data_columns[field]
            try:
                return results[child][field]
            except KeyError:
                raise KeyError(f"component {child!r} produced no output {field!r}; "
                               f"available: {sorted(results[child])}") from None

        out: dict[str, np.ndarray] = {"datetime": self._datetime}
        for out_name, kind, terms in self._output_plan:
            if kind == "alias":
                out[out_name] = lookup(*terms[0])
                continue
            acc = np.array(lookup(*terms[0]), dtype=float, copy=True)
            for child, field in terms[1:]:
                if kind == "sum":
                    acc += lookup(child, field)
                else:
                    acc *= lookup(child, field)
            out[out_name] = acc
        self._state = ModelState.PREDICTED
        return out

    def predict(self) -> pd.DataFrame:
        """Return ``pd.DataFrame(self.predict_arrays())``."""
        return pd.DataFrame(self.predict_arrays())

    def inequality_constraints(self) -> list[float]:
        """Child residuals, concatenated in component order."""
        self._sync_children()
        residuals: list[float] = []
        for child in self._components.values():
            residuals.extend(child.inequality_constraints())
        return residuals

    def finalize(self) -> None:
        """Finalize every child."""
        for child in self._components.values():
            child.finalize()
        self._datetime = None
        self._data_columns = {}
        self._state = ModelState.FINALIZED


# ---------------------------------------------------------------------------
# Preset: abstraction -> unit hydrograph (-> seasonality)
# ---------------------------------------------------------------------------

@registry.register
class AbstractionUHModel(CompositeModel):
    """Abstraction, then a unit hydrograph, optionally times a seasonal factor.

    A :class:`CompositeModel` preset:

    1. **abstraction** (default :class:`~sparsehydro.models.rdii.IAModel`)
       turns rainfall into effective rainfall;
    2. **uh** (default :class:`~sparsehydro.models.unithydrograph.PeakTailUH`)
       routes it to ``Q_pred``;
    3. **seasonality** (optional :class:`~sparsehydro.models.SeasonalityModel`)
       multiplies ``Q_pred`` by a peaking factor, with ``seas_baseline`` fixed
       at 1.

    Parameter names: abstraction parameters keep their names, the UH's are
    ``uh_{name}`` and the seasonality's ``seas_{name}``.

    :param abstraction: Model emitting effective rainfall as its first
        non-``datetime`` output.
    :param uh: Unit hydrograph component.
    :param seasonality: Optional multiplicative seasonal model.
    :param units: ``"imperial"`` (default) or ``"metric"``.
    """

    model_name: ClassVar[str] = "abstraction-uh"

    def __init__(
        self,
        abstraction: IModel | None = None,
        uh: IUnitHydroComponent | None = None,
        seasonality: SeasonalityModel | None = None,
        units: str = "imperial",
    ) -> None:
        if units not in ("imperial", "metric"):
            raise ValueError(f"units must be 'imperial' or 'metric'; got {units!r}")
        self._units = units
        self._rainfall_col = "rainfall_in" if units == "imperial" else "rainfall_mm"
        self._abstraction = abstraction if abstraction is not None else IAModel(units=units)
        self._uh = uh if uh is not None else PeakTailUH()
        self._seasonality = seasonality
        components: dict[str, IModel] = {"abstraction": self._abstraction, "uh": self._uh}
        prefixes = {"abstraction": "", "uh": "uh_"}
        overrides = {}
        if seasonality is not None:
            components["seasonality"] = seasonality
            prefixes["seasonality"] = "seas_"
            overrides["seas_baseline"] = {"value": 1.0, "calibrate": False}
        super().__init__(components, prefixes=prefixes, parameter_overrides=overrides,
                         outputs={"Q_pred": "uh.Q_pred"})

    def initialize(self) -> None:
        """Discover the excess and seasonal output names, then wire the preset."""
        self._abstraction.initialize()
        excess = next(n for n in self._abstraction.output_field_names if n != "datetime")
        uh_input = self._uh.input_field_names[0]
        uh_output = getattr(self._uh, "_output_column", "Q_pred")
        self._bindings = {f"uh.{uh_input}": f"abstraction.{excess}"}
        q = f"uh.{uh_output}"
        if self._seasonality is not None:
            self._seasonality.initialize()
            seasonal = next(n for n in self._seasonality.output_field_names if n != "datetime")
            self._outputs_spec = {"Q_pred": {"product": [q, f"seasonality.{seasonal}"]}}
        else:
            self._outputs_spec = {"Q_pred": q}
        super().initialize()
        self._output_fields["Q_pred"] = FieldRecord(
            "Q_pred", units="", description="Predicted stormflow", calibratable=True)

    def prepare(self, data: pd.DataFrame) -> None:
        """Prepare from a frame with ``datetime`` and a rainfall column.

        The rainfall column may be ``rain``, ``rainfall_in`` or ``rainfall_mm``;
        an optional ``temperature_c`` feeds a temperature-driven abstraction.

        :raises ValueError: If no recognised rainfall column is present.
        """
        df = data.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        rain_source = next((c for c in ("rain", self._rainfall_col, "rainfall_in", "rainfall_mm")
                            if c in df.columns), None)
        if rain_source is None:
            raise ValueError(
                "prepare() data needs a rainfall column ('rain', 'rainfall_in', or 'rainfall_mm')."
            )
        frame = pd.DataFrame({
            "datetime": df["datetime"].values,
            self._rainfall_col: df[rain_source].to_numpy(dtype=float),
        })
        if "temperature_c" in df.columns:
            frame["temperature_c"] = df["temperature_c"].to_numpy(dtype=float)
        super().prepare(frame)


__all__ = ["CompositeModel", "AbstractionUHModel"]
