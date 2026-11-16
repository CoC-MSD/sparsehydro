# Composing models

sparsehydro is built from small, registered **components**: an abstraction
store, a snow routine, a unit hydrograph, a sanitary baseline. A model for a
particular site is a **composition** of those components. You describe the
composition, in Python or as plain data; you do not write a new class for each
combination.

This page covers:

1. [Components and the registry](#components-and-the-registry)
2. [CompositeModel: components, bindings, outputs](#compositemodel)
3. [Parameters in a composite](#parameters-in-a-composite)
4. [Building from configuration files](#building-from-configuration)
5. [Presets](#presets)
6. [Writing a new component](#writing-a-new-component)

Runnable versions of these ideas are in `examples/` (start with
`examples/03_composite_from_config.py`).

## Components and the registry

Every component is an {class}`~sparsehydro.models.base.IModel` registered under
a short name:

```python
from sparsehydro import registry

registry.names()          # ['abstraction-uh', 'amm', 'centered-weekly-pattern', 'clark-uh', ...]
uh = registry.create("nash-uh", n=3.0, K=2.0)
```

Each component declares what it needs and what it produces:

| Component | Reads | Produces |
|---|---|---|
| unit hydrographs (`nash-uh`, `gamma-uh`, ...) | `rain` | `Q_pred` |
| `initial-abstraction` | `rainfall_in`, optional `temperature_c` | `p_excess_in` |
| `tank-constant`, `tank-linear`, `tank-sqrt` | `rainfall_in` | `p_excess_in` |
| `rdii` | `rainfall_in` (or an excess bound from another component) | `rdii_cfs`, `rdii_in`, `p_excess_in` |
| `degree-day-snow` | `rainfall_in`, `temperature_c` | `liquid_input_in`, `melt_in`, `SWE_in`, ... |

`model.output_field_names` lists a component's outputs, and
`model.input_field_names` lists the inputs another component can supply
through a binding (a unit hydrograph's `rain`, for example).

## CompositeModel

{class}`~sparsehydro.models.composite.CompositeModel` holds named components and
three pieces of wiring.

**Components** are `{name: model}`. Names are plain identifiers; `data` is
reserved for the input table.

**Bindings** are `{"target.input": "source.output"}`. Before each prediction
the source array is handed to the target. The source can be another component
(`"ia.p_excess_in"`) or a column of the input data (`"data.rainfall_in"`).
Components run in an order that respects the bindings; `model.run_order` shows
it.

**Outputs** say what the composite returns (besides `datetime`). Each entry is
an alias (`"q": "uh.Q_pred"`), a sum (a list of addresses, or
`{"sum": [...]}`) or a product (`{"product": [...]}`). Anything more elaborate
belongs in a component of its own. With `outputs=None` every component output
is returned as `"component.field"`.

An abstraction store feeding a fast and a slow unit hydrograph, summed:

```python
from sparsehydro.models import CompositeModel
from sparsehydro.models.rdii import IAModel
from sparsehydro.models.unithydrograph import GammaUH, NashUH

model = CompositeModel(
    {"ia": IAModel(), "fast": NashUH(n=2.0, K=0.5), "slow": GammaUH(alpha=1.0, theta=8.0)},
    bindings={"fast.rain": "ia.p_excess_in", "slow.rain": "ia.p_excess_in"},
    outputs={"stormflow_cfs": ["fast.Q_pred", "slow.Q_pred"]},
)
model.initialize()
model.run_order            # ['ia', 'fast', 'slow']
```

A composite is itself an `IModel`, so it calibrates like any other model and can
be a component of a larger composite.

## Parameters in a composite

Component parameters are registered on the composite as
`"<component>.<parameter>"`, for example `ia.ia_max` or `slow.theta`, so the
same name can appear in two components without clashing.

```python
model.scalar_parameter_names[:4]   # ['ia.ia_max', 'ia.ia_k0', 'ia.ia_kT', 'ia.ia_theta']
model.apply_flat_parameter("slow.theta", 12.0)
```

The constructor (and the configuration) accepts:

`prefixes`
: Change the prefix per component; `""` keeps the component's own names.

`shared`
: `{"name": ["a.param", "b.param"]}` registers one parameter that drives
  several component parameters, for example a drainage area used by both a
  groundwater store and an RDII model.

`parameter_overrides`
: `{"name": {"value": ..., "lower_bound": ..., "upper_bound": ...,
  "calibrate": ...}}`, applied after the component parameters are registered.
  `"calibrate": False` holds a parameter fixed at its value.

`parameter_order`
: Names placed first, in this order, in the calibration vector.

Constraints declared by a component (for example `T1 < T2`) are forwarded with
the same prefix, so optimisers see them.

## Building from configuration

{mod}`sparsehydro.configuration` builds anything registered from plain data. A
node `{"type": name, ...}` is created through the registry; the other keys are
constructor arguments, and nested nodes are built first. In a composite,
a component may carry `inputs` instead of separate `bindings`:

```python
from sparsehydro.configuration import build_model, build_objectives, build_solver

config = {
    "model": {
        "type": "composite",
        "components": {
            "ia": {"type": "initial-abstraction"},
            "fast": {"type": "nash-uh", "n": 2.0, "K": 0.5, "inputs": {"rain": "ia.p_excess_in"}},
            "slow": {"type": "gamma-uh", "alpha": 1.0, "theta": 8.0, "inputs": {"rain": "ia.p_excess_in"}},
        },
        "outputs": {"stormflow_cfs": {"sum": ["fast.Q_pred", "slow.Q_pred"]}},
    },
    "parameters": {
        "ia.ia_T_ref": {"calibrate": False},
        "slow.theta": {"lower_bound": 2.0, "upper_bound": 48.0},
    },
}
model = build_model(config)            # built and initialized
objectives = build_objectives(["nash_sutcliffe", {"type": "pbias"}])
solver = build_solver({"type": "scipy", "maxiter": 40})
```

`build_model` takes either a model node or a mapping with `model` and
`parameters`; `parameters` uses the same keys as `parameter_overrides`. The
same dictionary can be stored as JSON or YAML and read with
{func}`~sparsehydro.configuration.load_config` (YAML needs
`pip install "sparsehydro[yaml]"`):

```python
from sparsehydro.configuration import load_config

model = build_model(load_config("site_model.json"))
```

Unknown names fail early with a message listing the valid ones, for example
`KeyError: Unknown parameter 'nope'. Known: [...]`.

Registered names: `registry.names()` for models,
`objective_registry.names()` (from {mod}`sparsehydro.metrics`) for objectives,
and `solver_registry.names()` (from `sparsehydro.calibration.solvers`) for
solvers.

## Presets

A preset is a composite with the wiring already written: a convenient starting
point, not a different kind of model. Every parameter stays configurable.

- {class}`~sparsehydro.models.composite.AbstractionUHModel`
  (`abstraction-uh`): abstraction, then a unit hydrograph, optionally times a
  seasonal factor.
- {class}`~sparsehydro.models.presets.wholeflow.CompactWholeFlowModel`
  (`wholeflow-compact`) and
  {class}`~sparsehydro.models.presets.wholeflow.PatternWholeFlowModel`
  (`wholeflow-pattern-joint`): snow, a soil/groundwater store, RDII routing
  and a sanitary baseline, giving
  `total_cfs = sanitary + groundwater infiltration + RDII`.

## Components and internal states

Besides its outputs, every model can report a *trace*: the parts its outputs
are made of and the internal states behind them. `predict_arrays()` is
unchanged (calibration never computes the trace); `predict_trace()` runs the
model once and returns the outputs plus every trace field:

```python
model = build_model(config)          # e.g. the composite above
model.prepare(data)
trace = model.predict_trace()
model.trace_field_metadata()         # field, role, component_of, units, description
```

Each trace field is a {class}`~sparsehydro.parameters.FieldRecord` with a
`role`:

| Role | Meaning | Examples |
| --- | --- | --- |
| `component` | An additive part of another series, named by `component_of`; the components of a series sum to it. | each unit hydrograph of an RDII model (`rdii_component_1`, ...); each part of a mixture or peak-tail UH (`peak_Q_pred`, `tail_Q_pred`); each term of a composite `sum` output |
| `state` | A storage or level at the end of each step. | remaining initial abstraction, tank storage, soil and groundwater stores, snowpack |
| `flux` | A rate or per-step amount moving between stores. | rainfall excess, melt, recharge, groundwater infiltration |

A composite reports each child's outputs and trace under `child.field`
(`rdii.rdii_component_2`, `ia.p_excess_in`), at any depth of nesting, and marks
the terms of a summed output as its components. So for the model in the
diagram above, `fast.Q_pred` and `slow.Q_pred` are components of
`stormflow_cfs`, and their own parts, if any, follow.

To make a new component traceable, register its fields with
`register_trace_field(FieldRecord(name, role=..., component_of=...))` in
`initialize()` and return their arrays from `_trace_arrays(outputs)`, which
runs only when a trace is requested. `tests/test_trace.py` checks every
registered model: the outputs in the trace equal `predict_arrays()`, every
trace field has the right length, and components add up.

## Writing a new component

A new formulation only has to be a well-behaved component; composition comes
for free. In short (the full checklist is in `CONTRIBUTING.md`, "Adding a model
formulation"):

1. Subclass {class}`~sparsehydro.models.base.IModel` (or
   `UnitHydrographBase` for a unit hydrograph, which only needs its cumulative
   response).
2. Register parameters with physical units and sensible bounds in
   `initialize()`, and output fields with `FieldRecord`.
3. Implement `predict_arrays()` returning `{field: array}`.
4. To accept an input from another component, list it in
   `input_field_names` and store it in `set_input(name, values)`.
5. Decorate the class with `@registry.register` and give it a `model_name`.

It can then appear in any composite or configuration file by that name.

## Packaging an extension

A component does not have to live in sparsehydro. Registration happens when
the module defining the class is imported, so a separate package only has to
make sure its module gets imported. Declare it in the package's
`pyproject.toml` under the `sparsehydro.extensions` entry-point group:

```toml
[project.entry-points."sparsehydro.extensions"]
my_models = "my_package.models"
```

The same works for objectives and solvers: register them in that module with
`objective_registry.add(name, cls)` and `solver_registry.add(name, factory)`.

Applications (the desktop viewer, scripts, notebooks) then call
{func}`sparsehydro.load_extensions` once before building models by name:

```python
import sparsehydro
from sparsehydro.configuration import build_model

report = sparsehydro.load_extensions()          # installed extensions
report = sparsehydro.load_extensions(["work_in_progress.py"])  # and local files
print(report.loaded, report.failed)

model = build_model({"type": "my-uh"})
```

Entry points are imported once per process. A module that fails to import is
logged and listed in `report.failed` without stopping the others; pass
`strict=True` to raise instead, for example in a test suite or batch job.
