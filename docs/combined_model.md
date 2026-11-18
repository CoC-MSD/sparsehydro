# Plan: CombinedHydroModel — Configurable IA + Any-UH Composite

> **Historical design note.** Implemented as `sparsehydro.models.composite.AbstractionUHModel` on top of `CompositeModel`. Module paths below may be out of date; see the API reference for the current layout.

## Context

`RDIIModel` hard-codes the unit-hydrograph type (RTK triangles) and knows the
parameter names of `IAModel` by convention. This works well for the existing
three-triangle RDII use case but cannot be extended to Nash, Gamma, or other
UH shapes without a new model class each time.

The goal is a **single configurable composite** (`CombinedHydroModel`) that
accepts any `IModel`-based IA model and any list of UH component objects,
exposes the full combined parameter set to the calibration framework, and
produces the same `datetime / rdii_cfs / rdii_mm / p_excess_mm` DataFrame
as `RDIIModel`.

---

## Step 1 — Add `IUnitHydroComponent` to `sparsehydro/interfaces.py`

New abstract class that extends `IModel`:

```python
class IUnitHydroComponent(IModel, ABC):
    # Subclasses set this to the name of their R/A amplitude parameter (e.g. "R" or "A").
    # The composite will exclude this param from shape-param re-registration and
    # instead manage its own R_i fraction.
    _amplitude_param_name: ClassVar[str | None] = None

    @abstractmethod
    def get_kernel(self, dt_hours: float, n_steps: int | None = None) -> np.ndarray:
        """Return normalized UH ordinates: np.sum(result) * dt_hours ≈ 1.0."""
```

`IUnitHydroComponent` is still abstract, so `model_name` enforcement in
`__init_subclass__` fires only when a fully concrete subclass is created.

---

## Step 2 — Extend `RTKTriangle` (`sparsehydro/rdii/rtk_triangle.py`)

- Change base class from `IModel` to `IUnitHydroComponent`.
- Add class variable `_amplitude_param_name: ClassVar[str | None] = "R"`.
- Add `get_kernel(dt_hours, n_steps=None) -> np.ndarray`:
  delegates to the existing `triangular_uh(self, dt_hours, n_steps)`.
  No change to existing `predict()` — it still applies R-scaling for standalone use.

---

## Step 3 — Extend `UnitHydrographAdapter` (`sparsehydro/unithydrograph/adapter.py`)

- Change base class from `IModel` to `IUnitHydroComponent`.
- Dynamically set `_amplitude_param_name` in `initialize()`: inspect the
  underlying `UnitHydrograph._registry` for the model; if a param named `"A"`
  exists, set `_amplitude_param_name = "A"`, otherwise `None`.
- Add `get_kernel(dt_hours, n_steps=None) -> np.ndarray`:
  ```
  _sync_to_uh()
  raw = _uh.get_uh(norm=1)          # normalized (area=1), integer steps
  # Trim or zero-pad to n_steps if provided
  return raw[:n_steps] or raw
  ```
  Note: the UH adapter kernel is in data-resolution steps regardless of dt_hours
  (the UnitHydrograph library does not carry dt); dt_hours is accepted for API
  uniformity but only controls n_steps if explicit. Document this assumption.

---

## Step 4 — New `CombinedHydroModel` (`sparsehydro/rdii/combined_model.py`)

### Constructor
```python
def __init__(
    self,
    ia_model: IModel | None = None,
    uh_components: list[IUnitHydroComponent] | None = None,
) -> None
```
Defaults: `IAModel()` + 3 `RTKTriangle` instances (fast/medium/slow) — same
starting point as `RDIIModel(n_triangles=3)`.

Optional sub-model features flow through automatically: e.g.
`CombinedHydroModel(ia_model=IAModel(units="imperial", snow=True))` registers
the degree-day snow parameters (`snow_T`, `snow_ddf`) in the composite registry
alongside the other IA params, making them calibratable with no composite-side
changes (see `docs/rdii_design.md` for the snow model physics).

### Parameter layout in composite's flat registry

| Name pattern        | Source                                      | Notes                        |
|---------------------|---------------------------------------------|------------------------------|
| `area_acres`        | composite own                               | depth→flow conversion        |
| `R_1 … R_N`         | composite own                               | defaults from component's amplitude param |
| `ia_max`, `ia_k0`, … | re-registered from `ia_model` at init time | same names (IAModel already uses `ia_` prefix) |
| `uh1_T`, `uh1_K`, … | re-registered from component 1             | excludes `_amplitude_param_name` |
| `uh2_n`, `uh2_k`, … | re-registered from component 2             | etc.                         |

All re-registered params are **new `ScalarParameter` objects** (copies, not
references). `_sync_to_submodels()` pushes composite values back to sub-model
registries before each `predict()`.

### Lifecycle

**`initialize()`**
1. Call `ia_model.initialize()`, then each `uh.initialize()`.
2. Register `area_acres` (default 100.0, bounds [0.01, 100 000]).
3. For each component i: register `R_i` (default from component's amplitude
   param value if available, else 0.05; bounds [0.0, 1.0]).
4. Re-register all IA model params with original names.
5. Re-register all UH shape params with `uh{i}_` prefix, skipping
   `_amplitude_param_name`.

**`validate()`**
1. Call `ia_model.validate()`.
2. Check `parameters_valid()` for composite.
3. Check `ia_T_freeze < ia_T_ref` if those params exist.
4. Advance to VALIDATED.

**`prepare(data: pd.DataFrame)`**
Accepts same columns as `RDIIModel`: `datetime`, `rainfall_mm`, optional
`flow_cfs` / `temperature_c`.

1. Sort, infer `dt_hours`, fill missing temperature.
2. `_sync_to_submodels()`.
3. `ia_model.prepare(data)` (advances ia_model to PREPARED).
4. Store `_prepared_df`, `_dt_hours`.
5. Advance to PREPARED.

**`predict()`**
Called repeatedly during calibration from PREPARED (or previously PREDICTED) state.

1. `_sync_to_submodels()`.
2. `p_excess = ia_model.predict()["p_excess_mm"].to_numpy()`.
3. `rdii = zeros(n)`.
4. For each component i:
   - `kernel = uh_components[i-1].get_kernel(dt_hours)`
   - `rdii += R_i * fft_or_direct_convolve(p_excess, kernel)[:n]`
5. Clip, convert depth→CFS, return `datetime / rdii_cfs / rdii_mm / p_excess_mm`.

**`finalize()`**  
Release `_prepared_df`, call `ia_model.finalize()`.

**`inequality_constraints()`**  
Returns `[Σ R_i - 1.0]` — same constraint as `RDIIModel`.

### `_sync_to_submodels()`
Iterates over IA and UH registries, writes composite values back:
- IA: `ia_model.get_scalar_parameter(name).value = composite.get_scalar_parameter(name).value`
- UH i: `uh.get_scalar_parameter(orig_name).value = composite.get_scalar_parameter(f"uh{i}_{orig_name}").value`
- UH amplitude: `uh.get_scalar_parameter(amp_name).value = R_i` (keep sub-model's R in sync for standalone consistency)

---

## Step 5 — Update `sparsehydro/rdii/__init__.py`

Add `CombinedHydroModel` to imports and `__all__`.

---

## Critical files

| File | Change |
|------|--------|
| `sparsehydro/interfaces.py` | Add `IUnitHydroComponent` abstract class |
| `sparsehydro/rdii/rtk_triangle.py` | Extend `IUnitHydroComponent`; add `get_kernel()` |
| `sparsehydro/unithydrograph/adapter.py` | Extend `IUnitHydroComponent`; add `get_kernel()` |
| `sparsehydro/rdii/combined_model.py` | **New file** — `CombinedHydroModel` |
| `sparsehydro/rdii/__init__.py` | Export `CombinedHydroModel` |

---

## Reused utilities

- `triangular_uh()` in `sparsehydro/rdii/rtk_triangle.py` — unchanged, used by `RTKTriangle.get_kernel()`
- `IAModel.compute_excess_series()` static method is called internally by `IAModel.predict()` — composite relies on it indirectly
- `_MM_AC_PER_HR_TO_CFS` and `_FFT_THRESHOLD` constants from `model.py` — copy into `combined_model.py`
- `ScalarParameter` dataclass — used to clone sub-model params into composite registry

---

## Verification

```python
from sparsehydro.models.rdii import IAModel, RTKTriangle
from sparsehydro.models import AbstractionUHModel as CombinedHydroModel

# Default usage (matches RDIIModel(n_triangles=3))
model = CombinedHydroModel()
model.initialize()
assert len(model.scalar_parameter_names) == 8 + 2*3 + 1  # IA(7) + area(1) + R_i(3) + uh_i T+K(6)
model.validate()
model.prepare(df)
result = model.predict()
assert set(result.columns) == {"datetime", "rdii_cfs", "rdii_mm", "p_excess_mm"}
model.finalize()

# Mixed UH types (Nash + RTK)
from sparsehydro.models.unithydrograph import NashUH
model2 = CombinedHydroModel(
    ia_model=IAModel(),
    uh_components=[RTKTriangle(R=0.05, T=1.0, K=1.5), NashUH()],
)
model2.initialize()
model2.validate()
model2.prepare(df)
result2 = model2.predict()

# Calibration compatibility
from sparsehydro.calibration import CalibrationProblem, NSGAIISolver
problem = CalibrationProblem(model=model, data=df, objectives=[...], column_map={...})
result = NSGAIISolver(pop_size=20, n_gen=10).solve(problem)
```

Run existing test suite: `python -m pytest tests/` — no regressions expected since
`RTKTriangle` and `UnitHydrographAdapter` remain backward-compatible.
