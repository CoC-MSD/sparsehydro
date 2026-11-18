# Ensemble Visualization & Parameter Configuration Plan

> **Historical design note.** Implemented in `sparsehydro.models.ensemble` and `sparsehydro.visualization`. Module paths below may be out of date; see the API reference for the current layout.

## Goal

Extend `EnsembleModel` and the visualization layer so that after instantiating a
model the user can:

1. **Inspect and configure** all parameter starting values and search bounds in a
   Jupyter-friendly table before optimization.
2. **Visualize** individual component signals, the combined prediction, and a Pareto
   uncertainty band in a single multi-panel figure after calibration.

---

## 1. Extensions to `EnsembleModel` (`sparsehydro/ensemble.py`)

### 1.1 `_param_owner: dict[str, str]`

Built during `initialize()`.  Maps every prefixed parameter name to the alias of the
component that owns it (or `"weights"` for the mixing-weight parameters).

```
{
  "w_1":             "weights",
  "w_2":             "weights",
  "rdii_ia_max":     "rdii",
  "rdii_area_acres": "rdii",
  ...
  "seas_a_hour_of_day_1": "seas",
  ...
}
```

**Construction** — after `_param_maps` is built in `initialize()`:

```python
self._param_owner: dict[str, str] = {}
for i, alias in enumerate(self._aliases, 1):
    self._param_owner[f"w_{i}"] = "weights"
for alias, mapping in zip(self._aliases, self._param_maps):
    for prefixed_name in mapping.values():
        self._param_owner[prefixed_name] = alias
```

### 1.2 `parameter_table() -> pd.DataFrame`

Returns a plain DataFrame with one row per scalar parameter:

| column | description |
|---|---|
| `parameter` | prefixed name (e.g. `rdii_ia_max`) |
| `group` | owning alias (`rdii`, `seas`, `weights`) |
| `value` | current value |
| `lower_bound` | search lower bound |
| `upper_bound` | search upper bound |
| `units` | physical units string |
| `calibrate` | whether the optimizer touches this parameter |
| `description` | human-readable description |

Suitable for `display()` in Jupyter with `.style` applied on top.

### 1.3 `set_parameter(name, *, value, lower_bound, upper_bound, calibrate)`

Thin wrapper around `ScalarParameter.update()` for ergonomic in-notebook
configuration:

```python
ensemble.set_parameter("w_1", value=1.0, calibrate=False)
ensemble.set_parameter("rdii_area_acres", lower_bound=100.0, upper_bound=2000.0)
```

### 1.4 `collect_pareto_predictions(data, result, output_col, **prepare_kwargs) -> np.ndarray`

Applies every solution in `result.pareto_X` to the model and returns an array of
shape `(n_solutions, n_timesteps)` containing the combined output column for each
solution.  Parameter values are saved before the loop and restored in a `finally`
block so the model is left in its original state.  The user must call `prepare(data)`
again before the next `predict()`.

### 1.5 `param_owner` property

Read-only copy of `_param_owner`.

---

## 2. New Visualization Function (`sparsehydro/visualization/timeseries.py`)

### `plot_ensemble_components()`

**Signature:**

```python
def plot_ensemble_components(
    datetime,
    rainfall,                         # array or None
    observed: np.ndarray,
    pred_df: pd.DataFrame,            # output of EnsembleModel.predict()
    aliases: list[str],
    output_name: str = "ensemble_output",
    pareto_predictions: np.ndarray | None = None,   # (n_sol, n_t)
    confidence_percentiles: tuple = (10, 90),
    component_labels: dict | None = None,            # {alias: display_label}
    observed_label: str = "Observed",
    rainfall_label: str = "Rainfall [mm]",
    flow_label: str = "Flow",
    title: str = "Ensemble Components",
) -> go.Figure:
```

**Layout** (3 rows sharing X-axis; 2 rows when `rainfall=None`):

```
Row 1  [14%]  Rainfall bars (inverted axis)          — omitted if rainfall=None
Row 2  [36%]  Component signals                       — one filled-area trace per alias
Row 3  [50%]  Combined predicted + observed + band   — Pareto IQR band (optional)
```

**Row 2 — Component signals:**
Each alias gets a filled-area trace (`fill="tozeroy"`) with a semi-transparent fill
and a solid outline.  Components are plotted on the same axis (not stacked) so their
individual magnitudes can be compared directly.

**Row 3 — Combined + band:**
- **Pareto band**: `go.Scatter` with `fill="toself"` between the lower and upper
  percentile of `pareto_predictions`.  Color: `rgba(220,50,50,0.12)`.
- **Predicted total**: dashed crimson line.
- **Component outlines**: thin dotted lines (same colors as Row 2) repeated in this
  panel so the viewer can see how each component contributes to the total.
- **Observed**: solid black line.

---

## 3. Export Changes

### `sparsehydro/visualization/__init__.py`
- Add `plot_ensemble_components` to the `try` import block and `__all__`.

### `sparsehydro/__init__.py`
- Add `plot_ensemble_components` to the visualization import block and `__all__`.

---

## 4. Notebook Changes (`sparsehydro/notebooks/sheridan.ipynb`)

| Cell ID | Change |
|---|---|
| `b1c2d302` | Add `plot_ensemble_components` to imports; remove `plot_timeseries` |
| `c01183db` | Replace `SheridanEnsemble(...)` with `EnsembleModel(components=[...])` |
| `48c3cafd` | Use `ensemble.parameter_table()` instead of manual `rows` loop |
| *(new)* | Parameter configuration cell: `set_parameter()` examples + re-display table |
| `7b25faa7` | Use `collect_pareto_predictions()` + `plot_ensemble_components()` |

### Instantiation change

```python
ensemble = EnsembleModel(
    components=[
        (CombinedHydroModel(...), lambda df: df["rdii_cfs"].to_numpy()),
        (SeasonalityModel(...),   lambda df: df["sanitary_cfs"].to_numpy()),
    ],
    mode="sum",
    aliases=["rdii", "seas"],
    output_name="total_cfs",
)
ensemble.initialize()
ensemble.validate()
```

### Parameter table cell

```python
param_df = ensemble.parameter_table()
display(param_df.style.apply(_style_group, subset=["group"]).format({...}))
```

### Configuration cell (new)

```python
# Fix mixing weights — amplitude lives in R_i and Fourier coefficients
ensemble.set_parameter("w_1", value=1.0, calibrate=False)
ensemble.set_parameter("w_2", value=1.0, calibrate=False)
# Adjust any bounds before running the optimizer, e.g.:
# ensemble.set_parameter("rdii_area_acres", lower_bound=100.0, upper_bound=2000.0)
```

### Results cell

```python
pareto_preds = ensemble.collect_pareto_predictions(df, ens_result)
ensemble.prepare(df)
ens_sim = ensemble.predict()

fig = plot_ensemble_components(
    datetime=df["datetime"],
    rainfall=df["rainfall_in"].to_numpy(),
    observed=df["flow_cfs"].to_numpy(),
    pred_df=ens_sim,
    aliases=ensemble.aliases,
    output_name=ensemble.output_name,
    pareto_predictions=pareto_preds,
    confidence_percentiles=(10, 90),
    component_labels={"rdii": "RDII (wet weather)", "seas": "Sanitary (Fourier)"},
    flow_label="Flow (CFS)",
    rainfall_label="Rainfall (in)",
    title="Sheridan — Ensemble Components + Pareto Band",
)
fig.show()
```
