# RDII Physics-Based Module — Design Document

> **Historical design note.** Implemented in `sparsehydro.models.rdii`. Module paths below may be out of date; see the API reference for the current layout.

## Motivation

SWMM's standard RDII calibration requires 12 monthly RTK parameter sets because a single set conflates soil moisture (Initial Abstraction) state with infrastructure leakage fraction. This module replaces monthly RTK tables with a physics-based Initial Abstraction (IA) recovery model driven by temperature and elapsed dry time, producing emergent seasonal RDII variation from one parameter set per sewershed.

---

## Module Layout

```
sparsehydro/rdii/
├── __init__.py               # Public surface; guards viz imports
├── initial_abstraction.py    # IAModel class + static compute_excess_series()
├── rtk_triangle.py           # RTKTriangle(IUnitHydroComponent) + triangular_uh()
├── model.py                  # RDIIModel(IModel), model_name="rdii"
├── combined_model.py         # CombinedHydroModel — configurable IA + any-UH composite
├── objectives.py             # peak_weighted_mse(), nash_sutcliffe()
└── visualization.py          # plot_timeseries(), plot_pareto_evolution(),
                              # plot_parallel_coordinates()
```

Install optional dependencies with:

```bash
pip install sparsehydro[rdii]   # pymoo>=0.6, plotly>=5.0, scipy>=1.10
```

---

## Mathematics

### Initial Abstraction Recovery (dry interval)

During dry periods, available capacity recovers toward `IA_max` with an
exponential approach driven by a temperature-dependent rate:

$$IA_{avail}(t+\Delta t) = IA_{max} - \bigl(IA_{max} - IA_{avail}(t)\bigr)
  \cdot e^{-k_{rec}(T)\,\Delta t}$$

$$k_{rec}(T) = \begin{cases}
  k_0 + k_T \cdot e^{\,\theta\,(T - T_{ref})} & T \geq T_{freeze} \\
  0 & T < T_{freeze}
\end{cases}$$

- `k_0` — base rate capturing gravity drainage and capillary redistribution.
- `k_T · exp(θ(T − T_ref))` — thermally-driven evapotranspiration.
- Recovery is suppressed below `T_freeze`, producing high winter RDII naturally.

### Initial Abstraction Depletion and Rainfall Excess (storm pulse)

The wet step integrates the depletion ODE exactly over the step, assuming the
step's rainfall $\Delta P$ falls uniformly in time (p = cumulative rain within
the step):

$$\frac{d\,IA_{avail}}{dp} = -k_{dep}\,IA_{avail}
\quad\Rightarrow\quad
IA_{avail}(p) = IA_{avail}(t)\,e^{-k_{dep}\,p}$$

The instantaneous excess rate is $\max\bigl(0,\; 1 - k_{dep}\,IA_{avail}(p)\bigr)$,
which integrates to the closed form (`_wet_step_excess()`):

- $k_{dep}\,IA_{avail} \leq 1$ (bucket never absorbs the full rain rate):

$$P_{excess} = \Delta P - IA_{avail}\,\bigl(1 - e^{-k_{dep}\,\Delta P}\bigr)$$

- $k_{dep}\,IA_{avail} > 1$: no excess until $p^* = \ln(k_{dep}\,IA_{avail})/k_{dep}$
  of rain has been absorbed, then

$$P_{excess} = (\Delta P - p^*) - \Bigl(\tfrac{1}{k_{dep}} - IA_{avail}\,e^{-k_{dep}\,\Delta P}\Bigr)$$

End-of-step state in both regimes:

$$IA_{avail}(t+\Delta t) = IA_{avail}(t) \cdot e^{-k_{dep}\,\Delta P}$$

Because the step is integrated exactly, results are **invariant to sub-step
refinement** — running at a daily step is equivalent to uniformly
disaggregating each day to an arbitrarily fine timestep. The storage is
drained by the water it abstracts (mass-conserving), so antecedent wetness
directly controls the storm response. Note that total absorption is maximised
at an interior $k_{dep} \approx 1/IA_{max}$: $k_{dep} \to 0$ disables
abstraction entirely, while $k_{dep} \to \infty$ destroys the capacity within
the first instants of rain with negligible uptake.

### Degree-Day Snow Model (optional, `IAModel(snow=True)`)

Precipitation on cold days is stored as snow-water equivalent (SWE) and
released as melt during warm spells, feeding the IA wet/dry logic as liquid
input:

$$T \leq T_{snow}:\quad SWE \mathrel{+}= \Delta P,\qquad P_{liquid} = 0$$

$$T > T_{snow}:\quad melt = \min\bigl(SWE,\; ddf\,(T - T_{snow})\,\Delta t_{days}\bigr),
\qquad P_{liquid} = \Delta P + melt$$

A single threshold `snow_T` serves as both the rain/snow partition and the
melt base (no inter-parameter constraint needed). Rain-on-snow events add
melt to the day's rainfall. `snow_ddf = 0` disables melt, so the optimizer can
switch the snow influence off if the data does not support it. This produces
melt-driven flow on days with little or no rain — the signature of cold-season
wet-antecedent peak events.

### Triangular RTK Unit Hydrograph

For each triangle *i* with parameters $(R_i, T_i, K_i)$:

- **Time base:** $T_i \cdot (1 + K_i)$
- **Peak ordinate:** $\dfrac{2}{T_i \cdot (1 + K_i)}$ (ensures unit area)
- **RDII contribution:** $R_i \cdot \bigl(P_{excess} \circledast h_i\bigr)$

Total RDII flow:

$$Q_{RDII}(t) = \sum_{i=1}^{N} R_i \cdot \bigl(P_{excess} \circledast h_i\bigr)(t)$$

The convolution uses `numpy.convolve` for short series and
`scipy.signal.fftconvolve` when `max(n, kernel_len) > 500`.

---

## Parameter Specification

### Initial Abstraction — 7 parameters

| Name          | Default | Bounds        | Units | Description                              |
|---------------|---------|---------------|-------|------------------------------------------|
| `ia_max`      | 5.0     | [0.1, 50.0]   | mm    | Maximum abstraction capacity             |
| `ia_k0`       | 0.05    | [0.001, 1.0]  | 1/hr  | Base recovery rate (gravity drainage)    |
| `ia_kT`       | 0.02    | [0.0, 0.5]    | 1/hr  | Temperature-sensitive recovery coeff.   |
| `ia_theta`    | 0.1     | [0.0, 0.5]    | 1/°C  | Temperature sensitivity exponent         |
| `ia_T_ref`    | 20.0    | [0.0, 30.0]   | °C    | Reference temperature for ET scaling     |
| `ia_k_dep`    | 0.3     | [0.01, 5.0]   | 1/mm  | Depletion rate per mm of rainfall (rule of thumb ≈ 1/IA_max) |
| `ia_T_freeze` | 0.0     | [-5.0, 5.0]   | °C    | Recovery suppressed below this temp.    |

Defaults and bounds above are metric; imperial units (inches) use
`ia_max` default 0.2 in [0.004, 2.0] and `ia_k_dep` default 7.62 /in [0.25, 127].

### Snow — 2 additional parameters when `IAModel(snow=True)`

| Name       | Default | Bounds       | Units        | Description                                |
|------------|---------|--------------|--------------|--------------------------------------------|
| `snow_T`   | 1.0     | [-2.0, 4.0]  | °C           | Rain/snow partition threshold & melt base  |
| `snow_ddf` | 3.0     | [0.0, 12.0]  | mm/(°C·day)  | Degree-day snowmelt factor                 |

(Imperial: `snow_ddf` default 0.12 in/(°C·day), bounds [0.0, 0.5].)
With `snow=False` (default) neither parameter is registered — fully backward
compatible.

### RTK Triangle — 3 parameters per triangle i (1..N)

| Name  | Default (i=1/2/3)   | Bounds       | Units | Description                          |
|-------|---------------------|--------------|-------|--------------------------------------|
| `R_i` | 0.05 / 0.03 / 0.02  | [0.0, 1.0]   | —     | Fraction of P_excess entering sewer  |
| `T_i` | 1.0 / 12.0 / 72.0   | [0.1, 240.0] | hr    | Time to peak                         |
| `K_i` | 1.5 / 2.0 / 3.0     | [1.001, 10.0]| —     | Recession-to-peak ratio (strictly >1)|

**Constraint:** `sum(R_i) ≤ 1.0` — enforced in `RDIIModel.validate()`.

For N=3: 7 + 9 = **16 ScalarParameters** total.

---

## `RDIIModel` Usage

```python
import pandas as pd
from sparsehydro.models.rdii import RDIIModel

# Build input DataFrame
df = pd.DataFrame({
    "datetime":      pd.date_range("2024-01-01", periods=72, freq="h"),
    "rainfall_mm":   rainfall_array,    # required
    "flow_cfs":      observed_flow,     # optional (used by optimizer)
    "temperature_c": temperature_array, # optional (defaults to ia_T_ref)
})

model = RDIIModel(n_triangles=3)
model.initialize()
model.validate()
model.prepare(df)
result = model.predict()
# result columns: datetime, rdii_mm, p_excess_mm
model.finalize()
```

`prepare()` infers `dt_hours` from the median inter-row timedelta.
Missing `temperature_c` is filled with the current `ia_T_ref` parameter value.

---

## Multi-Objective Optimization

Calibration uses the generic `CalibrationProblem` + `ISolver` framework in
`sparsehydro.calibration`.  `CalibrationProblem` auto-discovers all
`ScalarParameter` bounds from the model registry and exposes any
`inequality_constraints()` to the solver.

### Objectives

1. **Peak-weighted MSE** (minimize) — `PeakWeightedMSE(power=1.0)`:

$$PWMSE = \frac{\sum_t w_t\,(Q_{obs,t} - Q_{pred,t})^2}{\sum_t w_t},
\quad w_t = \left(\frac{Q_{obs,t}}{\overline{Q}_{obs}}\right)^{p}$$

`power=1` (default) gives linear weighting; larger values sharpen the focus on
peaks (with `power=2` a flow at 4× the mean carries 16× the weight); `power=0`
reduces to plain MSE.

2. **Nash-Sutcliffe Efficiency** (maximize → minimized as −NSE internally):

$$NSE = 1 - \frac{\sum_t (Q_{obs,t}-Q_{pred,t})^2}{\sum_t (Q_{obs,t}-\overline{Q}_{obs})^2}$$

Additional objectives available in `sparsehydro.calibration`: `LogNSE(epsilon)`
(low-flow fidelity in log space — pairs well with a peak-focused objective for
a well-spread Pareto front), `KGE`, `PBIAS`, `VolumeRelativeError`,
`IndexOfAgreement`, `MSE`, `RMSE`, `MAE`. Note that `PeakWeightedMSE` and
`NashSutcliffe` are both squared-error metrics and strongly correlated;
pairing `PeakWeightedMSE(power=2)` with `LogNSE()` produces a genuine
peak-vs-baseflow trade-off.

### Usage

```python
from sparsehydro.models.rdii import RDIIModel
from sparsehydro.calibration import (
    CalibrationProblem, NSGAIISolver, PeakWeightedMSE, NashSutcliffe,
)

model = RDIIModel(n_triangles=3)
model.initialize()
model.validate()

problem = CalibrationProblem(
    model=model,
    data=df,                          # datetime, rainfall_mm, flow_cfs [, temperature_c]
    objectives=[PeakWeightedMSE(), NashSutcliffe()],
    column_map={
        "observed":  "flow_cfs",
        "predicted": "rdii_cfs",
    },
)
result = NSGAIISolver(pop_size=100, n_gen=200, seed=42).solve(problem)

print(result.pareto_X.shape)           # (n_solutions, n_params)
best_params = result.best_by("nash_sutcliffe")  # parameter Series
```

**NSGA-II settings:** SBX crossover (prob=0.9, η=15), polynomial mutation (η=20),
random float sampling, population stored every generation.

---

## Visualization

### Time Series

```python
from sparsehydro.visualization import plot_timeseries

fig = plot_timeseries(
    datetime=df["datetime"],
    rainfall_mm=df["rainfall_mm"],
    observed_flow=df["flow_cfs"],
    predicted_flow=result_df["rdii_mm"],
)
fig.show()
```

Dual-axis layout: rainfall bar chart (top row, Y-axis inverted) and
observed vs predicted flow lines (bottom row), shared X-axis.

### Pareto Front Evolution

```python
from sparsehydro.visualization import plot_pareto_evolution

fig = plot_pareto_evolution(opt_result)
fig.show()
```

Animated scatter plot with a generation slider. Each frame shows all
population solutions (grey) with Pareto-front solutions highlighted
(Viridis colorscale by NSE). A Play button animates the evolution.

### Parallel Coordinates

```python
from sparsehydro.visualization import plot_parallel_coordinates

fig = plot_parallel_coordinates(opt_result, color_by="nse")
fig.show()
```

One line per Pareto solution; axes = all calibrated parameters + PWMSE + NSE.
Drag the endpoints of any axis to filter solutions (Plotly `constraintrange`).
Color encodes NSE (or PWMSE when `color_by="pwmse"`).

---

## `RDIIResult` API

| Attribute / Method       | Description                                                   |
|--------------------------|---------------------------------------------------------------|
| `history`                | `list[GenerationRecord]` — one per generation                 |
| `pareto_X`               | `ndarray (m, n_params)` — Pareto parameter vectors            |
| `pareto_F`               | `ndarray (m, 2)` — [PWMSE, −NSE] for Pareto solutions         |
| `pareto_nse`             | `ndarray (m,)` — NSE values (−pareto_F[:, 1])                 |
| `pareto_pwmse`           | `ndarray (m,)` — PWMSE values                                 |
| `param_names`            | `list[str]` — parameter names in column order                 |
| `best_by_nse()`          | Parameter vector with highest NSE                             |
| `best_by_pwmse()`        | Parameter vector with lowest PWMSE                            |
| `to_pareto_dataframe()`  | Tidy DataFrame: generation, params, pwmse, nse, is_pareto     |

---

## Verification Checklist

1. `pytest tests/test_rdii_model.py -v` — all tests pass.
2. `np.sum(triangular_uh(tri, dt)) * dt ≈ 1.0` for multiple (T, K, dt) combos.
3. Zero-rainfall input → zero RDII output.
4. `registry.is_registered("rdii")` → `True` after `import sparsehydro`.
5. NSE of best Pareto solution improves between generation 1 and generation N.
6. Plotly figures render: dual-axis time series, slider navigates generations,
   parallel coord axis dragging filters solutions correctly.
