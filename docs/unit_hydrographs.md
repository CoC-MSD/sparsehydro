# Unit hydrographs

sparsehydro has one library of unit hydrographs (UHs). Every formulation is
registered by name, uses physical time units (hours), and gives the same
response whatever time step the model runs at.

## How a unit hydrograph is represented

Each UH is defined by the cumulative distribution $F(t)$ of its unit-area
instantaneous response, with $t$ in hours. For a time step $\Delta t$ the
kernel ordinate for step $j$ is the exact average over that step:

$$k_j = \frac{F\big((j+1)\Delta t\big) - F\big(j\,\Delta t\big)}{\Delta t} \qquad [1/\mathrm{hr}]$$

so $\sum_j k_j\,\Delta t = 1$ exactly. Convolving rainfall depth per step,
$r_i$, with these ordinates gives the instantaneous outflow at the end of each
step for rain falling uniformly within each step:

$$Q_j = A \sum_i r_i\, k_{j-i}$$

Two consequences:

* **The time step does not change the response.** An hourly storm split
  evenly into 5-minute steps gives, at every hour boundary, exactly the hourly
  prediction.
* **$A$ does not depend on the time step.** The flow volume is
  $\sum_j Q_j\,\Delta t = A \sum_i r_i$. With rain in inches and $Q$ in cfs,
  $A$ is in cfs per (in/hr), about $1.0083\times$ the contributing area in
  acres.

Responses with infinite support (gamma, lognormal, Clark, ...) are truncated
at the $1 - 10^{-6}$ quantile or 720 hours, whichever is earlier, and
renormalised. Any UH accepts an optional pure delay `td` [hr]
(`delay=` in the constructor), applied as $F(t - t_d)$.

## Formulations

| Class | Registry name | Parameters (besides `A`, optional `td`) | Shape flexibility |
|---|---|---|---|
| `NashUH` | `nash-uh` | `n` reservoirs, `K` [hr] | Medium |
| `GammaUH` | `gamma-uh` | `alpha`, `theta` [hr] | Medium |
| `GammaDelayUH` | `gamma-delay-uh` | `alpha`, `theta`, `td` [hr] | Medium |
| `SCSUH` | `scs-uh` | `Tp` [hr], `PRF` (fixed, 484) | Low |
| `SnyderUH` | `snyder-uh` | `tL` [hr], `Cp` | Low–medium |
| `ClarkUH` | `clark-uh` | `Tc` [hr], `Rs` [hr], optional time-area curve | Medium |
| `ModClarkUH` | `modclark-uh` | `Tc`, `Rs` + per-cell travel data | High |
| `WeibullUH` | `weibull-uh` | `k`, `lam` [hr] | Medium |
| `LognormalUH` | `lognormal-uh` | `mu`, `sigma` (of $\ln t$, $t$ in hours) | Medium–high |
| `GeneralizedGammaUH` | `gengamma-uh` | `a`, `c`, `scale` [hr] | High |
| `MixtureUH` | `mixture-uh` | weights + prefixed component parameters | Very high |
| `PeakTailUH` | `peak-tail-uh` | `w`, `peak_Tp`, `peak_Tb`, `tail_alpha`, `tail_theta` | High |
| `TriangleUH` | `triangle-uh` | `Tp`, `Tb` [hr] | Low |
| `RectangleUH` | `rectangle-uh` | `Tr` [hr] | Low |
| `DecayUH` | `decay-uh` | `K` [hr] | Low |
| `RTKTriangle` | `rtk-triangle` | `R` (fraction), `T` [hr], `K` | Low |

### Nash cascade and gamma

A cascade of $n$ equal linear reservoirs with storage constant $K$ has the
gamma density with shape $n$ and scale $K$ as its response
($F(t) = P(n, t/K)$, the regularised lower incomplete gamma function). Mean
$nK$, variance $nK^2$, peak at $(n-1)K$ for $n \ge 1$. `GammaUH` is the same
response under the statistical names `alpha` and `theta`. `DecayUH` is the
single-reservoir case ($n = 1$).

### SCS/NRCS dimensionless

The SCS peak is $q_p = \mathrm{PRF}\, A_{mi^2} Q_{in} / T_p$ [cfs], which for
a unit-area response is $q_p T_p = \mathrm{PRF}/645.33$ (0.75 for the
standard PRF of 484). sparsehydro uses the gamma form: shape $m+1$ and scale
$T_p/m$, with $m$ chosen so the peak ordinate matches the PRF (PRF 484 gives
$m \approx 3.7$). The peak is exactly at `Tp`; a common estimate is
$T_p \approx 0.6\,T_c$.

### Snyder

Snyder's peak is $q_p = 640\, C_p\, A_{mi^2} / t_L$ [cfs/in], i.e. a PRF of
$640\,C_p$ with the peak at the basin lag $t_L$. `SnyderUH` uses the same gamma
construction as `SCSUH` with those values. Snyder's empirical W50/W75 widths
are not matched separately.

### Clark and ModClark

Clark separates translation from storage. The time-area curve $A(\tau)$ (the
fraction of the basin contributing within travel time $\tau$, scaled to the
time of concentration $T_c$) is routed through one linear reservoir with
storage coefficient $R_s$:

$$F(t) = A(t) - \int_0^{\min(t, T_c)} a(s)\, e^{-(t-s)/R_s}\, ds$$

The default time-area curve is the HEC-HMS synthetic curve; pass
`time_area=(tau, cumulative_area)` for a measured one. The integral is
evaluated in closed form over the piecewise-linear curve, so it is exact at
any time step. As $R_s \to 0$ the response tends to the time-area curve; the
mean response time is the time-area mean plus $R_s$.

`ModClarkUH` builds the time-area curve from per-cell travel indices (travel
distance or time to the outlet) and cell areas: each cell's travel time is
$T_c \cdot d/d_{max}$. With basin-average rainfall, as here, this is exactly
Clark with that curve; spatially distributed rainfall is not modelled.

### Weibull, lognormal, generalized gamma

* `WeibullUH`: $F = 1 - e^{-(t/\lambda)^k}$; a stretched-exponential tail.
* `LognormalUH`: $\ln t \sim N(\mu, \sigma)$; strongly right-skewed, long
  recessions. Median $e^\mu$ hours.
* `GeneralizedGammaUH`: $F = P(a, (t/s)^c)$. $c = 1$ is the gamma
  ($\alpha = a$, $\theta = s$); $a = 1$ is the Weibull ($k = c$,
  $\lambda = s$).

### Mixtures

`MixtureUH` combines any UHs: $F(t) = \sum_i w_i F_i(t)$. Useful when a basin
has distinct fast and slow pathways (for example a mixture of two gammas).
Component $i$'s parameters appear as `c{i}_<name>` (or your own `prefixes`);
weights `w_2 ... w_K` are calibrated and $w_1 = 1 - \sum w_i$. With three or
more components the constraint `weights_sum_leq_1` keeps $w_1 \ge 0$.

```python
from sparsehydro.models.unithydrograph import MixtureUH, NashUH, ClarkUH

uh = MixtureUH([NashUH(n=3, K=0.3), ClarkUH(Tc=6, Rs=12)], weights=[0.7, 0.3], A=25.0)
```

`PeakTailUH` is a mixture of a triangle (`peak_`) and a gamma (`tail_`) with
the tail fraction named `w` and one shared delay.

## Using a unit hydrograph

Standalone, with rain depth per step in a `rain` column:

```python
from sparsehydro.models.unithydrograph import SCSUH

uh = SCSUH(A=20.0, Tp=1.5)
uh.initialize()
uh.prepare(df)                  # columns: datetime, rain
q = uh.predict()["Q_pred"]
```

By name, for configuration-driven code:

```python
from sparsehydro import registry
uh = registry.create("clark-uh", Tc=4.0, Rs=3.0)
```

Inside `RDIIModel`, any mix of UHs can route the rainfall excess. Each
component's kernel is scaled by the model's `R_i` (its `A` is ignored), and
its parameters appear as `uh{i}_<name>`:

```python
from sparsehydro.models.rdii import RDIIModel, RTKTriangle
from sparsehydro.models.unithydrograph import NashUH, ClarkUH

model = RDIIModel(uh_components=[RTKTriangle(R=0.03, T=1.0, K=2.0), NashUH(), ClarkUH()])
```

## Adding a formulation

Subclass `UnitHydrographBase`, implement `_cdf` and
`_register_shape_parameters`, and register the class:

```python
from typing import ClassVar
import numpy as np
from sparsehydro.parameters import ScalarParameter
from sparsehydro.registry import registry
from sparsehydro.models.unithydrograph import UnitHydrographBase


@registry.register
class LogisticUH(UnitHydrographBase):
    """Log-logistic response: F(t) = 1 / (1 + (t / scale)^-beta)."""

    model_name: ClassVar[str] = "loglogistic-uh"

    def __init__(self, A=1.0, scale=1.0, beta=3.0, *, delay=None):
        super().__init__(A, delay=delay)
        self._scale, self._beta = float(scale), float(beta)

    def _register_shape_parameters(self):
        self.register_scalar_parameter(ScalarParameter(
            "scale", self._scale, 0.01, 240.0, units="hr", description="Median time"))
        self.register_scalar_parameter(ScalarParameter(
            "beta", self._beta, 0.5, 20.0, description="Shape"))

    def _cdf(self, t):
        t = np.asarray(t, dtype=float)
        with np.errstate(divide="ignore"):
            return 1.0 / (1.0 + (t / self._p("scale")) ** -self._p("beta"))

    def _ppf(self, q):  # optional: closed-form quantile for fast truncation
        return self._p("scale") * (q / (1.0 - q)) ** (1.0 / self._p("beta"))
```

Then:

* For finite-support shapes, set `_finite_support = True` and override
  `_support_hours()`.
* Express relationships between parameters with a `ConstraintRecord`
  registered in `_register_shape_parameters` and a matching
  `_constraint_residuals()`.
* Add the class to the expected set in `tests/test_uh_library.py`; the
  parametrised tests (unit area, non-negativity, timestep invariance, registry
  round-trip) then cover it automatically. Add tests for its own analytical
  properties.

## Migrating from step-based parameters (before 1.0.0a3)

Earlier versions measured time parameters in time steps, sampled kernels at
points, and computed `Q = conv(rain, kernel * A * dt)`. Old names now raise a
`TypeError` that names the replacement.

| Old | New |
|---|---|
| `GammaUH(A, tt, tp)` | `GammaUH(A * dt, alpha=tt + 1, theta=tp * dt)` |
| `NashUH(A, n, k)` | `NashUH(A * dt, n, K=k * dt)` |
| `TriangleUH(A, tt, tp)` | `TriangleUH(A * dt, Tp=tp * dt, Tb=tt * dt)` |
| `RectangleUH(A, tr)` | `RectangleUH(A * dt, Tr=tr * dt)` |
| `DecayUH(A, alpha)` | `DecayUH(A * dt, K=-dt / ln(alpha))` |
| `GammaDelayUH(A, tt, tp, td)` | `GammaDelayUH(A * dt, alpha=tt + 1, theta=tp * dt, td=td * dt)` |
| `PeakTailUH(..., peak_tp, peak_tt, tail_tt, tail_tp, td)` | `peak_Tp=peak_tp*dt, peak_Tb=peak_tt*dt, tail_alpha=tail_tt+1, tail_theta=tail_tp*dt, td=td*dt` |

`dt` is the old model time step in hours. Results agree closely but not
exactly, because kernels are now bin averages rather than point samples and
long responses are no longer cut at 864 steps.
