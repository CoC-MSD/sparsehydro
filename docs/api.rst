API Reference
=============

.. contents:: Sections
   :local:
   :depth: 2

----

Core Package
------------

Top-level package
~~~~~~~~~~~~~~~~~

The top-level :mod:`sparsehydro` namespace re-exports only a small, stable
*core* API.  Everything else is accessed through its parent sub-package so that
import paths mirror the package structure:

.. code-block:: python

   # Core (top-level)
   from sparsehydro import (
       IModel, IUnitHydroComponent, ModelState,
       ScalarParameter, VectorParameter, FieldRecord,
       ModelRegistry, registry,
   )

   # Everything else via its parent sub-package
   from sparsehydro.metrics import nash_sutcliffe, kge, NashSutcliffe, KGE
   from sparsehydro.filters import apply_savgol_filter, compute_thresholds
   from sparsehydro.events import detect_events, events_to_dataframe
   from sparsehydro.models import CompositeModel, EnsembleModel, SeasonalityModel
   from sparsehydro.models.rdii import RDIIModel, IAModel, RTKTriangle
   from sparsehydro.models.unithydrograph import GammaUH, NashUH, TriangleUH
   from sparsehydro.calibration import (
       CalibrationProblem, CalibrationResult, SequentialFitter,
       NSGAIISolver, ScipySolver, PlatypusSolver, ParticleSwarmSolver,
   )
   from sparsehydro.configuration import build_model, build_objectives, build_solver
   from sparsehydro.visualization import plot_timeseries, plot_calibration_dashboard

Package layers
~~~~~~~~~~~~~~

Each package imports only from its own layer or the layers below it (enforced
by ``tests/test_layering.py``):

.. list-table::
   :header-rows: 1
   :widths: 10 35 55

   * - Layer
     - Package
     - Contents
   * - 0
     - core modules
     - ``enums``, ``parameters``, ``registry``, ``units``
   * - 1
     - ``metrics``, ``filters``, ``data``, ``sample_data``
     - metric functions and objectives; signal filters; file IO and data services
   * - 2
     - ``events``
     - storm event detection, screening and records
   * - 3
     - ``models``
     - model interface, components, composition and presets
   * - 4
     - ``calibration``
     - problems, solvers, splits, selection and sequential fitting
   * - 5
     - ``configuration``
     - building models, objectives and solvers from plain dicts
   * - 6
     - ``visualization``
     - Plotly figures
   * - 7
     - ``viewer``, ``studies``
     - applications and study-specific workflows (nothing imports ``studies``)

.. automodule:: sparsehydro
   :members:
   :show-inheritance:
   :noindex:

Enumerations
~~~~~~~~~~~~

.. automodule:: sparsehydro.enums
   :members:
   :show-inheritance:
   :undoc-members:

Parameters
~~~~~~~~~~

Parameters are the connective tissue between the model physics and the
calibration engine.  Every numeric quantity that may be optimised must be
registered with the model via a :class:`~sparsehydro.parameters.ScalarParameter`
(single value) or :class:`~sparsehydro.parameters.VectorParameter` (array).

Setting ``calibrate=False`` on a :class:`~sparsehydro.parameters.ScalarParameter`
freezes that parameter at its current value — it is excluded from the
optimisation search space but still applied during ``predict()``:

.. code-block:: python

   p = ScalarParameter("area_acres", value=250.0,
                       lower_bound=1.0, upper_bound=100_000.0,
                       calibrate=False)   # held fixed; not optimised

.. automodule:: sparsehydro.parameters
   :members:
   :show-inheritance:

Model Interface
~~~~~~~~~~~~~~~

All physical models in *sparsehydro* implement :class:`~sparsehydro.models.IModel`.
The lifecycle enforces a consistent prepare → predict → finalize pattern and
keeps the calibration engine decoupled from model internals.

:class:`~sparsehydro.models.IUnitHydroComponent` extends :class:`~sparsehydro.models.IModel`
with a :meth:`~sparsehydro.models.IUnitHydroComponent.get_kernel` method.
Any model implementing this interface can be used as a component inside
:class:`~sparsehydro.models.rdii.RDIIModel`.

.. automodule:: sparsehydro.models.base
   :members:
   :show-inheritance:

Model Registry
~~~~~~~~~~~~~~

.. automodule:: sparsehydro.registry
   :members:
   :show-inheritance:

PyTorch Interface
~~~~~~~~~~~~~~~~~

.. automodule:: sparsehydro.models.torch_model
   :members:
   :show-inheritance:

Units
~~~~~

.. automodule:: sparsehydro.units
   :members:
   :show-inheritance:


----

Metrics
-------

:mod:`sparsehydro.metrics` holds every goodness-of-fit formula exactly once.
Plain functions take observed and simulated arrays; the objective classes wrap
them for calibration (with a direction, an optional mask and a registry name).

.. code-block:: python

   from sparsehydro.metrics import nash_sutcliffe, pbias, kge, fit_summary

   nse = nash_sutcliffe(observed, simulated)
   bias = pbias(observed, simulated)        # signed: positive = over-prediction
   summary = fit_summary(observed, simulated)

.. automodule:: sparsehydro.metrics

Metric Functions
~~~~~~~~~~~~~~~~

.. automodule:: sparsehydro.metrics.functions
   :members:
   :show-inheritance:

Objectives
~~~~~~~~~~

+-------------------------------+---------------------------------------------------+
| Class                         | Description                                       |
+===============================+===================================================+
| :class:`MSE`                  | Mean squared error (minimise)                     |
+-------------------------------+---------------------------------------------------+
| :class:`RMSE`                 | Root mean squared error (minimise)                |
+-------------------------------+---------------------------------------------------+
| :class:`MAE`                  | Mean absolute error (minimise)                    |
+-------------------------------+---------------------------------------------------+
| :class:`PeakWeightedMSE`      | Flow-weighted MSE — penalises peak-flow errors    |
+-------------------------------+---------------------------------------------------+
| :class:`NashSutcliffe`        | Nash-Sutcliffe efficiency (maximise)              |
+-------------------------------+---------------------------------------------------+
| :class:`KGE`                  | Kling-Gupta efficiency (maximise)                 |
+-------------------------------+---------------------------------------------------+
| :class:`PBIAS`                | Absolute percent bias (minimise)                  |
+-------------------------------+---------------------------------------------------+
| :class:`Con15`                | Top-peak regression slope error (minimise)        |
+-------------------------------+---------------------------------------------------+

.. automodule:: sparsehydro.metrics.objectives
   :members:
   :show-inheritance:

Per-Event Scores
~~~~~~~~~~~~~~~~

.. automodule:: sparsehydro.metrics.events
   :members:
   :show-inheritance:


----

Stormflow Analysis
------------------

Savitzky-Golay filtering and derivative-based event detection used by the
sequential UH fitting workflow.

Filters
~~~~~~~

``apply_savgol_filter`` produces three aligned signals from a raw stormflow
series.  ``compute_thresholds`` attaches percentile-based detection thresholds
in a single immutable update step.

.. code-block:: python

   from sparsehydro.filters import apply_savgol_filter, compute_thresholds

   fr = apply_savgol_filter(rain_stormflow_df, window_length=48)
   fr = compute_thresholds(fr, sg_0_per=0.15, sg_1_per=0.25, sg_2_per=0.25)
   print(fr.thresholds)   # {'sg_0_th': ..., 'sg_1_th': ..., 'sg_2_th': ...}

.. automodule:: sparsehydro.filters.savgol
   :members:
   :show-inheritance:

Dry-weather patterns:

.. automodule:: sparsehydro.filters.dryweather
   :members:
   :show-inheritance:

Event Detection
~~~~~~~~~~~~~~~

``detect_events`` uses ``scipy.signal.find_peaks`` on the smoothed signal to
identify storm peaks, then walks backward and forward using derivative signals
to bound each event.  Overlapping events are merged or split at the trough.

.. note::

   ``height_perc``, ``prom_perc``, and ``width_perc`` are **percentile values
   in the 0–100 range** (not fractions).

.. code-block:: python

   from sparsehydro.events import detect_events, events_to_dataframe

   events, filter_result = detect_events(
       rain_stormflow_df,
       height_perc=25.0,
       prom_perc=25.0,
       width_perc=10.0,
   )
   print(events_to_dataframe(events))

.. automodule:: sparsehydro.events.detection
   :members:
   :show-inheritance:

Event Records
~~~~~~~~~~~~~

.. automodule:: sparsehydro.events.records
   :members:
   :show-inheritance:

Importing Events
~~~~~~~~~~~~~~~~

.. automodule:: sparsehydro.events.io
   :members:
   :show-inheritance:

Screening
~~~~~~~~~

Rules that drop, merge or split candidate events (rainfall depth, duration, flow response, inter-event time).

.. automodule:: sparsehydro.events.screening
   :members:
   :show-inheritance:

Hierarchical Detection
~~~~~~~~~~~~~~~~~~~~~~

Variable-window smoothing, peak definition, bimodality tests and event zones combine into :func:`~sparsehydro.events.detect_event_hierarchy`.

.. automodule:: sparsehydro.events.hierarchy
   :members:
   :show-inheritance:

.. automodule:: sparsehydro.events.smoothing
   :members:
   :show-inheritance:

.. automodule:: sparsehydro.events.peaks
   :members:
   :show-inheritance:

.. automodule:: sparsehydro.events.bimodality
   :members:
   :show-inheritance:

.. automodule:: sparsehydro.events.zones
   :members:
   :show-inheritance:

----

Data
----

File IO for meter data and TSF files, assembly of a modelling table from
downloaded inputs, and clients for external data services.

Files
~~~~~

.. automodule:: sparsehydro.data.utilities
   :members:
   :show-inheritance:

Assembly
~~~~~~~~

.. automodule:: sparsehydro.data.assembly
   :members:
   :show-inheritance:

Data Services
~~~~~~~~~~~~~

Credentials are passed to the client or read from environment variables; never write them into notebooks or source files.

.. automodule:: sparsehydro.data.services.flowfinity
   :members:
   :show-inheritance:

.. automodule:: sparsehydro.data.services.ayyeka
   :members:
   :show-inheritance:

Sample Data
~~~~~~~~~~~

A small bundled dataset for examples and tests:

.. code-block:: python

   from sparsehydro.sample_data import load_sample

   df = load_sample("tc-nb-001")   # datetime, rain [in], stormflow [cfs]

.. automodule:: sparsehydro.sample_data.samples
   :members:
   :show-inheritance:


----

Models
------

.. automodule:: sparsehydro.models

Composition
~~~~~~~~~~~

Formulations are combined declaratively: :class:`~sparsehydro.models.composite.CompositeModel`
wires registered children together by name (bindings, outputs, shared
parameters), so a new combination needs configuration, not a new class.

.. automodule:: sparsehydro.models.composite
   :members:
   :show-inheritance:

.. automodule:: sparsehydro.models.ensemble
   :members:
   :show-inheritance:

.. automodule:: sparsehydro.models.convolution
   :members:
   :show-inheritance:

Seasonality
~~~~~~~~~~~

.. automodule:: sparsehydro.models.seasonality
   :members:
   :show-inheritance:

Abstraction Stores
~~~~~~~~~~~~~~~~~~

.. automodule:: sparsehydro.models.abstraction.tank
   :members:
   :show-inheritance:

.. automodule:: sparsehydro.models.abstraction.soil_groundwater
   :members:
   :show-inheritance:

.. automodule:: sparsehydro.models.abstraction.external
   :members:
   :show-inheritance:

Forcing
~~~~~~~

.. automodule:: sparsehydro.models.forcing.snow
   :members:
   :show-inheritance:

Baseflow
~~~~~~~~

.. automodule:: sparsehydro.models.baseflow.components
   :members:
   :show-inheritance:

.. automodule:: sparsehydro.models.baseflow.fitting
   :members:
   :show-inheritance:

Presets
~~~~~~~

Ready-made whole-flow compositions of the components above. They are ordinary composites: every parameter remains configurable.

.. automodule:: sparsehydro.models.presets.wholeflow
   :members:
   :show-inheritance:


----

Unit Hydrograph
---------------

.. automodule:: sparsehydro.models.unithydrograph
   :members:
   :show-inheritance:
   :noindex:

Formulations
~~~~~~~~~~~~

All unit hydrographs share :class:`~sparsehydro.models.unithydrograph.UnitHydrographBase`:
time parameters in hours, kernels that are exact bin averages of the
continuous response (``sum(kernel) * dt_hours == 1``), and
``Q = A * convolve(rain, kernel)``. See :doc:`unit_hydrographs` for the
equations, the table of formulations and how to add one.

.. automodule:: sparsehydro.models.unithydrograph.base
   :members:
   :show-inheritance:

.. automodule:: sparsehydro.models.unithydrograph.models
   :members:
   :show-inheritance:

.. automodule:: sparsehydro.models.unithydrograph.synthetic
   :members:
   :show-inheritance:

.. automodule:: sparsehydro.models.unithydrograph.clark
   :members:
   :show-inheritance:

.. automodule:: sparsehydro.models.unithydrograph.distributions
   :members:
   :show-inheritance:

.. automodule:: sparsehydro.models.unithydrograph.mixture
   :members:
   :show-inheritance:

.. automodule:: sparsehydro.models.unithydrograph.peak_tail
   :members:
   :show-inheritance:

.. automodule:: sparsehydro.models.unithydrograph.kernels
   :members:

.. automodule:: sparsehydro.models.unithydrograph.rtk
   :members:
   :show-inheritance:


----

Antecedent Moisture Model (AMM)
-------------------------------

:class:`~sparsehydro.models.amm.AMMModel` implements the reparameterized
Antecedent Moisture Model of Edgren, Czachorski & Gonwa
(2024, *Journal of Water Management Modeling* 32: `C525
<https://doi.org/10.14796/JWMM.C525>`_).  A single configurable model
covers two component types selected via ``component_type``:

* ``"standard"`` — the full three-level wet-weather component (Eqs. 1–11).
  An antecedent-moisture recursion accumulates a reduced-wetness capture
  fraction ``RW`` whose temperature sensitivity is set by a seasonal
  hydrologic condition factor (``SHCF``).  Suitable for rainfall runoff or
  rain-derived infiltration/inflow (RDII).
* ``"baseflow"`` — the two-level baseflow component (Eqs. 1–3, 12–16).
  Level 2 is omitted and the capture fraction is driven directly from
  temperature.  Suitable for base flow / groundwater infiltration.

Both unit systems are supported via ``units``: ``"imperial"`` (``rainfall_in``,
acres, CFS) or ``"metric"`` (``rainfall_mm``).  The flow recursion (Eq. 1) is
the exact closed-form solution of the underlying linear-reservoir ODE, so the
result is invariant to the reporting time step.

The model self-registers under the name ``"amm"`` and may be created through
the registry:

.. code-block:: python

   from sparsehydro.registry import registry

   model = registry.create("amm", component_type="standard", units="imperial")
   model.initialize()
   model.validate()
   model.prepare(df)        # columns: datetime, rainfall_in, temperature
   result = model.predict() # datetime, amm_cfs, capture_fraction, rw, shcf, map, matemp

The registered parameters depend on ``component_type``:

+----------------+----------------------------------------------+------------------------------------+
| Parameter      | Meaning                                      | Component                          |
+================+==============================================+====================================+
| ``area_acres`` | Catchment area [acres]                       | both                               |
+----------------+----------------------------------------------+------------------------------------+
| ``PAT``        | Precipitation averaging time [hr]            | both                               |
+----------------+----------------------------------------------+------------------------------------+
| ``HHL``        | Hydrograph half-life [hr]                    | both                               |
+----------------+----------------------------------------------+------------------------------------+
| ``TAT``        | Temperature averaging time [hr]              | both                               |
+----------------+----------------------------------------------+------------------------------------+
| ``Cold_Temp``  | Cold-season temperature (Point 1)            | both                               |
+----------------+----------------------------------------------+------------------------------------+
| ``Hot_Temp``   | Hot-season temperature (Point 2)             | both                               |
+----------------+----------------------------------------------+------------------------------------+
| ``RD``         | Dry-weather capture fraction                 | standard                           |
+----------------+----------------------------------------------+------------------------------------+
| ``AMHL``       | Antecedent-moisture half-life [hr]           | standard                           |
+----------------+----------------------------------------------+------------------------------------+
| ``Cold_SHCF``  | Capture fraction at ``Cold_Temp``            | standard                           |
+----------------+----------------------------------------------+------------------------------------+
| ``Hot_SHCF``   | Capture fraction at ``Hot_Temp``             | standard                           |
+----------------+----------------------------------------------+------------------------------------+
| ``Cold_R``     | Capture fraction at ``Cold_Temp``            | baseflow                           |
+----------------+----------------------------------------------+------------------------------------+
| ``Hot_R``      | Capture fraction at ``Hot_Temp``             | baseflow                           |
+----------------+----------------------------------------------+------------------------------------+

The constraint ``Cold_Temp < Hot_Temp`` is registered as an inequality and is
also enforced by :meth:`~sparsehydro.models.amm.AMMModel.validate`.

.. note::

   Multiple AMM components (e.g. fast inflow + slow infiltration + baseflow,
   Figure 5 of the paper) are combined by summing several :class:`AMMModel`
   instances with :class:`~sparsehydro.models.EnsembleModel`.

.. automodule:: sparsehydro.models.amm
   :members:
   :show-inheritance:

----

RDII
----

The :mod:`sparsehydro.models.rdii` subpackage implements the full physics-based
Rainfall-Derived Inflow and Infiltration (RDII) modelling chain:

* Temperature-driven Initial Abstraction recovery/depletion (:class:`~sparsehydro.models.rdii.initial_abstraction.IAModel`)
  — the wet step integrates the depletion ODE exactly within each timestep
  (invariant to sub-step refinement; mass-conserving)
* Optional degree-day snow model (``IAModel(snow=True)``) — cold-day precipitation
  is stored as snow-water equivalent and released as melt during warm spells,
  capturing cold-season melt-driven peak events
* N triangular RTK unit hydrographs (:class:`~sparsehydro.models.rdii.rtk_triangle.RTKTriangle`)
* A configurable composite model mixing the IA model with any number of RTK
  triangles (:class:`~sparsehydro.models.rdii.model.RDIIModel`)
* ``area_acres`` parameter converts depth [mm] → flow [CFS] automatically
* Calibration via the generic :class:`~sparsehydro.calibration.problem.CalibrationProblem`
  — use ``column_map={"observed": "flow_cfs", "predicted": "rdii_cfs"}``

Requires the optional ``rdii`` extra:

.. code-block:: bash

   pip install sparsehydro[rdii]

.. automodule:: sparsehydro.models.rdii
   :members:
   :show-inheritance:
   :noindex:

RDII Model
~~~~~~~~~~

:class:`~sparsehydro.models.rdii.model.RDIIModel` couples an
:class:`~sparsehydro.models.rdii.initial_abstraction.IAModel` with a configurable
number of :class:`~sparsehydro.models.rdii.rtk_triangle.RTKTriangle` unit
hydrographs:

.. code-block:: python

   from sparsehydro.models.rdii import RDIIModel

   model = RDIIModel(n_triangles=3)
   model.initialize()
   model.validate()

.. automodule:: sparsehydro.models.rdii.model
   :members:
   :show-inheritance:

Initial Abstraction
~~~~~~~~~~~~~~~~~~~

.. automodule:: sparsehydro.models.rdii.initial_abstraction
   :members:
   :show-inheritance:

RTK ensemble
~~~~~~~~~~~~

.. automodule:: sparsehydro.models.rdii.rtk_triangle
   :members:
   :show-inheritance:

----

Calibration
-----------

The :mod:`sparsehydro.calibration` subpackage provides solver-agnostic
abstractions for parameter estimation:

* :class:`~sparsehydro.calibration.problem.CalibrationProblem` — wraps any
  :class:`~sparsehydro.models.IModel` with observed data and objectives.
* :class:`~sparsehydro.calibration.result.CalibrationResult` — stores the
  Pareto front, per-generation history, and metadata.
* Objectives from :mod:`sparsehydro.metrics` (re-exported here for convenience).
* Multiple solver backends (see :ref:`Solvers` below).

.. automodule:: sparsehydro.calibration
   :members:
   :show-inheritance:
   :noindex:

Calibration Problem
~~~~~~~~~~~~~~~~~~~

.. automodule:: sparsehydro.calibration.problem
   :members:
   :show-inheritance:

Calibration Result
~~~~~~~~~~~~~~~~~~

:class:`~sparsehydro.calibration.result.CalibrationResult` stores objective
values in **minimisation form** internally: maximised objectives (e.g. NSE,
KGE) are negated on the way in and restored via
:meth:`~sparsehydro.calibration.result.CalibrationResult.objective_display_values`
and :meth:`~sparsehydro.calibration.result.CalibrationResult.to_pareto_dataframe`.

.. automodule:: sparsehydro.calibration.result
   :members:
   :show-inheritance:

Callbacks
~~~~~~~~~

.. automodule:: sparsehydro.calibration.callbacks
   :members:
   :show-inheritance:

Train/Validation Splits
~~~~~~~~~~~~~~~~~~~~~~~

Time-block and whole-event splits, plus the scoring masks used to weight objectives toward dry weather, storms or snow melt.

.. automodule:: sparsehydro.calibration.splits.blocks
   :members:
   :show-inheritance:

.. automodule:: sparsehydro.calibration.splits.events
   :members:
   :show-inheritance:

.. automodule:: sparsehydro.calibration.splits.masks
   :members:
   :show-inheritance:

Selecting Solutions
~~~~~~~~~~~~~~~~~~~

.. automodule:: sparsehydro.calibration.selection
   :members:
   :show-inheritance:

Diagnostics
~~~~~~~~~~~

.. automodule:: sparsehydro.calibration.diagnostics
   :members:
   :show-inheritance:

Sequential Fitting
~~~~~~~~~~~~~~~~~~

:class:`~sparsehydro.calibration.sequential.SequentialFitter` fits a fresh
model instance to each storm event in chronological order, warm-starting from
the previous event's calibrated parameters (the first event starts from the
model's default values).  The predicted flow from each
fitted event is subtracted from the observed signal before the next event is
fitted (residual approach).

The ``model_factory`` argument is a zero-argument callable that returns a
fresh :class:`~sparsehydro.models.IModel` — this works for both single
UH models and :class:`~sparsehydro.models.EnsembleModel` composites:

.. code-block:: python

   from sparsehydro.calibration import SequentialFitter
   from sparsehydro.models.unithydrograph import GammaUH
   from sparsehydro.models import EnsembleModel

   # Single model
   fitter = SequentialFitter(lambda: GammaUH(), rain_stormflow_df, events)

   # Two-component ensemble
   def make_ensemble():
       fast = GammaUH(A=8.0, alpha=2.5, theta=0.25)
       slow = GammaUH(A=3.0, alpha=4.0, theta=1.7)
       ens = EnsembleModel(
           components=[(fast, lambda p: p["Q_pred"].to_numpy()),
                       (slow, lambda p: p["Q_pred"].to_numpy())],
           mode="sum", aliases=["fast", "slow"],
           output_name="Q_pred", normalize_weights=False,
       )
       ens.initialize()
       ens.set_parameter("w_1", value=1.0, calibrate=False)
       ens.set_parameter("w_2", value=1.0, calibrate=False)
       ens.validate()
       return ens

   fitter = SequentialFitter(make_ensemble, rain_stormflow_df, events)
   summary = fitter.fit(verbose=True)

   print(summary.metrics_summary())        # RMSE, NSE, KGE per event
   print(summary.parameter_evolution())    # fitted params per event

.. automodule:: sparsehydro.calibration.sequential
   :members:
   :show-inheritance:

----

.. _Solvers:

Solvers
-------

All solvers implement :class:`~sparsehydro.calibration.solvers.base.ISolver`
and return a :class:`~sparsehydro.calibration.result.CalibrationResult`.

.. list-table::
   :header-rows: 1
   :widths: 35 40 25

   * - Class
     - Algorithm
     - Extra dep
   * - :class:`~sparsehydro.calibration.solvers.nsga2.NSGAIISolver`
     - NSGA-II (pymoo)
     - ``pymoo``
   * - :class:`~sparsehydro.calibration.solvers.scipy_solver.ScipySolver`
     - SciPy minimisers
     - ``scipy``
   * - :class:`~sparsehydro.calibration.solvers.platypus_solver.PlatypusSolver`
     - Any Platypus algorithm
     - ``platypus-opt``
   * - :class:`~sparsehydro.calibration.solvers.platypus_solver.ParticleSwarmSolver`
     - SMPSO / OMOPSO (PSO)
     - ``platypus-opt``

.. automodule:: sparsehydro.calibration.solvers
   :members:
   :show-inheritance:
   :noindex:

Solver Base
~~~~~~~~~~~

.. automodule:: sparsehydro.calibration.solvers.base
   :members:
   :show-inheritance:

NSGA-II Solver
~~~~~~~~~~~~~~

Requires ``pymoo``:

.. code-block:: bash

   pip install pymoo

.. automodule:: sparsehydro.calibration.solvers.nsga2
   :members:
   :show-inheritance:

SciPy Solver
~~~~~~~~~~~~

Requires ``scipy``:

.. code-block:: bash

   pip install scipy

.. automodule:: sparsehydro.calibration.solvers.scipy_solver
   :members:
   :show-inheritance:

Platypus Solver
~~~~~~~~~~~~~~~

Wraps **any** :class:`platypus.Algorithm` subclass — NSGA-II, NSGA-III, SPEA2,
MOEA/D, GDE3, IBEA, ε-MOEA, and more — behind the uniform
:class:`~sparsehydro.calibration.solvers.base.ISolver` interface.

Requires ``platypus-opt``:

.. code-block:: bash

   pip install platypus-opt

.. automodule:: sparsehydro.calibration.solvers.platypus_solver
   :members:
   :show-inheritance:

Particle Swarm Solver
~~~~~~~~~~~~~~~~~~~~~

:class:`~sparsehydro.calibration.solvers.platypus_solver.ParticleSwarmSolver`
is defined in the same module as :class:`~sparsehydro.calibration.solvers.platypus_solver.PlatypusSolver`.
It wraps :class:`platypus.SMPSO` (default) or :class:`platypus.OMOPSO`
(when ``epsilons`` are provided) behind the standard
:class:`~sparsehydro.calibration.solvers.base.ISolver` interface.

Requires ``platypus-opt``:

.. code-block:: bash

   pip install platypus-opt

See :ref:`Platypus Solver` above — ``ParticleSwarmSolver`` is exported from
:mod:`sparsehydro.calibration.solvers.platypus_solver` and from the
top-level :mod:`sparsehydro.calibration` namespace.

CMA-ES Solver
~~~~~~~~~~~~~

.. automodule:: sparsehydro.calibration.solvers.cmaes_solver
   :members:
   :show-inheritance:

Solver Registry
~~~~~~~~~~~~~~~

.. automodule:: sparsehydro.calibration.solvers.registry
   :members:
   :show-inheritance:


----

Configuration
-------------

Build any registered model, objective or solver from a plain dictionary (for
example, loaded from JSON). ``{"type": name, ...}`` nodes are created through
the registries; nested specs build recursively.

.. code-block:: python

   from sparsehydro.configuration import build_model, build_objectives, build_solver

   model = build_model({"type": "gamma-uh", "A": 5.0})
   objectives = build_objectives(["nash_sutcliffe", {"type": "pbias"}])
   solver = build_solver({"type": "nsga2", "pop_size": 40})

.. automodule:: sparsehydro.configuration
   :members:
   :show-inheritance:

----

Visualization
-------------

The :mod:`sparsehydro.visualization` package provides interactive, browser-ready
Plotly charts for time-series diagnostics, calibration result exploration, and
RDII component analysis.  All functions require the optional ``plotly`` dependency:

.. code-block:: bash

   pip install plotly

All public names are importable directly from :mod:`sparsehydro.visualization`.

.. automodule:: sparsehydro.visualization
   :members:
   :show-inheritance:
   :noindex:

Time-Series Plots
~~~~~~~~~~~~~~~~~

.. automodule:: sparsehydro.visualization.timeseries
   :members:
   :show-inheritance:

Calibration Result Plots
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sparsehydro.visualization.calibration
   :members:
   :show-inheritance:

RDII-Specific Plots
~~~~~~~~~~~~~~~~~~~

.. automodule:: sparsehydro.visualization.rdii
   :members:
   :show-inheritance:

Seasonality Plots
~~~~~~~~~~~~~~~~~

.. automodule:: sparsehydro.visualization.seasonality
   :members:
   :show-inheritance:

Calibration Dashboard
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sparsehydro.visualization.dashboard
   :members:
   :show-inheritance:

Unit Hydrograph Plots
~~~~~~~~~~~~~~~~~~~~~

Six interactive Plotly figures for the sequential UH fitting workflow.
All functions return :class:`plotly.graph_objects.Figure` and are importable
directly from :mod:`sparsehydro.visualization`.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Function
     - Description
   * - :func:`plot_rainfall_flow_with_events`
     - Two-panel timeseries with event shading bands
   * - :func:`plot_filter_signals`
     - sg_0 / sg_1 / sg_2 with threshold lines
   * - :func:`plot_event_detection`
     - sg_0 + peak markers + event shading
   * - :func:`plot_sequential_fit`
     - Rainfall / obs+pred / residual three-row panel
   * - :func:`plot_parameter_evolution`
     - Fitted params over time, colored by NSE
   * - :func:`plot_effective_area`
     - Bar chart of effective area per event

.. automodule:: sparsehydro.visualization.unithydrograph
   :members:
   :show-inheritance:

Whole-Flow Diagnostics
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sparsehydro.visualization.wholeflow
   :members:
   :show-inheritance:

Bias and Event Splits
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sparsehydro.visualization.bias_multi
   :members:
   :show-inheritance:

.. automodule:: sparsehydro.visualization.eventsplit_plot
   :members:
   :show-inheritance:

Display Helpers
~~~~~~~~~~~~~~~

.. automodule:: sparsehydro.visualization.display
   :members:
   :show-inheritance:

Visualization Model
~~~~~~~~~~~~~~~~~~~

.. automodule:: sparsehydro.visualization.model
   :members:
   :show-inheritance:


----

Viewer
------

Interactive Streamlit app for stepping through
:func:`~sparsehydro.events.detect_event_hierarchy` +
:class:`~sparsehydro.calibration.sequential.GlobalSequentialFitter` results by
global event (Overview, Smoothing, UH Shape, Convolution).  Install the
``viewer`` extra and launch it from a shell:

.. code-block:: bash

   pip install "sparsehydro[viewer]"
   sparsehydro-viewer path/to/rain_stormflow.csv

.. automodule:: sparsehydro.viewer.data
   :members:

.. automodule:: sparsehydro.viewer.cli
   :members:

----

Studies
-------

Study-specific workflows built on the general components. Nothing outside
``sparsehydro.studies`` imports from here, so a study can be read as a worked
example of composing the library.

.. automodule:: sparsehydro.studies.msdgc

.. automodule:: sparsehydro.studies.msdgc.settings
   :members:
   :show-inheritance:

.. automodule:: sparsehydro.studies.msdgc.defaults
   :members:
   :show-inheritance:

.. automodule:: sparsehydro.studies.msdgc.data
   :members:
   :show-inheritance:

.. automodule:: sparsehydro.studies.msdgc.detector
   :members:
   :show-inheritance:

.. automodule:: sparsehydro.studies.msdgc.events
   :members:
   :show-inheritance:

.. automodule:: sparsehydro.studies.msdgc.assessment
   :members:
   :show-inheritance:

.. automodule:: sparsehydro.studies.msdgc.workflow
   :members:
   :show-inheritance:

.. automodule:: sparsehydro.studies.msdgc.pareto
   :members:
   :show-inheritance:

.. automodule:: sparsehydro.studies.msdgc.reporting
   :members:
   :show-inheritance:

.. automodule:: sparsehydro.studies.msdgc.plots
   :members:
   :show-inheritance:

.. automodule:: sparsehydro.studies.msdgc.notebook
   :members:
   :show-inheritance:
