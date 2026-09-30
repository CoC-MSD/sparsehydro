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
   from sparsehydro.models import AMMModel, EnsembleModel, SeasonalityModel
   from sparsehydro.models.rdii import RDIIModel, IAModel, RTKTriangle
   from sparsehydro.models.unithydrograph import (
       GammaUH, NashUH, TriangleUH, SequentialFitter, SequentialFitSummary,
   )
   from sparsehydro.filters import apply_savgol_filter, compute_thresholds
   from sparsehydro.events import detect_events, events_to_dataframe
   from sparsehydro.calibration import (
       CalibrationProblem, CalibrationResult,
       MSE, RMSE, NashSutcliffe, KGE, PeakWeightedMSE,
       NSGAIISolver, ScipySolver, PlatypusSolver, ParticleSwarmSolver,
   )
   from sparsehydro.visualization import plot_timeseries, plot_calibration_dashboard

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

----

Visualization
-------------

The :mod:`sparsehydro.visualization` package provides interactive, browser-ready
Plotly charts for time-series diagnostics, calibration result exploration, and
RDII component analysis.  All functions require the optional ``plotly`` dependency:

.. code-block:: bash

   pip install plotly

All public names are importable directly from :mod:`sparsehydro.visualization`
or from the top-level :mod:`sparsehydro` namespace.

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
directly from :mod:`sparsehydro.visualization` or the top-level namespace.

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

   from sparsehydro import apply_savgol_filter, compute_thresholds

   fr = apply_savgol_filter(rain_stormflow_df, window_length=48)
   fr = compute_thresholds(fr, sg_0_per=0.15, sg_1_per=0.25, sg_2_per=0.25)
   print(fr.thresholds)   # {'sg_0_th': ..., 'sg_1_th': ..., 'sg_2_th': ...}

.. automodule:: sparsehydro.filters
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

   from sparsehydro import detect_events, events_to_dataframe

   events, filter_result = detect_events(
       rain_stormflow_df,
       height_perc=25.0,
       prom_perc=25.0,
       width_perc=10.0,
   )
   print(events_to_dataframe(events))

.. automodule:: sparsehydro.events
   :members:
   :show-inheritance:

----

Viewer
------

Interactive Streamlit app for stepping through
:func:`~sparsehydro.events.detect_event_hierarchy` +
:class:`~sparsehydro.models.unithydrograph.GlobalSequentialFitter` results by
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

Unit Hydrograph
---------------

.. automodule:: sparsehydro.models.unithydrograph
   :members:
   :show-inheritance:
   :noindex:

Native UH Models
~~~~~~~~~~~~~~~~

Three self-contained :class:`~sparsehydro.models.IUnitHydroComponent`
implementations that follow the full SparseHydro lifecycle.  All return a
:class:`~pandas.DataFrame` with a ``"Q_pred"`` column so the same
``column_map`` works for single models and
:class:`~sparsehydro.models.EnsembleModel` composites.

**Kernel normalisation:** ``sum(kernel) * dt_hours ≈ 1.0``

**Convolution:** ``Q = convolve(rain, kernel * A * dt)[:n]``
where ``A`` is the effective area ratio (stormflow / rain volume).

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Class
     - Parameters
     - Kernel shape
   * - :class:`GammaUH`
     - A, tt (shape), tp (scale/steps)
     - ``(t/tp)^tt · exp(-t/tp)``, normalised
   * - :class:`NashUH`
     - A, n (reservoirs), k (storage)
     - ``t^(n-1) · exp(-t/k) / (k^n Γ(n))``
   * - :class:`TriangleUH`
     - A, tt (total steps), tp (peak)
     - Piecewise linear; ``tp < tt``

.. automodule:: sparsehydro.models.unithydrograph.models
   :members:
   :show-inheritance:

Sequential Fitting
~~~~~~~~~~~~~~~~~~

:class:`~sparsehydro.models.unithydrograph.sequential.SequentialFitter` fits a fresh
model instance to each storm event in chronological order, warm-starting from
the previous event's calibrated parameters.  The predicted flow from each
fitted event is subtracted from the observed signal before the next event is
fitted (residual approach).

The ``model_factory`` argument is a zero-argument callable that returns a
fresh :class:`~sparsehydro.models.IModel` — this works for both single
UH models and :class:`~sparsehydro.models.EnsembleModel` composites:

.. code-block:: python

   from sparsehydro.models.unithydrograph import GammaUH, SequentialFitter
   from sparsehydro.models import EnsembleModel

   # Single model
   fitter = SequentialFitter(lambda: GammaUH(), rain_stormflow_df, events)

   # Two-component ensemble
   def make_ensemble():
       fast = GammaUH(A=80.0, tt=1.5, tp=3.0)
       slow = GammaUH(A=30.0, tt=3.0, tp=20.0)
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

.. automodule:: sparsehydro.models.unithydrograph.sequential
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

RTK Triangle
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
* A library of :class:`~sparsehydro.calibration.objectives.IObjective`
  implementations covering MSE, NSE, KGE, and more.
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

.. automodule:: sparsehydro.calibration.objectives
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
