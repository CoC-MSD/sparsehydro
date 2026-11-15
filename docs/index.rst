.. include:: ../README.md
   :parser: myst_parser.sphinx_

----

Package Architecture
--------------------

The diagram below shows how the core abstractions relate to each other.
Models expose named :class:`~sparsehydro.parameters.ScalarParameter` objects
(each with an optional ``calibrate`` flag) which are discovered automatically
by :class:`~sparsehydro.calibration.problem.CalibrationProblem`.  A single
problem instance is passed unchanged to any solver backend.

.. mermaid::
   :caption: sparsehydro component architecture

   graph TB
       subgraph core["Core — sparsehydro"]
           IM["IModel\n(abstract lifecycle)"]
           SP["ScalarParameter\nvalue · bounds · calibrate"]
           MS["ModelState\nCREATED → FINALIZED"]
       end

       subgraph models["Concrete Models"]
           RDII["RDIIModel\nIA + N × RTK triangles"]
           CHM["CombinedHydroModel\nany IA + any UH mix"]
           AMM["AMMModel\nantecedent moisture\nstandard · baseflow"]
           UHA["UnitHydrographAdapter"]
           CUSTOM["Custom Model\n(subclass IModel)"]
       end

       subgraph calibration["Calibration — sparsehydro.calibration"]
           CP["CalibrationProblem\ndata · column_map · objectives"]
           OBJ["Objectives\nNSE · KGE · PeakWMSE · MSE"]
           IC["InequalityConstraints\ng_j ≤ 0"]

           subgraph solvers["Solvers  (ISolver)"]
               NS["NSGAIISolver\npymoo"]
               PS["ParticleSwarmSolver\nSMPSO / OMOPSO"]
               SC["ScipySolver\ndiff-evo / L-BFGS-B"]
               PL["PlatypusSolver\nany Platypus algo"]
           end

           CR["CalibrationResult\npareto_X · pareto_F · history"]
       end

       subgraph viz["Visualization — sparsehydro.visualization"]
           V1["plot_calibration_timeseries\n2-row dashboard + scatter"]
           V2["plot_pareto_evolution"]
           V3["plot_parallel_coordinates"]
           V4["plot_calibration_dashboard"]
       end

       IM --> RDII & CHM & UHA & CUSTOM & AMM
       IM -- "registers" --> SP
       RDII & UHA & CUSTOM --> CP
       AMM --> CP
       OBJ --> CP
       IC --> CP
       CP --> NS & PS & SC & PL
       NS & PS & SC & PL --> CR
       CR --> V1 & V2 & V3 & V4

       style core fill:#e8f4fd,stroke:#2c5f99
       style models fill:#eafaf1,stroke:#1e8449
       style calibration fill:#fef9e7,stroke:#d4ac0d
       style viz fill:#fdedec,stroke:#c0392b

Object Relationships
--------------------

Every model subclasses the abstract :class:`~sparsehydro.interfaces.IModel`
lifecycle and exposes named :class:`~sparsehydro.parameters.ScalarParameter` and
:class:`~sparsehydro.parameters.VectorParameter` objects through a built-in
registry.  Domain models extend either ``IModel`` directly or the
:class:`~sparsehydro.models.IUnitHydroComponent` kernel interface.

.. mermaid::
   :caption: Model and parameter class relationships

   classDiagram
       direction LR

       class ModelState {
           <<enumeration>>
           CREATED
           INITIALIZED
           VALIDATED
           PREPARED
           PREDICTED
           FINALIZED
       }

       class IModel {
           <<abstract>>
           +str model_name
           +ModelState state
           +initialize()
           +validate() bool
           +prepare(data)
           +predict() DataFrame
           +finalize()
           +register_scalar_parameter(p)
           +get_scalar_parameter(name) ScalarParameter
       }

       class ScalarParameter {
           +str name
           +float value
           +float lower_bound
           +float upper_bound
           +bool calibrate
           +is_valid() bool
           +normalize() float
           +clamp() ScalarParameter
       }

       class VectorParameter {
           +str name
           +ndarray values
           +int size
           +is_valid() bool
       }

       class IUnitHydroComponent {
           <<abstract>>
       }
       class ITorchModel {
           <<abstract>>
           +forward() Tensor
       }

       IModel "1" o-- "*" ScalarParameter : registers
       IModel "1" o-- "*" VectorParameter : registers
       IModel ..> ModelState : has state
       IUnitHydroComponent --|> IModel
       ITorchModel --|> IModel

       RDIIModel --|> IModel
       IAModel --|> IModel
       AMMModel --|> IModel
       EnsembleModel --|> IModel
       SeasonalityModel --|> IModel
       RTKTriangle --|> IUnitHydroComponent
       GammaUH --|> IUnitHydroComponent
       UnitHydrographAdapter --|> IUnitHydroComponent
       EnsembleModel "1" o-- "*" IModel : composes
       RDIIModel "1" *-- "1" IAModel : contains
       RDIIModel "1" *-- "*" RTKTriangle : contains

The calibration layer is solver-agnostic: a single
:class:`~sparsehydro.calibration.problem.CalibrationProblem` wraps a model, its
data, and one or more :class:`~sparsehydro.metrics.objectives.IObjective`
functions, then passes unchanged to any
:class:`~sparsehydro.calibration.solvers.base.ISolver`, which returns a
:class:`~sparsehydro.calibration.result.CalibrationResult`.

.. mermaid::
   :caption: Calibration class relationships

   classDiagram
       direction LR

       class CalibrationProblem {
           +IModel model
           +objectives IObjective[]
           +dict column_map
           +param_names str[]
           +int n_params
           +evaluate(x) ndarray
       }

       class IObjective {
           <<abstract>>
           +str name
           +bool minimize
           +evaluate(observed, predicted) float
       }

       class ISolver {
           <<abstract>>
           +solve(problem) CalibrationResult
       }

       class CalibrationResult {
           +ndarray pareto_X
           +ndarray pareto_F
           +history GenerationRecord[]
           +best_by(name) dict
           +to_pareto_dataframe() DataFrame
       }

       CalibrationProblem o-- IModel : wraps
       CalibrationProblem o-- "1..*" IObjective : scores with
       ISolver ..> CalibrationProblem : consumes
       ISolver ..> CalibrationResult : produces

       NSGAIISolver --|> ISolver
       ParticleSwarmSolver --|> ISolver
       ScipySolver --|> ISolver
       PlatypusSolver --|> ISolver

       MSE --|> IObjective
       NashSutcliffe --|> IObjective
       KGE --|> IObjective
       PeakWeightedMSE --|> IObjective

Calibration Workflow
--------------------

The typical end-to-end calibration loop — build the model, wrap it in a problem,
solve, then inspect and visualize the results.  The solver only ever talks to
the :class:`~sparsehydro.calibration.problem.CalibrationProblem`, which in turn
drives the model and objectives on every evaluation.

.. mermaid::
   :caption: Typical calibration workflow

   sequenceDiagram
       autonumber
       actor User
       participant Model as IModel
       participant Problem as CalibrationProblem
       participant Solver as ISolver
       participant Obj as IObjective
       participant Result as CalibrationResult
       participant Viz as Visualization

       User->>Model: initialize()
       User->>Model: validate()
       User->>Problem: CalibrationProblem(model, data, objectives, column_map)
       Problem->>Model: prepare(data)
       Note over Problem: discovers calibratable<br/>ScalarParameters and bounds

       User->>Solver: solve(problem)
       loop each candidate / generation
           Solver->>Problem: evaluate(x)
           Problem->>Model: set parameters and predict()
           Model-->>Problem: predicted series
           Problem->>Obj: evaluate(observed, predicted)
           Obj-->>Problem: scores
           Problem-->>Solver: objective vector F
       end
       Solver-->>Result: pareto_X, pareto_F, history
       Result-->>User: best_by(), to_pareto_dataframe()
       User->>Viz: plot_pareto_evolution(result)

.. toctree::
   :maxdepth: 1
   :hidden:

   getting_started
   unit_hydrographs
   api
