"""UI-independent settings and factories for the centered-pattern workflow.

The notebook is one client of this API. A desktop application can persist the same
settings as JSON and discover parameter units/bounds through ``parameter_table``.
Existing IModel, IObjective and ISolver interfaces remain the extension points.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from typing import ClassVar

import numpy as np

from sparsehydro.calibration.objectives import (
    ConcordanceCorrelationCoefficient, IObjective, NashSutcliffe, PBIAS, PeakWeightedMSE,
)
from sparsehydro.calibration.solvers.base import ISolver
from sparsehydro.models.rdii.initial_abstraction import SnowConfig
from sparsehydro.models.wholeflow_pattern import (
    CenteredWeeklyPattern, ObservedBackground, PatternWholeFlowModel, validate_background_options,
)
from sparsehydro.parameters import ScalarParameter


GROUNDWATER_PARAMETERS = ("k_div", "k_G", "g_max", "g_initial_fraction")


def _positive_integer(name: str, value: int, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


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


@dataclass(frozen=True)
class ModelSettings:
    """Model modules page; new studies default to the IA/RTK core without GWI."""

    groundwater_enabled: bool = False
    n_triangles: int = 3
    snow_enabled: bool = False
    calibrate_snow: bool = False
    snow_threshold_c: float = 0.0
    snow_melt_factor: float = 0.03
    initial_swe_in: float = 0.0
    seasonal_baseline: str = "auto"
    background_window_days: int = 21
    background_minimum_days: int = 3
    background_max_gap_days: int = 45
    background_endpoint_hold_days: int = 0

    def __post_init__(self) -> None:
        for name in ("groundwater_enabled", "snow_enabled", "calibrate_snow"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        _positive_integer("n_triangles", self.n_triangles)
        if self.calibrate_snow and not self.snow_enabled:
            raise ValueError("Snow calibration requires the snow module")
        if self.seasonal_baseline not in ("auto", "off", "observed"):
            raise ValueError(
                "seasonal_baseline must be 'auto', 'off' or 'observed'; the calendar 'annual' mode was removed"
            )
        if self.groundwater_enabled and self.seasonal_baseline == "observed":
            raise ValueError("Observed background and physical GWI are mutually exclusive")
        validate_background_options(
            self.background_window_days, self.background_minimum_days,
            self.background_max_gap_days, self.background_endpoint_hold_days)
        # Validate saved settings even when their module is currently disabled.
        SnowConfig(self.snow_threshold_c, self.snow_melt_factor, self.initial_swe_in)

    @property
    def snow_config(self) -> SnowConfig | None:
        if not self.snow_enabled:
            return None
        return SnowConfig(self.snow_threshold_c, self.snow_melt_factor, self.initial_swe_in)

    @property
    def uses_observed_background(self) -> bool:
        return self.seasonal_baseline == "observed" or (
            self.seasonal_baseline == "auto" and not self.groundwater_enabled)


@dataclass(frozen=True)
class CalibrationSettings:
    """Objectives and solver pages; explicit methods are independent of scoring mode."""

    scoring_mode: str = "multiobjective"
    solver: str = "auto"
    objectives: tuple[str, ...] = ("ccc", "nash_sutcliffe", "peak_weighted_mse")
    population: int = 100
    generations: int = 100
    seed: int = 42
    workers: int = 1
    cma_trials: int = 2000
    cma_starts: int = 4
    cma_jobs: int = 1
    cma_seed: int = 42
    cma_offspring_size: int = 10
    cma_progress_every: int = 200
    de_maxiter: int = 1000
    de_popsize: int = 15
    de_polish: bool = False

    OBJECTIVES: ClassVar[tuple[str, ...]] = (
        "ccc", "nash_sutcliffe", "con15_error", "peak_weighted_mse", "pbias",
    )
    SOLVERS: ClassVar[tuple[str, ...]] = (
        "auto", "nsga2", "spea2", "cmaes", "differential_evolution",
    )

    def __post_init__(self) -> None:
        if self.scoring_mode not in ("single", "multiobjective"):
            raise ValueError("scoring_mode must be 'single' or 'multiobjective'")
        if self.solver not in self.SOLVERS:
            raise ValueError(f"solver must be one of {self.SOLVERS}")
        if (not isinstance(self.objectives, (list, tuple)) or not self.objectives
                or any(name not in self.OBJECTIVES for name in self.objectives)
                or len(set(self.objectives)) != len(self.objectives)):
            raise ValueError(f"Choose nonempty unique objectives from {self.OBJECTIVES}")
        object.__setattr__(self, "objectives", tuple(self.objectives))
        if self.scoring_mode == "single" and len(self.objectives) != 1:
            raise ValueError("Single scoring mode requires exactly one objective")
        if self.resolved_solver in ("cmaes", "differential_evolution") and self.scoring_mode != "single":
            raise ValueError(f"{self.resolved_solver} requires single scoring mode")
        for name in ("seed", "cma_seed", "cma_progress_every"):
            _positive_integer(name, getattr(self, name), minimum=0)
        if not isinstance(self.de_polish, bool):
            raise ValueError("de_polish must be a boolean")
        # Inactive solver budgets are retained, but do not constrain the active method.
        if self.resolved_solver in ("nsga2", "spea2"):
            _positive_integer("population", self.population, minimum=2)
            _positive_integer("generations", self.generations)
            _positive_integer("workers", self.workers)
        elif self.resolved_solver == "cmaes":
            for name in ("cma_starts", "cma_jobs"):
                _positive_integer(name, getattr(self, name))
            for name in ("cma_trials", "cma_offspring_size"):
                _positive_integer(name, getattr(self, name), minimum=2)
        else:
            _positive_integer("de_maxiter", self.de_maxiter)
            _positive_integer("de_popsize", self.de_popsize)

    @property
    def resolved_solver(self) -> str:
        if self.solver == "auto":
            return "cmaes" if self.scoring_mode == "single" else "nsga2"
        return self.solver

    def build_objectives(
        self, train_mask: np.ndarray, storm_mask: np.ndarray, *,
        con15: IObjective | None = None,
    ) -> list[IObjective]:
        train, storm = np.asarray(train_mask), np.asarray(storm_mask)
        if (train.ndim != 1 or train.dtype != bool or storm.dtype != bool
                or train.shape != storm.shape or not train.any() or np.any(storm & ~train)):
            raise ValueError("Objective masks must be aligned booleans; storms must be training-only")
        if any(n in self.objectives for n in ("ccc", "peak_weighted_mse")) and not storm.any():
            raise ValueError("Storm objectives require training storm observations")
        if "con15_error" in self.objectives and (con15 is None or con15.name != "con15_error"):
            raise ValueError("con15_error requires an explicit training-only MSDGC objective")
        if "con15_error" in self.objectives and con15 is not None:
            mask = np.asarray(con15.mask)
            if mask.dtype != bool or mask.shape != train.shape or np.any(mask & ~train):
                raise ValueError("Con15 mask must be aligned and training-only")
        factories = {
            "ccc": lambda: ConcordanceCorrelationCoefficient(mask=storm),
            "nash_sutcliffe": lambda: NashSutcliffe(mask=train),
            "peak_weighted_mse": lambda: PeakWeightedMSE(power=2, mask=storm),
            "pbias": lambda: PBIAS(mask=train),
        }
        objectives: list[IObjective] = []
        for name in self.objectives:
            if name == "con15_error":
                if con15 is None:
                    raise ValueError("Missing Con15 objective")
                objectives.append(con15)
            else:
                objectives.append(factories[name]())
        return objectives

    def build_solver(self) -> ISolver:
        method = self.resolved_solver
        if method in ("nsga2", "spea2"):
            import platypus
            from sparsehydro.calibration.callbacks import ProgressCallback
            from sparsehydro.calibration.solvers.platypus_solver import PlatypusSolver
            return PlatypusSolver(
                {"nsga2": platypus.NSGAII, "spea2": platypus.SPEA2}[method],
                population_size=self.population, n_evaluations=self.population * self.generations,
                seed=self.seed, n_workers=self.workers, callback=ProgressCallback(frequency=10),
            )
        if method == "cmaes":
            from sparsehydro.calibration.solvers.cmaes_solver import PlatypusCmaEsSolver
            return PlatypusCmaEsSolver(
                objective=self.objectives[0], n_trials=self.cma_trials,
                offspring_size=self.cma_offspring_size, seed=self.cma_seed,
                multi_start=self.cma_starts, n_jobs=self.cma_jobs,
                allow_parallel_in_kernel=True, progress_every=self.cma_progress_every,
            )
        from sparsehydro.calibration.solvers.scipy_solver import ScipySolver
        return ScipySolver(
            method="differential_evolution", maxiter=self.de_maxiter, seed=self.seed,
            popsize=self.de_popsize, polish=self.de_polish,
        )

    def solver_metadata(self) -> dict[str, object]:
        method = self.resolved_solver
        if method in ("nsga2", "spea2"):
            return dict(
                solver={"nsga2": "platypus.NSGAII", "spea2": "platypus.SPEA2"}[method],
                population=self.population, generations=self.generations, seed=self.seed,
                calibration_workers=self.workers,
                evaluation_backend="ordered threads" if self.workers > 1 else "serial",
                requested_evaluations=self.population * self.generations,
            )
        if method == "cmaes":
            workers = min(self.cma_jobs, self.cma_starts)
            return dict(
                solver="platypus.CMAES", seed=self.cma_seed,
                trials_per_seed=self.cma_trials, multi_start=self.cma_starts,
                offspring_size=self.cma_offspring_size, calibration_workers=workers,
                evaluation_backend="seed processes" if workers > 1 else "serial",
                requested_evaluations=self.cma_trials * self.cma_starts,
            )
        return dict(
            solver="scipy.differential_evolution", seed=self.seed, calibration_workers=1,
            evaluation_backend="serial", maxiter=self.de_maxiter, popsize=self.de_popsize,
            polish=self.de_polish, requested_evaluations=None,
        )


@dataclass(frozen=True)
class StudySettings:
    """Serializable model/calibration settings plus client-owned workflow pages.

    Workflow pages retain the notebook's data, event-file, split and reporting
    switches without coupling the numerical core to file dialogs or data services.
    No credentials belong in these pages.
    """

    model: ModelSettings = field(default_factory=ModelSettings)
    calibration: CalibrationSettings = field(default_factory=CalibrationSettings)
    parameters: dict[str, ParameterSettings] = field(default_factory=dict)
    workflow: dict[str, dict[str, object]] = field(default_factory=dict)
    schema_version: int = 3

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 3:
            raise ValueError("Unsupported settings schema_version")
        if not isinstance(self.model, ModelSettings) or not isinstance(self.calibration, CalibrationSettings):
            raise ValueError("Study model and calibration must use their settings classes")
        if (not isinstance(self.parameters, dict)
                or any(not isinstance(n, str) or not isinstance(p, ParameterSettings)
                       for n, p in self.parameters.items())):
            raise ValueError("parameters must map names to ParameterSettings")
        if (not isinstance(self.workflow, dict)
                or any(not isinstance(n, str) or not isinstance(p, dict)
                       for n, p in self.workflow.items())):
            raise ValueError("workflow must map page names to JSON objects")
        json.dumps(asdict(self), allow_nan=False)
        object.__setattr__(self, "workflow", json.loads(json.dumps(self.workflow, allow_nan=False)))

    def build_model(
        self, pattern: CenteredWeeklyPattern, area_acres: float, dt_hours: float,
        sanitary_mean_cfs: float, sanitary_upper_cfs: float,
        *, observed_background: ObservedBackground | None = None,
    ) -> PatternWholeFlowModel:
        if self.model.uses_observed_background != (observed_background is not None):
            raise ValueError("Supply a training-derived observed_background exactly when seasonal_baseline enables it")
        if observed_background is not None and (
                observed_background.window_days != self.model.background_window_days
                or observed_background.minimum_days != self.model.background_minimum_days
                or observed_background.max_gap_days != self.model.background_max_gap_days
                or observed_background.endpoint_hold_days != self.model.background_endpoint_hold_days):
            raise ValueError("Observed background estimation options do not match model settings")
        model = PatternWholeFlowModel(
            pattern, area_acres, dt_hours, sanitary_mean_cfs, sanitary_upper_cfs,
            groundwater_enabled=self.model.groundwater_enabled,
            n_triangles=self.model.n_triangles, snow_config=self.model.snow_config,
            calibrate_snow=self.model.calibrate_snow, observed_background=observed_background,
        )
        model.initialize()
        inactive = set(GROUNDWATER_PARAMETERS) if not self.model.groundwater_enabled else set()
        for name, settings in self.parameters.items():
            if name in inactive:
                raise ValueError(f"Parameter {name} belongs to the disabled groundwater module")
            if name not in model.scalar_parameter_names:
                raise ValueError(f"Parameter {name} is not registered by the enabled modules")
            if name in ("snow_T", "snow_ddf") and not self.model.calibrate_snow:
                raise ValueError("Snow parameter overrides require snow calibration")
            settings.apply(model.get_scalar_parameter(name))
        if not model.validate():
            raise ValueError("Initial model settings violate physical constraints")
        return model

    def to_dict(self) -> dict[str, object]:
        # JSON normalization turns tuples into lists for a stable round trip.
        return json.loads(json.dumps(asdict(self), allow_nan=False))

    @classmethod
    def from_dict(cls, values: dict) -> StudySettings:
        data = dict(values)
        model_data = dict(data.get("model", {}))
        version = data.get("schema_version", 1)
        if type(version) is not int or version not in (1, 2, 3):
            raise ValueError("Unsupported settings schema_version")
        if version == 1:
            # Old saved no-GWI studies used a stationary weekly baseline.
            if "seasonal_baseline" in model_data:
                raise ValueError("seasonal_baseline requires settings schema_version 2")
            model_data["seasonal_baseline"] = "off"
        if version == 2:
            mode = model_data.get("seasonal_baseline", "auto")
            if mode == "annual" or (mode == "auto" and not model_data.get("groundwater_enabled", False)):
                raise ValueError(
                    "Schema 2 annual baseline is no longer supported. Explicitly choose schema_version 3 "
                    "with seasonal_baseline='observed' and refit training data, or choose 'off'."
                )
        data["schema_version"] = 3
        data["model"] = ModelSettings(**model_data)
        data["calibration"] = CalibrationSettings(**data.get("calibration", {}))
        data["parameters"] = {
            name: ParameterSettings(**settings) for name, settings in data.get("parameters", {}).items()
        }
        return cls(**data)
