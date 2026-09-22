"""sparsehydro.calibration.solvers — solver abstractions and implementations."""

from .base import ISolver
from .nsga2 import NSGAIISolver
from .optuna_solver import OptunaCmaEsSolver
from .platypus_solver import ParticleSwarmSolver, PlatypusSolver
from .scipy_solver import ScipySolver

__all__ = ["ISolver", "NSGAIISolver", "OptunaCmaEsSolver", "ParticleSwarmSolver", "PlatypusSolver", "ScipySolver"]
