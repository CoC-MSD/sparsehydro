"""Solvers by name for configuration (see :mod:`sparsehydro.calibration.solvers`)."""

from __future__ import annotations

from ...registry import NamedRegistry
from .cmaes_solver import PlatypusCmaEsSolver
from .nsga2 import NSGAIISolver
from .platypus_solver import ParticleSwarmSolver, PlatypusSolver
from .scipy_solver import ScipySolver

solver_registry = NamedRegistry("solver")


def _platypus_algorithm(name: str):
    def factory(**kwargs):
        import platypus

        return PlatypusSolver(getattr(platypus, name), **kwargs)
    return factory


solver_registry.add("nsga2", _platypus_algorithm("NSGAII"))
solver_registry.add("spea2", _platypus_algorithm("SPEA2"))
solver_registry.add("platypus", PlatypusSolver)
solver_registry.add("pso", ParticleSwarmSolver)
solver_registry.add("cmaes", PlatypusCmaEsSolver)
solver_registry.add("pymoo_nsga2", NSGAIISolver)
solver_registry.add("differential_evolution",
                    lambda **kwargs: ScipySolver(method="differential_evolution", **kwargs))
solver_registry.add("scipy", ScipySolver)


__all__ = ["solver_registry"]
