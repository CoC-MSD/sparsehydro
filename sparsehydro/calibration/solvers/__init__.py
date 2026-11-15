"""sparsehydro.calibration.solvers — solver abstractions and implementations.

:data:`solver_registry` maps names to solver factories for configuration:

==========================  ==================================================
Name                        Solver
==========================  ==================================================
``nsga2``                   :class:`PlatypusSolver` with ``platypus.NSGAII``
``spea2``                   :class:`PlatypusSolver` with ``platypus.SPEA2``
``platypus``                :class:`PlatypusSolver` (pass ``algorithm_class``)
``pso``                     :class:`ParticleSwarmSolver`
``cmaes``                   :class:`PlatypusCmaEsSolver`
``pymoo_nsga2``             :class:`NSGAIISolver` (pymoo)
``differential_evolution``  :class:`ScipySolver` (differential evolution)
``scipy``                   :class:`ScipySolver` (pass ``method``)
==========================  ==================================================

Keyword arguments are passed to the solver constructor. Register your own
with ``solver_registry.add("name", factory)``.
"""

from ...registry import NamedRegistry
from .base import ISolver
from .nsga2 import NSGAIISolver
from .cmaes_solver import PlatypusCmaEsSolver
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

__all__ = ["ISolver", "NSGAIISolver", "PlatypusCmaEsSolver", "ParticleSwarmSolver",
           "PlatypusSolver", "ScipySolver", "solver_registry"]
