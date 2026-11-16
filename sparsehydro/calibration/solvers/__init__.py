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
``pymoo_nsga2``             :class:`NSGAIISolver` (pymoo; deprecated, needs ``[pymoo]``)
``differential_evolution``  :class:`ScipySolver` (differential evolution)
``scipy``                   :class:`ScipySolver` (pass ``method``)
==========================  ==================================================

Keyword arguments are passed to the solver constructor. Register your own
with ``solver_registry.add("name", factory)``.
"""

from .base import ISolver
from .nsga2 import NSGAIISolver
from .cmaes_solver import PlatypusCmaEsSolver
from .platypus_solver import ParticleSwarmSolver, PlatypusSolver
from .scipy_solver import ScipySolver

from .registry import solver_registry

__all__ = ["ISolver", "NSGAIISolver", "PlatypusCmaEsSolver", "ParticleSwarmSolver",
           "PlatypusSolver", "ScipySolver", "solver_registry"]
