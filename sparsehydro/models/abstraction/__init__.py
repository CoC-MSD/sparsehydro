"""sparsehydro.models.abstraction — rainfall-abstraction (tank) models.

Ported from the Parsimonious Functions ``tank_models.py``.  Each model tracks a
storage "tank" whose fill state modulates a state-dependent effective-area
multiplier applied to rainfall, producing an *effective rainfall* series that
can be convolved with a unit hydrograph.

Models
------
- :class:`ConstantDrainTank` — ``V_C``: constant drainage per step.
- :class:`LinearDrainTank` — ``V_lin``: drainage proportional to storage (``k·V``).
- :class:`SqrtDrainTank` — ``V_sroot``: drainage proportional to ``k·sqrt(V)``.
- :class:`ExternalExcess` — no parameters; excess read from data or supplied
  by another model inside a composite.

All three expose the :class:`~sparsehydro.models.IModel` lifecycle and emit a
``p_excess_in`` / ``p_excess_mm`` column, matching
:class:`~sparsehydro.models.rdii.IAModel` so they are interchangeable as the
abstraction stage of :class:`~sparsehydro.models.AbstractionUHModel`.
"""

from .external import ExternalExcess
from .tank import (
    ConstantDrainTank,
    LinearDrainTank,
    SqrtDrainTank,
    TankAbstractionModel,
)

__all__ = [
    "ExternalExcess",
    "TankAbstractionModel",
    "ConstantDrainTank",
    "LinearDrainTank",
    "SqrtDrainTank",
]
