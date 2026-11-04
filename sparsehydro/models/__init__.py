"""sparsehydro.models — model interfaces, generic models, and domain sub-packages.

Sub-packages
------------
- :mod:`sparsehydro.models.rdii` — RDII physics models (IAModel, RDIIModel, RTKTriangle)
- :mod:`sparsehydro.models.unithydrograph` — native unit hydrograph implementations
- :mod:`sparsehydro.models.abstraction` — rainfall-abstraction (tank) models

Interfaces
----------
- :class:`~sparsehydro.models.IModel` — abstract lifecycle interface for all models
- :class:`~sparsehydro.models.IUnitHydroComponent` — extension for UH kernel models
- :class:`~sparsehydro.models.ITorchModel` — optional PyTorch-compatible interface

Generic models
--------------
- :class:`~sparsehydro.models.EnsembleModel` — weighted multi-model compositor
- :class:`~sparsehydro.models.SeasonalityModel` — discrete peaking-factor seasonal flow model
"""

from .base import IModel, IUnitHydroComponent
from .amm import AMMModel
from .ensemble import EnsembleModel
from .seasonality import SeasonalityModel
from .composite import AbstractionUHModel
from .rdii.initial_abstraction import SnowConfig
from .wholeflow_compact import CompactWholeFlowModel, SanitaryTemplate
from .wholeflow_pattern import CenteredWeeklyPattern, PatternWholeFlowModel
from .abstraction import (
    TankAbstractionModel,
    ConstantDrainTank,
    LinearDrainTank,
    SqrtDrainTank,
)

__all__ = [
    "IModel",
    "IUnitHydroComponent",
    "AMMModel",
    "EnsembleModel",
    "SeasonalityModel",
    "AbstractionUHModel",
    "SnowConfig",
    "CompactWholeFlowModel",
    "SanitaryTemplate",
    "CenteredWeeklyPattern",
    "PatternWholeFlowModel",
    "TankAbstractionModel",
    "ConstantDrainTank",
    "LinearDrainTank",
    "SqrtDrainTank",
    "ITorchModel",
]

try:
    from .torch_model import ITorchModel
except ImportError:  # pragma: no cover
    pass
