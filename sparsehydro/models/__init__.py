"""sparsehydro.models — model interfaces, components and composition.

Sub-packages
------------
- :mod:`sparsehydro.models.unithydrograph` — unit hydrograph library
- :mod:`sparsehydro.models.rdii` — initial abstraction and RDII routing
- :mod:`sparsehydro.models.abstraction` — abstraction stores (tanks, soil/groundwater, external excess)
- :mod:`sparsehydro.models.forcing` — forcing pre-processing (degree-day snow)
- :mod:`sparsehydro.models.baseflow` — sanitary/dry-weather baselines and their pre-fit estimators
- :mod:`sparsehydro.models.presets` — ready-made compositions (whole-flow models)

Interfaces
----------
- :class:`IModel` — lifecycle interface for all models
- :class:`IUnitHydroComponent` — extension for unit hydrograph kernels
- :class:`ITorchModel` — optional PyTorch-compatible interface

Composition
-----------
- :class:`CompositeModel` — named components wired by bindings, with combined outputs
- :class:`EnsembleModel` — weighted sum or product of models on the same data
- Presets: :class:`AbstractionUHModel`, :class:`CompactWholeFlowModel`,
  :class:`PatternWholeFlowModel`

Every concrete model is registered by name; see :data:`sparsehydro.registry`.
"""

from .base import IModel, IUnitHydroComponent
from .amm import AMMModel
from .ensemble import EnsembleModel
from .seasonality import SeasonalityModel
from .composite import AbstractionUHModel, CompositeModel
from .forcing import DegreeDaySnow, SnowConfig
from .abstraction import (
    TankAbstractionModel,
    ConstantDrainTank,
    LinearDrainTank,
    SqrtDrainTank,
    ExternalExcess,
    SoilGroundwaterStore,
)
from .baseflow import (
    CenteredWeeklyPattern,
    CenteredWeeklyPatternModel,
    ObservedBackground,
    ObservedBackgroundModel,
    SanitaryTemplate,
    SanitaryTemplateModel,
)
from .presets import CompactWholeFlowModel, PatternWholeFlowModel

__all__ = [
    "IModel",
    "IUnitHydroComponent",
    "AMMModel",
    "EnsembleModel",
    "SeasonalityModel",
    "CompositeModel",
    "AbstractionUHModel",
    "DegreeDaySnow",
    "SnowConfig",
    "TankAbstractionModel",
    "ConstantDrainTank",
    "LinearDrainTank",
    "SqrtDrainTank",
    "ExternalExcess",
    "SoilGroundwaterStore",
    "CenteredWeeklyPattern",
    "CenteredWeeklyPatternModel",
    "ObservedBackground",
    "ObservedBackgroundModel",
    "SanitaryTemplate",
    "SanitaryTemplateModel",
    "CompactWholeFlowModel",
    "PatternWholeFlowModel",
    "ITorchModel",
]

try:
    from .torch_model import ITorchModel
except ImportError:  # pragma: no cover
    pass
