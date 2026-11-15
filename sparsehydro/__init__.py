"""sparsehydro — interfaces and utilities for parsimonious hydrological models.

Only a small, stable *core* API is re-exported at the top level.  Everything
else is intentionally accessed through its parent sub-package so that the public
surface stays small and import paths mirror the package structure:

- Metrics and objectives — :mod:`sparsehydro.metrics`
- Stormflow filters — :mod:`sparsehydro.filters`
- Data IO and services — :mod:`sparsehydro.data`
- Event detection — :mod:`sparsehydro.events`
- Models — :mod:`sparsehydro.models` (plus :mod:`sparsehydro.models.rdii`,
  :mod:`sparsehydro.models.unithydrograph` and :mod:`sparsehydro.models.presets`)
- Calibration — :mod:`sparsehydro.calibration`
- Building from configuration — :mod:`sparsehydro.configuration`
- Visualization — :mod:`sparsehydro.visualization`
- Study workflows — :mod:`sparsehydro.studies`

Example::

    from sparsehydro import IModel, ModelState, ScalarParameter
    from sparsehydro.models.rdii import RDIIModel
    from sparsehydro.calibration import CalibrationProblem, NashSutcliffe
"""

from .enums import ModelState
from .parameters import FieldRecord, ScalarParameter, VectorParameter
from .registry import ModelRegistry, registry
from .models import IModel, ITorchModel, IUnitHydroComponent

__version__ = "1.0.0a1"

__all__ = [
    "ModelState",
    "FieldRecord",
    "ScalarParameter",
    "VectorParameter",
    "ModelRegistry",
    "registry",
    "IModel",
    "ITorchModel",
    "IUnitHydroComponent",
    "__version__",
]
