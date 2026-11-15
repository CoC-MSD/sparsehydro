"""MSDGC whole-flow study: everything specific to that study lives here.

* ``defaults`` — :func:`default_controls`, the study's default settings.
* ``settings`` — ``ModelSettings``, ``CalibrationSettings``, ``StudySettings``,
  ``study_settings_from_controls``.
* ``data`` — FlowFinity record selection and download, air-temperature alignment.
* ``detector`` — port of the MSDGC tool's event detector (``find_events``).
* ``events`` — frozen per-split detection, hybrid EVT inventories, melt windows.
* ``assessment`` — acceptance contract, MSDGC zone and Con15 assessment.
* ``workflow`` — data preparation, model building and calibration for the study.
* ``pareto`` — MSDGC Pareto evaluation and ranking.
* ``reporting`` — assessment, diagnostics and report/TSF export.
* ``plots`` — MSDGC Pareto, diagnostic and presentation figures.

The rest of sparsehydro never imports from this package.
"""

from .defaults import default_controls

__all__ = ["default_controls"]
