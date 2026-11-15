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

import logging
import sys

from .defaults import default_controls

__all__ = ["default_controls", "enable_notebook_logging"]


def enable_notebook_logging(level: int = logging.INFO) -> logging.Handler:
    """Show sparsehydro progress messages as plain lines (e.g. in a notebook).

    The library logs instead of printing; this attaches one stdout handler to
    the ``sparsehydro`` logger (idempotent) and returns it.
    """
    logger = logging.getLogger("sparsehydro")
    for handler in logger.handlers:
        if getattr(handler, "_sparsehydro_notebook", False):
            handler.setLevel(level)
            logger.setLevel(min(logger.level or level, level))
            return handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.setLevel(level)
    handler._sparsehydro_notebook = True
    logger.addHandler(handler)
    logger.setLevel(level)
    return handler
