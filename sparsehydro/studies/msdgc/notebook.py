"""Helpers for running the study interactively."""

from __future__ import annotations

import logging
import sys

__all__ = ["enable_notebook_logging"]


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
