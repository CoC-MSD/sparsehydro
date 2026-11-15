"""Unit conversion constants used across sparsehydro.

Depth-over-area to flow conversions assume a depth rate over an area in acres
and give flow in cubic feet per second (cfs).
"""

from __future__ import annotations

#: Cubic feet in one acre-inch (43560 ft^2 x 1/12 ft).
ACRE_INCH_TO_FT3: float = 43560.0 / 12.0

#: inches per hour over one acre -> cfs.
IN_AC_PER_HR_TO_CFS: float = 43560.0 / (12.0 * 3600.0)

#: millimetres per hour over one acre -> cfs.
MM_AC_PER_HR_TO_CFS: float = 43560.0 / (304.8 * 3600.0)

#: million US gallons per day -> cfs.
MGD_TO_CFS: float = 1e6 / (7.480519 * 86400.0)

__all__ = ["ACRE_INCH_TO_FT3", "IN_AC_PER_HR_TO_CFS", "MM_AC_PER_HR_TO_CFS", "MGD_TO_CFS"]
