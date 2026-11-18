"""Optional numba support shared by the compiled model kernels.

``njit`` is numba's decorator when numba is installed and a no-op otherwise, so
kernels decorated with it run as plain Python without numba. Install the
``speed`` extra (``pip install "sparsehydro[speed]"``) for the compiled path.
Kernels keep ``fastmath`` off so compiled and pure-Python results match bit
for bit.
"""

from __future__ import annotations

try:
    from numba import njit

    HAS_NUMBA = True
except Exception:  # pragma: no cover - depends on the optional dependency
    HAS_NUMBA = False

    def njit(*args, **kwargs):  # noqa: D401 - no-op decorator shim
        """Return the function unchanged (supports ``@njit`` and ``@njit(...)``)."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def _wrap(fn):
            return fn

        return _wrap


__all__ = ["HAS_NUMBA", "njit"]
