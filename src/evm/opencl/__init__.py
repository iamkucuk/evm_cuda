"""The portable backend: one set of OpenCL kernels for any vendor's hardware.

OpenCL drivers exist for Apple, AMD, Intel and NVIDIA graphics processors, and
for ordinary processors too. The kernels in ``kernels.cl`` are compiled by
whichever driver is present, so supporting a new card needs a driver rather
than new code here.

This is slower than the hand-written CUDA backend, which fuses stages together
and is tuned for one vendor. It is what runs everywhere else.
"""

from __future__ import annotations

from .runtime import available, device_name, unavailable_reason

__all__ = ["available", "unavailable_reason", "device_name", "OpenClOps",
           "ClArray"]


def __getattr__(name: str):
    # Deferred so that importing this package does not require pyopencl.
    if name == "OpenClOps":
        from .ops import OpenClOps
        return OpenClOps
    if name == "ClArray":
        from .array import ClArray
        return ClArray
    raise AttributeError(name)
