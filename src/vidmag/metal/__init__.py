"""The Metal backend, for Apple graphics hardware.

Apple has deprecated OpenCL. It still works today, and this exists so that when
it stops, Apple hardware keeps running through the interface Apple actually
supports.
"""

from __future__ import annotations

from .runtime import available, device_name, unavailable_reason

__all__ = ["available", "unavailable_reason", "device_name", "MetalOps", "MetalArray"]


def __getattr__(name: str):
    # Deferred so importing this package needs no Metal bindings.
    if name == "MetalOps":
        from .ops import MetalOps

        return MetalOps
    if name == "MetalArray":
        from .array import MetalArray

        return MetalArray
    raise AttributeError(name)
