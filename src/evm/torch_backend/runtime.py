"""Finding out whether PyTorch can run here, and on what.

PyTorch is an optional dependency and is never imported unless this backend is
asked for. Everything in this module answers the question "can it run" without
assuming it can.
"""

from __future__ import annotations

import functools
from typing import Any

__all__ = ["available", "unavailable_reason", "device_name", "pick_device", "torch"]


def _import_torch() -> Any:
    import torch

    return torch


def torch() -> Any:
    """The imported module, for callers that already know it is present."""
    return _import_torch()


def pick_device(preferred: str | None = None) -> str:
    """The device this backend will compute on.

    Order: an explicit request, then NVIDIA, then Apple, then the processor.
    PyTorch on the processor is a real option rather than a fallback — it is
    how this backend is checked on machines with no graphics hardware at all,
    including continuous integration.
    """
    t = _import_torch()
    if preferred:
        return preferred
    if t.cuda.is_available():
        return "cuda"
    if getattr(t.backends, "mps", None) is not None and t.backends.mps.is_available():
        return "mps"
    return "cpu"


def unavailable_reason() -> str | None:
    """Why this backend cannot run here, or ``None`` if it can."""
    try:
        _import_torch()
    except ImportError:
        return (
            "PyTorch is not installed; install this project's 'torch' extra "
            "(pip install evm-magnify[torch])"
        )
    return None


def available() -> bool:
    return unavailable_reason() is None


@functools.lru_cache(maxsize=None)
def device_name(preferred: str | None = None) -> str:
    """A readable name for the device, for the line that reports the choice."""
    t = _import_torch()
    device = pick_device(preferred)
    if device.startswith("cuda"):
        return f"{t.cuda.get_device_name(0)} (PyTorch, CUDA)"
    if device == "mps":
        return "Apple GPU (PyTorch, Metal Performance Shaders)"
    return "processor (PyTorch)"
