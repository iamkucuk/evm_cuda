"""Finding the Metal device and compiling the kernels for it.

Metal is Apple's own interface to their graphics hardware. This backend exists
alongside the OpenCL one because Apple has deprecated OpenCL: it still works
today, and it is not what Apple supports going forward.

Kernels are compiled from source at first use and cached for the process, the
same arrangement the OpenCL backend uses.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

__all__ = [
    "available",
    "unavailable_reason",
    "device",
    "queue",
    "pipeline",
    "device_name",
]

_KERNEL_SOURCE = Path(__file__).with_name("kernels.metal")


def _import_metal() -> Any:
    import Metal

    return Metal


def unavailable_reason() -> str | None:
    """Why this backend cannot run here, or ``None`` if it can.

    Distinguishes the two ways it can be missing, because the fixes differ:
    the Python bindings are installed with this project's ``metal`` extra,
    while Metal itself only exists on Apple hardware and cannot be installed at
    all elsewhere.
    """
    import platform

    if platform.system() != "Darwin":
        return (
            "Metal is an Apple interface and exists only on macOS; this "
            "machine runs " + platform.system()
        )
    try:
        Metal = _import_metal()
    except ImportError:
        return (
            "the Metal bindings are not installed; install this project's "
            "'metal' extra (pip install vidmag[metal])"
        )
    if Metal.MTLCreateSystemDefaultDevice() is None:
        return "Metal reported no default device"
    return None


def available() -> bool:
    return unavailable_reason() is None


@functools.lru_cache(maxsize=1)
def device() -> Any:
    Metal = _import_metal()
    dev = Metal.MTLCreateSystemDefaultDevice()
    if dev is None:
        raise RuntimeError("Metal reported no default device")
    return dev


@functools.lru_cache(maxsize=1)
def queue() -> Any:
    return device().newCommandQueue()


def device_name() -> str:
    return str(device().name())


@functools.lru_cache(maxsize=1)
def _library() -> Any:
    """Compile every kernel once."""
    source = _KERNEL_SOURCE.read_text()
    library, error = device().newLibraryWithSource_options_error_(source, None, None)
    if library is None:
        raise RuntimeError(f"the Metal kernels failed to compile: {error}")
    return library


@functools.lru_cache(maxsize=None)
def pipeline(name: str) -> Any:
    """A ready-to-dispatch handle for one kernel, built once per process."""
    function = _library().newFunctionWithName_(name)
    if function is None:
        raise RuntimeError(f"no Metal kernel named {name!r}")
    state, error = device().newComputePipelineStateWithFunction_error_(function, None)
    if state is None:
        raise RuntimeError(f"could not prepare Metal kernel {name!r}: {error}")
    return state


# ---------------------------------------------------------------------------
# Batching work
# ---------------------------------------------------------------------------
#
# Metal submits work in command buffers. Committing one per kernel and waiting
# for it makes every step a round trip to the hardware, which for a pipeline of
# several dozen small kernels costs more than the kernels do. Instead one
# buffer is kept open, every kernel is encoded into it, and it is submitted only
# when a result is actually read back. Nothing observes an unfinished result,
# because reading is the one thing that forces the wait.

_pending: Any = None


def encoder() -> Any:
    """A compute encoder on the open command buffer, opening one if needed."""
    global _pending
    if _pending is None:
        _pending = queue().commandBuffer()
    return _pending.computeCommandEncoder()


def flush() -> None:
    """Submit whatever has been encoded and wait for it to finish.

    Called before reading any buffer back. Safe to call when nothing is
    pending.
    """
    global _pending
    if _pending is None:
        return
    buffer, _pending = _pending, None
    buffer.commit()
    buffer.waitUntilCompleted()
