"""Finding an OpenCL device, and compiling the kernels for it.

OpenCL programs are compiled by the driver at run time from the source in
``kernels.cl``, which is what makes one source file work on hardware from
different vendors. The cost is a compile on first use; it is cached here so it
happens once per process.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

__all__ = ["available", "unavailable_reason", "context", "queue", "program",
           "device_name"]

_KERNEL_SOURCE = Path(__file__).with_name("kernels.cl")


def _import_pyopencl() -> Any:
    import pyopencl
    return pyopencl


def unavailable_reason() -> str | None:
    """Why this backend cannot run here, or ``None`` if it can.

    Says which of the two things is missing rather than reporting a bare
    failure: the Python package is installed with the project's ``opencl``
    extra, while the driver comes from the operating system or the graphics
    vendor, and the fixes are completely different.
    """
    try:
        cl = _import_pyopencl()
    except ImportError:
        return ("pyopencl is not installed; install this project's 'opencl' "
                "extra (pip install evm-cuda[opencl])")
    try:
        platforms = cl.get_platforms()
    except Exception as exc:            # driver missing or refusing to load
        return (f"no OpenCL driver found ({type(exc).__name__}: {exc}); a "
                f"driver is provided by the operating system or the graphics "
                f"vendor, not by this project")
    for platform in platforms:
        if platform.get_devices():
            return None
    return "an OpenCL driver is present but reports no devices"


def available() -> bool:
    return unavailable_reason() is None


def _pick_device(cl: Any) -> Any:
    """Prefer a graphics processor, and say so when falling back to the host.

    A processor-only OpenCL driver is a legitimate target — it is what runs
    these kernels in continuous integration, where there is no graphics card —
    but silently using one when a card was expected would look like the card
    was simply slow.
    """
    fallback = None
    for platform in cl.get_platforms():
        for device in platform.get_devices():
            if device.type & cl.device_type.GPU:
                return device
            fallback = fallback or device
    if fallback is None:
        raise RuntimeError("no OpenCL device found")
    return fallback


@functools.lru_cache(maxsize=1)
def context() -> Any:
    cl = _import_pyopencl()
    return cl.Context(devices=[_pick_device(cl)])


@functools.lru_cache(maxsize=1)
def queue() -> Any:
    cl = _import_pyopencl()
    return cl.CommandQueue(context())


def device_name() -> str:
    return context().devices[0].name.strip()


def _colour_matrix_defines() -> str:
    """Compiler definitions carrying the project's colour matrices.

    The kernels reference the nine forward and nine inverse coefficients by
    name rather than containing them. They are taken from the matrices in
    :mod:`evm.io.video`, which is what the project's agreement with the
    reference implementation rests on; a hand-typed second copy in the kernel
    source would be a second thing to get wrong, and the failure would look
    like a wrong colour rather than a wrong constant.
    """
    import numpy as np

    from ..io.video import rgb_to_yiq, yiq_to_rgb

    forward = rgb_to_yiq(np.eye(3, dtype=np.float32)).T
    inverse = yiq_to_rgb(np.eye(3, dtype=np.float32)).T

    parts = []
    for prefix, matrix in (("FWD", forward), ("INV", inverse)):
        for row in range(3):
            for col in range(3):
                parts.append(
                    f"-D {prefix}{row}{col}={float(matrix[row, col]):.10e}f")
    return " ".join(parts)


@functools.lru_cache(maxsize=None)
def kernel(name: str) -> Any:
    """One reusable handle per kernel.

    Reading ``program().some_kernel`` builds a fresh kernel object every time,
    which the driver charges for. These are looked up in a loop over pyramid
    levels, so the cost is paid repeatedly; caching removes it.
    """
    import pyopencl as cl
    return cl.Kernel(program(), name)


@functools.lru_cache(maxsize=1)
def program() -> Any:
    """Compile the kernels once per process."""
    cl = _import_pyopencl()
    source = _KERNEL_SOURCE.read_text()
    try:
        return cl.Program(context(), source).build(
            options=_colour_matrix_defines())
    except Exception as exc:
        raise RuntimeError(
            f"the OpenCL kernels failed to compile on "
            f"{device_name()!r}: {exc}"
        ) from exc
