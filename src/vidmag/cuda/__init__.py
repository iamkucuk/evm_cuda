"""EVM CUDA port — Python wrapper package.

This package wraps the compiled `_vidmag_cuda` extension (built by
`src/vidmag/cuda/CMakeLists.txt`) and exposes the four magnification pipelines as
drop-in replacements for the Python baseline `vidmag.magnify_*`.

Layout:
- ``_vidmag_cuda``  — the compiled pybind11 module (loaded lazily; raises on
                   machines without CUDA / nvcc-built .so).
- ``runtime``    — small helpers (cuFFT plan cache, version probe).
- ``batched``    — the OPTIMIZED, device-resident pipelines (color gdown+ideal,
                   motion lpyr+iir, both FP32 + FP16). The hot path.
- ``pipelines``  — the per-frame pipelines (motion lpyr+ideal, lpyr+butter);
                   the only implementations of those two rarer variants.
- ``benchmark``  — fair per-stage profiling harness (``run``/``summarize``).

Tests in `tests/cuda/` import from this package; they skip cleanly if the
CUDA module isn't built (see `tests/cuda/conftest.py`).
"""

from __future__ import annotations

import importlib

try:
    # `from . import _vidmag_cuda` would work, but it produces a false error on
    # the machines that matter most here — the ones without the extension. The
    # name is looked up while this package is still executing its own __init__,
    # so Python appends "most likely due to a circular import" to the
    # ImportError. There is no circular import; the .so was never compiled, and
    # that sentence sends readers to debug an imaginary problem.
    # importlib.import_module says what is actually true:
    # "No module named 'vidmag.cuda._vidmag_cuda'". A genuinely broken .so — a
    # missing libcudart, say — still reports the linker's own message, which is
    # the diagnostic worth keeping. Held by tests/test_api.py.
    #
    # It still binds the submodule as an attribute of this package, so the
    # `from . import _vidmag_cuda` in batched.py, ops.py and the rest is
    # unaffected; they run after this module finishes.
    _vidmag_cuda = importlib.import_module(f"{__name__}._vidmag_cuda")
    _have_cuda = True
    import_error: Exception | None = None
except ImportError as _e:  # pragma: no cover - exercised on Mac dev host
    _have_cuda = False
    import_error = _e

from .runtime import have_cuda, require_cuda

# Set runtime.import_error to the real captured error (it defaults to None).
# Done after the runtime import completes to avoid the circular import that
# previously swallowed the error.
from . import runtime as _runtime
_runtime.import_error = import_error
del _runtime


def __getattr__(name: str):
    # Surface the pipeline entry points lazily; avoids importing the
    # pipeline orchestration code (which needs _vidmag_cuda) on a non-CUDA host.
    #
    # The two hot pipelines (color gdown+ideal, motion lpyr+iir) resolve to
    # the OPTIMIZED batched path (batched.py) — the device-resident,
    # launch-collapsed implementation. The two rarer motion variants
    # (lpyr+ideal, lpyr+butter) resolve to the per-frame path (pipelines.py),
    # which is the only place they're implemented.
    #
    # The ``*_core`` names route the same way. That routing is what makes this
    # module the CUDA backend's Pipelines implementation (plan section 3c):
    # ``vidmag.backend.select("cuda")`` returns this package, and the facade calls
    # ``<stem>_core`` on it, so "fastest available CUDA implementation of this
    # pipeline" is decided in exactly one place — here.
    _BATCHED = {
        "magnify_color_gdown_ideal", "magnify_motion_lpyr_iir",
        "color_gdown_ideal_core", "motion_lpyr_iir_core",
        "color_gdown_ideal_fp16_core", "motion_lpyr_iir_fp16_core",
    }
    _PIPELINES = {
        "magnify_motion_lpyr_ideal", "magnify_motion_lpyr_butter",
        "motion_lpyr_ideal_core", "motion_lpyr_butter_core",
    }
    if name in _BATCHED:
        from . import batched
        return getattr(batched, name)
    if name in _PIPELINES:
        from . import pipelines
        return getattr(pipelines, name)
    # The public array type, resolved lazily like everything else here so that
    # importing vidmag.cuda on a machine without the extension still succeeds.
    if name == "DeviceArray":
        from .array import DeviceArray
        return DeviceArray
    # Allow ``from vidmag.cuda import batched`` / ``import vidmag.cuda.benchmark`` to
    # work even though __getattr__ is defined (a module-level __getattr__ would
    # otherwise shadow the standard submodule-import fallback).
    import importlib
    try:
        return importlib.import_module(f"{__name__}.{name}")
    except ImportError:
        pass
    raise AttributeError(name)


__all__ = [
    "have_cuda",
    "import_error",
    "require_cuda",
    "magnify_color_gdown_ideal",
    "magnify_motion_lpyr_ideal",
    "magnify_motion_lpyr_butter",
    "magnify_motion_lpyr_iir",
    # The Pipelines protocol (plan section 3c): array in, array out.
    "color_gdown_ideal_core",
    "motion_lpyr_ideal_core",
    "motion_lpyr_butter_core",
    "motion_lpyr_iir_core",
    # Lower-precision variants; only these two pipelines have one.
    "color_gdown_ideal_fp16_core",
    "motion_lpyr_iir_fp16_core",
    # The building blocks, for composing something this project does not
    # provide. `ops` is the Ops protocol of plan section 3c; `DeviceArray` is
    # the array type every one of them takes and returns.
    "ops",
    "DeviceArray",
]


def __dir__() -> list[str]:
    return sorted(__all__)
