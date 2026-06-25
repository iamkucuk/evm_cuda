"""EVM CUDA port — Python wrapper package.

This package wraps the compiled `_evm_cuda` extension (built by
`cuda/CMakeLists.txt`) and exposes the four magnification pipelines as
drop-in replacements for the Python baseline `evm.magnify_*`.

Layout:
- ``_evm_cuda``  — the compiled pybind11 module (loaded lazily; raises on
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

try:
    from . import _evm_cuda  # noqa: F401  (built by CMake into this package dir)
    _have_cuda = True
except ImportError as _e:  # pragma: no cover - exercised on Mac dev host
    _have_cuda = False
    _import_error = _e

from .runtime import have_cuda, import_error, require_cuda  # noqa: F401


def __getattr__(name: str):
    # Surface the pipeline entry points lazily; avoids importing the
    # pipeline orchestration code (which needs _evm_cuda) on a non-CUDA host.
    #
    # The two hot pipelines (color gdown+ideal, motion lpyr+iir) resolve to
    # the OPTIMIZED batched path (batched.py) — the device-resident,
    # launch-collapsed implementation. The two rarer motion variants
    # (lpyr+ideal, lpyr+butter) resolve to the per-frame path (pipelines.py),
    # which is the only place they're implemented.
    _BATCHED = {"magnify_color_gdown_ideal", "magnify_motion_lpyr_iir"}
    _PIPELINES = {"magnify_motion_lpyr_ideal", "magnify_motion_lpyr_butter"}
    if name in _BATCHED:
        from . import batched
        return getattr(batched, name)
    if name in _PIPELINES:
        from . import pipelines
        return getattr(pipelines, name)
    # Allow ``from evm_cuda import batched`` / ``import evm_cuda.benchmark`` to
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
]
