"""EVM CUDA port — Python wrapper package.

This package wraps the compiled `_evm_cuda` extension (built by
`cuda/CMakeLists.txt`) and exposes the four magnification pipelines as
drop-in replacements for the Python baseline `evm.magnify_*`.

Layout:
- ``_evm_cuda``  — the compiled pybind11 module (loaded lazily; raises on
                   machines without CUDA / nvcc-built .so).
- ``runtime``    — small helpers (cuFFT plan cache, version probe).
- ``pipelines``  — the four ``magnify_*`` pipeline orchestrators that call
                   into `_evm_cuda` kernels.

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
    # Surface the four pipeline entry points lazily; avoids importing the
    # pipeline orchestration code (which needs _evm_cuda) on a non-CUDA host.
    if name in {"magnify_color_gdown_ideal",
                "magnify_motion_lpyr_ideal",
                "magnify_motion_lpyr_butter",
                "magnify_motion_lpyr_iir"}:
        from . import pipelines
        return getattr(pipelines, name)
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
