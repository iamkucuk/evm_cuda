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
    # The ignore below is for a compiled .so that CMake writes into this
    # directory. It carries no stubs, so mypy can never resolve it, and the
    # try/except around this line is precisely the code that handles its
    # absence — there is no bug here to hide. warn_unused_ignores keeps it
    # honest: it turns red the day mypy can see the module.
    from . import _evm_cuda  # type: ignore[attr-defined]  # noqa: F401
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
    # pipeline orchestration code (which needs _evm_cuda) on a non-CUDA host.
    #
    # The two hot pipelines (color gdown+ideal, motion lpyr+iir) resolve to
    # the OPTIMIZED batched path (batched.py) — the device-resident,
    # launch-collapsed implementation. The two rarer motion variants
    # (lpyr+ideal, lpyr+butter) resolve to the per-frame path (pipelines.py),
    # which is the only place they're implemented.
    #
    # The ``*_core`` names route the same way. That routing is what makes this
    # module the CUDA backend's Pipelines implementation (plan section 3c):
    # ``evm.backend.select("cuda")`` returns this package, and the facade calls
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
    # importing evm.cuda on a machine without the extension still succeeds.
    if name == "DeviceArray":
        from .array import DeviceArray
        return DeviceArray
    # Allow ``from evm.cuda import batched`` / ``import evm.cuda.benchmark`` to
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
