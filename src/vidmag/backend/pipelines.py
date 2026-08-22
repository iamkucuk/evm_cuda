"""The Pipelines protocol — the four magnification cores, array in, array out.

Level two of the two-level backend interface in `docs/dev/PLAN.md` section 3c.
Nobody is *required* to implement it: a generic default written once against
:class:`vidmag.backend.Ops` serves every backend. A backend overrides it only when
it can do better by fusing stages — native CUDA does, because its speed comes
from collapsed kernel launches and buffers reused from a device pool across
stages, which op-by-op execution would throw away.

This file is a *description* of the two concrete implementations, not a sketch
they must be bent to fit. The names and signatures below were read off
:mod:`vidmag.cpu.magnify` (the correctness oracle) and :mod:`vidmag.cuda.batched` /
:mod:`vidmag.cuda.pipelines` (the fused override) and match them exactly.

An implementation is any object carrying the four functions — in practice a
module: ``vidmag.cpu.magnify`` is the CPU backend and ``vidmag.cuda`` is the CUDA one,
and both are handed to :func:`vidmag.backend.register` as-is.

``isinstance(vidmag.cpu.magnify, Pipelines)`` is ``True``. ``isinstance(vidmag.cuda,
Pipelines)`` is ``False``, and that is not a defect in either: since Python 3.12
``isinstance`` against a runtime-checkable protocol looks members up with
:func:`inspect.getattr_static`, which deliberately does not run a module's
``__getattr__`` — and ``vidmag/cuda/__init__.py`` resolves each core lazily through
exactly that hook, so importing ``vidmag.cuda`` on a CPU-only machine never reaches
for the compiled extension. Conformance is therefore checked the way the facade
uses it, with ``getattr``, not with ``isinstance``.

Relationship to the four public functions
-----------------------------------------

``vidmag.magnify_color_gdown_ideal`` and friends keep their exact current
signatures (path in, file out, ``float32``/255 array returned) and are thin
wrappers around these cores. The differences between a public function and its
core are deliberate:

* **no paths** — the core takes decoded frames and returns decoded frames;
  reading, writing and the H.264 encoder stay in the wrapper;
* **frame dropping is the wrapper's** — ``DROP_LAST = 10``
  (``vidmag/cpu/magnify.py:59``) is applied by the path-based reader. Plan decision
  D8: the path functions keep dropping the last ten frames, the array API
  (:func:`vidmag.magnify`) drops none unless asked. A core never drops anything.

``sampling_rate`` *is* part of three of the four cores, because it is part of
the public functions and of the MIT reference calls they reproduce: it is the
rate the temporal band is measured against, and ``None`` means "the rate the
frames arrived at", i.e. ``fps``. The fourth core (``motion_lpyr_iir_core``)
has no sampling rate anywhere in it — the r1/r2 recursion runs on frame index —
but still takes ``fps`` positionally so the facade can call any of the four the
same way.

Contract
--------

``frames_bgr_u8`` is a ``(T, H, W, 3)`` BGR ``uint8``
:class:`~vidmag.backend.Ops.Array` and the return value is the same shape and
dtype — the magnified clip, already quantised. ``uint8`` is what both existing
implementations produce naturally (the CPU oracle at
``vidmag/cpu/magnify.py:246``, the CUDA batched pipelines' ``d_out_u8`` buffer);
the ``.astype(np.float32) / 255.0`` in today's public functions is the
wrapper's business, not the core's. Today both implementations take and return
host arrays; Phase 4's ``DeviceArray`` will satisfy :class:`Array` too, which is
why the annotation is the protocol and not ``np.ndarray``.

Optional extras a backend may add
---------------------------------

* **extra keyword-only parameters.** The CUDA cores take an ``on_stage``
  profiling hook that ``vidmag/cuda/benchmark.py`` and ``tests/cuda/test_benchmark.py``
  drive; the CPU cores do not, and are not made to carry a parameter they would
  ignore. Additional optional arguments still satisfy the protocol.
* **lower-precision variants**, named ``<stem>_fp16_core``. ``vidmag.cuda`` has
  ``color_gdown_ideal_fp16_core`` and ``motion_lpyr_iir_fp16_core``; nothing
  else does. :func:`vidmag.magnify` looks the variant up by name and refuses
  ``precision="fp16"`` loudly on a backend that has none, rather than silently
  computing in fp32. The FP32 and FP16 CUDA bodies stay separate on purpose
  (`docs/dev/PLAN.md` section 3d rule 4): merging them risks numeric drift, and
  the accuracy figures in README.md are load-bearing.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..cpu.magnify import EXAGGERATION_FACTOR
from .ops import Array

__all__ = ["Pipelines"]


@runtime_checkable
class Pipelines(Protocol):
    """The four magnify cores. Parameter names, order and defaults mirror
    ``vidmag/cpu/magnify.py`` exactly, minus the path arguments."""

    def color_gdown_ideal_core(
        self,
        frames_bgr_u8: Array,
        fps: float,
        *,
        alpha: float,
        level: int,
        fl: float,
        fh: float,
        chrom_attenuation: float = 1.0,
        sampling_rate: float | None = None,
    ) -> Array:
        """Gaussian-downsampled stack + ideal bandpass — the pulse pipeline.

        MATLAB ``amplify_spatial_Gdown_temporal_ideal.m``; CPU implementation at
        ``vidmag/cpu/magnify.py:191``, public wrapper at ``:471``.
        """

    def motion_lpyr_ideal_core(
        self,
        frames_bgr_u8: Array,
        fps: float,
        *,
        alpha: float,
        lambda_c: float,
        fl: float,
        fh: float,
        chrom_attenuation: float = 0.0,
        sampling_rate: float | None = None,
        exaggeration_factor: float = EXAGGERATION_FACTOR,
    ) -> Array:
        """Laplacian pyramid + ideal bandpass.

        MATLAB ``amplify_spatial_lpyr_temporal_ideal.m``; CPU implementation at
        ``vidmag/cpu/magnify.py:267``, public wrapper at ``:498``.
        """

    def motion_lpyr_butter_core(
        self,
        frames_bgr_u8: Array,
        fps: float,
        *,
        alpha: float,
        lambda_c: float,
        fl: float,
        fh: float,
        chrom_attenuation: float = 0.0,
        sampling_rate: float | None = None,
        order: int = 1,
        exaggeration_factor: float = EXAGGERATION_FACTOR,
    ) -> Array:
        """Laplacian pyramid + first-order Butterworth bandpass.

        MATLAB ``amplify_spatial_lpyr_temporal_butter.m``; CPU implementation at
        ``vidmag/cpu/magnify.py:399``, public wrapper at ``:523``.
        """

    def motion_lpyr_iir_core(
        self,
        frames_bgr_u8: Array,
        fps: float,
        *,
        alpha: float,
        lambda_c: float,
        r1: float,
        r2: float,
        chrom_attenuation: float = 0.1,
        exaggeration_factor: float = EXAGGERATION_FACTOR,
    ) -> Array:
        """Laplacian pyramid + direct r1/r2 IIR bandpass — causal, streamable.

        MATLAB ``amplify_spatial_lpyr_temporal_iir.m``; CPU implementation at
        ``vidmag/cpu/magnify.py:431``, public wrapper at ``:550``. ``fps`` is
        accepted and ignored (see the module docstring).
        """
