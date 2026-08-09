"""The Ops protocol — the primitive operations every backend must implement.

Level one of the two-level backend interface in `docs/dev/PLAN.md` section 3c.
A backend that implements these operations gets all four magnification
pipelines for free; a backend that can fuse them may additionally override the
:mod:`~evm.backend.pipelines` level.

**The operation list is derived, not invented.** Every operation below is called
by one of the four CPU pipelines in ``evm/cpu/magnify.py`` — the correctness
oracle — and each has a counterpart in the CUDA bindings. Host-side scalar maths
that touches no pixels is deliberately *not* here: ``max_pyr_ht``
(``evm/cpu/pyramids.py:106``) and ``figure6_alpha_schedule``
(``evm/cpu/magnify.py:70``) stay plain functions shared by every backend.

+-------------------------+-----------------------------------+------------------------------------+
| Operation               | CPU baseline call site            | CUDA counterpart                   |
+=========================+===================================+====================================+
| ``bgr_u8_to_ntsc``      | ``magnify.py:116`` ``_rgb_frame_  | ``bgr_u8_to_ntsc_f32`` /           |
|                         | to_ntsc``, used :223 :240 :293    | ``batched_bgr_u8_to_ntsc_f32``     |
|                         | :352                              |                                    |
+-------------------------+-----------------------------------+------------------------------------+
| ``blur_dn``             | ``magnify.py:223`` ``blur_dn_clr``| ``blur_dn`` /                      |
|                         | (``pyramids.py:136``)             | ``batched_blur_dn_color``          |
+-------------------------+-----------------------------------+------------------------------------+
| ``build_lpyr``          | ``magnify.py:296`` and ``:355``   | ``lpyr_build`` /                   |
|                         | ``laplacian_pyramid_channels``    | ``batched_lpyr_build``             |
+-------------------------+-----------------------------------+------------------------------------+
| ``recon_lpyr``          | ``magnify.py:260`` ``reconstruct_ | ``lpyr_recon`` /                   |
|                         | from_channels``                   | ``batched_lpyr_recon``             |
+-------------------------+-----------------------------------+------------------------------------+
| ``ideal_bandpass``      | ``magnify.py:228`` (colour),      | ``ideal_bandpass`` /               |
|                         | ``:314`` (motion)                 | ``batched_ideal_bandpass``         |
+-------------------------+-----------------------------------+------------------------------------+
| ``butter_bandpass``     | ``magnify.py:420``                | ``butter_bandpass`` (batched       |
|                         |                                   | binding still missing, Phase 4)    |
+-------------------------+-----------------------------------+------------------------------------+
| ``iir_bandpass``        | ``magnify.py:453``                | ``iir_bandpass`` /                 |
|                         |                                   | ``batched_iir_bandpass``           |
+-------------------------+-----------------------------------+------------------------------------+
| ``apply_gain``          | ``magnify.py:234`` (colour gain), | ``apply_channel_gain`` /           |
|                         | ``:315``/``:384`` (per-level      | ``attenuate_chrom`` /              |
|                         | alpha), ``:261`` (chroma atten.)  | ``batched_apply_channel_gain``     |
+-------------------------+-----------------------------------+------------------------------------+
| ``upsample_bilinear``   | ``magnify.py:242`` ``cv2.resize`` | ``batched_bilinear_upsample_3ch``  |
+-------------------------+-----------------------------------+------------------------------------+
| ``add_and_quantize``    | ``magnify.py:245`` + ``:122``     | ``add_and_quantize`` /             |
|                         | ``_ntsc_to_bgr_uint8``; also      | ``batched_add_planar_quantize`` /  |
|                         | ``:263``+``:325`` and ``:390``    | ``batched_upsample_add_quantize``  |
+-------------------------+-----------------------------------+------------------------------------+

Ten pixel operations, plus :meth:`Ops.from_numpy` / :meth:`Ops.to_numpy` for
host transfer.

Conventions every implementation must honour
--------------------------------------------

* **Time-major batches.** Frame stacks are ``(T, H, W, 3)``; pyramid bands are
  ``(T, H_l, W_l, 3)``. Temporal filters run along axis 0. A backend is free to
  transpose internally — the CUDA path filters in an ``(N, T)`` layout — but the
  arrays crossing this interface are time-major.
* **Colour space.** ``bgr_u8_to_ntsc`` consumes OpenCV-order BGR ``uint8`` and
  produces NTSC YIQ floats in [0, 1]; ``add_and_quantize`` is the only way back,
  and it clips, rounds and returns BGR ``uint8``. The inverse colour conversion
  lives inside it because that is how both existing implementations do it
  (``_ntsc_to_bgr_uint8`` on the CPU, the fused ``batched_*_quantize`` kernels on
  the GPU); splitting it out would add an operation neither backend performs
  alone.
* **Precision is the backend's business.** The CPU oracle computes in float64;
  the CUDA path has FP32 and FP16 variants. The protocol fixes layout and
  semantics, not the working dtype — that is what
  :class:`~evm.backend.registry.Capabilities` advertises.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np

__all__ = ["Array", "Ops"]


@runtime_checkable
class Array(Protocol):
    """An opaque backend array handle.

    A plain :class:`numpy.ndarray` satisfies this protocol as-is — NumPy >= 2.0
    supplies ``.device`` (``'cpu'``) alongside ``.shape`` and ``.dtype`` — so the
    CPU backend needs no wrapper class. Host transfer is therefore expressed as
    :meth:`Ops.to_numpy` / :meth:`Ops.from_numpy` rather than as a ``to_numpy()``
    method on the handle, because ``ndarray`` has no such method and adding one
    would have meant wrapping every CPU array for no gain.

    ``isinstance(x, Array)`` works; ``issubclass`` does not (a runtime-checkable
    protocol with data members forbids it).
    """

    @property
    def shape(self) -> tuple[int, ...]:
        """Dimensions, time-major for anything with a temporal axis."""

    @property
    def dtype(self) -> np.dtype:
        """The element type, reported as a NumPy dtype on every backend.

        Device arrays report the dtype they mirror (``float32``, ``float16``,
        ``uint8``) so capability checks read the same everywhere.
        """

    @property
    def device(self) -> str:
        """Where the data lives: ``'cpu'``, ``'cuda:0'``, ...."""


@runtime_checkable
class Ops(Protocol):
    """The minimum a backend must provide. See the module docstring for the
    derivation of this list and the layout conventions."""

    # -- host transfer ------------------------------------------------------

    def from_numpy(self, array: np.ndarray) -> Array:
        """Move a host array onto this backend, preserving shape and dtype."""

    def to_numpy(self, array: Array) -> np.ndarray:
        """Copy a backend array back to the host."""

    # -- colour -------------------------------------------------------------

    def bgr_u8_to_ntsc(self, frames: Array) -> Array:
        """``(T, H, W, 3)`` BGR ``uint8`` -> NTSC YIQ float in [0, 1].

        MATLAB ``rgb2ntsc``; ``evm/cpu/magnify.py:116``.
        """

    def add_and_quantize(self, frames_ntsc: Array, delta: Array) -> Array:
        """``frames_ntsc + delta`` -> BGR ``uint8``, converted, clipped, rounded.

        The render step of all four pipelines (``evm/cpu/magnify.py:245``,
        ``:263``, ``:390``, each followed by ``_ntsc_to_bgr_uint8`` at ``:122``).
        Both operands are ``(T, H, W, 3)`` NTSC floats.
        """

    # -- spatial ------------------------------------------------------------

    def blur_dn(self, frames: Array, levels: int) -> Array:
        """``levels`` rounds of binom5 blur + decimate-by-2, per channel.

        matlabPyrTools ``blurDnClr``; the colour pipeline's spatial stage
        (``evm/cpu/magnify.py:223``, ``evm/cpu/pyramids.py:136``). The filter is
        ``BINOM5_SUM1`` with ``reflect1`` borders.
        """

    def build_lpyr(self, frames: Array, levels: int) -> list[Array]:
        """Laplacian pyramid per frame and per channel, finest band first.

        matlabPyrTools ``buildLpyr`` with ``BINOM5`` and ``reflect1`` borders
        (``evm/cpu/pyramids.py:147``); the motion pipelines call it through
        ``laplacian_pyramid_channels`` (``evm/cpu/magnify.py:296``, ``:355``).
        ``levels`` is explicit — the "auto" height is ``1 + max_pyr_ht(...)``,
        computed by the caller so the policy lives in one place.

        Returns one ``(T, H_l, W_l, 3)`` band per level. The MATLAB ``pind``
        table is not returned: it is exactly the bands' own shapes.
        """

    def recon_lpyr(self, bands: Sequence[Array]) -> Array:
        """Collapse the pyramid back to ``(T, H, W, 3)``.

        matlabPyrTools ``reconLpyr`` (``evm/cpu/pyramids.py:182``), reached via
        ``reconstruct_from_channels`` at ``evm/cpu/magnify.py:260``. Inverse of
        :meth:`build_lpyr`, same band order.
        """

    def upsample_bilinear(self, frames: Array, height: int, width: int) -> Array:
        """Bilinear resize of every frame to ``(height, width)``.

        MATLAB ``imresize`` default; the colour pipeline's render stage
        (``evm/cpu/magnify.py:242``, ``cv2.INTER_LINEAR``).
        """

    # -- temporal (all filter along axis 0) ---------------------------------

    def ideal_bandpass(
        self, series: Array, fl: float, fh: float, sampling_rate: float
    ) -> Array:
        """Brick-wall FFT bandpass, strict ``(fl, fh)`` on one-sided bins.

        MATLAB ``ideal_bandpassing`` (``evm/cpu/filters.py:34``); called at
        ``evm/cpu/magnify.py:228`` (colour) and ``:314`` (motion).
        """

    def butter_bandpass(
        self,
        series: Array,
        fl: float,
        fh: float,
        sampling_rate: float,
        order: int = 1,
    ) -> Array:
        """Difference of two Butterworth lowpasses, ``lowpass(fh) - lowpass(fl)``.

        ``evm/cpu/filters.py:64``; called at ``evm/cpu/magnify.py:420``. Cutoffs
        are in Hz and normalised to Nyquist by the implementation, as MATLAB does.
        """

    def iir_bandpass(self, series: Array, r1: float, r2: float) -> Array:
        """Two first-order IIR lowpasses subtracted, state initialised to x[0].

        ``evm/cpu/filters.py:93``; called at ``evm/cpu/magnify.py:453``. Requires
        ``r1 > r2``. Causal, which is why this is the streaming-capable filter.
        """

    # -- amplification ------------------------------------------------------

    def apply_gain(
        self, frames: Array, gain_y: float, gain_i: float, gain_q: float
    ) -> Array:
        """Scale the three NTSC channels independently.

        One operation covers all three amplification call sites: the colour
        pipeline's ``(alpha, alpha*chrom, alpha*chrom)``
        (``evm/cpu/magnify.py:234``), the motion pipelines' per-level Figure-6
        alpha applied to every channel (``:263``, ``:384``), and chrominance
        attenuation ``(1, chrom, chrom)`` on the reconstructed delta (``:261``).
        """
