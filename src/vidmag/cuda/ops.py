"""The GPU building blocks, as operations on :class:`~vidmag.cuda.array.DeviceArray`.

These are the primitives the four magnification pipelines are built from,
exposed so they can be used on their own or chained into something this project
does not provide. Every one takes and returns a ``DeviceArray``, so a chain of
them stays on the GPU: there is no copy back through host memory between steps.

Each function here has a counterpart of the same name in :mod:`vidmag.cpu.ops`
that computes the same thing with NumPy. That pairing is what the conformance
tests compare, and it is what a new backend has to reproduce.

Two things these wrappers do that the compiled bindings underneath do not.
They check shape and dtype, so a mistake raises here rather than reading past
the end of a buffer on the device. And they hide the layout shuffling the
kernels need — several of them work on a plane-per-channel arrangement, or on a
pixels-by-time one — so callers only ever see frames as (time, height, width, 3).
"""

from __future__ import annotations

import numpy as np

from . import _vidmag_cuda
from .array import DeviceArray
from .batched import _d_binom5, _d_binom5_sum1
from .runtime import butter_bandpass_coeffs

__all__ = [
    "bgr_u8_to_ntsc",
    "blur_dn",
    "build_lpyr",
    "recon_lpyr",
    "ideal_bandpass",
    "butter_bandpass",
    "iir_bandpass",
    "apply_gain",
    "upsample_add_quantize",
    "add_and_quantize",
    "level_sizes",
]


def _check(
    arr: DeviceArray,
    *,
    ndim: int | None = None,
    dtype: str | None = None,
    name: str = "input",
) -> None:
    if not isinstance(arr, DeviceArray):
        raise TypeError(f"{name}: expected a DeviceArray, got {type(arr).__name__}")
    if ndim is not None and len(arr.shape) != ndim:
        raise ValueError(f"{name}: expected {ndim} dimensions, got shape {arr.shape}")
    if dtype is not None and arr.dtype != np.dtype(dtype):
        raise TypeError(f"{name}: expected dtype {dtype}, got {arr.dtype}")


def level_sizes(height: int, width: int, levels: int) -> list[tuple[int, int]]:
    """The (height, width) of each pyramid level, finest first."""
    out, h, w = [], height, width
    for _ in range(levels):
        out.append((h, w))
        h, w = (h + 1) // 2, (w + 1) // 2
    return out


# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------


def bgr_u8_to_ntsc(frames: DeviceArray) -> DeviceArray:
    """Convert (T, H, W, 3) 8-bit blue-green-red frames to NTSC, float32.

    NTSC separates brightness from colour, which is what lets the pipelines
    amplify the two by different amounts.
    """
    _check(frames, ndim=4, dtype="uint8", name="frames")
    T, H, W, _ = frames.shape
    out = DeviceArray.empty((T, H, W, 3), np.float32)
    _vidmag_cuda.batched_bgr_u8_to_ntsc_f32(frames.ptr, out.ptr, T, H, W)
    return out


# ---------------------------------------------------------------------------
# Spatial
# ---------------------------------------------------------------------------


def blur_dn(frames: DeviceArray, levels: int) -> DeviceArray:
    """Blur and halve the resolution, ``levels`` times.

    This is the spatial stage of the colour pipeline. The kernel works on one
    plane per channel, so the conversion to that arrangement and back is done
    here rather than being the caller's problem.
    """
    _check(frames, ndim=4, dtype="float32", name="frames")
    if levels < 0:
        raise ValueError(f"blur_dn: levels must not be negative, got {levels}")
    T, H, W, _ = frames.shape
    hl, wl = level_sizes(H, W, levels + 1)[-1]

    planar = DeviceArray.empty((T * 3, H, W), np.float32)
    _vidmag_cuda.batched_to_planar_3ch(frames.ptr, planar.ptr, T, H, W)

    small_planar = DeviceArray.empty((T * 3, hl, wl), np.float32)
    _vidmag_cuda.batched_blur_dn_color(
        planar.ptr, small_planar.ptr, T * 3, H, W, levels, _d_binom5_sum1(), 5
    )

    out = DeviceArray.empty((T, hl, wl, 3), np.float32)
    _vidmag_cuda.batched_planar_to_interleaved_3ch(small_planar.ptr, out.ptr, T, hl, wl)
    return out


def build_lpyr(frames: DeviceArray, levels: int) -> list[DeviceArray]:
    """Build a Laplacian pyramid: one band per scale, finest first.

    Each band holds the detail at one scale and the last holds what is left
    over. :func:`recon_lpyr` sums them back together.

    The kernel writes every band into one flat allocation; this splits that
    into a band per level so each is an array with its own shape.
    """
    _check(frames, ndim=4, dtype="float32", name="frames")
    if levels < 1:
        raise ValueError(f"build_lpyr: levels must be at least 1, got {levels}")
    T, H, W, _ = frames.shape
    sizes = level_sizes(H, W, levels)

    planar = DeviceArray.empty((T * 3, H, W), np.float32)
    _vidmag_cuda.batched_to_planar_3ch_chan_outer(frames.ptr, planar.ptr, T, H, W)

    total = sum(h * w for h, w in sizes) * T * 3
    flat = DeviceArray.empty((total,), np.float32)
    _vidmag_cuda.batched_lpyr_build(
        planar.ptr, flat.ptr, T, H, W, levels, _d_binom5(), 5
    )

    bands, offset = [], 0
    for h, w in sizes:
        count = h * w * T * 3
        band = DeviceArray.empty((3, T, h, w), np.float32)
        _vidmag_cuda.copy_f32(flat.ptr, band.ptr, count, src_offset=offset)
        bands.append(band)
        offset += count
    return bands


def recon_lpyr(bands: list[DeviceArray], height: int, width: int) -> DeviceArray:
    """Sum a Laplacian pyramid back into frames, one plane per channel."""
    if not bands:
        raise ValueError("recon_lpyr: needs at least one band")
    for i, b in enumerate(bands):
        _check(b, ndim=4, dtype="float32", name=f"bands[{i}]")
    _, T, _, _ = bands[0].shape

    total = sum(int(np.prod(b.shape)) for b in bands)
    flat = DeviceArray.empty((total,), np.float32)
    offset = 0
    for b in bands:
        count = int(np.prod(b.shape))
        _vidmag_cuda.copy_f32(b.ptr, flat.ptr, count, dst_offset=offset)
        offset += count

    out = DeviceArray.empty((T * 3, height, width), np.float32)
    _vidmag_cuda.batched_lpyr_recon(
        flat.ptr, out.ptr, T, height, width, len(bands), _d_binom5(), 5
    )
    return out


# ---------------------------------------------------------------------------
# Temporal
# ---------------------------------------------------------------------------


def _to_signal(frames: DeviceArray) -> tuple[DeviceArray, int, int]:
    """Rearrange frames to the (pixels, time) layout the filters read."""
    _check(frames, ndim=4, dtype="float32", name="frames")
    T, H, W, C = frames.shape
    N = H * W * C
    sig = DeviceArray.empty((N, T), np.float32)
    _vidmag_cuda.batched_thwc_to_nt(frames.ptr, sig.ptr, T, N)
    return sig, T, N


def _to_frames(sig: DeviceArray, shape: tuple[int, ...]) -> DeviceArray:
    T, H, W, C = shape
    out = DeviceArray.empty(shape, np.float32)
    _vidmag_cuda.batched_nt_to_thwc_scaled(sig.ptr, out.ptr, T, H * W * C, 1.0)
    return out


def ideal_bandpass(
    frames: DeviceArray, fl: float, fh: float, sampling_rate: float
) -> DeviceArray:
    """Keep only frequencies strictly between ``fl`` and ``fh``.

    Computed with a Fourier transform over time, so it needs the whole clip and
    cannot run on a live stream. :func:`butter_bandpass` and
    :func:`iir_bandpass` can.

    A band narrower than the spacing between frequency bins selects nothing and
    returns zeros; :func:`vidmag.cpu.filters.ideal_bandpass` warns when that
    happens and explains how many frames would be needed.
    """
    sig, T, N = _to_signal(frames)
    filtered = DeviceArray.empty((N, T), np.float32)
    _vidmag_cuda.batched_ideal_bandpass(
        sig.ptr, filtered.ptr, T, N, fl, fh, sampling_rate
    )
    return _to_frames(filtered, frames.shape)


def butter_bandpass(
    frames: DeviceArray, fl: float, fh: float, sampling_rate: float, order: int = 1
) -> DeviceArray:
    """First-order Butterworth bandpass, applied along time.

    Runs forward in time only, so unlike :func:`ideal_bandpass` it can be used
    on frames as they arrive.
    """
    sig, T, N = _to_signal(frames)
    high, low = butter_bandpass_coeffs(fl, fh, sampling_rate, order)
    filtered = DeviceArray.empty((N, T), np.float32)
    _vidmag_cuda.batched_butter_bandpass(
        sig.ptr, filtered.ptr, T, N, high[0], high[1], high[2], low[0], low[1], low[2]
    )
    return _to_frames(filtered, frames.shape)


def iir_bandpass(frames: DeviceArray, r1: float, r2: float) -> DeviceArray:
    """The difference of two exponential moving averages, along time.

    ``r1`` is the faster decay and ``r2`` the slower; subtracting one from the
    other passes a band. This is the filter the motion pipeline uses, and it is
    causal, so it also works frame by frame.
    """
    sig, T, N = _to_signal(frames)
    filtered = DeviceArray.empty((N, T), np.float32)
    _vidmag_cuda.batched_iir_bandpass(sig.ptr, filtered.ptr, T, N, r1, r2)
    return _to_frames(filtered, frames.shape)


# ---------------------------------------------------------------------------
# Amplification and rendering
# ---------------------------------------------------------------------------


def apply_gain(
    frames: DeviceArray, gain_y: float, gain_i: float, gain_q: float
) -> DeviceArray:
    """Scale the three NTSC channels independently, in place.

    Returns the array it was given: the multiply happens on the existing
    memory rather than allocating a second copy.
    """
    _check(frames, ndim=4, dtype="float32", name="frames")
    T, H, W, _ = frames.shape
    _vidmag_cuda.batched_apply_channel_gain(frames.ptr, T, H, W, gain_y, gain_i, gain_q)
    return frames


def upsample_add_quantize(
    ntsc: DeviceArray, small: DeviceArray, chrom_attenuation: float = 1.0
) -> DeviceArray:
    """Scale the amplified signal back up, add it, and convert to 8-bit.

    This is the colour pipeline's final step: ``small`` holds the amplified
    signal at reduced resolution, ``ntsc`` the original full-resolution frames.
    """
    _check(ntsc, ndim=4, dtype="float32", name="ntsc")
    _check(small, ndim=4, dtype="float32", name="small")
    T, H, W, _ = ntsc.shape
    _, hl, wl, _ = small.shape
    out = DeviceArray.empty((T, H, W, 3), np.uint8)
    _vidmag_cuda.batched_upsample_add_quantize(
        ntsc.ptr, small.ptr, out.ptr, T, hl, wl, H, W, chrom_attenuation
    )
    return out


def add_and_quantize(
    ntsc: DeviceArray, delta_planar: DeviceArray, chrom_attenuation: float = 1.0
) -> DeviceArray:
    """Add a full-resolution amplified signal and convert to 8-bit.

    This is the motion pipeline's final step. ``delta_planar`` is the
    reconstructed pyramid, which is arranged one plane per channel.
    """
    _check(ntsc, ndim=4, dtype="float32", name="ntsc")
    _check(delta_planar, ndim=3, dtype="float32", name="delta_planar")
    T, H, W, _ = ntsc.shape
    out = DeviceArray.empty((T, H, W, 3), np.uint8)
    _vidmag_cuda.batched_add_planar_quantize(
        ntsc.ptr, delta_planar.ptr, out.ptr, T, H, W, chrom_attenuation
    )
    return out
