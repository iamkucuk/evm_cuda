"""The same building blocks as :mod:`evm.cuda.ops`, computed with NumPy.

This module adds no arithmetic of its own. It is a naming layer: the functions
here forward to the implementations in :mod:`evm.cpu.pyramids` and
:mod:`evm.cpu.filters`, under the names and argument order the GPU operations
use. Two reasons that is worth having.

A caller can write one chain of operations and choose the backend separately,
because ``evm.cpu.ops.blur_dn`` and ``evm.cuda.ops.blur_dn`` take the same
arguments in the same order.

And the conformance tests can run the same list of operations against every
backend, comparing each against this one. That makes this module the reference
the GPU is checked against, which is the role the whole project already gives
the NumPy code.
"""

from __future__ import annotations

import numpy as np

from . import filters as _filters
from . import pyramids as _pyramids
from .magnify import _ntsc_to_bgr_uint8, _rgb_frame_to_ntsc

__all__ = [
    "bgr_u8_to_ntsc",
    "blur_dn",
    "build_lpyr",
    "recon_lpyr",
    "ideal_bandpass",
    "butter_bandpass",
    "iir_bandpass",
    "apply_gain",
    "level_sizes",
]


def level_sizes(height: int, width: int, levels: int) -> list[tuple[int, int]]:
    """The (height, width) of each pyramid level, finest first."""
    out, h, w = [], height, width
    for _ in range(levels):
        out.append((h, w))
        h, w = (h + 1) // 2, (w + 1) // 2
    return out


def bgr_u8_to_ntsc(frames: np.ndarray) -> np.ndarray:
    """Convert (T, H, W, 3) 8-bit blue-green-red frames to NTSC, float32.

    The per-frame helper takes 8-bit input and does the channel reversal and
    the divide by 255 itself, so frames are handed to it unchanged.
    """
    if frames.dtype != np.uint8:
        raise TypeError(f"bgr_u8_to_ntsc: expected uint8, got {frames.dtype}")
    return np.stack([_rgb_frame_to_ntsc(f) for f in frames], axis=0).astype(np.float32)


def ntsc_to_bgr_u8(frames: np.ndarray) -> np.ndarray:
    """Convert NTSC frames back to 8-bit blue-green-red, with rounding."""
    return np.stack([_ntsc_to_bgr_uint8(f) for f in frames], axis=0)


def blur_dn(frames: np.ndarray, levels: int) -> np.ndarray:
    """Blur and halve the resolution, ``levels`` times."""
    return np.stack([_pyramids.blur_dn_clr(f, levels) for f in frames], axis=0)


def build_lpyr(frames: np.ndarray, levels: int) -> list[np.ndarray]:
    """Build a Laplacian pyramid: one band per scale, finest first.

    Returned bands are shaped (3, T, height, width) to match the GPU layout,
    which keeps each colour channel's planes together.
    """
    # The per-frame helper returns (levels, pind); each entry of `levels` is
    # one pyramid band shaped (height, width, channel), so the channel axis has
    # to be moved to the front to match the GPU's plane-per-channel layout.
    per_frame = [_pyramids.laplacian_pyramid_channels(f, levels)[0] for f in frames]
    bands = []
    for level in range(levels):
        stacked = np.stack([per_frame[t][level] for t in range(len(frames))])
        bands.append(np.ascontiguousarray(np.moveaxis(stacked, 3, 0), dtype=np.float32))
    return bands


def recon_lpyr(bands: list[np.ndarray], height: int, width: int) -> np.ndarray:
    """Sum a Laplacian pyramid back, returning one plane per channel.

    Bands arrive shaped (channel, time, height, width), matching the GPU
    layout. The per-frame helper wants the opposite: a list of levels each
    shaped (height, width, channel), plus the table of level dimensions it
    calls ``pind``. Both are rebuilt here.
    """
    T = bands[0].shape[1]
    pind = np.array([[b.shape[2], b.shape[3]] for b in bands], dtype=np.float64)
    out = np.empty((T * 3, height, width), dtype=np.float32)
    for t in range(T):
        levels = [np.moveaxis(b[:, t], 0, 2) for b in bands]
        frame = _pyramids.reconstruct_from_channels(levels, pind)
        for c in range(3):
            out[c * T + t] = frame[:, :, c]
    return out


def ideal_bandpass(
    frames: np.ndarray, fl: float, fh: float, sampling_rate: float
) -> np.ndarray:
    """Keep only frequencies strictly between ``fl`` and ``fh``, along time."""
    return _filters.ideal_bandpass(frames, fl, fh, sampling_rate, axis=0)


def butter_bandpass(
    frames: np.ndarray, fl: float, fh: float, sampling_rate: float, order: int = 1
) -> np.ndarray:
    """First-order Butterworth bandpass, along time."""
    return _filters.butter_bandpass(frames, fl, fh, sampling_rate, order=order, axis=0)


def iir_bandpass(frames: np.ndarray, r1: float, r2: float) -> np.ndarray:
    """The difference of two exponential moving averages, along time."""
    return _filters.iir_bandpass(frames, r1, r2, axis=0)


def apply_gain(
    frames: np.ndarray, gain_y: float, gain_i: float, gain_q: float
) -> np.ndarray:
    """Scale the three NTSC channels independently."""
    return frames * np.array([gain_y, gain_i, gain_q], dtype=frames.dtype)
