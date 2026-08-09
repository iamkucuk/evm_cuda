"""The NumPy implementation of :class:`evm.backend.Ops`.

This is the reference every other backend is measured against, and the one that
proves the generic pipelines in :mod:`evm.backend.generic` are correct: running
those pipelines through these operations must reproduce what
:mod:`evm.cpu.magnify` produces directly.

Arrays here are plain NumPy arrays. Nothing wraps them, because there is
nothing to wrap: they already carry a shape, a dtype and their own memory. A
backend that owns memory on a separate device needs its own array type; this
one does not, and inventing one would be ceremony.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import filters as _filters
from . import pyramids as _pyramids
from .magnify import _ntsc_to_bgr_uint8, _rgb_frame_to_ntsc

__all__ = ["NumpyOps", "OPS"]


class NumpyOps:
    """Every primitive the pipelines need, computed with NumPy."""

    name = "cpu"

    # -- transfer -----------------------------------------------------------

    def from_numpy(self, array: np.ndarray) -> np.ndarray:
        return array

    def to_numpy(self, array: np.ndarray) -> np.ndarray:
        return array

    # -- colour -------------------------------------------------------------

    def bgr_u8_to_ntsc(self, frames: np.ndarray) -> np.ndarray:
        return np.stack([_rgb_frame_to_ntsc(f) for f in frames], axis=0)

    def add_and_quantize(self, frames_ntsc: np.ndarray,
                         delta: np.ndarray) -> np.ndarray:
        return np.stack(
            [_ntsc_to_bgr_uint8(frames_ntsc[i] + delta[i])
             for i in range(len(frames_ntsc))], axis=0)

    # -- spatial ------------------------------------------------------------

    def blur_dn(self, frames: np.ndarray, levels: int) -> np.ndarray:
        return np.stack([_pyramids.blur_dn_clr(f, levels) for f in frames],
                        axis=0)

    def build_lpyr(self, frames: np.ndarray, levels: int) -> list[np.ndarray]:
        """One band per scale, each shaped (time, height, width, 3)."""
        per_frame = [_pyramids.laplacian_pyramid_channels(f, levels)[0]
                     for f in frames]
        return [np.stack([per_frame[t][level] for t in range(len(frames))],
                         axis=0)
                for level in range(levels)]

    def recon_lpyr(self, bands) -> np.ndarray:
        bands = list(bands)
        T = bands[0].shape[0]
        pind = np.array([[b.shape[1], b.shape[2]] for b in bands],
                        dtype=np.float64)
        return np.stack(
            [_pyramids.reconstruct_from_channels([b[t] for b in bands], pind)
             for t in range(T)], axis=0)

    def upsample_bilinear(self, frames: np.ndarray, height: int,
                          width: int) -> np.ndarray:
        # The reference uses MATLAB's imresize, whose default is bilinear;
        # OpenCV's INTER_LINEAR is the match, and is what evm.cpu.magnify
        # already calls, so both routes resize identically.
        return np.stack(
            [cv2.resize(f, (width, height), interpolation=cv2.INTER_LINEAR)
             for f in frames], axis=0)

    # -- temporal -----------------------------------------------------------

    def ideal_bandpass(self, series: np.ndarray, fl: float, fh: float,
                       sampling_rate: float) -> np.ndarray:
        return _filters.ideal_bandpass(series, fl, fh, sampling_rate, axis=0)

    def butter_bandpass(self, series: np.ndarray, fl: float, fh: float,
                        sampling_rate: float, order: int = 1) -> np.ndarray:
        return _filters.butter_bandpass(series, fl, fh, sampling_rate,
                                        order=order, axis=0)

    def iir_bandpass(self, series: np.ndarray, r1: float,
                     r2: float) -> np.ndarray:
        return _filters.iir_bandpass(series, r1, r2, axis=0)

    # -- amplification ------------------------------------------------------

    def apply_gain(self, frames: np.ndarray, gain_y: float, gain_i: float,
                   gain_q: float) -> np.ndarray:
        return frames * np.array([gain_y, gain_i, gain_q], dtype=np.float64)

    # -- streaming ----------------------------------------------------------

    def iir_step(self, fast: np.ndarray, slow: np.ndarray,
                 current: np.ndarray, r1: float, r2: float) -> np.ndarray:
        """Advance both running averages by one frame; return the difference.

        Updates ``fast`` and ``slow`` in place, matching what the device
        backends do, so that streaming behaves the same way whichever is in
        use.
        """
        fast *= (1.0 - r1)
        fast += current * r1
        slow *= (1.0 - r2)
        slow += current * r2
        return fast - slow


#: The single instance backends and tests share. These operations hold no
#: state, so there is no reason for a caller to build another one.
OPS = NumpyOps()
