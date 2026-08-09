"""The OpenCL implementation of :class:`evm.backend.Ops`.

Implementing these operations is the whole cost of supporting a backend: the
four magnification pipelines then come from :mod:`evm.backend.generic` without
being written again.

Everything here computes in float32, while the NumPy reference computes in
float64, so results differ in the last few digits. The conformance tests apply
tolerances for that; they are looser than the ones the hand-written CUDA code
is held to, because these kernels are written for portability rather than for
matching a particular vendor's arithmetic.
"""

from __future__ import annotations

import numpy as np

from ..cpu.pyramids import BINOM5, BINOM5_SUM1
from ..cuda.runtime import butter_bandpass_coeffs
from . import runtime
from .array import ClArray

__all__ = ["OpenClOps"]


class OpenClOps:
    """Every primitive the pipelines need, run through OpenCL."""

    name = "opencl"

    def __init__(self) -> None:
        self._filters: dict[tuple[float, ...], ClArray] = {}

    # -- helpers ------------------------------------------------------------

    @property
    def _q(self):
        return runtime.queue()

    @staticmethod
    def _k(name: str):
        """A reusable handle for one kernel; see runtime.kernel()."""
        return runtime.kernel(name)

    def _filter_buffer(self, taps: np.ndarray) -> ClArray:
        """Upload a filter once and keep it; the same few are used repeatedly."""
        key = tuple(float(t) for t in taps)
        if key not in self._filters:
            self._filters[key] = ClArray.from_numpy(
                np.ascontiguousarray(taps, dtype=np.float32)
            )
        return self._filters[key]

    # -- transfer -----------------------------------------------------------

    def from_numpy(self, array: np.ndarray) -> ClArray:
        return ClArray.from_numpy(array)

    def to_numpy(self, array: ClArray) -> np.ndarray:
        return array.numpy()

    # -- colour -------------------------------------------------------------

    def bgr_u8_to_ntsc(self, frames: ClArray) -> ClArray:
        T, H, W, _ = frames.shape
        out = ClArray.empty((T, H, W, 3), np.float32)
        count = T * H * W
        self._k("bgr_u8_to_ntsc")(
            self._q, (count,), None, frames.buffer, out.buffer, np.int32(count)
        )
        return out

    def add_and_quantize(self, frames_ntsc: ClArray, delta: ClArray) -> ClArray:
        T, H, W, _ = frames_ntsc.shape
        out = ClArray.empty((T, H, W, 3), np.uint8)
        count = T * H * W
        self._k("add_and_quantize")(
            self._q,
            (count,),
            None,
            frames_ntsc.buffer,
            delta.buffer,
            out.buffer,
            np.int32(count),
        )
        return out

    # -- spatial ------------------------------------------------------------

    def _corr_dn(self, src: ClArray, taps: np.ndarray) -> ClArray:
        """Filter and halve, rows then columns, as the reference does."""
        T, H, W, C = src.shape
        filt = self._filter_buffer(taps)
        flen = len(taps)

        out_h = (H + 1) // 2
        rows = ClArray.empty((T, out_h, W, C), np.float32)
        self._k("corr_dn_rows")(
            self._q,
            (C, W, T * out_h),
            None,
            src.buffer,
            rows.buffer,
            filt.buffer,
            np.int32(T),
            np.int32(H),
            np.int32(W),
            np.int32(C),
            np.int32(flen),
            np.int32(out_h),
        )

        out_w = (W + 1) // 2
        cols = ClArray.empty((T, out_h, out_w, C), np.float32)
        self._k("corr_dn_cols")(
            self._q,
            (C, out_w, T * out_h),
            None,
            rows.buffer,
            cols.buffer,
            filt.buffer,
            np.int32(T),
            np.int32(out_h),
            np.int32(W),
            np.int32(C),
            np.int32(flen),
            np.int32(out_w),
        )
        return cols

    def blur_dn(self, frames: ClArray, levels: int) -> ClArray:
        out = frames
        for _ in range(levels):
            out = self._corr_dn(out, BINOM5_SUM1)
        return out

    def _up_conv(
        self, src: ClArray, out_h: int, out_w: int, taps: np.ndarray
    ) -> ClArray:
        T, H, W, C = src.shape
        filt = self._filter_buffer(taps)
        flen = len(taps)

        rows = ClArray.empty((T, out_h, W, C), np.float32)
        self._k("up_conv_rows")(
            self._q,
            (C, W, T * out_h),
            None,
            src.buffer,
            rows.buffer,
            filt.buffer,
            np.int32(T),
            np.int32(H),
            np.int32(W),
            np.int32(C),
            np.int32(flen),
            np.int32(out_h),
        )

        cols = ClArray.empty((T, out_h, out_w, C), np.float32)
        self._k("up_conv_cols")(
            self._q,
            (C, out_w, T * out_h),
            None,
            rows.buffer,
            cols.buffer,
            filt.buffer,
            np.int32(T),
            np.int32(out_h),
            np.int32(W),
            np.int32(C),
            np.int32(flen),
            np.int32(out_w),
        )
        return cols

    def build_lpyr(self, frames: ClArray, levels: int) -> list[ClArray]:
        """Each band is what one scale keeps that the next coarser one loses."""
        bands: list[ClArray] = []
        current = frames
        for _ in range(levels - 1):
            T, H, W, C = current.shape
            smaller = self._corr_dn(current, BINOM5)
            back = self._up_conv(smaller, H, W, BINOM5)
            band = ClArray.empty((T, H, W, C), np.float32)
            count = T * H * W * C
            self._k("subtract")(
                self._q,
                (count,),
                None,
                current.buffer,
                back.buffer,
                band.buffer,
                np.int32(count),
            )
            bands.append(band)
            current = smaller
        bands.append(current)
        return bands

    def recon_lpyr(self, bands) -> ClArray:
        bands = list(bands)
        acc = bands[-1]
        for band in reversed(bands[:-1]):
            T, H, W, C = band.shape
            up = self._up_conv(acc, H, W, BINOM5)
            count = T * H * W * C
            self._k("add_into")(
                self._q, (count,), None, up.buffer, band.buffer, np.int32(count)
            )
            acc = up
        return acc

    def upsample_bilinear(self, frames: ClArray, height: int, width: int) -> ClArray:
        T, H, W, C = frames.shape
        out = ClArray.empty((T, height, width, C), np.float32)
        self._k("resize_bilinear")(
            self._q,
            (C, width, T * height),
            None,
            frames.buffer,
            out.buffer,
            np.int32(T),
            np.int32(H),
            np.int32(W),
            np.int32(height),
            np.int32(width),
            np.int32(C),
        )
        return out

    # -- temporal -----------------------------------------------------------

    def _to_series(self, frames: ClArray) -> tuple[ClArray, int, int]:
        T = frames.shape[0]
        N = int(np.prod(frames.shape[1:]))
        return frames.reshape((T, N)), T, N

    def ideal_bandpass(
        self, series: ClArray, fl: float, fh: float, sampling_rate: float
    ) -> ClArray:
        flat, T, N = self._to_series(series)
        matrix = _band_projection_matrix(T, fl, fh, sampling_rate)
        d_matrix = ClArray.from_numpy(matrix)
        out = ClArray.empty((T, N), np.float32)
        self._k("band_project")(
            self._q,
            (N, T),
            None,
            flat.buffer,
            out.buffer,
            d_matrix.buffer,
            np.int32(T),
            np.int32(N),
        )
        return out.reshape(series.shape)

    def butter_bandpass(
        self,
        series: ClArray,
        fl: float,
        fh: float,
        sampling_rate: float,
        order: int = 1,
    ) -> ClArray:
        flat, T, N = self._to_series(series)
        high, low = butter_bandpass_coeffs(fl, fh, sampling_rate, order)
        out = ClArray.empty((T, N), np.float32)
        self._k("butter_bandpass")(
            self._q,
            (N,),
            None,
            flat.buffer,
            out.buffer,
            np.int32(T),
            np.int32(N),
            np.float32(high[0]),
            np.float32(high[1]),
            np.float32(high[2]),
            np.float32(low[0]),
            np.float32(low[1]),
            np.float32(low[2]),
        )
        return out.reshape(series.shape)

    def iir_bandpass(self, series: ClArray, r1: float, r2: float) -> ClArray:
        flat, T, N = self._to_series(series)
        out = ClArray.empty((T, N), np.float32)
        self._k("iir_bandpass")(
            self._q,
            (N,),
            None,
            flat.buffer,
            out.buffer,
            np.int32(T),
            np.int32(N),
            np.float32(r1),
            np.float32(r2),
        )
        return out.reshape(series.shape)

    # -- amplification ------------------------------------------------------

    def apply_gain(
        self, frames: ClArray, gain_y: float, gain_i: float, gain_q: float
    ) -> ClArray:
        count = frames.size // 3
        # Copied on the device: this is called once per pyramid band per
        # frame, so a round trip through host memory here dominates.
        out = frames.copy()
        self._k("apply_gain")(
            self._q,
            (count,),
            None,
            out.buffer,
            np.int32(count),
            np.float32(gain_y),
            np.float32(gain_i),
            np.float32(gain_q),
        )
        return out


def _band_projection_matrix(
    T: int, fl: float, fh: float, sampling_rate: float
) -> np.ndarray:
    """The real matrix that keeps only frequencies strictly inside the band.

    Selecting frequency bins is a linear operation, so the whole filter can be
    written as one T-by-T matrix, built once on the host. Multiplying by it is
    exactly equal to transforming, zeroing the unwanted bins and transforming
    back, and it means this backend needs no Fourier transform library on the
    device — which is what lets the same kernels run on any vendor's hardware.

    The matrix is small: T is the number of frames, so a ten-second clip at
    thirty frames a second gives a 300-by-300 matrix.
    """
    freqs = np.arange(T) / T * sampling_rate
    keep = (freqs > fl) & (freqs < fh)
    basis = np.fft.fft(np.eye(T), axis=0)
    basis[~keep, :] = 0.0
    return np.ascontiguousarray(np.real(np.fft.ifft(basis, axis=0)), dtype=np.float32)

    # -- streaming ----------------------------------------------------------

    def iir_step(self, fast: ClArray, slow: ClArray, current: ClArray,
                 r1: float, r2: float) -> ClArray:
        """Advance both running averages by one frame and return the difference.

        Optional in the operations protocol. A backend without it still works —
        :mod:`evm.stream` falls back to doing this arithmetic on the host — but
        that fallback copies every pyramid band off the device and back on
        every frame, which costs more than the magnification itself. This keeps
        the state where the rest of the work already is.

        ``fast`` and ``slow`` are updated in place.
        """
        count = fast.size
        out = ClArray.empty(fast.shape, np.float32)
        self._k("iir_step")(self._q, (count,), None,
                            fast.buffer, slow.buffer, current.buffer,
                            out.buffer, np.int32(count),
                            np.float32(r1), np.float32(r2))
        return out
