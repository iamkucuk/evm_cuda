"""The Metal implementation of :class:`vidmag.backend.Ops`.

The same operations as the OpenCL backend, dispatched through Apple's own
interface. Implementing these is the whole cost of the backend: the four
magnification pipelines come from :mod:`vidmag.backend.generic` without being
written again.

Everything computes in float32 against a float64 reference, so results differ
in the last few digits; the conformance tests allow for that.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..cpu.pyramids import BINOM5, BINOM5_SUM1
from ..cuda.runtime import butter_bandpass_coeffs
from . import runtime
from .array import MetalArray

__all__ = ["MetalOps"]


def _shape_struct(
    T: int, H: int, W: int, C: int, flen: int, out_h: int, out_w: int
) -> bytes:
    """The `Shape` struct the spatial kernels take, as raw bytes."""
    return np.array([T, H, W, C, flen, out_h, out_w], dtype=np.int32).tobytes()


class MetalOps:
    """Every primitive the pipelines need, run through Metal."""

    name = "metal"

    def __init__(self) -> None:
        self._filters: dict[tuple[float, ...], MetalArray] = {}
        self._matrices: dict[str, bytes] | None = None

    # -- dispatch helpers ---------------------------------------------------

    def _run(
        self,
        kernel: str,
        buffers: list[Any],
        grid: tuple[int, ...],
    ) -> None:
        """Encode and run one kernel, waiting for it to finish.

        The work is encoded but not submitted. It is submitted, and waited
        for, the first time a result is read back — see runtime.flush(). That
        keeps a pipeline of several dozen small kernels to one round trip to
        the hardware instead of several dozen, which on this hardware is worth
        more than twice the total time.
        """
        import Metal

        state = runtime.pipeline(kernel)
        encoder = runtime.encoder()
        encoder.setComputePipelineState_(state)

        for index, item in enumerate(buffers):
            if isinstance(item, MetalArray):
                encoder.setBuffer_offset_atIndex_(item.buffer, 0, index)
            else:
                encoder.setBytes_length_atIndex_(item, len(item), index)

        width = state.threadExecutionWidth()
        if len(grid) == 1:
            per_group = Metal.MTLSizeMake(min(width, max(grid[0], 1)), 1, 1)
            total = Metal.MTLSizeMake(max(grid[0], 1), 1, 1)
        elif len(grid) == 2:
            per_group = Metal.MTLSizeMake(1, min(width, max(grid[1], 1)), 1)
            total = Metal.MTLSizeMake(max(grid[0], 1), max(grid[1], 1), 1)
        else:
            per_group = Metal.MTLSizeMake(1, 1, min(width, max(grid[2], 1)))
            total = Metal.MTLSizeMake(max(grid[0], 1), max(grid[1], 1), max(grid[2], 1))

        encoder.dispatchThreads_threadsPerThreadgroup_(total, per_group)
        encoder.endEncoding()

    def _filter_buffer(self, taps: np.ndarray) -> MetalArray:
        key = tuple(float(t) for t in taps)
        if key not in self._filters:
            self._filters[key] = MetalArray.from_numpy(
                np.ascontiguousarray(taps, dtype=np.float32)
            )
        return self._filters[key]

    def _colour_matrices(self) -> dict[str, bytes]:
        """The forward and inverse colour matrices, as raw bytes.

        Taken from :mod:`vidmag.io.video` rather than written into the kernel, so
        the numbers the whole project's agreement with the reference rests on
        exist in exactly one place.
        """
        if self._matrices is None:
            from ..io.video import rgb_to_yiq, yiq_to_rgb

            identity = np.eye(3, dtype=np.float32)
            self._matrices = {
                "forward": np.ascontiguousarray(
                    rgb_to_yiq(identity).T, dtype=np.float32
                ).tobytes(),
                "inverse": np.ascontiguousarray(
                    yiq_to_rgb(identity).T, dtype=np.float32
                ).tobytes(),
            }
        return self._matrices

    # -- transfer -----------------------------------------------------------

    def from_numpy(self, array: np.ndarray) -> MetalArray:
        return MetalArray.from_numpy(array)

    def to_numpy(self, array: MetalArray) -> np.ndarray:
        return array.numpy()

    # -- colour -------------------------------------------------------------

    def bgr_u8_to_ntsc(self, frames: MetalArray) -> MetalArray:
        T, H, W, _ = frames.shape
        out = MetalArray.empty((T, H, W, 3), np.float32)
        count = T * H * W
        self._run(
            "bgr_u8_to_ntsc",
            [
                frames,
                out,
                self._colour_matrices()["forward"],
                np.int32(count).tobytes(),
            ],
            (count,),
        )
        return out

    def add_and_quantize(
        self, frames_ntsc: MetalArray, delta: MetalArray
    ) -> MetalArray:
        T, H, W, _ = frames_ntsc.shape
        out = MetalArray.empty((T, H, W, 3), np.uint8)
        count = T * H * W
        self._run(
            "add_and_quantize",
            [
                frames_ntsc,
                delta,
                out,
                self._colour_matrices()["inverse"],
                np.int32(count).tobytes(),
            ],
            (count,),
        )
        return out

    # -- spatial ------------------------------------------------------------

    def _corr_dn(self, src: MetalArray, taps: np.ndarray) -> MetalArray:
        T, H, W, C = src.shape
        filt = self._filter_buffer(taps)
        flen = len(taps)

        out_h = (H + 1) // 2
        rows = MetalArray.empty((T, out_h, W, C), np.float32)
        self._run(
            "corr_dn_rows",
            [src, rows, filt, _shape_struct(T, H, W, C, flen, out_h, 0)],
            (C, W, T * out_h),
        )

        out_w = (W + 1) // 2
        cols = MetalArray.empty((T, out_h, out_w, C), np.float32)
        self._run(
            "corr_dn_cols",
            [rows, cols, filt, _shape_struct(T, out_h, W, C, flen, 0, out_w)],
            (C, out_w, T * out_h),
        )
        return cols

    def blur_dn(self, frames: MetalArray, levels: int) -> MetalArray:
        out = frames
        for _ in range(levels):
            out = self._corr_dn(out, BINOM5_SUM1)
        return out

    def _up_conv(
        self, src: MetalArray, out_h: int, out_w: int, taps: np.ndarray
    ) -> MetalArray:
        T, H, W, C = src.shape
        filt = self._filter_buffer(taps)
        flen = len(taps)

        rows = MetalArray.empty((T, out_h, W, C), np.float32)
        self._run(
            "up_conv_rows",
            [src, rows, filt, _shape_struct(T, H, W, C, flen, out_h, 0)],
            (C, W, T * out_h),
        )

        cols = MetalArray.empty((T, out_h, out_w, C), np.float32)
        self._run(
            "up_conv_cols",
            [rows, cols, filt, _shape_struct(T, out_h, W, C, flen, 0, out_w)],
            (C, out_w, T * out_h),
        )
        return cols

    def build_lpyr(self, frames: MetalArray, levels: int) -> list[MetalArray]:
        bands: list[MetalArray] = []
        current = frames
        for _ in range(levels - 1):
            T, H, W, C = current.shape
            smaller = self._corr_dn(current, BINOM5)
            back = self._up_conv(smaller, H, W, BINOM5)
            band = MetalArray.empty((T, H, W, C), np.float32)
            count = T * H * W * C
            self._run(
                "subtract", [current, back, band, np.int32(count).tobytes()], (count,)
            )
            bands.append(band)
            current = smaller
        bands.append(current)
        return bands

    def recon_lpyr(self, bands) -> MetalArray:
        bands = list(bands)
        acc = bands[-1]
        for band in reversed(bands[:-1]):
            T, H, W, C = band.shape
            up = self._up_conv(acc, H, W, BINOM5)
            count = T * H * W * C
            self._run("add_into", [up, band, np.int32(count).tobytes()], (count,))
            acc = up
        return acc

    def upsample_bilinear(
        self, frames: MetalArray, height: int, width: int
    ) -> MetalArray:
        T, H, W, C = frames.shape
        out = MetalArray.empty((T, height, width, C), np.float32)
        shape = np.array([T, H, W, height, width, C], dtype=np.int32).tobytes()
        self._run("resize_bilinear", [frames, out, shape], (C, width, T * height))
        return out

    # -- temporal -----------------------------------------------------------

    def _to_series(self, frames: MetalArray) -> tuple[MetalArray, int, int]:
        T = frames.shape[0]
        N = int(np.prod(frames.shape[1:]))
        return frames.reshape((T, N)), T, N

    def ideal_bandpass(
        self, series: MetalArray, fl: float, fh: float, sampling_rate: float
    ) -> MetalArray:
        from ..opencl.ops import _band_projection_matrix

        flat, T, N = self._to_series(series)
        matrix = MetalArray.from_numpy(
            _band_projection_matrix(T, fl, fh, sampling_rate)
        )
        out = MetalArray.empty((T, N), np.float32)
        self._run(
            "band_project",
            [flat, out, matrix, np.array([T, N], dtype=np.int32).tobytes()],
            (N, T),
        )
        return out.reshape(series.shape)

    def butter_bandpass(
        self,
        series: MetalArray,
        fl: float,
        fh: float,
        sampling_rate: float,
        order: int = 1,
    ) -> MetalArray:
        flat, T, N = self._to_series(series)
        high, low = butter_bandpass_coeffs(fl, fh, sampling_rate, order)
        out = MetalArray.empty((T, N), np.float32)
        coefficients = np.array(
            [high[0], high[1], high[2], low[0], low[1], low[2]], dtype=np.float32
        ).tobytes()
        self._run(
            "butter_bandpass",
            [flat, out, np.array([T, N], dtype=np.int32).tobytes(), coefficients],
            (N,),
        )
        return out.reshape(series.shape)

    def iir_bandpass(self, series: MetalArray, r1: float, r2: float) -> MetalArray:
        flat, T, N = self._to_series(series)
        out = MetalArray.empty((T, N), np.float32)
        self._run(
            "iir_bandpass",
            [
                flat,
                out,
                np.array([T, N], dtype=np.int32).tobytes(),
                np.array([r1, r2], dtype=np.float32).tobytes(),
            ],
            (N,),
        )
        return out.reshape(series.shape)

    # -- amplification and streaming ----------------------------------------

    def apply_gain(
        self, frames: MetalArray, gain_y: float, gain_i: float, gain_q: float
    ) -> MetalArray:
        count = frames.size // 3
        out = frames.copy()
        # float3 is padded to four floats in Metal's layout rules; writing four
        # keeps what the kernel reads aligned with what is sent.
        gains = np.array([gain_y, gain_i, gain_q, 0.0], dtype=np.float32).tobytes()
        self._run("apply_gain", [out, np.int32(count).tobytes(), gains], (count,))
        return out

    def iir_step(
        self,
        fast: MetalArray,
        slow: MetalArray,
        current: MetalArray,
        r1: float,
        r2: float,
    ) -> MetalArray:
        count = fast.size
        out = MetalArray.empty(fast.shape, np.float32)
        self._run(
            "iir_step",
            [
                fast,
                slow,
                current,
                out,
                np.int32(count).tobytes(),
                np.array([r1, r2], dtype=np.float32).tobytes(),
            ],
            (count,),
        )
        return out
