"""The Vulkan implementation of :class:`evm.backend.Ops`.

The same operations as the other backends, run through Vulkan compute shaders.
Implementing these is the whole cost of the backend: the four magnification
pipelines come from :mod:`evm.backend.generic` without being written again.

Everything computes in float32 against a float64 reference, so results differ
in the last few digits; the conformance tests allow for that.
"""

from __future__ import annotations


import numpy as np

from ..cpu.pyramids import BINOM5, BINOM5_SUM1
from ..cuda.runtime import butter_bandpass_coeffs
from . import runtime
from .array import VkArray

__all__ = ["VulkanOps"]


class VulkanOps:
    """Every primitive the pipelines need, run through Vulkan."""

    name = "vulkan"

    def __init__(self) -> None:
        self._filters: dict[tuple[float, ...], VkArray] = {}
        self._matrices: dict[str, VkArray] = {}

    # -- dispatch -----------------------------------------------------------

    def _run(
        self, shader: str, buffers: list[VkArray], push: bytes, threads: int
    ) -> None:
        """Record and submit one shader, waiting for it to finish.

        Each dispatch is submitted on its own and waited for. Batching several
        into one command buffer would be faster and is the obvious thing to do
        if this backend is ever tuned; correctness came first, and every result
        here is compared against the reference.
        """
        ctx = runtime.context()
        vk = ctx.vk

        pipeline, layout, set_layout = ctx.pipeline(shader, len(buffers), len(push))

        descriptor = vk.vkAllocateDescriptorSets(
            ctx.device,
            vk.VkDescriptorSetAllocateInfo(
                descriptorPool=ctx.descriptor_pool(), pSetLayouts=[set_layout]
            ),
        )[0]
        vk.vkUpdateDescriptorSets(
            ctx.device,
            len(buffers),
            [
                vk.VkWriteDescriptorSet(
                    dstSet=descriptor,
                    dstBinding=i,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    # The explicit size rather than the "whole buffer" constant:
                    # that constant is the largest 64-bit value, which the binding
                    # cannot represent.
                    pBufferInfo=[
                        vk.VkDescriptorBufferInfo(
                            buffer=b.buffer, offset=0, range=max(b.nbytes, 4)
                        )
                    ],
                )
                for i, b in enumerate(buffers)
            ],
            0,
            None,
        )

        command = vk.vkAllocateCommandBuffers(
            ctx.device,
            vk.VkCommandBufferAllocateInfo(
                commandPool=ctx.command_pool,
                level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                commandBufferCount=1,
            ),
        )[0]
        vk.vkBeginCommandBuffer(
            command,
            vk.VkCommandBufferBeginInfo(
                flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
            ),
        )
        vk.vkCmdBindPipeline(command, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline)
        vk.vkCmdBindDescriptorSets(
            command,
            vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            layout,
            0,
            1,
            [descriptor],
            0,
            None,
        )
        if push:
            from vulkan import ffi

            vk.vkCmdPushConstants(
                command,
                layout,
                vk.VK_SHADER_STAGE_COMPUTE_BIT,
                0,
                len(push),
                ffi.from_buffer(push),
            )
        groups = max((threads + 63) // 64, 1)
        vk.vkCmdDispatch(command, groups, 1, 1)
        vk.vkEndCommandBuffer(command)

        vk.vkQueueSubmit(
            ctx.queue,
            1,
            [vk.VkSubmitInfo(pCommandBuffers=[command])],
            vk.VK_NULL_HANDLE,
        )
        vk.vkQueueWaitIdle(ctx.queue)
        vk.vkFreeCommandBuffers(ctx.device, ctx.command_pool, 1, [command])
        ctx.reset_descriptors()

    # -- shared data --------------------------------------------------------

    def _filter_buffer(self, taps: np.ndarray) -> VkArray:
        key = tuple(float(t) for t in taps)
        if key not in self._filters:
            self._filters[key] = VkArray.from_numpy(
                np.ascontiguousarray(taps, dtype=np.float32)
            )
        return self._filters[key]

    def _matrix(self, which: str) -> VkArray:
        """The colour matrix, taken from the one place it is written down."""
        if which not in self._matrices:
            from ..io.video import rgb_to_yiq, yiq_to_rgb

            identity = np.eye(3, dtype=np.float32)
            source = rgb_to_yiq if which == "forward" else yiq_to_rgb
            self._matrices[which] = VkArray.from_numpy(
                np.ascontiguousarray(source(identity).T, dtype=np.float32)
            )
        return self._matrices[which]

    # -- transfer -----------------------------------------------------------

    def from_numpy(self, array: np.ndarray) -> VkArray:
        return VkArray.from_numpy(array)

    def to_numpy(self, array: VkArray) -> np.ndarray:
        return array.numpy()

    # -- colour -------------------------------------------------------------

    def bgr_u8_to_ntsc(self, frames: VkArray) -> VkArray:
        T, H, W, _ = frames.shape
        out = VkArray.empty((T, H, W, 3), np.float32)
        count = T * H * W
        self._run(
            "bgr_u8_to_ntsc",
            [frames, out, self._matrix("forward")],
            np.array([count], dtype=np.int32).tobytes(),
            count,
        )
        return out

    def add_and_quantize(self, frames_ntsc: VkArray, delta: VkArray) -> VkArray:
        T, H, W, _ = frames_ntsc.shape
        count = T * H * W
        # Zeroed, because the shader builds each 32-bit word by combining the
        # bytes written into it rather than overwriting the word.
        out = VkArray.zeros((T, H, W, 3), np.uint8)
        self._run(
            "add_and_quantize",
            [frames_ntsc, delta, out, self._matrix("inverse")],
            np.array([count], dtype=np.int32).tobytes(),
            count,
        )
        return out

    # -- spatial ------------------------------------------------------------

    @staticmethod
    def _spatial_push(
        T: int, H: int, W: int, C: int, flen: int, out_h: int, out_w: int
    ) -> bytes:
        return np.array([T, H, W, C, flen, out_h, out_w], dtype=np.int32).tobytes()

    def _corr_dn(self, src: VkArray, taps: np.ndarray) -> VkArray:
        T, H, W, C = src.shape
        filt = self._filter_buffer(taps)
        flen = len(taps)

        out_h = (H + 1) // 2
        rows = VkArray.empty((T, out_h, W, C), np.float32)
        self._run(
            "corr_dn_rows",
            [src, rows, filt],
            self._spatial_push(T, H, W, C, flen, out_h, 0),
            C * W * T * out_h,
        )

        out_w = (W + 1) // 2
        cols = VkArray.empty((T, out_h, out_w, C), np.float32)
        self._run(
            "corr_dn_cols",
            [rows, cols, filt],
            self._spatial_push(T, out_h, W, C, flen, 0, out_w),
            C * out_w * T * out_h,
        )
        return cols

    def blur_dn(self, frames: VkArray, levels: int) -> VkArray:
        out = frames
        for _ in range(levels):
            out = self._corr_dn(out, BINOM5_SUM1)
        return out

    def _up_conv(
        self, src: VkArray, out_h: int, out_w: int, taps: np.ndarray
    ) -> VkArray:
        T, H, W, C = src.shape
        filt = self._filter_buffer(taps)
        flen = len(taps)

        rows = VkArray.empty((T, out_h, W, C), np.float32)
        self._run(
            "up_conv_rows",
            [src, rows, filt],
            self._spatial_push(T, H, W, C, flen, out_h, 0),
            C * W * T * out_h,
        )

        cols = VkArray.empty((T, out_h, out_w, C), np.float32)
        self._run(
            "up_conv_cols",
            [rows, cols, filt],
            self._spatial_push(T, out_h, W, C, flen, 0, out_w),
            C * out_w * T * out_h,
        )
        return cols

    def build_lpyr(self, frames: VkArray, levels: int) -> list[VkArray]:
        bands: list[VkArray] = []
        current = frames
        for _ in range(levels - 1):
            T, H, W, C = current.shape
            smaller = self._corr_dn(current, BINOM5)
            back = self._up_conv(smaller, H, W, BINOM5)
            band = VkArray.empty((T, H, W, C), np.float32)
            count = T * H * W * C
            self._run(
                "subtract",
                [current, back, band],
                np.array([count], dtype=np.int32).tobytes(),
                count,
            )
            bands.append(band)
            current = smaller
        bands.append(current)
        return bands

    def recon_lpyr(self, bands) -> VkArray:
        bands = list(bands)
        acc = bands[-1]
        for band in reversed(bands[:-1]):
            T, H, W, C = band.shape
            up = self._up_conv(acc, H, W, BINOM5)
            count = T * H * W * C
            self._run(
                "add_into",
                [up, band],
                np.array([count], dtype=np.int32).tobytes(),
                count,
            )
            acc = up
        return acc

    def upsample_bilinear(self, frames: VkArray, height: int, width: int) -> VkArray:
        T, H, W, C = frames.shape
        out = VkArray.empty((T, height, width, C), np.float32)
        push = np.array([T, H, W, height, width, C], dtype=np.int32).tobytes()
        self._run("resize_bilinear", [frames, out], push, C * width * T * height)
        return out

    # -- temporal -----------------------------------------------------------

    def _to_series(self, frames: VkArray) -> tuple[VkArray, int, int]:
        T = frames.shape[0]
        N = int(np.prod(frames.shape[1:]))
        return frames.reshape((T, N)), T, N

    def ideal_bandpass(
        self, series: VkArray, fl: float, fh: float, sampling_rate: float
    ) -> VkArray:
        from ..opencl.ops import _band_projection_matrix

        flat, T, N = self._to_series(series)
        matrix = VkArray.from_numpy(_band_projection_matrix(T, fl, fh, sampling_rate))
        out = VkArray.empty((T, N), np.float32)
        self._run(
            "band_project",
            [flat, out, matrix],
            np.array([T, N], dtype=np.int32).tobytes(),
            N * T,
        )
        return out.reshape(series.shape)

    def butter_bandpass(
        self,
        series: VkArray,
        fl: float,
        fh: float,
        sampling_rate: float,
        order: int = 1,
    ) -> VkArray:
        flat, T, N = self._to_series(series)
        high, low = butter_bandpass_coeffs(fl, fh, sampling_rate, order)
        out = VkArray.empty((T, N), np.float32)
        push = (
            np.array([T, N], dtype=np.int32).tobytes()
            + np.array(
                [high[0], high[1], high[2], low[0], low[1], low[2]], dtype=np.float32
            ).tobytes()
        )
        self._run("butter_bandpass", [flat, out], push, N)
        return out.reshape(series.shape)

    def iir_bandpass(self, series: VkArray, r1: float, r2: float) -> VkArray:
        flat, T, N = self._to_series(series)
        out = VkArray.empty((T, N), np.float32)
        push = (
            np.array([T, N], dtype=np.int32).tobytes()
            + np.array([r1, r2], dtype=np.float32).tobytes()
        )
        self._run("iir_bandpass", [flat, out], push, N)
        return out.reshape(series.shape)

    # -- amplification and streaming ----------------------------------------

    def apply_gain(
        self, frames: VkArray, gain_y: float, gain_i: float, gain_q: float
    ) -> VkArray:
        count = frames.size // 3
        out = frames.copy()
        push = (
            np.array([count], dtype=np.int32).tobytes()
            + np.array([gain_y, gain_i, gain_q], dtype=np.float32).tobytes()
        )
        self._run("apply_gain", [out], push, count)
        return out

    def iir_step(
        self, fast: VkArray, slow: VkArray, current: VkArray, r1: float, r2: float
    ) -> VkArray:
        count = fast.size
        out = VkArray.empty(fast.shape, np.float32)
        push = (
            np.array([count], dtype=np.int32).tobytes()
            + np.array([r1, r2], dtype=np.float32).tobytes()
        )
        self._run("iir_step", [fast, slow, current, out], push, count)
        return out
