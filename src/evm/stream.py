"""Magnify a live feed, one frame at a time.

Every pipeline elsewhere in this library takes a whole clip. That is not a
limitation of the code but of one of the filters: selecting frequencies with a
Fourier transform needs all of time at once, so it can only run when all of time
exists. A camera does not offer that.

The other two filters can. Both run forward in time, carrying a little state
from one frame to the next, so a frame can be magnified as it arrives using
only what has already been seen. That is what this module does.

What it produces is identical to feeding the same frames to the batch pipeline —
not similar, identical, and :mod:`tests.test_streaming` asserts it. That
equality is the whole argument for trusting it: a streaming path that drifts
from the batch path would be a second implementation to verify separately.

    from evm.stream import MotionStream

    stream = MotionStream(height=480, width=640, alpha=10, lambda_c=16)
    for frame in camera:
        display(stream.push(frame))
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .backend import registry
from .cpu.magnify import EXAGGERATION_FACTOR, figure6_alpha_schedule
from .cpu.pyramids import BINOM5, max_pyr_ht

__all__ = ["MotionStream"]


class MotionStream:
    """Amplify motion in frames as they arrive.

    Holds the filter's state between frames, so each call needs only the new
    frame. Memory use does not grow with the length of the feed.

    Args:
        height: frame height in pixels. Fixed for the life of the stream.
        width: frame width in pixels.
        alpha: how much to amplify.
        lambda_c: spatial cutoff in pixels; detail finer than this is amplified
            progressively less. Raise it if the output shimmers.
        r1: the faster of the two decay rates.
        r2: the slower. Must be smaller than ``r1``.
        chrom_attenuation: how much to amplify colour relative to brightness.
        backend: which implementation to use. Defaults to the processor, which
            is measured to be the fastest choice here, and is not what
            ``"auto"`` would pick. The reason is that magnifying one frame at a
            time launches a few dozen small pieces of work on the graphics
            processor, and the cost of launching them does not shrink with the
            frame. Measured on an Apple M2 Max: at 320x240 the processor
            manages 56 frames a second against 6 through OpenCL, and at 640x480
            21 against 5. The graphics backends are far faster on a whole clip,
            where that cost is paid once instead of once per frame. Pass
            ``"auto"`` or a name to override.

    The first frame comes back unchanged. That is not a special case bolted on:
    the two running averages both start at the first frame's value, so their
    difference is zero, and the batch pipeline does exactly the same thing.
    """

    def __init__(
        self,
        height: int,
        width: int,
        *,
        alpha: float = 10.0,
        lambda_c: float = 16.0,
        r1: float = 0.4,
        r2: float = 0.05,
        chrom_attenuation: float = 0.1,
        exaggeration_factor: float = EXAGGERATION_FACTOR,
        backend: str = "auto",
    ) -> None:
        if r1 <= r2:
            raise ValueError(
                f"require r1 > r2 (a faster average minus a slower one); "
                f"got r1={r1}, r2={r2}"
            )
        self.height = int(height)
        self.width = int(width)
        self.r1 = float(r1)
        self.r2 = float(r2)
        self.chrom_attenuation = float(chrom_attenuation)

        if backend == "auto":
            backend = _fastest_streaming_backend()
        self.backend_name, self._impl = registry.select(backend)
        self._ops = _operations_for(self.backend_name, self._impl)

        self.levels = max_pyr_ht((self.height, self.width), len(BINOM5)) + 1
        self.schedule = figure6_alpha_schedule(
            self.levels,
            alpha,
            lambda_c,
            self.height,
            self.width,
            exaggeration_factor=exaggeration_factor,
        )

        # One pair of running averages per pyramid band, created on the first
        # frame when the band shapes are known.
        self._fast: list[Any] | None = None
        self._slow: list[Any] | None = None
        self.frames_seen = 0

    def push(self, frame: np.ndarray) -> np.ndarray:
        """Magnify one frame and return it.

        Args:
            frame: ``(height, width, 3)`` uint8, blue-green-red — the order
                cameras and video files decode into.

        Returns:
            The magnified frame, same shape and dtype.
        """
        frame = np.ascontiguousarray(frame)
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(
                f"frame is {frame.shape}, but this stream was created for "
                f"{(self.height, self.width, 3)}; make a new stream to change size"
            )
        if frame.dtype != np.uint8:
            raise TypeError(f"frame must be uint8, got {frame.dtype}")

        ops = self._ops
        # The operations work on a batch; a single frame is a batch of one.
        batch = ops.from_numpy(frame[None, ...])
        ntsc = ops.bgr_u8_to_ntsc(batch)
        bands = ops.build_lpyr(ntsc, self.levels)

        if self._fast is None:
            # First frame: both averages start at it, so their difference is
            # zero and nothing is amplified. Copies, not references — the two
            # must be able to diverge.
            self._fast = [_copy(ops, b) for b in bands]
            self._slow = [_copy(ops, b) for b in bands]

        amplified = []
        for index, band in enumerate(bands):
            difference = self._advance(index, band)
            level_alpha = float(self.schedule[index])
            amplified.append(
                ops.apply_gain(difference, level_alpha, level_alpha, level_alpha)
            )

        delta = ops.recon_lpyr(amplified)
        if self.chrom_attenuation != 1.0:
            delta = ops.apply_gain(
                delta, 1.0, self.chrom_attenuation, self.chrom_attenuation
            )
        out = ops.to_numpy(ops.add_and_quantize(ntsc, delta))
        self.frames_seen += 1
        return np.asarray(out[0])

    def _advance(self, index: int, band: Any) -> Any:
        """One step of the two running averages for one pyramid band.

        A backend that provides ``iir_step`` does this without leaving the
        device. Without it, the arithmetic happens on the host, which means
        copying the band off the device and back on every frame — correct, but
        on a graphics processor that copying costs several times more than the
        magnification itself. Measured on an Apple M2 Max at 960x544: 5 frames
        a second through the fallback, against 13 on the processor alone.
        """
        assert self._fast is not None and self._slow is not None, (
            "the running averages are created on the first frame; reaching "
            "here without them means push() called this out of order"
        )
        step = getattr(self._ops, "iir_step", None)
        if step is not None:
            return step(self._fast[index], self._slow[index], band, self.r1, self.r2)

        fast = _blend(self._ops, self._fast[index], band, self.r1)
        slow = _blend(self._ops, self._slow[index], band, self.r2)
        self._fast[index] = fast
        self._slow[index] = slow
        return _subtract(self._ops, fast, slow)

    def reset(self) -> None:
        """Forget everything seen so far, as if the stream were new."""
        self._fast = None
        self._slow = None
        self.frames_seen = 0

    def __repr__(self) -> str:
        return (
            f"MotionStream({self.height}x{self.width}, "
            f"backend={self.backend_name!r}, levels={self.levels}, "
            f"frames_seen={self.frames_seen})"
        )


# ---------------------------------------------------------------------------
# The three pieces of arithmetic the running averages need
# ---------------------------------------------------------------------------
#
# The operations protocol has no general elementwise arithmetic, and adding it
# for this would widen every backend's obligations. These three helpers work on
# whatever the backend's arrays are: on the processor that is a NumPy array and
# they are one line each; on a device they fall back to a copy through host
# memory. The fallback is correct everywhere and slow on a device, which is why
# the docstring below says what a fast implementation would need.


def _copy(ops: Any, array: Any) -> Any:
    if isinstance(array, np.ndarray):
        return array.copy()
    return ops.from_numpy(ops.to_numpy(array).copy())


def _blend(ops: Any, previous: Any, current: Any, rate: float) -> Any:
    """``previous * (1 - rate) + current * rate``, one step of a running average."""
    if isinstance(previous, np.ndarray):
        return previous * (1.0 - rate) + current * rate
    return ops.from_numpy(
        ops.to_numpy(previous) * (1.0 - rate) + ops.to_numpy(current) * rate
    )


def _subtract(ops: Any, left: Any, right: Any) -> Any:
    if isinstance(left, np.ndarray):
        return left - right
    return ops.from_numpy(ops.to_numpy(left) - ops.to_numpy(right))


def _fastest_streaming_backend() -> str:
    """The first backend in preference order that can actually stream.

    Plain ``select("auto")`` is wrong here. It walks the same order the
    whole-clip pipelines use, which puts the hand-written NVIDIA backend first
    — and that backend implements the four whole-clip pipelines but not the
    frame-at-a-time operations, so on an NVIDIA machine automatic selection
    would choose a backend that cannot do this at all.

    Skipping those leaves the ordinary preference order, which is what this
    walks. Until 2026-08-11 the default was the processor instead, on the
    stated grounds that launching many small pieces of work per frame costs
    more than it saves. Measured on an Apple M2 Max that is not so, at either
    size tried: 227.6 frames per second on Apple's interface against 57.9 on
    the processor at 320x240, and 58.8 against 8.1 at 720p.
    """
    for info in registry.list_backends():
        if info.available and info.capabilities.streaming:
            return info.name
    # Unreachable in practice: the NumPy baseline is always available and
    # streams. Naming it explicitly beats returning None and failing later.
    return "cpu"


def _operations_for(name: str, impl: Any) -> Any:
    """The primitive operations for a selected backend.

    A backend registered as a set of primitives already carries them. The CPU
    baseline is registered as its pipeline module, so its primitives are
    fetched separately.
    """
    from .backend.ops import Ops

    # Checking the interface, not the attribute name. `hasattr(impl, "ops")`
    # alone is not enough: evm.cuda is a package with a submodule called `ops`,
    # so it passed that test and then failed several calls later with
    # AttributeError: module 'evm.cuda.ops' has no attribute 'from_numpy'.
    # A backend that cannot stream should say so here, in one sentence, rather
    # than partway through the first frame.
    candidate = getattr(impl, "ops", None)
    if candidate is not None and isinstance(candidate, Ops):
        return candidate
    if name == "cpu":
        from .cpu.backend import OPS

        return OPS
    missing = [
        m
        for m in ("from_numpy", "bgr_u8_to_ntsc", "build_lpyr", "recon_lpyr")
        if not callable(getattr(candidate, m, None))
    ]
    raise NotImplementedError(
        f"the {name!r} backend cannot stream: it does not implement the "
        f"primitive operations streaming is built from"
        + (f" (missing {', '.join(missing)})" if missing else "")
        + ". Backends that can stream here: "
        + ", ".join(
            i.name
            for i in __import__(
                "evm.backend", fromlist=["registry"]
            ).registry.list_backends()
            if i.available and i.name != name
        )
        + "."
    )
