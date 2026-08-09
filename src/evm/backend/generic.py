"""The four magnification pipelines, written once against :class:`Ops`.

A backend only has to supply the primitive operations in
:mod:`evm.backend.ops` — colour conversion, blur and downsample, pyramid build
and reconstruct, three temporal filters, gain, upsample, quantize. The four
pipelines then come from here for free, which is what makes adding hardware
support a bounded job rather than a rewrite.

A backend may still replace any of these with its own version, and the two that
matter for speed do: the hand-written CUDA code fuses stages and collapses
kernel launches, which nothing expressed as a sequence of separate operations
can match. The functions here are the correct-by-construction fallback, and the
thing every new backend is measured against before it earns an override.

The arithmetic is the same as :mod:`evm.cpu.magnify`; only the spelling
differs, because here every step goes through the protocol rather than calling
NumPy directly.
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np

from ..cpu.magnify import EXAGGERATION_FACTOR, figure6_alpha_schedule
from ..cpu.pyramids import BINOM5, max_pyr_ht

__all__ = [
    "color_gdown_ideal_core",
    "motion_lpyr_ideal_core",
    "motion_lpyr_butter_core",
    "motion_lpyr_iir_core",
    "bind",
]


def _resolve_rate(fps: float, sampling_rate: float | None) -> float:
    return fps if sampling_rate is None else sampling_rate


def color_gdown_ideal_core(
    ops: Any,
    frames_bgr_u8: np.ndarray,
    fps: float,
    *,
    alpha: float,
    level: int,
    fl: float,
    fh: float,
    chrom_attenuation: float = 1.0,
    sampling_rate: float | None = None,
) -> np.ndarray:
    """Amplify colour change: blur down, bandpass over time, scale, add back.

    This is the pipeline that makes a pulse visible. Reducing the resolution
    first is what suppresses noise: a heartbeat changes a whole region of skin
    together, so averaging over a region keeps the signal and discards most of
    what is random.
    """
    rate = _resolve_rate(fps, sampling_rate)
    frames = ops.from_numpy(np.ascontiguousarray(frames_bgr_u8))

    ntsc = ops.bgr_u8_to_ntsc(frames)
    small = ops.blur_dn(ntsc, level)
    filtered = ops.ideal_bandpass(small, fl, fh, rate)
    amplified = ops.apply_gain(filtered, alpha,
                               alpha * chrom_attenuation,
                               alpha * chrom_attenuation)

    _, height, width, _ = ntsc.shape
    delta = ops.upsample_bilinear(amplified, height, width)
    out: np.ndarray = ops.to_numpy(ops.add_and_quantize(ntsc, delta))
    return out


def _motion_core(
    ops: Any,
    frames_bgr_u8: np.ndarray,
    fps: float,
    *,
    alpha: float,
    lambda_c: float,
    chrom_attenuation: float,
    exaggeration_factor: float,
    filter_band: Any,
) -> np.ndarray:
    """Shared body of the three motion pipelines.

    They differ only in which temporal filter runs on each pyramid band, so
    that one step arrives as ``filter_band``. Everything else — building the
    pyramid, the per-level amplification schedule from the reference paper's
    Figure 6, reconstruction, rendering — is identical, and duplicating it
    three times would mean three places for the schedule to drift.
    """
    frames = ops.from_numpy(np.ascontiguousarray(frames_bgr_u8))
    ntsc = ops.bgr_u8_to_ntsc(frames)
    _, height, width, _ = ntsc.shape

    # The reference builds the tallest pyramid the frame allows, which is one
    # more level than the deepest usable downsample. Getting this from the same
    # helper the reference uses, rather than recomputing it from the frame
    # size, is what keeps the two in step: an earlier version of this file used
    # its own formula and produced one level fewer, which changed the
    # per-level amplification and the output with it.
    levels = max_pyr_ht((height, width), len(BINOM5)) + 1
    bands = ops.build_lpyr(ntsc, levels)

    schedule = figure6_alpha_schedule(
        levels, alpha, lambda_c, height, width,
        exaggeration_factor=exaggeration_factor,
    )

    amplified = []
    for band, level_alpha in zip(bands, schedule):
        filtered = filter_band(band)
        amplified.append(ops.apply_gain(filtered, level_alpha, level_alpha,
                                        level_alpha))

    delta = ops.recon_lpyr(amplified)
    if chrom_attenuation != 1.0:
        delta = ops.apply_gain(delta, 1.0, chrom_attenuation, chrom_attenuation)
    out: np.ndarray = ops.to_numpy(ops.add_and_quantize(ntsc, delta))
    return out


def motion_lpyr_ideal_core(
    ops: Any,
    frames_bgr_u8: np.ndarray,
    fps: float,
    *,
    alpha: float,
    lambda_c: float,
    fl: float,
    fh: float,
    chrom_attenuation: float = 0.0,
    sampling_rate: float | None = None,
    exaggeration_factor: float = EXAGGERATION_FACTOR,
) -> np.ndarray:
    """Amplify motion, selecting the frequency band with a Fourier transform.

    Needs the whole clip, because the transform runs over all of time at once.
    """
    rate = _resolve_rate(fps, sampling_rate)
    return _motion_core(
        ops, frames_bgr_u8, fps, alpha=alpha, lambda_c=lambda_c,
        chrom_attenuation=chrom_attenuation,
        exaggeration_factor=exaggeration_factor,
        filter_band=lambda band: ops.ideal_bandpass(band, fl, fh, rate),
    )


def motion_lpyr_butter_core(
    ops: Any,
    frames_bgr_u8: np.ndarray,
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
) -> np.ndarray:
    """Amplify motion, selecting the band with a Butterworth filter.

    Runs forward in time only, so it also works on frames as they arrive.
    """
    rate = _resolve_rate(fps, sampling_rate)
    return _motion_core(
        ops, frames_bgr_u8, fps, alpha=alpha, lambda_c=lambda_c,
        chrom_attenuation=chrom_attenuation,
        exaggeration_factor=exaggeration_factor,
        filter_band=lambda band: ops.butter_bandpass(band, fl, fh, rate, order),
    )


def motion_lpyr_iir_core(
    ops: Any,
    frames_bgr_u8: np.ndarray,
    fps: float,
    *,
    alpha: float,
    lambda_c: float,
    r1: float,
    r2: float,
    chrom_attenuation: float = 0.1,
    exaggeration_factor: float = EXAGGERATION_FACTOR,
) -> np.ndarray:
    """Amplify motion, selecting the band by subtracting two running averages.

    The cheapest of the three and the only one that needs no history beyond the
    previous frame, which is what makes it the one a live stream can use.
    """
    return _motion_core(
        ops, frames_bgr_u8, fps, alpha=alpha, lambda_c=lambda_c,
        chrom_attenuation=chrom_attenuation,
        exaggeration_factor=exaggeration_factor,
        filter_band=lambda band: ops.iir_bandpass(band, r1, r2),
    )


class _BoundPipelines:
    """The four pipelines, with a particular set of operations already chosen.

    The registry hands the facade an object it can call ``<name>_core`` on. A
    backend that implements the whole :class:`Pipelines` protocol itself — the
    hand-written CUDA one, whose speed comes from fusing stages — is already
    such an object. A backend that supplies only the primitives becomes one by
    passing them through here.

    Each pipeline is bound with :func:`functools.partial` rather than wrapped in
    a method taking ``*args, **kwargs``. That is not a style choice: the entry
    point reads each core's signature to decide whether a frame rate is
    required, so a wrapper that hides the parameter names silently turns a
    clear error into a wrong answer.
    """

    __slots__ = ("ops", "name", "color_gdown_ideal_core",
                 "motion_lpyr_ideal_core", "motion_lpyr_butter_core",
                 "motion_lpyr_iir_core")

    def __init__(self, ops: Any) -> None:
        self.ops = ops
        self.name = getattr(ops, "name", type(ops).__name__)
        self.color_gdown_ideal_core = functools.partial(
            color_gdown_ideal_core, ops)
        self.motion_lpyr_ideal_core = functools.partial(
            motion_lpyr_ideal_core, ops)
        self.motion_lpyr_butter_core = functools.partial(
            motion_lpyr_butter_core, ops)
        self.motion_lpyr_iir_core = functools.partial(
            motion_lpyr_iir_core, ops)

    def __repr__(self) -> str:
        return f"<pipelines derived from {self.name!r} operations>"


def bind(ops: Any) -> _BoundPipelines:
    """Give a set of primitive operations all four pipelines.

    This is what makes supporting new hardware a bounded job: implement the
    operations in :mod:`evm.backend.ops`, call this, and the backend is
    complete.
    """
    return _BoundPipelines(ops)
