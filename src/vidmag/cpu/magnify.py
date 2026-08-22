"""Magnification pipelines, faithful to the MIT MATLAB reference.

Three entry points mirroring the three reference amplification functions:

* :func:`magnify_color_gdown_ideal`  — ``amplify_spatial_Gdown_temporal_ideal.m``
  Gaussian-downsampled stack + ideal bandpass. Used for face/baby/wrist pulse.
* :func:`magnify_motion_lpyr_ideal`   — ``amplify_spatial_lpyr_temporal_ideal.m``
  Laplacian pyramid + ideal bandpass.
* :func:`magnify_motion_lpyr_butter`  — ``amplify_spatial_lpyr_temporal_butter.m``
  Laplacian pyramid + 1st-order Butterworth bandpass (streaming).
* :func:`magnify_motion_lpyr_iir`     — ``amplify_spatial_lpyr_temporal_iir.m``
  Laplacian pyramid + direct r1/r2 IIR bandpass (streaming).

Each pipeline exists in two layers:

* an **array core** — ``color_gdown_ideal_core``, ``motion_lpyr_ideal_core``,
  ``motion_lpyr_butter_core``, ``motion_lpyr_iir_core`` — taking decoded frames
  already in memory and returning the magnified frames, touching no files. All
  four share one call shape, ``core(frames_bgr_u8, fps, *, ...) -> ndarray``;
  they are the reference implementation of the Pipelines protocol in
  ``docs/dev/PLAN.md`` section 3c, which every backend either inherits
  generically or overrides with a fused version.
* the **path wrapper** — ``magnify_color_gdown_ideal`` and friends — which
  decodes the input video (dropping the last :data:`DROP_LAST` frames exactly
  as the MATLAB code does), calls the core, writes the output file, and returns
  the frames as float32 in [0, 1] for testing.

Parameters and the per-level alpha schedule (Figure 6 of the paper) match the
reference bit-for-bit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from .filters import (
    butter_bandpass,
    ideal_bandpass,
    iir_bandpass,
)
from .pyramids import (
    blur_dn_clr,
    laplacian_pyramid_channels,
    reconstruct_from_channels,
)
from ..io.video import rgb_to_yiq, yiq_to_rgb

FilterKind = Literal["ideal", "butter", "iir"]

# The reference drops the last 10 frames of every input (see startIndex/endIndex
# in all four amplification functions).
DROP_LAST = 10

# Figure-6 exaggeration factor (hardcoded in the MATLAB reference).
EXAGGERATION_FACTOR = 2.0


# ---------------------------------------------------------------------------
# Figure-6 per-level amplification schedule
# ---------------------------------------------------------------------------


def figure6_alpha_schedule(
    n_levels: int,
    alpha: float,
    lambda_c: float,
    vid_h: int,
    vid_w: int,
    *,
    exaggeration_factor: float = EXAGGERATION_FACTOR,
) -> list[float]:
    """Compute the per-level amplification from the paper's Figure 6.

    Mirrors the loop in ``amplify_spatial_lpyr_temporal_{ideal,butter,iir}.m``:

        delta = lambda_c / 8 / (1 + alpha)
        lambda = sqrt(H^2 + W^2) / 3   # representative wavelength, coarsest band
        for l = nLevels:-1:1:
            currAlpha = (lambda/delta/8 - 1) * exaggeration_factor
            alpha_l = 0                       if l in {1, nLevels}  (drop edges)
                   = min(currAlpha, alpha)     otherwise
            lambda /= 2

    The list is returned finest-first (``alpha_l[0]`` is the finest band) to
    match the pyramid band ordering produced by :func:`build_lpyr`.
    """
    delta = lambda_c / 8.0 / (1.0 + alpha)
    lam = (vid_h ** 2 + vid_w ** 2) ** 0.5 / 3.0  # match MATLAB var

    # MATLAB iterates coarse->fine (nLevels..1) and appends per level; we build
    # coarse->fine then reverse to get the finest-first order of build_lpyr.
    coarse_first: list[float] = []
    for l in range(n_levels, 0, -1):  # match MATLAB var
        if l == n_levels or l == 1:
            a = 0.0
        else:
            curr_alpha = (lam / delta / 8.0 - 1.0) * exaggeration_factor
            a = min(curr_alpha, alpha) if curr_alpha > alpha else curr_alpha
        coarse_first.append(a)
        lam /= 2.0
    return list(reversed(coarse_first))  # finest-first


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _rgb_frame_to_ntsc(bgr_uint8: np.ndarray) -> np.ndarray:
    """OpenCV BGR uint8 frame -> NTSC YIQ float in [0,1] (MATLAB rgb2ntsc)."""
    rgb = bgr_uint8[:, :, ::-1].astype(np.float64) / 255.0
    return rgb_to_yiq(rgb)


def _ntsc_to_bgr_uint8(ntsc: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_rgb_frame_to_ntsc`, clipped and quantised for output."""
    rgb = yiq_to_rgb(ntsc)
    rgb = np.clip(rgb, 0.0, 1.0)
    bgr = rgb[:, :, ::-1]
    return np.round(bgr * 255.0).astype(np.uint8)


def _read_frames(
    path: str | Path, *, drop_last: int = DROP_LAST
) -> tuple[list[np.ndarray], float]:
    """Read all frames as BGR uint8 + fps, dropping the last ``drop_last``.

    The default is :data:`DROP_LAST`, which is what the four public path
    functions want (the MATLAB reference drops ten). :func:`vidmag.magnify` passes
    its own count — plan decision D8 — and drops nothing by default.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {path!r}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames: list[np.ndarray] = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if drop_last and len(frames) > drop_last:
        frames = frames[: len(frames) - drop_last]
    return frames, float(fps)


def _as_frames(frames_bgr_u8: np.ndarray) -> np.ndarray:
    """Validate decoded frames for the array cores: ``(T, H, W, 3)`` uint8 BGR.

    The cores are the array-in/array-out boundary, so a wrong shape or dtype
    has to be caught here. It raises rather than coercing: silently casting
    float frames to uint8, or accepting a ``(H, W, 3, T)`` stack, would render
    a plausible-looking wrong video instead of an error.
    """
    frames = np.asarray(frames_bgr_u8)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(
            f"frames must have shape (T, H, W, 3) BGR; got {frames.shape}"
        )
    if frames.dtype != np.uint8:
        raise TypeError(
            f"frames must be uint8 BGR in [0, 255]; got dtype {frames.dtype}"
        )
    if frames.shape[0] == 0:
        raise ValueError("frames is empty: nothing to magnify")
    return frames


# ---------------------------------------------------------------------------
# Array cores — the Pipelines protocol (plan section 3c)
#
# One call shape for all four: ``(frames_bgr_u8, fps, *, <params>) -> ndarray``
# with ``(T, H, W, 3)`` uint8 BGR in and out. Keyword order is fixed across the
# four: required magnification parameters first (``alpha``, then the spatial
# selector ``level``/``lambda_c``, then the temporal band ``fl``/``fh`` or
# ``r1``/``r2``), then the optional ones in the order ``chrom_attenuation``,
# ``sampling_rate``, ``order``, ``exaggeration_factor``. That is exactly the
# order the public path functions already used, so each wrapper is a
# pass-through with no re-ordering to get wrong.
# ---------------------------------------------------------------------------


def color_gdown_ideal_core(
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
    """``amplify_spatial_Gdown_temporal_ideal.m``, array in / array out.

    Builds a Gaussian-downsampled NTSC stack (``build_GDown_stack`` via
    :func:`blur_dn_clr` at ``level``), applies the ideal bandpass, scales the
    filtered Y channel by ``alpha`` and I/Q channels by
    ``alpha * chrom_attenuation``, then renders each frame by upsampling
    (``imresize``) the magnified signal back to full resolution and adding to
    the original NTSC frame.

    ``frames_bgr_u8`` is ``(T, H, W, 3)`` uint8 BGR — OpenCV's decode order —
    and the result has the same shape and dtype. ``sampling_rate`` defaults to
    ``fps``; pass it to filter at a rate other than the clip's own.
    """
    frames = _as_frames(frames_bgr_u8)
    if sampling_rate is None:
        sampling_rate = fps
    n = len(frames)
    h, w = frames[0].shape[:2]

    # --- Spatial: build the Gaussian-downsampled NTSC stack (T, h_l, w_l, 3).
    gdown = np.stack(
        [blur_dn_clr(_rgb_frame_to_ntsc(fr), level) for fr in frames], axis=0
    )

    # --- Temporal: ideal bandpass each channel along time.
    filtered = np.stack(
        [ideal_bandpass(gdown[..., c].astype(np.float64), fl, fh, sampling_rate)
         for c in range(3)],
        axis=-1,
    )

    # --- Amplify: Y by alpha, I/Q by alpha*chromAtt.
    gain = np.array([alpha, alpha * chrom_attenuation, alpha * chrom_attenuation])
    filtered = filtered * gain

    # --- Render: upsample back to full res, add to original NTSC frame.
    out = np.empty((n, h, w, 3), dtype=np.uint8)
    for i in range(n):
        ntsc_frame = _rgb_frame_to_ntsc(frames[i])
        # MATLAB uses imresize (bilinear by default) -> OpenCV INTER_LINEAR.
        upsampled = cv2.resize(
            filtered[i], (w, h), interpolation=cv2.INTER_LINEAR
        )
        rendered = ntsc_frame + upsampled
        out[i] = _ntsc_to_bgr_uint8(rendered)

    return out


def _amplify_lpyr_stack(
    frames_ntsc: list[np.ndarray],
    filtered_per_frame: list[list[np.ndarray]],
    pind,
    chrom_attenuation: float,
) -> list[np.ndarray]:
    """Reconstruct + attenuate chrominance for the motion pipelines."""
    out = []
    for i, ntsc_frame in enumerate(frames_ntsc):
        recon = reconstruct_from_channels(filtered_per_frame[i], pind)
        recon[..., 1] *= chrom_attenuation
        recon[..., 2] *= chrom_attenuation
        out.append(ntsc_frame + recon)
    return out


def motion_lpyr_ideal_core(
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
    """``amplify_spatial_lpyr_temporal_ideal.m``, array in / array out.

    Builds a per-frame Laplacian pyramid (auto height), stacks each band along
    time, applies the ideal bandpass per band, scales each band by the
    Figure-6 alpha schedule, reconstructs, attenuates chrominance, and renders.

    ``frames_bgr_u8`` is ``(T, H, W, 3)`` uint8 BGR and so is the result.
    """
    frames = _as_frames(frames_bgr_u8)
    if sampling_rate is None:
        sampling_rate = fps
    n = len(frames)
    h, w = frames[0].shape[:2]

    ntsc_frames = [_rgb_frame_to_ntsc(fr) for fr in frames]

    # Per-frame Laplacian pyramid (auto height == 1 + maxPyrHt).
    pyrs = [laplacian_pyramid_channels(f, "auto") for f in ntsc_frames]
    n_levels = pyrs[0][1].shape[0]
    pind = pyrs[0][1]

    # Stack each band along time: bands[l] is (T, h_l, w_l, 3).
    bands = [
        np.stack([pyrs[i][0][l] for i in range(n)], axis=0)
        for l in range(n_levels)
    ]

    # Figure-6 schedule (finest-first).
    alpha_sched = figure6_alpha_schedule(
        n_levels, alpha, lambda_c, h, w, exaggeration_factor=exaggeration_factor
    )

    # Temporal bandpass + per-level amplify.
    amplified_bands = []
    for l, band in enumerate(bands):
        filtered = ideal_bandpass(band.astype(np.float64), fl, fh, sampling_rate)
        amplified_bands.append(filtered * alpha_sched[l])

    # Reconstruct per frame.
    filtered_per_frame = [
        [amplified_bands[l][i] for l in range(n_levels)] for i in range(n)
    ]
    rendered_ntsc = _amplify_lpyr_stack(
        ntsc_frames, filtered_per_frame, pind, chrom_attenuation
    )

    return np.stack([_ntsc_to_bgr_uint8(x) for x in rendered_ntsc], axis=0)


def _streaming_lpyr_motion(
    frames_bgr_u8: np.ndarray,
    *,
    alpha: float,
    lambda_c: float,
    chrom_attenuation: float,
    filter_fn,
    exaggeration_factor: float = EXAGGERATION_FACTOR,
) -> np.ndarray:
    """Shared body of the butter / iir streaming motion cores.

    ``filter_fn(pyr_time_series)`` must take an array of shape (T, n_coeffs)
    (one pyramid flattened per frame, all 3 channels concatenated) and return
    the temporally-filtered series of the same shape. This matches the
    reference, which filters the *entire* pyramid coefficient vector as one
    temporal signal per pixel.

    Takes no sampling rate: the two callers have already baked theirs into
    ``filter_fn`` (Butterworth) or do not need one at all (the r1/r2 IIR).
    """
    frames = _as_frames(frames_bgr_u8)
    n = len(frames)
    h, w = frames[0].shape[:2]

    ntsc_frames = [_rgb_frame_to_ntsc(fr) for fr in frames]

    # Build the first pyramid to get pind / n_levels / per-level sizes.
    pyrs = [laplacian_pyramid_channels(f, "auto") for f in ntsc_frames]
    n_levels = pyrs[0][1].shape[0]
    pind = pyrs[0][1]

    # Flatten each frame's pyramid into a single (n_coeffs * 3) vector, stacked
    # along time -> (T, n_coeffs, 3). n_coeffs is the same for every frame
    # because pind is identical.
    n_coeffs = sum(int(pind[l, 0] * pind[l, 1]) for l in range(n_levels))
    series = np.empty((n, n_coeffs, 3), dtype=np.float64)
    for i in range(n):
        for l in range(n_levels):
            band = pyrs[i][0][l]
            sl = _level_slice(l, pind)
            series[i, sl, :] = band.reshape(-1, 3)

    # Temporal filter the whole coefficient vector (per channel, per coeff).
    filtered = filter_fn(series)

    # Figure-6 per-level amplification (finest-first).
    alpha_sched = figure6_alpha_schedule(
        n_levels, alpha, lambda_c, h, w, exaggeration_factor=exaggeration_factor
    )

    filtered_per_frame = []
    for i in range(n):
        bands = []
        for l in range(n_levels):
            sl = _level_slice(l, pind)
            lh, lw = int(pind[l, 0]), int(pind[l, 1])
            bands.append(filtered[i, sl, :].reshape(lh, lw, 3) * alpha_sched[l])
        filtered_per_frame.append(bands)

    rendered_ntsc = _amplify_lpyr_stack(
        ntsc_frames, filtered_per_frame, pind, chrom_attenuation
    )
    return np.stack([_ntsc_to_bgr_uint8(x) for x in rendered_ntsc], axis=0)


def _level_slice(level: int, pind: np.ndarray) -> slice:
    start = sum(int(pind[l, 0] * pind[l, 1]) for l in range(level))
    length = int(pind[level, 0] * pind[level, 1])
    return slice(start, start + length)


def motion_lpyr_butter_core(
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
    """``amplify_spatial_lpyr_temporal_butter.m``, array in / array out.

    ``frames_bgr_u8`` is ``(T, H, W, 3)`` uint8 BGR and so is the result.
    ``sampling_rate`` defaults to ``fps``, which is what sets the cutoffs.
    """
    sr = float(fps if sampling_rate is None else sampling_rate)

    def filt(s):
        return butter_bandpass(s, fl, fh, sr, order=order, axis=0)

    return _streaming_lpyr_motion(
        frames_bgr_u8,
        alpha=alpha, lambda_c=lambda_c,
        chrom_attenuation=chrom_attenuation,
        filter_fn=filt,
        exaggeration_factor=exaggeration_factor,
    )


def motion_lpyr_iir_core(
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
    """``amplify_spatial_lpyr_temporal_iir.m``, array in / array out.

    ``frames_bgr_u8`` is ``(T, H, W, 3)`` uint8 BGR and so is the result.

    ``fps`` is accepted and ignored: the r1/r2 IIR is a direct recursion on
    frame index with no sampling rate anywhere in it (that is exactly why the
    reference exposes r1/r2 rather than fl/fh here). It stays in the signature
    so all four cores share one call shape and the facade can dispatch to any
    of them without special-casing.
    """
    def filt(s):
        return iir_bandpass(s, r1, r2, axis=0)

    return _streaming_lpyr_motion(
        frames_bgr_u8,
        alpha=alpha, lambda_c=lambda_c,
        chrom_attenuation=chrom_attenuation,
        filter_fn=filt,
        exaggeration_factor=exaggeration_factor,
    )


# ---------------------------------------------------------------------------
# Public path pipelines — decode, call the core, write. Signatures and returned
# values are unchanged from before the core split; tests/test_golden.py holds
# their byte-level output fixed.
# ---------------------------------------------------------------------------


def magnify_color_gdown_ideal(
    vid_path: str | Path,
    out_path: str | Path,
    *,
    alpha: float,
    level: int,
    fl: float,
    fh: float,
    chrom_attenuation: float = 1.0,
    sampling_rate: float | None = None,
) -> np.ndarray:
    """Run :func:`color_gdown_ideal_core` on ``vid_path``, write ``out_path``.

    Reads the clip (dropping the last :data:`DROP_LAST` frames, as the MATLAB
    reference does) and returns the magnified frames as float32 in [0, 1].
    """
    frames, fps = _read_frames(vid_path)
    out = color_gdown_ideal_core(
        np.stack(frames, axis=0), fps,
        alpha=alpha, level=level, fl=fl, fh=fh,
        chrom_attenuation=chrom_attenuation,
        sampling_rate=sampling_rate,
    )
    _write(out_path, out, fps)
    return out.astype(np.float32) / 255.0


def magnify_motion_lpyr_ideal(
    vid_path: str | Path,
    out_path: str | Path,
    *,
    alpha: float,
    lambda_c: float,
    fl: float,
    fh: float,
    chrom_attenuation: float = 0.0,
    sampling_rate: float | None = None,
    exaggeration_factor: float = EXAGGERATION_FACTOR,
) -> np.ndarray:
    """Run :func:`motion_lpyr_ideal_core` on ``vid_path``, write ``out_path``."""
    frames, fps = _read_frames(vid_path)
    out = motion_lpyr_ideal_core(
        np.stack(frames, axis=0), fps,
        alpha=alpha, lambda_c=lambda_c, fl=fl, fh=fh,
        chrom_attenuation=chrom_attenuation,
        sampling_rate=sampling_rate,
        exaggeration_factor=exaggeration_factor,
    )
    _write(out_path, out, fps)
    return out.astype(np.float32) / 255.0


def magnify_motion_lpyr_butter(
    vid_path: str | Path,
    out_path: str | Path,
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
    """Run :func:`motion_lpyr_butter_core` on ``vid_path``, write ``out_path``."""
    frames, fps = _read_frames(vid_path)
    out = motion_lpyr_butter_core(
        np.stack(frames, axis=0), fps,
        alpha=alpha, lambda_c=lambda_c, fl=fl, fh=fh,
        chrom_attenuation=chrom_attenuation,
        sampling_rate=sampling_rate,
        order=order,
        exaggeration_factor=exaggeration_factor,
    )
    _write(out_path, out, fps)
    return out.astype(np.float32) / 255.0


def magnify_motion_lpyr_iir(
    vid_path: str | Path,
    out_path: str | Path,
    *,
    alpha: float,
    lambda_c: float,
    r1: float,
    r2: float,
    chrom_attenuation: float = 0.1,
    exaggeration_factor: float = EXAGGERATION_FACTOR,
) -> np.ndarray:
    """Run :func:`motion_lpyr_iir_core` on ``vid_path``, write ``out_path``."""
    frames, fps = _read_frames(vid_path)
    out = motion_lpyr_iir_core(
        np.stack(frames, axis=0), fps,
        alpha=alpha, lambda_c=lambda_c, r1=r1, r2=r2,
        chrom_attenuation=chrom_attenuation,
        exaggeration_factor=exaggeration_factor,
    )
    _write(out_path, out, fps)
    return out.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------


def _write(out_path: str | Path, frames_uint8: np.ndarray, fps: float) -> None:
    # Delegates to the shared H.264 encoder in vidmag.io.video so every writer
    # (batched CUDA, host CUDA, and this pure-Python path) emits identical
    # browser/VSCode-playable video.
    from ..io.video import encode_video
    encode_video(frames_uint8, out_path, fps)


# ---------------------------------------------------------------------------
# Phase-based magnification (the 2013 follow-up method)
# ---------------------------------------------------------------------------
#
# Named to match the other four so the facade and the registry find it the same
# way: `phase_core` takes frames, `magnify_phase` takes a path. The work itself
# is in vidmag.cpu.phase_magnify, which is a different enough algorithm to deserve
# its own module.


def phase_core(
    frames_bgr_u8: np.ndarray,
    fps: float,
    *,
    alpha: float,
    fl: float | None = None,
    fh: float | None = None,
    r1: float | None = None,
    r2: float | None = None,
    scales: int = 3,
    orientations: int = 4,
    sigma: float = 0.0,
    sampling_rate: float | None = None,
) -> np.ndarray:
    """Amplify motion by changing phase. Array in, array out.

    See :func:`vidmag.cpu.phase_magnify.phase_magnify` for what the parameters do
    and how this differs from the pyramid-based motion pipelines.
    """
    from .phase_magnify import phase_magnify

    rate = fps if sampling_rate is None else sampling_rate
    return phase_magnify(
        _as_frames(frames_bgr_u8), rate, alpha=alpha, fl=fl, fh=fh,
        r1=r1, r2=r2, scales=scales, orientations=orientations, sigma=sigma,
    )


def magnify_phase(
    vid_path: str | Path,
    out_path: str | Path,
    *,
    alpha: float,
    fl: float | None = None,
    fh: float | None = None,
    r1: float | None = None,
    r2: float | None = None,
    scales: int = 3,
    orientations: int = 4,
    sigma: float = 0.0,
    sampling_rate: float | None = None,
) -> np.ndarray:
    """Phase-based motion magnification, from a file to a file.

    Drops the last ten frames, as the other path functions do, so that a clip
    run through this and through them is the same length.
    """
    frames, fps = _read_frames(vid_path)
    stack = np.stack(frames, axis=0)
    out = phase_core(
        stack, fps, alpha=alpha, fl=fl, fh=fh, r1=r1, r2=r2,
        scales=scales, orientations=orientations, sigma=sigma,
        sampling_rate=sampling_rate,
    )
    _write(out_path, out, fps)
    return out.astype(np.float32) / 255.0
