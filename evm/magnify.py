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

Each function takes the *input video path* (not a preloaded array) so it can
stream frames one at a time exactly as the MATLAB code does, drop the last 10
frames, build per-frame pyramids, and write the output. They return the output
array too, for testing.

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
    build_lpyr,
    laplacian_pyramid_channels,
    max_pyr_ht,
    recon_lpyr,
    reconstruct_from_channels,
)
from .video import rgb_to_yiq, yiq_to_rgb

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
    lam = (vid_h ** 2 + vid_w ** 2) ** 0.5 / 3.0  # noqa: E741 - match MATLAB var

    # MATLAB iterates coarse->fine (nLevels..1) and appends per level; we build
    # coarse->fine then reverse to get the finest-first order of build_lpyr.
    coarse_first: list[float] = []
    for l in range(n_levels, 0, -1):  # noqa: E741 - match MATLAB var
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


def _read_frames(path: str | Path) -> tuple[list[np.ndarray], float]:
    """Read all frames as BGR uint8 + fps. Drops the last ``DROP_LAST``."""
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
    if len(frames) > DROP_LAST:
        frames = frames[: len(frames) - DROP_LAST]
    return frames, float(fps)


# ---------------------------------------------------------------------------
# Public pipelines
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
    """``amplify_spatial_Gdown_temporal_ideal.m``.

    Builds a Gaussian-downsampled NTSC stack (``build_GDown_stack`` via
    :func:`blur_dn_clr` at ``level``), applies the ideal bandpass, scales the
    filtered Y channel by ``alpha`` and I/Q channels by
    ``alpha * chrom_attenuation``, then renders each frame by upsampling
    (``imresize``) the magnified signal back to full resolution and adding to
    the original NTSC frame.
    """
    frames, fps = _read_frames(vid_path)
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

    _write(out_path, out, fps)
    return out.astype(np.float32) / 255.0


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
    """``amplify_spatial_lpyr_temporal_ideal.m``.

    Builds a per-frame Laplacian pyramid (auto height), stacks each band along
    time, applies the ideal bandpass per band, scales each band by the
    Figure-6 alpha schedule, reconstructs, attenuates chrominance, and renders.
    """
    frames, fps = _read_frames(vid_path)
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
        for l in range(n_levels)  # noqa: E741
    ]

    # Figure-6 schedule (finest-first).
    alpha_sched = figure6_alpha_schedule(
        n_levels, alpha, lambda_c, h, w, exaggeration_factor=exaggeration_factor
    )

    # Temporal bandpass + per-level amplify.
    amplified_bands = []
    for l, band in enumerate(bands):  # noqa: E741
        filtered = ideal_bandpass(band.astype(np.float64), fl, fh, sampling_rate)
        amplified_bands.append(filtered * alpha_sched[l])

    # Reconstruct per frame.
    filtered_per_frame = [
        [amplified_bands[l][i] for l in range(n_levels)] for i in range(n)
    ]
    rendered_ntsc = _amplify_lpyr_stack(
        ntsc_frames, filtered_per_frame, pind, chrom_attenuation
    )

    out = np.stack([_ntsc_to_bgr_uint8(x) for x in rendered_ntsc], axis=0)
    _write(out_path, out, fps)
    return out.astype(np.float32) / 255.0


def _streaming_lpyr_motion(
    vid_path: str | Path,
    out_path: str | Path,
    *,
    alpha: float,
    lambda_c: float,
    chrom_attenuation: float,
    filter_fn,
    exaggeration_factor: float = EXAGGERATION_FACTOR,
) -> np.ndarray:
    """Shared body of the butter / iir streaming motion pipelines.

    ``filter_fn(pyr_time_series)`` must take an array of shape (T, n_coeffs)
    (one pyramid flattened per frame, all 3 channels concatenated) and return
    the temporally-filtered series of the same shape. This matches the
    reference, which filters the *entire* pyramid coefficient vector as one
    temporal signal per pixel.
    """
    frames, fps = _read_frames(vid_path)
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
        for l in range(n_levels):  # noqa: E741
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
        for l in range(n_levels):  # noqa: E741
            sl = _level_slice(l, pind)
            lh, lw = int(pind[l, 0]), int(pind[l, 1])
            bands.append(filtered[i, sl, :].reshape(lh, lw, 3) * alpha_sched[l])
        filtered_per_frame.append(bands)

    rendered_ntsc = _amplify_lpyr_stack(
        ntsc_frames, filtered_per_frame, pind, chrom_attenuation
    )
    out = np.stack([_ntsc_to_bgr_uint8(x) for x in rendered_ntsc], axis=0)
    _write(out_path, out, fps)
    return out.astype(np.float32) / 255.0


def _level_slice(level: int, pind: np.ndarray) -> slice:
    start = sum(int(pind[l, 0] * pind[l, 1]) for l in range(level))
    length = int(pind[level, 0] * pind[level, 1])
    return slice(start, start + length)


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
    """``amplify_spatial_lpyr_temporal_butter.m``."""
    if sampling_rate is None:
        # need fps to set cutoffs; peek
        cap = cv2.VideoCapture(str(vid_path))
        sampling_rate = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
    sr = float(sampling_rate)

    def filt(s):
        return butter_bandpass(s, fl, fh, sr, order=order, axis=0)

    return _streaming_lpyr_motion(
        vid_path, out_path,
        alpha=alpha, lambda_c=lambda_c,
        chrom_attenuation=chrom_attenuation,
        filter_fn=filt,
        exaggeration_factor=exaggeration_factor,
    )


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
    """``amplify_spatial_lpyr_temporal_iir.m``."""
    def filt(s):
        return iir_bandpass(s, r1, r2, axis=0)

    return _streaming_lpyr_motion(
        vid_path, out_path,
        alpha=alpha, lambda_c=lambda_c,
        chrom_attenuation=chrom_attenuation,
        filter_fn=filt,
        exaggeration_factor=exaggeration_factor,
    )


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------


def _write(out_path: str | Path, frames_uint8: np.ndarray, fps: float) -> None:
    # Delegates to the shared H.264 encoder in evm.video so every writer
    # (batched CUDA, host CUDA, and this pure-Python path) emits identical
    # browser/VSCode-playable video.
    from .video import encode_video
    encode_video(frames_uint8, out_path, fps)
