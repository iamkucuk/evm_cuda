"""The per-frame EVM magnification pipelines, CUDA-accelerated.

This module implements the two rarer motion variants that the optimized
``batched`` module does not cover:

* ``magnify_motion_lpyr_ideal``  — ``amplify_spatial_lpyr_temporal_ideal``
* ``magnify_motion_lpyr_butter`` — ``amplify_spatial_lpyr_temporal_butter``

It also retains the original per-frame ``magnify_color_gdown_ideal`` /
``magnify_motion_lpyr_iir`` implementations, which are now superseded for
those two pipelines by the faster device-resident ``batched`` versions
(``evm_cuda.batched``). The package ``__init__`` routes the public names
to ``batched`` for the optimized pair and here for the two unique variants.

The per-frame path here allocates/frees device memory per call, which is why
``batched`` exists. It is kept as the canonical implementation of the two
motion-ideal/butter modes and as a correctness cross-check (its outputs must
match ``batched`` within tolerance — see ``tests/cuda/test_pipelines.py``).

Two layers, as in ``vidmag.cpu.magnify`` and ``vidmag.cuda.batched``
--------------------------------------------------------------
Each pipeline is an **array core** (``color_gdown_ideal_core``,
``motion_lpyr_ideal_core``, ``motion_lpyr_butter_core``,
``motion_lpyr_iir_core``) taking ``(T, H, W, 3)`` uint8 BGR frames already in
memory and returning the same shape and dtype, plus a **path wrapper**
(``magnify_*``) that decodes, drops the last ``drop_last`` frames, calls the
core and writes the file. ``motion_lpyr_ideal_core`` and
``motion_lpyr_butter_core`` are the only CUDA implementations of those two
variants, so they are what ``vidmag.cuda`` contributes to the Pipelines protocol
for them; the other two cores here are cross-checks for the faster fused
versions in ``batched``.

Strategy
--------
The host-side bookkeeping — frame reading, the Figure-6 alpha schedule,
pyramid level-size tables, drop-last-10 — stays in Python where it matches
the MATLAB structure and the Python baseline one-to-one. The hot loops
(per-pixel color convert, per-output convolution, per-pixel temporal filter,
per-pixel amplify+quantize) run on the GPU via ``_vidmag_cuda``.

To minimize host <-> device round-trips within a pipeline, we stage whole
arrays to the device once per pipeline call, run the per-frame operations
inside the per-kernel launches, and copy the final uint8 frame sequence
back. The current wrappers (in bindings.cpp) allocate/free device memory
per call for simplicity; a follow-up optimization can keep allocations
persistent across pipeline calls.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import _vidmag_cuda
from .runtime import butter_bandpass_coeffs
from ._common import figure6_alpha_schedule, read_frames as _read_frames
# Same validator as the CPU oracle and vidmag.cuda.batched: one set of messages for
# a wrong shape or dtype, whichever backend the user picked.
from ..cpu.magnify import _as_frames

DROP_LAST = _vidmag_cuda.drop_last
EXAGGERATION_FACTOR = _vidmag_cuda.exaggeration_factor
BINOM5 = np.array(_vidmag_cuda.binom5(), dtype=np.float32)
BINOM5_SUM1 = np.array(_vidmag_cuda.binom5_sum1(), dtype=np.float32)


# ---------------------------------------------------------------------------
# Helpers shared by all pipelines
# ---------------------------------------------------------------------------

def _bgr_u8_to_ntsc_f32(bgr: np.ndarray) -> np.ndarray:
    """(H,W,3) BGR uint8 -> (H,W,3) NTSC YIQ float32 via the GPU kernel."""
    return _vidmag_cuda.bgr_u8_to_ntsc_f32(np.ascontiguousarray(bgr))


def _ntsc_f32_to_bgr_u8(ntsc: np.ndarray) -> np.ndarray:
    return _vidmag_cuda.ntsc_f32_to_bgr_u8(np.ascontiguousarray(ntsc, dtype=np.float32))


def _write(out_path: str | Path, frames_uint8: np.ndarray, fps: float) -> None:
    # Delegates to the H.264-transcoding writer in batched.py so every CUDA
    # path emits browser/VSCode-playable video. Kept here only to avoid an
    # import cycle at module load.
    from .batched import _write as _batched_write
    _batched_write(out_path, frames_uint8, fps)


# ---------------------------------------------------------------------------
# Color pipeline (Gaussian-downsampled stack + ideal bandpass)
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
    """``vidmag.cpu.magnify.color_gdown_ideal_core``, GPU-accelerated per frame.

    Pipeline: per-frame bgr_u8->ntsc on GPU -> per-frame blur_dn on GPU
    (sum-normalized binom5) -> stack along time -> transpose to (N,T) ->
    ideal_bandpass via cuFFT -> per-channel gain -> per-frame bilinear upsample
    (cv2) -> add to ntsc frame -> ntsc->bgr u8 on GPU.

    ``frames_bgr_u8`` is ``(T, H, W, 3)`` uint8 BGR and so is the result.
    Superseded for speed by ``vidmag.cuda.batched.color_gdown_ideal_core``; kept as
    the per-frame cross-check (``tests/cuda/test_pipelines.py``).
    """
    frames = _as_frames(frames_bgr_u8)
    if sampling_rate is None:
        sampling_rate = fps
    n = len(frames)
    h, w = frames[0].shape[:2]

    # 1. Per-frame color convert + Gaussian downsample (both on GPU).
    gdown_frames: list[np.ndarray] = []
    for fr in frames:
        ntsc = _bgr_u8_to_ntsc_f32(fr)  # (H,W,3)
        # Per-channel blur_dn.
        small = np.empty(
            (ntsc.shape[0] // (2 ** level) + 1, ntsc.shape[1] // (2 ** level) + 1, 3),
            dtype=np.float32,
        )
        chans = [
            _vidmag_cuda.blur_dn(ntsc[:, :, c].astype(np.float32), level, BINOM5_SUM1)
            for c in range(3)
        ]
        small = np.stack(chans, axis=-1)
        gdown_frames.append(small)
    gdown = np.stack(gdown_frames, axis=0).astype(np.float32)  # (T,h_l,w_l,3)
    # Trim to actual size — blur_dn returns ceil-divided dims; np.stack should
    # have aligned them since all frames are identical-sized.

    # 2. Temporal bandpass per channel via the GPU ideal filter.
    filtered = np.empty_like(gdown)
    for c in range(3):
        # (T, h_l, w_l) -> transpose to (h_l*w_l, T) -> filter -> back.
        sig = gdown[..., c]  # (T, h_l, w_l)
        T_, H_, W_ = sig.shape
        flat = sig.reshape(T_, H_ * W_)
        # _vidmag_cuda.thwc_to_nt expects (T,H,W,C); use reshape instead.
        # Easiest: transpose to (N, T) inline.
        nt = np.ascontiguousarray(flat.T)  # (N=h_l*w_l, T)
        out = _vidmag_cuda.ideal_bandpass(nt, fl, fh, sampling_rate)
        filtered[..., c] = np.ascontiguousarray(out.T).reshape(T_, H_, W_)

    # 3. Per-channel gain (Y by alpha, I/Q by alpha*chromAtt).
    gain = np.array([alpha, alpha * chrom_attenuation, alpha * chrom_attenuation],
                    dtype=np.float32)
    filtered = filtered * gain

    # 4. Render: upsample + add + quantize, per frame.
    out = np.empty((n, h, w, 3), dtype=np.uint8)
    for i in range(n):
        ntsc_frame = _bgr_u8_to_ntsc_f32(frames[i])
        upsampled = cv2.resize(
            filtered[i].astype(np.float32), (w, h),
            interpolation=cv2.INTER_LINEAR,
        )
        rendered = ntsc_frame + upsampled
        out[i] = _ntsc_f32_to_bgr_u8(rendered)

    return out


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

    Reads the clip (dropping the last ``_vidmag_cuda.drop_last`` frames, as the
    MATLAB reference does) and returns float32 in [0, 1].
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


# ---------------------------------------------------------------------------
# Motion pipelines (Laplacian pyramid + temporal bandpass)
# ---------------------------------------------------------------------------

def _motion_lpyr_core(
    frames_bgr_u8: np.ndarray,
    fps: float,
    *,
    alpha: float,
    lambda_c: float,
    chrom_attenuation: float,
    filter_kind: str,            # "ideal" | "butter" | "iir"
    fl: float | None = None,
    fh: float | None = None,
    sampling_rate: float | None = None,
    r1: float | None = None,
    r2: float | None = None,
    exaggeration_factor: float = EXAGGERATION_FACTOR,
) -> np.ndarray:
    """Shared body for the three motion cores, array in / array out. Picks the
    temporal filter based on ``filter_kind``; the spatial Laplacian pyramid
    build/reconstruct and the Figure-6 schedule are common."""
    frames = _as_frames(frames_bgr_u8)
    if sampling_rate is None and filter_kind in ("ideal", "butter"):
        sampling_rate = fps
    n = len(frames)
    h, w = frames[0].shape[:2]

    # 1. NTSC convert all frames (GPU).
    ntsc_frames = [_bgr_u8_to_ntsc_f32(fr) for fr in frames]

    # 2. Per-frame Laplacian pyramid per channel (GPU). Auto height: we mirror
    # vidmag.pyramids.max_pyr_ht by iterating until both dims < 5.
    levels = 1
    hh, ww = h, w
    while hh >= 5 and ww >= 5:
        levels += 1
        hh = (hh + 1) // 2
        ww = (ww + 1) // 2
    # levels = 1 + max_pyr_ht((h,w), 5)

    # Build per-frame pyramids; stack each level along time.
    # pyrs_per_frame[i][c] = list of (H_l, W_l) arrays (finest-first).
    pyrs: list[list[list[np.ndarray]]] = []  # [frame][channel][level]
    for ntsc in ntsc_frames:
        frame_pyrs = []
        for c in range(3):
            bands, _ = _vidmag_cuda.lpyr_build(
                np.ascontiguousarray(ntsc[:, :, c], dtype=np.float32),
                levels, BINOM5,
            )
            frame_pyrs.append([np.ascontiguousarray(b, dtype=np.float32) for b in bands])
        pyrs.append(frame_pyrs)

    # 3. Stack each level along time, per channel: bands[l][c] = (T, H_l, W_l).
    level_sizes = [(int(pyrs[0][0][l].shape[0]),
                    int(pyrs[0][0][l].shape[1])) for l in range(levels)]
    stacked: list[list[np.ndarray]] = []  # [level][channel]
    for l in range(levels):
        lh, lw = level_sizes[l]
        chans = []
        for c in range(3):
            arr = np.stack([pyrs[i][c][l] for i in range(n)], axis=0)  # (T,lh,lw)
            chans.append(arr)
        stacked.append(chans)

    # 4. Temporal bandpass per (level, channel). For the streaming IIR/Butter
    # pipelines the reference filters the *entire* flattened pyramid per
    # pixel; for ideal it filters each level independently. We follow that
    # same structure here for fidelity.
    alpha_sched = figure6_alpha_schedule(
        levels, alpha, lambda_c, h, w, exaggeration_factor=exaggeration_factor
    )

    filtered: list[list[np.ndarray]] = []  # [level][channel]
    for l in range(levels):
        lh, lw = level_sizes[l]
        chans_out = []
        for c in range(3):
            sig = stacked[l][c]  # (T, lh, lw)
            T_, H_, W_ = sig.shape
            nt = np.ascontiguousarray(sig.reshape(T_, H_ * W_).T)  # (N, T)
            if filter_kind == "ideal":
                out = _vidmag_cuda.ideal_bandpass(nt, fl, fh, sampling_rate)
            elif filter_kind == "iir":
                out = _vidmag_cuda.iir_bandpass(nt, r1, r2)
            elif filter_kind == "butter":
                if fl is None or fh is None or sampling_rate is None:
                    # fl/fh are required keyword arguments on
                    # motion_lpyr_butter_core and sampling_rate defaults to fps
                    # above, so this fires only on an explicit None. Without it
                    # the None reaches scipy and comes back as a bare TypeError.
                    raise ValueError(
                        "the butter pipeline needs a band and a rate; got "
                        f"fl={fl!r} fh={fh!r} sampling_rate={sampling_rate!r}"
                    )
                (b0h, b1h, a1h), (b0l, b1l, a1l) = butter_bandpass_coeffs(
                    fl, fh, sampling_rate, order=1)
                out = _vidmag_cuda.butter_bandpass(nt, b0h, b1h, a1h, b0l, b1l, a1l)
            else:
                raise ValueError(f"unknown filter_kind {filter_kind!r}")
            chans_out.append(np.ascontiguousarray(out.T).reshape(T_, H_, W_)
                             * alpha_sched[l])
        filtered.append(chans_out)

    # 5. Per-frame reconstruct + chromAtt + add + quantize.
    out = np.empty((n, h, w, 3), dtype=np.uint8)
    for i in range(n):
        delta_chans = []
        for c in range(3):
            bands = [filtered[l][c][i] for l in range(levels)]
            recon = _vidmag_cuda.lpyr_recon(bands, BINOM5)
            delta_chans.append(recon)
        delta = np.stack(delta_chans, axis=-1)  # (H, W, 3)
        # ChromAtt on I,Q (motion pipelines attenuate chrominance post-recon).
        delta = _vidmag_cuda.attenuate_chrom(
            np.ascontiguousarray(delta, dtype=np.float32), chrom_attenuation)
        out[i] = _vidmag_cuda.add_and_quantize(ntsc_frames[i], delta)

    return out


# --- Array cores: the Pipelines protocol, one per filter --------------------

def motion_lpyr_ideal_core(
    frames_bgr_u8, fps, *, alpha, lambda_c, fl, fh,
    chrom_attenuation=0.0, sampling_rate=None,
    exaggeration_factor=EXAGGERATION_FACTOR,
):
    """Laplacian pyramid + ideal bandpass, array in / array out (uint8 BGR)."""
    return _motion_lpyr_core(
        frames_bgr_u8, fps, alpha=alpha, lambda_c=lambda_c,
        chrom_attenuation=chrom_attenuation, filter_kind="ideal",
        fl=fl, fh=fh, sampling_rate=sampling_rate,
        exaggeration_factor=exaggeration_factor,
    )


def motion_lpyr_butter_core(
    frames_bgr_u8, fps, *, alpha, lambda_c, fl, fh,
    chrom_attenuation=0.0, sampling_rate=None, order=1,
    exaggeration_factor=EXAGGERATION_FACTOR,
):
    """Laplacian pyramid + 1st-order Butterworth, array in / array out.

    ``order`` exists for signature parity with the CPU core; the CUDA kernel
    takes the six coefficients of a first-order bandpass, so anything else is
    refused rather than silently computed at order 1.
    """
    if order != 1:
        raise ValueError(
            f"the CUDA Butterworth kernel implements order=1 only; got "
            f"order={order!r}. Use backend='cpu' for higher orders."
        )
    return _motion_lpyr_core(
        frames_bgr_u8, fps, alpha=alpha, lambda_c=lambda_c,
        chrom_attenuation=chrom_attenuation, filter_kind="butter",
        fl=fl, fh=fh, sampling_rate=sampling_rate,
        exaggeration_factor=exaggeration_factor,
    )


def motion_lpyr_iir_core(
    frames_bgr_u8, fps, *, alpha, lambda_c, r1, r2,
    chrom_attenuation=0.1,
    exaggeration_factor=EXAGGERATION_FACTOR,
):
    """Laplacian pyramid + direct r1/r2 IIR, array in / array out.

    ``fps`` is accepted and ignored (the recursion runs on frame index).
    Superseded for speed by ``vidmag.cuda.batched.motion_lpyr_iir_core``.
    """
    return _motion_lpyr_core(
        frames_bgr_u8, fps, alpha=alpha, lambda_c=lambda_c,
        chrom_attenuation=chrom_attenuation, filter_kind="iir",
        r1=r1, r2=r2,
        exaggeration_factor=exaggeration_factor,
    )


# --- Path wrappers: decode, call the core, write ----------------------------

def magnify_motion_lpyr_ideal(
    vid_path, out_path, *, alpha, lambda_c, fl, fh,
    chrom_attenuation=0.0, sampling_rate=None,
    exaggeration_factor=EXAGGERATION_FACTOR,
):
    """``vidmag.magnify_motion_lpyr_ideal`` (Laplacian + ideal bandpass)."""
    frames, fps = _read_frames(vid_path)
    out = motion_lpyr_ideal_core(
        np.stack(frames, axis=0), fps, alpha=alpha, lambda_c=lambda_c,
        fl=fl, fh=fh, chrom_attenuation=chrom_attenuation,
        sampling_rate=sampling_rate,
        exaggeration_factor=exaggeration_factor,
    )
    _write(out_path, out, fps)
    return out.astype(np.float32) / 255.0


def magnify_motion_lpyr_butter(
    vid_path, out_path, *, alpha, lambda_c, fl, fh,
    chrom_attenuation=0.0, sampling_rate=None,
    exaggeration_factor=EXAGGERATION_FACTOR,
):
    """``vidmag.magnify_motion_lpyr_butter`` (Laplacian + 1st-order Butterworth)."""
    frames, fps = _read_frames(vid_path)
    out = motion_lpyr_butter_core(
        np.stack(frames, axis=0), fps, alpha=alpha, lambda_c=lambda_c,
        fl=fl, fh=fh, chrom_attenuation=chrom_attenuation,
        sampling_rate=sampling_rate,
        exaggeration_factor=exaggeration_factor,
    )
    _write(out_path, out, fps)
    return out.astype(np.float32) / 255.0


def magnify_motion_lpyr_iir(
    vid_path, out_path, *, alpha, lambda_c, r1, r2,
    chrom_attenuation=0.1,
    exaggeration_factor=EXAGGERATION_FACTOR,
):
    """``vidmag.magnify_motion_lpyr_iir`` (Laplacian + direct r1/r2 IIR)."""
    frames, fps = _read_frames(vid_path)
    out = motion_lpyr_iir_core(
        np.stack(frames, axis=0), fps, alpha=alpha, lambda_c=lambda_c,
        r1=r1, r2=r2, chrom_attenuation=chrom_attenuation,
        exaggeration_factor=exaggeration_factor,
    )
    _write(out_path, out, fps)
    return out.astype(np.float32) / 255.0
