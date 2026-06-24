"""The four EVM magnification pipelines, CUDA-accelerated.

Drop-in replacements for ``evm.magnify_color_gdown_ideal`` /
``magnify_motion_lpyr_ideal`` / ``magnify_motion_lpyr_butter`` /
``magnify_motion_lpyr_iir``. Same signatures, same parameters, same outputs
(within the per-stage tolerances documented in DESIGN.md).

Strategy
--------
The host-side bookkeeping — frame reading, the Figure-6 alpha schedule,
pyramid level-size tables, drop-last-10 — stays in Python where it matches
the MATLAB structure and the Python baseline one-to-one. The hot loops
(per-pixel color convert, per-output convolution, per-pixel temporal filter,
per-pixel amplify+quantize) run on the GPU via ``_evm_cuda``.

To minimize host <-> device round-trips within a pipeline, we stage whole
arrays to the device once per pipeline call, run the per-frame operations
inside the per-kernel launches, and copy the final uint8 frame sequence
back. The current wrappers (in bindings.cpp) allocate/free device memory
per call for simplicity; a follow-up optimization can keep allocations
persistent across pipeline calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from . import _evm_cuda
from .runtime import butter_bandpass_coeffs

DROP_LAST = _evm_cuda.drop_last
EXAGGERATION_FACTOR = _evm_cuda.exaggeration_factor
BINOM5 = np.array(_evm_cuda.binom5(), dtype=np.float32)
BINOM5_SUM1 = np.array(_evm_cuda.binom5_sum1(), dtype=np.float32)


# ---------------------------------------------------------------------------
# Helpers shared by all pipelines
# ---------------------------------------------------------------------------

def _read_frames(path: str | Path) -> Tuple[List[np.ndarray], float]:
    """Read all frames as BGR uint8 + fps; drop the last DROP_LAST frames."""
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


def _bgr_u8_to_ntsc_f32(bgr: np.ndarray) -> np.ndarray:
    """(H,W,3) BGR uint8 -> (H,W,3) NTSC YIQ float32 via the GPU kernel."""
    return _evm_cuda.bgr_u8_to_ntsc_f32(np.ascontiguousarray(bgr))


def _ntsc_f32_to_bgr_u8(ntsc: np.ndarray) -> np.ndarray:
    return _evm_cuda.ntsc_f32_to_bgr_u8(np.ascontiguousarray(ntsc, dtype=np.float32))


def _write(out_path: str | Path, frames_uint8: np.ndarray, fps: float) -> None:
    # Delegates to the H.264-transcoding writer in batched.py so every CUDA
    # path emits browser/VSCode-playable video (falls back to mp4v without
    # ffmpeg). Kept here only to avoid an import cycle at module load.
    from .batched import _write as _batched_write
    _batched_write(out_path, frames_uint8, fps)


def figure6_alpha_schedule(
    n_levels: int, alpha: float, lambda_c: float,
    vid_h: int, vid_w: int,
    *, exaggeration_factor: float = EXAGGERATION_FACTOR,
) -> List[float]:
    """Per-level amplification (Figure 6). Pure-Python mirror of
    evm.magnify.figure6_alpha_schedule — the CUDA kernels consume the result
    as a host-supplied per-level scale, so we don't need a GPU version.
    """
    delta = lambda_c / 8.0 / (1.0 + alpha)
    lam = (vid_h ** 2 + vid_w ** 2) ** 0.5 / 3.0
    coarse_first: list[float] = []
    for l in range(n_levels, 0, -1):
        if l == n_levels or l == 1:
            a = 0.0
        else:
            curr = (lam / delta / 8.0 - 1.0) * exaggeration_factor
            a = min(curr, alpha) if curr > alpha else curr
        coarse_first.append(a)
        lam /= 2.0
    return list(reversed(coarse_first))


# ---------------------------------------------------------------------------
# Color pipeline (Gaussian-downsampled stack + ideal bandpass)
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
    """``evm.magnify_color_gdown_ideal``, GPU-accelerated.

    Pipeline: read+drop-last -> per-frame bgr_u8->ntsc on GPU -> per-frame
    blur_dn on GPU (sum-normalized binom5) -> stack along time -> transpose
    to (N,T) -> ideal_bandpass via cuFFT -> per-channel gain -> per-frame
    bilinear upsample (cv2) -> add to ntsc frame -> ntsc->bgr u8 on GPU.
    """
    frames, fps = _read_frames(vid_path)
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
            _evm_cuda.blur_dn(ntsc[:, :, c].astype(np.float32), level, BINOM5_SUM1)
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
        # _evm_cuda.thwc_to_nt expects (T,H,W,C); use reshape instead.
        # Easiest: transpose to (N, T) inline.
        nt = np.ascontiguousarray(flat.T)  # (N=h_l*w_l, T)
        out = _evm_cuda.ideal_bandpass(nt, fl, fh, sampling_rate)
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

    _write(out_path, out, fps)
    return out.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Motion pipelines (Laplacian pyramid + temporal bandpass)
# ---------------------------------------------------------------------------

def _motion_lpyr(
    vid_path: str | Path,
    out_path: str | Path,
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
    """Shared body for the three motion pipelines. Picks the temporal filter
    based on ``filter_kind``; the spatial Laplacian pyramid build/reconstruct
    and the Figure-6 schedule are common."""
    frames, fps = _read_frames(vid_path)
    if sampling_rate is None and filter_kind in ("ideal", "butter"):
        sampling_rate = fps
    n = len(frames)
    h, w = frames[0].shape[:2]

    # 1. NTSC convert all frames (GPU).
    ntsc_frames = [_bgr_u8_to_ntsc_f32(fr) for fr in frames]

    # 2. Per-frame Laplacian pyramid per channel (GPU). Auto height: we mirror
    # evm.pyramids.max_pyr_ht by iterating until both dims < 5.
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
            bands, _ = _evm_cuda.lpyr_build(
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
                out = _evm_cuda.ideal_bandpass(nt, fl, fh, sampling_rate)
            elif filter_kind == "iir":
                out = _evm_cuda.iir_bandpass(nt, r1, r2)
            elif filter_kind == "butter":
                (b0h, b1h, a1h), (b0l, b1l, a1l) = butter_bandpass_coeffs(
                    fl, fh, sampling_rate, order=1)
                out = _evm_cuda.butter_bandpass(nt, b0h, b1h, a1h, b0l, b1l, a1l)
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
            recon = _evm_cuda.lpyr_recon(bands, BINOM5)
            delta_chans.append(recon)
        delta = np.stack(delta_chans, axis=-1)  # (H, W, 3)
        # ChromAtt on I,Q (motion pipelines attenuate chrominance post-recon).
        delta = _evm_cuda.attenuate_chrom(
            np.ascontiguousarray(delta, dtype=np.float32), chrom_attenuation)
        out[i] = _evm_cuda.add_and_quantize(ntsc_frames[i], delta)

    _write(out_path, out, fps)
    return out.astype(np.float32) / 255.0


def magnify_motion_lpyr_ideal(
    vid_path, out_path, *, alpha, lambda_c, fl, fh,
    chrom_attenuation=0.0, sampling_rate=None,
    exaggeration_factor=EXAGGERATION_FACTOR,
):
    """``evm.magnify_motion_lpyr_ideal`` (Laplacian + ideal bandpass)."""
    return _motion_lpyr(
        vid_path, out_path, alpha=alpha, lambda_c=lambda_c,
        chrom_attenuation=chrom_attenuation, filter_kind="ideal",
        fl=fl, fh=fh, sampling_rate=sampling_rate,
        exaggeration_factor=exaggeration_factor,
    )


def magnify_motion_lpyr_butter(
    vid_path, out_path, *, alpha, lambda_c, fl, fh,
    chrom_attenuation=0.0, sampling_rate=None,
    exaggeration_factor=EXAGGERATION_FACTOR,
):
    """``evm.magnify_motion_lpyr_butter`` (Laplacian + 1st-order Butterworth)."""
    return _motion_lpyr(
        vid_path, out_path, alpha=alpha, lambda_c=lambda_c,
        chrom_attenuation=chrom_attenuation, filter_kind="butter",
        fl=fl, fh=fh, sampling_rate=sampling_rate,
        exaggeration_factor=exaggeration_factor,
    )


def magnify_motion_lpyr_iir(
    vid_path, out_path, *, alpha, lambda_c, r1, r2,
    chrom_attenuation=0.1,
    exaggeration_factor=EXAGGERATION_FACTOR,
):
    """``evm.magnify_motion_lpyr_iir`` (Laplacian + direct r1/r2 IIR)."""
    return _motion_lpyr(
        vid_path, out_path, alpha=alpha, lambda_c=lambda_c,
        chrom_attenuation=chrom_attenuation, filter_kind="iir",
        r1=r1, r2=r2,
        exaggeration_factor=exaggeration_factor,
    )
