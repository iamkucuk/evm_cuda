"""Helpers shared across the CUDA pipeline modules (``batched``, ``pipelines``).

Keeps the Figure-6 amplification schedule and the frame reader in ONE place so
``batched.py`` (the optimized path) and ``pipelines.py`` (the per-frame path)
can't drift apart on algorithm details. Both consume the compiled extension
for ``drop_last`` / ``exaggeration_factor`` defaults.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import _evm_cuda


def read_frames(path: str | Path) -> tuple[list[np.ndarray], float]:
    """Read all frames as BGR uint8 + fps; drop the last ``drop_last`` frames."""
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
    if len(frames) > _evm_cuda.drop_last:
        frames = frames[: len(frames) - _evm_cuda.drop_last]
    return frames, float(fps)


def figure6_alpha_schedule(
    n_levels: int, alpha: float, lambda_c: float,
    vid_h: int, vid_w: int,
    *, exaggeration_factor: float = _evm_cuda.exaggeration_factor,
) -> list[float]:
    """Per-level amplification schedule (EVM Figure 6).

    Pure-Python mirror of ``evm.magnify.figure6_alpha_schedule`` — the CUDA
    kernels consume the result as a host-supplied per-level scale, so we don't
    need a GPU version. The two CUDA paths must use the IDENTICAL schedule, so
    it lives here once.
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
