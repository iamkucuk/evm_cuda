"""End-to-end tests for the magnification pipelines.

These build a tiny synthetic video on disk (the pipelines read from a path to
mirror the MATLAB streaming behaviour) and check:

* the Figure-6 alpha schedule zeroes the boundary bands and clamps to alpha;
* a flat input is left unchanged (the bandpass produces zero);
* a clip with in-band energy is amplified.

The face/baby numerical comparison against MIT's own output videos lives in
``test_against_mit_reference.py`` as a slow integration test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evm import (  # noqa: E402
    figure6_alpha_schedule,
    magnify_color_gdown_ideal,
    magnify_motion_lpyr_iir,
)


def _write_synth_video(path: Path, frames: np.ndarray, fps: float = 30.0) -> None:
    """Write a (T,H,W,3) float [0,1] array as a BGR uint8 mp4."""
    path.parent.mkdir(parents=True, exist_ok=True)
    t, h, w, _ = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, fps, (w, h), isColor=True)
    u8 = np.clip(np.round(frames * 255), 0, 255).astype(np.uint8)
    for i in range(t):
        vw.write(u8[i][:, :, ::-1])
    vw.release()


def _flat_clip(t: int = 40, h: int = 32, w: int = 32) -> np.ndarray:
    return np.full((t, h, w, 3), 0.5, dtype=np.float32)


def _pulse_clip(t: int = 120, h: int = 64, w: int = 64) -> np.ndarray:
    # 0.9 Hz sits well inside the [0.5, 1.5] test band below.
    n = np.arange(t)
    flicker = 0.08 * np.sin(2 * np.pi * 0.9 * n / 30.0)
    intensity = (0.5 + flicker).astype(np.float32)
    return intensity[:, None, None, None] * np.ones((t, h, w, 3), dtype=np.float32)


# --- Figure-6 schedule ---------------------------------------------------


def test_figure6_zeroes_boundary_bands() -> None:
    sched = figure6_alpha_schedule(
        n_levels=8, alpha=10.0, lambda_c=16.0, vid_h=960, vid_w=544
    )
    assert sched[0] == 0.0  # finest
    assert sched[-1] == 0.0  # coarsest residual
    for a in sched[1:-1]:
        assert 0.0 < a <= 10.0


def test_figure6_clamps_to_alpha_when_lambda_small() -> None:
    # Tiny lambda_c makes delta tiny -> currAlpha huge everywhere -> clamp to alpha
    # everywhere except the zeroed boundary bands (matches MATLAB branch).
    sched = figure6_alpha_schedule(
        n_levels=6, alpha=5.0, lambda_c=1.0, vid_h=960, vid_w=544
    )
    assert sched[0] == 0.0 and sched[-1] == 0.0
    for a in sched[1:-1]:
        assert a == pytest.approx(5.0)


def test_figure6_small_lambda_c_gives_negative_alpha() -> None:
    # This is the literal MATLAB behaviour: with lambda_c huge relative to the
    # representative wavelengths, currAlpha = lambda*(1+alpha)/lambda_c - 1 ~ -1,
    # and the `else` branch keeps that negative value (no clamping to alpha).
    sched = figure6_alpha_schedule(
        n_levels=6, alpha=5.0, lambda_c=1e6, vid_h=960, vid_w=544
    )
    assert sched[0] == 0.0 and sched[-1] == 0.0
    for a in sched[1:-1]:
        assert a < 0.0  # matches MATLAB; produces mild attenuation


# --- color pipeline ------------------------------------------------------


def test_color_flat_input_unchanged(tmp_path: Path) -> None:
    src = tmp_path / "flat.mp4"
    dst = tmp_path / "out.mp4"
    _write_synth_video(src, _flat_clip())
    out = magnify_color_gdown_ideal(
        str(src), str(dst), alpha=50, level=2, fl=0.83, fh=0.99, chrom_attenuation=1.0
    )
    assert out.shape == (30, 32, 32, 3)  # 40 frames - 10 dropped
    # A flat image has no temporal variation to amplify.
    assert np.abs(out - 0.5).max() < 0.05


def test_color_amplifies_pulse(tmp_path: Path) -> None:
    src = tmp_path / "pulse.mp4"
    dst = tmp_path / "out.mp4"
    clip = _pulse_clip()
    _write_synth_video(src, clip)
    # Use a band centred on the 0.9 Hz pulse so the ideal filter passes it
    # cleanly (the edge of [0.83, 0.99] would attenuate it too aggressively).
    out = magnify_color_gdown_ideal(
        str(src), str(dst), alpha=30, level=2, fl=0.5, fh=1.5, chrom_attenuation=1.0
    )
    in_swing = clip.mean(axis=(1, 2, 3)).std()
    out_swing = out.mean(axis=(1, 2, 3)).std()
    assert out_swing > in_swing * 1.5


# --- motion pipeline -----------------------------------------------------


def test_iir_flat_input_unchanged(tmp_path: Path) -> None:
    src = tmp_path / "flat.mp4"
    dst = tmp_path / "out.mp4"
    _write_synth_video(src, _flat_clip())
    out = magnify_motion_lpyr_iir(
        str(src), str(dst), alpha=10, lambda_c=16, r1=0.4, r2=0.05, chrom_attenuation=0.1
    )
    assert np.abs(out - 0.5).max() < 0.05
