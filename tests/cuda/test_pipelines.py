"""End-to-end CUDA pipeline tests vs the Python baseline.

Two flavours:
1. Synthetic-clip end-to-end — fast, deterministic, no external data.
2. face.mp4 / baby.mp4 comparison — the ultimate validation, gated on the
   MIT samples being present in data/ (same skipif as the Python suite).
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
CUDA_DIR = ROOT / "cuda"
for p in (str(ROOT), str(CUDA_DIR), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import evm  # noqa: E402
from conftest import TOL, have_cuda, skip_no_cuda  # noqa: E402

if have_cuda:
    from evm_cuda import pipelines as cu  # noqa: E402
    from evm_cuda import batched as cu_batched  # noqa: E402

DATA = ROOT / "data"
TMP = ROOT / "output" / "_test"


def _write_synth(path: Path, frames: np.ndarray, fps: float = 30.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t, h, w, _ = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, fps, (w, h), isColor=True)
    u8 = np.clip(np.round(frames * 255), 0, 255).astype(np.uint8)
    for i in range(t):
        vw.write(u8[i][:, :, ::-1])
    vw.release()


def _pulse_clip(t=60, h=32, w=32):
    n = np.arange(t)
    flicker = 0.08 * np.sin(2 * np.pi * 0.9 * n / 30.0)
    intensity = (0.5 + flicker).astype(np.float32)
    return intensity[:, None, None, None] * np.ones((t, h, w, 3), dtype=np.float32)


def _flat_clip(t=40, h=32, w=32):
    return np.full((t, h, w, 3), 0.5, dtype=np.float32)


def _rmse(a, b):
    return float(np.sqrt(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean()))


# --- synthetic end-to-end --------------------------------------------------


@skip_no_cuda
def test_color_pipeline_matches_python(tmp_path):
    src = tmp_path / "pulse.mp4"
    _write_synth(src, _pulse_clip())
    py = evm.magnify_color_gdown_ideal(
        str(src), str(tmp_path / "py.mp4"),
        alpha=30, level=2, fl=0.5, fh=1.5, chrom_attenuation=1.0)
    cu_out = cu.magnify_color_gdown_ideal(
        str(src), str(tmp_path / "cu.mp4"),
        alpha=30, level=2, fl=0.5, fh=1.5, chrom_attenuation=1.0)
    assert py.shape == cu_out.shape
    assert _rmse(py, cu_out) < TOL["end_to_end_rmse"]


@skip_no_cuda
def test_iir_pipeline_matches_python(tmp_path):
    src = tmp_path / "flat.mp4"
    _write_synth(src, _flat_clip())
    py = evm.magnify_motion_lpyr_iir(
        str(src), str(tmp_path / "py.mp4"),
        alpha=10, lambda_c=16, r1=0.4, r2=0.05, chrom_attenuation=0.1)
    cu_out = cu.magnify_motion_lpyr_iir(
        str(src), str(tmp_path / "cu.mp4"),
        alpha=10, lambda_c=16, r1=0.4, r2=0.05, chrom_attenuation=0.1)
    assert py.shape == cu_out.shape
    assert _rmse(py, cu_out) < TOL["end_to_end_rmse"]


# --- MIT samples (the ultimate validation) ---------------------------------

def _have(*names):
    return all((DATA / n).exists() for n in names)


@skip_no_cuda
@pytest.mark.skipif(
    not _have("face.mp4"),
    reason="download data/face.mp4 first (python scripts/download_samples.py face)",
)
def test_face_color_cuda_matches_python(tmp_path):
    """The CUDA color pipeline's output matches the Python baseline's,
    within the 0.01 RMSE end-to-end tolerance."""
    py = evm.magnify_color_gdown_ideal(
        str(DATA / "face.mp4"), str(tmp_path / "py.mp4"),
        alpha=50, level=4, fl=50/60, fh=60/60, chrom_attenuation=1.0,
        sampling_rate=30.0,
    )
    cu_out = cu.magnify_color_gdown_ideal(
        str(DATA / "face.mp4"), str(tmp_path / "cu.mp4"),
        alpha=50, level=4, fl=50/60, fh=60/60, chrom_attenuation=1.0,
        sampling_rate=30.0,
    )
    assert py.shape == cu_out.shape
    assert _rmse(py, cu_out) < TOL["end_to_end_rmse"]


@skip_no_cuda
@pytest.mark.skipif(
    not _have("baby.mp4"),
    reason="download data/baby.mp4 first",
)
def test_baby_iir_cuda_matches_python(tmp_path):
    py = evm.magnify_motion_lpyr_iir(
        str(DATA / "baby.mp4"), str(tmp_path / "py.mp4"),
        alpha=10, lambda_c=16, r1=0.4, r2=0.05, chrom_attenuation=0.1,
    )
    cu_out = cu.magnify_motion_lpyr_iir(
        str(DATA / "baby.mp4"), str(tmp_path / "cu.mp4"),
        alpha=10, lambda_c=16, r1=0.4, r2=0.05, chrom_attenuation=0.1,
    )
    assert py.shape == cu_out.shape
    assert _rmse(py, cu_out) < TOL["end_to_end_rmse"]


# --- optimized (batched) pipeline vs Python baseline -----------------------
#
# The batched pipeline (evm_cuda.batched) is the speed-optimized path. It
# uses a different code flow (to_planar_3ch, batched_blur_dn_color, CUDA
# bilinear upsample, device-resident NTSC) but must produce the same output
# as the Python baseline within the end-to-end RMSE tolerance.


@skip_no_cuda
def test_batched_color_matches_python_synth(tmp_path):
    """Batched color pipeline vs Python baseline on a synthetic pulse clip."""
    src = tmp_path / "pulse.mp4"
    _write_synth(src, _pulse_clip())
    py = evm.magnify_color_gdown_ideal(
        str(src), str(tmp_path / "py.mp4"),
        alpha=30, level=2, fl=0.5, fh=1.5, chrom_attenuation=1.0)
    batched = cu_batched.magnify_color_gdown_ideal(
        str(src), str(tmp_path / "batched.mp4"),
        alpha=30, level=2, fl=0.5, fh=1.5, chrom_attenuation=1.0)
    assert py.shape == batched.shape
    assert _rmse(py, batched) < TOL["end_to_end_rmse"]


@skip_no_cuda
@pytest.mark.skipif(
    not _have("face.mp4"),
    reason="download data/face.mp4 first (python scripts/download_samples.py face)",
)
def test_batched_color_matches_python_face(tmp_path):
    """The OPTIMIZED batched color pipeline vs Python baseline on face.mp4.

    This is the critical end-to-end validation for the speed-optimized path:
    it exercises every Phase 1c-1h change (planar transpose, batched blur_dn,
    CUDA bilinear upsample, device-resident NTSC, cudaMalloc warmup) and must
    still match the baseline within RMSE < 0.01."""
    py = evm.magnify_color_gdown_ideal(
        str(DATA / "face.mp4"), str(tmp_path / "py.mp4"),
        alpha=50, level=4, fl=50/60, fh=60/60, chrom_attenuation=1.0,
        sampling_rate=30.0,
    )
    batched = cu_batched.magnify_color_gdown_ideal(
        str(DATA / "face.mp4"), str(tmp_path / "batched.mp4"),
        alpha=50, level=4, fl=50/60, fh=60/60, chrom_attenuation=1.0,
        sampling_rate=30.0,
    )
    assert py.shape == batched.shape
    assert _rmse(py, batched) < TOL["end_to_end_rmse"]


# --- FP16 motion (batched) --------------------------------------------------
# Storage is half; compute stays FP32 in spatial kernels and FP64 in IIR.
# Must match the Python FP32 oracle within the same end-to-end RMSE budget.


@skip_no_cuda
def test_batched_motion_fp16_matches_python_synth(tmp_path):
    """Batched FP16 motion pipeline vs Python FP32 baseline on a flat synth clip."""
    src = tmp_path / "flat.mp4"
    _write_synth(src, _flat_clip())
    py = evm.magnify_motion_lpyr_iir(
        str(src), str(tmp_path / "py.mp4"),
        alpha=10, lambda_c=16, r1=0.4, r2=0.05, chrom_attenuation=0.1)
    cu16 = cu_batched.magnify_motion_lpyr_iir_fp16(
        str(src), str(tmp_path / "cu16.mp4"),
        alpha=10, lambda_c=16, r1=0.4, r2=0.05, chrom_attenuation=0.1)
    assert py.shape == cu16.shape
    assert _rmse(py, cu16) < TOL["end_to_end_rmse"]


@skip_no_cuda
@pytest.mark.skipif(not _have("baby.mp4"), reason="baby.mp4 not in data/")
def test_batched_motion_fp16_matches_python_baby(tmp_path):
    """Batched FP16 motion pipeline vs Python FP32 baseline on baby.mp4."""
    py = evm.magnify_motion_lpyr_iir(
        str(DATA / "baby.mp4"), str(tmp_path / "py.mp4"),
        alpha=20, lambda_c=16, r1=0.4, r2=0.05, chrom_attenuation=0.1)
    cu16 = cu_batched.magnify_motion_lpyr_iir_fp16(
        str(DATA / "baby.mp4"), str(tmp_path / "cu16.mp4"),
        alpha=20, lambda_c=16, r1=0.4, r2=0.05, chrom_attenuation=0.1)
    assert py.shape == cu16.shape
    assert _rmse(py, cu16) < TOL["end_to_end_rmse"]


@skip_no_cuda
@pytest.mark.skipif(not _have("baby.mp4"), reason="baby.mp4 not in data/")
def test_batched_motion_fp16_close_to_fp32_baby(tmp_path):
    """FP16 motion storage vs FP32 motion CUDA on baby.mp4.

    DESIGN notes FP16/FP32 motion RMSE ~0.0016 and max per-pixel error around
    a few uint8 steps. Bound to 5 LSB (measured peak on 3090); RMSE still
    under the end-to-end 0.01 gate.
    """
    fp32 = cu_batched.magnify_motion_lpyr_iir(
        str(DATA / "baby.mp4"), str(tmp_path / "fp32.mp4"),
        alpha=20, lambda_c=16, r1=0.4, r2=0.05, chrom_attenuation=0.1)
    fp16 = cu_batched.magnify_motion_lpyr_iir_fp16(
        str(DATA / "baby.mp4"), str(tmp_path / "fp16.mp4"),
        alpha=20, lambda_c=16, r1=0.4, r2=0.05, chrom_attenuation=0.1)
    assert fp32.shape == fp16.shape
    a = np.clip(np.rint(fp32 * 255.0), 0, 255).astype(np.int16)
    b = np.clip(np.rint(fp16 * 255.0), 0, 255).astype(np.int16)
    diff = np.abs(a - b)
    assert int(diff.max()) <= 5, f"FP16 vs FP32 max LSB {diff.max()} > 5"
    assert _rmse(fp32, fp16) < TOL["end_to_end_rmse"]
