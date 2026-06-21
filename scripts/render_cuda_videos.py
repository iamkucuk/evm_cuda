#!/usr/bin/env python3
"""Render magnified output videos using the CUDA pipelines.

Produces four comparisons on the MIT samples, using the same parameters MIT
used in their published filenames so the outputs are directly comparable:

  face.mp4   -> face_color_cuda.mp4   (alpha=50, level=4, fl=0.8333, fh=1.0)
  baby.mp4   -> baby_motion_cuda.mp4  (alpha=10, lambda_c=16, r1=0.4, r2=0.05)

Also re-renders the Python-baseline versions side-by-side so you can A/B them.

Run on a GPU node. Outputs land in <repo>/output/cuda_render/.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CUDA_DIR = ROOT / "cuda"
for p in (str(ROOT), str(CUDA_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

import evm  # Python baseline  # noqa: E402
from evm_cuda import pipelines as cu  # CUDA pipelines  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "output" / "cuda_render"
OUT.mkdir(parents=True, exist_ok=True)


def rmse(a, b):
    return float(np.sqrt(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean()))


def run(label, fn, **kwargs):
    t0 = time.time()
    out = fn(str(DATA / kwargs.pop("src")), str(OUT / kwargs.pop("dst")), **kwargs)
    dt = time.time() - t0
    print(f"  [{label}] {dt:.1f}s  shape={out.shape}  "
          f"mean={out.mean():.4f}  range=[{out.min():.3f},{out.max():.3f}]")
    return out


def main():
    print(f"=== rendering to {OUT} ===\n")

    # face.mp4 — color (pulse) magnification, MIT params.
    print("face.mp4 — color magnification (alpha=50, level=4)")
    py_face = run("python",
        src="face.mp4", dst="face_color_python.mp4",
        fn=evm.magnify_color_gdown_ideal,
        alpha=50, level=4, fl=50/60, fh=60/60,
        chrom_attenuation=1.0, sampling_rate=30.0)
    cu_face = run("cuda  ",
        src="face.mp4", dst="face_color_cuda.mp4",
        fn=cu.magnify_color_gdown_ideal,
        alpha=50, level=4, fl=50/60, fh=60/60,
        chrom_attenuation=1.0, sampling_rate=30.0)
    print(f"  -> face RMSE(cuda, python) = {rmse(cu_face, py_face):.5f}\n")

    # baby.mp4 — motion (IIR) magnification, MIT params.
    print("baby.mp4 — motion magnification (alpha=10, lambda_c=16, r1=0.4, r2=0.05)")
    py_baby = run("python",
        src="baby.mp4", dst="baby_motion_python.mp4",
        fn=evm.magnify_motion_lpyr_iir,
        alpha=10, lambda_c=16, r1=0.4, r2=0.05, chrom_attenuation=0.1)
    cu_baby = run("cuda  ",
        src="baby.mp4", dst="baby_motion_cuda.mp4",
        fn=cu.magnify_motion_lpyr_iir,
        alpha=10, lambda_c=16, r1=0.4, r2=0.05, chrom_attenuation=0.1)
    print(f"  -> baby RMSE(cuda, python) = {rmse(cu_baby, py_baby):.5f}\n")

    # Side-by-side comparison PNGs at the peak-difference frame.
    _side_by_side(cu_face, py_face, "face", OUT / "face_compare.png")
    _side_by_side(cu_baby, py_baby, "baby", OUT / "baby_compare.png")

    print(f"=== outputs in {OUT} ===")
    for p in sorted(OUT.iterdir()):
        print(f"  {p.name:40s} {p.stat().st_size//1024} KB")


def _side_by_side(cu, py, name, out_path):
    """PNG: left=python baseline, right=CUDA, at the frame with the largest
    temporal deviation in the CUDA output (where the magnified signal peaks)."""
    import cv2
    dev = np.abs(cu.astype(np.float64) - cu.mean(axis=0, keepdims=True)).mean(axis=(1, 2, 3))
    peak = int(np.argmax(dev))
    cu_bgr = (np.clip(cu[peak], 0, 1) * 255).astype(np.uint8)[:, :, ::-1]
    py_bgr = (np.clip(py[peak], 0, 1) * 255).astype(np.uint8)[:, :, ::-1]
    side = np.concatenate([py_bgr, cu_bgr], axis=1)
    cv2.putText(side, "python baseline", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(side, "CUDA", (py_bgr.shape[1] + 10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imwrite(str(out_path), side)
    print(f"  -> side-by-side PNG: {out_path.name} (peak frame {peak})")


if __name__ == "__main__":
    main()
