# Eulerian Video Magnification on CUDA

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](#)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green?logo=nvidia&logoColor=white)](#)
[![C++](https://img.shields.io/badge/C%2B%2B-17-orange?logo=c%2B%2B&logoColor=white)](#)
[![Tests](https://img.shields.io/badge/tests-61%20passed-brightgreen)](#)
[![Speedup](https://img.shields.io/badge/speedup-557x-success)](#)
[![License: BSD-3-Clause-NC](https://img.shields.io/badge/License-BSD--3--NC-yellow.svg)](LICENSE)

**A CUDA-accelerated implementation of Eulerian Video Magnification (EVM) that
reveals invisible temporal changes in video — a person's pulse, a baby's
breathing, the vibration of machinery — by amplifying sub-pixel color and
motion variations that the eye cannot detect.**

This project ports the MIT SIGGRAPH 2012 reference implementation from
MATLAB to raw CUDA C++, achieving **557x compute-only speedup** (273x full
pipeline including H2D/D2H transfers) over the Python/NumPy baseline on an
NVIDIA H100, while producing bit-for-bit equivalent output (RMSE < 0.01).

---

### Pulse magnification (color pipeline)

<p align="center">
  <img src="docs/img/face_demo.gif" alt="Pulse magnification: blood flow becomes visible" width="600">
</p>

<p align="center"><sub>Left: original. Right: amplified. The green tint shows amplified
blood flow — each heartbeat causes sub-pixel skin color changes that EVM makes visible.</sub></p>

### Motion magnification (IIR pipeline)

<p align="center">
  <img src="docs/img/baby_demo.gif" alt="Motion magnification: subtle breathing amplified" width="600">
</p>

<p align="center"><sub>Left: original. Right: amplified. Submillimeter chest movements
from breathing are amplified to be clearly visible, enabling non-contact vital sign monitoring.</sub></p>

---

## Performance

Measured on an NVIDIA H100-80GB (sm_90) and the Python/NumPy CPU baseline.
Each stage — including every H2D/D2H transfer — is timed separately with
`cudaDeviceSynchronize`, median of 5 runs after one warmup. The speedup is
reported at three inclusion levels because transfers are a comparable cost to
compute:

| Pipeline | Python CPU | GPU compute | + H2D | + H2D + D2H (full) |
|----------|-----------:|------------:|------:|-------------------:|
| Color FP32 | 11,194 ms | 9 ms (**1,302x**) | 47 ms (238x) | 119 ms (94x) |
| Motion FP32 | 44,190 ms | 85 ms (**522x**) | 135 ms (329x) | 162 ms (**273x**) |
| Motion FP16 | 44,190 ms | 79 ms (**557x**) | 141 ms (314x) | 183 ms (242x) |

- **compute-only** = kernels (data already on GPU) — the headline speedup.
- **+ H2D** = realistic cost of feeding the GPU from host memory.
- **+ H2D + D2H** = full accelerator-offload cost, reading the result back.

On color, transfers dominate total time (the output D2H alone is 66 ms), so
the realistic full speedup is ~94x. On motion, compute is heavier and the full
speedup stays ~273x. FP16 is a wash on color (bandwidth-bound transfers) but
the fastest compute-only motion (557x).

FP16 motion fits in 12 GB VRAM (down from 23 GB), running on 16 GB GPUs like
the Tesla P100 and T4. Full per-stage breakdown (with transfers) and the
multi-GPU (P100/A100/H100) comparison in the
[optimization writeup](docs/blog_speedup.md).

## How it works

Every EVM variant follows the same four-stage pipeline:

```
input video (T frames, H x W, RGB)
   |
   1. COLOR    BGR u8 -> NTSC YIQ float (per-pixel matrix multiply)
   2. SPATIAL  Gaussian downsample (color) OR Laplacian pyramid (motion)
   3. TEMPORAL Bandpass filter along time (FFT / Butterworth / IIR)
   4. AMPLIFY  Multiply by alpha, add back, render to RGB
   |
output video (magnified)
```

The CUDA port implements each stage as one or more kernels, with the entire
pipeline running device-resident (zero per-frame host-device transfers).
See [`cuda/DESIGN.md`](cuda/DESIGN.md) for the kernel-by-kernel mapping and
[`docs/blog_speedup.md`](docs/blog_speedup.md) for the full optimization story.

## Quick start

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make download          # fetch MIT sample videos

# Build (needs CUDA toolkit + nvcc)
make build

# Run
make run-color         # pulse magnification on face.mp4
make run-motion        # motion magnification on baby.mp4

# Test
make test              # 61 tests (25 Python baseline + 36 CUDA)

# Profile
make profile           # CPU vs FP32 vs FP16 comparison
make help              # all targets
```

## Tech stack

| Layer | Technology | Why |
|-------|-----------|-----|
| GPU kernels | CUDA C++ (raw nvcc) | Maximum control, no framework overhead |
| Python bindings | pybind11 | Thin, zero-copy device pointer passing |
| Build | CMake + Ninja | Standard, portable |
| FFT | cuFFT (batched C2C) | Hardware-accelerated temporal filtering |
| Color | OpenCV (VideoCapture) | Input video decode |
| Encode | PyAV (libx264) | H.264 yuv420p +faststart output (browser/VSCode-playable) |
| Compute | NumPy / SciPy (baseline) | The correctness oracle |

No PyTorch, no CuPy, no Numba — every kernel is hand-written CUDA C++.

## Architecture highlights

- **Device-resident pipeline** — the entire clip is staged to GPU memory once;
  all 50+ kernel launches execute without a single host-device round-trip
- **Batched spatial kernels** — `grid.z = M` collapses ~35,000 launches into ~50
- **cuFFT plan caching** — eliminates per-call autotuning overhead
- **Templated FP16 storage** — all kernels compile in both FP32 and FP16 variants
  via `cvt_in`/`cvt_out` helpers; compute stays FP32, storage halves
- **Multiple-elements-per-thread** — render and transpose kernels process
  4 pixels per thread to pipeline independent memory reads (22% speedup)
- **61 tests** validating every kernel against the Python baseline, including
  end-to-end RMSE checks and MIT reference output comparison

## Project structure

```
evm_cuda/
├── evm/                  # Python baseline (the correctness oracle)
├── cuda/                 # CUDA port
│   ├── kernels/          # 10 .cu files (color, spatial, lpyr, iir, render...)
│   ├── bindings.cpp      # pybind11 module
│   ├── evm_cuda/         # Python wrapper (pipelines, DeviceBuffer)
│   └── DESIGN.md         # kernel-by-kernel mapping + tolerance contract
├── docs/
│   ├── blog_speedup.md   # full optimization writeup
│   └── img/              # demo images
├── scripts/              # CLI + profilers
├── tests/                # 25 Python + 36 CUDA tests
├── kaggle/               # free-GPU benchmark harness
└── Makefile              # build, test, run, profile targets
```

## Citation

If you use this work in your research, please cite it:

```bibtex
@misc{kucuk2026evm_cuda,
  title     = {Eulerian Video Magnification on {CUDA}},
  author    = {Kucuk, Furkan},
  year      = {2026},
  url       = {https://github.com/iamkucuk/eulerian-video-magnification-cuda},
}
```

This project builds on the original EVM work:

> Wu, Rubinstein, Freeman, Durand, Guttag. "Eulerian Video Magnification for
> Revealing Subtle Changes in the World." SIGGRAPH 2012.
> <http://people.csail.mit.edu/mrub/vidmag/>

## License

[BSD 3-Clause (Non-Commercial Research Use)](LICENSE) — free for academic
research and non-commercial educational use. Commercial use requires written
permission. Citation is required for any derived publication.
