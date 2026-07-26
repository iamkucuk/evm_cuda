# Eulerian Video Magnification (CUDA)

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](#)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green?logo=nvidia&logoColor=white)](#)
[![C++](https://img.shields.io/badge/C%2B%2B-17-orange?logo=c%2B%2B&logoColor=white)](#)
[![Tests](https://img.shields.io/badge/tests-92%20passed-brightgreen)](#)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iamkucuk/eulerian-video-magnification-cuda/blob/main/colab/evm_cuda_benchmark.ipynb)
[![License: BSD-3-Clause-NC](https://img.shields.io/badge/License-BSD--3--NC-yellow.svg)](LICENSE)

**[Eulerian Video Magnification](http://people.csail.mit.edu/mrub/vidmag/)** (EVM)
reveals invisible temporal changes in video by amplifying sub-pixel color and
motion variations that the eye cannot detect. This repository is an open-source
**CUDA C++** implementation of the MIT SIGGRAPH 2012 method (Wu, Rubinstein,
Freeman, Durand, Guttag), with pulse and motion demos, benchmarks, and writeups.

It ports the MIT reference from MATLAB to raw CUDA C++. On a consumer RTX 3090,
motion compute is about 75 ms (FP16) for baby.mp4: roughly 3,900 FPS of pure
compute, or about 30 concurrent 1080p@30 streams if the frames are already on
the GPU. Output matches the Python baseline within end-to-end RMSE < 0.01.

---

### Pulse magnification (color pipeline)

<p align="center">
  <img src="docs/img/face_demo.gif" alt="Pulse magnification: blood flow becomes visible" width="600">
</p>

<p align="center"><sub>Left: original. Right: amplified. The green tint shows amplified
blood flow. Each heartbeat causes sub-pixel skin color changes that EVM makes visible.</sub></p>

### Motion magnification (IIR pipeline)

<p align="center">
  <img src="docs/img/baby_demo.gif" alt="Motion magnification: subtle breathing amplified" width="600">
</p>

<p align="center"><sub>Left: original. Right: amplified. Submillimeter chest movements
from breathing are amplified to be clearly visible, enabling non-contact vital sign monitoring.</sub></p>

---

## Performance

Primary numbers are from a consumer RTX 3090 24GB (the common card for this
workload). Each stage, including every H2D/D2H transfer, is timed with
`cudaDeviceSynchronize`; harness is `evm_cuda.benchmark` (1 warmup, median of
7). We report three inclusion levels because they answer different questions:

| Pipeline | ① Compute only | ② + H2D | ③ + H2D + D2H | ① vs CPU |
|----------|---------------:|--------:|--------------:|---------:|
| Motion FP32 (`baby.mp4`) | 90.4 ms | 128.0 ms | 203.5 ms | ~490x |
| Motion FP16 (`baby.mp4`) | 75.1 ms | 107.1 ms | 179.5 ms | ~590x |
| Color FP32 (`face.mp4`) | 10.1 ms | 29.5 ms | 73.2 ms | ~1,110x |
| Color FP16 (`face.mp4`) | 7.8 ms | 26.9 ms | 71.1 ms | ~1,440x |
| Color FP32 (`baby.mp4`) | 15.8 ms | 48.6 ms | 121.0 ms | ~710x |
| Color FP16 (`baby.mp4`) | 12.2 ms | 45.0 ms | 117.9 ms | ~920x |

Python CPU baselines: color 11,194 ms, motion 44,190 ms.

- **① Compute only** is pure kernel time (data already on the GPU). Useful when
  EVM is one stage inside a larger device-resident graph.
- **② + H2D** is the realistic *inference* cost: the input starts on the host,
  so you pay the upload. The output D2H is deliberately left out. In most real
  uses the amplified signal is consumed on the GPU (heart-rate estimation,
  motion features, a downstream net). You do not need a viewable video on the
  host to extract information from it.
- **③ + H2D + D2H** is the full standalone "decode, magnify, encode" path, when
  you must materialize a viewable video on the host.

For motion, ② is within about 1.4x of ① (107 vs 75 ms FP16). The upload is a
tax, but the GPU is still doing the work. For color, D2H alone is most of ③, so
② is the honest headline for any real invocation that keeps the result on
device. Motion FP16 uses the same spatial templates as FP32 (dense half smem,
float MAC); the FP16 column is the same algorithm with half storage.

### Throughput and real-time capacity

Pixel rates are scaled from the measured clips (face 291x592x528, baby
291x960x544). 1080p@30 needs about 62.2 Mpx/s. Stream counts below are
**compute-only** (data already on the GPU). They are not file-to-file capacity;
decode, encode, and PCIe still dominate those paths.

| Pipeline (FP16, ①) | Throughput | ~ concurrent 1080p@30 streams |
|--------------------|-----------:|------------------------------:|
| Motion | ~2.0 Gpx/s | ~30 |
| Color (face) | ~12 Gpx/s | ~190 |

At the inference tier (②), stream counts drop because the upload is in the
budget. The full path (③) drops further and is usually PCIe-bound on the
download. A single baby-class motion clip is still roughly 100x real-time in
pure compute; that is the GPU ceiling, not a promise about concurrent
file-to-file jobs.

Standalone motion FP16 peaks around 8 to 9 GB VRAM (measured), so it fits 16 GB
cards when the harness does not leave residual pool allocations from other
configs.

### Other GPUs (compute only)

| GPU | Motion FP32 / FP16 | Color face FP32 / FP16 | Motion FP16 streams @1080p30 |
|-----|-------------------:|-----------------------:|-----------------------------:|
| RTX 3090 24GB | 90.4 / 75.1 ms | 10.1 / 7.8 ms | ~30 |
| A100 80GB | 54.4 / 48.2 ms | 8.8 / 8.2 ms | ~50 |
| H100 80GB | 35.8 / 34.5 ms | 4.9 / 4.4 ms | ~70 |
| P100 16GB |  | 31.6 / 27.1 ms | (color ~50) |

Relative to 3090 motion FP16 compute: A100 is about 1.6x faster, H100 about
2.2x. Color on A100 is roughly the same wall time as 3090; H100 is about 1.8x
faster. P100 is the color-only reference here (motion OOM'd in the
multi-config harness).

Raw JSON: `benches/bench_rtx3090.json`, `bench_a100.json`, `bench_h100.json`,
`bench_p100.json`. Per-stage breakdown and the optimization arc live in
[blog_speedup.md](docs/blog_speedup.md) and
[blog_further_optimizations.md](docs/blog_further_optimizations.md).

### Accuracy

| Compare | RMSE | max LSB |
|---------|-----:|--------:|
| Motion FP16 vs CUDA FP32 (baby) | 0.00232 | 5 |
| Color FP16 vs CUDA FP32 (face) | 0.00071 | 1 |

End-to-end vs Python stays under RMSE < 0.01.

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
make test              # 92 tests (32 Python baseline + 60 CUDA parametrized)

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

No PyTorch, no CuPy, no Numba. Every kernel is hand-written CUDA C++.

## Architecture highlights

- Device-resident pipeline: the entire clip is staged to GPU memory once;
  all 50+ kernel launches execute without a single host-device round-trip
- Batched spatial kernels: `grid.z = M` collapses ~35,000 launches into ~50
- cuFFT plan caching eliminates per-call autotuning overhead
- Templated FP16 storage: all kernels compile in both FP32 and FP16 variants
  via `cvt_in`/`cvt_out` helpers; compute stays FP32, storage halves
- Multiple-elements-per-thread render and transpose kernels process
  4 pixels per thread to pipeline independent memory reads (22% speedup)
- **92 tests** (67 functions, parametrized to 92 cases) validating every kernel
  end-to-end RMSE checks and MIT reference output comparison

## Project structure

```
evm_cuda/
├── evm/                  # Python baseline (the correctness oracle)
├── cuda/                 # CUDA port
│   ├── kernels/          # .cu files (color, spatial, lpyr, iir, render...)
│   ├── evm_cuda/         # Python package: batched.py, pipelines, benchmark
│   ├── bindings.cpp      # pybind11 + DeviceMemPool + sticky scratch
│   └── DESIGN.md         # kernel map, tolerances, production path
├── docs/
│   ├── blog_speedup.md                 # first optimization writeup
│   ├── blog_further_optimizations.md   # layout, pool, smem (unified)
│   ├── img/                            # demo images
│   └── video/                          # Pages demo clips
├── scripts/              # CLI + profilers
├── tests/                # 32 Python + 60 CUDA cases (92 collected)
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

[BSD 3-Clause (Non-Commercial Research Use)](LICENSE), free for academic
research and non-commercial educational use. Commercial use requires written
permission. Citation is required for any derived publication.
