# Eulerian Video Magnification (CUDA)

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](#)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green?logo=nvidia&logoColor=white)](#)
[![C++](https://img.shields.io/badge/C%2B%2B-17-orange?logo=c%2B%2B&logoColor=white)](#)
[![Tests](https://img.shields.io/badge/tests-92%20passed-brightgreen)](#)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iamkucuk/eulerian-video-magnification-cuda/blob/main/colab/evm_cuda_benchmark.ipynb)
[![License: BSD-3-Clause-NC](https://img.shields.io/badge/License-BSD--3--NC-yellow.svg)](LICENSE)

**[Eulerian Video Magnification](http://people.csail.mit.edu/mrub/vidmag/)** (EVM)
reveals invisible temporal changes in video — a person's pulse, a baby's
breathing, the vibration of machinery — by amplifying sub-pixel color and
motion variations that the eye cannot detect. This repository is an open-source
**CUDA C++** implementation of the MIT SIGGRAPH 2012 method (Wu, Rubinstein,
Freeman, Durand, Guttag), with pulse and motion demos, benchmarks, and writeups.

It ports the MIT reference from MATLAB to raw CUDA C++, achieving **~1,280×**
motion compute-only speedup (and **~2,500×** color) over the Python/NumPy
baseline on an NVIDIA H100, while matching the Python baseline within
end-to-end RMSE &lt; 0.01.

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

Current production path. Harness: `evm_cuda.benchmark` — **1 warmup + median of
7 timed runs**, `cudaDeviceSynchronize` per stage. Sources:
`benches/bench_truba_h100.json`, `benches/bench_truba_a100.json`,
`benches/bench_osiris_3090.json`, `benches/bench_kaggle_p100.json`.

Python CPU baselines: color **11,194 ms**, motion **44,190 ms**.

### H100-80GB

| Pipeline | ① Compute | ② + H2D | ③ + H2D+D2H | ① vs CPU |
|----------|----------:|--------:|------------:|---------:|
| Color FP32 (`face.mp4`) | **4.9 ms** | 34.6 ms | 103.6 ms | **~2,290×** |
| Color FP16 (`face.mp4`) | **4.4 ms** | 34.2 ms | 102.8 ms | **~2,540×** |
| Motion FP32 (`baby.mp4`) | **35.8 ms** | 82.1 ms | 196.0 ms | **~1,230×** |
| Motion FP16 (`baby.mp4`) | **34.5 ms** (**0.96×** FP32) | 81.0 ms | 189.3 ms | **~1,280×** |

- **①** kernels only · **②** upload + compute · **③** full H2D+D2H  
On H100, transfers dominate wall time.

### Other GPUs (compute only)

| GPU | Motion FP32 / FP16 | Color face FP32 / FP16 |
|-----|-------------------:|-----------------------:|
| **RTX 3090** 24GB | **90.4 / 75.1 ms** (0.83×) | **10.1 / 7.8 ms** (0.77×) |
| **A100** 80GB | **54.4 / 48.2 ms** (0.89×) | **8.8 / 8.2 ms** (0.93×) |
| **P100** 16GB | — | **31.6 / 27.1 ms** (0.86×) |

3090 also: color baby **15.8 / 12.2 ms** (0.77×). Motion compute FPS on 3090
(291 frames): FP32 **~3,220**, FP16 **~3,870**. P100 motion was OOM in the
multi-config harness; standalone motion FP16 peaks **~8–9 GB**.

### Accuracy

| Compare | RMSE | max LSB |
|---------|-----:|--------:|
| Motion FP16 vs CUDA FP32 (baby) | **0.00232** | **5** |
| Color FP16 vs CUDA FP32 (face) | **0.00071** | **1** |

End-to-end vs Python stays under RMSE &lt; 0.01.

Optimization history and stage-by-stage improvements:
[blog_speedup.md](docs/blog_speedup.md) ·
[blog_further_optimizations.md](docs/blog_further_optimizations.md).

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

[BSD 3-Clause (Non-Commercial Research Use)](LICENSE) — free for academic
research and non-commercial educational use. Commercial use requires written
permission. Citation is required for any derived publication.
