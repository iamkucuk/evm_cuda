# Eulerian Video Magnification (CUDA)

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](#)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green?logo=nvidia&logoColor=white)](#)
[![C++](https://img.shields.io/badge/C%2B%2B-17-orange?logo=c%2B%2B&logoColor=white)](#)
[![Tests](https://img.shields.io/badge/tests-92%20passed-brightgreen)](#)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iamkucuk/eulerian-video-magnification-cuda/blob/main/colab/evm_cuda_benchmark.ipynb)
[![License: BSD-3-Clause-NC](https://img.shields.io/badge/License-BSD--3--NC-yellow.svg)](LICENSE)

**A CUDA-accelerated implementation of [Eulerian Video Magnification](http://people.csail.mit.edu/mrub/vidmag/) (EVM) that
reveals invisible temporal changes in video by amplifying sub-pixel color and
motion variations that the eye cannot detect.**

This project ports the MIT SIGGRAPH 2012 reference (Wu, Rubinstein, Freeman,
Durand, Guttag) from MATLAB to raw CUDA C++. On a consumer RTX 3090 it reaches
about 730x compute-only speedup on motion (FP16) and about 1,470x on color
(FP16) over the Python/NumPy baseline, while matching that baseline within
end-to-end RMSE < 0.01.

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

Measured on a consumer RTX 3090 24GB vs the Python/NumPy CPU baseline. Each
stage, including every H2D/D2H transfer, is timed with `cudaDeviceSynchronize`
(harness: `evm_cuda.benchmark`, 1 warmup, median of 7). We report the speedup
at three inclusion levels because they answer different questions:

| Pipeline | Python CPU | ① Compute only | ② + H2D (inference) | ③ + H2D + D2H (full) |
|----------|-----------:|---------------:|--------------------:|---------------------:|
| Color FP32 (`face.mp4`) | 11,194 ms | 9.7 ms (~1,150x) | 29.1 ms (~385x) | 72.9 ms (~155x) |
| Color FP16 (`face.mp4`) | 11,194 ms | 7.6 ms (~1,470x) | 26.7 ms (~420x) | 71.0 ms (~160x) |
| Motion FP32 (`baby.mp4`) | 44,190 ms | 75.4 ms (~590x) | 113.0 ms (~390x) | 188.4 ms (~235x) |
| Motion FP16 (`baby.mp4`) | 44,190 ms | 60.4 ms (~730x) | 92.4 ms (~480x) | 164.8 ms (~270x) |

- **① Compute only** is pure kernel time (data already on the GPU), e.g. as one
  stage inside a larger device-resident graph.
- **② + H2D** is the realistic *inference* cost: the input starts on the host,
  so you pay the upload. The output D2H is deliberately left out. In most real
  uses the amplified signal is consumed on the GPU (heart-rate estimation,
  motion features, a downstream net). You do not need a viewable video on the
  host to extract information from it. This is the cost of an embedded EVM
  stage in a GPU pipeline.
- **③ + H2D + D2H** is the full standalone "decode → magnify → encode" path,
  when you must materialize a viewable video on the host.

For motion, the inference speedup (②, ~480x FP16) is within about 1.5x of the
compute speedup (①, ~730x). The upload is a tax, but the GPU is still doing the
work. For color, D2H alone is most of ③, so ② is the honest headline for any
real invocation that keeps the result on device. Motion FP16 uses the same
spatial templates as FP32 (dense half smem, float MAC); half storage, same
algorithm.

Color on `baby.mp4` (same clip as motion): FP32 15.3 / 48.2 / 120.5 ms and
FP16 11.4 / 44.3 / 117.2 ms for ① / ② / ③.

### Throughput and real-time capacity

At the realistic *inference* tier (②, input upload included, output kept on the
GPU), pixel rates scale from the measured clips as:

| Pipeline (FP16) | Throughput (②) | ~ 1080p @ 30 fps |
|-----------------|---------------:|-----------------:|
| Color (face) | ~3.4 Gpx/s | ~55 streams |
| Motion (baby) | ~1.6 Gpx/s | ~26 streams |

The compute-only ceiling (①, data already on the GPU) is higher: about
~12 Gpx/s color (~190 streams) or ~2.5 Gpx/s motion (~40 streams). The full
decode→magnify→encode path (③) drops further and is usually PCIe-bound on the
download. 1080p@30 needs about 62.2 Mpx/s; stream counts are pixel-scaled
capacity, not a measured multi-stream harness.

Standalone motion FP16 peaks around 8 to 9 GB VRAM (measured), so it fits many
16 GB cards when the process is not holding residual allocations from other
configs. Per-stage breakdown and multi-GPU numbers (A100 / H100 / P100) live in
[blog_speedup.md](docs/blog_speedup.md); the mid-pipeline arc (TN IIR, sticky
scratch, free-list pool, smem downsample) is in
[blog_further_optimizations.md](docs/blog_further_optimizations.md).

### Other GPUs (compute only)

| GPU | Motion FP32 / FP16 | Color face FP32 / FP16 |
|-----|-------------------:|-----------------------:|
| RTX 3090 24GB | 75.4 / 60.4 ms | 9.7 / 7.6 ms |
| A100 80GB † | 54.4 / 48.2 ms | 8.8 / 8.2 ms |
| H100 80GB † | 35.8 / 34.5 ms | 4.9 / 4.4 ms |
| P100 16GB | OOM / 139.7 ms | 26.3 / 21.8 ms |
| T4 16GB ‡ | OOM / 228.8 ms | 48.9 / 39.7 ms |

† Measured before the up_conv work (smaller tiles, divide-free `reflect1`,
even-tap loop) and not yet re-run. That change is worth about 1.2x on motion
compute on the 3090 and nothing about it is architecture specific, so those
two rows are pessimistic by roughly that much. Cross-GPU ratios below are
left from the pre-change measurement and will shift once the others are re-run.

‡ One run of `colab/evm_cuda_benchmark.ipynb` on Colab's shared hardware, with
no stored JSON. Indicative only: repeated runs on that class of machine moved
by tens of percent.

P100 and T4 were measured on main. Motion FP32 needs 16.3 GB and does not fit a
16 GB card. Motion FP16 peaks at 8.4 GB and now runs on both; it used to fail
there only because the device pool held on to every earlier config's memory.
Raw JSON:
`benches/bench_rtx3090.json`, `bench_a100.json`, `bench_h100.json`,
`bench_p100.json`.

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
- 92 tests (67 functions, parametrized to 92 cases) covering every kernel,
  end-to-end RMSE checks, and MIT reference comparison

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
