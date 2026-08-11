# Eulerian Video Magnification

[![Documentation](https://img.shields.io/badge/documentation-read-blue)](https://iamkucuk.github.io/eulerian-video-magnification-cuda/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)](#)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green?logo=nvidia&logoColor=white)](#)
[![OpenCL](https://img.shields.io/badge/OpenCL-Apple%20%7C%20AMD%20%7C%20Intel-orange)](#)
[![Metal](https://img.shields.io/badge/Metal-Apple-silver?logo=apple&logoColor=white)](#)
[![Vulkan](https://img.shields.io/badge/Vulkan-any%20vendor-red?logo=vulkan&logoColor=white)](#)
[![C++](https://img.shields.io/badge/C%2B%2B-17-orange?logo=c%2B%2B&logoColor=white)](#)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iamkucuk/eulerian-video-magnification-cuda/blob/main/colab/evm_cuda_benchmark.ipynb)
[![License: BSD-3-Clause-NC](https://img.shields.io/badge/License-BSD--3--NC-yellow.svg)](LICENSE)

**Amplify changes in video that are too small to see: the flush of blood
through a face with each heartbeat, the millimetre rise of a sleeping child's
chest, the vibration of a guitar string.** An implementation of
[Eulerian Video Magnification](http://people.csail.mit.edu/mrub/vidmag/),
checked against the original authors' published output, running on the
processor, on NVIDIA hardware through hand-written CUDA, and on Apple, AMD and
Intel hardware through OpenCL.

```bash
pip install evm-magnify
```

```python
import evm

evm.magnify("face.mp4", preset="pulse", out="pulse.mp4")
```

**[Documentation](https://iamkucuk.github.io/eulerian-video-magnification-cuda/)**
— installing, worked examples for pulse, vibration and motion, what each
parameter does, and what to do when the output looks identical to the input.

This project ports the MIT SIGGRAPH 2012 reference (Wu, Rubinstein, Freeman,
Durand, Guttag) from MATLAB to raw CUDA C++, and adds a portable OpenCL
implementation for hardware CUDA cannot reach. Both are compared against the
NumPy implementation, which is itself compared against the original authors'
published output.

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

Measured on a consumer RTX 3090 24GB against the NumPy implementation. Each
stage, including every H2D/D2H transfer, is timed with `cudaDeviceSynchronize`
(harness: `evm.cuda.benchmark`, 1 warmup, median of 7). We report the speedup
at three inclusion levels because they answer different questions.

The two motion rows were re-measured after three rounds of work on that path:
an FP32 IIR state, the band combine folded into the up_conv store, and shared
memory taken out of the two enlargement kernels. They are the median of three
such runs. Everything else is from the original session. Colour re-measured at
9.9 ms against the 9.7 ms below, so the two sessions agree to about 2% —
colour is the control here, since none of the three changes touches a pipeline
without a Laplacian pyramid.

**Read the ratios with the reference in mind.** The "Python CPU" column below
was measured on the machine that produced these numbers, and every ratio is
relative to it. The same GPU timings against an Apple M2 Max — 6,347 ms colour
and 28,303 ms motion for the same clips — give roughly 835x and 1,050x for
FP16 compute rather than 1,470x and 1,635x. The GPU side does not change; the
comparison does.

| Pipeline | Python CPU | ① Compute only | ② + H2D (inference) | ③ + H2D + D2H (full) |
|----------|-----------:|---------------:|--------------------:|---------------------:|
| Color FP32 (`face.mp4`) | 11,194 ms | 9.7 ms (~1,150x) | 29.1 ms (~385x) | 72.9 ms (~155x) |
| Color FP16 (`face.mp4`) | 11,194 ms | 7.6 ms (~1,470x) | 26.7 ms (~420x) | 71.0 ms (~160x) |
| Motion FP32 (`baby.mp4`) | 44,190 ms | 40.5 ms (~1,090x) | 74.1 ms (~595x) | 151.1 ms (~290x) |
| Motion FP16 (`baby.mp4`) | 44,190 ms | 27.0 ms (~1,635x) | 61.1 ms (~725x) | 138.7 ms (~320x) |

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

For motion, the inference speedup (②, ~605x FP16) is within about 2x of the
compute speedup (①, ~1,200x). The upload is a tax, but the GPU is still doing the
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
[blog_speedup.md](docs/internals/blog_speedup.md); the mid-pipeline arc (TN IIR, sticky
scratch, free-list pool, smem downsample) is in
[blog_further_optimizations.md](docs/internals/blog_further_optimizations.md).

### On hardware that is not NVIDIA

The portable OpenCL backend, on an Apple M2 Max, against the same machine's
processor cores. This hardware previously had no acceleration at all.

| Pipeline | Clip | Processor | Apple GPU | Ratio |
|---|---|---:|---:|---:|
| Colour | `face.mp4`, 301 frames | 6,347 ms | 222 ms | 28.6x |
| Motion | `baby.mp4`, 301 frames | 28,303 ms | 1,504 ms | 18.8x |

Details in `benches/apple_m2_max_opencl_2026-08-10.md`. AMD and Intel hardware
should work through the same kernels, but nobody has run it there, so no result
is claimed.

### Other NVIDIA GPUs (compute only)

| GPU | Motion FP32 / FP16 | Color face FP32 / FP16 |
|-----|-------------------:|-----------------------:|
| RTX 3090 24GB | 40.5 / 27.0 ms | 9.7 / 7.6 ms |
| A100 80GB † | 54.4 / 48.2 ms | 8.8 / 8.2 ms |
| H100 80GB † | 35.8 / 34.5 ms | 4.9 / 4.4 ms |
| P100 16GB | OOM / 139.7 ms | 26.3 / 21.8 ms |
| T4 16GB ‡ | OOM / 228.8 ms | 48.9 / 39.7 ms |

† Measured before two later rounds of work on the motion path and not yet
re-run: the up_conv change (smaller tiles, divide-free `reflect1`, even-tap
loop), then an FP32 IIR state and the band combine folded into the up_conv
store. The 3090 is the only row re-measured. Building the same tree with just
those last two changes reverted, and running both builds back to back on that
card, gives 76.67 ms against 47.92 ms FP32 (1.60x) and 60.91 ms against
36.83 ms FP16 (1.65x); the reverted build lands within 1.7% of the stored
`bench_rtx3090.json`, so that file is a fair record of the older code. None of
this is architecture specific, so the other motion figures are pessimistic by
roughly that much. Colour came out unchanged across the same back-to-back pair
(9.90 against 9.85 ms), which is the check that the numbers are comparable. Cross-GPU ratios below are left from
the pre-change measurement and will shift once the others are re-run.

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
| Motion FP16 vs CUDA FP32 (baby) | 0.00199 | 5 |
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
[`docs/blog_speedup.md`](docs/internals/blog_speedup.md) for the full optimization story.

## Quick start

```bash
# Setup — installs evm-cuda and its dev/build tooling into the venv.
# Works with or without nvcc; without it you get the CPU-only package.
python3 -m venv .venv && source .venv/bin/activate
make install-dev
make download          # fetch MIT sample videos

# Rebuild after editing cuda/ (compiles the GPU kernels when nvcc is present)
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
├── src/evm/              # the installed package (`import evm`)
│   ├── cpu/              # NumPy baseline (the correctness oracle)
│   ├── io/               # video decode/encode + the shared H.264 writer
│   └── cuda/             # GPU wrapper: batched, pipelines, benchmark
│                         #   (CMake writes _evm_cuda*.so in here)
├── src/evm_cuda/         # deprecated shim, forwards to evm.cuda
├── cuda/                 # CUDA sources
│   ├── kernels/          # .cu files (color, spatial, lpyr, iir, render...)
│   ├── bindings.cpp      # pybind11 + DeviceMemPool + sticky scratch
│   ├── CMakeLists.txt    # CUDA-optional, driven by scikit-build-core
│   └── DESIGN.md         # kernel map, tolerances, production path
├── docs/
│   ├── blog_speedup.md                 # first optimization writeup
│   ├── blog_further_optimizations.md   # layout, pool, smem (unified)
│   ├── img/                            # demo images
│   └── video/                          # Pages demo clips
├── scripts/              # CLI + profilers
├── tests/                # 32 Python + 60 CUDA cases (92 collected)
├── kaggle/               # free-GPU benchmark harness
├── pyproject.toml        # one distribution, `pip install .`, CUDA optional
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

[BSD 3-Clause (Non-Commercial Research Use)](LICENSE). Free to use for research
(including research inside a company), for teaching, for personal and evaluation
use, and inside open-source software that is distributed at no charge under a
licence permitting those same uses.

Selling it, or building it into a product or service that is sold or run for
commercial advantage, requires written permission. Any publication that uses
this software must cite it; the required citation is in the licence file.
