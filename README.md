# vidmag

**Eulerian Video Magnification. Hand-written CUDA at the core, five more
backends so it runs anywhere.**

[![Documentation](https://img.shields.io/badge/documentation-read-blue)](https://iamkucuk.github.io/eulerian-video-magnification-cuda/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)](#)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green?logo=nvidia&logoColor=white)](#)
[![OpenCL](https://img.shields.io/badge/OpenCL-Apple%20%7C%20AMD%20%7C%20Intel-orange)](#)
[![Metal](https://img.shields.io/badge/Metal-Apple-silver?logo=apple&logoColor=white)](#)
[![Vulkan](https://img.shields.io/badge/Vulkan-any%20vendor-red?logo=vulkan&logoColor=white)](#)
[![C++](https://img.shields.io/badge/C%2B%2B-17-orange?logo=c%2B%2B&logoColor=white)](#)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iamkucuk/eulerian-video-magnification-cuda/blob/main/scripts/cloud/colab_benchmark.ipynb)
[![License: BSD-3-Clause-NC](https://img.shields.io/badge/License-BSD--3--NC-yellow.svg)](LICENSE)

**Amplify changes in video that are too small to see: the flush of blood
through a face with each heartbeat, the millimetre rise of a sleeping child's
chest, the vibration of a guitar string.** An implementation of
[Eulerian Video Magnification](http://people.csail.mit.edu/mrub/vidmag/),
checked against the original authors' published output. The NVIDIA path is
hand-written CUDA and is what this project exists to make fast. Five more
backends — OpenCL, Vulkan, Apple's Metal, PyTorch, and the NumPy baseline —
mean it still runs when there is no NVIDIA card, and every one of them is
checked against that baseline.

```bash
pip install vidmag
```

```python
import vidmag

vidmag.magnify("face.mp4", preset="pulse", out="pulse.mp4")
```

**[Documentation](https://iamkucuk.github.io/eulerian-video-magnification-cuda/)**
— installing, worked examples for pulse, vibration and motion, what each
parameter does, and what to do when the output looks identical to the input.

This project ports the MIT SIGGRAPH 2012 reference (Wu, Rubinstein, Freeman,
Durand, Guttag) from MATLAB to raw CUDA C++ — no PyTorch, no CuPy, no Numba —
and adds four further implementations for hardware CUDA cannot reach: OpenCL,
Vulkan, Apple's Metal, and PyTorch. All five are compared against the NumPy
implementation, which is itself compared against the original authors'
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
(harness: `vidmag.cuda.benchmark`, 1 warmup, median of 7, a fresh process per
configuration). We report the speedup at three inclusion levels because they
answer different questions.

**Every figure in this section comes from one measurement session on
2026-08-18** and is stored in `benches/bench_rtx3090.json`, which
`scripts/dev/record_gpu_bench.py` regenerates in full. That matters because
these numbers previously accumulated across sessions months apart, and two of
them stopped being true: the processor column below read 11,194 ms for colour,
which is within 9% of the figure for `baby.mp4` rather than the `face.mp4` this
row is measured on, and it inflated every colour ratio by about 1.7 times.

**Read the ratios with the reference in mind.** The "Python CPU" column is the
NumPy implementation on the same machine, median of three runs, and every ratio
divides by it. It is not a stable quantity: successive runs of it on an idle
machine varied by about 14%, so treat the ratios as approximate and the
millisecond figures as the real result. Against a different processor the same
GPU timings give completely different ratios; the GPU side does not change.

| Pipeline | Python CPU | ① Compute only | ② + H2D (inference) | ③ + H2D + D2H (full) |
|----------|-----------:|---------------:|--------------------:|---------------------:|
| Color FP32 (`face.mp4`) | 5,585 ms | 9.8 ms (~570x) | 29.7 ms (~190x) | 77.1 ms (~72x) |
| Color FP16 (`face.mp4`) | 5,585 ms | 7.6 ms (~730x) | 27.9 ms (~200x) | 79.6 ms (~70x) |
| Motion FP32 (`baby.mp4`) | 31,981 ms | 40.3 ms (~790x) | 74.7 ms (~430x) | 154.1 ms (~210x) |
| Motion FP16 (`baby.mp4`) | 31,981 ms | 26.8 ms (~1,190x) | 61.2 ms (~520x) | 140.0 ms (~230x) |

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

All three are sums of per-stage timings, and none of them is what a caller
waits for. On an RTX 3090 the motion clip measures 40.3 ms at ①, 154.1 ms at ③,
and **232.2 ms** as wall-clock time through `vidmag.magnify()` — the extra 78 ms is
preparing the input array, the entry point's own checks, and allocating the
output, none of which a stage timer counts. Quote ① against another
implementation's kernel time and ③ against another implementation's stage sum;
comparing a stage sum against somebody's wall clock overstates this project by
about six times, and comparing across backends here needs wall clock for both,
because only the NVIDIA backend has per-stage timing.

For motion, the inference speedup (②, ~520x FP16) is within about 2.3x of the
compute speedup (①, ~1,190x). The upload is a tax, but the GPU is still doing the
work. For color, D2H alone is most of ③, so ② is the honest headline for any
real invocation that keeps the result on device. Motion FP16 uses the same
spatial kernel templates as FP32, with half-precision storage and
single-precision accumulation — the same algorithm. The shrinking kernel still
stages tiles in on-chip memory; the two enlargement kernels no longer do, which
was the third of the three changes below.

Color on `baby.mp4` (same clip as motion): FP32 15.4 / 54.7 / 133.8 ms and
FP16 11.4 / 47.9 / 135.6 ms for ① / ② / ③. The NumPy implementation does that
same colour job on `baby.mp4` in 10,280 ms, against 5,585 ms on `face.mp4` —
the two clips are not interchangeable, which is what the old figure got wrong.

### Throughput and real-time capacity

At the realistic *inference* tier (②, input upload included, output kept on the
GPU), pixel rates scale from the measured clips as:

| Pipeline (FP16) | Throughput (②) | ~ 1080p @ 30 fps |
|-----------------|---------------:|-----------------:|
| Color (face) | ~3.3 Gpx/s | ~52 streams |
| Motion (baby) | ~2.5 Gpx/s | ~40 streams |

The compute-only ceiling (①, data already on the GPU) is higher: about
~12 Gpx/s color (~190 streams) or ~5.7 Gpx/s motion (~90 streams). The full
decode→magnify→encode path (③) drops further and is usually PCIe-bound on the
download. 1080p@30 needs about 62.2 Mpx/s; stream counts are pixel-scaled
capacity, not a measured multi-stream harness.

Standalone motion FP16 peaks around 8 to 9 GB VRAM (measured), so it fits many
16 GB cards when the process is not holding residual allocations from other
configs. How this got from a correct-but-slow port to here — three rounds, what each
measured, and the two decisions a later round reversed — is in
[how it was made fast](docs/internals/making-it-fast.md).

### On hardware that is not NVIDIA

The portable OpenCL backend, on an Apple M2 Max, against the same machine's
processor cores. This hardware previously had no acceleration at all.

| Pipeline | Clip | Processor | Apple GPU | Ratio |
|---|---|---:|---:|---:|
| Colour | `face.mp4` | 7,014 ms | 217 ms | 32x |
| Motion | `baby.mp4` | 23,634 ms | 1,020 ms | 23x |

From the all-backends session of 2026-08-11, which is the one the documentation
site uses; see [Performance](docs/performance.md) for the other four backends on
the same machine. A single-backend session the day before gave different figures
(222 ms and 1,504 ms against a 6,347 ms and 28,303 ms processor baseline) and is
kept, superseded, in `benches/apple_m2_max_opencl_2026-08-10.md`.

**These Apple figures have not been re-verified since.** An attempt on
2026-08-18 could not reproduce them: the machine's graphics processor was at
100% utilisation from unrelated software, so every graphics backend measured
five to seven times slower while the processor baseline was only 18% slower.
That measurement proves nothing either way, and none of it was published.
`scripts/dev/record_backend_bench.py` regenerates this table on an idle machine.

AMD and Intel hardware should work through the same kernels, but nobody has run
it there, so no result is claimed.

### Other NVIDIA GPUs (compute only)

| GPU | Motion FP32 / FP16 | Color face FP32 / FP16 |
|-----|-------------------:|-----------------------:|
| H100 80GB (Hopper) ◊ | 17.0 / 13.8 ms | 4.2 / 3.7 ms |
| RTX 3090 24GB (Ampere) | 40.3 / 26.8 ms | 9.8 / 7.6 ms |
| P100 16GB (Pascal) | does not fit / 82.8 ms | 26.4 / 21.9 ms |
| T4 16GB (Turing) ‡ | does not fit / 137.2 ms | 43.2 / 38.6 ms |
| A100 80GB † | 54.4 / 48.2 ms | 8.8 / 8.2 ms |

**Only the A100 row is still on old code**, and all four other cards are four
different architectures. That matters, because three rounds of work on the
motion path sit between the current code and that row: the up_conv change
(smaller tiles, divide-free `reflect1`, even-tap loop), then an FP32 IIR state
with the band combine folded into the up_conv store, then shared memory taken
out of the two enlargement kernels.

Each of the four was measured before and after those rounds, on its own
hardware, and all four improve:

| | Motion FP16, before | after | Colour FP16, before | after |
|---|---:|---:|---:|---:|
| RTX 3090 (Ampere, sm_86) | 60.9 ms | 26.8 ms (2.3x) | 7.6 ms | 7.6 ms |
| P100 (Pascal, sm_60) | 139.7 ms | 82.8 ms (1.7x) | 21.8 ms | 21.9 ms |
| H100 (Hopper, sm_90) ◊ | 34.5 ms | 13.8 ms (2.5x) | 4.4 ms | 3.7 ms |
| T4 (Turing, sm_75) ‡ | 228.8 ms | 137.2 ms (1.7x) | 39.7 ms | 38.6 ms |

Colour is the control: it builds no Laplacian pyramid, so none of the three
changes can reach it, and any movement in the colour column is the measurement
rather than the code. On the RTX 3090 and the P100 it is flat, which makes those
two a controlled comparison. On the H100 and the T4 it is not, and each of those
two rows should be read with its own control in mind:

- **T4, colour moved 12%** in single precision (48.9 to 43.2 ms). Both T4 runs
  are single runs on Colab's shared hardware, so about 12% is that machine's
  noise floor. Motion moved 67%, well outside it.
- **H100, colour moved 15%.** The colour kernels are byte-identical between the
  two runs — the only change to those three files since is comment text, so this
  is entirely the environment, and the older run records no date or commit to
  narrow it further. Motion moved 150%, an order of magnitude outside it.

So on all four cards the direction and rough size hold; the exact factor is
trustworthy only for the RTX 3090 and the P100. Treat the remaining A100 motion
figure as pessimistic, by an amount nobody has measured.

† Measured before those three rounds and not re-run since, so the cross-GPU
ratios in this table mix two versions of the code and will shift once that card
is re-run. It could not be re-run on 2026-08-22: the cluster partition holding
the A100s was down for a reboot and not responding.

‡ One run of `scripts/cloud/colab_benchmark.ipynb` on Colab's shared hardware,
median of 5 rather than the 7 used elsewhere. Indicative only: repeated runs on
that class of machine move by tens of percent, as the colour control above
shows.

◊ Measured on a shared cluster node holding one of its four GPUs. The kernel
figures had a GPU to themselves; the transfer figures in `bench_h100.json` share
that node's PCIe with other jobs and are looser than the rest of that file.

Motion FP32 needs 16.3 GB and does not fit a 16 GB card — the P100 and the T4
both skip it and say so rather than failing partway. Motion FP16 peaks at 8.4 GB
and runs on both; it used to fail there only because the device pool held on to
every earlier config's memory.

Raw JSON in `benches/`. The four current ones record the date and commit they
were taken at: `bench_rtx3090.json` (2026-08-18, by
`scripts/dev/record_gpu_bench.py`), `bench_h100.json` (2026-08-22, by the same
script on a private HPC cluster node), `bench_p100.json` (2026-08-22, by
`scripts/cloud/kaggle/run_gpu_comparison.py`, with its console log in
`benches/kaggle_runs/`), and `bench_t4.json` (2026-08-22, transcribed by hand
from the Colab notebook's printed output, since that notebook writes no JSON of
its own). `bench_a100.json` is the pre-change measurement and records neither a
date nor a commit, which is why the marker on that row says only that it is
older.

### Accuracy

| Compare | RMSE | max LSB |
|---------|-----:|--------:|
| Motion FP16 vs CUDA FP32 (baby) | 0.00140 | 2 |
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
See [`src/vidmag/cuda/DESIGN.md`](src/vidmag/cuda/DESIGN.md) for the kernel-by-kernel mapping and
[how it was made fast](docs/internals/making-it-fast.md) for the full optimisation story.

## Quick start

```bash
# Setup — installs vidmag and its dev/build tooling into the venv.
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
make test              # 380 tests; 98 need an NVIDIA GPU and skip without one

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
- 380 tests covering every kernel, every backend against the NumPy
  baseline, end-to-end RMSE checks, and MIT reference comparison

## Project structure

```
.
├── src/vidmag/              # the installed package (`import vidmag`)
│   ├── api.py            # magnify() — the one entry point
│   ├── presets.py        # named parameter sets;  _cli.py — vidmag
│   ├── stream.py         # live magnification over a running capture
│   ├── backend/          # the backend interface, registry, generic pipelines
│   ├── cpu/              # NumPy baseline (the correctness oracle for the rest)
│   ├── cuda/             # NVIDIA: wrapper (batched, pipelines, benchmark, ops)
│   │   ├── kernels/      #   10 .cu files (color, spatial, lpyr, iir, render...)
│   │   ├── include/      #   shared device headers
│   │   ├── bindings.cpp  #   pybind11 + DeviceMemPool + sticky scratch
│   │   ├── CMakeLists.txt#   CUDA-optional, driven by scikit-build-core
│   │   └── DESIGN.md     #   kernel map, tolerances, production path
│   ├── opencl/           # kernels.cl + runtime, array, ops
│   ├── metal/            # kernels.metal + the same three
│   ├── vulkan/           # shaders/*.comp with committed *.spv + the same three
│   └── io/               # video decode/encode + the shared H.264 writer
├── docs/                 # the documentation site (mkdocs), incl. internals/
│                         #   with the two optimisation writeups, img/, video/
├── scripts/              # sample download, profilers, dev helpers
├── tests/                # 380 collected; 98 of them need an NVIDIA GPU
├── benches/              # stored benchmark results per GPU
├── colab/ and kaggle/    # free-GPU benchmark harnesses
├── pyproject.toml        # one distribution, `pip install .`, CUDA optional
└── Makefile              # build, test, run, profile targets
```

## Citation

If you use this work in your research, please cite it:

```bibtex
@misc{kucuk2026evm,
  title     = {Eulerian Video Magnification, on processors and any graphics hardware},
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
