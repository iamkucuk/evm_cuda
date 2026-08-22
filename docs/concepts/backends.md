# Backends and hardware

The same magnification is implemented six times, but they are not six equals.
**The NVIDIA backend is what this project is for** — hand-written CUDA C++ with
no tensor framework underneath, where optimisation work starts, and whose
figures are on [Performance](../performance.md). The other five exist so the
library still works when there is no NVIDIA card, and so hardware appearing later
has a path that needs no new code here.

Which one runs is chosen automatically, and never changed behind your back.

| Backend | Runs on | Written in | Needs |
|---|---|---|---|
| `cuda` | NVIDIA graphics processors | Hand-written CUDA, tuned | a compiler at install |
| `metal` | Apple graphics processors | Metal compute kernels | `[metal]` |
| `vulkan` | Anything with a Vulkan driver: AMD, Intel, NVIDIA, Apple via a translation layer, mobile/embedded | GLSL compiled to SPIR-V | `[vulkan]` |
| `opencl` | Anything with an OpenCL driver: Apple, AMD, Intel, NVIDIA; also processors via a software driver | OpenCL kernels | `[opencl]` |
| `torch` | Wherever PyTorch runs: NVIDIA, Apple, processors | PyTorch tensor operations | `[torch]` |
| `cpu` | Anything | NumPy — the reference the others are checked against | nothing |

The overlap is deliberate: on an Apple laptop, four of them work. Which
interface a device supports is not something this project controls, and it
changes — Apple has deprecated OpenCL, Vulkan ships on most new hardware, Metal
is Apple's path forward. Covering several means a later device needs no new code
here.

Only `cuda` and `cpu` install by default; the rest are optional extras, so
nothing is downloaded for hardware you do not have:

```bash
pip install "vidmag[metal]"      # or [vulkan], [opencl], [torch]
```

## How one is chosen

`backend="auto"`, the default, tries them in the table's order and uses the
first that can run: tuned NVIDIA, then Metal, Vulkan, OpenCL, PyTorch, then the
processor. Where several would work, the earlier one is the shorter path to the
hardware — Vulkan on a Mac runs through a translation layer onto Metal, so using
Metal directly removes a layer. PyTorch sits second-to-last because it reaches no
hardware the others miss, so it is never preferred over a native backend for
whole-clip work.

```python
from vidmag import backend

name, _ = backend.select("auto")
print(name)
```

Naming one is honoured exactly. If it cannot run, the call raises and says why;
it is never quietly replaced with a slower one — the gap between fastest and
slowest here is a factor of several hundred, far too large to happen by accident.

```python
from vidmag.backend import registry

for info in registry.list_backends():
    print(f"{info.name:8} {info.unavailable_reason or 'available'}")
```

## Measured

Apple M2 Max, the sample clips, magnification only — reading and writing the
video excluded. One warm-up, then the best of two. Measured 2026-08-11.

| Backend | Colour, `face.mp4` | Motion, `baby.mp4` | Against the processor |
|---|---:|---:|---:|
| OpenCL | 217 ms | 1,020 ms | 32× / 23× |
| Vulkan | 255 ms | 1,462 ms | 28× / 16× |
| Metal | 362 ms | 1,698 ms | 19× / 14× |
| PyTorch | 653 ms | 2,320 ms | 11× / 10× |
| Processor (NumPy) | 7,014 ms | 23,634 ms | — |

Every one agrees with the reference to within one step of the final 8-bit
rounding.

*Not re-verified since.* A reproduction attempt on 2026-08-18 could not confirm
these: the machine's graphics processor was at 100% utilisation from unrelated
software, so every graphics backend measured five to seven times slower while
the processor baseline was only 18% slower. That measurement proves nothing
either way, and none of it was published.
`scripts/dev/record_backend_bench.py` regenerates this table on an idle machine.

**There is no single winner, and the ordering is not stable.** OpenCL is fastest
here on both pipelines; on a 60-frame clip Metal was faster than OpenCL. The
ranking changes with clip length and would change again on other hardware. That
is why the choice is not hard-coded, and why these numbers describe one machine
rather than the backends in general.

## Live streaming is a different question

Whole-clip speed says nothing about keeping up with a camera. Pushing 720p
frames one at a time, the same day:

| Machine and backend | frames/s |
|---|---:|
| RTX 3090, PyTorch | 107.6 |
| Apple M2 Max, Metal | 58.8 |
| Apple M2 Max, PyTorch | 44.6 |
| Apple M2 Max, Vulkan | 20.5 |
| Apple M2 Max, processor | 8.1 |
| RTX 3090, processor | 6.3 |
| Apple M2 Max, OpenCL | 3.9 |

OpenCL is fastest on whole clips here and slowest on live frames — batching is
where it wins. And the hand-written NVIDIA backend **cannot stream at all**: it
implements the four whole-clip pipelines but not the frame-at-a-time operations,
and says so rather than failing partway. On NVIDIA hardware, PyTorch is currently
the only way to magnify a live stream. See [magnify a live feed](../recipes/streaming.md).

## The two backends that need explaining

**Why the NumPy version still exists.** It is the reference: every operation in
every other backend is compared against it, and it is the version compared
against the authors' published output. It is slow on purpose — written to be
obviously correct, not fast.

**Why a PyTorch backend exists.** It was nearly skipped, on the reasoning that it
reaches no hardware the native backends miss. That holds for whole-clip work,
where it is the slowest graphics backend and the hand-written NVIDIA code is
2.5× faster on the same card. It does not hold for three things no other backend
covers:

- **The only way to stream on NVIDIA hardware**, and the fastest streaming
  measured here at 107.6 frames per second on 720p.
- **Results stay tensors**, so magnification can sit inside a larger tensor
  computation with no round trip through the host.
- **An independent implementation** — a different library expressing the same
  definitions, so its agreement with NumPy is evidence about the definitions,
  not about one way of writing them.

PyTorch is never imported unless this backend is asked for.

## Adding hardware this does not cover

Implement about a dozen primitive operations for the new device, then:

```python
from vidmag.backend import generic

backend = generic.bind(my_operations)
```

All four pipelines follow, and the conformance suite compares them against the
NumPy reference. A backend may override a pipeline for speed — the CUDA backend
does — but need not.

## Why the portable backends are slower than CUDA

The CUDA backend fuses stages together and keeps intermediate results in fast
local memory. The portable backends run each operation as its own piece of work,
because that is what lets one source compile under every vendor's driver. Slower,
on hardware that would otherwise have no acceleration at all, is the trade.

**Not implemented: devices with no standard driver.** Some accelerators are only
reachable through a vendor-specific compiler. Nothing general-purpose can target
those, and this project does not claim to.
