# Backends and hardware

The same magnification is implemented six times over, but they are not six
equals. **The NVIDIA backend is what this project is for.** It is hand-written
CUDA C++ with no tensor framework underneath it, it is where optimisation work
starts, and the figures on [Performance](../performance.md) are its figures.
The other five exist so the library still works when there is no NVIDIA card
present, and so that hardware appearing later has a path that needs no new code
here.

Which one runs is chosen automatically, and never changed behind your back.

| Backend | Runs on | Written in | Needs |
|---|---|---|---|
| `cuda` | NVIDIA graphics processors | Hand-written CUDA, tuned | a compiler at install |
| `metal` | Apple graphics processors | Metal compute kernels | `[metal]` |
| `vulkan` | Anything with a Vulkan driver: AMD, Intel, NVIDIA, Apple through a translation layer, mobile and embedded hardware | GLSL compiled to SPIR-V | `[vulkan]` |
| `opencl` | Anything with an OpenCL driver: Apple, AMD, Intel, NVIDIA; also processors, through a software driver | OpenCL kernels | `[opencl]` |
| `torch` | Wherever PyTorch runs: NVIDIA, Apple, and processors | PyTorch tensor operations | `[torch]` |
| `cpu` | Anything | NumPy — the reference the others are checked against | nothing |

Five different ways to reach a graphics processor, and they overlap: on an
Apple laptop, four of them work. That is on purpose. Which interface a given
piece of hardware supports is not something this project controls, and it
changes: Apple has deprecated OpenCL, Vulkan is what most new hardware ships
with, and Metal is what Apple supports going forward. Covering several means a
device that appears later has a path that needs no new code here.

Only `cuda` and `cpu` are installed by default. The rest are optional extras,
so nothing is downloaded for hardware you do not have:

```bash
pip install "vidmag[metal]"      # or [vulkan], [opencl], [torch]
```

## Measured

Apple M2 Max, the sample clips, magnification only — reading and writing the
video are excluded. One warm-up run, then the best of two. Measured
2026-08-11.

| Backend | Colour, `face.mp4` | Motion, `baby.mp4` | Against the processor |
|---|---:|---:|---:|
| OpenCL | 217 ms | 1,020 ms | 32x / 23x |
| Vulkan | 255 ms | 1,462 ms | 28x / 16x |
| Metal | 362 ms | 1,698 ms | 19x / 14x |
| PyTorch | 653 ms | 2,320 ms | 11x / 10x |
| Processor (NumPy) | 7,014 ms | 23,634 ms | — |

Every one agrees with the reference to within one step of the final 8-bit
rounding.

*Not re-verified since.* A reproduction attempt on 2026-08-18 could not confirm
these: the machine's graphics processor was at 100% utilisation from unrelated
software, and every graphics backend measured five to seven times slower while
the processor baseline was only 18% slower. That measurement proves nothing
either way and none of it was published.
`scripts/dev/record_backend_bench.py` regenerates this table on an idle machine.


**There is no single winner, and the ordering is not stable.** OpenCL is
fastest here on both pipelines, and on a 60-frame clip Apple's own interface
was faster than OpenCL — the ranking changes with clip length, and would change
again on other hardware. That is why the choice is not hard-coded, and why
these numbers describe one machine rather than the backends in general.

## Live streaming is a different question

Whole-clip speed says nothing about keeping up with a camera. Pushing 720p
frames one at a time, the same day:

| Machine and backend | frames per second |
|---|---:|
| RTX 3090, PyTorch | 107.6 |
| Apple M2 Max, Metal | 58.8 |
| Apple M2 Max, PyTorch | 44.6 |
| Apple M2 Max, Vulkan | 20.5 |
| Apple M2 Max, processor | 8.1 |
| RTX 3090, processor | 6.3 |
| Apple M2 Max, OpenCL | 3.9 |

Two things worth knowing before choosing. OpenCL is fastest on whole clips here
and slowest on live frames — batching is where it wins. And the hand-written
NVIDIA backend **cannot stream at all**: it implements the four whole-clip
pipelines but not the frame-at-a-time operations, and says so rather than
failing partway through. On NVIDIA hardware, PyTorch is currently the only way
to magnify a live stream.

## How one is chosen

`backend="auto"`, the default, tries them in the order in the table above and
uses the first that can run: the tuned NVIDIA path first, then Apple's own
interface, then Vulkan, then OpenCL, then PyTorch, then the processor. PyTorch
sits second to last because it reaches no hardware the others miss, so it
should never be preferred over a native backend for whole-clip work. Where several would
work, the earlier one is the shorter path to the hardware — Vulkan on a Mac
runs through a translation layer onto Metal, so using Metal directly removes a
layer. The choice is reported through the `vidmag` logger, and can be asked
for in advance:

```python
from vidmag import backend

name, _ = backend.select("auto")
print(name)
```

Naming one is honoured exactly. If it cannot run, the call raises and says why;
it is never quietly replaced with a slower one. The difference between the
fastest and slowest here is a factor of several hundred, which is far too large
to happen by accident.

```python
from vidmag.backend import registry

for info in registry.list_backends():
    print(f"{info.name:8} {info.unavailable_reason or 'available'}")
```

## Why the NumPy version still exists

It is the reference. Every operation in both graphics implementations is
compared against it by the test suite, and it is the version compared against
the original authors' published output. It is slow on purpose: it is written to
be obviously correct, not fast.

## Why the portable ones are slower than the CUDA one

The CUDA backend fuses stages together and keeps intermediate results in fast
local memory. The portable backends run each operation as its own piece of
work, because that is what lets one source be compiled by every vendor's
driver. Slower, on hardware that would otherwise have no acceleration at all,
is the trade being made.

## Adding hardware this does not cover

Implement about a dozen primitive operations for the new device, then:

```python
from vidmag.backend import generic

backend = generic.bind(my_operations)
```

All four pipelines follow from those operations, and the conformance suite
compares them against the NumPy reference. There is no need to reimplement any
pipeline; a backend may override one, and the CUDA backend does, but only for
speed.

## Why a PyTorch backend exists as well

It was planned as optional and nearly skipped, on the reasoning that it reaches
no hardware the native backends miss. That reasoning holds for whole-clip work,
where it is the slowest of the graphics backends and the hand-written NVIDIA
code is 2.5 times faster on the same card. It does not hold for everything
else, and three things it does are not covered by any other backend:

- **It is the only way to stream on NVIDIA hardware**, and the fastest streaming
  measured in this project at 107.6 frames per second on 720p.
- **It keeps results as tensors**, so magnification can sit inside a larger
  tensor computation with no round trip through the host.
- **It is an independent implementation**: written in a different library from
  the same definitions, so its agreement with the NumPy reference is evidence
  about the definitions rather than about one way of expressing them.

PyTorch is never imported unless this backend is asked for. A machine without
it is unaffected.

## What is deliberately not implemented

**Devices with no standard driver.** Some accelerators are only reachable
through a vendor-specific compiler. Nothing general-purpose can target those,
and this project does not claim to.
