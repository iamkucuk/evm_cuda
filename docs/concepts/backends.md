# Backends and hardware

The same magnification is implemented three times over. Which one runs is
chosen automatically, and never changed behind your back.

| Backend | Runs on | Written in |
|---|---|---|
| `cuda` | NVIDIA graphics processors | Hand-written CUDA, tuned |
| `metal` | Apple graphics processors | Metal compute kernels |
| `vulkan` | Anything with a Vulkan driver: AMD, Intel, NVIDIA, Apple through a translation layer, mobile and embedded hardware | GLSL compiled to SPIR-V |
| `opencl` | Anything with an OpenCL driver: Apple, AMD, Intel, NVIDIA; also processors, through a software driver | OpenCL kernels |
| `cpu` | Anything | NumPy — the reference the others are checked against |

Four different interfaces to graphics hardware, and they overlap: on this Mac,
three of them work. That is on purpose. Which interface a given piece of
hardware supports is not something this project controls, and it changes: Apple
has deprecated OpenCL, Vulkan is what most new hardware ships with, and Metal
is what Apple supports going forward. Covering several means a device that
appears later has a path that needs no new code here.

## Measured

An Apple M2 Max, on the sample clips, against the same machine's processor.
Single runs, so treat them as approximate.

| Pipeline | Processor | Metal | Vulkan | OpenCL |
|---|---:|---:|---:|---:|
| Colour, `face.mp4`, 301 frames | 6,769 ms | 281 ms | 212 ms | 218 ms |
| Motion, `baby.mp4`, 301 frames | 30,172 ms | 1,923 ms | 2,343 ms | 5,953 ms |

All three agree with the reference to within one step of the final 8-bit
rounding. There is no single winner: Vulkan is fastest on the colour pipeline
and Metal on the motion one, and the ordering would differ on other hardware,
which is why the choice is not hard-coded.

## How one is chosen

`backend="auto"`, the default, tries them in the order in the table above and
uses the first that can run: the tuned NVIDIA path first, then Apple's own
interface, then Vulkan, then OpenCL, then the processor. Where several would
work, the earlier one is the shorter path to the hardware — Vulkan on a Mac
runs through a translation layer onto Metal, so using Metal directly removes a
layer. The choice is reported through the `evm` logger, and can be asked
for in advance:

```python
from evm import backend

name, _ = backend.select("auto")
print(name)
```

Naming one is honoured exactly. If it cannot run, the call raises and says why;
it is never quietly replaced with a slower one. The difference between the
fastest and slowest here is a factor of several hundred, which is far too large
to happen by accident.

```python
from evm.backend import registry

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
from evm.backend import generic

backend = generic.bind(my_operations)
```

All four pipelines follow from those operations, and the conformance suite
compares them against the NumPy reference. There is no need to reimplement any
pipeline; a backend may override one, and the CUDA backend does, but only for
speed.

## What is deliberately not implemented

**A PyTorch backend.** It would add a very large dependency to reach hardware
already reached natively.

**Devices with no standard driver.** Some accelerators are only reachable
through a vendor-specific compiler. Nothing general-purpose can target those,
and this project does not claim to.
