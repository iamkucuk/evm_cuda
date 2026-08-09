# Backends and hardware

The same magnification is implemented three times over. Which one runs is
chosen automatically, and never changed behind your back.

| Backend | Runs on | Written in | Speed |
|---|---|---|---|
| `cuda` | NVIDIA graphics processors | Hand-written CUDA, tuned | Fastest |
| `opencl` | Apple, AMD, Intel and NVIDIA graphics processors; also processors, through a software driver | One set of OpenCL kernels | In between |
| `cpu` | Anything | NumPy | Slowest, and the reference |

## How one is chosen

`backend="auto"`, the default, tries them in the order above and uses the first
that can run. The choice is reported through the `evm` logger, and can be asked
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

## Why the OpenCL one is slower than the CUDA one

The CUDA backend fuses stages together and keeps intermediate results in fast
local memory. The OpenCL backend runs each operation as its own kernel, because
that is what allows one source file to be compiled by every vendor's driver. A
few times slower, on hardware that previously had no acceleration at all, is the
trade being made.

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

**Vulkan and Metal.** Every device this project targets is reachable through an
OpenCL driver, so a second and third set of kernels would add maintenance
without adding a supported device. Two things would change that: Apple removing
OpenCL, which it has deprecated, or a decision to support Android.

**A PyTorch backend.** It would add a very large dependency to reach hardware
already reached natively.

**Devices with no standard driver.** Some accelerators are only reachable
through a vendor-specific compiler. Nothing general-purpose can target those,
and this project does not claim to.
