# Installing

```bash
pip install vidmag
```

That is the whole thing on any machine. It always gives you a working library
on the processor cores; whether it also builds graphics acceleration depends on
the hardware. The two sections below say how to get each kind.

## NVIDIA graphics processor

The fastest path is hand-written CUDA, compiled during installation. It needs
the CUDA toolkit — specifically `nvcc` — already on the machine.

```bash
nvcc --version          # prints a version => the extension will be built
pip install vidmag
```

If `nvcc` is absent the install still succeeds and prints that the extension
was skipped — a missing compiler should not stop you installing a library that
works without it. Two environment variables change the defaults:

| Variable | Effect |
|---|---|
| `VIDMAG_CUDA_REQUIRE=1` | Turn a missing `nvcc` into a build error instead of a skip |
| `VIDMAG_CUDA_ARCHS=all` | Build for a range of NVIDIA cards, not just the one compiling |

Check what you got:

```python
import vidmag.cuda

print(vidmag.cuda.have_cuda)
```

## Apple, AMD or Intel graphics processor

These are reached through OpenCL, which needs one extra package:

```bash
pip install "vidmag[opencl]"
```

The driver comes from your operating system or graphics vendor, not from this
project. macOS ships one; on Linux it comes from the vendor's driver package.
Installing a Python package and installing a driver are different jobs, so ask
which half is missing:

```python
from vidmag.opencl import runtime

print(runtime.unavailable_reason() or f"ready: {runtime.device_name()}")
```

That prints the device name if it works, and otherwise says what is missing.
Metal and Vulkan are alternatives on the same hardware — see
[backends and hardware](../concepts/backends.md).

## Which one gets used

By default, the fastest that will run: NVIDIA, then Metal, Vulkan, OpenCL,
PyTorch, then the processor cores. The choice is reported through the `vidmag`
logger, and you can ask before running:

```python
from vidmag import backend

name, _ = backend.select("auto")
print(name)
```

Naming one explicitly is honoured exactly and fails loudly if it cannot run — it
is never quietly swapped for a slower one.

## From a checkout

```bash
git clone https://github.com/iamkucuk/eulerian-video-magnification-cuda
cd eulerian-video-magnification-cuda
pip install -e ".[dev]"
python -m pytest tests/ -q
```

The suite reports skips as well as passes. On a machine with no graphics
processor the whole hardware-comparison suite skips, so a green run there says
nothing about that hardware — read the skip count, not just the pass count.
