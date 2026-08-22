# Installing

```bash
pip install vidmag
```

That is the whole thing on any machine. It always gives you a working library
running on the processor cores; whether it also builds graphics acceleration
depends on what the machine has, and the next two sections say how to get it.

## Using an NVIDIA graphics processor

The fastest path in this project is hand-written CUDA. It is compiled during
installation, which needs the CUDA toolkit — specifically `nvcc` — already on
the machine.

```bash
nvcc --version          # if this prints a version, the extension will be built
pip install vidmag
```

If `nvcc` is absent the install still succeeds and prints a message saying the
extension was skipped. That is deliberate: a missing compiler should not stop
you installing a library that works without it. To turn the absence into an
error instead, set `VIDMAG_CUDA_REQUIRE=1` before installing.

By default the extension is compiled for the graphics processor in the machine
doing the compiling. To build one that runs on a range of NVIDIA hardware, set
`VIDMAG_CUDA_ARCHS=all`.

Check what you got:

```python
import vidmag.cuda

print(vidmag.cuda.have_cuda)
```

## Using an Apple, AMD or Intel graphics processor

These are reached through OpenCL, which needs one extra Python package:

```bash
pip install "vidmag[opencl]"
```

The driver itself comes from your operating system or your graphics vendor, not
from this project. macOS ships one. On Linux it comes from the vendor's driver
package. To find out whether both halves are present:

```python
from vidmag.opencl import runtime

print(runtime.unavailable_reason() or f"ready: {runtime.device_name()}")
```

That prints the device name if it works, and otherwise says which of the two
things is missing, because installing a Python package and installing a driver
are different jobs.

## Which one gets used

By default, the fastest that will run: hand-written CUDA, then OpenCL, then the
processor cores. The choice is reported through the `vidmag` logger, and you can
ask before running:

```python
from vidmag import backend

name, _ = backend.select("auto")
print(name)
```

Naming one explicitly is honoured exactly, and fails loudly if it cannot run —
it is never quietly swapped for a slower one. See
[backends and hardware](../concepts/backends.md).

## Installing from a checkout

```bash
git clone https://github.com/iamkucuk/eulerian-video-magnification-cuda
cd eulerian-video-magnification-cuda
pip install -e ".[dev]"
python -m pytest tests/ -q
```

The test suite reports how many tests it skipped as well as how many passed.
The skips are real information: on a machine with no graphics processor, the
whole hardware-comparison suite skips, and a green run there says nothing about
that hardware.
