# Use the building blocks

The presets cover the jobs the original paper covers. For anything else — a
different filter, an extra step, one stage on its own — the parts are available
separately, and they are the same parts the pipelines are built from.

## The parts

| Operation | What it does |
|---|---|
| `bgr_u8_to_ntsc` / `ntsc_to_bgr_u8` | Convert 8-bit colour to a brightness-plus-colour form, and back |
| `blur_dn` | Blur and halve the resolution, repeatedly |
| `build_lpyr` / `recon_lpyr` | Split an image into detail at each scale, and reconstruct it |
| `ideal_bandpass` | Keep a band of frequencies, using a Fourier transform |
| `butter_bandpass` | Keep a band with a Butterworth filter, running forward in time |
| `iir_bandpass` | Keep a band by subtracting two running averages |
| `apply_gain` | Scale the three channels — brightness and two colour — separately |

Same names and argument order on every backend, so a chain written against one
runs on another. Full signatures are in the [building blocks reference](../reference/ops.md).

## On the processor

```python
import numpy as np
from vidmag.cpu import ops

frames = np.zeros((60, 64, 64, 3), dtype=np.uint8)  # your frames, 8-bit BGR

ntsc = ops.bgr_u8_to_ntsc(frames)
small = ops.blur_dn(ntsc, 2)
band = ops.iir_bandpass(small, 0.4, 0.05)
amplified = ops.apply_gain(band, 20.0, 2.0, 2.0)
print(amplified.shape)
```

## On an NVIDIA graphics processor, without copying between steps

The GPU versions take and return a `DeviceArray`, which stays on the card.
Nothing is copied back until you ask:

```python
import numpy as np
from vidmag.cuda import ops
from vidmag.cuda.array import DeviceArray

frames = DeviceArray.from_numpy(np.zeros((60, 64, 64, 3), dtype=np.uint8))

ntsc = ops.bgr_u8_to_ntsc(frames)
small = ops.blur_dn(ntsc, 2)
band = ops.iir_bandpass(small, 0.4, 0.05)
result = band.numpy()  # the one copy back
```

## Handing the result to PyTorch without copying

A `DeviceArray` implements the protocol array libraries use to share memory, so
a result can go straight into a tensor — no copy, and the two then refer to the
same memory:

```python
import torch  # doctest: +SKIP

tensor = torch.from_dlpack(band)  # doctest: +SKIP
assert tensor.data_ptr() == band.ptr  # doctest: +SKIP
```

This is what makes the library usable as one stage inside a larger pipeline
already living on the graphics processor.

## A pipeline that runs on every backend

The four pipelines are written once in terms of the operations above, and each
backend supplies its own versions. Do the same, and your code runs wherever the
library does:

```python
import numpy as np
from vidmag.backend import registry

frames = np.zeros((60, 64, 64, 3), dtype=np.uint8)  # your frames, 8-bit BGR

name, impl = registry.select("auto")
print("running on", name)
out = impl.motion_lpyr_iir_core(
    frames, 30.0, alpha=10.0, lambda_c=16.0, r1=0.4, r2=0.05
)
print(out.shape)
```

## Adding support for new hardware

Implement the operations above for the hardware, then:

```python
from vidmag.backend import generic, registry

backend = generic.bind(my_operations)  # all four pipelines, derived
```

That is the whole job. The pipelines follow from the operations, and the
conformance tests compare them against the NumPy reference. See
[backends and hardware](../concepts/backends.md).
