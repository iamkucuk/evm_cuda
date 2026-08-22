# Magnify a live feed

Everything else in this library takes a whole clip. This does not: it magnifies
each frame as it arrives, using only what has already been seen, so it works on
a camera.

```bash
vidmag stream 0 --display
```

That opens the first camera, amplifies motion, and shows the result. Press `q`
to stop. It prints how fast it kept up when it exits.

From Python:

```python
import numpy as np
from vidmag.stream import MotionStream

stream = MotionStream(height=240, width=320, alpha=10, lambda_c=16)
for _ in range(3):  # your camera loop here
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    magnified = stream.push(frame)
```

## Why only motion

The colour pipeline selects its frequency band with a Fourier transform, which
needs all of time at once. A camera does not have all of time. The motion
pipeline's filter runs forward in time, carrying a small amount of state from
frame to frame, so it can work on what has arrived so far.

## It gives the same answer as the whole-clip version

Not similar — identical. Feeding frames one at a time produces exactly what
magnifying the whole clip produces, and the test suite asserts it with no
tolerance at all. That matters because it means the streaming path does not
need verifying separately: it inherits the comparison against the reference
implementation that the batch path already carries.

The first frame comes back unchanged. Both running averages start at its value,
so their difference is zero and nothing is amplified. The batch pipeline does
the same.

## Speed, measured

Measured 2026-08-11, frames pushed one at a time as a camera would deliver
them.

At 720p (1280x720), which is what most webcams produce:

| Machine and backend | frames per second | Keeps up with 30 fps? |
|---|---:|---|
| RTX 3090, PyTorch | 107.6 | yes |
| Apple M2 Max, Metal | 58.8 | yes |
| Apple M2 Max, PyTorch | 44.6 | yes |
| Apple M2 Max, Vulkan | 20.5 | no |
| Apple M2 Max, processor | 8.1 | no |
| Apple M2 Max, OpenCL | 3.9 | no |

At 320x240, on an Apple M2 Max: Metal 227.6, PyTorch 84.6, the processor 57.9.

**Use a graphics backend**, which is what the default now does. An earlier
version of this page advised the opposite — that launching many small pieces of
work per frame costs more than it saves — and measurement does not support it
at either size tried: Apple's interface is about four times the processor's
rate at 320x240 and about seven times at 720p.

**The hand-written NVIDIA backend cannot stream.** It implements the four
whole-clip pipelines but not the frame-at-a-time operations, and says so rather
than failing partway through a frame. That is why `MotionStream` does not
simply take whatever `"auto"` would choose for a whole clip: it walks the same
preference order but skips backends that cannot stream. On an NVIDIA machine
that means PyTorch, which is both the only option there and the fastest
streaming measured in this project.

## Keeping up with a camera

If magnification is slower than the camera delivers, something has to give.
Reduce the frame size first — it is the setting with the largest effect and the
least cost to quality:

```python
import cv2
import numpy as np
from vidmag.stream import MotionStream

small = (320, 240)
stream = MotionStream(height=small[1], width=small[0])
frame = np.zeros((480, 640, 3), dtype=np.uint8)  # from your camera
magnified = stream.push(cv2.resize(frame, small))
```

Dropping frames also works and is what a live view should usually do, but note
that the filter's idea of frequency is in frames, so dropping them changes what
band you are actually selecting.

## Writing the result to a file

```bash
vidmag stream 0 --out session.mp4 --max-frames 300
```

## Choosing the amount

The same parameters as the whole-clip motion pipeline, and the same advice: see
[amplify motion](motion.md). On a live feed the effect of too much `alpha` is
easier to spot, since you can raise it while watching.
