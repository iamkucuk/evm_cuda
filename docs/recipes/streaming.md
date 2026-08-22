# Magnify a live feed

Everything else in this library takes a whole clip. This does not: it magnifies
each frame as it arrives, using only what has already been seen, so it works on
a camera.

```bash
vidmag stream 0 --display
```

That opens the first camera, amplifies motion, and shows the result. Press `q`
to stop; it prints how fast it kept up when it exits.

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

The colour pipeline selects its band with a Fourier transform, which needs all
of time at once — a camera does not have that. The motion filter runs forward
in time, carrying a little state from frame to frame, so it works on what has
arrived so far. The first frame comes back unchanged: both running averages
start at its value, so their difference is zero.

## It gives the same answer as the whole-clip version

Not similar — identical. Feeding frames one at a time produces exactly what
magnifying the whole clip produces, and the test suite asserts it with no
tolerance. So the streaming path needs no separate verification: it inherits the
batch path's comparison against the reference implementation.

## Speed, measured

Measured 2026-08-11, frames pushed one at a time as a camera would deliver them.
At 720p (1280×720), what most webcams produce:

| Machine and backend | frames/s | Keeps 30 fps? |
|---|---:|---|
| RTX 3090, PyTorch | 107.6 | yes |
| Apple M2 Max, Metal | 58.8 | yes |
| Apple M2 Max, PyTorch | 44.6 | yes |
| Apple M2 Max, Vulkan | 20.5 | no |
| Apple M2 Max, processor | 8.1 | no |
| Apple M2 Max, OpenCL | 3.9 | no |

At 320×240 on the Apple M2 Max: Metal 227.6, PyTorch 84.6, the processor 57.9.

Two things follow. **Use a graphics backend**, which the default now does — an
earlier version of this page advised the opposite, and measurement does not
support it at either size. **The hand-written NVIDIA backend cannot stream:** it
implements the four whole-clip pipelines but not the frame-at-a-time operations,
and says so rather than failing partway. So `MotionStream` walks the same
preference order as `"auto"` but skips backends that cannot stream; on an NVIDIA
machine that means PyTorch, the fastest streaming measured here.

## Keeping up with a camera

If magnification is slower than the camera delivers, reduce the frame size
first — the largest effect for the least cost to quality:

```python
import cv2
import numpy as np
from vidmag.stream import MotionStream

small = (320, 240)
stream = MotionStream(height=small[1], width=small[0])
frame = np.zeros((480, 640, 3), dtype=np.uint8)  # from your camera
magnified = stream.push(cv2.resize(frame, small))
```

Dropping frames also works and is usually right for a live view, but the
filter's idea of frequency is in frames, so dropping them changes the band you
are actually selecting.

## Writing the result to a file

```bash
vidmag stream 0 --out session.mp4 --max-frames 300
```

## Choosing the amount

Same parameters and same advice as the whole-clip motion pipeline — see
[amplify motion](motion.md). On a live feed, too much `alpha` is easier to spot,
since you can raise it while watching.
