# Magnify a live feed

Everything else in this library takes a whole clip. This does not: it magnifies
each frame as it arrives, using only what has already been seen, so it works on
a camera.

```bash
evm-magnify stream 0 --display
```

That opens the first camera, amplifies motion, and shows the result. Press `q`
to stop. It prints how fast it kept up when it exits.

From Python:

```python
import numpy as np
from evm.stream import MotionStream

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

On an Apple M2 Max, using the processor:

| Frame size | Rate |
|---|---:|
| 320x240 | 56 frames a second |
| 640x480 | 21 frames a second |
| 960x544 | 13 frames a second |

So a small camera feed runs comfortably faster than real time; a large one does
not.

**Use the processor for this, not the graphics processor.** That is the
opposite of the advice for whole clips, and it is measured rather than assumed:
at 320x240 the graphics path manages 6 frames a second against the processor's
56. Magnifying one frame at a time launches a few dozen small pieces of work,
and the cost of launching them does not shrink with the frame, so it dominates.
Over a whole clip that cost is paid once instead of once per frame, which is why
the graphics backends win there by a wide margin. `MotionStream` therefore
defaults to the processor rather than to whatever `"auto"` would choose.

## Keeping up with a camera

If magnification is slower than the camera delivers, something has to give.
Reduce the frame size first — it is the setting with the largest effect and the
least cost to quality:

```python
import cv2
import numpy as np
from evm.stream import MotionStream

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
evm-magnify stream 0 --out session.mp4 --max-frames 300
```

## Choosing the amount

The same parameters as the whole-clip motion pipeline, and the same advice: see
[amplify motion](motion.md). On a live feed the effect of too much `alpha` is
easier to spot, since you can raise it while watching.
