# vidmag

**Amplify changes in a video that are too small to see** — the flush of blood
through a face with each heartbeat, the millimetre rise of a sleeping child's
chest, the vibration of a guitar string. This is [Eulerian Video
Magnification](http://people.csail.mit.edu/mrub/vidmag/) (Wu and colleagues,
MIT, 2012): hand-written CUDA at the core, five more backends so it runs on
whatever hardware you have.

<p align="center">
  <img src="img/face_demo.gif" alt="A face before and after colour magnification" width="600">
</p>
<p align="center"><sub>Left: the original clip. Right: colour changes amplified.
The green tint is blood flow, appearing and fading with each heartbeat.</sub></p>

## Run it

```bash
pip install vidmag
```

```python
import vidmag

vidmag.magnify("face.mp4", preset="pulse", out="pulse.mp4")
```

That reads a video, amplifies the colour changes in the frequency band a
resting heart rate falls in, and writes the result. A graphics processor is
used if one is present; otherwise the same call runs on the processor cores.

Frames already in memory work too, with no file:

```python
import numpy as np
import vidmag

frames = np.zeros((60, 64, 64, 3), dtype=np.uint8)  # your frames, 8-bit BGR
amplified = vidmag.magnify(frames, preset="motion")
```

## Where to go next

| To | Read |
|---|---|
| Install it | [Installing](getting-started/install.md) |
| See it work on a real clip | [First result](getting-started/first-result.md) |
| Make a pulse visible | [See a pulse](recipes/pulse.md) |
| Measure a vibrating object | [Measure vibration](recipes/vibration.md) |
| Amplify small movements | [Amplify motion](recipes/motion.md) |
| Build past the presets | [Use the building blocks](recipes/building-blocks.md) |
| Understand the method | [How it works](concepts/how-it-works.md) |
| Pick a backend | [Backends and hardware](concepts/backends.md) |

## Why it is built this way

The method is defined by the authors' original MATLAB code. Here it is written
once in NumPy, checked against the authors' own published output, then
reimplemented five more times — hand-written CUDA, OpenCL, Vulkan, Apple's
Metal, and PyTorch — with every step of every one compared against the NumPy
version by an automated test suite.

The comparison is the point. Six independent implementations, in six languages
across five kinds of hardware, agreeing to within one step of the final 8-bit
value. A magnification nobody has checked is a picture, not a measurement.
Speed, measured rather than estimated, is on the [performance](performance.md)
page.

## Licence

Free for research, teaching, personal use, and inclusion in open-source
software given away at no charge. Selling it, or building it into a product run
for commercial advantage, needs written permission. See
[licence and citation](licence.md).
