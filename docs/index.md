# Eulerian Video Magnification

Some changes in a video are real but too small to see: the flush of blood
through a face with each heartbeat, the millimetre rise and fall of a sleeping
child's chest, the vibration of a guitar string or a machine housing. This
library amplifies those changes so they become visible, using the method
published by Wu and colleagues at MIT in 2012.

<p align="center">
  <img src="img/face_demo.gif" alt="A face before and after colour magnification" width="600">
</p>
<p align="center"><sub>Left: the original clip. Right: the same clip with colour
changes amplified. The green tint is blood flow, appearing and fading with each
heartbeat.</sub></p>

## In three lines

```python
import evm

evm.magnify("face.mp4", preset="pulse", out="pulse.mp4")
```

That reads a video, amplifies the colour changes in the frequency band a
resting heart rate falls in, and writes the result. If a graphics processor is
available it is used; if not, the same call runs on the processor cores.

Frames already in memory work too, with no file involved:

```python
import numpy as np
import evm

frames = np.zeros((60, 64, 64, 3), dtype=np.uint8)  # your frames here
amplified = evm.magnify(frames, preset="motion")
```

## Where to go next

| If you want to | Read |
|---|---|
| Get it installed | [Installing](getting-started/install.md) |
| See it work on a real clip | [First result](getting-started/first-result.md) |
| Make a pulse visible | [See a pulse](recipes/pulse.md) |
| Measure a vibrating object | [Measure vibration](recipes/vibration.md) |
| Amplify small movements | [Amplify motion](recipes/motion.md) |
| Build something the presets do not cover | [Use the building blocks](recipes/building-blocks.md) |
| Understand what it is doing | [How it works](concepts/how-it-works.md) |
| Know which hardware it runs on | [Backends and hardware](concepts/backends.md) |

## What this project is

A careful reimplementation, not a demonstration. The method is defined by the
authors' original MATLAB code; the version here is written in NumPy, checked
against the authors' own published output, and then reimplemented five more
times — hand-written CUDA for NVIDIA hardware, OpenCL, Vulkan, Apple's Metal,
and PyTorch — with every step of every one compared against the NumPy version
by an automated test suite. The comparison is the point: a magnification you
cannot check is a picture, not a measurement.

That the five agree with a NumPy reference which itself agrees with the
original authors' published output is what makes the numbers here worth
something. Six independent implementations of the same definitions, written in
six languages against five different pieces of hardware, converging to within
one step of the final 8-bit value.

Speed, measured rather than estimated, is on the [performance](performance.md)
page.

## Licence

Free for research, teaching, personal use, and inclusion in open-source
software given away at no charge. Selling it, or building it into a product run
for commercial advantage, needs written permission. Details on the
[licence and citation](licence.md) page.
