# Your first result

About five minutes, ending in two video files you can play side by side.

## Get the sample clips

The clips the original authors used are `face.mp4` and `baby.mp4` from the
[MIT project page](http://people.csail.mit.edu/mrub/vidmag/). From a checkout of
this repository:

```bash
vidmag download face baby
```

That needs the repository, not just the installed package: it wraps
`scripts/download_samples.py`, which holds the addresses and is not in the
wheel. With only the library installed, fetch the two files yourself into a
`data/` directory.

## Make a pulse visible

```bash
vidmag magnify data/face.mp4 pulse.mp4 --preset pulse
```

Play `pulse.mp4` next to the original. The face takes on a green flush that
appears and fades a little more than once a second — blood arriving with each
heartbeat, changing the skin's colour by less than one step in 255, far too
little for an eye to catch. The command prints which backend it used before it
starts.

## Make a small movement visible

```bash
vidmag magnify data/baby.mp4 breathing.mp4 --preset motion
```

The sleeping child's chest now visibly rises and falls. The real movement is
under a millimetre.

## The same thing from Python

```python
import vidmag

vidmag.magnify("data/face.mp4", preset="pulse", out="pulse.mp4")
```

Or on frames you already have, with no file:

```python
import numpy as np
import vidmag

frames = np.zeros((90, 64, 64, 3), dtype=np.uint8)  # your frames, 8-bit BGR
amplified = vidmag.magnify(frames, preset="motion")
print(amplified.shape, amplified.dtype)
```

## If the result looks identical to the input

Common, and usually the frequency band. The `pulse` preset keeps 0.83 to 1.0
cycles per second — a resting heart rate — and a clip shorter than about six
seconds at 30 frames per second cannot resolve that band at all. The library
warns when that happens and says how many frames it needs.
[What can go wrong](../concepts/pitfalls.md) covers the rest.
