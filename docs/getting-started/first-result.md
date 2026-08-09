# Your first result

This takes about five minutes and ends with two video files you can play side
by side.

## Get the sample clips

The clips used by the original authors are `face.mp4` and `baby.mp4` from the
[project page at MIT](http://people.csail.mit.edu/mrub/vidmag/). Download them
into a `data/` directory next to where you are working.

From a checkout of this repository there is a command for it:

```bash
evm-magnify download face baby
```

That one needs the repository, not just the installed package: it wraps
`scripts/download_samples.py`, which holds the addresses and is not part of the
wheel. With only the library installed, fetch the two files yourself.

## Make a pulse visible

```bash
evm-magnify magnify data/face.mp4 pulse.mp4 --preset pulse
```

Play `pulse.mp4` next to the original. The face takes on a green flush that
appears and fades a little over once a second. That is blood arriving with each
heartbeat: it was in the original clip all along, changing the colour of the
skin by less than one step in 255, which is far too little for an eye to catch.

The command prints which backend it used before it starts, so you know whether
the graphics processor did the work.

## Make a small movement visible

```bash
evm-magnify magnify data/baby.mp4 breathing.mp4 --preset motion
```

The sleeping child's chest now visibly rises and falls. The real movement is
under a millimetre.

## The same thing from Python

```python
import evm

evm.magnify("data/face.mp4", preset="pulse", out="pulse.mp4")
```

Or on frames you already have, with no file involved:

```python
import numpy as np
import evm

frames = np.zeros((90, 64, 64, 3), dtype=np.uint8)  # your frames, 8-bit BGR
amplified = evm.magnify(frames, preset="motion")
print(amplified.shape, amplified.dtype)
```

## If the result looks identical to the input

That is a real and common outcome, and it usually means the thing you are
looking for is not inside the frequency band the preset selects. The `pulse`
preset keeps 0.83 to 1.0 cycles per second, which is a resting heart rate, and
a clip shorter than about six seconds at 30 frames per second cannot resolve
that band at all — the library warns when that happens and says how many frames
it would need.

[What can go wrong](../concepts/pitfalls.md) covers the rest.
