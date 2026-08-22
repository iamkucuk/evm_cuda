# Choosing parameters

Five numbers control everything. This is what each does and how to pick it.

## `alpha` — how much to amplify

Start with the preset, raise it until the change is clear, then stop.

Too high looks different by pipeline. Colour saturates: regions go flat and lose
detail. Motion tears: edges break into ripples and haloes, because the
method's assumption fails at edges first. There is no correct value — it depends
on how large the real change is and how much noise the camera adds, and both
vary by clip.

## `fl` and `fh` — which frequencies to keep

In cycles per second; everything outside is discarded. Pick it from what you are
looking for: a resting heart rate of 60 bpm is 1 cycle per second, a guitar's
low E string is 82. Two constraints apply.

- **Below half the frame rate.** A camera at 30 fps cannot represent anything
  faster than 15 cycles per second; above that limit the movement appears as a
  slower one that is not really there.
- **Wide enough for the clip.** A clip of `N` frames at `f` fps resolves
  frequencies in steps of `f / N`. A band narrower than one step contains no
  frequency, the filter returns zeros, and the output equals the input. The
  library warns and says how many frames it needs. The `pulse` preset needs
  about 181 frames — six seconds at 30 fps — before its band contains anything.

## `level` — how far to shrink, for colour

How many times to halve the resolution before filtering. More shrinking means
less noise and a smoother result, at the cost of fine detail. The preset uses 4,
so a 512-pixel-wide frame is filtered at 32 wide. Shrink too far and the region
you care about disappears into one sample.

## `lambda_c` — the spatial cutoff, for motion

In pixels; detail finer than this is amplified progressively less. This is the
artefact control. Raise it when the result shimmers; lower it to amplify finer
movement and accept more noise. The presets use 16.

## `chrom_attenuation` — how much colour to amplify

A multiplier on the colour channels relative to brightness. For colour work,
1.0: the colour change is the signal. For motion, about 0.1: the movement is in
brightness, and amplifying colour just amplifies colour noise, which is very
visible.

## Where the shipped values come from

Every preset records its own provenance:

```python
from vidmag.presets import PRESETS

print(PRESETS["pulse"].params)
print(PRESETS["pulse"].source)
```

The `pulse` and `motion` presets are the authors' own values for their own
clips. The `vibration` preset comes from an example in this repository, and its
description says so — that is weaker evidence, and the reader should know.
