# What can go wrong

Most of these are not bugs. They are the method working exactly as defined on
input it cannot help with.

## The output is identical to the input

The most common outcome, and almost always the frequency band.

- **The band contains no frequency.** A clip of `N` frames at `f` fps resolves
  steps of `f / N`; a band narrower than one step selects nothing and the filter
  returns zeros. The library warns and says how many frames it needs. Record
  longer or widen the band.
- **The band is above what the camera can capture.** Nothing faster than half
  the frame rate can be represented. The `vibration` preset's 72–92 cycle band
  needs more than 184 fps; at 30 it selects nothing.
- **The change is not in the band.** Measure rather than assume — see
  [measure vibration](../recipes/vibration.md).

## The result shimmers, ripples, or haloes at edges

Too much amplification for the spatial cutoff. The method assumes movement is
small compared with the detail it shifts, and this is what a broken assumption
looks like. Raise `lambda_c` first (less amplification of fine detail); lower
`alpha` if that is not enough.

## The whole frame pulses or sways

The camera moved. Everything here is relative to the frame, so camera shake is
amplified too, usually far larger than the thing you are measuring. Use a
tripod, or align the frames to each other first.

## Coloured speckle appears

Colour noise being amplified. Lower `chrom_attenuation` — 0.1 is a good default
for motion, and 0 removes colour amplification entirely.

## A slow brightness ripple across the whole frame

Often the lighting. Mains-powered lights flicker at 50 or 60 cycles per second,
and a camera exposure beating against that makes a slow ripple stronger than a
pulse. Use daylight or a direct-current lamp.

## It runs out of memory

Every pipeline holds the whole clip at once, because the Fourier filter needs
all of time together. Work in overlapping chunks — there is an example in
[amplify motion](../recipes/motion.md) — or reduce the resolution first.

## It runs on the processor when a graphics processor is present

The library never falls back silently, so the backend genuinely could not run.
Ask it why:

```python
from vidmag.backend import registry

for info in registry.list_backends():
    print(info.name, info.unavailable_reason or "available")
```

The reason distinguishes a missing Python package from a missing driver, which
need different fixes.
