# What can go wrong

The failures here are mostly not bugs. They are the method working exactly as
defined on input it cannot help with.

## The output is identical to the input

The most common outcome, and almost always the frequency band.

**The band contains no frequency at all.** A clip of `N` frames at `f` frames
per second resolves steps of `f / N`; a band narrower than one step selects
nothing and the filter returns zeros. The library warns and says how many
frames it needs. Record longer or widen the band.

**The band is above what the camera can capture.** Nothing faster than half the
frame rate can be represented. The `vibration` preset's 72-to-92-cycle band
needs more than 184 frames per second; at 30 it selects nothing.

**The change is not in the band.** Check by measuring rather than assuming, as
shown in [measure vibration](../recipes/vibration.md).

## The result shimmers, ripples, or has haloes at edges

Too much amplification for the spatial cutoff in use. The method assumes
movement is small compared with the detail it is shifting, and this is what it
looks like when that assumption fails.

Raise `lambda_c` first, which reduces amplification of fine detail. Lower
`alpha` if that is not enough.

## The whole frame pulses or sways

The camera moved. Everything here is relative to the frame, so camera shake is
amplified along with everything else, and it is usually far larger than the
thing you are measuring. Use a tripod, or align the frames to each other first.

## Coloured speckle appears

Colour noise being amplified. Lower `chrom_attenuation` — 0.1 for motion work
is a good default, and 0 removes colour amplification entirely.

## A slow brightness ripple appears across the whole frame

Often the lighting. Mains-powered lights flicker at 50 or 60 cycles per second,
and a camera exposure that beats against that produces a slow ripple which can
be stronger than a pulse. Use daylight or a direct-current lamp.

## It runs out of memory

Every pipeline holds the whole clip at once, because the Fourier filter needs
all of time together. Work in overlapping chunks — there is an example in
[amplify motion](../recipes/motion.md) — or reduce the resolution first.

## It runs on the processor when a graphics processor is present

The library never falls back silently, so this means the backend genuinely
could not run. Ask it why:

```python
from evm.backend import registry

for info in registry.list_backends():
    print(info.name, info.unavailable_reason or "available")
```

The reason distinguishes a missing Python package from a missing driver, which
need different fixes.
