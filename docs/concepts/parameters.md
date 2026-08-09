# Choosing parameters

Five numbers control everything. This page says what each does and how to pick
it.

## `alpha` — how much to amplify

Start with the preset, then raise it until the change is clear and stop.

Too high looks different depending on the pipeline. For colour, it saturates:
regions go flat and lose detail. For motion, edges tear into ripples and
haloes, because the assumption the method rests on breaks down at edges first.

There is no correct value. It depends on how large the real change is and how
much noise the camera contributes, and those vary by clip.

## `fl` and `fh` — which frequencies to keep

In cycles per second. Everything outside is discarded.

Pick it from the thing you are looking for. A resting heart rate of 60 beats
per minute is 1 cycle per second; a guitar's low E string is 82.

Two constraints apply:

**Below half the frame rate.** A camera recording at 30 frames per second
cannot represent anything faster than 15 cycles per second. Above that limit
the movement appears as a different, slower one that is not really there.

**Wide enough for the clip's length.** A clip of `N` frames at `f` frames per
second resolves frequencies in steps of `f / N`. A band narrower than one step
contains no frequency at all and the filter returns zeros, so the output equals
the input. The library warns when this happens and says how many frames it
would need. The `pulse` preset needs about 181 frames — six seconds at 30 per
second — before its band contains anything.

## `level` — how far to shrink, for colour

How many times to halve the resolution before filtering. More shrinking means
less noise and a smoother result, at the cost of fine detail. The preset uses
4, so a 512-pixel-wide frame is filtered at 32 pixels wide.

Shrink too far and the region you care about disappears into a single sample.

## `lambda_c` — the spatial cutoff, for motion

In pixels. Detail finer than this is amplified progressively less.

This is the artefact control. Raise it when the result shimmers; lower it to
amplify finer movement, and accept more noise. The presets use 16.

## `chrom_attenuation` — how much colour to amplify

A multiplier on the colour channels relative to brightness.

For colour work, 1.0: the colour change is the signal. For motion, around 0.1:
the movement is in brightness, and amplifying colour just amplifies colour
noise, which is very visible.

## Where the shipped values come from

Every preset records its own provenance:

```python
from evm.presets import PRESETS
print(PRESETS["pulse"].params)
print(PRESETS["pulse"].source)
```

The `pulse` and `motion` presets are the values the original authors used for
their own clips. The `vibration` preset comes from an example in this
repository and its description says so, because that is weaker evidence and the
reader should know.
