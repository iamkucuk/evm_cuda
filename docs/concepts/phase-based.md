# Phase-based magnification

A second way to amplify motion, from the 2013 follow-up to the original paper.
Both methods make small movement visible; they differ in what they amplify, and
the difference decides which one to use.

## The difference

The original method builds a pyramid of image detail and scales the difference
between neighbouring scales. That difference approximates what a small shift
does to a picture, so adding a multiple of it back approximates shifting
further. The approximation holds while the movement is small compared with the
detail it moves — and when it stops holding, edges tear into ripples and haloes.

The phase-based method builds a decomposition in which every part of the image
has a phase as well as a strength, arranged so that a shift shows up as a
change in phase. Instead of approximating a shift, it changes the phase, which
*is* a shift for that part of the image. Nothing has to stay small for that to
be true, so it holds together at amplifications where the original breaks.

## Using it

```python
import numpy as np
import evm

frames = np.zeros((60, 64, 64, 3), dtype=np.uint8)     # your frames
out = evm.magnify(frames, preset="motion_phase", backend="cpu", fps=30.0)
```

Or directly, with more control:

```python
import numpy as np
from evm.cpu.phase_magnify import phase_magnify

frames = np.zeros((60, 64, 64, 3), dtype=np.uint8)
out = phase_magnify(frames, 30.0, alpha=15, fl=0.5, fh=1.5,
                    scales=3, orientations=4)
```

## When to choose it

**Choose it** when the original method produces artefacts before it produces a
visible movement — when raising `alpha` makes edges shimmer rather than making
the thing you want to see clearer.

**Choose the original** when it already works. It is far faster: the
phase-based method transforms every frame into the frequency domain and back
once for every scale and direction, where the original does a handful of small
convolutions.

## Its parameters

| Parameter | What it does |
|---|---|
| `alpha` | How much to amplify. Tolerates larger values than the original method. |
| `fl`, `fh` | The band to amplify, in cycles per second |
| `r1`, `r2` | An alternative to the band: two decay rates, as the original method's cheapest filter uses. Give one pair or the other. |
| `scales` | How many scales to split into. More reaches larger movement. |
| `orientations` | How many directions per scale. Four is usual. |
| `sigma` | Smooths the measured phase spatially, weighted by strength. Raise it if flat areas look speckled. |

Only brightness is processed. Colour is carried through untouched, because
movement is a brightness phenomenon at edges and amplifying colour phase adds
speckle without adding movement.

## How this one is verified, and how that differs

Everything else in this project is checked by comparison against the original
authors' published output. **That is not the case here**, and the difference
matters enough to state plainly.

The authors' rendered results for the phase-based method are not among the
files this project can fetch, so there is nothing to compare against. Instead
it is checked against motion constructed for the purpose: a texture shifted by
a known fraction of a pixel, at a known rate, with the output measured to see
whether the movement grew by the predicted amount. It does — the growth matches
`1 + alpha` times how much of the movement the temporal filter passed, and with
`alpha` set to zero the clip comes back unchanged.

That is evidence the method does what it claims. It is not evidence that this
implementation matches the authors' code in every detail, and it should not be
read as such. If you need that assurance, compare against their released
results yourself before relying on it.

## What is not implemented

**Graphics acceleration.** This runs on the processor only. The pipeline is
built from the same primitive operations as everything else, so a backend could
implement it, but none does yet — asking for another backend raises rather than
silently running somewhere unexpected.

**Half-octave scales and the more elaborate filter sets** from the paper. The
implementation here uses octave-spaced scales, which is the simplest form that
works.
