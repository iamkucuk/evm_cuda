# Amplify motion

Breathing, a pulse visible as movement rather than colour, a structure settling,
the small sway of something that looks still. The motion pipeline makes these
visible by amplifying how much the picture's detail shifts over time.

```python
import evm

evm.magnify("baby.mp4", preset="motion", out="breathing.mp4")
```

## Which of the three filters to use

The motion pipelines differ only in how they pick out the frequencies to
amplify, and the choice matters more than it looks.

| Filter | Choose it when | Cost |
|---|---|---|
| Two running averages (`motion` preset) | You want a broad band and the simplest behaviour, or you are working frame by frame | Cheapest; needs only the previous frame |
| Butterworth | You want a defined band with a gentle edge, and still want it to work frame by frame | Cheap; needs only the previous frame |
| Fourier | You want a sharply defined band and have the whole clip | Needs every frame at once |

The `motion` preset uses the first. It is what the original authors used for
the sleeping-child clip.

## The two parameters that matter

**`alpha`** is how much to amplify. Raise it until the movement is clear, then
stop. Past that point the picture tears into ripples and haloes, because the
method assumes the movement is small enough that shifting the image is nearly
the same as scaling its detail — and that assumption fails first at edges.

**`lambda_c`** is the spatial cutoff, in pixels. Detail finer than this is
amplified less, on purpose: fine detail is where the small-movement assumption
breaks, and where the sensor noise is. Raising it reduces artefacts and
amplifies less; lowering it does the reverse.

```python
gentle = evm.magnify("clip.mp4", preset="motion", alpha=5, lambda_c=32)
strong = evm.magnify("clip.mp4", preset="motion", alpha=25, lambda_c=8)
```

If the result looks like it is made of shimmering ripples, that is too much
`alpha` for the `lambda_c` in use. Raise `lambda_c` before lowering `alpha`.

## Amplifying colour or not

`chrom_attenuation` scales how much the colour channels are amplified relative
to brightness. The `motion` preset sets it to 0.1, so colour is amplified about
a tenth as much. Colour noise is very visible and rarely carries the movement
you want, so keeping it low is usually right.

## A clip that is too long to fit in memory

Every pipeline here takes the whole clip at once. For something long, work in
overlapping chunks and discard the overlap:

```python
import numpy as np
import evm
from evm.io.video import load_video

video, info = load_video("long.mp4")
frames = np.clip(np.rint(video * 255), 0, 255).astype(np.uint8)

chunk, overlap, pieces = 300, 30, []
for start in range(0, len(frames), chunk):
    block = frames[max(0, start - overlap) : start + chunk]
    out = evm.magnify(block, preset="motion", fps=info.fps)
    pieces.append(out[overlap:] if start else out)

result = np.concatenate(pieces)
```

The overlap exists because the filters need history: without it, every chunk
would restart from nothing and the joins would show.
