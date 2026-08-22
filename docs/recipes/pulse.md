# See a pulse

Blood arriving in the skin with each heartbeat changes its colour by a fraction
of one step in 255. The colour pipeline amplifies exactly that.

```python
import vidmag

vidmag.magnify("face.mp4", preset="pulse", out="pulse.mp4")
```

## What the preset does

| Parameter | Value | What it controls |
|---|---|---|
| `alpha` | 50 | How much the change is amplified |
| `level` | 4 | How far the image is shrunk before filtering |
| `fl`, `fh` | 0.833 to 1.0 Hz | The band kept: 50 to 60 beats per minute |
| `chrom_attenuation` | 1.0 | How much colour, as opposed to brightness, is amplified |

These are the values the original authors used for their own face clip. They
are in `vidmag.presets.PRESETS`, along with where each came from.

## Adjusting it for your clip

**The heart rate is not 50 to 60 beats per minute.** Divide by 60 to get the
band. For a resting adult at 72 beats per minute, that is 1.2 cycles per
second, so a band of roughly 1.0 to 1.4:

```python
vidmag.magnify("face.mp4", preset="pulse", fl=1.0, fh=1.4, out="pulse.mp4")
```

**The clip is short.** The band has to be wide enough for the clip's length to
resolve it. A clip of `N` frames at `f` frames per second resolves frequencies
in steps of `f / N`, and a band narrower than one step selects nothing at all —
the result is then identical to the input. The library warns when this happens
and says how many frames it would need. Either record longer or widen the band.

**The result flickers or looks blotchy.** `alpha` is too high for the noise in
the clip. Halve it. Amplification does not distinguish signal from sensor
noise, and the noise is usually the thing that breaks first.

**Nothing happens even though the band is right.** Check the lighting is
steady. A room lit by mains electricity flickers at 50 or 60 cycles per second,
and a camera exposure that beats against it produces a slow brightness ripple
that can swamp a pulse.

## Reading a heart rate out, rather than looking at it

The amplified signal is an array, so you can take a number from it instead of a
video. Average each frame over a patch of skin, then find the strongest
frequency:

```python
import numpy as np
import vidmag

amplified = vidmag.magnify("face.mp4", preset="pulse")
patch = amplified[:, 100:200, 100:200, :].mean(axis=(1, 2, 3))

fps = 30.0
spectrum = np.abs(np.fft.rfft(patch - patch.mean()))
freqs = np.fft.rfftfreq(len(patch), 1 / fps)
band = (freqs > 0.7) & (freqs < 3.0)  # 42 to 180 beats per minute
print(f"{freqs[band][spectrum[band].argmax()] * 60:.0f} beats per minute")
```

Choose the patch over skin, avoiding hair and the edges of the face. This is a
demonstration of the idea, not a medical measurement.
