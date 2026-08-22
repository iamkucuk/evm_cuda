# See a pulse

Blood arriving in the skin with each heartbeat changes its colour by a fraction
of one step in 255. The colour pipeline amplifies exactly that.

```python
import vidmag

vidmag.magnify("face.mp4", preset="pulse", out="pulse.mp4")
```

## What the preset does

| Parameter | Value | Controls |
|---|---|---|
| `alpha` | 50 | How much the change is amplified |
| `level` | 4 | How far the image is shrunk before filtering |
| `fl`, `fh` | 0.833–1.0 Hz | The band kept: 50–60 beats per minute |
| `chrom_attenuation` | 1.0 | How much colour, versus brightness, is amplified |

These are the authors' own values for their face clip. They live in
`vidmag.presets.PRESETS`, each with a note on where it came from.

## Adjusting it for your clip

**The heart rate is not 50–60 bpm.** Divide beats-per-minute by 60 to get the
band. A resting adult at 72 bpm is 1.2 cycles per second, so roughly 1.0 to 1.4:

```python
vidmag.magnify("face.mp4", preset="pulse", fl=1.0, fh=1.4, out="pulse.mp4")
```

**The clip is short.** A clip of `N` frames at `f` frames per second resolves
frequencies in steps of `f / N`; a band narrower than one step selects nothing,
and the result equals the input. The library warns and says how many frames it
needs. Record longer or widen the band.

**It flickers or looks blotchy.** `alpha` is too high for the noise in the clip
— halve it. Amplification cannot tell signal from sensor noise, and the noise
usually breaks first.

**Nothing happens even with the right band.** Check the lighting is steady.
Mains-powered light flickers at 50 or 60 cycles per second, and a camera
exposure beating against it makes a slow brightness ripple that swamps a pulse.

## Reading a heart rate out

The amplified result is an array, so you can take a number from it instead of
watching a video. Average each frame over a patch of skin, then find the
strongest frequency:

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

Pick the patch over skin, away from hair and the edges of the face. This shows
the idea; it is not a medical measurement.
