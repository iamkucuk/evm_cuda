# Measure vibration

A guitar string, a spinning machine, a bridge under load: all move by amounts
too small to see, at frequencies far above anything a body does. The motion
pipeline amplifies them, and the result can be read as a measurement.

```python
import vidmag

vidmag.magnify("guitar.mp4", preset="vibration", out="vibration.mp4")
```

## The frame rate decides whether it is possible

A camera recording at `f` frames per second cannot represent anything vibrating
faster than `f / 2`; above that, the movement appears as a slower one that is
not really there.

The shipped `vibration` preset keeps 72 to 92 cycles per second — a guitar's
low E and A strings. Resolving 92 needs more than 184 frames per second, so a
high-speed camera. At an ordinary 30 fps that band is far above the limit and
the preset returns your input unchanged. So pick the band from what you are
measuring, and check the camera can reach it.

| Measuring | Typical band | Frame rate needed |
|---|---|---:|
| Guitar low E string | 82 Hz | over 165/s |
| Mains-powered motor hum | 50 or 60 Hz | over 120/s |
| A tall building or bridge swaying | 0.2 to 2 Hz | 30/s is plenty |
| A washing machine drum | 10 to 20 Hz | over 40/s |

## Finding the frequency instead of assuming it

Run the movement through a wide band first and look at what comes out:

```python
import numpy as np
import vidmag

fps = 240.0
amplified = vidmag.magnify(
    "machine.mp4", preset="motion", fps=fps, alpha=25, lambda_c=16, r1=0.4, r2=0.05
)

# One number per frame, from a patch on the moving part.
signal = amplified[:, 200:260, 300:360, :].mean(axis=(1, 2, 3))
spectrum = np.abs(np.fft.rfft(signal - signal.mean()))
freqs = np.fft.rfftfreq(len(signal), 1 / fps)
print(f"strongest movement at {freqs[1:][spectrum[1:].argmax()]:.1f} Hz")
```

Then narrow the band around that peak and re-run to see it clearly.

## Keep the camera still

Everything here is relative to the frame. A hand-held phone moves far more than
the thing being measured, and the amplification faithfully makes that movement
enormous. Use a tripod. If the camera did move, align the frames to each other
before magnifying.
