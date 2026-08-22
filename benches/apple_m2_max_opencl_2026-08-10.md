# Apple M2 Max, portable OpenCL backend, 2026-08-10

The first measurements of this project running on a graphics processor that is
not made by NVIDIA. The kernels are the ones in `src/vidmag/opencl/kernels.cl`,
compiled by Apple's OpenCL driver.

Machine: Apple M2 Max, 38 compute units, macOS. Timing covers magnification
only; decoding and encoding the video are excluded, as in every other
measurement recorded here. One warm-up run first, so the figures exclude the
one-off cost of the driver compiling the kernels.

| Pipeline | Clip | Processor cores | Graphics processor | Ratio |
|---|---|---:|---:|---:|
| Colour (`pulse` preset) | `face.mp4`, 301 frames, 528x592 | 6,347 ms | **222 ms** | 28.6x |
| Motion (`motion` preset) | `baby.mp4`, 301 frames, 960x544 | 28,303 ms | **1,504 ms** | 18.8x |

Both outputs differ from the NumPy reference by at most one step of the final
8-bit rounding, which is the same bar the whole-pipeline conformance tests
apply.

## How this compares with the hand-written CUDA code

The CUDA backend on an RTX 3090 does the same colour clip in about 9.5 ms and
the same motion clip in about 74 ms. The portable backend is therefore roughly
twenty times slower than the tuned one, on different hardware. That gap is
expected and is the trade the design makes: the CUDA path fuses stages together
and collapses kernel launches, while the portable path runs each primitive
operation as its own kernel because that is what lets one source file work on
every vendor's driver.

The comparison to draw is not between the two graphics processors. It is that a
machine which previously had no acceleration at all now runs the motion
pipeline in one and a half seconds instead of twenty-eight.

## Reproducing

```bash
pip install -e ".[opencl]"
python -c "
import numpy as np, time
from vidmag.io.video import load_video
from vidmag.presets import PRESETS
from vidmag.backend import generic
from vidmag.opencl.ops import OpenClOps
from vidmag.cpu import magnify as direct

ops = OpenClOps()
video, info = load_video('data/face.mp4')
frames = np.clip(np.rint(video * 255), 0, 255).astype(np.uint8)
params = dict(PRESETS['pulse'].params)

generic.color_gdown_ideal_core(ops, frames[:8], info.fps, **params)   # warm up
t = time.perf_counter()
generic.color_gdown_ideal_core(ops, frames, info.fps, **params)
print('graphics processor:', (time.perf_counter() - t) * 1000, 'ms')
t = time.perf_counter()
direct.color_gdown_ideal_core(frames, info.fps, **params)
print('processor cores:   ', (time.perf_counter() - t) * 1000, 'ms')
"
```

## Not measured

AMD and Intel graphics processors. The same kernels should run there, because
the driver is what compiles them, but nobody has run them on that hardware and
this project does not claim results it has not measured. A contributed run is
welcome.

## Superseded, and one failed reproduction (added 2026-08-18)

**Superseded.** An all-backends session the next day, 2026-08-11, measured this
same machine again and got different figures: colour 217 ms and motion 1,020 ms
on OpenCL, against a processor baseline of 7,014 ms and 23,634 ms. Those are the
numbers `docs/performance.md` and `docs/concepts/backends.md` carry, and they
are the ones to use. The figures above are kept because they are what was
recorded on the day.

**A reproduction attempt on 2026-08-18 was inconclusive and is recorded so
nobody repeats it under the same conditions.** Running the snippet above
verbatim on the same machine gave 1,220 ms for the colour pipeline, about five
and a half times the 222 ms above. It was not a regression in the backend and
it does not disprove either session: `ioreg` reported the graphics processor at
100% utilisation from unrelated software at the time. The check that separates
the two explanations is the processor baseline, which shares none of that
contention and came out only 18% slow. Every graphics backend measured five to
seven times slow that day; none of those numbers was published.

Re-run on an idle machine with:

```bash
python scripts/dev/record_backend_bench.py --machine "Apple M2 Max" \
    --date YYYY-MM-DD --out benches/backends_apple_m2_max.json
```
