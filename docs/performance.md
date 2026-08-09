# Performance

Every number here was measured on the machine named beside it. Timing covers
the magnification only; reading and writing the video are excluded, since those
depend on the codec rather than on this library.

## NVIDIA RTX 3090, hand-written CUDA

Against the NumPy reference on the same clips. Median of seven runs after one
warm-up.

| Pipeline | Clip | Compute | With transfers to and from the card |
|---|---|---:|---:|
| Colour, single precision | `face.mp4`, 301 frames | 9.5 ms | 73 ms |
| Colour, half precision | `face.mp4` | 7.9 ms | 74 ms |
| Motion, single precision | `baby.mp4`, 301 frames | 73.6 ms | 178 ms |
| Motion, half precision | `baby.mp4` | 58.7 ms | 170 ms |

Note how much of the total is moving data rather than computing. If the result
is consumed on the graphics processor by something else, only the first column
applies. If you need a video file at the end, the second is the honest figure.

## Apple M2 Max, portable OpenCL

The same clips, on a machine that previously had no acceleration at all.

| Pipeline | Clip | Processor cores | Graphics processor | Ratio |
|---|---|---:|---:|---:|
| Colour | `face.mp4`, 301 frames | 6,347 ms | 222 ms | 28.6x |
| Motion | `baby.mp4`, 301 frames | 28,303 ms | 1,504 ms | 18.8x |

Full details in
[the recorded measurement](https://github.com/iamkucuk/eulerian-video-magnification-cuda/blob/main/benches/apple_m2_max_opencl_2026-08-10.md).

## Reading these numbers honestly

**Speed-up figures depend entirely on what you compare against.** The same
graphics measurements divided by an Apple M2 Max give roughly 730x for the
colour pipeline; divided by a slower processor they give over 1,100x. The
graphics side did not change. Whenever a ratio appears here, the processor it
was measured against is named.

**The two tables are not comparable to each other.** Different machines,
different implementations. The portable backend runs each operation as its own
step so that one source works on every vendor's hardware, while the CUDA
backend fuses them; a few times slower is the cost of that portability.

**Half precision is not free.** It is faster and uses less memory, and it
differs from single precision by a measurable amount. The accuracy figures are
in the project's internal notes.
