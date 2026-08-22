# Performance

Every number here was measured on the machine named beside it. Timing covers
the magnification only; reading and writing the video are excluded, since those
depend on the codec rather than on this library.

## What is being timed

Four different things get called "how fast is it", and they differ by a factor
of six on the same clip. Whenever a figure appears below, it says which one it
is.

| Level | What it includes | Use it when |
|---|---|---|
| **Kernels** | The graphics processor's work alone, with the data already on the card | Comparing against another implementation's kernel time, or embedding this as one stage of a larger device-resident computation |
| **Kernels + upload** | The clip starts on the host | The result is consumed on the card — heart-rate estimation, motion features, a downstream network. You do not need a viewable video to extract information from one |
| **Kernels + upload + download** | The result comes back to the host | You need a video file at the end |
| **Wall clock** | What a caller of `vidmag.magnify()` actually waits | Comparing backends against each other, or estimating how long a job takes |

On an RTX 3090, one motion clip of 301 frames of which the pipelines process
291, measured 2026-08-18:

| Level | Time |
|---|---:|
| Kernels | 40.3 ms |
| Kernels + upload | 74.7 ms |
| Kernels + upload + download | 154.1 ms |
| Wall clock through `vidmag.magnify()` | 232.2 ms |

More than half the wall clock is not computation. Moving the clip to the card
and the result back costs 113.8 ms — reading the result back alone, at 79.4 ms,
costs about twice what every kernel put together does — and the remaining
78.1 ms is preparing the input array, checking it, and allocating the output.

Every figure on this page for this card comes from that one session and is
stored in `benches/bench_rtx3090.json`, which
`scripts/dev/record_gpu_bench.py` regenerates.

**Only compare like with like.** Quoting a kernel figure against another
project's wall clock overstates this library by about six times.

## NVIDIA RTX 3090, hand-written CUDA

Kernel time, median of seven runs after one warm-up, fresh process per
configuration.

| Pipeline | Clip | Single precision | Half precision |
|---|---|---:|---:|
| Colour | `face.mp4`, 291 frames processed | 9.8 ms | 7.6 ms |
| Motion | `baby.mp4`, 291 frames processed | 40.3 ms | 26.8 ms |

The motion pipeline was 76.7 ms in single precision earlier in this project's
life. Three changes account for the difference, all described in
[the internals notes](internals/design.md): the temporal filter's running state
moved from 64-bit to 32-bit floating point, which this class of card executes
sixty-four times faster; the add-and-subtract pass that formed each pyramid
level was folded into the write that was already happening; and the two kernels
that enlarge an image stopped staging their input in on-chip memory, which cost
more than it saved.

Those two enlargement kernels now run at 92% to 96% of what the card's memory
can sustain, and the shrinking kernel at 94%. There is little left to win in
the kernels; the remaining cost is the transfers above.

## Apple M2 Max, every backend

The same clips on the same machine, magnification only, one warm-up then the
best of two. Measured 2026-08-11.

| Backend | Colour, `face.mp4` | Motion, `baby.mp4` | Against this machine's processor |
|---|---:|---:|---:|
| OpenCL | 217 ms | 1,020 ms | 32x / 23x |
| Vulkan | 255 ms | 1,462 ms | 28x / 16x |
| Metal | 362 ms | 1,698 ms | 19x / 14x |
| PyTorch | 653 ms | 2,320 ms | 11x / 10x |
| Processor (NumPy) | 7,014 ms | 23,634 ms | — |

*Not re-verified since.* A reproduction attempt on 2026-08-18 could not confirm
these: the machine's graphics processor was at 100% utilisation from unrelated
software, and every graphics backend measured five to seven times slower while
the processor baseline was only 18% slower. That measurement proves nothing
either way and none of it was published.
`scripts/dev/record_backend_bench.py` regenerates this table on an idle machine.

**The ordering is not a property of the backends.** OpenCL leads here; on a
60-frame clip Apple's own interface was faster than OpenCL. The ranking changes
with clip length and would change again on other hardware, which is why the
library does not hard-code a preference beyond "native before PyTorch, anything
before the processor".

## Live streaming, 720p

Frames pushed one at a time, as a camera would. The target for real-time use is
30 frames per second.

| Machine and backend | frames per second | Keeps up? |
|---|---:|---|
| RTX 3090, PyTorch | 107.6 | yes |
| Apple M2 Max, Metal | 58.8 | yes |
| Apple M2 Max, PyTorch | 44.6 | yes |
| Apple M2 Max, Vulkan | 20.5 | no |
| Apple M2 Max, processor | 8.1 | no |
| RTX 3090, processor | 6.3 | no |
| Apple M2 Max, OpenCL | 3.9 | no |

Three backends keep up with a 30 frames-per-second camera at 720p. Note that
OpenCL, fastest on whole clips here, is slowest on live frames: batching is
where it wins.

**The hand-written NVIDIA backend does not appear because it cannot stream.**
It implements the four whole-clip pipelines but not the frame-at-a-time
operations, and refuses with an explanation rather than failing partway through
a frame. On NVIDIA hardware, PyTorch is the only way to magnify a live stream
today.

## Hand-written against PyTorch, same card

The comparison people ask for. Same RTX 3090, same clip, same entry point,
array in and array out, one backend per process.

| Backend | 291 frames processed | frames per second |
|---|---:|---:|
| Hand-written CUDA | 238 ms | 1,223 |
| PyTorch | 596 ms | 488 |

The hand-written code is 2.5 times faster, which is the expected result and the
reason this project exists. That the PyTorch backend is the fastest at live
streaming is a separate fact about a different kind of work; neither figure
generalises to the other.

*One backend per process matters here.* With both in one process, the NVIDIA
backend's device memory pool starves whatever runs next: measured on
2026-08-18, PyTorch takes 4,003 ms that way against 596 ms alone, which is 6.7
times slower for no reason other than the measurement setup.

## Other NVIDIA cards

| Card | Motion, kernels, single / half precision |
|---|---:|
| RTX 3090 | 40.3 / 26.8 ms |
| A100 80GB † | 54.4 / 48.2 ms |
| H100 80GB † | 35.8 / 34.5 ms |
| P100 16GB † | does not fit / 139.7 ms |

† Measured before the three motion-path changes described above, and not
re-run. Those are worth about 1.9 times on the RTX 3090 and none of it is
specific to one card, so treat these three rows as pessimistic by roughly that
much. The RTX 3090 row was re-measured on 2026-08-18 and is current.

Motion in single precision needs 16.3 GB and does not fit a 16 GB card; in half
precision it peaks at 8.4 GB and does.

## Reading these numbers honestly

**Speed-up figures depend entirely on what you compare against.** The same
graphics measurements divided by an Apple M2 Max give one ratio; divided by a
slower processor they give a much larger one. The graphics side does not
change. Wherever a ratio appears here, the processor it was measured against is
named.

**Machines drift.** This project's RTX 3090 host reads about 4% slower after a
restart, measured by rebuilding identical code and running it again in the same
session. Differences smaller than that between two figures here mean nothing.

**Half precision is not free.** It is faster and uses less memory, and it
differs from single precision by a measurable amount: on the motion pipeline
the two differ by 0.00140 of full scale in root-mean-square error, with a
largest single difference of 2 output levels out of 255 (measured 2026-08-18 on
`baby.mp4`, comparing the 8-bit outputs). The colour pipeline differs by
0.00071 and 1 level.
