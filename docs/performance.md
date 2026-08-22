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
| H100 80GB (Hopper) ◊ | 17.0 / 13.8 ms |
| RTX 3090 (Ampere) | 40.3 / 26.8 ms |
| P100 16GB (Pascal) | does not fit / 82.8 ms |
| T4 16GB (Turing) ‡ | does not fit / 137.2 ms |
| A100 80GB † | 54.4 / 48.2 ms |

† Measured before the three motion-path changes described above, and not
re-run, so this table mixes two versions of the code. It could not be re-run on
2026-08-22: the cluster partition holding the A100s was down for a reboot.

The other four rows are current, and they are four different architectures —
which is what tells you the three changes are not an Ampere trick. Each card was
measured before and after, on its own hardware:

| | Motion, half precision, before | after | Colour, half precision, before | after |
|---|---:|---:|---:|---:|
| RTX 3090 (Ampere, sm_86) | 60.9 ms | 26.8 ms (2.3x) | 7.6 ms | 7.6 ms |
| P100 (Pascal, sm_60) | 139.7 ms | 82.8 ms (1.7x) | 21.8 ms | 21.9 ms |
| H100 (Hopper, sm_90) ◊ | 34.5 ms | 13.8 ms (2.5x) | 4.4 ms | 3.7 ms |
| T4 (Turing, sm_75) ‡ | 228.8 ms | 137.2 ms (1.7x) | 39.7 ms | 38.6 ms |

Colour is the control: it builds no Laplacian pyramid, so none of the three
changes reaches it, and movement in the colour column is the measurement rather
than the code. It is flat on the RTX 3090 and the P100, so those two are
controlled comparisons. It is not flat on the other two, and each says by how
much its own machine moves:

‡ Both T4 runs are single runs of `scripts/cloud/colab_benchmark.ipynb` on
Colab's shared hardware, median of 5 rather than the 7 used elsewhere. Between
them colour moved 2.8% in half precision and 12% in single precision, and colour
cannot have changed — so roughly 12% is that machine's noise floor. Motion moved
67%, well outside it.

◊ On the H100 colour moved 15%. The colour kernels are byte-identical between
the two runs: the only change to those three source files in between is comment
text. So that 15% is the environment, and the older run records neither a date
nor a commit to narrow it further. Motion moved 150%, an order of magnitude
outside it. The run also held one of four GPUs on a shared cluster node, which
leaves its transfer figures looser than its kernel figures.

The direction and rough size hold on all four cards. The exact factor is
trustworthy only for the RTX 3090 and the P100. The remaining A100 row is
pessimistic by an amount nobody has measured.

Motion in single precision needs 16.3 GB and does not fit a 16 GB card; in half
precision it peaks at 8.4 GB and does. The P100 and T4 runs report the skip
rather than failing partway.

Sources, each recording the commit it was taken at:
`benches/bench_rtx3090.json` (2026-08-18, `scripts/dev/record_gpu_bench.py`),
`benches/bench_h100.json` (2026-08-22, the same script on a private HPC
cluster node),
`benches/bench_p100.json` (2026-08-22,
`scripts/cloud/kaggle/run_gpu_comparison.py`, console log in
`benches/kaggle_runs/`), and `benches/bench_t4.json` (2026-08-22, transcribed
from the Colab notebook's printed output, which writes no JSON of its own).

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
