# PyTorch backend: what the validation found, 2026-08-11

Plan step 4P.1 asked for a probe before any implementation: does PyTorch on
Apple's graphics processor support the operations the four pipelines need, and
does it agree with the NumPy baseline? This records the answers, including two
that changed the implementation.

Machine: Apple M2 Max, macOS, PyTorch 2.13.0, computing on the Apple GPU
through Metal Performance Shaders.

## Every operation the pipelines need is supported

| Operation | Result | Largest difference from the baseline |
|---|---|---:|
| Blur-and-halve (conv2d, reflect padding, stride) | works | 1.30e-07 |
| The r1/r2 recursive filter | works | 4.57e-07 |
| Real Fourier transform along time | works | 4.95e-06 |
| Inverse transform, round trip | works | 4.77e-07 |
| Enlarge by zero-insert and filter | works | — |
| Half-precision arithmetic | works | — |
| Complex arithmetic and angle | works | — |

The plan flagged the Fourier transform as the operation least likely to exist
on Apple's GPU, and named two fallbacks to pick between if it were missing: a
processor-side transform, or shipping only the recursive filters there.
**Neither is needed.** No pipeline is degraded on any device.

## Three findings that changed the implementation

**Apple's GPU has no double precision at all.** Not slow — absent. It refuses
the transfer with `Cannot convert a MPS Tensor to float64`. The baseline
computes in double throughout, so `from_numpy` narrows double to single on the
way in. Integer frames pass through untouched. This is the same single-versus-
double gap the other device backends already have, and the conformance
tolerance already covers it.

**The frequency filter is not a real-input transform.** Writing it the obvious
way — `rfft`, mask, `irfft` — produced output unrelated to the reference,
differing by the full amplitude of the signal. The reference follows the
original MATLAB: it takes the *full complex* transform and masks against a
one-sided frequency ramp `arange(n)/n*rate` across every bin, which discards
the upper half of the spectrum rather than treating it as the conjugate mirror,
then takes the real part. Those are different filters. The implementation now
does what the reference does.

**Rounding differs at exact halves.** The reference rounds half away from zero;
`torch.round` rounds half to even. On the final 8-bit conversion that is a
one-level difference on any pixel landing exactly on .5. Adding 0.5 and
truncating reproduces the reference.

## A near miss worth recording

The first run of the probe reported the Fourier transform as diverging by 7.15
— which, had it been believed, was an argument for one of the fallbacks or for
abandoning the phase. It was wrong: the comparison cast complex results to real
and silently discarded the imaginary half. The round-trip test, which recovered
the original signal to 4.77e-07, already contradicted it.

The lesson is not about complex numbers. A probe that decides whether a phase
goes ahead needs its own result checked as carefully as the thing it measures.

## What the finished backend agrees with, and what it costs

Against the NumPy baseline, both complete pipelines land within **one step of
the 8-bit output** — the same bar the OpenCL, Apple and Vulkan backends meet.
Streaming through it matches the baseline to the same bar.

Speed on the Apple M2 Max, magnification only, one warm-up then the best of two
runs, alongside the other backends measured the same way:

| Backend | Colour, `face.mp4` | Motion, `baby.mp4` |
|---|---:|---:|
| OpenCL | 217 ms | 1,020 ms |
| Vulkan | 255 ms | 1,462 ms |
| Apple Metal | 362 ms | 1,698 ms |
| **PyTorch** | **653 ms** | **2,320 ms** |
| Processor (NumPy) | 7,014 ms | 23,634 ms |

It is the fastest backend for live, frame-at-a-time work, which is a different
question from batch throughput. Pushing 720p frames one at a time:

| Machine and backend | frames per second |
|---|---:|
| **RTX 3090, PyTorch** | **107.6** |
| Apple M2 Max, Apple's own interface | 58.8 |
| Apple M2 Max, PyTorch | 44.6 |
| Apple M2 Max, Vulkan | 20.5 |
| Apple M2 Max, processor baseline | 8.1 |
| RTX 3090, processor baseline | 6.3 |
| Apple M2 Max, OpenCL | 3.9 |

The hand-written NVIDIA backend does not appear because it cannot stream: it
has no implementation of the primitive operations, and refuses. So on NVIDIA
hardware this backend is currently the only way to magnify a live stream, which
is a concrete reason for it beyond convenience.

**That is not a general ranking, and reading it as one is a mistake worth
guarding against.** Being fastest at frame-at-a-time work says nothing about
whole-clip work, where the hand-written backend wins clearly. Same RTX 3090,
same clip, same entry point, array in and array out, each in its own process:

| Backend | Whole clip, 301 frames | frames/sec |
|---|---:|---:|
| **Hand-written NVIDIA (CUDA)** | **239 ms** | **1,261** |
| PyTorch | 664 ms | 453 |

The hand-written code is 2.8 times faster, which is the expected result and the
reason this project exists. The two figures answer different questions: how
quickly can a whole clip be processed, and can a live stream be kept up with.

For batch work it is the slowest of the four graphics backends and roughly ten
times faster than the processor. That is the expected trade, and the reason the
plan put it last in the order automatic selection walks: for whole-clip work it
reaches no hardware the others miss, so it should never be chosen over a native
backend that can do the job.

What it adds is four things, and only the last was anticipated when the plan
called this optional: it is the only way to stream on NVIDIA hardware; it is the
fastest streaming measured anywhere here; it runs wherever PyTorch is already
set up, with no driver work; and it keeps results as tensors, so magnification
can sit inside a larger tensor computation. Being an independent implementation
is a fifth: written in a different library from the same definitions, so
agreement with the baseline is evidence about the definitions rather than about
one way of expressing them.
