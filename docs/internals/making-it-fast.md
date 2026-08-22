# Making it fast: three rounds on one pipeline

A correct implementation of Eulerian Video Magnification, ported to CUDA, ran
the motion pipeline in about 934 ms of computation on an RTX 3090. It now runs
in 40.5 ms. This is what the three rounds of work in between changed, what each
one measured, and — twice — what each got wrong.

Everything below is the motion pipeline on `baby.mp4`, 301 frames at 960x544,
on an NVIDIA RTX 3090, unless another machine is named. "Computation" means the
kernels and the work between them, with the clip already on the card; the
transfers are counted separately at the end, because they turn out to matter
more than any of this.

## The three rounds in one table

| | Motion computation | What the round was about |
|---|---:|---|
| A correct port, per-kernel transfers | over 95% transfer | Nothing was wrong except the architecture |
| **Round 1** — keep the data on the card | ~934 ms | Stop copying between every kernel |
| **Round 2** — layout and allocation | ~76 ms | Where the data sits, and not freeing it constantly |
| **Round 3** — measure against the hardware | **40.5 ms** | Compare each stage to what the card can sustain |

Round 2 is the large one, twelve times over. Round 3 is the interesting one,
because two of its three changes reverse decisions made in Round 2 — decisions
that were themselves measured, and correct against what they were measured
against.

---

# Round 1: the architecture, not the kernels

The first CUDA port wrapped each kernel in its own binding that allocated
memory, copied the input to the card, ran, copied the result back, and freed.
With 291 frames processed one at a time, that is roughly 1,773 such calls per
run. Profiling showed **over 95% of the time was transfer and allocation**, not
computation.

No kernel was slow. The architecture was wrong.

The fix is to make the whole pipeline device-resident: the clip enters the card
once, stays there through colour conversion, pyramid construction, temporal
filtering, amplification and reconstruction, and leaves once. Everything else
in this round follows from that:

- **Batched launches.** One kernel over all frames of a pyramid level rather
  than one per frame, which turns 1,773 launches into a few dozen.
- **A plan cache for the Fourier transform**, keyed on the transform's shape, so
  the colour pipeline's frequency filter does not rebuild it every call.
- **Half precision as a storage option.** Bands, filtered bands and the
  amplified result stored as 16-bit halves the memory and lets the motion
  pipeline fit a 16 GB card, which in single precision it does not — it needs
  16.3 GB and peaks at 8.4 GB in half.

The lesson of this round is the least surprising of the three and the most
often needed: **measure where the time goes before optimising anything.** The
kernels were never the problem.

---

# Round 2: layout, allocation, and one failure worth keeping

With the pipeline device-resident, computation was about 934 ms and every stage
looked plausible. This round took it to about 76 ms, in steps, each measured on
the same clip and the same card:

| Step | Computation | What changed |
|---|---:|---|
| Start of the round | ~934 ms | |
| Filter where the data already lies | ~718 ms | |
| Stop discarding pyramid scratch each call | ~377–407 ms | |
| Reuse device buffers instead of freeing them | ~113–133 ms | |
| Channel-outer layout, contiguous bands | ~106–110 ms | no measurable win on its own |
| Shared memory in the shrinking kernel | ~98–102 ms | |
| Shared memory in the enlarging kernel | 127–189 ms | **worse; reverted** |
| Retune the enlargement kernels | ~76 ms | smaller tiles, no integer divide, skip dead taps |

Four things in that table are worth more than the numbers.

**The temporal filter was not slow at arithmetic; it was reading memory in the
wrong order.** It ran on data laid out with time as the slowest-varying axis,
so consecutive threads read addresses far apart. Filtering where the bands
already lie — time contiguous — made the same arithmetic match the speed of a
plain copy. No algorithm changed.

**Allocation was the largest single cost, twice.** After the filter was fixed,
the next slow stages were not slow at computing. They were rebuilding
multi-gigabyte scratch buffers on every pyramid call, and freeing device memory
on the hot path. Keeping the scratch and pooling the buffers took the pipeline
from about 407 ms to about 120 ms without touching a single kernel.

**A correct change can be worth nothing.** Reordering the data so each colour
channel's planes sit together, and making band writes contiguous, is better
engineering than what preceded it. It moved wall time by an amount
indistinguishable from noise. It stayed because it made later changes possible,
not because it paid for itself.

**Shared memory helped the shrinking kernel and hurt the enlarging one.** Adding
it to the shrink was a small real win. Adding it to the enlargement made things
worse — twice, on two attempts — and was reverted. The conclusion drawn at the
time was that enlargement has "weak access patterns" and is not worth fusing.

That conclusion was half right, and the half that was wrong survived into Round
3.

---

# Round 3: measuring against the hardware instead of against yesterday

Both earlier rounds optimise by comparison: try a change, keep it if the stage
got faster. That works, and it can never say when a stage is finished. A stage
that went from 40 ms to 30 ms is better; whether 30 ms is *good* is a different
question, and answering it needs something to compare against that is not the
previous version.

## First, find the ceiling

The card's specification says 936 GB/s. That is not the right number — it is
what the memory achieves under conditions no real kernel meets. So measure what
this card actually sustains, with a program that does nothing but move memory:

| Access pattern | Measured | Share of the 936 GB/s specification |
|---|---:|---:|
| Read and write | **863 GB/s** | 92% |
| Read only | 910 GB/s | 97% |
| Write only | 909 GB/s | 97% |

863 GB/s is the honest ceiling for almost every stage here, because almost
every stage reads and writes.

## Then count what each stage must move

Given the sizes of the arrays a stage touches, how many bytes does it have to
move? Divide by the measured time and there is a number to compare against 863.

That count is a floor, not a measurement — a kernel can move more than it must.
So a stage at 50% might be moving twice the necessary bytes efficiently, or the
necessary bytes at half speed. It says *look here*, not *here is the fault*.
The hardware counters that would settle it need administrator rights this
machine does not have. Arithmetic and a stopwatch were enough.

Where the pipeline stood at the start of Round 3:

| Stage | GB moved | Time | Achieved | Of ceiling |
|---|---:|---:|---:|---:|
| Colour convert | 2.28 | 2.70 ms | 843 GB/s | 98% |
| Pyramid build | 10.94 | 20.58 ms | 532 GB/s | **62%** |
| Temporal filter | 4.86 | 6.12 ms | 795 GB/s | 92% |
| Pyramid reconstruct | 7.90 | 13.36 ms | 591 GB/s | **69%** |
| Amplify and render | 4.10 | 4.59 ms | 894 GB/s | 104% |

Two rows need explaining before the two that look bad. Render reads two 32-bit
arrays and writes a smaller 8-bit one; read-heavy work has a higher ceiling —
910 GB/s measured — so 104% of the read-and-write figure means it is at its own
limit. And the temporal filter at 92% had been at 28% a week earlier.

## The temporal filter was doing arithmetic, not moving memory

The r1/r2 filter keeps two running averages per pixel across time. Its state
was 64-bit floating point, on a defensible argument: a 300-frame recursion
accumulates rounding error roughly as the square root of its length, and 32-bit
arithmetic would spend most of the accuracy budget on it.

Sound in general, wrong for this filter. r1/r2 is a pair of *decaying*
averages: each step multiplies the old value by less than one, so an error
introduced at step 50 has nearly vanished by step 300. A decaying average
forgets its own error rather than accumulating it, and the square-root bound
never comes close to binding.

Measured over the largest pyramid level, the largest difference between 32-bit
state and 64-bit state is **4.023e-07**, against a budget of 1e-5.

The cost of that unnecessary precision was severe, because this class of card
runs 64-bit arithmetic at one sixty-fourth the rate of 32-bit:

| State | Time | Achieved |
|---|---:|---:|
| 64-bit | 4.95 ms | 246 GB/s |
| 32-bit | 1.50 ms | 809 GB/s |

The stage went from arithmetic-bound to memory-bound, which is where a filter
this simple belongs. In the pipeline it fell from 24.65 ms to 6.36 ms.

The Butterworth filter keeps its 64-bit state: it is a true recursion with
feedback on its own output, the square-root argument does apply to it, and it
is not on the hot path.

## The pyramid stages wrote things to read them straight back

Building one pyramid level: shrink the image, enlarge it back, subtract the
result from the original. Reconstruction is the same with an addition. Both did
that last step as a separate pass — the enlargement wrote a full-resolution
intermediate, and a second kernel immediately read it back to combine it with a
band.

At the finest level that intermediate is 1.8 GB. Writing and re-reading it is
3.6 GB of traffic for no arithmetic reason, on stages whose whole cost is
traffic. Handing the band to the enlargement kernel, so the combination happens
inside the store that was already occurring, removed both:

| Stage | Before | After |
|---|---:|---:|
| Pyramid build | 27.83 ms | 21.44 ms |
| Pyramid reconstruct | 20.51 ms | 13.99 ms |

## The shared memory was the problem, not the solution

Both pyramid stages were still near 60% of the ceiling. The enlargement kernels
staged their input in shared memory — fast on-chip storage — as the shrinking
kernel beside them did and still does.

**Round 2 put that shared memory there deliberately.** Its attempt at *fusing*
the enlargement lost twice, and the conclusion drawn was that enlargement has
weak access patterns. The staging itself was never questioned.

Testing it four ways, all producing bit-identical output:

| Variant | Enlarge rows | Enlarge columns |
|---|---:|---:|
| Shared memory, one output per thread (what shipped) | 474 GB/s (55%) | 471 GB/s (55%) |
| **No shared memory**, one output per thread | 720 GB/s (83%) | 671 GB/s (78%) |
| Shared memory, four outputs per thread | — | 683 GB/s (79%) |
| **No shared memory**, four outputs per thread | 862 GB/s (100%) | 864 GB/s (100%) |

Removing the staging alone recovered most of it. The reason is in what an
enlargement does: it inserts a gap between every pair of samples, so of the five
filter taps only two or three land on real data, and each output reads only
those two or three inputs. Neighbouring threads read overlapping inputs, and the
ordinary cache already serves that overlap. Staging bought nothing and cost a
synchronisation barrier plus a loading loop shaped by the tile rather than by
the hardware — in the column kernel, 96 of 256 threads sat idle during the load,
and each group of 32 threads had its reads split across two short misaligned
pieces.

The shrinking kernel keeps its shared memory, and should: it reads a 2x2
neighbourhood plus halo per output, genuinely reuses staged data, and measures
94%.

**What shipped is not the fastest variant.** The 100% form uses 16-byte
vectorised access, which needs widths divisible by four; two pyramid levels of
this clip have widths 15 and 30. It would need a second kernel, a fallback for
those levels, and separate treatment for half precision. The shipped form
spaces each thread's four outputs one warp apart, so every store is still 32
consecutive values, no alignment is required, and one kernel serves every level
and both number formats. That trades about six percentage points for not
maintaining three kernels where one does.

---

# Where it stands

Per stage, after all three rounds:

| Stage | GB moved | Time | Achieved | Of ceiling |
|---|---:|---:|---:|---:|
| Colour convert | 2.28 | 2.66 ms | 858 GB/s | 99% |
| Pyramid build | 10.94 | 17.16 ms | 637 GB/s | 74% |
| Temporal filter | 4.86 | 5.96 ms | 816 GB/s | 95% |
| Pyramid reconstruct | 7.90 | 9.39 ms | 842 GB/s | 98% |
| Amplify and render | 4.10 | 4.45 ms | 922 GB/s | 107% |
| **All computation** | **30.09** | **39.62 ms** | **759 GB/s** | **88%** |

Round 3's effect alone, measured against the same code with its three changes
removed and rebuilt on the same machine in the same session:

| | Before | After | |
|---|---:|---:|---:|
| Motion, single precision | 76.7 ms | 40.5 ms | 1.89x |
| Motion, half precision | 60.9 ms | 27.0 ms | 2.25x |

The colour pipeline is untouched by all three and measured unchanged, which is
the control that makes those figures a comparison rather than a difference
between sessions.

*Both sides of that comparison, and every stage figure on this page, come from
one session. A fresh measurement on 2026-08-18 put the same motion pipeline at
40.3 ms and 26.8 ms — within about half a percent — and is what
`docs/performance.md` and `benches/bench_rtx3090.json` now carry. The figures
here are deliberately not restated from it: the value of a before-and-after pair
is that both halves were measured on the same machine in the same session, and
swapping one half for a number from a different day would destroy that.*

*Added 2026-08-22: this held on a second architecture. Round 3 was designed
against an RTX 3090, and one card cannot tell you whether a result is a property
of the algorithm or of Ampere. A Tesla P100 (Pascal, sm_60) had been measured on
the same harness before Round 3 and was re-run on it after: motion in half
precision went from 139.7 ms to 82.8 ms, 1.7x, against 2.25x here. Colour went
26.3 → 26.4 ms and 21.8 → 21.9 ms, unchanged, the same control as above. Less
gain on the older card and still a real one — the two runs are
`benches/bench_p100.json` and its predecessor in that file's history.*

Pyramid build is the one row still short of the others at 74%, and the number
probably understates it: the shrinking kernel inside it stages overlapping
tiles, so it genuinely re-reads data the model does not count. Timed on its
own, that kernel measures 94%.

## What is left, and it is not the kernels

| | Time |
|---|---:|
| All kernels | 39.1 ms |
| Copying the clip to the card and the result back | 104.5 ms |
| Everything else `vidmag.magnify()` does | 85.2 ms |

Reading the result back costs 72.3 ms on its own — more than every kernel put
together. That is not bandwidth: a plain copy into a freshly allocated host
array runs at 3.0 GB/s against 12 GB/s into a reused one, because every page of
a new 456 MB array is touched for the first time as the copy fills it. Making
the transfer page-locked was tried and made it worse, 84.5 ms to 155.4 ms,
because locking 456 MB per call costs more than it saves.

The fix is to stop allocating a fresh output array per call, which means
letting callers supply one. That is a change to the public interface rather
than to a kernel, and it has not been made.

---

# What the three rounds taught

**Measure where the time goes before optimising anything.** Round 1's entire
gain came from noticing that 95% of the time was transfer. No kernel needed
touching.

**Allocation is a cost, and it hides.** Twice in Round 2 the slow stage was not
slow at computing — it was rebuilding scratch or freeing device memory on the
hot path. A kernel at 93% of bandwidth inside a stage taking 60 ms means the
stage is not the kernel.

**Optimising against yesterday cannot tell you when to stop.** Rounds 1 and 2
both improved every step they kept, and both left the pipeline at roughly 60%
of what the card could do, because nothing in the method could see the ceiling.
"532 GB/s against 863" is a question. "20.58 ms" is not.

**A failed experiment can close a question it never asked.** Round 2 tried
fusing the enlargement kernels, lost twice, and concluded enlargement had weak
access patterns. The shared memory inside those kernels was never the thing
under test, and it survived for months as the largest single inefficiency in
the pipeline.

**Good defaults are still defaults.** Use 64-bit accumulators in long
recursions; stage reused data in shared memory. Both are correct advice. Both
were wrong here, and the second was wrong *in this project's own earlier work*,
which measured — but measured against itself.

---

# Provenance

Every figure here was measured on the machine named beside it, with one warm-up
run and the median of seven, one configuration per process. The scripts are in
the repository. Raw per-card results are in `benches/`.

This document replaces three earlier write-ups, one per round, which are kept
unedited at `docs/dev/archive/` as the dated records of what was measured when.
They contain more detail per round than is reproduced here, including the
attempts that failed and were reverted.
