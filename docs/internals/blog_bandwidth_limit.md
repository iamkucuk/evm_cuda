# Finding the ceiling, and getting to it

The third of three write-ups. The [first](blog_speedup.md) made a correct CUDA
port fast. The [second](blog_further_optimizations.md) went after memory layout
and allocation. This one asks a question neither of them did: **what is the
most this hardware can do, and how close are we?**

Everything below was measured on an NVIDIA RTX 3090 in August 2026, on
`baby.mp4` — 301 frames at 960x544, the motion pipeline. Every figure is
reproducible with the scripts named beside it.

## The question changes when you have a number to compare against

Both earlier write-ups optimise by comparison: try a change, keep it if the
stage got faster. That works, and it never tells you when to stop. A stage that
went from 40 ms to 30 ms is better; whether 30 ms is good is a different
question, and answering it needs a ceiling.

The card's specification says 936 GB/s. That is not the right number either —
it is what the memory can do under conditions no real kernel meets. So the
first step was to measure what this card actually sustains, with a program that
does nothing but move memory:

| Access pattern | Measured | Share of the 936 GB/s specification |
|---|---:|---:|
| Read and write | **863 GB/s** | 92% |
| Read only | 910 GB/s | 97% |
| Write only | 909 GB/s | 97% |

863 GB/s is the honest ceiling for almost every stage here, because almost
every stage reads and writes. Anything at 800 GB/s is essentially finished;
anything at 400 GB/s has half its time going somewhere other than moving data.

## Counting the bytes each stage must move

With a ceiling, the second half is arithmetic: how many bytes does each stage
have to move, given the sizes of the arrays it touches? Divide by the measured
time and you have a number to compare against 863.

That count is a floor, not a measurement — a kernel can move more than it must,
through poor access patterns or re-reading data. So a stage measuring 50% might
be moving twice the necessary bytes efficiently, or the necessary bytes at half
speed. It says *look here*, not *here is the fault*.

The hardware counters that would settle it need administrator rights on this
machine, which is not available. Arithmetic and a stopwatch turned out to be
enough.

Where the pipeline stood at the start of this work:

| Stage | GB moved | Time | Achieved | Of ceiling |
|---|---:|---:|---:|---:|
| Colour convert | 2.28 | 2.70 ms | 843 GB/s | 98% |
| Pyramid build | 10.94 | 20.58 ms | 532 GB/s | **62%** |
| Temporal filter | 4.86 | 6.12 ms | 795 GB/s | 92% |
| Pyramid reconstruct | 7.90 | 13.36 ms | 591 GB/s | **69%** |
| Amplify and render | 4.10 | 4.59 ms | 894 GB/s | 104% |

Two rows need explaining before the two that look bad. The render stage reads
two 32-bit arrays and writes a smaller 8-bit one; read-heavy work has a higher
ceiling — 910 GB/s measured read-only — so 104% of the read-and-write figure
means it is at its own limit, not that the measurement is wrong. And the
temporal filter at 92% had been at 28% a week earlier, for a reason worth its
own section.

## The temporal filter was doing arithmetic, not moving memory

The r1/r2 filter keeps two running averages per pixel across time. Its state
was 64-bit floating point, on a defensible argument: a 300-frame recursion
accumulates rounding error roughly as the square root of its length, and 32-bit
arithmetic would spend most of the accuracy budget on it.

The argument is sound in general and wrong for this filter. r1/r2 is a pair of
*decaying* averages: each step multiplies the old value by less than one, so an
error introduced at step 50 has almost vanished by step 300. A decaying average
forgets its own error rather than accumulating it, and the square-root bound
never comes close to binding.

Measured over the largest pyramid level, the largest difference between a
32-bit state and a 64-bit one is **4.023e-07**, against a budget of 1e-5.

The cost of that unnecessary precision was severe, because this class of card
runs 64-bit arithmetic at one sixty-fourth the rate of 32-bit. Same kernel,
same access pattern, only the state type changed:

| State | Time | Achieved |
|---|---:|---:|
| 64-bit | 4.95 ms | 246 GB/s |
| 32-bit | 1.50 ms | 809 GB/s |

The stage went from arithmetic-bound to memory-bound, which is where a filter
this simple belongs. In the whole pipeline it fell from 24.65 ms to 6.36 ms.

The Butterworth filter keeps its 64-bit state: it is a true recursion with
feedback on its own output, the square-root argument does apply to it, and it
is not on the hot path.

## The pyramid stages were writing things to read them straight back

Building one level of a Laplacian pyramid: shrink the image, enlarge it back,
subtract the result from the original. Reconstruction is the same with an
addition. Both were doing the last step as its own pass — the enlargement wrote
a full-resolution intermediate, and a second kernel immediately read it back to
combine it with a band.

At the finest level that intermediate is 1.8 GB. Writing it and reading it back
is 3.6 GB of traffic for no arithmetic reason, on stages whose whole cost is
traffic.

Handing the band to the enlargement kernel, so the combination happens inside
the store that was already occurring, removed both:

| Stage | Before | After |
|---|---:|---:|
| Pyramid build | 27.83 ms | 21.44 ms |
| Pyramid reconstruct | 20.51 ms | 13.99 ms |

## The shared memory was the problem, not the solution

Both pyramid stages were still near 60% of the ceiling after that, and the
obvious suspects had been checked. The enlargement kernels staged their input
in shared memory — fast on-chip storage — as the shrinking kernel beside them
did and still does. That is the textbook technique, and this project's earlier
write-ups both add shared memory to make things faster.

Testing it rather than assuming it, four ways, all producing bit-identical
output:

| Variant | Enlarge rows | Enlarge columns |
|---|---:|---:|
| Shared memory, one output per thread (what shipped) | 474 GB/s (55%) | 471 GB/s (55%) |
| **No shared memory**, one output per thread | 720 GB/s (83%) | 671 GB/s (78%) |
| Shared memory, four outputs per thread | — | 683 GB/s (79%) |
| **No shared memory**, four outputs per thread | 862 GB/s (100%) | 864 GB/s (100%) |

Removing the staging alone recovered most of it. The reason is in what an
enlargement does: it inserts a gap between every pair of samples, so of the
five filter taps only two or three land on real data, and each output reads
only those two or three inputs. Neighbouring threads read overlapping inputs,
and the ordinary cache already serves that overlap. Staging it bought nothing
and cost a synchronisation barrier plus a loading loop shaped by the tile
rather than by the hardware — in the column kernel, 96 of 256 threads sat idle
during the load and each group of 32 threads had its reads split across two
short misaligned pieces.

The shrinking kernel keeps its shared memory, and should: it reads a 2x2
neighbourhood plus halo per output, genuinely reuses staged data, and measures
94%.

**What shipped is not the fastest variant.** The 100% form uses 16-byte
vectorised access, which needs image widths divisible by four; two pyramid
levels of this clip have widths 15 and 30. It would need a second kernel, a
fallback for those levels, and separate treatment for half precision. The
shipped form spaces each thread's four outputs one warp apart, so every store
is still 32 consecutive values, no alignment is required, and one kernel serves
every level and both number formats. That trades about six percentage points
for not maintaining three kernels where one does.

## Where it ended up

The same table, on the same clip, after all three changes:

| Stage | GB moved | Time | Achieved | Of ceiling |
|---|---:|---:|---:|---:|
| Colour convert | 2.28 | 2.66 ms | 858 GB/s | 99% |
| Pyramid build | 10.94 | 17.16 ms | 637 GB/s | 74% |
| Temporal filter | 4.86 | 5.96 ms | 816 GB/s | 95% |
| Pyramid reconstruct | 7.90 | 9.39 ms | 842 GB/s | 98% |
| Amplify and render | 4.10 | 4.45 ms | 922 GB/s | 107% |
| **All computation** | **30.09** | **39.62 ms** | **759 GB/s** | **88%** |

Whole-pipeline computation, measured against the same code with the changes
removed and built on the same machine in the same session:

| | Before | After | |
|---|---:|---:|---:|
| Motion, single precision | 76.7 ms | 40.5 ms | 1.89x |
| Motion, half precision | 60.9 ms | 27.0 ms | 2.25x |

The colour pipeline is untouched by all three and measured unchanged, which is
the control that makes the motion figures a comparison rather than a difference
between sessions.

**Pyramid build is the one row still short of the others at 74%, and the number
probably understates it.** The shrinking kernel inside it stages overlapping
tiles, so it genuinely re-reads data the model does not count; its true traffic
is higher than the floor and its true efficiency correspondingly better. Timed
on its own, that kernel measures 94%.

## What is left, and it is not the kernels

At 88% of what the card sustains, the computation has little room left. The
transfers do:

| | Time |
|---|---:|
| All kernels | 39.1 ms |
| Copying the clip to the card and the result back | 104.5 ms |
| Everything else `evm.magnify()` does | 85.2 ms |

Reading the result back costs 72.3 ms on its own — more than every kernel put
together. That is not bandwidth: a plain copy into a freshly allocated host
array runs at 3.0 GB/s against 12 GB/s into a reused one, because every page of
a new 456 MB array is touched for the first time as the copy fills it. Making
the transfer page-locked was tried and made it worse, 84.5 ms to 155.4 ms,
because locking 456 MB per call costs more than it saves.

The fix is to stop allocating a fresh output array per call, which means
letting callers supply one. That is a change to the public interface rather
than to a kernel, and it has not been made.

## What this exercise was actually worth

The three changes are each small — a type, a fused store, a deleted
optimisation. None would have been found by the method the earlier write-ups
used, because none of the stages looked wrong; they looked like they had
already been optimised, and two of them had been.

What found them was having a number to compare against. 532 GB/s against a
ceiling of 863 is a question. 20.58 ms is not.

Two of the three also went against received practice: use 64-bit for
accumulators in long recursions, and stage reused data in shared memory. Both
are good defaults. Neither survived being measured on this particular pipeline
on this particular hardware, and the second had been added deliberately, by
this project, in an earlier round of optimisation that also measured — but
measured against itself rather than against the hardware's limit.
