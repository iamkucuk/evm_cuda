# The three original optimisation write-ups

One per round of work, written as each round finished, and kept unedited.

| File | Round | Written |
|---|---|---|
| `blog_speedup.md` | Making a correct CUDA port fast: device-residency, batched launches, half precision | 2026-07 |
| `blog_further_optimizations.md` | Memory layout and allocation, taking motion computation from ~934 ms to ~76 ms | 2026-08 |
| `blog_bandwidth_limit.md` | Measuring against the card's sustained bandwidth rather than against the previous version | 2026-08-11 |

`docs/internals/making-it-fast.md` tells the same story as one narrative and is
what the published site links to. These are here because they are dated records:
each says what was measured at the time, including the attempts that failed and
the conclusions later rounds overturned. The merged version necessarily reads
back from the answer; these do not.

They live under `docs/dev/`, which `mkdocs.yml` excludes from the site, so they
are in the repository and not on the published pages.

**Every GPU timing in these three files has since been superseded, and they were
deliberately left alone for the same reason.** They quote figures for five cards
that were current when each was written. Four of the five were re-measured on
2026-08-22, on code that includes three later rounds of work on the motion path,
and the motion figures roughly halved:

| Card | Motion, half precision, as written here | Current |
|---|---:|---:|
| RTX 3090 | 75.1 ms, then 27.0 ms † | 26.8 ms |
| H100 80GB | 34.5 ms | 13.8 ms |
| Tesla P100 | 139.7 ms | 82.8 ms |
| Tesla T4 | 228.8 ms | 137.2 ms |
| A100 80GB | 48.2 ms | not re-measured — those cards were down |

† The RTX 3090 is the one card these files partly caught up with on their own:
`blog_speedup.md` says 75.1 ms, and `blog_further_optimizations.md` carries a
2026-08-11 line giving 40.5 and 27.0 ms after the three later changes. The other
four cards are quoted only at their original values.

The current figures live in `benches/bench_{rtx3090,h100,p100,t4,a100}.json`,
each recording the date and commit it was taken at, and are what `README.md`,
`docs/performance.md` and `docs/internals/making-it-fast.md` publish. Read every
number below as "what was true when this was written", which is what a dated
record is for.

**The package was renamed to `vidmag` on 2026-08-18 and these three files were
deliberately left alone.** They say `evm`, `evm-cuda` or `evm-magnify`, and quote
command output that printed those names; read them as `vidmag`, and read a path
written `cuda/` or `src/evm/` as `src/vidmag/cuda/`. Rewriting a dated record to
use a name it predates would make it a worse record, which is the whole reason
these are kept unedited. `docs/dev/PLAN.md` decision D1 has the reasons for the
rename.
