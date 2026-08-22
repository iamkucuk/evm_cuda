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

**The package was renamed to `vidmag` on 2026-08-18 and these three files were
deliberately left alone.** They say `evm`, `evm-cuda` or `evm-magnify`, and quote
command output that printed those names; read them as `vidmag`, and read a path
written `cuda/` or `src/evm/` as `src/vidmag/cuda/`. Rewriting a dated record to
use a name it predates would make it a worse record, which is the whole reason
these are kept unedited. `docs/dev/PLAN.md` decision D1 has the reasons for the
rename.
