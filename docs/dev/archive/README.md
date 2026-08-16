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
