# Kaggle run logs

Raw console logs from benchmark and test runs executed on Kaggle GPU instances,
produced by `scripts/cloud/kaggle/run_gpu_comparison.py`. Each is the notebook's own JSON
stream log: a list of `{stream_name, time, data}` records in execution order.

| File | Branch it ran | What the run did |
|---|---|---|
| `evm-cuda-gpu-comparison_2026-08-22_p100.log` | `library-restructure` | Processor versus FP32 versus FP16 on a Tesla P100. **The current one** — recorded in `benches/bench_p100.json` |
| `evm-cuda-gpu-comparison.log` | `feature/true-fp16-motion` (2026-08-09) | The same comparison, on the same class of card, before the three motion-path changes |
| `evm-cuda-baseline.log` | `feature/true-fp16-motion` | Baseline benchmark run |
| `evm-cuda-fp16-profile.log` | `feature/true-fp16-motion` | Per-stage FP16 profiling run |
| `evm-cuda-tests.log` | `feature/true-fp16-motion` | Full test suite on a Kaggle GPU |

The four 2026-08-09 logs predate the rename to `vidmag`; read `evm`, `evm_cuda`
and `evm-magnify` in them as the package now called `vidmag`. They are dated
records of what was run, so they are not edited to use a name they predate.

These are kept because they record measurements taken on hardware that is not
otherwise available for this project, and they cannot be reproduced without
re-running on the same instance types. The four older ones were recovered from
directories under `kaggle/` that held complete repository snapshots (178 MB) and
were removed once their unique contents had been preserved.

The source code from those snapshots, including work that existed nowhere else,
is preserved on the branches named `rescue/kaggle-*`. See `docs/dev/PLAN.md`,
Phase 0 step 0.6, for what each branch contains.
