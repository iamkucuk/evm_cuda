# Kaggle run logs

Raw console logs from benchmark and test runs executed on Kaggle GPU instances,
produced by `kaggle/run_gpu_comparison.py`. Each is the notebook's own JSON
stream log: a list of `{stream_name, time, data}` records in execution order.

| File | What the run did |
|---|---|
| `evm-cuda-baseline.log` | Baseline benchmark run |
| `evm-cuda-gpu-comparison.log` | CPU versus FP32 versus FP16 comparison on a Tesla P100 |
| `evm-cuda-fp16-profile.log` | Per-stage FP16 profiling run |
| `evm-cuda-tests.log` | Full test suite on a Kaggle GPU |

These are kept because they record measurements taken on hardware that is not
otherwise available for this project, and they cannot be reproduced without
re-running on the same instance types. They were recovered from directories
under `kaggle/` that held complete repository snapshots (178 MB) and were
removed once their unique contents had been preserved.

The source code from those snapshots, including work that existed nowhere else,
is preserved on the branches named `rescue/kaggle-*`. See `docs/dev/PLAN.md`,
Phase 0 step 0.6, for what each branch contains.
