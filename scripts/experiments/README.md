# One-off measurements

These were written to answer a specific question once, and their answers are
recorded in `docs/internals/making-it-fast.md`. They are kept so those claims
can be re-checked, not because anything runs them.

| Script | The question it answered |
|---|---|
| `bound_microbench.py` | Is a stage limited by memory bandwidth or by arithmetic, on a machine where the profiler cannot read hardware counters? |
| `measure_fp16_experiments.py` | Which half-precision storage choices actually pay, stage by stage? |
| `prove_fp16_cost.py` | Does half precision cost anything on the fused downsample and the full pyramid build? |

They need an NVIDIA GPU with the extension built. Nothing else in the
repository imports them, and no automated check runs them; if one stops working
because the code beneath it moved on, that is a signal about this directory
rather than a broken build.

The tools that are still used live one level up in `scripts/`: `run_evm.py`,
`download_samples.py`, `profile_full_comparison.py` and `render_cuda_videos.py`.
