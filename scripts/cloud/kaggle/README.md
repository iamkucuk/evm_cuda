# Kaggle GPU Benchmark

Run the EVM CUDA profiler on Kaggle's free GPU (T4 or P100) via CLI.

The active kernel is `run_gpu_comparison.py` — it clones `main`, installs the
repo with `pip install .` (one step; that also compiles the CUDA extension for
the detected GPU arch), runs color+motion × FP32+FP16 via `vidmag.cuda.benchmark`
(1 warmup + median of 7), and renders output videos (skips OOM configs on
16 GB).

CPU reference numbers are measured on the Kaggle machine itself, so its speedup
ratios describe that machine and are not comparable with the ratios in the
project README, which divide by the NumPy baseline of the RTX 3090 host
(colour 5,585 ms / motion 31,981 ms, measured 2026-08-18). Compare the
millisecond figures across machines, not the ratios.
## Setup (one-time)

1. Create a Kaggle account at [kaggle.com](https://kaggle.com)
2. Go to Account -> Settings -> Create New API Token
3. Save the token to `~/.kaggle/kaggle.json`:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```
4. Install the Kaggle CLI:
   ```bash
   pip install kaggle
   ```

## Push and run

```bash
cd kaggle/
kaggle kernels push -p .
```

This uploads `run_gpu_comparison.py`, starts a GPU kernel, builds the
CUDA extension, and runs all four profiler configurations.

## Check status

```bash
kaggle kernels status furkankucuk/evm-cuda-gpu-comparison
```

## Pull results

```bash
kaggle kernels output furkankucuk/evm-cuda-gpu-comparison -p ./results_gpu
```

This downloads:
- `gpu_comparison_results.json` (per-stage timing for all 4 configs)
- `output/face_fp32.mp4`, `output/face_fp16.mp4`
- `output/baby_fp32.mp4`, `output/baby_fp16.mp4`
- Log files with the full profiler output

## Limits

- 30 hours of GPU time per week (resets weekly)
- Sessions up to 12 hours
- T4 (16 GB) or P100 (16 GB) GPU
- Internet access enabled (needed for git clone + pip install)
