# Kaggle GPU Benchmark

Run the EVM CUDA profiler on Kaggle's free GPU (T4 or P100) via CLI.

The active kernel is `run_gpu_comparison.py` — it clones the repo, builds
the CUDA extension for the detected GPU arch, runs the GPU-only profiler
(FP32 + FP16 for both pipelines with per-stage breakdown), and renders
all 4 output videos (skipping any that OOM on 16 GB GPUs).

CPU reference numbers come from the TRUBA A100 run (hardcoded in the
script) since the CPU baseline doesn't depend on the GPU.

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
