# Kaggle Benchmark

Run the EVM CUDA benchmark on Kaggle's free GPU (T4 or P100) via CLI.

## Setup (one-time)

1. Create a Kaggle account at [kaggle.com](https://kaggle.com)
2. Go to Account → Settings → Create New API Token
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

This uploads `run_benchmark.py` to Kaggle, starts a GPU kernel, builds the
CUDA extension from source, and runs both profilers.

## Check status

```bash
kaggle kernels status fkucuk/evm-cuda-benchmark
```

## Pull results

```bash
kaggle kernels output fkucuk/evm-cuda-benchmark -p ./results
```

This downloads:
- `benchmark_results.json` (timing summary)
- `output/face_color.mp4` (rendered pulse magnification)
- `output/baby_motion.mp4` (rendered motion magnification)
- Log files with the full profiler output

## Limits

- 30 hours of GPU time per week (resets weekly)
- Sessions up to 12 hours
- T4 (16 GB) or P100 (16 GB) GPU
- Internet access enabled (needed for git clone + pip install)

## Running locally

The same script works on any machine with nvcc + a GPU:

```bash
python kaggle/run_benchmark.py
```

It clones the repo, builds the extension for the detected GPU architecture,
and runs the profilers.
