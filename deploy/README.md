# Deploying & validating the EVM CUDA port on TRUBA

The kernels can only be compiled and run on an NVIDIA GPU. This directory
holds everything you need to build `_evm_cuda.so` and run the full test suite
on TRUBA HPC (ARF-ACC, the GPU cluster behind `ssh truba`).

## Prerequisites (one-time, from your laptop)

1. **VPN up.** TRUBA requires OpenVPN for SSH access.
2. **Sync the project** to WEKA scratch (visible from compute nodes), carrying
   `data/` along so the sample videos are already on the cluster:
   ```bash
   rsync -avz --exclude='.git' --exclude='.venv' --exclude='output' \
       /Users/furkan/Documents/projects/evm_cuda/ \
       truba:/arf/scratch/fkucuk/projects/evm_cuda/
   ```
   `data/` is small (~10 MB: `face.mp4`, `baby.mp4` plus the two MIT
   reference outputs) so it's cheaper to push from the laptop than to
   re-download on the cluster. If it's missing for any reason, the login
   node has internet and you can repopulate with
   `python scripts/download_samples.py face baby --with-references`.

## Modules on cuda-ui

The cluster uses **Tcl Environment Modules** (not Lmod). The real module
names (verified Jun 2026) that `deploy/truba_env.sh` loads:

| Tool | Module | Version |
|---|---|---|
| CUDA | `lib/cuda/12.6` | nvcc 12.6.20 (covers sm_60..sm_90) |
| Python | `comp/python/miniconda3` | 3.12.2 (with venv + pip + `_ctypes`) |
| GCC | `comp/gcc/12.3.0` | 12.3.0 (matches CUDA 12.6's host-compiler requirement) |
| CMake | `comp/cmake/3.31.1` | 3.31.1 |

Other CUDA modules available: `lib/cuda/11.8`, `12.4`, `13.0`. Override any
via env vars, e.g. `EVM_CUDA_MODULE=lib/cuda/13.0 sbatch deploy/submit.slurm`.

Note: the `module` function is only defined in a **login shell**, so all
scripts in `deploy/` use `#!/bin/bash -l` (the SLURM shebang) or
`bash -lc '...'` when invoked manually.

## Build & validate

### Option A — interactive (recommended for first bring-up)

```bash
ssh truba
cd /arf/scratch/fkucuk/projects/evm_cuda
mkdir -p /arf/scratch/fkucuk/logs/evm_cuda

# Grab one GPU for an hour, build, and run tests in it.
srun --partition=palamut-cuda --gres=gpu:1 --time=01:00:00 --cpus-per-task=8 \
    --pty bash
# (inside the allocation:)
source deploy/truba_env.sh
bash deploy/build.sh
PYTHONPATH=$PWD/cuda pytest tests/ tests/cuda/ -v
```

### Option B — batch submission

```bash
ssh truba
cd /arf/scratch/fkucuk/projects/evm_cuda
sbatch deploy/submit.slurm
# Watch:
tail -f /arf/scratch/fkucuk/logs/evm_cuda/<jobid>.out
```

The job script:
1. Loads `CUDA/12.1.1` + Python + the project venv (`deploy/truba_env.sh`).
2. Builds the extension via CMake + nvcc (`deploy/build.sh`).
3. Runs `pytest tests/` (the Python baseline, the correctness oracle).
4. Runs `pytest tests/cuda/` (CUDA kernels vs the Python baseline).

Default queue: `palamut-cuda` (A100 / V100 / P100). For Hopper:
```bash
sbatch --partition=kolyoz-cuda --gres=gpu:h100:1 deploy/submit.slurm
```

## Target architectures

`cuda/CMakeLists.txt` builds for `sm_60 sm_70 sm_80 sm_89 sm_90` — covering
every GPU on TRUBA's ARF-ACC queues:

| Queue | GPU | sm |
|---|---|---|
| palamut-cuda | P100 | 60 |
| palamut-cuda | V100 | 70 |
| palamut-cuda | A100 | 80 |
| kolyoz-cuda | H100 | 90 |

The single `.so` therefore runs on any node SLURM assigns. Override with
`-DCMAKE_CUDA_ARCHITECTURES=80` (e.g.) if you want to speed up the build for
a known target.

## What "passing" means

- `tests/` — 29 Python baseline tests, including the 2 MIT-reference
  integration tests (`test_against_mit_reference.py`). These must pass
  regardless of GPU.
- `tests/cuda/` — 30 CUDA-vs-Python tests, gated on the extension building.
  Tolerances per AGENTS.md §2 (e.g. color_cvt `<1e-6`, IIR/Butter `<1e-5`,
  ideal `<1e-4`, end-to-end RMSE `<0.01`).
- The 4 end-to-end pipeline tests (`test_pipelines.py`) compare the CUDA
  pipeline's output to the Python baseline's on the **same input** — this is
  the validation contract from AGENTS.md §1 ("the CUDA port is validated
  against `evm/`, not against MIT directly").

## Common issues

- **`nvcc not found`** — you didn't `source deploy/truba_env.sh` first.
- **`cufft` linking error** — `CUDAToolkit` not found. Check
  `module load CUDA/12.1.1` succeeded; run `module avail CUDA` on cuda-ui.
- **`ImportError: _evm_cuda` after build** — the `.so` is at
  `cuda/evm_cuda/_evm_cuda.so`; make sure `PYTHONPATH` includes `cuda/`.
  The job script sets this; interactive runs need `export PYTHONPATH=$PWD/cuda`.
- **Pipeline tests skip** — `data/face.mp4` or `data/baby.mp4` missing. Run
  the download step from the login node (above).
- **Job hangs on SSH** — VPN is down. Reconnect OpenVPN.

## Pulling results back

```bash
# From your laptop:
rsync -avz \
    truba:/arf/scratch/fkucuk/projects/evm_cuda/output/ \
    /Users/furkan/Documents/projects/evm_cuda/output/
```
