# Deploying & validating the EVM CUDA port on TRUBA

The kernels can only be compiled and run on an NVIDIA GPU. This directory
holds everything you need to build `_evm_cuda.so` and run the full test suite
on TRUBA HPC (ARF-ACC, the GPU cluster behind `ssh truba`).

Most commands have Makefile equivalents (`make build-truba`, `make slurm`).
The files here exist because TRUBA needs module loading and SLURM
batch scripts that can't be expressed in a portable Makefile.

## Files

| File | Purpose |
|---|---|
| `truba_env.sh` | Loads gcc/12.3.0 + miniconda3 + cuda/12.6 + cmake modules, creates venv |
| `build.sh` | CMake + nvcc build wrapper (sources truba_env.sh, supports `CLEAN=1`) |
| `submit_profile.slurm` | The canonical SLURM job: build + test + full profiler in one job |

## Prerequisites (one-time, from your laptop)

1. **VPN up.** TRUBA requires OpenVPN for SSH access.
2. **Sync the project** to WEKA scratch (visible from compute nodes):
   ```bash
   rsync -avz --exclude='.git' --exclude='.venv' --exclude='output' \
       /Users/furkan/Documents/projects/evm_cuda/ \
       truba:/arf/scratch/fkucuk/projects/evm_cuda/
   ```

## Build & validate

### Option A — interactive (recommended for first bring-up)

```bash
ssh truba
cd /arf/scratch/fkucuk/projects/evm_cuda

# Grab one GPU for an hour
srun --partition=palamut-cuda --gres=gpu:1 --time=01:00:00 --cpus-per-task=8 \
    --pty bash

# (inside the allocation:)
make build-truba     # sources truba_env.sh + builds the extension
make test            # all 125 tests (77 baseline + 48 CUDA)
```

### Option B — batch submission

```bash
ssh truba
cd /arf/scratch/fkucuk/projects/evm_cuda
make slurm           # sbatch deploy/submit_profile.slurm
# Watch:
tail -f /arf/scratch/fkucuk/logs/evm_cuda/<jobid>.out
```

The job: sources env, builds, runs all tests, then runs the full profiler
(CPU vs FP32 vs FP16, both pipelines, renders output videos).

Default queue: `palamut-cuda` (A100 / V100 / P100). For Hopper:
```bash
sbatch --partition=kolyoz-cuda --gres=gpu:h100:1 deploy/submit_profile.slurm
```

## Modules on cuda-ui

| Tool | Module | Version |
|---|---|---|
| CUDA | `lib/cuda/12.6` | nvcc 12.6.20 (covers sm_60..sm_90) |
| Python | `comp/python/miniconda3` | 3.12.2 |
| GCC | `comp/gcc/12.3.0` | 12.3.0 (matches CUDA 12.6's host-compiler requirement) |
| CMake | `comp/cmake/3.31.1` | 3.31.1 |

Override via env vars, e.g. `EVM_CUDA_MODULE=lib/cuda/13.0 make slurm`.

Note: the `module` function is only defined in a **login shell**, so all
scripts use `#!/bin/bash -l`.

## Target architectures

`cuda/CMakeLists.txt` builds for `sm_60 sm_70 sm_80 sm_89 sm_90`:

| Queue | GPU | sm |
|---|---|---|
| palamut-cuda | P100 | 60 |
| palamut-cuda | V100 | 70 |
| palamut-cuda | A100 | 80 |
| kolyoz-cuda | H100 | 90 |

The single `.so` runs on any node SLURM assigns.

## What "passing" means

- `tests/` — 77 Python baseline tests, including MIT-reference integration
  tests. These must pass regardless of GPU.
- `tests/cuda/` — 48 CUDA-vs-Python tests (per-kernel tolerances from 1e-6
  to 1e-4, end-to-end RMSE < 0.01).
- Total: **125 tests**.

## Pulling results back

```bash
rsync -avz \
    truba:/arf/scratch/fkucuk/projects/evm_cuda/output/ \
    /Users/furkan/Documents/projects/evm_cuda/output/
```
