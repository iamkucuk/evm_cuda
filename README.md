# evm_cuda

Eulerian Video Magnification (EVM) — CUDA-accelerated.

Reference: Wu, Rubinstein, Freeman, Durand, Guttag.
**Eulerian Video Magnification for Revealing Subtle Changes in the World.**
SIGGRAPH 2012. <http://people.csail.mit.edu/mrub/vidmag/>

## Status

- [x] Python baseline (correctness oracle for the CUDA port)
  - [x] Color magnification (pulse / heart-rate)
  - [x] Motion magnification (collapsible Laplacian pyramid)
  - [x] Temporal filters: ideal (FFT), Butterworth, causal IIR
- [x] CUDA port — validated vs Python baseline (125/125 tests pass)
- [x] Speed optimization — **144x** color, **222x** motion (FP32, A100, compute-only)
- [x] FP16 storage — **269x** motion (FP16, A100), fits 16 GB GPUs

See [`docs/blog_speedup.md`](docs/blog_speedup.md) for the full optimization
write-up and [`HANDOFF.md`](HANDOFF.md) for current state.

## Quick start with make

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make download          # fetch MIT sample videos + references

# Build (needs nvcc)
make build             # compile _evm_cuda.so

# Test
make test              # all 125 tests (77 baseline + 48 CUDA)
make test-baseline     # Python oracle only (no GPU, ~40s)

# Run
make run-color         # pulse magnification on face.mp4
make run-motion        # motion magnification on baby.mp4

# Profile
make profile           # CPU vs FP32 vs FP16, both pipelines + videos
make help              # list all targets
```

## Manual usage

```bash
# Color (pulse) magnification
python scripts/run_evm.py data/face.mp4 output/face_color.mp4 \
    --mode color --alpha 50 --level 4 --fl 0.8333 --fh 1.0 --chromatt 1

# Motion magnification (IIR)
python scripts/run_evm.py data/baby.mp4 output/baby_motion.mp4 \
    --mode iir --alpha 10 --lambda-c 16 --r1 0.4 --r2 0.05 --chromatt 0.1
```

Run `python scripts/run_evm.py --help` for the full parameter list.

### Tests

```bash
python -m pytest tests/ -q          # baseline only (no GPU needed)
python -m pytest tests/ tests/cuda/ -q   # + CUDA kernels (needs GPU)
```

## CUDA port

Raw CUDA kernels (no PyTorch/CuPy) with a thin pybind11 binding. Every kernel
is validated against the Python baseline within tight tolerances — the CUDA
output is bit-for-bit equivalent to the MIT MATLAB reference.

See [`cuda/DESIGN.md`](cuda/DESIGN.md) for the kernel architecture and
[`docs/blog_speedup.md`](docs/blog_speedup.md) for the optimization story.

### Build

```bash
make build     # cmake + nvcc, produces cuda/evm_cuda/_evm_cuda.so
```

### Profiling

```bash
make profile           # CPU vs FP32 vs FP16, both pipelines + output videos
make profile-color     # color FP32 stages only
make profile-motion    # motion FP32 stages only
```
