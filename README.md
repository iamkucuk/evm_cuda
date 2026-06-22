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
- [x] CUDA port — validated bit-for-bit vs Python baseline (64/64 tests pass)
- [x] Speed optimization — **10x speedup** on the color pipeline

See [`docs/blog_speedup.md`](docs/blog_speedup.md) for the full optimization
write-up.

## Baseline

### Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_samples.py          # fetches MIT EVM samples into data/
```

### Usage

```bash
# Color (pulse) magnification
python scripts/run_evm.py data/face.mp4 output/face_color.mp4 \
    --mode color --fl 0.83 --fh 1.0 --chromatt 1.0 --alpha 50

# Motion magnification
python scripts/run_evm.py data/baby.mp4 output/baby_motion.mp4 \
    --mode motion --fl 0.4 --fh 3.0 --alpha 25 --levels 6
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

### Build (TRUBA)

```bash
source deploy/truba_env.sh
bash deploy/build.sh
```

### Profiling

```bash
python scripts/profile_cuda.py    # per-stage breakdown, run on a GPU node
```
