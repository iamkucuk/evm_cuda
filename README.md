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
- [ ] CUDA port

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
python -m pytest tests/ -q
```
