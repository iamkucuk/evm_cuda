#!/bin/bash
#SBATCH -J evm_full_bench
#SBATCH -p kolyoz-cuda
#SBATCH -A romer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH -o /arf/scratch/fkucuk/logs/evm_full_bench-%j.out
#SBATCH -e /arf/scratch/fkucuk/logs/evm_full_bench-%j.err
set -euo pipefail
export TERM=xterm
PROJ=/arf/scratch/fkucuk/projects/evm_cuda
cd "$PROJ"
mkdir -p /arf/scratch/fkucuk/logs "$PROJ/output"

source /usr/share/Modules/init/bash
module load lib/cuda/12.6

# Build extension for this node's SM
bash scripts/truba_build.sh

source "$PROJ/.venv/bin/activate"
export PYTHONPATH=cuda
export PATH="$PROJ/.venv/bin:$PATH"

python - <<'PY'
import json
from pathlib import Path
from evm_cuda.benchmark import run
from evm_cuda import batched, _evm_cuda
import numpy as np

def rmse_u8(a, b):
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    u8 = np.abs(
        np.clip(np.rint(a * 255), 0, 255).astype(np.int16)
        - np.clip(np.rint(b * 255), 0, 255).astype(np.int16)
    )
    return float(np.sqrt((d * d).mean())), float(d.max()), int(u8.max())

gpu = ""
try:
    import subprocess
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        text=True,
    ).strip()
except Exception as e:
    gpu = str(e)

out = {
    "host": "truba",
    "gpu": gpu,
    "n_iter": 7,
    "results": {},
}

def pack(r):
    return {
        "compute_ms": r.compute_ms,
        "transfer_ms": r.transfer_ms,
        "total_ms": r.total_ms,
        "stages": {
            s.name: {"median": s.median_ms, "min": s.min_ms, "max": s.max_ms}
            for s in r.stages
        },
        "notes": r.notes,
        "gpu": r.gpu,
    }

configs = [
    ("motion_fp32", "motion", "fp32",
     dict(vid="data/baby.mp4", alpha=20.0, lambda_c=16.0, r1=0.4, r2=0.05, chrom_attenuation=0.1)),
    ("motion_fp16", "motion", "fp16",
     dict(vid="data/baby.mp4", alpha=20.0, lambda_c=16.0, r1=0.4, r2=0.05, chrom_attenuation=0.1)),
    ("color_face_fp32", "color", "fp32",
     dict(vid="data/face.mp4", alpha=50.0, level=4, fl=50/60, fh=60/60, chrom_attenuation=1.0, sampling_rate=30.0)),
    ("color_face_fp16", "color", "fp16",
     dict(vid="data/face.mp4", alpha=50.0, level=4, fl=50/60, fh=60/60, chrom_attenuation=1.0, sampling_rate=30.0)),
    ("color_baby_fp32", "color", "fp32",
     dict(vid="data/baby.mp4", alpha=50.0, level=4, fl=0.4, fh=3.0, chrom_attenuation=1.0)),
    ("color_baby_fp16", "color", "fp16",
     dict(vid="data/baby.mp4", alpha=50.0, level=4, fl=0.4, fh=3.0, chrom_attenuation=1.0)),
]

for key, pipe, prec, params in configs:
    print("===", key, flush=True)
    r = run(pipe, prec, params, n_iter=7)
    print(r, flush=True)
    out["results"][key] = pack(r)

# Accuracy
print("=== accuracy ===", flush=True)
a = batched.magnify_motion_lpyr_iir(
    "data/baby.mp4", "", alpha=20, lambda_c=16, r1=0.4, r2=0.05, chrom_attenuation=0.1)
b = batched.magnify_motion_lpyr_iir_fp16(
    "data/baby.mp4", "", alpha=20, lambda_c=16, r1=0.4, r2=0.05, chrom_attenuation=0.1)
rmse, mx, u8 = rmse_u8(a, b)
out["results"]["motion_fp16_vs_fp32"] = {"rmse": rmse, "max": mx, "u8_max": u8}
print("motion", rmse, mx, u8, flush=True)

a = batched.magnify_color_gdown_ideal(
    "data/face.mp4", "", alpha=50, level=4, fl=50/60, fh=60/60, chrom_attenuation=1.0, sampling_rate=30.0)
b = batched.magnify_color_gdown_ideal_fp16(
    "data/face.mp4", "", alpha=50, level=4, fl=50/60, fh=60/60, chrom_attenuation=1.0, sampling_rate=30.0)
rmse, mx, u8 = rmse_u8(a, b)
out["results"]["color_face_fp16_vs_fp32"] = {"rmse": rmse, "max": mx, "u8_max": u8}
print("color face", rmse, mx, u8, flush=True)

path = Path("output/bench_truba_h100.json")
path.write_text(json.dumps(out, indent=2))
print("WROTE", path, flush=True)
for k, v in out["results"].items():
    if isinstance(v, dict) and "compute_ms" in v:
        print(f"SUMMARY {k}: compute={v['compute_ms']:.2f} total={v['total_ms']:.2f} notes={v.get('notes','')}", flush=True)
PY
