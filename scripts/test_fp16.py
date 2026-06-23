"""Compare FP16 vs FP32 motion pipeline output and measure memory."""
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'cuda'))
import numpy as np

# Run both pipelines
from evm_cuda.batched import magnify_motion_lpyr_iir, magnify_motion_lpyr_iir_fp16

print("Running FP32 pipeline...")
out32 = magnify_motion_lpyr_iir(
    'data/baby.mp4', '/tmp/baby_fp32.mp4',
    alpha=10, lambda_c=16, r1=0.4, r2=0.05, chrom_attenuation=0.1)

print("Running FP16 pipeline...")
out16 = magnify_motion_lpyr_iir_fp16(
    'data/baby.mp4', '/tmp/baby_fp16.mp4',
    alpha=10, lambda_c=16, r1=0.4, r2=0.05, chrom_attenuation=0.1)

rmse = np.sqrt(np.mean((out32 - out16) ** 2))
max_err = np.max(np.abs(out32 - out16))
mae = np.mean(np.abs(out32 - out16))
print(f"\nFP32 vs FP16 comparison:")
print(f"  RMSE:      {rmse:.6f}")
print(f"  Max error: {max_err:.6f}")
print(f"  MAE:       {mae:.6f}")
print(f"  Tolerance: <0.01 RMSE")
print(f"  PASS: {rmse < 0.01}")
