// evm_common.cuh — shared constants and device helpers for the EVM CUDA port.
//
// Every value here is copied verbatim from the Python baseline (evm/) so the
// CUDA kernels reproduce the reference numerics bit-for-bit. Do NOT tweak
// these "for performance" — the tolerances in tests/cuda/ assume these
// exact constants. File:line references point into evm/.

#pragma once

#include <cuda_runtime.h>

namespace evm {

// ---------------------------------------------------------------------------
// Algorithm-wide constants (evm/magnify.py:50,53)
// ---------------------------------------------------------------------------

// All four reference amplification functions drop the last 10 frames of the
// input (MATLAB endIndex = len - 10).
constexpr int kDropLast = 10;

// Figure-6 per-level exaggeration factor. Hardcoded (=2) in every MATLAB
// amplification function; we expose it as a default so tests can override.
constexpr float kExaggerationFactor = 2.0f;

// ---------------------------------------------------------------------------
// binom5 pyramid filter, dual form (evm/pyramids.py:33-35)
// ---------------------------------------------------------------------------
//
// matlabPyrTools stores binom5 L2-normalized (scaled by sqrt(2)). buildLpyr
// uses it verbatim; blurDn re-normalizes to sum=1. We hardcode BOTH forms
// because the float division that derives one from the other is not exact.

// BINOM5_SUM1: [1,4,6,4,1]/16, sum == 1.0. Used by blur_dn / blur_dn_clr.
__constant__ const float kBinom5Sum1[5] = {
    0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f};

// BINOM5: sqrt(2)*[1,4,6,4,1]/16, L2 norm == 1.0, sum == sqrt(2).
// Used by build_lpyr / recon_lpyr.
__constant__ const float kBinom5[5] = {
    0.08838834764831843f,
    0.35355339059327373f,
    0.5303300858899106f,
    0.35355339059327373f,
    0.08838834764831843f};

// ---------------------------------------------------------------------------
// NTSC YIQ color matrices (evm/video.py:117-137)
// ---------------------------------------------------------------------------
//
// Match MATLAB rgb2ntsc / ntsc2rgb exactly, INCLUDING the ~1e-6 and ~1e-7
// entries — they are part of the documented inverse and contribute at the
// 1e-6 tolerance level we assert in tests.

// Row order (Y, I, Q): yiq = rgb @ kRgbToYiq.T
__constant__ const float kRgbToYiq[3][3] = {
    { 0.299f,     0.587f,      0.114f    },
    {-0.168736f, -0.331264f,   0.5f      },
    { 0.5f,      -0.418688f,  -0.081312f }};

// rgb = yiq @ kYiqToRgb.T  (inverse of kRgbToYiq, full precision)
__constant__ const float kYiqToRgb[3][3] = {
    {1.0f, -1.21889419e-06f,  1.40199959f   },
    {1.0f, -3.44135678e-01f, -7.14136156e-01f},
    {1.0f,  1.77200007f,      4.06298063e-07f}};

// ---------------------------------------------------------------------------
// reflect1 boundary handling (numpy mode='reflect' == MATLAB reflect1)
// ---------------------------------------------------------------------------
//
// Half-sample symmetric reflection about the edge pixel WITHOUT duplicating
// it. For source length n, padded coordinate j (0-based, with the array
// starting at padded index 'pad') maps to source coordinate i = j - pad; if
// i is outside [0, n), reflect. Each iteration pulls i closer to the centre,
// so for the 5-tap binom5 (pad=2) and n>=2 the loop terminates in at most a
// few iterations.
//
// Caller passes a source-space coordinate (already shifted); we return the
// reflected in-range index. Used by corr_dn / up_conv to gather edge samples.

__device__ __forceinline__ int reflect1(int i, int n) {
    // Mirror without duplicating the edge sample. The period is 2*(n-1).
    if (n == 1) return 0;
    const int period = 2 * (n - 1);
    // Fold i into [0, period) using the same modulo convention as numpy
    // (result takes the sign of the divisor).
    i = i % period;
    if (i < 0) i += period;
    if (i >= n) i = period - i;
    return i;
}

// ---------------------------------------------------------------------------
// Launch-size helper
// ---------------------------------------------------------------------------

__host__ __device__ __forceinline__ int div_up(int a, int b) {
    return (a + b - 1) / b;
}

}  // namespace evm
