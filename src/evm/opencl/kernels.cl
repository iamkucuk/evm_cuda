// OpenCL kernels for the portable backend.
//
// One source file, compiled at run time by whatever OpenCL driver is present.
// That is the whole reason this backend exists: the same text runs on Apple,
// AMD, Intel and NVIDIA hardware, so supporting a new graphics card needs a
// driver rather than new code here.
//
// Everything is float32. The NumPy reference computes in float64, so a small
// difference is expected and is what the conformance tolerances allow for.
//
// Border handling is `reflect1`, matching the reference: index -1 reads
// element 1, and index n reads element n-2. Getting this wrong shifts every
// pyramid level by a fraction of a pixel, which is why it is written once in
// reflect_index() and used everywhere.

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

inline int reflect_index(int i, int n) {
    // Mirror without repeating the edge sample, for arbitrary overshoot.
    if (n == 1) return 0;
    int period = 2 * n - 2;
    int m = i % period;
    if (m < 0) m += period;
    return (m < n) ? m : (period - m);
}

// ---------------------------------------------------------------------------
// Colour
// ---------------------------------------------------------------------------

// Blue-green-red bytes to NTSC, which separates brightness from colour.
//
// The nine FWD_* constants are not written here. They are substituted into
// this source before compilation from the matrix in evm.io.video, so the two
// cannot drift apart: that matrix is what the whole project's agreement with
// the reference implementation rests on, and a second hand-typed copy of it
// would be a second thing to get wrong.
__kernel void bgr_u8_to_ntsc(__global const uchar* src,
                             __global float* dst,
                             const int count) {
    int i = get_global_id(0);
    if (i >= count) return;
    float b = src[3 * i + 0] / 255.0f;
    float g = src[3 * i + 1] / 255.0f;
    float r = src[3 * i + 2] / 255.0f;
    dst[3 * i + 0] = FWD00 * r + FWD01 * g + FWD02 * b;
    dst[3 * i + 1] = FWD10 * r + FWD11 * g + FWD12 * b;
    dst[3 * i + 2] = FWD20 * r + FWD21 * g + FWD22 * b;
}

// Add the amplified signal, convert back, clamp and round to bytes.
__kernel void add_and_quantize(__global const float* base,
                               __global const float* delta,
                               __global uchar* dst,
                               const int count) {
    int i = get_global_id(0);
    if (i >= count) return;
    // INV_* are substituted from the inverse matrix in evm.io.video, for the
    // same reason as the forward direction above.
    float c0 = base[3 * i + 0] + delta[3 * i + 0];
    float c1 = base[3 * i + 1] + delta[3 * i + 1];
    float c2 = base[3 * i + 2] + delta[3 * i + 2];

    float r = INV00 * c0 + INV01 * c1 + INV02 * c2;
    float g = INV10 * c0 + INV11 * c1 + INV12 * c2;
    float b = INV20 * c0 + INV21 * c1 + INV22 * c2;

    r = clamp(r, 0.0f, 1.0f);
    g = clamp(g, 0.0f, 1.0f);
    b = clamp(b, 0.0f, 1.0f);

    // Stored blue-green-red, matching how the frames were read.
    dst[3 * i + 0] = (uchar)(round(b * 255.0f));
    dst[3 * i + 1] = (uchar)(round(g * 255.0f));
    dst[3 * i + 2] = (uchar)(round(r * 255.0f));
}

// ---------------------------------------------------------------------------
// Spatial: separable filter along one axis, then keep every other sample
// ---------------------------------------------------------------------------

// Every spatial kernel below takes an explicit frame index. Flattening the
// frames into one tall image would be simpler, but the reflected border would
// then read from the neighbouring frame at the top and bottom edges, quietly
// mixing frames together. Each work item is (channel, column, frame*row).

// Filter down the rows. Input (T, H, W, C), output (T, ceil(H/2), W, C).
__kernel void corr_dn_rows(__global const float* src,
                           __global float* dst,
                           __global const float* filt,
                           const int T, const int H, const int W, const int C,
                           const int flen, const int out_h) {
    int c = get_global_id(0);
    int x = get_global_id(1);
    int idx = get_global_id(2);
    int t = idx / out_h;
    int oy = idx - t * out_h;
    if (c >= C || x >= W || t >= T) return;

    int pad = flen / 2;
    int y = oy * 2;
    size_t in_frame = (size_t)t * H * W * C;
    size_t out_frame = (size_t)t * out_h * W * C;
    float acc = 0.0f;
    for (int k = 0; k < flen; ++k) {
        // The reference reverses the kernel and takes a valid convolution,
        // which is the same as this correlation with the original taps.
        int sy = reflect_index(y + k - pad, H);
        acc += filt[k] * src[in_frame + (size_t)(sy * W + x) * C + c];
    }
    dst[out_frame + (size_t)(oy * W + x) * C + c] = acc;
}

// Filter across the columns. Input (T, H, W, C), output (T, H, ceil(W/2), C).
__kernel void corr_dn_cols(__global const float* src,
                           __global float* dst,
                           __global const float* filt,
                           const int T, const int H, const int W, const int C,
                           const int flen, const int out_w) {
    int c = get_global_id(0);
    int ox = get_global_id(1);
    int idx = get_global_id(2);
    int t = idx / H;
    int y = idx - t * H;
    if (c >= C || ox >= out_w || t >= T) return;

    int pad = flen / 2;
    int x = ox * 2;
    size_t in_frame = (size_t)t * H * W * C;
    size_t out_frame = (size_t)t * H * out_w * C;
    float acc = 0.0f;
    for (int k = 0; k < flen; ++k) {
        int sx = reflect_index(x + k - pad, W);
        acc += filt[k] * src[in_frame + (size_t)(y * W + sx) * C + c];
    }
    dst[out_frame + (size_t)(y * out_w + ox) * C + c] = acc;
}

// Insert a zero between every pair of rows, filter, crop. The transpose of
// corr_dn_rows.
__kernel void up_conv_rows(__global const float* src,
                           __global float* dst,
                           __global const float* filt,
                           const int T, const int H, const int W, const int C,
                           const int flen, const int out_h) {
    int c = get_global_id(0);
    int x = get_global_id(1);
    int idx = get_global_id(2);
    int t = idx / out_h;
    int oy = idx - t * out_h;
    if (c >= C || x >= W || t >= T) return;

    int pad = flen / 2;
    int up_h = H * 2;
    size_t in_frame = (size_t)t * H * W * C;
    size_t out_frame = (size_t)t * out_h * W * C;
    float acc = 0.0f;
    for (int k = 0; k < flen; ++k) {
        int sy = reflect_index(oy + k - pad, up_h);
        // Only even rows of the upsampled image carry a sample.
        if ((sy & 1) == 0) {
            int row = sy >> 1;
            if (row < H) acc += filt[k] * src[in_frame + (size_t)(row * W + x) * C + c];
        }
    }
    dst[out_frame + (size_t)(oy * W + x) * C + c] = acc;
}

__kernel void up_conv_cols(__global const float* src,
                           __global float* dst,
                           __global const float* filt,
                           const int T, const int H, const int W, const int C,
                           const int flen, const int out_w) {
    int c = get_global_id(0);
    int ox = get_global_id(1);
    int idx = get_global_id(2);
    int t = idx / H;
    int y = idx - t * H;
    if (c >= C || ox >= out_w || t >= T) return;

    int pad = flen / 2;
    int up_w = W * 2;
    size_t in_frame = (size_t)t * H * W * C;
    size_t out_frame = (size_t)t * H * out_w * C;
    float acc = 0.0f;
    for (int k = 0; k < flen; ++k) {
        int sx = reflect_index(ox + k - pad, up_w);
        if ((sx & 1) == 0) {
            int col = sx >> 1;
            if (col < W) acc += filt[k] * src[in_frame + (size_t)(y * W + col) * C + c];
        }
    }
    dst[out_frame + (size_t)(y * out_w + ox) * C + c] = acc;
}

// Subtract one image from another, used to form each pyramid band.
__kernel void subtract(__global const float* a, __global const float* b,
                       __global float* dst, const int count) {
    int i = get_global_id(0);
    if (i >= count) return;
    dst[i] = a[i] - b[i];
}

__kernel void add_into(__global float* acc, __global const float* b,
                       const int count) {
    int i = get_global_id(0);
    if (i >= count) return;
    acc[i] += b[i];
}

// ---------------------------------------------------------------------------
// Bilinear resize, matching what the reference uses to scale the amplified
// colour signal back to full resolution.
// ---------------------------------------------------------------------------

__kernel void resize_bilinear(__global const float* src,
                              __global float* dst,
                              const int T,
                              const int in_h, const int in_w,
                              const int out_h, const int out_w,
                              const int C) {
    int c = get_global_id(0);
    int ox = get_global_id(1);
    int idx = get_global_id(2);
    int t = idx / out_h;
    int oy = idx - t * out_h;
    if (c >= C || ox >= out_w || t >= T) return;

    size_t in_frame = (size_t)t * in_h * in_w * C;
    size_t out_frame = (size_t)t * out_h * out_w * C;

    // Half-pixel centres: the same convention OpenCV's INTER_LINEAR uses.
    float fy = ((float)oy + 0.5f) * ((float)in_h / (float)out_h) - 0.5f;
    float fx = ((float)ox + 0.5f) * ((float)in_w / (float)out_w) - 0.5f;

    int y0 = (int)floor(fy);
    int x0 = (int)floor(fx);
    float wy = fy - (float)y0;
    float wx = fx - (float)x0;

    int y1 = min(max(y0 + 1, 0), in_h - 1);
    int x1 = min(max(x0 + 1, 0), in_w - 1);
    y0 = min(max(y0, 0), in_h - 1);
    x0 = min(max(x0, 0), in_w - 1);

    float v00 = src[in_frame + (size_t)(y0 * in_w + x0) * C + c];
    float v01 = src[in_frame + (size_t)(y0 * in_w + x1) * C + c];
    float v10 = src[in_frame + (size_t)(y1 * in_w + x0) * C + c];
    float v11 = src[in_frame + (size_t)(y1 * in_w + x1) * C + c];

    float top = v00 + wx * (v01 - v00);
    float bot = v10 + wx * (v11 - v10);
    dst[out_frame + (size_t)(oy * out_w + ox) * C + c] = top + wy * (bot - top);
}

// ---------------------------------------------------------------------------
// Temporal filters. Each work item owns one pixel's whole time series, so the
// recursions below stay sequential in time while every pixel runs in parallel.
// Layout is (time, pixel): element t of pixel i is at t * n_pixels + i.
// ---------------------------------------------------------------------------

// Difference of two exponential moving averages.
__kernel void iir_bandpass(__global const float* src,
                           __global float* dst,
                           const int T, const int N,
                           const float r1, const float r2) {
    int i = get_global_id(0);
    if (i >= N) return;

    float low1 = src[i];
    float low2 = src[i];
    dst[i] = 0.0f;                      // first sample: the two agree exactly
    for (int t = 1; t < T; ++t) {
        float x = src[t * N + i];
        low1 = (1.0f - r1) * low1 + r1 * x;
        low2 = (1.0f - r2) * low2 + r2 * x;
        dst[t * N + i] = low1 - low2;
    }
}

// Two first-order sections, their outputs subtracted: a Butterworth bandpass.
__kernel void butter_bandpass(__global const float* src,
                              __global float* dst,
                              const int T, const int N,
                              const float b0h, const float b1h, const float a1h,
                              const float b0l, const float b1l, const float a1l) {
    int i = get_global_id(0);
    if (i >= N) return;

    float x_prev = 0.0f;
    float yh_prev = 0.0f;
    float yl_prev = 0.0f;
    for (int t = 0; t < T; ++t) {
        float x = src[t * N + i];
        float yh = b0h * x + b1h * x_prev - a1h * yh_prev;
        float yl = b0l * x + b1l * x_prev - a1l * yl_prev;
        dst[t * N + i] = yh - yl;
        x_prev = x;
        yh_prev = yh;
        yl_prev = yl;
    }
}

// Keep only a band of frequencies, as a matrix multiply.
//
// Selecting frequencies is a linear map, so it can be written as one T-by-T
// real matrix built on the host from the kept bins. That is exactly equal to
// transforming, zeroing the unwanted bins and transforming back, and it needs
// no Fourier transform library on the device — which is what lets this backend
// run anywhere without pulling in a per-vendor maths library.
__kernel void band_project(__global const float* src,
                           __global float* dst,
                           __global const float* matrix,
                           const int T, const int N) {
    int i = get_global_id(0);
    int t = get_global_id(1);
    if (i >= N || t >= T) return;

    float acc = 0.0f;
    for (int s = 0; s < T; ++s) {
        acc += matrix[t * T + s] * src[s * N + i];
    }
    dst[t * N + i] = acc;
}

// ---------------------------------------------------------------------------
// Amplification
// ---------------------------------------------------------------------------

__kernel void apply_gain(__global float* data, const int count,
                         const float gy, const float gi, const float gq) {
    int i = get_global_id(0);
    if (i >= count) return;
    data[3 * i + 0] *= gy;
    data[3 * i + 1] *= gi;
    data[3 * i + 2] *= gq;
}

// One step of the two running averages, and their difference.
//
// This exists so that magnifying a live feed does not have to copy every
// pyramid band back to host memory between frames. Doing the update here keeps
// the state on the device, where the rest of the work already is.
//
//   fast = fast * (1 - r1) + current * r1
//   slow = slow * (1 - r2) + current * r2
//   out  = fast - slow
//
// `fast` and `slow` are updated in place; they are the state carried from one
// frame to the next.
__kernel void iir_step(__global float* fast,
                       __global float* slow,
                       __global const float* current,
                       __global float* out,
                       const int count,
                       const float r1, const float r2) {
    int i = get_global_id(0);
    if (i >= count) return;
    float x = current[i];
    float f = fast[i] * (1.0f - r1) + x * r1;
    float s = slow[i] * (1.0f - r2) + x * r2;
    fast[i] = f;
    slow[i] = s;
    out[i] = f - s;
}
