// Metal compute kernels for the Apple backend.
//
// The same operations as src/evm/opencl/kernels.cl, in Metal Shading Language.
// They are kept as separate files rather than shared through macros because
// the two are different languages: Metal passes buffers as `device` pointers
// with explicit binding indices and takes the thread index as an attributed
// parameter, where OpenCL uses address-space qualifiers and a function call.
// A macro layer thin enough to hide that would be harder to read than either
// file, and a bug in it would appear as a wrong picture rather than a
// compile error.
//
// Why this exists when OpenCL already runs on Apple hardware: Apple has
// deprecated OpenCL. Metal is what Apple supports, and what will still be here
// when OpenCL is not.
//
// Everything is float32. The NumPy reference computes in float64, so a small
// difference is expected and is what the conformance tolerances allow.
//
// Border handling is `reflect1`, matching the reference: index -1 reads
// element 1, and index n reads element n-2.

#include <metal_stdlib>
using namespace metal;

inline int reflect_index(int i, int n) {
    if (n == 1) return 0;
    int period = 2 * n - 2;
    int m = i % period;
    if (m < 0) m += period;
    return (m < n) ? m : (period - m);
}

// Sizes shared by the spatial kernels, passed as one struct so a kernel needs
// one binding for all of them rather than six.
struct Shape {
    int T;
    int H;
    int W;
    int C;
    int flen;
    int out_h;
    int out_w;
};

// ---------------------------------------------------------------------------
// Colour
// ---------------------------------------------------------------------------

struct ColourMatrix {
    float m[9];
};

kernel void bgr_u8_to_ntsc(device const uchar* src [[buffer(0)]],
                           device float* dst [[buffer(1)]],
                           constant ColourMatrix& fwd [[buffer(2)]],
                           constant int& count [[buffer(3)]],
                           uint i [[thread_position_in_grid]]) {
    if ((int)i >= count) return;
    float b = src[3 * i + 0] / 255.0f;
    float g = src[3 * i + 1] / 255.0f;
    float r = src[3 * i + 2] / 255.0f;
    dst[3 * i + 0] = fwd.m[0] * r + fwd.m[1] * g + fwd.m[2] * b;
    dst[3 * i + 1] = fwd.m[3] * r + fwd.m[4] * g + fwd.m[5] * b;
    dst[3 * i + 2] = fwd.m[6] * r + fwd.m[7] * g + fwd.m[8] * b;
}

kernel void add_and_quantize(device const float* base [[buffer(0)]],
                             device const float* delta [[buffer(1)]],
                             device uchar* dst [[buffer(2)]],
                             constant ColourMatrix& inv [[buffer(3)]],
                             constant int& count [[buffer(4)]],
                             uint i [[thread_position_in_grid]]) {
    if ((int)i >= count) return;
    float c0 = base[3 * i + 0] + delta[3 * i + 0];
    float c1 = base[3 * i + 1] + delta[3 * i + 1];
    float c2 = base[3 * i + 2] + delta[3 * i + 2];

    float r = inv.m[0] * c0 + inv.m[1] * c1 + inv.m[2] * c2;
    float g = inv.m[3] * c0 + inv.m[4] * c1 + inv.m[5] * c2;
    float b = inv.m[6] * c0 + inv.m[7] * c1 + inv.m[8] * c2;

    r = clamp(r, 0.0f, 1.0f);
    g = clamp(g, 0.0f, 1.0f);
    b = clamp(b, 0.0f, 1.0f);

    dst[3 * i + 0] = (uchar)(round(b * 255.0f));
    dst[3 * i + 1] = (uchar)(round(g * 255.0f));
    dst[3 * i + 2] = (uchar)(round(r * 255.0f));
}

// ---------------------------------------------------------------------------
// Spatial. Each thread is one (channel, column, frame*row) position, so the
// reflected border never reads across a frame boundary.
// ---------------------------------------------------------------------------

kernel void corr_dn_rows(device const float* src [[buffer(0)]],
                         device float* dst [[buffer(1)]],
                         device const float* filt [[buffer(2)]],
                         constant Shape& s [[buffer(3)]],
                         uint3 gid [[thread_position_in_grid]]) {
    int c = (int)gid.x, x = (int)gid.y, idx = (int)gid.z;
    int t = idx / s.out_h, oy = idx - t * s.out_h;
    if (c >= s.C || x >= s.W || t >= s.T) return;

    int pad = s.flen / 2, y = oy * 2;
    size_t in_frame = (size_t)t * s.H * s.W * s.C;
    size_t out_frame = (size_t)t * s.out_h * s.W * s.C;
    float acc = 0.0f;
    for (int k = 0; k < s.flen; ++k) {
        int sy = reflect_index(y + k - pad, s.H);
        acc += filt[k] * src[in_frame + (size_t)(sy * s.W + x) * s.C + c];
    }
    dst[out_frame + (size_t)(oy * s.W + x) * s.C + c] = acc;
}

kernel void corr_dn_cols(device const float* src [[buffer(0)]],
                         device float* dst [[buffer(1)]],
                         device const float* filt [[buffer(2)]],
                         constant Shape& s [[buffer(3)]],
                         uint3 gid [[thread_position_in_grid]]) {
    int c = (int)gid.x, ox = (int)gid.y, idx = (int)gid.z;
    int t = idx / s.H, y = idx - t * s.H;
    if (c >= s.C || ox >= s.out_w || t >= s.T) return;

    int pad = s.flen / 2, x = ox * 2;
    size_t in_frame = (size_t)t * s.H * s.W * s.C;
    size_t out_frame = (size_t)t * s.H * s.out_w * s.C;
    float acc = 0.0f;
    for (int k = 0; k < s.flen; ++k) {
        int sx = reflect_index(x + k - pad, s.W);
        acc += filt[k] * src[in_frame + (size_t)(y * s.W + sx) * s.C + c];
    }
    dst[out_frame + (size_t)(y * s.out_w + ox) * s.C + c] = acc;
}

kernel void up_conv_rows(device const float* src [[buffer(0)]],
                         device float* dst [[buffer(1)]],
                         device const float* filt [[buffer(2)]],
                         constant Shape& s [[buffer(3)]],
                         uint3 gid [[thread_position_in_grid]]) {
    int c = (int)gid.x, x = (int)gid.y, idx = (int)gid.z;
    int t = idx / s.out_h, oy = idx - t * s.out_h;
    if (c >= s.C || x >= s.W || t >= s.T) return;

    int pad = s.flen / 2, up_h = s.H * 2;
    size_t in_frame = (size_t)t * s.H * s.W * s.C;
    size_t out_frame = (size_t)t * s.out_h * s.W * s.C;
    float acc = 0.0f;
    for (int k = 0; k < s.flen; ++k) {
        int sy = reflect_index(oy + k - pad, up_h);
        if ((sy & 1) == 0) {
            int row = sy >> 1;
            if (row < s.H)
                acc += filt[k] * src[in_frame + (size_t)(row * s.W + x) * s.C + c];
        }
    }
    dst[out_frame + (size_t)(oy * s.W + x) * s.C + c] = acc;
}

kernel void up_conv_cols(device const float* src [[buffer(0)]],
                         device float* dst [[buffer(1)]],
                         device const float* filt [[buffer(2)]],
                         constant Shape& s [[buffer(3)]],
                         uint3 gid [[thread_position_in_grid]]) {
    int c = (int)gid.x, ox = (int)gid.y, idx = (int)gid.z;
    int t = idx / s.H, y = idx - t * s.H;
    if (c >= s.C || ox >= s.out_w || t >= s.T) return;

    int pad = s.flen / 2, up_w = s.W * 2;
    size_t in_frame = (size_t)t * s.H * s.W * s.C;
    size_t out_frame = (size_t)t * s.H * s.out_w * s.C;
    float acc = 0.0f;
    for (int k = 0; k < s.flen; ++k) {
        int sx = reflect_index(ox + k - pad, up_w);
        if ((sx & 1) == 0) {
            int col = sx >> 1;
            if (col < s.W)
                acc += filt[k] * src[in_frame + (size_t)(y * s.W + col) * s.C + c];
        }
    }
    dst[out_frame + (size_t)(y * s.out_w + ox) * s.C + c] = acc;
}

kernel void subtract(device const float* a [[buffer(0)]],
                     device const float* b [[buffer(1)]],
                     device float* dst [[buffer(2)]],
                     constant int& count [[buffer(3)]],
                     uint i [[thread_position_in_grid]]) {
    if ((int)i >= count) return;
    dst[i] = a[i] - b[i];
}

kernel void add_into(device float* acc [[buffer(0)]],
                     device const float* b [[buffer(1)]],
                     constant int& count [[buffer(2)]],
                     uint i [[thread_position_in_grid]]) {
    if ((int)i >= count) return;
    acc[i] += b[i];
}

// ---------------------------------------------------------------------------
// Bilinear resize
// ---------------------------------------------------------------------------

struct ResizeShape {
    int T;
    int in_h;
    int in_w;
    int out_h;
    int out_w;
    int C;
};

kernel void resize_bilinear(device const float* src [[buffer(0)]],
                            device float* dst [[buffer(1)]],
                            constant ResizeShape& s [[buffer(2)]],
                            uint3 gid [[thread_position_in_grid]]) {
    int c = (int)gid.x, ox = (int)gid.y, idx = (int)gid.z;
    int t = idx / s.out_h, oy = idx - t * s.out_h;
    if (c >= s.C || ox >= s.out_w || t >= s.T) return;

    size_t in_frame = (size_t)t * s.in_h * s.in_w * s.C;
    size_t out_frame = (size_t)t * s.out_h * s.out_w * s.C;

    // Half-pixel centres, the same convention the reference resize uses.
    float fy = ((float)oy + 0.5f) * ((float)s.in_h / (float)s.out_h) - 0.5f;
    float fx = ((float)ox + 0.5f) * ((float)s.in_w / (float)s.out_w) - 0.5f;

    int y0 = (int)floor(fy), x0 = (int)floor(fx);
    float wy = fy - (float)y0, wx = fx - (float)x0;

    int y1 = min(max(y0 + 1, 0), s.in_h - 1);
    int x1 = min(max(x0 + 1, 0), s.in_w - 1);
    y0 = min(max(y0, 0), s.in_h - 1);
    x0 = min(max(x0, 0), s.in_w - 1);

    float v00 = src[in_frame + (size_t)(y0 * s.in_w + x0) * s.C + c];
    float v01 = src[in_frame + (size_t)(y0 * s.in_w + x1) * s.C + c];
    float v10 = src[in_frame + (size_t)(y1 * s.in_w + x0) * s.C + c];
    float v11 = src[in_frame + (size_t)(y1 * s.in_w + x1) * s.C + c];

    float top = v00 + wx * (v01 - v00);
    float bot = v10 + wx * (v11 - v10);
    dst[out_frame + (size_t)(oy * s.out_w + ox) * s.C + c] = top + wy * (bot - top);
}

// ---------------------------------------------------------------------------
// Temporal. One thread owns one pixel's whole time series, so the recursions
// stay sequential in time while every pixel runs in parallel.
// Layout is (time, pixel): element t of pixel i is at t * N + i.
// ---------------------------------------------------------------------------

struct TemporalShape {
    int T;
    int N;
};

kernel void iir_bandpass(device const float* src [[buffer(0)]],
                         device float* dst [[buffer(1)]],
                         constant TemporalShape& s [[buffer(2)]],
                         constant float2& rates [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
    if ((int)i >= s.N) return;
    float r1 = rates.x, r2 = rates.y;
    float low1 = src[i], low2 = src[i];
    dst[i] = 0.0f;
    for (int t = 1; t < s.T; ++t) {
        float x = src[t * s.N + i];
        low1 = (1.0f - r1) * low1 + r1 * x;
        low2 = (1.0f - r2) * low2 + r2 * x;
        dst[t * s.N + i] = low1 - low2;
    }
}

struct ButterCoefficients {
    float b0h, b1h, a1h;
    float b0l, b1l, a1l;
};

kernel void butter_bandpass(device const float* src [[buffer(0)]],
                            device float* dst [[buffer(1)]],
                            constant TemporalShape& s [[buffer(2)]],
                            constant ButterCoefficients& c [[buffer(3)]],
                            uint i [[thread_position_in_grid]]) {
    if ((int)i >= s.N) return;
    float x_prev = 0.0f, yh_prev = 0.0f, yl_prev = 0.0f;
    for (int t = 0; t < s.T; ++t) {
        float x = src[t * s.N + i];
        float yh = c.b0h * x + c.b1h * x_prev - c.a1h * yh_prev;
        float yl = c.b0l * x + c.b1l * x_prev - c.a1l * yl_prev;
        dst[t * s.N + i] = yh - yl;
        x_prev = x;
        yh_prev = yh;
        yl_prev = yl;
    }
}

// Keep a band of frequencies, as a matrix multiply. Selecting frequency bins
// is a linear map, so one matrix built on the host is exactly equal to a
// transform, a mask and an inverse transform — and needs no maths library on
// the device, which is what lets these kernels run anywhere.
kernel void band_project(device const float* src [[buffer(0)]],
                         device float* dst [[buffer(1)]],
                         device const float* matrix [[buffer(2)]],
                         constant TemporalShape& s [[buffer(3)]],
                         uint2 gid [[thread_position_in_grid]]) {
    int i = (int)gid.x, t = (int)gid.y;
    if (i >= s.N || t >= s.T) return;
    float acc = 0.0f;
    for (int k = 0; k < s.T; ++k) acc += matrix[t * s.T + k] * src[k * s.N + i];
    dst[t * s.N + i] = acc;
}

// ---------------------------------------------------------------------------
// Amplification and streaming
// ---------------------------------------------------------------------------

kernel void apply_gain(device float* data [[buffer(0)]],
                       constant int& count [[buffer(1)]],
                       constant float3& gains [[buffer(2)]],
                       uint i [[thread_position_in_grid]]) {
    if ((int)i >= count) return;
    data[3 * i + 0] *= gains.x;
    data[3 * i + 1] *= gains.y;
    data[3 * i + 2] *= gains.z;
}

// One step of the two running averages, and their difference. Keeps a live
// feed's state on the device instead of copying it to the host every frame.
kernel void iir_step(device float* fast [[buffer(0)]],
                     device float* slow [[buffer(1)]],
                     device const float* current [[buffer(2)]],
                     device float* out [[buffer(3)]],
                     constant int& count [[buffer(4)]],
                     constant float2& rates [[buffer(5)]],
                     uint i [[thread_position_in_grid]]) {
    if ((int)i >= count) return;
    float x = current[i];
    float f = fast[i] * (1.0f - rates.x) + x * rates.x;
    float s = slow[i] * (1.0f - rates.y) + x * rates.y;
    fast[i] = f;
    slow[i] = s;
    out[i] = f - s;
}
