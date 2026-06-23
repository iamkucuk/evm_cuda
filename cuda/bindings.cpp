// bindings.cpp — pybind11 module exposing EVM CUDA kernels to Python.
//
// Each `m.def()` is a thin wrapper that:
//   1. Validates the input numpy array (dtype, contiguity, shape).
//   2. Allocates device memory and copies the input up.
//   3. Launches the kernel.
//   4. Copies the result back and returns a new numpy array.
//
// We keep these wrappers deliberately small and stateless so the kernel
// implementations in kernels/*.cu are the only place CUDA math lives. The
// higher-level orchestration (pyramid level loops, pipeline composition,
// cuFFT plan caching) lives in evm_cuda/pipelines.py on the Python side.
//
// Naming convention: every Python-callable function is snake_case matching
// the evm/ Python baseline (e.g. `corr_dn_rows` mirrors `corr_dn_axis(...,0)`),
// so tests can swap `evm.X` <-> `_evm_cuda.X` with minimal edits.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cufft.h>
#include <cuda_fp16.h>
#include <memory>
#include <unordered_map>
#include <vector>
#include <utility>

#include "../include/evm_common.cuh"
#include "../include/evm_check.cuh"

namespace py = pybind11;

// pybind11 array flags: force C-contiguous, row-major layout. The Python
// baseline (numpy) is happy to hand us F-order transposed views (e.g. the
// output of corr_dn_axis on axis=0 is F-contiguous); our kernels assume
// row-major, so we ask pybind11 to make a contiguous cast on the way in.
// Lives at global scope (not in evm::) so the PYBIND11_MODULE lambdas can
// name it without qualification.
template <typename T>
using carray_t = py::array_t<T, py::array::c_style | py::array::forcecast>;

namespace evm {

// --- launcher decls (defined in each kernels/*.cu) -------------------------
void launch_bgr_u8_to_ntsc_f32(const unsigned char* bgr, float* yiq,
                                int H, int W, cudaStream_t stream);
void launch_ntsc_f32_to_bgr_u8(const float* yiq, unsigned char* bgr,
                                int H, int W, cudaStream_t stream);
void launch_corr_dn_rows(const float* in, float* out, int H, int W,
                         const float* filt, int filt_len, cudaStream_t stream);
void launch_corr_dn_cols(const float* in, float* out, int H, int W,
                         const float* filt, int filt_len, cudaStream_t stream);
void launch_up_conv_rows(const float* in, float* out,
                         int in_H, int out_H, int W,
                         const float* filt, int filt_len, cudaStream_t stream);
void launch_up_conv_cols(const float* in, float* out,
                         int H, int in_W, int out_W,
                         const float* filt, int filt_len, cudaStream_t stream);
void launch_thwc_to_nt(const float* src, float* dst, int T, int N,
                       cudaStream_t stream);
void launch_nt_to_thwc(const float* src, float* dst, int T, int N,
                       cudaStream_t stream);
void launch_nt_to_thwc_scaled(const float* src, float* dst, int T, int N,
                              float scale, cudaStream_t stream);
void launch_to_planar_3ch(const float* src, float* dst, int n, int H, int W,
                          cudaStream_t stream);
void launch_planar_to_interleaved_3ch(const float* src, float* dst,
                                      int n, int H, int W, cudaStream_t stream);
void launch_iir_bandpass(const float* in, float* out, int T, int N,
                         double r1, double r2, cudaStream_t stream);
void launch_butter_bandpass(const float* in, float* out, int T, int N,
                            double b0_high, double b1_high, double a1_high,
                            double b0_low,  double b1_low,  double a1_low,
                            cudaStream_t stream);
void launch_apply_channel_gain(float* sig, int H, int W,
                               float g0, float g1, float g2,
                               cudaStream_t stream);
void launch_attenuate_chrom(float* delta, int H, int W, float chrom_att,
                            cudaStream_t stream);
void launch_scale_inplace(float* data, int n, float scale, cudaStream_t stream);
void launch_bilinear_upsample_3ch(const float* src, float* dst,
                                  int M, int in_H, int in_W,
                                  int out_H, int out_W, cudaStream_t stream);
void launch_add_and_quantize(const float* ntsc_frame, const float* delta,
                             unsigned char* bgr_out, int H, int W,
                             float chrom_att, cudaStream_t stream);
void launch_add_planar_quantize(const float* ntsc, const float* delta_planar,
                                unsigned char* bgr_out,
                                int n, int H, int W, float chrom_att,
                                cudaStream_t stream);
void launch_upsample_add_quantize(const float* ntsc, const float* filt,
                                  unsigned char* bgr_out,
                                  int M, int in_H, int in_W,
                                  int out_H, int out_W, float chrom_att,
                                  cudaStream_t stream);

// ideal_bandpass.cu — self-contained cuFFT fwd+mask+inv pipeline.
void launch_ideal_bandpass(
    const float* in, float* out, cufftComplex* tmp,
    int T, int N, float wl, float wh, float sampling_rate,
    cufftHandle plan_fwd, cufftHandle plan_inv,
    cudaStream_t stream);

// --- high-level orchestrators (defined in lpyr.cu, blur_dn.cu) -------------
//
// These take pre-allocated device pointers and the (host) per-level size
// table; the pybind11 wrappers below own the device memory lifecycle and
// the scratch allocations.

struct LevelSize { int h; int w; };

// lpyr.cu
std::vector<std::pair<int, int>> lpyr_level_sizes(int H, int W, int levels);
void lpyr_build_device(
    const float* img, int H, int W,
    float** band_ptrs, const std::pair<int, int>* sizes, int levels,
    const float* filt, int filt_len,
    float* scratch_a, float* scratch_b, float* scratch_c,
    cudaStream_t stream);
void lpyr_recon_device(
    const float* const* band_ptrs,
    const std::pair<int, int>* sizes, int levels,
    const float* filt, int filt_len,
    float* out,
    float* scratch_lo, float* scratch_hi,
    cudaStream_t stream);

// Batched spatial kernels (spatial.cu) — B slices per launch via grid.z.
void launch_corr_dn_rows_batched(const float* in, float* out,
                                 int H, int W, const float* filt, int filt_len,
                                 int stride_in, int stride_out, int B,
                                 cudaStream_t stream);
void launch_corr_dn_cols_batched(const float* in, float* out,
                                 int H, int W, const float* filt, int filt_len,
                                 int stride_in, int stride_out, int B,
                                 cudaStream_t stream);
void launch_up_conv_rows_batched(const float* in, float* out,
                                 int in_H, int out_H, int W,
                                 const float* filt, int filt_len,
                                 int stride_in, int stride_out, int B,
                                 cudaStream_t stream);
void launch_up_conv_cols_batched(const float* in, float* out,
                                 int H, int in_W, int out_W,
                                 const float* filt, int filt_len,
                                 int stride_in, int stride_out, int B,
                                 cudaStream_t stream);

// Scatter/gather for channel-major band layout (lpyr.cu).
void launch_scatter_subtract(const float* a, const float* b, float* dst,
                             const int* offsets, int n_per_slice, int B,
                             cudaStream_t stream);
void launch_scatter(const float* src, float* dst,
                    const int* offsets, int n_per_slice, int B,
                    cudaStream_t stream);
void launch_gather_add(const float* src, const float* b, float* dst,
                       const int* offsets, int n_per_slice, int B,
                       cudaStream_t stream);
void launch_gather(const float* src, float* dst,
                   const int* offsets, int n_per_slice, int B,
                   cudaStream_t stream);

// fp16_cvt.cu — FP16↔FP32 conversion at buffer boundaries.
void launch_f32_to_f16(const float* src, __half* dst, int n, cudaStream_t stream);
void launch_f16_to_f32(const __half* src, float* dst, int n, cudaStream_t stream);

// blur_dn.cu
void blur_dn_device(
    const float* in, int H, int W,
    float* out, int nlevs,
    const float* filt, int filt_len,
    float* scratch_a, float* scratch_b,
    cudaStream_t stream);

}  // namespace evm

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

template <typename T>
T* device_alloc(size_t n) {
    T* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
    return p;
}

void device_free(void* p) {
    if (p) CUDA_CHECK(cudaFree(p));
}

// cuFFT plan cache for ideal_bandpass. cufftPlanMany does internal autotuning
// (kernel selection, workspace sizing) on every call — measured at ~5-10ms
// per plan on H200, called 2x per channel x 3 channels = ~30-60ms per
// pipeline invocation. The plan depends only on (T, N): same length-T batched
// over N signals. Caching by that key makes the 2nd+ pipeline call skip plan
// creation entirely.
//
// Thread-safety: the GIL serializes Python-side calls into this module, so
// the cache access is single-threaded in practice.
struct FftPlanPair { cufftHandle fwd; cufftHandle inv; };
std::unordered_map<long long, FftPlanPair> g_fft_cache;

FftPlanPair get_or_create_fft_plans(int T, int N) {
    // Key combines T and N into a single int64 (T in high 32 bits, N in low).
    // T (frame count) and N (spatial locations) are both << 2^31 in practice.
    long long key = (static_cast<long long>(T) << 32) | static_cast<long long>(N);
    auto it = g_fft_cache.find(key);
    if (it != g_fft_cache.end()) return it->second;

    FftPlanPair p;
    int n_arr[1] = {T};
    int in_emb[2] = {T, 1};
    CUFFT_CHECK(cufftPlanMany(&p.fwd, 1, n_arr,
                              in_emb, 1, T, in_emb, 1, T, CUFFT_C2C, N));
    CUFFT_CHECK(cufftPlanMany(&p.inv, 1, n_arr,
                              in_emb, 1, T, in_emb, 1, T, CUFFT_C2C, N));
    g_fft_cache[key] = p;
    return p;
}

// RAII device memory buffer. Holds a cudaMalloc'd region for the lifetime of
// a pipeline call; the Python-facing DeviceBuffer class wraps this.
struct DeviceBuffer {
    void* ptr = nullptr;
    size_t nbytes = 0;
    explicit DeviceBuffer(size_t n) : nbytes(n) {
        if (n > 0) CUDA_CHECK(cudaMalloc(&ptr, n));
    }
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

// Validate a numpy array is C-contiguous float32 with a trailing channel
// axis of size 3.
void require_chw3_float32(const py::array& a, const char* name) {
    if (a.ndim() != 3 || a.shape(2) != 3)
        throw std::runtime_error(std::string(name) + ": expected (H,W,3)");
    if (a.dtype().char_() != 'f' || !(py::str(a.dtype()).equal(py::str(py::dtype::of<float>()))))
        throw std::runtime_error(std::string(name) + ": expected float32");
    if (!(a.flags() & py::array::c_style))
        throw std::runtime_error(std::string(name) + ": expected C-contiguous");
}

}  // namespace

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

PYBIND11_MODULE(_evm_cuda, m) {
    m.doc() = "EVM CUDA kernels (raw nvcc + pybind11).";

    // --- color_cvt --------------------------------------------------------

    m.def("bgr_u8_to_ntsc_f32", [](carray_t<unsigned char> bgr) {
        auto buf = bgr.request();
        if (buf.ndim != 3 || buf.shape[2] != 3)
            throw std::runtime_error("bgr_u8_to_ntsc_f32: expected (H,W,3) uint8");
        int H = buf.shape[0], W = buf.shape[1];
        carray_t<float> yiq({H, W, 3});
        auto y = yiq.request();
        auto* d_in  = device_alloc<unsigned char>(H * W * 3);
        auto* d_out = device_alloc<float>(H * W * 3);
        CUDA_CHECK(cudaMemcpy(d_in, buf.ptr, H * W * 3 * sizeof(unsigned char),
                              cudaMemcpyHostToDevice));
        evm::launch_bgr_u8_to_ntsc_f32(d_in, d_out, H, W, 0);
        CUDA_CHECK(cudaMemcpy(y.ptr, d_out, H * W * 3 * sizeof(float),
                              cudaMemcpyDeviceToHost));
        device_free(d_in); device_free(d_out);
        return yiq;
    }, py::arg("bgr"));

    m.def("ntsc_f32_to_bgr_u8", [](carray_t<float> ntsc) {
        auto buf = ntsc.request();
        if (buf.ndim != 3 || buf.shape[2] != 3)
            throw std::runtime_error("ntsc_f32_to_bgr_u8: expected (H,W,3) float32");
        int H = buf.shape[0], W = buf.shape[1];
        carray_t<unsigned char> bgr({H, W, 3});
        auto b = bgr.request();
        auto* d_in  = device_alloc<float>(H * W * 3);
        auto* d_out = device_alloc<unsigned char>(H * W * 3);
        CUDA_CHECK(cudaMemcpy(d_in, buf.ptr, H * W * 3 * sizeof(float),
                              cudaMemcpyHostToDevice));
        evm::launch_ntsc_f32_to_bgr_u8(d_in, d_out, H, W, 0);
        CUDA_CHECK(cudaMemcpy(b.ptr, d_out, H * W * 3 * sizeof(unsigned char),
                              cudaMemcpyDeviceToHost));
        device_free(d_in); device_free(d_out);
        return bgr;
    }, py::arg("ntsc"));

    // --- spatial primitives ------------------------------------------------

    m.def("corr_dn_rows", [](carray_t<float> in, carray_t<float> filt) {
        auto b = in.request(); auto f = filt.request();
        if (b.ndim != 2) throw std::runtime_error("corr_dn_rows: expected (H,W)");
        int H = b.shape[0], W = b.shape[1], fl = f.shape[0];
        carray_t<float> out({(H + 1) / 2, W});
        auto o = out.request();
        auto* d_in = device_alloc<float>(H * W);
        auto* d_f  = device_alloc<float>(fl);
        auto* d_o  = device_alloc<float>(((H + 1) / 2) * W);
        CUDA_CHECK(cudaMemcpy(d_in, b.ptr, H * W * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_f,  f.ptr, fl * sizeof(float), cudaMemcpyHostToDevice));
        evm::launch_corr_dn_rows(d_in, d_o, H, W, d_f, fl, 0);
        CUDA_CHECK(cudaMemcpy(o.ptr, d_o, ((H + 1) / 2) * W * sizeof(float), cudaMemcpyDeviceToHost));
        device_free(d_in); device_free(d_f); device_free(d_o);
        return out;
    }, py::arg("in"), py::arg("filt"));

    m.def("corr_dn_cols", [](carray_t<float> in, carray_t<float> filt) {
        auto b = in.request(); auto f = filt.request();
        if (b.ndim != 2) throw std::runtime_error("corr_dn_cols: expected (H,W)");
        int H = b.shape[0], W = b.shape[1], fl = f.shape[0];
        carray_t<float> out({H, (W + 1) / 2});
        auto o = out.request();
        auto* d_in = device_alloc<float>(H * W);
        auto* d_f  = device_alloc<float>(fl);
        auto* d_o  = device_alloc<float>(H * ((W + 1) / 2));
        CUDA_CHECK(cudaMemcpy(d_in, b.ptr, H * W * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_f,  f.ptr, fl * sizeof(float), cudaMemcpyHostToDevice));
        evm::launch_corr_dn_cols(d_in, d_o, H, W, d_f, fl, 0);
        CUDA_CHECK(cudaMemcpy(o.ptr, d_o, H * ((W + 1) / 2) * sizeof(float), cudaMemcpyDeviceToHost));
        device_free(d_in); device_free(d_f); device_free(d_o);
        return out;
    }, py::arg("in"), py::arg("filt"));

    m.def("up_conv_rows", [](carray_t<float> in, int out_H, carray_t<float> filt) {
        auto b = in.request(); auto f = filt.request();
        if (b.ndim != 2) throw std::runtime_error("up_conv_rows: expected (in_H,W)");
        int in_H = b.shape[0], W = b.shape[1], fl = f.shape[0];
        carray_t<float> out({out_H, W});
        auto o = out.request();
        auto* d_in = device_alloc<float>(in_H * W);
        auto* d_f  = device_alloc<float>(fl);
        auto* d_o  = device_alloc<float>(out_H * W);
        CUDA_CHECK(cudaMemcpy(d_in, b.ptr, in_H * W * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_f,  f.ptr, fl * sizeof(float), cudaMemcpyHostToDevice));
        evm::launch_up_conv_rows(d_in, d_o, in_H, out_H, W, d_f, fl, 0);
        CUDA_CHECK(cudaMemcpy(o.ptr, d_o, out_H * W * sizeof(float), cudaMemcpyDeviceToHost));
        device_free(d_in); device_free(d_f); device_free(d_o);
        return out;
    }, py::arg("in"), py::arg("out_H"), py::arg("filt"));

    m.def("up_conv_cols", [](carray_t<float> in, int out_W, carray_t<float> filt) {
        auto b = in.request(); auto f = filt.request();
        if (b.ndim != 2) throw std::runtime_error("up_conv_cols: expected (H,in_W)");
        int H = b.shape[0], in_W = b.shape[1], fl = f.shape[0];
        carray_t<float> out({H, out_W});
        auto o = out.request();
        auto* d_in = device_alloc<float>(H * in_W);
        auto* d_f  = device_alloc<float>(fl);
        auto* d_o  = device_alloc<float>(H * out_W);
        CUDA_CHECK(cudaMemcpy(d_in, b.ptr, H * in_W * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_f,  f.ptr, fl * sizeof(float), cudaMemcpyHostToDevice));
        evm::launch_up_conv_cols(d_in, d_o, H, in_W, out_W, d_f, fl, 0);
        CUDA_CHECK(cudaMemcpy(o.ptr, d_o, H * out_W * sizeof(float), cudaMemcpyDeviceToHost));
        device_free(d_in); device_free(d_f); device_free(d_o);
        return out;
    }, py::arg("in"), py::arg("out_W"), py::arg("filt"));

    // --- transpose (T,H,W,C) <-> (N,T) -------------------------------------

    m.def("thwc_to_nt", [](carray_t<float> in) {
        auto b = in.request();
        if (b.ndim != 4 || b.shape[3] != 3)
            throw std::runtime_error("thwc_to_nt: expected (T,H,W,3)");
        int T = b.shape[0], H = b.shape[1], W = b.shape[2];
        int N = H * W * 3;
        carray_t<float> out({N, T});
        auto o = out.request();
        auto* d_in = device_alloc<float>(T * N);
        auto* d_o  = device_alloc<float>(N * T);
        CUDA_CHECK(cudaMemcpy(d_in, b.ptr, T * N * sizeof(float), cudaMemcpyHostToDevice));
        evm::launch_thwc_to_nt(d_in, d_o, T, N, 0);
        CUDA_CHECK(cudaMemcpy(o.ptr, d_o, N * T * sizeof(float), cudaMemcpyDeviceToHost));
        device_free(d_in); device_free(d_o);
        return out;
    }, py::arg("in"));

    m.def("nt_to_thwc", [](carray_t<float> in, int H, int W) {
        auto b = in.request();
        if (b.ndim != 2) throw std::runtime_error("nt_to_thwc: expected (N,T)");
        int N = b.shape[0], T = b.shape[1];
        if (N != H * W * 3) throw std::runtime_error("nt_to_thwc: N mismatch");
        carray_t<float> out({T, H, W, 3});
        auto o = out.request();
        auto* d_in = device_alloc<float>(N * T);
        auto* d_o  = device_alloc<float>(T * N);
        CUDA_CHECK(cudaMemcpy(d_in, b.ptr, N * T * sizeof(float), cudaMemcpyHostToDevice));
        evm::launch_nt_to_thwc(d_in, d_o, T, N, 0);
        CUDA_CHECK(cudaMemcpy(o.ptr, d_o, T * N * sizeof(float), cudaMemcpyDeviceToHost));
        device_free(d_in); device_free(d_o);
        return out;
    }, py::arg("in"), py::arg("H"), py::arg("W"));

    // --- temporal filters --------------------------------------------------

    m.def("iir_bandpass", [](carray_t<float> in, double r1, double r2) {
        auto b = in.request();
        if (b.ndim != 2) throw std::runtime_error("iir_bandpass: expected (N,T)");
        int N = b.shape[0], T = b.shape[1];
        carray_t<float> out({N, T});
        auto o = out.request();
        auto* d_in = device_alloc<float>(N * T);
        auto* d_o  = device_alloc<float>(N * T);
        CUDA_CHECK(cudaMemcpy(d_in, b.ptr, N * T * sizeof(float), cudaMemcpyHostToDevice));
        evm::launch_iir_bandpass(d_in, d_o, T, N, r1, r2, 0);
        CUDA_CHECK(cudaMemcpy(o.ptr, d_o, N * T * sizeof(float), cudaMemcpyDeviceToHost));
        device_free(d_in); device_free(d_o);
        return out;
    }, py::arg("in"), py::arg("r1"), py::arg("r2"));

    m.def("butter_bandpass", [](carray_t<float> in,
                                double b0_h, double b1_h, double a1_h,
                                double b0_l, double b1_l, double a1_l) {
        auto b = in.request();
        if (b.ndim != 2) throw std::runtime_error("butter_bandpass: expected (N,T)");
        int N = b.shape[0], T = b.shape[1];
        carray_t<float> out({N, T});
        auto o = out.request();
        auto* d_in = device_alloc<float>(N * T);
        auto* d_o  = device_alloc<float>(N * T);
        CUDA_CHECK(cudaMemcpy(d_in, b.ptr, N * T * sizeof(float), cudaMemcpyHostToDevice));
        evm::launch_butter_bandpass(d_in, d_o, T, N,
                                    b0_h, b1_h, a1_h, b0_l, b1_l, a1_l, 0);
        CUDA_CHECK(cudaMemcpy(o.ptr, d_o, N * T * sizeof(float), cudaMemcpyDeviceToHost));
        device_free(d_in); device_free(d_o);
        return out;
    }, py::arg("in"),
       py::arg("b0_high"), py::arg("b1_high"), py::arg("a1_high"),
       py::arg("b0_low"),  py::arg("b1_low"),  py::arg("a1_low"));

    // --- amplify_render helpers -------------------------------------------

    m.def("apply_channel_gain", [](carray_t<float> sig,
                                   float g0, float g1, float g2) {
        auto b = sig.request();
        if (b.ndim != 3 || b.shape[2] != 3)
            throw std::runtime_error("apply_channel_gain: expected (H,W,3)");
        int H = b.shape[0], W = b.shape[1];
        auto* d = device_alloc<float>(H * W * 3);
        CUDA_CHECK(cudaMemcpy(d, b.ptr, H * W * 3 * sizeof(float), cudaMemcpyHostToDevice));
        evm::launch_apply_channel_gain(d, H, W, g0, g1, g2, 0);
        carray_t<float> out({H, W, 3});
        auto o = out.request();
        CUDA_CHECK(cudaMemcpy(o.ptr, d, H * W * 3 * sizeof(float), cudaMemcpyDeviceToHost));
        device_free(d);
        return out;
    }, py::arg("sig"), py::arg("g0"), py::arg("g1"), py::arg("g2"));

    m.def("attenuate_chrom", [](carray_t<float> delta, float chrom_att) {
        auto b = delta.request();
        if (b.ndim != 3 || b.shape[2] != 3)
            throw std::runtime_error("attenuate_chrom: expected (H,W,3)");
        int H = b.shape[0], W = b.shape[1];
        auto* d = device_alloc<float>(H * W * 3);
        CUDA_CHECK(cudaMemcpy(d, b.ptr, H * W * 3 * sizeof(float), cudaMemcpyHostToDevice));
        evm::launch_attenuate_chrom(d, H, W, chrom_att, 0);
        carray_t<float> out({H, W, 3});
        auto o = out.request();
        CUDA_CHECK(cudaMemcpy(o.ptr, d, H * W * 3 * sizeof(float), cudaMemcpyDeviceToHost));
        device_free(d);
        return out;
    }, py::arg("delta"), py::arg("chrom_att"));

    m.def("add_and_quantize", [](carray_t<float> ntsc_frame,
                                 carray_t<float> delta) {
        auto bf = ntsc_frame.request(); auto bd = delta.request();
        if (bf.ndim != 3 || bf.shape[2] != 3 || bd.shape[2] != 3)
            throw std::runtime_error("add_and_quantize: expected (H,W,3)");
        int H = bf.shape[0], W = bf.shape[1];
        carray_t<unsigned char> bgr({H, W, 3});
        auto bo = bgr.request();
        auto* d_f = device_alloc<float>(H * W * 3);
        auto* d_d = device_alloc<float>(H * W * 3);
        auto* d_o = device_alloc<unsigned char>(H * W * 3);
        CUDA_CHECK(cudaMemcpy(d_f, bf.ptr, H * W * 3 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_d, bd.ptr, H * W * 3 * sizeof(float), cudaMemcpyHostToDevice));
        evm::launch_add_and_quantize(d_f, d_d, d_o, H, W, 1.0f, 0);
        CUDA_CHECK(cudaMemcpy(bo.ptr, d_o, H * W * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost));
        device_free(d_f); device_free(d_d); device_free(d_o);
        return bgr;
    }, py::arg("ntsc_frame"), py::arg("delta"));

    // --- ideal_bandpass (self-contained cuFFT plan lifecycle) --------------
    //
    // Plans are created and destroyed within the call. For sustained
    // throughput the pipeline layer should batch many bandpass calls
    // together; per-call plan creation is acceptable for the baseline
    // accuracy comparison. (A plan cache can be added later if profiling
    // shows it matters.)

    m.def("ideal_bandpass", [](carray_t<float> in,
                               float wl, float wh, float sampling_rate) {
        auto b = in.request();
        if (b.ndim != 2) throw std::runtime_error("ideal_bandpass: expected (N,T)");
        int N = b.shape[0], T = b.shape[1];
        const size_t real_bytes = static_cast<size_t>(N) * T * sizeof(float);
        const size_t cplx_bytes = static_cast<size_t>(N) * T * sizeof(cufftComplex);

        auto* d_in  = device_alloc<float>(static_cast<size_t>(N) * T);
        auto* d_out = device_alloc<float>(static_cast<size_t>(N) * T);
        auto* d_tmp = device_alloc<cufftComplex>(static_cast<size_t>(N) * T);

        CUDA_CHECK(cudaMemcpy(d_in, b.ptr, real_bytes, cudaMemcpyHostToDevice));

        // Create batched C2C plans. cufftPlanMany with stride=1, dist=T makes
        // each of the N rows a contiguous length-T signal — matches the
        // (N, T) layout produced by thwc_to_nt.
        cufftHandle plan_fwd, plan_inv;
        int n_arr[1] = {T};
        int in_emb[2]  = {T, 1};   // (stride-on-stack, dist)
        // For cufftPlanMany: n, inembed, istride, idist, onembed, ostride, odist
        CUFFT_CHECK(cufftPlanMany(&plan_fwd, 1, n_arr,
                                  in_emb, 1, T,  in_emb, 1, T,
                                  CUFFT_C2C, N));
        CUFFT_CHECK(cufftPlanMany(&plan_inv, 1, n_arr,
                                  in_emb, 1, T,  in_emb, 1, T,
                                  CUFFT_C2C, N));

        evm::launch_ideal_bandpass(d_in, d_out, d_tmp,
                                    T, N, wl, wh, sampling_rate,
                                    plan_fwd, plan_inv, 0);

        CUFFT_CHECK(cufftDestroy(plan_fwd));
        CUFFT_CHECK(cufftDestroy(plan_inv));

        carray_t<float> out({N, T});
        auto o = out.request();
        CUDA_CHECK(cudaMemcpy(o.ptr, d_out, real_bytes, cudaMemcpyDeviceToHost));
        device_free(d_in); device_free(d_out); device_free(d_tmp);
        return out;
    }, py::arg("in"), py::arg("wl"), py::arg("wh"), py::arg("sampling_rate"));

    // --- Laplacian pyramid build/reconstruct (host orchestration) ----------
    //
    // Returns the per-level sizes alongside the flat band vector so the
    // Python pipeline layer can manage pind. Input is (H, W) float32;
    // output is a list of (H_l, W_l) float32 arrays, finest-first.

    m.def("lpyr_build", [](carray_t<float> img, int levels,
                           carray_t<float> filt) {
        auto b = img.request(); auto f = filt.request();
        if (b.ndim != 2) throw std::runtime_error("lpyr_build: expected (H,W)");
        int H = b.shape[0], W = b.shape[1], fl = f.shape[0];
        auto sizes_vec = evm::lpyr_level_sizes(H, W, levels);

        // Device buffers: input, per-level bands, scratch (3 buffers each
        // large enough for the largest level = H*W).
        auto* d_img = device_alloc<float>(H * W);
        auto* d_filt = device_alloc<float>(fl);
        CUDA_CHECK(cudaMemcpy(d_img,  b.ptr, H * W * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_filt, f.ptr, fl * sizeof(float),   cudaMemcpyHostToDevice));

        std::vector<float*> band_ptrs(levels);
        std::vector<float*> band_storage(levels);
        for (int l = 0; l < levels; ++l) {
            int sz = sizes_vec[l].first * sizes_vec[l].second;
            band_storage[l] = device_alloc<float>(sz);
            band_ptrs[l] = band_storage[l];
        }
        auto* scratch_a = device_alloc<float>(H * W);
        auto* scratch_b = device_alloc<float>(H * W);
        auto* scratch_c = device_alloc<float>(H * W);

        evm::lpyr_build_device(d_img, H, W, band_ptrs.data(),
                               sizes_vec.data(), levels,
                               d_filt, fl,
                               scratch_a, scratch_b, scratch_c, 0);

        // Pull each band back to a numpy array.
        py::list out_bands;
        for (int l = 0; l < levels; ++l) {
            int hl = sizes_vec[l].first, wl = sizes_vec[l].second;
            carray_t<float> band({hl, wl});
            auto bo = band.request();
            CUDA_CHECK(cudaMemcpy(bo.ptr, band_storage[l],
                                  hl * wl * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            out_bands.append(band);
        }
        for (int l = 0; l < levels; ++l) device_free(band_storage[l]);
        device_free(d_img); device_free(d_filt);
        device_free(scratch_a); device_free(scratch_b); device_free(scratch_c);
        return py::make_tuple(out_bands, sizes_vec);
    }, py::arg("img"), py::arg("levels"), py::arg("filt"));

    m.def("lpyr_recon", [](py::list bands_in, carray_t<float> filt) {
        auto f = filt.request();
        int fl = f.shape[0];
        int levels = py::len(bands_in);
        if (levels == 0) throw std::runtime_error("lpyr_recon: empty pyramid");

        // Read the bands + sizes from the python list.
        std::vector<std::pair<int, int>> sizes(levels);
        std::vector<float*> band_ptrs(levels);
        std::vector<float*> band_storage(levels);
        std::vector<carray_t<float>> bands_keep(levels);  // keep alive
        for (int l = 0; l < levels; ++l) {
            auto band = bands_in[l].cast<carray_t<float>>();
            bands_keep[l] = band;
            auto br = band.request();
            sizes[l] = {static_cast<int>(br.shape[0]),
                        static_cast<int>(br.shape[1])};
            band_storage[l] = device_alloc<float>(br.shape[0] * br.shape[1]);
            CUDA_CHECK(cudaMemcpy(band_storage[l], br.ptr,
                                  br.shape[0] * br.shape[1] * sizeof(float),
                                  cudaMemcpyHostToDevice));
            band_ptrs[l] = band_storage[l];
        }
        int H = sizes[0].first, W = sizes[0].second;
        auto* d_filt = device_alloc<float>(fl);
        CUDA_CHECK(cudaMemcpy(d_filt, f.ptr, fl * sizeof(float), cudaMemcpyHostToDevice));
        auto* d_out = device_alloc<float>(H * W);
        auto* scratch_lo = device_alloc<float>(H * W);
        auto* scratch_hi = device_alloc<float>(H * W);

        evm::lpyr_recon_device(const_cast<const float* const*>(band_ptrs.data()),
                               sizes.data(), levels,
                               d_filt, fl, d_out,
                               scratch_lo, scratch_hi, 0);

        carray_t<float> out({H, W});
        auto o = out.request();
        CUDA_CHECK(cudaMemcpy(o.ptr, d_out, H * W * sizeof(float), cudaMemcpyDeviceToHost));
        for (int l = 0; l < levels; ++l) device_free(band_storage[l]);
        device_free(d_filt); device_free(d_out);
        device_free(scratch_lo); device_free(scratch_hi);
        return out;
    }, py::arg("bands"), py::arg("filt"));

    // --- blur_dn (single channel, host-orchestrated level loop) ------------

    m.def("blur_dn", [](carray_t<float> img, int nlevs,
                        carray_t<float> filt) {
        auto b = img.request(); auto f = filt.request();
        if (b.ndim != 2) throw std::runtime_error("blur_dn: expected (H,W)");
        int H = b.shape[0], W = b.shape[1], fl = f.shape[0];
        // Compute final coarse size.
        int fh = H, fw = W;
        for (int l = 0; l < nlevs; ++l) { fh = (fh + 1) / 2; fw = (fw + 1) / 2; }

        auto* d_in  = device_alloc<float>(H * W);
        auto* d_out = device_alloc<float>(H * W);
        auto* d_filt = device_alloc<float>(fl);
        auto* scratch_a = device_alloc<float>(H * W);
        auto* scratch_b = device_alloc<float>(H * W);
        CUDA_CHECK(cudaMemcpy(d_in,  b.ptr, H * W * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_filt, f.ptr, fl * sizeof(float),   cudaMemcpyHostToDevice));

        evm::blur_dn_device(d_in, H, W, d_out, nlevs,
                            d_filt, fl, scratch_a, scratch_b, 0);

        carray_t<float> out({fh, fw});
        auto o = out.request();
        CUDA_CHECK(cudaMemcpy(o.ptr, d_out, fh * fw * sizeof(float), cudaMemcpyDeviceToHost));
        device_free(d_in); device_free(d_out); device_free(d_filt);
        device_free(scratch_a); device_free(scratch_b);
        return out;
    }, py::arg("img"), py::arg("nlevs"), py::arg("filt"));

    // Expose the constant filter arrays so Python tests can verify they match
    // the Python baseline's BINOM5 / BINOM5_SUM1 exactly.
    m.def("binom5",     []() { return std::vector<float>(evm::kBinom5, evm::kBinom5 + 5); });
    m.def("binom5_sum1", []() { return std::vector<float>(evm::kBinom5Sum1, evm::kBinom5Sum1 + 5); });
    m.attr("drop_last")            = evm::kDropLast;
    m.attr("exaggeration_factor")  = evm::kExaggerationFactor;

    // =====================================================================
    // Phase 1: device-resident API for batched (whole-clip) execution.
    //
    // The numpy wrappers above each do cudaMalloc + H2D + kernel + D2H +
    // cudaFree per call. The profiler (docs/profile_baseline.txt) showed
    // >95% of wall time is that overhead. The DeviceTensor below is a
    // GC-managed device buffer; the batched_* wrappers take device pointers
    // (as ints) and do NO host transfers. The whole clip stays on-device
    // from upload at pipeline entry to download at pipeline exit.
    // =====================================================================

    py::class_<DeviceBuffer>(m, "DeviceBuffer")
        .def(py::init([](size_t nbytes) {
            return std::make_unique<DeviceBuffer>(nbytes);
        }), py::arg("nbytes"))
        .def(py::init([](py::array arr) {
            // Treat the input as raw bytes — NO dtype cast. We must NOT use
            // py::array_t<char>::ensure() with forcecast: that would CAST each
            // element to char (1 byte), truncating float32 data (e.g. 0.5 -> 0).
            // Require C-contiguity (callers in our pipeline always satisfy this)
            // and copy raw bytes via the buffer protocol — no Python calls.
            if (!(arr.flags() & py::array::c_style))
                throw std::runtime_error(
                    "DeviceBuffer: input must be C-contiguous; call numpy.ascontiguousarray first");
            auto info = arr.request();
            size_t nbytes = arr.nbytes();
            auto db = std::make_unique<DeviceBuffer>(nbytes);
            CUDA_CHECK(cudaMemcpy(db->ptr, info.ptr, nbytes,
                                  cudaMemcpyHostToDevice));
            return db;
        }), py::arg("array"))
        .def("upload", [](DeviceBuffer& self, py::array arr) {
            // Same raw-bytes semantics as the constructor (see comment above).
            if (!(arr.flags() & py::array::c_style))
                throw std::runtime_error(
                    "DeviceBuffer.upload: input must be C-contiguous");
            auto info = arr.request();
            size_t nbytes = arr.nbytes();
            if (nbytes > self.nbytes)
                throw std::runtime_error("DeviceBuffer.upload: buffer too small");
            CUDA_CHECK(cudaMemcpy(self.ptr, info.ptr, nbytes, cudaMemcpyHostToDevice));
        }, py::arg("array"))
        .def("download_f32", [](DeviceBuffer& self, py::ssize_t count) {
            carray_t<float> out(count);
            auto o = out.request();
            size_t nbytes = count * sizeof(float);
            if (nbytes > self.nbytes)
                throw std::runtime_error("DeviceBuffer.download_f32: too much");
            CUDA_CHECK(cudaMemcpy(o.ptr, self.ptr, nbytes, cudaMemcpyDeviceToHost));
            return out;
        }, py::arg("count"))
        .def("download_u8", [](DeviceBuffer& self, py::ssize_t count) {
            carray_t<unsigned char> out(count);
            auto o = out.request();
            size_t nbytes = count;
            if (nbytes > self.nbytes)
                throw std::runtime_error("DeviceBuffer.download_u8: too much");
            CUDA_CHECK(cudaMemcpy(o.ptr, self.ptr, nbytes, cudaMemcpyDeviceToHost));
            return out;
        }, py::arg("count"))
        .def_readonly("nbytes", &DeviceBuffer::nbytes)
        .def_property_readonly("ptr", [](DeviceBuffer& self) {
            return reinterpret_cast<uintptr_t>(self.ptr);
        });

    // --- batched color_cvt: whole clip (T,H,W,3) at once -------------------
    // Per-pixel independent op; treat the clip as a (T*H, W, 3) image.
    m.def("batched_bgr_u8_to_ntsc_f32",
        [](uintptr_t d_in, uintptr_t d_out, int T, int H, int W) {
            evm::launch_bgr_u8_to_ntsc_f32(
                reinterpret_cast<unsigned char*>(d_in),
                reinterpret_cast<float*>(d_out),
                H * T, W, 0);
        }, py::arg("d_in"), py::arg("d_out"), py::arg("T"), py::arg("H"), py::arg("W"));

    // --- whole-clip planar layout transpose: (n,H,W,3) -> (n*3,H,W) --------
    // Bit-exact layout transform that lets the color pipeline operate on
    // contiguous per-frame-channel slices via pointer offsets.
    m.def("batched_to_planar_3ch",
        [](uintptr_t d_in, uintptr_t d_out, int n, int H, int W) {
            evm::launch_to_planar_3ch(
                reinterpret_cast<float*>(d_in),
                reinterpret_cast<float*>(d_out), n, H, W, 0);
        }, py::arg("d_in"), py::arg("d_out"), py::arg("n"),
           py::arg("H"), py::arg("W"));

    // --- batched blur_dn over M=n*3 contiguous planar slices ----------------
    // Processes all M slices simultaneously through the level loop using the
    // batched spatial kernels (grid.z = M). Collapses the old M-iteration host
    // loop into nlevs iterations, each with 2 batched kernel launches.
    //
    // Output is frame-major (M, hl, wl) — regular strides, no scatter needed.
    // Scratch: 2 M-sized buffers (ping-pong between downsample passes).
    m.def("batched_blur_dn_color",
        [](uintptr_t d_in, uintptr_t d_out, int M, int H, int W, int nlevs,
           uintptr_t d_filt, int filt_len) {
            const float* in_p  = reinterpret_cast<const float*>(d_in);
            float*       out_p = reinterpret_cast<float*>(d_out);
            const float* filt  = reinterpret_cast<const float*>(d_filt);

            // 2 frame-major scratch buffers, each M*H*W floats.
            float* scratch_a = device_alloc<float>(static_cast<size_t>(M) * H * W);
            float* scratch_b = device_alloc<float>(static_cast<size_t>(M) * H * W);

            // cur := input (M slices, batched D2D copy).
            CUDA_CHECK(cudaMemcpyAsync(scratch_a, in_p,
                static_cast<size_t>(M) * H * W * sizeof(float),
                cudaMemcpyDeviceToDevice, 0));

            float* cur = scratch_a;
            float* nxt = scratch_b;
            int ch = H, cw = W;
            for (int l = 0; l < nlevs; ++l) {
                int wn = (cw + 1) / 2;
                int hn = (ch + 1) / 2;
                // cols downsample: cur (M, ch, cw) -> nxt (M, ch, wn)
                evm::launch_corr_dn_cols_batched(
                    cur, nxt, ch, cw, filt, filt_len,
                    ch * cw, ch * wn, M, 0);
                // rows downsample: nxt (M, ch, wn) -> cur (M, hn, wn)
                evm::launch_corr_dn_rows_batched(
                    nxt, cur, ch, wn, filt, filt_len,
                    ch * wn, hn * wn, M, 0);
                ch = hn; cw = wn;
            }

            // Copy final result to output.
            CUDA_CHECK(cudaMemcpyAsync(out_p, cur,
                static_cast<size_t>(M) * ch * cw * sizeof(float),
                cudaMemcpyDeviceToDevice, 0));

            device_free(scratch_a); device_free(scratch_b);
        }, py::arg("d_in"), py::arg("d_out"), py::arg("M"),
           py::arg("H"), py::arg("W"), py::arg("nlevs"),
           py::arg("d_filt"), py::arg("filt_len"));

    // --- batched lpyr_build: M planar slices -> multi-level band output -----
    // Processes all M=n_frames*3 slices simultaneously through the level loop.
    // At each level, the batched spatial kernels process all M slices in one
    // launch (grid.z = M). Band writes go through scatter_subtract (channel-
    // major output layout has irregular per-slice offsets).
    //
    // This collapses the old M-iteration host loop (~35k launches) into ~40
    // launches total. Scratch is M-sized (4 buffers, each M*H*W floats).
    m.def("batched_lpyr_build",
        [](uintptr_t d_in, uintptr_t d_out, int n_frames, int H, int W, int levels,
           uintptr_t d_filt, int filt_len) {
            int M = n_frames * 3;
            auto sizes = evm::lpyr_level_sizes(H, W, levels);
            const float* filt = reinterpret_cast<const float*>(d_filt);
            const float* in_base = reinterpret_cast<const float*>(d_in);
            float* out_base = reinterpret_cast<float*>(d_out);

            // Per-level offset table (host-side, for pre-computing scatter offsets).
            std::vector<size_t> level_offsets(levels), level_sizes_vec(levels);
            size_t total = 0;
            for (int l = 0; l < levels; ++l) {
                level_sizes_vec[l] = static_cast<size_t>(sizes[l].first) * sizes[l].second;
                level_offsets[l] = total;
                total += level_sizes_vec[l] * M;
            }

            // 4 frame-major scratch buffers, each M*H*W floats.
            // cur = current image, lo = corr_dn(cols), lo2 = corr_dn(rows),
            // hi2 = upsample(lo2) back to current size (needed alive for subtract).
            float* scratch_cur = device_alloc<float>(static_cast<size_t>(M) * H * W);
            float* scratch_lo  = device_alloc<float>(static_cast<size_t>(M) * H * W);
            float* scratch_lo2 = device_alloc<float>(static_cast<size_t>(M) * H * W);
            float* scratch_hi2 = device_alloc<float>(static_cast<size_t>(M) * H * W);
            // Device-side per-slice offset table for scatter (reused per level).
            int* d_offsets = device_alloc<int>(M);

            // cur := input (M slices, batched D2D copy).
            CUDA_CHECK(cudaMemcpyAsync(scratch_cur, in_base,
                                       static_cast<size_t>(M) * H * W * sizeof(float),
                                       cudaMemcpyDeviceToDevice, 0));

            for (int l = 0; l < levels - 1; ++l) {
                const int h = sizes[l].first, w = sizes[l].second;
                const int hn = (h + 1) / 2;
                const int wn = (w + 1) / 2;

                // lo  = corr_dn(cur, axis=1)  -> (M, h, wn)
                evm::launch_corr_dn_cols_batched(
                    scratch_cur, scratch_lo, h, w, filt, filt_len,
                    h * w, h * wn, M, 0);
                // lo2 = corr_dn(lo,  axis=0)  -> (M, hn, wn)
                evm::launch_corr_dn_rows_batched(
                    scratch_lo, scratch_lo2, h, wn, filt, filt_len,
                    h * wn, hn * wn, M, 0);
                // hi  = up_conv(lo2, axis=0, out_size=h) -> (M, h, wn)
                //      reuse scratch_lo (lo no longer needed)
                evm::launch_up_conv_rows_batched(
                    scratch_lo2, scratch_lo, hn, h, wn, filt, filt_len,
                    hn * wn, h * wn, M, 0);
                // hi2 = up_conv(hi,  axis=1, out_size=w) -> (M, h, w)
                evm::launch_up_conv_cols_batched(
                    scratch_lo, scratch_hi2, h, wn, w, filt, filt_len,
                    h * wn, h * w, M, 0);

                // band[l] = cur - hi2, scattered into channel-major storage.
                // Build per-slice offset table for this level and upload.
                std::vector<int> h_offsets(M);
                for (int m = 0; m < M; ++m) {
                    int frame = m / 3, chan = m % 3;
                    size_t slice_off = static_cast<size_t>(chan) * n_frames + frame;
                    h_offsets[m] = static_cast<int>(
                        level_offsets[l] + slice_off * level_sizes_vec[l]);
                }
                CUDA_CHECK(cudaMemcpy(d_offsets, h_offsets.data(),
                                      M * sizeof(int), cudaMemcpyHostToDevice));
                evm::launch_scatter_subtract(
                    scratch_cur, scratch_hi2, out_base,
                    d_offsets, h * w, M, 0);

                // Descend: cur := lo2 (next level's input).
                CUDA_CHECK(cudaMemcpyAsync(scratch_cur, scratch_lo2,
                    static_cast<size_t>(M) * hn * wn * sizeof(float),
                    cudaMemcpyDeviceToDevice, 0));
            }

            // Coarsest level (l = levels-1): residual lowpass = cur.
            {
                int l = levels - 1;
                const int h = sizes[l].first, w = sizes[l].second;
                std::vector<int> h_offsets(M);
                for (int m = 0; m < M; ++m) {
                    int frame = m / 3, chan = m % 3;
                    size_t slice_off = static_cast<size_t>(chan) * n_frames + frame;
                    h_offsets[m] = static_cast<int>(
                        level_offsets[l] + slice_off * level_sizes_vec[l]);
                }
                CUDA_CHECK(cudaMemcpy(d_offsets, h_offsets.data(),
                                      M * sizeof(int), cudaMemcpyHostToDevice));
                evm::launch_scatter(
                    scratch_cur, out_base, d_offsets, h * w, M, 0);
            }

            device_free(scratch_cur); device_free(scratch_lo);
            device_free(scratch_lo2); device_free(scratch_hi2);
            device_free(d_offsets);
        }, py::arg("d_in"), py::arg("d_out"), py::arg("n_frames"),
           py::arg("H"), py::arg("W"), py::arg("levels"),
           py::arg("d_filt"), py::arg("filt_len"));

    // --- batched lpyr_recon: multi-level band input -> M planar slices -----
    // Mirror of batched_lpyr_build for the motion pipeline's Stage D. Walks
    // coarsest→finest, batching all M slices per spatial kernel launch. Band
    // reads go through gather_add (channel-major input → frame-major output).
    m.def("batched_lpyr_recon",
        [](uintptr_t d_bands, uintptr_t d_out, int n_frames, int H, int W, int levels,
           uintptr_t d_filt, int filt_len) {
            int M = n_frames * 3;
            auto sizes = evm::lpyr_level_sizes(H, W, levels);
            const float* filt = reinterpret_cast<const float*>(d_filt);
            const float* bands_base = reinterpret_cast<const float*>(d_bands);
            float* out_base = reinterpret_cast<float*>(d_out);

            std::vector<size_t> level_offsets(levels), level_sizes_vec(levels);
            size_t total = 0;
            for (int l = 0; l < levels; ++l) {
                level_sizes_vec[l] = static_cast<size_t>(sizes[l].first) * sizes[l].second;
                level_offsets[l] = total;
                total += level_sizes_vec[l] * M;
            }

            // 2 frame-major scratch buffers: cur (current reconstruction level)
            // and res (up_conv output). Plus device-side offset table.
            float* scratch_cur = device_alloc<float>(static_cast<size_t>(M) * H * W);
            float* scratch_res = device_alloc<float>(static_cast<size_t>(M) * H * W);
            int* d_offsets = device_alloc<int>(M);

            // Start: gather coarsest band (l=levels-1) into frame-major scratch_cur.
            {
                int l = levels - 1;
                const int h = sizes[l].first, w = sizes[l].second;
                std::vector<int> h_offsets(M);
                for (int m = 0; m < M; ++m) {
                    int frame = m / 3, chan = m % 3;
                    size_t slice_off = static_cast<size_t>(chan) * n_frames + frame;
                    h_offsets[m] = static_cast<int>(
                        level_offsets[l] + slice_off * level_sizes_vec[l]);
                }
                CUDA_CHECK(cudaMemcpy(d_offsets, h_offsets.data(),
                                      M * sizeof(int), cudaMemcpyHostToDevice));
                evm::launch_gather(
                    bands_base, scratch_cur, d_offsets, h * w, M, 0);
            }

            // Walk coarsest→finest (l = levels-2 down to 0).
            for (int l = levels - 2; l >= 0; --l) {
                const int h = sizes[l].first, w = sizes[l].second;
                const int ph = sizes[l + 1].first, pw = sizes[l + 1].second;

                // res = up_conv(cur, axis=0, out_size=h) -> (M, h, pw)
                evm::launch_up_conv_rows_batched(
                    scratch_cur, scratch_res, ph, h, pw, filt, filt_len,
                    ph * pw, h * pw, M, 0);
                // out = up_conv(res, axis=1, out_size=w) -> (M, h, w)
                //      write into scratch_cur (reuse, cur no longer needed)
                evm::launch_up_conv_cols_batched(
                    scratch_res, scratch_cur, h, pw, w, filt, filt_len,
                    h * pw, h * w, M, 0);

                // out = band[l] + res. Gather band[l] (scattered) + scratch_cur,
                // write back into scratch_cur via gather_add.
                std::vector<int> h_offsets(M);
                for (int m = 0; m < M; ++m) {
                    int frame = m / 3, chan = m % 3;
                    size_t slice_off = static_cast<size_t>(chan) * n_frames + frame;
                    h_offsets[m] = static_cast<int>(
                        level_offsets[l] + slice_off * level_sizes_vec[l]);
                }
                CUDA_CHECK(cudaMemcpy(d_offsets, h_offsets.data(),
                                      M * sizeof(int), cudaMemcpyHostToDevice));
                // gather_add: dst[di] = src[offsets[m]+px] + b[di]
                // src = bands_base, b = scratch_cur (res), dst = scratch_cur (in-place)
                evm::launch_gather_add(
                    bands_base, scratch_cur, scratch_cur,
                    d_offsets, h * w, M, 0);
            }

            // Copy final reconstruction from scratch_cur to output.
            CUDA_CHECK(cudaMemcpyAsync(out_base, scratch_cur,
                static_cast<size_t>(M) * H * W * sizeof(float),
                cudaMemcpyDeviceToDevice, 0));

            device_free(scratch_cur); device_free(scratch_res);
            device_free(d_offsets);
        }, py::arg("d_bands"), py::arg("d_out"), py::arg("n_frames"),
           py::arg("H"), py::arg("W"), py::arg("levels"),
           py::arg("d_filt"), py::arg("filt_len"));

    // --- batched spatial primitives: corr_dn / up_conv on device pointers ---

    // Transpose (T,N) <-> (N,T) on device pointers. Used by the motion
    // pipeline's device-resident Stage C to rearrange pyramid band data for
    // the temporal IIR filter. Takes raw uintptr_t so the caller can offset.
    m.def("batched_thwc_to_nt",
        [](uintptr_t d_in, uintptr_t d_out, int T, int N) {
            evm::launch_thwc_to_nt(
                reinterpret_cast<const float*>(d_in),
                reinterpret_cast<float*>(d_out), T, N, 0);
        }, py::arg("d_in"), py::arg("d_out"), py::arg("T"), py::arg("N"));

    // Scaled transpose: folds a per-call scalar multiply into the (N,T)->(T,N)
    // transpose. Used by the motion pipeline's Stage C to apply per-level alpha
    // amplification to IIR-filtered bands without a separate scale_inplace pass.
    m.def("batched_nt_to_thwc_scaled",
        [](uintptr_t d_in, uintptr_t d_out, int T, int N, float scale) {
            evm::launch_nt_to_thwc_scaled(
                reinterpret_cast<const float*>(d_in),
                reinterpret_cast<float*>(d_out), T, N, scale, 0);
        }, py::arg("d_in"), py::arg("d_out"), py::arg("T"),
           py::arg("N"), py::arg("scale"));

    // --- batched temporal filters on device pointers -----------------------
    // These take (N, T) row-major float32 on device, filter in-place semantics
    // (input and output may be different buffers).
    m.def("batched_iir_bandpass",
        [](uintptr_t d_in, uintptr_t d_out, int T, int N, double r1, double r2) {
            evm::launch_iir_bandpass(
                reinterpret_cast<float*>(d_in), reinterpret_cast<float*>(d_out),
                T, N, r1, r2, 0);
        }, py::arg("d_in"), py::arg("d_out"), py::arg("T"), py::arg("N"),
           py::arg("r1"), py::arg("r2"));

    // Fused planar-delta add + quantize (motion pipeline render). Reads delta
    // from planar (n*3,H,W) layout directly, folding the planar->interleaved
    // transpose inline. Eliminates the intermediate interleaved buffer + one
    // full-res kernel pass.
    m.def("batched_add_planar_quantize",
        [](uintptr_t d_ntsc, uintptr_t d_delta_planar, uintptr_t d_bgr,
           int n, int H, int W, float chrom_att) {
            evm::launch_add_planar_quantize(
                reinterpret_cast<float*>(d_ntsc),
                reinterpret_cast<float*>(d_delta_planar),
                reinterpret_cast<unsigned char*>(d_bgr),
                n, H, W, chrom_att, 0);
        }, py::arg("d_ntsc"), py::arg("d_delta_planar"), py::arg("d_bgr"),
           py::arg("n"), py::arg("H"), py::arg("W"), py::arg("chrom_att"));

    // --- batched bilinear upsample: M frames (in_H,in_W,3) -> (out_H,out_W,3) -
    // Replaces host-side cv2.resize(INTER_LINEAR) in the color pipeline render
    // stage. Coordinate convention: half-pixel centers + replicate border,
    // reverse-engineered to match cv2 bit-exactly.
    m.def("batched_bilinear_upsample_3ch",
        [](uintptr_t d_in, uintptr_t d_out, int M,
           int in_H, int in_W, int out_H, int out_W) {
            evm::launch_bilinear_upsample_3ch(
                reinterpret_cast<float*>(d_in),
                reinterpret_cast<float*>(d_out),
                M, in_H, in_W, out_H, out_W, 0);
        }, py::arg("d_in"), py::arg("d_out"), py::arg("M"),
           py::arg("in_H"), py::arg("in_W"), py::arg("out_H"), py::arg("out_W"));

    // --- batched fused upsample+add+quantize (color pipeline render) -------
    // Combines bilinear_upsample_3ch + add_and_quantize into one kernel that
    // reads the small filtered signal + the full-res NTSC frame, interpolates
    // inline, adds, and writes the uint8 output. Eliminates the M*out_H*out_W*3
    // float32 intermediate buffer and one kernel launch.
    m.def("batched_upsample_add_quantize",
        [](uintptr_t d_ntsc, uintptr_t d_filt, uintptr_t d_bgr,
           int M, int in_H, int in_W, int out_H, int out_W, float chrom_att) {
            evm::launch_upsample_add_quantize(
                reinterpret_cast<float*>(d_ntsc),
                reinterpret_cast<float*>(d_filt),
                reinterpret_cast<unsigned char*>(d_bgr),
                M, in_H, in_W, out_H, out_W, chrom_att, 0);
        }, py::arg("d_ntsc"), py::arg("d_filt"), py::arg("d_bgr"),
           py::arg("M"), py::arg("in_H"), py::arg("in_W"),
           py::arg("out_H"), py::arg("out_W"), py::arg("chrom_att"));

    // --- batched ideal_bandpass: needs cuFFT plans, so orchestrate here -----
    // Plans are cached by (T, N) — cuFFT plan creation does internal autotuning
    // (~5-10ms each on H200), and this is called 3x per pipeline (once per
    // channel) with identical (T, N). First call pays the autotuning cost;
    // subsequent calls (same clip, or repeated pipeline runs) skip it.
    m.def("batched_ideal_bandpass",
        [](uintptr_t d_in, uintptr_t d_out, int T, int N,
           float wl, float wh, float sampling_rate) {
            auto* d_tmp = device_alloc<cufftComplex>(static_cast<size_t>(N) * T);
            FftPlanPair plans = get_or_create_fft_plans(T, N);
            evm::launch_ideal_bandpass(
                reinterpret_cast<float*>(d_in), reinterpret_cast<float*>(d_out),
                d_tmp, T, N, wl, wh, sampling_rate, plans.fwd, plans.inv, 0);
            device_free(d_tmp);
        }, py::arg("d_in"), py::arg("d_out"), py::arg("T"), py::arg("N"),
           py::arg("wl"), py::arg("wh"), py::arg("sampling_rate"));

    // Warm up the CUDA driver's memory pool. The first large cudaMalloc
    // (~100MB+) takes ~1s because the driver lazily sets up page tables. A
    // quick alloc+free of `nbytes` primes the pool so all subsequent
    // allocations are O(1). Call once at pipeline entry.
    m.def("warmup_device_pool", [](size_t nbytes) {
        void* p = nullptr;
        CUDA_CHECK(cudaMalloc(&p, nbytes));
        CUDA_CHECK(cudaFree(p));
    }, py::arg("nbytes"));

    // Block the host until all outstanding async work on the default stream
    // completes. The batched_* wrappers are all fire-and-forget (they return
    // immediately after queueing kernels on stream 0); without an explicit
    // sync, perf_counter() wall-clock measurements only capture host overhead
    // and the actual GPU compute piles up at the next blocking D2H memcpy.
    // The profilers (scripts/profile_*.py) call this between every stage so
    // the per-stage breakdown reflects real GPU time, not host queue time.
    m.def("device_synchronize", []() {
        CUDA_CHECK(cudaDeviceSynchronize());
    });

    // FP16↔FP32 batch conversion for the FP16 storage path.
    // Converts n elements between float32 and __half. Used to halve VRAM for
    // intermediate buffers (NTSC, bands, scratch) while keeping compute in FP32.
    m.def("f32_to_f16", [](uintptr_t d_src, uintptr_t d_dst, int n) {
        evm::launch_f32_to_f16(
            reinterpret_cast<const float*>(d_src),
            reinterpret_cast<__half*>(d_dst), n, 0);
    }, py::arg("d_src"), py::arg("d_dst"), py::arg("n"));

    m.def("f16_to_f32", [](uintptr_t d_src, uintptr_t d_dst, int n) {
        evm::launch_f16_to_f32(
            reinterpret_cast<const __half*>(d_src),
            reinterpret_cast<float*>(d_dst), n, 0);
    }, py::arg("d_src"), py::arg("d_dst"), py::arg("n"));

    // Upload the binom5 filters lazily (on first call, not at module import).
    // Allocating at import time runs cudaMalloc before any explicit device
    // context setup, which can segfault on some systems.
    m.def("d_binom5_ptr", []() -> uintptr_t {
        static float* p = nullptr;
        if (!p) {
            p = device_alloc<float>(5);
            CUDA_CHECK(cudaMemcpy(p, evm::kBinom5, 5 * sizeof(float),
                                  cudaMemcpyHostToDevice));
        }
        return reinterpret_cast<uintptr_t>(p);
    });
    m.def("d_binom5_sum1_ptr", []() -> uintptr_t {
        static float* p = nullptr;
        if (!p) {
            p = device_alloc<float>(5);
            CUDA_CHECK(cudaMemcpy(p, evm::kBinom5Sum1, 5 * sizeof(float),
                                  cudaMemcpyHostToDevice));
        }
        return reinterpret_cast<uintptr_t>(p);
    });
}
