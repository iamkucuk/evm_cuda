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
void launch_add_and_quantize(const float* ntsc_frame, const float* delta,
                             unsigned char* bgr_out, int H, int W,
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
        evm::launch_add_and_quantize(d_f, d_d, d_o, H, W, 0);
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

    m.def("lpyr_level_sizes", [](int H, int W, int levels) {
        auto sizes = evm::lpyr_level_sizes(H, W, levels);
        py::list out;
        for (auto& s : sizes) out.append(py::make_tuple(s.first, s.second));
        return out;
    }, py::arg("H"), py::arg("W"), py::arg("levels"));

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
}
