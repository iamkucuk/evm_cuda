# CUDA EVM Development Diary

**Project:** Eulerian Video Magnification (EVM) – CUDA Acceleration  
**Maintainer:** AI Agent (Cascade)  
**Last Updated:** 2025-04-17T20:28:24+03:00

---

## 1. Objective

Accelerate the EVM pipeline (Gaussian and Laplacian pathways) using CUDA, so the entire video processing can run efficiently on the GPU. The goal is to match the output of the existing C++/OpenCV CPU implementation, and to provide clear, maintainable CUDA code that can be extended and debugged as needed.

---

## 2. Current Status

### Done

- **CUDA Gaussian Blur Kernel:**  
  - Implemented and tested for 3-channel float32 images.  
  - Output numerically matches CPU (OpenCV/naive) implementation.

- **CUDA Downsampling (pyrDown) and Upsampling (pyrUp):**  
  - Implemented and tested (including zero-insert, blur with kernel * 4).  
  - Outputs match CPU reference within 1e-7 tolerance.  
  - Test suite covers all three operations.

- **Test Infrastructure:**  
  - Standalone CUDA test executable ([test_gaussian_blur.cu](../tests/test_gaussian_blur.cu)) with random image generation and CPU/CUDA comparison.  
  - Build and run instructions documented.

### In Progress

- **Integration into EVM Pipeline:**  
  - The main pipeline ([cpp/src/main.cpp](../../cpp/src/main.cpp), `processVideoGaussianBatch`, etc.) still uses CPU-based pyramid construction.  
  - CUDA kernels are not yet called from the main C++ pipeline.

---

## 3. Next Steps

### 3.1. Immediate TODOs

1. **Wrap CUDA Kernels for C++ Integration**
   - Expose CUDA pyramid functions (`cudaSpatiallyFilterGaussian`, `cudaPyrDown`, `cudaPyrUp`) via C++ wrappers (extern "C" or C++/CUDA interop).
   - Ensure memory layout and data transfer (cv::Mat ↔ float*) is handled correctly.

2. **Replace CPU Pyramid Construction with CUDA**
   - In Gaussian and Laplacian pipeline code (e.g., `spatiallyFilterGaussian`, `pyrDown`, `pyrUp` in `gaussian_pyramid.cpp`/`laplacian_pyramid.cpp`), add CUDA-accelerated code paths.
   - Add a config flag or auto-detect CUDA availability for fallback.

3. **Test Full Pipeline with CUDA**
   - Run the full EVM pipeline on a test video, compare output frames (CPU vs. CUDA).
   - Profile speedup and check for correctness (visual and numerical).

4. **Error Handling and Logging**
   - Ensure CUDA errors are caught and reported in the main pipeline.
   - Add logging for GPU/CPU fallback.

### 3.2. Longer-Term TODOs

- **Full GPU Memory Pipeline:**  
  - Minimize host-device transfers by keeping video frames and pyramids on the GPU as much as possible.
  - Implement batch processing for multiple frames if memory allows.

- **Laplacian Pyramid CUDA Implementation:**  
  - Implement CUDA versions of Laplacian pyramid construction and reconstruction (if not already done).

- **Butterworth Filter on GPU:**  
  - Port temporal filtering (Butterworth) to CUDA if it becomes a bottleneck.

- **Color Conversion on GPU:**  
  - Port RGB↔YIQ conversion to CUDA if needed.

---

## 4. Current Issues

- **Integration Layer:**  
  - Need to ensure seamless data transfer between OpenCV (cv::Mat) and CUDA (float*).
  - Need to decide on memory management strategy (who allocates/frees GPU memory).

- **Testing at Pipeline Level:**  
  - Need to validate not just individual kernels, but the whole video pipeline (frame-by-frame).

- **Fallback/Compatibility:**  
  - Ensure the code gracefully falls back to CPU if CUDA is unavailable.

---

## 5. Instructions for Next Agent/Developer

1. **Start by reviewing the CUDA kernels in [cuda/src/gaussian_pyramid_cuda.cu](src/gaussian_pyramid_cuda.cu) and their headers.**
2. **Implement C++ wrappers (extern "C" or direct CUDA interop) if not already present.**
3. **In the main pipeline ([cpp/src/main.cpp](../../cpp/src/main.cpp), `gaussian_pyramid.cpp`), replace CPU pyramid calls with CUDA versions.**
4. **Test with [cuda/tests/test_gaussian_blur.cu](../tests/test_gaussian_blur.cu) and then with a full video using the main pipeline.**
5. **Document any new dependencies, build steps, or caveats in the README.**
6. **Update this diary with every significant change, including issues found, design decisions, and next steps.**

---

## 6. Example Build/Run Instructions

- **Build CUDA tests:**  
  `cd cuda && nvcc -o test_gaussian_blur tests/test_gaussian_blur.cu src/gaussian_pyramid_cuda.cu -Iinclude -lstdc++ -lcudart`

- **Run CUDA tests:**  
  `./test_gaussian_blur`

- **Build main pipeline (example):**  
  *(Update as needed for CUDA integration)*

---

## 7. References

- `cuda/include/gaussian_pyramid_cuda.hpp`
- [cuda/src/gaussian_pyramid_cuda.cu](src/gaussian_pyramid_cuda.cu)
- [cuda/tests/test_gaussian_blur.cu](../tests/test_gaussian_blur.cu)
- `cpp/src/gaussian_pyramid.cpp`
- [cpp/src/main.cpp](../../cpp/src/main.cpp)

---

## 8. Environment and Platform Notes

- **OpenCV CUDA Module:**
  - OpenCV's CUDA-accelerated functions (e.g., `cv::cuda::GpuMat`, `cv::cuda::pyrDown`, etc.) are **not available in the default local environment**.
  - These can only be used inside a Docker container that has OpenCV compiled with CUDA support. If you need to use OpenCV's CUDA features, refer to the project's Dockerfile and run the pipeline inside Docker.
  - **Current CUDA EVM implementation is independent of OpenCV CUDA** and uses custom CUDA kernels for Gaussian blur, downsampling, and upsampling.

- **General CUDA Requirements:**
  - CUDA toolkit and compatible GPU drivers must be installed on the host or inside Docker.
  - All custom CUDA code is located in the `cuda/` directory and is tested independently of OpenCV's CUDA modules.

---

## 9. Optional Enhancements and FAQ

### 9.1. Directory Structure Summary

```
/evm
  |-- cpp/                 # Main C++ pipeline (CPU reference)
  |-- cuda/                # Custom CUDA kernels and tests
  |-- docker/              # Docker setup for CUDA/OpenCV if needed
  |-- ...
```

### 9.2. Example Input/Output
- **Input:** Video file (e.g., `.mp4`), processed frame-by-frame.
- **Output:** Magnified video (same format), written to disk.

### 9.3. Dependencies
- CUDA Toolkit (version as per Dockerfile or system)
- OpenCV (CPU version required; CUDA version optional and only via Docker)
- C++17 or newer

### 9.4. Common Pitfalls
- Forgetting to convert `cv::Mat` data to float32 planar/interleaved format before passing to CUDA kernels.
- Not freeing GPU memory after use (memory leaks).
- Mismatched image dimensions after pyramid operations.
- Trying to use OpenCV CUDA modules outside Docker (will fail).

### 9.5. FAQ
- **Q: Can I use OpenCV's CUDA functions directly?**
  - A: Not unless you are inside the Docker container with OpenCV built with CUDA support. All custom CUDA kernels are designed to work independently of OpenCV's CUDA modules.
- **Q: How do I run the pipeline on GPU?**
  - A: Integrate the custom CUDA kernels into the main pipeline as described above, and ensure you are running on a CUDA-capable machine (or Docker with GPU support).

---

**Continue updating this diary as you progress.**  
If you encounter blockers, document them here and suggest possible solutions.
