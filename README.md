# Eulerian Video Magnification (EVM) - C++ and Python Implementation

This repository contains implementations of the Eulerian Video Magnification (EVM) technique, primarily focusing on a C++ version (`evmcpp`) with a reference Python implementation (`evmpy`). EVM allows for the visualization of subtle temporal variations (motion or color changes) in standard video sequences.

## EVM Paper Explanation

The core technique is based on the SIGGRAPH 2012 paper:

**Eulerian Video Magnification for Revealing Subtle Changes in the World**
*Hao-Yu Wu, Michael Rubinstein, Eugene Shih, John Guttag, Frédo Durand, William Freeman*
*MIT CSAIL, Quanta Research Cambridge Inc.*

[Link to Paper PDF](http://people.csail.mit.edu/mrub/papers/vidmag.pdf) | [Link to Project Page](http://people.csail.mit.edu/mrub/vidmag/)

### Core Idea

Instead of tracking features (Lagrangian approach), EVM operates in an Eulerian framework, analyzing temporal changes in pixel values at *fixed spatial locations*. By filtering these temporal signals within a frequency band of interest and amplifying them, subtle variations like small motions or physiological changes (e.g., pulse) become visible.

### Methodology Overview

The general pipeline involves:

1.  **Spatial Decomposition (Optional but Recommended):**
    *   Input video frames are decomposed into spatial frequency bands using a Laplacian or Gaussian pyramid.
    *   *Why?* This improves Signal-to-Noise Ratio (SNR) by averaging noise at coarser levels and allows applying different amplification factors to different spatial frequencies, mitigating artifacts, especially for motion magnification.

2.  **Temporal Filtering:**
    *   The time series for each pixel (within each spatial band) is filtered temporally using a bandpass filter (e.g., Butterworth, Ideal FFT-based).
    *   This isolates the specific frequencies corresponding to the subtle variation of interest (e.g., 0.8-1.0 Hz for pulse). Let `I(x, y, t)` be the pixel intensity and `F` the filter; the filtered signal is `B(x, y, t) = F(I(x, y, t))`.

3.  **Amplification:**
    *   The filtered signal `B(x, y, t)` is multiplied by an amplification factor `α`.
    *   `MagnifiedSignal(x, y, t) = α * B(x, y, t)`
    *   `α` controls the magnification strength.

4.  **Reconstruction:**
    *   The amplified signal is added back to the original signal (for that band).
    *   `OutputBand(x, y, t) = I(x, y, t) + MagnifiedSignal(x, y, t)`

5.  **Spatial Collapse (If Decomposed):**
    *   The modified spatial bands are recombined (e.g., summing Laplacian pyramid levels) to synthesize the final output video frame.

### Mathematical Foundation: Motion Magnification

EVM approximates motion magnification using a first-order Taylor series expansion. For a 1D signal `I(x, t) = f(x + δ(t))` undergoing motion `δ(t)`, the goal is `Î_target(x, t) = f(x + (1 + α)δ(t))`.

The EVM process `Î(x, t) = I(x, t) + α * B(x, t)` (where `B(x, t)` is the temporally filtered signal) approximates this target:

`Î(x, t) ≈ f(x) + (1 + α)δ(t) * (∂f(x) / ∂x)`

This matches the Taylor expansion of the target `Î_target(x, t)`, provided the approximation holds for both `δ(t)` and the amplified motion `(1 + α)δ(t)`.

### Limits & Motion Magnification Bounds

The Taylor approximation breaks down if the amplified displacement `(1 + α)δ(t)` is large relative to the spatial wavelength `λ` of the image features. The paper derives a bound:

`(1 + α) * |δ(t)| < λ / 8`

*   **Implication:** High spatial frequencies (small `λ`, like edges) tolerate less amplification `α` before artifacts appear.
*   **Solution:** Use spatial pyramids. Coarser levels (large `λ`) tolerate higher `α`. Finer levels (small `λ`) require `α` to be attenuated or zeroed, especially if the overall `α` or motion `δ(t)` is large. A spatial cutoff wavelength `λ_c` is often used, below which `α` is reduced.

### Color Amplification

The same pipeline applies to color changes (e.g., pulse detection):

1.  **Spatial Pooling:** Low-pass filtering (e.g., Gaussian pyramid, coarser Laplacian levels) is *crucial* to average sensor noise and improve SNR for subtle color signals.
2.  **Color Space:** Processing in YIQ or YCbCr allows selective amplification of luminance (Y) vs. chrominance (IQ/CbCr), potentially using a `chrom_attenuation` factor to prevent unrealistic colors.
3.  **Temporal Filtering:** Isolate the frequency band of interest (e.g., heart rate).
4.  **Amplification & Reconstruction:** As above.

### Key Implementation Parameters

*   **Spatial Representation:** Laplacian pyramid, Gaussian pyramid, or simple filtering.
*   **Temporal Filter:** Ideal (FFT), FIR, IIR (Butterworth). Defined by cutoff frequencies (`fl`, `fh`).
*   **Amplification Factor (`α`):** Main gain.
*   **Spatial Cutoff Wavelength (`λ_c`):** (Motion) Wavelength below which `α` is attenuated.
*   **Attenuation Factor:** How `α` is reduced for high frequencies.
*   **Color Space:** RGB, YIQ, YCbCr.
*   **Chrominance Attenuation:** Factor to reduce amplification of color channels relative to luminance.

## Python Implementation (`evmpy`)

The `evmpy` directory contains a reference implementation written in Python using libraries like NumPy and OpenCV.

*   **Purpose:** Serves as a baseline and for generating test data.
*   **Structure:**
    *   `src/`: Contains the core logic.
        *   `evm.py`: Main script orchestrating the EVM process.
        *   `laplacian_pyramid.py`: Laplacian pyramid construction.
        *   `gaussian_pyramid.py`: Gaussian pyramid construction.
        *   `processing.py`: Filtering, amplification, reconstruction logic.
    *   `data/`: Sample input videos.
    *   `generate_test_data.py`: Script used to create intermediate data files for verifying the C++ implementation.

## C++ Implementation (`evmcpp`)

The `evmcpp` directory contains a C++ implementation of both Laplacian and Gaussian EVM pathways, aiming for better performance than the Python reference. It is built using CMake and OpenCV.

### Architecture

*   **`evm_core` (Static Library):** Encapsulates the core EVM algorithms (pyramids, filtering, processing). Defined in `evmcpp/CMakeLists.txt`.
    *   Headers: `include/evmcpp/`
    *   Sources: `src/evmcpp/`
*   **`evm_app` (Executable):** A command-line application that uses `evm_core` to process videos.
    *   Source: `src/main.cpp`

### Key Components

*   **Laplacian Pathway (`laplacian_pyramid.cpp/.hpp`):**
    *   Implements Laplacian pyramid construction (`generateLaplacianPyramid`) using standard OpenCV functions `cv::pyrDown` and `cv::pyrUp` for spatial decomposition.
    *   Temporal filtering (`filterLaplacianPyramids`) is performed per-level, per-frame using a 1st-order IIR Butterworth filter. The filter coefficients (b, a) are calculated in `butterworth.cpp` (based on analog prototype and bilinear transform) and applied directly in the filtering loop.
    *   Spatial attenuation is applied during temporal filtering based on the pyramid level and `lambda_cutoff`.
    *   Reconstruction (`reconstructLaplacianImage`) involves upsampling the filtered levels (using `cv::pyrUp`) and adding them back to the original YIQ image before converting to RGB.

*   **Gaussian Pathway (`gaussian_pyramid.cpp/.hpp`):**
    *   Implements spatial filtering (`spatiallyFilterGaussian`) by converting the input RGB frame to YIQ, then repeatedly applying the custom `evmcpp::pyrDown` and `evmcpp::pyrUp` functions (defined in `processing.cpp`) for the specified number of levels. This produces a spatially blurred YIQ frame.
    *   Temporal filtering and amplification (`temporalFilterGaussianBatch`) operates on a batch of these spatially filtered YIQ frames. It uses an ideal bandpass filter implemented via FFT for each pixel's time series:
        *   `cv::dft` computes the Discrete Fourier Transform along the time axis.
        *   A frequency mask based on `fl` and `fh` selects the desired band.
        *   `cv::idft` computes the Inverse Discrete Fourier Transform.
        *   The resulting filtered signal is amplified by `alpha`, with chrominance channels attenuated by `chromAttenuation`.
    *   Frame reconstruction (`reconstructGaussianFrame`) takes the original RGB frame and the corresponding filtered/amplified YIQ signal. It converts the original frame to YIQ, adds the filtered signal, converts the result back to RGB (float), clips values to [0, 255], and converts to the final `uint8` format.

*   **Temporal Filtering (`butterworth.cpp/.hpp`, `gaussian_pyramid.cpp`):**
    *   **Laplacian:** Uses a time-domain IIR Butterworth filter implemented in `laplacian_pyramid.cpp`. Coefficients are calculated in `butterworth.cpp` based on the desired frequency band (`fl`, `fh`) and video `fps`.
    *   **Gaussian:** Uses a frequency-domain ideal bandpass filter implemented within `temporalFilterGaussianBatch` (in `gaussian_pyramid.cpp`) using `cv::dft` and `cv::idft`. The filter directly selects frequencies between `fl` and `fh`.

*   **Shared Processing (`processing.cpp/.hpp`):**
    *   Contains common functions for RGB <-> YIQ color space conversions (`rgb2yiq`, `yiq2rgb`) using `cv::transform` with standard conversion matrices.
    *   Defines custom `pyrDown` and `pyrUp` functions that mimic Python's behavior using `cv::filter2D` with a Gaussian kernel and explicit down/upsampling logic. These are used by the Gaussian pathway's `spatiallyFilterGaussian` function. (Note: The C++ Laplacian pathway implementation was updated to also use these custom functions for consistency, although the original README stated it used OpenCV's built-ins).

*   **Main Application (`main.cpp`):**
    *   Parses command-line arguments (`--input`, `--output`, `--alpha`, `--level`, `--fl`, `--fh`, `--lambda_cutoff`, `--chrom_atten`, `--mode`).
    *   Loads the input video using `cv::VideoCapture`, converting frames from BGR to RGB.
    *   Based on the selected `--mode`:
        *   **Laplacian:** Calls `getLaplacianPyramids`, `filterLaplacianPyramids`, and `reconstructLaplacianImage` sequentially, processing frame by frame.
        *   **Gaussian:** Calls the batch function `processVideoGaussianBatch` (defined in `processing.cpp`) which handles loading frames, calling `spatiallyFilterGaussian` for each, calling `temporalFilterGaussianBatch` on the results, and then calling `reconstructGaussianFrame` frame-by-frame.
    *   Saves the resulting magnified frames as an output video using `cv::VideoWriter`, converting frames back from RGB to BGR. Defines the 5x5 Gaussian kernel used by `cv::pyrDown`/`cv::pyrUp` in the Laplacian path.

*   **Build System (`CMakeLists.txt`):** Uses CMake to manage the build process. Requires OpenCV (core, imgproc, videoio) to be installed and findable.
*   **Testing (`tests/`):**
    *   Uses GoogleTest framework (`gtest`). Configuration in `evmcpp/tests/CMakeLists.txt`.
    *   Includes unit tests for individual components (processing helpers, Butterworth filter, Laplacian pathway, Gaussian pathway). Test files include `test_processing.cpp`, `test_butterworth.cpp`, `test_laplacian.cpp`, and `test_gaussian.cpp`.
    *   Compares results against pre-computed data generated by the Python implementation (`evmpy/generate_test_data.py`) stored in `evmcpp/tests/data/` to ensure correctness and consistency.

### Build and Run

1.  **Prerequisites:**
    *   CMake (>= 3.10)
    *   A C++17 compliant compiler (GCC, Clang, MSVC)
    *   OpenCV (>= 4.x recommended) installed and configured (you might need to set the `OpenCV_DIR` environment variable).

2.  **Build Steps:**
    ```bash
    cd evmcpp
    mkdir build
    cd build
    cmake ..
    make # Or your specific build system generator command (e.g., ninja)
    ```
    This will create the `libevm_core.a` library and the `evm_app` executable inside the `build` directory.

3.  **Running the Application:**
    Execute `evm_app` from the `build` directory.

    ```bash
    ./evm_app [options]
    ```

    **Common Options (see `./evm_app --help` for full list):**

    *   `--input <path>`: Path to the input video (e.g., `../../evmpy/data/face.mp4`).
    *   `--output <path>`: Path for the output video (e.g., `face_magnified.avi`).
    *   `--mode <name>`: Processing mode: `laplacian` (default) or `gaussian`.
    *   `--alpha <float>`: Amplification factor (e.g., `50`).
    *   `--level <int>`: Number of pyramid levels (e.g., `4`).
    *   `--fl <float>`: Low frequency cutoff in Hz (e.g., `0.83`).
    *   `--fh <float>`: High frequency cutoff in Hz (e.g., `1.0`).
    *   `--lambda_cutoff <float>`: Spatial cutoff wavelength (motion mode, e.g., `16`).
    *   `--chrom_atten <float>`: Chrominance attenuation factor (e.g., `1.0` for no attenuation, `0.1` to reduce color shifts).

    **Example (Color Magnification - Pulse):**
    ```bash
    ./evm_app --input ../../evmpy/data/face.mp4 --output face_pulse.avi --mode gaussian --alpha 50 --level 4 --fl 0.83 --fh 1.0 --chrom_atten 0.1
    ```

    **Example (Motion Magnification - Baby):**
    ```bash
    ./evm_app --input ../../evmpy/data/baby.mp4 --output baby_motion.avi --mode laplacian --alpha 20 --level 4 --fl 0.4 --fh 3.0 --lambda_cutoff 10
    ```

## Future Optimizations & CUDA Implementation

### C++ Optimizations

While the C++ version offers performance gains over Python, further optimizations are possible:

*   **Memory Management:** Pre-allocate memory where possible, reuse buffers (`cv::Mat`) to reduce allocations/deallocations within loops (especially frame processing). Analyze `cv::Mat` copying vs. referencing.
*   **Loop Unrolling/Vectorization:** Profile key loops (filtering, pyramid construction) and investigate compiler optimizations or manual vectorization (e.g., using SIMD intrinsics if necessary, though OpenCV often handles this).
*   **Algorithmic Improvements:** Explore alternative filtering techniques if needed.
*   **Parallelism (CPU):** Utilize multi-threading (e.g., OpenMP, TBB, `std::thread`) for frame-level parallelism or potentially within pyramid level processing if beneficial.

### CUDA Implementation Status (April 2025)

The repository now includes a **working, verified CUDA implementation** for the **Gaussian EVM pathway**, with partial support for the Laplacian pathway. The CUDA code resides in `evmcpp/src/evmcuda/` with headers in `evmcpp/include/evmcuda/`. It uses the CUDA Runtime API directly (raw device pointers, `cudaMallocPitch`, `cudaMemcpy2D`, etc.) and is built as a separate static library.

**Current Status:**

- **Gaussian Pathway: Fully Implemented and Verified**
  - **Color Conversion:** `rgb2yiq_gpu` and `yiq2rgb_gpu` kernels convert between RGB and YIQ color spaces. Verified against CPU implementations.
  - **Temporal Filtering:** `temporalFilterGaussianBatch_gpu` uses cuFFT for batch 1D FFTs, with a custom kernel applying frequency masks and amplification. Verified against CPU results.
  - **Reconstruction:** CUDA kernels perform addition of filtered signals, color space conversion, clipping, and conversion back to uint8. Verified against CPU pipeline.
  - **Tests:** Extensive tests compare CUDA outputs to CPU reference implementations, all passing within tight tolerances.

- **Laplacian Pathway: Partially Implemented**
  - **Temporal Filtering:** A CUDA kernel for single-frame IIR Butterworth filtering (`filterLaplacianLevelFrame_gpu`) is implemented and verified against the CPU reference across multiple frames, correctly handling state propagation.
  - **Pyramid Operations:** Custom CUDA kernels for `pyrDown`/`pyrUp` were attempted but abandoned due to numerical discrepancies. Use OpenCV's `cv::cuda::pyrDown`/`pyrUp` if GPU acceleration is needed.

**Summary of CUDA Kernels:**

- **Implemented and Verified:**
  - `rgb2yiq_gpu`, `yiq2rgb_gpu`
  - `temporalFilterGaussianBatch_gpu` (cuFFT-based)
  - Gaussian pathway reconstruction (addition, clipping, conversion)
- **Implemented and Verified:**
  - Laplacian temporal filtering (`filterLaplacianLevelFrame_gpu`, IIR-based, multi-frame state verified)
- **Not Implemented:**
  - Custom pyramid construction (recommend OpenCV CUDA functions)
  - Full Laplacian reconstruction pipeline (Requires pyramid ops and accumulation)

**Integration Strategy:**

- Keep data on the GPU throughout the Gaussian pipeline: color conversion, temporal filtering, amplification, reconstruction.
- Use CUDA streams for asynchronous execution.
- Use cuFFT for temporal filtering.
- Use raw device pointers with pitched memory for efficient 2D data handling.
- Transfer final frames back to CPU for display or encoding, or use GPU-accelerated codecs to avoid transfers.

**Performance:**

- The CUDA Gaussian pathway provides significant acceleration over the CPU implementation.
- Numerical results are consistent with the CPU pipeline, verified by automated tests.

**Next Steps:**

- Optional: Integrate GPU-accelerated video decoding/encoding (e.g., `cv::cudacodec`).
- Optional: Further optimize memory management and kernel launches.
- Optional: Implement CUDA Laplacian reconstruction (pyramid ops + accumulation).
- Optional: Integrate GPU-accelerated video decoding/encoding (e.g., `cv::cudacodec`).
- Optional: Further optimize memory management and kernel launches.

This CUDA implementation enables fast, accurate Gaussian EVM processing on supported NVIDIA GPUs.