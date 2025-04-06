# EVM C++: Eulerian Video Magnification Implementation

This project provides a C++ implementation of core functionalities inspired by the research paper "Eulerian Video Magnification for Revealing Subtle Changes in the World" by Hao-Yu Wu, Michael Rubinstein, Eugene Shih, John Guttag, Fredo Durand, and William T. Freeman. It aims to reproduce the numerical results of a corresponding Python implementation (`evmpy`) found in the parent directory.

## 1. Introduction: Eulerian Video Magnification (EVM)

Eulerian Video Magnification is a computational technique that reveals subtle, almost invisible changes in standard videos. Instead of tracking specific features (Lagrangian approach), EVM analyzes the temporal variations of pixel values at fixed locations (Eulerian approach).

**Core Idea:**
1.  **Spatial Decomposition:** The video frames are decomposed into different spatial frequency bands using image pyramids (typically Gaussian or Laplacian pyramids). This separates coarse structures from fine details.
2.  **Temporal Filtering:** The time series of pixel values within each spatial frequency band is filtered temporally (e.g., using bandpass filters like Butterworth) to isolate specific frequency ranges of interest (e.g., heart rate, structural vibrations).
3.  **Amplification:** The filtered temporal signals are amplified (magnified).
4.  **Reconstruction:** The amplified signals are added back to the original signal (or a component of it), and the image is reconstructed from the modified pyramid levels.

This process effectively makes subtle temporal variations (like color changes due to blood flow or tiny motions) visible to the naked eye in the output video.

This C++ project replicates both the **Laplacian pyramid** and **Gaussian pyramid** pathways as implemented in the accompanying `evmpy` Python project.

## 2. Python Implementation (`evmpy`) Walkthrough

The reference Python implementation is located in the `../evmpy` directory relative to this C++ project's root (`evmcpp`).

### 2.1. Project Structure (`evmpy`)

```
evmpy/
├── data/             # Input video files (e.g., face.mp4)
├── results/          # Default output directory for processed videos
├── src/              # Source code modules
│   ├── constants.py  # Shared constants (kernels, color matrices)
│   ├── processing.py # Video I/O, color conversion, pyramid ops, reconstruction
│   ├── laplacian_pyramid.py # Laplacian pyramid generation and filtering
│   ├── gaussian_pyramid.py  # Gaussian pyramid generation and filtering
│   └── evm.py        # Main script, argument parsing, pipeline orchestration
├── generate_test_data.py # Script added to generate reference data for C++ tests
├── requirements.txt  # Python dependencies (numpy, opencv-python, scipy, tqdm)
└── README            # Original Python project README
```

### 2.2. Key Modules (`evmpy/src`)

*   **`constants.py`:** Defines shared numerical values like the 5x5 Gaussian kernel used for pyramids and the RGB <-> YIQ color conversion matrices.
*   **`processing.py`:** Contains fundamental operations:
    *   `loadVideo`: Loads video using OpenCV, extracts frames (converting BGR to RGB), and gets FPS.
    *   `rgb2yiq`, `yiq2rgb`: Performs color space conversions using matrix multiplication based on constants.
    *   `pyrDown`, `pyrUp`: **Crucially, these are *custom* implementations.** They do *not* directly use `cv2.pyrDown`/`cv2.pyrUp`. Instead, they use `cv2.filter2D` with the `gaussian_kernel` followed by manual downsampling (`[::2, ::2]`) or manual zero-padding and filtering, respectively. This detail is vital for numerical replication.
    *   `reconstructLaplacianImage`, `reconstructGaussianImage`: Reconstructs a single output frame by combining the original frame (converted to YIQ) with the appropriately upsampled and summed filtered pyramid levels, then converting back to RGB and clipping/casting to uint8.
    *   `getLaplacianOutputVideo`, `getGaussianOutputVideo`: Helper functions to iterate through frames and call the reconstruction function.
    *   `saveVideo`: Saves the processed frames to an output video file using OpenCV's `VideoWriter`.
    *   `idealTemporalBandpassFilter`: An alternative FFT-based temporal filter (not used by the main Laplacian/Gaussian paths implemented here).
*   **`laplacian_pyramid.py`:**
    *   `generateLaplacianPyramid`: Creates the Laplacian pyramid for a single YIQ image using the custom `pyrDown` and `pyrUp` from `processing.py`. The pyramid levels `L_i` are calculated as `L_i = G_i - pyrUp(G_{i+1})`, where `G_i` is the i-th Gaussian level.
    *   `getLaplacianPyramids`: Processes a sequence of images, calling `rgb2yiq` and `generateLaplacianPyramid` for each.
    *   `filterLaplacianPyramids`: Applies temporal filtering to the batch of pyramids. It uses `scipy.signal.butter` to get 1st-order IIR filter coefficients and applies them frame-by-frame. It also includes spatial attenuation logic based on `lambda_cutoff` and `alpha`.
*   **`gaussian_pyramid.py`:** Contains logic for Gaussian pyramid spatial filtering and FFT-based temporal filtering.
*   **`evm.py`:**
    *   Uses `argparse` to handle command-line arguments for input/output paths, pyramid type (`--mode`), levels, alpha, frequency range, lambda cutoff, etc.
    *   Orchestrates the pipeline: load video -> select mode (Laplacian/Gaussian) -> generate pyramids -> filter pyramids -> reconstruct video -> save video.

### 2.3. Running the Python Code

From the `evmpy` directory:
```bash
# Example for Laplacian mode (matching C++ defaults)
python src/evm.py \
    -v data/face.mp4 \
    -s results/face_py_alpha50.avi \
    -m laplacian \
    -l 4 \
    -a 50 \
    -lc 16 \
    -lo 0.8333 \
    -ho 1.0 \
    -at 1.0
```

## 3. C++ Implementation (`evmcpp`) Walkthrough

This project replicates both the Laplacian and Gaussian pathways of the `evmpy` project in C++.

### 3.1. Project Structure (`evmcpp`)

```
evmcpp/
├── CMakeLists.txt          # Main CMake configuration file
├── README.md               # This file
├── include/                # Header files (.hpp)
│   └── evmcpp/             # Namespace directory
│       ├── laplacian_pyramid.hpp
│       ├── processing.hpp
│       ├── butterworth.hpp
│       └── gaussian_pyramid.hpp
├── src/                    # C++ source files (.cpp)
│   ├── main.cpp            # Main application entry point & pipeline
│   └── evmcpp/             # Implementation files matching headers
│       ├── laplacian_pyramid.cpp
│       ├── processing.cpp
│       ├── butterworth.cpp
│       └── gaussian_pyramid.cpp
├── tests/                  # Unit tests
│   ├── CMakeLists.txt      # CMake config for tests (uses GoogleTest)
│   ├── data/               # Reference data generated by Python script
│   ├── test_laplacian.cpp
│   ├── test_processing.cpp
│   ├── test_butterworth.cpp
│   └── test_gaussian.cpp
└── build/                  # Build output directory (created by CMake)
```

### 3.2. Key Modules (`evmcpp`)

*   **`include/evmcpp/*.hpp` & `src/evmcpp/*.cpp`:** These implement the core logic, mirroring the Python structure:
    *   `processing`: Contains `rgb2yiq`, `yiq2rgb`, and the crucial *custom* `pyrDown` and `pyrUp` implementations that match the Python versions (using `cv::filter2D` and manual sampling/padding). Also holds the color conversion matrices and Gaussian kernel constant.
    *   `butterworth`: Contains `calculateButterworthCoeffs` which manually implements the 1st-order low-pass Butterworth filter design (pre-warping, analog poles, bilinear transform, coefficient calculation) to match `scipy.signal.butter` for the specific case used in the Laplacian filter.
    *   `laplacian_pyramid`: Contains `generateLaplacianPyramid`, `getLaplacianPyramids`, `filterLaplacianPyramids` (using IIR Butterworth filter), and `reconstructLaplacianImage`, replicating the logic from the corresponding Python files.
    *   `gaussian_pyramid`: Contains `spatiallyFilterGaussian` (using custom `pyrDown`/`pyrUp`), `temporalFilterGaussianBatch` (using FFT-based ideal bandpass filter via `cv::dft`/`cv::idft`), and `reconstructGaussianFrame`, replicating the Gaussian pathway logic from Python.
*   **`src/main.cpp`:**
    *   Provides the main application entry point.
    *   Includes simple command-line argument parsing for key hyperparameters (level, alpha, frequencies, etc.) and input/output paths.
    *   Orchestrates the full pipeline based on the selected `--mode`: Load video -> Perform spatial filtering -> Perform temporal filtering -> Reconstruct frames -> Save video. Calls either `processVideoLaplacian` or `processVideoGaussianBatch` (defined in `processing.cpp`).
*   **`CMakeLists.txt` (root):** Configures the project, finds OpenCV, defines the `evm_core` static library and the `evm_app` executable, and includes the `tests` subdirectory.
*   **`tests/CMakeLists.txt`:** Configures the test build. It uses `FetchContent` to download GoogleTest, defines the `evm_tests` executable, and links it against `gtest`, `gtest_main`, and the `evm_core` library. It also copies the reference data from `tests/data` into the build directory.
*   **`tests/*.cpp`:** Contain unit tests using GoogleTest. They load reference data generated by `evmpy/generate_test_data.py` and compare the output of C++ functions against the Python reference numerically using helper functions (`loadMatrixFromTxt`, `loadVectorFromTxt`, `CompareMatrices`, `CompareVectors`).

### 3.3. Building and Running (`evmcpp`)

1.  **Prerequisites:** CMake (>=3.10 recommended), a C++17 compatible compiler (like GCC or Clang), and OpenCV (core, imgproc, videoio modules) installed and findable by CMake (you might need to set the `OpenCV_DIR` environment variable or CMake variable).
2.  **Configure:**
    ```bash
    cd evmcpp
    mkdir build
    cd build
    cmake ..
    ```
3.  **Build:**
    ```bash
    make # Or your chosen build system command
    ```
4.  **Run Tests:**
    ```bash
    ctest # Or ctest --output-on-failure
    ```
5.  **Run Application:**
    ```bash
    ./evm_app --input ../../evmpy/data/face.mp4 --output face_processed.avi [other options...]
    # Run ./evm_app --help to see all options
    ```

## 4. Python-to-C++ Conversion Details

This project replicates both the **Laplacian pyramid** and **Gaussian pyramid** pathways.

### 4.1. Implemented Components

The following Python components have corresponding, numerically verified C++ implementations:

*   **Color Conversion:** `rgb2yiq`, `yiq2rgb` (using `cv::transform`).
*   **Pyramid Operations:** Custom `pyrDown`, `pyrUp` (using `cv::filter2D` and manual sampling/padding).
*   **Laplacian Pathway:** `generateLaplacianPyramid`, `getLaplacianPyramids`, `filterLaplacianPyramids` (using IIR Butterworth filter), `reconstructLaplacianImage`.
*   **Gaussian Pathway:** `spatiallyFilterGaussian`, `temporalFilterGaussianBatch` (using FFT filter), `reconstructGaussianFrame`.
*   **Butterworth Coefficients:** `calculateButterworthCoeffs` (manual implementation for 1st order low-pass).
*   **Constants:** Gaussian kernel, RGB/YIQ matrices.
*   **Pipeline:** The overall flow in `main.cpp` and `processing.cpp` (`processVideoLaplacian`, `processVideoGaussianBatch`) mirrors `evm.py`.

### 4.2. Key Challenges & Differences

*   **Custom Pyramids:** The most critical finding was that `evmpy` uses custom `pyrDown`/`pyrUp` based on `filter2D`, not standard `cv2.pyrDown`/`cv2.pyrUp`. The C++ code *must* replicate this custom logic for numerical equivalence.
*   **Butterworth Filter:** `scipy.signal.butter` involves complex DSP calculations. The C++ version manually implements these steps (pre-warping, analog poles, bilinear transform, coefficient generation) for the specific 1st-order low-pass case needed.
*   **Data Types:** Python uses `float32` extensively. C++ uses `float` (`CV_32F`) for most image processing and `double` for filter coefficient calculations to maintain precision. Type conversions (e.g., `convertTo`) are used carefully.
*   **Color Order:** OpenCV typically uses BGR, while the Python code loaded frames and immediately converted to RGB. The C++ `main.cpp` mimics this by converting to RGB after loading and back to BGR before saving. Internal processing uses RGB or YIQ.
*   **Array Handling:** NumPy provides flexible array operations. C++ uses `cv::Mat` and standard library vectors, requiring more explicit loops or OpenCV functions (`cv::add`, `cv::subtract`, `cv::split`, `cv::merge`, `cv::resize`).
*   **Testing:** A rigorous TDD approach was essential. This involved:
    *   Modifying the Python code (`generate_test_data.py`) to dump intermediate results to text files.
    *   Creating C++ helper functions to load this text data.
    *   Using GoogleTest to compare C++ outputs against Python references with appropriate floating-point tolerances. Tolerances sometimes needed slight adjustment due to minor differences in floating-point accumulation between NumPy and OpenCV/C++.

## 5. Future Enhancements

*   **Extended Butterworth Filter:** The C++ `calculateButterworthCoeffs` currently only implements the 1st-order low-pass case required by the Laplacian filter. It could be extended to support higher orders or other filter types (`high`, `bandpass`, `bandstop`) for greater flexibility.
*   **CUDA Acceleration:** Implement GPU-accelerated versions of key components (pyramids, filtering, color conversion, reconstruction) using CUDA, cuFFT, and potentially NPP, as outlined in the main project README.
*   **Performance Profiling & Optimization:** Profile the C++ CPU implementation and apply further optimizations (memory management, vectorization, CPU parallelism).
