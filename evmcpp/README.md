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

This C++ project focuses on replicating the **Laplacian pyramid-based motion magnification** pathway as implemented in the accompanying `evmpy` Python project.

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
*   **`gaussian_pyramid.py`:** (Not converted to C++ yet) Contains similar logic but for Gaussian pyramids and a different filtering approach.
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

This project aims to replicate the Laplacian pathway of the `evmpy` project in C++.

### 3.1. Project Structure (`evmcpp`)

```
evmcpp/
├── CMakeLists.txt          # Main CMake configuration file
├── README.md               # This file
├── include/                # Header files (.hpp)
│   └── evmcpp/             # Namespace directory
│       ├── laplacian_pyramid.hpp
│       ├── processing.hpp
│       └── butterworth.hpp
├── src/                    # C++ source files (.cpp)
│   ├── main.cpp            # Main application entry point & pipeline
│   └── evmcpp/             # Implementation files matching headers
│       ├── laplacian_pyramid.cpp
│       ├── processing.cpp
│       └── butterworth.cpp
├── tests/                  # Unit tests
│   ├── CMakeLists.txt      # CMake config for tests (uses GoogleTest)
│   ├── data/               # Reference data generated by Python script
│   ├── test_laplacian.cpp
│   ├── test_processing.cpp
│   └── test_butterworth.cpp
└── build/                  # Build output directory (created by CMake)
```

### 3.2. Key Modules (`evmcpp`)

*   **`include/evmcpp/*.hpp` & `src/evmcpp/*.cpp`:** These implement the core logic, mirroring the Python structure:
    *   `processing`: Contains `rgb2yiq`, `yiq2rgb`, and the crucial *custom* `pyrDown` and `pyrUp` implementations that match the Python versions (using `cv::filter2D` and manual sampling/padding). Also holds the color conversion matrices and Gaussian kernel constant.
    *   `butterworth`: Contains `calculateButterworthCoeffs` which manually implements the 1st-order low-pass Butterworth filter design (pre-warping, analog poles, bilinear transform, coefficient calculation) to match `scipy.signal.butter` for the specific case used in the Laplacian filter.
    *   `laplacian_pyramid`: Contains `generateLaplacianPyramid`, `getLaplacianPyramids`, `filterLaplacianPyramids`, and `reconstructLaplacianImage`, replicating the logic from the corresponding Python files using the C++ helper functions.
*   **`src/main.cpp`:**
    *   Provides the main application entry point.
    *   Includes simple command-line argument parsing for key hyperparameters (level, alpha, frequencies, etc.) and input/output paths.
    *   Orchestrates the full pipeline: Load video -> Build pyramids -> Filter pyramids -> Reconstruct frames -> Save video.
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

This project focused on replicating the **Laplacian pyramid motion magnification** path.

### 4.1. Implemented Components

The following Python components have corresponding, numerically verified C++ implementations:

*   **Color Conversion:** `rgb2yiq`, `yiq2rgb` (using pixel iteration and matrix multiplication).
*   **Pyramid Operations:** Custom `pyrDown`, `pyrUp` (using `cv::filter2D` and manual sampling/padding).
*   **Pyramid Generation:** `generateLaplacianPyramid`, `getLaplacianPyramids`.
*   **Temporal Filtering:** `filterLaplacianPyramids` (including spatial attenuation logic).
*   **Butterworth Coefficients:** `calculateButterworthCoeffs` (manual implementation for 1st order low-pass, matching `scipy.signal.butter` for the required cases).
*   **Reconstruction:** `reconstructLaplacianImage`.
*   **Constants:** Gaussian kernel, RGB/YIQ matrices.
*   **Pipeline:** The overall flow in `main.cpp` mirrors `evm.py` for the Laplacian path.

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

## 5. Remaining Work (Future Implementation)

This C++ implementation currently only covers the Laplacian pyramid pathway. The following components from the `evmpy` project have **not** been converted:

*   **Gaussian Pyramid Path:**
    *   `src/gaussian_pyramid.py` (`getGaussianPyramids`, `filterGaussianPyramids`)
    *   `src/processing.py` (`reconstructGaussianImage`)
    *   The corresponding logic branch in `src/evm.py` / C++ `main.cpp`.
*   **Ideal Temporal Filter:** The FFT-based `idealTemporalBandpassFilter` from `processing.py` was not implemented. The C++ code uses the IIR Butterworth approach from the Laplacian path.
*   **Extended Butterworth:** The C++ `calculateButterworthCoeffs` only implements the 1st-order low-pass case required by the Laplacian filter. It does not yet support higher orders or other filter types (`high`, `bandpass`, `bandstop`) which might be needed for other applications or extensions.

## 6. How to Implement Remaining Parts

To implement the remaining **Gaussian pyramid path**, the same methodology used for the Laplacian path can be followed:

1.  **Analyze:** Study the Python code in `gaussian_pyramid.py` and the `reconstructGaussianImage` function in `processing.py`. Understand the data flow and calculations.
2.  **Design (C++):**
    *   Declare necessary functions (e.g., `getGaussianPyramids`, `filterGaussianPyramids`, `reconstructGaussianImage`) in a new header file `include/evmcpp/gaussian_pyramid.hpp`.
    *   Create the corresponding source file `src/evmcpp/gaussian_pyramid.cpp`.
3.  **Implement (C++):** Write the C++ code for these functions, reusing helpers from `processing.hpp` where applicable. Pay attention to the specific filtering logic used in `filterGaussianPyramids`.
4.  **Generate Reference Data (Python):** Modify `evmpy/generate_test_data.py` to:
    *   Call the Python Gaussian functions (`getGaussianPyramids`, `filterGaussianPyramids`, `reconstructGaussianImage`).
    *   Save the outputs of these functions (pyramid levels, filtered levels, reconstructed frame) to new text files in `evmcpp/tests/data/`.
5.  **Test (C++):**
    *   Create a new test file `evmcpp/tests/test_gaussian.cpp`.
    *   Add this file to the `evm_tests` target in `evmcpp/tests/CMakeLists.txt`.
    *   Write GoogleTest cases in `test_gaussian.cpp` that load the new reference data and compare it against the output of the C++ Gaussian functions using the existing comparison helpers.
6.  **Integrate (C++):**
    *   Add the new source file (`src/evmcpp/gaussian_pyramid.cpp`) to the `evm_core` library target in the root `evmcpp/CMakeLists.txt`.
    *   Modify `src/main.cpp` to:
        *   Accept a `--mode gaussian` command-line argument.
        *   Call the Gaussian pathway functions (`getGaussianPyramids`, `filterGaussianPyramids`, `reconstructGaussianImage`) when the Gaussian mode is selected.

Implementing other filter types or higher orders for the Butterworth filter would involve extending the `calculateButterworthCoeffs` function in `butterworth.cpp` with the appropriate DSP math for those cases and adding corresponding tests.

## 7. Prompt for SPARC Orchestrator (Future Work)

```text
<task>
Extend the existing `evmcpp` C++ project to implement the Gaussian pyramid pathway for Eulerian Video Magnification, ensuring numerical equivalence with the reference Python implementation in `../evmpy`.

**Context:**
- The `evmcpp` project currently implements the Laplacian pyramid pathway and has verified C++ functions for `rgb2yiq`, `yiq2rgb`, custom `pyrDown`/`pyrUp`, `generateLaplacianPyramid`, `getLaplacianPyramids`, `filterLaplacianPyramids` (using 1st order Butterworth), and `reconstructLaplacianImage`.
- Unit tests using GoogleTest and reference data generated from Python (`evmpy/generate_test_data.py`) are already set up in `evmcpp/tests`.
- The Python reference code for the Gaussian path is in `evmpy/src/gaussian_pyramid.py` and `evmpy/src/processing.py` (`reconstructGaussianImage`).

**Methodology:**
Follow the established Test-Driven Development (TDD) and modular implementation approach used for the Laplacian path:
1.  **Analyze Python Code:** Understand the logic in `evmpy/src/gaussian_pyramid.py` (specifically `getGaussianPyramids`, `filterGaussianPyramids`) and `evmpy/src/processing.py` (`reconstructGaussianImage`). Note the filtering method used.
2.  **C++ Skeletons:** Create `include/evmcpp/gaussian_pyramid.hpp` and `src/evmcpp/gaussian_pyramid.cpp`. Declare the necessary functions (`getGaussianPyramids`, `filterGaussianPyramids`, `reconstructGaussianImage`).
3.  **Generate Reference Data:** Modify `evmpy/generate_test_data.py` to execute the Python Gaussian pathway for the first few frames of `evmpy/data/face.mp4` and save the outputs of `getGaussianPyramids` (each level), `filterGaussianPyramids` (each filtered level), and `reconstructGaussianImage` (the final frame) to appropriately named text files in `evmcpp/tests/data/`.
4.  **Implement C++ Functions:** Implement the functions declared in `gaussian_pyramid.hpp` within `gaussian_pyramid.cpp`, reusing helpers from `processing.hpp` as needed. Ensure the filtering logic matches the Python version.
5.  **Implement C++ Tests:**
    *   Create `evmcpp/tests/test_gaussian.cpp`.
    *   Add `test_gaussian.cpp` to the `evm_tests` target in `evmcpp/tests/CMakeLists.txt`.
    *   Write GoogleTest cases in `test_gaussian.cpp` to:
        *   Load reference input data (e.g., original frames).
        *   Call the C++ Gaussian functions (`getGaussianPyramids`, `filterGaussianPyramids`, `reconstructGaussianImage`).
        *   Load the corresponding reference output data generated in step 3.
        *   Compare the C++ results numerically against the Python reference data using the existing helper functions (`loadMatrixFromTxt`, `CompareMatrices`) with appropriate tolerances.
6.  **Integrate into Main App:**
    *   Add `src/evmcpp/gaussian_pyramid.cpp` to the `evm_core` library target in the root `evmcpp/CMakeLists.txt`.
    *   Modify `src/main.cpp`:
        *   Add a `--mode <gaussian|laplacian>` command-line argument (defaulting to laplacian or requiring specification).
        *   Based on the mode, call either the existing Laplacian pipeline functions or the new Gaussian pipeline functions.
7.  **Build and Verify:** Ensure the entire project builds successfully and all tests (including the new Gaussian tests) pass via `ctest`. Ensure the `evm_app` executable runs correctly in Gaussian mode.

**Goal:** A fully functional `evmcpp` project that can perform both Laplacian and Gaussian EVM, with both pathways numerically verified against the Python reference.
</task>