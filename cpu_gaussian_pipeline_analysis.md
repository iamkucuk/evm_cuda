# CPU Gaussian Pipeline Analysis

## Overview

The Gaussian path in the Eulerian Video Magnification implementation uses spatial filtering followed by temporal filtering to amplify subtle changes in video. This document provides a comprehensive analysis of each component in the CPU implementation.

## Pipeline Architecture

The Gaussian EVM pipeline consists of three main stages:
1. **Spatial Filtering** (`spatiallyFilterGaussian`)
2. **Temporal Filtering** (`temporalFilterGaussianBatch`) 
3. **Reconstruction** (`reconstructGaussianFrame`)

## Core Components Analysis

### 1. Spatial Filtering Component

**Function**: `spatiallyFilterGaussian` (gaussian_pyramid.cpp:26)

**Purpose**: Applies spatial blur by downsampling then upsampling, creating a low-pass spatial filter effect.

**Algorithm**:
```cpp
Input: RGB frame (CV_8UC3), pyramid levels, Gaussian kernel
1. Convert RGB → YIQ (float32)
2. Downsample 'level' times using pyrDown
3. Upsample 'level' times using pyrUp 
4. Return spatially filtered YIQ frame (CV_32FC3)
```

**Key Implementation Details**:
- Uses custom `pyrDown`/`pyrUp` functions that mirror Python's OpenCV behavior
- Stores intermediate shapes to ensure proper reconstruction
- Level 0 = no filtering (direct RGB→YIQ conversion)
- Final resize to original dimensions if needed

**Critical Dependencies**:
- `pyrDown()`: processing.cpp:72
- `pyrUp()`: processing.cpp:119
- `rgb2yiq()`: processing.cpp:30

### 2. Pyramid Operations (Core Spatial Processing)

#### 2.1 pyrDown Function (processing.cpp:72)

**Purpose**: Gaussian blur followed by 2:1 downsampling

**Algorithm**:
```cpp
Input: Image (CV_32FC3), Gaussian kernel
1. Apply cv::filter2D with kernel and BORDER_REFLECT_101
2. Downsample by 2x using cv::resize with INTER_NEAREST
Output: Downsampled image (half width/height)
```

**OpenCV Equivalency**: Matches `cv::pyrDown` with specific kernel and border handling

#### 2.2 pyrUp Function (processing.cpp:119)

**Purpose**: Zero-injection upsampling followed by Gaussian blur

**Algorithm**:
```cpp
Input: Image (CV_32FC3), kernel, target dimensions
1. Create zeros matrix at target size
2. Place original pixels at even indices (0,2,4...)
3. Apply cv::filter2D with 4×kernel and BORDER_REFLECT_101
4. Resize to exact target dimensions if needed
Output: Upsampled image
```

**Critical Details**:
- 4× kernel scaling compensates for zero injection
- Exact placement at even indices only
- Target size specification crucial for reconstruction

### 3. Color Space Conversion

#### 3.1 RGB to YIQ Conversion (processing.cpp:30)

**Purpose**: Convert RGB to YIQ color space for processing

**Algorithm**:
```cpp
RGB2YIQ_MATRIX = [
  0.299,     0.587,     0.114,     // Y (luminance)
  0.59590059, -0.27455667, -0.32134392, // I (chrominance)
  0.21153661, -0.52273617,  0.31119955  // Q (chrominance)
]
```

**Implementation**: `cv::transform(rgb_float, yiq, RGB2YIQ_MATRIX)`

#### 3.2 YIQ to RGB Conversion (processing.cpp:52)

**Purpose**: Convert processed YIQ back to RGB

**Algorithm**: `cv::transform(yiq, rgb, YIQ2RGB_MATRIX)` where `YIQ2RGB_MATRIX = RGB2YIQ_MATRIX.inv()`

### 4. Temporal Filtering Component

**Function**: `temporalFilterGaussianBatch` (gaussian_pyramid.cpp:118)

**Purpose**: Apply FFT-based bandpass filtering across time for each pixel

**Algorithm**:
```cpp
Input: Spatially filtered YIQ frames, fps, fl, fh, alpha, chromAttenuation
For each pixel (r,c,ch):
  1. Extract time series across all frames
  2. Forward FFT (cv::dft)
  3. Apply frequency mask (keep fl ≤ |f| ≤ fh)
  4. Inverse FFT (cv::idft)
  5. Apply amplification: α for Y, α×chromAttenuation for I,Q
  6. Store amplified signal
Output: Temporally filtered YIQ frames
```

**Key Features**:
- Pixel-wise temporal processing
- Frequency mask based on `np.fft.fftfreq` logic
- Separate amplification for luminance vs chrominance
- Uses OpenCV's DFT functions with proper scaling

### 5. Reconstruction Component

**Function**: `reconstructGaussianFrame` (gaussian_pyramid.cpp:273)

**Purpose**: Combine original frame with amplified signal

**Algorithm**:
```cpp
Input: Original RGB frame, filtered YIQ signal
1. Convert original RGB → YIQ 
2. Add: combined_yiq = original_yiq + filtered_signal
3. Convert combined YIQ → RGB (float)
4. Clip values to [0, 255]
5. Convert to uint8
Output: Final RGB frame (CV_8UC3)
```

## Data Flow and Processing Steps

### Step-by-Step Data Pipeline

Based on test data files, the processing follows these steps:

1. **Input**: RGB frame (frame_0_rgb.txt)
2. **Step 2**: Convert to YIQ (frame_0_step2_yiq.txt)
3. **Step 3**: Spatial filtering (frame_0_step3_spatial_filtered_yiq.txt)
4. **Step 4**: Temporal filtering (frame_0_step4_temporal_filtered_yiq.txt)
5. **Step 5**: Amplification (frame_0_step5_amplified_filtered_yiq.txt)
6. **Step 6b**: Combine with original (frame_0_step6b_combined_yiq.txt)
7. **Step 6c**: Convert to RGB float (frame_0_step6c_reconstructed_rgb_float.txt)
8. **Step 6d**: Clip values (frame_0_step6d_clipped_rgb_float.txt)
9. **Step 6e**: Final uint8 (frame_0_step6e_final_rgb_uint8.txt)

## Performance Characteristics

### Computational Complexity
- **Spatial Filtering**: O(WHK²L) where W=width, H=height, K=kernel_size, L=levels
- **Temporal Filtering**: O(WH×3×N×log(N)) where N=number_of_frames
- **Total**: Dominated by temporal FFT processing

### Memory Usage
- Stores all frames in memory for batch temporal processing
- Peak usage: ~N×W×H×3×4 bytes (for N frames in float32)

## Critical Implementation Notes

### Gaussian Kernel
- Uses 5×5 separable Gaussian kernel: [1,4,6,4,1]/256 in both dimensions
- Total kernel weights sum to 1.0 (normalized)
- Defined in processing.hpp:14-21

### Border Handling
- pyrDown/pyrUp use `BORDER_REFLECT_101` for edge pixels
- Matches OpenCV's default pyramid operations

### Numerical Precision
- All processing in float32 for consistency with Python implementation
- Critical for maintaining >30 dB PSNR vs reference

### Error Handling
- Extensive input validation at each stage
- Graceful degradation with informative error messages
- Empty matrix returns on failures

## Test Validation Framework

### Reference Data
The implementation validates against exact Python outputs:
- Frame dimensions: 592×528 (face.mp4 frame 0)
- Pyramid levels: 4 levels tested
- Multiple test data files for each processing step

### Validation Metrics
- PSNR comparison against Python reference
- Pixel-by-pixel difference analysis
- Statistical error measurements (mean, max error)

## Implementation Dependencies

### External Libraries
- OpenCV 4.x: Core matrices, image processing, FFT operations
- Standard C++: STL containers, error handling

### Internal Dependencies
- processing.hpp: Core pyramid and color conversion functions
- test_helpers.hpp: Validation utilities and data loading

## Performance Optimization Opportunities

### Current Implementation
- Sequential pixel processing in temporal filtering
- Full frame storage in memory
- Single-threaded execution

### CUDA Conversion Targets
- Parallel temporal filtering across pixels
- GPU-resident pyramid operations
- Batch processing of multiple frames simultaneously
- Memory coalescing optimization for 3D data access

This analysis provides the foundation for understanding each component that needs to be converted to CUDA while maintaining numerical accuracy and improving performance.