# Eulerian Video Magnification - CUDA Implementation

This repository contains a CUDA implementation of the Eulerian Video Magnification (EVM) algorithm, which amplifies subtle temporal variations in videos to reveal imperceptible changes.

## Overview

Eulerian Video Magnification is a technique that enhances small temporal variations in video sequences, making imperceptible motions or color changes visible to the naked eye. Applications include amplifying the pulse visible in a person's face, subtle movements in structures, or small color changes due to blood flow.

This implementation uses CUDA to accelerate the computation, providing significant performance improvements over CPU-based implementations.

## Features

- Full GPU acceleration of the entire EVM pipeline
- Support for both motion magnification and color amplification
- Laplacian pyramid-based spatial decomposition
- Temporal bandpass filtering using IIR Butterworth filters
- Customizable parameters for different types of videos and signals
- Command-line interface for easy integration into other tools

## Requirements

- CUDA-capable GPU with compute capability 7.5 or higher
- CUDA Toolkit 11.0 or higher
- OpenCV 4.x (for video I/O)
- CMake 3.8 or higher
- A C++14 compatible compiler

## Building the Project

1. Make sure you have CUDA and OpenCV installed on your system
2. Clone the repository
3. Configure and build with CMake:

```bash
cd cuda
mkdir build
cd build
cmake ..
make
```

## Usage

```
Eulerian Video Magnification (CUDA Implementation)
Usage: ./evm_cuda [options]
Options:
  -i, --input <file>       Input video file (required)
  -o, --output <file>      Output video file (required)
  -l, --levels <int>       Number of pyramid levels [default: 4]
  -a, --alpha <float>      Magnification factor [default: 10]
  -c, --cutoff <float>     Spatial wavelength cutoff [default: 16]
  -fl, --freq-low <float>  Low frequency cutoff for bandpass [default: 0.05]
  -fh, --freq-high <float> High frequency cutoff for bandpass [default: 0.4]
  -ca, --chrom-att <float> Chrominance attenuation [default: 0.1]
  -h, --help               Display this help message
```

### Example Commands

For motion amplification (e.g., small movements):
```bash
./evm_cuda -i input.mp4 -o output.mp4 -a 20 -l 4 -fl 0.05 -fh 0.4 -c 20 -ca 0.1
```

For pulse/color amplification (e.g., heartbeat):
```bash
./evm_cuda -i face.mp4 -o face_pulse.mp4 -a 100 -l 6 -fl 0.8 -fh 1.0 -c 16 -ca 1.0
```

## Parameter Tuning

- `alpha`: Amplification factor. Higher values give stronger amplification but may introduce artifacts. 
  - For motion: 10-20 is a good range
  - For color/pulse: 50-100 is a good range

- `levels`: Number of pyramid levels. More levels allow for larger motions to be amplified but require more memory.
  - 4-6 is a good range for most videos

- `freq-low` and `freq-high`: Bandpass filter range in Hz. This should be tuned to the frequency of interest.
  - For pulse: 0.8-1.0 Hz (48-60 BPM)
  - For subtle motions: 0.05-0.4 Hz
  - For breathing: 0.2-0.5 Hz

- `cutoff`: Spatial wavelength cutoff for attenuating amplification of higher spatial frequencies.
  - Lower values (e.g., 16) reduce noise amplification but may reduce signal
  - Higher values (e.g., 32) allow more amplification but may increase noise

- `chrom-att`: Chrominance attenuation. Lower values reduce color amplification.
  - 0.1 for motion emphasis with minimal color changes
  - 1.0 for full color amplification

## Implementation Details

The implementation consists of several key components:

1. **Color Space Conversion**: RGB ↔ YIQ conversion to separate intensity and chrominance
2. **Gaussian Pyramid Operations**: Multi-scale decomposition using pyrDown and pyrUp operations
3. **Laplacian Pyramid**: Created by subtracting upsampled downsampled images from original images
4. **Temporal Filtering**: Butterworth bandpass filtering applied to each pixel's time series
5. **Spatial Attenuation**: Lambda-based attenuation to control magnification at different spatial frequencies
6. **Reconstruction**: Combining filtered pyramid levels to create the final magnified video

All operations are implemented as CUDA kernels to maximize performance on the GPU, with careful attention to memory management and proper synchronization.

## Performance

The CUDA implementation provides significant speedup over the CPU implementation, with the exact performance gain depending on:

- Video resolution and length
- Number of pyramid levels
- GPU model and memory bandwidth
- CPU model (for comparison)

Typical speedups range from 5-20x depending on the above factors.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on the original Eulerian Video Magnification paper by Wu et al. (MIT CSAIL)
- CPU implementation based on the evmcpp and evmpy repositories