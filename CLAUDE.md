# Eulerian Video Magnification CUDA Implementation

## Project Overview

This project implements the Eulerian Video Magnification algorithm in CUDA, based on the existing C++ implementation. The goal is to convert the entire pipeline to run on CUDA devices, optimizing for performance while maintaining numerical accuracy with the original CPU implementation.

### What is Eulerian Video Magnification?
Eulerian Video Magnification is a technique to reveal temporal variations in videos that are difficult or impossible to see with the naked eye. The method amplifies small temporal changes in the video signal by using spatial decomposition and temporal filtering, followed by signal amplification.

### Project Structure
- `cpp/`: Contains the original CPU implementation (reference code, should not be touched!)
- `cuda/`: Will contain the new CUDA implementation (our target)
- Documentation: AI-DIARY.md, README.AI.md

## Implementation Strategy

### Overall Approach
1. Convert each component of the CPU implementation to its CUDA counterpart kernel by kernel
2. For each kernel:
   - Identify input/output data structures
   - Implement the CUDA kernel
   - Validate against CPU implementation using fixed inputs
   - Document results and any discrepancies

### Kernel-by-Kernel Validation Process
For each kernel conversion:
1. Run the CPU implementation with fixed inputs and record outputs
2. Run the CUDA implementation with identical inputs
3. Compare results for numerical accuracy (exact match or within acceptable error margin)
4. If mismatch:
   - Analyze potential causes
   - Debug and fix issues
   - Revalidate until match is achieved
5. Document the validation process and results in AI-DIARY.md

## CPU Implementation Details

### Core Components
- **Color Conversion**: RGB to YIQ and back
- **Pyramid Construction**: Gaussian and Laplacian pyramid building
- **Temporal Filtering**: Butterworth bandpass filter
- **Signal Processing**: Amplification and reconstruction
- **Video Processing**: End-to-end pipeline

### File Structure Reference
- `include/`: Header files
  - `butterworth.hpp`: Butterworth filter implementation
  - `color_conversion.hpp`: RGB ↔ YIQ conversion
  - `gaussian_pyramid.hpp`: Gaussian pyramid implementation
  - `laplacian_pyramid.hpp`: Laplacian pyramid implementation
  - `processing.hpp`: Signal processing functions
  - `pyramid.hpp`: Base pyramid class
  - `temporal_filter.hpp`: Temporal filtering functions
- `src/`: Implementation files
  - Corresponding .cpp files for each header
  - `main.cpp`: Entry point for the application
- `tests/`: Test files for each component
  - `data/`: Test data files with expected outputs

### Key Algorithms

#### Color Conversion
- RGB to YIQ conversion: Used to separate intensity (Y) from chrominance (I, Q)
- YIQ to RGB conversion: Used for final output reconstruction

#### Pyramid Construction
- Gaussian pyramid: Multi-scale representation with progressive downsampling
- Laplacian pyramid: Band-pass representation derived from Gaussian pyramid
- Pyramid operations include:
  - Downsampling (pyrDown): Blur + downscale
  - Upsampling (pyrUp): Upscale + blur
  - Reconstruction: Combining levels to recreate original signal

#### Temporal Filtering
- Butterworth bandpass filter: Isolates frequency band of interest
- Applied to each pixel's time series across frames

#### Signal Processing
- Magnification: Amplifies filtered signals
- Reconstruction: Combining original and amplified signals

## CUDA Implementation Requirements

### Memory Management
- Use appropriate CUDA memory types (global, shared, constant)
- Minimize host-device transfers
- Consider pinned memory for efficient transfers when necessary
- Use appropriate memory allocation/deallocation patterns

### Kernel Design
- Design for coalesced memory access
- Use shared memory for frequently accessed data
- Consider occupancy and register usage
- Implement proper error checking

### Optimization Techniques
- Thread block size optimization
- Memory access pattern optimization
- Kernel fusion when appropriate
- Stream concurrency for overlapping operations

### Testing Framework
- Develop unit tests for each kernel
- Compare results against CPU implementation
- Use fixed test data for validation
- Document error margins and validation criteria

## Development Process

### Conversion Order
1. Basic utility functions
2. Color conversion kernels
3. Gaussian pyramid kernels
4. Laplacian pyramid kernels
5. Butterworth filter implementation
6. Temporal filtering kernels
7. Signal processing kernels
8. End-to-end pipeline integration

### Validation Points
- Pixel-by-pixel comparison for image processing kernels
- Level-by-level comparison for pyramid operations
- Frame-by-frame comparison for temporal filters
- End-to-end comparison for full pipeline

## Documentation Standards

### AI-DIARY.md
- Chronological record of development process
- Detailed notes on challenges, solutions, and decisions
- Record of validation results for each kernel
- Insights and observations during development

### README.AI.md
- Current state of the implementation
- Comprehensive knowledge base
- Algorithmic details and implementation choices
- Performance metrics and comparisons
- Known issues and limitations

## Technical Considerations

### CUDA-Specific
- CUDA kernels should follow CUDA best practices
- Use appropriate thread/block organization
- Consider memory coalescing and bank conflicts
- Implement proper error handling

### Numerical Accuracy
- Aim for bit-exact matches when possible
- Document acceptable error margins when exact matches aren't possible
- Understand floating-point precision differences between CPU and GPU
- Use double precision when necessary for accuracy

### Performance Optimization
- Start with correct implementation, then optimize
- Benchmark each kernel against CPU counterpart
- Profile to identify bottlenecks
- Apply optimizations incrementally with validation

## Testing Resources

### Test Data
- Use test data files in `cpp/tests/data/` for validation
- Create custom test cases for edge conditions
- Document expected outputs for each test case

### Validation Methods
- Element-wise comparison for arrays
- Statistical measures (mean error, max error) for floating-point comparisons
- Visual comparison for image outputs when appropriate

## Subagent Guidelines

1. Always begin by reading AI-DIARY.md, README.AI.md, and relevant code files
2. Focus on one kernel implementation at a time
3. Document your approach, challenges, and solutions
4. Include validation results with statistical metrics
5. Update AI-DIARY.md with your findings
6. Ensure your implementation is correct before optimizing
7. Hand off a working, validated implementation

## Environment Constraints

- A conda environment called `cuda_class` will be used, as it contains the nvcc binaries. Ideally, nvcc should be already at the environment, but if not, it's in that conda environment.
- No OpenCV with CUDA support available
- Implement all functionality directly with CUDA
- Exception: Video encoding/decoding can use OpenCV

## Common Pitfalls and Solutions

### Thread Synchronization
- Be aware of race conditions in kernels
- Use appropriate synchronization primitives
- Consider atomic operations when necessary

### Memory Limitations
- Be mindful of GPU memory capacity
- Implement tiling or streaming for large inputs
- Release memory when no longer needed

### Numerical Precision
- Be aware of floating-point precision issues
- Watch for accumulated errors in iterative operations
- Document precision requirements for each operation

### Kernel Launch Configuration
- Choose appropriate grid and block dimensions
- Consider hardware limitations (max threads per block)
- Optimize occupancy for performance

## Implementation Roadmap

1. Set up CUDA project structure
2. Implement and validate color conversion kernels
3. Implement and validate Gaussian pyramid kernels
4. Implement and validate Laplacian pyramid kernels
5. Implement and validate Butterworth filter
6. Implement and validate temporal filtering
7. Implement and validate signal processing
8. Integrate full pipeline
9. Optimize and benchmark
10. Final validation and documentation

## Critical Success Factors

- Numerical accuracy with CPU implementation
- Performance improvement over CPU version
- Complete end-to-end GPU pipeline
- Robust validation and documentation
- Maintainable, well-structured code