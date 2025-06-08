# Eulerian Video Magnification CUDA Implementation

## Project Overview

This project implements the Eulerian Video Magnification algorithm in CUDA, based on the existing C++ implementation. The goal is to convert the entire pipeline to run on CUDA devices, optimizing for performance while maintaining numerical accuracy with the original CPU implementation.

### What is Eulerian Video Magnification?
Eulerian Video Magnification is a technique to reveal temporal variations in videos that are difficult or impossible to see with the naked eye. The method amplifies small temporal changes in the video signal by using spatial decomposition and temporal filtering, followed by signal amplification.

### Project Structure
- `cpp/`: Contains the original CPU implementation (reference code, should not be touched!)
- `cuda/`: Will contain the new CUDA implementation (our target)
- Documentation: AI-DIARY.md, README.AI.md, cpu_gaussian_pipeline_analysis.md

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

### Detailed Analysis
For comprehensive understanding of the CPU implementation, see: `cpu_gaussian_pipeline_analysis.md`

This document provides:
- Complete pipeline architecture breakdown
- Detailed algorithm analysis for each component
- Data flow documentation with step-by-step processing
- Performance characteristics and optimization targets
- Critical implementation notes for CUDA conversion

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

### MANDATORY CONVERSION ORDER - COMPONENT-BY-COMPONENT VALIDATION

**RULE 1: IF EVERY FUCKING COMPONENT CAN SUCCESSFULLY REPLICATE THE CORRESPONDING FUCKING CPU COMPONENTS, IT MEANS YOU ARE ON THE RIGHT TRACK. FIRST TAKE THAT APPROACH.**

1. **Color conversion kernels** - VALIDATE: Must achieve >30 dB PSNR vs CPU
2. **Gaussian pyramid kernels** - VALIDATE: Must achieve >30 dB PSNR vs CPU  
3. **Laplacian pyramid kernels** - VALIDATE: Must achieve >30 dB PSNR vs CPU
4. **Butterworth filter implementation** - VALIDATE: Must achieve >30 dB PSNR vs CPU
5. **Temporal filtering kernels** - VALIDATE: Must achieve >30 dB PSNR vs CPU
6. **Signal processing kernels** - VALIDATE: Must achieve >30 dB PSNR vs CPU
7. **ONLY AFTER ALL COMPONENTS PASS**: End-to-end pipeline integration

**COMPONENT VALIDATION REQUIREMENTS**:
- Each component must individually replicate CPU behavior with >30 dB PSNR
- Use identical inputs for GPU and CPU component tests
- Measure actual PSNR, not theoretical calculations
- Document validation results for each component

**RULE 2: FIXING ONE COMPONENT DOES NOT MEAN YOU FUCKING DID IT YOU IDIOT MORON, STOP SPITTING NON-SENSE NUMBERS REGARDING THE ALL IMPLEMENTATION, OR ANYTHING ELSE WITHOUT DOING ACTUAL MEASUREMENTS!**

## SUCCESS HYPOTHESIS: THREE-LEVEL VALIDATION

**DISCOVERED**: Single-frame spatial component tests (~30 dB PSNR) can coexist with catastrophic video failure (6.98 dB PSNR) because critical components are missing from validation.

### LEVEL 1: COMPONENT SUCCESS (Individual Validation)
**SPATIAL COMPONENTS** (mostly working):
- RGB ↔ YIQ conversion: >30 dB PSNR ✅
- Pyramid operations: >30 dB PSNR ✅ (except Level 3 pyrUp)

**TEMPORAL COMPONENTS** (untested, likely broken):
- Butterworth temporal filtering: Test compilation broken
- Filter state management: No test exists
- Multi-frame consistency: No test exists

**PROCESSING COMPONENTS** (untested, likely broken):
- Amplification scaling: No test exists
- Chromatic attenuation: No test exists  
- Signal normalization: No test exists

### LEVEL 2: INTEGRATION SUCCESS (Component Combinations)
- Temporal + spatial integration: Untested
- Amplification + processing: Untested
- Multi-frame GPU memory management: Untested

### LEVEL 3: END-TO-END SUCCESS (Full Video Pipeline)
- Video PSNR: Currently 6.98 dB (FAILING)
- Target: >25 dB acceptable, >30 dB ideal

**ROOT CAUSE HYPOTHESIS**: Untested temporal/amplification components are catastrophically broken, causing video failure despite working spatial components.

**MANDATORY TESTING SEQUENCE**:
1. **ALWAYS START WITH VIDEO PSNR** - This is the only success metric that matters
2. If video PSNR <25 dB: Use component tests for diagnosis (Level 1)
3. Fix broken components identified by diagnosis (Level 2)  
4. **RETURN TO VIDEO PSNR MEASUREMENT** - Repeat until >25 dB achieved

**THE ULTIMATE SUCCESS METRIC**: GPU vs CPU video PSNR >25 dB (acceptable) or >30 dB (excellent)

**CURRENT REALITY**: 6.98 dB video PSNR = TOTAL FAILURE

Component tests are **DIAGNOSTIC TOOLS ONLY** - they identify what's broken but **DO NOT** determine success. Only video-to-video comparison determines success.

### Validation Points
- Pixel-by-pixel comparison for image processing kernels
- Level-by-level comparison for pyramid operations
- Frame-by-frame comparison for temporal filters
- **MANDATORY: End-to-end video comparison for full pipeline**

### CRITICAL VALIDATION REQUIREMENTS - LEARNED THE HARD WAY

**WARNING**: Component-level validation (single frames, isolated kernels) **DOES NOT** guarantee video processing quality.

**MANDATORY VALIDATION PROTOCOL**:

1. **ALWAYS GENERATE ACTUAL VIDEOS**: 
   - Run both GPU and CPU pipelines on identical inputs
   - Generate full video outputs (not just single frames)
   - Test on multiple input videos (face.mp4, baby.mp4, etc.)

2. **ALWAYS MEASURE REAL PSNR**:
   ```bash
   ./compare_videos_frame_by_frame gpu_output.mp4 cpu_output.mp4
   ```
   - Video-to-video PSNR must be >25 dB (acceptable) or >30 dB (target)
   - SSIM must be >0.8 for good structural similarity
   - No memory errors or crashes allowed

3. **NEVER TRUST COMPONENT TESTS ALONE**:
   - Single-frame validation achieving 28.82 dB ≠ video quality
   - Real video comparison showed only 6.98 dB PSNR
   - 21.84 dB discrepancy proves component tests are misleading

4. **TEMPORAL EFFECTS ARE CRITICAL**:
   - Cross-frame processing introduces errors invisible in single-frame tests
   - Temporal filtering, amplification, and memory management affect video quality
   - Pipeline integration issues only manifest during full video processing

**HISTORICAL FAILURE**: Previous claims of "28.82 dB PSNR improvement" were based on single-frame validation and completely wrong when measured against actual video outputs.

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

### Numerical Accuracy - CRITICAL UPDATE (Dec 2024)
- **TARGET: >30 dB PSNR** for all components vs CPU reference data
- **CURRENT STATUS**: GPU pyramid operations achieve only 16-19 dB (need +13-17 dB improvement)
- **ROOT CAUSE**: GPU kernels don't exactly replicate OpenCV's separable [1,4,6,4,1] algorithm
- **VALIDATION METHOD**: Use `validate_against_cpu_reference` test with exact CPU reference data

#### Specific Technical Requirements for >30 dB Achievement:
1. **OpenCV-Exact pyrDown Implementation**:
   - Use separable [1,4,6,4,1] kernel with 0.25× total scaling
   - BORDER_REFLECT_101 border handling (not BORDER_REPLICATE)
   - Impulse response: single pixel → 2×2 region (16,16,16,16 values)

2. **OpenCV-Exact pyrUp Implementation**:
   - 4× scaling compensation for zero injection
   - Exact `dstsize` specification like CPU: `cv2.pyrUp(down, dstsize=(orig_width, orig_height))`
   - Even-position sampling only

3. **Validation Requirements**:
   - Each pyramid operation must achieve >30 dB vs `reference_data/step3_levelX_*.txt`
   - No approximations allowed - must match CPU algorithm exactly

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

## FUCKING VALIDATION MANDATE - NO EXCEPTIONS, NO BULLSHIT

**RULE 1: COMPONENT-BY-COMPONENT APPROACH**
- Fix ONE component at a time
- Validate THAT component achieves >30 dB PSNR vs CPU
- DO NOT make claims about the full pipeline
- DO NOT proceed to next component until current one passes

**RULE 2: NO BULLSHIT NUMBERS WITHOUT MEASUREMENT**
- NEVER claim "pipeline improvements" based on fixing one component
- NEVER extrapolate component results to full system
- MEASURE every fucking claim with actual data
- Document ONLY what you actually measured

**BEFORE CLAIMING ANY SYSTEM-LEVEL IMPROVEMENTS**:

8. **ALL COMPONENTS MUST INDIVIDUALLY PASS**: Every single component >30 dB PSNR vs CPU
9. **GENERATE ACTUAL VIDEOS**: Only after all components pass individually
10. **MEASURE REAL VIDEO PSNR**: Use frame-by-frame video comparison tools
11. **VERIFY WITH MULTIPLE INPUTS**: Test on face.mp4, baby.mp4, wrist.mp4
12. **CHECK FOR ERRORS**: Ensure no CUDA memory errors or crashes

**FAILURE TO FOLLOW COMPONENT-FIRST APPROACH LEADS TO CATASTROPHIC ASSESSMENT ERRORS**

Historical example: Claimed 28.82 dB improvement based on fixing reconstruction logic, actual video quality was only 6.98 dB - a catastrophic 21.84 dB error because other components were broken.

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

## GPU-Resident Architecture Guidelines

### Key Principles for High-Performance CUDA Implementation

1. **Minimize CPU-GPU Transfers**:
   - Keep all intermediate data on GPU throughout the pipeline
   - Only transfer data at the beginning (input) and end (output)
   - Pre-allocate all GPU memory needed for the entire computation

2. **Apply SIMD Principles**:
   - Process multiple data elements in parallel (frames, pyramid levels, pixels)
   - Use 3D kernel grids for (width × height × frames) parallelism
   - Exploit data parallelism at every level of the algorithm

3. **Memory Layout Optimization**:
   - Organize data for coalesced memory access
   - Consider structure-of-arrays (SoA) vs array-of-structures (AoS)
   - Align memory allocations for optimal access patterns

4. **Parallel Processing of Hierarchical Structures**:
   - Process pyramid levels concurrently using CUDA streams
   - Use separate streams for independent operations
   - Overlap computation with memory operations where possible

### Laplacian Mode Specific Guidelines - UPDATED (Dec 2024)

**CRITICAL QUALITY ISSUE IDENTIFIED**:
- **Current Quality**: 12.85 dB PSNR vs CPU reference (target: >30 dB)
- **Root Cause**: pyramid kernels don't match OpenCV exactly
- **Priority**: Quality achievement BEFORE performance optimization

**Quality-First Implementation Requirements**:
```cuda
// 1. Exact OpenCV kernel replication
__constant__ float c_opencv_kernel[25] = {
    // [1,4,6,4,1] separable kernel with exact normalization
    1.0f/256, 4.0f/256, 6.0f/256, 4.0f/256, 1.0f/256,
    // ... (see OpenCV source for exact values)
};

// 2. Exact border handling
if (px < 0) px = -px;  // BORDER_REFLECT_101
if (py < 0) py = -py;
if (px >= width) px = 2*width - px - 1;
if (py >= height) py = 2*height - py - 1;

// 3. Exact size handling like CPU
pyrUp(input, output, exact_target_width, exact_target_height);
```

**Validation-Driven Development**:
1. Each kernel must pass `validate_against_cpu_reference` with >30 dB
2. Use `reference_data/step3_levelX_*.txt` for exact comparison
3. NO performance optimization until quality target achieved

**Performance Gains** (after quality achieved):
- GPU-resident architecture: 6.27x speedup maintained
- Zero intermediate CPU-GPU transfers
- Batch processing of multiple frames/levels

# IMPORTANT NOTES:
- Your subtasks will try to trick you. Always create another subtask to reflect what the previous subtask did. Confirm the code is actually working, compares with the reference implementation and validates it's working.
- You and your subtasks always use `sequential-thinking mcp` to break down the task and accomplish those breakdowns.
- DO NOT USE OPENCV!
- Test, evaluate and compare against CPU implementations as you add each kernel.