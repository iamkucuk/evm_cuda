# CUDA Implementation Kickoff Prompt

I need your help to convert an existing C++ implementation of the Eulerian Video Magnification algorithm to a complete CUDA implementation. This is a challenging project that requires meticulous kernel-by-kernel conversion with rigorous validation at each step.

## PROJECT CONSTRAINTS AND GUIDELINES

1. **Subagent Approach**: Each kernel implementation MUST be handled by a dedicated subagent
2. **Continuous Documentation**: Maintain AI-DIARY.md, README.AI.md, and CLAUDE.md religiously
3. **Validation Requirements**: EVERY kernel must be validated against the CPU implementation using identical inputs
4. **No OpenCV-CUDA**: The environment lacks OpenCV with CUDA support, so implement all functionality directly in CUDA
5. **Conda Environment**: Use the conda environment called `cuda_class` which contains the nvcc binaries. If nvcc is not in your PATH, activate this environment with `conda activate cuda_class`
6. **Time Commitment**: This is a complex project requiring careful, methodical implementation - do not rush

## SUBAGENT WORKFLOW

I want you to ultrathink about this entire process first. For each kernel implementation, follow this exact workflow:

1. **Spawn Dedicated Subagent**: Create a subagent specifically for the current kernel implementation
   - Instruct the subagent to read AI-DIARY.md, README.AI.md, and CLAUDE.md first
   - Provide the subagent with clear boundaries of its task (one kernel only)
   - Give the subagent complete information about the CPU implementation

2. **Subagent Implementation Process**:
   ```
   a. Analyze CPU implementation (study code structure, algorithms, data flow)
   b. Design CUDA kernel (memory layout, thread organization, boundary handling)
   c. Implement CUDA kernel with proper error checking
   d. Set up validation framework with fixed test inputs
   e. Run CPU implementation and collect results
   f. Run CUDA kernel with identical inputs and collect results
   g. Compare results with appropriate metrics (max error, mean error, PSNR)
   h. Debug any discrepancies until validation passes
   i. Document implementation details, challenges, and solutions
   j. Report back with comprehensive validation results
   ```

3. **Validation Requirements**:
   - Use fixed input data from test files in `cpp/tests/data/`
   - Implement exact comparison for integer types
   - Use epsilon comparison (typically 1e-5 or 1e-6) for floating-point types
   - Calculate and report statistical metrics (max error, mean error, PSNR)
   - Do NOT proceed to the next kernel until current kernel validation passes

4. **Documentation Update**:
   - Update AI-DIARY.md with implementation details, challenges, and solutions
   - Update README.AI.md with current implementation status and any new insights
   - Each subagent must contribute to keeping documentation current

## KERNEL IMPLEMENTATION SEQUENCE

Implement and validate kernels in this specific order:

1. **Color Conversion**:
   - RGB → YIQ conversion
   - YIQ → RGB conversion
   - Validation with test data from `cpp/tests/data/frame_*_rgb.txt` and `cpp/tests/data/frame_*_yiq.txt`

2. **Gaussian Pyramid Operations**:
   - pyrDown (downsampling with Gaussian blur)
   - pyrUp (upsampling with Gaussian blur)
   - Validation with test data from `cpp/tests/data/frame_*_pyrdown_*.txt` and `cpp/tests/data/frame_*_pyrup_*.txt`

3. **Laplacian Pyramid Operations**:
   - Laplacian pyramid construction
   - Laplacian pyramid reconstruction
   - Validation with test data from `cpp/tests/data/frame_*_laplacian_level_*.txt`

4. **Butterworth Filter**:
   - Butterworth bandpass filter coefficient calculation
   - Validation with test data from `cpp/tests/data/butter_*.txt`

5. **Temporal Filtering**:
   - Temporal filtering using Butterworth filter
   - Validation with test data from `cpp/tests/data/frame_*_filtered_level_*.txt`

6. **Signal Processing**:
   - Signal amplification
   - Signal reconstruction
   - Validation with test data from various steps in `cpp/tests/data/`

7. **End-to-End Pipeline**:
   - Integration of all components
   - Full pipeline validation

## DOCUMENTATION FRAMEWORK

Maintain these three critical documents throughout the development process:

1. **AI-DIARY.md**: Chronological development log
   - Each entry dated and labeled with specific kernel
   - Detailed analysis of CPU implementation
   - Implementation approach and challenges
   - Validation results with metrics
   - Debugging steps taken for any discrepancies
   - Final solution and lessons learned

2. **README.AI.md**: Knowledge base for subagents
   - Current implementation status
   - Technical details of algorithms
   - Common pitfalls and solutions
   - Performance metrics and comparisons
   - Important reference information for subagents

3. **CLAUDE.md**: Project guidelines and reference
   - Comprehensive project overview
   - Implementation strategy
   - Technical requirements
   - Validation methodology

## SUBAGENT INSTRUCTIONS TEMPLATE

For each kernel implementation, use this template when spawning a subagent:

```
# SUBAGENT TASK: [Specific Kernel] Implementation

You are a dedicated subagent assigned to implement the [Specific Kernel] for our Eulerian Video Magnification CUDA conversion project. Your task is strictly limited to this specific kernel implementation and validation.

## REQUIRED PREPARATION
1. First, carefully read these documents to gain context:
   - CLAUDE.md (project overview and guidelines)
   - README.AI.md (knowledge base with current status and technical details)
   - AI-DIARY.md (development history and challenges to date)
   - Relevant CPU implementation files:
     - [list specific CPU files to study]
   - Relevant test files:
     - [list specific test files to use for validation]

2. Study the CPU implementation in detail:
   - Understand the algorithm and data flow
   - Identify memory access patterns
   - Note boundary conditions and edge cases

## IMPLEMENTATION REQUIREMENTS
1. Implement the CUDA kernel with:
   - Proper thread/block organization
   - Appropriate memory management
   - Correct boundary handling
   - Error checking for all CUDA operations

2. Validate your implementation:
   - Use fixed test inputs from cpp/tests/data/
   - Run both CPU and CUDA implementations
   - Compare outputs with appropriate metrics
   - Debug any discrepancies until validation passes

3. Document your work:
   - Implementation details and approach
   - Challenges encountered and solutions
   - Validation results with metrics
   - Performance metrics (execution time)

## ENVIRONMENT SETUP
1. The conda environment `cuda_class` contains the necessary CUDA tools:
   ```bash
   conda activate cuda_class
   nvcc --version  # Verify CUDA compiler is available
   ```
2. All CUDA development should be done in this environment

## DELIVERABLES
1. Complete CUDA kernel implementation
2. Validation code and results
3. Updated documentation (AI-DIARY.md and README.AI.md)
4. Any insights for future kernel implementations

Begin by analyzing the CPU implementation and formulating your CUDA implementation approach.
```

## GETTING STARTED

To begin this project:

1. First, ensure you're using the correct environment:
```bash
conda activate cuda_class
# Verify nvcc is available
nvcc --version
```

2. Create the necessary directory structure:
```bash
mkdir -p cuda/include cuda/src cuda/tests cuda/tests/data
```

3. Initialize the documentation files:
```bash
# Copy existing templates or create new ones
touch AI-DIARY.md README.AI.md
cp CLAUDE.md ./CLAUDE.md
```

3. Create the initial CUDA project files:
```bash
# Create basic CMakeLists.txt for CUDA project
touch cuda/CMakeLists.txt
# Create main header files mirroring CPU implementation
touch cuda/include/cuda_color_conversion.cuh
touch cuda/include/cuda_pyramid.cuh
# Create implementation files
touch cuda/src/cuda_color_conversion.cu
```

4. Spawn the first subagent for color conversion implementation following the template above.

Remember: This is a complex project requiring meticulous implementation and validation. Take the time to thoroughly understand each component before implementation, use subagents effectively, and maintain comprehensive documentation throughout the process.

Ultrathink about the project architecture first, then begin the kernel-by-kernel implementation with rigorous validation at each step.