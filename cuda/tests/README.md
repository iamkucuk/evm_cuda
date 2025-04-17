# CUDA Gaussian Blur Test

This test compares the output of the CUDA Gaussian blur implementation with a naive CPU implementation on a random 3-channel float image. The test prints the maximum absolute difference between the two results and reports pass/fail.

## Build Instructions

Assuming you are inside the `cuda/` directory and have CUDA toolkit installed:

```
nvcc -Iinclude -o tests/test_gaussian_blur tests/test_gaussian_blur.cu src/gaussian_pyramid_cuda.cu
```

## Run Instructions

```
./tests/test_gaussian_blur
```

The output should look like:

```
Max abs diff (CUDA vs CPU): 1.2e-06
Test PASSED!
```

A difference below 1e-4 is considered a pass (matches C++/Python tolerance in EVM tests).
