#include <iostream>
#include <cuda_runtime.h>
#include "cuda_color_conversion.cuh"
#include "cuda_pyramid.cuh"
#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
    std::cout << "Eulerian Video Magnification CUDA Implementation" << std::endl;
    
    // Initialize CUDA modules
    if (!evmcuda::init_color_conversion()) {
        std::cerr << "Failed to initialize color conversion module" << std::endl;
        return 1;
    }
    
    if (!evmcuda::init_pyramid()) {
        std::cerr << "Failed to initialize pyramid module" << std::endl;
        return 1;
    }
    
    // Check command line arguments
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <input_video> <output_video> [options]" << std::endl;
        std::cout << "Options:" << std::endl;
        std::cout << "  --levels <int>        : Number of pyramid levels (default: 4)" << std::endl;
        std::cout << "  --alpha <float>       : Amplification factor (default: 50)" << std::endl;
        std::cout << "  --fl <float>          : Low frequency cutoff (default: 0.05)" << std::endl;
        std::cout << "  --fh <float>          : High frequency cutoff (default: 0.4)" << std::endl;
        std::cout << "  --chrom <float>       : Chrominance attenuation (default: 0.1)" << std::endl;
        return 0;
    }
    
    // For now, just print that the implementation is not complete
    std::cout << "The full implementation is in progress." << std::endl;
    std::cout << "Color conversion and pyramid operations have been implemented." << std::endl;
    std::cout << "Run test_cuda_color_conversion and test_cuda_pyramid to validate." << std::endl;
    
    // Clean up CUDA modules
    evmcuda::cleanup_color_conversion();
    evmcuda::cleanup_pyramid();
    
    return 0;
}