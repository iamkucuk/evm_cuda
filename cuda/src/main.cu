#include <iostream>
#include <string>
#include <cstdlib>
#include <cuda_runtime.h>
#include "cuda_evm.cuh"
#include <opencv2/opencv.hpp>

// Configuration structure
struct Config {
    std::string input_video_path;
    std::string output_video_path;
    std::string mode = "laplacian"; // Default mode
    int levels = 4;
    double alpha = 50.0;
    double lambda_cutoff = 16.0; // Only used in Laplacian mode
    double fl = 0.05;
    double fh = 0.4;
    double chrom_attenuation = 0.1;
};

void print_usage(const char* prog_name) {
    std::cout << "Eulerian Video Magnification CUDA Implementation" << std::endl;
    std::cout << "Usage: " << prog_name << " <input_video> <output_video> [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "EVM Modes:" << std::endl;
    std::cout << "  --mode <mode>         : Processing mode: 'gaussian' or 'laplacian' (default: laplacian)" << std::endl;
    std::cout << "    gaussian            : Color/intensity amplification (pulse detection)" << std::endl;
    std::cout << "    laplacian           : Motion amplification (mechanical vibrations)" << std::endl;
    std::cout << std::endl;
    std::cout << "Common Options:" << std::endl;
    std::cout << "  --levels <int>        : Number of pyramid/spatial levels (default: 4)" << std::endl;
    std::cout << "  --alpha <float>       : Amplification factor (default: 50)" << std::endl;
    std::cout << "  --fl <float>          : Low frequency cutoff (default: 0.05)" << std::endl;
    std::cout << "  --fh <float>          : High frequency cutoff (default: 0.4)" << std::endl;
    std::cout << "  --chrom <float>       : Chrominance attenuation (default: 0.1)" << std::endl;
    std::cout << std::endl;
    std::cout << "Laplacian Mode Only:" << std::endl;
    std::cout << "  --lambda_cutoff <f>   : Spatial wavelength cutoff (default: 16.0)" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  # Pulse detection (Gaussian mode)" << std::endl;
    std::cout << "  " << prog_name << " face.mp4 face_pulse.mp4 --mode gaussian --alpha 100 --fl 0.8 --fh 1.0" << std::endl;
    std::cout << std::endl;
    std::cout << "  # Motion amplification (Laplacian mode)" << std::endl;
    std::cout << "  " << prog_name << " baby.mp4 baby_motion.mp4 --mode laplacian --alpha 20 --fl 0.05 --fh 0.4" << std::endl;
}

bool parse_args(int argc, char* argv[], Config& config) {
    if (argc < 3) {
        print_usage(argv[0]);
        return false;
    }
    
    config.input_video_path = argv[1];
    config.output_video_path = argv[2];
    
    for (int i = 3; i < argc; i += 2) {
        if (i + 1 >= argc) {
            std::cerr << "Error: Missing value for argument " << argv[i] << std::endl;
            return false;
        }
        
        std::string arg = argv[i];
        std::string value = argv[i + 1];
        
        if (arg == "--mode") {
            config.mode = value;
            if (config.mode != "gaussian" && config.mode != "laplacian") {
                std::cerr << "Error: Invalid mode '" << config.mode << "'. Must be 'gaussian' or 'laplacian'." << std::endl;
                return false;
            }
        } else if (arg == "--levels") {
            config.levels = std::stoi(value);
        } else if (arg == "--alpha") {
            config.alpha = std::stod(value);
        } else if (arg == "--lambda_cutoff") {
            config.lambda_cutoff = std::stod(value);
        } else if (arg == "--fl") {
            config.fl = std::stod(value);
        } else if (arg == "--fh") {
            config.fh = std::stod(value);
        } else if (arg == "--chrom") {
            config.chrom_attenuation = std::stod(value);
        } else {
            std::cerr << "Error: Unknown argument " << arg << std::endl;
            return false;
        }
    }
    
    // Validate parameters
    if (config.levels <= 0) {
        std::cerr << "Error: Levels must be positive" << std::endl;
        return false;
    }
    
    if (config.fl >= config.fh || config.fl <= 0 || config.fh <= 0) {
        std::cerr << "Error: Invalid frequency range (fl=" << config.fl << ", fh=" << config.fh << "). Requires 0 < fl < fh." << std::endl;
        return false;
    }
    
    return true;
}

int main(int argc, char** argv) {
    std::cout << "Eulerian Video Magnification CUDA Implementation" << std::endl;
    std::cout << "Supporting both Gaussian and Laplacian modes" << std::endl;
    std::cout << std::endl;
    
    // Parse command line arguments
    Config config;
    if (!parse_args(argc, argv, config)) {
        return 1;
    }
    
    // Print configuration
    std::cout << "--- Configuration ---" << std::endl;
    std::cout << "Mode: " << config.mode << std::endl;
    std::cout << "Input Video: " << config.input_video_path << std::endl;
    std::cout << "Output Video: " << config.output_video_path << std::endl;
    std::cout << "Levels: " << config.levels << std::endl;
    std::cout << "Alpha: " << config.alpha << std::endl;
    if (config.mode == "laplacian") {
        std::cout << "Lambda Cutoff: " << config.lambda_cutoff << std::endl;
    }
    std::cout << "Freq Low (fl): " << config.fl << " Hz" << std::endl;
    std::cout << "Freq High (fh): " << config.fh << " Hz" << std::endl;
    std::cout << "Chrom Attenuation: " << config.chrom_attenuation << std::endl;
    std::cout << "---------------------" << std::endl;
    
    try {
        // Initialize EVM modules
        if (!evmcuda::init_evm()) {
            std::cerr << "Failed to initialize EVM CUDA modules" << std::endl;
            return 1;
        }
        
        // Process video based on selected mode
        if (config.mode == "gaussian") {
            std::cout << "Starting Gaussian mode processing..." << std::endl;
            evmcuda::process_video_gaussian(
                config.input_video_path,
                config.output_video_path,
                config.levels,
                config.alpha,
                config.fl,
                config.fh,
                config.chrom_attenuation
            );
        } else if (config.mode == "laplacian") {
            std::cout << "Starting Laplacian mode processing..." << std::endl;
            evmcuda::process_video_laplacian(
                config.input_video_path,
                config.output_video_path,
                config.levels,
                config.alpha,
                config.lambda_cutoff,
                config.fl,
                config.fh,
                config.chrom_attenuation
            );
        } else {
            std::cerr << "Error: Unknown mode '" << config.mode << "'" << std::endl;
            return 1;
        }
        
        std::cout << std::endl;
        std::cout << "Processing complete! Output saved to: " << config.output_video_path << std::endl;
        
        // Clean up CUDA modules
        evmcuda::cleanup_evm();
        
    } catch (const std::exception& e) {
        std::cerr << "Error during processing: " << e.what() << std::endl;
        evmcuda::cleanup_evm();
        return 1;
    }
    
    return 0;
}